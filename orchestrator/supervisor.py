"""CRRA Lab C4 - Portfolio orchestrator (vibe-coded version).

    analysis -> policy_check -> [hitl if needed] -> report

The agent may recommend, but policy says it may not commit Zensar to anything
on its own. policy_check decides, in plain Python, when a human must sign off.

Before running:
    Tab 1:  python mcp_server/contract_shim.py
    Tab 2:  python orchestrator/supervisor.py              (default 3 contracts)
            python orchestrator/supervisor.py CTR-1004     (or pick your own)
"""
import json
import os
import sys
from pathlib import Path
from typing import TypedDict

# `python orchestrator/supervisor.py` only puts orchestrator/ on the import path.
# Add the project root first so `guardrails` and `data` can be imported.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import anthropic                                   # noqa: E402
import chromadb                                    # noqa: E402
import requests                                    # noqa: E402
from dotenv import load_dotenv                     # noqa: E402
from langgraph.graph import END, StateGraph        # noqa: E402

from guardrails.audit_logger import AuditLogger    # noqa: E402

load_dotenv(ROOT / ".env")

MODEL = "claude-opus-5"
CONTRACT_API = "http://localhost:5001"
KB_COLLECTION = "crra_policy"
MAX_ROUNDS = 5
DEFAULT_PORTFOLIO = ["CTR-1010", "CTR-1012", "CTR-1006"]  # auto / gated / gated

audit = AuditLogger()
client = anthropic.Anthropic()


class ContractState(TypedDict):
    contract_id: str
    contract: dict
    recommendation: str
    confidence: str
    rationale: str
    policy_citation: str
    estimated_annual_impact_inr: int
    hitl_required: bool
    hitl_reason: str
    hitl_approved: bool
    approver: str
    final_status: str


def extract_text(response):
    """First block with text - a thinking block may come first."""
    for block in response.content:
        if hasattr(block, "text"):
            return block.text.strip()
    return ""


# ---------------------------------------------------------------- policy KB
def load_kb():
    """Lab C1's ChromaDB is in-memory, so rebuild the collection if missing."""
    kb_client = chromadb.Client()
    try:
        return kb_client.get_collection(KB_COLLECTION)
    except Exception:
        from data.kb_setup import chunk_article
        collection = kb_client.create_collection(KB_COLLECTION)
        chunks = [c for md in sorted((ROOT / "data" / "kb").glob("*.md"))
                  for c in chunk_article(md.read_text(encoding="utf-8"), md.name)]
        collection.add(ids=[c["id"] for c in chunks],
                       documents=[c["document"] for c in chunks],
                       metadatas=[c["metadata"] for c in chunks])
        print(f"Policy KB built in memory: {len(chunks)} sections.")
        return collection


KB = load_kb()


def search_policy(query):
    res = KB.query(query_texts=[query], n_results=8)
    best = {}
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if meta["source"] not in best or dist < best[meta["source"]][0]:
            best[meta["source"]] = (dist, meta["heading"], text)
    top2 = sorted(best.items(), key=lambda kv: kv[1][0])[:2]
    return [{"source": s, "section": h, "confidence": round(max(0.0, 1 - d), 2), "text": t}
            for s, (d, h, t) in top2]


# ---------------------------------------------------------------- node 1: analysis
TOOLS = [
    {"name": "search_policy",
     "description": "Search the procurement policy. Returns the best section from each of "
                    "the top 2 policy files. Use at most twice.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}},
                      "required": ["query"]}},
    {"name": "submit_recommendation",
     "description": "Record the final recommendation. Call exactly once, last.",
     "input_schema": {"type": "object", "properties": {
         "recommendation": {"type": "string",
                            "enum": ["RENEW", "RENEGOTIATE", "CONSOLIDATE", "TERMINATE"]},
         "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
         "rationale": {"type": "string", "description": "2-3 sentences with the numbers."},
         "policy_citation": {"type": "string", "description": "file.md §Section"},
         "estimated_annual_impact_inr": {"type": "integer",
                                         "description": "Negative = saving, 0 = no change."}},
         "required": ["recommendation", "confidence", "rationale", "policy_citation",
                      "estimated_annual_impact_inr"]}},
]

SYSTEM_PROMPT = """You are the Renewal Analysis Agent for Zensar BizOps.
You are given one contract's facts. Recommend exactly one of RENEW, RENEGOTIATE,
CONSOLIDATE or TERMINATE. Search the policy (at most twice), then call
submit_recommendation exactly once. You may request both tools in one turn.

Guidance:
- Utilisation below 40% with an overlapping vendor in the category: CONSOLIDATE.
- Proposed uplift above 15% is never accepted at first offer: RENEGOTIATE.
- High utilisation with a modest uplift: RENEW.
- TERMINATE only when the capability is no longer needed. An UNASSIGNED owner
  means nobody has confirmed that - recommend conservatively and say why.
- utilisation_pct is null for AMC/support contracts; judge on other facts.

LOW confidence is a valid, useful answer. Submit it honestly instead of inventing
certainty or searching again hoping for a cleaner picture. A human reviews it.
Cite the specific policy file and section."""

CONTRACT_FIELDS = ("contract_id", "vendor", "category", "owner", "annual_value_inr",
                   "approval_band", "renewal_date", "notice_deadline", "notice_state",
                   "days_to_notice_deadline", "auto_renew", "seats_purchased",
                   "seats_active", "utilisation_pct", "proposed_uplift_pct")


def manual_review(reason):
    """Fallback when the agent cannot decide. LOW confidence forces the human gate."""
    return {"recommendation": "MANUAL_REVIEW", "confidence": "LOW", "rationale": reason,
            "policy_citation": "-", "estimated_annual_impact_inr": 0}


def analysis_node(state):
    cid = state["contract_id"]
    print(f"\n{'=' * 72}\nCONTRACT {cid}\n{'=' * 72}\n> ANALYSIS")

    try:
        r = requests.get(f"{CONTRACT_API}/api/contracts/{cid}", timeout=10)
    except requests.exceptions.RequestException:
        msg = "Contract API unreachable - is contract_shim.py running on port 5001?"
        print(f"  x {msg}")
        audit.log("AnalysisAgent", "fetch_failed", cid, msg)
        return {"final_status": "ERROR_API_UNREACHABLE"}
    if r.status_code == 404:
        audit.log("AnalysisAgent", "fetch_failed", cid, "contract not found")
        return {"final_status": "ERROR_NOT_FOUND"}

    contract = r.json()
    util = contract["utilisation_pct"]
    print(f"  {contract['vendor']} | {contract['category']} | band {contract['approval_band']} | "
          f"INR {contract['annual_value_inr']:,}/yr | util {'n/a' if util is None else f'{util}%'} | "
          f"uplift {contract['proposed_uplift_pct']}% | {contract['notice_state']}")
    audit.log("AnalysisAgent", "contract_fetched", cid,
              f"band {contract['approval_band']}, {contract['notice_state']}")

    facts = json.dumps({k: contract.get(k) for k in CONTRACT_FIELDS}, indent=2)
    messages = [{"role": "user", "content": f"Analyse this contract:\n{facts}"}]

    for _ in range(MAX_ROUNDS):
        try:
            response = client.messages.create(
                model=MODEL, max_tokens=4000, output_config={"effort": "medium"},
                system=SYSTEM_PROMPT, tools=TOOLS, messages=messages)
        except anthropic.APIError as e:
            print(f"  x Claude API error: {e}")
            audit.log("AnalysisAgent", "model_error", cid, str(e)[:200])
            return {"contract": contract, **manual_review(f"Claude API error: {e}")}

        if response.stop_reason != "tool_use":
            text = extract_text(response)
            print(f"  Agent stopped without submitting: {text[:150]}")
            audit.log("AnalysisAgent", "no_recommendation", cid, text[:200])
            return {"contract": contract,
                    **manual_review(text or "Agent produced no recommendation.")}

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "submit_recommendation":
                rec = block.input
                print(f"  > {rec['recommendation']} (confidence {rec['confidence']})")
                audit.log("AnalysisAgent", "recommendation", cid,
                          f"{rec['recommendation']}/{rec['confidence']} - {rec['policy_citation']}")
                return {"contract": contract,
                        "recommendation": rec["recommendation"],
                        "confidence": rec["confidence"],
                        "rationale": rec["rationale"],
                        "policy_citation": rec["policy_citation"],
                        "estimated_annual_impact_inr": rec.get("estimated_annual_impact_inr", 0)}
            if block.name == "search_policy":
                hits = search_policy(block.input["query"])
                print(f"  > policy '{block.input['query'][:40]}': " + ", ".join(
                    f"{h['source']} §{h['section']} ({h['confidence']:.0%})" for h in hits))
                result = {"results": hits}
            else:
                result = {"error": f"unknown tool {block.name}"}
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(result)})
        messages.append({"role": "user", "content": results})

    print(f"  ! No recommendation within {MAX_ROUNDS} rounds.")
    audit.log("AnalysisAgent", "max_rounds_hit", cid, f"stopped after {MAX_ROUNDS} rounds")
    return {"contract": contract,
            **manual_review(f"Agent did not converge within {MAX_ROUNDS} rounds.")}


# ---------------------------------------------------------------- node 2: policy check
def policy_check_node(state):
    """Pure Python - no model call. Collects EVERY trigger that applies."""
    print("\n> POLICY CHECK")
    if state.get("final_status", "").startswith("ERROR"):
        return {"hitl_required": False, "hitl_reason": "skipped - no contract data"}

    c = state["contract"]
    reasons = []
    if c["approval_band"] in ("B", "C"):
        reasons.append(f"approval band {c['approval_band']} needs a named human approver")
    if c["notice_state"] == "INSIDE_WINDOW":
        reasons.append("inside the notice window - leverage already lost")
    if state["recommendation"] == "TERMINATE":
        reasons.append("every termination needs human sign-off")
    if state["confidence"] == "LOW":
        reasons.append("analysis confidence is LOW")
    if str(c.get("owner", "")).strip().upper() == "UNASSIGNED":
        reasons.append("no business owner on record")
    # Lab C4 Step 6: add your own extra trigger here, e.g.
    # if c["proposed_uplift_pct"] > 15: reasons.append("...")

    hitl_reason = "; ".join(reasons) or "no policy trigger"
    print(f"  HITL required: {bool(reasons)}")
    for reason in reasons:
        print(f"    - {reason}")
    audit.log("PolicyCheck", "evaluate_triggers", state["contract_id"], hitl_reason)
    return {"hitl_required": bool(reasons), "hitl_reason": hitl_reason}


# ---------------------------------------------------------------- node 3: human gate
def ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:          # no keyboard (e.g. piped input): fail safe = reject
        return ""


def hitl_node(state):
    cid = state["contract_id"]
    print("\n> HUMAN APPROVAL GATE")
    print(f"  Contract : {cid} ({state['contract'].get('vendor')})")
    print(f"  Proposed : {state['recommendation']} (confidence {state['confidence']})")
    print(f"  Impact   : INR {state['estimated_annual_impact_inr']:,}/yr")
    print(f"  Because  : {state['hitl_reason']}")
    print(f"  Policy   : {state['policy_citation']}")
    print(f"  Rationale: {state['rationale']}")
    audit.log("HITLGate", "approval_request", cid,
              f"{state['recommendation']} - {state['hitl_reason']}", "PENDING")

    answer = ""
    while answer not in ("y", "n"):
        answer = ask("\n  Approve this recommendation? [y/n]: ").lower()[:1] or "n"
    approved = answer == "y"

    approver = ""
    if approved:
        while not approver:
            approver = ask("  Approver name: ")
            if not approver:
                print("  A name is required - the audit trail must show who approved.")
    reviewer = approver or ask("  Your name (for the audit trail): ") or "unnamed reviewer"

    status = "APPROVED" if approved else "REJECTED"
    audit.log("HITLGate", "approval_decision", cid,
              f"{state['recommendation']} {status.lower()}", status, actor=reviewer)
    print(f"  > {status} by {reviewer}")
    return {"hitl_approved": approved, "approver": approver}


# ---------------------------------------------------------------- node 4: report
def report_node(state):
    print("\n> REPORT")
    cid = state["contract_id"]
    if state.get("final_status", "").startswith("ERROR"):
        print(f"  Skipped - {state['final_status']}")
        return {}

    rec = state["recommendation"]
    if not state["hitl_required"]:
        final, note, status = f"{rec}_AUTO", "No policy trigger - cleared automatically.", "N/A"
    elif state["hitl_approved"]:
        final, note, status = (f"{rec}_APPROVED",
                               f"Approved by {state['approver']}. Cleared to action.", "APPROVED")
    else:
        final, note, status = ("ON_HOLD_REJECTED",
                               "Rejected at the gate. No commitment made to the vendor.", "REJECTED")
    audit.log("Reporting", "final_status", cid, f"{final}: {note}", status)
    print(f"  FINAL STATUS: {final}\n  {note}")
    return {"final_status": final}


# ---------------------------------------------------------------- graph
def build_graph():
    g = StateGraph(ContractState)
    g.add_node("analysis", analysis_node)
    g.add_node("policy_check", policy_check_node)
    g.add_node("hitl", hitl_node)
    g.add_node("report", report_node)
    g.set_entry_point("analysis")
    g.add_edge("analysis", "policy_check")
    g.add_conditional_edges("policy_check",
                            lambda s: "hitl" if s["hitl_required"] else "report",
                            {"hitl": "hitl", "report": "report"})
    g.add_edge("hitl", "report")
    g.add_edge("report", END)
    return g.compile()


def initial_state(cid):
    return {"contract_id": cid, "contract": {}, "recommendation": "", "confidence": "",
            "rationale": "", "policy_citation": "", "estimated_annual_impact_inr": 0,
            "hitl_required": False, "hitl_reason": "", "hitl_approved": False,
            "approver": "", "final_status": ""}


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not found. Add it to .env in the project root.")
    try:
        requests.get(f"{CONTRACT_API}/health", timeout=3)
    except requests.exceptions.RequestException:
        raise SystemExit("Contract API unreachable. Start Tab 1: python mcp_server/contract_shim.py")

    portfolio = [a.upper() for a in sys.argv[1:]] or DEFAULT_PORTFOLIO
    graph = build_graph()
    outcomes = [graph.invoke(initial_state(cid)) for cid in portfolio]

    print(f"\n\n{'=' * 72}\nPORTFOLIO REVIEW COMPLETE\n{'=' * 72}")
    print(f"{'Contract':<10}{'Action':<15}{'Conf':<8}{'Gate':<8}{'Approver':<16}Final status")
    print("-" * 72)
    for o in outcomes:
        print(f"{o['contract_id']:<10}{o['recommendation'] or '-':<15}{o['confidence'] or '-':<8}"
              f"{'HUMAN' if o['hitl_required'] else 'auto':<8}{o['approver'] or '-':<16}"
              f"{o['final_status']}")
    audit.summary()


if __name__ == "__main__":
    main()