"""CRRA Lab C3 - Renewal Analysis Agent (vibe-coded version).

Decides RENEW / RENEGOTIATE / CONSOLIDATE / TERMINATE for a contract, with a
confidence level and a citation to the procurement policy.

Before running:
    Tab 1:  python mcp_server/contract_shim.py      (Lab C2 - leave running)
    Tab 2:  python agents/renewal_agent.py          (from the project root)
"""
import json
import os
import sys
from pathlib import Path

import anthropic
import chromadb
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # so "from data.kb_setup import ..." works
load_dotenv(ROOT / ".env")

MODEL = "claude-opus-5"
CONTRACT_API = "http://localhost:5001"
KB_COLLECTION = "crra_policy"
MAX_ROUNDS = 5          # hard cap: an agent that cannot converge must still stop
TEST_CONTRACTS = ["CTR-1003", "CTR-1012", "CTR-1005", "CTR-1006", "CTR-1004"]


# ---------------------------------------------------------------- helpers
def extract_text(response):
    """First block that has text. A thinking block may come before it."""
    for block in response.content:
        if hasattr(block, "text"):
            return block.text.strip()
    return ""


def load_kb():
    """Lab C1 uses an in-memory ChromaDB client, so the collection does not
    survive between scripts. Rebuild it from data/kb/*.md if it is missing."""
    kb_client = chromadb.Client()
    try:
        return kb_client.get_collection(KB_COLLECTION)
    except Exception:
        from data.kb_setup import chunk_article   # reuse Lab C1's chunker
        collection = kb_client.create_collection(KB_COLLECTION)
        chunks = [c for md in sorted((ROOT / "data" / "kb").glob("*.md"))
                  for c in chunk_article(md.read_text(encoding="utf-8"), md.name)]
        if not chunks:
            raise SystemExit("No policy files found in data/kb/ - check Lab C1.")
        collection.add(ids=[c["id"] for c in chunks],
                       documents=[c["document"] for c in chunks],
                       metadatas=[c["metadata"] for c in chunks])
        print(f"Policy KB built in memory: {len(chunks)} sections.")
        return collection


# ---------------------------------------------------------------- tools
def get_contract(contract_id):
    try:
        r = requests.get(f"{CONTRACT_API}/api/contracts/{contract_id}", timeout=10)
    except requests.exceptions.RequestException:
        return {"error": "Contract API unreachable. Is contract_shim.py running on "
                         "port 5001? Check http://localhost:5001/health"}
    if r.status_code == 404:
        return {"error": f"Contract {contract_id} not found"}
    return r.json()


def search_policy(query):
    res = KB.query(query_texts=[query], n_results=8)
    best = {}   # source file -> (distance, heading, text)
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                                res["distances"][0]):
        src = meta["source"]
        if src not in best or dist < best[src][0]:
            best[src] = (dist, meta["heading"], text)
    top2 = sorted(best.items(), key=lambda kv: kv[1][0])[:2]
    return {"results": [{"source": src, "section": heading,
                         "confidence": round(max(0.0, 1 - dist), 2), "text": text}
                        for src, (dist, heading, text) in top2]}


def find_category_overlap(category):
    try:
        r = requests.get(f"{CONTRACT_API}/api/categories", timeout=10)
    except requests.exceptions.RequestException:
        return {"error": "Contract API unreachable. Is contract_shim.py running?"}
    for entry in r.json()["categories"]:
        if entry["category"].lower() == category.lower():
            return entry
    return {"category": category, "vendor_count": 0, "vendors": [],
            "note": "No contracts in this category."}


TOOLS = [
    {"name": "get_contract",
     "description": "Fetch one contract with its derived fields: approval_band, notice_state, "
                    "days_to_notice_deadline, utilisation_pct (null for AMC/support), "
                    "proposed_uplift_pct, annual_value_inr.",
     "input_schema": {"type": "object",
                      "properties": {"contract_id": {"type": "string", "description": "e.g. CTR-1004"}},
                      "required": ["contract_id"]}},
    {"name": "search_policy",
     "description": "Search the procurement policy. Returns the best-matching section from "
                    "each of the top 2 policy files. Use at most twice per contract.",
     "input_schema": {"type": "object",
                      "properties": {"query": {"type": "string",
                                               "description": "Plain-English policy question"}},
                      "required": ["query"]}},
    {"name": "find_category_overlap",
     "description": "All vendors in a category with value and utilisation. Use only when "
                    "considering CONSOLIDATE.",
     "input_schema": {"type": "object",
                      "properties": {"category": {"type": "string", "description": "e.g. Observability"}},
                      "required": ["category"]}},
    {"name": "submit_recommendation",
     "description": "Record the final recommendation. Call exactly once, as the last step.",
     "input_schema": {"type": "object",
                      "properties": {
                          "contract_id": {"type": "string"},
                          "recommendation": {"type": "string",
                                             "enum": ["RENEW", "RENEGOTIATE", "CONSOLIDATE", "TERMINATE"]},
                          "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                          "rationale": {"type": "string",
                                        "description": "2-3 sentences naming the numbers that drove the call."},
                          "policy_citation": {"type": "string",
                                              "description": "File and section, e.g. 'renegotiation_levers.md §Price uplift benchmarks'"},
                          "estimated_annual_impact_inr": {"type": "integer",
                                                          "description": "Negative = saving. 0 if no change."},
                          "human_approval_required": {"type": "boolean"}},
                      "required": ["contract_id", "recommendation", "confidence", "rationale",
                                   "policy_citation", "estimated_annual_impact_inr",
                                   "human_approval_required"]}},
]

TOOL_FUNCS = {"get_contract": get_contract, "search_policy": search_policy,
              "find_category_overlap": find_category_overlap}

SYSTEM_PROMPT = """You are the Renewal Analysis Agent for Zensar BizOps.
For the contract you are given, recommend exactly one of RENEW, RENEGOTIATE,
CONSOLIDATE or TERMINATE.

Method, in order:
1. get_contract - read the real numbers. Never assume them.
2. search_policy - find the rule that governs this case (at most twice).
3. find_category_overlap - only if you are considering CONSOLIDATE.
4. submit_recommendation - exactly once, to finish.
You have a small budget of turns; you may request several tools in one turn.

Guidance:
- Utilisation below 40% with a viable overlapping vendor points to CONSOLIDATE.
- Proposed uplift above 15% is never accepted at first offer: RENEGOTIATE.
- High utilisation with a modest uplift is a healthy RENEW.
- TERMINATE only when the capability itself is no longer needed. A missing owner
  is a reason to escalate to a human, not proof the tool is unneeded.
- utilisation_pct is null for AMC and support contracts; judge those on other facts.
- INSIDE_WINDOW means the notice deadline has passed and leverage is weakened; say so.

Confidence:
- HIGH: the numbers and the policy clearly agree.
- MEDIUM: sound, but rests on an assumption - name it in the rationale.
- LOW: genuinely unclear. LOW confidence is a valid and useful answer. Submit it
  honestly rather than inventing certainty or calling more tools hoping for clarity.

Set human_approval_required to true for approval bands B and C, for any contract
INSIDE_WINDOW, and for every TERMINATE.
Cite the specific policy file and section that supports the recommendation."""


# ---------------------------------------------------------------- agent loop
def no_decision(contract_id, reason):
    return {"contract_id": contract_id, "recommendation": "NO DECISION",
            "confidence": "LOW", "human_approval_required": True,
            "estimated_annual_impact_inr": 0, "policy_citation": "-", "rationale": reason}


def describe(name, args, result):
    if "error" in result:
        return f"  x {name}: {result['error']}"
    if name == "get_contract":
        util = result["utilisation_pct"]
        return (f"  > contract: band {result['approval_band']}, {result['notice_state']}, "
                f"util {'n/a' if util is None else f'{util}%'}, "
                f"uplift {result['proposed_uplift_pct']}%, INR {result['annual_value_inr']:,}")
    if name == "search_policy":
        hits = ", ".join(f"{h['source']} §{h['section']} ({h['confidence']:.0%})"
                         for h in result["results"])
        return f"  > policy '{args['query'][:40]}': {hits}"
    vendors = ", ".join(v["vendor"] for v in result.get("vendors", []))
    return f"  > overlap in {args['category']}: {vendors or 'none'}"


def analyse_contract(client, contract_id):
    print(f"\n{'=' * 64}\nANALYSING {contract_id}\n{'=' * 64}")
    messages = [{"role": "user",
                 "content": f"Analyse contract {contract_id} and recommend an action."}]

    for round_no in range(1, MAX_ROUNDS + 1):
        try:
            response = client.messages.create(
                model=MODEL, max_tokens=4000, output_config={"effort": "medium"},
                system=SYSTEM_PROMPT, tools=TOOLS, messages=messages)
        except anthropic.APIError as e:
            print(f"  x API error: {e}")
            return no_decision(contract_id, f"API error: {e}")

        if response.stop_reason != "tool_use":
            text = extract_text(response)
            print(f"  Agent stopped without submitting: {text[:200]}")
            return no_decision(contract_id, text or f"stop_reason={response.stop_reason}")

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "submit_recommendation":
                rec = dict(block.input)
                print(f"\n  RECOMMENDATION: {rec['recommendation']}  ({rec['confidence']})")
                print(f"  Policy: {rec['policy_citation']}")
                print(f"  Human approval required: {rec['human_approval_required']}")
                print(f"  {rec['rationale']}")
                return rec
            func = TOOL_FUNCS.get(block.name)
            result = func(**block.input) if func else {"error": f"unknown tool {block.name}"}
            print(describe(block.name, block.input, result))
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(result)})
        messages.append({"role": "user", "content": results})

    print(f"  ! Hit MAX_ROUNDS ({MAX_ROUNDS}) without a recommendation.")
    return no_decision(contract_id, f"No recommendation within {MAX_ROUNDS} rounds.")


def print_summary(recs):
    print(f"\n\n{'=' * 64}\nPORTFOLIO SUMMARY\n{'=' * 64}")
    print(f"{'Contract':<10}{'Action':<14}{'Conf':<8}{'Approval':<10}{'Impact (INR/yr)':>18}")
    print("-" * 64)
    for r in recs:
        impact = r.get("estimated_annual_impact_inr") or 0
        print(f"{r['contract_id']:<10}{r['recommendation']:<14}{r['confidence']:<8}"
              f"{'YES' if r['human_approval_required'] else 'no':<10}"
              f"{f'{impact:,}' if impact else '-':>18}")
    print("-" * 64)
    decided = sum(r["recommendation"] != "NO DECISION" for r in recs)
    humans = sum(bool(r["human_approval_required"]) for r in recs)
    print(f"{decided}/{len(recs)} decided, {humans} need human approval before action")


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not found. Add it to .env in the project root.")
    try:
        requests.get(f"{CONTRACT_API}/health", timeout=3)
    except requests.exceptions.RequestException:
        raise SystemExit("Contract API unreachable. Start Tab 1: python mcp_server/contract_shim.py")

    client = anthropic.Anthropic()
    print_summary([analyse_contract(client, cid) for cid in TEST_CONTRACTS])


KB = load_kb()

if __name__ == "__main__":
    main()