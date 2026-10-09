"""CRRA Lab C2 - Mock Contract API (vibe-coded version).

Run from the project root:  python mcp_server/contract_shim.py
Then open http://localhost:5001/health in a browser.
"""
import csv
from datetime import date, timedelta
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "contracts.csv"
SIMULATED_TODAY = date(2025, 4, 1)  # fixed so every participant gets identical results
CONTRACTS = []


def derive(row):
    """Convert raw CSV strings to types and add the four policy fields."""
    for key in ("annual_value_inr", "notice_days", "seats_purchased",
                "seats_active", "proposed_uplift_pct"):
        row[key] = int(row[key])
    row["auto_renew"] = row["auto_renew"].strip().upper() == "Y"

    renewal = date.fromisoformat(row["renewal_date"])
    deadline = renewal - timedelta(days=row["notice_days"])
    days_to_deadline = (deadline - SIMULATED_TODAY).days
    row["notice_deadline"] = deadline.isoformat()
    row["days_to_renewal"] = (renewal - SIMULATED_TODAY).days
    row["days_to_notice_deadline"] = days_to_deadline

    if renewal < SIMULATED_TODAY:
        row["notice_state"] = "EXPIRED"
    elif deadline <= SIMULATED_TODAY:
        row["notice_state"] = "INSIDE_WINDOW"   # leverage is gone
    elif days_to_deadline <= 30:
        row["notice_state"] = "APPROACHING"
    else:
        row["notice_state"] = "OPEN"

    purchased = row["seats_purchased"]
    row["utilisation_pct"] = (round(100 * row["seats_active"] / purchased)
                              if purchased > 0 else None)  # AMC / support plans

    value = row["annual_value_inr"]
    row["approval_band"] = "A" if value < 1_000_000 else "B" if value <= 5_000_000 else "C"
    return row


def load_contracts():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        CONTRACTS[:] = [derive(row) for row in csv.DictReader(f)]


def find(contract_id):
    return next((c for c in CONTRACTS
                 if c["contract_id"].upper() == contract_id.upper()), None)


def not_found(contract_id):
    return jsonify({"error": f"Contract {contract_id} not found"}), 404


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "mock-contract-api",
                    "contracts_loaded": len(CONTRACTS),
                    "simulated_today": SIMULATED_TODAY.isoformat()})


@app.get("/api/contracts")
def list_contracts():
    results = CONTRACTS
    for param, field in (("category", "category"), ("band", "approval_band"),
                         ("notice_state", "notice_state")):
        wanted = request.args.get(param)
        if wanted:
            results = [c for c in results if c[field].lower() == wanted.lower()]
    return jsonify({"count": len(results), "contracts": results})


# Declared before /<contract_id> for readability; Flask matches the literal path first anyway.
@app.get("/api/contracts/expiring")
def expiring():
    try:
        window = int(request.args.get("days", 90))
    except ValueError:
        return jsonify({"error": "days must be a whole number"}), 400
    results = sorted((c for c in CONTRACTS if 0 <= c["days_to_renewal"] <= window),
                     key=lambda c: c["days_to_renewal"])
    return jsonify({"count": len(results), "window_days": window, "contracts": results})


@app.get("/api/contracts/<contract_id>")
def get_contract(contract_id):
    contract = find(contract_id)
    return jsonify(contract) if contract else not_found(contract_id)


@app.get("/api/categories")
def categories():
    grouped = {}
    for c in CONTRACTS:
        grouped.setdefault(c["category"], []).append(
            {key: c[key] for key in ("contract_id", "vendor",
                                     "annual_value_inr", "utilisation_pct")})
    summary = [{"category": cat, "vendor_count": len(items),
                "total_annual_value_inr": sum(i["annual_value_inr"] for i in items),
                "vendors": items}
               for cat, items in sorted(grouped.items())]
    return jsonify({"count": len(summary), "categories": summary})


@app.patch("/api/contracts/<contract_id>")
def update_contract(contract_id):
    """In-memory only - restarting the server undoes every change."""
    contract = find(contract_id)
    if not contract:
        return not_found(contract_id)
    payload = request.get_json(silent=True) or {}
    changed = {k: payload[k] for k in ("status", "owner", "proposed_uplift_pct")
               if k in payload}
    if not changed:
        return jsonify({"error": "Send JSON with status, owner or proposed_uplift_pct"}), 400
    contract.update(changed)
    return jsonify({"updated": True, "contract": contract})


if __name__ == "__main__":
    load_contracts()
    print(f"Mock Contract API: {len(CONTRACTS)} contracts loaded, "
          f"simulated today {SIMULATED_TODAY}")
    print("Open http://localhost:5001/health")
    app.run(port=5001, debug=False)