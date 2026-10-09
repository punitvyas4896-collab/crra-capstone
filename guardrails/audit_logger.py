"""CRRA Lab C4 - Audit trail (vibe-coded version).

Append-only: every run adds to logs/audit_trail.jsonl, one JSON object per line.
Delete the file by hand if you want a clean log for a demo.
"""
import json
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "audit_trail.jsonl"


class AuditLogger:
    def __init__(self, log_path=LOG_PATH):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries = []   # this run only, for the summary table

    def log(self, agent, action, contract_id, rationale="",
            approval_status="N/A", actor="system"):
        entry = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                 "agent": agent, "action": action, "contract_id": contract_id,
                 "rationale": rationale, "approval_status": approval_status,
                 "actor": actor}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self.entries.append(entry)
        short = rationale if len(rationale) <= 70 else rationale[:67] + "..."
        print(f"  [AUDIT] {agent}: {action} ({approval_status}) - {short}")
        return entry

    def summary(self):
        print(f"\n{'=' * 72}\nAUDIT TRAIL - {len(self.entries)} entries this run\n{'=' * 72}")
        print(f"{'Time':<10}{'Contract':<10}{'Agent':<15}{'Action':<20}{'Status':<10}Actor")
        print("-" * 72)
        for e in self.entries:
            print(f"{e['timestamp'][11:]:<10}{e['contract_id']:<10}{e['agent']:<15}"
                  f"{e['action'][:19]:<20}{e['approval_status']:<10}{e['actor']}")
        print(f"-" * 72 + f"\nAppended to {self.log_path}")