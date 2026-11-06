#!/usr/bin/env python3
"""experiment_ledger.py — append-only experiment log for WSP runs.

Every experiment (inspect, patch, merge, ablation, etc.) should be logged here.
The ledger is a JSONL file — one record per line, never overwritten.

Usage:
    python experiment_ledger.py log --ledger experiments.jsonl --event patch \\
        --model model.pt --notes "scaled layer 0 attention Q by 0.98" \\
        --result '{"metric": 0.82, "baseline": 0.79}'

    python experiment_ledger.py show --ledger experiments.jsonl
    python experiment_ledger.py show --ledger experiments.jsonl --filter patch
    python experiment_ledger.py show --ledger experiments.jsonl --last 5
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def log_event(
    ledger_path: str,
    event_type: str,
    model: str = None,
    notes: str = None,
    result: dict = None,
    tags: list = None,
    extra: dict = None,
) -> dict:
    """Append one experiment record to the ledger. Returns the written record."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "model": model,
        "notes": notes,
        "result": result or {},
        "tags": tags or [],
        **(extra or {}),
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def read_ledger(ledger_path: str) -> list:
    p = Path(ledger_path)
    if not p.exists():
        return []
    records = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def show_ledger(records: list, filter_type: str = None, last_n: int = None):
    if filter_type:
        records = [r for r in records if r.get("event_type") == filter_type]
    if last_n:
        records = records[-last_n:]

    if not records:
        print("No records found.")
        return

    for r in records:
        ts = r.get("timestamp", "?")[:19]
        event = r.get("event_type", "?")
        model = r.get("model", "?")
        notes = r.get("notes", "")
        result_summary = ""
        if r.get("result"):
            result_summary = " | " + ", ".join(f"{k}={v}" for k, v in list(r["result"].items())[:3])
        print(f"[{ts}] {event:12s} {model}{result_summary} — {notes}")


def main():
    p = argparse.ArgumentParser(description="WSP experiment ledger — append-only log")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("log", help="Append an experiment record")
    pl.add_argument("--ledger", default="experiments.jsonl")
    pl.add_argument("--event", required=True, help="Event type: inspect|patch|merge|ablation|eval|note")
    pl.add_argument("--model", help="Checkpoint path or model name")
    pl.add_argument("--notes", help="Free-text notes")
    pl.add_argument("--result", help="JSON dict of metrics/results")
    pl.add_argument("--tags", nargs="*", help="Optional tags")

    ps = sub.add_parser("show", help="Display ledger records")
    ps.add_argument("--ledger", default="experiments.jsonl")
    ps.add_argument("--filter", help="Filter by event type")
    ps.add_argument("--last", type=int, help="Show last N records")
    ps.add_argument("--json", action="store_true", help="Output as JSON")

    args = p.parse_args()

    if args.cmd == "log":
        result = json.loads(args.result) if args.result else {}
        rec = log_event(args.ledger, args.event, args.model, args.notes, result, args.tags)
        print(json.dumps(rec, indent=2))

    elif args.cmd == "show":
        records = read_ledger(args.ledger)
        if args.json:
            print(json.dumps(records, indent=2))
        else:
            show_ledger(records, args.filter, args.last)


if __name__ == "__main__":
    main()
