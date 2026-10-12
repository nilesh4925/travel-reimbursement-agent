"""Claim intake: JSON (file, folder, pasted text) or CSV (one row per line item)."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .models import Claim


def _bool(v) -> bool:
    return str(v).strip().lower() in {"yes", "y", "true", "1"}


def parse_json(text: str) -> list[Claim]:
    data = json.loads(text)
    if isinstance(data, dict):
        data = [data]
    return [Claim(**d) for d in data]


def parse_csv(text: str) -> list[Claim]:
    groups: dict[str, dict] = {}
    for row in csv.DictReader(io.StringIO(text)):
        cid = row["claim_id"].strip()
        g = groups.setdefault(cid, {
            "claim_id": cid, "employee": row["employee"], "purpose": row["purpose"],
            "trip_start": row["trip_start"], "trip_end": row["trip_end"],
            "submitted_date": row["submitted_date"],
            "total_claimed": float(row["total_claimed"]) if row.get("total_claimed") else None,
            "items": [],
        })
        item = {"category": row["category"], "description": row["description"],
                "amount": float(row["amount"]), "receipt_attached": _bool(row["receipt_attached"])}
        for k in ("units", "attendees"):
            if row.get(k):
                item[k] = int(row[k])
        for k in ("fare_class", "expense_date"):
            if row.get(k):
                item[k] = row[k]
        g["items"].append(item)
    return [Claim(**g) for g in groups.values()]


def load_claims(path) -> list[Claim]:
    p = Path(path)
    if p.is_dir():
        return [c for f in sorted(p.glob("*.json")) for c in parse_json(f.read_text())]
    text = p.read_text()
    return parse_csv(text) if p.suffix.lower() == ".csv" else parse_json(text)
