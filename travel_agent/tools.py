"""Agent tools. Each is a small deterministic function over the claim + policy data.

Numbers (caps, totals, tiers) always come from these tools - the LLM decides *which* tools to
call and how to combine their results, but never does the arithmetic itself.
"""
from __future__ import annotations

import json
import re

from .config import DATA_DIR
from .models import Claim, ClaimItem
from .policy_store import APPROVAL, LIMITS, search_policy

R2 = lambda x: round(float(x) + 1e-9, 2)  # noqa: E731


# ---------- helpers ----------
def normalize_category(raw: str) -> str:
    c = re.sub(r"[\s\-]+", "_", raw.strip().lower())
    return LIMITS["aliases"].get(c, c)


def classify_item(item: ClaimItem) -> tuple[str, str]:
    cat = normalize_category(item.category)
    if cat in LIMITS["categories"]["ineligible"]:
        return "ineligible", cat
    if cat in LIMITS["categories"]["eligible"]:
        return "eligible", cat
    return "unknown", cat


def fare_class(item: ClaimItem) -> str | None:
    raw = item.fare_class or (re.search(r"\b(economy|coach|business|first|premium)\b", item.description.lower()) or [None, None])[1]
    if not raw:
        return None
    raw = raw.strip().lower()
    return "economy" if raw == "coach" else raw


def load_receipts(claim_id: str) -> list[dict] | None:
    p = DATA_DIR / "receipts" / f"{claim_id}.json"
    return json.loads(p.read_text())["receipts"] if p.exists() else None


# ---------- tools ----------
def lookup_policy(query: str) -> dict:
    return {"rules": search_policy(query)}


def check_claim_consistency(claim: Claim) -> dict:
    conflicts = []
    if claim.trip_end < claim.trip_start:
        conflicts.append("Trip end date is before trip start date.")
    if claim.submitted_date < claim.trip_start:
        conflicts.append("Claim was submitted before the trip started.")
    if claim.total_claimed is not None and abs(claim.total_claimed - claim.total) > 0.01:
        conflicts.append(f"Stated total ${claim.total_claimed:,.2f} does not match sum of line items ${claim.total:,.2f}.")
    for it in claim.items:
        if it.expense_date and not (claim.trip_start <= it.expense_date <= claim.trip_end):
            conflicts.append(f"'{it.description}' is dated {it.expense_date}, outside the trip window.")
    return {"consistent": not conflicts, "conflicts": conflicts, "sum_of_items": claim.total}


def check_category_eligibility(claim: Claim) -> dict:
    rows = []
    for i, it in enumerate(claim.items):
        status, cat = classify_item(it)
        flags, rule, note = [], "POL-CAT-01", ""
        if status == "ineligible":
            rule, note = "POL-CAT-02", f"'{cat}' is never reimbursable."
        elif status == "unknown":
            note = f"Category '{cat}' is not covered by the policy."
        else:
            if cat == "airfare":
                fc = fare_class(it)
                if fc is None:
                    flags.append("AIRFARE_CLASS_UNSPECIFIED")
                elif fc != "economy":
                    flags.append("AIRFARE_NOT_ECONOMY")
                if flags:
                    rule = "POL-AIR-01"
            if cat == "meals" and it.attendees > 1:
                flags.append("GROUP_MEAL")
        rows.append({"index": i, "category": cat, "description": it.description, "amount": R2(it.amount),
                     "status": status, "flags": flags, "rule": rule, "note": note})
    return {"items": rows, "ineligible_total": R2(sum(r["amount"] for r in rows if r["status"] == "ineligible"))}


def check_receipts(claim: Claim) -> dict:
    meta = load_receipts(claim.claim_id)
    threshold, always = LIMITS["receipt"]["required_above"], LIMITS["receipt"]["always_required"]
    missing, conflicts, required = [], [], 0
    for i, it in enumerate(claim.items):
        status, cat = classify_item(it)
        if status == "ineligible":
            continue  # nothing to reimburse, so no receipt needed
        if not (it.amount > threshold or cat in always):
            continue
        required += 1
        label = f"Itemized receipt - {cat} (${it.amount:,.2f})"
        if not it.receipt_attached:
            missing.append({"index": i, "category": cat, "amount": R2(it.amount), "label": label})
            continue
        if meta is not None:  # cross-check the attachment metadata on file
            rec = next((r for r in meta if normalize_category(r["category"]) == cat and abs(r["amount"] - it.amount) < 0.01), None)
            if rec is None:
                conflicts.append(f"Receipt marked attached for {cat} (${it.amount:,.2f}) but no matching receipt is on file.")
            elif not rec.get("itemized", True):
                missing.append({"index": i, "category": cat, "amount": R2(it.amount), "label": label + " - attached receipt is not itemized"})
    return {"required_count": required, "missing": missing, "conflicts": conflicts}


def check_per_diem_limits(claim: Claim) -> dict:
    rows, ineligible_total = [], 0.0
    for i, it in enumerate(claim.items):
        status, cat = classify_item(it)
        if status == "ineligible":
            ineligible_total += it.amount
            continue
        row = {"index": i, "category": cat, "claimed": R2(it.amount), "cap": None, "rate": None, "unit": None, "units": it.units, "rule": None}
        pd = LIMITS["per_diem"].get(cat)
        if pd:
            cap = pd["cap"] * it.units
            allowed = min(it.amount, cap)
            row.update(cap=R2(cap), rate=pd["cap"], unit=pd["unit"], rule=pd["rule"])
        else:
            allowed = it.amount
        row["allowed"], row["excess"] = R2(allowed), R2(it.amount - allowed)
        rows.append(row)
    return {
        "items": rows,
        "per_diem_excess_total": R2(sum(r["excess"] for r in rows)),
        "ineligible_total": R2(ineligible_total),
        "total_reimbursable": R2(sum(r["allowed"] for r in rows)),
        "total_claimed": claim.total,
    }


def check_timeliness(claim: Claim) -> dict:
    window = LIMITS["submission_window_days"]
    worst = max((claim.submitted_date - (it.expense_date or claim.trip_start)).days for it in claim.items)
    return {"max_days_elapsed": worst, "window_days": window, "late": worst > window, "rule": "POL-TIME-01"}


def check_approval_threshold(total_amount: float) -> dict:
    for t in APPROVAL["tiers"]:
        if t["max"] is None or total_amount <= t["max"]:
            return {"tier_id": t["id"], "tier_name": t["name"], "action": t["action"],
                    "within_agent_authority": t["action"] != "MANUAL_REVIEW", "total_evaluated": R2(total_amount)}
    raise ValueError("approval matrix has no matching tier")


# ---------- tool registry / schemas exposed to the LLM ----------
_NO_ARGS = {"type": "object", "properties": {}, "additionalProperties": False}

TOOL_SCHEMAS = [
    {"name": "lookup_policy",
     "description": "Retrieve the policy rules (with POL-* ids) most relevant to a topic, e.g. 'meals per diem' or 'missing receipt'.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "check_claim_consistency",
     "description": "Detect conflicting information: stated total vs line items, impossible or out-of-window dates.",
     "input_schema": _NO_ARGS},
    {"name": "check_category_eligibility",
     "description": "Classify each line item as eligible / ineligible / unknown (POL-CAT-01/02) and flag airfare class (POL-AIR-01) and group meals.",
     "input_schema": _NO_ARGS},
    {"name": "check_receipts",
     "description": "Check receipt completeness (POL-RCT-01/02): which items need an itemized receipt, which are missing, and any attachment conflicts.",
     "input_schema": _NO_ARGS},
    {"name": "check_per_diem_limits",
     "description": "Apply per-diem caps (meals, lodging, ground transport). Returns allowed/excess per item and total_reimbursable.",
     "input_schema": _NO_ARGS},
    {"name": "check_timeliness",
     "description": "Check the 30-day submission window (POL-TIME-01).",
     "input_schema": _NO_ARGS},
    {"name": "check_approval_threshold",
     "description": "Map the total reimbursable amount (from check_per_diem_limits.total_reimbursable) to an approval tier (POL-APR-01/02/03).",
     "input_schema": {"type": "object", "properties": {"total_amount": {"type": "number"}}, "required": ["total_amount"]}},
]


class ToolBox:
    """Tools bound to ONE claim, so the model cannot mix data between claims. Records every call."""

    def __init__(self, claim: Claim):
        self.claim, self.calls, self.results = claim, [], {}
        self._fns = {
            "lookup_policy": lambda a: lookup_policy(a.get("query", "")),
            "check_claim_consistency": lambda a: check_claim_consistency(claim),
            "check_category_eligibility": lambda a: check_category_eligibility(claim),
            "check_receipts": lambda a: check_receipts(claim),
            "check_per_diem_limits": lambda a: check_per_diem_limits(claim),
            "check_timeliness": lambda a: check_timeliness(claim),
            "check_approval_threshold": lambda a: check_approval_threshold(float(a["total_amount"])),
        }

    def call(self, name: str, args: dict | None = None) -> dict:
        if name not in self._fns:
            return {"error": f"unknown tool '{name}'"}
        try:
            out = self._fns[name](args or {})
        except Exception as exc:  # tool errors go back to the model instead of crashing the run
            return {"error": f"{type(exc).__name__}: {exc}"}
        if name not in self.calls:
            self.calls.append(name)
        self.results[name] = out
        return out
