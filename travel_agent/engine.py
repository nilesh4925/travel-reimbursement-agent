"""Policy decision engine: turns tool outputs into a decision, amounts, refs and explanation.

Used (a) directly by the offline planner and (b) as the independent verifier of the LLM's decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import Claim
from .policy_store import APPROVAL
from .tools import (check_approval_threshold, check_category_eligibility, check_claim_consistency,
                    check_per_diem_limits, check_receipts, check_timeliness)

AMBIGUOUS_FLAGS = {"GROUP_MEAL", "AIRFARE_CLASS_UNSPECIFIED"}
FLAG_TEXT = {
    "AIRFARE_NOT_ECONOMY": "non-economy airfare is a policy exception - a pre-approval may exist, so it must be verified by a reviewer (POL-AIR-01)",
    "AIRFARE_CLASS_UNSPECIFIED": "airfare class is not stated, so the economy-only rule cannot be confirmed (POL-AIR-01)",
    "GROUP_MEAL": "multi-attendee meal - the per-diem policy does not say how the cap applies to group meals (POL-PD-01)",
}

# Confidence is a transparent function of explicit signals (not a magic number per decision):
#   start from a base by decision type, subtract a penalty for each ambiguity signal, and a small
#   one when an automatic approval sits close to a tier boundary. The evaluation suite reports how
#   well this lines up with observed accuracy (see travel_agent/evaluation.py).
BASE_CONFIDENCE = {"REJECT": 0.98, "APPROVE": 0.97, "PARTIAL_APPROVE": 0.95, "MANUAL_REVIEW": 0.90}
SIGNAL_PENALTY = {"CONFLICT": 0.15, "UNKNOWN_CATEGORY": 0.15, "GROUP_MEAL": 0.15, "AIRFARE_CLASS_UNSPECIFIED": 0.10}
NEAR_BOUNDARY = 0.05  # within 5% of a tier limit


def compute_confidence(decision: str, signals: list[str], total: float) -> float:
    conf = BASE_CONFIDENCE[decision] - sum(SIGNAL_PENALTY[s] for s in set(signals))
    limits = [t["max"] for t in APPROVAL["tiers"] if t["max"]]
    if decision in ("APPROVE", "PARTIAL_APPROVE") and any(abs(total - b) <= NEAR_BOUNDARY * b for b in limits):
        conf -= 0.05
    return round(max(conf, 0.5), 2)


def _m(x: float) -> str:
    return f"${x:,.2f}"


@dataclass
class Assessment:
    decision: str
    approved: float
    deducted: float
    pending_approved: float
    pending_deducted: float
    missing_docs: list[str]
    policy_refs: list[str]
    confidence: float
    reasons: list[str] = field(default_factory=list)
    explanation: str = ""


def run_all_checks(claim: Claim) -> dict:
    o = {
        "check_claim_consistency": check_claim_consistency(claim),
        "check_category_eligibility": check_category_eligibility(claim),
        "check_receipts": check_receipts(claim),
        "check_per_diem_limits": check_per_diem_limits(claim),
        "check_timeliness": check_timeliness(claim),
    }
    o["check_approval_threshold"] = check_approval_threshold(o["check_per_diem_limits"]["total_reimbursable"])
    return o


def decide(claim: Claim, o: dict) -> Assessment:
    cons, elig = o["check_claim_consistency"], o["check_category_eligibility"]["items"]
    rc, pd, tim, tier = o["check_receipts"], o["check_per_diem_limits"], o["check_timeliness"], o["check_approval_threshold"]
    live = [e for e in elig if e["status"] != "ineligible"]
    refs: set[str] = set()
    manual: list[str] = []
    signals: list[str] = []

    # --- collect policy refs for every rule that was actually applied ---
    if any(e["status"] == "ineligible" for e in elig):
        refs.add("POL-CAT-02")
    if live:
        refs.add("POL-CAT-01")
        refs.update(r["rule"] for r in pd["items"] if r["rule"])
        refs.add(tier["tier_id"])
        if any(e["category"] == "airfare" for e in live):
            refs.add("POL-AIR-01")
    if rc["required_count"]:
        refs.add("POL-RCT-01")

    # --- manual-review triggers (and the ambiguity signals that lower confidence) ---
    for c in cons["conflicts"] + rc["conflicts"]:
        manual.append(f"conflicting information: {c}")
        signals.append("CONFLICT")
    for e in live:
        if e["status"] == "unknown":
            manual.append(f"'{e['description']}' has category '{e['category']}' which the policy does not cover")
            signals.append("UNKNOWN_CATEGORY")
        for f in e["flags"]:
            manual.append(f"'{e['description']}' ({_m(e['amount'])}): {FLAG_TEXT[f]}")
            if f in AMBIGUOUS_FLAGS:
                signals.append(f)
            refs.add("POL-AIR-01" if f.startswith("AIRFARE") else "POL-PD-01")
    for m in rc["missing"]:
        manual.append(f"{m['label']} is missing and the reviewer should request it (POL-RCT-01/02)")
        refs.add("POL-RCT-02")
    if tim["late"]:
        manual.append(f"submitted {tim['max_days_elapsed']} days after the expense date, beyond the {tim['window_days']}-day window (POL-TIME-01)")
        refs.add("POL-TIME-01")
    if live and not tier["within_agent_authority"]:
        manual.append(f"reimbursable total {_m(tier['total_evaluated'])} exceeds the agent's auto-approval authority; director approval is required (POL-APR-03)")

    # --- amounts ---
    claimed, reimbursable = pd["total_claimed"], pd["total_reimbursable"]
    deducted_full = round(claimed - reimbursable, 2)
    lines = [f"{e['description']} ({_m(e['amount'])}) is ineligible [{e['category']}] under POL-CAT-02" for e in elig if e["status"] == "ineligible"]
    lines += [f"{r['category']} {_m(r['claimed'])} exceeds the {_m(r['cap'])} cap ({r['units']} x {_m(r['rate'])} per {r['unit']}) by {_m(r['excess'])} [{r['rule']}]" for r in pd["items"] if r["excess"] > 0]

    # --- decision ---
    if not live and not cons["conflicts"]:
        decision, approved, deducted = "REJECT", 0.0, claimed
        expl = f"No reimbursable items: {'; '.join(lines)}. The full {_m(claimed)} is deducted."
    elif manual:
        decision, approved, deducted = "MANUAL_REVIEW", 0.0, 0.0
        expl = f"Routed to Manual Review: {'; '.join(manual)}."
        expl += f" If the reviewer clears these, the indicative outcome is {_m(reimbursable)} approved and {_m(deducted_full)} deducted"
        expl += (f" ({'; '.join(lines)})." if lines else ".")
    elif deducted_full > 0:
        decision, approved, deducted = "PARTIAL_APPROVE", reimbursable, deducted_full
        expl = (f"The claim is valid but {_m(deducted_full)} is deducted: {'; '.join(lines)}. "
                f"The remaining {_m(approved)} falls in the {tier['tier_name']} tier ({tier['tier_id']}) and is approved.")
    else:
        decision, approved, deducted = "APPROVE", reimbursable, 0.0
        expl = (f"All {len(elig)} items are eligible (POL-CAT-01), receipts are present where required (POL-RCT-01) and amounts are within "
                f"per-diem caps. The total {_m(approved)} falls in the {tier['tier_name']} tier ({tier['tier_id']}), so it is approved in full.")

    return Assessment(decision, round(approved, 2), round(deducted, 2), reimbursable, deducted_full,
                      [m["label"] for m in rc["missing"]], sorted(refs),
                      compute_confidence(decision, signals, reimbursable), manual, expl)
