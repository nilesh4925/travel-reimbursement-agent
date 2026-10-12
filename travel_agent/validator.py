"""Output validator: schema + business-consistency checks on every result before it is returned."""
from __future__ import annotations

from pydantic import ValidationError

from .models import Claim, ClaimResult
from .policy_store import RULES


def validate_result(payload: dict, claim: Claim) -> list[str]:
    try:
        r = ClaimResult(**payload)
    except ValidationError as exc:
        return [f"schema: {'.'.join(map(str, e['loc']))} {e['msg']}" for e in exc.errors()]
    errs = []
    if r.claim_id != claim.claim_id:
        errs.append("claim_id does not match the claim")
    errs += [f"unknown policy ref {p}" for p in r.policy_refs if p not in RULES]
    if not r.tools_used:
        errs.append("tools_used is empty")
    if not r.explanation.strip():
        errs.append("explanation is empty")
    if r.decision == "MANUAL_REVIEW":
        if r.approved_amount or r.deducted_amount:
            errs.append("MANUAL_REVIEW must not finalise any amount")
    else:
        if abs(r.approved_amount + r.deducted_amount - claim.total) > 0.01:
            errs.append("approved + deducted must equal the total claimed")
        if r.decision == "APPROVE" and r.deducted_amount > 0:
            errs.append("APPROVE cannot have deductions")
        if r.decision == "REJECT" and r.approved_amount > 0:
            errs.append("REJECT cannot have an approved amount")
        if r.decision == "PARTIAL_APPROVE" and not (r.approved_amount > 0 and r.deducted_amount > 0):
            errs.append("PARTIAL_APPROVE needs both approved and deducted amounts")
    return errs
