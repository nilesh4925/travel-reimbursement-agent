from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["APPROVE", "PARTIAL_APPROVE", "REJECT", "MANUAL_REVIEW"]
DECISIONS = ["APPROVE", "PARTIAL_APPROVE", "REJECT", "MANUAL_REVIEW"]


class ClaimItem(BaseModel):
    category: str
    description: str
    amount: float = Field(gt=0)
    receipt_attached: bool
    units: int = Field(default=1, ge=1)      # days (meals/ground) or nights (lodging)
    attendees: int = Field(default=1, ge=1)  # people covered by a meal
    fare_class: Optional[str] = None         # airfare only
    expense_date: Optional[date] = None      # defaults to trip_start when absent


class Claim(BaseModel):
    claim_id: str
    employee: str
    purpose: str
    trip_start: date
    trip_end: date
    submitted_date: date
    currency: str = "USD"
    total_claimed: Optional[float] = None    # as stated on the form; cross-checked
    items: list[ClaimItem]

    @property
    def total(self) -> float:
        return round(sum(i.amount for i in self.items), 2)


class ClaimResult(BaseModel):
    """Exact output contract required by the assignment."""
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    decision: Decision
    approved_amount: float = Field(ge=0)
    deducted_amount: float = Field(ge=0)
    missing_docs: list[str]
    policy_refs: list[str]
    confidence: float = Field(ge=0, le=1)
    explanation: str
    tools_used: list[str]
