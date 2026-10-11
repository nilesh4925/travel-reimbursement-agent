"""Results dashboard: data summary (used by the Streamlit UI) and a static PNG."""
from __future__ import annotations

import pandas as pd

from .models import DECISIONS, Claim, ClaimResult

COLORS = {"APPROVE": "#2e7d32", "PARTIAL_APPROVE": "#f9a825", "REJECT": "#c62828", "MANUAL_REVIEW": "#1565c0"}


def summarize(results: list[ClaimResult], claims: list[Claim]) -> tuple[pd.DataFrame, pd.Series]:
    totals = {c.claim_id: c.total for c in claims}
    df = pd.DataFrame([{
        "claim_id": r.claim_id, "decision": r.decision, "claimed": totals[r.claim_id],
        "approved": r.approved_amount, "deducted": r.deducted_amount,
        "pending_review": round(totals[r.claim_id] - r.approved_amount - r.deducted_amount, 2),
        "confidence": r.confidence,
    } for r in results])
    counts = df["decision"].value_counts().reindex(DECISIONS, fill_value=0)
    return df, counts


def save_png(results: list[ClaimResult], claims: list[Claim], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df, counts = summarize(results, claims)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.bar(counts.index, counts.values, color=[COLORS[d] for d in counts.index])
    a1.set_title("Decision breakdown")
    a1.set_ylabel("Claims")
    a1.set_yticks(range(0, int(counts.max()) + 2))
    a1.tick_params(axis="x", rotation=20)
    bottom = [0.0] * len(df)
    for col, color, label in (("approved", "#2e7d32", "Approved"), ("deducted", "#c62828", "Deducted"), ("pending_review", "#1565c0", "Pending manual review")):
        a2.bar(df["claim_id"], df[col], bottom=bottom, color=color, label=label)
        bottom = [b + v for b, v in zip(bottom, df[col])]
    a2.set_title("Claimed amount by outcome (USD)")
    a2.legend()
    fig.suptitle(f"Travel Reimbursement Agent - {len(df)} claims | approved ${df['approved'].sum():,.2f} | "
                 f"deducted ${df['deducted'].sum():,.2f} | pending ${df['pending_review'].sum():,.2f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
