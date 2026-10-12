"""Evaluation harness: run labelled cases through an agent and measure correctness + confidence quality."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .models import Claim


def load_eval_cases(path) -> list[dict]:
    return json.loads(Path(path).read_text())


def _evaluate_case(agent, case: dict) -> dict:
    claim, exp = Claim(**case["claim"]), case["expected"]
    r = agent.evaluate(claim)
    decision_ok = r.decision == exp["decision"]
    amounts_ok = abs(r.approved_amount - exp["approved_amount"]) < 0.01 and abs(r.deducted_amount - exp["deducted_amount"]) < 0.01
    refs_ok = set(exp.get("policy_refs_include", [])) <= set(r.policy_refs)
    return {
        "case": case["name"], "expected": exp["decision"], "got": r.decision,
        "approved": r.approved_amount, "deducted": r.deducted_amount, "confidence": r.confidence,
        "decision_ok": decision_ok, "amounts_ok": amounts_ok, "refs_ok": refs_ok,
        "passed": decision_ok and amounts_ok and refs_ok,
    }


def run_evaluation(agent, cases: list[dict], workers: int = 1, progress: bool = False) -> pd.DataFrame:
    """Each case: {name, description, claim, expected: {decision, approved_amount, deducted_amount, policy_refs_include}}.

    workers > 1 evaluates cases in parallel threads (useful for LLM mode); row order is preserved.
    progress=True prints one line per finished case (with seconds taken) so long LLM runs are visibly alive.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows: list = [None] * len(cases)

    def work(i: int):
        t0 = time.time()
        row = _evaluate_case(agent, cases[i])
        row["seconds"] = round(time.time() - t0, 1)
        return i, row

    def report(done: int, row: dict) -> None:
        if progress:
            print(f"  [{done}/{len(cases)}] {row['case']}: {row['got']} ({row['seconds']}s) {'ok' if row['passed'] else 'FAIL (expected ' + row['expected'] + ')'}", flush=True)

    if workers <= 1:
        for n, i in enumerate(range(len(cases)), 1):
            _, rows[i] = work(i)
            report(n, rows[i])
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for n, fut in enumerate(as_completed([pool.submit(work, i) for i in range(len(cases))]), 1):
                i, row = fut.result()
                rows[i] = row
                report(n, row)
    return pd.DataFrame(rows)


def summarize_evaluation(df: pd.DataFrame) -> dict:
    """Accuracy plus a simple confidence check: Brier score and mean confidence vs observed accuracy."""
    correct = df["passed"].astype(float)
    return {
        "cases": len(df),
        "passed": int(df["passed"].sum()),
        "accuracy": round(float(correct.mean()), 3),
        "mean_confidence": round(float(df["confidence"].mean()), 3),
        "brier_score": round(float(((df["confidence"] - correct) ** 2).mean()), 4),  # 0 = perfect, lower is better
        "manual_review_precision": _precision(df),
    }


def _precision(df: pd.DataFrame) -> float | None:
    routed = df[df["got"] == "MANUAL_REVIEW"]
    return None if routed.empty else round(float((routed["expected"] == "MANUAL_REVIEW").mean()), 3)


def confidence_bands(df: pd.DataFrame) -> pd.DataFrame:
    bands = pd.cut(df["confidence"], [0, 0.7, 0.85, 0.95, 1.0], labels=["<=0.70", "0.70-0.85", "0.85-0.95", ">0.95"])
    out = df.groupby(bands, observed=True).agg(cases=("passed", "size"), accuracy=("passed", "mean"), mean_confidence=("confidence", "mean"))
    return out.round(3)
