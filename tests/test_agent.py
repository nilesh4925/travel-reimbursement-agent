"""Automated tests:  pytest -q

Covers every tool, the decision engine on the 5 provided claims + 15 edge cases, the output contract,
validator guardrails, CSV intake, and the OpenAI tool-calling loop (mocked - no network or key needed).
"""
import json
from datetime import date
from types import SimpleNamespace as NS

import pytest

from travel_agent import config
from travel_agent.agent import OPENAI_TOOLS, TravelReimbursementAgent
from travel_agent.engine import compute_confidence
from travel_agent.evaluation import confidence_bands, load_eval_cases, run_evaluation, summarize_evaluation
from travel_agent.intake import load_claims
from travel_agent.models import Claim, ClaimResult
from travel_agent.policy_store import RULES, search_policy
from travel_agent.tools import (check_approval_threshold, check_category_eligibility, check_per_diem_limits,
                                check_receipts, check_timeliness)
from travel_agent.validator import validate_result

CASES = load_eval_cases(config.DATA_DIR / "eval_cases.json")
CLAIMS = {c.claim_id: c for c in load_claims(config.DATA_DIR / "claims")}
OUTPUT_FIELDS = ["claim_id", "decision", "approved_amount", "deducted_amount", "missing_docs", "policy_refs", "confidence", "explanation", "tools_used"]


@pytest.fixture(scope="module")
def offline():
    return TravelReimbursementAgent(mode="offline")


def make_claim(items, sub="2026-06-12", **kw):
    base = dict(claim_id="T-1", employee="T", purpose="t", trip_start="2026-06-10", trip_end="2026-06-10", submitted_date=sub, items=items)
    return Claim(**{**base, **kw})


def it(cat, amt, rec=True, **kw):
    return dict(category=cat, description=cat, amount=amt, receipt_attached=rec, **kw)


# ---------- labelled cases: 5 provided claims + edge cases ----------
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_labelled_case(offline, case):
    r = offline.evaluate(Claim(**case["claim"]))
    exp = case["expected"]
    assert r.decision == exp["decision"]
    assert r.approved_amount == pytest.approx(exp["approved_amount"])
    assert r.deducted_amount == pytest.approx(exp["deducted_amount"])
    assert set(exp.get("policy_refs_include", [])) <= set(r.policy_refs)


def test_five_provided_claims_summary(offline):
    got = {c.claim_id: offline.evaluate(c).decision for c in CLAIMS.values()}
    assert got == {"CLM-001": "APPROVE", "CLM-002": "REJECT", "CLM-003": "PARTIAL_APPROVE", "CLM-004": "MANUAL_REVIEW", "CLM-005": "MANUAL_REVIEW"}


# ---------- output contract ----------
def test_output_contract_exact_fields(offline):
    for c in CLAIMS.values():
        r = offline.evaluate(c)
        assert list(r.model_dump().keys()) == OUTPUT_FIELDS
        assert validate_result(r.model_dump(), c) == []
        assert 0 <= r.confidence <= 1 and len(r.tools_used) >= 2
        assert set(r.policy_refs) <= set(RULES)


def test_amounts_reconcile_for_non_manual(offline):
    for case in CASES:
        c = Claim(**case["claim"]); r = offline.evaluate(c)
        if r.decision != "MANUAL_REVIEW":
            assert r.approved_amount + r.deducted_amount == pytest.approx(c.total)
        else:
            assert r.approved_amount == 0 and r.deducted_amount == 0


def test_missing_docs_reported(offline):
    assert any("lodging" in d for d in offline.evaluate(CLAIMS["CLM-004"]).missing_docs)
    assert any("meals" in d for d in offline.evaluate(CLAIMS["CLM-005"]).missing_docs)
    assert offline.evaluate(CLAIMS["CLM-001"]).missing_docs == []


# ---------- individual tools ----------
def test_per_diem_tool():
    out = check_per_diem_limits(CLAIMS["CLM-003"])
    lodging = next(r for r in out["items"] if r["category"] == "lodging")
    assert (lodging["cap"], lodging["excess"], lodging["allowed"]) == (400, 100, 400)
    assert out["total_reimbursable"] == 840 and out["per_diem_excess_total"] == 100


def test_receipt_tool_thresholds():
    assert check_receipts(make_claim([it("meals", 25, rec=False)]))["missing"] == []
    assert len(check_receipts(make_claim([it("meals", 25.01, rec=False)]))["missing"]) == 1
    assert len(check_receipts(make_claim([it("lodging", 10, rec=False)]))["missing"]) == 1  # always required
    assert check_receipts(make_claim([it("spa", 300, rec=False)]))["missing"] == []          # ineligible: no receipt needed


def test_timeliness_boundary():
    assert check_timeliness(make_claim([it("meals", 10)], sub="2026-07-10"))["late"] is False
    assert check_timeliness(make_claim([it("meals", 10)], sub="2026-07-11"))["late"] is True


@pytest.mark.parametrize("total,tier", [(0.01, "POL-APR-01"), (500, "POL-APR-01"), (500.01, "POL-APR-02"), (2000, "POL-APR-02"), (2000.01, "POL-APR-03")])
def test_approval_threshold_tiers(total, tier):
    assert check_approval_threshold(total)["tier_id"] == tier


def test_category_eligibility_flags():
    out = check_category_eligibility(CLAIMS["CLM-004"])["items"][0]
    assert "AIRFARE_NOT_ECONOMY" in out["flags"]
    assert check_category_eligibility(CLAIMS["CLM-002"])["ineligible_total"] == 380


def test_policy_retrieval():
    assert search_policy("minibar alcohol")[0]["id"] == "POL-CAT-02"
    assert search_policy("missing receipt")[0]["id"] in ("POL-RCT-02", "POL-RCT-01")
    assert search_policy("show me POL-APR-03")[0]["id"] == "POL-APR-03"


# ---------- confidence ----------
def test_confidence_lower_when_ambiguous():
    assert compute_confidence("MANUAL_REVIEW", ["GROUP_MEAL"], 75) < compute_confidence("MANUAL_REVIEW", [], 75)
    assert compute_confidence("APPROVE", [], 495) < compute_confidence("APPROVE", [], 300)  # near $500 boundary


def test_evaluation_summary_and_bands(offline):
    df = run_evaluation(offline, CASES)
    s = summarize_evaluation(df)
    assert s["accuracy"] == 1.0 and s["cases"] == len(CASES) and s["brier_score"] < 0.05
    assert not confidence_bands(df).empty


# ---------- validator guardrails ----------
def good_payload(offline):
    return offline.evaluate(CLAIMS["CLM-003"]).model_dump()


def test_validator_catches_bad_outputs(offline):
    c = CLAIMS["CLM-003"]
    p = good_payload(offline)
    assert validate_result(p, c) == []
    assert validate_result({**p, "decision": "MAYBE"}, c)                         # not an allowed decision
    assert validate_result({**p, "approved_amount": 900.0}, c)                    # amounts don't add up
    assert validate_result({**p, "policy_refs": ["POL-FAKE-99"]}, c)              # invented rule id
    assert validate_result({**p, "confidence": 1.4}, c)                           # out of range
    assert validate_result({**p, "extra_field": 1}, c)                            # exact fields only
    assert validate_result({**p, "decision": "APPROVE"}, c)                       # APPROVE with deductions
    assert validate_result({**p, "decision": "MANUAL_REVIEW"}, c)                 # manual must not finalise amounts


# ---------- intake ----------
def test_csv_intake_matches_json(offline):
    from_csv = {c.claim_id: c for c in load_claims(config.DATA_DIR / "sample_claims.csv")}
    assert set(from_csv) == set(CLAIMS)
    for cid in CLAIMS:
        assert offline.evaluate(from_csv[cid]).model_dump() == offline.evaluate(CLAIMS[cid]).model_dump()


# ---------- LLM (OpenAI) tool-calling loop with a mocked client ----------
def _tc(i, name, args):
    return NS(id=i, function=NS(name=name, arguments=json.dumps(args)))


def _mock_client(final, first_calls=None):
    state = {"n": 0, "tool_msgs": 0}

    def create(**kw):
        state["n"] += 1
        if state["n"] == 1:
            calls = first_calls or [_tc("a", "lookup_policy", {"query": "lodging"}), _tc("b", "check_per_diem_limits", {}), _tc("c", "check_receipts", {})]
        else:
            state["tool_msgs"] = sum(1 for m in kw["messages"] if isinstance(m, dict) and m.get("role") == "tool")
            calls = [_tc("z", "submit_decision", final)]
        return NS(choices=[NS(message=NS(tool_calls=calls, content=None))])

    return NS(chat=NS(completions=NS(create=create))), state


def _llm_agent(client):
    a = TravelReimbursementAgent(mode="offline")
    a.client, a.mode = client, "llm"
    return a


def test_openai_tools_schema():
    names = [t["function"]["name"] for t in OPENAI_TOOLS]
    assert "submit_decision" in names and len(names) == 8 and all(t["type"] == "function" for t in OPENAI_TOOLS)


def test_llm_agreeing_decision_is_kept():
    client, state = _mock_client(dict(decision="PARTIAL_APPROVE", confidence=0.93, explanation="Lodging exceeds cap."))
    r = _llm_agent(client).evaluate(CLAIMS["CLM-003"])
    assert r.decision == "PARTIAL_APPROVE" and r.explanation == "Lodging exceeds cap." and r.approved_amount == 840
    assert r.tools_used == ["lookup_policy", "check_per_diem_limits", "check_receipts", "validate_output"]
    assert state["tool_msgs"] == 3  # tool results were fed back to the model


def test_llm_disagreement_goes_to_manual_review():
    client, _ = _mock_client(dict(decision="REJECT", confidence=0.99, explanation="wrong"))
    r = _llm_agent(client).evaluate(CLAIMS["CLM-001"])
    assert r.decision == "MANUAL_REVIEW" and r.confidence <= 0.5 and r.approved_amount == 0


def test_llm_cannot_auto_approve_a_policy_exception():
    client, _ = _mock_client(dict(decision="APPROVE", confidence=0.95, explanation="looks fine"))
    assert _llm_agent(client).evaluate(CLAIMS["CLM-004"]).decision == "MANUAL_REVIEW"


def test_llm_garbage_confidence_is_handled():
    client, _ = _mock_client(dict(decision="APPROVE", confidence="high", explanation="ok"))
    r = _llm_agent(client).evaluate(CLAIMS["CLM-001"])
    assert r.decision == "APPROVE" and 0 <= r.confidence <= 1


def test_api_failure_falls_back_to_offline_and_stops_retrying():
    class APIConnectionError(Exception):
        pass

    calls = {"n": 0}

    def boom(**kw):
        calls["n"] += 1
        raise APIConnectionError("Connection error.")

    agent = _llm_agent(NS(chat=NS(completions=NS(create=boom))))
    results = agent.evaluate_all(list(CLAIMS.values()))
    assert calls["n"] == 1 and agent.mode == "offline"
    assert [r.decision for r in results] == ["APPROVE", "REJECT", "PARTIAL_APPROVE", "MANUAL_REVIEW", "MANUAL_REVIEW"]


def test_llm_with_no_submission_falls_back():
    state = {"n": 0}

    def create(**kw):
        state["n"] += 1
        return NS(choices=[NS(message=NS(tool_calls=None, content="I think it is fine."))])

    r = _llm_agent(NS(chat=NS(completions=NS(create=create)))).evaluate(CLAIMS["CLM-002"])
    assert r.decision == "REJECT"  # offline planner result


# ---------- evaluation harness: progress + parallel execution ----------
def test_parallel_evaluation_matches_sequential(offline):
    seq = run_evaluation(offline, CASES)
    par = run_evaluation(offline, CASES, workers=4)
    cols = ["case", "got", "approved", "deducted", "confidence", "passed"]
    assert seq[cols].equals(par[cols])  # same results, same order


def test_evaluation_progress_output(offline, capsys):
    run_evaluation(offline, CASES[:3], progress=True)
    out = capsys.readouterr().out
    assert "[3/3]" in out and out.count("ok") == 3
