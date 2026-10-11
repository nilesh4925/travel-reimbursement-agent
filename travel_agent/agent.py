"""Travel Reimbursement Approval Agent.

LLM mode   : an OpenAI model drives a function-calling loop (it chooses which tools to call, combines the results, and
             submits a decision + explanation). The decision is then cross-checked against the policy
             engine; any disagreement is routed to MANUAL_REVIEW.
Offline mode: no API key needed - a deterministic planner calls the same tools in a fixed order.
Both modes pass the result through the output validator; failures fall back to MANUAL_REVIEW.
"""
from __future__ import annotations

import json
import sys

from . import config, engine
from .models import Claim, ClaimResult
from .policy_store import RULES
from .tools import TOOL_SCHEMAS, ToolBox
from .validator import validate_result

SYSTEM_PROMPT = """You are a Travel Reimbursement Approval Agent for a company. You evaluate ONE employee claim at a time against the company travel policy and return a recommendation.

Process:
1. Call lookup_policy to ground yourself in the relevant POL-* rules.
2. Call the check_* tools you need (normally all of them). check_approval_threshold needs total_reimbursable from check_per_diem_limits.
3. Combine the tool results. NEVER do your own arithmetic when a tool gives you the number; never invent policy rules.
4. Call submit_decision exactly once with your decision, confidence and explanation.

Decision guidance:
- APPROVE: every item eligible, all required receipts present, all within per-diem, total in an approvable tier (POL-APR-01/02).
- PARTIAL_APPROVE: claim is valid but some amounts exceed per-diem caps (and/or some items are ineligible alongside eligible ones); reimburse up to the cap, deduct the rest.
- REJECT: every item is ineligible (POL-CAT-02); nothing reimbursable.
- MANUAL_REVIEW: any ambiguity, policy exception (e.g. non-economy airfare), total above $2,000 (POL-APR-03), a missing required receipt (POL-RCT-02), a late claim, or conflicting information. Prefer MANUAL_REVIEW over forcing a decision.

Explanation: 2-4 sentences, plain English, cite POL-* ids and use the amounts returned by the tools. confidence is 0-1 and should be lower when the case is ambiguous."""

SUBMIT_TOOL = {
    "name": "submit_decision",
    "description": "Submit the final recommendation for this claim. Call exactly once, after the checks.",
    "input_schema": {"type": "object", "properties": {
        "decision": {"type": "string", "enum": ["APPROVE", "PARTIAL_APPROVE", "REJECT", "MANUAL_REVIEW"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "explanation": {"type": "string"}},
        "required": ["decision", "confidence", "explanation"]},
}

# Tool schemas are stored provider-neutral; convert to OpenAI's function-calling format.
OPENAI_TOOLS = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
                for t in TOOL_SCHEMAS + [SUBMIT_TOOL]]


class AgentError(Exception):
    pass


class TravelReimbursementAgent:
    def __init__(self, mode: str = "auto"):
        self.client = None
        if mode in ("auto", "llm") and config.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)
            except ImportError:
                pass
        if mode == "llm" and self.client is None:
            raise RuntimeError("mode 'llm' needs OPENAI_API_KEY and the `openai` package")
        self.mode = "llm" if self.client else "offline"

    # ---------- public ----------
    def evaluate_all(self, claims: list[Claim]) -> list[ClaimResult]:
        out = []
        for c in claims:
            print(f"  evaluating {c.claim_id} ...", file=sys.stderr)
            out.append(self.evaluate(c))
        return out

    def evaluate(self, claim: Claim) -> ClaimResult:
        if self.client:
            try:
                return self._evaluate_llm(claim)
            except Exception as exc:  # network/API/format problems -> deterministic fallback
                print(f"  ! LLM path failed for {claim.claim_id} ({type(exc).__name__}: {exc}); using offline planner", file=sys.stderr)
        return self._evaluate_offline(claim)

    # ---------- offline planner ----------
    def _evaluate_offline(self, claim: Claim) -> ClaimResult:
        tb = ToolBox(claim)
        cats = " ".join(sorted({i.category for i in claim.items}))
        tb.call("lookup_policy", {"query": f"{cats} eligible receipt per diem approval threshold submission window"})
        for name in ("check_claim_consistency", "check_category_eligibility", "check_receipts", "check_per_diem_limits", "check_timeliness"):
            tb.call(name)
        tb.call("check_approval_threshold", {"total_amount": tb.results["check_per_diem_limits"]["total_reimbursable"]})
        ref = engine.decide(claim, tb.results)
        return self._finalize(claim, ref, ref.decision, ref.explanation, ref.confidence, tb.calls)

    # ---------- LLM tool-use loop ----------
    def _evaluate_llm(self, claim: Claim) -> ClaimResult:
        tb = ToolBox(claim)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Evaluate this travel reimbursement claim:\n" + claim.model_dump_json(indent=2)}]
        submission, nudged = None, False
        for _ in range(config.MAX_AGENT_TURNS):
            resp = self.client.chat.completions.create(model=config.OPENAI_MODEL, messages=messages,
                                                       tools=OPENAI_TOOLS, tool_choice="auto", temperature=0)
            msg = resp.choices[0].message
            messages.append(msg)
            if not msg.tool_calls:
                if submission or nudged:
                    break
                nudged = True
                messages.append({"role": "user", "content": "Please call submit_decision now."})
                continue
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if tc.function.name == "submit_decision":
                    submission = args
                    payload = {"status": "recorded"}
                else:
                    payload = tb.call(tc.function.name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(payload)})
            if submission:
                break
        if not submission or submission.get("decision") not in ("APPROVE", "PARTIAL_APPROVE", "REJECT", "MANUAL_REVIEW"):
            raise AgentError("model did not submit a valid decision")

        # Independent verification against the policy engine
        ref = engine.decide(claim, engine.run_all_checks(claim))
        llm_decision = submission["decision"]
        try:
            conf = min(1.0, max(0.0, float(submission.get("confidence", ref.confidence))))
        except (TypeError, ValueError):
            conf = ref.confidence
        explanation = str(submission.get("explanation", "")).strip() or ref.explanation
        decision = llm_decision
        if llm_decision != ref.decision:
            decision = "MANUAL_REVIEW"
            conf = min(conf, 0.5)
            explanation = (f"Agent proposed {llm_decision} but the policy checks indicate {ref.decision}; routed to Manual Review "
                           f"for human confirmation. Policy-check summary: {ref.explanation}")
        return self._finalize(claim, ref, decision, explanation, conf, tb.calls)

    # ---------- shared ----------
    def _finalize(self, claim: Claim, ref: engine.Assessment, decision: str, explanation: str, confidence: float, tools: list[str]) -> ClaimResult:
        manual = decision == "MANUAL_REVIEW"
        payload = {
            "claim_id": claim.claim_id,
            "decision": decision,
            "approved_amount": 0.0 if manual else ref.approved,
            "deducted_amount": 0.0 if manual else ref.deducted,
            "missing_docs": ref.missing_docs,
            "policy_refs": [p for p in ref.policy_refs if p in RULES],
            "confidence": round(confidence, 2),
            "explanation": explanation,
            "tools_used": list(tools) + ["validate_output"],
        }
        errors = validate_result(payload, claim)
        if errors:  # reliability fallback: never emit an invalid or inconsistent result
            payload.update(decision="MANUAL_REVIEW", approved_amount=0.0, deducted_amount=0.0, confidence=0.3,
                           explanation=f"Automated output failed validation ({'; '.join(errors)}); routed to Manual Review. {ref.explanation}")
        return ClaimResult(**payload)
