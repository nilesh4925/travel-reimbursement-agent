# Design Notes & Reasoning

## Approach in one paragraph
The policy is encoded as data (`data/policy_rules.json`, `limits.json`, `approval_matrix.json`) and exposed
through seven small deterministic tools. In LLM mode, an OpenAI model (function calling) decides which tools to call, reads the results, and
submits a decision and explanation. In offline mode, a fixed planner calls the same tools. In both modes the
numbers (caps, totals, tiers) come from tools, never from the model, and the final decision is cross-checked by
an independent policy engine and an output validator before it is returned.

## Key design choices and trade-offs
| Choice | Why | Trade-off |
|---|---|---|
| Deterministic tools do all arithmetic and rule checks | LLMs are unreliable at money math; results are auditable and repeatable | Policy changes mean editing data/code, not just a prompt |
| LLM (OpenAI function calling) chooses tools, combines results, writes the explanation | Demonstrates agentic reasoning; explanations read naturally | Slightly slower and costs API calls vs. pure rules |
| Verifier: LLM decision != engine decision -> `MANUAL_REVIEW` | A wrong auto-approval is the costly failure; humans are the safe fallback | May over-route if the model is conservative |
| Offline planner using the same tools | Runs with no key/network; makes demos and review reproducible | Not "agentic" - it is the control baseline |
| Keyword retrieval over 12 rules instead of a vector DB | At this size, retrieval is trivial; keeps setup light | Would not scale to a long policy document |
| Tools are bound to one claim (`ToolBox`) | The model can't mix data between claims; tools take no claim id | - |
| Pydantic output model + validator | Guarantees the exact output contract and internal consistency | - |
| Streamlit UI + static PNG | Minimal interactive UI, plus a screenshot-style artifact | Streamlit is one more dependency |

## Why certain cases go to Manual Review
* **CLM-004** - Three independent triggers. Business-class fare is an exception that must not be auto-deducted
  because a pre-approval may exist (POL-AIR-01). The lodging receipt is missing, so the item is not silently
  rejected (POL-RCT-02). The reimbursable total ($3,000) exceeds the agent's $2,000 authority (POL-APR-03).
* **CLM-005** - The $220 dinner has no receipt (POL-RCT-01/02). Also, the policy's $75/day cap doesn't say how it
  applies to a client dinner for four, so the agent flags an ambiguity instead of guessing it should deduct $145.
* General rule: missing required receipts, policy exceptions, unknown categories, late claims (POL-TIME-01),
  conflicting data, and totals above $2,000 always route to Manual Review. Ambiguity lowers confidence (0.75 vs 0.90).

## Assumptions
1. Per-diem caps scale by `units` on each line item (days for meals/ground transport, nights for lodging); `units`
   is part of the claim data (e.g. "2 nights" -> `units: 2`).
2. Approval tiers are evaluated on the total reimbursable amount after per-diem deductions, and include items that
   are pending review (so CLM-004's $3,000 hits POL-APR-03).
3. The "expense date" is the item's `expense_date` if given, else the trip start date (the conservative choice).
4. For `MANUAL_REVIEW`, no amount is finalised: `approved_amount = deducted_amount = 0`; the indicative amounts
   appear in the explanation. For the other decisions, `approved + deducted = total claimed`.
5. A claim with eligible items plus ineligible items (and nothing else wrong) is `PARTIAL_APPROVE`.
6. Group meals (`attendees > 1`) are treated as ambiguous because the policy is silent on them.
7. Business purpose is taken from the claim's `purpose` text; it is not independently verified.
8. All data is mock: no real employee or company data is used.

## Known gaps / limitations
* Receipts are mock metadata + text files; there is no OCR or image/PDF parsing.
* No duplicate-claim detection across claims, no currency conversion (USD only), no per-item travel dates beyond
  the optional `expense_date`.
* The LLM path was verified with a mocked client for the control flow (agreement, disagreement, API failure);
  run it with your own key to see real model behaviour.
* Single-claim, in-memory processing; no persistence or authentication.

## What I would improve next
* Parse real receipt PDFs/images and reconcile vendor/date/amount with line items.
* Duplicate and split-claim detection across a batch and history.
* Calibrated confidence (e.g. from evaluation data) instead of rule-based confidence.
* A larger labelled claim set and an evaluation script to measure agreement between LLM and engine.
* Reviewer workflow: request-missing-document messages, approver routing by tier, and an audit trail.
