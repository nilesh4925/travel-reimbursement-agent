# Design Notes & Reasoning

## Approach in one paragraph
The policy is encoded as data (policy rules, a limit table and an approval matrix) and exposed through seven small
deterministic tools. In LLM mode, an OpenAI model (function calling) decides which tools to call, reads the results,
and submits a decision and explanation. In offline mode, a fixed planner calls the same tools. In both modes the
numbers (caps, totals, tiers) come from tools, never from the model, and the final decision is cross-checked by an
independent policy engine and an output validator before it is returned.

## Key design choices and trade-offs
| Choice | Why | Trade-off |
|---|---|---|
| Deterministic tools do all arithmetic and rule checks | LLMs are unreliable at money math; results are auditable and repeatable | Policy changes mean editing data/code, not just a prompt |
| LLM (OpenAI function calling) chooses tools, combines results, writes the explanation | Demonstrates agentic reasoning; explanations read naturally | Slower and costs API calls vs. pure rules |
| Verifier: LLM decision differs from engine decision -> `MANUAL_REVIEW` | A wrong auto-approval is the costly failure; humans are the safe fallback | May over-route if the model is conservative |
| Offline planner using the same tools | Runs with no key/network; reproducible demos and tests | Not "agentic" - it is the control baseline |
| Keyword retrieval over 12 rules instead of a vector DB | At this size retrieval is trivial; keeps setup light | Would not scale to a long policy document |
| Tools are bound to one claim (`ToolBox`) | The model cannot mix data between claims; tools take no claim id | - |
| Pydantic output model + validator | Guarantees the exact output contract and internal consistency | - |
| Fail fast on fatal API errors, then continue offline | One bad key/endpoint should not stall or crash the batch | Rest of the run is not LLM-driven |

## Why certain cases go to Manual Review
* **CLM-004** - Three independent triggers. Business-class fare is an exception that must not be auto-deducted
  because a pre-approval may exist (POL-AIR-01). The lodging receipt is missing, so the item is not silently
  rejected (POL-RCT-02). The reimbursable total ($3,000) exceeds the agent's $2,000 authority (POL-APR-03).
* **CLM-005** - The $220 dinner has no receipt (POL-RCT-01/02). Also, the $75/day cap does not say how it applies to a
  client dinner for four, so the agent flags an ambiguity instead of guessing it should deduct $145.
* General rule: missing required receipts, policy exceptions, unknown categories, late claims (POL-TIME-01),
  conflicting data, and totals above $2,000 always route to Manual Review.

## Evaluation, testing and confidence
* **Labelled evaluation suite** (25 cases): the 5 provided claims plus 20 edge cases that probe every boundary -
  exactly 30 vs 31 days, receipt at $25.00 vs $25.01, totals of $500 / $2,000 / $2,000.01, unknown categories,
  mismatched totals, receipt-amount conflicts, category aliases, ground-transport caps and more. Expected outcomes
  were written from the policy text, not from the program's output.
* **Automated tests** (`pytest`, 52 tests): each tool, the decision engine on all 25 cases, the exact output
  contract, validator guardrails, CSV vs JSON intake, and the OpenAI tool-calling loop with a mocked client
  (agreement, disagreement, garbage confidence, no submission, API failure). The notebook runs the same suite.
* **Confidence** is a transparent function of explicit signals rather than a fixed number per decision: a base by
  decision type, minus a penalty for each ambiguity signal (conflicting data, unknown category, group meal,
  unspecified airfare class), minus a small one when an auto-approval is within 5% of a tier boundary. The
  evaluation reports accuracy, mean confidence, Brier score and accuracy per confidence band. With 25 cases this
  is a sanity check, not true calibration - see "Known gaps".

## Assumptions
1. Per-diem caps scale by `units` on each line item (days for meals/ground transport, nights for lodging); `units`
   is part of the claim data (e.g. "2 nights" -> `units: 2`).
2. Approval tiers are evaluated on the total reimbursable amount after per-diem deductions, and include items that
   are pending review (so CLM-004's $3,000 hits POL-APR-03).
3. The "expense date" is the item's `expense_date` if given, else the trip start date (the conservative choice).
4. For `MANUAL_REVIEW`, no amount is finalised: `approved_amount = deducted_amount = 0`; the indicative amounts
   appear in the explanation. For other decisions, `approved + deducted = total claimed`.
5. A claim with eligible items plus ineligible items (and nothing else wrong) is `PARTIAL_APPROVE`.
6. Group meals (`attendees > 1`) are treated as ambiguous because the policy is silent on them.
7. Business purpose is taken from the claim's `purpose` text; it is not independently verified.
8. All data is mock: no real employee or company data is used.

## Known gaps / limitations
* Receipts are mock metadata; there is no OCR or image/PDF parsing.
* No duplicate-claim detection across claims, no currency conversion (USD only).
* Confidence is signal-based and checked on 25 labelled cases; real calibration needs a larger labelled set,
  ideally with human reviewer outcomes. The labelled cases are also written by the author, so agreement with the
  engine is partly self-consistency; the five provided claims are the independent anchor.
* Real-model behaviour is only exercised when an `OPENAI_API_KEY` is set. The committed notebook outputs and the
  automated tests use offline mode and a mocked client respectively.
* Single-claim, in-memory processing; no persistence or authentication.

## What I would improve next
* Parse real receipt PDFs/images and reconcile vendor/date/amount with line items.
* Duplicate and split-claim detection across a batch and history.
* Calibrate confidence on a larger labelled set with human reviewer outcomes.
* Run the evaluation suite against several OpenAI models and report agreement with the policy engine.
* Reviewer workflow: request-missing-document messages, approver routing by tier, and an audit trail.
