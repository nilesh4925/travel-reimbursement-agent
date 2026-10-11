# Travel Reimbursement Approval Agent

A small, runnable GenAI / agentic prototype that reviews employee travel-reimbursement claims against a
written policy, receipts, limits and approval rules, and returns a structured recommendation
(`APPROVE`, `PARTIAL_APPROVE`, `REJECT`, `MANUAL_REVIEW`).

* **LLM mode** - an OpenAI model runs a function-calling loop: it decides which policy/limit tools to call, combines their
  results and submits a decision + explanation.
* **Offline mode** - no API key needed; a deterministic planner runs the same tools in a fixed order.
* Every result is **validated** and **cross-checked** against the policy engine. Disagreement, invalid output,
  or an API failure ends in `MANUAL_REVIEW` / the offline planner - never a silently wrong decision.

## Project structure

```
travel-reimbursement-agent/
├── README.md
├── requirements.txt
├── .env.example                  # OPENAI_API_KEY (optional), OPENAI_MODEL, OPENAI_BASE_URL
├── run_demo.py                   # CLI: evaluate claims -> outputs/results.json + dashboard.png
├── app.py                        # Streamlit UI (intake, run, dashboard, per-claim decisions)
├── travel_agent/
│   ├── models.py                 # Pydantic: Claim, ClaimItem, ClaimResult (exact output contract)
│   ├── intake.py                 # JSON / CSV / pasted-JSON claim intake
│   ├── policy_store.py           # loads policy rules, limit table, approval matrix; keyword retrieval
│   ├── tools.py                  # the 7 agent tools + tool schemas + per-claim ToolBox
│   ├── engine.py                 # policy decision engine (offline planner + verifier of LLM output)
│   ├── agent.py                  # LLM tool-use loop, offline planner, fallbacks
│   ├── validator.py              # schema + business-consistency output validation
│   └── dashboard.py              # dashboard data + PNG
├── data/
│   ├── policy.md                 # Appendix A, human-readable
│   ├── policy_rules.json         # the same 12 POL-* rules, machine-readable (retrieval corpus)
│   ├── limits.json               # limit table: per-diem caps, receipt rules, categories, window
│   ├── approval_matrix.json      # POL-APR-01/02/03 tiers
│   ├── claims/CLM-001..005.json  # Appendix B sample claims (JSON intake)
│   ├── sample_claims.csv         # the same 5 claims in CSV intake format
│   └── receipts/                 # mock receipt metadata (CLM-00x.json) + mock receipt text files
├── docs/DESIGN_NOTES.md          # Design Notes & Reasoning, assumptions, limitations, next steps
└── outputs/
    ├── results.json              # sample output (final JSON array for the 5 claims)
    ├── dashboard.png             # results dashboard  (copy: "UI SS_1.png")
    └── UI SS_1.png
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # optional: add OPENAI_API_KEY for LLM mode
```

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | No | - | Enables LLM mode. Without it the agent runs offline. |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Any OpenAI chat model that supports function calling (e.g. `gpt-4o`, `gpt-4.1`). |
| `OPENAI_BASE_URL` | No | OpenAI default | Optional: any OpenAI-compatible endpoint, e.g. a local Ollama server (`http://localhost:11434/v1`). |
| `MAX_AGENT_TURNS` | No | `12` | Safety cap on tool-use turns per claim. |

## Run

```bash
python run_demo.py                      # auto: LLM if key set, else offline
python run_demo.py --mode offline       # force offline (no key needed)
python run_demo.py --mode llm           # force LLM mode (fails fast if no key)
python run_demo.py --claims data/sample_claims.csv     # CSV intake
python run_demo.py --claims path/to/my_claim.json      # your own claim(s)

streamlit run app.py                    # UI: pick/upload/paste claims -> Evaluate -> dashboard
```

`run_demo.py` prints the final **JSON array** (one object per claim) to stdout and writes
`outputs/results.json` and `outputs/dashboard.png`.

### Output contract (exactly these fields per claim)
`claim_id, decision, approved_amount, deducted_amount, missing_docs, policy_refs, confidence, explanation, tools_used`

For `MANUAL_REVIEW`, no amount is finalised (`approved_amount` and `deducted_amount` are `0.0`); the explanation
states the indicative outcome if the reviewer clears the issue.

## Results on the five sample claims

| Claim | Decision | Approved | Deducted | Why |
|---|---|---:|---:|---|
| CLM-001 | APPROVE | $1,110.00 | $0.00 | All eligible, receipts present, within caps; Manager tier (POL-APR-02) |
| CLM-002 | REJECT | $0.00 | $380.00 | Spa + minibar are ineligible (POL-CAT-02) |
| CLM-003 | PARTIAL_APPROVE | $840.00 | $100.00 | Lodging $500 vs $400 cap (2 nights x $200) (POL-PD-02) |
| CLM-004 | MANUAL_REVIEW | $0.00 | $0.00 | Business-class airfare (POL-AIR-01), missing lodging receipt (POL-RCT-02), total > $2,000 (POL-APR-03) |
| CLM-005 | MANUAL_REVIEW | $0.00 | $0.00 | Missing receipt > $25 (POL-RCT-02) and a 4-person dinner the per-diem policy doesn't address |

## How the agent works

```
claim (JSON/CSV/UI) -> intake -> Agent
   LLM mode:  OpenAI model <-> tools (lookup_policy, check_claim_consistency, check_category_eligibility,
                              check_receipts, check_per_diem_limits, check_timeliness, check_approval_threshold)
              -> submit_decision(decision, confidence, explanation)
   Offline:   same tools, fixed order
        -> policy engine cross-check (disagreement => MANUAL_REVIEW)
        -> output validator (invalid => MANUAL_REVIEW) -> ClaimResult
```

See [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md) for assumptions, trade-offs, why cases go to Manual Review,
known gaps and next steps.
