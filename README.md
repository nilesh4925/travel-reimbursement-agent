# Travel Reimbursement Approval Agent

A runnable GenAI / agentic prototype that reviews employee travel-reimbursement claims against a written policy,
receipts, limits and approval rules, and returns a structured recommendation
(`APPROVE`, `PARTIAL_APPROVE`, `REJECT`, `MANUAL_REVIEW`).

## The deliverable: `nilesh.ipynb`
**Rename it to `<namesurname>.ipynb` before submitting.** The notebook is the single, self-contained deliverable:
setup/README, embedded policy + sample claims + receipts, tools, agent, sample outputs, **Dashboard** (charts +
interactive UI), **Evaluation & tests**, **Design Notes & Reasoning**, and a final cell that prints the JSON array
for the five provided claims. Saved outputs are included (offline mode); add an `OPENAI_API_KEY` to run LLM mode.

```bash
pip install -r requirements.txt jupyter
jupyter notebook nilesh.ipynb        # Kernel -> Restart & Run All
```

The rest of the repository is the same code organised as a normal Python project (CLI, Streamlit UI, tests).
`nilesh.ipynb` is generated from it by `python build_notebook.py`, so the two never drift apart.

## How it works
```
claim (JSON/CSV/UI) -> intake -> Agent
   LLM mode : OpenAI model <-> tools (lookup_policy, check_claim_consistency, check_category_eligibility,
              check_receipts, check_per_diem_limits, check_timeliness, check_approval_threshold)
              -> submit_decision(decision, confidence, explanation)
   Offline  : same tools, fixed order (no API key needed)
        -> policy-engine cross-check (disagreement => MANUAL_REVIEW)
        -> output validator (invalid => MANUAL_REVIEW) -> ClaimResult
```
Tools do all arithmetic; the LLM chooses tools and explains. See [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md).

## Results on the five provided claims

| Claim | Decision | Approved | Deducted | Why |
|---|---|---:|---:|---|
| CLM-001 | APPROVE | $1,110.00 | $0.00 | All eligible, receipts present, within caps; Manager tier (POL-APR-02) |
| CLM-002 | REJECT | $0.00 | $380.00 | Spa + minibar are ineligible (POL-CAT-02) |
| CLM-003 | PARTIAL_APPROVE | $840.00 | $100.00 | Lodging $500 vs $400 cap (2 nights x $200) (POL-PD-02) |
| CLM-004 | MANUAL_REVIEW | $0.00 | $0.00 | Business-class airfare (POL-AIR-01), missing lodging receipt (POL-RCT-02), total > $2,000 (POL-APR-03) |
| CLM-005 | MANUAL_REVIEW | $0.00 | $0.00 | Missing receipt > $25 (POL-RCT-02); 4-person dinner the per-diem policy doesn't address |

For `MANUAL_REVIEW` no amount is finalised (`approved_amount` and `deducted_amount` are `0.0`); the explanation states
the indicative outcome if the reviewer clears the issue.

## Tests and evaluation
Run these from the project root, with the virtual environment active:
```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q                          # 52 tests; no API key or internet needed (expect "52 passed")
python -m pytest -q                # same, if plain `pytest` cannot import travel_agent (pytest.ini fixes this)
python evaluate.py                 # labelled evaluation, OFFLINE baseline: pass/fail table, accuracy, Brier score, confidence bands
python evaluate.py --mode llm      # same 25 cases against the OpenAI agent (needs OPENAI_API_KEY); prints live progress
python evaluate.py --mode llm --limit 5      # quick smoke test: only the first 5 cases
python evaluate.py --mode llm --workers 4    # run 4 cases in parallel (much faster)
python evaluate.py --only CLM-004            # only cases whose name contains this text
python evaluate.py --mode auto     # LLM if OPENAI_API_KEY is set, otherwise offline
```
**Which mode is used?** `evaluate.py` defaults to `--mode offline` on purpose: the baseline is deterministic and
needs no key, so having `OPENAI_API_KEY` in `.env` does **not** switch it to the LLM. The first line of the output
shows the mode (`[mode: offline]` or `[mode: llm]`). In LLM mode, any case where the model disagrees with the policy
engine appears as `MANUAL_REVIEW` and is marked as failed - those are the cases worth reading. (`run_demo.py`,
by contrast, defaults to `--mode auto`.) `pytest` and the notebook's evaluation cell always use the offline baseline
and a mocked OpenAI client, so they stay fast and reproducible.

**LLM runs look slow?** Each case is a multi-step conversation with the model (typically 5-20 seconds), so the full
suite takes a few minutes sequentially. `evaluate.py` prints the model, endpoint and one line per finished case, so you can
see it working. If the first case takes more than ~1 minute or prints `! LLM path failed`, run `python openai_check.py`
to check the key, model and network; use `--limit 3` to test quickly and `--workers 4` for the full run.

Useful pytest variations: `pytest -v`, `pytest -k "CLM-004"`, `pytest -x` (stop at first failure).
`evaluate.py` also writes `outputs/evaluation_report.csv` and exits non-zero if any case fails.

The 25 labelled cases (`data/eval_cases.json`) are the 5 provided claims plus 20 edge cases around every policy
boundary (30 vs 31 days, $25.00 vs $25.01, $500 / $2,000 / $2,000.01, unknown categories, conflicting data, ...).

## Run as a normal project
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # optional: add OPENAI_API_KEY for LLM mode
python openai_check.py                                 # verify your .env key/model/endpoint
python run_demo.py                                     # auto: LLM if key set, else offline -> outputs/
python run_demo.py --mode offline | --mode llm
python run_demo.py --claims data/sample_claims.csv     # CSV intake
streamlit run app.py                                   # Streamlit UI
```

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | No | - | Enables LLM mode. Without it the agent runs offline. |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Any OpenAI chat model that supports function calling. |
| `OPENAI_BASE_URL` | No | OpenAI default | Optional OpenAI-compatible endpoint. Use the base URL only (e.g. `https://api.openai.com/v1`), never a full path like `.../models`. |
| `MAX_AGENT_TURNS` | No | `12` | Safety cap on tool-calling turns per claim. |

## Project structure
```
├── nilesh.ipynb              # THE deliverable (self-contained, outputs saved)
├── build_notebook.py         # generates the notebook from the package + data
├── run_demo.py  app.py       # CLI and Streamlit UI
├── evaluate.py  tests/       # evaluation script and pytest suite
├── openai_check.py           # one-call OpenAI connectivity test (reads .env)
├── travel_agent/             # models, intake, policy_store, tools, engine, agent, validator, dashboard, evaluation
├── data/                     # policy.md, policy_rules.json, limits.json, approval_matrix.json,
│                             # claims/ (5 JSON) + sample_claims.csv, receipts/ (mock metadata + text),
│                             # eval_cases.json (25 labelled cases)
├── docs/DESIGN_NOTES.md      # design notes, assumptions, limitations, next steps
└── outputs/                  # results.json, dashboard.png, UI SS_1.png, evaluation_report.csv
```

## Troubleshooting LLM mode

| Symptom | Likely cause / fix |
|---|---|
| `APIConnectionError: Connection error` | The endpoint is unreachable. Check internet/VPN/proxy, and that `OPENAI_BASE_URL` is empty (OpenAI) or correct. A stray `OPENAI_BASE_URL` in your shell overrides `.env` - run `echo $OPENAI_BASE_URL` and `unset OPENAI_BASE_URL`. |
| `AuthenticationError` | `OPENAI_API_KEY` is wrong or missing in `.env`. |
| `NotFoundError` | `OPENAI_MODEL` is not available for your key, or `OPENAI_BASE_URL` includes an extra path such as `/models`. |

On these errors the agent prints the endpoint and model, then continues in offline mode so you still get results.
