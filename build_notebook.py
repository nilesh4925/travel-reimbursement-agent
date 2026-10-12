"""Builds the single-notebook deliverable (nilesh.ipynb) from the package source + data files.

The notebook is fully self-contained (policy, claims, receipts and eval cases are embedded), so it runs
top-to-bottom on its own. Regenerate with:  python build_notebook.py
"""
import json
import re
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
PKG, DATA = ROOT / "travel_agent", ROOT / "data"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "nilesh.ipynb"


def module_source(name: str) -> str:
    """Package module -> notebook cell: drop relative imports, send progress prints to stdout."""
    code = (PKG / f"{name}.py").read_text()
    code = re.sub(r"^from __future__ import annotations\n+", "", code, flags=re.M)
    code = re.sub(r"^from \.[\w.]* import \([^)]*\)\n", "", code, flags=re.M)
    code = re.sub(r"^from \.[\w.]* import [^(\n]*\n", "", code, flags=re.M)
    code = code.replace(", file=sys.stderr", "")
    return code.strip() + "\n"


def jload(path):
    return json.loads(Path(path).read_text())


cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))   # noqa: E731
code = lambda t: cells.append(nbf.v4.new_code_cell(t.strip("\n")))     # noqa: E731

# ------------------------------------------------------------------ README
md('''
# Travel Reimbursement Approval Agent

A runnable GenAI / agentic prototype that reviews employee travel-reimbursement claims against a written policy,
receipts, per-diem limits and approval rules, and returns a structured recommendation:
`APPROVE`, `PARTIAL_APPROVE`, `REJECT` or `MANUAL_REVIEW`.

## README

**What this notebook contains (top to bottom):** configuration + embedded data -> models -> policy grounding ->
tools -> decision engine -> validator -> agent -> run on the 5 provided claims -> sample outputs ->
**Dashboard** (charts + interactive UI) -> **Evaluation & tests** -> **Design Notes & Reasoning** -> final JSON.

### Setup
1. Python **3.10+**.
2. Install dependencies (once):
   ```bash
   pip install openai pandas matplotlib pydantic python-dotenv ipywidgets jupyter
   ```
3. *(Optional, for LLM mode)* create a `.env` file next to this notebook:
   ```
   OPENAI_API_KEY=sk-...
   OPENAI_MODEL=gpt-4o-mini        # any OpenAI chat model with function calling
   OPENAI_BASE_URL=                # leave empty for OpenAI; or any OpenAI-compatible endpoint (e.g. local Ollama)
   ```
4. **Run all cells** (`Kernel -> Restart & Run All`). No manual steps are needed.

### Environment variables
| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | No | - | Enables **LLM mode**. Without it the agent runs in **offline mode** (same tools, deterministic planner). |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | OpenAI chat model used in LLM mode. |
| `OPENAI_BASE_URL` | No | OpenAI default | Optional OpenAI-compatible endpoint. Use the base URL only (e.g. `https://api.openai.com/v1`). |
| `MAX_AGENT_TURNS` | No | `12` | Safety cap on tool-calling turns per claim. |

> The outputs saved in this notebook were produced in **offline mode** (no API key). With a key set, the same
> notebook runs the OpenAI tool-calling agent; every LLM decision is still cross-checked by the policy engine.

### How it works
```
claim (JSON/CSV/UI) -> intake -> Agent
  LLM mode : OpenAI model <-> tools (lookup_policy, check_claim_consistency, check_category_eligibility,
             check_receipts, check_per_diem_limits, check_timeliness, check_approval_threshold)
             -> submit_decision(decision, confidence, explanation)
  Offline  : same tools, fixed order
      -> policy-engine cross-check (disagreement => MANUAL_REVIEW)
      -> output validator (invalid => MANUAL_REVIEW) -> structured result
```
Key design choices: tools do all arithmetic; the LLM chooses tools and explains; independent verification and
validation guard every result; uncertain cases go to Manual Review. Details in **Design Notes & Reasoning** below.
''')

# ------------------------------------------------------------------ config + embedded data
md('## 1. Configuration and embedded data\nPolicy rules, limits, approval matrix, the five provided claims (Appendix B), mock receipt metadata and the labelled evaluation cases are embedded so the notebook needs no other files.')
embedded = {
    "policy_rules": jload(DATA / "policy_rules.json"),
    "limits": jload(DATA / "limits.json"),
    "approval_matrix": jload(DATA / "approval_matrix.json"),
    "claims": [jload(f) for f in sorted((DATA / "claims").glob("*.json"))],
    "receipts": {f.stem: jload(f)["receipts"] for f in sorted((DATA / "receipts").glob("*.json"))},
}
code('''
import json, os, re, sys
from types import SimpleNamespace

try:  # optional: read OPENAI_* settings from a .env file next to the notebook
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

config = SimpleNamespace(
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
    OPENAI_MODEL=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    OPENAI_BASE_URL=os.getenv("OPENAI_BASE_URL") or None,
    MAX_AGENT_TURNS=int(os.getenv("MAX_AGENT_TURNS", "12")),
)
''')
code("EMBEDDED = json.loads(r'''\n" + json.dumps(embedded, indent=1) + "\n''')\nprint({k: len(v) for k, v in EMBEDDED.items()})")

policy_md = re.sub(r"^## ", "#### ", (DATA / "policy.md").read_text().replace("# Appendix A - Travel Reimbursement Policy\n", ""), flags=re.M)
md("### Policy used (Appendix A)\n\n" + policy_md.strip())

# ------------------------------------------------------------------ models + intake
md('## 2. Claim models and intake\nPydantic models define a claim and the **exact output contract**. Intake accepts JSON (object/array) or CSV.')
code(module_source("models") + "\n\n" + module_source("intake"))

# ------------------------------------------------------------------ policy store
md('## 3. Policy grounding\nThe agent retrieves relevant `POL-*` rules (keyword retrieval over the 12 rules) and uses the limit table and approval matrix.')
ps = module_source("policy_store")
a, b = ps.index("# --- data access"), ps.index("# --- end data access ---") + len("# --- end data access ---")
ps = ps[:a] + '''# --- data access: embedded data ---
LIMITS = EMBEDDED["limits"]
APPROVAL = EMBEDDED["approval_matrix"]
RULES = {r["id"]: r for r in EMBEDDED["policy_rules"]}


def load_receipts(claim_id):
    """Receipt metadata on file for a claim (None if the claim has no receipt record)."""
    return EMBEDDED["receipts"].get(claim_id)
''' + ps[b:]
code(ps)

# ------------------------------------------------------------------ tools / engine / validator / agent
md('## 4. Tools\nSeven deterministic tools. The LLM decides *which* to call; numbers always come from the tools. A `ToolBox` binds the tools to one claim and records every call (`tools_used`).')
code(module_source("tools"))

md('## 5. Policy engine\nTurns tool outputs into a decision, amounts, policy references, explanation and a signal-based confidence. It drives the offline planner and independently verifies the LLM.')
code(module_source("engine") + "\nengine = SimpleNamespace(decide=decide, run_all_checks=run_all_checks, Assessment=Assessment)")

md('## 6. Output validator\nSchema and business-consistency checks on every result (exact fields, allowed decisions, known policy ids, amounts reconcile).')
code(module_source("validator"))

md('## 7. The agent\nOpenAI function-calling loop (LLM mode) with an offline deterministic planner, plus guardrails: cross-check against the policy engine, output validation, and fallbacks (API failure -> offline planner; invalid/conflicting output -> `MANUAL_REVIEW`).')
code(module_source("agent"))

# ------------------------------------------------------------------ run
md('## 8. Run the agent on the five provided claims')
code('''
claims = [Claim(**c) for c in EMBEDDED["claims"]]
agent = TravelReimbursementAgent()  # LLM mode if OPENAI_API_KEY is set, otherwise offline
print(f"[mode: {agent.mode}]" + (f" model={config.OPENAI_MODEL}" if agent.mode == "llm" else " (set OPENAI_API_KEY in .env for LLM mode)"))
results = agent.evaluate_all(claims)
''')

md('## Sample outputs\nDecisions and explanations for all five provided claims.')
code('''
for c, r in zip(claims, results):
    print(f"{r.claim_id} | {c.purpose}")
    print(f"  decision   : {r.decision}   approved ${r.approved_amount:,.2f} / deducted ${r.deducted_amount:,.2f}   (confidence {r.confidence})")
    print(f"  policy refs: {', '.join(r.policy_refs)}")
    print(f"  missing    : {r.missing_docs or 'none'}")
    print(f"  tools used : {', '.join(r.tools_used)}")
    print(f"  explanation: {r.explanation}\\n")
''')

# ------------------------------------------------------------------ dashboard
md('## Dashboard\nData-driven summary of the actual claim outcomes, followed by an interactive UI to evaluate any claim.')
code(module_source("dashboard") + '''

import matplotlib.pyplot as plt
from IPython.display import display

summary_df, decision_counts = summarize(results, claims)
display(summary_df)
fig = draw_dashboard(results, claims)
plt.show()
''')

md('### Interactive UI\nPick a provided claim, or paste your own claim JSON (object or array) and click **Evaluate**. Uses the same agent as above.')
code('''
try:
    import ipywidgets as w
    from IPython.display import HTML, clear_output, display

    COLORS_UI = {"APPROVE": "#2e7d32", "PARTIAL_APPROVE": "#f9a825", "REJECT": "#c62828", "MANUAL_REVIEW": "#1565c0"}
    picker = w.Dropdown(options=[c.claim_id for c in claims] + ["Custom JSON"], description="Claim:")
    box = w.Textarea(placeholder='Paste claim JSON here (used when "Custom JSON" is selected)', layout=w.Layout(width="100%", height="140px"))
    btn = w.Button(description="Evaluate", button_style="primary")
    out = w.Output()

    def _show(claim, r):
        color = COLORS_UI[r.decision]
        display(HTML(
            f"<div style='border-left:6px solid {color};padding:8px 14px;margin:8px 0'>"
            f"<b>{r.claim_id}</b> - <span style='color:{color};font-weight:bold'>{r.decision}</span> "
            f"&nbsp; approved <b>${r.approved_amount:,.2f}</b> / deducted <b>${r.deducted_amount:,.2f}</b> "
            f"&nbsp; confidence {r.confidence}<br>{r.explanation}<br>"
            f"<small>policy refs: {', '.join(r.policy_refs)} | missing: {', '.join(r.missing_docs) or 'none'} | tools: {', '.join(r.tools_used)}</small></div>"))
        display(pd.DataFrame([i.model_dump() for i in claim.items]))

    def _on_click(_):
        with out:
            clear_output()
            try:
                todo = parse_json(box.value) if picker.value == "Custom JSON" else [c for c in claims if c.claim_id == picker.value]
                for c in todo:
                    _show(c, agent.evaluate(c))
            except Exception as exc:
                print(f"Could not evaluate: {type(exc).__name__}: {exc}")

    btn.on_click(_on_click)
    display(w.VBox([w.HBox([picker, btn]), box, out]))
except ImportError:
    print("Interactive UI needs ipywidgets:  pip install ipywidgets  (the charts above work without it)")
''')

# ------------------------------------------------------------------ evaluation
md('''
## Evaluation and tests
A labelled suite of **25 cases**: the 5 provided claims plus 20 edge cases around every policy boundary (30 vs 31 days,
$25.00 vs $25.01, $500 / $2,000 / $2,000.01 totals, unknown categories, mismatched totals, receipt conflicts, ...).
Expected outcomes were written from the policy text, not from the program's output. The same suite runs as
`pytest` in the repository (52 tests, including a mocked OpenAI tool-calling loop).

The cell below runs the deterministic baseline and **asserts every case passes**. It also reports how well the
signal-based confidence lines up with observed accuracy (accuracy, Brier score, accuracy per confidence band).
Set `RUN_LLM_EVAL = True` to also run the OpenAI agent over the suite (makes one API call loop per case).
''')
code(module_source("evaluation") + "\nEVAL_CASES = json.loads(r'''\n" + json.dumps(jload(DATA / "eval_cases.json"), indent=1) + "\n''')")
code('''
RUN_LLM_EVAL = False  # True -> also evaluate the OpenAI agent (needs OPENAI_API_KEY)

baseline = TravelReimbursementAgent(mode="offline")
eval_df = run_evaluation(baseline, EVAL_CASES)
print(eval_df[["case", "expected", "got", "approved", "deducted", "confidence", "passed"]].to_string(index=False))
print("\\nSummary :", summarize_evaluation(eval_df))
print("\\nConfidence vs observed accuracy:\\n", confidence_bands(eval_df))
assert eval_df["passed"].all(), "evaluation failures: " + ", ".join(eval_df.loc[~eval_df["passed"], "case"])
print(f"\\nAll {len(eval_df)} labelled cases passed (deterministic baseline).")

if RUN_LLM_EVAL and agent.mode == "llm":
    llm_df = run_evaluation(agent, EVAL_CASES)
    print("\\nOpenAI agent:", summarize_evaluation(llm_df))
    print(llm_df.loc[~llm_df["passed"], ["case", "expected", "got"]].to_string(index=False) or "no disagreements")
''')

# ------------------------------------------------------------------ design notes
notes = (ROOT / "docs" / "DESIGN_NOTES.md").read_text()
notes = notes.replace("# Design Notes & Reasoning", "## Design Notes & Reasoning", 1)
notes = re.sub(r"^## (?!Design Notes)", "### ", notes, flags=re.M)
md(notes)

# ------------------------------------------------------------------ final results
md('## Final results\nJSON array with one object per provided claim (exact fields required by the assignment).')
code('''
FIELDS = ["claim_id", "decision", "approved_amount", "deducted_amount", "missing_docs", "policy_refs", "confidence", "explanation", "tools_used"]
final_output = [r.model_dump() for r in results]
assert len(final_output) == 5 and all(list(o) == FIELDS for o in final_output), "output does not match the required contract"
print(json.dumps(final_output, indent=2))
''')

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
