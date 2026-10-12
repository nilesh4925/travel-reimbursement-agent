"""Run the labelled evaluation suite.

  python evaluate.py                          # offline deterministic baseline (default)
  python evaluate.py --mode llm               # OpenAI agent (needs OPENAI_API_KEY), with live progress
  python evaluate.py --mode llm --limit 5     # only the first 5 cases (quick smoke test)
  python evaluate.py --mode llm --workers 4   # evaluate 4 cases in parallel
  python evaluate.py --only CLM-004           # cases whose name contains this text
"""
import argparse
import sys

import pandas as pd

from travel_agent import config
from travel_agent.agent import TravelReimbursementAgent
from travel_agent.evaluation import confidence_bands, load_eval_cases, run_evaluation, summarize_evaluation


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["auto", "llm", "offline"], default="offline")
    ap.add_argument("--cases", default=str(config.DATA_DIR / "eval_cases.json"))
    ap.add_argument("--limit", type=int, help="only run the first N cases")
    ap.add_argument("--only", help="only run cases whose name contains this text")
    ap.add_argument("--workers", type=int, default=1, help="parallel cases (LLM mode); default 1")
    args = ap.parse_args()

    cases = load_eval_cases(args.cases)
    if args.only:
        cases = [c for c in cases if args.only.lower() in c["name"].lower()]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("No cases selected.")
        return 2

    agent = TravelReimbursementAgent(mode=args.mode)
    print(f"[mode: {agent.mode}] {len(cases)} cases", end="")
    if agent.mode == "llm":
        print(f" | model={config.OPENAI_MODEL} | endpoint={config.OPENAI_BASE_URL or 'https://api.openai.com/v1 (default)'} | workers={args.workers}")
        print("Each case is a multi-step tool-calling conversation; expect roughly 5-20 seconds per case.")
    else:
        print()
    print()
    df = run_evaluation(agent, cases, workers=args.workers, progress=True)

    pd.set_option("display.width", 200)
    print()
    print(df[["case", "expected", "got", "approved", "deducted", "confidence", "passed"]].to_string(index=False))
    print("\nSummary:", summarize_evaluation(df))
    print("\nConfidence vs observed accuracy:\n", confidence_bands(df))
    failed = df[~df["passed"]]
    if len(failed):
        print("\nFailed cases (in LLM mode a disagreement with the policy engine shows up as MANUAL_REVIEW):")
        print(failed[["case", "expected", "got"]].to_string(index=False))
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(config.OUTPUT_DIR / "evaluation_report.csv", index=False)
    return 0 if df["passed"].all() else 1


if __name__ == "__main__":
    sys.exit(main())
