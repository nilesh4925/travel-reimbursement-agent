"""CLI: evaluate claims and write outputs/results.json + outputs/dashboard.png."""
import argparse
import json
import sys
from pathlib import Path

from travel_agent import config
from travel_agent.agent import TravelReimbursementAgent
from travel_agent.dashboard import save_png, summarize
from travel_agent.intake import load_claims


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--claims", default=str(config.DATA_DIR / "claims"), help="folder of JSON claims, a JSON file, or a CSV file")
    ap.add_argument("--mode", choices=["auto", "llm", "offline"], default="auto", help="auto = use the LLM if OPENAI_API_KEY is set")
    ap.add_argument("--out", default=str(config.OUTPUT_DIR))
    args = ap.parse_args()

    claims = load_claims(args.claims)
    agent = TravelReimbursementAgent(mode=args.mode)
    print(f"[mode: {agent.mode}] loaded {len(claims)} claims from {args.claims}", file=sys.stderr)
    if agent.mode == "llm":
        print(f"[llm] model={config.OPENAI_MODEL} endpoint={config.OPENAI_BASE_URL or 'https://api.openai.com/v1 (default)'}", file=sys.stderr)
    results = agent.evaluate_all(claims)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = [r.model_dump() for r in results]
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    save_png(results, claims, out / "dashboard.png")

    df, _ = summarize(results, claims)
    print("\n" + df.to_string(index=False), file=sys.stderr)
    print(f"\nSaved {out/'results.json'} and {out/'dashboard.png'}\n", file=sys.stderr)
    print(json.dumps(payload, indent=2))  # final output: JSON array, one object per claim


if __name__ == "__main__":
    main()
