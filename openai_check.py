"""Minimal OpenAI connectivity check: reads .env, makes one LLM call, prints the reply.

Run:  python openai_check.py
Needs: pip install openai python-dotenv
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent / ".env")

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
base_url = os.getenv("OPENAI_BASE_URL") or None  # empty -> official OpenAI endpoint

print(f"model    : {model}")
print(f"endpoint : {base_url or 'https://api.openai.com/v1 (default)'}")
print(f"api key  : {'set (' + api_key[:7] + '...)' if api_key else 'MISSING - add OPENAI_API_KEY to .env'}")
if not api_key:
    sys.exit(1)

client = OpenAI(api_key=api_key, base_url=base_url, timeout=30, max_retries=1)
try:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: OpenAI connection OK"}],
        temperature=0,
    )
    print("\nresponse :", resp.choices[0].message.content)
except Exception as exc:
    print(f"\nFAILED   : {type(exc).__name__}: {exc}")
    sys.exit(1)
