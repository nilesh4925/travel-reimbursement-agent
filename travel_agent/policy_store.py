"""Policy grounding: rules (POL-*), limit table and approval matrix loaded from /data."""
from __future__ import annotations

import json
import re

from .config import DATA_DIR


def _load(name: str):
    return json.loads((DATA_DIR / name).read_text())


LIMITS = _load("limits.json")
APPROVAL = _load("approval_matrix.json")
RULES = {r["id"]: r for r in _load("policy_rules.json")}

_STOP = {"the", "a", "an", "of", "to", "and", "or", "for", "is", "in", "on", "per", "be", "at", "by", "any"}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP}


def search_policy(query: str, k: int = 5) -> list[dict]:
    """Lightweight keyword retrieval over the 12 policy rules (no vector DB needed at this size)."""
    wanted = set(re.findall(r"POL-[A-Z]+-\d+", query.upper()))
    q = _tokens(query)
    scored = []
    for rid, r in RULES.items():
        tag_tokens = _tokens(" ".join(r["tags"]))
        body = _tokens(f"{r['title']} {r['text']}")
        score = 3 * len(q & tag_tokens) + len(q & body) + (100 if rid in wanted else 0)
        if score:
            scored.append((-score, rid))
    scored.sort()
    return [{"id": rid, "title": RULES[rid]["title"], "text": RULES[rid]["text"]} for _, rid in scored[:k]]
