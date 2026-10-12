"""Minimal results UI:  streamlit run app.py"""
import json

import pandas as pd
import streamlit as st

from travel_agent import config
from travel_agent.agent import TravelReimbursementAgent
from travel_agent.dashboard import summarize
from travel_agent.intake import load_claims, parse_csv, parse_json

st.set_page_config(page_title="Travel Reimbursement Agent", layout="wide")


@st.cache_resource
def get_agent() -> TravelReimbursementAgent:
    return TravelReimbursementAgent()


agent = get_agent()
st.title("Travel Reimbursement Approval Agent")
st.caption(f"Engine: {'LLM tool-use agent (' + config.OPENAI_MODEL + ')' if agent.mode == 'llm' else 'offline deterministic planner (set OPENAI_API_KEY for LLM mode)'}")

# ---------- 1. claim intake ----------
source = st.sidebar.radio("Claim source", ["Sample claims", "Upload JSON / CSV", "Paste JSON"])
claims, error = [], None
try:
    if source == "Sample claims":
        claims = load_claims(config.DATA_DIR / "claims")
        picked = st.sidebar.multiselect("Claims", [c.claim_id for c in claims], default=[c.claim_id for c in claims])
        claims = [c for c in claims if c.claim_id in picked]
    elif source == "Upload JSON / CSV":
        f = st.sidebar.file_uploader("Claim file", type=["json", "csv"])
        if f:
            text = f.getvalue().decode()
            claims = parse_csv(text) if f.name.lower().endswith(".csv") else parse_json(text)
    else:
        text = st.sidebar.text_area("Claim JSON (object or array)", height=300)
        if text.strip():
            claims = parse_json(text)
except Exception as exc:
    error = f"Could not read claims: {exc}"

if error:
    st.error(error)
st.subheader(f"Claims to evaluate: {len(claims)}")
if claims:
    st.dataframe(pd.DataFrame([{"claim_id": c.claim_id, "employee": c.employee, "purpose": c.purpose, "items": len(c.items), "total_claimed": c.total} for c in claims]), hide_index=True, width="stretch")

# ---------- 2. run agent ----------
if st.button("Evaluate claims", type="primary", disabled=not claims):
    with st.spinner("Agent is reviewing claims..."):
        st.session_state["claims"] = claims
        st.session_state["results"] = agent.evaluate_all(claims)

# ---------- 3. results + dashboard ----------
if "results" in st.session_state:
    claims_done, results = st.session_state["claims"], st.session_state["results"]
    df, counts = summarize(results, claims_done)
    st.header("Dashboard")
    k = st.columns(4)
    k[0].metric("Claims", len(df))
    k[1].metric("Approved", f"${df['approved'].sum():,.2f}")
    k[2].metric("Deducted", f"${df['deducted'].sum():,.2f}")
    k[3].metric("Pending manual review", f"${df['pending_review'].sum():,.2f}")
    c1, c2 = st.columns(2)
    c1.caption("Decision breakdown")
    c1.bar_chart(counts)
    c2.caption("Claimed amount by outcome (USD)")
    c2.bar_chart(df.set_index("claim_id")[["approved", "deducted", "pending_review"]])
    st.dataframe(df, hide_index=True, width="stretch")

    st.header("Decisions")
    by_id = {c.claim_id: c for c in claims_done}
    for r in results:
        with st.expander(f"{r.claim_id} - {r.decision} - approved ${r.approved_amount:,.2f} / deducted ${r.deducted_amount:,.2f}"):
            st.write(r.explanation)
            st.write(f"**Confidence:** {r.confidence}  |  **Policy refs:** {', '.join(r.policy_refs)}  |  **Tools:** {', '.join(r.tools_used)}")
            if r.missing_docs:
                st.warning("Missing documents: " + "; ".join(r.missing_docs))
            st.dataframe(pd.DataFrame([i.model_dump() for i in by_id[r.claim_id].items]), hide_index=True, width="stretch")

    st.download_button("Download results.json", json.dumps([r.model_dump() for r in results], indent=2), "results.json", "application/json")
