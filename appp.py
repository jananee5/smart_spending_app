# app.py
import os
import re
import io
import json
import traceback
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd
import numpy as np
import streamlit as st
from fpdf import FPDF

# Lightweight, pure-python PDF reader
from pypdf import PdfReader

# HF client (lightweight)
from huggingface_hub import InferenceClient
from huggingface_hub.utils._errors import HfHubHTTPError

from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

# -------------------- Config / ENV --------------------
HF_API_KEY = None
try:
    HF_API_KEY = st.secrets.get("huggingface", {}).get("token", os.getenv("HF_API_KEY"))
except Exception:
    HF_API_KEY = os.getenv("HF_API_KEY")

# Default remote FLAN model (recommended)
DEFAULT_FLAN_MODEL = "google/flan-t5-small"

# Create HF client only if token is present
client = InferenceClient(token=HF_API_KEY) if HF_API_KEY else None

executor = ThreadPoolExecutor(max_workers=2)

# -------------------- Streamlit Page Config --------------------
st.set_page_config(page_title="SpendWise", layout="wide")

def set_background(url: str):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{url}");
            background-size: cover;
            background-attachment: fixed;
        }}
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.85);
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background(
    "https://static.vecteezy.com/system/resources/previews/019/154/472/large_2x/inflation-illustration-set-characters-buying-food-in-supermarket-and-worries-about-groceries-rising-price-vector.jpg"
)

st.title("📊 SpendWise — Smart UPI Analyzer (Safe FLAN Deployment)")

# -------------------- UTIL: PDF Parsing --------------------
def parse_pdf_bytes(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        try:
            text = page.extract_text()
        except Exception:
            text = None
        if text:
            pages.append(text)
    return "\n".join(pages)

# -------------------- UTIL: Transaction Extraction --------------------
DEFAULT_REGEX = re.compile(
    r"(Received|Paid|Credited|Debited|Deposit|Withdrawal)\s*[₹]?\s*([\d,]+\.\d{2}).*?(?:to|from)?\s*([A-Za-z0-9@&\-\._\/\s]+)?\s*\n?.*?([A-Za-z]{3}\s+\d{1,2},\s+\d{4})",
    re.DOTALL
)

def extract_transactions(text: str, regex: Optional[re.Pattern] = None) -> pd.DataFrame:
    regex = regex or DEFAULT_REGEX
    matches = regex.findall(text)
    transactions = []
    for ttype, amount, party, date in matches:
        parsed_date = pd.to_datetime(date, errors="coerce")
        transactions.append({
            "Date": parsed_date,
            "Type": normalize_type(ttype),
            "Amount": safe_float(amount),
            "Party": (party.strip() if party else "Self"),
            "RawDate": date
        })
    df = pd.DataFrame(transactions)
    if df.empty:
        return df
    df = df.dropna(subset=["Date"]).sort_values("Date", ascending=False).reset_index(drop=True)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    return df

def safe_float(val: str) -> float:
    try:
        return float(val.replace(",", ""))
    except Exception:
        return 0.0

def normalize_type(t):
    t = (t or "").lower()
    if t in ["received", "credited", "deposit"]:
        return "Received"
    if t in ["paid", "debited", "withdrawal"]:
        return "Paid"
    return t.capitalize()

# -------------------- CATEGORIZATION --------------------
BROAD_CATEGORIES = {
    "swiggy": "Food Delivery", "zomato": "Food Delivery", "zepto": "Groceries",
    "blinkit": "Groceries", "dunzo": "Food & Essentials", "amazon": "Shopping",
    "flipkart": "Shopping", "dmart": "Groceries", "reliance": "Groceries",
    "bigbasket": "Groceries", "apollo": "Medical", "pharmacy": "Medical",
    "uber": "Transport", "ola": "Transport", "fuel": "Fuel", "hpcl": "Fuel",
    "paytm": "Wallet/UPI", "phonepe": "Wallet/UPI", "gpay": "Wallet/UPI",
    "pvr": "Movies", "bookmyshow": "Movies & Events", "spotify": "Entertainment",
    "netflix": "Entertainment"
}

def map_category(party: str) -> str:
    name = (party or "").lower().strip()
    name = re.sub(r'\b(ltd|pvt|private|limited|india|inc|co)\b', '', name)
    for keyword, category in BROAD_CATEGORIES.items():
        if keyword in name:
            return category
    if "neft" in name or "imps" in name or "rtgs" in name or "transfer" in name:
        return "Bank Transfer"
    if "@" in name and ("upi" in name or "paytm" in name or "icici" in name or "sbi" in name):
        return "UPI Payment"
    if re.match(r'^[\d\-\/]+$', name):
        return "Bank/Other"
    return "Other"

def categorize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["Category"] = df["Party"].apply(map_category)
    return df

# -------------------- HELPERS & PROMPTS --------------------
def sanitize_text(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r'[^\x00-\xFF]', '?', text)

def build_prompts(df: pd.DataFrame, question: str) -> Dict[str, str]:
    income = df[df.Type == "Received"].Amount.sum() if not df.empty else 0.0
    expense = df[df.Type == "Paid"].Amount.sum() if not df.empty else 0.0
    net = income - expense

    monthly_summary = pd.DataFrame()
    cat_summary = pd.Series(dtype=float)
    try:
        monthly_summary = df.groupby(["Month", "Type"]).Amount.sum().unstack(fill_value=0).reset_index()
        cat_summary = df[df.Type == "Paid"].groupby("Category").Amount.sum().sort_values(ascending=False).head(10)
    except Exception:
        pass

    summary_prompt = f"""You are a helpful financial assistant.
Total Income: ₹{income:.2f}
Total Expenses: ₹{expense:.2f}
Net Balance: ₹{net:.2f}

Monthly breakdown:
{monthly_summary.to_string(index=False)}

Top spending categories:
{cat_summary.to_string()}

Give 2–3 short bullets summarizing financial state."""
    waste_prompt = f"""Given these transactions, identify up to 5 wasteful or recurring expenses (with approximate monthly cost) and why they might be optimized:

{df.head(30).to_string(index=False)}"""
    advice_prompt = f"""User Question: {question}

Based on the above, give 3 actionable recommendations for saving and budgeting (include one automation tip and one recurring cost reduction).
Output: short numbered bullets. If insufficient data, say 'insufficient data'."""
    return {"summary": summary_prompt, "waste": waste_prompt, "advice": advice_prompt}

def _normalize_hf_output(output: Any) -> str:
    try:
        if isinstance(output, str):
            return output
        if isinstance(output, list) and len(output) > 0:
            first = output[0]
            if isinstance(first, dict):
                return first.get("generated_text") or first.get("text") or str(first)
            return str(first)
        if isinstance(output, dict):
            return output.get("generated_text") or output.get("text") or str(output)
        return str(output)
    except Exception:
        return str(output)

# -------------------- REMOTE INFERENCE (with temperature support) --------------------
def hf_remote_infer(model_id: str, prompt: str, max_tokens: int = 512, temperature: Optional[float] = None) -> str:
    """
    Attempts remote inference using the InferenceClient and returns plain string.
    Accepts temperature if the API supports it.
    """
    if not HF_API_KEY or client is None:
        raise RuntimeError("HF_API_KEY not set for remote inference.")
    try:
        params = {"model": model_id, "prompt": prompt, "max_new_tokens": max_tokens}
        # add optional params if given
        if temperature is not None:
            params["temperature"] = float(temperature)
        out = client.text_generation(**params)
        return _normalize_hf_output(out)
    except HfHubHTTPError:
        raise
    except Exception:
        raise

# -------------------- DETERMINISTIC FALLBACK --------------------
def generate_simple_advice(df: pd.DataFrame, question: str) -> str:
    income = df[df.Type == "Received"].Amount.sum() if not df.empty else 0.0
    expense = df[df.Type == "Paid"].Amount.sum() if not df.empty else 0.0
    net = income - expense
    top_cats = df[df.Type == "Paid"].groupby("Category").Amount.sum().sort_values(ascending=False).head(5) if not df.empty else pd.Series(dtype=float)

    lines = []
    lines.append("Summary (fallback):")
    lines.append(f"Total Income: ₹{income:.2f}")
    lines.append(f"Total Expenses: ₹{expense:.2f}")
    lines.append(f"Net Balance: ₹{net:.2f}\n")

    if not top_cats.empty:
        lines.append("Top spending categories (fallback):")
        for cat, amt in top_cats.items():
            lines.append(f"- {cat}: ₹{amt:.2f}")
        lines.append("")

    qlow = question.lower()
    if "save" in qlow or "cut" in qlow or "saving" in qlow:
        lines.append("Actionable tips (fallback):")
        lines.append("- Reduce the largest spending category by 10–20% and re-evaluate next month.")
        lines.append("- Review and cancel unused subscriptions.")
        lines.append("- Automate small monthly transfer to savings (start with 5–10% of income).")
    else:
        lines.append("General tips (fallback):")
        lines.append("- Keep a weekly budget and track discretionary spending.")
        lines.append("- Automate 10% of income to savings if possible.")
    return "\n".join(lines)

# -------------------- VERIFICATION / SAFETY LAYER --------------------
AMOUNT_REGEX = re.compile(r"₹\s*([\d,]+(?:\.\d{1,2})?)")
PERCENT_REGEX = re.compile(r"(\d{1,3})\s*%")

def parse_amounts_from_text(text: str) -> Tuple[float, List[float]]:
    amounts = []
    for m in AMOUNT_REGEX.findall(text):
        try:
            amounts.append(float(m.replace(",", "")))
        except:
            continue
    return sum(amounts), amounts

def parse_percents_from_text(text: str) -> List[int]:
    percents = []
    for m in PERCENT_REGEX.findall(text):
        try:
            percents.append(int(m))
        except:
            continue
    return percents

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def basic_sanity_checks(advice_text: str, df: pd.DataFrame) -> Tuple[bool, List[str]]:
    reasons = []
    income = df[df.Type == "Received"].Amount.sum() if not df.empty else 0.0
    expense = df[df.Type == "Paid"].Amount.sum() if not df.empty else 0.0
    net = income - expense

    total_mentioned, amounts = parse_amounts_from_text(advice_text)
    percents = parse_percents_from_text(advice_text)

    for a in amounts:
        if income > 0 and a > income * 1.5:
            reasons.append(f"Explicit amount ₹{a:.2f} exceeds income (₹{income:.2f}).")
        if net < 0 and a > 0:
            reasons.append(f"Net is negative (₹{net:.2f}) but advice suggests saving/transferring ₹{a:.2f}.")

    for p in percents:
        if p < 0 or p > 1000:
            reasons.append(f"Suspicious percent value: {p}%.")
        if p > 100 and p <= 1000:
            reasons.append(f"Suggested percent {p}% seems unrealistic.")

    # Heuristic contradictions
    low_income_threshold = 1000.0
    if income < low_income_threshold and any(word in advice_text.lower() for word in ["invest", "save", "transfer"]):
        reasons.append("Income is very low but advice recommends aggressive financial moves.")

    is_valid = len(reasons) == 0
    return is_valid, reasons

def validate_and_finalize_advice(raw_outputs: str, df: pd.DataFrame, question: str, model_id: str, retry: bool = True) -> Tuple[str, bool, List[str]]:
    """
    Validate LLM outputs with 3 safeguards:
     1) Rule-based numeric checks on the output
     2) Cross-check using deterministic summary generation
     3) Confidence/consistency check: run FLAN twice (temp 0 and 0.6) and compare; if unstable -> regenerate or fallback
    Returns (final_text, used_llm_bool, reasons)
    """
    # 1) Basic sanity on raw outputs
    is_valid, reasons = basic_sanity_checks(raw_outputs, df)
    if is_valid:
        # Also perform a lightweight consistency check: run a second call with low temp and compare
        if HF_API_KEY and client is not None:
            try:
                # attempt a quick consistency call: temperature 0 (deterministic)
                deterministic_out = hf_remote_infer(model_id, raw_outputs + "\n\nPlease repeat succinctly.", max_tokens=200, temperature=0.0)
                sim = similarity(raw_outputs, deterministic_out)
                # If similarity is low, mark as unstable
                if sim < 0.6:
                    reasons.append(f"Low consistency between outputs (similarity={sim:.2f}).")
                    is_valid = False
                else:
                    return raw_outputs, True, []
            except Exception as e:
                # if the extra call fails, we still accept original if primary checks passed
                return raw_outputs, True, []
        else:
            return raw_outputs, True, []

    # If invalid or inconsistent, attempt regeneration with a strict safety prompt
    if retry and HF_API_KEY and client is not None:
        safety_prompt = (
            "You are a strict financial assistant that must NOT recommend impossible or unsafe amounts.\n"
            "Only use numeric values consistent with the provided totals. If unsure, say 'insufficient data'.\n\n"
            f"User question: {question}\n\n"
            f"Context summary: Income=₹{df[df.Type=='Received'].Amount.sum():.2f}, "
            f"Expenses=₹{df[df.Type=='Paid'].Amount.sum():.2f}, Net=₹{(df[df.Type=='Received'].Amount.sum() - df[df.Type=='Paid'].Amount.sum()):.2f}.\n\n"
            "Provide 2 clear numbered recommendations, any amounts must be <= income, keep answers short."
        )
        try:
            # Try with temperature 0 (deterministic)
            out = hf_remote_infer(model_id, safety_prompt, max_tokens=256, temperature=0.0)
            new_text = _normalize_hf_output(out)
            is_valid2, reasons2 = basic_sanity_checks(new_text, df)
            if is_valid2:
                # Also ensure deterministic vs original is reasonably similar or clearly safer
                return new_text, True, []
            else:
                # fallback to deterministic advice
                fallback = generate_simple_advice(df, question)
                reasons_combined = reasons + reasons2 + ["Regeneration failed checks — used deterministic fallback."]
                return fallback, False, reasons_combined
        except Exception as e:
            fallback = generate_simple_advice(df, question)
            reasons.append(f"Remote regeneration failed: {type(e).__name__}")
            return fallback, False, reasons
    else:
        fallback = generate_simple_advice(df, question)
        reasons.append("LLM output failed sanity checks — deterministic fallback used.")
        return fallback, False, reasons

# -------------------- RUN PIPELINE (uses FLAN if available) --------------------
def run_pipeline_with_remote(df: pd.DataFrame, question: str, model_id: str) -> str:
    """
    Orchestrates prompts, remote FLAN calls, validation and fallback.
    Returns an audit header + final advice.
    """
    prompts = build_prompts(df, question)
    combined_outputs = []
    try:
        # Primary run: prefer FLAN model if selected
        for name, prompt in prompts.items():
            # Append context marker so it's clear to the model
            full_prompt = f"### {name.upper()} ###\n{prompt}\n\n"
            # Attempt to call model at medium temperature for creativity
            out = None
            try:
                out = hf_remote_infer(model_id, full_prompt, max_tokens=512, temperature=0.6)
            except Exception:
                # Try deterministic fallback to get something
                out = hf_remote_infer(model_id, full_prompt, max_tokens=512, temperature=0.0) if HF_API_KEY and client is not None else None
            combined_outputs.append(f"--- {name.upper()} ---\n{_normalize_hf_output(out)}\n")
        raw = "\n".join(combined_outputs)

        final_text, used_llm, reasons = validate_and_finalize_advice(raw, df, question, model_id, retry=True)

        audit_lines = []
        if used_llm:
            audit_lines.append("✅ LLM (FLAN) output validated by numeric & consistency checks.")
        else:
            audit_lines.append("⚠️ LLM output rejected or unavailable — deterministic fallback used.")
        if reasons:
            audit_lines.append("Validation issues: " + "; ".join(reasons))

        return "\n".join(audit_lines) + "\n\n" + final_text
    except Exception:
        # Full remote pipeline failed — fallback to deterministic
        print("Remote inference failed — falling back to deterministic advice.")
        traceback.print_exc()
        fallback = generate_simple_advice(df, question)
        return (
            "⚠️ Remote model unavailable or returned an error. Showing fallback advice below.\n\n"
            + fallback
        )

# -------------------- REPORT (PDF) --------------------
def create_pdf_report(text: str, df: pd.DataFrame) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    title = sanitize_text("SpendWise - Personalized Financial Advice")
    pdf.multi_cell(0, 7, title, 0, 1)
    pdf.ln(2)
    try:
        income = df[df.Type == "Received"].Amount.sum()
        expense = df[df.Type == "Paid"].Amount.sum()
        pdf.multi_cell(0, 6, sanitize_text(f"Total Income: ₹{income:.2f}"))
        pdf.multi_cell(0, 6, sanitize_text(f"Total Expenses: ₹{expense:.2f}"))
        pdf.ln(2)
    except Exception:
        pass
    safe_text = sanitize_text(text)
    for line in safe_text.splitlines():
        pdf.multi_cell(0, 6, line)
    return pdf.output(dest="S").encode("latin-1", "replace")

# -------------------- UI / App --------------------
def main():
    st.sidebar.header("Settings")

    hf_token_present = bool(HF_API_KEY)
    if hf_token_present:
        st.sidebar.success("Hugging Face token: ✅ set (from Streamlit secrets or env)")
    else:
        st.sidebar.warning("Hugging Face token: ❌ not set")

    st.sidebar.markdown("---")

    model_options = [
        "gpt2",                  # tiny fallback
        DEFAULT_FLAN_MODEL,      # Google FLAN small (recommended)
        "google/flan-t5-base",   # larger FLAN (if available)
        "facebook/bart-large-cnn"
    ]
    # select recommended FLAN by default if token is present
    default_index = model_options.index(DEFAULT_FLAN_MODEL) if DEFAULT_FLAN_MODEL in model_options else 0
    selected_model = st.sidebar.selectbox("Choose model (hosted)", model_options, index=default_index)

    if not hf_token_present:
        st.sidebar.markdown(
            "Set HF token in Streamlit Secrets (Manage app → Settings → Secrets) as:\n\n"
            "```toml\n[huggingface]\ntoken = \"hf_xxx\"\n```"
        )
    else:
        st.sidebar.markdown("Model will use your Hugging Face token for inference (keeps the token secret).")

    if st.sidebar.button("Test model connectivity"):
        st.sidebar.info("Testing model... (this runs a small inference call)")
        try:
            test_client = InferenceClient(token=HF_API_KEY) if HF_API_KEY else None
            if not test_client:
                st.sidebar.error("No HF token available; cannot test.")
            else:
                out = test_client.text_generation(model=selected_model, prompt="Hello! Provide a 1-line summary.", max_new_tokens=16)
                txt = _normalize_hf_output(out)
                st.sidebar.success("Test OK — model responded")
                st.sidebar.write(txt[:300])
        except Exception as e:
            st.sidebar.error(f"Model test failed: {type(e).__name__}: {str(e)[:200]}")

    st.session_state["selected_model"] = selected_model
    st.sidebar.markdown("---")
    st.sidebar.caption("If remote model fails, app will show deterministic fallback advice.")

    uploaded = st.file_uploader("Upload bank / UPI statement (PDF)", type=["pdf"])
    df = pd.DataFrame()

    if uploaded:
        bytes_data = uploaded.read()
        raw_text = parse_pdf_bytes(bytes_data)
        df = extract_transactions(raw_text)
        df = categorize_df(df)

        if df.empty:
            st.error("No transactions found. Try a different statement or check parsing regex.")
            st.code(raw_text[:1000])
            return

        st.success(f"Parsed {len(df)} transactions.")
        st.dataframe(df.head(50))

        with st.expander("Monthly Summary Chart"):
            monthly = df.groupby(["Month", "Type"]).Amount.sum().unstack(fill_value=0)
            st.bar_chart(monthly)

        with st.expander("Top Spending Categories"):
            cats = df[df.Type == "Paid"].groupby("Category").Amount.sum().sort_values(ascending=False)
            st.bar_chart(cats)

        if st.sidebar.checkbox("Show parsed merchant names (debug)", value=False):
            st.sidebar.write(df["Party"].unique().tolist())

        question = st.text_area("Ask a financial question:", "Where can I cut spending or save more?")

        if st.button("Generate AI Advice"):
            st.info("Running pipeline... this may take a few seconds.")
            model_to_use = st.session_state.get("selected_model", DEFAULT_FLAN_MODEL)
            result_text = run_pipeline_with_remote(df, question, model_to_use)

            st.markdown("### 📋 Financial Advice (AI)")

            lines = result_text.splitlines()
            # show up to first 3 audit lines
            if lines and ("LLM (FLAN) output validated" in lines[0] or "LLM output rejected" in lines[0] or "Remote model unavailable" in lines[0]):
                audit_block = "\n".join(lines[:3])
                if "validated" in lines[0]:
                    st.success(audit_block)
                else:
                    st.warning(audit_block)
                st.markdown("---")
                st.write("\n".join(lines[3:]))
            else:
                st.write(result_text)

            pdf_bytes = create_pdf_report(result_text, df)
            st.download_button("Download PDF Report", data=pdf_bytes, file_name="spendwise_report.pdf", mime="application/pdf")

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download transactions CSV", data=csv_bytes, file_name="transactions.csv", mime="text/csv")

    else:
        st.info("Upload a UPI or bank statement PDF to begin.")


if __name__ == "__main__":
    main()
