# app.py
import os
import re
import io
from typing import Optional, Dict, Any

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
import traceback

# -------------------- Config / ENV --------------------
HF_API_KEY = None
try:
    HF_API_KEY = st.secrets.get("huggingface", {}).get("token", os.getenv("HF_API_KEY"))
except Exception:
    HF_API_KEY = os.getenv("HF_API_KEY")

# Model choices (commented options kept for evaluator)
# Note: Some models on HF are not hosted by the free Inference API and will return 404.
# For reliable demo on Streamlit Cloud, keep a small hosted model or rely on fallback.
#REMOTE_MISTRAL_ID = "google/flan-t5-base"
#REMOTE_MISTRAL_ID = "tiiuae/falcon-7b-instruct"
#REMOTE_MISTRAL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
# A safe default: choose a small model or use fallback if inference fails.
REMOTE_MISTRAL_ID = "gpt2"

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
        /* Semi-opaque overlay so text remains readable over busy images */
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.85); /* adjust opacity if needed */
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Use the Vecteezy image you provided
set_background(
    "https://static.vecteezy.com/system/resources/previews/019/154/472/large_2x/inflation-illustration-set-characters-buying-food-in-supermarket-and-worries-about-groceries-rising-price-vector.jpg"
)

st.title("📊 SpendWise — Smart UPI Analyzer")

# -------------------- UTIL: PDF Parsing --------------------
def parse_pdf_bytes(file_bytes: bytes) -> str:
    """
    Extract text from PDF bytes using pypdf (pure Python).
    Returns concatenated page text.
    """
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
# Looser regex to capture more merchant formats, with date at the end
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
    # Food & Delivery
    "swiggy": "Food Delivery",
    "swiggy limited": "Food Delivery",
    "zomato": "Food Delivery",
    "zepto": "Groceries",
    "blinkit": "Groceries",
    "bbdaily": "Groceries",
    "dunzo": "Food & Essentials",

    # Groceries & Retail
    "amazon": "Shopping",
    "flipkart": "Shopping",
    "dmart": "Groceries",
    "reliance": "Groceries",
    "bigbasket": "Groceries",
    "more supermarket": "Groceries",
    "big basket": "Groceries",

    # Medical & Healthcare
    "apollo": "Medical",
    "pharmacy": "Medical",
    "medplus": "Medical",
    "hospital": "Medical",
    "clinic": "Medical",

    # Transport & Fuel
    "uber": "Transport",
    "ola": "Transport",
    "rapido": "Transport",
    "metro": "Transport",
    "fuel": "Fuel",
    "hpcl": "Fuel",
    "indianoil": "Fuel",
    "bharat petroleum": "Fuel",

    # Jewellery & Luxury
    "jewellers": "Jewellery",
    "tanishq": "Jewellery",
    "kalyan": "Jewellery",
    "malabar": "Jewellery",
    "pc jeweller": "Jewellery",

    # Utilities & Bills
    "electricity": "Utilities",
    "water": "Utilities",
    "gas": "Utilities",
    "bsnl": "Utilities",
    "airtel": "Utilities",
    "jio": "Utilities",
    "vodafone": "Utilities",
    "reliance jio": "Utilities",

    # Rent & Loans
    "rent": "Rent",
    "emi": "Loan EMI",
    "loan": "Loan Payment",
    "insurance": "Insurance",
    "lic": "Insurance",

    # Payments / Wallets
    "phonepe": "Wallet/UPI",
    "paytm": "Wallet/UPI",
    "gpay": "Wallet/UPI",
    "google pay": "Wallet/UPI",
    "upi": "UPI Payment",

    # Entertainment
    "pvr": "Movies",
    "inox": "Movies",
    "bookmyshow": "Movies & Events",
    "spotify": "Entertainment",
    "netflix": "Entertainment",
    "hotstar": "Entertainment",
    "prime video": "Entertainment",
    "primevideo": "Entertainment",
}

def map_category(party: str) -> str:
    name = (party or "").lower().strip()

    # Normalize common noise words
    name = re.sub(r'\b(ltd|pvt|private|limited|india|inc|co)\b', '', name)

    # Detailed keyword mapping
    for keyword, category in BROAD_CATEGORIES.items():
        if keyword in name:
            return category

    # Bank transfers
    if "neft" in name or "imps" in name or "rtgs" in name or "transfer" in name:
        return "Bank Transfer"

    # UPI handle detection
    if "@" in name and ("ok" in name or "upi" in name or "paytm" in name or "icici" in name or "sbi" in name):
        return "UPI Payment"

    # If numeric-only party or card txn id -> treat as Other/Banks
    if re.match(r'^[\d\-\/]+$', name):
        return "Bank/Other"

    # Fallback
    return "Other"

def categorize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["Category"] = df["Party"].apply(map_category)
    return df

# -------------------- HELPERS & PIPELINE --------------------
def sanitize_text(text: str) -> str:
    """Replace any non-latin-1 character with '?' to keep fpdf happy."""
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

Based on the above, give 3 actionable recommendations for saving and budgeting (include one automation tip and one recurring cost reduction)."""

    return {"summary": summary_prompt, "waste": waste_prompt, "advice": advice_prompt}

# Create HF client only if token exists (safe guard)
client = InferenceClient(token=HF_API_KEY) if HF_API_KEY else None

def _normalize_hf_output(output: Any) -> str:
    """Normalize HF client return values to a readable string."""
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

# -------------------- REMOTE INFERENCE (robust with fallback) --------------------
def hf_remote_infer(model_id: str, prompt: str, max_tokens: int = 512) -> str:
    """
    Attempts remote inference using the InferenceClient and returns plain string.
    Raises exceptions to let caller fallback.
    """
    if not HF_API_KEY or client is None:
        raise RuntimeError("HF_API_KEY not set for remote inference.")
    try:
        out = client.text_generation(
            model=model_id,
            prompt=prompt,
            max_new_tokens=max_tokens
        )
        return _normalize_hf_output(out)
    except HfHubHTTPError:
        # re-raise to allow caller to fallback gracefully
        raise
    except Exception:
        # re-raise for caller
        raise

def generate_simple_advice(df: pd.DataFrame, question: str) -> str:
    """Deterministic fallback advice based on numeric summaries."""
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

    if "save" in question.lower() or "cut" in question.lower():
        lines.append("Actionable tips (fallback):")
        lines.append("- Reduce spending in top categories shown above by 10–20%.")
        lines.append("- Review subscriptions and recurring payments; cancel unused ones.")
        lines.append("- Automate small monthly transfers to savings.")
    else:
        lines.append("General tips (fallback):")
        lines.append("- Keep a weekly budget and track discretionary spending.")
        lines.append("- Automate 10% of income to savings if possible.")

    return "\n".join(lines)

def run_pipeline_with_remote(df: pd.DataFrame, question: str, model_id: str) -> str:
    """
    Try remote LLM; if it errors (404/403/etc.), fall back to deterministic advice.
    """
    prompts = build_prompts(df, question)
    outputs = []
    try:
        for name, prompt in prompts.items():
            out = hf_remote_infer(model_id, prompt, max_tokens=512)
            outputs.append(f"--- {name.upper()} ---\n{out}\n")
        return "\n".join(outputs)
    except Exception:
        # Log stacktrace to server logs for debugging
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
    # sanitize title and header
    title = sanitize_text("SpendWise - Personalized Financial Advice")
    pdf.multi_cell(0, 7, title, 0, 1)
    pdf.ln(2)

    # Optional: include a short summary table (sanitized)
    try:
        income = df[df.Type == "Received"].Amount.sum()
        expense = df[df.Type == "Paid"].Amount.sum()
        pdf.multi_cell(0, 6, sanitize_text(f"Total Income: ₹{income:.2f}"))
        pdf.multi_cell(0, 6, sanitize_text(f"Total Expenses: ₹{expense:.2f}"))
        pdf.ln(2)
    except Exception:
        pass

    # Sanitize AI text before writing
    safe_text = sanitize_text(text)
    for line in safe_text.splitlines():
        pdf.multi_cell(0, 6, line)

    # Return bytes in latin-1 to satisfy fpdf internals
    return pdf.output(dest="S").encode("latin-1", "replace")

# -------------------- UI / App --------------------
def main():
    st.sidebar.header("Settings")

    # Show whether HF token is loaded (do NOT display the token)
    hf_token_present = bool(HF_API_KEY)
    if hf_token_present:
        st.sidebar.success("Hugging Face token: ✅ set (from Streamlit secrets or env)")
    else:
        st.sidebar.warning("Hugging Face token: ❌ not set")

    st.sidebar.markdown("---")

    # Small list of hosted/testable models for the dropdown (you can change these)
    model_options = [
        "gpt2",                  # very small, usually available
        "google/flan-t5-small",  # small FLAN variant (test)
        "facebook/bart-large-cnn" # summarization model (test)
    ]
    selected_model = st.sidebar.selectbox("Choose model (hosted)", model_options, index=0)

    if not hf_token_present:
        st.sidebar.markdown(
            "Set HF token in Streamlit Secrets (Manage app → Settings → Secrets) as:\n\n"
            "```toml\n[huggingface]\ntoken = \"hf_xxx\"\n```"
        )
    else:
        st.sidebar.markdown("Model will use your Hugging Face token for inference (keeps the token secret).")

    # Quick connectivity test button
    if st.sidebar.button("Test model connectivity"):
        st.sidebar.info("Testing model... (this runs a small inference call)")
        try:
            test_client = InferenceClient(token=HF_API_KEY) if HF_API_KEY else None
            if not test_client:
                st.sidebar.error("No HF token available; cannot test.")
            else:
                out = test_client.text_generation(model=selected_model, prompt="Hello!", max_new_tokens=8)
                # Normalize output to string
                if isinstance(out, str):
                    txt = out
                elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
                    txt = out[0].get("generated_text") or out[0].get("text") or str(out[0])
                else:
                    txt = str(out)
                st.sidebar.success("Test OK — model responded")
                st.sidebar.write(txt[:300])
        except Exception as e:
            st.sidebar.error(f"Model test failed: {type(e).__name__}: {str(e)[:200]}")

    # Store selection so other parts of app can use it
    st.session_state["selected_model"] = selected_model
    st.sidebar.markdown("---")
    st.sidebar.caption("If the remote model fails, app will show fallback (deterministic) advice.")

    # --- Main UI ---
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

        # show unique parties for debugging
        if st.sidebar.checkbox("Show parsed merchant names (debug)", value=False):
            st.sidebar.write(df["Party"].unique().tolist())

        question = st.text_area("Ask a financial question:", "Where can I cut spending or save more?")

        if st.button("Generate AI Advice"):
            st.info("Running pipeline... this may take a few seconds.")
            # Use selected model if available, otherwise fallback to REMOTE_MISTRAL_ID
            model_to_use = st.session_state.get("selected_model", REMOTE_MISTRAL_ID)
            result_text = run_pipeline_with_remote(df, question, model_to_use)

            st.markdown("### 📋 Financial Advice (AI)")
            st.write(result_text)

            pdf_bytes = create_pdf_report(result_text, df)
            st.download_button("Download PDF Report", data=pdf_bytes, file_name="spendwise_report.pdf", mime="application/pdf")

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download transactions CSV", data=csv_bytes, file_name="transactions.csv", mime="text/csv")

    else:
        st.info("Upload a UPI or bank statement PDF to begin.")


if __name__ == "__main__":
    main()

