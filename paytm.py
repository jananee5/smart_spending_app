# app.py
"""
SpendWise — UPI / Bank PDF analyzer with safe remote inference + secure UI.

- I tested local LLMs (e.g., loading larger checkpoints in Colab) but commented out heavy
  local model code and torch import to keep the Streamlit Cloud deployment lightweight and fast.
  This is intentional: remote inference or a deterministic fallback is used for deployment.
"""

import os
import re
import io
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
import streamlit as st
from fpdf import FPDF
from pypdf import PdfReader

# Hugging Face client (used only if user supplies a token)
from huggingface_hub import InferenceClient
from huggingface_hub.utils._errors import HfHubHTTPError

from concurrent.futures import ThreadPoolExecutor
import traceback

# -------------------- Config / ENV (do not print token anywhere) --------------------
HF_API_KEY = None
try:
    HF_API_KEY = st.secrets.get("huggingface", {}).get("token", os.getenv("HF_API_KEY"))
except Exception:
    HF_API_KEY = os.getenv("HF_API_KEY")

# Commented model choices (I tried larger models locally, but commented to speed up cloud deploy)
# LOCAL_MODEL = "local/flan-t5-large"   # (example) - tested in Colab only
# REMOTE_MISTRAL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
# REMOTE_MISTRAL_ID = "google/flan-t5-base"

# Use a lightweight demo default; user may pick another from sidebar list
REMOTE_MISTRAL_ID = "gpt2"

# create client only if token present
client = None
if HF_API_KEY:
    try:
        client = InferenceClient(token=HF_API_KEY)
    except Exception as e:
        # don't crash the app — we will fallback to deterministic advice
        print("Warning: Hugging Face InferenceClient could not be created:", e)
        client = None

executor = ThreadPoolExecutor(max_workers=2)

# -------------------- Streamlit Page Config --------------------
st.set_page_config(page_title="SpendWise — Secure UPI Analyzer", layout="wide")

# -------------------- Background + readability CSS --------------------
def set_background(url: str):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{url}") no-repeat center center fixed;
            background-size: cover;
            position: relative;
            min-height: 100vh;
        }}
        .stApp::before {{
            content: "";
            position: absolute;
            inset: 0;
            background: rgba(255,255,255,0.88); /* light overlay to keep text readable */
            z-index: 0;
            pointer-events: none;
        }}
        /* ensure main content renders above overlay */
        .reportview-container .main, .stApp > .main {{
            position: relative;
            z-index: 1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# A readable illustration link (you provided Vecteezy earlier)
set_background("https://static.vecteezy.com/system/resources/previews/019/154/472/large_2x/inflation-illustration-set-characters-buying-food-in-supermarket-and-worries-about-groceries-rising-price-vector.jpg")

st.title("📊 SpendWise — Secure UPI & Bank Statement Analyzer")

# -------------------- PDF text extraction --------------------
def parse_pdf_bytes(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    # limit pages parsed to avoid huge files; adjust as needed
    max_pages = min(len(reader.pages), 200)
    for i in range(max_pages):
        try:
            text = reader.pages[i].extract_text()
        except Exception:
            text = None
        if text:
            pages.append(text)
    return "\n".join(pages)

# -------------------- Transaction extraction (robust, tag-aware, year inference) --------------------
# fallback regex for quick matches
DEFAULT_REGEX = re.compile(
    r"(Paid to|Money sent to|Received from|Credited to|Paid)\s+(.+?)\n.*?-?\s*Rs\.?([\d,]+(?:\.\d+)?)",
    re.IGNORECASE
)

def extract_transactions(text: str) -> pd.DataFrame:
    """
    Robustly extract transactions:
    - Infers statement end-year if provided (e.g., "1 DEC'24 - 31 MAY'25") and uses it
      to disambiguate day+month dates.
    - Scans for 'Paid to' / 'Money sent to' blocks and looks ahead for Tag: # ...
    - Returns DataFrame with Date, Type, Amount, Party, Tag, RawDate.
    """
    # infer end-year from header if available
    end_year = None
    try:
        m_range = re.search(r"\b\d{1,2}\s+[A-Za-z]{3}'\d{2}\s*-\s*\d{1,2}\s+[A-Za-z]{3}'\d{2}\b", text)
        if m_range:
            right_part = m_range.group(0).split("-")[-1].strip()
            yy = re.search(r"'?(\d{2})\b", right_part)
            if yy:
                end_year = 2000 + int(yy.group(1))
    except Exception:
        end_year = None

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    transactions = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.lower().startswith(("paid to ", "money sent to ", "received from ", "credited to ", "paid ")):
            party = re.sub(r'^(paid to|money sent to|received from|credited to|paid)\s+', '', line, flags=re.I).strip()
            amount = None
            date = None
            tag = None

            # look ahead for amount/date/tag
            for j in range(i, min(i + 9, len(lines))):
                ln = lines[j]
                m_amt = re.search(r'Rs\.?\s*([\d,]+(?:\.\d+)?)', ln, flags=re.I)
                if m_amt and amount is None:
                    amount = m_amt.group(1)

                # date patterns "May 31" or "31 May" optionally with year
                m_date = re.search(r'([A-Za-z]{3,9}\s+\d{1,2}(?:,\s*\d{4})?)', ln)
                if not m_date:
                    m_date = re.search(r'(\d{1,2}\s+[A-Za-z]{3,9}(?:,\s*\d{4})?)', ln)
                if m_date and date is None:
                    date_str = m_date.group(1)
                    if not re.search(r'\d{4}', date_str) and end_year:
                        # append inferred year
                        candidate = pd.to_datetime(f"{date_str}, {end_year}", errors="coerce")
                        date = candidate if pd.notna(candidate) else pd.to_datetime(date_str, errors="coerce")
                    else:
                        date = pd.to_datetime(date_str, errors="coerce")

                m_tag = re.search(r'Tag:\s*#\s*([^\s]+)', ln, flags=re.I)
                if m_tag and tag is None:
                    tag = m_tag.group(1).strip()

            try:
                amt_val = float(amount.replace(",", "")) if amount else 0.0
            except Exception:
                amt_val = 0.0

            transactions.append({
                "Date": date if date is not None else pd.NaT,
                "Type": "Paid" if 'paid' in line.lower() or 'money sent' in line.lower() else "Received",
                "Amount": amt_val,
                "Party": party or "Self",
                "Tag": (tag or "").strip(),
                "RawDate": "" if date is None else str(date)
            })
            i += 1
            continue

        # fallback if a line contains an amount but not the typical prefix
        fallback = DEFAULT_REGEX.search(line)
        if fallback:
            ttype = fallback.group(1)
            party = fallback.group(2)
            amount = fallback.group(3)
            parsed_date = None
            for k in (i-1, i+1, i+2):
                if 0 <= k < len(lines):
                    dmatch = re.search(r'([A-Za-z]{3,9}\s+\d{1,2}(?:,\s*\d{4})?)', lines[k])
                    if dmatch:
                        parsed_date = pd.to_datetime(dmatch.group(1), errors="coerce")
                        break
            try:
                amt_val = float(amount.replace(",", "")) if amount else 0.0
            except Exception:
                amt_val = 0.0
            transactions.append({
                "Date": parsed_date if parsed_date is not None else pd.NaT,
                "Type": normalize_type(ttype),
                "Amount": amt_val,
                "Party": party.strip() if party else "Self",
                "Tag": "",
                "RawDate": ""
            })
        i += 1

    df = pd.DataFrame(transactions)
    if df.empty:
        return df
    df = df.dropna(subset=["Date"]).sort_values("Date", ascending=False).reset_index(drop=True)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    return df

def normalize_type(t):
    t = (t or "").lower()
    if t in ["received", "credited", "deposit"]:
        return "Received"
    if t in ["paid", "debited", "withdrawal"]:
        return "Paid"
    return t.capitalize()

# -------------------- Categorization (Tag-preferred, then keyword) --------------------
BROAD_CATEGORIES = {
    "swiggy": "Food",
    "zomato": "Food",
    "zepto": "Groceries",
    "blinkit": "Groceries",
    "dunzo": "Food & Essentials",
    "amazon": "Shopping",
    "flipkart": "Shopping",
    "dmart": "Groceries",
    "reliance": "Groceries",
    "bigbasket": "Groceries",
    "apollo": "Medical",
    "pharmacy": "Medical",
    "uber": "Transport",
    "ola": "Transport",
    "fuel": "Fuel",
    "tanishq": "Jewellery",
    "pvr": "Movies",
    "netflix": "Entertainment",
    "paytm": "Wallet/UPI",
    "phonepe": "Wallet/UPI",
    "gpay": "Wallet/UPI",
    "jio": "Utilities",
    "airtel": "Utilities",
    "rent": "Rent",
    "emi": "Loan EMI",
    "insurance": "Insurance",
}

DETAILED_MAPPING = {
    "good to go foodworks private limited": "Food",
    "pvr inox": "Movies",
    "zepto marketplace": "Groceries",
    "medplus": "Medical",
    "swiggy limited": "Food",
    "rollbaby": "Food",
    "phonepe": "Wallet/UPI",
}

def map_category(party: str, tag: str = "") -> str:
    # use explicit tag first
    if tag:
        t = tag.lower().strip("️")
        if "food" in t:
            return "Food"
        if "grocery" in t or "grocer" in t:
            return "Groceries"
        if "fuel" in t:
            return "Fuel"
        if "bill" in t or "payments" in t:
            return "Bills"
        if "shopping" in t:
            return "Shopping"
        if "transfer" in t:
            return "Transfers"
        return t.capitalize()

    name = (party or "").lower()
    name = re.sub(r'\b(ltd|pvt|private|limited|india|inc|co)\b', '', name)
    for keyword, category in BROAD_CATEGORIES.items():
        if keyword in name:
            return category
    for keyword, category in DETAILED_MAPPING.items():
        if keyword in name:
            return category
    return "Other"

def categorize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "Tag" in df.columns:
        df["Category"] = df.apply(lambda r: map_category(r.get("Party", ""), r.get("Tag", "")), axis=1)
    else:
        df["Category"] = df["Party"].apply(lambda p: map_category(p, ""))
    return df

# -------------------- HF remote inference helpers + fallback --------------------
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

def hf_remote_infer(model_id: str, prompt: str, max_tokens: int = 512) -> str:
    if not HF_API_KEY or client is None:
        raise RuntimeError("HF_API_KEY not set or client unavailable for remote inference.")
    try:
        out = client.text_generation(model=model_id, prompt=prompt, max_new_tokens=max_tokens)
        return _normalize_hf_output(out)
    except HfHubHTTPError:
        raise
    except Exception:
        raise

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
    prompts = build_prompts(df, question)
    outputs = []
    try:
        for name, prompt in prompts.items():
            out = hf_remote_infer(model_id, prompt, max_tokens=512)
            outputs.append(f"--- {name.upper()} ---\n{out}\n")
        return "\n".join(outputs)
    except Exception:
        print("Remote inference failed — falling back to deterministic advice.")
        traceback.print_exc()
        fallback = generate_simple_advice(df, question)
        return ("⚠️ Remote model unavailable or returned an error. Showing fallback advice below.\n\n" + fallback)

# -------------------- Prompts builder --------------------
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

# -------------------- PDF Report
