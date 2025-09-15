# app.py
import os
import re
import io
from typing import Optional, Dict

import pandas as pd
import numpy as np
import streamlit as st
from fpdf import FPDF

from huggingface_hub import InferenceClient 
from concurrent.futures import ThreadPoolExecutor

# -------------------- Config / ENV --------------------
HF_API_KEY = None
try:
    HF_API_KEY = st.secrets.get("huggingface", {}).get("token", os.getenv("HF_API_KEY"))
except Exception:
    HF_API_KEY = os.getenv("HF_API_KEY")
#REMOTE_MISTRAL_ID = "google/flan-t5-base"
#REMOTE_MISTRAL_ID = "tiiuae/falcon-7b-instruct"
#REMOTE_MISTRAL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
REMOTE_MISTRAL_ID = "gpt2"

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
        /* Add white overlay with 80% transparency */
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.8);
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_background("https://static.vecteezy.com/system/resources/previews/019/154/472/large_2x/inflation-illustration-set-characters-buying-food-in-supermarket-and-worries-about-groceries-rising-price-vector.jpg")

st.title("📊 SpendWise — Smart UPI Analyzer")

# -------------------- UTIL: PDF Parsing --------------------
from pypdf import PdfReader

def parse_pdf_bytes(file_bytes: bytes) -> str:
    """
    Extract text from PDF bytes using pypdf (pure Python).
    Returns concatenated page text.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)

# -------------------- UTIL: Transaction Extraction --------------------
DEFAULT_REGEX = re.compile(
    r"(Received|Paid|Credited|Debited|Deposit|Withdrawal)\s*[₹]?\s*([\d,]+\.\d{2}).*?(?:to|from)?\s*([A-Za-z0-9&\-\s\.]+)?\s*\n?.*?([A-Za-z]{3}\s+\d{1,2},\s+\d{4})",
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
            "Amount": float(amount.replace(",", "")),
            "Party": party.strip() if party else "Self",
            "RawDate": date
        })
    df = pd.DataFrame(transactions)
    if df.empty:
        return df
    df = df.dropna(subset=["Date"]).sort_values("Date", ascending=False).reset_index(drop=True)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    return df

def normalize_type(t):
    t = t.lower()
    if t in ["received", "credited", "deposit"]:
        return "Received"
    if t in ["paid", "debited", "withdrawal"]:
        return "Paid"
    return t.capitalize()

# -------------------- CATEGORIZATION --------------------
BROAD_CATEGORIES = {
    # Food & Delivery
    "swiggy": "Food Delivery",
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
}

def map_category(party: str) -> str:
    name = party.lower()

    # Match detailed keyword mapping
    for keyword, category in BROAD_CATEGORIES.items():
        if keyword in name:
            return category

    # Bank transfers
    if "neft" in name or "imps" in name or "rtgs" in name or "transfer" in name:
        return "Bank Transfer"

    # UPI handle detection
    if "@ok" in name or "@upi" in name or "@paytm" in name or "@icici" in name or "@sbi" in name:
        return "UPI Payment"

    # Fallback
    return "Other"

def categorize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["Category"] = df["Party"].apply(map_category)
    return df

from huggingface_hub import InferenceClient
from huggingface_hub.utils._errors import HfHubHTTPError
import traceback

# create client (ensure HF_API_KEY is defined earlier)
client = InferenceClient(token=HF_API_KEY)

def hf_remote_infer(model_id: str, prompt: str, max_tokens: int = 512) -> str:
    """
    Attempts remote inference. On any error, raises the exception up so the caller
    can decide to fallback. We return the text (string) when successful.
    """
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY not set for remote inference.")
    try:
        # Use text_generation (non-streaming)
        out = client.text_generation(
            model=model_id,
            prompt=prompt,
            max_new_tokens=max_tokens
        )
        return out
    except HfHubHTTPError as e:
        # Hugging Face specific HTTP error (404/403/429/etc.)
        # Re-raise so the caller can fallback gracefully
        raise
    except Exception as e:
        # Any other error (timeouts, network issues)
        raise

def generate_simple_advice(df: pd.DataFrame, question: str) -> str:
    """
    Simple deterministic fallback advice generator using dataframe summaries.
    This ensures your app works even if remote LLM is unavailable.
    """
    income = df[df.Type == "Received"].Amount.sum()
    expense = df[df.Type == "Paid"].Amount.sum()
    net = income - expense
    top_cats = df[df.Type == "Paid"].groupby("Category").Amount.sum().sort_values(ascending=False).head(5)

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

    # Question-aware simple advice
    if "save" in question.lower() or "cut" in question.lower():
        lines.append("Actionable tips (fallback):")
        lines.append("- Reduce spending in top categories shown above by 10–20%.")
        lines.append("- Review subscriptions and recurring payments; cancel unused ones.")
        lines.append("- Move small savings into a recurring deposit or sweep account.")
    else:
        lines.append("General tips (fallback):")
        lines.append("- Keep a weekly budget and track daily discretionary spends.")
        lines.append("- Automate 10% of income to savings if possible.")

    return "\n".join(lines)

def run_pipeline_with_remote(df: pd.DataFrame, question: str, model_id: str) -> str:
    """
    Try remote LLM; if it errors (404/403/etc.), fall back to deterministic advice.
    """
    prompts = build_prompts(df, question)
    outputs = []
    # Try-catch per-prompt so partial success is possible (but we fallback to deterministic if any fail)
    try:
        for name, prompt in prompts.items():
            out = hf_remote_infer(model_id, prompt, max_tokens=512)
            outputs.append(f"--- {name.upper()} ---\n{out}\n")
        return "\n".join(outputs)
    except Exception as e:
        # Log error server-side (so you can view in Streamlit Cloud logs)
        print("Remote inference failed — falling back to deterministic advice.")
        traceback.print_exc()

        # Return clear user-visible message + fallback advice
        fallback = generate_simple_advice(df, question)
        return (
            "⚠️ Remote model unavailable or returned an error. Showing fallback advice below.\n\n"
            + fallback
        )



def sanitize_text(text: str) -> str:
    """
    Replace any character outside Latin-1 range with a safe ASCII substitute.
    Keeps basic punctuation and numbers; replaces others with '?' to avoid FPDF errors.
    """
    if text is None:
        return ""
    # Replace non-latin1 with '?'
    return re.sub(r'[^\x00-\xFF]', '?', text)

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
    model_choice = st.sidebar.radio("Choose Model:", ("Remote HuggingFace (fast)",))

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

        question = st.text_area("Ask a financial question:", "Where can I cut spending or save more?")

        if st.button("Generate AI Advice"):
            st.info("Running pipeline... this may take a few seconds.")
            result_text = run_pipeline_with_remote(df, question, REMOTE_MISTRAL_ID)

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











