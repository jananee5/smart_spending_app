# app.py
import os
import re
import io
import base64
from typing import Optional, Tuple, Any, Dict

import pandas as pd
import numpy as np
import streamlit as st
import pdfplumber
from fpdf import FPDF
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import requests
from concurrent.futures import ThreadPoolExecutor

# -------------------- Config / ENV --------------------
# Hugging Face key from secrets
HF_API_KEY = None
try:
    HF_API_KEY = st.secrets.get("huggingface", {}).get("token", os.getenv("HF_API_KEY"))
except Exception:
    HF_API_KEY = os.getenv("HF_API_KEY")

DEFAULT_MODEL = "flan-t5-small"   # local fallback model (lighter)
REMOTE_MISTRAL_ID = "mistral-inference-model-id"

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
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("https://wallpaperaccess.com/full/3457552.jpg")

st.title("📊 SpendWise — Smart UPI Analyzer")

# -------------------- UTIL: PDF Parsing --------------------
def parse_pdf_bytes(file_bytes: bytes) -> str:
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages = []
        for p in pdf.pages:
            text = p.extract_text() or ""
            pages.append(text)
    return "\n".join(pages)

# -------------------- UTIL: Transaction Extraction --------------------
DEFAULT_REGEX = re.compile(
    r"(Received|Paid|Credited|Debited|Deposit|Withdrawal)\s*[₹]?\s*([\d,]+\.\d{2})\s*(?:to\s(.*?))?\s*\n?.*?([A-Za-z]{3}\s+\d{1,2},\s+\d{4})",
    re.DOTALL
)

def extract_transactions(text: str, regex: Optional[re.Pattern] = None) -> pd.DataFrame:
    if regex is None:
        regex = DEFAULT_REGEX
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
    "amazon": "Shopping", "flipkart": "Shopping", "swiggy": "Food",
    "zomato": "Food", "apollo": "Medical", "pharmacy": "Medical",
    "hospital": "Medical", "uber": "Transport", "ola": "Transport",
    "fuel": "Transport", "grocery": "Groceries", "dmart": "Groceries",
    "reliance": "Groceries", "rent": "Rent", "insurance": "Insurance",
    "phonepe": "Payments", "paytm": "Payments", "google": "Payments",
    "gpay": "Payments",
}

DETAILED_MAPPING = {
    "good to go foodworks private limited": "Food Delivery",
    "pvr inox": "Entertainment: Movies",
    "zepto marketplace": "Groceries",
    "medplus": "Healthcare: Pharmacy",
    "swiggy limited": "Food Delivery",
    "rollbaby": "Food & Dining",
    "phonepe": "Telecom & Recharge",
}

def map_category(party: str) -> str:
    name = party.lower()
    for k, v in BROAD_CATEGORIES.items():
        if k in name:
            return v
    for k, v in DETAILED_MAPPING.items():
        if k in name:
            return v
    return "Other"

def categorize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["Category"] = df["Party"].apply(map_category)
    return df

# -------------------- MODEL LOADING (LOCAL - optional) --------------------
@st.cache_resource(show_spinner=False)
def load_local_flan(model_name: str = DEFAULT_MODEL) -> Tuple[Any, Any]:
    # ⚠️ Local model loading is slow on Streamlit Cloud (2–3 minutes).
    tok = AutoTokenizer.from_pretrained(f"google/{model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(f"google/{model_name}")
    return tok, model

# -------------------- REMOTE INFERENCE --------------------
def hf_remote_infer(model_id: str, prompt: str, max_tokens: int = 512) -> str:
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY not set for remote inference.")
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    output = resp.json()
    if isinstance(output, list) and "generated_text" in output[0]:
        return output[0]["generated_text"]
    if isinstance(output, dict) and "generated_text" in output:
        return output["generated_text"]
    if isinstance(output, str):
        return output
    return str(output)

# -------------------- PIPELINE (Prompts + Run) --------------------
def build_prompts(df: pd.DataFrame, question: str) -> Dict[str, str]:
    income = df[df.Type == "Received"].Amount.sum()
    expense = df[df.Type == "Paid"].Amount.sum()
    net = income - expense

    monthly_summary = (
        df.groupby(["Month", "Type"]).Amount.sum().unstack(fill_value=0).reset_index()
    )
    cat_summary = df[df.Type == "Paid"].groupby("Category").Amount.sum().sort_values(ascending=False).head(10)

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

def run_pipeline_with_local(df: pd.DataFrame, question: str, tokenizer, model, max_new_tokens=256) -> str:
    prompts = build_prompts(df, question)
    outputs = []
    for name, prompt in prompts.items():
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        result = model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = tokenizer.decode(result[0], skip_special_tokens=True)
        outputs.append(f"--- {name.upper()} ---\n{decoded}\n")
    return "\n".join(outputs)

def run_pipeline_with_remote(df: pd.DataFrame, question: str, model_id: str) -> str:
    prompts = build_prompts(df, question)
    outputs = []
    for name, prompt in prompts.items():
        out = hf_remote_infer(model_id, prompt, max_tokens=512)
        outputs.append(f"--- {name.upper()} ---\n{out}\n")
    return "\n".join(outputs)

# -------------------- LANGFLOW (commented out) --------------------
# ⚠️ NOTE: Langflow integration was implemented, but commented out
# for deployment speed. Works locally via localhost, not in cloud.
#
# def get_spendwise_advice(transactions: str, question: str) -> str:
#     if not LANGFLOW_URL:
#         raise RuntimeError("LANGFLOW_URL not configured.")
#     payload = {"flow": "SpendWise","inputs": {"transactions": transactions, "question": question}}
#     resp = requests.post(LANGFLOW_URL, json=payload, timeout=120)
#     resp.raise_for_status()
#     return resp.json().get("advice", "No advice returned.")

# -------------------- REPORT (PDF) --------------------
def create_pdf_report(text: str, df: pd.DataFrame) -> bytes:
    imgs = []
    monthly = df.groupby(["Month", "Type"]).Amount.sum().unstack(fill_value=0)
    if not monthly.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        monthly.plot(kind="bar", ax=ax)
        ax.set_title("Monthly Income vs Expense")
        ax.set_ylabel("₹")
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        imgs.append(buf.read())
    cats = df[df.Type == "Paid"].groupby("Category").Amount.sum().sort_values(ascending=False).head(10)
    if not cats.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        cats.plot(kind="bar", ax=ax)
        ax.set_title("Top Spending Categories")
        ax.set_ylabel("₹")
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        imgs.append(buf.read())
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 7, "SpendWise — Personalized Financial Advice", 0, 1)
    pdf.ln(2)
    for line in text.splitlines():
        pdf.multi_cell(0, 6, line)
    pdf.ln(4)
    for i, img_bytes in enumerate(imgs):
        pdf.add_page()
        img_path = f"/tmp/tmp_chart_{i}.png"
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        pdf.image(img_path, x=10, y=20, w=190)
    return pdf.output(dest="S").encode("utf-8")

# -------------------- UI / App --------------------
def main():
    st.sidebar.header("Settings")
    model_choice = st.sidebar.radio("Choose Model:", ("Remote HuggingFace (fast)", "Local FLAN (slow)"))

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
            st.info("Running pipeline... this may take a few seconds depending on model choice.")

            if model_choice.startswith("Local"):
                st.warning("⚠️ Local model may take 2–3 minutes to load. Use Remote for faster demo.")
                tokenizer, model = load_local_flan()
                result_text = run_pipeline_with_local(df, question, tokenizer, model, 256)
            else:
                if not HF_API_KEY:
                    st.error("HF_API_KEY not set. Unable to use remote Hugging Face inference.")
                    return
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
