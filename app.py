# app.py
import os
import re
import io
import time
import base64
import threading
from functools import partial
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import numpy as np
import streamlit as st
import pdfplumber
from fpdf import FPDF
import matplotlib.pyplot as plt

# Transformers and HTTP for remote inference
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import requests
from concurrent.futures import ThreadPoolExecutor

# -------------------- Config / ENV --------------------
HF_API_KEY = os.getenv("HF_API_KEY")           # for remote Mistral inference (optional)
LANGFLOW_URL = os.getenv("LANGFLOW_URL")      # e.g. http://localhost:8080/execute (optional)
DEFAULT_MODEL = "flan-t5-base"                # local fallback model
REMOTE_MISTRAL_ID = "mistral-inference-model-id"  # replace with actual HF model id if using remote

# ThreadPool for background tasks (keeps UI responsive while working)
executor = ThreadPoolExecutor(max_workers=2)

# -------------------- Streamlit Page Config --------------------
st.set_page_config(page_title="SpendWise — Refactored", layout="wide")

st.title("📊 SpendWise — Refactored: Mistral + FLAN + Langflow-ready")

# -------------------- UTIL: PDF Parsing --------------------
def parse_pdf_bytes(file_bytes: bytes) -> str:
    """Return extracted text from uploaded pdf bytes using pdfplumber."""
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
    """Extract transactions from text. Returns DataFrame with standardized columns."""
    if regex is None:
        regex = DEFAULT_REGEX

    matches = regex.findall(text)
    transactions = []
    for ttype, amount, party, date in matches:
        try:
            parsed_date = pd.to_datetime(date, format="%b %d, %Y", errors="coerce")
        except Exception:
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
    "amazon": "Shopping",
    "flipkart": "Shopping",
    "swiggy": "Food",
    "zomato": "Food",
    "apollo": "Medical",
    "pharmacy": "Medical",
    "hospital": "Medical",
    "uber": "Transport",
    "ola": "Transport",
    "fuel": "Transport",
    "grocery": "Groceries",
    "dmart": "Groceries",
    "reliance": "Groceries",
    "rent": "Rent",
    "insurance": "Insurance",
    "phonepe": "Payments",
    "paytm": "Payments",
    "google": "Payments",
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
    # Extend with more
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

# -------------------- MODEL LOADING (cached) --------------------
@st.cache_resource(show_spinner=False)
def load_local_flan(model_name: str = DEFAULT_MODEL) -> Tuple[Any, Any]:
    """Load local flan model and tokenizer (cached)."""
    tok = AutoTokenizer.from_pretrained(f"google/{model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(f"google/{model_name}")
    return tok, model

# -------------------- REMOTE INFERENCE (Hugging Face Inference) --------------------
def hf_remote_infer(model_id: str, prompt: str, max_tokens: int = 512) -> str:
    """Call HF Inference API (synchronous). Requires HF_API_KEY env var."""
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY not set for remote inference.")
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    output = resp.json()
    # Many HF text models return [{"generated_text": "..."}]
    if isinstance(output, list) and "generated_text" in output[0]:
        return output[0]["generated_text"]
    # Some return text directly
    if isinstance(output, dict) and "generated_text" in output:
        return output["generated_text"]
    # Fallback: try string
    if isinstance(output, str):
        return output
    # If huggingface returns tokens or unusual format:
    return str(output)

# -------------------- LANGFLOW INTEGRATION (if available) --------------------
def call_langflow(flow_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Call a Langflow REST endpoint if LANGFLOW_URL is set.
    Expects the Langflow server to expose an execute endpoint that accepts JSON payload:
    { "flow": "<flow_name_or_json>", "inputs": {..} }
    """
    if not LANGFLOW_URL:
        raise RuntimeError("LANGFLOW_URL not configured.")
    payload = {"flow": flow_name, "inputs": inputs}
    resp = requests.post(LANGFLOW_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()

# -------------------- PIPELINE (3-step) --------------------
def build_prompts(df: pd.DataFrame, question: str) -> Dict[str, str]:
    """Create three prompts:
      - summary_prompt: top-line numbers + monthly table
      - waste_prompt: detect recurring/wasteful spends
      - advice_prompt: actionable steps + address user question
    """
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

Answer concisely (2-3 short bullets) summarizing the user's financial state."""
    waste_prompt = f"""Given the transactions below, identify up to 5 wasteful or recurring expenses (with approximate monthly cost) and explain why they might be optimized. Provide concise bullets.

Transactions (showing sample top rows):
{df.head(30).to_string(index=False)}

Focus on subscriptions, frequent food delivery, or transport pickups."""
    advice_prompt = f"""User Question: {question}

Based on the summary and wasteful spending insights, provide 3 personalized, actionable recommendations for budgeting, saving, and quick wins. Include one suggestion for automating savings and one for reducing recurring charges."""
    return {"summary": summary_prompt, "waste": waste_prompt, "advice": advice_prompt}

def run_pipeline_with_local(df: pd.DataFrame, question: str, tokenizer, model, max_new_tokens=256) -> str:
    """Run the 3-step pipeline locally using the provided tokenizer+model."""
    prompts = build_prompts(df, question)
    all_outputs = []
    for name in ["summary", "waste", "advice"]:
        inputs = tokenizer(prompts[name], return_tensors="pt", truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        all_outputs.append(f"--- {name.upper()} ---\n{decoded}\n")
    return "\n".join(all_outputs)

def run_pipeline_with_remote(df: pd.DataFrame, question: str, model_id: str) -> str:
    """Run pipeline by calling remote HF inference for each prompt."""
    prompts = build_prompts(df, question)
    all_outputs = []
    for name in ["summary", "waste", "advice"]:
        out = hf_remote_infer(model_id, prompts[name], max_tokens=512)
        all_outputs.append(f"--- {name.upper()} ---\n{out}\n")
    return "\n".join(all_outputs)

def run_pipeline_with_langflow(flow_name: str, df: pd.DataFrame, question: str) -> str:
    """Call Langflow to execute a saved flow. Returns flow outputs combined."""
    inputs = {
        "transactions": df.to_dict(orient="records"),
        "question": question
    }
    resp = call_langflow(flow_name, inputs)
    # Expecting resp to be dict with keys like 'summary','waste','advice'
    parts = []
    for k in ("summary", "waste", "advice"):
        if k in resp:
            parts.append(f"--- {k.upper()} ---\n{resp[k]}\n")
    return "\n".join(parts)
def get_spendwise_advice(transactions: str, question: str) -> str:
    """
    Call the specific Langflow SpendWise flow directly and return the AI's advice.
    """
    if not LANGFLOW_URL:
        raise RuntimeError("LANGFLOW_URL not configured in Streamlit secrets.")

    payload = {
        "input_type": "text",
        "output_type": "chat",
        "input_value": "",
        "tweaks": {
            "Prompt-f7hjD": {
                "transactions": transactions,
                "question": question
            }
        }
    }

    resp = requests.post(LANGFLOW_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["outputs"][0]["outputs"][0]["data"]["text"]

# -------------------- REPORT (PDF) --------------------
def create_pdf_report(text: str, df: pd.DataFrame) -> bytes:
    """Create a PDF with the advice and charts embedded (returns bytes)."""
    # Create charts to images in-memory
    imgs = []
    # Monthly bar chart
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

    # Category chart
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

    # Build PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 7, "SpendWise — Personalized Financial Advice", 0, 1)
    pdf.ln(2)
    for line in text.splitlines():
        pdf.multi_cell(0, 6, line)
    pdf.ln(4)
    for img_bytes in imgs:
        pdf.add_page()
        img_path = "/tmp/tmp_chart.png"
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        # Fit image to page with margins
        pdf.image(img_path, x=10, y=20, w=190)
    return pdf.output(dest="S").encode("latin-1")

# -------------------- UI / App --------------------
def main():
    st.sidebar.header("Settings")
    model_choice = st.sidebar.radio("Model (local vs remote):", ("Local Flan-T5 (fast)", "Remote Mistral (HF)"))
    use_langflow = st.sidebar.checkbox("Use Langflow flow (if LANGFLOW_URL configured)", value=False)
    st.sidebar.markdown("Set HF_API_KEY and LANGFLOW_URL as environment variables for remote features.")

    uploaded = st.file_uploader("Upload bank / UPI statement (PDF)", type=["pdf"])
    raw_text = None
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

        # Action button triggers pipeline in background thread (but we wait with spinner)
        if st.button("Generate AI Advice"):
            st.info("Running pipeline... this may take 10-60s depending on model choice.")
            future = None

            if use_langflow and LANGFLOW_URL:
                # Build transactions string for Langflow
                transactions_text = "; ".join(
                f"{row['Type']} ₹{row['Amount']} to {row['Party']} on {row['Date'].strftime('%Y-%m-%d')}"
                for _, row in df.iterrows()
                )
                future = executor.submit(get_spendwise_advice, transactions_text, question)
            
            else:
                if model_choice.startswith("Local"):
                    # Load cached local flan
                    tokenizer, model = load_local_flan("flan-t5-base")
                    # run in thread
                    future = executor.submit(run_pipeline_with_local, df, question, tokenizer, model, 256)
                else:
                    # remote mistral
                    if not HF_API_KEY:
                        st.error("HF_API_KEY not set. Unable to use remote Mistral.")
                        return
                    future = executor.submit(run_pipeline_with_remote, df, question, REMOTE_MISTRAL_ID)

            with st.spinner("Generating advice..."):
                try:
                    result_text = future.result(timeout=180)  # block while still showing spinner
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    return

            # display results
            st.markdown("### 📋 Financial Advice (AI)")
            st.write(result_text)

            # Provide download buttons for report and CSV
            pdf_bytes = create_pdf_report(result_text, df)
            st.download_button("Download PDF Report", data=pdf_bytes, file_name="spendwise_report.pdf", mime="application/pdf")

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download transactions CSV", data=csv_bytes, file_name="transactions.csv", mime="text/csv")

    else:
        st.info("Upload a UPI or bank statement PDF to begin.")

if __name__ == "__main__":
    main()

