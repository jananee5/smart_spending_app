import streamlit as st
import pdfplumber
import pandas as pd
import re
import threading
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    TextIteratorStreamer,
)
from fpdf import FPDF

# ------------------------------
# CONFIG
# ------------------------------
HF_TOKEN = st.secrets["huggingface"]["token"]

st.set_page_config(page_title="Smart Spending Coach", page_icon="💰", layout="wide")

# ------------------------------
# THEME & BACKGROUND
# ------------------------------
page_bg_css = """
<style>
[data-testid="stAppViewContainer"] {
  background-image:
    linear-gradient(rgba(0,0,0,0.35), rgba(0,0,0,0.35)),
    url("bg.png");
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
.block-container {
  background: rgba(255, 255, 255, 0.82);
  backdrop-filter: blur(2px);
  border-radius: 14px;
  padding: 1.25rem;
}
.block-container h1 {
  color: #111111;
  letter-spacing: 0.2px;
  margin-bottom: 0.5rem;
}
.block-container h2, .block-container h3 {
  color: #222222;
}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

# ------------------------------
# SESSION STATE
# ------------------------------
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "model" not in st.session_state:
    st.session_state.model = None
if "model_choice" not in st.session_state:
    st.session_state.model_choice = None
if "df" not in st.session_state:
    st.session_state.df = None

# ------------------------------
# MODEL LOADING HELPERS (GPU if available)
# ------------------------------
@st.cache_resource
def load_causal_model(model_id: str):
    device = 0 if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=HF_TOKEN,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None
    ).to(device)
    return tokenizer, model

@st.cache_resource
def load_seq2seq_model(model_id: str):
    device = 0 if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        token=HF_TOKEN,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None
    ).to(device)
    return tokenizer, model

# ------------------------------
# PDF GENERATOR
# ------------------------------
def generate_advice_pdf(text: str) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        for chunk in [line[i:i+90] for i in range(0, len(line), 90)]:
            pdf.cell(0, 6, chunk, ln=True)
    return pdf.output(dest="S").encode("latin-1")

# ------------------------------
# STREAMING GENERATION
# ------------------------------
def stream_generate(model, tokenizer, prompt, max_new_tokens=256):
    device = 0 if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    generation_kwargs = dict(**inputs, max_new_tokens=max_new_tokens, streamer=streamer)
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    for new_text in streamer:
        yield new_text

# ------------------------------
# PDF UPLOAD & PARSING
# ------------------------------
st.header("📂 Upload Your Transaction PDF")
uploaded_pdf = st.file_uploader("Upload your bank/payment statement (PDF)", type="pdf")

if uploaded_pdf is not None:
    with pdfplumber.open(uploaded_pdf) as pdf:
        full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    pattern = re.compile(
        r"(Received|Paid) ₹([\d,]+\.\d{2})"
        r"(?: to (.*?))?\s*\n"
        r"([A-Za-z]{3} \d{1,2}, \d{4}, \d{1,2}:\d{2}:\d{2} [APM]+) GMT"
        r".*?Details:\s*([\w/]+)\s*\n"
        r"(Completed|Failed)",
        re.DOTALL
    )

    matches = pattern.findall(full_text)
    records = []
    for ttype, amt_str, recv, ts_str, txid, status in matches:
        dt = pd.to_datetime(ts_str, format="%b %d, %Y, %I:%M:%S %p", errors="coerce")
        amt = float(amt_str.replace(",", ""))
        recv = recv or "Self (Received)"
        cat = "Income" if ttype == "Received" else "Expense"
        records.append({
            "datetime": dt,
            "transaction_type": ttype,
            "is_income": ttype == "Received",
            "is_expense": ttype == "Paid",
            "amount": amt,
            "merchant": recv,
            "transaction_id": txid,
            "status": status,
            "spend_category": "Other"
        })

    df = pd.DataFrame(records).dropna(subset=["datetime"])
    st.session_state.df = df
    st.success(f"✅ Parsed {len(df)} transactions from your PDF")
    st.dataframe(df.head())

# ------------------------------
# MODEL UI
# ------------------------------
st.title("💰 Smart Spending Coach")

model_choice = st.selectbox(
    "Select model",
    ["mistralai/Mistral-7B-Instruct-v0.1", "google/flan-t5-large"],
    index=0,
)
st.session_state.model_choice = model_choice

load_clicked = st.button("Load model")
if load_clicked:
    with st.spinner(f"Loading '{model_choice}'..."):
        if "Mistral" in model_choice:
            tok, mdl = load_causal_model(model_choice)
        else:
            tok, mdl = load_seq2seq_model(model_choice)
        st.session_state.tokenizer = tok
        st.session_state.model = mdl
    st.success(f"Model '{model_choice}' loaded ✅")

user_input = st.text_area("Enter your question or request for advice:")

# ------------------------------
# GENERATE ADVICE
# ------------------------------
generate_disabled = st.session_state.model is None
gen_clicked = st.button("Generate advice", disabled=generate_disabled)

if gen_clicked:
    if st.session_state.df is None:
        st.warning("Please upload a PDF first.")
    elif not user_input.strip():
        st.warning("Please enter a question.")
    else:
        df = st.session_state.df
        total_spent = df[df.is_expense].amount.sum()
        top_categories = (
            df[df.is_expense]
            .groupby("spend_category").amount.sum()
            .sort_values(ascending=False)
            .head(3)
            .to_dict()
        )
        trend = (
            df[df.is_expense]
            .groupby(df.datetime.dt.to_period("M"))
            .amount.sum()
            .to_dict()
        )
        threshold = df[df.is_expense].amount.quantile(0.75)
        wasteful = df[(df.is_expense) & (df.amount > threshold)][
            ["datetime", "merchant", "amount"]
        ].to_dict(orient="records")

        top2 = list(top_categories.items())[:2]
        top2_cat_str = "\n".join(f"- {cat}: ₹{amt}" for cat, amt in top2)
        months = sorted(trend.keys())
        if len(months) >= 2:
            prev, last = months[-2], months[-1]
            delta = trend[last] - trend[prev]
            trend_delta = f"₹{abs(delta):.2f} {'↑' if delta>0 else '↓'}"
        else:
            trend_delta = "N/A"
        top_waste = sorted(wasteful, key=lambda r: r["amount"], reverse=True)[:3]
        wasteful_str = "\n".join(
            f"- {r['datetime'].strftime('%Y-%m-%d')}: {r['merchant']} (₹{r['amount']})"
            for r in top_waste
        )

        # Build richer prompt for the model
        prompt = f"""
        You are a sharp personal finance coach. Based on the following spending data, give clear, actionable advice.

        Total spent: ₹{total_spent}
        Top categories:
        {top2_cat_str}
        Trend change: {trend_delta}
        High-value transactions:
        {wasteful_str}

        User question: {user_input}

        Instructions:
        1. Give two behavioural insights based on spending patterns.
        2. Suggest one practical way to reduce monthly spend.
        3. Recommend one bold action for long-term financial control.

        Format your response in markdown with clear headings and bullet points.
        """

        st.subheader("💬 Advice (streaming)")
        advice_container = st.empty()
        full_text = ""

        model = st.session_state.model
        tokenizer = st.session_state.tokenizer

        with st.spinner("Thinking..."):
            for chunk in stream_generate(model, tokenizer, prompt, max_new_tokens=512):
                full_text += chunk
                advice_container.write(full_text)

        # PDF download
        pdf_bytes = generate_advice_pdf(full_text)
        st.download_button(
            label="📄 Download advice as PDF",
            data=pdf_bytes,
            file_name="smart_spending_advice.pdf",
            mime="application/pdf",
        )
