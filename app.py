import streamlit as st
import pandas as pd
import pdfplumber
import re
from fpdf import FPDF
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="SpendWise: AI Financial Coach", layout="wide")

st.title("💰 SpendWise: AI‑Powered Financial Coach")

# ----------------------
# PDF Upload and Parsing
# ----------------------
def parse_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    return text

def extract_transactions(text):
    pattern = re.compile(
        r"(Received|Paid)\s₹([\d,]+\.\d{2})\s(?:to\s(.*?))?\s*\n.*?(\w{3}\s\d{1,2},\s\d{4})",
        re.DOTALL
    )
    matches = pattern.findall(text)
    transactions = []
    for ttype, amount, party, date in matches:
        transactions.append({
            "Date": pd.to_datetime(date),
            "Type": ttype,
            "Amount": float(amount.replace(",", "")),
            "Party": party or "Self",
            "Category": "Other"
        })
    return pd.DataFrame(transactions)

# ----------------------
# AI Advice Generator
# ----------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

def generate_advice(df, user_question, tokenizer, model):
    total_income = df[df.Type == "Received"].Amount.sum()
    total_expense = df[df.Type == "Paid"].Amount.sum()
    top_expenses = (
        df[df.Type == "Paid"]
        .groupby("Party")
        .Amount.sum()
        .sort_values(ascending=False)
        .head(3)
        .to_dict()
    )

    prompt = f"""
    You are a financial advisor. Analyze this user's spending and income from a bank statement.

    Total Income: ₹{total_income:.2f}
    Total Expenses: ₹{total_expense:.2f}
    Top Expenses:
    {"; ".join([f"{k}: ₹{v:.2f}" for k, v in top_expenses.items()])}

    Question: {user_question}

    Provide 3 personalized pieces of financial advice.
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=256)
    advice = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return advice

# ----------------------
# PDF Report Generator
# ----------------------
def create_pdf_report(advice_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "SpendWise Financial Advice\n\n" + advice_text)
    return pdf.output(dest='S').encode("latin-1")

# ----------------------
# Streamlit UI
# ----------------------
uploaded_pdf = st.file_uploader("📄 Upload your bank statement (PDF)", type="pdf")

if uploaded_pdf:
    st.success("PDF uploaded successfully!")
    raw_text = parse_pdf(uploaded_pdf)
    df = extract_transactions(raw_text)

    if not df.empty:
        st.subheader("📊 Transactions Preview")
        st.dataframe(df.head(10))

        question = st.text_area("🧠 Ask your financial question:", "How can I reduce unnecessary spending?")

        if st.button("💡 Get AI Advice"):
            with st.spinner("Generating advice..."):
                tokenizer, model = load_model()
                advice = generate_advice(df, question, tokenizer, model)
                st.subheader("📋 AI-Powered Financial Advice")
                st.markdown(advice)

                pdf_bytes = create_pdf_report(advice)
                st.download_button(
                    label="📥 Download Advice as PDF",
                    data=pdf_bytes,
                    file_name="financial_advice.pdf",
                    mime="application/pdf"
                )
    else:
        st.error("❌ No valid transactions found in your PDF.")
