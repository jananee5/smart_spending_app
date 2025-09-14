import streamlit as st
import pandas as pd
import pdfplumber
import re
from fpdf import FPDF
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="SpendWise: Smart Financial Advisor", layout="wide")

# -------------------- Background Styling --------------------
st.markdown(
    """
    <style>
    .main {
        background-image: url("https://images.unsplash.com/photo-1556745757-8d76bdb6984b");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }
    div.stButton > button {
        color: white;
        background-color: #ff4b4b;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- PDF Upload and Parsing --------------------
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
        try:
            parsed_date = pd.to_datetime(date, format="%b %d, %Y")
        except:
            parsed_date = pd.to_datetime(date, errors="coerce")

        transactions.append({
            "Date": parsed_date,
            "Type": ttype,
            "Amount": float(amount.replace(",", "")),
            "Party": party or "Self",
        })

    df = pd.DataFrame(transactions)

    # Drop rows with invalid dates (NaT)
    df = df.dropna(subset=["Date"])

    return categorize_transactions(df)


# -------------------- Categorization with Deep Subcategories --------------------
def categorize_transactions(df):
    # Broad categories first
    categories = {
        "Amazon": "Shopping",
        "Flipkart": "Shopping",
        "Swiggy": "Food",
        "Zomato": "Food",
        "Apollo": "Medical",
        "Pharmacy": "Medical",
        "Hospital": "Medical",
        "Uber": "Transport",
        "Ola": "Transport",
        "Fuel": "Transport",
        "Grocery": "Groceries",
        "Dmart": "Groceries",
        "Reliance": "Groceries",
        "Rent": "Rent",
        "Insurance": "Insurance"
    }

    # Your detailed deep sub-category mapping for 'Other'
    detailed_mapping = {
        "Good To Go Foodworks Private Limited": "Food Delivery",
        "Thiruvalluvar Salai Mugappair East": "Shopping",
        "Pvr Inox Limited": "Entertainment: Movies",
        "Zepto Marketplace Private Limited": "Groceries",
        "M/S.Ligo Pharmacy": "Healthcare: Pharmacy",
        "Swiggy Limited": "Food Delivery",
        "Aladiyan Karupatti Shop": "Groceries",
        "Medplus Padi Kumaran Nagar": "Healthcare: Pharmacy",
        "Punitha Rani R": "Personal Transfer",
        "M/S.Saravana Egg Mart": "Groceries",
        "Rollbaby Roll Vr Mall": "Food & Dining",
        "Phonepe": "Telecom & Recharge",
        "M Jayabharathi": "Personal Transfer",
        # Add more mappings here if needed
    }

    def map_category(party):
        # First try broad categories
        for key, val in categories.items():
            if key.lower() in party.lower():
                return val

        # If no match, try detailed mapping for better subcategories
        for key, val in detailed_mapping.items():
            if key.lower() in party.lower():
                return val

        # Default fallback
        return "Other"

    df["Category"] = df["Party"].apply(map_category)
    df["Month"] = df["Date"].dt.strftime("%Y-%m")
    return df

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

# -------------------- Generate Advice --------------------
def generate_advice(df, question, tokenizer, model):
    income = df[df.Type == "Received"].Amount.sum()
    expense = df[df.Type == "Paid"].Amount.sum()
    net = income - expense

    monthly_summary = (
        df.groupby(["Month", "Type"])
        .Amount.sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    category_summary = (
        df[df.Type == "Paid"]
        .groupby("Category")
        .Amount.sum()
        .sort_values(ascending=False)
        .head(10)  # show top 10 categories for better detail
    )

    prompt = f"""
Analyze the user's financial behavior based on the data below.

Total Income: ₹{income:.2f}
Total Expenses: ₹{expense:.2f}
Net Balance: ₹{net:.2f}

Monthly Breakdown:
{monthly_summary.to_string(index=False)}

Top Spending Categories:
{category_summary.to_string()}

Question: {question}

Provide three actionable, personalized financial insights.
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# -------------------- PDF Export --------------------
def create_pdf_report(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    return pdf.output(dest="S").encode("latin-1")

# -------------------- UI Layout --------------------
st.title("📊 SpendWise — AI-Powered Financial Coach")

uploaded = st.file_uploader("📄 Upload your bank statement (PDF)", type="pdf")

if uploaded:
    text = parse_pdf(uploaded)
    df = extract_transactions(text)

    if df.empty:
        st.error("No valid transactions found.")
    else:
        st.success(f"✅ Parsed {len(df)} transactions.")
        st.dataframe(df.head())

        with st.expander("📈 Monthly Summary"):
            monthly = df.groupby(["Month", "Type"]).Amount.sum().unstack(fill_value=0)
            st.bar_chart(monthly)

        with st.expander("📂 Spending by Category"):
            categories = df[df.Type == "Paid"].groupby("Category").Amount.sum().sort_values(ascending=False)
            st.bar_chart(categories)

        question = st.text_area("🧠 Ask a financial question:", "Where can I cut spending or save more?")

        if st.button("💡 Generate AI Advice"):
            with st.spinner("Analyzing your finances..."):
                tokenizer, model = load_model()
                advice = generate_advice(df, question, tokenizer, model)
                st.markdown("### 📋 Financial Advice")
                st.write(advice)

                pdf = create_pdf_report(advice)
                st.download_button("📥 Download as PDF", data=pdf, file_name="financial_advice.pdf")
