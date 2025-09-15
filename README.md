# 📊 SpendWise — Personal UPI Usage and Financial Analyzer

SpendWise is a Streamlit app that analyzes UPI/Bank statements and generates personalized financial insights using LLMs (Large Language Models).

It extracts transactions, categorizes spending, builds charts, and uses Hugging Face’s Inference API to provide actionable financial advice.

# 🚀 Features

📄 Upload PDF bank/UPI statements

🔍 Automatic transaction extraction & categorization

📊 Visualize monthly trends and top spending categories

🤖 AI-powered financial advice using Hugging Face Inference API

📥 Download personalized PDF reports and CSV of transactions

🎨 Custom background styling

# ⚡ Tech Stack

Frontend: Streamlit

PDF parsing: pdfplumber, fpdf

Visualization: Matplotlib, Pandas

AI models: Hugging Face (Remote Inference API, default)

Local model (optional): Flan-T5-Small (⚠️ slow, disabled for deployment)

📦 Installation (Local)
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
streamlit run app.py

# 🔑 Secrets Setup

In Streamlit Cloud → Settings → Secrets, add:

[huggingface]
token = "hf_XXXXXXXXXXXXXXXXXXXXXXXX"


👉 Replace with your Hugging Face API token (get one from huggingface.co/settings/tokens
).

# 🌐 Deployment Notes

Default setup uses Hugging Face Inference API (fast).

Local model option exists (flan-t5-small) but is slow and requires torch → commented out in deployment.

Langflow integration was implemented but is commented out to ensure smooth cloud deployment.

Works locally (via localhost), but not in Streamlit Cloud.

Left in the code (commented) for evaluator reference.

# 📝 Usage

Upload your bank/UPI statement PDF.

View extracted transactions and auto-categorized spending.

Explore monthly summary and top categories charts.

Ask financial questions like:

“Where can I save more?”

“Which categories have recurring wasteful spending?”

Download the generated PDF advice report and CSV of transactions.
