import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    TextIteratorStreamer,
)
from fpdf import FPDF
import threading

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
/* Full-app background with subtle dark overlay for contrast */
[data-testid="stAppViewContainer"] {
  background-image:
    linear-gradient(rgba(0,0,0,0.35), rgba(0,0,0,0.35)),
    url("bg.png");
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
}

/* Transparent top bar */
[data-testid="stHeader"] { background: rgba(0,0,0,0); }

/* Make main content readable on image */
.block-container {
  background: rgba(255, 255, 255, 0.82);
  backdrop-filter: blur(2px);
  border-radius: 14px;
  padding: 1.25rem 1.25rem 1.5rem 1.25rem;
}

/* Title styling */
.block-container h1 {
  color: #111111;
  letter-spacing: 0.2px;
  margin-bottom: 0.5rem;
}

/* Subtle accent for subheaders */
.block-container h2, .block-container h3 {
  color: #222222;
}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

# ------------------------------
# SESSION STATE (persist across reruns)
# ------------------------------
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "model" not in st.session_state:
    st.session_state.model = None
if "model_choice" not in st.session_state:
    st.session_state.model_choice = None

# ------------------------------
# MODEL LOADING HELPERS
# ------------------------------
@st.cache_resource
def load_causal_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=HF_TOKEN,
        trust_remote_code=True,
        device_map=None,        # keep on CPU as requested
        low_cpu_mem_usage=True,
    ).to("cpu")
    return tokenizer, model

@st.cache_resource
def load_seq2seq_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        token=HF_TOKEN,
        device_map=None,        # keep on CPU
        low_cpu_mem_usage=True,
    ).to("cpu")
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
    """Yields generated text chunks as they are produced."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cpu")
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    generation_kwargs = dict(**inputs, max_new_tokens=max_new_tokens, streamer=streamer)

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text

# ------------------------------
# APP UI
# ------------------------------
st.title("💰 Smart Spending Coach")

model_choice = st.selectbox(
    "Select model",
    ["mistralai/Mistral-7B-Instruct-v0.1", "google/flan-t5-large"],
    index=0,
)
st.session_state.model_choice = model_choice

load_clicked = st.button("Load model to CPU")

if load_clicked:
    with st.spinner(f"Loading '{model_choice}' to CPU..."):
        if "Mistral" in model_choice:
            tok, mdl = load_causal_model(model_choice)
        else:
            tok, mdl = load_seq2seq_model(model_choice)
        st.session_state.tokenizer = tok
        st.session_state.model = mdl
    st.success(f"Model '{model_choice}' loaded on CPU ✅")

user_input = st.text_area("Enter your spending data or question:")

# Only enable generate if model is loaded
generate_disabled = st.session_state.model is None
gen_clicked = st.button("Generate advice", disabled=generate_disabled)

if gen_clicked:
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        st.subheader("💬 Advice (streaming)")
        advice_container = st.empty()
        full_text = ""

        model = st.session_state.model
        tokenizer = st.session_state.tokenizer

        with st.spinner("Thinking..."):
            for chunk in stream_generate(model, tokenizer, user_input, max_new_tokens=512):
                full_text += chunk
                advice_container.write(full_text)

        pdf_bytes = generate_advice_pdf(full_text)
        st.download_button(
            label="📄 Download advice as PDF",
            data=pdf_bytes,
            file_name="smart_spending_advice.pdf",
            mime="application/pdf",
        )
