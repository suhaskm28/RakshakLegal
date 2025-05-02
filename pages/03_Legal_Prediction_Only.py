import streamlit as st
import sys
import os
import torch
import time
import psutil
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# --- DEV MODE Toggle ---
DEV_MODE = "interface" in sys.argv
if DEV_MODE:
    st.warning("🛠️ DEV MODE ENABLED — Model not loaded.")

# App start time
if "start_time" not in st.session_state:
    st.session_state["start_time"] = datetime.now()

# Sidebar system monitor
with st.sidebar:
    st.markdown("### 📊 Live System Monitor")
    mem_placeholder = st.empty()
    cpu_placeholder = st.empty()
    core_placeholder = st.empty()
    uptime_placeholder = st.empty()
    core_placeholder.markdown(f"🧩 CPU Cores: **{psutil.cpu_count(logical=True)}**")


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # in MB


if not DEV_MODE:
    # Load model and tokenizer only in Production Mode
    MODEL_PATH = "InLegalBertDeep4\\InLegalBertDeep4"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    @st.cache_resource
    def load_model():
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        model.eval()
        return tokenizer, model

    tokenizer, model = load_model()

    def predict_outcome(text):
        encoded = tokenizer(text, return_tensors="pt", truncation=True,
                            padding=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = F.softmax(logits, dim=-1).squeeze()
        predicted_label = "Accepted" if torch.argmax(probs) == 1 else "Rejected"
        confidence = round(probs[torch.argmax(probs)].item(), 4)
        return predicted_label, confidence, probs.tolist()

# App UI
st.title("⚖️ Legal Case Outcome Predictor")
st.markdown("Predict whether a legal case document is likely to be **Accepted** or **Rejected**.")

uploaded_file = st.file_uploader(
    "Upload a file containing legal case documents (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        import fitz
        text = ""
        with fitz.open(uploaded_file) as doc:
            for page in doc:
                text += page.get_text()
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        from docx import Document
        doc = Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])

    st.subheader("📝 Document Preview")
    st.text_area("Document Preview", text[:2000], height=300)

    if DEV_MODE:
        st.info("🔍 Prediction disabled in DEV MODE.")
    else:
        predicted_label, confidence, probs = predict_outcome(text)
        st.subheader("🔎 Prediction Result")
        st.markdown(f"**{predicted_label}** with **{confidence * 100:.2f}%** confidence.")
        st.markdown(f"📊 Probabilities: `{[round(p, 6) for p in probs]}`")

    st.subheader("🔍 Explanation / Highlights (Future RAG integration)")
    st.markdown("This section would highlight the key parts of the document based on attention scores or proof sentences once the RAG pipeline is integrated.")
    st.write("Currently, no specific highlights or explanations are provided, but this will be extended in the future.")

else:
    user_input = st.text_area("📄 Enter your legal case document or summary here:")
    if st.button("🔍 Predict", key="text_predict_button"):
        if not user_input.strip():
            st.warning("Please enter some text before predicting.")
        elif DEV_MODE:
            st.info("🔍 Prediction disabled in DEV MODE.")
        else:
            predicted_label, confidence, probs = predict_outcome(user_input)
            st.subheader("🔎 Prediction")
            st.markdown(f"**{predicted_label}** with **{confidence * 100:.2f}%** confidence.")
            st.markdown(f"📊 Probabilities: `{[round(p, 6) for p in probs]}`")

# Sidebar live system info
with st.sidebar:
    process = psutil.Process(os.getpid())
    while True:
        mem = process.memory_info().rss / (1024 * 1024)
        cpu = psutil.cpu_percent(interval=1)
        uptime = str(datetime.now() - st.session_state["start_time"]).split(".")[0]
        mem_placeholder.markdown(f"🧠 RAM Used: **{mem:.2f} MB**")
        cpu_placeholder.markdown(f"⚙️ CPU Usage: **{cpu:.1f}%**")
        uptime_placeholder.markdown(f"⏱️ Uptime: **{uptime}**")
        time.sleep(2)
        st.rerun()
