import streamlit as st

# --- Page Configuration ---
st.set_page_config(page_title="⚖️ LegalIndia | Home", layout="wide")

# --- Title ---
st.markdown("<h1 style='text-align: center;'>⚖️ Welcome to LegalIndia</h1>", unsafe_allow_html=True)

# --- Subtitle ---
st.markdown("""
<div style='text-align: center; font-size: 18px; margin-bottom: 40px;'>
    An AI-powered assistant for legal prediction, explanation, summarization, and question answering.
</div>
""", unsafe_allow_html=True)

# --- Introduction Section ---
st.header("📌 About LegalIndia")
st.markdown("""
LegalIndia is a smart assistant that leverages transformer-based language models to provide:
- **Legal Case Prediction**: Predict whether a case petition is likely to be accepted or rejected.
- **Legal Explanation**: Understand the rationale behind predictions.
- **Summarization & QA**: Summarize complex legal texts or answer legal questions with clarity.

Whether you're a lawyer, law student, or researcher, LegalIndia can assist you in understanding and analyzing legal documents quickly.
""")

# --- Features Overview ---
st.header("🔍 Key Features")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📘 Legal QA & Summarization")
    st.markdown("Ask legal questions or summarize long legal cases.")

with col2:
    st.subheader("⚖️ Prediction & Explanation")
    st.markdown("Predict outcomes with clear, model-backed explanations.")

with col3:
    st.subheader("📤 Prediction Only")
    st.markdown("Quickly get predictions for legal cases without rationale.")

# --- Navigation Instruction ---
st.markdown("---")
st.markdown("""
👉 Use the **sidebar** to navigate to one of the features.  
Each section has its own interactive interface tailored to the legal task at hand.
""")

st.markdown("<div style='text-align: center;'>Made with ❤️ for Indian legal understanding</div>", unsafe_allow_html=True)
