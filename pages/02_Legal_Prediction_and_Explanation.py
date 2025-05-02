import json
from transformers import TextStreamer
import streamlit as st
import time
import sys
# --- Development Mode Switch ---
DEV_MODE = "interface" in sys.argv

# --- Model Loading (Only when not in DEV mode) ---
if not DEV_MODE:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, PeftConfig
    import os
    import bitsandbytes as bnb

    base_model_path = "quantized"          # Local path to the 4-bit quantized base model
    lora_adapter_path = "LoRaAdapter"      # Local path to the fine-tuned LoRA adapter

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # Load quantized base model (4-bit)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",                      # Multi-GPU/Auto CPU-GPU assignment
        load_in_4bit=True,                      # Enable 4-bit loading via bitsandbytes
        torch_dtype=torch.float16,             # Efficient dtype
        trust_remote_code=True                 # Set to True if using custom architectures
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        model,
        lora_adapter_path,
        torch_dtype=torch.float16
    )

    model.eval()


# --- Page Configuration ---
st.set_page_config(page_title="⚖️ LegalIndia | Predict & Explain", layout="wide")

# --- Sidebar: Inference Settings ---
st.sidebar.title("🔧 Inference Settings")

# Temperature slider (0.0 to 1.0 for random sampling)
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.5,
    value=0.7,
    step=0.1,
    help="Controls randomness. Lower = more deterministic output."
)

# Top-p (nucleus sampling) slider (0.0 to 1.0 for limiting probability)
top_p = st.sidebar.slider(
    "Top-p (nucleus sampling)",
    min_value=0.0,
    max_value=1.0,
    value=0.9,
    step=0.05,
    help="Limits sampling to top cumulative probability tokens."
)

# Max Tokens slider (define the maximum tokens for output)
max_tokens = st.sidebar.slider(
    "Max New Tokens",
    min_value=64,
    max_value=512,
    value=256,
    step=64,  # Standard step values (64, 128, 192, 256, 320, 384, 448, 512)
    help="Maximum length of generated explanation."
)
do_sample = st.sidebar.checkbox(
    "Use Sampling",
    value=True,
    help="Enable sampling during generation. When disabled, the model uses greedy decoding (always picks highest probability next token). Recommended to keep ON for diverse, human-like responses."
)

st.sidebar.markdown("---")
st.sidebar.info("These settings control how the AI generates predictions.")

# --- Title and Subtitle ---
st.markdown("""
    <h2 style='text-align: center;'>⚖️ LegalIndia: Legal Case Prediction & Explanation</h2>
    <p style='text-align: center; font-size:16px;'>Paste a legal case or select a sample. The AI will predict whether it is accepted or rejected and explain the reasoning.</p>
""", unsafe_allow_html=True)

# --- Load Case Data ---
try:
    with open("case/case_data.json", "r") as f:
        case_data = json.load(f)
except Exception as e:
    st.error(f"Failed to load case_data.json: {e}")
    st.stop()

# --- Case Loader UI ---
st.markdown("### 📂 Load Sample Case")

col1, col2 = st.columns(2)
if col1.button("🏛️ Supreme Court Case"):
    st.session_state["selected_court"] = "Supreme Court"
if col2.button("🏢 High Court Case"):
    st.session_state["selected_court"] = "High Court"

# --- Reset Option ---
if "selected_court" in st.session_state:
    st.markdown(f"**Selected Court:** 🏷️ {st.session_state['selected_court']}")

    if st.button("🔁 Reset Selection"):
        st.session_state.pop("selected_court", None)
        st.session_state.pop("case_input", None)
        st.rerun()

    court_type = st.session_state["selected_court"]
    year = st.selectbox("📅 Select Year", list(case_data[court_type].keys()), key=f"{court_type}_year")
    cases = case_data[court_type][year]
    case_names = ["-- Select a Case --"] + [c["case_name"] for c in cases]
    
    selected_case_name = st.selectbox("📜 Select Case", case_names, key=f"{court_type}_case")

    if selected_case_name != "-- Select a Case --":
        selected_case = next(c for c in cases if c["case_name"] == selected_case_name)
        case_text = selected_case["case_details"]

        with st.expander("🔍 Preview Case Details", expanded=False):
            st.markdown(f"**Case Name:** {selected_case_name}")
            st.markdown(f"```{case_text}```")

        # Load selected case into input box
        st.session_state["case_input"] = case_text


# --- Main Input Area ---
st.markdown("### 📝 Enter Your Legal Case")
# Pre-fill case input with session state value
# Ensure default is initialized before the widget renders
if "case_input" not in st.session_state:
    st.session_state["case_input"] = ""

st.text_area(
    "Input the full legal case description (facts, issues, arguments):",
    height=300,
    key="case_input"
)


# --- Predict Button ---
submit = st.button("🔍 Predict & Explain")

# --- Custom Streamer Class ---
class StreamToStreamlit(TextStreamer):
    def __init__(self, tokenizer, prediction_placeholder, explanation_placeholder):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.buffer = ""
        self.tokenizer = tokenizer
        self.prediction_placeholder = prediction_placeholder
        self.explanation_placeholder = explanation_placeholder

    def on_text(self, text, **kwargs):
        # Decode the streamed token text
        decoded_text = self.tokenizer.decode(self.tokenizer.encode(text), skip_special_tokens=True)
        self.buffer += decoded_text
        
        # Split the text into lines
        lines = self.buffer.strip().split("\n")
        
        # Prediction result (first line)
        prediction = lines[0] if lines else "..."
        
        # Explanation (full response)
        explanation = self.buffer.strip()

        # Update the prediction and explanation in the Streamlit placeholders
        self.prediction_placeholder.markdown(f"### 📤 Prediction Result\n\n**{prediction}**")
        with st.expander("🧠 Explanation Rationale", expanded=True):
            self.explanation_placeholder.markdown(f"```\n{explanation}\n```")


# --- Inference Block ---
if not DEV_MODE and submit and st.session_state["case_input"].strip():
    with st.spinner("Generating legal prediction and explanation..."):
        prompt = (
            f"You are a legal assistant. Given the following case, predict whether the petition should be "
            f"accepted or rejected, and explain your reasoning.\n\nCase: {st.session_state['case_input'].strip()}\n\nPredict and explain."
        )

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Create placeholders for prediction and explanation
            prediction_placeholder = st.empty()
            explanation_placeholder = st.empty()

            # Initialize the custom Streamer
            streamer = StreamToStreamlit(tokenizer, prediction_placeholder, explanation_placeholder)

            # Generate with streaming
            model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Save final output to session state
            st.session_state.response = streamer.buffer.strip()

        except Exception as e:
            st.error(f"Inference failed: {e}")


# --- Output ---
if "response" in st.session_state:
    st.markdown("### 📤 Prediction Result")
    st.success(st.session_state.response.split("\n")[0])  # Assume first line is prediction

    with st.expander("🧠 Explanation Rationale", expanded=True):
        st.markdown(f"```\n{st.session_state.response}\n```")

    # --- Auto-Scroll to the output area ---
    st.markdown("""
        <script>
        window.scrollTo(0, document.body.scrollHeight);
        </script>
    """, unsafe_allow_html=True)
