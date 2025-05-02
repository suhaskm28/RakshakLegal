import streamlit as st
import sys

# ========== Interface Mode ==========
def is_interface_mode():
    return "interface" in sys.argv

INTERFACE_MODE = is_interface_mode()

# ========== Imports for RAG ==========
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# ========== Prompt Template ==========
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# ========== Model and Retrieval ==========
DB_FAISS_PATH = "vectorstore/db_faiss"

def load_llm():
    return CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )

def retrieval_qa_chain(llm, prompt, db):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    prompt = set_custom_prompt()
    return retrieval_qa_chain(llm, prompt, db)

def final_result(query):
    qa = qa_bot()
    return qa({"query": query})

# ========== Streamlit Chat UI ==========
st.set_page_config(page_title="Legal Chatbot", page_icon="⚖️")
st.title("⚖️ Indian Legal Chatbot")
st.markdown("Ask legal questions about Indian law and get helpful answers based on real judgments.")

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box (ChatGPT style)
user_query = st.chat_input("Type your legal question here...")

if user_query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Processing..."):
            if INTERFACE_MODE:
                answer = "**[Interface Mode]** This is a mock response. QA system is skipped."
                sources = []
            else:
                try:
                    result = final_result(user_query)
                    answer = result["result"]
                    sources = result.get("source_documents", [])
                except Exception as e:
                    answer = f"⚠️ Error: {str(e)}"
                    sources = []

        st.markdown(answer.strip())

        if sources:
            with st.expander("📚 View Sources"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Source {i+1}:**\n\n{doc.page_content.strip()}\n")

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer.strip()})
