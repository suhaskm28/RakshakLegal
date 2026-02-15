# RakshakLegal (LegalIndia)

RakshakLegal is an AI-powered legal intelligence system that helps users understand Indian legal cases through **question answering, summarization, and outcome prediction**.
The platform combines **Retrieval-Augmented Generation (RAG)**, **fine-tuned transformer models**, and **explainable AI** to provide context-grounded legal insights.

Designed for law students, researchers, and legal-tech developers, the system demonstrates how modern AI can improve legal accessibility and decision support.

---

## What This System Does

✔ Answer legal questions using real judgments
✔ Summarize complex legal documents
✔ Predict whether a petition may be accepted or rejected
✔ Explain legal reasoning behind predictions
✔ Provide context-backed responses to reduce hallucinations

---

## Core Modules

### ⚖️ 1. Legal QA & Summarization (RAG)

* Retrieves relevant legal judgments from a FAISS vector database
* Uses semantic embeddings to find context
* Injects retrieved context into an LLM prompt
* Generates grounded answers with source references

**Stack**

* SentenceTransformers embeddings
* FAISS vector store
* LangChain RetrievalQA
* Llama-2 (via CTransformers)

---

### 🧠 2. Prediction & Explanation (Generative Reasoning)

* Predicts case outcome
* Generates legal reasoning & explanation
* Uses quantized LLM + LoRA fine-tuned adapter
* Streams explanation in real time

**Stack**

* Quantized transformer model
* PEFT LoRA adapter
* bitsandbytes 4-bit inference

---

### 📤 3. Prediction Only (Fast Classifier)

* Predicts Accepted / Rejected outcome
* Provides confidence score
* Supports TXT, PDF, DOCX input

**Stack**

* Fine-tuned InLegalBERT classifier
* PyTorch inference pipeline

---

## System Architecture & Flow

### 🔎 RAG Legal QA Flow

User Question
→ Sentence Transformer Embedding
→ FAISS Similarity Search
→ Retrieve Relevant Judgments
→ Context injected into prompt
→ LLM generates grounded answer
→ Sources displayed

This ensures answers are based on real legal context.

---

### ⚖️ Legal Prediction & Explanation Flow

Legal Case Input
→ Quantized LLM
→ LoRA fine-tuned adapter
→ Legal reasoning generation
→ Prediction + explanation

Provides explainable AI reasoning.

---

### 📊 Outcome Classification Flow

Legal Document
→ Tokenization
→ InLegalBERT classifier
→ Prediction (Accepted/Rejected)
→ Confidence score

Optimized for fast predictions.

---

## Tech Stack

**Language**

* Python

**AI & NLP**

* HuggingFace Transformers
* LangChain
* Sentence Transformers
* FAISS vector search

**Models**

* Llama-2 (GGML)
* LoRA fine-tuned legal reasoning model
* InLegalBERT outcome classifier

**Interface**

* Streamlit multi-page application

**Optimization**

* bitsandbytes quantization
* PEFT (LoRA)
* GPU/CPU auto device mapping

---

## Project Structure

RakshakLegal/
├── Home.py
├── pages/
│   ├── Legal QA & Summarization (RAG)
│   ├── Prediction & Explanation
│   └── Prediction Only
├── vectorstore/db_faiss/
├── LoRaAdapter/
├── quantized/
├── InLegalBERT model/
└── case_data.json

---

## Run the App

streamlit run Home.py

---

## Who This Is For

* Law students & legal researchers
* Legal-tech developers
* AI/NLP practitioners
* Explainable AI experimentation
* Legal analytics prototyping

---

## Why RAG Matters Here

Legal accuracy requires grounding answers in real judgments.
The RAG pipeline ensures:

✔ context-backed responses
✔ reduced hallucinations
✔ traceable legal sources
✔ trustworthy legal insights

---

## Disclaimer

This system is for educational and research purposes only and does not constitute legal advice.

---
