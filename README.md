# 🏥 Medical RAG Chatbot

A specialized medical Question-Answering chatbot built using the **MedQuAD dataset**.

The system uses **Sentence Transformers and FAISS** for semantic retrieval of relevant medical information and provides an interactive web interface using **Streamlit**.

The project also includes basic **medical entity recognition** for identifying diseases, symptoms, treatments, and medications from user queries and retrieved medical information.

---

## 📌 Project Overview

This project implements a retrieval-based Medical Q&A chatbot using the **MedQuAD dataset**.

Instead of generating unsupported medical information, the system retrieves relevant answers from the MedQuAD knowledge base.

The user's question is converted into a semantic embedding using a Sentence Transformer model. The embedding is then compared against the medical question embeddings using FAISS similarity search.

The most relevant medical Q&A pairs are retrieved and displayed through the Streamlit interface.

---

## ✨ Features

### 🔎 Semantic Medical Retrieval

User questions are converted into vector embeddings using:

`all-MiniLM-L6-v2`

The embeddings are normalized and searched using FAISS similarity search to identify semantically related medical questions.

### 🧠 FAISS Vector Search

The project uses:

`faiss.IndexFlatIP`

with normalized embeddings for similarity-based retrieval across the MedQuAD dataset.

### 🏥 Medical Q&A

The chatbot retrieves relevant medical answers directly from the MedQuAD knowledge base.

The retrieval system does not use a generative LLM to invent medical answers.

### 🧬 Medical Entity Recognition

The application performs basic medical entity extraction for:

- Diseases
- Symptoms
- Treatments
- Medications

The detected entities are displayed in the Streamlit interface.

### 📊 Retrieval Results

The system can display retrieved medical questions and similarity scores, making the retrieval process more transparent.

### 💻 Streamlit Interface

A simple interactive web interface allows users to enter medical questions and receive relevant information from the knowledge base.

### ⚡ Cached Semantic Index

The application supports cached question embeddings and a FAISS index.

This avoids regenerating all **14,979 question embeddings** every time the application starts.

---

# 🏗️ System Architecture

```text
                    User
                     │
                     ▼
            ┌─────────────────┐
            │  Streamlit UI   │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ User Question   │
            └────────┬────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Query Embedding     │
          │ SentenceTransformer │
          └──────────┬──────────┘
                     │
                     ▼
             ┌───────────────┐
             │ FAISS Search  │
             └───────┬───────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Relevant MedQuAD    │
          │ Q&A Results         │
          └──────────┬──────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
 ┌─────────────────┐   ┌──────────────────┐
 │ Medical Answer  │   │ Entity Extraction│
 └─────────────────┘   └──────────────────┘
          │                     │
          └──────────┬──────────┘
                     ▼
              ┌─────────────┐
              │  Streamlit  │
              │  Response   │
              └─────────────┘
