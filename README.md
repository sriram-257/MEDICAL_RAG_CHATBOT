# 🏥 Medical RAG Chatbot

A specialized medical Question-Answering chatbot built using the **MedQuAD dataset**. The system uses semantic retrieval with **Sentence Transformers and FAISS** to retrieve relevant medical information and provides a simple interactive interface using **Streamlit**.

The project also includes basic **medical entity recognition** for identifying diseases, symptoms, treatments, and medications from user queries and retrieved medical information.

---

## 📌 Project Overview

This project was developed as a Medical Q&A chatbot using the MedQuAD dataset.

The system follows a retrieval-based approach rather than generating unsupported medical information. A user's question is converted into a semantic embedding and compared against pre-computed embeddings of medical questions stored in a FAISS index.

The most relevant medical Q&A pairs are retrieved and presented through the Streamlit interface.

### Main Components

- MedQuAD medical Q&A dataset
- Sentence Transformer embeddings
- FAISS similarity search
- Semantic retrieval
- TF-IDF retrieval
- Medical entity extraction
- Streamlit user interface
- Cached embeddings and FAISS index

---

## ✨ Features

### 🔎 Semantic Medical Retrieval

User questions are converted into vector embeddings using:

`all-MiniLM-L6-v2`

The embeddings are compared using FAISS similarity search to find semantically related medical questions.

### 🧠 FAISS Vector Search

The project uses a FAISS `IndexFlatIP` index with normalized embeddings.

This allows efficient similarity-based retrieval across the MedQuAD dataset.

### 🏥 Medical Q&A

The chatbot retrieves relevant answers from the MedQuAD knowledge base.

The system does not depend on a generative LLM to invent medical answers.

### 🧬 Medical Entity Recognition

The application identifies basic medical entities such as:

- Diseases
- Symptoms
- Treatments
- Medications

These entities are displayed in the Streamlit interface.

### 📊 Retrieval Results

The application can display retrieved medical questions and their similarity scores, allowing the retrieval process to be inspected.

### 💻 Streamlit Interface

The project provides a simple web interface where users can enter medical questions and receive retrieved medical information.

### ⚡ Cached Semantic Index

The project stores pre-computed:

- Question embeddings
- FAISS index

This prevents the application from regenerating all 14,979 embeddings every time it starts.

---

# 🏗️ System Architecture

```text
                    User
                     │
                     ▼
            ┌─────────────────┐
            │ Streamlit UI    │
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
 │ Medical Answer │   │ Entity Extraction │
 └─────────────────┘   └──────────────────┘
          │                     │
          └──────────┬──────────┘
                     ▼
              ┌─────────────┐
              │ Streamlit   │
              │ Response    │
              └─────────────┘