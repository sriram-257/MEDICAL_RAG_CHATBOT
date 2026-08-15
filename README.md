# 🏥 Medical RAG Chatbot

A specialized medical Question-Answering chatbot built using the **MedQuAD dataset**.

The system uses **Sentence Transformers and FAISS** for semantic retrieval of relevant medical information and provides an interactive web interface using **Streamlit**.

The project also includes basic **medical entity recognition** for identifying diseases, symptoms, treatments, and medications from user queries and retrieved medical information.

---

## 📌 Project Overview

This project implements a retrieval-based Medical Q&A chatbot using the **MedQuAD dataset**.

Instead of generating unsupported medical information, the system retrieves relevant answers from the MedQuAD knowledge base.

The user's question is converted into a semantic embedding using the **Sentence Transformer** model `all-MiniLM-L6-v2`.

The resulting embedding is compared against medical question embeddings using **FAISS similarity search**.

The most relevant medical Q&A pairs are then retrieved and displayed through the Streamlit interface.

---

## ✨ Features

### 🔎 Semantic Medical Retrieval

User questions are converted into vector embeddings using:

```text
all-MiniLM-L6-v2
````

The embeddings are normalized and searched using FAISS similarity search to identify semantically related medical questions.

---

### 🧠 FAISS Vector Search

The project uses:

```text
faiss.IndexFlatIP
```

with normalized embeddings for similarity-based retrieval across the MedQuAD dataset.

---

### 🏥 Medical Q&A

The chatbot retrieves relevant medical answers directly from the MedQuAD knowledge base.

The retrieval system does not use a generative LLM to invent medical answers.

---

### 🧬 Medical Entity Recognition

The application performs basic medical entity extraction for:

* 🦠 Diseases
* 🤒 Symptoms
* 💊 Treatments
* 💉 Medications

The detected entities are displayed in the Streamlit interface.

---

### 🎯 Intent Detection

The system also performs basic intent detection for medical queries.

For example:

```text
What are the symptoms of diabetes?
```

can be classified as:

```text
Symptoms
```

This provides additional context about the user's question.

---

### 📊 Retrieval Results

The application displays information such as:

* Match confidence score
* Matched MedQuAD question
* Retrieved answer
* Detected intent
* Detected medical entities
* Other relevant matches

This makes the retrieval process more transparent.

---

### 💻 Streamlit Interface

The interactive web interface allows users to:

* Ask medical questions
* View retrieved answers
* View similarity/confidence scores
* View detected entities
* View detected intent
* Inspect other relevant matches
* Maintain conversation history
* Clear the conversation

---

### ⚡ Cached Semantic Index

The project includes pre-computed semantic retrieval files:

```text
data/index/embeddings.npy
data/index/medquad.faiss
```

These files contain the question embeddings and FAISS vector index generated for the processed MedQuAD dataset.

Therefore, the application can load the existing semantic index instead of regenerating all question embeddings every time it starts.

The current cached index contains:

```text
14,979 vectors
384 dimensions
```

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
                ┌──────────────────────┐
                │ Sentence Transformer │
                │  all-MiniLM-L6-v2    │
                └──────────┬───────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Query Embedding │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   FAISS Search  │
                  │   IndexFlatIP   │
                  └────────┬────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Relevant MedQuAD    │
                │ Q&A Results         │
                └──────────┬──────────┘
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
    ┌──────────────────┐       ┌──────────────────┐
    │ Medical Answer   │       │ Entity Extraction│
    └────────┬─────────┘       └─────────┬────────┘
             │                           │
             │                    ┌──────┴──────┐
             │                    │             │
             │                    ▼             ▼
             │                Diseases      Symptoms
             │
             │                Treatments    Medications
             │
             └──────────────────┬────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Streamlit     │
                       │    Response     │
                       └─────────────────┘
```

---

# 📂 Project Structure

```text
MEDICAL_RAG_CHATBOT/
│
├── data/
│   ├── index/
│   │   ├── embeddings.npy
│   │   └── medquad.faiss
│   │
│   └── processed/
│       └── medquad.csv
│
├── src/
│   ├── app.py
│   ├── chatbot.py
│   ├── data_loader.py
│   ├── dataset_analysis.py
│   ├── entity_extractor.py
│   ├── semantic_retriever.py
│   └── tfidf_retriever.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

### Important Data Files

#### `data/processed/medquad.csv`

Processed MedQuAD question-answer dataset used by the application.

#### `data/index/embeddings.npy`

Pre-computed Sentence Transformer embeddings for the MedQuAD questions.

#### `data/index/medquad.faiss`

FAISS vector index containing the pre-computed question embeddings.

These cached files allow the application to start without regenerating all 14,979 question embeddings.

---

# 📚 Dataset

This project uses the **MedQuAD (Medical Question Answering Dataset)**.

Official repository:

[https://github.com/abachaa/MedQuAD](https://github.com/abachaa/MedQuAD)

The processed dataset used by the application contains approximately:

```text
14,979 Q&A pairs
```

The application expects the processed dataset at:

```text
data/processed/medquad.csv
```

The dataset provides medical questions and corresponding answers collected from medical information sources.

---

# 🛠️ Technologies Used

| Technology            | Purpose                             |
| --------------------- | ----------------------------------- |
| Python                | Core programming language           |
| Pandas                | Dataset loading and processing      |
| NumPy                 | Numerical operations and embeddings |
| Scikit-learn          | TF-IDF retrieval                    |
| Sentence Transformers | Semantic embeddings                 |
| FAISS                 | Vector similarity search            |
| spaCy                 | Basic medical entity processing     |
| Streamlit             | Interactive web interface           |
| MedQuAD               | Medical Q&A knowledge base          |

---

# 🔬 How the System Works

## Step 1 — Load the Dataset

The processed MedQuAD dataset is loaded using Pandas.

```text
data/processed/medquad.csv
```

---

## Step 2 — Load the Sentence Transformer

The system loads:

```text
all-MiniLM-L6-v2
```

The model converts medical questions into numerical vector representations.

---

## Step 3 — Load Cached Embeddings

The application checks whether pre-computed embeddings are available.

```text
data/index/embeddings.npy
```

If available, the existing embeddings are loaded.

The cached embeddings have:

```text
14,979 questions
384 dimensions
```

---

## Step 4 — Load Cached FAISS Index

The application loads:

```text
data/index/medquad.faiss
```

The FAISS index contains the pre-computed question vectors.

This avoids rebuilding the complete vector index during normal application startup.

---

## Step 5 — Normalize Embeddings

The question embeddings are normalized before similarity search.

This allows inner-product similarity to be used effectively for semantic retrieval.

---

## Step 6 — Convert User Query

When a user enters a medical question, the same Sentence Transformer model converts the query into an embedding.

---

## Step 7 — Retrieve Relevant Questions

FAISS compares the query embedding against the MedQuAD question embeddings.

The system retrieves the top relevant results.

---

## Step 8 — Apply Relevance Threshold

Weak semantic matches can be filtered using a similarity threshold.

The current default threshold is:

```text
0.45
```

---

## Step 9 — Extract Medical Entities

The application identifies basic medical entities from the query and retrieved information.

Entities include:

```text
Diseases
Symptoms
Treatments
Medications
```

---

## Step 10 — Detect Query Intent

The application performs basic intent detection.

Example:

```text
What are the symptoms of diabetes?
```

Intent:

```text
Symptoms
```

Possible categories include:

```text
Symptoms
Treatment
Medication
Disease
```

---

## Step 11 — Display the Answer

The best relevant MedQuAD answer is displayed through the Streamlit interface along with retrieval information and detected medical entities.

---

# 🔄 Retrieval Methods

The project includes two retrieval approaches.

## Semantic Retrieval

Implemented in:

```text
src/semantic_retriever.py
```

Uses:

```text
Sentence Transformers
        +
FAISS
```

This provides semantic similarity-based retrieval.

---

## TF-IDF Retrieval

Implemented in:

```text
src/tfidf_retriever.py
```

TF-IDF provides a lightweight keyword-based retrieval method.

The Streamlit application allows the retrieval method to be selected from the sidebar.

Available methods:

```text
TF-IDF
```

and:

```text
Semantic (Sentence-Transformers)
```

---

# 🎯 Medical Intent Detection

The application performs basic intent detection to understand the type of medical information requested.

Example:

```text
What are the symptoms of diabetes?
```

Intent:

```text
Symptoms
```

Other supported medical intents can include:

```text
Symptoms
Treatment
Medication
Disease
```

depending on the query and entity extraction logic.

---

# 🧬 Medical Entity Recognition

The application performs basic medical entity extraction.

The system identifies categories including:

```text
Diseases
Symptoms
Treatments
Medications
```

The detected entities are displayed in separate sections within the Streamlit interface.

Example:

```text
Diseases:
• Diabetes
• High Blood Pressure

Symptoms:
• Fatigue
• Swelling

Treatments:
• Exercise
• Diet

Medications:
• Insulin
```

The entity extraction is intended for basic demonstration purposes and is not a clinical-grade medical NER system.

---

# 🖥️ Streamlit Interface

The application provides a simple medical Q&A interface.

The interface includes:

* Medical question input
* Retrieved medical answer
* Similarity score
* Matched MedQuAD question
* Intent detection
* Disease detection
* Symptom detection
* Treatment detection
* Medication detection
* Other relevant matches
* Conversation history
* Clear conversation functionality
* Medical safety disclaimer

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/sriram-257/MEDICAL_RAG_CHATBOT.git
```

Navigate into the project:

```bash
cd MEDICAL_RAG_CHATBOT
```

---

## 2. Create a Virtual Environment

On Windows:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\activate
```

---

## 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

# 📦 Dataset and Cached Index Setup

The repository contains the processed dataset and pre-computed semantic index:

```text
data/
├── index/
│   ├── embeddings.npy
│   └── medquad.faiss
│
└── processed/
    └── medquad.csv
```

Therefore, **embedding generation is not required for normal execution**.

The application loads the existing cached embeddings and FAISS index when they are available.

If the processed dataset needs to be recreated in a fresh setup, the original MedQuAD dataset can be obtained from:

[https://github.com/abachaa/MedQuAD](https://github.com/abachaa/MedQuAD)

The data preparation script is:

```bash
python src/data_loader.py
```

---

# ▶️ Running the Application

From the project root directory:

```bash
streamlit run src/app.py
```

After Streamlit starts, open the local URL shown in the terminal.

Usually:

```text
http://localhost:8501
```

---

# 🧪 Testing

The chatbot can be tested using questions such as the following.

## Test 1 — Symptoms

```text
What are the symptoms of diabetes?
```

Expected behavior:

* Retrieve a relevant MedQuAD question
* Display the medical answer
* Detect diabetes as a disease
* Detect relevant symptoms
* Identify the intent as `Symptoms`

---

## Test 2 — Treatment

```text
What are the treatments for diabetes?
```

Expected behavior:

* Retrieve relevant medical information
* Detect diabetes as a disease
* Identify treatment-related information when available
* Identify the treatment intent

---

## Test 3 — Medical Condition

```text
What are the symptoms of kidney disease?
```

Expected behavior:

* Retrieve relevant kidney disease information
* Display the best matching answer
* Extract relevant medical entities

---

## Test 4 — Out-of-Domain Question

```text
What is the capital of France?
```

Expected behavior:

The chatbot should recognize that the question is not a suitable medical query and avoid presenting unrelated medical information.

---

# 📊 Retrieval Configuration

Current semantic retrieval configuration:

```text
Dataset:
MedQuAD

Approximate Q&A pairs:
14,979

Embedding Model:
all-MiniLM-L6-v2

Embedding Dimension:
384

Vector Index:
FAISS IndexFlatIP

Similarity:
Normalized Inner Product

Default Top-K:
3

Default Similarity Threshold:
0.45
```

---

# ⚡ Performance

The project uses pre-computed question embeddings and a FAISS index.

Cached files:

```text
data/index/embeddings.npy
data/index/medquad.faiss
```

The application loads these files when available.

This avoids regenerating embeddings for all 14,979 MedQuAD questions every time the application starts.

Only the new user query needs to be converted into an embedding during semantic search.

This significantly reduces startup time compared with rebuilding the entire semantic index.

---

# 🔐 Repository and Data Management

The repository contains the files required for normal execution:

```text
src/
data/processed/medquad.csv
data/index/embeddings.npy
data/index/medquad.faiss
requirements.txt
README.md
```

Development and environment-specific files such as the Python virtual environment and cache files should not be committed.

The `.gitignore` file is used to prevent unnecessary local files from being added to the repository.

---

# 🩺 Medical Safety Disclaimer

> **Important:** This chatbot is an educational and research-oriented medical information retrieval system.

This system is **not a doctor** and its responses should not be considered:

* Medical diagnosis
* Personalized medical advice
* Medical prescriptions
* Professional treatment recommendations
* A replacement for a qualified healthcare professional

The chatbot retrieves information from the MedQuAD dataset and may not contain information about every medical condition or situation.

For personal medical concerns, diagnosis, treatment decisions, or emergencies, consult a qualified healthcare professional.

---

# ⚠️ Limitations

The current system has several limitations:

1. The chatbot is limited to information available in the MedQuAD dataset.
2. Retrieval quality depends on semantic similarity.
3. Questions outside the dataset may not receive useful answers.
4. Medical entity recognition is a basic NLP implementation.
5. Entity recognition is not a clinical-grade medical NER system.
6. Similarity scores do not represent medical certainty.
7. The system does not diagnose medical conditions.
8. The system does not provide personalized treatment plans.
9. Retrieved information should be verified with qualified healthcare professionals.
10. The system is not intended for emergency medical decision-making.

---

# 🚀 Future Improvements

Possible future improvements include:

* Hybrid semantic and keyword retrieval
* Advanced medical Named Entity Recognition
* Retrieval reranking
* Improved medical terminology recognition
* Better source attribution
* Medical knowledge graph integration
* Retrieval evaluation metrics
* Improved out-of-domain question detection
* Multilingual medical question support
* Cloud deployment
* Automated testing
* Improved conversational context
* Larger medical knowledge bases
* Better retrieval evaluation and benchmarking
* Improved medical safety filtering

---

# 📌 Project Summary

The **Medical RAG Chatbot** combines:

```text
MedQuAD Dataset
        +
Sentence Transformers
        +
FAISS
        +
Semantic Retrieval
        +
TF-IDF Retrieval
        +
Medical Entity Recognition
        +
Intent Detection
        +
Streamlit
```

to provide an interactive retrieval-based medical Question-Answering system.

The project demonstrates practical implementation of:

* Natural Language Processing
* Semantic Search
* Vector Similarity Search
* Information Retrieval
* Medical Entity Recognition
* Intent Detection
* Retrieval-Based Question Answering
* Streamlit Application Development

---

# 📁 Main Project Files

### `src/app.py`

Main Streamlit application and user interface.

### `src/chatbot.py`

Medical chatbot logic and response handling.

### `src/semantic_retriever.py`

Sentence Transformer-based semantic retrieval and FAISS search.

### `src/tfidf_retriever.py`

TF-IDF-based keyword retrieval.

### `src/entity_extractor.py`

Basic medical entity extraction.

### `src/data_loader.py`

Dataset loading and preprocessing utilities.

### `src/dataset_analysis.py`

Dataset analysis utilities.

### `requirements.txt`

Python dependencies required to run the project.

---

# 👨‍💻 Author

**Sri Ram Koduru**

B.Sc. Physical Science
Specialization in Artificial Intelligence & Machine Learning

---

# 📄 Dataset Attribution

This project uses the **MedQuAD dataset** for educational and academic purposes.

Dataset repository:

[https://github.com/abachaa/MedQuAD](https://github.com/abachaa/MedQuAD)

Please refer to the original MedQuAD repository for dataset information, authorship, and licensing details.

---

## ⭐ Project Highlights

This project demonstrates a complete retrieval-based medical chatbot pipeline:

```text
Medical Dataset
      ↓
Data Preprocessing
      ↓
Question Embeddings
      ↓
FAISS Vector Index
      ↓
User Query
      ↓
Semantic Retrieval
      ↓
Relevance Filtering
      ↓
Medical Entity Extraction
      ↓
Intent Detection
      ↓
Retrieved Medical Answer
      ↓
Streamlit Interface
```

The application is designed to provide **retrieval-based medical information rather than generating unsupported medical responses**.

```
