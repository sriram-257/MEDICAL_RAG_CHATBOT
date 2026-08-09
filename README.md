
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

This helps provide additional context about the user's query.

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

A simple interactive web interface allows users to:

* Ask medical questions
* View retrieved answers
* View retrieval confidence
* View detected entities
* View detected intent
* Inspect other relevant matches
* Maintain conversation history
* Clear the conversation

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
                       │  Streamlit      │
                       │  Response       │
                       └─────────────────┘
```

---

# 📂 Project Structure

```text
MEDICAL_RAG_CHATBOT/
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
├── data/
│   └── processed/
│       └── medquad.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

> The complete MedQuAD source dataset and large generated embedding/index files may be kept locally and excluded from GitHub when they exceed repository size limits.

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

The dataset provides medical questions and corresponding answers collected from trusted medical information sources.

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

## Step 3 — Generate Question Embeddings

Each MedQuAD question is converted into a semantic embedding.

The model produces:

```text
384-dimensional embeddings
```

for the questions.

---

## Step 4 — Normalize Embeddings

The embeddings are normalized before similarity search.

This allows inner-product similarity to be used effectively for semantic retrieval.

---

## Step 5 — FAISS Similarity Search

The project uses:

```python
faiss.IndexFlatIP
```

to search for the most semantically similar medical questions.

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

## Step 10 — Display the Answer

The best relevant MedQuAD answer is displayed through the Streamlit interface.

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

Other supported medical intents can include categories such as:

```text
Symptoms
Treatment
Medication
Disease
```

depending on the query and entity extraction logic.

---

# 🖥️ Streamlit Interface

The application provides a simple medical Q&A interface.

The interface includes:

* Medical question input
* Retrieved answer
* Match confidence score
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

```bash
pip install -r requirements.txt
```

---

# 📦 Dataset Setup

Download the MedQuAD dataset from:

[https://github.com/abachaa/MedQuAD](https://github.com/abachaa/MedQuAD)

Prepare the processed dataset so that the following file exists:

```text
data/processed/medquad.csv
```

If the processed dataset has not yet been created, run the project's data preparation script:

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

The chatbot can be tested using questions such as:

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

The Sentence Transformer model generates embeddings for the medical questions used by the semantic retriever.

Because embedding generation can take time on the first run, the Streamlit application uses resource caching for the retriever.

This prevents the retriever object from being unnecessarily recreated during normal Streamlit interactions.

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

# 👨‍💻 Author

**Sri Ram Koduru**

B.Sc. Physical Science
Specialization in Artificial Intelligence & Machine Learning

---

# 📄 Dataset Attribution

This project uses the MedQuAD dataset for educational and academic purposes.

Dataset repository:

[https://github.com/abachaa/MedQuAD](https://github.com/abachaa/MedQuAD)

Please refer to the original dataset repository for dataset information, authorship, and licensing details.

```
```
