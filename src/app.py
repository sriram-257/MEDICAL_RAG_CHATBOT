import streamlit as st

from tfidf_retriever import TfidfRetriever
from semantic_retriever import SemanticRetriever
from entity_extractor import extract_entities, detect_intent


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/medquad.csv"

st.set_page_config(
    page_title="MedQuAD Medical Q&A Chatbot",
    page_icon="🩺",
    layout="wide",
)


# ============================================================
# MEDICAL QUERY DETECTION
# ============================================================

MEDICAL_KEYWORDS = {
    # General medical terms
    "medical",
    "health",
    "healthcare",
    "disease",
    "diseases",
    "condition",
    "conditions",
    "disorder",
    "disorders",
    "illness",
    "illnesses",
    "syndrome",
    "symptom",
    "symptoms",
    "sign",
    "signs",
    "diagnosis",
    "diagnose",
    "diagnostic",
    "treatment",
    "treatments",
    "therapy",
    "therapies",
    "medicine",
    "medication",
    "medications",
    "drug",
    "drugs",
    "dose",
    "dosage",
    "prescription",
    "doctor",
    "physician",
    "hospital",
    "patient",
    "patients",
    "clinic",
    "clinical",
    "surgery",
    "surgical",
    "infection",
    "infected",
    "pain",
    "fever",
    "swelling",
    "injury",
    "injuries",
    "blood",
    "pressure",
    "heart",
    "kidney",
    "liver",
    "lung",
    "lungs",
    "brain",
    "diabetes",
    "cancer",
    "stroke",
    "cholesterol",
    "asthma",
    "obesity",
    "pregnancy",
    "pregnant",
    "vitamin",
    "allergy",
    "allergies",
    "rash",
    "headache",
    "migraine",
    "cough",
    "breathing",
    "breath",
    "vomiting",
    "nausea",
    "diarrhea",
    "diarrhoea",
    "blood pressure",
    "heart attack",
    "kidney disease",
    "heart disease",
    "type 1 diabetes",
    "type 2 diabetes",
    "gestational diabetes",
}


def is_medical_query(question):
    """
    Determine whether a question is likely to be medical.

    This prevents unrelated questions such as:
    'What is the capital of France?'
    from receiving an unrelated medical answer.
    """

    question_lower = question.lower().strip()

    if not question_lower:
        return False

    # Direct keyword detection
    for keyword in MEDICAL_KEYWORDS:
        if keyword in question_lower:
            return True

    return False


# ============================================================
# RETRIEVER LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading TF-IDF retriever...")
def load_tfidf_retriever():
    return TfidfRetriever(DATA_PATH)


@st.cache_resource(
    show_spinner="Loading semantic retriever..."
)
def load_semantic_retriever():
    return SemanticRetriever(DATA_PATH)


def get_retriever(method):

    if method == "Semantic (Sentence-Transformers)":
        return load_semantic_retriever()

    return load_tfidf_retriever()


# ============================================================
# ENTITY CONFIGURATION
# ============================================================

ENTITY_LABELS = {
    "diseases": "🦠 Diseases",
    "symptoms": "🤒 Symptoms",
    "treatments": "💊 Treatments",
    "medications": "💉 Medications",
}


# ============================================================
# ENTITY CLEANING
# ============================================================

def clean_entity_list(items):

    if not items:
        return []

    cleaned = []

    for item in items:

        if item is None:
            continue

        item = str(item).strip()

        if not item:
            continue

        # Remove accidental bullets
        item = item.lstrip("-•*").strip()

        if not item:
            continue

        # Remove duplicates
        if item.lower() not in [
            existing.lower()
            for existing in cleaned
        ]:
            cleaned.append(item)

    return cleaned


def normalize_entities(entities):

    normalized = {}

    for key in ENTITY_LABELS:

        if isinstance(entities, dict):

            normalized[key] = clean_entity_list(
                entities.get(key, [])
            )

        else:

            normalized[key] = []

    return normalized


# ============================================================
# RENDER MEDICAL ENTITIES
# ============================================================

def render_entities(entities):

    entities = normalize_entities(entities)

    st.markdown(
        "## 🔬 Detected Medical Entities"
    )

    columns = st.columns(4)

    for index, (key, label) in enumerate(
        ENTITY_LABELS.items()
    ):

        items = entities[key]

        with columns[index]:

            st.markdown(
                f"### {label}"
            )

            st.markdown("---")

            if items:

                for item in items:

                    st.markdown(
                        f"• **{item.title()}**"
                    )

            else:

                st.caption(
                    "None detected"
                )


# ============================================================
# QUESTION INTENT
# ============================================================

def render_intent(intents):

    st.markdown(
        "## 🎯 Question Intent"
    )

    if not intents:

        st.info(
            "General Medical Question"
        )

        return

    intent_icons = {

        "Symptoms": "🤒",
        "Symptom": "🤒",

        "Treatment": "💊",
        "Treatments": "💊",

        "Medication": "💉",
        "Medications": "💉",

        "Disease": "🦠",
        "Diseases": "🦠",

        "Diagnosis": "🔬",

        "Prevention": "🛡️",

        "Cause": "❓",
        "Causes": "❓",
    }

    intent_text = ", ".join(
        intents
    )

    icon = "🎯"

    if len(intents) == 1:

        icon = intent_icons.get(
            intents[0],
            "🎯"
        )

    st.success(
        f"{icon} {intent_text}"
    )


# ============================================================
# RETRIEVAL INFORMATION
# ============================================================

def render_retrieval_information(results):

    if not results:
        return

    best = results[0]

    st.markdown(
        "## 📊 Retrieval Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Match Confidence",
            f"{best.get('score', 0):.3f}"
        )

    with col2:

        st.metric(
            "Retrieved Results",
            len(results)
        )

    st.markdown(
        "### 📚 Matched MedQuAD Question"
    )

    st.info(
        best.get(
            "question",
            "No matched question available."
        )
    )

    if len(results) > 1:

        st.markdown(
            "### 🔎 Other Relevant Matches"
        )

        for index, result in enumerate(
            results[1:],
            start=2
        ):

            score = result.get(
                "score",
                0
            )

            question = result.get(
                "question",
                ""
            )

            st.markdown(
                f"**{index}. "
                f"({score:.3f}) "
                f"{question}**"
            )


# ============================================================
# ANSWER SECTION
# ============================================================

def render_answer(answer):

    st.markdown(
        "## 💬 Medical Answer"
    )

    st.write(answer)


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    with st.sidebar:

        st.title(
            "🩺 MedQuAD"
        )

        st.markdown(
            """
            ### About

            This chatbot retrieves medical
            question-answer pairs from the
            **MedQuAD dataset**.

            ### Components

            - TF-IDF retrieval
            - Semantic search
            - Sentence Transformers
            - FAISS similarity search
            - Medical entity extraction
            - Question intent detection
            - Streamlit interface
            """
        )

        st.markdown("---")

        st.markdown(
            "### 🔍 Retrieval System"
        )

        method = st.radio(
            "Retrieval method",
            [
                "Semantic (Sentence-Transformers)",
                "TF-IDF",
            ],
            index=0,
            help=(
                "Semantic search uses sentence "
                "embeddings for semantic matching. "
                "TF-IDF provides lexical matching."
            )
        )

        top_k = st.slider(
            "Number of results",
            min_value=1,
            max_value=5,
            value=3
        )

        st.markdown("---")

        st.markdown(
            "### 📊 Dataset"
        )

        st.write(
            "MedQuAD"
        )

        st.caption(
            "Medical question-answer dataset"
        )

        st.markdown("---")

        if st.button(
            "🗑️ Clear Conversation",
            use_container_width=True
        ):

            st.session_state.history = []

            st.rerun()

        st.markdown("---")

        st.caption(
            "⚠️ For informational purposes only. "
            "This chatbot is not medical advice."
        )

    return method, top_k


# ============================================================
# CHAT HISTORY
# ============================================================

def render_history():

    if "history" not in st.session_state:

        st.session_state.history = []

    for turn in st.session_state.history:

        # ------------------------------
        # USER QUESTION
        # ------------------------------

        with st.chat_message("user"):

            st.markdown(
                "### ❓ User Question"
            )

            st.write(
                turn["question"]
            )

        # ------------------------------
        # ASSISTANT RESPONSE
        # ------------------------------

        with st.chat_message("assistant"):

            render_answer(
                turn["answer"]
            )

            st.markdown("---")

            render_entities(
                turn["entities"]
            )

            st.markdown("---")

            render_intent(
                turn["intents"]
            )

            st.markdown("---")

            render_retrieval_information(
                turn["results"]
            )


# ============================================================
# PROCESS QUESTION
# ============================================================

def process_question(
    question,
    retriever,
    top_k
):

    # --------------------------------------------------------
    # MEDICAL QUERY VALIDATION
    # --------------------------------------------------------

    if not is_medical_query(question):

        return {
            "type": "non_medical",
            "message": (
                "I’m designed to answer medical "
                "questions using the MedQuAD "
                "knowledge base.\n\n"
                "Please ask a medical question, "
                "for example:\n\n"
                "• What are the symptoms of diabetes?\n"
                "• What causes high blood pressure?\n"
                "• What treatments are available for asthma?"
            )
        }

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    with st.spinner(
        "🔎 Searching the MedQuAD knowledge base..."
    ):

        results = retriever.search(
            question,
            top_k=top_k
        )

    # --------------------------------------------------------
    # NO RESULTS
    # --------------------------------------------------------

    if not results:

        return {
            "type": "no_results",
            "message": (
                "I couldn't find a relevant medical "
                "answer in the MedQuAD knowledge base."
            )
        }

    # --------------------------------------------------------
    # BEST RESULT
    # --------------------------------------------------------

    best = results[0]

    # --------------------------------------------------------
    # ENTITY EXTRACTION
    # --------------------------------------------------------

    try:

        entities = extract_entities(
            question,
            result=best
        )

    except Exception:

        entities = {}

    entities = normalize_entities(
        entities
    )

    # --------------------------------------------------------
    # INTENT DETECTION
    # --------------------------------------------------------

    try:

        intents = detect_intent(
            question
        )

    except Exception:

        intents = []

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "type": "success",

        "answer": best.get(
            "answer",
            "No answer available."
        ),

        "entities": entities,

        "intents": intents,

        "results": results,
    }


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # ========================================================
    # HEADER
    # ========================================================

    st.title(
        "🩺 MedQuAD Medical Q&A Chatbot"
    )

    st.caption(
        "A semantic medical question-answering "
        "system powered by MedQuAD, "
        "Sentence Transformers and FAISS."
    )

    st.markdown("---")

    # ========================================================
    # SIDEBAR
    # ========================================================

    method, top_k = render_sidebar()

    # ========================================================
    # LOAD RETRIEVER
    # ========================================================

    try:

        retriever = get_retriever(
            method
        )

    except FileNotFoundError:

        st.error(
            f"""
            Could not find:

            `{DATA_PATH}`

            Please make sure the processed
            MedQuAD dataset exists.
            """
        )

        st.stop()

    except Exception as e:

        st.error(
            f"Failed to load the retriever: {e}"
        )

        st.stop()

    # ========================================================
    # SESSION STATE
    # ========================================================

    if "history" not in st.session_state:

        st.session_state.history = []

    # ========================================================
    # PREVIOUS CONVERSATION
    # ========================================================

    render_history()

    # ========================================================
    # QUESTION INPUT
    # ========================================================

    question = st.chat_input(
        "Ask a medical question..."
    )

    # ========================================================
    # NEW QUESTION
    # ========================================================

    if question:

        question = question.strip()

        if not question:
            st.stop()

        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        with st.chat_message("user"):

            st.markdown(
                "### ❓ User Question"
            )

            st.write(
                question
            )

        # ----------------------------------------------------
        # ASSISTANT
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            result = process_question(
                question,
                retriever,
                top_k
            )

            # =================================================
            # NON-MEDICAL QUESTION
            # =================================================

            if result["type"] == "non_medical":

                st.warning(
                    result["message"]
                )

                st.session_state.history.append(
                    {
                        "question": question,
                        "answer": result["message"],
                        "entities": {
                            "diseases": [],
                            "symptoms": [],
                            "treatments": [],
                            "medications": [],
                        },
                        "intents": [],
                        "results": [],
                    }
                )

                st.stop()

            # =================================================
            # NO RESULTS
            # =================================================

            if result["type"] == "no_results":

                st.warning(
                    result["message"]
                )

                st.session_state.history.append(
                    {
                        "question": question,
                        "answer": result["message"],
                        "entities": {
                            "diseases": [],
                            "symptoms": [],
                            "treatments": [],
                            "medications": [],
                        },
                        "intents": [],
                        "results": [],
                    }
                )

                st.stop()

            # =================================================
            # ANSWER
            # =================================================

            render_answer(
                result["answer"]
            )

            st.markdown("---")

            # =================================================
            # ENTITIES
            # =================================================

            render_entities(
                result["entities"]
            )

            st.markdown("---")

            # =================================================
            # INTENT
            # =================================================

            render_intent(
                result["intents"]
            )

            st.markdown("---")

            # =================================================
            # RETRIEVAL
            # =================================================

            render_retrieval_information(
                result["results"]
            )

        # ====================================================
        # SAVE HISTORY
        # ====================================================

        st.session_state.history.append(
            {
                "question": question,

                "answer": result["answer"],

                "entities": result["entities"],

                "intents": result["intents"],

                "results": result["results"],
            }
        )

    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown("---")

    st.caption(
        "⚠️ This chatbot retrieves information "
        "from the MedQuAD medical Q&A dataset. "
        "It is intended for educational and "
        "informational purposes only and does "
        "not provide medical advice, diagnosis, "
        "or treatment."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()