# ============================================================
# entity_extractor.py
# MedQuAD Medical Entity Extraction
# ============================================================

import re


# ============================================================
# MEDICAL VOCABULARIES
# ============================================================

DISEASES = {
    # Diabetes
    "diabetes",
    "diabetes mellitus",
    "type 1 diabetes",
    "type 2 diabetes",
    "type 1 diabetes mellitus",
    "type 2 diabetes mellitus",
    "gestational diabetes",
    "prediabetes",

    # Cardiovascular
    "heart disease",
    "heart attack",
    "myocardial infarction",
    "cardiovascular disease",
    "coronary artery disease",
    "high blood pressure",
    "hypertension",
    "cholesterol",
    "high cholesterol",
    "stroke",

    # Kidney
    "kidney disease",
    "chronic kidney disease",
    "kidney failure",
    "renal disease",
    "renal failure",

    # Respiratory
    "asthma",
    "pneumonia",
    "bronchitis",
    "copd",
    "chronic obstructive pulmonary disease",

    # Common diseases
    "cancer",
    "breast cancer",
    "lung cancer",
    "skin cancer",
    "prostate cancer",
    "colon cancer",

    "flu",
    "influenza",
    "covid-19",
    "covid",
    "coronavirus",

    "migraine",
    "arthritis",
    "rheumatoid arthritis",
    "osteoarthritis",

    "anemia",
    "iron deficiency anemia",

    "depression",
    "anxiety",

    "obesity",
    "overweight",

    "thyroid disease",
    "hypothyroidism",
    "hyperthyroidism",

    "ulcer",
    "stomach ulcer",
    "peptic ulcer",

    "acid reflux",
    "gerd",
    "gastroesophageal reflux disease",

    "osteoporosis",

    "alzheimer's disease",
    "dementia",
}


SYMPTOMS = {
    # Diabetes-related symptoms
    "losing weight without trying",
    "weight loss",
    "unexplained weight loss",
    "sores that heal slowly",
    "slow healing",
    "slow healing wounds",
    "tingling in the feet",
    "tingling",
    "numbness",
    "loss of feeling",
    "feeling very hungry",
    "very hungry",
    "increased hunger",
    "excessive hunger",
    "being very thirsty",
    "very thirsty",
    "excessive thirst",
    "increased thirst",
    "frequent urination",
    "frequent urination",
    "blurry eyesight",
    "blurred vision",
    "blurry vision",
    "dry itchy skin",
    "dry skin",
    "itchy skin",
    "fatigue",
    "tiredness",
    "feeling tired",

    # General symptoms
    "fever",
    "headache",
    "cough",
    "sore throat",
    "runny nose",
    "stuffy nose",
    "shortness of breath",
    "difficulty breathing",
    "chest pain",
    "abdominal pain",
    "stomach pain",
    "back pain",
    "joint pain",
    "muscle pain",
    "nausea",
    "vomiting",
    "diarrhea",
    "constipation",
    "dizziness",
    "weakness",
    "fatigue",
    "swelling",
    "rash",
    "itching",
    "pain",
    "bleeding",
    "blood in urine",
    "blood in stool",
    "loss of appetite",
    "increased appetite",
    "difficulty sleeping",
    "insomnia",
    "confusion",
    "memory loss",
}


TREATMENTS = {
    # Diabetes
    "treatment",
    "treatments",
    "diabetes treatment",
    "diabetes management",
    "blood sugar control",
    "glucose control",
    "diet",
    "healthy diet",
    "exercise",
    "physical activity",
    "weight loss",
    "lifestyle changes",
    "lifestyle modification",

    # General treatments
    "surgery",
    "operation",
    "chemotherapy",
    "radiation therapy",
    "radiotherapy",
    "immunotherapy",
    "physical therapy",
    "physiotherapy",
    "occupational therapy",
    "speech therapy",
    "psychotherapy",
    "counseling",
    "counselling",
    "dialysis",
    "kidney transplant",
    "organ transplant",
    "blood transfusion",
    "oxygen therapy",
    "insulin therapy",
    "hormone therapy",
    "antibiotic treatment",
    "antiviral treatment",
    "rest",
    "hydration",
}


MEDICATIONS = {
    # Diabetes
    "insulin",
    "metformin",
    "glipizide",
    "glyburide",
    "glimepiride",
    "pioglitazone",
    "sitagliptin",
    "linagliptin",
    "empagliflozin",
    "dapagliflozin",
    "canagliflozin",
    "semaglutide",
    "liraglutide",

    # Common medications
    "aspirin",
    "ibuprofen",
    "acetaminophen",
    "paracetamol",
    "naproxen",
    "amoxicillin",
    "azithromycin",
    "penicillin",
    "prednisone",
    "prednisolone",
    "atorvastatin",
    "simvastatin",
    "rosuvastatin",
    "lisinopril",
    "losartan",
    "amlodipine",
    "omeprazole",
    "pantoprazole",
    "levothyroxine",
    "warfarin",
    "heparin",
    "clopidogrel",
    "morphine",
    "codeine",
    "gabapentin",
    "pregabalin",
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text for matching.
    """

    if not text:
        return ""

    text = str(text).lower()

    # Normalize apostrophes
    text = text.replace("’", "'")

    # Remove unnecessary punctuation
    text = re.sub(r"[^a-z0-9\s\-']", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# PHRASE MATCHING
# ============================================================

def phrase_in_text(phrase, text):
    """
    Check whether a complete medical phrase occurs in text.
    """

    phrase = normalize_text(phrase)
    text = normalize_text(text)

    if not phrase or not text:
        return False

    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"

    return re.search(pattern, text) is not None


# ============================================================
# FIND ENTITIES
# ============================================================

def find_entities(text, vocabulary):
    """
    Find entities from a vocabulary.

    Longer phrases are checked first so that:
        'type 2 diabetes'
    is detected before:
        'diabetes'
    """

    if not text:
        return []

    found = []

    # Longest phrases first
    sorted_vocab = sorted(
        vocabulary,
        key=lambda x: len(x),
        reverse=True
    )

    for entity in sorted_vocab:

        if phrase_in_text(entity, text):

            # Avoid duplicates caused by overlapping phrases
            if entity not in found:
                found.append(entity)

    return found


# ============================================================
# REMOVE OVERLAPPING ENTITIES
# ============================================================

def remove_overlapping_entities(entities):
    """
    Remove shorter entities when they are already represented
    by a longer entity.

    Example:

        type 2 diabetes
        diabetes

    becomes:

        type 2 diabetes
    """

    entities = sorted(
        set(entities),
        key=lambda x: len(x),
        reverse=True
    )

    result = []

    for entity in entities:

        normalized_entity = normalize_text(entity)

        is_duplicate = False

        for existing in result:

            normalized_existing = normalize_text(existing)

            if normalized_entity == normalized_existing:
                is_duplicate = True
                break

            if (
                normalized_entity in normalized_existing
                and len(normalized_existing) > len(normalized_entity)
            ):
                is_duplicate = True
                break

        if not is_duplicate:
            result.append(entity)

    # Put longer/more meaningful entities first
    return sorted(
        result,
        key=lambda x: (-len(x), x)
    )


# ============================================================
# EXTRACT ENTITIES
# ============================================================

def extract_entities(question, result=None):
    """
    Extract medical entities from:

    1. User question
    2. Matched MedQuAD question
    3. Retrieved MedQuAD answer

    Parameters
    ----------
    question : str
        User's question.

    result : dict, optional
        Best retrieval result containing:
            question
            answer

    Returns
    -------
    dict
        {
            "diseases": [...],
            "symptoms": [...],
            "treatments": [...],
            "medications": [...]
        }
    """

    # --------------------------------------------------------
    # User question
    # --------------------------------------------------------

    question_text = normalize_text(question)

    # --------------------------------------------------------
    # Retrieved result
    # --------------------------------------------------------

    matched_question = ""
    answer = ""

    if isinstance(result, dict):

        matched_question = result.get(
            "question",
            ""
        )

        answer = result.get(
            "answer",
            ""
        )

    # --------------------------------------------------------
    # Build searchable text
    # --------------------------------------------------------

    searchable_text = " ".join(
        [
            question_text,
            normalize_text(matched_question),
            normalize_text(answer),
        ]
    )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    diseases = find_entities(
        searchable_text,
        DISEASES
    )

    symptoms = find_entities(
        searchable_text,
        SYMPTOMS
    )

    treatments = find_entities(
        searchable_text,
        TREATMENTS
    )

    medications = find_entities(
        searchable_text,
        MEDICATIONS
    )

    # --------------------------------------------------------
    # Remove overlaps
    # --------------------------------------------------------

    diseases = remove_overlapping_entities(
        diseases
    )

    symptoms = remove_overlapping_entities(
        symptoms
    )

    treatments = remove_overlapping_entities(
        treatments
    )

    medications = remove_overlapping_entities(
        medications
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "diseases": diseases,
        "symptoms": symptoms,
        "treatments": treatments,
        "medications": medications,
    }


# ============================================================
# INTENT DETECTION
# ============================================================

def detect_intent(question):
    """
    Detect the user's medical question intent.
    """

    text = normalize_text(question)

    intents = []

    # --------------------------------------------------------
    # Symptoms
    # --------------------------------------------------------

    symptom_patterns = [
        "symptom",
        "symptoms",
        "signs",
        "sign",
        "how do i know",
        "what happens",
        "what does it feel like",
        "indications",
    ]

    if any(
        pattern in text
        for pattern in symptom_patterns
    ):
        intents.append("Symptoms")

    # --------------------------------------------------------
    # Treatment
    # --------------------------------------------------------

    treatment_patterns = [
        "treatment",
        "treatments",
        "treat",
        "cure",
        "cured",
        "therapy",
        "therapies",
        "how is it treated",
        "how to treat",
        "management",
        "manage",
    ]

    if any(
        pattern in text
        for pattern in treatment_patterns
    ):
        intents.append("Treatment")

    # --------------------------------------------------------
    # Medication
    # --------------------------------------------------------

    medication_patterns = [
        "medication",
        "medications",
        "medicine",
        "medicines",
        "drug",
        "drugs",
        "tablet",
        "tablets",
        "pill",
        "pills",
        "prescription",
        "what medicine",
        "what medication",
    ]

    if any(
        pattern in text
        for pattern in medication_patterns
    ):
        intents.append("Medication")

    # --------------------------------------------------------
    # Diagnosis
    # --------------------------------------------------------

    diagnosis_patterns = [
        "diagnosis",
        "diagnose",
        "diagnostic",
        "test",
        "tests",
        "testing",
        "blood test",
        "how is it diagnosed",
        "how do doctors diagnose",
    ]

    if any(
        pattern in text
        for pattern in diagnosis_patterns
    ):
        intents.append("Diagnosis")

    # --------------------------------------------------------
    # Causes
    # --------------------------------------------------------

    cause_patterns = [
        "cause",
        "causes",
        "why",
        "reason",
        "reasons",
        "risk factor",
        "risk factors",
        "what causes",
    ]

    if any(
        pattern in text
        for pattern in cause_patterns
    ):
        intents.append("Causes")

    # --------------------------------------------------------
    # Prevention
    # --------------------------------------------------------

    prevention_patterns = [
        "prevent",
        "prevention",
        "avoid",
        "avoiding",
        "reduce the risk",
        "how can i prevent",
    ]

    if any(
        pattern in text
        for pattern in prevention_patterns
    ):
        intents.append("Prevention")

    # --------------------------------------------------------
    # Complications
    # --------------------------------------------------------

    complication_patterns = [
        "complication",
        "complications",
        "side effect",
        "side effects",
        "long term effects",
        "long-term effects",
    ]

    if any(
        pattern in text
        for pattern in complication_patterns
    ):
        intents.append("Complications")

    # --------------------------------------------------------
    # If nothing detected
    # --------------------------------------------------------

    if not intents:
        intents.append("General")

    # Remove duplicates while preserving order
    intents = list(dict.fromkeys(intents))

    return intents


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = "What are the symptoms of diabetes?"

    sample_result = {
        "question": "What are the symptoms of Diabetes?",
        "answer": """
        Common signs of diabetes include being very thirsty,
        frequent urination, feeling very hungry or tired,
        losing weight without trying, sores that heal slowly,
        dry itchy skin, blurry eyesight and tingling in the feet.
        Treatment may include diet, exercise and physical activity.
        Some people may require insulin.
        """
    }

    print("\nDetected entities:\n")

    entities = extract_entities(
        question,
        result=sample_result
    )

    for category, values in entities.items():

        print(category)

        if values:

            for value in values:
                print(" -", value)

        else:
            print(" - None detected")

    print("\nIntent:")

    print(
        detect_intent(question)
    )