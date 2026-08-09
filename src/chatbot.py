from semantic_retriever import SemanticRetriever


# =============================================================
# MEDICAL CHATBOT
# =============================================================

class MedicalChatbot:

    def __init__(self):

        print("\nInitializing Medical Chatbot...")

        # -----------------------------------------------------
        # Initialize semantic retriever
        # -----------------------------------------------------

        self.retriever = SemanticRetriever(
            "data/processed/medquad.csv"
        )

        print("Medical Chatbot ready.")

    # =========================================================
    # CLEAN ANSWER
    # =========================================================

    def _clean_answer(self, answer):

        if not answer:
            return ""

        # Convert to string
        answer = str(answer)

        # Remove unnecessary whitespace
        answer = " ".join(
            answer.split()
        )

        return answer.strip()

    # =========================================================
    # DETERMINE CONFIDENCE
    # =========================================================

    def _get_confidence(self, score):

        if score >= 0.75:
            return "High"

        elif score >= 0.60:
            return "Good"

        elif score >= 0.45:
            return "Moderate"

        return "Low"

    # =========================================================
    # REMOVE DUPLICATE RESULTS
    # =========================================================

    def _remove_duplicates(self, results):

        unique_results = []

        seen_questions = set()

        for result in results:

            question = (
                result.get("question", "")
                .strip()
                .lower()
            )

            if question in seen_questions:
                continue

            seen_questions.add(question)

            unique_results.append(result)

        return unique_results

    # =========================================================
    # ASK QUESTION
    # =========================================================

    def ask(
        self,
        question,
        top_k=3,
        threshold=0.45
    ):

        # -----------------------------------------------------
        # Validate input
        # -----------------------------------------------------

        if question is None:

            return {
                "answer": "Please enter a medical question.",
                "results": [],
                "relevant": False,
                "confidence": "None",
                "score": 0.0,
                "source_question": ""
            }

        question = str(question).strip()

        if not question:

            return {
                "answer": "Please enter a medical question.",
                "results": [],
                "relevant": False,
                "confidence": "None",
                "score": 0.0,
                "source_question": ""
            }

        # -----------------------------------------------------
        # Retrieve relevant information
        # -----------------------------------------------------

        results = self.retriever.search(
            question,
            top_k=top_k,
            threshold=threshold
        )

        # -----------------------------------------------------
        # Remove duplicate results
        # -----------------------------------------------------

        results = self._remove_duplicates(
            results
        )

        # -----------------------------------------------------
        # No relevant information
        # -----------------------------------------------------

        if not results:

            return {
                "answer": (
                    "I couldn't find sufficiently relevant "
                    "medical information for this question "
                    "in the MedQuAD knowledge base. "
                    "Please try rephrasing your question "
                    "or ask another medical question."
                ),
                "results": [],
                "relevant": False,
                "confidence": "None",
                "score": 0.0,
                "source_question": ""
            }

        # -----------------------------------------------------
        # Best retrieved result
        # -----------------------------------------------------

        best_result = results[0]

        score = float(
            best_result.get(
                "score",
                0.0
            )
        )

        answer = self._clean_answer(
            best_result.get(
                "answer",
                ""
            )
        )

        source_question = self._clean_answer(
            best_result.get(
                "question",
                ""
            )
        )

        # -----------------------------------------------------
        # Extra safety check
        # -----------------------------------------------------

        if not answer:

            return {
                "answer": (
                    "I found a relevant entry, but "
                    "the available answer was empty. "
                    "Please try another medical question."
                ),
                "results": results,
                "relevant": False,
                "confidence": "None",
                "score": score,
                "source_question": source_question
            }

        # -----------------------------------------------------
        # Confidence
        # -----------------------------------------------------

        confidence = self._get_confidence(
            score
        )

        # -----------------------------------------------------
        # Return structured response
        # -----------------------------------------------------

        return {
            "answer": answer,
            "results": results,
            "relevant": True,
            "confidence": confidence,
            "score": score,
            "source_question": source_question
        }


# =============================================================
# DIRECT TEST
# =============================================================

if __name__ == "__main__":

    chatbot = MedicalChatbot()

    print("\n")
    print("=" * 60)
    print("MEDICAL CHATBOT")
    print("=" * 60)

    while True:

        question = input(
            "\nAsk a medical question "
            "(or type 'exit'): "
        )

        # -----------------------------------------------------
        # Exit
        # -----------------------------------------------------

        if question.lower().strip() == "exit":

            print("\nExiting...")

            break

        # -----------------------------------------------------
        # Ask chatbot
        # -----------------------------------------------------

        response = chatbot.ask(
            question
        )

        print(
            "\n" + "-" * 60
        )

        print(
            "\nANSWER:"
        )

        print(
            response["answer"]
        )

        print(
            "\nRelevant:",
            response["relevant"]
        )

        print(
            "Confidence:",
            response["confidence"]
        )

        print(
            "Similarity:",
            f"{response['score']:.4f}"
        )

        # -----------------------------------------------------
        # Source
        # -----------------------------------------------------

        if response["source_question"]:

            print(
                "\nRetrieved MedQuAD Question:"
            )

            print(
                response["source_question"]
            )

        # -----------------------------------------------------
        # Retrieved results
        # -----------------------------------------------------

        if response["results"]:

            print(
                "\nRetrieved Results:"
            )

            for i, result in enumerate(
                response["results"],
                start=1
            ):

                print(
                    f"\n{i}. "
                    f"{result['score']:.4f} - "
                    f"{result['question']}"
                )