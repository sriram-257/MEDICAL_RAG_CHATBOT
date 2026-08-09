import os
import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# =============================================================
# PATHS
# =============================================================

DATA_PATH = "data/processed/medquad.csv"

INDEX_DIR = "data/index"

EMBEDDINGS_PATH = os.path.join(
    INDEX_DIR,
    "embeddings.npy"
)

FAISS_PATH = os.path.join(
    INDEX_DIR,
    "medquad.faiss"
)

MODEL_NAME = "all-MiniLM-L6-v2"


# =============================================================
# SEMANTIC RETRIEVER
# =============================================================

class SemanticRetriever:

    def __init__(self, data_path=DATA_PATH):

        print("=" * 60)
        print("INITIALIZING SEMANTIC RETRIEVER")
        print("=" * 60)

        # -----------------------------------------------------
        # Load dataset
        # -----------------------------------------------------

        print("\nLoading MedQuAD dataset...")

        self.df = pd.read_csv(data_path)

        self.df["question"] = (
            self.df["question"]
            .fillna("")
            .astype(str)
        )

        self.df["answer"] = (
            self.df["answer"]
            .fillna("")
            .astype(str)
        )

        print(
            f"Loaded {len(self.df):,} Q&A pairs."
        )

        # -----------------------------------------------------
        # Load Sentence Transformer
        # -----------------------------------------------------

        print("\nLoading Sentence Transformer model...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        print(
            "Sentence Transformer loaded."
        )

        # -----------------------------------------------------
        # Check for existing cached index
        # -----------------------------------------------------

        if (
            os.path.exists(EMBEDDINGS_PATH)
            and os.path.exists(FAISS_PATH)
        ):

            print("\n" + "-" * 60)
            print("FOUND EXISTING EMBEDDINGS AND FAISS INDEX")
            print("-" * 60)

            try:

                # -------------------------------------------------
                # Load embeddings
                # -------------------------------------------------

                print(
                    "\nLoading cached embeddings..."
                )

                self.embeddings = np.load(
                    EMBEDDINGS_PATH
                )

                self.embeddings = np.asarray(
                    self.embeddings,
                    dtype="float32"
                )

                print(
                    f"Loaded embeddings: "
                    f"{self.embeddings.shape}"
                )

                # -------------------------------------------------
                # Load FAISS index
                # -------------------------------------------------

                print(
                    "\nLoading cached FAISS index..."
                )

                self.index = faiss.read_index(
                    FAISS_PATH
                )

                print(
                    "FAISS index loaded successfully."
                )

                print(
                    f"Indexed vectors: "
                    f"{self.index.ntotal:,}"
                )

                # -------------------------------------------------
                # Validate cached files
                # -------------------------------------------------

                if len(self.df) != self.embeddings.shape[0]:

                    raise ValueError(
                        "Dataset size does not match "
                        "cached embeddings."
                    )

                if self.index.ntotal != len(self.df):

                    raise ValueError(
                        "FAISS index size does not match "
                        "dataset size."
                    )

                if self.embeddings.shape[0] != self.index.ntotal:

                    raise ValueError(
                        "Embedding count does not match "
                        "FAISS index."
                    )

                print(
                    "\nCached semantic index validated."
                )

                print("=" * 60)
                print("SEMANTIC RETRIEVER READY")
                print("=" * 60)

                return

            except Exception as e:

                print(
                    "\nWARNING: Cached index could not "
                    "be used."
                )

                print(
                    f"Reason: {e}"
                )

                print(
                    "\nRebuilding embeddings and "
                    "FAISS index..."
                )

        # -----------------------------------------------------
        # Create embeddings only if cache is unavailable
        # -----------------------------------------------------

        print("\n" + "-" * 60)
        print("CREATING NEW QUESTION EMBEDDINGS")
        print("-" * 60)

        questions = (
            self.df["question"]
            .tolist()
        )

        embeddings = self.model.encode(
            questions,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        self.embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        print(
            f"\nEmbedding shape: "
            f"{self.embeddings.shape}"
        )

        # -----------------------------------------------------
        # Create FAISS index
        # -----------------------------------------------------

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            self.embeddings
        )

        print(
            "\nFAISS index created successfully."
        )

        print(
            f"Indexed vectors: "
            f"{self.index.ntotal:,}"
        )

        # -----------------------------------------------------
        # Save cache
        # -----------------------------------------------------

        os.makedirs(
            INDEX_DIR,
            exist_ok=True
        )

        print(
            "\nSaving embeddings..."
        )

        np.save(
            EMBEDDINGS_PATH,
            self.embeddings
        )

        print(
            "Embeddings saved."
        )

        print(
            "\nSaving FAISS index..."
        )

        faiss.write_index(
            self.index,
            FAISS_PATH
        )

        print(
            "FAISS index saved."
        )

        print("=" * 60)
        print("SEMANTIC RETRIEVER READY")
        print("=" * 60)


    # =========================================================
    # SEARCH
    # =========================================================

    def search(
        self,
        query,
        top_k=3,
        threshold=0.45
    ):

        # -----------------------------------------------------
        # Validate query
        # -----------------------------------------------------

        if not query or not query.strip():

            return []

        query = query.strip()

        # -----------------------------------------------------
        # Create query embedding
        # -----------------------------------------------------

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        # -----------------------------------------------------
        # Search FAISS
        # -----------------------------------------------------

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        # -----------------------------------------------------
        # Process results
        # -----------------------------------------------------

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            # Ignore invalid FAISS index
            if index < 0:
                continue

            score = float(score)

            # -------------------------------------------------
            # Relevance threshold
            # -------------------------------------------------

            if score < threshold:
                continue

            results.append(
                {
                    "question": self.df.iloc[index]["question"],
                    "answer": self.df.iloc[index]["answer"],
                    "score": score
                }
            )

        return results


# =============================================================
# TESTING
# =============================================================

if __name__ == "__main__":

    retriever = SemanticRetriever(
        DATA_PATH
    )

    print("\n")
    print("=" * 60)
    print("SEMANTIC MEDICAL RETRIEVER TEST")
    print("=" * 60)

    while True:

        query = input(
            "\nEnter medical question "
            "(or type 'exit'): "
        )

        if query.lower().strip() == "exit":

            print(
                "\nExiting..."
            )

            break

        results = retriever.search(
            query,
            top_k=3,
            threshold=0.45
        )

        print(
            "\n" + "-" * 60
        )

        print(
            "SEARCH RESULTS"
        )

        print(
            "-" * 60
        )

        # -----------------------------------------------------
        # No relevant result
        # -----------------------------------------------------

        if not results:

            print(
                "\nNo sufficiently relevant medical "
                "information was found."
            )

            print(
                "\nThis system is designed to answer "
                "medical questions from the MedQuAD dataset."
            )

            continue

        # -----------------------------------------------------
        # Display results
        # -----------------------------------------------------

        for i, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\nResult {i}"
            )

            print(
                f"Similarity: "
                f"{result['score']:.4f}"
            )

            print(
                "\nQuestion:"
            )

            print(
                result["question"]
            )

            print(
                "\nAnswer:"
            )

            print(
                result["answer"][:1000]
            )

    print(
        "\nRetriever stopped."
    )