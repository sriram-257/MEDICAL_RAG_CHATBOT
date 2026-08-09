import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/processed/medquad.csv"


class TfidfRetriever:

    def __init__(self, data_path):

        print("Loading MedQuAD dataset...")

        self.df = pd.read_csv(data_path)

        self.df["question"] = self.df["question"].fillna("")
        self.df["answer"] = self.df["answer"].fillna("")

        print(f"Loaded {len(self.df):,} Q&A pairs.")

        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=50000
        )

        print("Building TF-IDF index...")

        self.question_vectors = self.vectorizer.fit_transform(
            self.df["question"]
        )

        print("TF-IDF index ready!")

    def search(self, query, top_k=3):

        # Convert user question into TF-IDF vector
        query_vector = self.vectorizer.transform([query])

        # Calculate cosine similarity
        similarities = cosine_similarity(
            query_vector,
            self.question_vectors
        ).flatten()

        # Get highest scores
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []

        for index in top_indices:

            results.append({
                "question": self.df.iloc[index]["question"],
                "answer": self.df.iloc[index]["answer"],
                "score": float(similarities[index])
            })

        return results


if __name__ == "__main__":

    retriever = TfidfRetriever(DATA_PATH)

    print("\n" + "=" * 60)
    print("TF-IDF MEDICAL RETRIEVER")
    print("=" * 60)

    while True:

        query = input("\nEnter medical question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        results = retriever.search(query, top_k=3)

        print("\n" + "-" * 60)
        print("TOP RESULTS")
        print("-" * 60)

        for i, result in enumerate(results, 1):

            print(f"\nResult {i}")
            print(f"Similarity: {result['score']:.4f}")

            print(f"\nQuestion:")
            print(result["question"])

            print(f"\nAnswer:")
            print(result["answer"][:1000])