import pandas as pd


DATA_PATH = "data/processed/medquad.csv"


def main():

    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("MEDQUAD DATASET ANALYSIS")
    print("=" * 60)

    print(f"\nTotal records: {len(df):,}")

    # Question length
    df["question_length"] = df["question"].str.len()
    df["answer_length"] = df["answer"].str.len()

    print("\nQuestion statistics:")
    print(df["question_length"].describe())

    print("\nAnswer statistics:")
    print(df["answer_length"].describe())

    # Sources
    print("\nNumber of unique sources:")
    print(df["source"].nunique())

    print("\nTop sources:")
    print(df["source"].value_counts().head(15))

    # Question examples
    print("\nSample Q&A pairs:")
    for i, row in df.head(5).iterrows():

        print("\n" + "-" * 60)
        print("QUESTION:")
        print(row["question"])

        print("\nANSWER:")
        print(row["answer"][:500])

        print("\nSOURCE:")
        print(row["source"])

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()