import os
import xml.etree.ElementTree as ET
import pandas as pd


DATASET_PATH = "data/MedQuAD"
OUTPUT_PATH = "data/processed/medquad.csv"


def load_medquad(dataset_path):
    records = []

    for root_dir, _, files in os.walk(dataset_path):

        for filename in files:

            if not filename.lower().endswith(".xml"):
                continue

            file_path = os.path.join(root_dir, filename)

            try:
                tree = ET.parse(file_path)
                root = tree.getroot()

                # Extract source information if available
                source = root.findtext("Source")
                url = root.findtext("URL")

                for qa_pair in root.findall(".//QAPair"):

                    question = qa_pair.findtext("Question")
                    answer = qa_pair.findtext("Answer")

                    if question and answer:

                        records.append({
                            "question": question.strip(),
                            "answer": answer.strip(),
                            "source": source.strip() if source else "",
                            "url": url.strip() if url else ""
                        })

            except ET.ParseError:
                print(f"XML parsing error: {file_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return pd.DataFrame(records)


def clean_dataset(df):

    # Remove missing questions/answers
    df = df.dropna(subset=["question", "answer"])

    # Remove empty strings
    df = df[
        (df["question"].str.strip() != "") &
        (df["answer"].str.strip() != "")
    ]

    # Remove duplicate questions
    df = df.drop_duplicates(subset=["question"])

    # Reset index
    df = df.reset_index(drop=True)

    return df


def main():

    print("=" * 60)
    print("MEDQUAD DATASET LOADER")
    print("=" * 60)

    print("\nLoading MedQuAD...")

    df = load_medquad(DATASET_PATH)

    print(f"\nRaw Q&A pairs: {len(df):,}")

    if df.empty:
        print("\nNo data found.")
        print("Check that your MedQuAD folder is:")
        print("data/MedQuAD/")
        return

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nCleaning dataset...")

    df = clean_dataset(df)

    print(f"Clean Q&A pairs: {len(df):,}")

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nSample questions:")
    for i, question in enumerate(df["question"].head(5), 1):
        print(f"{i}. {question}")

    # Create processed directory
    os.makedirs("data/processed", exist_ok=True)

    # Save
    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("DATASET READY")
    print("=" * 60)
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Total records: {len(df):,}")


if __name__ == "__main__":
    main()