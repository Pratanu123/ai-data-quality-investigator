
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "amazon.csv"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place the Kaggle CSV at data/amazon.csv"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n--- DATASET SHAPE ---")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n--- COLUMN NAMES ---")
    print(df.columns.tolist())

    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

    print("\n--- EXACT DUPLICATE ROWS ---")
    print(df.duplicated().sum())

    print("\n--- SAMPLE RECORDS ---")
    print(df.head(5).to_string())

    print("\n--- NUMERIC SUMMARY ---")
    print(df.describe(include="all").to_string())


if __name__ == "__main__":
    main()
