import argparse
import pathlib
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression


def load_data(path: pathlib.Path) -> pd.DataFrame:
    """Read CSV → DataFrame and basic sanity checks."""
    if not path.exists():
        sys.exit(f"[ERROR] CSV file not found: {path}")
    df = pd.read_csv(path)
    required = {"temperature", "humidity", "hour", "is_weekend", "consumption"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        sys.exit(f"[ERROR] Missing columns in CSV: {', '.join(missing)}")
    return df


def build_pipeline(include_categorical: bool, df: pd.DataFrame) -> Pipeline:
    """Create preprocessing + LinearRegression pipeline."""
    numeric_features = ["temperature", "humidity", "hour", "is_weekend"]
    transformers = [("num", StandardScaler(), numeric_features)]

    if include_categorical:
        cat_features = ["season", "district_type"]
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_features))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    model = LinearRegression()
    return Pipeline(steps=[("prep", preprocessor), ("lr", model)])


def run_experiment(df: pd.DataFrame, include_categorical: bool, tag: str) -> None:
    """Train, evaluate, plot."""
    features = ["temperature", "humidity", "hour", "is_weekend"]
    if include_categorical:
        if {"season", "district_type"}.issubset(df.columns):
            features += ["season", "district_type"]
        else:
            print(f"[WARN] {tag}: categorical columns not found – skipping Task 2.")
            return

    X = df[features]
    y = df["consumption"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_pipeline(include_categorical, df)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print(f"\n=== {tag} RESULTS ===")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--")
    plt.xlabel("True Consumption (kWh)")
    plt.ylabel("Predicted Consumption (kWh)")
    plt.title(f"{tag}: True vs Predicted (MAPE {mape:.2f}%)")
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Energy consumption regression")
    parser.add_argument("--data", type=pathlib.Path, default="data.csv",
                        help="Path to CSV file (default: data.csv)")
    args = parser.parse_args()

    df = load_data(args.data)

    run_experiment(df, include_categorical=False, tag="Task 1 (Numeric)")

    run_experiment(df, include_categorical=True, tag="Task 2 (Numeric+Categorical)")


if __name__ == "__main__":
    main()
