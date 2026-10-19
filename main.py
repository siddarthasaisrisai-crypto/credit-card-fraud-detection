from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "transactions.csv"
LOG_PATH = ROOT / "data" / "scored_transactions.csv"
FEATURES = [
    "amount",
    "transactions_last_24h",
    "international",
    "account_age_days",
    "late_night",
]
TARGET = "is_fraud"


def load_data():
    data = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if data[TARGET].nunique() < 2:
        raise ValueError("Training data must contain both fraud and non-fraud examples.")
    return data


def train_model(data):
    x = data[FEATURES]
    y = data[TARGET].astype(int)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced"
        )),
    ])
    # Evaluation is only illustrative for this tiny synthetic sample.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    model.fit(x_train, y_train)
    print("\nIllustrative evaluation on synthetic sample data:")
    print(classification_report(y_test, model.predict(x_test), zero_division=0))
    model.fit(x, y)
    return model


def read_number(prompt, minimum=None):
    while True:
        try:
            value = float(input(prompt).strip())
            if minimum is not None and value < minimum:
                print(f"Enter a value greater than or equal to {minimum}.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")


def score_transaction(model):
    transaction = {
        "amount": read_number("Transaction amount: ", 0),
        "transactions_last_24h": read_number("Transactions in last 24 hours: ", 0),
        "international": int(read_number("International transaction? (0=no, 1=yes): ", 0)),
        "account_age_days": read_number("Account age in days: ", 0),
        "late_night": int(read_number("Late-night transaction? (0=no, 1=yes): ", 0)),
    }
    if transaction["international"] not in (0, 1) or transaction["late_night"] not in (0, 1):
        print("International and late-night values must be 0 or 1.")
        return
    frame = pd.DataFrame([transaction], columns=FEATURES)
    probability = float(model.predict_proba(frame)[0][1])
    prediction = int(probability >= 0.5)
    label = "POTENTIALLY SUSPICIOUS" if prediction else "NOT FLAGGED BY DEMO MODEL"
    print(f"\nResult: {label}")
    print(f"Model fraud score: {probability:.1%}")
    print("Note: this demo model is not suitable for real payment decisions.")
    row = transaction | {"fraud_score": probability, "prediction": prediction}
    exists = LOG_PATH.exists()
    pd.DataFrame([row]).to_csv(LOG_PATH, mode="a", header=not exists, index=False)
    print(f"Saved to: {LOG_PATH}")


def main():
    try:
        data = load_data()
        model = train_model(data)
    except Exception as exc:
        print(f"Could not initialize model: {exc}")
        return

    while True:
        print("\n=== Credit Card Fraud Detection ===")
        print("1. Score a transaction")
        print("2. Show training data")
        print("3. Exit")
        choice = input("Choose an option: ").strip()
        if choice == "1":
            score_transaction(model)
        elif choice == "2":
            print(data.to_string(index=False))
        elif choice == "3":
            print("Goodbye.")
            break
        else:
            print("Choose 1, 2, or 3.")


if __name__ == "__main__":
    main()
