import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

TEST_SIZE = 0.4

FEATURE_NAMES = [
    "Income",
    "Employment Status",
    "Credit Score",
    "Loan Amount",
    "Marital Status",
    "Loan Term (months)",
    "Existing Debt"
]


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python task2.py data.csv")

    evidence, labels = load_data(sys.argv[1])

    if not evidence:
        sys.exit("No valid data loaded.")

    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE, random_state=42
    )

    model = train_model(X_train, y_train)

    print("Feature Importance")
    feature_importance(model, FEATURE_NAMES)
    print()

    predictions = model.predict(X_test)

    accuracy, precision, recall, f1 = evaluate(y_test, predictions)

    print("Model Evaluation")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print()

    new_applicant = [[55000, 2, 720, 15000, 1, 36, 3000]]
    predicted_label = model.predict(new_applicant)

    outcome = "Approved" if predicted_label[0] == 1 else "Rejected"

    print("New Applicant Prediction")
    print(outcome)


def load_data(filename):
    evidence = []
    labels = []

    employment_map = {
        "Unemployed": 0,
        "Self-Employed": 1,
        "Employed": 2
    }

    marital_map = {
        "Single": 0,
        "Married": 1,
        "Divorced": 2
    }

    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            try:
                income = float(row[1])
                employment = employment_map[row[2]]
                credit = float(row[3])
                loan_amount = float(row[4])
                marital = marital_map[row[5]]
                term = float(row[6])
                debt = float(row[7])
                approved = int(row[8])

                evidence.append([
                    income,
                    employment,
                    credit,
                    loan_amount,
                    marital,
                    term,
                    debt
                ])

                labels.append(approved)

            except:
                continue

    return evidence, labels


def train_model(evidence, labels):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(evidence, labels)
    return model


def feature_importance(model, feature_names):
    importances = list(zip(feature_names, model.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)

    for name, score in importances:
        print(name, round(score, 4))


def evaluate(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    return accuracy, precision, recall, f1


if __name__ == "__main__":
    main()