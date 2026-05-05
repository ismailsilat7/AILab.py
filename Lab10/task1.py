import csv
import sys
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TEST_SIZE = 0.4

FEATURE_NAMES = [
    "Study Hours",
    "Attendance (%)",
    "Previous Grade",
    "Participation Level",
    "Internet Usage (hrs)",
    "Assignments Submitted"
]

def main():

    if len(sys.argv) != 2:
        sys.exit("Usage: python task1.py data.csv")

    evidence, labels = load_data(sys.argv[1])

    if not evidence:
        sys.exit("No valid data loaded. Check your CSV file.")

    print(f"Loaded {len(evidence)} valid records.\n")

    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE, random_state=42
    )

    model = train_model(X_train, y_train)

    print("Feature Importance (Linear Coefficients)")
    feature_importance(model, FEATURE_NAMES)
    print()

    predictions = model.predict(X_test)

    mae, rmse, r2 = evaluate(y_test, predictions)

    print("Model Evaluation")
    print(f"MAE:      {mae:.2f}")
    print(f"RMSE:     {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    print()

    # predict for a new student
    # [Study Hours, Attendance (%), Previous Grade, Participation (0=Low,1=Med,2=High), Internet Usage, Assignments]
    new_student = [[6.5, 88.0, 75.0, 2, 3.5, 5.0]]
    predicted_score = model.predict(new_student)
    print("New Student Prediction")
    print(f"Features : Study Hours=6.5, Attendance=88%, Prev Grade=75, "
          f"Participation=High, Internet=3.5hrs, Assignments=5")
    print(f"Predicted Final Score: {predicted_score[0]:.2f}")


def load_data(filename):
    """
    Load and clean dataset from CSV.
    - Skips rows with missing or invalid values (handles missing attendance/study hours).
    - Encodes participation level: Low=0, Medium=1, High=2.
    - Computes column means from valid rows to allow imputation if desired.
    """
    evidence = []
    labels = []
    skipped = 0

    participation_map = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            try:
                study_hours     = float(row[1]) if row[1].strip() else None
                attendance      = float(row[2]) if row[2].strip() else None
                previous_grade  = float(row[3]) if row[3].strip() else None
                participation   = participation_map.get(row[4].strip(), None)
                internet_usage  = float(row[5]) if row[5].strip() else None
                assignments     = float(row[6]) if row[6].strip() else None
                final_score     = float(row[7]) if row[7].strip() else None
                if None in [study_hours, attendance, previous_grade,
                            participation, internet_usage, assignments, final_score]:
                    skipped += 1
                    continue

                evidence.append([
                    study_hours,
                    attendance,
                    previous_grade,
                    participation,
                    internet_usage,
                    assignments
                ])
                labels.append(final_score)

            except (ValueError, IndexError):
                skipped += 1
                continue

    if skipped > 0:
        print(f"Warning: {skipped} rows skipped due to missing/invalid data.\n")

    return evidence, labels


def train_model(evidence, labels):
    model = LinearRegression()
    model.fit(evidence, labels)
    return model


def feature_importance(model, feature_names):
    coefficients = list(zip(feature_names, model.coef_))
    coefficients.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, coef in coefficients:
        direction = "+" if coef >= 0 else "-"
        print(f"  {direction} {name:<30} coefficient: {coef:.4f}")


def evaluate(labels, predictions):
    mae  = mean_absolute_error(labels, predictions)
    rmse = mean_squared_error(labels, predictions) ** 0.5
    r2   = r2_score(labels, predictions)
    return mae, rmse, r2


if __name__ == "__main__":
    main()