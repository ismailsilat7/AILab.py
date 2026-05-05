import csv
import sys
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier

TEST_SIZE = 0.2

# choose 1 model at a time
# MODEL, MODEL_NAME = KNeighborsClassifier(n_neighbors=5),"KNN (k=5)"
# MODEL, MODEL_NAME = KNeighborsClassifier(n_neighbors=11),"KNN (k=11)"
# MODEL, MODEL_NAME = GaussianNB(),"Naive Bayes"
# MODEL, MODEL_NAME = LogisticRegression(max_iter=5000, random_state=42),"Logistic Regression"
# MODEL, MODEL_NAME = DecisionTreeClassifier(random_state=42),"Decision Tree"
MODEL, MODEL_NAME = DecisionTreeClassifier(max_depth=7, random_state=42), "Decision Tree (d=7)"
"""
MODEL, MODEL_NAME = DecisionTreeClassifier(
    max_depth=7,
    min_samples_split=5,
    min_samples_leaf=8,
    min_impurity_decrease=0.0,
    max_features=None,
    criterion="entropy",
    random_state=42
), "Decision Tree (d=7, tuned)"
"""
# MODEL, MODEL_NAME = RandomForestClassifier(n_estimators=50, random_state=42),"Random Forest"


def main():
    """
    # flow:
    train_model() --> model --> model.predict() --> evaluate --> confusion matrix
    """

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python train.py data.csv")

    # Load dataset
    evidence, labels = load_data(sys.argv[1])

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        evidence,
        labels,
        test_size=TEST_SIZE,
        random_state=42
    )

    # Train model
    model = train_model(X_train, y_train)

    # K-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        MODEL, X_train, y_train,
        cv=kf,
        scoring='balanced_accuracy'
    )

    # Make predictions
    predictions = model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    # Evaluate model
    test_score = evaluate(y_test, predictions)

    # Print results
    print(f"Model: {MODEL_NAME}")
    print(f"CV Score (5-Fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Test Score:        {test_score:.4f}")
    print(f"Confusion Matrix:")
    print(cm)


def load_data(filename):
    evidence = []
    labels = []

    # LABEL ENCODING
    soil_type_map =      { "Clay": 0, "Loamy": 1, "Sandy": 2, "Silt": 3 }
    crop_type_map =      { "Cotton": 0, "Maize": 1, "Potato": 2, "Rice": 3, "Sugarcane": 4, "Wheat": 5 }
    growth_stage_map =   { "Flowering": 0, "Harvest": 1, "Sowing": 2, "Vegetative": 3 }
    season_map =         { "Kharif": 0, "Rabi": 1, "Zaid": 2 }
    irrigation_type_map= { "Canal": 0, "Drip": 1, "Rainfed": 2, "Sprinkler": 3 }
    water_source_map =   { "Groundwater": 0, "Rainwater": 1, "Reservoir": 2, "River": 3 }
    mulching_map =       { "No": 0, "Yes": 1 }
    region_map =         { "Central": 0, "North": 1, "East": 2, "West": 3, "South": 4 }
    label_map =          { "Low": 0, "Medium": 1, "High": 2 }

    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            evidence.append([
                soil_type_map[row[1]],
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
                float(row[6]),
                float(row[7]),
                float(row[8]),
                float(row[9]),
                float(row[10]),
                crop_type_map[row[11]],
                growth_stage_map[row[12]],
                season_map[row[13]],
                irrigation_type_map[row[14]],
                water_source_map[row[15]],
                float(row[16]),
                mulching_map[row[17]],
                float(row[18]),
                region_map[row[19]]
            ])
            labels.append(label_map[row[20]])
    return evidence, labels


def train_model(X_training, y_training):
    """
    MODEL defined glabally
    """
    MODEL.fit(X_training, y_training)
    return MODEL


def evaluate(y_true, y_pred):
    # Balanced accuracy required as evaluation metric by kaggle
    score = balanced_accuracy_score(y_true, y_pred)
    return score


if __name__ == "__main__":
    main()