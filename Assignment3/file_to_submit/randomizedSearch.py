import csv
import sys
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier

TEST_SIZE = 0.2
SAMPLE_SIZE = 0.3   # fraction of training data used for grid search
N_ITER = 20         # number of random combinations to try

MODEL_NAME = "Decision Tree (d=7, RandomizedSearch)"

PARAM_DIST = {
    "min_samples_split":     [2, 5, 10, 20],
    "min_samples_leaf":      [1, 2, 4, 8],
    "criterion":             ["gini", "entropy"],
    "max_features":          [None, "sqrt", "log2"],
    "min_impurity_decrease": [0.0, 0.001, 0.01]
}

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python train.py data.csv")

    evidence, labels = load_data(sys.argv[1])

    X_train, X_test, y_train, y_test = train_test_split(
        evidence,
        labels,
        test_size=TEST_SIZE,
        random_state=42
    )

    print(f"Training rows:       {len(X_train):,}")
    print(f"Sample for search:   {int(len(X_train) * SAMPLE_SIZE):,}")
    print(f"Random combinations: {N_ITER}\n")

    # find best params on a sample
    best_params = find_best_params(X_train, y_train)
    print(f"\nBest Params: {best_params}\n")

    # refit best model on FULL training data
    model = DecisionTreeClassifier(max_depth=7, random_state=42, **best_params)
    model.fit(X_train, y_train)
    print("Refit on full training data: done\n")

    # K-Fold CV on full training data with best model
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=kf,
        scoring='balanced_accuracy',
        n_jobs=-1
    )

    # evaluate on held-out test set
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    test_score = evaluate(y_test, predictions)

    print(f"Model:             {MODEL_NAME}")
    print(f"CV Score (5-Fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Test Score:        {test_score:.4f}")
    print(f"Confusion Matrix:")
    print(cm)


def find_best_params(X_train, y_train):
    # Sample 30% of training data for the search
    X_sample, _, y_sample, _ = train_test_split(
        X_train, y_train,
        train_size=SAMPLE_SIZE,
        random_state=42,
        stratify=y_train       # keep class balance in sample
    )

    base = DecisionTreeClassifier(max_depth=7, random_state=42)

    search = RandomizedSearchCV(
        base,
        PARAM_DIST,
        n_iter=N_ITER,
        cv=3,                  # 3-fold on sample is enough to find direction
        scoring='balanced_accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=2
    )
    search.fit(X_sample, y_sample)

    print(f"Best CV score on sample: {search.best_score_:.4f}")
    return search.best_params_


def load_data(filename):
    evidence = []
    labels = []

    soil_type_map =       { "Clay": 0, "Loamy": 1, "Sandy": 2, "Silt": 3 }
    crop_type_map =       { "Cotton": 0, "Maize": 1, "Potato": 2, "Rice": 3, "Sugarcane": 4, "Wheat": 5 }
    growth_stage_map =    { "Flowering": 0, "Harvest": 1, "Sowing": 2, "Vegetative": 3 }
    season_map =          { "Kharif": 0, "Rabi": 1, "Zaid": 2 }
    irrigation_type_map = { "Canal": 0, "Drip": 1, "Rainfed": 2, "Sprinkler": 3 }
    water_source_map =    { "Groundwater": 0, "Rainwater": 1, "Reservoir": 2, "River": 3 }
    mulching_map =        { "No": 0, "Yes": 1 }
    region_map =          { "Central": 0, "North": 1, "East": 2, "West": 3, "South": 4 }
    label_map =           { "Low": 0, "Medium": 1, "High": 2 }

    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            evidence.append([
                soil_type_map[row[1]],
                float(row[2]), float(row[3]), float(row[4]),
                float(row[5]), float(row[6]), float(row[7]),
                float(row[8]), float(row[9]), float(row[10]),
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


def evaluate(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)


if __name__ == "__main__":
    main()