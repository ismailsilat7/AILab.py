import csv
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# best performing model
MODEL, MODEL_NAME = DecisionTreeClassifier(
    max_depth=7,
    min_samples_split=5,
    min_samples_leaf=8,
    min_impurity_decrease=0.0,
    max_features=None,
    criterion="entropy",
    random_state=42
), "Decision Tree (d=7, tuned)"


def main():
    print(f"Training model: {MODEL_NAME}")

    # Load full training data (train on ALL of it, no holdout)
    evidence, labels = load_train_data("train.csv")
    MODEL.fit(evidence, labels)
    print("Training complete.")

    # Load test data
    test_ids, test_evidence = load_test_data("test.csv")
    print(f"Loaded {len(test_ids)} test samples.")

    # Make predictions
    predictions = MODEL.predict(test_evidence)

    # Convert numeric predictions back to class labels
    reverse_label_map = { 0: "Low", 1: "Medium", 2: "High" }
    pred_labels = [reverse_label_map[p] for p in predictions]

    # Save submission.csv
    save_submission(test_ids, pred_labels)
    print("Saved: submission.csv")


def load_train_data(filename):
    evidence = []
    labels = []

    # LABEL ENCODING
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
        next(reader)  # skip header
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


def load_test_data(filename):
    test_ids = []
    test_evidence = []

    # LABEL ENCODING
    soil_type_map =       { "Clay": 0, "Loamy": 1, "Sandy": 2, "Silt": 3 }
    crop_type_map =       { "Cotton": 0, "Maize": 1, "Potato": 2, "Rice": 3, "Sugarcane": 4, "Wheat": 5 }
    growth_stage_map =    { "Flowering": 0, "Harvest": 1, "Sowing": 2, "Vegetative": 3 }
    season_map =          { "Kharif": 0, "Rabi": 1, "Zaid": 2 }
    irrigation_type_map = { "Canal": 0, "Drip": 1, "Rainfed": 2, "Sprinkler": 3 }
    water_source_map =    { "Groundwater": 0, "Rainwater": 1, "Reservoir": 2, "River": 3 }
    mulching_map =        { "No": 0, "Yes": 1 }
    region_map =          { "Central": 0, "North": 1, "East": 2, "West": 3, "South": 4 }

    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            test_ids.append(int(row[0]))  # id column
            test_evidence.append([
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

    return test_ids, test_evidence


def save_submission(test_ids, pred_labels, filename="submission.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "Irrigation_Need"])  # header
        for id_, label in zip(test_ids, pred_labels):
            writer.writerow([id_, label])

if __name__ == "__main__":
    main()