# Build and Save The Best Tree
# Data
# Data Filtered
# XGBRegressor

# Save Tree
# Print Validation Accuracy / Loss
from xgboost import XGBRegressor, XGBClassifier

from team_based_approach.tools import DataTools
import numpy as np


def build_best_tree(data, features):
    indices = DataTools.keep_features(data, features)
    keep_index = data["outputs"].index("home_wins")

    training_data = data["training_data"]
    X = DataTools.keep_indexes(indices, np.array(training_data[0]))
    y = np.array(training_data[1])[:, keep_index]

    validation_data = data["validation_data"]
    vx = DataTools.keep_indexes(indices, np.array(validation_data[0]))
    vy = np.array(validation_data[1])[:, keep_index]

    model = XGBClassifier(n_estimators=1000)
    model.fit(X, y)

    out = model.predict(vx)
    target_shape = len(vy), 1
    errors = DataTools.compute_errors(np.reshape(vy, target_shape), np.reshape(np.array(out), target_shape))

    print("Accuracy (Decision Tree): " + str(errors[0]))
    print("Loss (Decision Tree): " + str(errors[1]))

    return model


def tree_output(data, features, tree):
    indices = DataTools.keep_features(data, features)
    training_data = data["training_data"]
    X = DataTools.keep_indexes(indices, np.array(training_data[0]))

    validation_data = data["validation_data"]
    vx = DataTools.keep_indexes(indices, np.array(validation_data[0]))

    return tree.predict(X), tree.predict(vx)
