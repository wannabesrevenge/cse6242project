from typing import List

import numpy as np

from neural_structures import NeuralNetwork
from team_based_approach.data_building import TeamDataBuilder
from team_based_approach.tools import DataTools


def compute_accuracy(hist, keep_features: List):
    data = TeamDataBuilder.build_data(hist, postseason=False, preseason=False)

    indices = []
    kept_labels = []
    for i in range(len(data["features"])):
        v = data["features"][i]
        if any([x in v for x in keep_features]) and not any([x in v for x in ["delta", "std"]]):
            indices.append(i)
            kept_labels.append(data["features"][i])

    keep_index = data["outputs"].index("home_wins")

    training_data = data["training_data"]
    X = DataTools.keep_indexes(indices, np.array(training_data[0]))
    y = np.array(training_data[1])[:, keep_index]

    validation_data = data["validation_data"]
    vx = DataTools.keep_indexes(indices, np.array(validation_data[0]))
    vy = np.array(validation_data[1])[:, keep_index]

    network = NeuralNetwork.create_network(len(kept_labels), 1, int(len(kept_labels) * 2.0), 3, activations=['selu'])
    network.train((X, y), (vx, vy))

    out = network.predict(vx)

    target_shape = len(vy), 1
    errors = DataTools.compute_errors(np.reshape(vy, target_shape), np.reshape(np.array(out), target_shape))
    # single_value.append((best, errors[0], errors[1]))
    print("Features: " + str(keep_features))
    print("History: " + str(hist))
    print("Accuracy: ", errors[0])
    print("Loss: ", errors[1])
    print("RMSE: ", errors[2])
    print("\n")
    # Print top 3 most important features.
    # Join Features together [Name], [Name_delta], [Name_std], [Name_delta_std]
    return errors[1], errors[0]


# def _filter_features()
feature_options = ["penalties", "penalty_yards", "punts", "punt_yards",
                   "rushing_yards", "passing_yards",
                   "first_downs", "turnovers", "total_yards", "last_played"]
features = ["win", "score"]

single_value = [i for i in range(1, 36)]

max_value = 0.0
max_feature = features
values = []

for f in single_value:
    loss, accuracy = compute_accuracy([f], features)
    values.append((f, loss[0], accuracy[0]))

for v in values:
    print(str(v[0])+", "+str(v[1])+", "+str(v[2]))