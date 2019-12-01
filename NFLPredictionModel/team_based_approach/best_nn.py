#  Build and save the best nueral_network

from neural_structures import NeuralNetwork
from team_based_approach.tools import DataTools
import numpy as np


def build_best_nn(data, features):
    indices = DataTools.keep_features(data, features)
    keep_index = data["outputs"].index("home_wins")

    training_data = data["training_data"]
    X = DataTools.keep_indexes(indices, np.array(training_data[0]))
    y = np.array(training_data[1])[:, keep_index]

    validation_data = data["validation_data"]
    vx = DataTools.keep_indexes(indices, np.array(validation_data[0]))
    vy = np.array(validation_data[1])[:, keep_index]

    network = NeuralNetwork.create_network(len(indices), 1, int(len(indices) * 2.0), 3, activations=['relu'])
    network.train((X, y), (vx, vy))

    out = network.predict(vx)
    target_shape = len(vy), 1
    errors = DataTools.compute_errors(np.reshape(vy, target_shape), np.reshape(np.array(out), target_shape))

    print("Accuracy (Neural Network): " + str(errors[0]))
    print("Loss (Neural Network): " + str(errors[1]))

    return network


def nn_output(data, features, network):
    indices = DataTools.keep_features(data, features)
    training_data = data["training_data"]
    X = DataTools.keep_indexes(indices, np.array(training_data[0]))

    validation_data = data["validation_data"]
    vx = DataTools.keep_indexes(indices, np.array(validation_data[0]))

    return network.predict(X), network.predict(vx)
