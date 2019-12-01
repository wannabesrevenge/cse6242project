from typing import List

import numpy as np
from keras.layers import LeakyReLU

from neural_structures import NeuralNetwork
from team_based_approach.data_building import TeamDataBuilder
from team_based_approach.tools import DataTools


def compute_accuracy(keep_features: List, data):
    indices = []
    kept_labels = []
    for i in range(len(data["features"])):
        v = data["features"][i]
        if any([x in v for x in keep_features]):
            indices.append(i)
            kept_labels.append(data["features"][i])

    keep_index = data["outputs"].index("home_wins")

    training_data = data["training_data"]
    X = DataTools.keep_indexes(indices, np.array(training_data[0]))
    y = np.array(training_data[1])[:, keep_index]

    validation_data = data["validation_data"]
    vx = DataTools.keep_indexes(indices, np.array(validation_data[0]))
    vy = np.array(validation_data[1])[:, keep_index]

    network = NeuralNetwork.create_network(len(kept_labels), 1, int(len(kept_labels) * 2.0), 3, activations=[lambda x: LeakyReLU(alpha=0.1)(x)])
    network.train((X, y), (vx, vy))

    out = network.predict(vx)

    target_shape = len(vy), 1
    errors = DataTools.compute_errors(np.reshape(vy, target_shape), np.reshape(np.array(out), target_shape))
    return errors[1], errors[0]


values = []

history = [2, 7, 18, 27]
data_set = TeamDataBuilder.build_data(history, postseason=False, preseason=False, randomness=.01, loops=32)
# For each type...
base_features = set([f.replace("away", "").replace("home", "") for f in data_set['features']])
print("Finished Building Features")

current_features = []
best_features = []
best_value = 1000.0
best_accuracy = 0.0
exists_better = True
for i in range(20):
    values = []
    cnt = 0
    for b in base_features:
        cnt += 1
        ratio = cnt/len(base_features)
        print("Complete: "+str(ratio))
        features = [b] + current_features
        loss, accuracy = compute_accuracy(features, data_set)
        values.append((features, b, loss[0], accuracy[0]))
        if best_value > loss[0]:
            best_features = features
            best_value = loss[0]
            best_accuracy = accuracy[0]
            # exists_better = True

    m = min(values, key=lambda x: x[2])
    base_features.remove(m[1])
    print(str(len(m[0]))+", "+str(m[2])+", "+str(m[3])+", "+str(m[1]))
    current_features.append(m[1])

print("Best Features: ", best_features)
print("TODO? Split Each Feature, Progressively Add One")


