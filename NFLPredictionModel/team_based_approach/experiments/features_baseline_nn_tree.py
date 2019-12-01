# small_features
import numpy as np
from neural_structures import NeuralNetwork
from team_based_approach.data_building import TeamDataBuilder
from team_based_approach.tools import DataTools

# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()

print("Loading Data")
# small_features = LoadingTools.read_data("baseline_series_short_memory_1_1_1_1.bson")
# small_features = LoadingTools.read_data("baseline_series_medium_memory_1_2_5_25_75.bson")
# small_features = LoadingTools.read_data("baseline_series_short_memory_simple.bson")
print("Data Loaded")

# Initial, Depth, Decay, -> Total Nuerons, Train Epochs,Train Time, log-loss in-sample, log-loss validation

# features = len(small_features["features"])
# print("# of features: "+str(features))
# outputs = len(small_features["outputs"])
# keep_index = small_features["outputs"].index("home_wins")
#
# training_data = small_features["training_data"]
# X = np.array(training_data[0])
# y = np.array(training_data[1])[:, keep_index]
#
# validation_data = small_features["validation_data"]
# vx = np.array(validation_data[0])
# vy = np.array(validation_data[1])[:, keep_index]
#
# model = XGBClassifier(n_estimators=1000)
# model.fit(X, y)
#
# out = model.predict(vx)
#
# target_shape = len(vy), 1
# errors = DataTools.compute_errors(np.reshape(vy, target_shape), np.reshape(np.array(out), target_shape))
# print("Accuracy: ", errors[0])
# print("Loss: ", errors[1])
# print("RMSE: ", errors[2])
history = [1, 2, 5, 7, 18, 27, 50]
data_set = TeamDataBuilder.build_data(history, postseason=False, preseason=False, randomness=0.01, warmup=50*32,loops=16)
# initial_size = [2.0]
# for s in initial_size:

# network = NeuralNetwork.create_network(len(kept_labels), 1, int(len(kept_labels) * 2.0), 3, activations=['selu'])
# network.train((X, y), (vx, vy))
# out = network.predict(vx)
features = len(data_set["features"])
network = NeuralNetwork.create_network(features, 1, int(features * 6.0), 6, activations=['selu'])
keep_index = data_set["outputs"].index("home_wins")

training_data = data_set["training_data"]
X = np.array(training_data[0])  # DataTools.keep_indexes(indices, np.array(training_data[0]))
y = np.array(training_data[1])[:, keep_index]

validation_data = data_set["validation_data"]
vx = np.array(validation_data[0])  # DataTools.keep_indexes(indices, np.array(validation_data[0]))
vy = np.array(validation_data[1])[:, keep_index]

network.train((X, y), (vx, vy))

out = network.predict(vx)

target_shape = len(vy), 1
errors = DataTools.compute_errors(np.reshape(vy, target_shape), np.reshape(np.array(out), target_shape))

print("Accuracy: ", errors[0])
print("Loss: ", errors[1])
print("RMSE: ", errors[2])
#64.4