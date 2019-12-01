import numpy as np

from neural_structures import NeuralNetwork
from team_based_approach.best_nn import build_best_nn, nn_output
from team_based_approach.best_tree import build_best_tree, tree_output
from team_based_approach.data_building import TeamDataBuilder
from team_based_approach.tools import DataTools
import pandas as pd

history = [2, 7, 18, 27]
data_set = TeamDataBuilder.build_data(history, postseason=False, preseason=False, randomness=0.0, loops=0)

tree_features = ["_rushing_yards_delta_3_9", "win_delta_3_9", "_last_played_std_1_2", "_penalty_yards_delta_std_3_9", "_last_played_delta_std_1_2"]
nn_features = ["_score_delta_3_9", "_turnovers_delta_10_27", "_penalty_yards_delta_10_27", "_first_downs_delta_std_1_2", "_first_downs_std_1_2", "_penalty_yards_delta_3_9"]
best_tree = build_best_tree(data_set, tree_features)
best_nn = build_best_nn(data_set, nn_features)

data_set_total = TeamDataBuilder.build_data(history, team_output=["score", "total_yards"], postseason=False, preseason=False, randomness=0.0, loops=0)
training_data = data_set_total["training_data"]
validation_data = data_set_total["validation_data"]

# Now Build The Best Ensemble To Minimize All The Things
tree_data = tree_output(data_set_total, tree_features, best_tree)
nn_data = nn_output(data_set_total, nn_features, best_nn)

# Keep All The Score & Yard Totals
final_feature_indices = DataTools.keep_features(data_set_total, ["total_yards_1", "total_yards_3", "total_yards_10", "score_1", "score_3", "score_10"])
X = DataTools.keep_indexes(final_feature_indices, np.array(training_data[0]))

# TODO: Join Ensemble Predictions
X = np.column_stack([tree_data[0], nn_data[0], X])

y = np.array(training_data[1])

vx = DataTools.keep_indexes(final_feature_indices, np.array(validation_data[0]))

# TODO: Join Ensemble Predictions
vx = np.column_stack([tree_data[1], nn_data[1], vx])

vy = np.array(validation_data[1])
output_size = len(training_data[1][0])
network = NeuralNetwork.create_network(len(final_feature_indices) + 2, output_size, int(len(final_feature_indices) * 2.0), 3, activations=['linear', 'relu'])
network.train((X, y), (vx, vy))

out_training = network.predict(X)
out_validation = network.predict(vx)
# Replace with tree
# Replace with tree
out_validation[:, 0] = tree_data[1]
out_training[:, 0] = tree_data[0]

target_shape = len(vy), output_size
target_shape2 = len(y), output_size
errors_validation = DataTools.compute_errors(np.reshape(vy, target_shape), np.reshape(np.array(out_validation), target_shape))
errors_training = DataTools.compute_errors(np.reshape(y, target_shape2), np.reshape(np.array(out_training), target_shape2))

# Output All The Final Data
#  nfl_id,home_won,home_win_probability,home_score,predicted_home_score,away_score,predicted_away_score,home_total_yards,predicted_home_total_yards,away_yards,predicted_away_total_yards
# Create Validation Records
validation_records = []
for x in range(len(vy)):
    row = []
    row.append(validation_data[2][x][0])  # Game ID
    row.append(validation_data[2][x][1])  # Home Team
    row.append(validation_data[2][x][2])  # Away Team
    row.append(vy[x][0])  # Did Win
    row.append(out_validation[x][0])  # Win Probability

    row.append(out_validation[x][1] * data_set_total['normalization']['score'])  # Home Score Predicted
    row.append(vy[x][1] * data_set_total['normalization']['score'])  # Home Score Actual

    row.append(out_validation[x][3] * data_set_total['normalization']['score'])  # Away Score Predicted
    row.append(vy[x][3] * data_set_total['normalization']['score'])  # Away Score Actual

    row.append(out_validation[x][2] * data_set_total['normalization']['total_yards'])  # Home Total Yards Predicted
    row.append(vy[x][2] * data_set_total['normalization']['total_yards'])  # Home Total Yards Actual

    row.append(out_validation[x][4] * data_set_total['normalization']['total_yards'])  # Away Total Yards Predicted
    row.append(vy[x][4] * data_set_total['normalization']['total_yards'])  # Away Total Yards Actual

    validation_records.append(row)

# Create Training Records
training_records = []
for x in range(len(vy)):
    row = []
    row.append(training_data[2][x][0])  # Game ID
    row.append(training_data[2][x][1])  # Home Team
    row.append(training_data[2][x][2])  # Away Team
    row.append(y[x][0])  # Did Win
    row.append(out_training[x][0])  # Win Probability

    row.append(out_training[x][1] * data_set_total['normalization']['score'])  # Home Score Predicted
    row.append(y[x][1] * data_set_total['normalization']['score'])  # Home Score Actual

    row.append(out_training[x][3] * data_set_total['normalization']['score'])  # Away Score Predicted
    row.append(y[x][3] * data_set_total['normalization']['score'])  # Away Score Actual

    row.append(out_training[x][2] * data_set_total['normalization']['total_yards'])  # Home Total Yards Predicted
    row.append(y[x][2] * data_set_total['normalization']['total_yards'])  # Home Total Yards Actual

    row.append(out_training[x][4] * data_set_total['normalization']['total_yards'])  # Away Total Yards Predicted
    row.append(y[x][4] * data_set_total['normalization']['total_yards'])  # Away Total Yards Actual

    training_records.append(row)

# Save Data To CSV
df = pd.DataFrame(validation_records,
                  columns=["nfl_id", "home_team", "away_team", "home_won", "home_win_probability", "home_score", "predicted_home_score", "away_score", "predicted_away_score", "home_total_yards",
                           "predicted_home_total_yards", "away_yards", "predicted_away_total_yards"])
df.to_csv('validation_results.csv', index=False)
df = pd.DataFrame(validation_records,
                  columns=["nfl_id", "home_team", "away_team", "home_won", "home_win_probability", "home_score", "predicted_home_score", "away_score", "predicted_away_score", "home_total_yards",
                           "predicted_home_total_yards", "away_yards", "predicted_away_total_yards"])
df.to_csv('training_results.csv', index=False)
# df = pd.DataFrame(training_records + validation_records,
#                   columns=["nfl_id", "home_team", "away_team", "home_won", "home_win_probability", "home_score", "predicted_home_score", "away_score", "predicted_away_score", "home_total_yards",
#                            "predicted_home_total_yards", "away_yards", "predicted_away_total_yards"])
# df.to_csv('all_results.csv', index=False)

# Print Final Statistics
# Print Gap Between Training Set & ...
print("TODO...")
print("Accuracy (who wins): "+str(100.0*errors_validation[0][0])+"%")
print("Loss (who wins): "+str(errors_validation[1][0]))
total_score_errors = data_set_total['normalization']['score']*(np.abs(vy[:,1]-out_validation[:,1]) + np.abs(vy[:,3]-out_validation[:,3]))
total_yards_errors = data_set_total['normalization']['total_yards']*(np.abs(vy[:,2]-out_validation[:,2]) + np.abs(vy[:,4]-out_validation[:,4]))
print("Error Score (avg): "+str(np.mean(total_score_errors)))
print("Error Score (median): "+str(np.median(total_score_errors)))
print("Error Total Yards (avg): "+str(np.mean(total_yards_errors)))
print("Error Total Yards (median): "+str(np.median(total_yards_errors)))

# Outcome, Score Total Avg Error, Total Yard, Score Spread Error
# Preseason Vs All
