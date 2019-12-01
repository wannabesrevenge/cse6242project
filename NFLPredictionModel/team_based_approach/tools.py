from typing import List, Tuple

import bson

import os

import numpy as np
from keras import losses

local_path = os.path.dirname(os.path.abspath(__file__))


class LoadingTools:
    @staticmethod
    def write_data(name: str, description: str, axis_labels: List, training_data: Tuple[List, List, List], validation_data: Tuple[List, List, List]):
        data = bson.dumps({
            "description": description,
            "features": axis_labels[0],
            "outputs": axis_labels[1],
            "training_data": training_data,
            "validation_data": validation_data
        })
        f = open(local_path + "/data/" + name + ".bson", 'wb')
        f.write(data)
        f.close()

    @staticmethod
    def read_data(name: str):
        f = open(local_path + "/data/" + name, 'rb')
        data = f.read()
        f.close()
        return bson.loads(data)

    @staticmethod
    def get_data(name: str, description: str, axis_labels: List, training_data: Tuple[List, List, List], validation_data: Tuple[List, List, List], feature_normalization):
        return {
            "description": description,
            "features": axis_labels[0],
            "outputs": axis_labels[1],
            "training_data": training_data,
            "validation_data": validation_data,
            "normalization": feature_normalization
        }


class GameTools:
    @staticmethod
    def max_time_between_games(sorted_games):
        game_time_deltas = []
        team_delta = {}
        for r in sorted_games:
            away = r.away_team
            home = r.home_team

            if away not in team_delta:
                team_delta[away] = r.time

            if home not in team_delta:
                team_delta[home] = r.time

            # ??
            game_time_deltas.append(r.time - team_delta[away])
            game_time_deltas.append(r.time - team_delta[home])

            team_delta[home] = r.time
            team_delta[away] = r.time
        return max(game_time_deltas)


class DataTools:
    @staticmethod
    def compute_errors(true_y, predicted_y):
        samples = len(predicted_y)
        features = true_y.shape[1]
        accuracy = np.zeros(features)
        loss = np.zeros(features)
        mse = np.zeros(features)

        for x in range(samples):
            r = predicted_y[x]
            for p in range(features):
                delta = abs(r[p] - true_y[x][p])
                mse[p] += (true_y[x][p] - r[p]) ** 2
                loss[p] += delta
                accuracy[p] += abs(1.0 if delta < 0.5 else 0.0)

        mse = mse / samples
        rmse = (mse / samples) ** .5
        avg_rmse = np.average(losses.mean_squared_error(true_y, predicted_y))
        avg_rmse2 = sum(losses.mean_squared_error(true_y, predicted_y)) / samples
        return accuracy / samples, loss / samples, rmse

    @staticmethod
    def keep_indexes(indices, X):
        new_x = []
        # new_y = []
        for i in indices:
            new_x.append(X[:, i])
            # new_y.append(y[:, i])
        return np.column_stack(new_x)

    @staticmethod
    def keep_features(data, keep_features: List):
        indices = []
        kept_labels = []
        for i in range(len(data["features"])):
            v = data["features"][i]
            if any([x in v for x in keep_features]):
                indices.append(i)
                kept_labels.append(data["features"][i])
        return indices


