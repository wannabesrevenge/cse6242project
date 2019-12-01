import random
from typing import Dict, List, Tuple
import bson
import math
import pandas as pd
import numpy as np
from team_based_approach.data_structures import TeamData, build_data_series, SEASON_VALUE
import os

# Testing Bson Data
# c = bson.dumps({"data": [["A", "B", "C"], [[1.1, 1.2], [1.1, 1.2]], [1.1, 2.2], ["a"]]})
# print("?")
# open("test.out","wb").write(c)
from team_based_approach.tools import LoadingTools, GameTools


def max_feature(f, new_f):
    for k, v in list(f.items()):
        if v < new_f[k]:
            f[k] = new_f[k]
    return f


def _make_features(row, pre_fix, time_since_last=0.0, home=False) -> Dict[str, float]:
    # if home > away
    home_win = 1.0 if row["home_score"] > row["away_score"] else 0.0
    data = {
        "penalties": row[pre_fix + "pen"],
        "penalty_yards": row[pre_fix + "penyds"],
        "punts": row[pre_fix + "pt"],
        "punt_yards": row[pre_fix + "ptyds"],
        "score": row[pre_fix + "score"],
        "rushing_yards": row[pre_fix + "ryds"],
        "passing_yards": row[pre_fix + "pyds"],
        "first_downs": row[pre_fix + "totfd"],
        "total_yards": row[pre_fix + "totyds"],
        "turnovers": row[pre_fix + "trnovr"],
        "win": home_win if home else abs(1.0 - home),  # Tie is technically possible...
        "last_played": time_since_last,  # TODO: log step over all the data
        "preseason": SEASON_VALUE[row["season_type"]] if isinstance(row["season_type"], str) else 0.0
    }
    # if data["preseason"] != 0.5:
    #     print("wtf?")
    return data


def _make_features_random(randomness=0.0):
    def _make_features2(row, pre_fix, time_since_last=0.0, home=False) -> Dict[str, float]:
        # if home > away
        home_win = 1.0 if row["home_score"] > row["away_score"] else 0.0
        h_value = home_win if home else abs(1.0 - home)
        data = {
            "penalties": row[pre_fix + "pen"] + row[pre_fix + "pen"]*randomness*random.random() * random.choice((-1, 1)),
            "penalty_yards": row[pre_fix + "penyds"] + row[pre_fix + "penyds"]*randomness*random.random() * random.choice((-1, 1)),
            "punts": row[pre_fix + "pt"] + row[pre_fix + "pt"]*randomness*random.random() * random.choice((-1, 1)),
            "punt_yards": row[pre_fix + "ptyds"] + row[pre_fix + "ptyds"]*randomness*random.random() * random.choice((-1, 1)),
            "score": row[pre_fix + "score"] + row[pre_fix + "score"]*randomness*random.random() * random.choice((-1, 1)),
            "rushing_yards": row[pre_fix + "ryds"] + row[pre_fix + "ryds"]*randomness*random.random() * random.choice((-1, 1)),
            "passing_yards": row[pre_fix + "pyds"] + row[pre_fix + "pyds"]*randomness*random.random() * random.choice((-1, 1)),
            "first_downs": row[pre_fix + "totfd"] + row[pre_fix + "totfd"]*randomness*random.random() * random.choice((-1, 1)),
            "total_yards": row[pre_fix + "totyds"] + row[pre_fix + "totyds"]*randomness*random.random() * random.choice((-1, 1)),
            "turnovers": row[pre_fix + "trnovr"] + row[pre_fix + "trnovr"]*randomness*random.random() * random.choice((-1, 1)),
            "win": h_value + 1.0 * randomness*random.random() * random.choice((-1, 1)),  # Tie is technically possible...
            "last_played": time_since_last + time_since_last * randomness*random.random() * random.choice((-1, 1)),  # TODO: log step over all the data
            "preseason": SEASON_VALUE[row["season_type"]] if isinstance(row["season_type"], str) else 0.0 + randomness*random.random()
        }
        # if data["preseason"] != 0.5:
        #     print("wtf?")
        return data
    return _make_features2

local_path = os.path.dirname(os.path.abspath(__file__))
game_states = pd.read_csv(local_path + "/../data/game_stats.csv", engine='python')
sorted_games = sorted([x[1] for x in list(game_states.iterrows())], key=lambda x: x["time"])

feature_normalization = _make_features(sorted_games[0], "home_")
feature_normalization["last_played"] = GameTools.max_time_between_games(sorted_games)
for s in sorted_games:
    max_feature(feature_normalization, _make_features(s, "home_", time_since_last=sorted_games[-1].time - sorted_games[0].time))
    max_feature(feature_normalization, _make_features(s, "away_", time_since_last=sorted_games[-1].time - sorted_games[0].time))


# % Validation
# Count Warm-UP
# Reg season only
class TeamDataBuilder:
    @staticmethod
    def build_data(feature_hist: List[int], team_output: List[str] = [], warmup=150, validation=.15, preseason=True, postseason=True, loops=0, randomness=0.01):
        # Filter game data
        this_data = sorted_games[:]
        if not preseason:
            this_data = filter(lambda x: x["season_type"] != "PRE", this_data)

        if not postseason:
            this_data = filter(lambda x: x["season_type"] != "POST", this_data)

        this_data = filter(lambda x: isinstance(x["season_type"], str), this_data)
        this_data = list(this_data)

        size = len(this_data)
        validation_cnt = int(size * validation)

        training_data_series = [(0, size - validation_cnt, warmup)]
        validation_data_series = [(0, size, size - validation_cnt)]  # ["validation_data"]
        for x in range(loops):
            training_data_series.append(training_data_series[0])
        X_training = []
        Y_training = []
        Z_training = []
        X_validation = []
        Y_validation = []
        Z_validation = []
        labels = None
        for training_data in training_data_series:
            X, y, z, headers = build_data_series(this_data[training_data[0]:training_data[1]], training_data[2], feature_hist, feature_normalization, per_team_output=team_output,
                                                 feature_maker=_make_features_random(randomness))
            X_training.extend(X)
            Y_training.extend(y)
            Z_training.extend(z)

        for training_data in validation_data_series:
            X, y, z, headers = build_data_series(this_data[training_data[0]:training_data[1]], training_data[2], feature_hist, feature_normalization, per_team_output=team_output,
                                                 feature_maker=_make_features)
            X_validation.extend(X)
            Y_validation.extend(y)
            Z_validation.extend(z)
            labels = headers

        # print("Writing Dataset " + ml_set["name"])
        return LoadingTools.get_data("auto", "todo", labels, (X_training, Y_training, Z_training), (X_validation, Y_validation, Z_validation), feature_normalization)

    @staticmethod
    def build_data_sets(ml_data_sets):
        pass

# TODO: If not win or preseason, grow it by 1.05
# WARMUP = 150
# LEAVE_OUT = 500
# VALIDATION_OFFSET = len(sorted_games) - LEAVE_OUT
#
# # Start Index, End Index, Offset
# ml_data_sets = [
#     {
#         "training_data": [(0, VALIDATION_OFFSET, WARMUP)],
#         "validation_data": [(0, len(sorted_games), VALIDATION_OFFSET)],
#         "name": "baseline_series_short_memory_simple",
#         "feature_history": [1, 1, 1, 1],
#         "team_output" : ["score", "total_yards"]
#     }, {
#         "training_data": [(0, VALIDATION_OFFSET, WARMUP)],
#         "validation_data": [(0, len(sorted_games), VALIDATION_OFFSET)],
#         "name": "baseline_series_medium_memory_simple",
#         "feature_history": [1, 2, 5, 25, 75],
#         "team_output": ["score", "total_yards"]
#     }
# ]


# Questions To Answer:
# 1) Does RE-Sampling Random Series Help? If so by what degree?
# 2) How do parameters effect things? Is more history useful? What's the threshold?
# --> in NN? in Forests/Other?
