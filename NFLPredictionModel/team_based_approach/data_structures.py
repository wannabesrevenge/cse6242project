from typing import Dict, List
import numpy as np


# THEME:
# Scoring 10 points on a team with a bad defense is less impressive than scoring 10 points on a team with strong defense
# Solution: Track the relative values between the teams as well
# Model Assumes Actual Team Doesn't Matter, only the underlying statistics do. Could play with that concept....

class TeamData:
    def __init__(self, name: str):
        """

        :param team_id:
        :param name: Name of team
        :param list_history_stores: histories & delta's to store on team based attributes
        e.g. [1, 5, 10, 25] -> last game features/delta, aggregate last [2 through 6] games, aggregate last [7 - 17] games
        """
        self.game_dates = {}
        # self.team_id = team_id
        self.name = name
        self.game_attributes = {}  # GID -> Tuple(Attributes, Attribute Deltas)

        # Some Helpful Optimizations For Better Look Up Times
        self.game_to_gid = {}  # Index to GID
        self.gid_to_index = {}  # GID to Index
        self.sorted = False

    def add_game(self,
                 gid: int, game_datetime: float,
                 team_attributes: Dict[str, float], opposing_attributes: Dict[str, float]):
        delta_attr = {}
        for k, v in team_attributes.items():
            delta_attr[k] = v - opposing_attributes[k]

        self.game_attributes[gid] = (team_attributes, delta_attr)
        self.game_dates[game_datetime] = gid
        self.sorted = False

    def build_result(self, gid: int, data: Dict, feature_normalization: Dict[str, float]):
        """

        :param gid:
        :param data: Existing dictionary to add to
        :param feature_normalization:
        :return:
        """
        for f, v in feature_normalization.items():
            data[f] = self.game_attributes[gid][f] / v

    def build_feature(self, gid: int, history_length: int, skip_length: int, feature_normalization: Dict[str, float]):
        """
        For Each Feature Stored.
            Aggregate history length
            (average value)
            (average "delta")
            (std "delta")
            (std "value")


        :param feature_normalization:
        :param gid:
        :param history_length:
        :param skip_length:
        :return: Dictionary of each feature in feature_normalized. If there is no game data the values are 0.0
        """
        if not self.sorted:
            self.game_to_gid = {}  # Index to GID
            self.gid_to_index = {}  # GID to Index
            cnt = 0
            for k in sorted(self.game_dates.keys()):  # k=> game time
                self.gid_to_index[self.game_dates[k]] = cnt
                self.game_to_gid[cnt] = self.game_dates[k]
                cnt += 1
            self.sorted = True

        if gid not in self.gid_to_index:
            raise Exception("Game Doesn't Exist In Records")

        initial_index = self.gid_to_index[gid] - skip_length
        features = []
        for g in range(history_length):
            index = initial_index - g
            if index in self.game_to_gid:
                features.append(self.game_attributes[self.game_to_gid[index]])

        tmp_deltas = []
        tmp_values = []

        out_data = {}

        for k, v in feature_normalization.items():
            tmp_deltas.clear()
            tmp_values.clear()

            if len(features) > 0:

                for f in features:
                    tmp_values.append(f[0][k])
                    tmp_deltas.append(f[1][k])

                out_data[k] = np.mean(tmp_values) / feature_normalization[k]
                out_data[k + "_delta"] = np.mean(tmp_deltas) / feature_normalization[k]

                # Knowledge Compression: TODO experiment with effectiveness
                # if history_length > 1:
                out_data[k + "_std"] = np.std(tmp_values) / feature_normalization[k]
                out_data[k + "_delta_std"] = np.std(tmp_deltas) / feature_normalization[k]
            else:
                out_data[k] = 0.0
                out_data[k + "_delta"] = 0.0

                # Knowledge Compression: TODO experiment with effectiveness

                # if history_length > 1:
                out_data[k + "_delta_std"] = 0.0
                out_data[k + "_std"] = 0.0

        return out_data


SEASON_VALUE = {"PRE": 0.0, "REG": 0.5, "POST": 1.0}


def build_data_series(game_data: List, offset: int,
                      history_parameters, feature_normalization,
                      per_team_output: List[str], feature_maker):
    teams: Dict[str, TeamData] = {}
    last_played: Dict[str, TeamData] = {}

    for x in range(len(game_data)):
        # Build Team Data
        r = game_data[x]
        away = r.away_team
        home = r.home_team

        if away not in teams:
            teams[away] = TeamData(away)
            last_played[away] = r.time

        if r.home_team not in teams:
            teams[home] = TeamData(home)
            last_played[home] = r.time

        home_features = feature_maker(r, "home_", time_since_last=r.time - last_played[home], home=True)
        away_features = feature_maker(r, "away_", time_since_last=r.time - last_played[away], home=False)
        teams[home].add_game(r.nfl_id, r.time, home_features, away_features)
        teams[away].add_game(r.nfl_id, r.time, away_features, home_features)

        # Set Last Played
        last_played[home] = r.time
        last_played[away] = r.time

    X = []
    y = []
    z = []  # GID, HOME, AWAY (utility for reporting)
    ordered_keys = list(sorted(list(teams[game_data[0].home_team].build_feature(game_data[0].nfl_id, history_length=1, skip_length=0, feature_normalization=feature_normalization).keys())))

    for x in range(offset, len(game_data)):
        r = game_data[x]
        away = r.away_team
        home = r.home_team
        row_dt = []

        hd = []
        ad = []
        skip = 1
        for hlen in history_parameters:
            hd.append(teams[home].build_feature(r.nfl_id, history_length=hlen, skip_length=skip, feature_normalization=feature_normalization))
            ad.append(teams[away].build_feature(r.nfl_id, history_length=hlen, skip_length=skip, feature_normalization=feature_normalization))
            skip += hlen

        if isinstance(r.season_type, str):
            row_dt.append(SEASON_VALUE[r.season_type])
        else:
            row_dt.append(0.0)

        for q in [hd, ad]:
            for h in q:
                for ok in ordered_keys:
                    row_dt.append(h[ok])

        home_values = teams[home].build_feature(r.nfl_id, history_length=1, skip_length=0, feature_normalization=feature_normalization)
        away_values = teams[away].build_feature(r.nfl_id, history_length=1, skip_length=0, feature_normalization=feature_normalization)
        X.append(row_dt)
        y.append([1.0 if r.home_score > r.away_score else 0.0] + [home_values[f] for f in per_team_output] + [away_values[f] for f in per_team_output])
        z.append([r.nfl_id, r.home_team, r.away_team])

    # Feature Headers
    header_x = ["preseason"]
    for q in ["home_", "away_"]:
        sk = 1
        for px in history_parameters:
            for ok in ordered_keys:
                header_x.append(q + ok + "_" + str(sk) + "_" + str(sk + px - 1))
            sk += px

    headers = [
        header_x, ["home_wins"] + ["home_" + x for x in per_team_output] + ["away_" + x for x in per_team_output]
    ]

    return X, y, z, headers
