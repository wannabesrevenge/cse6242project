import csv
import json
import os

import requests

HOST = "https://profootballapi.com"
API_KEY = "FAKE API KEY"

MIN_YEAR = 2009
MAX_YEAR = 2019

api_sess = requests.Session()

def get(path, params):
    params["api_key"] = API_KEY
    res = api_sess.post(f"{HOST}/{path}", params=params)
    res.raise_for_status()
    return res.json()

def get_schedule(year):
    return get("schedule", {"year": year})

def get_game(gid):
    return get("game", {"game_id": gid})

def get_plays(gid):
    return get("plays", {"game_id": gid})

def process_schedule(sched):
    return [game["id"] for game in sched]

def get_player_stats(gid, team):
    # The possible values for stats_type:
    #   offense, defense, special_teams, passing, rushing, receiving, kicking, punting, returning
    offense = get("players", {"stats_type": "offense", "game_id": gid, "team": team})
    defense = get("players", {"stats_type": "defense", "game_id": gid, "team": team})
    special = get("players", {"stats_type": "special_teams", "game_id": gid, "team": team})
    return {"team": team, "game": gid, "offense": offense, "defense": defense, "special_teams": special}

def process_offense_stats(stats):
    flat_stats = {}
    def _ensure_player(player):
        if player not in flat_stats:
            flat_stats[player] = {
                "name": player,
                "fumbles": 0,
                "fumbles_lost": 0,
                "pass_attempts": 0,
                "pass_completions": 0,
                "pass_yards": 0,
                "pass_touchdowns": 0,
                "pass_interceptions": 0,
                "pass_two_point_attempts": 0,
                "pass_two_point_makes": 0,
                "receptions": 0,
                "recv_yards": 0,
                "recv_touchdowns": 0,
                "recv_long": 0,
                "recv_long_touchdown": 0,
                "recv_two_point_attempts": 0,
                "recv_two_point_makes": 0,
                "rush_attempts": 0,
                "rush_yards": 0,
                "rush_touchdowns": 0,
                "rush_long": 0,
                "rush_long_touchdown": 0,
                "rush_two_point_attempts": 0,
                "rush_two_point_makes": 0
            }
    for k, v in stats.items():
        for play_id, play_stats in v.items():
            passing = play_stats.get("passing")
            rush = play_stats.get("rushing")
            recv = play_stats.get("receiving")
            fumb = play_stats.get("fumbles")

            for s in (passing, rush, recv, fumb):
                if s:
                    _ensure_player(s["name"])

            if fumb:
                flat_stats[fumb["name"]]["fumbles"] += fumb["total_fumbles"]
                flat_stats[fumb["name"]]["fumbles_lost"] += fumb["fumbles_lost"]
            if passing:
                stat = flat_stats[passing["name"]]
                stat["pass_attempts"] += passing["attempts"]
                stat["pass_completions"] += passing["completions"]
                stat["pass_yards"] += passing["yards"]
                stat["pass_touchdowns"] += passing["touchdowns"]
                stat["pass_interceptions"] += passing["interceptions"]
                stat["pass_two_point_attempts"] += passing["two_point_attempts"]
                stat["pass_two_point_makes"] += passing["two_point_makes"]
            if recv:
                stat = flat_stats[recv["name"]]
                stat["receptions"] += recv["receptions"]
                stat["recv_yards"] += recv["yards"]
                stat["recv_touchdowns"] += recv["touchdowns"]
                stat["recv_long"] += recv["long"]
                stat["recv_long_touchdown"] += recv["long_touchdown"]
                stat["recv_two_point_attempts"] += recv["two_point_attempts"]
                stat["recv_two_point_makes"] += recv["two_point_makes"]
            if rush:
                stat = flat_stats[rush["name"]]
                stat["rush_attempts"] += rush["attempts"]
                stat["rush_yards"] += rush["yards"]
                stat["rush_touchdowns"] += rush["touchdowns"]
                stat["rush_long"] += rush["long"]
                stat["rush_long_touchdown"] += rush["long_touchdown"]
                stat["rush_two_point_attempts"] += rush["two_point_attempts"]
                stat["rush_two_point_makes"] += rush["two_point_makes"]

    return [v for _, v in flat_stats.items()]
    

def process_defense_stats(stats):
    flat_stats = {}
    for k, v in stats.items():
        # Skip the key if it doesn't have a play ID associated with it
        if "defense" in v:
            continue
        
        for play_id, play_stats in v.items():
            # Skip empty defense stats
            if "defense" not in play_stats or not play_stats["defense"]:
                continue

            play_stats = play_stats["defense"]

            if play_stats["name"] not in flat_stats:
                flat_stats[play_stats["name"]] = play_stats
            else:
                stats = flat_stats[play_stats["name"]]
                stats["tackles"] += play_stats["tackles"]
                stats["assisted_tackles"] += play_stats["assisted_tackles"]
                stats["sacks"] += play_stats["sacks"]
                stats["interceptions"] += play_stats["interceptions"]
                stats["forced_fumbles"] += play_stats["forced_fumbles"]
    return [v for _, v in flat_stats.items()]


kick_keys = ["attempts", "made", "yards", "xp_attempt", "xp_made", "xp_missed", "xp_blocked", "xp_total"]
punt_keys = ["punts", "yards", "inside_20", "long"]
return_keys = ["returns", "touchdowns", "long", "long_touchdown"]
def process_special_team_stats(stats):
    punt_stats = {}
    return_stats = {}
    kick_stats = {}
    for k, v in stats.items():
        for play_id, play_stats in v.items():
            kicking = play_stats.get("kicking")
            punting = play_stats.get("punting")
            kick_r = play_stats.get("kick_returns")
            punt_r = play_stats.get("punt_returns")

            if kicking:
                if kicking["name"] not in kick_stats:
                    kicking.pop("percent")
                    kick_stats[kicking["name"]] = kicking
                else:
                    ks = kick_stats[kicking["name"]]
                    for k in kick_keys:
                        ks[k] += kicking[k]

            if punting:
                if punting["name"] not in punt_stats:
                    punting.pop("average")
                    punt_stats[punting["name"]] = punting
                else:
                    ps = prunt_stats[punting["name"]]
                    for k in punt_keys:
                        ps[k] += punting[k]

            for r in (kick_r, punt_r):
                if not r:
                    continue
                if r["name"] not in return_stats:
                    r.pop("average")
                    return_stats[r["name"]] = r
                else:
                    rs = return_stats[r["name"]]
                    for k in return_keys:
                        rs[k] += r[k]
    punt_stats = [v for _, v in punt_stats.items()]
    kick_stats = [v for _, v in kick_stats.items()]
    return_stats = [v for _, v in return_stats.items()]
    return punt_stats, kick_stats, return_stats

def add_game_meta(stats, team, game_info):
    for s in stats:
        s["team"] = team
        s["home_team"] = game_info["home"]["team"]
        s["away_team"] = game_info["away"]["team"]
        s["is_home_team"] = team == game_info["home"]["team"]
        s["game_id"] = game_info["nfl_id"]

def process_game_info(game_info):
    data = {}
    data["nfl_id"] = game_info["nfl_id"]
    data["day"] = game_info["day"]
    data["month"] = game_info["month"]
    data["time"] = game_info["time"]
    data["season_type"] = game_info["season_type"]
    data["week"] = game_info["week"]
    data["year"] = game_info["year"]
    data["final"] = game_info["final"]
    data["home_team"] = game_info["home"]["team"]
    data["home_score"] = game_info["home_score"]
    data["home_totfd"] = game_info["home"]["totfd"]
    data["home_totyds"] = game_info["home"]["totyds"]
    data["home_pyds"] = game_info["home"]["pyds"]
    data["home_ryds"] = game_info["home"]["ryds"]
    data["home_pen"] = game_info["home"]["pen"]
    data["home_penyds"] = game_info["home"]["penyds"]
    data["home_trnovr"] = game_info["home"]["trnovr"]
    data["home_pt"] = game_info["home"]["pt"]
    data["home_ptyds"] = game_info["home"]["ptyds"]
    data["home_ptavg"] = game_info["home"]["ptavg"]
    data["away_team"] = game_info["away"]["team"]
    data["away_score"] = game_info["away_score"]
    data["away_totfd"] = game_info["away"]["totfd"]
    data["away_totyds"] = game_info["away"]["totyds"]
    data["away_pyds"] = game_info["away"]["pyds"]
    data["away_ryds"] = game_info["away"]["ryds"]
    data["away_pen"] = game_info["away"]["pen"]
    data["away_penyds"] = game_info["away"]["penyds"]
    data["away_trnovr"] = game_info["away"]["trnovr"]
    data["away_pt"] = game_info["away"]["pt"]
    data["away_ptyds"] = game_info["away"]["ptyds"]
    data["away_ptavg"] = game_info["away"]["ptavg"]
    return data

def process_game_stats(game_info, home_stats, away_stats):
    home_team = home_stats["team"]
    away_team = away_stats["team"]
    off_home = process_offense_stats(home_stats["offense"])
    off_away = process_offense_stats(away_stats["offense"])
    add_game_meta(off_home, home_team, game_info)
    add_game_meta(off_away, away_team, game_info)

    def_home = process_defense_stats(home_stats["defense"])
    def_away = process_defense_stats(away_stats["defense"])
    add_game_meta(def_home, home_team, game_info)
    add_game_meta(def_away, away_team, game_info)

    punt_home, kick_home, return_home = process_special_team_stats(home_stats["special_teams"])
    punt_away, kick_away, return_away = process_special_team_stats(away_stats["special_teams"])
    add_game_meta(punt_home, home_team, game_info)
    add_game_meta(punt_away, away_team, game_info)
    add_game_meta(kick_home, home_team, game_info)
    add_game_meta(kick_away, away_team, game_info)
    add_game_meta(return_home, home_team, game_info)
    add_game_meta(return_away, away_team, game_info)

    off_stats = off_home + off_away
    def_stats = def_home + def_away
    punt_stats = punt_home + punt_away
    kick_stats = kick_home + kick_away
    return_stats = return_home + return_away

    return off_stats, def_stats, punt_stats, kick_stats, return_stats


def process_games(game_ids):
    game_stats = []
    off_stats = []
    def_stats = []
    punt_stats = []
    kick_stats = []
    return_stats = []

    for gid in game_ids:
        try:
            game_info = get_game(gid)
            plays = get_plays(gid)
            home_team = game_info["home"]["team"]
            away_team = game_info["away"]["team"]
            home_stats = get_player_stats(gid, home_team)
            away_stats = get_player_stats(gid, away_team)

            ostat, dstat, pstat, kstat, rstat = process_game_stats(game_info, home_stats, away_stats)
            flat_game_info = process_game_info(game_info)

            game_stats.append(flat_game_info)
            off_stats += ostat
            def_stats += dstat
            punt_stats += pstat
            kick_stats += kstat
            return_stats += rstat
        except requests.exceptions.HTTPError as e:
            print(e)
            print(f"HTTP error, skipping game {gid}")
            continue

    return game_stats, off_stats, def_stats, punt_stats, kick_stats, return_stats

def fill_data_by_year(year, output_dir):
    sched = get_schedule(year)
    game_ids = process_schedule(sched)
    game_stats, off_stats, def_stats, punt_stats, kick_stats, return_stats = process_games(game_ids)

    write_data_to_file(os.path.join(output_dir, "game_stats.csv"), game_stats)
    write_data_to_file(os.path.join(output_dir, "offense_stats.csv"), off_stats)
    write_data_to_file(os.path.join(output_dir, "defense_stats.csv"), def_stats)
    write_data_to_file(os.path.join(output_dir, "punt_stats.csv"), punt_stats)
    write_data_to_file(os.path.join(output_dir, "kick_stats.csv"), kick_stats)
    write_data_to_file(os.path.join(output_dir, "return_stats.csv"), return_stats)


def write_data_to_file(fname, data):
    with open(fname, 'a+', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(data[0].keys()))

        # Only write the header if we are at the start of the file
        if csvfile.tell() == 0:
            writer.writeheader()

        for d in data:
            writer.writerow(d)


OUTDIR = 'scraped_data'
for year in range(2009, 2020):
    print(f"Filling year {year}")
    fill_data_by_year(year, OUTDIR)
