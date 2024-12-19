import gc
import math
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import sklearn.model_selection
import streamlit as st
import tqdm
import xmltodict

import importlib

from accessible_space.utility import get_unused_column_name
from accessible_space.interface import per_object_frameify_tracking_data, get_expected_pass_completion
from accessible_space.core import PARAMETER_BOUNDS

importlib.reload(sys.modules["accessible_space.utility"])
importlib.reload(sys.modules["accessible_space.interface"])
importlib.reload(sys.modules["accessible_space.core"])

cache_dir = os.path.join(os.path.dirname(__file__), ".joblib-cache")
memory = joblib.Memory(cache_dir, verbose=0)

metrica_open_data_base_dir = "https://raw.githubusercontent.com/metrica-sports/sample-data/refs/heads/master/data"


@memory.cache
def get_metrica_tracking_data(dataset_nr):
    # TODO remove kloppy dependency
    home_data_url = f"{metrica_open_data_base_dir}/Sample_Game_{dataset_nr}/Sample_Game_{dataset_nr}_RawTrackingData_Home_Team.csv"
    away_data_url = f"{metrica_open_data_base_dir}/Sample_Game_{dataset_nr}/Sample_Game_{dataset_nr}_RawTrackingData_Away_Team.csv"
    df_tracking_home = pd.read_csv(home_data_url)
    df_tracking_away = pd.read_csv(away_data_url)
    df_tracking_home["team_id"] = "Home"
    df_tracking_away["team_id"] = "Away"
    df_tracking = pd.concat([df_tracking_home, df_tracking_away])

    # dataset = kloppy.metrica.load_open_data(dataset_nr)  # , limit=100)
    # df_tracking = dataset.to_df()
    return df_tracking


@st.cache_resource
def get_kloppy_events(dataset_nr):
    if dataset_nr in [1, 2]:
        # df = pd.read_csv(f"C:/Users/Jonas/Desktop/ucloud/Arbeit/Spielanalyse/soccer-analytics/football1234/datasets/metrica/sample-data-master/data/Sample_Game_{dataset_nr}/Sample_Game_{dataset_nr}_RawEventsData.csv")
        df = pd.read_csv(f"{metrica_open_data_base_dir}/Sample_Game_{dataset_nr}/Sample_Game_{dataset_nr}_RawEventsData.csv")
        df["body_part_type"] = df["Subtype"].where(df["Subtype"].isin(["HEAD"]), None)
        df["set_piece_type"] = df["Subtype"].where(
            df["Subtype"].isin(["THROW IN", "GOAL KICK", "FREE KICK", "CORNER KICK"]), None).map(
            lambda x: x.replace(" ", "_") if x is not None else None
        )
        df["Type"] = df["Type"].str.replace(" ", "_")
        df["Start X"] = (df["Start X"] - 0.5) * 105
        df["Start Y"] = -(df["Start Y"] - 0.5) * 68
        df["End X"] = (df["End X"] - 0.5) * 105
        df["End Y"] = -(df["End Y"] - 0.5) * 68
        df = df.rename(columns={
            "Type": "event_type",
            "Period": "period_id",
            "Team": "team_id",
            "From": "player_id",
            "To": "receiver_player_id",
            "Start X": "coordinates_x",
            "Start Y": "coordinates_y",
            "End X": "end_coordinates_x",
            "End Y": "end_coordinates_y",
            "Start Frame": "frame_id",
            "End Frame": "end_frame_id",
        })
        player_id_to_column_id = {}
        column_id_to_team_id = {}
        for team_id in df["team_id"].unique():
            df_players = df[df["team_id"] == team_id]
            team_player_ids = set(
                df_players["player_id"].dropna().tolist() + df_players["receiver_player_id"].dropna().tolist())
            player_id_to_column_id.update(
                {player_id: f"{team_id.lower().strip()}_{player_id.replace('Player', '').strip()}" for player_id in
                 team_player_ids})
            column_id_to_team_id.update({player_id_to_column_id[player_id]: team_id for player_id in team_player_ids})

        df["player_id"] = df["player_id"].map(player_id_to_column_id)
        df["receiver_player_id"] = df["receiver_player_id"].map(player_id_to_column_id)
        df["receiver_team_id"] = df["receiver_player_id"].map(column_id_to_team_id)

        df["tmp_next_player"] = df["player_id"].shift(-1)
        df["tmp_next_team"] = df["team_id"].shift(-1)
        df["tmp_receiver_player"] = df["receiver_player_id"].where(df["receiver_player_id"].notna(), df["tmp_next_player"])
        df["tmp_receiver_team"] = df["tmp_receiver_player"].map(column_id_to_team_id)

        df["success"] = df["tmp_receiver_team"] == df["team_id"]

        df["is_pass"] = (df["event_type"].isin(["PASS", "BALL_LOST", "BALL_OUT"])) \
                        & (~df["Subtype"].isin(["CLEARANCE", "HEAD-CLEARANCE", "HEAD-INTERCEPTION-CLEARANCE"])) \
                        & (df["frame_id"] != df["end_frame_id"])

        df["is_high"] = df["Subtype"].isin([
            "CROSS",
            # "CLEARANCE",
            "CROSS-INTERCEPTION",
            # "HEAD-CLEARANCE",
            # "HEAD-INTERCEPTION-CLEARANCE"
        ])

        #     df_passes["xc"], _, _ = dangerous_accessible_space.get_expected_pass_completion(
        #         df_passes, df_tracking, event_frame_col="td_frame", tracking_frame_col="frame", event_start_x_col="start_x",
        #         event_start_y_col="start_y", event_end_x_col="end_x", event_end_y_col="end_y",
        #         event_player_col="tracking_player_id",
        #     )

        return df.drop(columns=["tmp_next_player", "tmp_next_team", "tmp_receiver_player", "tmp_receiver_team"])
    else:
        # dataset = kloppy.metrica.load_event(
        #     event_data="C:/Users/Jonas/Desktop/ucloud/Arbeit/Spielanalyse/soccer-analytics/football1234/datasets/metrica/sample-data-master/data/Sample_Game_3/Sample_Game_3_events.json",
        #     # meta_data="https://raw.githubusercontent.com/metrica-sports/sample-data/refs/heads/master/data/Sample_Game_3/Sample_Game_3_metadata.xml",
        #     meta_data="C:/Users/Jonas/Desktop/ucloud/Arbeit/Spielanalyse/soccer-analytics/football1234/datasets/metrica/sample-data-master/data/Sample_Game_3/Sample_Game_3_metadata.xml",
        #     coordinates="secondspectrum",
        # )
        # json_data = json.load(open("C:/Users/Jonas/Desktop/ucloud/Arbeit/Spielanalyse/soccer-analytics/football1234/datasets/metrica/sample-data-master/data/Sample_Game_3/Sample_Game_3_events.json"))
        # json_data = json.loads(open(f"{metrica_open_data_base_dir}/Sample_Game_3/Sample_Game_3_events.json"))
        json_data = requests.get(f"{metrica_open_data_base_dir}/Sample_Game_3/Sample_Game_3_events.json").json()

        df = pd.json_normalize(json_data["data"])

        expanded_df = pd.DataFrame(df['subtypes'].apply(pd.Series))
        expanded_df.columns = [f'subtypes.{col}' for col in expanded_df.columns]

        new_dfs = []
        for expanded_col in expanded_df.columns:
            expanded_df2 = pd.json_normalize(expanded_df[expanded_col])
            expanded_df2.columns = [f'{expanded_col}.{col}' for col in expanded_df2.columns]
            new_dfs.append(expanded_df2)

        expanded_df = pd.concat(new_dfs, axis=1)

        df = pd.concat([df, expanded_df], axis=1)

        i_subtypes_nan = ~df["subtypes.name"].isna()
        i_subtypes_0_nan = ~df["subtypes.0.name"].isna()

        # check if the true's are mutually exclusive
        assert not (i_subtypes_nan & i_subtypes_0_nan).any()

        df.loc[i_subtypes_nan, "subtypes.0.name"] = df.loc[i_subtypes_nan, "subtypes.name"]
        df.loc[i_subtypes_nan, "subtypes.0.id"] = df.loc[i_subtypes_nan, "subtypes.id"]
        df = df.drop(columns=["subtypes.name", "subtypes.id", "subtypes"])
        subtype_cols = [col for col in df.columns if col.startswith("subtypes.") and col.endswith("name")]

        player2team = df[['from.id', 'team.id']].set_index('from.id')['team.id'].to_dict()
        df["receiver_team_id"] = df["to.id"].map(player2team)
        # df["tmp_next_player"] = df["player_id"].shift(-1)
        # df["tmp_next_team"] = df["team_id"].shift(-1)
        # df["tmp_receiver_player"] = df["receiver_player_id"].where(df["receiver_player_id"].notna(), df["tmp_next_player"])
        # df["tmp_receiver_team"] = df["tmp_receiver_player"].map(column_id_to_team_id)
        df["success"] = df["receiver_team_id"] == df["team.id"]

        df["success"] = df["success"].astype(bool)

        df["is_pass"] = (df["type.name"].isin(["PASS", "BALL LOST", "BALL OUT"])) \
                        & ~df[subtype_cols].isin(["CLEARANCE"]).any(axis=1) \
                        & (df["start.frame"] != df["end.frame"])

        # df[df[['Name', 'Age']].isin(['Alice', 30]).any(axis=1)]
        df["is_high"] = df[subtype_cols].isin(["CROSS"]).any(axis=1)

        df = df.rename(columns={
            "type.name": "event_type",
            "from.id": "player_id",
            "team.id": "team_id",
            "to.id": "receiver_player_id",
            "period": "period_id",
            "start.frame": "frame_id",
            "end.frame": "end_frame_id",
            "start.x": "coordinates_x",
            "start.y": "coordinates_y",
            "end.x": "end_coordinates_x",
            "end.y": "end_coordinates_y",
        }).drop(columns=[
            "to",
        ])
        df["coordinates_x"] = (df["coordinates_x"] - 0.5) * 105
        df["coordinates_y"] = (df["coordinates_y"] - 0.5) * 68
        df["end_coordinates_x"] = (df["end_coordinates_x"] - 0.5) * 105
        df["end_coordinates_y"] = (df["end_coordinates_y"] - 0.5) * 68

        meta_data = xmltodict.parse(requests.get(f"{metrica_open_data_base_dir}/Sample_Game_3/Sample_Game_3_metadata.xml").text)

        df_player = pd.json_normalize(meta_data, record_path=["main", "Metadata", "Players", "Player"])
        player2team = df_player[["@id", "@teamId"]].set_index("@id")["@teamId"].to_dict()
        df["team_id"] = df["player_id"].map(player2team)

        return df


@st.cache_resource
def get_metrica_data():
    datasets = []
    dfs_event = []
    st.write(" ")
    st.write(" ")
    progress_bar_text = st.empty()
    progress_bar = st.progress(0)
    for dataset_nr in [1, 2, 3]:
    # for dataset_nr in [3]:
        progress_bar_text.text(f"Loading dataset {dataset_nr}")
        # dataset = kloppy.metrica.load_tracking_csv(
        #     home_data=f"https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_{dataset_nr}/Sample_Game_{dataset_nr}_RawTrackingData_Home_Team.csv",
        #     away_data=f"https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_{dataset_nr}/Sample_Game_{dataset_nr}_RawTrackingData_Away_Team.csv",
        #     # sample_rate=1 / 5,
        #     # limit=100,
        #     coordinates="secondspectrum"
        # )
        # df_events1 = pd.read_csv(f"https://raw.githubusercontent.com/metrica-sports/sample-data/refs/heads/master/data/Sample_Game_{dataset_nr}/Sample_Game_{dataset_nr}_RawEventsData.csv")
        # df_passes1 = df_events1[df_events1["Type"] == "PASS"]

        with st.spinner(f"Downloading events from dataset {dataset_nr}"):
            df_events = get_kloppy_events(dataset_nr).copy()
        event_frames = df_events["frame_id"].unique()

        delta_frames_to_load = 5

        frames_to_load = [set(range(event_frame, event_frame + delta_frames_to_load)) for event_frame in event_frames]
        frames_to_load = sorted(list(set([frame for frames in frames_to_load for frame in frames])))

        with st.spinner(f"Downloading tracking data from dataset {dataset_nr}"):
            df_tracking = get_metrica_tracking_data(dataset_nr)

        df_tracking = df_tracking[df_tracking["frame_id"].isin(frames_to_load)]

        df_tracking[[col for col in df_tracking.columns if col.endswith("_x")]] = (df_tracking[[col for col in df_tracking.columns if col.endswith("_x")]].astype(float) - 0.5) * 105
        df_tracking[[col for col in df_tracking.columns if col.endswith("_y")]] = (df_tracking[[col for col in df_tracking.columns if col.endswith("_y")]].astype(float) - 0.5) * 68

        df_tracking = df_tracking.drop(columns=[col for col in df_tracking.columns if col.endswith("_d") or col.endswith("_s")])

        players = [col.replace("_x", "") for col in df_tracking.columns if col.endswith("_x")]
        x_cols = [f"{player}_x" for player in players]
        y_cols = [f"{player}_y" for player in players]
        vx_cols = [f"{player}_vx" for player in players]
        vy_cols = [f"{player}_vy" for player in players]
        v_cols = [f"{player}_velocity" for player in players]
        frame_col = "frame_id"

        # dt = df_tracking["timestamp"].diff().mean()

        # df_tracking["ball_vx"] = df_tracking["ball_x"].diff() / df_tracking["timestamp"].dt.total_seconds().diff()
        # df_tracking["ball_vy"] = df_tracking["ball_y"].diff() / df_tracking["timestamp"].dt.total_seconds().diff()
        # df_tracking["ball_velocity"] = np.sqrt(df_tracking["ball_vx"]**2 + df_tracking["ball_vy"]**2)
        for player in players:
            df_tracking[f"{player}_x"] = df_tracking[f"{player}_x"].astype(float)
            xdiff = df_tracking[f"{player}_x"].diff().fillna(method="bfill")
            xdiff2 = -df_tracking[f"{player}_x"].diff(periods=-1).fillna(method="ffill")
            tdiff = df_tracking["timestamp"].diff().dt.total_seconds().fillna(method="bfill")
            tdiff2 = -df_tracking["timestamp"].diff(periods=-1).dt.total_seconds().fillna(method="ffill")
            vx = (xdiff + xdiff2) / (tdiff + tdiff2)
            df_tracking[f"{player}_vx"] = vx

            df_tracking[f"{player}_y"] = df_tracking[f"{player}_y"].astype(float)
            ydiff = df_tracking[f"{player}_y"].diff().fillna(method="bfill")
            ydiff2 = -df_tracking[f"{player}_y"].diff(periods=-1).fillna(method="ffill")
            vy = (ydiff + ydiff2)  # / (tdiff + tdiff2)
            df_tracking[f"{player}_vy"] = vy
            df_tracking[f"{player}_velocity"] = np.sqrt(vx ** 2 + vy ** 2)

            i_nan_x = df_tracking[f"{player}_x"].isna()
            df_tracking.loc[i_nan_x, f"{player}_vx"] = np.nan
            i_nan_y = df_tracking[f"{player}_y"].isna()
            df_tracking.loc[i_nan_y, f"{player}_vy"] = np.nan
            df_tracking.loc[i_nan_x | i_nan_y, f"{player}_velocity"] = np.nan

        player_to_team = {}
        if dataset_nr in [1, 2]:
            for player in players:
                if "home" in player:
                    player_to_team[player] = "Home"
                elif "away" in player:
                    player_to_team[player] = "Away"
                else:
                    player_to_team[player] = None
        else:
            player_to_team = df_events[['player_id', 'team_id']].set_index('player_id')['team_id'].to_dict()

        df_tracking_obj = per_object_frameify_tracking_data(
            df_tracking, frame_col,
            coordinate_cols=[[x_cols[i], y_cols[i], vx_cols[i], vy_cols[i], v_cols[i]] for i, _ in enumerate(players)],
            players=players, player_to_team=player_to_team,
            new_coordinate_cols=["x", "y", "vx", "vy", "v"],
            new_team_col="team_id", new_player_col="player_id",
        )

        # get ball control
        fr2control = df_events.set_index("frame_id")["team_id"].to_dict()
        df_tracking_obj["ball_possession"] = df_tracking_obj["frame_id"].map(fr2control)
        df_tracking_obj = df_tracking_obj.sort_values("frame_id")
        df_tracking_obj["ball_possession"] = df_tracking_obj["ball_possession"].ffill()

        datasets.append(df_tracking_obj)

        dfs_event.append(df_events)

        progress_bar.progress(dataset_nr / 3)

    return datasets, dfs_event


def check_synthetic_pass(p4ss, df_tracking_frame_attacking, v_receiver, v_receiver_threshold=4, v_players=10, pass_duration_threshold=0.5, pass_length_threshold=15):
    """ Checks whether a synthetic pass is guaranteed to be unsuccessful according to the criteria of our validation """

    p4ss["angle"] = math.atan2(p4ss["end_coordinates_y"] - p4ss["coordinates_y"], p4ss["end_coordinates_x"] - p4ss["coordinates_x"])

    if v_receiver > v_receiver_threshold:
        return False  # Criterion 1: Receiver is not too fast

    v0_pass = p4ss["v0"]
    v0x_pass = v0_pass * math.cos(p4ss["angle"])
    v0y_pass = v0_pass * math.sin(p4ss["angle"])
    x0_pass = p4ss["coordinates_x"]
    y0_pass = p4ss["coordinates_y"]

    pass_length = math.sqrt((p4ss["coordinates_x"] - p4ss["end_coordinates_x"]) ** 2 + (p4ss["coordinates_y"] - p4ss["end_coordinates_y"]) ** 2)
    pass_duration = pass_length / v0_pass
    if pass_duration < pass_duration_threshold or pass_length < pass_length_threshold:
        return False  # Criterion 2: Pass is not too short

    df_tracking_frame_attacking = df_tracking_frame_attacking[(df_tracking_frame_attacking["team_id"] == p4ss["team_id"])]
    for _, row in df_tracking_frame_attacking.iterrows():
        x_player = row["x"]
        y_player = row["y"]

        distance_to_target = math.sqrt((x_player - p4ss["end_coordinates_x"]) ** 2 + (y_player - p4ss["end_coordinates_y"]) ** 2)
        necessary_speed_to_reach_target = distance_to_target / pass_duration

        def can_intercept(x0b, y0b, vxb, vyb, x_A, y_A, v_A, duration):
            # Constants
            C = (x0b - x_A) ** 2 + (y0b - y_A) ** 2
            B = 2 * ((x0b - x_A) * vxb + (y0b - y_A) * vyb)
            A = v_A ** 2 - (vxb ** 2 + vyb ** 2)

            if A <= 0:
                # If A is non-positive, agent A cannot intercept object B
                return False

            # Calculate the discriminant of the quadratic equation
            discriminant = B ** 2 + 4 * A * C

            # Check if the discriminant is non-negative and if there are real, positive roots
            if discriminant >= 0:
                # Roots of the quadratic equation
                sqrt_discriminant = math.sqrt(discriminant)
                t1 = (B - sqrt_discriminant) / (2 * A)
                t2 = (B + sqrt_discriminant) / (2 * A)

                # Check if any of the roots are non-negative
                if t1 >= 0 or t2 >= 0 and t1 < duration and t2 < duration:
                    return True

            return False

        if necessary_speed_to_reach_target < v_players or can_intercept(x0_pass, y0_pass, v0x_pass, v0y_pass, x_player, y_player, v_players, pass_duration):
            return False  # Criterion 3: Pass cannot be received by any teammate

    return True


def add_synthetic_passes(
    df_passes, df_tracking, n_synthetic_passes=5, event_frame_col="frame_id", tracking_frame_col="frame_id",
    event_team_col="team_id", tracking_team_col="team_id", event_player_col="player_id",
    tracking_player_col="player_id", x_col="x", y_col="y",
    new_is_synthetic_col="is_synthetic"
):
    st.write("add_synthetic_passes...")
    df_passes[new_is_synthetic_col] = False
    synthetic_passes = []

    teams = df_tracking[tracking_team_col].unique()

    for _, p4ss in df_passes.sample(frac=1).iterrows():
        # for attacking_team in df_tracking[tracking_team_col].unique():
        for attacking_team in teams:
            df_frame_players = df_tracking[
                (df_tracking[event_frame_col] == p4ss[event_frame_col]) &
                (df_tracking[x_col].notna()) &
                (df_tracking[event_team_col].notna())  # ball
            ]
            df_frames_defenders = df_frame_players[df_frame_players[tracking_team_col] != attacking_team]
            df_frame_attackers = df_frame_players[df_frame_players[tracking_team_col] == attacking_team]

            for _, attacker_frame in df_frame_attackers.iterrows():
                for _, defender_frame in df_frames_defenders.iterrows():
                    for v0 in [10]:  # [5, 10, 15, 20]:
                        synthetic_pass = {
                            "frame_id": p4ss[event_frame_col],
                            "coordinates_x": attacker_frame[x_col],
                            "coordinates_y": attacker_frame[y_col],
                            "end_coordinates_x": defender_frame[x_col],
                            "end_coordinates_y": defender_frame[y_col],
                            "event_type": None,
                            "Subtype": None,
                            "period": None,
                            "end_frame_id": None,
                            "v0": v0,
                            "player_id": attacker_frame[tracking_player_col],
                            "team_id": attacker_frame[tracking_team_col],
                            "success": False,
                            new_is_synthetic_col: True,
                        }
                        # assert p4ss[event_team_col] == attacker_frame[tracking_team_col]
                        # i += 1
                        # if i > 15:
                        #     st.stop()

                        if check_synthetic_pass(synthetic_pass, df_frame_players, v_receiver=defender_frame["v"]):
                            synthetic_passes.append(synthetic_pass)
                            if len(synthetic_passes) >= n_synthetic_passes:
                                break
                    if len(synthetic_passes) >= n_synthetic_passes:
                        break
                if len(synthetic_passes) >= n_synthetic_passes:
                    break
            if len(synthetic_passes) >= n_synthetic_passes:
                break
        if len(synthetic_passes) >= n_synthetic_passes:
            break

    df_synthetic_passes = pd.DataFrame(synthetic_passes)

    assert len(
        df_synthetic_passes) == n_synthetic_passes, f"len(df_synthetic_passes)={len(df_synthetic_passes)} != n_synthetic_passes={n_synthetic_passes}, (len(synthetic_passes)={len(synthetic_passes)}"

    return pd.concat([df_passes, df_synthetic_passes], axis=0)


def get_scores(_df, baseline_accuracy, outcome_col="success"):
    df = _df.copy()

    data = {}

    # Descriptives
    data["average_accuracy"] = df[outcome_col].mean()
    data["synthetic_share"] = df["is_synthetic"].mean()

    # Baselines
    data["baseline_brier"] = sklearn.metrics.brier_score_loss(df[outcome_col], [baseline_accuracy] * len(df))
    try:
        data["baseline_logloss"] = sklearn.metrics.log_loss(df[outcome_col], [baseline_accuracy] * len(df))
    except ValueError:
        data["baseline_logloss"] = np.nan
    try:
        data["baseline_auc"] = sklearn.metrics.roc_auc_score(df[outcome_col], [baseline_accuracy] * len(df))
    except ValueError:
        data["baseline_auc"] = np.nan

    if "xc" in df.columns:
        data["avg_xc"] = df["xc"].mean()
        # Model scores
        data["brier_score"] = (df[outcome_col] - df["xc"]).pow(2).mean()

        # data["brier_score"] = sklearn.metrics.brier_score_loss(df[outcome_col], df["xc"])

        try:
            data["logloss"] = sklearn.metrics.log_loss(df[outcome_col], df["xc"])
        except ValueError:
            data["logloss"] = np.nan
        try:
            data["auc"] = sklearn.metrics.roc_auc_score(df[outcome_col], df["xc"])
        except ValueError:
            data["auc"] = np.nan
    else:
        data["brier_score"] = np.nan
        data["logloss"] = np.nan
        data["auc"] = np.nan

    # Model scores by syntheticness
    for is_synthetic in [False, True]:
        synth_str = "synthetic" if is_synthetic else "real"
        df_synth = df[df["is_synthetic"] == is_synthetic]

        if "xc" in df.columns:
            try:
                data[f"brier_score_{synth_str}"] = sklearn.metrics.brier_score_loss(df_synth[outcome_col], df_synth["xc"])
            except ValueError:
                data[f"brier_score_{synth_str}"] = np.nan
            try:
                data[f"logloss_{synth_str}"] = sklearn.metrics.log_loss(df_synth[outcome_col], df_synth["xc"])
            except ValueError:
                data[f"logloss_{synth_str}"] = np.nan
            try:
                data[f"auc_{synth_str}"] = sklearn.metrics.roc_auc_score(df_synth[outcome_col], df_synth["xc"])
            except ValueError:
                data[f"auc_{synth_str}"] = np.nan

        data[f"average_accuracy_{synth_str}"] = df_synth[outcome_col].mean()
        data[f"synthetic_share_{synth_str}"] = df_synth["is_synthetic"].mean()
        try:
            data[f"baseline_brier_{synth_str}"] = sklearn.metrics.brier_score_loss(df_synth[outcome_col], [baseline_accuracy] * len(df_synth))
        except ValueError:
            data[f"baseline_brier_{synth_str}"] = np.nan
        try:
            data[f"baseline_loglos_{synth_str}"] = sklearn.metrics.log_loss(df_synth[outcome_col], [baseline_accuracy] * len(df_synth))
        except ValueError:
            data[f"baseline_loglos_{synth_str}"] = np.nan
        try:
            data[f"baseline_auc_{synth_str}"] = sklearn.metrics.roc_auc_score(df_synth[outcome_col], [baseline_accuracy] * len(df_synth))
        except ValueError:
            data[f"baseline_auc_{synth_str}"] = np.nan

    return data


def bin_nr_calibration_plot(df, prediction_col="xc", outcome_col="success", n_bins=None, binsize=None, add_text=True):
    bin_col = get_unused_column_name(df.columns, "bin")
    if binsize is None and n_bins is not None:
        df[bin_col] = pd.qcut(df[prediction_col], n_bins, labels=False, duplicates="drop")
    elif binsize is not None and n_bins is None:
        min_val = df[prediction_col].min()
        max_val = df[prediction_col].max()
        bin_edges = [min_val + i * binsize for i in range(int((max_val - min_val) / binsize) + 2)]
        df[bin_col] = pd.cut(df[prediction_col], bins=bin_edges, labels=False, include_lowest=True)
    else:
        raise ValueError("Either n_bins or binsize must be specified")

    df_calibration = df.groupby(bin_col).agg({outcome_col: "mean", prediction_col: "mean"}).reset_index()
    df_calibration[bin_col] = df_calibration[bin_col]
    fig, ax = plt.subplots()
    ax.plot(df_calibration[prediction_col], df_calibration[outcome_col], marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black")

    # Annotate each point with the number of samples
    if add_text:
        for i, row in df_calibration.iterrows():
            count = len(df[df[bin_col] == row[bin_col]])  # Count of samples in the bin
            ax.annotate(
                f"n={count}",
                (row[prediction_col], row[outcome_col] - 0.03),  # Position of the text
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='lightblue'),
                fontsize=7, ha='center', va='center'
            )

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed probability")
    ax.set_title("Calibration plot")
    return fig


def plot_pass(p4ss, df_tracking, pass_x_col):
    plt.figure()
    plt.arrow(x=p4ss["coordinates_x"], y=p4ss["coordinates_y"], dx=p4ss["end_coordinates_x"] - p4ss["coordinates_x"],
              dy=p4ss["end_coordinates_y"] - p4ss["coordinates_y"], head_width=1, head_length=1, fc="red", ec="red")
    df_frame = df_tracking[df_tracking["frame_id"] == p4ss["frame_id"]]
    for team in df_frame["team_id"].unique():
        df_frame_team = df_frame[df_frame["team_id"] == team]
        x = df_frame_team["x"].tolist()
        y = df_frame_team["y"].tolist()
        plt.scatter(x, y, c="red" if team == p4ss["team_id"] else "blue")

        vx = df_frame_team["vx"].tolist()
        vy = df_frame_team["vy"].tolist()
        for i in range(len(x)):
            plt.arrow(x=x[i], y=y[i], dx=vx[i] / 5, dy=vy[i] / 5, head_width=0.5, head_length=0.5, fc="black", ec="black")

    # plot passing start point with colored X
    plt.scatter(p4ss["coordinates_x"], p4ss["coordinates_y"], c="red", marker="x", s=100)

    # plot ball position
    df_frame_ball = df_frame[df_frame["player_id"] == "ball"]
    x_ball = df_frame_ball["x"].iloc[0]
    y_ball = df_frame_ball["y"].iloc[0]
    plt.scatter(x_ball, y_ball, c="black", marker="x", s=100)

    plt.plot([-52.5, 52.5], [-34, -34], c="black")
    plt.plot([-52.5, 52.5], [34, 34], c="black")
    plt.plot([-52.5, -52.5], [-34, 34], c="black")
    plt.plot([52.5, 52.5], [-34, 34], c="black")
    plt.title(f"Pass: {p4ss['success']}")
    st.write(plt.gcf())


def validate_multiple_matches(
    dfs_tracking, dfs_passes, n_steps=100, training_size=0.7, use_prefit=True,
    outcome_col="success", tracking_team_col="team_id", event_team_col="team_id",
):
    random_state = 1893

    plot_synthetic_passes = st.button("Plot synthetic passes (unused)")
    exclude_synthetic_passes_from_training_set = st.checkbox("Exclude synthetic passes from training set", value=False)
    exclude_synthetic_passes_from_test_set = st.checkbox("Exclude synthetic passes from test set", value=False)
    chunk_size = st.number_input("Chunk size", value=200, min_value=1, max_value=None)

    ## Add synthetic passes
    @st.cache_resource
    def _get_dfs_passes_with_synthetic():
        dfs_passes_with_synthetic = []
        for df_tracking, df_passes in zip(dfs_tracking, dfs_passes):
            # n_synthetic_passes = 5
            n_synthetic_passes = len(df_passes[df_passes["success"]]) - len(df_passes[~df_passes["success"]])
            st.write("n_synthetic_passes", n_synthetic_passes)

            df_passes = add_synthetic_passes(df_passes, df_tracking, n_synthetic_passes=n_synthetic_passes,
                                             tracking_frame_col="frame_id", event_frame_col="frame_id")
            dfs_passes_with_synthetic.append(df_passes)

            # if plot_synthetic_passes:
            #     columns = st.columns(2)
            #     for pass_nr, (_, synthetic_pass) in enumerate(df_passes[df_passes["is_synthetic"]].iterrows()):
            #         df_frame = df_tracking[df_tracking["frame_id"] == synthetic_pass["frame_id"]]
            #         fig = plt.figure()
            #         plt.arrow(synthetic_pass["coordinates_x"], synthetic_pass["coordinates_y"], synthetic_pass["end_coordinates_x"] - synthetic_pass["coordinates_x"], synthetic_pass["end_coordinates_y"] - synthetic_pass["coordinates_y"], head_width=1, head_length=1, fc='k', ec='k')
            #
            #         plt.plot([-52.5, -52.5], [-34, 34], color="black", alpha=0.5)
            #         plt.plot([52.5, 52.5], [-34, 34], color="black", alpha=0.5)
            #         plt.plot([-52.5, 52.5], [-34, -34], color="black", alpha=0.5)
            #         plt.plot([-52.5, 52.5], [34, 34], color="black", alpha=0.5)
            #
            #         df_frame_home = df_frame[df_frame[tracking_team_col] == synthetic_pass[event_team_col]]
            #         plt.scatter(df_frame_home["x"], df_frame_home["y"], color="red", alpha=1)
            #         df_frame_def = df_frame[(df_frame[tracking_team_col] != synthetic_pass[event_team_col]) & (df_frame[tracking_team_col].notna())]
            #         plt.scatter(df_frame_def["x"], df_frame_def["y"], color="blue", alpha=1)
            #         df_frame_ball = df_frame[df_frame[tracking_team_col].isna()]
            #         plt.scatter(df_frame_ball["x"], df_frame_ball["y"], color="black", alpha=1, marker="x", s=100)
            #
            #         columns[pass_nr % 2].write(f"Pass {pass_nr} (frame {synthetic_pass['frame_id']})")
            #         columns[pass_nr % 2].write(fig)
            #
            #         plt.close()

        return dfs_passes_with_synthetic

    dfs_passes_with_synthetic = _get_dfs_passes_with_synthetic()

    ##
    dfs_training = []
    dfs_test = []
    for dataset_nr, df_passes in enumerate(dfs_passes_with_synthetic):
        df_passes = df_passes.copy()
        dataset_nr_col = get_unused_column_name(df_passes.columns, "dataset_nr")
        df_passes[dataset_nr_col] = dataset_nr
        df_passes["stratification_var"] = df_passes[outcome_col].astype(str) + "_" + df_passes["is_synthetic"].astype(str)

        df_passes = df_passes.reset_index(drop=True)

        df_passes["identifier"] = df_passes["dataset_nr"].astype(str) + "_" + df_passes.index.astype(str)

        assert len(df_passes["identifier"]) == len(set(df_passes["identifier"]))
        assert len(df_passes.index) == len(set(df_passes.index))

        df_training, df_test = sklearn.model_selection.train_test_split(
            df_passes, stratify=df_passes["stratification_var"], train_size=training_size, random_state=random_state
        )

        if exclude_synthetic_passes_from_training_set:
            df_training = df_training[~df_training["is_synthetic"]]
        if exclude_synthetic_passes_from_test_set:
            df_test = df_test[~df_test["is_synthetic"]]

        assert len(set(df_training.index).intersection(set(df_test.index))) == 0
        assert len(set(df_training["identifier"]).intersection(set(df_test["identifier"]))) == 0

        dfs_training.append(df_training.copy())
        dfs_test.append(df_test.copy())

    df_training = pd.concat(dfs_training).reset_index(drop=True).copy()
    st.write("df_training", df_training.shape)
    df_test = pd.concat(dfs_test).reset_index(drop=True).copy()
    st.write("df_test", df_test.shape)

    # assert no duplicate "identifier"
    assert len(df_training["identifier"]) == len(set(df_training["identifier"]))
    assert len(df_test["identifier"]) == len(set(df_test["identifier"]))
    # assert no overlapping "identifier"
    assert len(set(df_training["identifier"]).intersection(set(df_test["identifier"]))) == 0

    st.write("Number of training passes", len(df_training), f"avg. accuracy={df_training[outcome_col].mean():.1%}")
    st.write("Number of test passes", len(df_test), f"avg. accuracy={df_test[outcome_col].mean():.1%}")

    training_scores = get_scores(df_training, df_training[outcome_col].mean(), outcome_col=outcome_col)

    # test_scores = get_scores(df_test, df_training[outcome_col].mean(), outcome_col=outcome_col)
    # st.write("Training scores")
    # st.write(training_scores)
    # st.write("Test scores")
    # st.write(test_scores)

    def _choose_random_parameters(parameter_to_bounds):
        random_parameters = {}
        for param, bounds in parameter_to_bounds.items():
            # st.write("B", param, bounds, str(type(bounds[0])), str(type(bounds[-1])), "bool", isinstance(bounds[0], bool), isinstance(bounds[0], int), isinstance(bounds[0], float))
            if isinstance(bounds[0], bool):  # order matters, bc bool is also int
                random_parameters[param] = np.random.choice([bounds[0], bounds[-1]])
            elif isinstance(bounds[0], int) or isinstance(bounds[0], float):
                random_parameters[param] = np.random.uniform(bounds[0], bounds[-1])
            elif isinstance(bounds[0], str):
                random_parameters[param] = np.random.choice(bounds)
            else:
                raise NotImplementedError(f"Unknown type: {type(bounds[0])}")
        return random_parameters

    data = {
        "brier_score": [],
        "logloss": [],
        "auc": [],
        "brier_score_synthetic": [],
        # "logloss_synthetic": [],
        # "auc_synthetic": [],
        "brier_score_real": [],
        "logloss_real": [],
        "auc_real": [],
        "passes_json": [],
    }
    data.update({key: [] for key in training_scores.keys()})
    data["parameters"] = []
    st.write("n_steps")
    st.write(n_steps)
    progress_bar_text = st.empty()
    progress_bar = st.progress(0)
    display_df = st.empty()
    for i in tqdm.tqdm(range(n_steps), desc="Simulation", total=n_steps):
        gc.collect()
        progress_bar_text.text(f"Simulation {i + 1}/{n_steps}")
        progress_bar.progress((i + 1) / n_steps)
        if use_prefit:
            random_paramter_assignment = {}
        else:
            random_paramter_assignment = _choose_random_parameters(PARAMETER_BOUNDS)

        data_simres = {
            "xc": [],
            "success": [],
            "is_synthetic": [],
        }
        dfs_training_passes = []
        for dataset_nr, df_training_passes in df_training.groupby("dataset_nr"):
            df_training_passes = df_training_passes.copy()
            df_tracking = dfs_tracking[dataset_nr].copy()
            ret = get_expected_pass_completion(
                df_training_passes, df_tracking, event_frame_col="frame_id", tracking_frame_col="frame_id",
                event_start_x_col="coordinates_x",
                event_start_y_col="coordinates_y", event_end_x_col="end_coordinates_x",
                event_end_y_col="end_coordinates_y",
                event_team_col="team_id",
                event_player_col="player_id", tracking_player_col="player_id", tracking_team_col="team_id",
                ball_tracking_player_id="ball",
                tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy", tracking_v_col="v",
                tracking_team_in_possession_col="ball_possession",

                n_frames_after_pass_for_v0=5, fallback_v0=10,
                chunk_size=chunk_size,

                **random_paramter_assignment,
            )
            xc = ret.xc
            df_training_passes["xc"] = xc
            data_simres["xc"].extend(xc.tolist())
            data_simres["success"].extend(df_training_passes[outcome_col].tolist())
            data_simres["is_synthetic"].extend(df_training_passes["is_synthetic"].tolist())

            print("lens data_simres", {k: len(v) for k, v in data_simres.items()})

            dfs_training_passes.append(df_training_passes.copy())

        df_training_passes = pd.concat(dfs_training_passes)
        training_passes_json = df_training_passes.to_json(orient="records")
        data["passes_json"].append(training_passes_json)

        df_simres = pd.DataFrame(data_simres)
        data["parameters"].append(random_paramter_assignment)
        for key, value in random_paramter_assignment.items():
            data.setdefault(key, []).append(value)

        scores = get_scores(df_simres, df_training[outcome_col].mean(), outcome_col=outcome_col)
        for key, value in scores.items():
            # data[key].append(value)
            data.setdefault(key, []).append(value)

        df_to_display = pd.DataFrame(data).sort_values("logloss", ascending=True)
        df_to_display.iloc[1:, df_to_display.columns.get_loc("passes_json")] = np.nan
        df_to_display.iloc[1:, df_to_display.columns.get_loc("parameters")] = np.nan
        display_df.write(df_to_display.head(20))

        gc.collect()

        # write size of "data" in GB
        # import sys
        # st.write("Size of 'data' in GB:", round(sum(sys.getsizeof(value) for value in data.values()) / 1024**3, 2))

        if use_prefit:
            break

    df_training_results = pd.DataFrame(data)
    # st.write("df_training_results")
    # st.write(df_training_results)

    best_index = df_training_results["logloss"].idxmin()
    best_parameters = df_training_results["parameters"][best_index]
    best_passes = df_training_results["passes_json"][best_index]
    df_best_passes = pd.read_json(best_passes).copy()
    df_best_passes["error"] = (df_best_passes["success"] - df_best_passes["xc"]).abs()

    # st.write("df_best_passes")
    # st.write(df_best_passes.sort_values("xc"))

    st.write("### Training results")

    @st.fragment
    def frag1():
        n_bins = st.number_input("Number of bins for calibration plot", value=10, min_value=1, max_value=None)
        st.write(bin_nr_calibration_plot(df_best_passes, outcome_col=outcome_col, n_bins=n_bins))

    @st.fragment
    def frag2():
        binsize = st.number_input("Binsize for calibration plot", value=0.1, min_value=0.01, max_value=None)
        st.write(bin_nr_calibration_plot(df_best_passes, outcome_col=outcome_col, binsize=binsize))

    frag1()
    frag2()

    # st.stop()

    for (text, df) in [
        # ("Worst predictions", df_best_passes.sort_values("error", ascending=False)),
        # ("Best predictions", df_best_passes.sort_values("error", ascending=True)),
        ("Worst synthetic predictions",
         df_best_passes[df_best_passes["is_synthetic"]].sort_values("error", ascending=False)),
        ("Best synthetic predictions",
         df_best_passes[df_best_passes["is_synthetic"]].sort_values("error", ascending=True)),
        ("Worst real predictions",
         df_best_passes[~df_best_passes["is_synthetic"]].sort_values("error", ascending=False)),
        ("Best real predictions", df_best_passes[~df_best_passes["is_synthetic"]].sort_values("error", ascending=True)),
    ]:
        with st.expander(text):
            for pass_nr, (_, p4ss) in enumerate(df.iterrows()):
                st.write("#### Pass", pass_nr, "xc=", p4ss["xc"], "success=", p4ss["success"], "error=", p4ss["error"],
                         "is_synthetic=", p4ss["is_synthetic"])
                st.write(p4ss)
                plot_pass(p4ss, dfs_tracking[p4ss["dataset_nr"]])

                if pass_nr > 20:
                    break

    # with st.expander("Worst predictions"):
    #     for pass_nr, (_, p4ss) in enumerate(df_best_passes.sort_values("error", ascending=False).iterrows()):
    #         # st.write("Pass", pass_nr, "xc=", p4ss["xc"], "success=", p4ss["success"], "error=", p4ss["error"], "is_synthetic=", p4ss["is_synthetic"])
    #         st.write("#### Pass", pass_nr, "xc=", p4ss["xc"], "success=", p4ss["success"], "error=", p4ss["error"], "is_synthetic=", p4ss["is_synthetic"])
    #         plot_pass(p4ss, dfs_tracking[p4ss["dataset_nr"]])
    #
    #         if pass_nr > 1:
    #             break
    #
    # with st.expander("Best predictions"):
    #     for pass_nr, (_, p4ss) in enumerate(df_best_passes.sort_values("error", ascending=True).iterrows()):
    #         st.write("#### Pass", pass_nr, "xc=", p4ss["xc"], "success=", p4ss["success"], "error=", p4ss["error"], "is_synthetic=", p4ss["is_synthetic"])
    #         plot_pass(p4ss, dfs_tracking[p4ss["dataset_nr"]])
    #
    #         if pass_nr > 1:
    #             break

    data_simres = {
        "xc": [],
        "success": [],
        "is_synthetic": [],
    }
    for dataset_nr, df_test_passes in df_test.groupby("dataset_nr"):
        df_test_passes = df_test_passes.copy()
        df_tracking = dfs_tracking[dataset_nr].copy()
        ret = get_expected_pass_completion(
            df_test_passes, df_tracking, event_frame_col="frame_id", tracking_frame_col="frame_id",
            event_start_x_col="coordinates_x",
            event_start_y_col="coordinates_y", event_end_x_col="end_coordinates_x", event_end_y_col="end_coordinates_y",
            event_team_col="team_id",
            event_player_col="player_id", tracking_player_col="player_id", tracking_team_col="team_id",
            ball_tracking_player_id="ball",
            tracking_team_in_possession_col="ball_possession",
            tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy", tracking_v_col="v",
            n_frames_after_pass_for_v0=5, fallback_v0=10, chunk_size=chunk_size,
            **best_parameters,
        )
        data_simres["xc"].extend(ret.xc)
        data_simres["success"].extend(df_test_passes[outcome_col].tolist())
        data_simres["is_synthetic"].extend(df_test_passes["is_synthetic"].tolist())

    df_simres_test = pd.DataFrame(data_simres).copy()

    test_scores = get_scores(df_simres_test.copy(), df_test[outcome_col].mean(), outcome_col=outcome_col)
    st.write("### Test scores")
    df_test_scores = pd.DataFrame(test_scores, index=[0])

    # order cols like training
    df_test_scores = df_test_scores[[col for col in df_training_results.columns if col in df_test_scores.columns]]

    st.write("df_test_scores")
    st.write(df_test_scores)

    st.write("df_simres_test")
    st.write(df_simres_test)

    st.write(bin_nr_calibration_plot(df_simres_test, outcome_col=outcome_col, n_bins=10))
    st.write(bin_nr_calibration_plot(df_simres_test, outcome_col=outcome_col, n_bins=20))

    # for n_bins in [5, 10, 20]:
    #     st.write(f"Calibration plot with {n_bins} bins")
    #     fig = bin_nr_calibration_plot(df_simres_test, outcome_col=outcome_col, n_bins=n_bins)
    #     st.pyplot(fig)
    #
    # for binsize in [0.2, 0.1, 0.05]:
    #     st.write(f"Calibration plot with binsize {binsize}")
    #     fig = bin_nr_calibration_plot(df_simres_test, outcome_col=outcome_col, binsize=binsize)
    #     st.pyplot(fig)
    #
    st.stop()

    # brier = sklearn.metrics.brier_score_loss(df_simres_test["success"], df_simres_test["xc"])
    # logloss = sklearn.metrics.log_loss(df_simres_test["success"], df_simres_test["xc"], labels=[0, 1])
    # try:
    #     auc = sklearn.metrics.roc_auc_score(df_simres_test["success"], df_simres_test["xc"])
    # except ValueError as e:
    #     auc = e
    #
    # # brier = sklearn.metrics.brier_score_loss(df_test[outcome_col], df_test["xc"])
    # # logloss = sklearn.metrics.log_loss(df_test[outcome_col], df_test["xc"])
    # # auc = sklearn.metrics.roc_auc_score(df_test[outcome_col], df_test["xc"])
    # st.write("#### Test results")
    # st.write(f"Brier: {brier}")
    # st.write(f"Logloss: {logloss}")
    # st.write(f"AUC: {auc}")
    #
    # # brier_skill_score = 1 - brier / baseline_brier
    # # st.write(f"Brier skill score: {brier_skill_score}")
    #
    # for is_synthetic in [True, False]:
    #     df_synth = df_simres_test[df_simres_test["is_synthetic"] == is_synthetic]
    #     brier = sklearn.metrics.brier_score_loss(df_synth["success"], df_synth["xc"])
    #     logloss = sklearn.metrics.log_loss(df_synth["success"], df_synth["xc"], labels=[0, 1])
    #     try:
    #         auc = sklearn.metrics.roc_auc_score(df_synth["success"], df_synth["xc"])
    #     except ValueError as e:
    #         auc = e
    #     st.write(f"#### Test results (synthetic={is_synthetic})")
    #     st.write(f"Brier (synthetic={is_synthetic}): {brier}")
    #     st.write(f"Logloss (synthetic={is_synthetic}): {logloss}")
    #     st.write(f"AUC (synthetic={is_synthetic}): {auc}")
    #
    # return


def validation_dashboard():
    st.write(f"Getting kloppy data...")
    dfs_tracking, dfs_event = get_metrica_data()

    ### DAS vs x_norm
    # for df_tracking, df_event in zip(dfs_tracking, dfs_event):
    #     das_vs_xnorm(df_tracking, df_event)
    #     break

    ### Validation
    dfs_passes = []
    for i, (df_tracking, df_events) in enumerate(zip(dfs_tracking, dfs_event)):
        df_events["player_id"] = df_events["player_id"].str.replace(" ", "")
        df_events["receiver_player_id"] = df_events["receiver_player_id"].str.replace(" ", "")

        ### Prepare data -> TODO put into other function
        dataset_nr = i + 1
        st.write(f"### Dataset {dataset_nr}")
        # if dataset_nr == 1 or dataset_nr == 2:
        #     continue
        # df_tracking = dataset
        # st.write(f"Getting events...")
        # df_events = get_kloppy_events(dataset_nr)

        st.write("Pass %", f'{df_events[df_events["is_pass"]]["success"].mean():.2%}',
                 f'Passes: {len(df_events[df_events["is_pass"]])}')

        st.write("df_tracking", df_tracking.shape)
        st.write(df_tracking.head())
        st.write("df_events", df_events.shape)
        st.write(df_events)

        ### Do validation with this data
        dfs_event.append(df_events)
        df_passes = df_events[(df_events["is_pass"]) & (~df_events["is_high"])]

        df_passes = df_passes.drop_duplicates(subset=["frame_id"])

        # st.write("df_passes", df_passes.shape)
        # st.write("df_passes_fr_unique", len(df_passes["frame_id"].unique()))

        # duplicate_frames = df_passes["frame_id"].value_counts()
        # duplicate_frames = duplicate_frames[duplicate_frames > 1]

        # st.write(df_passes)
        # dfs_passes.append(df_passes.iloc[125:126])
        dfs_passes.append(df_passes)

        for _, p4ss in df_passes.iloc[125:126].iterrows():
            plot_pass(p4ss, df_tracking)

    # validate()
    n_steps = st.number_input("Number of simulations", value=100)
    use_prefit = st.checkbox("Use prefit", value=True)
    validate_multiple_matches(
        dfs_tracking=dfs_tracking, dfs_passes=dfs_passes, outcome_col="success", n_steps=n_steps, use_prefit=use_prefit
    )
    return


if __name__ == '__main__':
    validation_dashboard()
