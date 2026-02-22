import ast
import os
import sys

import streamlit.runtime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_auc_score
from pandas.errors import PerformanceWarning

import warnings

import io
import functools
import gc
import random
import math
import subprocess
import concurrent.futures

import matplotlib.patches

import mplsoccer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import sklearn.model_selection
import streamlit as st
import xmltodict
import kloppy.metrica


SEED = 1221871
rng = np.random.default_rng(SEED)
# {
# "bit_generator":"PCG64"
# "state":{
# "state":8.74101197170723e+37
# "inc":6.034160481890525e+37
# }
# "has_uint32":0
# "uinteger":0
# }
# st.write("rng.bit_generator.state start")
# st.write(rng.bit_generator.state)
# print(rng.bit_generator.state)
# assert rng.bit_generator.state == {'bit_generator': 'PCG64', 'state': {'state': 87410119717072287666137457930238493692, 'inc': 60341604818905247986700519057288636087}, 'has_uint32': 0, 'uinteger': 0}

plt.close()
gc.collect()

from accessible_space.utility import get_unused_column_name, progress_bar
from accessible_space.interface import per_object_frameify_tracking_data, get_expected_pass_completion, \
    get_dangerous_accessible_space, plot_expected_completion_surface, _DEFAULT_N_V0_FOR_XC, _DEFAULT_V0_MAX_FOR_XC, \
    _DEFAULT_USE_POSS_FOR_XC, _DEFAULT_V0_MIN_FOR_XC, _DEFAULT_USE_FIXED_V0_FOR_XC, get_das_gained
from accessible_space.core import _DEFAULT_PASS_START_LOCATION_OFFSET, _DEFAULT_B0, _DEFAULT_TIME_OFFSET_BALL, _DEFAULT_A_MAX, \
    _DEFAULT_USE_MAX, _DEFAULT_USE_APPROX_TWO_POINT, _DEFAULT_B1, _DEFAULT_PLAYER_VELOCITY, _DEFAULT_V_MAX, \
    _DEFAULT_KEEP_INERTIAL_VELOCITY, _DEFAULT_INERTIAL_SECONDS, _DEFAULT_TOL_DISTANCE, _DEFAULT_RADIAL_GRIDSIZE, \
    _DEFAULT_V0_PROB_AGGREGATION_MODE, _DEFAULT_NORMALIZE, _DEFAULT_USE_EFFICIENT_SIGMOID, _DEFAULT_FACTOR, \
    _DEFAULT_FACTOR2, _DEFAULT_RESPECT_OFFSIDE


metrica_open_data_base_dir = "https://raw.githubusercontent.com/metrica-sports/sample-data/refs/heads/master/data"

PARAMETER_BOUNDS = {
    # Core simulation model
    "pass_start_location_offset": [-3, 3],
    "time_offset_ball": [-3, 3],
    "radial_gridsize": [3, 7],
    "b0": [-5, 5],
    "b1": [-200, 0],
    "player_velocity": [5, 40],
    "keep_inertial_velocity": [True],
    "use_max": [False, True],
    "v_max": [5, 40],
    "a_max": [5, 20],
    "inertial_seconds": [0.0, 1.5],  # , True],
    "tol_distance": [0, 15],
    "use_approx_two_point": [False, True],
    "v0_prob_aggregation_mode": ["mean", "max"],
    "normalize": [False, True],
    "respect_offside": [False, True],
    "use_efficient_sigmoid": [False, True],
    "factor": [0.0001, 3],
    "factor2": [-2, 2],

    # xC
    "exclude_passer": [True],
    # "use_poss": [False, True],  # , True],#, True],
    "use_poss": [False, True],  # , True],#, True],
    # "use_fixed_v0": [False, True],
    "use_fixed_v0": [False, True],
    "v0_min": [1, 49.999],
    "v0_max": [50, 70],
    "n_v0": [15, 30],
}

PREFIT_PARAMS = {"a_max": _DEFAULT_A_MAX, "b0": _DEFAULT_B0, "b1": _DEFAULT_B1,
                 "exclude_passer": True, "factor": _DEFAULT_FACTOR, "factor2": _DEFAULT_FACTOR2,
                 "inertial_seconds": _DEFAULT_INERTIAL_SECONDS, "keep_inertial_velocity": _DEFAULT_KEEP_INERTIAL_VELOCITY,
                 "n_v0": _DEFAULT_N_V0_FOR_XC,
                 "normalize": _DEFAULT_NORMALIZE, "pass_start_location_offset": _DEFAULT_PASS_START_LOCATION_OFFSET,
                 "player_velocity": _DEFAULT_PLAYER_VELOCITY, "radial_gridsize": _DEFAULT_RADIAL_GRIDSIZE,
                 "time_offset_ball": _DEFAULT_TIME_OFFSET_BALL, "tol_distance": _DEFAULT_TOL_DISTANCE,
                 "use_approx_two_point": _DEFAULT_USE_APPROX_TWO_POINT, "use_efficient_sigmoid": _DEFAULT_USE_EFFICIENT_SIGMOID, "use_fixed_v0": _DEFAULT_USE_FIXED_V0_FOR_XC, "use_max": _DEFAULT_USE_MAX,
                 "respect_offside": _DEFAULT_RESPECT_OFFSIDE,
                 "use_poss": _DEFAULT_USE_POSS_FOR_XC, "v0_max": _DEFAULT_V0_MAX_FOR_XC, "v0_min": _DEFAULT_V0_MIN_FOR_XC,
                 "v0_prob_aggregation_mode": _DEFAULT_V0_PROB_AGGREGATION_MODE, "v_max": _DEFAULT_V_MAX}


def eval_benchmark():
    url = f"https://api.github.com/repos/EAISI/OJN-EPV-benchmark/contents/OJN-Pass-EPV-benchmark"

    response = requests.get(url)
    files = response.json()
    import accessible_space
    # Iterate over the files and print file names
    data = []
    for dir_nr, dir in accessible_space.progress_bar(enumerate(files), total=len(files), desc="Loading files"):
        dir = dir["name"]
        datapoint = {}
        file_mod = f"https://raw.githubusercontent.com/EAISI/OJN-EPV-benchmark/refs/heads/main/OJN-Pass-EPV-benchmark/{dir}/modification.csv"
        df_mod = pd.read_csv(file_mod, encoding="utf-8")
        higher_state = df_mod["higher_state_id"].iloc[0]
        datapoint["higher_state"] = df_mod["higher_state_id"].iloc[0]
        cols = st.columns(2)

        index2df = {}

        for index in [1, 2]:
            file_gamestate = f"https://raw.githubusercontent.com/EAISI/OJN-EPV-benchmark/refs/heads/main/OJN-Pass-EPV-benchmark/{dir}/game_state_{index}.csv"
            df = pd.read_csv(file_gamestate, encoding="utf-8")

            player2team = df[['player', 'team']].drop_duplicates().set_index('player').to_dict()['team']
            assert len(set(df["event_player"])) == 1
            event_player = df["event_player"].iloc[0]
            event_team = player2team[event_player]
            df["team_in_possession"] = event_team
            df["frame_id"] = 0
            df["playing_direction_event"] = df["playing_direction_event"].map({True: 1, False: -1})

            passer_x = df[df["player"] == event_player]["pos_x"].iloc[0]
            passer_y = df[df["player"] == event_player]["pos_y"].iloc[0]

            ret = accessible_space.get_dangerous_accessible_space(
                df, frame_col='frame_id', player_col='player', team_col='team', x_col='pos_x', y_col='pos_y',
                vx_col='smooth_x_speed', vy_col='smooth_y_speed', team_in_possession_col="team_in_possession",
                period_col=None, ball_player_id=0, attacking_direction_col="playing_direction_event",
                infer_attacking_direction=False, player_in_possession_col="event_player",
            )
            das = ret.das.iloc[0]
            acc_space = ret.acc_space.iloc[0]

            datapoint[f"as_{index}"] = acc_space
            datapoint[f"das_{index}"] = das

            plt.figure()
            df["team_is_in_possession"] = df["team"] == df["team_in_possession"]
            df["team_is_ball"] = df["player"] == 0
            df["team_is_defending"] = (df["team"] != df["team_in_possession"]) & ~df["team_is_ball"]
            df["color"] = df["team_is_in_possession"].map({True: "red", False: "blue"})
            df["color"] = df["color"].where(~df["team_is_ball"], "black")
            plt.scatter(df["pos_x"], df["pos_y"], c=df["color"], cmap="viridis", alpha=1)

            # plot the passer extra
            plt.scatter(passer_x, passer_y, c="orange", marker="x", s=10, label="Passer")

            # plot velocities
            for i, row in df.iterrows():
                plt.arrow(row["pos_x"], row["pos_y"], row["smooth_x_speed"], row["smooth_y_speed"],
                          head_width=0.5, head_length=0.5, fc='black', ec='black', alpha=1)

            plt.xlim(-52.5, 52.5)
            plt.plot([0, 0], [-34, 34], color="black", linewidth=2)
            plt.ylim(-34, 34)
            # accessible_space.plot_expected_completion_surface(ret.simulation_result, 0, color="blue")
            accessible_space.plot_expected_completion_surface(ret.dangerous_result, 0, color="red")
            cols[index-1].write(f"{index}, {df['playing_direction_event'].iloc[0]}, {higher_state=}, {das=}, {acc_space=}")
            cols[index-1].write(plt.gcf())
            cols[index-1].write(f"DAS: {das}m^2, DAS: {acc_space}m^2")

            plt.close()

            index2df[index] = df

        # check if dfs only differ by z coordinate of the ball
        index2df[1]["pos_z"] = None
        index2df[2]["pos_z"] = None

        # check if the two dataframes are equal
        if index2df[1].equals(index2df[2]):
            datapoint["differs_only_by_z"] = True
        else:
            datapoint["differs_only_by_z"] = False

        st.write(f"{higher_state=}")
        st.write("-----")

        # st.stop()
        data.append(datapoint)

    df = pd.DataFrame(data)

    def is_correct(row):
        if row["higher_state"] == 1:
            return row["das_1"] > row["das_2"]
        elif row["higher_state"] == 2:
            return row["das_1"] < row["das_2"]
        else:
            raise ValueError(f"Unknown higher state: {row['higher_state']}")

    df["correct"] = df.apply(lambda row: is_correct(row), axis=1)
    st.write("df")
    st.write(df)

    mean_correct = df["correct"].mean()

    mean_correct_not_only_differs_by_z = df[df["differs_only_by_z"] == False]["correct"].mean()

    st.write("mean_correct")
    st.write(mean_correct)

    st.write("mean_correct_not_only_differs_by_z")
    st.write(mean_correct_not_only_differs_by_z)

    return mean_correct, mean_correct_not_only_differs_by_z


def bootstrap_metric_ci(y_true, y_pred, fnc, n_iterations, conf_level=0.95, **kwargs):
    bs_loglosses = []
    # for i in progress_bar(range(n_iterations), total=n_iterations):
    # for i in range(n_iterations):
    i = 0
    for i in progress_bar(range(n_iterations), total=n_iterations):
        i += 1
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
        res = fnc(y_true_sample, y_pred_sample, **kwargs)
        if res is not None:
            bs_loglosses.append(res)

    bs_loglosses = np.array(sorted(bs_loglosses))

    logloss = fnc(y_true, y_pred, **kwargs)

    if len(bs_loglosses) == 0:
        return logloss, None, None

    percentile_alpha = ((1 - conf_level) / 2) * 100
    ci_lower = np.percentile(bs_loglosses, percentile_alpha)
    ci_higher = np.percentile(bs_loglosses, 100 - percentile_alpha)

    return logloss, ci_lower, ci_higher



def bootstrap_logloss_ci(y_true, y_pred, n_iterations, all_labels=np.array([0, 1]), conf_level=0.95):
    return bootstrap_metric_ci(y_true, y_pred, sklearn.metrics.log_loss, n_iterations, conf_level, labels=all_labels)


def bootstrap_brier_ci(y_true, y_pred, n_iterations, conf_level=0.95):
    return bootstrap_metric_ci(y_true, y_pred, sklearn.metrics.brier_score_loss, n_iterations, conf_level)


def ece(y_true, y_pred, bins=10):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    if y_pred.ndim == 2 and y_pred.shape[1] > 1:
        confidences = np.max(y_pred, axis=1)
        accuracies = (np.argmax(y_pred, axis=1) == y_true).astype(float)
    else:
        confidences = y_pred
        accuracies = y_true

    bins = int(bins)
    bin_boundaries = np.linspace(0, 1, bins + 1)
    ece = 0.0

    for i in range(bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        if np.any(in_bin):
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            bin_weight = np.sum(in_bin) / len(confidences)

            ece += np.abs(bin_accuracy - bin_confidence) * bin_weight

    return ece


def ece_ci(y_true, y_pred, n_iterations, conf_level=0.95):
    return bootstrap_metric_ci(y_true, y_pred, ece, n_iterations, conf_level)


def bootstrap_auc_ci(y_true, y_pred, n_iterations, conf_level=0.95):
    def error_handled_auc(y_true, y_pred):
        try:
            return sklearn.metrics.roc_auc_score(y_true, y_pred)
        except ValueError:
            return None
    return bootstrap_metric_ci(y_true, y_pred, error_handled_auc, n_iterations, conf_level)


def get_metrica_tracking_data(dataset_nr, limit=None):
    dataset = kloppy.metrica.load_open_data(dataset_nr, limit=None)
    df_tracking = dataset.to_df()
    return df_tracking


# TODO rename
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
        df["is_cross"] = df["Subtype"] == "CROSS"
        df["is_clearance"] = df["Subtype"] == "CLEARANCE"
        st.write(df["is_clearance"].sum(), "/", len(df), "clearances")
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

        # expanded_df = pd.DataFrame(df['subtypes'].apply(pd.Series))
        expanded_df = pd.DataFrame(df['subtypes'].apply(
            lambda x: pd.Series(x, dtype='float64') if x is None or isinstance(x, float) and pd.isna(x) else pd.Series(x)
        ))
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

        # check if the True's are mutually exclusive
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
def get_metrica_data(dummy=False):
    datasets = []
    dfs_event = []
    st.write(" ")
    st.write(" ")
    progress_bar_text = st.empty()
    st_progress_bar = st.progress(0)
    dataset_nrs = [1, 2, 3] if not dummy else [1, 3]
    for dataset_nr in dataset_nrs:
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
            xdiff = df_tracking[f"{player}_x"].diff().bfill()
            xdiff2 = -df_tracking[f"{player}_x"].diff(periods=-1).ffill()
            tdiff = df_tracking["timestamp"].diff().dt.total_seconds().bfill()
            tdiff2 = -df_tracking["timestamp"].diff(periods=-1).dt.total_seconds().ffill()
            vx = (xdiff + xdiff2) / (tdiff + tdiff2)
            df_tracking[f"{player}_vx"] = vx

            df_tracking[f"{player}_y"] = df_tracking[f"{player}_y"].astype(float)
            ydiff = df_tracking[f"{player}_y"].diff().bfill()
            ydiff2 = -df_tracking[f"{player}_y"].diff(periods=-1).ffill()
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

        if dummy:
            df_events = df_events.iloc[:100]
            df_tracking_obj = df_tracking_obj[df_tracking_obj["frame_id"].isin(df_events["frame_id"].unique())]

        datasets.append(df_tracking_obj)
        dfs_event.append(df_events)

        st_progress_bar.progress(dataset_nr / 3)

    return datasets, dfs_event


def check_synthetic_pass(p4ss, df_tracking_frame_attacking, v_receiver, v_receiver_threshold=4, v_players=10, pass_duration_threshold=0.5, pass_length_threshold=15, distance_to_origin_threshold=7.5):
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

    df_tracking_frame_attacking = df_tracking_frame_attacking[
        (df_tracking_frame_attacking["team_id"] == p4ss["team_id"]) &
        (df_tracking_frame_attacking["player_id"] != p4ss["player_id"])
    ]
    for _, row in df_tracking_frame_attacking.iterrows():
        x_player = row["x"]
        y_player = row["y"]

        distance_to_target = math.sqrt((x_player - p4ss["end_coordinates_x"]) ** 2 + (y_player - p4ss["end_coordinates_y"]) ** 2)
        necessary_speed_to_reach_target = distance_to_target / pass_duration
        distance_to_origin = math.sqrt((x_player - p4ss["coordinates_x"]) ** 2 + (y_player - p4ss["coordinates_y"]) ** 2)

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

        # st.write("distance_to_origin", distance_to_origin, distance_to_target)
        if necessary_speed_to_reach_target < v_players or distance_to_origin < distance_to_origin_threshold or can_intercept(x0_pass, y0_pass, v0x_pass, v0y_pass, x_player, y_player, v_players, pass_duration):
            # st.write("False")
            return False  # Criterion 3: Pass cannot be received by any teammate

    return True


def add_synthetic_passes(
    df_passes, df_tracking, n_synthetic_passes=5, event_frame_col="frame_id", tracking_frame_col="frame_id",
    event_team_col="team_id", tracking_team_col="team_id", event_player_col="player_id",
    tracking_player_col="player_id", x_col="x", y_col="y",
    new_is_synthetic_col="is_synthetic"
):
    st.write("Adding", n_synthetic_passes, "synthetic passes")
    df_passes[new_is_synthetic_col] = False
    synthetic_passes = []

    teams = df_tracking[tracking_team_col].unique()

    local_rng = np.random.default_rng(SEED)

    for _, p4ss in df_passes.sample(frac=1, random_state=local_rng.bit_generator).iterrows():
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

    assert len(df_synthetic_passes) == n_synthetic_passes, f"len(df_synthetic_passes)={len(df_synthetic_passes)} != n_synthetic_passes={n_synthetic_passes}, (len(synthetic_passes)={len(synthetic_passes)}"

    return pd.concat([df_passes, df_synthetic_passes], axis=0)

# TODO change definition of impossible pass to exclude passes where opponent is around passer (e.g. 5 meter radius)
def get_scores(_df, baseline_accuracy, outcome_col="success", add_confidence_intervals=True, n_bootstrap_samples=1000):
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

        data["ece"] = ece(df[outcome_col].values, df["xc"].values)

        if add_confidence_intervals:
            logloss_from_ci, logloss_ci_lower, logloss_ci_upper = bootstrap_logloss_ci(df[outcome_col].values, df["xc"].values, n_iterations=n_bootstrap_samples)
            data["logloss_ci_lower"] = logloss_ci_lower
            data["logloss_ci_upper"] = logloss_ci_upper
            _, brier_ci_lower, brier_ci_upper = bootstrap_brier_ci(df[outcome_col].values, df["xc"].values, n_iterations=n_bootstrap_samples)
            data["brier_ci_lower"] = brier_ci_lower
            data["brier_ci_upper"] = brier_ci_upper
            _, auc_ci_lower, auc_ci_upper = bootstrap_auc_ci(df[outcome_col].values, df["xc"].values, n_iterations=n_bootstrap_samples)
            data["auc_ci_lower"] = auc_ci_lower
            data["auc_ci_upper"] = auc_ci_upper
            _, ecll_ci_lower, ecll_ci_upper = ece_ci(df[outcome_col].values, df["xc"].values, n_iterations=n_bootstrap_samples)
            data["ece_ci_lower"] = ecll_ci_lower
            data["ece_ci_upper"] = ecll_ci_upper

        # data["brier_score"] = sklearn.metrics.brier_score_loss(df[outcome_col], df["xc"])

        try:
            data["logloss"] = sklearn.metrics.log_loss(df[outcome_col], df["xc"], labels=np.array([0, 1]))
        except ValueError:
            data["logloss"] = np.nan
        try:
            data["auc"] = sklearn.metrics.roc_auc_score(df[outcome_col], df["xc"], labels=np.array([0, 1]))
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

        baseline_accuracy_synth = df_synth[outcome_col].mean()

        if "xc" in df.columns:
            try:
                data[f"brier_score_{synth_str}"] = sklearn.metrics.brier_score_loss(df_synth[outcome_col], df_synth["xc"])
            except ValueError:
                data[f"brier_score_{synth_str}"] = np.nan
            try:
                data[f"logloss_{synth_str}"] = sklearn.metrics.log_loss(df_synth[outcome_col], df_synth["xc"], labels=np.array([0, 1]))
            except ValueError:
                data[f"logloss_{synth_str}"] = np.nan
            try:
                data[f"auc_{synth_str}"] = sklearn.metrics.roc_auc_score(df_synth[outcome_col], df_synth["xc"], labels=np.array([0, 1]))
            except ValueError:
                data[f"auc_{synth_str}"] = np.nan
            try:
                data[f"ece_{synth_str}"] = ece(df_synth[outcome_col].values, df_synth["xc"])
            except (ValueError, AssertionError):
                data[f"ece_{synth_str}"] = np.nan

            if add_confidence_intervals:
                logloss_from_ci, logloss_ci_lower, logloss_ci_upper = bootstrap_logloss_ci(df_synth[outcome_col].values, df_synth["xc"].values, n_iterations=n_bootstrap_samples)
                data[f"logloss_ci_lower_{synth_str}"] = logloss_ci_lower
                data[f"logloss_ci_upper_{synth_str}"] = logloss_ci_upper
                _, brier_ci_lower, brier_ci_upper = bootstrap_brier_ci(df_synth[outcome_col].values, df_synth["xc"].values, n_iterations=n_bootstrap_samples)
                data[f"brier_ci_lower_{synth_str}"] = brier_ci_lower
                data[f"brier_ci_upper_{synth_str}"] = brier_ci_upper
                _, auc_ci_lower, auc_ci_upper = bootstrap_auc_ci(df_synth[outcome_col].values, df_synth["xc"].values, n_iterations=n_bootstrap_samples)
                data[f"auc_ci_lower_{synth_str}"] = auc_ci_lower
                data[f"auc_ci_upper_{synth_str}"] = auc_ci_upper
                _, ecll_ci_lower, ecll_ci_upper = ece_ci(df[outcome_col].values, df["xc"].values, n_iterations=n_bootstrap_samples)
                data["ece_ci_lower"] = ecll_ci_lower
                data["ece_ci_upper"] = ecll_ci_upper

        data[f"average_accuracy_{synth_str}"] = df_synth[outcome_col].mean()
        data[f"synthetic_share_{synth_str}"] = df_synth["is_synthetic"].mean()
        try:
            data[f"baseline_brier_{synth_str}"] = sklearn.metrics.brier_score_loss(df_synth[outcome_col], [baseline_accuracy_synth] * len(df_synth))
        except ValueError:
            data[f"baseline_brier_{synth_str}"] = np.nan
        try:
            data[f"baseline_loglos_{synth_str}"] = sklearn.metrics.log_loss(df_synth[outcome_col], [baseline_accuracy_synth] * len(df_synth))
        except ValueError:
            data[f"baseline_loglos_{synth_str}"] = np.nan
        try:
            data[f"baseline_auc_{synth_str}"] = sklearn.metrics.roc_auc_score(df_synth[outcome_col], [baseline_accuracy_synth] * len(df_synth))
        except ValueError:
            data[f"baseline_auc_{synth_str}"] = np.nan
        try:
            data[f"ece_{synth_str}"] = np.nan  # ece(df_synth[outcome_col].values, [baseline_accuracy_synth] * len(df_synth))
        except ValueError:
            data[f"ece_{synth_str}"] = np.nan

    return data


def calibration_histogram(df, hist_col="xc", synth_col="is_synthetic", n_bins=None, binsize=None, add_text=True, use_boken_axis=True, ylim1=450, ylim2=1550, ylim3=1600, is_training=False):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use("seaborn-v0_8")
    plt.figure()

    # reset style

    if use_boken_axis:
        import brokenaxes
        bax = brokenaxes.brokenaxes(xlims=((0, 1),), ylims=((0, ylim1), (ylim2, ylim3)), hspace=0.125)
    else:
        bax = plt.gca()
    # plt.title("Distribution of predicted pass success rates in training set")

    # x = np.linspace(0, 1, 100)
    # bax.plot(x, np.sin(10 * x), label='sin')
    # bax.plot(x, np.cos(10 * x), label='cos')
    # bax.legend(loc=3)
    # bax.set_xlabel('time')
    # bax.set_ylabel('value')

    # if binsize is None and n_bins is not None:
    #     df[bin_col] = pd.qcut(df[hist_col], n_bins, labels=False, duplicates="drop")
    # elif binsize is not None and n_bins is None:
    #     min_val = df[hist_col].min()
    #     max_val = df[hist_col].max()
    #     bin_edges = [min_val + i * binsize for i in range(int((max_val - min_val) / binsize) + 2)]
    #     df[bin_col] = pd.cut(df[hist_col], bins=bin_edges, labels=False, include_lowest=True)
    # else:
    #     raise ValueError("Either n_bins or binsize must be specified")
    custom_style = {
        'axes.edgecolor': 'gray',
        'axes.facecolor': 'whitesmoke',
        'axes.grid': True,
        'grid.color': 'lightgray',
        'grid.linestyle': '--',
        'axes.spines.right': False,
        'axes.spines.top': False,
    }
    plt.rcParams.update({
        'axes.facecolor': 'gray',
        'axes.edgecolor': 'black',
        'axes.grid': True,
        'grid.color': '.8',
        'grid.linestyle': '-',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.bottom': True,
        'ytick.left': True,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.frameon': False,
        'legend.fontsize': 12,
        'figure.facecolor': 'gray',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
    })
    plt.rcParams.update(custom_style)

    # plt.hist(data, bins=30, stacked=True, label=['Data 1', 'Data 2', 'Data 3'], color=['blue', 'green', 'red'])
    import matplotlib.cm
    cmap = matplotlib.cm.get_cmap('viridis')  # or 'plasma', 'inferno', etc.

    groups = [(is_synthetic, df) for is_synthetic, df in df.groupby(synth_col)]
    dfs = [group[1] for group in groups]
    # plt.hist([df["xc"] for df in dfs], stacked=True, bins=n_bins, label=[f"Synthetic={group[0]}" for group in groups])
    bax.hist([df[hist_col] for df in dfs], stacked=True, bins=n_bins, density=False,
             label=[f"{'Synthetic passes' if group[0] else 'Real passes'}" for group in groups],
             color=[cmap(i / len(groups)) for i in range(len(groups))]
             )

    # set x limit
    bax.set_xlim(0, 1)
    # plt.ylim(0, 400)

    # set xticks
    bax.set_xticks(np.arange(0, 1.1, 0.1))
    plt.gca().set_xticks(np.arange(0, 1.1, 0.1))
    plt.gca().set_xticklabels([f"{i:.1f}" for i in np.arange(0, 1.1, 0.1)])

    # thin y grid


    # plt.gca().set_xlabel("X Axis", labelpad=20)  # Move X label down
    # plt.gca().set_ylabel("Y Axis", labelpad=20)  # Move Y label left

    bax.set_xlabel("Predicted pass completion probability", labelpad=7)
    bax.set_ylabel(f"Number of passes in {'test' if not is_training else 'training'} set", labelpad=35)

    # bax.annotate('xxxx (synthetic passes)', xy=(0.1, 400), xytext=(0.1, 300),
    #             arrowprops=dict(facecolor='yellow', arrowstyle='->', lw=2, ls='dashed', color='yellow'),
    #             fontsize=12, color='yellow', ha='center', va='center')

    bax.set_axisbelow(True)
    plt.gca().xaxis.grid(color='gray', linestyle='dashed', alpha=0.5)
    plt.gca().yaxis.grid(color='gray', linestyle='dashed', alpha=0.5)

    bax.legend(loc='upper right', fontsize=12, frameon=True)

    plt.rcParams.update(plt.rcParamsDefault)

    return plt.gcf()


def get_bins(df, prediction_col="xc", outcome_col="success", new_bin_col="bin", n_bins=None, binsize=None):
    if binsize is None and n_bins is not None:
        df[new_bin_col] = pd.qcut(df[prediction_col], n_bins, labels=False, duplicates="drop")
    elif binsize is not None and n_bins is None:
        min_val = df[prediction_col].min()
        max_val = df[prediction_col].max()
        bin_edges = [min_val + i * binsize for i in range(int((max_val - min_val) / binsize) + 2)]
        df[new_bin_col] = pd.cut(df[prediction_col], bins=bin_edges, labels=False, include_lowest=True)
    else:
        raise ValueError("Either n_bins or binsize must be specified")

    df_calibration = df.groupby(new_bin_col).agg({outcome_col: "mean", prediction_col: "mean"}).reset_index()
    df_calibration[new_bin_col] = df_calibration[new_bin_col]
    return df_calibration


def bin_nr_calibration_plot(df, prediction_col="xc", outcome_col="success", n_bins=None, binsize=None, add_text=True, style="seaborn-v0_8", add_interval=True, interval_confidence_level=0.95, n_bootstrap_samples=1000, method="wilson"):
    bin_col = get_unused_column_name(df.columns, "bin")

    df_calibration = get_bins(df, prediction_col, outcome_col, bin_col, n_bins, binsize)

    plt.style.use(style)
    fig, ax = plt.subplots()

    ax.plot(df_calibration[prediction_col], df_calibration[outcome_col], marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black")

    if add_interval:
        avg_predictions = []
        uppers = []
        lowers = []
        if method == "bootstrap":
            avg_avg_outcomes = []
            for bin, df_bin in df.groupby(bin_col):
                avg_outcomes = []
                for i in range(n_bootstrap_samples):
                    df_sample = df_bin.sample(n=len(df_bin), replace=True)
                    avg_outcomes.append(df_sample[outcome_col].mean())

                avg_outcome_lower = np.percentile(avg_outcomes, 100 * (1 - interval_confidence_level) / 2)
                avg_outcome_upper = np.percentile(avg_outcomes, 100 * (1 - (1 - interval_confidence_level) / 2))
                avg_predictions.append(df_calibration.loc[df_calibration[bin_col] == bin, prediction_col].iloc[0])
                avg_avg_outcomes.append(np.mean(avg_outcomes))
                uppers.append(avg_outcome_upper)
                lowers.append(avg_outcome_lower)

            avg_predictions = np.array(avg_predictions)
            uppers = np.array(uppers)
            lowers = np.array(lowers)

            df_bootstrap = pd.DataFrame({"avg_prediction": avg_predictions, "avg_outcome": avg_avg_outcomes, "upper": uppers, "lower": lowers})
        # else:
        #     for bin, df_bin in df.groupby(bin_col):
        #         avg_outcomes = []
        #         for i in range(n_bootstrap_samples):
        #             df_sample = df_bin.sample(n=len(df_bin), replace=True)
        #             avg_outcomes.append(df_sample[outcome_col].mean())
        #
        #         k = df_bin[outcome_col].sum()
        #         n = len(df_bin)
        #         ci_low, ci_upp = statsmodels.api.stats.proportion_confint(k, n, alpha=0.05, method='wilson')
        #         avg_predictions.append(df_calibration.loc[df_calibration[bin_col] == bin, prediction_col].iloc[0])
        #         uppers.append(ci_upp)
        #         lowers.append(ci_low)

        plt.fill_between(avg_predictions, lowers, uppers, color='grey', alpha=0.3, label='Uncertainty Area')

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

    # xticks every 0.1
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels([f"{i:.1f}" for i in np.arange(0, 1.1, 0.1)])
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f"{i:.1f}" for i in np.arange(0, 1.1, 0.1)])

    ax.set_xlabel("Predicted pass completion probability")
    ax.set_ylabel("Observed pass completion rate")
    # ax.set_title(title)
    return fig


def plot_pass(p4ss, df_tracking, add_legend=True, add_as=False, add_das=False, flip=False, legend_loc="best",
              legend_bbox_to_anchor=None, use_green_background=True, add_pass_to_legend=True):
    # from mplsoccer import Pitch, VerticalPitch

    if use_green_background:
        pitch = mplsoccer.Pitch(pitch_type="secondspectrum", pitch_length=105, pitch_width=68, pitch_color='#aabb97', line_color='white',
                  stripe_color='#c2d59d', stripe=True)
    else:
        pitch = mplsoccer.Pitch(pitch_type="secondspectrum", pitch_length=105, pitch_width=68, pitch_color="white", shade_color="white")

#    pitch = mplsoccer.Pitch(pitch_type="secondspectrum", pitch_length=105, pitch_width=68, pitch_color="white", shade_color="white")
    # specifying figure size (width, height)
    fig, ax = pitch.draw(figsize=(8, 4))

    plt.style.use("seaborn-v0_8-white")  # ?

    # plt.title(f"Pass: {p4ss['success']}")

    df_frame = df_tracking[df_tracking["frame_id"] == p4ss["frame_id"]].copy()
    df_frame["ball_owning_team_id"] = p4ss["team_id"]


    red_color = "#ff6666"
    red_color_pass = "#b30000"
    blue_color = "#6864b0"

    if flip:
        df_frame["x"] = -df_frame["x"]
        df_frame["y"] = -df_frame["y"]
        df_frame["vx"] = -df_frame["vx"]
        df_frame["vy"] = -df_frame["vy"]

        if "coordinates_x" in p4ss:
            p4ss["coordinates_x"] = -p4ss["coordinates_x"]
            p4ss["coordinates_y"] = -p4ss["coordinates_y"]
            p4ss["end_coordinates_x"] = -p4ss["end_coordinates_x"]
            p4ss["end_coordinates_y"] = -p4ss["end_coordinates_y"]

    if add_das or add_as:
        assert len(df_frame) > 0
        df_frame["ball_owning_team_id"] = p4ss["team_id"]

        das = get_dangerous_accessible_space(
            df_frame, team_in_possession_col="ball_owning_team_id", period_col=None,
            n_v0=150, radial_gridsize=1.83, n_angles=40,
        )
        # 1.3419858566843872	3749.9871613033556
        df_frame["DAS"] = das.das
        df_frame["AS"] = das.acc_space
        st.write(df_frame)

        if add_das:
            plot_expected_completion_surface(das.dangerous_result, color="red", plot_gridpoints=False)
        if add_as:
            plot_expected_completion_surface(das.simulation_result, color="red", plot_gridpoints=False)

    try:
        arrow = matplotlib.patches.FancyArrowPatch(
            (p4ss["coordinates_x"], p4ss["coordinates_y"]), (p4ss["end_coordinates_x"], p4ss["end_coordinates_y"]),
            arrowstyle="->", mutation_scale=30, color=red_color_pass, linewidth=2, ec=red_color_pass, fc=red_color_pass

        )
        plt.gca().add_artist(arrow)
    except KeyError:
        pass
    # plt.scatter(p4ss["coordinates_x"], p4ss["coordinates_y"], c=red_color_pass, marker="x", s=150, label="Pass origin (event data)")

    # df_frame = df_tracking[df_tracking["frame_id"] == p4ss["frame_id"]]
    for team_nr, team in enumerate(df_frame["team_id"].unique()):
        team_color = red_color if team == p4ss["team_id"] else blue_color
        team_name = "Attacking team" if team == p4ss["team_id"] else "Defending team"

        if team is None:
            continue
        df_frame_team = df_frame[df_frame["team_id"] == team].copy()

        df_frame_team["player_name"] = df_frame_team["player_id"].map(lambda x: f"Player{str(x).replace('home_', '').replace('away_', '').replace('PlayerP', 'P')}")

        x = df_frame_team["x"].tolist()
        y = df_frame_team["y"].tolist()

        vx = df_frame_team["vx"].tolist()
        vy = df_frame_team["vy"].tolist()

        player_names = df_frame_team["player_name"].tolist()

        for i in range(len(x)):
            label = None if i == 0 and team_nr == 0 else None
            if vx[i] ** 2 + vy[i] ** 2 > 0:
                plt.arrow(x=x[i], y=y[i], dx=vx[i]/2, dy=vy[i]/2, width=0.425/1.5, head_width=1.5/1.5, head_length=1.5/1.5, fc="black", ec="black", label=label)

            # plot names
            # if i == 0:
            if not np.isnan(x[i]) and not np.isnan(y[i]):
                plt.text(x[i], y[i] - 2.65, player_names[i], fontsize=8, color=team_color, ha="center", va="center")

        plt.scatter(x, y, c=team_color, label=team_name, edgecolors="black", s=50, linewidth=1)

    # plot ball position
    df_frame_ball = df_frame[df_frame["player_id"] == "ball"]
    x_ball = df_frame_ball["x"].iloc[0]
    y_ball = df_frame_ball["y"].iloc[0]
    plt.scatter(x_ball, y_ball, c="black", marker="x", s=100, label="Ball position (tracking data)")

    if add_legend:
        plt.legend(frameon=True, loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)

    # handles, labels = plt.gca().get_legend_handles_labels()

    plt.scatter([], [], marker=r'$\rightarrow$', label='Velocity', color='black', s=100)  # dummy scatter to add an item to the legend
    if add_pass_to_legend:
        plt.scatter([], [], marker=r'$\rightarrow$', label='Pass', color=red_color_pass, s=100)  # dummy scatter to add an item to the legend

    # handles[0] = matplotlib.patches.FancyArrowPatch((0, 0), (1, 0), color="blue", mutation_scale=0.0000, mutation_aspect=0.0)
    # handles.append(matplotlib.lines.Line2D([], [], color=red_color_pass, marker='>', markersize=10, label="Arrow"))
    # handles.append(matplotlib.patches.FancyArrowPatch((0, 0), (1, 0), color="blue", mutation_scale=0.0000, mutation_aspect=0.0))
    # labels.append(None)
    # arrow =
    # plt.legend(handles=handles, labels=labels, frameon=True)
    if add_legend:
        plt.legend(frameon=True, prop={'size': 9}, loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)

    st.write(plt.gcf())

    if add_legend:
        filename = f"{p4ss['frame_id']}_legend_{add_das}_{add_as}"
    else:
        filename = f"{p4ss['frame_id']}_{add_das}_{add_as}"

    plt.savefig(os.path.join(os.path.dirname(__file__), f"{filename}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(os.path.dirname(__file__), f"{filename}.pdf"), bbox_inches="tight")

    plt.close()

    return plt.gcf()


def _choose_random_parameters(parameter_to_bounds):
    random_parameters = {}
    for param, bounds in parameter_to_bounds.items():
        if isinstance(bounds[0], bool):  # order matters, bc bool is also int
            random_parameters[param] = np.random.choice([bounds[0], bounds[-1]])
        elif isinstance(bounds[0], int) or isinstance(bounds[0], float):
            random_parameters[param] = np.random.uniform(bounds[0], bounds[-1])
        elif isinstance(bounds[0], str):
            random_parameters[param] = np.random.choice(bounds)
        else:
            raise NotImplementedError(f"Unknown type: {type(bounds[0])}")
    return random_parameters


def simulate_parameters(df_training, dfs_tracking, use_prefit, add_confidence_intervals, chunk_size=200, outcome_col="success", calculate_passes_json=False, parameter_assignment=None):
    gc.collect()

    data = {}

    if parameter_assignment is None:
        if use_prefit:
            parameter_assignment = PREFIT_PARAMS
        else:
            parameter_assignment = _choose_random_parameters(PARAMETER_BOUNDS)

    assert "respect_offside" in PREFIT_PARAMS
    assert "respect_offside" in parameter_assignment

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
            tracking_period_col="period_id",

            n_frames_after_pass_for_v0=5, fallback_v0=10,
            chunk_size=chunk_size,
            use_progress_bar=False,
            use_event_coordinates_as_ball_position=True,  # necessary because validation uses duplicate frames (artificial passes)

            **parameter_assignment,
        )
        xc = ret.xc
        df_training_passes["xc"] = xc
        data_simres["xc"].extend(xc.tolist())
        data_simres["success"].extend(df_training_passes[outcome_col].tolist())
        data_simres["is_synthetic"].extend(df_training_passes["is_synthetic"].tolist())

        dfs_training_passes.append(df_training_passes.copy())

    df_training_passes = pd.concat(dfs_training_passes)
    training_passes_json = df_training_passes.to_json(orient="records")
    if calculate_passes_json:
        data["passes_json"] = training_passes_json
    else:
        data["passes_json"] = ""

    df_simres = pd.DataFrame(data_simres)
    data["parameters"] = parameter_assignment
    for key, value in parameter_assignment.items():
        data[key] = value

    scores = get_scores(df_simres, df_training[outcome_col].mean(), outcome_col=outcome_col, add_confidence_intervals=add_confidence_intervals)
    for key, value in scores.items():
        data[key] = value

    gc.collect()

    return data


def validate_multiple_matches(
    dfs_tracking, dfs_passes, n_steps=100, training_size=0.7, use_prefit=True, outcome_col="success", run_asserts=True,
):
    @st.fragment
    def frag_plot_das():
        if not st.toggle("frag_plot_das", value=True):
            return

        dataset_nr = st.number_input("Dataset nr", value=0, min_value=0, max_value=len(dfs_tracking) - 1, key="frag_plot_das2")
        # default_frame = 48767  # 7404
        default_frame = 7404
        frames = dfs_tracking[dataset_nr]["frame_id"].unique().tolist()
        frame = st.selectbox("Frame", frames, key="frag_plot_das", index=frames.index(default_frame) if default_frame in frames else frames[0])
        flip = st.checkbox("Flip", True)

        # flip_x = st.checkbox("Flip X (breaks data!)", False)
        # if flip_x:
        #     dfs_tracking[dataset_nr]["x"] *= -1
        #     dfs_tracking[dataset_nr]["vx"] *= -1

        teams = dfs_tracking[dataset_nr]["team_id"].dropna().unique()
        selected_team = st.selectbox("Team", teams, key="frag_plot_das3")

        plot_all_passes = st.checkbox("All passes", False)
        if plot_all_passes :
            for add_legend in [False, True]:
                for pass_nr, (_, p4ss) in enumerate(dfs_passes[dataset_nr].iloc[183+24:].iterrows()):
                    st.write(f"{pass_nr=}")
                    try:
                        plot_pass(p4ss, dfs_tracking[dataset_nr], add_legend=add_legend,
                                  legend_loc="lower left", add_as=True, add_das=False, flip=False,
                                  use_green_background=False,
                                  legend_bbox_to_anchor=(0.05, 0.0), add_pass_to_legend=False)
                    except AssertionError as e:
                        pass
                    plt.close()

        for das in ["as", "das"]:
            plot_pass({"frame_id": frame, "team_id": selected_team}, dfs_tracking[dataset_nr], add_legend=False,
                      legend_loc="lower left", add_as=das == "as", add_das=das == "das", flip=flip, use_green_background=False,
                      legend_bbox_to_anchor=(0.05, 0.0), add_pass_to_legend=False)

        return

    if run_asserts:
        p4ss = dfs_passes[0][dfs_passes[0]["frame_id"] == 7404].iloc[0]
        df_frame = dfs_tracking[0][dfs_tracking[0]["frame_id"] == p4ss["frame_id"]].copy()
        df_frame["ball_owning_team_id"] = p4ss["team_id"]
        df_frame["x"] = -df_frame["x"]
        df_frame["y"] = -df_frame["y"]
        df_frame["vx"] = -df_frame["vx"]
        df_frame["vy"] = -df_frame["vy"]
        das = get_dangerous_accessible_space(
            df_frame, team_in_possession_col="ball_owning_team_id", period_col=None, n_v0=150, radial_gridsize=1.83,
            n_angles=40,
        )
        df_frame["DAS"] = das.das
        df_frame["AS"] = das.acc_space
        assert round(das.das.iloc[0], 2) == 1.34
        assert round(das.acc_space.iloc[0], 0) == 3750

    frag_plot_das()

    exclude_synthetic_passes_from_training_set = st.checkbox("Exclude synthetic passes from training set", value=False)
    exclude_synthetic_passes_from_test_set = st.checkbox("Exclude synthetic passes from test set", value=False)
    chunk_size = st.number_input("Chunk size", value=50, min_value=1, max_value=None)
    max_workers = st.number_input("Max workers", value=4, min_value=1, max_value=None)

    ## Add synthetic passes
    @st.cache_resource  # REMOVE FOR REPRO
    def _get_dfs_passes_with_synthetic():
        dfs_passes_with_synthetic = []
        for df_tracking, df_passes in progress_bar(zip(dfs_tracking, dfs_passes)):
            n_synthetic_passes = len(df_passes[df_passes["success"]]) - len(df_passes[~df_passes["success"]])

            with st.spinner(f"Adding synthetic passes to dataset ({n_synthetic_passes} synthetic passes)"):
                df_passes = add_synthetic_passes(df_passes, df_tracking, n_synthetic_passes=n_synthetic_passes, tracking_frame_col="frame_id", event_frame_col="frame_id")
                dfs_passes_with_synthetic.append(df_passes)

        return dfs_passes_with_synthetic

    dfs_passes_with_synthetic = _get_dfs_passes_with_synthetic()

    dfs_training = []
    dfs_test = []
    for dataset_nr, df_passes in enumerate(dfs_passes_with_synthetic):
        df_passes = df_passes.copy()
        dataset_nr_col = "dataset_nr"
        df_passes[dataset_nr_col] = dataset_nr
        df_passes["stratification_var"] = df_passes[outcome_col].astype(str) + "_" + df_passes["is_synthetic"].astype(str)

        df_passes = df_passes.reset_index(drop=True)

        df_passes["identifier"] = df_passes["dataset_nr"].astype(str) + "_" + df_passes.index.astype(str)

        assert len(df_passes["identifier"]) == len(set(df_passes["identifier"]))
        assert len(df_passes.index) == len(set(df_passes.index))

        df_training, df_test = sklearn.model_selection.train_test_split(
            df_passes, stratify=df_passes["stratification_var"], train_size=training_size, random_state=1893
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
    df_test = pd.concat(dfs_test).reset_index(drop=True).copy()

    # assert no duplicate "identifier"
    # st.write("df_training")
    # st.write(df_training)
    # st.write("df_test")
    # st.write(df_test)
    # i_dupes = df_training[df_training.duplicated(subset=["identifier"], keep=False)]
    # st.write("df_training.loc[i_dupes]")
    # st.write(i_dupes)
    assert len(df_training["identifier"]) == len(set(df_training["identifier"]))
    assert len(df_test["identifier"]) == len(set(df_test["identifier"]))
    # assert no overlapping "identifier"
    assert len(set(df_training["identifier"]).intersection(set(df_test["identifier"]))) == 0

    st.write("Number of training passes", len(df_training), f"avg. accuracy={df_training[outcome_col].mean():.1%}")
    st.write("Number of test passes", len(df_test), f"avg. accuracy={df_test[outcome_col].mean():.1%}")

    training_scores = get_scores(df_training, df_training[outcome_col].mean(), outcome_col=outcome_col, add_confidence_intervals=False)

    data = {
        "brier_score": [],
        "logloss": [],
        "auc": [],
        "brier_score_synthetic": [],
        "brier_score_real": [],
        "logloss_real": [],
        "auc_real": [],
        "passes_json": [],
    }
    data.update({key: [] for key in training_scores.keys()})
    data["parameters"] = []

    expensive_cols = ["passes_json", "parameters"]

    simulate_params_partial = functools.partial(simulate_parameters, df_training, dfs_tracking, use_prefit, chunk_size=chunk_size, outcome_col=outcome_col, calculate_passes_json=False, add_confidence_intervals=False)

    use_parallel_processing = st.checkbox("Use parallel processing", value=True)

    optimization_target = st.selectbox("Select optimization target", [
        "logloss", "brier_score", "auc", "logloss_real", "brier_score_real", "auc_real", "logloss_synthetic", "brier_score_synthetic", "auc_synthetic",
    ])
    ret = simulate_parameters(df_training, dfs_tracking, True,
                              add_confidence_intervals=True, chunk_size=chunk_size, outcome_col=outcome_col,
                              calculate_passes_json=True)
    data = ret
    df_data = pd.Series(data).to_frame().T
    df_data["step_nr"] = 0
    front_cols = ["step_nr", "logloss", "brier_score", "auc", "brier_score_synthetic", "logloss_synthetic",
                  "auc_synthetic", "brier_score_real", "logloss_real", "auc_real"]
    cols = front_cols + [col for col in df_data.columns if col not in front_cols]
    df_data = df_data[cols]
    df = df_data
    st.write("df_prefit")
    st.write(df)

    st.write("display_df")
    display_df = st.empty()

    if not use_prefit:
        if use_parallel_processing:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                tasks = [executor.submit(simulate_params_partial) for _ in range(n_steps)]

                for i, future in enumerate(progress_bar(concurrent.futures.as_completed(tasks), total=n_steps, desc="MP Simulation")):
                    data = future.result()
                    df_data = pd.Series(data).to_frame().T
                    df_data["step_nr"] = i
                    front_cols = ["step_nr", "logloss", "brier_score", "auc", "brier_score_synthetic", "logloss_synthetic", "auc_synthetic", "brier_score_real", "logloss_real", "auc_real"]
                    cols = front_cols + [col for col in df_data.columns if col not in front_cols]
                    df_data = df_data[cols]

                    if df is None:
                        df = df_data
                    else:
                        df = pd.concat([df, df_data], axis=0)
                        df = df.sort_values(optimization_target, ascending=True).reset_index(drop=True)
                        # if len(df) > 20:
                        #     df.loc[20:, expensive_cols] = np.nan
                        if len(df) > 1:
                            df.loc[1:, expensive_cols] = np.nan

                    display_df.write(df.head(20))

                try:
                    del data
                    del df_data
                except Exception as e:
                    st.write(e)

                future = None
                del future
                gc.collect()

        else:
            for i in progress_bar(range(n_steps), desc="Simulation", total=n_steps):
                data = simulate_params_partial()
                df_data = pd.Series(data).to_frame().T
                df_data["step_nr"] = i
                front_cols = ["step_nr", "logloss", "brier_score", "auc", "brier_score_synthetic", "logloss_synthetic",
                              "auc_synthetic", "brier_score_real", "logloss_real", "auc_real"]
                cols = front_cols + [col for col in df_data.columns if col not in front_cols]
                df_data = df_data[cols]

                if df is None:
                    df = df_data
                else:
                    df = pd.concat([df, df_data], axis=0)
                    df = df.sort_values(optimization_target, ascending=True).reset_index(drop=True)
                    if len(df) > 1:
                        df.loc[1:, expensive_cols] = np.nan

                display_df.write(df.head(20))

            try:
                del data
                del df_data
            except Exception as e:
                st.write(e)

            future = None
            del future
            gc.collect()

    df_training_results = df.sort_values(optimization_target, ascending=True).reset_index(drop=True)

    training_data_simres = df_training_results["passes_json"][0]
    df_training_passes = pd.read_json(io.StringIO(training_data_simres)).copy()

    # move logloss column and brier and auc etc to front
    cols = df_training_results.columns.tolist()
    front_cols = ["step_nr", "logloss", "brier_score", "auc", "brier_score_synthetic", "logloss_synthetic", "auc_synthetic", "brier_score_real", "logloss_real", "auc_real"]
    df_training_results[front_cols] = df_training_results[front_cols].astype(float)
    cols = front_cols + [col for col in cols if col not in front_cols]
    df_training_results = df_training_results[cols]
    st.write("df_training_results")
    st.write(df_training_results)

    df_training_results.to_csv("df_training_results.csv", sep=";")
    st.write(f"Wrote results to {os.path.abspath('df_training_results.csv')}")

    best_index = df_training_results[optimization_target].idxmin()
    best_parameters = df_training_results["parameters"][best_index]
    best_passes = df_training_results["passes_json"][best_index]
    df_best_passes = pd.read_json(io.StringIO(best_passes)).copy()
    df_best_passes["error"] = (df_best_passes["success"] - df_best_passes["xc"]).abs()

    st.write("### Training results")

    @st.fragment
    def frag1():
        n_bins = st.number_input("Number of bins for calibration plot", value=10, min_value=1, max_value=None, key="frag1")
        add_text = st.checkbox("Add text to calibration plot", value=True, key="add_text1")
        style = st.selectbox("Style", plt.style.available, index=plt.style.available.index("seaborn-v0_8"), key="style1")
        # method = st.selectbox("CI-method", ["bootstrap", "wilson"], index=0, key="ci_method1")
        method = "bootstrap"
        st.write(bin_nr_calibration_plot(df_best_passes, outcome_col=outcome_col, n_bins=n_bins, add_text=add_text, style=style, method=method))

    @st.fragment
    def frag2():
        binsize = st.number_input("Binsize for calibration plot", value=0.1, min_value=0.01, max_value=None, key="frag2")
        add_text = st.checkbox("Add text to calibration plot", value=True, key="add_text2")
        style = st.selectbox("Style", plt.style.available, index=plt.style.available.index("seaborn-v0_8"), key="style2")
        # method = st.selectbox("CI-method", ["bootstrap", "wilson"], index=0, key="ci_method2")
        method = "bootstrap"
        plt.savefig(os.path.join(os.path.dirname(__file__), "frag2_training.pdf"))
        plt.savefig(os.path.join(os.path.dirname(__file__), "frag2_training.png"), dpi=300)
        st.write(bin_nr_calibration_plot(df_best_passes, outcome_col=outcome_col, binsize=binsize, add_text=add_text, style=style, method=method))

    @st.fragment
    def frag3():
        use_boken_axis = st.checkbox("Use broken axis", value=True, key="frag3")
        ylim1 = st.number_input("Y-axis max", value=450.0, min_value=0.0, max_value=None, key="frag3_1")
        ylim2 = st.number_input("Y-axis min 2", value=1550.0, min_value=0.0, max_value=None, key="frag3_2")
        ylim3 = st.number_input("Y-axis max 2", value=1600.0, min_value=0.0, max_value=None, key="frag3_3")
        st.write(calibration_histogram(df_best_passes, n_bins=40, use_boken_axis=use_boken_axis, ylim1=ylim1, ylim2=ylim2, ylim3=ylim3))
        st.write("Max xC", df_best_passes["xc"].max())
        plt.savefig(os.path.join(os.path.dirname(__file__), "frag3_training.png"), dpi=300)
        plt.savefig(os.path.join(os.path.dirname(__file__), "frag3_training.pdf"))

    frag1()
    frag2()
    frag3()

    # st.stop()

    if st.toggle("Show example passes", value=False):
        df_best_passes["ambiguity"] = (df_best_passes["xc"] - 0.5).abs()
        df_best_passes["closeness_to_55%"] = (df_best_passes["xc"] - 0.55).abs()
        df_best_passes["closeness_to_40%"] = (df_best_passes["xc"] - 0.4).abs()
        df_best_passes["closeness_to_72%"] = (df_best_passes["xc"] - 0.72).abs()
        for (text, df) in [
            ("Most ambiguous real predictions", df_best_passes[~df_best_passes["is_synthetic"]].sort_values("ambiguity", ascending=True)),
            ("Worst real unsuccessful pass predictions", df_best_passes[(~df_best_passes["success"]) & (~df_best_passes["is_synthetic"])].sort_values("error", ascending=False)),
            ("Random synthetic passes", df_best_passes[df_best_passes["is_synthetic"]].sample(frac=1).reset_index(drop=True)),
            ("Worst synthetic predictions", df_best_passes[df_best_passes["is_synthetic"]].sort_values("error", ascending=False)),
            ("Best synthetic predictions", df_best_passes[df_best_passes["is_synthetic"]].sort_values("error", ascending=True)),
            ("Worst real successful predictions", df_best_passes[df_best_passes["success"] & (~df_best_passes["is_synthetic"])].sort_values("error", ascending=False)),
            ("Best real predictions", df_best_passes[~df_best_passes["is_synthetic"]].sort_values("error", ascending=True)),
            ("Real predictions closest to 72%", df_best_passes[~df_best_passes["is_synthetic"]].sort_values("closeness_to_72%", ascending=True)),
            ("Real predictions closest to 40%", df_best_passes[~df_best_passes["is_synthetic"]].sort_values("closeness_to_40%", ascending=True)),
            ("Real predictions closest to 55%", df_best_passes[~df_best_passes["is_synthetic"]].sort_values("closeness_to_55%", ascending=True)),
        ]:
            with st.expander(text):
                for pass_nr, (_, p4ss) in enumerate(df.iterrows()):
                    st.write("#### Pass", pass_nr, "xc=", p4ss["xc"], "success=", p4ss["success"], "error=", p4ss["error"],
                             "is_synthetic=", p4ss["is_synthetic"])
                    for add_legend in [True]:#, False]:
                        plot_pass(p4ss, dfs_tracking[p4ss["dataset_nr"]], add_legend=add_legend, use_green_background=False)

                    if pass_nr > 20:
                        break

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
            tracking_period_col="period_id",

            tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy", tracking_v_col="v",
            n_frames_after_pass_for_v0=5, fallback_v0=10, chunk_size=chunk_size,

            use_event_coordinates_as_ball_position=True,

            **best_parameters,
        )
        data_simres["xc"].extend(ret.xc)
        data_simres["success"].extend(df_test_passes[outcome_col].tolist())
        data_simres["is_synthetic"].extend(df_test_passes["is_synthetic"].tolist())

    df_simres_test = pd.DataFrame(data_simres).copy()

    df_simres_total = pd.concat([df_training_passes, df_test_passes]).reset_index(drop=True)
    df_simres_total_only_success = df_simres_total[df_simres_total["success"]].copy()
    df_simres_total_only_fail = df_simres_total[~df_simres_total["success"]].copy()
    st.write("df_simres_total_only_success")
    st.write(df_simres_total_only_success)
    st.write("Average xC training + test", df_simres_total["xc"].mean())
    avg_xc_total_only_success = df_simres_total_only_success["xc"].mean()
    avg_xc_total_only_success_test = df_simres_test[df_simres_test["success"]]["xc"].mean()
    avg_xc_total_only_failure_test = df_simres_total_only_fail["xc"].mean()
    st.write("Average xC training + test only successful", avg_xc_total_only_success)
    st.write("Average xC test only successful", avg_xc_total_only_success_test)
    st.write("Average xC test only failure", avg_xc_total_only_failure_test)

    test_scores = get_scores(df_simres_test.copy(), df_test[outcome_col].mean(), outcome_col=outcome_col, add_confidence_intervals=True, n_bootstrap_samples=2000)
    st.write("### Test scores")
    df_test_scores = pd.DataFrame(test_scores, index=[0])

    # order cols like training
    df_test_scores = df_test_scores[[col for col in df_training_results.columns if col in df_test_scores.columns]]
    st.write("df_test_scores")
    st.write(df_test_scores.T)

    df_test_scores.to_csv("df_test_scores.csv", sep=";")
    st.write(f"Wrote test scores to {os.path.abspath('df_test_scores.csv')}")

    st.write("df_simres_test")
    st.write(df_simres_test)

    @st.fragment
    def frag1_test():
        n_bins = st.number_input("Number of bins for calibration plot", value=10, min_value=1, max_value=None, key="frag1_test")
        add_text = st.checkbox("Add text to calibration plot", value=True, key="add_text1_test")
        # method = st.selectbox("CI-method", ["bootstrap", "wilson"], index=0, key="ci_method4")
        method = "bootstrap"
        st.write(bin_nr_calibration_plot(df_simres_test, outcome_col=outcome_col, n_bins=n_bins, add_text=add_text, method=method))
        plt.savefig(os.path.join(os.path.dirname(__file__), "frag1_test.png"), dpi=300)
        plt.savefig(os.path.join(os.path.dirname(__file__), "frag1_test.pdf"))

    @st.fragment
    def frag2_test():
        binsize = st.number_input("Binsize for calibration plot", value=0.1, min_value=0.01, max_value=None, key="frag2_test")
        add_text = st.checkbox("Add text to calibration plot", value=True, key="add_text2_test")
        # method = st.selectbox("CI-method", ["bootstrap", "wilson"], index=0, key="ci_method5")
        method = "bootstrap"
        st.write(bin_nr_calibration_plot(df_simres_test, outcome_col=outcome_col, binsize=binsize, add_text=add_text, method=method))
        plt.savefig(os.path.join(os.path.dirname(__file__), "frag2_test.pdf"))
        plt.savefig(os.path.join(os.path.dirname(__file__), "frag2_test.png"), dpi=300)

    biggest_xc_in_test_set = df_simres_test["xc"].max()

    @st.fragment
    def frag3_test():
        use_boken_axis = st.checkbox("Use broken axis", value=True, key="frag4_test")
        ylim1 = st.number_input("Y-axis max", value=175.0, min_value=0.0, max_value=None, key="frag3_12")
        ylim2 = st.number_input("Y-axis min 2", value=675.0, min_value=0.0, max_value=None, key="frag3_22")
        ylim3 = st.number_input("Y-axis max 2", value=700.0, min_value=0.0, max_value=None, key="frag3_32")
        st.write(calibration_histogram(df_simres_test, n_bins=40, use_boken_axis=use_boken_axis, ylim1=ylim1, ylim2=ylim2, ylim3=ylim3))
        st.write("Max xC", biggest_xc_in_test_set)
        plt.savefig(os.path.join(os.path.dirname(__file__), "frag3_test.png"), dpi=300)
        plt.savefig(os.path.join(os.path.dirname(__file__), "frag3_test.pdf"))

    frag1_test()
    frag2_test()
    frag3_test()

    df_test_results = df_test_scores

    top_result = df_test_results.iloc[0]

    def _assert(a, b):
        if a != b:
            print(f"Assertion failed: {a} != {b}")
            st.warning(f"Assertion failed: {a} != {b}")
        # assert a == b

    if run_asserts:
        assert round(biggest_xc_in_test_set, 3) == 0.982
        assert round(avg_xc_total_only_success_test, 3) == 0.852

        # Validation results must equal the published results
        _assert(round(top_result["logloss"], 3), 0.246)
        _assert(round(top_result["logloss_real"], 3), 0.387),
        _assert(round(top_result["brier_score"], 3), 0.076),
        _assert(round(top_result["brier_score_real"], 3), 0.119),
        _assert(round(top_result["ece"], 3), 0.031),
        _assert(round(df_test_results["auc"].iloc[0], 3), 0.958),

        _assert(round(df_test_results["logloss_ci_lower"].iloc[0], 3), 0.220),
        _assert(round(df_test_results["logloss_ci_upper"].iloc[0], 3), 0.273),
        _assert(round(df_test_results["brier_ci_lower"].iloc[0], 3), 0.067),
        _assert(round(df_test_results["brier_ci_upper"].iloc[0], 3), 0.085),
        _assert(round(df_test_results["auc_ci_lower"].iloc[0], 3), 0.949),
        _assert(round(df_test_results["auc_ci_upper"].iloc[0], 3), 0.967),
        _assert(round(df_test_results["ece_ci_lower"].iloc[0], 3), 0.022),
        _assert(round(df_test_results["ece_ci_upper"].iloc[0], 3), 0.045),

        _assert(round(df_test_results["logloss_real"].iloc[0], 3), 0.387),
        _assert(round(df_test_results["brier_score_real"].iloc[0], 3), 0.119),
        _assert(round(df_test_results["auc_real"].iloc[0], 3), 0.832),
        _assert(round(df_test_results["logloss_ci_lower_real"].iloc[0], 3), 0.349),
        _assert(round(df_test_results["logloss_ci_upper_real"].iloc[0], 3), 0.428),
        _assert(round(df_test_results["brier_ci_lower_real"].iloc[0], 3), 0.106),
        _assert(round(df_test_results["brier_ci_upper_real"].iloc[0], 3), 0.133),
        _assert(round(df_test_results["auc_ci_lower_real"].iloc[0], 3), 0.800),
        _assert(round(df_test_results["auc_ci_upper_real"].iloc[0], 3), 0.862),

        _assert(round(df_test_results["baseline_logloss"].iloc[0], 3), 0.693),
        _assert(round(df_test_results["baseline_brier"].iloc[0], 3), 0.250),
        _assert(round(df_test_results["baseline_auc"].iloc[0], 3), 0.500),
        _assert(round(df_test_results["baseline_loglos_real"].iloc[0], 3), 0.494),
        _assert(round(df_test_results["baseline_brier_real"].iloc[0], 3), 0.157),
        _assert(round(df_test_results["baseline_auc_real"].iloc[0], 3), 0.500),

        st.markdown('<div id="done-flag">DONE</div>', unsafe_allow_html=True)

    return df_test_scores, biggest_xc_in_test_set, avg_xc_total_only_success_test, avg_xc_total_only_failure_test


def validation_dashboard(dummy=False, run_asserts=True):
    # suppress bs warnings
    warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)  # because this case is handled (just return nan)
    warnings.simplefilter(action="ignore", category=PerformanceWarning)  # because this code is not performance-critical
    warnings.simplefilter(action="ignore", category=UserWarning)
    warnings.simplefilter(action="ignore", category=PendingDeprecationWarning)
    warnings.simplefilter(action="ignore", category=FutureWarning)

    np.random.seed(SEED)
    random.seed(343431)

    do_das = st.toggle("Validate DAS", value=True)
    do_benchmark = st.toggle("Validate DAS against benchmark", value=True)

    ### DAS vs x_norm
    # for df_tracking, df_event in zip(dfs_tracking, dfs_event):
    #     das_vs_xnorm(df_tracking, df_event)
    #     break

    if do_benchmark:
        with st.expander("Benchmark"):
            mean_correct, mean_correct_not_only_differs_by_z = eval_benchmark()
            if run_asserts:
                assert mean_correct == 0.74
                assert round(mean_correct_not_only_differs_by_z, 3) == 0.804

    dfs_tracking, dfs_event = get_metrica_data(dummy=dummy)

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

        dfs_passes.append(df_passes)

        for _, p4ss in df_passes.iloc[:1].iterrows():
            plot_pass(p4ss, df_tracking)

    np.random.seed(SEED)
    random.seed(343431)

    if do_das:
        df_passes, target_density_success, target_density_fail = validate_das(dfs_tracking, dfs_passes)
        if run_asserts:
            assert round(target_density_fail, 3) == 0.346
            assert round(target_density_success, 3) == 0.843
            st.markdown('<div id="done-flag">DONE 1</div>', unsafe_allow_html=True)

    np.random.seed(SEED)
    random.seed(343431)

    n_steps = st.number_input("Number of simulations", value=25000)
    use_prefit = st.checkbox("Use prefit parameters", value=True)

    df_test_scores, biggest_xc_in_test_set, avg_xc_total_only_success_test, avg_xc_total_only_failure_test = validate_multiple_matches(
        dfs_tracking=dfs_tracking, dfs_passes=dfs_passes, outcome_col="success", n_steps=n_steps, use_prefit=use_prefit,
        run_asserts=run_asserts,
    )
    # {
    # "bit_generator":"PCG64"
    # "state":{
    # "state":2.8625763554376696e+38
    # "inc":6.034160481890525e+37
    # }
    # "has_uint32":1
    # "uinteger":1775787077
    # }
    # st.write("rng.bit_generator.state end")
    # st.write(rng.bit_generator.state)
    # print(rng.bit_generator.state)

    assert rng.bit_generator.state == {'bit_generator': 'PCG64', 'state': {'state': 286257635543766940493387507884471841288, 'inc': 60341604818905247986700519057288636087}, 'has_uint32': 1, 'uinteger': 1775787077}

    return df_test_scores, biggest_xc_in_test_set, avg_xc_total_only_success_test, avg_xc_total_only_failure_test, target_density_success, target_density_fail, mean_correct, mean_correct_not_only_differs_by_z


def validate_das(dfs_tracking, dfs_passes):
    @st.cache_resource
    def _get_das_of_dataset(dataset_nr):
        st.write("1")
        df_passes = dfs_passes[dataset_nr]
        df_tracking = dfs_tracking[dataset_nr]
        st.write("df_tracking")
        st.write(df_tracking.head())
        das_result = get_das_gained(df_passes, df_tracking, event_success_col="success",
                                    event_target_frame_col="end_frame_id", event_start_x_col="coordinates_x",
                                    event_start_y_col="coordinates_y", event_target_x_col="end_coordinates_x",
                                    event_target_y_col="end_coordinates_y", tracking_period_col="period_id",
                                    )
        return das_result

    dfs = []
    for dataset_nr in progress_bar([0, 1, 2], total=3):
        das_result = _get_das_of_dataset(dataset_nr)

        phi_grid = das_result.simulation_result.phi_grid[0]
        r_grid = das_result.simulation_result.r_grid

        dr = r_grid[1] - r_grid[0]

        df_passes = dfs_passes[dataset_nr]
        df_passes["frame_index"] = das_result.frame_index
        df_passes["DAS_Gained"] = das_result.das_gained

        target_densities = []
        for pass_index, p4ss in progress_bar(dfs_passes[dataset_nr].iterrows(), total=len(dfs_passes[dataset_nr])):
            p4ss["angle"] = (2*math.pi + math.atan2(p4ss["end_coordinates_y"] - p4ss["coordinates_y"], p4ss["end_coordinates_x"] - p4ss["coordinates_x"])) % (2*math.pi)
            p4ss["distance"] = np.sqrt((p4ss["end_coordinates_y"] - p4ss["coordinates_y"])**2 + (p4ss["end_coordinates_x"] - p4ss["coordinates_x"])**2)
            phi_index = np.abs(phi_grid - p4ss["angle"]).argmin()
            r_index = np.abs(r_grid - p4ss["distance"]).argmin()
            F = p4ss["frame_index"]
            target_density = das_result.simulation_result.attack_poss_density[F, phi_index, r_index] * dr
            target_densities.append(target_density)

        df_passes["target_density"] = target_densities
        df_passes["dataset_nr"] = dataset_nr
        dfs.append(df_passes)

    df_passes = pd.concat(dfs)

    # write target density mean
    st.write("Normalized AS density at target location of any passes")
    st.write(df_passes["target_density"].mean())
    st.write("Normalized AS density at target location of successful passes")
    target_density_success = df_passes.loc[df_passes["success"] == True, "target_density"].mean()
    st.write(target_density_success)
    dfg_mean = df_passes.loc[df_passes["success"] == True].groupby("dataset_nr")["target_density"].mean()
    st.write("... by data set", dfg_mean)
    st.write("Normalized AS density at target location of unsuccessful passes")
    target_density_fail = df_passes.loc[df_passes["success"] == False, "target_density"].mean()
    st.write(target_density_fail)

    # plot DAS Gained histogram
    st.write("DAS Gained histogram")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.histplot(df_passes[df_passes["DAS_Gained"].notna()], x="DAS_Gained", bins=60, kde=True)
    st.write(plt.gcf())
    plt.savefig(os.path.join(os.path.dirname(__file__), "das_gained_histogram.png"), dpi=300)
    plt.savefig(os.path.join(os.path.dirname(__file__), "das_gained_histogram.pdf"))
    # import plotly.express as px
    # fig = px.histogram(df_passes, x="DAS_Gained", color="dataset_nr", marginal="box", title="DAS Gained histogram")
    # fig.update_traces(opacity=0.75)
    # st.plotly_chart(fig)

    return df_passes, target_density_success, target_density_fail


def main(run_as_streamlit_app=True, dummy=False, run_asserts=True):
    if streamlit.runtime.exists() or not run_as_streamlit_app:
        return validation_dashboard(dummy=dummy, run_asserts=run_asserts)
    else:
        return subprocess.run(['streamlit', 'run', os.path.abspath(__file__), f"{dummy}", str(run_asserts)], check=True)

    # if run_as_streamlit_app:
    #     key_argument = "run_dashboard"
    #     if len(sys.argv) == 4 and sys.argv[1] == key_argument:
    #         return validation_dashboard(dummy=dummy, run_asserts=run_asserts)
    #     else:  # if script is called directly, call it again with streamlit
    #         return subprocess.run(['streamlit', 'run', os.path.abspath(__file__), key_argument, f"{dummy}", str(run_asserts)], check=True)
    # else:
    #     return validation_dashboard(dummy=dummy, run_asserts=run_asserts)


if __name__ == '__main__':
    if streamlit.runtime.exists():
        dummy = ast.literal_eval(sys.argv[1]) if len(sys.argv) > 1 else False
        run_asserts = ast.literal_eval(sys.argv[2]) if len(sys.argv) > 2 else True
        main(dummy=dummy, run_asserts=run_asserts)
    else:
        main()
