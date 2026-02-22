import pytest
import accessible_space
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xmltodict
import requests


def _get_metrica_data(
    dataset_nr=1, new_team_col="TEAM", new_player_col="PLAYER", new_x_col="X", new_y_col="Y",
    passer_team_col="passer_team", receiver_team_col="receiver_team",
):
    # @st.cache_resource
    def get_events(dataset_nr):
        metrica_open_data_base_dir = "https://raw.githubusercontent.com/metrica-sports/sample-data/refs/heads/master/data"
        if dataset_nr in [1, 2]:
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
                team_player_ids = set(df_players["player_id"].dropna().tolist() + df_players["receiver_player_id"].dropna().tolist())
                player_id_to_column_id.update({player_id: f"{team_id.lower().strip()}_{player_id.replace('Player', '').strip()}" for player_id in team_player_ids})
                column_id_to_team_id.update({player_id_to_column_id[player_id]: team_id for player_id in team_player_ids})


            df["PLAYER"] = df["player_id"].map(player_id_to_column_id)
            df["RECEIVER"] = df["receiver_player_id"].map(player_id_to_column_id)
            df["RECEIVING_TEAM"] = df["receiver_player_id"].map(column_id_to_team_id)

            df["tmp_next_player"] = df["player_id"].shift(-1)
            df["tmp_next_team"] = df["team_id"].shift(-1)
            df["tmp_receiver_player"] = df["receiver_player_id"].where(df["receiver_player_id"].notna(),
                                                                       df["tmp_next_player"])
            df["tmp_receiver_team"] = df["tmp_receiver_player"].map(column_id_to_team_id)

            df["is_successful"] = df["tmp_receiver_team"] == df["team_id"]

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

            meta_data = xmltodict.parse(
                requests.get(f"{metrica_open_data_base_dir}/Sample_Game_3/Sample_Game_3_metadata.xml").text)

            df_player = pd.json_normalize(meta_data, record_path=["main", "Metadata", "Players", "Player"])
            player2team = df_player[["@id", "@teamId"]].set_index("@id")["@teamId"].to_dict()
            df["team_id"] = df["player_id"].map(player2team)

            return df

    # @st.cache_resource
    def _get():
        #                             https://raw.githubusercontent.com/metrica-sports/sample-data/refs/heads/master/data/Sample_Game_1/Sample_Game_1_RawEventsData.csv
        metrica_open_data_base_dir = "https://raw.githubusercontent.com/metrica-sports/sample-data/refs/heads/master/data"
        # home_data=f"https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_{dataset_nr}/Sample_Game_{dataset_nr}_RawTrackingData_Home_Team.csv"
        home_data_url = f"{metrica_open_data_base_dir}/Sample_Game_{dataset_nr}/Sample_Game_{dataset_nr}_RawTrackingData_Home_Team.csv"
        # away_data=f"https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_{dataset_nr}/Sample_Game_{dataset_nr}_RawTrackingData_Away_Team.csv"
        away_data_url = f"{metrica_open_data_base_dir}/Sample_Game_{dataset_nr}/Sample_Game_{dataset_nr}_RawTrackingData_Away_Team.csv"
        event_url = f"{metrica_open_data_base_dir}/Sample_Game_{dataset_nr}/Sample_Game_{dataset_nr}_RawEventsData.csv"
        # with st.spinner(f"Loading data 1/3 from {event_url}"):
        df_events = pd.read_csv(event_url)
        # with st.spinner(f"Loading data 2/3 from {home_data_url}"):
        df_home = pd.read_csv(home_data_url, skiprows=2)
        # with st.spinner(f"Loading data 3/3 from {away_data_url}"):
        df_away = pd.read_csv(away_data_url, skiprows=2)
        return df_home, df_away, df_events

    df_home, df_away, df_events = _get()
    df_events = get_events(dataset_nr)
    df_passes = df_events[df_events["is_pass"]].copy()

    # st.write("df_events")
    # st.write(df_events)

    df_tracking = []
    for team, df_team in [("Home", df_home), ("Away", df_away)]:
        x_cols = [col for col in df_team.columns if col.startswith("Player") or col.startswith("Ball")]
        y_cols = [col for col in df_team.columns if col.startswith("Unnamed")]
        player_cols = x_cols

        df_tracking.append(accessible_space.per_object_frameify_tracking_data(
            df_team,
            frame_col="Frame",
            coordinate_cols=[[x, y] for x, y in zip(x_cols, y_cols)],
            players=player_cols,
            player_to_team={player: team for player in player_cols},
            new_coordinate_cols=[new_x_col, new_y_col],
            new_team_col=new_team_col,
            new_player_col=new_player_col,
        ))
    df_tracking = pd.concat(df_tracking)
    df_tracking = df_tracking.drop_duplicates(subset=["Frame", new_player_col], keep="first")
    i_ball = df_tracking[new_player_col] == "Ball"
    df_tracking.loc[i_ball, new_team_col] = None

    df_tracking["vx"] = 0  # df["x"].diff() / 25  # v not present -> allow infer?
    df_tracking["vy"] = 0  # df["y"].diff() / 25  # TODO check v correctness in validation

    # i_pass = (df_events["Type"].isin(["PASS", "BALL LOST", "BALL OUT"])) \
    #          & (~df_events["Subtype"].isin(["CLEARANCE", "HEAD-CLEARANCE", "HEAD-INTERCEPTION-CLEARANCE"]))

    # df_passes = df_events[i_pass]
    player2team = df_tracking[[new_player_col, new_team_col]].set_index(new_player_col)[new_team_col].to_dict()
    df_passes[passer_team_col] = df_passes["player_id"].map(player2team)
    df_passes[receiver_team_col] = df_passes["receiver_player_id"].map(player2team)

    # for x_col in ["coordinates_x"]:
    #     df_passes[x_col] = (df_passes[x_col] - 0.5) * 105
    # for y_col in ["coordinates_y"]:
    #     df_passes[y_col] = (df_passes[y_col] - 0.5) * 68
    for x_col in [new_x_col]:
        df_tracking[x_col] = (df_tracking[x_col] - 0.5) * 105
    for y_col in [new_y_col]:
        df_tracking[y_col] = (df_tracking[y_col] - 0.5) * 68 * -1

    df_passes["is_successful"] = df_passes[passer_team_col] == df_passes[receiver_team_col]
    # st.write("df_passes")
    # st.write(df_passes)
    # st.write("Success rate", df_passes["is_successful"].mean())
    assert df_passes["is_successful"].mean() < 1.0

    return df_passes, df_tracking


@pytest.mark.slow
@pytest.mark.parametrize("dataset_nr", [1, 2])
def test_real_world_data(dataset_nr):
    df_passes, df_tracking = _get_metrica_data(dataset_nr)

# KeyError: "Missing columns in df_passes: event_frame_col='Start Frame', event_start_x_col='Start X', event_start_y_col='Start Y', event_end_x_col='End X', event_end_y_col='End Y', event_team_col='Team', event_player_col='From'."

    ret = accessible_space.get_expected_pass_completion(
        df_passes=df_passes, df_tracking=df_tracking, tracking_frame_col="Frame", event_frame_col="frame_id",
        event_start_x_col="coordinates_x", event_end_x_col="end_coordinates_x", event_start_y_col="coordinates_y", event_end_y_col="end_coordinates_y",
        event_player_col="player_id", use_event_coordinates_as_ball_position=True, ball_tracking_player_id="Ball",
        event_team_col="passer_team", tracking_x_col="X", tracking_y_col="Y", tracking_team_col="TEAM",
        tracking_player_col="PLAYER",
    )
    df_passes["xc"] = ret.xc

    df_tracking_passes = df_tracking[df_tracking["Frame"].isin(df_passes["frame_id"].unique())]
    df_tracking_passes = df_tracking_passes.merge(df_passes[["frame_id", "passer_team"]], right_on="frame_id", left_on="Frame")
    ret_das = accessible_space.interface.get_individual_dangerous_accessible_space(
        df_tracking_passes, frame_col="Frame", x_col="X", y_col="Y", ball_player_id="Ball",
        return_cropped_result=True, player_col="PLAYER", team_col="TEAM", team_in_possession_col="passer_team",
        period_col="Period",
    )
    df_tracking_passes["AS"] = ret_das.acc_space
    df_tracking_passes["DAS"] = ret_das.das
    df_tracking_passes["frame_index"] = ret_das.frame_index
    df_tracking_passes["AS_player"] = ret_das.player_acc_space
    df_tracking_passes["DAS_player"] = ret_das.player_das

    for team, df_tracking_passes_team in df_tracking_passes.groupby("passer_team"):
        df_tracking_passes_team = df_tracking_passes_team[df_tracking_passes_team["TEAM"] == team]
        assert df_tracking_passes_team.groupby("Frame").agg({"DAS": "nunique"})["DAS"].max() == 1
        assert df_tracking_passes_team.groupby("Frame").agg({"AS": "nunique"})["AS"].max() == 1

        ### TODO fix this test
        # dfg = df_tracking_passes_team.groupby("Frame").agg({"DAS": "first", "AS": "first", "DAS_player": "sum", "AS_player": "sum"})
        # st.write("dfg")
        # st.write(dfg)
        # st.write(dfg[["DAS", "DAS_player"]])
        # i_path = dfg["DAS"] > dfg["DAS_player"]
        # st.write("i_path")
        # st.write(i_path)
        # st.write(dfg[i_path])
        # assert (dfg["DAS"] <= dfg["DAS_player"]).all()  # DAS/AS is non-exclusive within each team
        # assert (dfg["AS"] <= dfg["AS_player"]).all()

    df_passes["DAS_from_das"] = df_passes["frame_id"].map(df_tracking_passes.set_index("frame_id")["DAS"].to_dict())
    df_passes["AS_from_das"] = df_passes["frame_id"].map(df_tracking_passes.set_index("frame_id")["AS"].to_dict())
    df_passes["frame_index"] = df_passes["frame_id"].map(df_tracking_passes.set_index("frame_id")["frame_index"].to_dict())
    # st.write("df_passes")
    # st.write(df_passes)

    # st.write("Plotting xC")
    for pass_nr, (pass_index, p4ss) in enumerate(df_passes.iloc[:30].iterrows()):
        plt.figure()
        plt.arrow(p4ss["coordinates_x"], p4ss["coordinates_y"], p4ss["end_coordinates_x"]-p4ss["coordinates_x"], p4ss["end_coordinates_y"]-p4ss["coordinates_y"], color="black", head_width=2, head_length=3)
        df_tracking_frame = df_tracking[df_tracking["Frame"] == p4ss["frame_id"]]
        for team, df_tracking_frame_team in df_tracking_frame.groupby("TEAM"):
            team2color = {"Home": "red", "Away": "blue"}
            plt.scatter(df_tracking_frame_team["X"], df_tracking_frame_team["Y"], color=team2color[team])

        # df_tracking_frame["team_in_possession"] = p4ss["passer_team"]
        # ret = accessible_space.get_dangerous_accessible_space(
        #     df_tracking_frame, frame_col="Frame", x_col="X", y_col="Y", ball_player_id="Ball",
        #     return_cropped_result=True, player_col="PLAYER", team_col="TEAM",
        #
        #     **parameters,
        # )
        plt.xlim([-52.5, 52.5])
        plt.ylim([-34, 34])

        plt.title(f"Frame: {p4ss['frame_id']}, xC: {p4ss['xc']:1f}, Success: {p4ss['is_successful']}, DAS: {p4ss['DAS_from_das']}, AS: {p4ss['AS_from_das']}")

        accessible_space.plot_expected_completion_surface(ret_das.simulation_result, p4ss["frame_index"])
        # st.write(plt.gcf())
        # df_passes.loc[pass_index, "DAS_from_das"] = ret.das.iloc[0]
        # df_passes.loc[pass_index, "AS_from_das"] = ret.acc_space.iloc[0]

        plt.close()

    # average_xc = df_passes["xc"].mean()
    # average_success = df_passes["success"].mean()
    # st.write("xc", average_xc, "success", average_success)
    # assert average_xc > 0.5
    # assert average_success < 1.0

    def calculate_log_loss(y_true, y_pred):
        epsilon = 1e-15  # Avoid log(0) errors
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip probabilities to avoid extreme values
        log_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return log_loss

    # logloss = sklearn.metrics.log_loss(df_passes["is_successful"], df_passes["xc"], labels=np.array([0, 1]))
    logloss = calculate_log_loss(df_passes["is_successful"], df_passes["xc"])
    # st.write("logloss", logloss)
    # brier = sklearn.metrics.brier_score_loss(df_passes["is_successful"], df_passes["xc"])
    brier = np.mean((df_passes["is_successful"] - df_passes["xc"]) ** 2)
    # st.write("brier", brier)

    assert logloss < 0.6
    assert brier < 0.2

    # @st.cache_resource
    def _get2():
        ret2 = accessible_space.get_das_gained(
            df_passes, df_tracking,
            tracking_frame_col="Frame",
            tracking_x_col="X",
            tracking_y_col="Y",
            tracking_team_col="TEAM",
            tracking_player_col="PLAYER",
            tracking_period_col="Period",
            ball_tracking_player_id="Ball",
            event_frame_col="frame_id",
            event_target_frame_col="end_frame_id",
            event_success_col="is_successful",
            event_team_col="passer_team",
            event_receiver_team_col="receiver_team",
            # event_start_x_col="Start X",
            # event_start_y_col="Start Y",
            # event_target_x_col="End X",
            # event_target_y_col="End Y",
            use_event_coordinates_as_ball_position=False,
            use_event_team_as_team_in_possession=True,
            use_progress_bar=True,
        )
        return ret2

    ret2 = _get2()

    df_passes["DAS_gained"] = ret2.das_gained
    df_passes["AS_gained"] = ret2.as_gained
    df_passes["AS_from_gained"] = ret2.acc_space
    df_passes["AS_reception"] = ret2.acc_space_reception
    df_passes["DAS_from_gained"] = ret2.das
    df_passes["DAS_reception"] = ret2.das_reception
    df_passes["frame_index"] = ret2.frame_index
    # st.write("df_passes")
    # st.write(df_passes)

    # sort by das gained
    df_best_passes = df_passes.sort_values("DAS_gained", ascending=False)
    df_worst_passes = df_passes.sort_values("DAS_gained", ascending=True)

    # plot most dangerous passes
    for (df, string) in [
        (df_best_passes, "best"),
        (df_worst_passes, "worst"),
    ]:
        # st.write(f"### {string} passes")
        for _, p4ss in df.iloc[:30].iterrows():
            plt.figure()
            plt.arrow(p4ss["coordinates_x"], p4ss["coordinates_y"], p4ss["end_coordinates_x"]-p4ss["coordinates_x"], p4ss["end_coordinates_y"]-p4ss["coordinates_y"], color="black", head_width=2, head_length=3)
            df_tracking_frame = df_tracking[df_tracking["Frame"] == p4ss["frame_id"]]
            for team, df_tracking_frame_team in df_tracking_frame.groupby("TEAM"):
                team2color = {"Home": "red", "Away": "blue"}
                plt.scatter(df_tracking_frame_team["X"], df_tracking_frame_team["Y"], color=team2color[team])

            plt.xlim([-52.5, 52.5])
            plt.ylim([-34, 34])
            plt.title(f"Frame: {p4ss['frame_id']}, xC: {p4ss['xc']:1f}, Success: {p4ss['is_successful']}, DAS gained: {p4ss['DAS_gained']}, AS gained: {p4ss['AS_gained']}")

            accessible_space.plot_expected_completion_surface(ret2.simulation_result, p4ss["frame_index"])

            # st.write(plt.gcf())
            plt.close()

    df_passes["diff1"] = df_passes["DAS_from_das"] - df_passes["DAS_from_gained"]
    df_passes["diff2"] = df_passes["AS_from_das"] - df_passes["AS_from_gained"]

    # st.write(df_passes[['DAS_from_das', 'DAS_from_gained', 'AS_from_das', 'AS_from_gained', "diff1", "diff2"]])

    # check all close
    assert np.mean(np.isclose(df_passes["DAS_from_das"], df_passes["DAS_from_gained"])) >= 0.9  # DAS and DAS Gained sometimes don't match when team in possession is not clear for a frame.
    assert np.mean(np.isclose(df_passes["AS_from_das"], df_passes["AS_from_gained"])) >= 0.9
