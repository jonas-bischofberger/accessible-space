import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import accessible_space
import accessible_space.utility
import databallpy
# import databallpy.features
import databallpy.visualize


@st.cache_resource
def get_data():
    match = databallpy.get_open_match()
    return match


@st.cache_resource
def get_preprocessed_data():
    match = get_data()
    match.synchronise_tracking_and_event_data()
    databallpy.add_team_possession(match.tracking_data, match.event_data, match.home_team_id, inplace=True)

    databallpy.add_velocity(
        match.tracking_data, inplace=True, column_ids=match.home_players_column_ids() + match.away_players_column_ids(),
        frame_rate=match.frame_rate
    )

    return match


@st.cache_resource
def _get_preprocessed_data():
    match = get_preprocessed_data()

    players = match.home_players_column_ids() + match.away_players_column_ids() + ["ball"]
    frame_col = "frame"

    player_to_team = {}
    for player in players:
        if player in match.home_players_column_ids():
            player_to_team[player] = match.home_team_id
        elif player in match.away_players_column_ids():
            player_to_team[player] = match.away_team_id
        else:
            player_to_team[player] = None

    coordinate_columns = [[f"{player}_{coord}" for coord in ["x", "y", "vx", "vy", "velocity"]] for player in players]
    df_tracking = dangerous_accessible_space.per_object_frameify_tracking_data(match.tracking_data, frame_col, coordinate_columns, players, player_to_team, new_coordinate_cols=["x", "y", "vx", "vy", "v"])
    df_tracking["ball_possession"] = df_tracking["ball_possession"].map({"home": match.home_team_id, "away": match.away_team_id})

    df_events = match.event_data
    df_events["tracking_player_id"] = df_events["player_id"].map(match.player_id_to_column_id)

    return match, df_tracking, df_events


def das_vs_xnorm(df_tracking, df_event):
    df_tracking["attacking_direction"] = dangerous_accessible_space.infer_playing_direction(df_tracking)
    df_event["attacking_direction"] = df_event["frame_id"].map(df_tracking.set_index("frame_id")["attacking_direction"].to_dict())

    df_passes = df_event[(df_event["is_pass"]) & (~df_event["is_high"])]
    df_tracking = df_tracking[df_tracking["frame_id"].isin(df_passes["frame_id"])]
    df_tracking["AS"], df_tracking["DAS"], df_tracking["result_index"], _, _ = dangerous_accessible_space.get_dangerous_accessible_space(
        df_tracking, tracking_frame_col="frame_id", tracking_player_col="player_id", tracking_team_col="team_id",
    )
    df_passes["AS"] = df_passes["frame_id"].map(df_tracking.set_index("frame_id")["AS"].to_dict())
    df_passes["DAS"] = df_passes["frame_id"].map(df_tracking.set_index("frame_id")["DAS"].to_dict())
    df_passes["result_index"] = df_passes["frame_id"].map(df_tracking.set_index("frame_id")["result_index"].to_dict())

    # correlate x_norm and DAS
    df_passes["x_norm"] = df_passes["coordinates_x"] * df_passes["attacking_direction"]
    df_passes["y_norm"] = df_passes["coordinates_y"] * df_passes["attacking_direction"]
    corr = df_passes[["x_norm", "DAS"]].corr().iloc[0, 1]
    st.write("Correlation between x_norm and DAS", corr)

    # plot it
    fig, ax = plt.subplots()
    ax.scatter(df_passes["x_norm"], df_passes["DAS"])
    ax.set_xlabel("x_norm")
    ax.set_ylabel("DAS")
    st.pyplot(fig)


def demo_dashboard():
    match, df_tracking, df_event = _get_preprocessed_data()

    df_passes = df_event[df_event["databallpy_event"] == "pass"].reset_index()

    df_passes["xc"] = np.nan
    df_passes["AS"] = np.nan

    # df_passes["index"] = df_passes.index

    ### xC
    df_passes["xc"], _, _ = dangerous_accessible_space.get_expected_pass_completion(
        df_passes, df_tracking, event_frame_col="td_frame", tracking_frame_col="frame", event_start_x_col="start_x",
        event_start_y_col="start_y", event_end_x_col="end_x", event_end_y_col="end_y",
        event_player_col="tracking_player_id",
    )

    df_passes = df_passes.iloc[6:30]

    ### AS
    st.write("df_tracking")
    st.write(df_tracking.head(50))
    df_tracking = df_tracking[df_tracking["frame"].isin(df_passes["td_frame"])]
    df_tracking["AS"], df_tracking["DAS"], df_tracking["result_index"], simulation_result, dangerous_result = dangerous_accessible_space.get_dangerous_accessible_space(
        df_tracking, infer_attacking_direction=True, tracking_frame_col="frame", tracking_player_col="player_id",
        tracking_team_col="team_id",
    )
    df_passes["AS"] = df_passes["td_frame"].map(df_tracking.set_index("frame")["AS"].to_dict())
    df_passes["DAS"] = df_passes["td_frame"].map(df_tracking.set_index("frame")["DAS"].to_dict())
    df_passes["result_index"] = df_passes["td_frame"].map(df_tracking.set_index("frame")["result_index"].to_dict())

    # df_passes = df_passes.sort_values("xc", ascending=True)

    for i, (frame, row) in enumerate(df_passes.iterrows()):
        plt.figure()
        fig, ax = databallpy.visualize.plot_soccer_pitch(pitch_color="white")
        databallpy.visualize.plot_tracking_data(
            match,
            row["td_frame"],
            team_colors=["blue", "red"],
            title=f"Pass completion: {row['outcome']}",
            add_velocities=True,
            variable_of_interest=f"AS={row['AS']:.0f} m^2, xC={row['xc']:.1%}, DAS={row['DAS']:.2f} m^2",
            # variable_of_interest=f"AS={row['AS']:.0f} m^2",
            ax=ax,
        )
        team_color = "blue" if row["team_id"] == match.home_team_id else "red"
        def_team_color = "red" if row["team_id"] == match.home_team_id else "blue"
        plt.arrow(
            row["start_x"], row["start_y"], row["end_x"] - row["start_x"], row["end_y"] - row["start_y"], head_width=1,
            head_length=1, fc=team_color, ec=team_color
        )

        try:
            fig = dangerous_accessible_space.plot_expected_completion_surface(
                dangerous_result, row["result_index"], plot_type_off="poss",
                # plot_type_def="poss",
                color_off=team_color, color_def=def_team_color, plot_gridpoints=True,
            )
        except NameError as e:
            pass

        st.write(fig)
        plt.close(fig)

        if i > 30:
            break

    st.write(fig)

    # profiler.stop()


if __name__ == '__main__':
    demo_dashboard()
