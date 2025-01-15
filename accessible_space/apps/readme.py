import os
import subprocess
import sys

import matplotlib.pyplot as plt
import streamlit as st

import accessible_space
from accessible_space.tests.resources import df_passes, df_tracking  # Example data


def readme_dashboard():
    st.set_page_config(layout="wide")

    st.write("#### Prepare example data")
    st.code("""from accessible_space.tests.resources import df_passes, df_tracking  # Example data""", language="python")
    st.write("df_passes")
    st.write(df_passes)
    st.write("df_tracking")
    st.write(df_tracking)

    ### 1. Add expected completion to passes
    st.write("#### 1. Add expected completion to passes")
    st.code("""pass_result = accessible_space.get_expected_pass_completion(df_passes, df_tracking, event_frame_col="frame_id", event_player_col="player_id", event_team_col="team_id", event_start_x_col="x", event_start_y_col="y", event_end_x_col="x_target", event_end_y_col="y_target", tracking_frame_col="frame_id", tracking_player_col="player_id", tracking_team_col="team_id", tracking_team_in_possession_col="team_in_possession", tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy", ball_tracking_player_id="ball")""", language="python")
    pass_result = accessible_space.get_expected_pass_completion(df_passes, df_tracking, event_frame_col="frame_id", event_player_col="player_id", event_team_col="team_id", event_start_x_col="x", event_start_y_col="y", event_end_x_col="x_target", event_end_y_col="y_target", tracking_frame_col="frame_id", tracking_player_col="player_id", tracking_team_col="team_id", tracking_team_in_possession_col="team_in_possession", tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy", ball_tracking_player_id="ball")
    st.code("""df_passes["xC"] = pass_result.xc""", language="python")
    df_passes["xC"] = pass_result.xc  # Expected pass completion rate
    st.write(df_passes[["event_string", "xC"]])

    ### 2. Add DAS Gained to passes
    st.write("#### 2. Add DAS Gained to passes")
    st.code("""das_gained_result = accessible_space.get_das_gained(df_passes, df_tracking, event_frame_col="frame_id", event_success_col="pass_outcome", event_target_frame_col="target_frame_id", tracking_frame_col="frame_id", tracking_period_col="period_id", tracking_player_col="player_id", tracking_team_col="team_id", tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy", tracking_team_in_possession_col="team_in_possession", x_pitch_min=-52.5, x_pitch_max=52.5, y_pitch_min=-34, y_pitch_max=34)""", language="python")
    das_gained_result = accessible_space.get_das_gained(df_passes, df_tracking, event_frame_col="frame_id", event_success_col="pass_outcome", event_target_frame_col="target_frame_id", tracking_frame_col="frame_id", tracking_period_col="period_id", tracking_player_col="player_id", tracking_team_col="team_id", tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy", tracking_team_in_possession_col="team_in_possession", x_pitch_min=-52.5, x_pitch_max=52.5, y_pitch_min=-34, y_pitch_max=34)
    st.code("""df_passes["DAS_Gained"] = das_gained_result.das_gained
df_passes["AS_Gained"] = das_gained_result.as_gained""")
    df_passes["DAS_Gained"] = das_gained_result.das_gained
    df_passes["AS_Gained"] = das_gained_result.as_gained
    st.write(df_passes[["event_string", "DAS_Gained", "AS_Gained"]])

    ### 3. Add Dangerous Accessible Space to tracking frames
    st.write("#### 3. Add Dangerous Accessible Space to tracking frames")
    st.code("""pitch_result = accessible_space.get_dangerous_accessible_space(df_tracking, frame_col="frame_id", period_col="period_id", player_col="player_id", team_col="team_id", x_col="x", y_col="y", vx_col="vx", vy_col="vy", team_in_possession_col="team_in_possession", x_pitch_min=-52.5, x_pitch_max=52.5, y_pitch_min=-34, y_pitch_max=34)""", language="python")
    pitch_result = accessible_space.get_dangerous_accessible_space(df_tracking, frame_col="frame_id", period_col="period_id", player_col="player_id", team_col="team_id", x_col="x", y_col="y", vx_col="vx", vy_col="vy", team_in_possession_col="team_in_possession", x_pitch_min=-52.5, x_pitch_max=52.5, y_pitch_min=-34, y_pitch_max=34)
    st.code("""df_tracking["AS"] = pitch_result.acc_space
df_tracking["DAS"] = pitch_result.das""", language="python")
    df_tracking["AS"] = pitch_result.acc_space  # Accessible space
    df_tracking["DAS"] = pitch_result.das  # Dangerous accessible space
    st.write(df_tracking[["frame_id", "team_in_possession", "AS", "DAS"]].drop_duplicates())

    ### 4. Access raw simulation results
    # Example 4.1: Expected interception rate = last value of the cumulative interception probability of the defending team
    st.write("#### Example 4.1: Expected interception rate")
    pass_result = accessible_space.get_expected_pass_completion(df_passes, df_tracking, additional_fields_to_return=["defense_cum_prob"])
    pass_frame = 0  # We consider the pass at frame 0
    df_passes["frame_index"] = pass_result.event_frame_index  # frame_index implements a mapping from original frame number to indexes of the numpy arrays in the raw simulation_result.
    df_pass = df_passes[df_passes["frame_id"] == pass_frame]  # Consider the pass at frame 0
    frame_index = int(df_pass["frame_index"].iloc[0])
    expected_interception_rate = pass_result.simulation_result.defense_cum_prob[frame_index, 0, -1]  # Frame x Angle x Distance
    st.write(f"Expected interception rate: {expected_interception_rate:.1%}")

    # Example 4.2: Plot accessible space and dangerous accessible space
    st.write("#### Example 4.2: Plot accessible space and dangerous accessible space")
    df_tracking["frame_index"] = pitch_result.frame_index

    def plot_constellation(df_tracking_frame):
        plt.figure()
        plt.xlim([-52.5, 52.5])
        plt.ylim([-34, 34])
        plt.scatter(df_tracking_frame["x"], df_tracking_frame["y"], c=df_tracking_frame["team_id"].map({"Home": "red", "Away": "blue"}).fillna("black"), marker="o")
        for _, row in df_tracking_frame.iterrows():
            plt.text(row["x"], row["y"], row["player_id"] if row["player_id"] != "ball" else "")
        plt.gca().set_aspect('equal', adjustable='box')

    df_tracking_frame = df_tracking[df_tracking["frame_id"] == 0]  # Plot frame 0
    frame_index = df_tracking_frame["frame_index"].iloc[0]

    columns = st.columns(2)
    plot_constellation(df_tracking_frame)
    accessible_space.plot_expected_completion_surface(pitch_result.simulation_result, frame_index=frame_index)
    plt.title(f"Accessible space: {df_tracking_frame['AS'].iloc[0]:.0f} m²")
    with columns[0]:
        st.write(plt.gcf())

    plot_constellation(df_tracking_frame)
    accessible_space.plot_expected_completion_surface(pitch_result.dangerous_result, frame_index=frame_index, color="red")
    plt.title(f"Dangerous accessible space: {df_tracking_frame['DAS'].iloc[0]:.2f} m²")
    with columns[1]:
        st.write(plt.gcf())

    # Example 4.3: Get (dangerous) accessible space of individual players
    st.write("#### Example 4.3: Get (dangerous) accessible space of individual players")
    df_tracking["player_index"] = pitch_result.player_index  # Mapping from player to index in simulation_result
    pitch_result = accessible_space.get_dangerous_accessible_space(df_tracking, additional_fields_to_return=["player_poss_density"], period_col="period_id")
    areas = accessible_space.integrate_surfaces(pitch_result.simulation_result)  # Calculate surface integrals
    dangerous_areas = accessible_space.integrate_surfaces(pitch_result.dangerous_result)
    columns = st.columns(2)
    for row_nr, (_, row) in enumerate(df_tracking[(df_tracking["frame_id"] == 0) & (df_tracking["player_id"] != "ball")].iterrows()):  # Consider frame 0
        is_attacker = row["team_id"] == row["team_in_possession"]
        acc_space = areas.player_poss[int(frame_index), int(row["player_index"])]
        das = dangerous_areas.player_poss[int(frame_index), int(row["player_index"])]

        plot_constellation(df_tracking_frame)
        accessible_space.plot_expected_completion_surface(pitch_result.simulation_result, frame_index, "player_poss_density", player_index=int(row["player_index"]))
        accessible_space.plot_expected_completion_surface(pitch_result.dangerous_result, frame_index, "player_poss_density", player_index=int(row["player_index"]), color="red")
        plt.title(f"{row['player_id']} ({'attacker' if is_attacker else 'defender'}) {acc_space:.0f}m² AS and {das:.2f} m² DAS.")
        with columns[row_nr % 2]:
            st.write(plt.gcf())
        # Note: Individual space is not exclusive within a team. This is intentional because your team mates do not take away space from you in the competitive way that your opponents do.


def main(run_as_streamlit_app=True):
    if run_as_streamlit_app:
        if len(sys.argv) == 2 and sys.argv[1] == "run_dashboard":
            readme_dashboard()
        else:  # if script is called directly, call it again with streamlit
            subprocess.run(['streamlit', 'run', os.path.abspath(__file__), "run_dashboard"], check=True)
    else:
        readme_dashboard()


if __name__ == "__main__":
    main()
