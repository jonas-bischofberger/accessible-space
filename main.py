import time

import matplotlib.pyplot as plt
import pandas as pd

import accessible_space
import numpy as np

import accessible_space.utility
import accessible_space.tests.qualitative_profiling

# import streamlit as st


def readme():
    import accessible_space
    from accessible_space.tests.resources import df_passes, df_tracking  # Example data
    import matplotlib.pyplot as plt

    ### 1. Add expected completion to passes
    pass_result = accessible_space.get_expected_pass_completion(df_passes, df_tracking, event_frame_col="frame_id", event_player_col="player_id", event_team_col="team_id", event_start_x_col="x", event_start_y_col="y", event_end_x_col="x_target", event_end_y_col="y_target", tracking_frame_col="frame_id", tracking_player_col="player_id", tracking_team_col="team_id", tracking_team_in_possession_col="team_in_possession", tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy", ball_tracking_player_id="ball")
    df_passes["xC"] = pass_result.xc  # Expected pass completion rate
    print(df_passes[["event_string", "xC"]])

    ### 2. Add DAS Gained to passes
    das_gained_result = accessible_space.get_das_gained(df_passes, df_tracking, event_frame_col="frame_id", event_success_col="pass_outcome", event_target_frame_col="target_frame_id", tracking_frame_col="frame_id", tracking_period_col="period_id", tracking_player_col="player_id", tracking_team_col="team_id", tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy", tracking_team_in_possession_col="team_in_possession", x_pitch_min=-52.5, x_pitch_max=52.5, y_pitch_min=-34, y_pitch_max=34)
    df_passes["DAS_Gained"] = das_gained_result.das_gained
    df_passes["AS_Gained"] = das_gained_result.as_gained
    print(df_passes[["event_string", "DAS_Gained", "AS_Gained"]])

    ### 3. Add Dangerous Accessible Space to tracking frames
    pitch_result = accessible_space.get_dangerous_accessible_space(df_tracking, frame_col="frame_id", period_col="period_id", player_col="player_id", team_col="team_id", x_col="x", y_col="y", vx_col="vx", vy_col="vy", team_in_possession_col="team_in_possession", x_pitch_min=-52.5, x_pitch_max=52.5, y_pitch_min=-34, y_pitch_max=34)
    df_tracking["AS"] = pitch_result.acc_space  # Accessible space
    df_tracking["DAS"] = pitch_result.das  # Dangerous accessible space
    print(df_tracking[["frame_id", "team_in_possession", "AS", "DAS"]].drop_duplicates())

    ### 4. Access raw simulation results
    # Example 4.1: Expected interception rate = last value of the cumulative interception probability of the defending team
    pass_result = accessible_space.get_expected_pass_completion(df_passes, df_tracking)
    pass_frame = 0  # We consider the pass at frame 0
    df_passes["frame_index"] = pass_result.event_frame_index  # frame_index implements a mapping from original frame number to indexes of the numpy arrays in the raw simulation_result.
    df_pass = df_passes[df_passes["frame_id"] == pass_frame]  # Consider the pass at frame 0
    frame_index = int(df_pass["frame_index"].iloc[0])
    expected_interception_rate = pass_result.simulation_result.defense_cum_prob[frame_index, 0, -1]  # Frame x Angle x Distance
    print(f"Expected interception rate: {expected_interception_rate:.1%}")

    # Example 4.2: Plot accessible space and dangerous accessible space
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

    plot_constellation(df_tracking_frame)
    accessible_space.plot_expected_completion_surface(pitch_result.simulation_result, frame_index=frame_index)
    plt.title(f"Accessible space: {df_tracking_frame['AS'].iloc[0]:.0f} m²")

    plot_constellation(df_tracking_frame)
    accessible_space.plot_expected_completion_surface(pitch_result.dangerous_result, frame_index=frame_index, color="red")
    plt.title(f"Dangerous accessible space: {df_tracking_frame['DAS'].iloc[0]:.2f} m²")
    plt.show()

    # Example 4.3: Get (dangerous) accessible space of individual players
    df_tracking["player_index"] = pitch_result.player_index  # Mapping from player to index in simulation_result
    areas = accessible_space.integrate_surfaces(pitch_result.simulation_result)  # Calculate surface integrals
    dangerous_areas = accessible_space.integrate_surfaces(pitch_result.dangerous_result)
    for _, row in df_tracking[(df_tracking["frame_id"] == 0) & (df_tracking["player_id"] != "ball")].iterrows():  # Consider frame 0
        is_attacker = row["team_id"] == row["team_in_possession"]
        acc_space = areas.player_poss[int(frame_index), int(row["player_index"])]
        das = dangerous_areas.player_poss[int(frame_index), int(row["player_index"])]

        plot_constellation(df_tracking_frame)
        accessible_space.plot_expected_completion_surface(pitch_result.simulation_result, "player_poss_density", frame_index=frame_index, player_index=int(row["player_index"]))
        accessible_space.plot_expected_completion_surface(pitch_result.dangerous_result, "player_poss_density", frame_index=frame_index, player_index=int(row["player_index"]), color="red")
        plt.title(f"{row['player_id']} ({'attacker' if is_attacker else 'defender'}) {acc_space:.0f}m² AS and {das:.2f} m² DAS.")
        plt.show()
        # Note: Individual space is not exclusive within a team. This is intentional because your team mates do not take away space from you in the competitive way that your opponents do.
        print(f"Player {row['player_id']} ({'attacker' if is_attacker else 'defender'}) controls {acc_space:.0f}m² AS and {das:.2f} m² DAS.")


if __name__ == '__main__':
    import importlib
    import accessible_space.tests.test_model_plausibility
    import accessible_space.interface
    import accessible_space.core
    import accessible_space.validation
    from accessible_space.utility import progress_bar

    importlib.reload(accessible_space.tests.test_model_plausibility)
    importlib.reload(accessible_space.interface)
    # importlib.reload(accessible_space.core)
    # importlib.reload(accessible_space.validation)

    accessible_space.tests.test_model_plausibility.test_no_poss_artifact_around_passer()
    # accessible_space.tests.test_model_plausibility.test_infer_playing_direction(_get_data=accessible_space.tests.test_model_plausibility._get_butterfly_data)

    # accessible_space.tests.test_model_plausibility.test_xc_parameters(accessible_space.tests.test_model_plausibility._get_butterfly_data, False, False, True, False)
    # accessible_space.tests.test_model_plausibility.test_bad_data_das(pd.DataFrame({"frame_id": [1, 2], "player_id": ["a", "b"], "team_id": ["H", "A"], "x": [0, 0], "y": [0, 0], "vx": [0, 0], "vy": [0, 0], "team_in_possession": ["H", "H"]}), ValueError, "Ball flag ball_tracking_player_id='ball' does not exist in column ")
    # accessible_space.tests.test_model_plausibility.test_real_world_data(1)
    # accessible_space.validation_dashboard()
    # accessible_space.tests.qualitative_profiling.profiling_dashboard()
    # accessible_space.tests.test_model_plausibility.test_as_symmetry(_get_data=accessible_space.tests.test_model_plausibility._get_butterfly_data)
    # readme()
