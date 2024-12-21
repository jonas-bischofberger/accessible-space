import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import accessible_space

try:
    import streamlit as st
except ImportError:
    pass


def _get_butterfly_data():
    df_tracking = pd.DataFrame({
        "frame_id": [0, 0, 0, 0],
        "player_id": ["a", "b", "x", "ball"],
        "x": [0, -50, 50, 0],
        "y": [0, 0, 0, 0],
        "vx": [0, 0, 0, 15],
        "vy": [0, 0, 0, 0],
        "team_id": ["H", "H", "A", None],
        "player_color": ["blue", "blue", "red", "black"],
        "team_in_possession": ["H"] * 4,
        "player_in_possession": ["a"] * 4,
        "attacking_direction": [1] * 4,
        # "frame_id": [0, 0, 0, 0, 1],
        # "player_id": ["a", "b", "x", "ball", "ball"],
        # "x": [0, -50, 50, 0, 1],
        # "y": [0, 0, 0, 0, 0],
        # "vx": [0, 0, 0, 15, 15],
        # "vy": [0, 0, 0, 0, 0],
        # "team_id": ["H", "H", "A", None, None],
        # "player_color": ["blue", "blue", "red", "black", "black"],
        # "team_in_possession": ["H"] * 5,
        # "player_in_possession": ["a"] * 5,
        # "attacking_direction": [1] * 5,
    })

    ### Plotting
    # plt.scatter(df_tracking["x"], df_tracking["y"], color=df_tracking["player_color"])
    # plt.show()

    df_pass_safe = pd.DataFrame({
        "frame_id": [0],
        "player_id": ["a"],
        "team_id": ["H"],
        "x": [0],
        "y": [0],
        "x_target": [-50],
        "y_target": [0],
        "v0": [15],
    })
    df_pass_risky = df_pass_safe.copy()
    df_pass_risky["x_target"] = 50

    return df_pass_safe, df_pass_risky, df_tracking

def _get_butterfly_data_with_nans():
    df_tracking = pd.DataFrame({
        "frame_id": [0, 0, 0, 0, 0, 0],
        "player_id": ["a", "b", "x", "ball", "c", "d"],
        "x": [0, -50, 50, 0, np.nan, np.nan],
        "y": [0, 0, 0, 0, np.nan, np.nan],
        "vx": [0, 0, 0, 15, np.nan, np.nan],
        "vy": [0, 0, 0, 0, np.nan, np.nan],
        "team_id": ["H", "H", "A", None, "H", "A"],
        "player_color": ["blue", "blue", "red", "black", "white", "white"],
        "team_in_possession": ["H"] * 6,
        "player_in_possession": ["a"] * 6,
        "attacking_direction": [1] * 6,
    })

    ### Plotting
    # plt.scatter(df_tracking["x"], df_tracking["y"], color=df_tracking["player_color"])
    # plt.show()

    df_pass_safe = pd.DataFrame({
        "frame_id": [0],
        "player_id": ["a"],
        "team_id": ["H"],
        "x": [0],
        "y": [0],
        "x_target": [-50],
        "y_target": [0],
        "v0": [15],
    })
    df_pass_risky = df_pass_safe.copy()
    df_pass_risky["x_target"] = 50

    return df_pass_safe, df_pass_risky, df_tracking


def _get_double_butterfly_data():
    df_tracking = pd.DataFrame({
        "frame_id": [0, 0, 0, 0, 0, 0],
        "player_id": ["a", "b", "c", "x", "y", "ball"],
        "team_id": ["H", "H", "H", "A", "A", None],
        "x": [0, -50, -10, 50, 10, 0],
        "y": [0, 0, 0, 0, 0, 0],
        "vx": [0, 0, 0, 0, 0, 0],
        "vy": [0, 0, 0, 0, 0, 0],
        "player_color": ["blue", "blue", "blue", "red", "red", "black"],
        "team_in_possession": ["H"] * 6,
        "player_in_possession": ["a"] * 6,
    })

    ### Plotting
    # plt.scatter(df_tracking["x"], df_tracking["y"], color=df_tracking["player_color"])
    # st.write(plt.gcf())
    # plt.show()

    return df_tracking


def test_no_poss_artifact_around_passer():
    _, _, df_tracking = _get_butterfly_data()

    # Using the default parameters should not lead to inaccessible artifact around passer
    ret = accessible_space.get_dangerous_accessible_space(df_tracking)

    ### Plotting
    # fig = accessible_space.plot_expected_completion_surface(ret.simulation_result, 0, "attack_poss_density", color="blue")
    # plt.xlim([-52.5, 52.5])
    # plt.ylim([-34, 34])
    # st.write(fig)
    assert (ret.simulation_result.attack_cum_poss[0, :, 0] > 0.5).any()


@pytest.mark.parametrize("_get_data", [_get_butterfly_data, _get_butterfly_data_with_nans])
def test_as_symmetry(_get_data):
    _, _, df_tracking = _get_data()

    reaches_half_space = False
    for angles, v0_min, n_v0, gridsize in [
        (64, 0.01, 250, 1),
        (32, 0.01, 250, 1),
        (48, 0.01, 250, 1),
        (12, 0.01, 250, 1),
        (24, 0.01, 250, 1),
        (24, 0.001, 250, 1),
        (24, 0.0001, 250, 1),
        (24, 0.1, 250, 1),
        (24, 1, 250, 1),
    ]:
        ret = accessible_space.get_dangerous_accessible_space(
            df_tracking, passer_to_exclude_col="player_in_possession", n_angles=angles, v0_min=v0_min,
            n_v0=n_v0, radial_gridsize=gridsize
        )
        if np.isclose(ret.acc_space.iloc[0], 3570, atol=10):  # allow 10m² error here, 3570 = 105*68/2. This is very sensitive to the parameters though - we merely ensure that it's POSSIBLE to achieve this value.
            reaches_half_space = True
    assert reaches_half_space

    accessible_space.plot_expected_completion_surface(ret.simulation_result, 0, "attack_poss_density", color="blue")

    ### Plotting
    # plt.xlim([-52.5, 52.5])
    # plt.ylim([-34, 34])
    # st.write(plt.gcf())
    # st.write("ret")
    # st.write("ret.acc_space", ret.acc_space)
    # st.write("ret.simulation_result.attack_poss_density[0]", ret.simulation_result.attack_poss_density[0])


@pytest.mark.parametrize("_get_data", [_get_butterfly_data, _get_butterfly_data_with_nans])
def test_xc_symmetry(_get_data):
    df_pass_safe, df_pass_risky, df_tracking = _get_data()

    ret_safe = accessible_space.get_expected_pass_completion(df_pass_safe, df_tracking)
    assert ret_safe.xc > 0.95
    ret_risky = accessible_space.get_expected_pass_completion(df_pass_risky, df_tracking)
    assert ret_risky.xc < 0.05

    assert np.isclose(ret_safe.xc, 1 - ret_risky.xc, atol=1e-3)


@pytest.mark.parametrize("_get_data", [_get_butterfly_data, _get_butterfly_data_with_nans])
@pytest.mark.parametrize("use_approx_two_point", [False, True])
@pytest.mark.parametrize("keep_inertial_velocity", [False, True])
@pytest.mark.parametrize("use_event_coordinates_as_ball_position", [False, True])
@pytest.mark.parametrize("use_fixed_v0", [False, True])
def test_xc_parameters(_get_data, use_approx_two_point, keep_inertial_velocity, use_event_coordinates_as_ball_position, use_fixed_v0):
    df_pass_safe, df_pass_risky, df_tracking = _get_data()

    ret_safe = accessible_space.get_expected_pass_completion(df_pass_safe, df_tracking, use_approx_two_point=use_approx_two_point, keep_inertial_velocity=keep_inertial_velocity, use_event_coordinates_as_ball_position=use_event_coordinates_as_ball_position, use_fixed_v0=use_fixed_v0, clip_to_pitch=False)
    assert ret_safe.xc[0] > 0.95

    df_tracking["vx"] = -df_tracking["vx"]
    df_tracking["x"] = -df_tracking["x"]
    ret_risky = accessible_space.get_expected_pass_completion(df_pass_safe, df_tracking, use_approx_two_point=use_approx_two_point, keep_inertial_velocity=keep_inertial_velocity, use_event_coordinates_as_ball_position=use_event_coordinates_as_ball_position, use_fixed_v0=use_fixed_v0, clip_to_pitch=False)
    df_tracking["vx"] = -df_tracking["vx"]
    df_tracking["x"] = -df_tracking["x"]

    assert ret_risky.xc[0] < 0.05
    assert np.isclose(ret_safe.xc[0], 1 - ret_risky.xc[0], atol=1e-3)


@pytest.mark.parametrize("_get_data", [_get_butterfly_data, _get_butterfly_data_with_nans])
def test_coordinate_systems(_get_data):
    _, _, df_tracking = _get_data()
    for x_min, x_max, y_min, y_max in [
        (-52.5, 52.5, -34, 34),
        (0, 105, 0, 68),
        (-1, 1, -1, 1),
        (-2500, 2000, -2500, 1600),
        (-200, -20, -200, -20),
    ]:
        ret = accessible_space.get_dangerous_accessible_space(
            df_tracking, x_pitch_min=x_min, x_pitch_max=x_max, y_pitch_min=y_min, y_pitch_max=y_max, radial_gridsize=np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2) / 50
        )
        assert np.all(np.any(ret.simulation_result.x_grid <= x_min, axis=(1, 2)))
        assert np.all(np.any(ret.simulation_result.x_grid >= x_max, axis=(1, 2)))
        assert np.all(np.any(ret.simulation_result.y_grid <= y_min, axis=(1, 2)))
        assert np.all(np.any(ret.simulation_result.y_grid >= y_max, axis=(1, 2)))

        ### Plotting
        # plt.figure()
        # plt.xlim([x_min, x_max])
        # plt.ylim([y_min, y_max])
        # accessible_space.plot_expected_completion_surface(ret.simulation_result, 0, "attack_poss_density", color="blue")
        # accessible_space.plot_expected_completion_surface(ret.dangerous_result, 0, "attack_poss_density", color="red")
        # plt.title(f"Accessible space: {ret.acc_space.iloc[0]:.0f} m², DAS: {ret.das.iloc[0]:.2f} m²")
        # st.write(plt.gcf())


def test_das_gained():
    from .resources import df_passes, df_tracking

    ret_das_gained = accessible_space.get_das_gained(df_passes, df_tracking, use_event_coordinates_as_ball_position=False, use_event_team_as_team_in_possession=False)
    df_passes["DAS_gained"] = ret_das_gained.das_gained
    df_passes["AS_gained"] = ret_das_gained.as_gained
    df_passes["AS"] = ret_das_gained.acc_space
    df_passes["DAS"] = ret_das_gained.das
    df_passes["AS_reception"] = ret_das_gained.acc_space_reception
    df_passes["DAS_reception"] = ret_das_gained.das_reception
    df_passes["frame_index"] = ret_das_gained.frame_index
    df_passes["target_frame_index"] = ret_das_gained.target_frame_index

    assert df_passes.apply(lambda row: row["frame_index"] != row["target_frame_index"], axis=1).all()

    i_unsuccessful = df_passes["pass_outcome"] == 0
    i_successful = df_passes["pass_outcome"] == 1

    assert (df_passes.loc[i_unsuccessful, "DAS_gained"] < 0).all()
    assert (df_passes.loc[i_unsuccessful, "AS_gained"] < 0).all()

    assert (df_passes.loc[i_successful, "DAS_gained"] == df_passes.loc[i_successful, "DAS_reception"] - df_passes.loc[i_successful, "DAS"]).all()

    ret_das = accessible_space.get_dangerous_accessible_space(df_tracking)
    df_tracking["DAS"] = ret_das.das
    df_tracking["AS"] = ret_das.acc_space

    for _, p4ss in df_passes.iterrows():
        df_tracking_frame = df_tracking[df_tracking["frame_id"] == p4ss["frame_id"]]
        assert p4ss["DAS"] == df_tracking_frame["DAS"].iloc[0]
        assert p4ss["AS"] == df_tracking_frame["AS"].iloc[0]

        df_tracking_target_frame = df_tracking[df_tracking["frame_id"] == p4ss["target_frame_id"]]
        if p4ss["pass_outcome"] == 1:
            assert p4ss["DAS_reception"] == df_tracking_target_frame["DAS"].iloc[0]
            assert p4ss["AS_reception"] == df_tracking_target_frame["AS"].iloc[0]
        else:
            assert p4ss["DAS_reception"] == 0
            assert p4ss["AS_reception"] == 0


def test_chunk_wise_simulation():
    from .resources import df_tracking, df_passes

    F_tracking = len(df_tracking["frame_id"].unique())
    assert F_tracking > 1
    F_event = len(df_passes["frame_id"].unique())
    assert F_event > 1

    for chunk_size in [500, 1, 2, 3, None, 0, -1]:
        ret = accessible_space.get_expected_pass_completion(df_passes, df_tracking, chunk_size=chunk_size)
        assert ret.xc.shape[0] == len(df_passes)
        assert ret.event_frame_index.shape[0] == len(df_passes)
        assert ret.tracking_frame_index.shape[0] == len(df_tracking)
        assert ret.simulation_result.defense_cum_poss.shape[0] == F_event
        ret_field = accessible_space.get_dangerous_accessible_space(df_tracking, chunk_size=chunk_size)
        assert ret_field.acc_space.shape[0] == len(df_tracking)
        assert ret_field.das.shape[0] == len(df_tracking)
        assert ret_field.simulation_result.defense_poss_density.shape[0] == F_tracking


@pytest.mark.parametrize("_get_data", [_get_butterfly_data, _get_butterfly_data_with_nans])
def test_cum_prob_sum_is_1(_get_data):
    _, _, df_tracking = _get_data()
    ret = accessible_space.get_dangerous_accessible_space(
        df_tracking,
    )
    p_sum = ret.simulation_result.attack_cum_prob[0] + ret.simulation_result.defense_cum_prob[0] + ret.simulation_result.cum_p0[0]
    assert np.allclose(p_sum, 1)


@pytest.mark.parametrize("_get_data", [_get_butterfly_data, _get_butterfly_data_with_nans])
def test_simulation_result_dimensions(_get_data):
    _, _, df_tracking = _get_data()
    F = len(df_tracking["frame_id"].unique())
    P = len(df_tracking.loc[df_tracking["player_id"] != "ball", "player_id"].unique())
    n_angles = 24
    importlib.reload(accessible_space)
    importlib.reload(accessible_space)
    ret = accessible_space.get_dangerous_accessible_space(df_tracking, n_angles=n_angles)

    fields = [
        ret.simulation_result.attack_poss_density,
        ret.simulation_result.defense_poss_density,
        ret.simulation_result.attack_cum_poss,
        ret.simulation_result.defense_cum_poss,
        ret.simulation_result.attack_prob_density,
        ret.simulation_result.defense_prob_density,
        ret.simulation_result.attack_cum_prob,
        ret.simulation_result.defense_cum_prob,
        ret.simulation_result.cum_p0,
        ret.simulation_result.p0_density,
        ret.simulation_result.x_grid,
        ret.simulation_result.y_grid,
    ]
    T = ret.simulation_result.attack_poss_density.shape[2]
    for field in fields:
        assert len(field.shape) == 3
        assert field.shape[0] == F
        assert field.shape[1] == n_angles
        assert field.shape[2] == T

    individual_fields = [
        ret.simulation_result.player_cum_prob,
        ret.simulation_result.player_cum_poss,
        ret.simulation_result.player_prob_density,
        ret.simulation_result.player_poss_density,
    ]
    for field in individual_fields:
        assert len(field.shape) == 4
        assert field.shape[0] == F
        assert field.shape[1] == P
        assert field.shape[2] == n_angles
        assert field.shape[3] == T

    assert len(ret.simulation_result.r_grid.shape) == 1
    assert ret.simulation_result.r_grid.shape[0] == T
    assert len(ret.simulation_result.phi_grid.shape) == 2
    assert ret.simulation_result.phi_grid.shape[0] == F
    assert ret.simulation_result.phi_grid.shape[1] == n_angles


### TODO Re-introduce after figuring out correct normalization
# def test_integrated_prob_density_sum_is_1():
#     _, _, df_tracking = _get_butterfly_data()
#     normalize2ret = {}
#     for normalize in [True]:
#         st.write(f"#### normalize={normalize}")
#         ret = accessible_space.get_dangerous_accessible_space(df_tracking)
#         r_grid = ret.simulation_result.r_grid
#         p_cum_att_from_density = scipy.integrate.cumulative_trapezoid(y=ret.simulation_result.attack_prob_density, x=r_grid[np.newaxis, np.newaxis, :], initial=0, axis=-1)
#         p_cum_def_from_density = scipy.integrate.cumulative_trapezoid(y=ret.simulation_result.defense_prob_density, x=r_grid[np.newaxis, np.newaxis, :], initial=0, axis=-1)
#         p0_cum_from_density = 1 + scipy.integrate.cumulative_trapezoid(y=ret.simulation_result.p0_density, x=r_grid[np.newaxis, np.newaxis, :], initial=0, axis=-1)
#
#         st.write("p_cum_att_from_density", p_cum_att_from_density.shape)
#         st.write(p_cum_att_from_density[0])
#         st.write("p_cum_def_from_density", p_cum_def_from_density.shape)
#         st.write(p_cum_def_from_density[0])
#         st.write("p0_cum_from_density", p0_cum_from_density.shape)
#         st.write(p0_cum_from_density[0])
#         st.write("ret.simulation_result.cum_p0", ret.simulation_result.cum_p0.shape)
#         st.write(ret.simulation_result.cum_p0[0])
#         st.write((p_cum_att_from_density + p_cum_def_from_density + p0_cum_from_density)[0])
#
#         for field in [
#             p_cum_att_from_density,
#             p_cum_def_from_density,
#             p0_cum_from_density,
#         ]:
#             assert np.all(field >= 0)
#             assert np.all(field <= 1)
#
#         normalize2ret[normalize] = ret
#
#     norm_res = normalize2ret[True].simulation_result
#     assert ((norm_res.p_cum_att_from_density + norm_res.p_cum_def_from_density + norm_res.p0_cum_from_density) == 1).all()


@pytest.mark.parametrize("_get_data", [_get_butterfly_data, _get_butterfly_data_with_nans])
def test_probability_ranges(_get_data):
    _, _, df_tracking = _get_data()
    ret = accessible_space.get_dangerous_accessible_space(df_tracking)

    for p_cum in [
        ret.simulation_result.attack_cum_prob,
        ret.simulation_result.defense_cum_prob,
        ret.simulation_result.attack_cum_poss,
        ret.simulation_result.defense_cum_poss,
        ret.simulation_result.cum_p0,
        ret.simulation_result.player_cum_prob,
        ret.simulation_result.player_cum_poss,
    ]:
        assert (p_cum >= 0).all()
        assert (p_cum <= 1).all()

    dx = ret.simulation_result.r_grid[1] - ret.simulation_result.r_grid[0]
    for p_density in [
        ret.simulation_result.attack_prob_density,
        ret.simulation_result.defense_prob_density,
        ret.simulation_result.attack_poss_density,
        ret.simulation_result.defense_poss_density,
        ret.simulation_result.player_prob_density,
        ret.simulation_result.player_poss_density,
    ]:
        assert ((p_density * dx) >= 0).all()
        assert ((p_density * dx) <= 1).all()

    assert ((ret.simulation_result.p0_density * dx) <= 0).all()
    assert ((ret.simulation_result.p0_density * dx) >= -1).all()


@pytest.mark.parametrize("_get_data", [_get_butterfly_data, _get_butterfly_data_with_nans])
def test_das_is_smaller_than_as(_get_data):
    _, _, df_tracking = _get_data()
    ret = accessible_space.get_dangerous_accessible_space(
        df_tracking,
    )
    assert (ret.das <= ret.acc_space).all()


@pytest.mark.parametrize("_get_data", [_get_butterfly_data, _get_butterfly_data_with_nans])
def test_player_level_consistent_with_team_level(_get_data):
    _, _, df_tracking = _get_data()
    ret = accessible_space.get_dangerous_accessible_space(df_tracking)

    df_tracking["frame_index"], df_tracking["player_index"] = ret.frame_index, ret.player_index

    i_ball = df_tracking["team_id"] == "ball"
    i_att = df_tracking["team_id"] == df_tracking["team_in_possession"]
    attacking_player_indices = df_tracking.loc[~i_ball & i_att, "player_index"].dropna().astype(int).unique()
    defending_player_indices = df_tracking.loc[~i_ball & ~i_att, "player_index"].dropna().astype(int).unique()

    p_density_att_from_players = ret.simulation_result.player_prob_density[0, attacking_player_indices, :, :].sum(axis=0)
    p_density_def_from_players = ret.simulation_result.player_prob_density[0, defending_player_indices, :, :].sum(axis=0)
    assert np.allclose(p_density_att_from_players, ret.simulation_result.attack_prob_density[0])
    assert np.allclose(p_density_def_from_players, ret.simulation_result.defense_prob_density[0])

    poss_density_att_from_players = ret.simulation_result.player_poss_density[0, attacking_player_indices, :, :].max(axis=0)
    poss_density_def_from_players = ret.simulation_result.player_poss_density[0, defending_player_indices, :, :].max(axis=0)
    assert np.allclose(poss_density_att_from_players, ret.simulation_result.attack_poss_density[0])
    assert np.allclose(poss_density_def_from_players, ret.simulation_result.defense_poss_density[0])

    p_cum_att_from_players = ret.simulation_result.player_cum_prob[0, attacking_player_indices, :, :].sum(axis=0)
    p_cum_def_from_players = ret.simulation_result.player_cum_prob[0, defending_player_indices, :, :].sum(axis=0)
    assert np.allclose(p_cum_att_from_players, ret.simulation_result.attack_cum_prob[0])
    assert np.allclose(p_cum_def_from_players, ret.simulation_result.defense_cum_prob[0])

    poss_cum_att_from_players = ret.simulation_result.player_cum_poss[0, attacking_player_indices, :, :].max(axis=0)
    poss_cum_def_from_players = ret.simulation_result.player_cum_poss[0, defending_player_indices, :, :].max(axis=0)
    assert np.allclose(poss_cum_att_from_players, ret.simulation_result.attack_cum_poss[0])
    assert np.allclose(poss_cum_def_from_players, ret.simulation_result.defense_cum_poss[0])


@pytest.mark.parametrize("_get_data", [_get_butterfly_data, _get_butterfly_data_with_nans])
def test_infer_playing_direction(_get_data):
    _, _, df_tracking = _get_data()
    df_tracking["playing_direction"] = accessible_space.infer_playing_direction(df_tracking, period_col=None)
    assert (df_tracking["playing_direction"] == 1).all()

@pytest.mark.parametrize("df_tracking,period_col,exception,exception_message_substring", [
    (pd.DataFrame(), None, KeyError, "Missing columns"),
    (pd.DataFrame({"frame_id": [1, 2]}), None, KeyError, "Missing columns"),
    (pd.DataFrame({"frame_id": [1, 2], "team_id": ["H", "A"], "x": [0, 0]}), None, KeyError, "Missing column"),
    (pd.DataFrame({"frame_id": [1, 2], "team_id": ["H", "A"], "x": [0, 0], "team_in_possession": ["H", "H"]}), "period_col", KeyError, "Missing column"),
])
def test_bad_data_infer_playing_direction(df_tracking, period_col, exception, exception_message_substring):
    with pytest.raises(exception, match=exception_message_substring):
        accessible_space.infer_playing_direction(df_tracking, period_col=period_col)


@pytest.mark.parametrize("df_tracking,exception,exception_message_substring", [
    (pd.DataFrame({"frame_id": [1, 2], "team_id": ["H", "A"], "x": [0, 0], "team_in_possession": ["H", "H"]}), KeyError, "specify period_col or set to None if your data has no separate periods"),
])
def test_bad_data_infer_playing_direction_with_default_period(df_tracking, exception, exception_message_substring):
    with pytest.raises(exception, match=exception_message_substring):
        accessible_space.infer_playing_direction(df_tracking)


@pytest.mark.parametrize("_get_data", [_get_butterfly_data, _get_butterfly_data_with_nans])
def test_poss_never_below_prob(_get_data):
    _, _, df_tracking = _get_data()
    ret = accessible_space.get_dangerous_accessible_space(df_tracking, normalize=True)

    for (prob, poss) in [
        ### TODO: The remaining ones are not working yet (normalization)
        # (ret.simulation_result.player_prob_density, ret.simulation_result.player_poss_density),
        # (ret.simulation_result.player_cum_prob, ret.simulation_result.player_cum_poss),
        # (ret.simulation_result.attack_prob_density, ret.simulation_result.attack_poss_density),
        # (ret.simulation_result.defense_prob_density, ret.simulation_result.defense_poss_density),
        (ret.simulation_result.attack_cum_prob, ret.simulation_result.attack_cum_poss),
        (ret.simulation_result.defense_cum_prob, ret.simulation_result.defense_cum_poss),
    ]:
        assert np.all(prob <= poss)  # all smaller or equal


@pytest.mark.parametrize("x_min,x_max,y_min,y_max,_get_data", [
    (-52.5, 52.5, -34, 34, _get_butterfly_data),
    (0, 105, 0, 68, _get_butterfly_data),
    (-1, 1, -1, 1, _get_butterfly_data),
    (1500, 2000, 1500, 1600, _get_butterfly_data),
    (20, 200, 20, 200, _get_butterfly_data),
    (-52.5, 52.5, -34, 34, _get_butterfly_data_with_nans),
    (0, 105, 0, 68, _get_butterfly_data_with_nans),
    (-1, 1, -1, 1, _get_butterfly_data_with_nans),
    (1500, 2000, 1500, 1600, _get_butterfly_data_with_nans),
    (20, 200, 20, 200, _get_butterfly_data_with_nans),
])
def test_pitch_clipping(x_min, x_max, y_min, y_max, _get_data):
    _, _, df_tracking = _get_data()
    importlib.reload(accessible_space)
    ret = accessible_space.get_dangerous_accessible_space(df_tracking)
    cropped_result = accessible_space.clip_simulation_result_to_pitch(ret.simulation_result, x_min, x_max, y_min, y_max)

    x_grid = ret.simulation_result.x_grid
    y_grid = ret.simulation_result.y_grid
    i_in_pitch = (x_grid >= x_min) & (x_grid <= x_max) & (y_grid >= y_min) & (y_grid <= y_max)

    for field_str in [
        "attack_poss_density", "defense_poss_density", "attack_prob_density", "defense_prob_density", "p0_density",
    ]:
        field = getattr(ret.simulation_result, field_str)
        cropped_field = getattr(cropped_result, field_str)
        i_in_pitch_player = np.repeat(i_in_pitch[:, np.newaxis, :, :], field.shape[1], axis=1) if len(field.shape) == 4 else i_in_pitch

        assert np.all(field[i_in_pitch_player] == cropped_field[i_in_pitch_player])
        assert np.all(cropped_field[~i_in_pitch_player] == 0)

    for field_str in [
        "attack_cum_prob", "defense_cum_prob", "attack_cum_poss", "defense_cum_poss", "cum_p0", "player_cum_prob",
        "player_cum_poss"
    ]:
        cum_field = getattr(ret.simulation_result, field_str)
        cropped_field = getattr(cropped_result, field_str)
        i_in_pitch_field = i_in_pitch if len(cum_field.shape) == 3 else np.repeat(i_in_pitch[:, np.newaxis, :, :], cum_field.shape[1], axis=1)

        assert np.all(cum_field[i_in_pitch_field] == cropped_field[i_in_pitch_field])

        def array_to_list_of_non_nan_1d_slices(array):
            reshaped_array = array.reshape(-1, array.shape[-1])
            list_of_arrays = [row[~np.isnan(row)] for row in reshaped_array]
            return list_of_arrays

        field_outside_pitch = np.where(~i_in_pitch_field, cropped_field, np.nan)
        for time_slice_outside_pitch in array_to_list_of_non_nan_1d_slices(field_outside_pitch):
            assert len(np.unique(time_slice_outside_pitch)) == 1


@pytest.mark.parametrize("_get_data", [_get_butterfly_data, _get_butterfly_data_with_nans])
def test_surface_integration(_get_data):
    _, _, df_tracking = _get_data()
    ret = accessible_space.get_dangerous_accessible_space(df_tracking)
    areas = accessible_space.integrate_surfaces(ret.simulation_result)
    for field in areas:
        assert np.all(field >= 0)
        assert np.all(field <= 105*68)


@pytest.mark.parametrize("df_tracking,exception,exception_message_substring", [
    (pd.DataFrame(), KeyError, "Missing columns in tracking data"),
    (pd.DataFrame({"frame_id": [1, 2]}), KeyError, "Missing columns in tracking data"),
    (pd.DataFrame({"frame_id": [1, 2], "player_id": ["a", "b"], "team_id": ["H", "A"], "x": [0, 0], "vx": [0, 0]}), KeyError, "Missing columns in tracking data: y_col='y', vy_col='vy'"),
    (pd.DataFrame({"frame_id": [1, 2], "player_id": ["a", "b"], "team_id": ["H", "A"], "x": [0, 0], "y": [0, 0], "vx": [0, 0]}), KeyError, "Missing columns in tracking data: vy_col='vy'"),
    (pd.DataFrame({"frame_id": [1, 2], "player_id": ["a", "b"], "team_id": ["H", "A"], "x": [0, 0], "y": [0, 0], "vx": [0, 0], "vy": [0, 0]}), KeyError, "Missing column in tracking data: team_in_possession_col='team_in_possession'"),
    (pd.DataFrame({"frame_id": [1, 2], "player_id": ["a", "b"], "team_id": ["H", "A"], "x": [0, 0], "y": [0, 0], "vx": [0, 0], "vy": [0, 0], "team_in_possession": ["H", "H"]}), ValueError, "Ball flag ball_tracking_player_id='ball' does not exist in column "),
])
def test_bad_data_das(df_tracking, exception, exception_message_substring):
    with pytest.raises(exception, match=exception_message_substring):
        accessible_space.get_dangerous_accessible_space(df_tracking)

#     df_tracking =
@pytest.mark.parametrize("df_passes,df_tracking,exception,use_event_ball_pos,exception_message_substring", [
    (pd.DataFrame({"frame_id": [1, 1], "player_id": ["a", "a"], "team_id": ["H", "H"], "x": [0, 0], "x_target": [1, 1], "y": [2, 2], "y_target": [3, 3]}), pd.DataFrame(), KeyError, True, "Missing columns in df_tracking"),
    (pd.DataFrame(), pd.DataFrame({"frame_id": [1, 1, 1, 1], "player_id": ["a", "ball", "b", "c"], "team_id": ["H", None, "A", "H"], "x": [0, 0, 1, 2], "y": [0, 0, 1, 2], "vx": [0, 0, 1, 2], "vy": [0, 0, 1, 2], "team_in_possession": ["H", "H", "H", "H"]}), KeyError, True, "Missing columns in df_passes"),
    (pd.DataFrame({"frame_id": [1, 1], "player_id": ["a", "a"], "team_id": ["H", "H"], "x": [0, 0], "x_target": [1, 1], "y": [2, 2], "y_target": [3, 3]}), pd.DataFrame({"frame_id": [1, 1], "player_id": ["a", "b"], "team_id": ["H", "A"], "x": [0, 0], "y": [0, 0], "vx": [0, 0], "vy": [0, 0], "team_in_possession": ["H", "H"]}), ValueError, True, "Ball flag ball_tracking_player_id='ball' does not exist in column"),
])
def test_bad_data_xc(df_passes, df_tracking, use_event_ball_pos, exception, exception_message_substring):
    with pytest.raises(exception, match=exception_message_substring):
        accessible_space.get_expected_pass_completion(df_passes, df_tracking)


def test_duplicate_frames():
    df_passes = pd.DataFrame({"frame_id": [5, 5], "player_id": ["a", "a"], "team_id": ["H", "H"], "x": [0, 0], "x_target": [1, 1], "y": [2, 2], "y_target": [3, 3]})
    df_tracking = pd.DataFrame({"frame_id": [5, 5, 5, 5], "player_id": ["a", "ball", "b", "c"], "team_id": ["H", None, "A", "H"], "x": [0, 0, 1, 2], "y": [0, 0, 1, 2], "vx": [0, 0, 1, 2], "vy": [0, 0, 1, 2], "team_in_possession": ["H", "H", "H", "H"]})
    ret = accessible_space.get_expected_pass_completion(df_passes, df_tracking)
    assert len(ret.xc) == len(df_passes)
    assert len(ret.event_frame_index) == len(df_passes)
    assert len(ret.tracking_frame_index) == len(df_tracking)
    assert len(ret.tracking_player_index) == len(df_tracking)

    df_passes["xC"] = ret.xc
    df_passes["frame_index"] = ret.event_frame_index
    df_tracking["frame_index"] = ret.tracking_frame_index
    df_tracking["player_index"] = ret.tracking_player_index

    assert len(df_passes["frame_index"].unique()) == len(df_passes)
    assert ret.simulation_result.attack_poss_density.shape[0] == len(df_passes)


def test_minimal_das_runs_error_free():
    df_tracking = pd.DataFrame({"frame_id": [1, 1, 1, 1], "player_id": ["a", "ball", "b", "c"], "team_id": ["H", None, "A", "H"], "x": [0, 0, 1, 2], "y": [0, 0, 1, 2], "vx": [0, 0, 1, 2], "vy": [0, 0, 1, 2], "team_in_possession": ["H", "H", "H", "H"]})
    accessible_space.get_dangerous_accessible_space(df_tracking)


def test_minimal_xc_runs_error_free():
    df_tracking = pd.DataFrame({"frame_id": [1, 1, 1, 1], "player_id": ["a", "ball", "b", "c"], "team_id": ["H", None, "A", "H"], "x": [0, 0, 1, 2], "y": [0, 0, 1, 2], "vx": [0, 0, 1, 2], "vy": [0, 0, 1, 2], "team_in_possession": ["H", "H", "H", "H"]})
    df_passes = pd.DataFrame({"frame_id": [1, 1], "player_id": ["a", "a"], "team_id": ["H", "H"], "x": [0, 0], "x_target": [1, 1], "y": [2, 2], "y_target": [3, 3]})
    accessible_space.get_expected_pass_completion(df_passes, df_tracking)



@pytest.mark.parametrize("_get_data", [_get_butterfly_data, _get_butterfly_data_with_nans])
def test_fields_to_return_are_present(_get_data):
    _, _, df_tracking = _get_data()

    for field in [
        "attack_cum_prob",
        "attack_cum_poss",
        "attack_prob_density",
        "attack_poss_density",
        "defense_cum_prob",
        "defense_cum_poss",
        "defense_prob_density",
        "defense_poss_density",
        "cum_p0",
        "p0_density",
        "player_cum_prob",
        "player_cum_poss",
        "player_prob_density",
        "player_poss_density",
    ]:
        ret = accessible_space.get_dangerous_accessible_space(df_tracking, additional_fields_to_return=[field])
        assert getattr(ret.simulation_result, field) is not None


@pytest.mark.parametrize("_get_data", [_get_butterfly_data, _get_butterfly_data_with_nans])
def test_fields_to_return_others_are_not_present(_get_data):
    _, _, df_tracking = _get_data()

    all_fields = [
        "attack_cum_prob",
        "attack_cum_poss",
        "attack_prob_density",
        "attack_poss_density",
        "defense_cum_prob",
        "defense_cum_poss",
        "defense_prob_density",
        "defense_poss_density",
        "cum_p0",
        "p0_density",
        "player_cum_prob",
        "player_cum_poss",
        "player_prob_density",
        "player_poss_density",
    ]
    das_fields = ["attack_poss_density"]

    for field in all_fields:
        ret = accessible_space.get_dangerous_accessible_space(df_tracking, additional_fields_to_return=[field])

        none_fields = [ret_field for ret_field in all_fields if getattr(ret.simulation_result, ret_field) is None]
        present_fields = [ret_field for ret_field in all_fields if isinstance(getattr(ret.simulation_result, ret_field), np.ndarray)]
        remaining_fields = [ret_field for ret_field in all_fields if ret_field not in none_fields and ret_field not in present_fields]

        expected_fields = {field}.union(set(das_fields))

        assert len(remaining_fields) == 0
        assert set(present_fields) == expected_fields

    for field1 in all_fields:
        for field2 in all_fields:
            if field1 == field2:
                continue
            ret = accessible_space.get_dangerous_accessible_space(df_tracking, additional_fields_to_return=[field1, field2])

            none_fields = [ret_field for ret_field in all_fields if getattr(ret.simulation_result, ret_field) is None]
            present_fields = [ret_field for ret_field in all_fields if isinstance(getattr(ret.simulation_result, ret_field), np.ndarray)]
            remaining_fields = [ret_field for ret_field in all_fields if ret_field not in none_fields and ret_field not in present_fields]

            expected_fields = {field1, field2}.union(set(das_fields))

            assert len(remaining_fields) == 0
            assert set(present_fields) == expected_fields
            for present_field in present_fields:
                field_data = getattr(ret.simulation_result, present_field)
                assert np.all(~np.isnan(field_data))
                assert np.any(field_data != 0)


@pytest.mark.parametrize("_get_data", [_get_butterfly_data, _get_butterfly_data_with_nans])
def test_surface_plot(_get_data):
    _, _, df_tracking = _get_data()
    ret = accessible_space.get_dangerous_accessible_space(df_tracking)

    def _plot():
        plt.figure()
        plt.xlim([-52.5, 52.5])
        plt.ylim([-34, 34])
        plt.scatter(df_tracking["x"], df_tracking["y"], color=df_tracking["player_color"])

    _plot()
    accessible_space.plot_expected_completion_surface(ret.simulation_result, 0, "attack_poss_density")
    # st.write("attack_poss_density")
    # st.write(plt.gcf())
    plt.close()
    _plot()
    accessible_space.plot_expected_completion_surface(ret.simulation_result, 0, "attack_prob_density")
    # st.write("attack_prob_density")
    # st.write(plt.gcf())
    plt.close()
    _plot()
    accessible_space.plot_expected_completion_surface(ret.simulation_result, 0, "defense_poss_density")
    # st.write("defense_poss_density")
    # st.write(plt.gcf())
    plt.close()
    _plot()
    accessible_space.plot_expected_completion_surface(ret.simulation_result, 0, "defense_prob_density")
    # st.write("defense_prob_density")
    # st.write(plt.gcf())
    plt.close()
    _plot()
    accessible_space.plot_expected_completion_surface(ret.simulation_result, 0, "player_prob_density", player_index=0)
    # st.write("player_prob_density")
    # st.write(plt.gcf())
    plt.close()
    _plot()
    accessible_space.plot_expected_completion_surface(ret.simulation_result, 0, "player_poss_density", player_index=0)
    # st.write("player_poss_density")
    # st.write(plt.gcf())
    plt.close()

    with pytest.raises(ValueError, match=f"no player index is given"):
        _plot()
        accessible_space.plot_expected_completion_surface(ret.simulation_result, 0, "player_prob_density")

    plt.close()


@pytest.mark.parametrize("_get_data", [_get_butterfly_data, _get_butterfly_data_with_nans])
def test_additional_defender_decreases_as_and_additional_attacker_increases_as(_get_data):
    _, _, df_tracking = _get_data()

    def get_as_and_das(_df_tracking):
        ret_baseline = accessible_space.get_dangerous_accessible_space(
            _df_tracking, infer_attacking_direction=False, attacking_direction_col="attacking_direction",
        )
        return ret_baseline.acc_space.iloc[0], ret_baseline.das.iloc[0]

    baseline_as, baseline_das = get_as_and_das(df_tracking)

    defending_team = [team for team in df_tracking["team_id"].unique() if team != df_tracking["team_in_possession"].iloc[0]][0]
    attacking_team = df_tracking["team_in_possession"].iloc[0]

    for new_x in [0, 10, -53]:
        for new_y in [-10, 40]:
            for new_vx in [0, -20]:
                for new_vy in [0, -2]:
                    df_tracking_extra_defender = df_tracking.copy()
                    extra_defender_data = {
                        "frame_id": df_tracking["frame_id"].iloc[0],
                        "player_id": "extra_player",
                        "x": new_x, "y": new_y, "vx": new_vx, "vy": new_vy,
                        "team_id": defending_team,
                        "player_color": None,
                        "attacking_direction": df_tracking["attacking_direction"].iloc[0],
                        "team_in_possession": df_tracking["team_in_possession"].iloc[0],
                        "player_in_possession": df_tracking["player_in_possession"].iloc[0],
                    }
                    df_tracking_extra_defender.loc[len(df_tracking_extra_defender)] = pd.Series(extra_defender_data)

                    as_with_extra_defender, das_with_extra_defender = get_as_and_das(df_tracking_extra_defender)
                    assert as_with_extra_defender <= baseline_as
                    assert das_with_extra_defender <= baseline_das

                    df_tracking_extra_attacker = df_tracking.copy()
                    extra_attacker_data = {
                        "frame_id": df_tracking["frame_id"].iloc[0],
                        "player_id": "extra_player",
                        "x": new_x, "y": new_y, "vx": new_vx, "vy": new_vy,
                        "team_id": attacking_team,
                        "player_color": None,
                        "attacking_direction": df_tracking["attacking_direction"].iloc[0],
                        "team_in_possession": df_tracking["team_in_possession"].iloc[0],
                        "player_in_possession": df_tracking["player_in_possession"].iloc[0],
                    }
                    # df_tracking_extra_attacker = df_tracking_extra_attacker.append(pd.Series(extra_attacker_data), ignore_index=True)
                    df_tracking_extra_attacker.loc[len(df_tracking_extra_attacker)] = pd.Series(extra_attacker_data)
                    as_with_extra_attacker, das_with_extra_attacker = get_as_and_das(df_tracking_extra_attacker)
                    assert as_with_extra_attacker >= baseline_as, f"new_x={new_x}, new_y={new_y}, new_vx={new_vx} new_vy={new_vy}"
                    assert das_with_extra_attacker >= baseline_das


def example_function():
    """
    >>> example_function()
    52.5
    """
    # Convert a number to float64
    result = np.float64(52.5) if np.__version__ >= '1.20.0' else np.float64(52.5).astype(np.float64)
    return result


@pytest.mark.parametrize("dataset_nr", [1, 2])
def rest_real_world_data(dataset_nr):
    @st.cache_resource
    def get_metrica_data(
            dataset_nr=1, new_team_col="TEAM", new_player_col="PLAYER", new_x_col="X", new_y_col="Y",
            passer_team_col="passer_team", receiver_team_col="receiver_team",
    ):
        def _get():
            #                             https://raw.githubusercontent.com/metrica-sports/sample-data/refs/heads/master/data/Sample_Game_1/Sample_Game_1_RawEventsData.csv
            metrica_open_data_base_dir = "https://raw.githubusercontent.com/metrica-sports/sample-data/refs/heads/master/data"
            # home_data=f"https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_{dataset_nr}/Sample_Game_{dataset_nr}_RawTrackingData_Home_Team.csv"
            home_data_url = f"{metrica_open_data_base_dir}/Sample_Game_{dataset_nr}/Sample_Game_{dataset_nr}_RawTrackingData_Home_Team.csv"
            # away_data=f"https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_{dataset_nr}/Sample_Game_{dataset_nr}_RawTrackingData_Away_Team.csv"
            away_data_url = f"{metrica_open_data_base_dir}/Sample_Game_{dataset_nr}/Sample_Game_{dataset_nr}_RawTrackingData_Away_Team.csv"
            event_url = f"{metrica_open_data_base_dir}/Sample_Game_{dataset_nr}/Sample_Game_{dataset_nr}_RawEventsData.csv"
            with st.spinner(f"Loading data 1/3 from {event_url}"):
                df_events = pd.read_csv(event_url)
            with st.spinner(f"Loading data 2/3 from {home_data_url}"):
                df_home = pd.read_csv(home_data_url, skiprows=2)
            with st.spinner(f"Loading data 3/3 from {away_data_url}"):
                df_away = pd.read_csv(away_data_url, skiprows=2)
            return df_home, df_away, df_events

        df_home, df_away, df_events = _get()

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

        df_passes = df_events[df_events["Type"] == "PASS"]
        player2team = df_tracking[[new_player_col, new_team_col]].set_index(new_player_col)[new_team_col].to_dict()
        df_passes[passer_team_col] = df_passes["From"].map(player2team)
        df_passes[receiver_team_col] = df_passes["To"].map(player2team)

        for x_col in ["Start X"]:
            df_passes[x_col] = (df_passes[x_col] - 0.5) * 105
        for y_col in ["Start Y"]:
            df_passes[y_col] = (df_passes[y_col] - 0.5) * 68
        for x_col in [new_x_col]:
            df_tracking[x_col] = (df_tracking[x_col] - 0.5) * 105
        for y_col in [new_y_col]:
            df_tracking[y_col] = (df_tracking[y_col] - 0.5) * 68

        df_passes["is_successful"] = df_passes[passer_team_col] == df_passes[receiver_team_col]

        return df_passes, df_tracking

    df_passes, df_tracking = get_metrica_data(dataset_nr)

    st.write("df_passes after parse")
    st.write(df_passes)

    ret = accessible_space.get_expected_pass_completion(
        df_passes=df_passes, df_tracking=df_tracking, tracking_frame_col="Frame", event_frame_col="Start Frame",
        event_start_x_col="Start X", event_end_x_col="End X", event_start_y_col="Start Y", event_end_y_col="End Y",
        event_player_col="From", use_event_coordinates_as_ball_position=True, ball_tracking_player_id="Ball",
        event_team_col="Team", tracking_x_col="X", tracking_y_col="Y", tracking_team_col="TEAM",
        tracking_player_col="PLAYER"
    )
    df_passes["xc"] = ret.xc

    for pass_nr, (pass_index, p4ss) in enumerate(df_passes.iterrows()):
        plt.figure()
        plt.arrow(p4ss["Start X"], p4ss["Start Y"], p4ss["End X"]-p4ss["Start X"], p4ss["End Y"]-p4ss["Start Y"], color="black", head_width=2, head_length=3)
        df_tracking_frame = df_tracking[df_tracking["Frame"] == p4ss["Start Frame"]]
        for team, df_tracking_frame_team in df_tracking_frame.groupby("TEAM"):
            team2color = {"Home": "red", "Away": "blue"}
            plt.scatter(df_tracking_frame_team["X"], df_tracking_frame_team["Y"], color=team2color[team])

        df_tracking_frame["team_in_possession"] = p4ss["Team"]
        ret = accessible_space.get_dangerous_accessible_space(
            df_tracking_frame, frame_col="Frame", x_col="X", y_col="Y", ball_player_id="Ball",
            return_cropped_result=True, player_col="PLAYER", team_col="TEAM",
        )
        plt.xlim([-52.5, 52.5])
        plt.ylim([-34, 34])

        plt.title(f"Frame: {p4ss['Start Frame']}, xC: {p4ss['xc']:1f}, Success: {p4ss['success']}, DAS: {ret.das.iloc[0]}, AS: {ret.acc_space.iloc[0]}")

        accessible_space.plot_expected_completion_surface(ret.simulation_result)
        st.write(plt.gcf())

        df_passes.loc[pass_index, "DAS_from_das"] = ret.das.iloc[0]
        df_passes.loc[pass_index, "AS_from_das"] = ret.acc_space.iloc[0]

        if pass_nr > 10:
            break

        plt.close()

    # average_xc = df_passes["xc"].mean()
    # average_success = df_passes["success"].mean()
    # st.write("xc", average_xc, "success", average_success)
    # assert average_xc > 0.5
    # assert average_success < 1.0

    st.write("df_passes")
    st.write(df_passes)

    @st.cache_resource
    def _get2():
        ret2 = accessible_space.get_das_gained(
            df_passes, df_tracking,
            tracking_frame_col="Frame",
            tracking_x_col=new_x_col,
            tracking_y_col="Y",
            tracking_team_col=new_team_col,
            tracking_player_col=new_player_col,
            ball_tracking_player_id="Ball",
            event_frame_col="Start Frame",
            event_target_frame_col="End Frame",
            event_success_col="success",
            event_team_col="Team",
            event_receiver_team_col="rec_team_id",
            # event_start_x_col="Start X",
            # event_start_y_col="Start Y",
            # event_target_x_col="End X",
            # event_target_y_col="End Y",
            use_event_coordinates_as_ball_position=False,
            use_event_team_as_team_in_possession=True,
        )
        return ret2

    ret2 = _get2()

    st.write("ret2")
    st.write(ret2)


# TODO add back
# def test_validation_runs_error_free():
#     accessible_space.validation_dashboard()
