import importlib
import pytest
import numpy as np
import pandas as pd
import streamlit as st
import accessible_space


def _get_butterfly_data():
    df_tracking = pd.DataFrame({
        "frame_id": [0, 0, 0, 0],
        "player_id": ["a", "b", "x", "ball"],
        "x": [0, -50, 50, 0],
        "y": [0, 0, 0, 0],
        "vx": [0, 0, 0, 0],
        "vy": [0, 0, 0, 0],
        "team_id": ["H", "H", "A", None],
        "player_color": ["blue", "blue", "red", "black"],
        "team_in_possession": ["H"] * 4,
        "player_in_possession": ["a"] * 4,
        "attacking_direction": [1] * 4,
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

    ### Plotting, TODO how to incorporate plotting cleanly into pytests
    # plt.scatter(df_tracking["x"], df_tracking["y"], color=df_tracking["player_color"])
    # st.write(plt.gcf())
    # plt.show()

    return df_tracking


def test_as_symmetry():
    df_tracking = _get_double_butterfly_data()

    reaches_half_space = False
    for angles, v0_min, n_v0 in [
        (64, 0.01, 250),
        (32, 0.01, 250),
        (48, 0.01, 250),
        (12, 0.01, 250),
        (24, 0.01, 250),
        (24, 0.001, 250),
        (24, 0.0001, 250),
        (24, 0.1, 250),
        (24, 1, 250),
    ]:
        ret = accessible_space.get_dangerous_accessible_space(
            df_tracking, passer_to_exclude_col="player_in_possession", n_angles=angles, v0_min=v0_min, n_v0=n_v0
        )
        st.write(angles, v0_min, n_v0, ret.acc_space.iloc[0], 3570)
        if np.isclose(ret.acc_space.iloc[0], 3570, atol=10):  # allow 10m² error here, 3570 = 105*68/2. This is very sensitive to the parameters though - we merely ensure that it's POSSIBLE to achieve this value.
            reaches_half_space = True
    assert reaches_half_space

    accessible_space.plot_expected_completion_surface(ret.simulation_result, "attack_poss_density", 0, color="blue")

    ### Plotting
    # plt.xlim([-52.5, 52.5])
    # plt.ylim([-34, 34])
    # st.write(plt.gcf())
    # st.write("ret")
    # st.write("ret.acc_space", ret.acc_space)
    # st.write("ret.simulation_result.attack_poss_density[0]", ret.simulation_result.attack_poss_density[0])


def test_xc_symmetry():
    df_pass_safe, df_pass_risky, df_tracking = _get_butterfly_data()

    ret_safe = accessible_space.get_expected_pass_completion(df_pass_safe, df_tracking)
    assert ret_safe.xc > 0.95
    ret_risky = accessible_space.get_expected_pass_completion(df_pass_risky, df_tracking)
    assert ret_risky.xc < 0.05

    assert np.isclose(ret_safe.xc, 1 - ret_risky.xc, atol=1e-3)


# TODO parameterize with different data
def test_cum_prob_sum_is_1():
    _, _, df_tracking = _get_butterfly_data()
    ret = accessible_space.get_dangerous_accessible_space(
        df_tracking,
    )
    p_sum = ret.simulation_result.attack_cum_prob[0] + ret.simulation_result.defense_cum_prob[0] + ret.simulation_result.cum_p0[0]
    assert np.allclose(p_sum, 1)


def test_simulation_result_dimensions():
    _, _, df_tracking = _get_butterfly_data()
    F = len(df_tracking["frame_id"].unique())
    P = len(df_tracking.loc[df_tracking["player_id"] != "ball", "player_id"].unique())
    n_angles = 24
    importlib.reload(accessible_space)
    importlib.reload(accessible_space)
    ret = accessible_space.get_dangerous_accessible_space(df_tracking, n_angles=n_angles)
    st.write("ret.simulation_result.attack_cum_poss", ret.simulation_result.attack_cum_poss.shape)

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
        st.write("a")
        st.write(field[0])
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
#         st.write(f"#### {normalize=}")
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
#             assert np.alltrue(field >= 0)
#             assert np.alltrue(field <= 1)
#
#         normalize2ret[normalize] = ret
#
#     norm_res = normalize2ret[True].simulation_result
#     assert ((norm_res.p_cum_att_from_density + norm_res.p_cum_def_from_density + norm_res.p0_cum_from_density) == 1).all()


def test_probability_ranges():
    _, _, df_tracking = _get_butterfly_data()
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


def test_das_is_smaller_than_as():
    _, _, df_tracking = _get_butterfly_data()
    ret = accessible_space.get_dangerous_accessible_space(
        df_tracking,
    )
    assert (ret.das <= ret.acc_space).all()


# def test_cum_prob_is_consistent_with_prob_density():
#     _, _, df_tracking = _get_butterfly_data()
#     ret = accessible_space.get_dangerous_accessible_space(
#         df_tracking, passer_to_exclude_col="player_in_possession", n_angles=4, n_v0=1, radial_gridsize=5,
#     )
#
#     prob_cum = ret.simulation_result.attack_cum_prob
#     prob_density = ret.simulation_result.attack_prob_density
#     r = ret.simulation_result.r_grid
#
#     st.write("prob_cum", prob_cum.shape)
#     st.write("prob_density", prob_density.shape)
#     st.write("r", r.shape)
#
#     prob_cum2 = scipy.integrate.cumulative_trapezoid(y=prob_density, x=r[np.newaxis, np.newaxis, :], initial=0, axis=-1)
#     st.write("prob_cum2", prob_cum2.shape)
#     st.write(prob_cum2[0])
#     st.write("prob_cum", prob_cum.shape)
#     st.write(prob_cum[0])
#     st.write((prob_cum - prob_cum2)[0])
#
#     assert np.allclose(prob_cum, prob_cum2), prob_cum - prob_cum2


def test_player_level_consistent_with_team_level():
    _, _, df_tracking = _get_butterfly_data()
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


def test_poss_never_below_prob():
    _, _, df_tracking = _get_butterfly_data()
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
        assert np.alltrue(prob <= poss)  # all smaller or equal


def test_pitch_clipping():
    _, _, df_tracking = _get_butterfly_data()
    ret = accessible_space.get_dangerous_accessible_space(df_tracking)
    cropped_result = accessible_space.crop_density_to_pitch(ret.simulation_result)

    x_grid = ret.simulation_result.x_grid
    y_grid = ret.simulation_result.y_grid

    i_in_pitch = (x_grid >= -52.5) & (x_grid <= 52.5) & (y_grid <= 34) & (y_grid >= -34)
    for field, cropped_field in [
        (ret.simulation_result.attack_poss_density, cropped_result.attack_poss_density),
        (ret.simulation_result.defense_poss_density, cropped_result.defense_poss_density),
        (ret.simulation_result.attack_prob_density, cropped_result.attack_prob_density),
        (ret.simulation_result.defense_prob_density, cropped_result.defense_prob_density),
        (ret.simulation_result.p0_density, cropped_result.p0_density),
    ]:
        field[~i_in_pitch] = 0
        assert np.alltrue(field[i_in_pitch] == cropped_field[i_in_pitch])
        assert np.alltrue(field[~i_in_pitch] == 0)

    for player_field, cropped_field in [
        (ret.simulation_result.player_poss_density, cropped_result.player_poss_density),
        (ret.simulation_result.player_prob_density, cropped_result.player_prob_density),
    ]:
        i_in_pitch_player = np.repeat(i_in_pitch[:, np.newaxis, :, :], player_field.shape[1], axis=1)
        player_field[~i_in_pitch_player] = 0  # TODO boolean comparison newaxis
        assert np.alltrue(player_field[i_in_pitch_player] == cropped_field[i_in_pitch_player])
        assert np.alltrue(player_field[~i_in_pitch_player] == 0)


def test_surface_integration():
    _, _, df_tracking = _get_butterfly_data()
    ret = accessible_space.get_dangerous_accessible_space(df_tracking)
    areas = accessible_space.integrate_surfaces(ret.simulation_result)
    for field in areas:
        assert np.alltrue(field >= 0)
        assert np.alltrue(field <= 105*68)


@pytest.mark.parametrize("df_tracking,exception,exception_message_substring", [
    (pd.DataFrame(), ValueError, "Tracking data is empty"),
    (pd.DataFrame({"frame_id": [1, 2]}), KeyError, "Missing columns in tracking data"),
    (pd.DataFrame({"frame_id": [1, 2], "player_id": ["a", "b"], "team_id": ["H", "A"], "x": [0, 0], "vx": [0, 0]}), KeyError, "Missing columns in tracking data: y_col='y', vy_col='vy'"),
    (pd.DataFrame({"frame_id": [1, 2], "player_id": ["a", "b"], "team_id": ["H", "A"], "x": [0, 0], "y": [0, 0], "vx": [0, 0]}), KeyError, "Missing column in tracking data: vy_col='vy'"),
    (pd.DataFrame({"frame_id": [1, 2], "player_id": ["a", "b"], "team_id": ["H", "A"], "x": [0, 0], "y": [0, 0], "vx": [0, 0], "vy": [0, 0]}), KeyError, "Tracking data does not contain column 'team_in_possession'"),
    (pd.DataFrame({"frame_id": [1, 2], "player_id": ["a", "b"], "team_id": ["H", "A"], "x": [0, 0], "y": [0, 0], "vx": [0, 0], "vy": [0, 0], "team_in_possession": ["H", "H"]}), ValueError, "Tracking data contains no ball data"),
])
def test_bad_data_das(df_tracking, exception, exception_message_substring):
    with pytest.raises(exception, match=exception_message_substring):
        accessible_space.get_dangerous_accessible_space(df_tracking)

#     df_tracking =
@pytest.mark.parametrize("df_passes,df_tracking,exception,use_event_ball_pos,exception_message_substring", [
    (pd.DataFrame({"frame_id": [1, 1], "player_id": ["a", "a"], "team_id": ["H", "H"], "x": [0, 0], "x_target": [1, 1], "y": [2, 2], "y_target": [3, 3]}), pd.DataFrame(), ValueError, True, "Tracking data is empty"),
    (pd.DataFrame(), pd.DataFrame({"frame_id": [1, 1, 1, 1], "player_id": ["a", "ball", "b", "c"], "team_id": ["H", None, "A", "H"], "x": [0, 0, 1, 2], "y": [0, 0, 1, 2], "vx": [0, 0, 1, 2], "vy": [0, 0, 1, 2], "team_in_possession": ["H", "H", "H", "H"]}), ValueError, True, "Passes data is empty"),
    (pd.DataFrame({"frame_id": [1, 1], "player_id": ["a", "a"], "team_id": ["H", "H"], "x": [0, 0], "x_target": [1, 1], "y": [2, 2], "y_target": [3, 3]}), pd.DataFrame({"frame_id": [1, 1], "player_id": ["a", "b"], "team_id": ["H", "A"], "x": [0, 0], "y": [0, 0], "vx": [0, 0], "vy": [0, 0], "team_in_possession": ["H", "H"]}), ValueError, True, "Tracking data contains no ball data"),
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


def test_fields_to_return_are_present():  # TODO fix
    _, _, df_tracking = _get_butterfly_data()

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


def test_fields_to_return_others_are_not_present():
    _, _, df_tracking = _get_butterfly_data()

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
        st.write(f"### {field}")
        ret = accessible_space.get_dangerous_accessible_space(df_tracking, additional_fields_to_return=[field])

        none_fields = [ret_field for ret_field in all_fields if getattr(ret.simulation_result, ret_field) is None]
        present_fields = [ret_field for ret_field in all_fields if isinstance(getattr(ret.simulation_result, ret_field), np.ndarray)]
        remaining_fields = [ret_field for ret_field in all_fields if ret_field not in none_fields and ret_field not in present_fields]

        expected_fields = {field}.union(set(das_fields))

        assert len(remaining_fields) == 0
        assert set(present_fields) == expected_fields

    for field1, field2 in zip(all_fields, all_fields):
        st.write(f"### {field1}, {field2}")
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
            assert np.alltrue(field_data > 0)
            assert np.alltrue(~np.isnan(field_data))

    for field1, field2, field3 in zip(all_fields, all_fields, all_fields):
        st.write(f"### {field1}, {field2}, {field3}")
        if field1 == field2:
            continue
        ret = accessible_space.get_dangerous_accessible_space(df_tracking, additional_fields_to_return=[field1, field2, field3])

        none_fields = [ret_field for ret_field in all_fields if getattr(ret.simulation_result, ret_field) is None]
        present_fields = [ret_field for ret_field in all_fields if isinstance(getattr(ret.simulation_result, ret_field), np.ndarray)]
        remaining_fields = [ret_field for ret_field in all_fields if ret_field not in none_fields and ret_field not in present_fields]

        expected_fields = {field1, field2, field3}.union(set(das_fields))

        assert len(remaining_fields) == 0
        assert set(present_fields) == expected_fields
        for present_field in present_fields:
            field_data = getattr(ret.simulation_result, present_field)
            assert np.alltrue(field_data > 0)
            assert np.alltrue(~np.isnan(field_data))


    for field1, field2, field3, field4, field5 in zip(all_fields, all_fields, all_fields, all_fields, all_fields):
        st.write(f"### {field1}, {field2}, {field3}, {field4}, {field5}")
        if field1 == field2:
            continue
        ret = accessible_space.get_dangerous_accessible_space(df_tracking, additional_fields_to_return=[field1, field2, field3, field4, field5])

        none_fields = [ret_field for ret_field in all_fields if getattr(ret.simulation_result, ret_field) is None]
        present_fields = [ret_field for ret_field in all_fields if isinstance(getattr(ret.simulation_result, ret_field), np.ndarray)]
        remaining_fields = [ret_field for ret_field in all_fields if ret_field not in none_fields and ret_field not in present_fields]

        expected_fields = {field1, field2, field3, field4, field5}.union(set(das_fields))

        assert len(remaining_fields) == 0
        assert set(present_fields) == expected_fields
        for present_field in present_fields:
            field_data = getattr(ret.simulation_result, present_field)
            assert np.alltrue(field_data > 0)
            assert np.alltrue(~np.isnan(field_data))


def test_additional_defender_decreases_as_and_additional_attacker_increases_as():
    _, _, df_tracking = _get_butterfly_data()

    def get_as_and_das(_df_tracking):
        ret_baseline = accessible_space.get_dangerous_accessible_space(
            _df_tracking, infer_attacking_direction=False, attacking_direction_col="attacking_direction",
            n_v0=100, v0_min=1  # TODO removing this breaks the test - check why. Likely AS/DAS dont converge.
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
                    assert as_with_extra_attacker >= baseline_as, f"{new_x=}, {new_y=}, {new_vx=} {new_vy=}"
                    assert das_with_extra_attacker >= baseline_das
