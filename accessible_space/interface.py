import collections
import warnings

import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.tri
import numpy as np
import pandas as pd
import streamlit as st

from . import as_dangerous_result
from .core import _DEFAULT_PASS_START_LOCATION_OFFSET, _DEFAULT_B0, _DEFAULT_TIME_OFFSET_BALL, _DEFAULT_A_MAX, \
    _DEFAULT_USE_MAX, _DEFAULT_USE_APPROX_TWO_POINT, _DEFAULT_B1, _DEFAULT_PLAYER_VELOCITY, _DEFAULT_V_MAX, \
    _DEFAULT_KEEP_INERTIAL_VELOCITY, _DEFAULT_INERTIAL_SECONDS, _DEFAULT_TOL_DISTANCE, _DEFAULT_RADIAL_GRIDSIZE, \
    _DEFAULT_V0_PROB_AGGREGATION_MODE, _DEFAULT_NORMALIZE, _DEFAULT_USE_EFFICIENT_SIGMOID, \
    simulate_passes_chunked, clip_simulation_result_to_pitch, integrate_surfaces, SimulationResult
from .utility import get_unused_column_name, _dist_to_opp_goal, _opening_angle_to_goal, _adjust_saturation, _unset, \
    _replace_column_values_except_nans

ReturnValueDASGained = collections.namedtuple("ReturnValueDASGained", [
    "acc_space", "das", "acc_space_reception", "das_reception", "as_gained", "das_gained", "simulation_result",
    "frame_index", "target_frame_index"
])
ReturnValueXC = collections.namedtuple("ReturnValueXC", [
    "xc", "event_frame_index", "tracking_frame_index", "tracking_player_index", "simulation_result"
])
ReturnValueDAS = collections.namedtuple("ReturnValueDAS", [
    "acc_space", "das", "frame_index", "player_index", "simulation_result", "dangerous_result"
])

_DEFAULT_N_FRAMES_AFTER_PASS_FOR_V0 = 3
_DEFAULT_FALLBACK_V0 = 10
_DEFAULT_USE_POSS_FOR_XC = False
_DEFAULT_USE_FIXED_V0_FOR_XC = True
_DEFAULT_V0_MAX_FOR_XC = 15.108273248071049
_DEFAULT_V0_MIN_FOR_XC = 4.835618861117393
_DEFAULT_N_V0_FOR_XC = 6

_DEFAULT_N_ANGLES_FOR_DAS = 60
_DEFAULT_PHI_OFFSET = 0
_DEFAULT_N_V0_FOR_DAS = 60
_DEFAULT_V0_MIN_FOR_DAS = 0.01
_DEFAULT_V0_MAX_FOR_DAS = 30


def _check_presence_of_required_columns(df, str_data, column_names, column_values, additional_message=None):
    missing_tracking_cols = [(col_name, col_value) for (col_name, col_value) in zip(column_names, column_values) if col_value not in df.columns]
    if len(missing_tracking_cols) > 0:
        raise KeyError(f"""Missing column{'s' if len(missing_tracking_cols) > 1 else ''} in {str_data}: {', '.join(['='.join([str(parameter_name), "'" + str(col) + "'"]) for (parameter_name, col) in missing_tracking_cols])}.{' ' + additional_message if additional_message is not None else ''}""")


def _check_tracking_coordinate_ranges(df_tracking, x_col, y_col):
    x_max_real = df_tracking[x_col].max()
    x_min_real = df_tracking[x_col].min()
    y_max_real = df_tracking[y_col].max()
    y_min_real = df_tracking[y_col].min()

    if x_max_real - x_min_real < 5:
        warnings.warn(f"Range of tracking X coordinates ({x_min_real} to {x_max_real}) is very small. Make sure your coordinates are in meters.")
    if y_max_real - y_min_real < 5:
        warnings.warn(f"Range of tracking Y coordinates ({y_min_real} to {y_max_real}) is very small. Make sure your coordinates are in meters.")


def _check_pitch_dimensions(x_pitch_min, x_pitch_max, y_pitch_min, y_pitch_max):
    if x_pitch_max - x_pitch_min < 5:
        warnings.warn(f"Pitch X dimension x_pitch_min={x_pitch_min} to x_pitch_max={x_pitch_max} is very small. Make sure your coordinates are in meters.")
    if y_pitch_max - y_pitch_min < 5:
        warnings.warn(f"Pitch Y dimension y_pitch_min={y_pitch_min} to y_pitch_max={y_pitch_max} is very small. Make sure your coordinates are in meters.")
    if x_pitch_max - x_pitch_min > 500:
        warnings.warn(f"Pitch X dimension x_pitch_min={x_pitch_min} to x_pitch_max={x_pitch_max} is very big. Make sure your coordinates are in meters.")
    if y_pitch_max - y_pitch_min > 500:
        warnings.warn(f"Pitch Y dimension y_pitch_min={y_pitch_min} to y_pitch_max={y_pitch_max} is very big. Make sure your coordinates are in meters.")


def _check_ball_in_tracking_data(df_tracking, tracking_player_col, ball_tracking_player_id):
    if ball_tracking_player_id not in df_tracking[tracking_player_col].values:
        raise ValueError(f"Ball flag ball_tracking_player_id='{ball_tracking_player_id}' does not exist in column df_tracking['{tracking_player_col}']. Make sure to pass the correct identifier of the ball with the parameter 'ball_tracking_player_id'")


def _get_unique_frame_col(df_passes):
    return np.arange(df_passes.shape[0])


def _get_matrix_coordinates(
    df_tracking, frame_col="frame_id", player_col="player_id", ball_player_id="ball", team_col="team_id",
    controlling_team_col="team_in_possession", x_col="x", y_col="y", vx_col="vx", vy_col="vy"
):
    """
    Convert tracking data from a DataFrame to numpy matrices as used within this package to compute the passing model.

    >>> df_tracking = pd.DataFrame({"frame_id": [5, 5, 6, 6, 5, 6], "player_id": ["A", "B", "A", "B", "ball", "ball"], "team_id": ["H", "A", "H", "A", None, None], "team_in_possession": ["H", "H", "H", "H", "H", "H"], "x": [1.0, 2, 3, 4, 5, 6], "y": [5.0, 6, 7, 8, 9, 10], "vx": [9.0, 10, 11, 12, 13, 14], "vy": [13.0, 14, 15, 16, 17, 18]})
    >>> df_tracking
       frame_id player_id team_id team_in_possession    x     y    vx    vy
    0         5         A       H                  H  1.0   5.0   9.0  13.0
    1         5         B       A                  H  2.0   6.0  10.0  14.0
    2         6         A       H                  H  3.0   7.0  11.0  15.0
    3         6         B       A                  H  4.0   8.0  12.0  16.0
    4         5      ball    None                  H  5.0   9.0  13.0  17.0
    5         6      ball    None                  H  6.0  10.0  14.0  18.0
    >>> PLAYER_POS, BALL_POS, players, player_teams, controlling_teams, frame_to_index, player_to_index = _get_matrix_coordinates(df_tracking)
    >>> PLAYER_POS, PLAYER_POS.shape
    (array([[[ 1.,  5.,  9., 13.],
            [ 2.,  6., 10., 14.]],
    <BLANKLINE>
           [[ 3.,  7., 11., 15.],
            [ 4.,  8., 12., 16.]]]), (2, 2, 4))
    >>> BALL_POS, BALL_POS.shape
    (array([[ 5.,  9., 13., 17.],
           [ 6., 10., 14., 18.]]), (2, 4))
    >>> players, players.shape
    (array(['A', 'B'], dtype=object), (2,))
    >>> player_teams, player_teams.shape
    (array(['H', 'A'], dtype=object), (2,))
    >>> controlling_teams, controlling_teams.shape
    (array(['H', 'H'], dtype=object), (2,))
    >>> frame_to_index
    {5: 0, 6: 1}
    """
    if controlling_team_col not in df_tracking.columns:
        raise KeyError(f"Tracking data does not contain column '{controlling_team_col}'")
    if not set(df_tracking[team_col].unique()).issubset(set(df_tracking[team_col].unique())):
        raise ValueError(f"Tracking data contains teams that are not present in the controlling team column (team_col={team_col}, controlling_team_col{controlling_team_col})")

    df_tracking = df_tracking.sort_values(by=[frame_col, team_col])

    i_player = df_tracking[player_col] != ball_player_id

    df_players = df_tracking.loc[i_player].pivot(
        index=frame_col, columns=player_col, values=[x_col, y_col, vx_col, vy_col]
    )
    F = df_players.shape[0]  # number of frames
    C = 4  # number of coordinates per player
    P = df_tracking.loc[i_player, player_col].nunique()  # number of players

    dfp = df_players.stack(level=1, dropna=False)

    PLAYER_POS = dfp.values.reshape(F, P, C)
    frame_to_index = {frame: i for i, frame in enumerate(df_players.index)}
    player_to_index = {player: i for i, player in enumerate(df_players.columns.get_level_values(1).unique())}

    players = np.array(df_players.columns.get_level_values(1).unique())  # P
    player2team = df_tracking.loc[i_player, [player_col, team_col]].drop_duplicates().set_index(player_col)[team_col]
    player_teams = player2team.loc[players].values

    df_ball = df_tracking.loc[~i_player].set_index(frame_col)[[x_col, y_col, vx_col, vy_col]]
    BALL_POS = df_ball.values  # F x C

    controlling_teams = df_tracking.groupby(frame_col)[controlling_team_col].first().values

    F = PLAYER_POS.shape[0]
    assert F == BALL_POS.shape[0]
    assert F == controlling_teams.shape[0], f"Dimension F is {F} (from PLAYER_POS: {PLAYER_POS.shape}), but passer_team shape is {controlling_teams.shape}"
    P = PLAYER_POS.shape[1]
    assert P == player_teams.shape[0]
    assert P == players.shape[0]
    assert PLAYER_POS.shape[2] >= 4  # >= or = ?
    assert BALL_POS.shape[1] >= 2  # ...

    return PLAYER_POS, BALL_POS, players, player_teams, controlling_teams, frame_to_index, player_to_index


def per_object_frameify_tracking_data(
    df_tracking,
    frame_col,
    coordinate_cols,  # P x C
    players,  # P
    player_to_team,
    new_coordinate_cols=("x", "y", "vx", "vy"),  # C
    new_player_col="player_id",
    new_team_col="team_id"
):
    """
    Convert tracking data with '1 row per frame' into '1 row per frame + player' format

    >>> df_tracking = pd.DataFrame({"frame_id": [0, 1], "A_x": [1.2, 1.3], "A_y": [-5.1, -4.9], "B_x": [15.0, 15.0], "B_y": [0.0, 0.1]})
    >>> df_tracking
       frame_id  A_x  A_y   B_x  B_y
    0         0  1.2 -5.1  15.0  0.0
    1         1  1.3 -4.9  15.0  0.1
    >>> per_object_frameify_tracking_data(df_tracking, "frame_id", [["A_x", "A_y"], ["B_x", "B_y"]], ["Player A", "Player B"], {"Player A": "Home", "Player B": "Guest"}, ["x", "y"])
       frame_id     x    y player_id team_id
    0         0   1.2 -5.1  Player A    Home
    1         1   1.3 -4.9  Player A    Home
    2         0  15.0  0.0  Player B   Guest
    3         1  15.0  0.1  Player B   Guest
    """
    dfs_player = []
    for player_nr, player in enumerate(players):
        coordinate_cols_player = coordinate_cols[player_nr]
        df_player = df_tracking[[frame_col] + coordinate_cols_player]
        df_player = df_player.rename(columns={coord_col: new_coord_col for coord_col, new_coord_col in zip(coordinate_cols_player, new_coordinate_cols)})
        df_player[new_player_col] = player
        df_player[new_team_col] = player_to_team.get(player, None)
        dfs_player.append(df_player)

    df_player = pd.concat(dfs_player, axis=0)

    remaining_cols = [col for col in df_tracking.columns if col not in [frame_col] + [col for col_list in coordinate_cols for col in col_list]]

    return df_player.merge(df_tracking[[frame_col] + remaining_cols], on=frame_col, how="left")


def get_pass_velocity(
    df_passes, df_tracking_ball, event_frame_col="frame_id", tracking_frame_col="frame_id",
    n_frames_after_pass_for_v0=_DEFAULT_N_FRAMES_AFTER_PASS_FOR_V0, fallback_v0=_DEFAULT_FALLBACK_V0,
    tracking_vx_col="vx", tracking_vy_col="vy", tracking_v_col=None,
):
    """
    Add initial velocity to passes according to the first N frames of ball tracking data after the pass

    >>> df_passes = pd.DataFrame({"frame_id": [0, 3]})
    >>> df_tracking = pd.DataFrame({"frame_id": [0, 1, 2, 3, 4, 5, 6], "v": [0.5] * 5 + [1] * 2})
    >>> df_passes
       frame_id
    0         0
    1         3
    >>> df_tracking
       frame_id    v
    0         0  0.5
    1         1  0.5
    2         2  0.5
    3         3  0.5
    4         4  0.5
    5         5  1.0
    6         6  1.0
    >>> df_passes["v0"] = get_pass_velocity(df_passes, df_tracking, tracking_v_col="v", n_frames_after_pass_for_v0=3)
    >>> df_passes
       frame_id        v0
    0         0  0.500000
    1         3  0.666667
    """
    df_passes = df_passes.copy()
    df_tracking_ball = df_tracking_ball.copy()
    pass_nr_col = get_unused_column_name(df_passes.columns, "pass_nr")
    frame_end_col = get_unused_column_name(df_passes.columns, "frame_end")
    ball_velocity_col = get_unused_column_name(df_tracking_ball.columns, "ball_velocity")

    df_passes[pass_nr_col] = df_passes.index
    df_tracking_ball = df_tracking_ball.merge(df_passes[[event_frame_col, pass_nr_col]], left_on=tracking_frame_col, right_on=event_frame_col, how="left")

    fr_max = df_tracking_ball[tracking_frame_col].max()
    df_passes[frame_end_col] = np.minimum(df_passes[event_frame_col] + n_frames_after_pass_for_v0 - 1, fr_max)

    all_valid_frame_list = np.concatenate([np.arange(start, end + 1) for start, end in zip(df_passes[event_frame_col], df_passes[frame_end_col])])

    df_tracking_ball_v0 = df_tracking_ball[df_tracking_ball[tracking_frame_col].isin(all_valid_frame_list)].copy()
    df_tracking_ball_v0[pass_nr_col] = df_tracking_ball_v0[pass_nr_col].ffill()
    if tracking_v_col is not None:
        df_tracking_ball_v0[ball_velocity_col] = df_tracking_ball_v0[tracking_v_col]
    else:
        df_tracking_ball_v0[ball_velocity_col] = np.sqrt(df_tracking_ball_v0[tracking_vx_col] ** 2 + df_tracking_ball_v0[tracking_vy_col] ** 2)

    dfg_v0 = df_tracking_ball_v0.groupby(pass_nr_col)[ball_velocity_col].mean()

    v0 = df_passes[pass_nr_col].map(dfg_v0)
    v0 = v0.fillna(fallback_v0)  # Set a reasonable default if no ball data was available during the first N frames
    return v0


def get_das_gained(
    df_passes, df_tracking,

    # Event data schema - pass these parameters according to your data
    event_frame_col="frame_id",
    event_target_frame_col="target_frame_id",
    event_success_col="pass_outcome",
    event_start_x_col="x", event_start_y_col="y", event_target_x_col="x_target", event_target_y_col="y_target",
    event_team_col="team_id", event_receiver_team_col="receiver_team_id",

    # Tracking data schema - pass these parameters according to your data
    tracking_frame_col="frame_id", tracking_player_col="player_id", tracking_team_col="team_id",
    tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy",
    tracking_team_in_possession_col="team_in_possession",
    ball_tracking_player_id="ball", tracking_attacking_direction_col=None,
    tracking_period_col=_unset, tracking_passer_to_exclude_col=None,

    # Pitch dimensions
    x_pitch_min=-52.5, x_pitch_max=52.5, y_pitch_min=-34, y_pitch_max=34,

    # Options
    infer_attacking_direction=True,
    use_event_coordinates_as_ball_position=True,
    use_event_team_as_team_in_possession=True,
    danger_weight=2,
    chunk_size=50,
    return_cropped_result=False,
    additional_fields_to_return=(
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
        "player_poss_density"
    ),  # Set to None to speed up calculation

    # DAS Parameters
    n_angles=_DEFAULT_N_ANGLES_FOR_DAS,
    phi_offset=_DEFAULT_PHI_OFFSET,
    n_v0=_DEFAULT_N_V0_FOR_DAS,
    v0_min=_DEFAULT_V0_MIN_FOR_DAS,
    v0_max=_DEFAULT_V0_MAX_FOR_DAS,

    # Simulation parameters
    pass_start_location_offset=_DEFAULT_PASS_START_LOCATION_OFFSET,
    time_offset_ball=_DEFAULT_TIME_OFFSET_BALL,
    radial_gridsize=_DEFAULT_RADIAL_GRIDSIZE,
    b0=_DEFAULT_B0,
    b1=_DEFAULT_B1,
    player_velocity=_DEFAULT_PLAYER_VELOCITY,
    keep_inertial_velocity=_DEFAULT_KEEP_INERTIAL_VELOCITY,
    use_max=_DEFAULT_USE_MAX,
    v_max=_DEFAULT_V_MAX,
    a_max=_DEFAULT_A_MAX,
    inertial_seconds=_DEFAULT_INERTIAL_SECONDS,
    tol_distance=_DEFAULT_TOL_DISTANCE,
    use_approx_two_point=_DEFAULT_USE_APPROX_TWO_POINT,
    v0_prob_aggregation_mode=_DEFAULT_V0_PROB_AGGREGATION_MODE,
    normalize=_DEFAULT_NORMALIZE,
    use_efficient_sigmoid=_DEFAULT_USE_EFFICIENT_SIGMOID,
):
    _check_presence_of_required_columns(df_passes, "df_passes", ["event_success_col", "event_frame_col", "event_target_frame_col"], [event_success_col, event_frame_col, event_target_frame_col])
    _check_presence_of_required_columns(df_tracking, "df_tracking", ["tracking_frame_col", "tracking_x_col", "tracking_y_col", "tracking_player_col", "tracking_team_col"], [tracking_frame_col, tracking_x_col, tracking_y_col, tracking_player_col, tracking_team_col])
    _check_tracking_coordinate_ranges(df_tracking, tracking_x_col, tracking_y_col)
    _check_pitch_dimensions(x_pitch_min, x_pitch_max, y_pitch_min, y_pitch_max)
    _check_ball_in_tracking_data(df_tracking, tracking_player_col, ball_tracking_player_id)

    df_passes = df_passes.copy()
    df_passes[event_success_col] = df_passes[event_success_col].astype(int)

    unique_frame_col = get_unused_column_name(df_passes.columns.tolist() + df_tracking.columns.tolist(), "unique_frame")
    df_passes[unique_frame_col] = _get_unique_frame_col(df_passes)

    relevant_frames = set(df_passes[event_frame_col].tolist() + df_passes[event_target_frame_col].tolist())

    df_tracking_passes_and_receptions = df_tracking[df_tracking[tracking_frame_col].isin(relevant_frames)].copy()

    if use_event_coordinates_as_ball_position:
        _check_presence_of_required_columns(df_passes, "df_passes", ["event_start_x_col", "event_start_y_col", "event_target_x_col", "event_target_y_col"], [event_start_x_col, event_start_y_col, event_target_x_col, event_target_y_col], additional_message="Either specify these columns or set 'use_event_coordinates_as_ball_position' to 'False'")

        i_ball = df_tracking_passes_and_receptions[tracking_player_col] == ball_tracking_player_id
        df_tracking_passes_and_receptions.loc[i_ball, :] = _replace_column_values_except_nans(df_tracking_passes_and_receptions.loc[i_ball, :], tracking_frame_col, tracking_x_col, df_passes, event_target_frame_col, event_target_x_col)
        df_tracking_passes_and_receptions.loc[i_ball, :] = _replace_column_values_except_nans(df_tracking_passes_and_receptions.loc[i_ball, :], tracking_frame_col, tracking_y_col, df_passes, event_target_frame_col, event_target_y_col)
        df_tracking_passes_and_receptions.loc[i_ball, :] = _replace_column_values_except_nans(df_tracking_passes_and_receptions.loc[i_ball, :], tracking_frame_col, tracking_x_col, df_passes, event_frame_col, event_start_x_col)
        df_tracking_passes_and_receptions.loc[i_ball, :] = _replace_column_values_except_nans(df_tracking_passes_and_receptions.loc[i_ball, :], tracking_frame_col, tracking_y_col, df_passes, event_frame_col, event_start_y_col)

    if use_event_team_as_team_in_possession:
        _check_presence_of_required_columns(df_passes, "df_passes", ["event_team_col", "event_receiver_team_col"], [event_team_col, event_receiver_team_col], additional_message="Either specify these columns or set 'use_event_team_as_team_in_possession' to 'False' and specify 'tracking_team_in_possession_col' yourself.")
        if (df_passes[event_success_col] != (df_passes[event_team_col] == df_passes[event_receiver_team_col])).any():
            warnings.warn(f"Success column '{event_success_col}' is not consistent with team columns '{event_team_col}' == '{event_receiver_team_col}'.")

        df_tracking_passes_and_receptions[tracking_team_in_possession_col] = None
        df_tracking_passes_and_receptions = _replace_column_values_except_nans(df_tracking_passes_and_receptions, tracking_frame_col, tracking_team_in_possession_col, df_passes, event_frame_col, event_team_col)
        df_tracking_passes_and_receptions = _replace_column_values_except_nans(df_tracking_passes_and_receptions, tracking_frame_col, tracking_team_in_possession_col, df_passes, event_target_frame_col, event_receiver_team_col)

    ret = get_dangerous_accessible_space(
        df_tracking_passes_and_receptions,
        frame_col=tracking_frame_col, player_col=tracking_player_col, team_col=tracking_team_col,
        x_col=tracking_x_col, y_col=tracking_y_col, vx_col=tracking_vx_col, vy_col=tracking_vy_col,
        team_in_possession_col=tracking_team_in_possession_col, ball_player_id=ball_tracking_player_id,
        attacking_direction_col=tracking_attacking_direction_col, period_col=tracking_period_col,
        passer_to_exclude_col=tracking_passer_to_exclude_col,
        x_pitch_min=x_pitch_min, x_pitch_max=x_pitch_max, y_pitch_min=y_pitch_min, y_pitch_max=y_pitch_max,

        # Options
        infer_attacking_direction=infer_attacking_direction,
        danger_weight=danger_weight,
        chunk_size=chunk_size,
        return_cropped_result=return_cropped_result,
        additional_fields_to_return=additional_fields_to_return,

        # DAS Parameters
        n_angles=n_angles,
        phi_offset=phi_offset,
        n_v0=n_v0,
        v0_min=v0_min,
        v0_max=v0_max,

        # Simulation parameters
        pass_start_location_offset=pass_start_location_offset,
        time_offset_ball=time_offset_ball,
        radial_gridsize=radial_gridsize,
        b0=b0,
        b1=b1,
        player_velocity=player_velocity,
        keep_inertial_velocity=keep_inertial_velocity,
        use_max=use_max,
        v_max=v_max,
        a_max=a_max,
        inertial_seconds=inertial_seconds,
        tol_distance=tol_distance,
        use_approx_two_point=use_approx_two_point,
        v0_prob_aggregation_mode=v0_prob_aggregation_mode,
        normalize=normalize,
        use_efficient_sigmoid=use_efficient_sigmoid,
    )
    df_tracking_passes_and_receptions["DAS"] = ret.das
    df_tracking_passes_and_receptions["AS"] = ret.acc_space
    df_tracking_passes_and_receptions["frame_index"] = ret.frame_index

    fr2das = df_tracking_passes_and_receptions[[tracking_frame_col, "DAS"]].set_index(tracking_frame_col)["DAS"].to_dict()
    fr2as = df_tracking_passes_and_receptions[[tracking_frame_col, "AS"]].set_index(tracking_frame_col)["AS"].to_dict()
    fr2index = df_tracking_passes_and_receptions[[tracking_frame_col, "frame_index"]].set_index(tracking_frame_col)["frame_index"].to_dict()

    das = df_passes[event_frame_col].map(fr2das)
    acc_space = df_passes[event_frame_col].map(fr2as)
    frame_index = df_passes[event_frame_col].map(fr2index)
    target_frame_index = df_passes[event_target_frame_col].map(fr2index)
    das_reception = df_passes[event_target_frame_col].map(fr2das)
    acc_space_reception = df_passes[event_target_frame_col].map(fr2as)

    # Unsuccessful passes = All (D)AS is lost.
    i_unsuccessful = df_passes[event_success_col] == 0
    das_reception[i_unsuccessful] = 0
    acc_space_reception[i_unsuccessful] = 0

    res = ReturnValueDASGained(
        acc_space=acc_space,
        das=das,
        acc_space_reception=acc_space_reception,
        das_reception=das_reception,
        as_gained=acc_space_reception - acc_space, das_gained=das_reception - das,
        simulation_result=ret.simulation_result, frame_index=frame_index, target_frame_index=target_frame_index
    )
    return res


def get_expected_pass_completion(
    df_passes, df_tracking,

    # Event data schema - pass these parameters according to your data
    event_frame_col="frame_id",
    event_player_col="player_id",
    event_team_col="team_id",
    event_start_x_col="x", event_start_y_col="y", event_end_x_col="x_target", event_end_y_col="y_target",
    event_v0_col=None,

    # Tracking data schema - pass these parameters according to your data
    tracking_frame_col="frame_id",
    tracking_player_col="player_id",
    tracking_team_col="team_id",
    tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy", tracking_v_col=None,
    tracking_team_in_possession_col=None,
    ball_tracking_player_id="ball",

    # Pitch dimensions
    x_pitch_min=-52.5, x_pitch_max=52.5, y_pitch_min=-34, y_pitch_max=34,

    # Options
    clip_to_pitch=True,  # Whether to calculate xC as aggregated interception probability at the pitch boundary or at the end of the simulation (beyond pitch boundary)
    use_event_coordinates_as_ball_position=False,
    use_event_team_as_team_in_possession=True,
    chunk_size=50,
    additional_fields_to_return=("attack_cum_prob", "attack_cum_poss", "attack_prob_density", "attack_poss_density", "defense_cum_prob", "defense_cum_poss", "defense_prob_density", "defense_poss_density", "cum_p0", "p0_density", "player_cum_prob", "player_cum_poss", "player_prob_density", "player_poss_density"),  # Set to None to speed up calculation

    # xC Parameters
    n_frames_after_pass_for_v0=5,
    fallback_v0=10,
    exclude_passer=True,
    use_poss=_DEFAULT_USE_POSS_FOR_XC,
    use_fixed_v0=_DEFAULT_USE_FIXED_V0_FOR_XC,
    v0_min=_DEFAULT_V0_MIN_FOR_XC,
    v0_max=_DEFAULT_V0_MAX_FOR_XC,
    n_v0=_DEFAULT_N_V0_FOR_XC,

    # Core model parameters
    pass_start_location_offset=_DEFAULT_PASS_START_LOCATION_OFFSET,
    time_offset_ball=_DEFAULT_TIME_OFFSET_BALL,
    radial_gridsize=_DEFAULT_RADIAL_GRIDSIZE,
    b0=_DEFAULT_B0,
    b1=_DEFAULT_B1,
    player_velocity=_DEFAULT_PLAYER_VELOCITY,
    keep_inertial_velocity=_DEFAULT_KEEP_INERTIAL_VELOCITY,
    use_max=_DEFAULT_USE_MAX,
    v_max=_DEFAULT_V_MAX,
    a_max=_DEFAULT_A_MAX,
    inertial_seconds=_DEFAULT_INERTIAL_SECONDS,
    tol_distance=_DEFAULT_TOL_DISTANCE,
    use_approx_two_point=_DEFAULT_USE_APPROX_TWO_POINT,
    v0_prob_aggregation_mode=_DEFAULT_V0_PROB_AGGREGATION_MODE,
    normalize=_DEFAULT_NORMALIZE,
    use_efficient_sigmoid=_DEFAULT_USE_EFFICIENT_SIGMOID,
):
    """
    Calculate Expected Pass Completion (xC) for the given passes, using the given tracking data.    

    >>> pd.set_option("display.max_columns", None)
    >>> pd.set_option("display.expand_frame_repr", False)
    >>> import accessible_space.tests.resources as res
    >>> df_passes, df_tracking = res.df_passes, res.df_tracking
    >>> df_passes
       frame_id  target_frame_id player_id receiver_id team_id     x     y  x_target  y_target  pass_outcome receiver_team_id         event_string
    0         0                6         A           B    Home  -0.1   0.0        20        30             1             Home   0: Pass A -> B (1)
    1         6                9         B           X    Home  25.0  30.0        15        30             0             Away   6: Pass B -> X (0)
    2        14               16         C           Y    Home -13.8  40.1        49        -1             0             Away  14: Pass C -> Y (0)
    >>> result = get_expected_pass_completion(df_passes, df_tracking, tracking_frame_col="frame_id", event_frame_col="frame_id", tracking_player_col="player_id", tracking_team_col="team_id", ball_tracking_player_id="ball", tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy", event_start_x_col="x", event_start_y_col="y", event_end_x_col="x_target", event_end_y_col="y_target", event_team_col="team_id", event_player_col="player_id")
    >>> df_passes["xC"], df_passes["frame_index"], simulation_result = result.xc, result.event_frame_index, result.simulation_result
    >>> df_passes
       frame_id  target_frame_id player_id receiver_id team_id     x     y  x_target  y_target  pass_outcome receiver_team_id         event_string        xC  frame_index
    0         0                6         A           B    Home  -0.1   0.0        20        30             1             Home   0: Pass A -> B (1)  0.460387            0
    1         6                9         B           X    Home  25.0  30.0        15        30             0             Away   6: Pass B -> X (0)  0.944628            1
    2        14               16         C           Y    Home -13.8  40.1        49        -1             0             Away  14: Pass C -> Y (0)  0.099345            2
    >>> simulation_result.attack_cum_prob[df_passes["frame_index"].iloc[0], 0, -1]
    0.4833537059537944
    """
    _check_presence_of_required_columns(df_tracking, "df_tracking", column_names=["tracking_x_col", "tracking_y_col", "tracking_vx_col", "tracking_vy_col", "tracking_frame_col", "tracking_team_col", "tracking_player_col"], column_values=[tracking_x_col, tracking_y_col, tracking_vx_col, tracking_vy_col, tracking_frame_col, tracking_team_col, tracking_player_col])
    _check_presence_of_required_columns(df_passes, "df_passes", column_names=["event_frame_col", "event_start_x_col", "event_start_y_col", "event_end_x_col", "event_end_y_col", "event_team_col", "event_player_col"], column_values=[event_frame_col, event_start_x_col, event_start_y_col, event_end_x_col, event_end_y_col, event_team_col, event_player_col])
    _check_tracking_coordinate_ranges(df_tracking, tracking_x_col, tracking_y_col)
    _check_pitch_dimensions(x_pitch_min, x_pitch_max, y_pitch_min, y_pitch_max)

    if not set(df_passes[event_team_col].unique()).issubset(set(df_tracking[tracking_team_col].unique())):
        warnings.warn(f"Teams in passes data do not match teams in tracking data: {set(df_passes[event_team_col].unique())} is not a subset of {set(df_tracking[tracking_team_col].unique())}")

    if not use_event_team_as_team_in_possession:
        if tracking_team_in_possession_col is None:
            raise ValueError("Lacking information about the ball-possessing team in the tracking data. Either specify 'tracking_team_in_possession_col' or set 'use_event_team_as_ball_possessing_team'='True'")
        if tracking_team_in_possession_col not in df_tracking.columns:
            raise ValueError(f"Column 'tracking_team_in_possession_col'='{tracking_team_in_possession_col}' not found in tracking data while 'use_event_team_as_ball_possessing_team' is 'False'. Specify the correct 'tracking_team_in_possession_col' or set 'use_event_team_as_ball_possessing_team' to 'True'.")

    _check_ball_in_tracking_data(df_tracking, tracking_player_col, ball_tracking_player_id)

    if exclude_passer:
        if not set(df_tracking[tracking_player_col]).issuperset(set(df_passes[event_player_col])):
            raise ValueError("Tracking data does not contain all players that are making passes.")

    if not set(df_passes[event_frame_col]).issubset(set(df_tracking[tracking_frame_col])):
        raise ValueError(f"Pass frames are not present in tracking data: {set(df_passes[event_frame_col])} not subset of {set(df_tracking[tracking_frame_col])}")

    df_tracking = df_tracking.copy()
    df_passes = df_passes.copy()

    unique_frame_col = get_unused_column_name(df_passes.columns.tolist() + df_tracking.columns.tolist(), "unique_frame")
    df_passes[unique_frame_col] = _get_unique_frame_col(df_passes)

    df_tracking_passes = df_passes[[event_frame_col, unique_frame_col]].merge(df_tracking, left_on=event_frame_col, right_on=tracking_frame_col, how="left")
    # if use_event_coordinates_as_ball_position:
    #     _check_presence_of_required_columns(df_passes, ["event_start_x_col", "event_start_y_col"], [event_start_x_col, event_start_y_col])
    #
    #     df_tracking_passes = df_tracking_passes.set_index(unique_frame_col)
    #     df_passes_copy = df_passes.copy().set_index(unique_frame_col)
    #
    #     df_tracking_passes.loc[df_tracking_passes[tracking_player_col] == ball_tracking_player_id, tracking_x_col] = df_passes_copy[event_start_x_col]
    #     df_tracking_passes.loc[df_tracking_passes[tracking_player_col] == ball_tracking_player_id, tracking_y_col] = df_passes_copy[event_start_y_col]
    #     df_tracking_passes = df_tracking_passes.reset_index()

    if use_event_coordinates_as_ball_position:
        _check_presence_of_required_columns(df_passes, "df_passes", ["event_start_x_col", "event_start_y_col", "event_target_x_col", "event_target_y_col"], [event_start_x_col, event_start_y_col], additional_message="Either specify these columns or set 'use_event_coordinates_as_ball_position' to 'False'")
        i_ball = df_tracking_passes[tracking_player_col] == ball_tracking_player_id
        df_tracking_passes.loc[i_ball, :] = _replace_column_values_except_nans(df_tracking_passes.loc[i_ball, :], tracking_frame_col, tracking_x_col, df_passes, event_frame_col, event_start_x_col)
        df_tracking_passes.loc[i_ball, :] = _replace_column_values_except_nans(df_tracking_passes.loc[i_ball, :], tracking_frame_col, tracking_y_col, df_passes, event_frame_col, event_start_y_col)

    if use_event_team_as_team_in_possession:
        tracking_team_in_possession_col = get_unused_column_name(df_tracking_passes.columns, "team_in_possession")
        df_tracking_passes[tracking_team_in_possession_col] = None
        df_tracking_passes = _replace_column_values_except_nans(df_tracking_passes, tracking_frame_col, tracking_team_in_possession_col, df_passes, event_frame_col, event_team_col)

    PLAYER_POS, BALL_POS, players, player_teams, _, unique_frame_to_index, player_to_index = _get_matrix_coordinates(
        df_tracking_passes, frame_col=unique_frame_col, player_col=tracking_player_col,
        ball_player_id=ball_tracking_player_id, team_col=tracking_team_col, x_col=tracking_x_col, y_col=tracking_y_col,
        vx_col=tracking_vx_col, vy_col=tracking_vy_col, controlling_team_col=tracking_team_in_possession_col,
    )

    # 2. Add v0 to passes
    if event_v0_col is None:
        event_v0_col = get_unused_column_name(df_passes.columns, "v0")
        df_passes[event_v0_col] = get_pass_velocity(
            df_passes, df_tracking[df_tracking[tracking_player_col] == ball_tracking_player_id],
            event_frame_col=event_frame_col, tracking_frame_col=tracking_frame_col,
            n_frames_after_pass_for_v0=n_frames_after_pass_for_v0, fallback_v0=fallback_v0, tracking_vx_col=tracking_vx_col,
            tracking_vy_col=tracking_vy_col, tracking_v_col=tracking_v_col
        )
    if use_fixed_v0:
        v0_grid = np.linspace(start=v0_min, stop=v0_max, num=round(n_v0))[np.newaxis, :].repeat(df_passes.shape[0], axis=0)  # F x V0
    else:
        v0_grid = df_passes[event_v0_col].values[:, np.newaxis]  # F x V0=1, only simulate actual passing speed

    # 3. Add angle to passes
    phi_col = get_unused_column_name(df_passes.columns, "phi")
    df_passes[phi_col] = np.arctan2(df_passes[event_end_y_col] - df_passes[event_start_y_col], df_passes[event_end_x_col] - df_passes[event_start_x_col])
    phi_grid = df_passes[phi_col].values[:, np.newaxis]  # F x PHI

    # 4. Extract player team info
    passer_teams = df_passes[event_team_col].values  # F
    player_teams = np.array(player_teams)  # P
    if exclude_passer:
        passers_to_exclude = df_passes[event_player_col].values  # F
    else:
        passers_to_exclude = None

    # 5. Simulate passes to get expected completion
    fields_to_return = ["attack_cum_poss"] if use_poss else ["attack_cum_prob"]
    if additional_fields_to_return is None:
        additional_fields_to_return = []
    fields_to_return = list(set(fields_to_return + list(additional_fields_to_return)))
    simulation_result = simulate_passes_chunked(
        # xC parameters
        PLAYER_POS, BALL_POS, phi_grid, v0_grid, passer_teams, player_teams, players,
        passers_to_exclude=passers_to_exclude,

        fields_to_return=fields_to_return,

        # Core model parameters
        pass_start_location_offset=pass_start_location_offset,
        time_offset_ball=time_offset_ball,
        radial_gridsize=radial_gridsize,
        b0=b0,
        b1=b1,
        player_velocity=player_velocity,
        keep_inertial_velocity=keep_inertial_velocity,
        use_max=use_max,
        v_max=v_max,
        a_max=a_max,
        inertial_seconds=inertial_seconds,
        tol_distance=tol_distance,
        use_approx_two_point=use_approx_two_point,
        v0_prob_aggregation_mode=v0_prob_aggregation_mode,
        normalize=normalize,
        use_efficient_sigmoid=use_efficient_sigmoid,

        chunk_size=chunk_size,
        x_pitch_min=x_pitch_min, x_pitch_max=x_pitch_max, y_pitch_min=y_pitch_min, y_pitch_max=y_pitch_max,
    )

    def last_non_nan_value(arr):
        return arr[::-1][~np.isnan(arr[::-1])][0]

    if clip_to_pitch:
        cropped_result = clip_simulation_result_to_pitch(simulation_result, x_pitch_min=x_pitch_min, x_pitch_max=x_pitch_max, y_pitch_min=y_pitch_min, y_pitch_max=y_pitch_max)
        xc_field = cropped_result.attack_cum_poss if use_poss else cropped_result.attack_cum_prob
        xc_field = xc_field[:, 0, :]  # F x PHI x T ---> F x T
        xc = []
        for i_f in range(xc_field.shape[0]):
            xc.append(last_non_nan_value(xc_field[i_f, :]))
        xc = np.array(xc)
    else:
        xc = simulation_result.attack_cum_poss if use_poss else simulation_result.attack_cum_prob
        xc = xc[:, 0, -1]

    if np.any(np.isnan(xc)):
        warnings.warn("Some xC values are NaN.")

    # if use_poss:
    #     xc = cropped_result.attack_cum_poss[:, 0, -1]  # F x PHI x T ---> F
    # else:
    #     xc = cropped_result.attack_cum_prob[:, 0, -1]  # F x PHI x T ---> F

    frame_to_unique_frame = df_passes[[event_frame_col, unique_frame_col]].drop_duplicates(event_frame_col, keep="first").set_index(event_frame_col)[unique_frame_col].to_dict()
    event_frame_index = df_passes[unique_frame_col].map(unique_frame_to_index)
    tracking_frame_index = df_tracking[tracking_frame_col].map(frame_to_unique_frame)
    tracking_player_index = df_tracking[tracking_player_col].map(player_to_index)

    return ReturnValueXC(xc, event_frame_index, tracking_frame_index, tracking_player_index, simulation_result)


def _get_danger(dist_to_goal, opening_angle):
    """
    Simple prefit xG model

    >>> _get_danger(20, np.pi/2)
    0.058762795476666185
    """
    coefficients = [-0.14447723, 0.40579492]
    intercept = -0.52156283
    logit = intercept + coefficients[0] * dist_to_goal + coefficients[1] * opening_angle
    with np.errstate(over='ignore'):  # overflow leads to inf which is handled gracefully by the division operation
        prob_true = 1 / (1 + np.exp(-logit))
    return prob_true


def get_dangerous_accessible_space(
    # Data
    df_tracking,

    # Tracking schema
    frame_col="frame_id", player_col="player_id", ball_player_id="ball", team_col="team_id", x_col="x",
    y_col="y", vx_col="vx", vy_col="vy", attacking_direction_col=None, period_col=_unset,
    team_in_possession_col="team_in_possession", passer_to_exclude_col=None,

    # Pitch coordinate system
    x_pitch_min=-52.5, x_pitch_max=52.5, y_pitch_min=-34, y_pitch_max=34,

    # Options
    infer_attacking_direction=True,
    danger_weight=2,
    chunk_size=50,
    return_cropped_result=False,
    additional_fields_to_return=("attack_cum_prob", "attack_cum_poss", "attack_prob_density", "attack_poss_density", "defense_cum_prob", "defense_cum_poss", "defense_prob_density", "defense_poss_density", "cum_p0", "p0_density", "player_cum_prob", "player_cum_poss", "player_prob_density", "player_poss_density"),

    # DAS Parameters
    n_angles=_DEFAULT_N_ANGLES_FOR_DAS,
    phi_offset=_DEFAULT_PHI_OFFSET,
    n_v0=_DEFAULT_N_V0_FOR_DAS,
    v0_min=_DEFAULT_V0_MIN_FOR_DAS,
    v0_max=_DEFAULT_V0_MAX_FOR_DAS,

    # Simulation parameters
    pass_start_location_offset=_DEFAULT_PASS_START_LOCATION_OFFSET,
    time_offset_ball=_DEFAULT_TIME_OFFSET_BALL,
    radial_gridsize=_DEFAULT_RADIAL_GRIDSIZE,
    b0=_DEFAULT_B0,
    b1=_DEFAULT_B1,
    player_velocity=_DEFAULT_PLAYER_VELOCITY,
    keep_inertial_velocity=_DEFAULT_KEEP_INERTIAL_VELOCITY,
    use_max=_DEFAULT_USE_MAX,
    v_max=_DEFAULT_V_MAX,
    a_max=_DEFAULT_A_MAX,
    inertial_seconds=_DEFAULT_INERTIAL_SECONDS,
    tol_distance=_DEFAULT_TOL_DISTANCE,
    use_approx_two_point=_DEFAULT_USE_APPROX_TWO_POINT,
    v0_prob_aggregation_mode=_DEFAULT_V0_PROB_AGGREGATION_MODE,
    normalize=_DEFAULT_NORMALIZE,
    use_efficient_sigmoid=_DEFAULT_USE_EFFICIENT_SIGMOID,
):
    """
    >>> pd.set_option("display.max_columns", None)
    >>> pd.set_option("display.expand_frame_repr", False)
    >>> import accessible_space.tests.resources as res
    >>> df_tracking = res.df_tracking
    >>> df_tracking.head(5)
       frame_id player_id team_id    x     y   vx    vy team_in_possession  period_id
    0         0         A    Home -0.1  0.00  0.1  0.05               Home          0
    1         1         A    Home  0.0  0.05  0.1  0.05               Home          0
    2         2         A    Home  0.1  0.10  0.1  0.05               Home          0
    3         3         A    Home  0.2  0.15  0.1  0.05               Home          0
    4         4         A    Home  0.3  0.20  0.1  0.05               Home          0
    >>> ret = get_dangerous_accessible_space(df_tracking, frame_col="frame_id", player_col="player_id", team_col="team_id", ball_player_id="ball", x_col="x", y_col="y", vx_col="vx", vy_col="vy", attacking_direction_col="attacking_direction", period_col="period_id", team_in_possession_col="team_in_possession", infer_attacking_direction=True)
    >>> df_tracking["AS"], df_tracking["DAS"], df_tracking["frame_index"], df_tracking["player_index"] = ret.acc_space, ret.das, ret.frame_index, ret.player_index
    >>> df_tracking.head(5)
       frame_id player_id team_id    x     y   vx    vy team_in_possession  period_id           AS        DAS  frame_index  player_index
    0         0         A    Home -0.1  0.00  0.1  0.05               Home          0  2558.530803  24.369262            0           0.0
    1         1         A    Home  0.0  0.05  0.1  0.05               Home          0  2545.716748  24.118275            1           0.0
    2         2         A    Home  0.1  0.10  0.1  0.05               Home          0  2539.492197  24.160716            2           0.0
    3         3         A    Home  0.2  0.15  0.1  0.05               Home          0  2529.022848  24.016915            3           0.0
    4         4         A    Home  0.3  0.20  0.1  0.05               Home          0  2526.241382  24.098972            4           0.0
    """

    missing_columns = [(parameter_name, col) for parameter_name, col in [("frame_col", frame_col), ("player_col", player_col), ("team_col", team_col), ("x_col", x_col), ("y_col", y_col), ("vx_col", vx_col), ("vy_col", vy_col), ("team_in_possession_col", team_in_possession_col)] if col not in df_tracking.columns]
    if len(missing_columns) > 0:
        raise KeyError(f"""Missing column{'s' if len(missing_columns) > 1 else ''} in tracking data: {', '.join(['='.join([parameter_name, "'" + missing_columns + "'"]) for (parameter_name, missing_columns) in missing_columns])}""")
    if period_col == _unset:
        period_col = None
        if infer_attacking_direction:
            warnings.warn("Inferring attacking direction but 'period_col' is unset. If you have data across multiple halfs, specify 'period_col', otherwise pass 'period_col'=None.", UserWarning)

    _check_ball_in_tracking_data(df_tracking, player_col, ball_player_id)

    df_tracking = df_tracking.copy()

    # Center coordinates
    # x_center = (x_pitch_max + x_pitch_min) / 2
    # y_center = (y_pitch_max + y_pitch_min) / 2
    # df_tracking[x_col] = df_tracking[x_col] - x_center
    # df_tracking[y_col] = df_tracking[y_col] - y_center
    # x_pitch_max_centered = x_pitch_max - x_center
    # x_pitch_min_centered = x_pitch_min - x_center
    # y_pitch_max_centered = y_pitch_max - y_center
    # y_pitch_min_centered = y_pitch_min - y_center

    PLAYER_POS, BALL_POS, players, player_teams, controlling_teams, frame_to_index, player_to_index = _get_matrix_coordinates(
        df_tracking, frame_col=frame_col, player_col=player_col,
        ball_player_id=ball_player_id, team_col=team_col, x_col=x_col, y_col=y_col,
        vx_col=vx_col, vy_col=vy_col, controlling_team_col=team_in_possession_col,
    )
    F = PLAYER_POS.shape[0]
    if passer_to_exclude_col is not None:
        PASSERS_TO_EXCLUDE = df_tracking.drop_duplicates(frame_col)[passer_to_exclude_col].values  # F
        assert F == PASSERS_TO_EXCLUDE.shape[0]
    else:
        PASSERS_TO_EXCLUDE = None

    phi_grid = np.tile(np.linspace(phi_offset, 2*np.pi+phi_offset, n_angles, endpoint=False), (F, 1))  # F x PHI
    v0_grid = np.tile(np.linspace(v0_min, v0_max, n_v0), (F, 1))  # F x V0

    if additional_fields_to_return is None:
        additional_fields_to_return = []
    fields_to_return = list(set(["attack_poss_density"] + list(additional_fields_to_return)))

    simulation_result = simulate_passes_chunked(
        PLAYER_POS, BALL_POS, phi_grid, v0_grid, controlling_teams, player_teams, players,
        passers_to_exclude=PASSERS_TO_EXCLUDE,
        pass_start_location_offset=pass_start_location_offset,
        time_offset_ball=time_offset_ball,
        radial_gridsize=radial_gridsize,
        b0=b0,
        b1=b1,
        player_velocity=player_velocity,
        keep_inertial_velocity=keep_inertial_velocity,
        use_max=use_max,
        v_max=v_max,
        a_max=a_max,
        inertial_seconds=inertial_seconds,
        tol_distance=tol_distance,
        use_approx_two_point=use_approx_two_point,
        v0_prob_aggregation_mode=v0_prob_aggregation_mode,
        normalize=normalize,
        use_efficient_sigmoid=use_efficient_sigmoid,

        fields_to_return=fields_to_return,
        chunk_size=chunk_size,

        x_pitch_min=x_pitch_min, x_pitch_max=x_pitch_max, y_pitch_min=y_pitch_min, y_pitch_max=y_pitch_max,
    )
    if return_cropped_result:
        simulation_result = clip_simulation_result_to_pitch(simulation_result, x_pitch_min=x_pitch_min, x_pitch_max=x_pitch_max, y_pitch_min=y_pitch_min, y_pitch_max=y_pitch_max)

    ### Add danger to simulation result
    # 1. Get attacking direction
    if infer_attacking_direction:
        attacking_direction_col = get_unused_column_name(df_tracking.columns, "attacking_direction")
        df_tracking[attacking_direction_col] = infer_playing_direction(
            df_tracking, team_col=team_col, period_col=period_col, team_in_possession_col=team_in_possession_col, x_col=x_col
        )
    if attacking_direction_col is not None:
        fr2playingdirection = df_tracking[[frame_col, attacking_direction_col]].set_index(frame_col).to_dict()[attacking_direction_col]
        ATTACKING_DIRECTION = np.array([fr2playingdirection[frame] for frame in frame_to_index])  # F
    else:
        ATTACKING_DIRECTION = np.ones(F)  # if no attacking direction is given and we didn't infer it, we assume always left-to-right
        warnings.warn("Neither 'attacking_direction_col' nor 'infer_attacking_direction' are specified, thus we assume always left-to-right to calculate danger and DAS")

    # 2. Calculate danger
    x_center = (x_pitch_max + x_pitch_min) / 2
    y_center = (y_pitch_max + y_pitch_min) / 2
    X_CENTERED = simulation_result.x_grid - x_center  # F x PHI x T
    Y_CENTERED = simulation_result.y_grid - y_center  # F x PHI x T
    X_NORM = X_CENTERED * ATTACKING_DIRECTION[:, np.newaxis, np.newaxis]  # F x PHI x T
    Y_NORM = Y_CENTERED * ATTACKING_DIRECTION[:, np.newaxis, np.newaxis]  # F x PHI x T
    DIST_TO_GOAL = _dist_to_opp_goal(X_NORM, Y_NORM, x_pitch_max - x_center)  # F x PHI x T
    OPENING_ANGLE = _opening_angle_to_goal(X_NORM, Y_NORM)  # F x PHI x T
    DANGER = _get_danger(DIST_TO_GOAL, OPENING_ANGLE)  # F x PHI x T

    # 3. Add danger to simulation result
    dangerous_result = as_dangerous_result(simulation_result, DANGER, danger_weight)

    # Get AS and DAS
    accessible_space = integrate_surfaces(simulation_result, x_pitch_min, x_pitch_max, y_pitch_min, y_pitch_max).attack_poss  # F
    das = integrate_surfaces(dangerous_result, x_pitch_min, x_pitch_max, y_pitch_min, y_pitch_max).attack_poss  # F
    fr2AS = pd.Series(accessible_space, index=df_tracking[frame_col].unique())
    fr2DAS = pd.Series(das, index=df_tracking[frame_col].unique())
    as_series = df_tracking[frame_col].map(fr2AS)
    das_series = df_tracking[frame_col].map(fr2DAS)

    frame_index = df_tracking[frame_col].map(frame_to_index)
    player_index = df_tracking[player_col].map(player_to_index)

    return ReturnValueDAS(as_series, das_series, frame_index, player_index, simulation_result, dangerous_result)


def infer_playing_direction(
    df_tracking, team_col="team_id", period_col="period_id", team_in_possession_col="team_in_possession", x_col="x",
):
    """
    Automatically infer playing direction based on the mean x position of each teams in each period.

    >>> df_tracking = pd.DataFrame({"period": [0, 0, 1, 1], "team_id": ["H", "A", "H", "A"], "team_in_possession": ["H", "H", "A", "A"], "x": [1, 2, 3, 4], "y": [5, 6, 7, 8]})
    >>> df_tracking
       period team_id team_in_possession  x  y
    0       0       H                  H  1  5
    1       0       A                  H  2  6
    2       1       H                  A  3  7
    3       1       A                  A  4  8
    >>> df_tracking["playing_direction"] = infer_playing_direction(df_tracking, team_col="team_id", period_col="period", team_in_possession_col="team_in_possession", x_col="x")
    >>> df_tracking
       period team_id team_in_possession  x  y  playing_direction
    0       0       H                  H  1  5                1.0
    1       0       A                  H  2  6                1.0
    2       1       H                  A  3  7               -1.0
    3       1       A                  A  4  8               -1.0
    """
    _check_presence_of_required_columns(df_tracking, "df_tracking", column_names=["team_col", "team_in_possession_col", "x_col"], column_values=[team_col, team_in_possession_col, x_col])
    if period_col is not None:
        _check_presence_of_required_columns(df_tracking, "df_tracking", ["period_col"], [period_col], "Either specify period_col or set to None if your data has no separate periods.")

    playing_direction = {}
    if period_col is None:
        _period_col = get_unused_column_name(df_tracking.columns, "period_id")
        df_tracking[_period_col] = 0  # assume same period everywhere
    else:
        _period_col = period_col

    for period_id, df_tracking_period in df_tracking.groupby(_period_col):
        x_mean = df_tracking_period.groupby(team_col)[x_col].mean()
        smaller_x_team = x_mean.idxmin()
        greater_x_team = x_mean.idxmax()
        playing_direction[period_id] = {smaller_x_team: 1, greater_x_team: -1}

    new_attacking_direction = pd.Series(index=df_tracking.index, dtype=np.float64)

    for period_id in playing_direction:
        i_period = df_tracking[_period_col] == period_id
        for team_id, direction in playing_direction[period_id].items():
            i_period_team_possession = i_period & (df_tracking[team_in_possession_col] == team_id)
            new_attacking_direction.loc[i_period_team_possession] = direction

    return new_attacking_direction


def plot_expected_completion_surface(simulation_result: SimulationResult, frame_index=0, attribute="attack_poss_density", player_index=None, color="blue", plot_gridpoints=True):  # TODO gridpoints False
    """ Plot a pass completion surface. """
    # if not isinstance(simulation_result, SimulationResult):
    #     if isinstance(simulation_result, ReturnValueDAS) or isinstance(simulation_result, ReturnValueDASGained) or isinstance(simulation_result, ReturnValueXC):
    #         raise ValueError(f"Parameter 'simulation_result' is not of type 'SimulationResult' but of type '{type(simulation_result).__name__}': Did you mean to pass <{type(simulation_result).__name__}>.simulation_result?")
    #     else:
    #         raise ValueError(f"Parameter 'simulation_result' is not of type 'SimulationResult' but of type '{type(simulation_result).__name__}'")

    st.write("x_grid", simulation_result.x_grid.shape)
    st.write("frame_index", frame_index)

    x_grid = simulation_result.x_grid[frame_index, :, :]
    y_grid = simulation_result.y_grid[frame_index, :, :]

    x = np.ravel(x_grid)  # F*PHI*T
    y = np.ravel(y_grid)  # F*PHI*T

    field = getattr(simulation_result, attribute)

    if len(field.shape) == 3:
        p = field[frame_index, :, :]
    elif len(field.shape) == 4:
        if player_index is None:
            raise ValueError("Field is player-level (= 4D) but no player index is given: Pass 'player_index' to plot a player-level field.")
        p = field[frame_index, player_index, :, :]
    else:
        raise ValueError(f"Internal package error: Field {attribute} simulation result to plot is not 3D or 4D")

    z = np.ravel(p)  # F*PHI*T

    areas = 10
    levels = np.linspace(start=0, stop=np.max(z)+0.00001, num=areas + 1, endpoint=True)
    saturations = [x / (areas) for x in range(areas)]
    base_color = matplotlib.colors.to_rgb(color)

    colors = [_adjust_saturation(base_color, s) for s in saturations]

    # Create a triangulation
    triang = matplotlib.tri.Triangulation(x, y)
    cp = plt.tricontourf(x, y, z.T, colors=colors, alpha=0.1, cmap=None, levels=levels)  # Comment in to use [0, 1] scale
    plt.tricontourf(triang, z.T, colors=colors, alpha=0.1, cmap=None, levels=levels)  # Comment in to use [0, 1] scale

    if plot_gridpoints:
        plt.plot(x, y, 'ko', ms=0.5)

    return plt.gcf()
