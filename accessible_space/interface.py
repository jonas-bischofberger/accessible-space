import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri
import matplotlib.colors

from .core import _DEFAULT_PASS_START_LOCATION_OFFSET, _DEFAULT_B0, _DEFAULT_TIME_OFFSET_BALL, _DEFAULT_A_MAX, \
    _DEFAULT_USE_MAX, _DEFAULT_USE_APPROX_TWO_POINT, _DEFAULT_B1, _DEFAULT_PLAYER_VELOCITY, _DEFAULT_V_MAX, \
    _DEFAULT_KEEP_INERTIAL_VELOCITY, _DEFAULT_INERTIAL_SECONDS, _DEFAULT_TOL_DISTANCE, _DEFAULT_RADIAL_GRIDSIZE, \
    simulate_passes_chunked, crop_result_to_pitch, integrate_attacking_surface
from .utility import get_unused_column_name, _dist_to_opp_goal, _opening_angle_to_goal, _adjust_saturation

_DEFAULT_N_FRAMES_AFTER_PASS_FOR_V0 = 3
_DEFAULT_FALLBACK_V0 = 10
_DEFAULT_USE_POSS_FOR_XC = False
_DEFAULT_USE_FIXED_V0_FOR_XC = True
_DEFAULT_V0_MAX_FOR_XC = 15.108273248071049
_DEFAULT_V0_MIN_FOR_XC = 4.835618861117393
_DEFAULT_N_V0_FOR_XC = 6

_DEFAULT_N_ANGLES_FOR_DAS = 60
_DEFAULT_PHI_OFFSET = 0
_DEFAULT_N_V0_FOR_DAS = 20
_DEFAULT_V0_MIN_FOR_DAS = 3
_DEFAULT_V0_MAX_FOR_DAS = 30


def get_matrix_coordinates(
    df_tracking, frame_col="frame_id", player_col="player_id", ball_player_id="ball", team_col="team_id",
    controlling_team_col="ball_possession", x_col="x", y_col="y", vx_col="vx", vy_col="vy"
):
    """
    Convert tracking data from a DataFrame to numpy matrices as used within this package to compute the passing model.

    >>> df_tracking = pd.DataFrame({"frame_id": [5, 5, 6, 6, 5, 6], "player_id": ["A", "B", "A", "B", "ball", "ball"], "team_id": ["H", "A", "H", "A", None, None], "ball_possession": ["H", "H", "H", "H", "H", "H"], "x": [1, 2, 3, 4, 5, 6], "y": [5, 6, 7, 8, 9, 10], "vx": [9, 10, 11, 12, 13, 14], "vy": [13, 14, 15, 16, 17, 18]})
    >>> df_tracking
       frame_id player_id team_id ball_possession  x   y  vx  vy
    0         5         A       H               H  1   5   9  13
    1         5         B       A               H  2   6  10  14
    2         6         A       H               H  3   7  11  15
    3         6         B       A               H  4   8  12  16
    4         5      ball    None               H  5   9  13  17
    5         6      ball    None               H  6  10  14  18
    >>> PLAYER_POS, BALL_POS, players, player_teams, controlling_teams, frame_to_idx = get_matrix_coordinates(df_tracking)
    >>> PLAYER_POS, PLAYER_POS.shape
    (array([[[ 1,  5,  9, 13],
            [ 2,  6, 10, 14]],
    <BLANKLINE>
           [[ 3,  7, 11, 15],
            [ 4,  8, 12, 16]]], dtype=int64), (2, 2, 4))
    >>> BALL_POS, BALL_POS.shape
    (array([[ 5,  9, 13, 17],
           [ 6, 10, 14, 18]], dtype=int64), (2, 4))
    >>> players, players.shape
    (Index(['A', 'B'], dtype='object', name='player_id'), (2,))
    >>> player_teams, player_teams.shape
    (array(['H', 'A'], dtype=object), (2,))
    >>> controlling_teams, controlling_teams.shape
    (array(['H', 'H'], dtype=object), (2,))
    >>> frame_to_idx
    {5: 0, 6: 1}
    """
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
    frame_to_idx = {frame: i for i, frame in enumerate(df_players.index)}

    players = df_players.columns.get_level_values(1).unique()  # P
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

    return PLAYER_POS, BALL_POS, players, player_teams, controlling_teams, frame_to_idx


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
    pass_nr_col = get_unused_column_name(df_passes, "pass_nr")
    frame_end_col = get_unused_column_name(df_passes, "frame_end")
    ball_velocity_col = get_unused_column_name(df_tracking_ball, "ball_velocity")

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


def get_expected_pass_completion(
    df_passes, df_tracking, tracking_frame_col="frame_id", event_frame_col="frame_id",
    tracking_player_col="player_id", tracking_team_col="team_id", ball_tracking_player_id="ball",
    n_frames_after_pass_for_v0=5, fallback_v0=10, tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx",
    tracking_vy_col="vy", tracking_v_col=None, event_start_x_col="x", event_start_y_col="y",
    event_end_x_col="x_target", event_end_y_col="y_target", event_team_col="team_id", event_player_col="player_id",
    tracking_ball_possession_col=None,
    use_event_ball_position=False,
    chunk_size=200,

    # xC Parameters
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
):
    """
    Calculate Expected Pass Completion (xC) for the given passes, using the given tracking data.    

    >>> pd.set_option("display.max_columns", None)
    >>> pd.set_option("display.expand_frame_repr", False)
    >>> import accessible_space.tests.resources as res
    >>> df_passes, df_tracking = res.df_passes, res.df_tracking
    >>> df_passes
       frame_id player_id receiver_id  team_id     x     y  x_target  y_target pass_outcome
    0         0         A           B        0  -0.1   0.0       -10        11   successful
    1         6         B           X        0  -9.6  10.5        15        30       failed
    2        14         C           Y        0 -13.8 -12.9        49        -1       failed
    >>> df_passes["xC"], df_passes["matrix_index"], simulation_result = get_expected_pass_completion(df_passes, df_tracking, tracking_frame_col="frame_id", event_frame_col="frame_id", tracking_player_col="player_id", tracking_team_col="team_id", ball_tracking_player_id="ball", tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy", event_start_x_col="x", event_start_y_col="y", event_end_x_col="x_target", event_end_y_col="y_target", event_team_col="team_id", event_player_col="player_id")
    >>> df_passes
       frame_id player_id receiver_id  team_id     x     y  x_target  y_target pass_outcome        xC  matrix_index
    0         0         A           B        0  -0.1   0.0       -10        11   successful  0.972250           0.0
    1         6         B           X        0  -9.6  10.5        15        30       failed  0.091146           NaN
    2        14         C           Y        0 -13.8 -12.9        49        -1       failed  0.081058           NaN
    >>> simulation_result.poss_density_att.shape
    (3, 1, 50)
    >>> simulation_result.prob_cum_att[int(df_passes["matrix_index"].iloc[0]), 0, -1]
    0.9722499353573427
    """
    df_tracking = df_tracking.copy()
    df_passes = df_passes.copy()

    # 1. Extract player and ball positions at passes
    assert set(df_passes[event_frame_col]).issubset(set(df_tracking[tracking_frame_col]))

    unique_frame_col = get_unused_column_name(df_passes, "unique_frame")
    df_passes[unique_frame_col] = np.arange(df_passes.shape[0])

    df_tracking_passes = df_passes[[event_frame_col, unique_frame_col]].merge(df_tracking, left_on=event_frame_col, right_on=tracking_frame_col, how="left")
    if use_event_ball_position:
        df_tracking_passes = df_tracking_passes.set_index(unique_frame_col)
        df_passes_copy = df_passes.copy().set_index(unique_frame_col)
        df_tracking_passes.loc[df_tracking_passes[tracking_player_col] == ball_tracking_player_id, tracking_x_col] = df_passes_copy[event_start_x_col]
        df_tracking_passes.loc[df_tracking_passes[tracking_player_col] == ball_tracking_player_id, tracking_y_col] = df_passes_copy[event_start_y_col]
        df_tracking_passes = df_tracking_passes.reset_index()

    if tracking_ball_possession_col is None:
        tracking_ball_possession_col = event_team_col

    PLAYER_POS, BALL_POS, players, player_teams, _, frame_to_idx = get_matrix_coordinates(
        df_tracking_passes, frame_col=unique_frame_col, player_col=tracking_player_col,
        ball_player_id=ball_tracking_player_id, team_col=tracking_team_col, x_col=tracking_x_col, y_col=tracking_y_col,
        vx_col=tracking_vx_col, vy_col=tracking_vy_col, controlling_team_col=tracking_ball_possession_col,
    )

    # 2. Add v0 to passes
    v0_col = get_unused_column_name(df_passes, "v0")
    df_passes[v0_col] = get_pass_velocity(
        df_passes, df_tracking[df_tracking[tracking_player_col] == ball_tracking_player_id],
        event_frame_col=event_frame_col, tracking_frame_col=tracking_frame_col,
        n_frames_after_pass_for_v0=n_frames_after_pass_for_v0, fallback_v0=fallback_v0, tracking_vx_col=tracking_vx_col,
        tracking_vy_col=tracking_vy_col, tracking_v_col=tracking_v_col
    )
    if use_fixed_v0:
        v0_grid = np.linspace(start=v0_min, stop=v0_max, num=round(n_v0))[np.newaxis, :].repeat(df_passes.shape[0], axis=0)  # F x V0
    else:
        v0_grid = df_passes[v0_col].values[:, np.newaxis]  # F x V0=1, only simulate actual passing speed

    # 3. Add angle to passes
    phi_col = get_unused_column_name(df_passes, "phi")
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
    simulation_result = simulate_passes_chunked(
        # xC parameters
        PLAYER_POS, BALL_POS, phi_grid, v0_grid, passer_teams, player_teams, players,
        passers_to_exclude=passers_to_exclude,

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

        # Chunk size
        chunk_size=chunk_size,
    )
    if use_poss:
        xc = simulation_result.poss_cum_att[:, 0, -1]  # F x PHI x T ---> F
    else:
        xc = simulation_result.prob_cum_att[:, 0, -1]  # F x PHI x T ---> F

    matrix_index = df_passes[event_frame_col].map(frame_to_idx)

    return xc, matrix_index, simulation_result


def _get_danger(dist_to_goal, opening_angle):
    """
    Simple prefit xG model

    >>> _get_danger(20, np.pi/2)
    0.058762795476666185
    """
    coefficients = [-0.14447723, 0.40579492]
    intercept = -0.52156283
    logit = intercept + coefficients[0] * dist_to_goal + coefficients[1] * opening_angle
    prob_true = 1 / (1 + np.exp(-logit))
    return prob_true


def as_dangerous_result(result, danger, danger_weight=None):
    """
    Convert a simulation result to a dangerous simulation result by multiplying density with danger.

    >>> res = simulate_passes_chunked(np.array([[[0, 0, 0, 0], [50, 0, 0, 0]]]), np.array([[0, 0]]), np.array([[0]]), np.array([[10]]), np.array([0]), np.array([0, 1]), players=np.array(["A", "B"]), passers_to_exclude=np.array(["A"]), radial_gridsize=15)
    >>> res.poss_density_def
    array([[[3.64076555e-05, 6.78000534e-05, 4.92186270e-04, 6.01130886e-02,
             6.00108990e-02, 2.16895102e-03, 1.68588297e-04, 8.77026250e-05,
             5.92672504e-05, 4.47561819e-05]]])
    >>> danger = np.array([[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]])
    >>> dangerous_res = as_dangerous_result(res, danger)
    >>> dangerous_res.poss_density_def
    array([[[3.64076555e-06, 1.35600107e-05, 1.47655881e-04, 2.40452354e-02,
             3.00054495e-02, 1.30137061e-03, 1.18011808e-04, 7.01621000e-05,
             5.33405253e-05, 4.47561819e-05]]])
    """
    weighted_multiplication = lambda x, y, weight=danger_weight: x**weight * y**(1-weight) if weight is not None else x * y
    return result._replace(
        poss_cum_att=None,
        prob_cum_att=None,
        poss_density_att=weighted_multiplication(danger, result.poss_density_att),
        prob_density_att=weighted_multiplication(danger, result.prob_density_att),
        poss_cum_def=None,
        prob_cum_def=None,
        poss_density_def=weighted_multiplication(danger, result.poss_density_def),
        prob_density_def=weighted_multiplication(danger, result.prob_density_def),
    )


def get_dangerous_accessible_space(
    # Data
    df_tracking, tracking_frame_col="frame_id", tracking_player_col="player_id", tracking_team_col="team_id",
    ball_tracking_player_id="ball", tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy",
    attacking_direction_col="attacking_direction", period_col="period_id", possession_team_col="ball_possession",
    infer_attacking_direction=True,

    # Options
    return_cropped_result=False,

    # Parameters
    n_angles=_DEFAULT_N_ANGLES_FOR_DAS,
    phi_offset=_DEFAULT_PHI_OFFSET,
    n_v0=_DEFAULT_N_V0_FOR_DAS,
    v0_min=_DEFAULT_V0_MIN_FOR_DAS,
    v0_max=_DEFAULT_V0_MAX_FOR_DAS,
):
    """
    >>> pd.set_option("display.max_columns", None)
    >>> pd.set_option("display.expand_frame_repr", False)
    >>> import accessible_space.tests.resources as res
    >>> df_tracking = res.df_tracking
    >>> df_tracking["AS"], df_tracking["DAS"], df_tracking["matrix_index"], simulation_result, dangerous_result = get_dangerous_accessible_space(df_tracking, tracking_frame_col="frame_id", tracking_player_col="player_id", tracking_team_col="team_id", ball_tracking_player_id="ball", tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy", attacking_direction_col="attacking_direction", period_col="period_id", possession_team_col="ball_possession", infer_attacking_direction=True)
    >>> df_tracking
         frame_id player_id  team_id    x     y   vx    vy  ball_possession  period_id  attacking_direction           AS       DAS  matrix_index
    0           0         A      0.0 -0.1  0.00  0.1  0.05                0          0                  1.0  4479.436833  2.235315             0
    1           1         A      0.0  0.0  0.05  0.1  0.05                0          0                  1.0  4511.233023  2.253153             1
    2           2         A      0.0  0.1  0.10  0.1  0.05                0          0                  1.0  4502.846128  2.227248             2
    3           3         A      0.0  0.2  0.15  0.1  0.05                0          0                  1.0  4495.100201  2.198680             3
    4           4         A      0.0  0.3  0.20  0.1  0.05                0          0                  1.0  4474.157453  2.109178             4
    ..        ...       ...      ...  ...   ...  ...   ...              ...        ...                  ...          ...       ...           ...
    114        15      ball      NaN  1.5  0.00  0.1  0.00                1          0                 -1.0  1916.019280  0.076284            15
    115        16      ball      NaN  1.6  0.00  0.1  0.00                1          0                 -1.0  1922.945790  0.078280            16
    116        17      ball      NaN  1.7  0.00  0.1  0.00                1          0                 -1.0  1926.031171  0.077382            17
    117        18      ball      NaN  1.8  0.00  0.1  0.00                1          0                 -1.0  1934.867991  0.076520            18
    118        19      ball      NaN  1.9  0.00  0.1  0.00                1          0                 -1.0  1081.484989  0.073956            19
    <BLANKLINE>
    [119 rows x 13 columns]
    """
    PLAYER_POS, BALL_POS, players, player_teams, controlling_teams, frame_to_idx = get_matrix_coordinates(
        df_tracking, frame_col=tracking_frame_col, player_col=tracking_player_col,
        ball_player_id=ball_tracking_player_id, team_col=tracking_team_col, x_col=tracking_x_col, y_col=tracking_y_col,
        vx_col=tracking_vx_col, vy_col=tracking_vy_col, controlling_team_col=possession_team_col,
    )
    F = PLAYER_POS.shape[0]

    phi_grid = np.tile(np.linspace(phi_offset, 2*np.pi+phi_offset, n_angles, endpoint=False), (F, 1))  # F x PHI
    v0_grid = np.tile(np.linspace(v0_min, v0_max, n_v0), (F, 1))  # F x V0

    simulation_result = simulate_passes_chunked(
        PLAYER_POS, BALL_POS, phi_grid, v0_grid, controlling_teams, player_teams, players, passers_to_exclude=None,
    )
    if return_cropped_result:
        simulation_result = crop_result_to_pitch(simulation_result)

    ### Add danger to simulation result
    # 1. Get attacking direction
    if infer_attacking_direction:
        attacking_direction_col = get_unused_column_name(df_tracking, "attacking_direction")
        df_tracking[attacking_direction_col] = infer_playing_direction(
            df_tracking, team_col=tracking_team_col, period_col=period_col, possession_team_col=possession_team_col,
            x_col=tracking_x_col
        )
    if attacking_direction_col is not None:
        fr2playingdirection = df_tracking[[tracking_frame_col, attacking_direction_col]].set_index(tracking_frame_col).to_dict()[attacking_direction_col]
        ATTACKING_DIRECTION = np.array([fr2playingdirection[frame] for frame in frame_to_idx])  # F
    else:
        ATTACKING_DIRECTION = np.ones(F)  # if no attacking direction is given, we assume always left-to-right
    # 2. Calculate danger
    X = simulation_result.x_grid  # F x PHI x T
    Y = simulation_result.y_grid  # F x PHI x T
    X_NORM = X * ATTACKING_DIRECTION[:, np.newaxis, np.newaxis]  # F x PHI x T
    Y_NORM = Y * ATTACKING_DIRECTION[:, np.newaxis, np.newaxis]  # F x PHI x T
    DIST_TO_GOAL = _dist_to_opp_goal(X_NORM, Y_NORM)  # F x PHI x T
    OPENING_ANGLE = _opening_angle_to_goal(X_NORM, Y_NORM)  # F x PHI x T
    DANGER = _get_danger(DIST_TO_GOAL, OPENING_ANGLE)  # F x PHI x T

    # 3. Add danger to simulation result
    dangerous_result = as_dangerous_result(simulation_result, DANGER)

    # Get AS and DAS
    accessible_space = integrate_attacking_surface(simulation_result)  # F
    das = integrate_attacking_surface(dangerous_result)  # F
    fr2AS = pd.Series(accessible_space, index=df_tracking[tracking_frame_col].unique())
    fr2DAS = pd.Series(das, index=df_tracking[tracking_frame_col].unique())
    as_series = df_tracking[tracking_frame_col].map(fr2AS)
    das_series = df_tracking[tracking_frame_col].map(fr2DAS)

    idx = df_tracking[tracking_frame_col].map(frame_to_idx)

    return as_series, das_series, idx, simulation_result, dangerous_result


def infer_playing_direction(
    df_tracking, team_col="team_id", period_col="period_id", possession_team_col="ball_possession", x_col="x",
):
    """
    Automatically infer playing direction based on the mean x position of each teams in each period.

    >>> df_tracking = pd.DataFrame({"frame_id": [0, 0, 1, 1], "team_id": ["H", "A", "H", "A"], "ball_possession": ["H", "H", "A", "A"], "x": [1, 2, 3, 4], "y": [5, 6, 7, 8]})
    >>> df_tracking["playing_direction"] = infer_playing_direction(df_tracking, team_col="team_id", period_col="frame_id", possession_team_col="ball_possession", x_col="x")
    >>> df_tracking
       frame_id team_id ball_possession  x  y  playing_direction
    0         0       H               H  1  5                1.0
    1         0       A               H  2  6                1.0
    2         1       H               A  3  7               -1.0
    3         1       A               A  4  8               -1.0
    """
    playing_direction = {}
    for period_id, df_tracking_period in df_tracking.groupby(period_col):
        x_mean = df_tracking_period.groupby(team_col)[x_col].mean()
        smaller_x_team = x_mean.idxmin()
        greater_x_team = x_mean.idxmax()
        playing_direction[period_id] = {smaller_x_team: 1, greater_x_team: -1}

    new_attacking_direction = pd.Series(index=df_tracking.index, dtype=np.float64)

    for period_id in playing_direction:
        i_period = df_tracking[period_col] == period_id
        for team_id, direction in playing_direction[period_id].items():
            i_period_team_possession = i_period & (df_tracking[possession_team_col] == team_id)
            new_attacking_direction.loc[i_period_team_possession] = direction

    return new_attacking_direction


def plot_expected_completion_surface(
    das_simulation_result, frame_index, plot_type_off="poss", plot_type_def=None, color_off="blue", color_def="red",
    plot_gridpoints=True,
):
    """ Plot a pass completion surface. """
    x_grid = das_simulation_result.x_grid[frame_index, :, :]
    y_grid = das_simulation_result.y_grid[frame_index, :, :]

    x = np.ravel(x_grid)  # F*PHI*T
    y = np.ravel(y_grid)  # F*PHI*T

    for offdef, plot_type, color in [("off", plot_type_off, color_off), ("def", plot_type_def, color_def)]:
        if plot_type is None:
            continue
        if offdef == "off":
            if plot_type == "poss":
                p = das_simulation_result.poss_density_att[frame_index, :, :]
            elif plot_type == "prob":
                p = das_simulation_result.prob_density_att[frame_index, :, :]
            else:
                raise ValueError(f"Unknown plot type: {plot_type}. Must be 'poss' or 'prob'.")
        else:
            if plot_type == "poss":
                p = das_simulation_result.poss_density_def[frame_index, :, :]
            elif plot_type == "prob":
                p = das_simulation_result.prob_density_def[frame_index, :, :]
            else:
                raise ValueError(f"Unknown plot type: {plot_type}. Must be 'poss' or 'prob'.")

        z = np.ravel(p)  # F*PHI*T

        areas = 10
        absolute_scale = False
        if absolute_scale:
            levels = np.linspace(start=0, stop=1.1, num=areas + 1, endpoint=True)
        else:
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
