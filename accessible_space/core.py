import numpy as np
import math
import scipy.integrate
import collections

from .motion_models import constant_velocity_time_to_arrive_1d, approx_two_point_time_to_arrive, constant_velocity_time_to_arrive

# Result object to hold simulation results
_result_fields = [
    "poss_cum_att",  # F x PHI x T
    "prob_cum_att",  # F x PHI x T
    "poss_density_att",  # F x PHI x T
    "prob_density_att",  # F x PHI x T
    "poss_cum_def",  # F x PHI x T
    "prob_cum_def",  # F x PHI x T
    "poss_density_def",  # F x PHI x T
    "prob_density_def",  # F x PHI x T

    "phi_grid",  # PHI
    "r_grid",  # T
    "x_grid",  # F x PHI x T
    "y_grid",  # F x PHI x T
]
Result = collections.namedtuple("Result", _result_fields, defaults=[None] * len(_result_fields))

# Default model parameters
_DEFAULT_B0 = -1.3075312012275244
_DEFAULT_B1 = -65.57184250749606
_DEFAULT_PASS_START_LOCATION_OFFSET = 0.2821895970952328
_DEFAULT_TIME_OFFSET_BALL = -0.09680365586691105
_DEFAULT_TOL_DISTANCE = 2.5714050933456036
_DEFAULT_PLAYER_VELOCITY = 3.984451038279267
_DEFAULT_KEEP_INERTIAL_VELOCITY = True
_DEFAULT_A_MAX = 14.256003027575932
_DEFAULT_V_MAX = 12.865546440947865
_DEFAULT_USE_MAX = True
_DEFAULT_USE_APPROX_TWO_POINT = False  # True
_DEFAULT_INERTIAL_SECONDS = 0.6164609802178712
_DEFAULT_RADIAL_GRIDSIZE = 3


PARAMETER_BOUNDS = {
    # Core simulation model
    "pass_start_location_offset": [-5, 5],
    "time_offset_ball": [-5, 5],
    "radial_gridsize": [4.99, 5.01],
    "b0": [-20, 15],
    "b1": [-250, 0],
    "player_velocity": [2, 35],
    "keep_inertial_velocity": [True],  # , False],
    "use_max": [False, True],
    "v_max": [5, 40],
    "a_max": [10, 45],
    "inertial_seconds": [0.0, 1.5],  # , True],
    "tol_distance": [0, 7],
    "use_approx_two_point": [False, True],

    # xC
    "exclude_passer": [True],
    "use_poss": [False, True],  # , True],#, True],
    "use_fixed_v0": [False, True],
    "v0_min": [1, 14.999],
    "v0_max": [15, 45],
    "n_v0": [0.5, 7.5],
}


def _sigmoid(x):
    """
    Computational efficient sigmoid function

    >>> _sigmoid(np.array([-1, 0, 1])), 1 / (1 + np.exp(-np.array([-1, 0, 1])))
    (array([0.25, 0.5 , 0.75]), array([0.26894142, 0.5       , 0.73105858]))
    """
    return 0.5 * (x / (1 + np.abs(x)) + 1)


def _integrate_trapezoid(y, x):
    """
    Integrate y over x using the trapezoid rule

    >>> _integrate_trapezoid(np.array([1, 2, 3]), np.array([0, 1, 2]))
    array([0. , 1.5, 4. ])
    """
    return scipy.integrate.cumulative_trapezoid(y=y, x=x, initial=0, axis=-1)


def _assert_matrix_consistency(PLAYER_POS, BALL_POS, phi_grid, v0_grid, passer_team, team_list, players=None, passers_to_exclude=None):
    F = PLAYER_POS.shape[0]
    assert F == BALL_POS.shape[0], f"Dimension F is {F} (from PLAYER_POS: {PLAYER_POS.shape}), but BALL_POS shape is {BALL_POS.shape}"
    assert F == phi_grid.shape[0], f"Dimension F is {F} (from PLAYER_POS: {PLAYER_POS.shape}), but phi_grid shape is {phi_grid.shape}"
    assert F == v0_grid.shape[0], f"Dimension F is {F} (from PLAYER_POS: {PLAYER_POS.shape}), but v0_grid shape is {v0_grid.shape}"
    assert F == passer_team.shape[0], f"Dimension F is {F} (from PLAYER_POS: {PLAYER_POS.shape}), but passer_team shape is {passer_team.shape}"
    P = PLAYER_POS.shape[1]
    assert P == team_list.shape[0], f"Dimension P is {P} (from PLAYER_POS: {PLAYER_POS.shape}), but team_list shape is {team_list.shape}"
    assert PLAYER_POS.shape[2] >= 4  # >= or = ?
    assert BALL_POS.shape[1] >= 2  # ...
    if passers_to_exclude is not None:
        assert F == passers_to_exclude.shape[0], f"Dimension F is {F} (from PLAYER_POS: {PLAYER_POS.shape}), but passers_to_exclude shape is {passers_to_exclude.shape}"
        assert P == players.shape[0], f"Dimension P is {P} (from PLAYER_POS: {PLAYER_POS.shape}), but players shape is {players.shape}"


def simulate_passes(
    # Input data
    PLAYER_POS,  # F x P x 4[x, y, vx, vy], player positions
    BALL_POS,  # F x 2[x, y], ball positions
    phi_grid,  # F x PHI, pass angles
    v0_grid,  # F x V0, pass speeds
    passer_teams,  # F, frame-wise team of passers
    player_teams,  # P, player teams
    players=None,  # P, players
    passers_to_exclude=None,  # F, frame-wise passer, but only if we want to exclude the passer

    # Model parameters
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
) -> Result:
    """ Calculate the pass simulation model using numpy matrices - Core functionality of this package

    # Simulate a pass from player A straight to the right towards a defender B who is 50m away.
    >>> res = simulate_passes(np.array([[[0, 0, 0, 0], [50, 0, 0, 0]]]), np.array([[0, 0]]), np.array([[0]]), np.array([[10]]), np.array([0]), np.array([0, 1]), players=np.array(["A", "B"]), passers_to_exclude=np.array(["A"]), radial_gridsize=15)
    >>> res.poss_density_def.shape, res.poss_density_def
    ((1, 1, 10), array([[[3.64076555e-05, 6.78000534e-05, 4.92186270e-04, 6.01130886e-02,
             6.00108990e-02, 2.16895102e-03, 1.68588297e-04, 8.77026250e-05,
             5.92672504e-05, 4.47561819e-05]]]))
    >>> res.prob_density_def.shape, res.prob_density_def
    ((1, 1, 10), array([[[4.81845022e-05, 8.96538328e-05, 6.47811510e-04, 4.78308182e-02,
             1.76086850e-02, 3.79740781e-04, 2.89490486e-05, 1.50277900e-05,
             1.01430190e-05, 7.65297540e-06]]]))
    >>> res.poss_cum_def.shape, res.poss_cum_def
    ((1, 1, 10), array([[[5.46114833e-04, 1.01700080e-03, 7.38279405e-03, 9.01696329e-01,
             9.01696329e-01, 9.01696329e-01, 9.01696329e-01, 9.01696329e-01,
             9.01696329e-01, 9.01696329e-01]]]))
    >>> res.prob_cum_def.shape, res.prob_cum_def
    ((1, 1, 10), array([[[0.00000000e+00, 8.64911915e-04, 5.49242530e-03, 3.39981656e-01,
             7.64645132e-01, 8.62980264e-01, 8.65617423e-01, 8.65903149e-01,
             8.66066720e-01, 8.66182372e-01]]]))
    >>> res.prob_cum_att.shape, res.prob_cum_att  # F x PHI x T
    ((1, 1, 10), array([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]))
    >>> res.phi_grid.shape, res.phi_grid
    ((1, 1), array([[0]]))
    >>> res.r_grid.shape, res.r_grid
    ((10,), array([  0.2821896 ,  16.91750186,  33.55281413,  50.1881264 ,
            66.82343867,  83.45875093, 100.0940632 , 116.72937547,
           133.36468773, 150.        ]))
    >>> res.x_grid.shape, res.x_grid
    ((1, 1, 10), array([[[  0.2821896 ,  16.91750186,  33.55281413,  50.1881264 ,
              66.82343867,  83.45875093, 100.0940632 , 116.72937547,
             133.36468773, 150.        ]]]))
    >>> res.y_grid.shape, res.y_grid
    ((1, 1, 10), array([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]))
    """
    _assert_matrix_consistency(PLAYER_POS, BALL_POS, phi_grid, v0_grid, passer_teams, player_teams, players, passers_to_exclude)

    ### 1. Calculate ball trajectory
    # 1.1 Calculate spatial grid
    D_BALL_SIM = np.linspace(start=pass_start_location_offset, stop=150, num=math.ceil(150 / radial_gridsize))  # T

    # 1.2 Calculate temporal grid
    T_BALL_SIM = constant_velocity_time_to_arrive_1d(
        x=D_BALL_SIM[0], v=v0_grid[:, :, np.newaxis], x_target=D_BALL_SIM[np.newaxis, np.newaxis, :],
    )  # F x V0 x T
    T_BALL_SIM += time_offset_ball

    # 1.3 Calculate 2D points along ball trajectory
    cos_phi, sin_phi = np.cos(phi_grid), np.sin(phi_grid)  # F x PHI
    X_BALL_SIM = BALL_POS[:, 0][:, np.newaxis, np.newaxis] + cos_phi[:, :, np.newaxis] * D_BALL_SIM[np.newaxis, np.newaxis, :]  # F x PHI x T
    Y_BALL_SIM = BALL_POS[:, 1][:, np.newaxis, np.newaxis] + sin_phi[:, :, np.newaxis] * D_BALL_SIM[np.newaxis, np.newaxis, :]  # F x PHI x T

    ### 2 Calculate player interception rates
    # 2.1 Calculate time to arrive for each player along ball trajectory
    if use_approx_two_point:
        TTA_PLAYERS = approx_two_point_time_to_arrive(  # F x P x PHI x T
            x=PLAYER_POS[:, :, 0][:, :, np.newaxis, np.newaxis],
            y=PLAYER_POS[:, :, 1][:, :, np.newaxis, np.newaxis],
            vx=PLAYER_POS[:, :, 2][:, :, np.newaxis, np.newaxis],
            vy=PLAYER_POS[:, :, 3][:, :, np.newaxis, np.newaxis],
            x_target=X_BALL_SIM[:, np.newaxis, :, :],
            y_target=Y_BALL_SIM[:, np.newaxis, :, :],

            # Parameters
            use_max=use_max, velocity=player_velocity, keep_inertial_velocity=keep_inertial_velocity, v_max=v_max,
            a_max=a_max, inertial_seconds=inertial_seconds, tol_distance=tol_distance,
        )
    else:
        TTA_PLAYERS = constant_velocity_time_to_arrive(  # F x P x PHI x T
            x=PLAYER_POS[:, :, 0][:, :, np.newaxis, np.newaxis],
            y=PLAYER_POS[:, :, 1][:, :, np.newaxis, np.newaxis],
            x_target=X_BALL_SIM[:, np.newaxis, :, :],
            y_target=Y_BALL_SIM[:, np.newaxis, :, :],
            player_velocity=player_velocity,
        )

    if passers_to_exclude is not None:
        i_passers_to_exclude = np.array([list(players).index(passer) for passer in passers_to_exclude])
        i_frames = np.arange(TTA_PLAYERS.shape[0])
        TTA_PLAYERS[i_frames, i_passers_to_exclude, :, :] = np.inf  # F x P x PHI x T

    TTA_PLAYERS = np.nan_to_num(TTA_PLAYERS, nan=np.inf)  # Handle players not participating in the game by setting their TTA to infinity

    # 2.2 Transform time to arrive into interception rates
    X = TTA_PLAYERS[:, :, np.newaxis, :, :] - T_BALL_SIM[:, np.newaxis, :, np.newaxis, :]  # F x P x PHI x T - F x PHI x T = F x P x V0 x PHI x T
    with np.errstate(over='ignore'):  # overflow leads to inf which will be handled gracefully
        X[:] = b0 + b1 * X  # 1 + 1 * F x P x V0 x PHI x T = F x P x V0 x PHI x T
    with np.errstate(invalid='ignore'):  # inf -> nan
        X[:] = _sigmoid(X)
    X = np.nan_to_num(X, nan=0)  # F x P x V0 x PHI x T, gracefully handle overflow
    DT = T_BALL_SIM[:, :, 1] - T_BALL_SIM[:, :, 0]  # F x V0
    ar_time = X / DT[:, np.newaxis, :, np.newaxis, np.newaxis]  # F x P x V0 x PHI x T

    ## 3. Use interception rates to calculate probabilities
    # 3.1 Sums of interception rates over players
    sum_ar = np.nansum(ar_time, axis=1)  # F x V0 x PHI x T

    # poss-specific
    player_is_attacking = player_teams[np.newaxis, :] == passer_teams[:, np.newaxis]  # F x P
    sum_ar_att = np.nansum(np.where(player_is_attacking[:, :, np.newaxis, np.newaxis, np.newaxis], ar_time, 0), axis=1)  # F x V0 x PHI x T
    sum_ar_def = np.nansum(np.where(~player_is_attacking[:, :, np.newaxis, np.newaxis, np.newaxis], ar_time, 0), axis=1)  # F x V0 x PHI x T

    # poss-specific
    int_sum_ar = _integrate_trapezoid(y=sum_ar, x=T_BALL_SIM[:, :, np.newaxis, :])  # F x V0 x PHI x T
    int_sum_ar_att = _integrate_trapezoid(y=sum_ar_att, x=T_BALL_SIM[:, :, np.newaxis, :])  # F x V0 x PHI x T
    int_sum_ar_def = _integrate_trapezoid(y=sum_ar_def, x=T_BALL_SIM[:, :, np.newaxis, :])  # F x V0 x PHI x T

    # Cumulative probabilities from integrals
    p0_cum = np.exp(-int_sum_ar) #if "prob" in ptypes else None  # F x V0 x PHI x T, cumulative probability that no one intercepted
    p0_cum_only_att = np.exp(-int_sum_ar_att) #if "poss" in ptypes else None  # F x V0 x PHI x T
    p0_cum_only_def = np.exp(-int_sum_ar_def) #if "poss" in ptypes else None  # F x V0 x PHI x T
    p0_only_opp = np.where(
        player_is_attacking[:, :, np.newaxis, np.newaxis, np.newaxis],
        p0_cum_only_def[:, np.newaxis, :, :, :], p0_cum_only_att[:, np.newaxis, :, :, :]
    )  # F x P x V0 x PHI x T

    # Individual probability densities
    dpr_over_dt = p0_cum[:, np.newaxis, :, :, :] * ar_time  # if "prob" in ptypes else None  # F x P x V0 x PHI x T
    pr_cum_prob = _integrate_trapezoid(  # F x P x V0 x PHI x T, cumulative probability that player P intercepted
        y=dpr_over_dt,  # F x P x V0 x PHI x T
        x=T_BALL_SIM[:, np.newaxis, :, np.newaxis, :]  # F x V0 x T
    )

    dpr_poss_over_dt = p0_only_opp * ar_time  # if "poss" in ptypes else None  # F x P x V0 x PHI x T
    pr_cum_poss = _integrate_trapezoid(  # F x P x V0 x PHI x T
        y=dpr_poss_over_dt,  # F x P x V0 x PHI x T
        x=T_BALL_SIM[:, np.newaxis, :, np.newaxis, :]  # F x V0 x T
    )

    dp0_over_dt = -p0_cum * sum_ar  # F x V0 x PHI x T

    # Go from dt -> dx
    DX = D_BALL_SIM[1] - D_BALL_SIM[0]
    dpr_over_dx = dpr_over_dt * DT[:, np.newaxis, :, np.newaxis, np.newaxis] / DX  # F x P x V0 x PHI x T
    dpr_poss_over_dx = dpr_poss_over_dt * DT[:, np.newaxis, :, np.newaxis, np.newaxis] / DX  # F x P x V0 x PHI x T
    dp0_over_dx = dp0_over_dt * DT[:, np.newaxis, :, np.newaxis, np.newaxis] / DX

    # Aggregate over v0
    dpr_over_dx_vagg_prob = np.average(dpr_over_dx, axis=2)  # if "prob" in ptypes else None  # F x P x PHI x T, Take the average over all V0 in v0_grid
    dpr_over_dx_vagg_poss = np.max(dpr_poss_over_dx, axis=2)  # if "poss" in ptypes else None  # F x P x PHI x T, np.max not supported yet with numba using axis https://github.com/numba/numba/issues/1269
    dp0_over_dx_vagg = np.average(dp0_over_dx, axis=1)  # F x PHI x T

    # Normalize 3/4: poss density
    dpr_over_dx_vagg_poss_times_dx = dpr_over_dx_vagg_poss * DX  # F x P x PHI x T
    num_max = np.max(dpr_over_dx_vagg_poss_times_dx, axis=(1, 3))  # F x PHI
    dpr_over_dx_vagg_poss = dpr_over_dx_vagg_poss / num_max[:, np.newaxis, :, np.newaxis]  # F x P x PHI x T

    # Normalize 2/4: prob density
    dpr_over_dx_vagg_prob_sum = np.sum(dpr_over_dx_vagg_prob * radial_gridsize, axis=(1, 3))  # F x PHI
    dpr_over_dx_vagg_prob = dpr_over_dx_vagg_prob / dpr_over_dx_vagg_prob_sum[:, np.newaxis, :, np.newaxis]  # F x P x PHI x T

    p0_cum_vagg = np.mean(p0_cum, axis=1)  # if add_receiver else None  # F x PHI x T
    pr_cum_prob_vagg = np.mean(pr_cum_prob, axis=2)  # if add_receiver else None  # F x P x PHI x T
    pr_cum_poss_vagg = np.max(pr_cum_poss, axis=2)  # if add_receiver else None  # F x P x V0 x PHI x T -> F x P x V0 x PHI x T

    dpr_over_dx_vagg_att_prob = np.nanmax(np.where(player_is_attacking[:, :, np.newaxis, np.newaxis], dpr_over_dx_vagg_prob, 0), axis=1)  # F x PHI x T
    dpr_over_dx_vagg_def_prob = np.nanmax(np.where(~player_is_attacking[:, :, np.newaxis, np.newaxis], dpr_over_dx_vagg_prob, 0), axis=1)  # F x PHI x T
    dpr_over_dx_vagg_att_poss = np.nanmax(np.where(player_is_attacking[:, :, np.newaxis, np.newaxis], dpr_over_dx_vagg_poss, 0), axis=1)  #if add_receiver else None  # F x PHI x T
    dpr_over_dx_vagg_def_poss = np.nanmax(np.where(~player_is_attacking[:, :, np.newaxis, np.newaxis], dpr_over_dx_vagg_poss, 0), axis=1)  #if add_receiver else None  # F x PHI x T

    pr_cum_att = np.nansum(np.where(player_is_attacking[:, :, np.newaxis, np.newaxis], pr_cum_prob_vagg, 0), axis=1) #if add_receiver else None  # F x PHI x T
    pr_cum_def = np.nansum(np.where(~player_is_attacking[:, :, np.newaxis, np.newaxis], pr_cum_prob_vagg, 0), axis=1) #if add_receiver else None  # F x PHI x T
    pr_cum_poss_att = np.maximum.accumulate(dpr_over_dx_vagg_att_poss, axis=2) * radial_gridsize  # possibility CDF uses cummax instead of cumsum to emerge from PDF
    pr_cum_poss_def = np.maximum.accumulate(dpr_over_dx_vagg_def_poss, axis=2) * radial_gridsize

    # normalize 1/4: prob cum
    p_sum = pr_cum_att + pr_cum_def + p0_cum_vagg
    pr_cum_att = pr_cum_att / p_sum
    pr_cum_def = pr_cum_def / p_sum

    # normalize 3/4: poss cum
    pr_cum_poss_att = np.minimum(pr_cum_poss_att, 1)
    pr_cum_poss_def = np.minimum(pr_cum_poss_def, 1)

    result = Result(
        # Prob/poss (cumulative and densities) along simulated ball trajectories
        poss_cum_att=pr_cum_poss_att,  # F x PHI x T
        prob_cum_att=pr_cum_att,  # F x PHI x T
        poss_density_att=dpr_over_dx_vagg_att_poss,  # F x PHI x T
        prob_density_att=dpr_over_dx_vagg_att_prob,  # F x PHI x T
        poss_cum_def=pr_cum_poss_def,  # F x PHI x T
        prob_cum_def=pr_cum_def,  # F x PHI x T
        poss_density_def=dpr_over_dx_vagg_def_poss,  # F x PHI x T
        prob_density_def=dpr_over_dx_vagg_def_prob,  # F x PHI x T

        # Trajectory grids
        phi_grid=phi_grid,  # F x PHI
        r_grid=D_BALL_SIM,  # T
        x_grid=X_BALL_SIM,  # F x PHI x T
        y_grid=Y_BALL_SIM,  # F x PHI x T
    )

    return result


def simulate_passes_chunked(
    PLAYER_POS, BALL_POS, phi_grid, v0_grid, passer_teams, player_teams, players=None, passers_to_exclude=None,
    chunk_size=200,

    # Model parameters
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
) -> Result:
    """
    Execute pass simulation in chunks to avoid OOM.

    >>> res = simulate_passes_chunked(np.array([[[0, 0, 0, 0], [50, 0, 0, 0]]]), np.array([[0, 0]]), np.array([[0]]), np.array([[10]]), np.array([0]), np.array([0, 1]), players=np.array(["A", "B"]), passers_to_exclude=np.array(["A"]), radial_gridsize=15)
    >>> res.poss_density_def.shape, res.poss_density_def
    ((1, 1, 10), array([[[3.64076555e-05, 6.78000534e-05, 4.92186270e-04, 6.01130886e-02,
             6.00108990e-02, 2.16895102e-03, 1.68588297e-04, 8.77026250e-05,
             5.92672504e-05, 4.47561819e-05]]]))
    """
    _assert_matrix_consistency(PLAYER_POS, BALL_POS, phi_grid, v0_grid, passer_teams, player_teams, players, passers_to_exclude)

    F = PLAYER_POS.shape[0]

    # i_chunks = list(np.arange(0, F, chunk_size))
    i_chunks = range(0, F, chunk_size)

    full_result = None

    for chunk_nr, i in enumerate(i_chunks):
        i_chunk_end = min(i + chunk_size, F)

        PLAYER_POS_chunk = PLAYER_POS[i:i_chunk_end, ...]
        BALL_POS_chunk = BALL_POS[i:i_chunk_end, ...]
        phi_grid_chunk = phi_grid[i:i_chunk_end, ...]
        v0_grid_chunk = v0_grid[i:i_chunk_end, ...]
        passer_team_chunk = passer_teams[i:i_chunk_end, ...]
        if passers_to_exclude is not None:
            passers_to_exclude_chunk = passers_to_exclude[i:i_chunk_end, ...]
        else:
            passers_to_exclude_chunk = None

        result = simulate_passes(
            PLAYER_POS_chunk, BALL_POS_chunk, phi_grid_chunk, v0_grid_chunk, passer_team_chunk, player_teams, players,
            passers_to_exclude_chunk,

            pass_start_location_offset,
            time_offset_ball,
            radial_gridsize,
            b0,
            b1,
            player_velocity,
            keep_inertial_velocity,
            use_max,
            v_max,
            a_max,
            inertial_seconds,
            tol_distance,
            use_approx_two_point,
        )

        if full_result is None:
            full_result = result
        else:
            full_p_cum = np.concatenate([full_result.prob_cum_att, result.prob_cum_att], axis=0)
            full_poss_cum = np.concatenate([full_result.poss_cum_att, result.poss_cum_att], axis=0)
            full_p_density = np.concatenate([full_result.poss_density_att, result.poss_density_att], axis=0)
            full_prob_density = np.concatenate([full_result.prob_density_att, result.prob_density_att], axis=0)
            full_p_cum_def = np.concatenate([full_result.prob_cum_def, result.prob_cum_def], axis=0)
            full_poss_cum_def = np.concatenate([full_result.poss_cum_def, result.poss_cum_def], axis=0)
            full_p_density_def = np.concatenate([full_result.poss_density_def, result.poss_density_def], axis=0)
            full_prob_density_def = np.concatenate([full_result.prob_density_def, result.prob_density_def], axis=0)
            full_phi = np.concatenate([full_result.phi_grid, result.phi_grid], axis=0)
            full_x0 = np.concatenate([full_result.x_grid, result.x_grid], axis=0)
            full_y0 = np.concatenate([full_result.y_grid, result.y_grid], axis=0)
            full_result = Result(
                poss_cum_att=full_poss_cum,
                prob_cum_att=full_p_cum,
                poss_density_att=full_p_density,
                prob_density_att=full_prob_density,
                poss_cum_def=full_poss_cum_def,
                prob_cum_def=full_p_cum_def,
                poss_density_def=full_p_density_def,
                prob_density_def=full_prob_density_def,

                phi_grid=full_phi,
                r_grid=full_result.r_grid,
                x_grid=full_x0,
                y_grid=full_y0,
            )

    return full_result


def crop_result_to_pitch(simulation_result: Result) -> Result:
    """
    Set all data points that are outside the pitch to zero (e.g. for DAS computation)

    >>> res = simulate_passes(np.array([[[0, 0, 0, 0], [50, 0, 0, 0]]]), np.array([[0, 0]]), np.array([[0]]), np.array([[10]]), np.array([0]), np.array([0, 1]), players=np.array(["A", "B"]), passers_to_exclude=np.array(["A"]), radial_gridsize=15)
    >>> res.poss_density_def
    array([[[3.64076555e-05, 6.78000534e-05, 4.92186270e-04, 6.01130886e-02,
             6.00108990e-02, 2.16895102e-03, 1.68588297e-04, 8.77026250e-05,
             5.92672504e-05, 4.47561819e-05]]])
    >>> crop_result_to_pitch(res).poss_density_def
    array([[[3.64076555e-05, 6.78000534e-05, 4.92186270e-04, 6.01130886e-02,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00]]])
    """
    x = simulation_result.x_grid
    y = simulation_result.y_grid

    on_pitch_mask = ((x >= -52.5) & (x <= 52.5) & (y >= -34) & (y <= 34))  # F x PHI x T

    simulation_result = simulation_result._replace(
        prob_cum_att=np.where(on_pitch_mask, simulation_result.prob_cum_att, 0),
        poss_cum_att=np.where(on_pitch_mask, simulation_result.poss_cum_att, 0),
        poss_density_att=np.where(on_pitch_mask, simulation_result.poss_density_att, 0),
        prob_density_att=np.where(on_pitch_mask, simulation_result.prob_density_att, 0),
        poss_cum_def=np.where(on_pitch_mask, simulation_result.prob_cum_def, 0),
        prob_cum_def=np.where(on_pitch_mask, simulation_result.prob_cum_def, 0),
        poss_density_def=np.where(on_pitch_mask, simulation_result.poss_density_def, 0),
        prob_density_def=np.where(on_pitch_mask, simulation_result.prob_density_def, 0),
    )
    return simulation_result


def integrate_attacking_surface(result: Result):
    """
    Integrate attacking possibility density in result to obtain surface area (AS/DAS)

    >>> res = simulate_passes(np.array([[[0, 0, 0, 0], [50, 0, 0, 0]]]), np.array([[0, 0]]), np.array([[0, 1*np.pi/3, 2*np.pi/3]]), np.array([[10, 10, 10]]), np.array([0]), np.array([0, 1]), radial_gridsize=15)
    >>> res.poss_density_att
    array([[[2.31757060e-03, 1.73020357e-04, 8.94507051e-05, 3.65230575e-05,
             1.01681461e-05, 4.87298124e-06, 3.99332684e-06, 3.42206651e-06,
             2.99492538e-06, 2.66281387e-06],
            [6.01130886e-02, 4.48853184e-03, 2.32930637e-03, 1.57168554e-03,
             1.18542697e-03, 9.51327057e-04, 7.94322996e-04, 6.81737630e-04,
             5.97067435e-04, 5.31080571e-04],
            [6.01130886e-02, 4.48907742e-03, 2.33040676e-03, 1.57321409e-03,
             1.18718728e-03, 9.53159905e-04, 7.96136039e-04, 6.83486440e-04,
             5.98734076e-04, 5.32660414e-04]]])
    >>> integrate_attacking_surface(res)
    array([97.49734999])
    """
    result = crop_result_to_pitch(result)

    # 1. Get r-part of area elements
    r_grid = result.r_grid  # T

    r_lower_bounds = np.zeros_like(r_grid)  # Initialize with zeros
    r_lower_bounds[1:] = (r_grid[:-1] + r_grid[1:]) / 2  # Midpoint between current and previous element
    r_lower_bounds[0] = r_grid[0]  # Set lower bound for the first element

    r_upper_bounds = np.zeros_like(r_grid)  # Initialize with zeros
    r_upper_bounds[:-1] = (r_grid[:-1] + r_grid[1:]) / 2  # Midpoint between current and next element
    r_upper_bounds[-1] = r_grid[-1]  # Arbitrarily high upper bound for the last element

    dr = r_upper_bounds - r_lower_bounds  # T

    # 2. Get phi-part of area elements
    phi_grid = result.phi_grid  # F x PHI

    phi_lower_bounds = np.zeros_like(phi_grid)  # F x PHI
    phi_lower_bounds[:, 1:] = (phi_grid[:, :-1] + phi_grid[:, 1:]) / 2  # Midpoint between current and previous element
    phi_lower_bounds[:, 0] = phi_grid[:, 0]

    phi_upper_bounds = np.zeros_like(phi_grid)  # Initialize with zeros
    phi_upper_bounds[:, :-1] = (phi_grid[:, :-1] + phi_grid[:, 1:]) / 2  # Midpoint between current and next element
    phi_upper_bounds[:, -1] = phi_grid[:, -1]  # Arbitrarily high upper bound for the last element

    dphi = phi_upper_bounds - phi_lower_bounds  # F x PHI

    # 3. Calculate area elements
    outer_bound_circle_slice_area = dphi[:, :, np.newaxis]/(2*np.pi) * (np.pi * r_upper_bounds[np.newaxis, np.newaxis, :]**2)  # T
    inner_bound_circle_slice_area = dphi[:, :, np.newaxis]/(2*np.pi) * (np.pi * r_lower_bounds[np.newaxis, np.newaxis, :]**2)  # T
    dA = outer_bound_circle_slice_area - inner_bound_circle_slice_area  # F x PHI x T

    # 4. Calculate surface area
    p = result.poss_density_att * dr[np.newaxis, np.newaxis, :]  # F x PHI x T
    return np.sum(p * dA, axis=(1, 2))  # F
