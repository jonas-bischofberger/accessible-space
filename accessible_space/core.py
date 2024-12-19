import collections
import math
import sys

import numpy as np
import scipy.integrate
# import streamlit as st

from .motion_models import constant_velocity_time_to_arrive_1d, approx_two_point_time_to_arrive, constant_velocity_time_to_arrive
from .utility import progress_bar

# SimulationResult object to hold simulation results
_result_fields = [
    "attack_cum_prob",  # F x PHI x T
    "attack_cum_poss",  # F x PHI x T
    "attack_prob_density",  # F x PHI x T
    "attack_poss_density",  # F x PHI x T
    "defense_cum_prob",  # F x PHI x T
    "defense_cum_poss",  # F x PHI x T
    "defense_prob_density",  # F x PHI x T
    "defense_poss_density",  # F x PHI x T

    "cum_p0",  # F x PHI x T
    "p0_density",  # F x PHI x T

    "player_cum_prob",  # F x P x PHI x T
    "player_cum_poss",  # F x P x PHI x T
    "player_prob_density",  # F x P x PHI x T
    "player_poss_density",  # F x P x PHI x T

    "phi_grid",  # PHI
    "r_grid",  # T
    "x_grid",  # F x PHI x T
    "y_grid",  # F x PHI x T
]
SimulationResult = collections.namedtuple("SimulationResult", _result_fields, defaults=[None] * len(_result_fields))

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
_DEFAULT_V0_PROB_AGGREGATION_MODE = "max"
_DEFAULT_NORMALIZE = True
_DEFAULT_USE_EFFICIENT_SIGMOID = True

PARAMETER_BOUNDS = {
    # Core simulation model
    "pass_start_location_offset": [-5, 5],
    "time_offset_ball": [-5, 5],  # very small and negative values can lead to artifacts around the passer (bc passer cannot reach the ball), also in conjunction with location offset
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
    "v0_prob_aggregation_mode": ["mean", "max"],
    "normalize": [False, True],
    "use_efficient_sigmoid": [False, True],

    # xC
    "exclude_passer": [True],
    "use_poss": [False, True],  # , True],#, True],
    "use_fixed_v0": [False, True],
    "v0_min": [1, 14.999],
    "v0_max": [15, 45],
    "n_v0": [0.5, 7.5],

    # Validation-only parameters
    "use_event_coordinates_as_ball_position": [False, True],
}


def _approximate_sigmoid(x):
    """
    Computational efficient sigmoid approximation.

    >>> _approximate_sigmoid(np.array([-1, 0, 1])), 1 / (1 + np.exp(-np.array([-1, 0, 1])))
    (array([0.25, 0.5 , 0.75]), array([0.26894142, 0.5       , 0.73105858]))
    """
    return 0.5 * (x / (1 + np.abs(x)) + 1)


def _sigmoid(x):
    """
    >>> _sigmoid(np.array([-1, 0, 1]))
    array([0.26894142, 0.5       , 0.73105858])
    """
    return 1 / (1 + np.exp(-x))


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
    fields_to_return=(
        "attack_cum_prob",  # F x PHI x T
        "attack_cum_poss",  # F x PHI x T
        "attack_prob_density",  # F x PHI x T
        "attack_poss_density",  # F x PHI x T
        "defense_cum_prob",  # F x PHI x T
        "defense_cum_poss",  # F x PHI x T
        "defense_prob_density",  # F x PHI x T
        "defense_poss_density",  # F x PHI x T
        "cum_p0",  # F x PHI x T
        "p0_density",  # F x PHI x T
        "player_cum_prob",  # F x P x PHI x T
        "player_cum_poss",  # F x P x PHI x T
        "player_prob_density",  # F x P x PHI x T
        "player_poss_density",  # F x P x PHI x T
    ),
    x_pitch_min=-52.5, x_pitch_max=52.5, y_pitch_min=-34, y_pitch_max=34,

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
    v0_prob_aggregation_mode=_DEFAULT_V0_PROB_AGGREGATION_MODE,
    normalize=_DEFAULT_NORMALIZE,
    use_efficient_sigmoid=_DEFAULT_USE_EFFICIENT_SIGMOID,
) -> SimulationResult:
    """ Calculate the pass simulation model using numpy matrices - Core functionality of this package

    # Simulate a pass from player A straight to the right towards a defender B who is 50m away.
    >>> res = simulate_passes(np.array([[[0, 0, 0, 0], [50, 0, 0, 0]]]), np.array([[0, 0]]), np.array([[0]]), np.array([[10]]), np.array([0]), np.array([0, 1]), players=np.array(["A", "B"]), passers_to_exclude=np.array(["A"]), radial_gridsize=15)
    >>> res.defense_poss_density.shape, res.defense_poss_density
    ((1, 1, 13), array([[[4.04061215e-05, 6.93665348e-05, 2.44880664e-04, 6.66617232e-02,
             6.66666667e-02, 6.63364931e-02, 4.10293354e-04, 1.45308093e-04,
             8.82879207e-05, 6.34066099e-05, 4.94660755e-05, 4.05506393e-05,
             3.43581543e-05]]]))
    >>> res.attack_cum_prob.shape, res.attack_cum_prob  # F x PHI x T
    ((1, 1, 13), array([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]))
    >>> res.phi_grid.shape, res.phi_grid
    ((1, 1), array([[0]]))
    >>> res.r_grid.shape, res.r_grid
    ((13,), array([  0.2821896,  15.2821896,  30.2821896,  45.2821896,  60.2821896,
            75.2821896,  90.2821896, 105.2821896, 120.2821896, 135.2821896,
           150.2821896, 165.2821896, 180.2821896]))
    >>> res.x_grid.shape, res.x_grid
    ((1, 1, 13), array([[[  0.2821896,  15.2821896,  30.2821896,  45.2821896,
              60.2821896,  75.2821896,  90.2821896, 105.2821896,
             120.2821896, 135.2821896, 150.2821896, 165.2821896,
             180.2821896]]]))
    >>> res.y_grid.shape, res.y_grid
    ((1, 1, 13), array([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]))
    """
    _assert_matrix_consistency(PLAYER_POS, BALL_POS, phi_grid, v0_grid, passer_teams, player_teams, players, passers_to_exclude)

    def _should_return_any_of(fields):
        return any([field in fields for field in fields_to_return])

    ### 1. Calculate ball trajectory
    # 1.1 Calculate spatial grid
    max_pass_length = np.sqrt((x_pitch_max - x_pitch_min) ** 2 + (y_pitch_max - y_pitch_min) ** 2) + radial_gridsize * 3  # T
    D_BALL_SIM = np.arange(pass_start_location_offset, max_pass_length + pass_start_location_offset + radial_gridsize, radial_gridsize)  # T

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
            x=PLAYER_POS[:, :, 0][:, :, np.newaxis, np.newaxis], y=PLAYER_POS[:, :, 1][:, :, np.newaxis, np.newaxis],
            vx=PLAYER_POS[:, :, 2][:, :, np.newaxis, np.newaxis], vy=PLAYER_POS[:, :, 3][:, :, np.newaxis, np.newaxis],
            x_target=X_BALL_SIM[:, np.newaxis, :, :], y_target=Y_BALL_SIM[:, np.newaxis, :, :],

            # Parameters
            use_max=use_max, velocity=player_velocity, keep_inertial_velocity=keep_inertial_velocity, v_max=v_max,
            a_max=a_max, inertial_seconds=inertial_seconds, tol_distance=tol_distance,
        )
    else:
        TTA_PLAYERS = constant_velocity_time_to_arrive(  # F x P x PHI x T
            x=PLAYER_POS[:, :, 0][:, :, np.newaxis, np.newaxis], y=PLAYER_POS[:, :, 1][:, :, np.newaxis, np.newaxis],
            x_target=X_BALL_SIM[:, np.newaxis, :, :], y_target=Y_BALL_SIM[:, np.newaxis, :, :],

            # Parameter
            player_velocity=player_velocity,
        )

    if passers_to_exclude is not None:
        i_passers_to_exclude = np.array([list(players).index(passer) for passer in passers_to_exclude])
        i_frames = np.arange(TTA_PLAYERS.shape[0])
        TTA_PLAYERS[i_frames, i_passers_to_exclude, :, :] = np.inf  # F x P x PHI x T

    TTA_PLAYERS = np.nan_to_num(TTA_PLAYERS, nan=np.inf)  # Handle players not participating in the game by setting their TTA to infinity

    # 2.2 Transform time to arrive into interception rates
    TMP = TTA_PLAYERS[:, :, np.newaxis, :, :] - T_BALL_SIM[:, np.newaxis, :, np.newaxis, :]  # F x P x PHI x T - F x PHI x T = F x P x V0 x PHI x T
    with np.errstate(over='ignore'):  # overflow leads to inf which will be handled gracefully later
        TMP[:] = b0 + b1 * TMP  # 1 + 1 * F x P x V0 x PHI x T = F x P x V0 x PHI x T
    with np.errstate(invalid='ignore'):  # inf -> nan
        TMP[:] = _approximate_sigmoid(TMP) if use_efficient_sigmoid else _sigmoid(TMP)
    TMP = np.nan_to_num(TMP, nan=0)  # F x P x V0 x PHI x T, gracefully handle overflow
    DT = T_BALL_SIM[:, :, 1] - T_BALL_SIM[:, :, 0]  # F x V0
    ar_time = TMP / DT[:, np.newaxis, :, np.newaxis, np.newaxis]  # F x P x V0 x PHI x T
    del TMP

    ## 3. Use interception rates to calculate probabilities
    # 3.1 Sums of interception rates over players
    sum_ar = np.nansum(ar_time, axis=1) if _should_return_any_of(["attack_prob_density", "defense_prob_density", "player_prob_density", "attack_cum_prob", "defense_cum_prob", "player_cum_prob", "p0_density", "cum_p0"]) else None  # F x V0 x PHI x T
    player_is_attacking = (player_teams[np.newaxis, :] == passer_teams[:, np.newaxis]) if _should_return_any_of(["attack_cum_poss", "attack_poss_density", "defense_cum_poss", "defense_poss_density", "player_cum_poss", "player_poss_density", "attack_cum_prob", "defense_cum_prob", "attack_prob_density", "defense_prob_density"]) else None  # F x P
    sum_ar_att = np.nansum(np.where(player_is_attacking[:, :, np.newaxis, np.newaxis, np.newaxis], ar_time, 0), axis=1) if _should_return_any_of(["attack_cum_poss", "attack_poss_density", "defense_cum_poss", "defense_poss_density", "player_cum_poss", "player_poss_density"]) else None  # F x V0 x PHI x T
    sum_ar_def = np.nansum(np.where(~player_is_attacking[:, :, np.newaxis, np.newaxis, np.newaxis], ar_time, 0), axis=1) if _should_return_any_of(["attack_cum_poss", "attack_poss_density", "defense_cum_poss", "defense_poss_density", "player_cum_poss", "player_poss_density"]) else None  # F x V0 x PHI x T

    # 3.2 Integral of sum of interception rates
    int_sum_ar = scipy.integrate.cumulative_trapezoid(y=sum_ar, x=T_BALL_SIM[:, :, np.newaxis, :], initial=0, axis=-1) if _should_return_any_of(["attack_prob_density", "defense_prob_density", "player_prob_density", "attack_cum_prob", "defense_cum_prob", "player_cum_prob", "p0_density", "cum_p0"]) else None  # F x V0 x PHI x T
    int_sum_ar_att = scipy.integrate.cumulative_trapezoid(y=sum_ar_att, x=T_BALL_SIM[:, :, np.newaxis, :], initial=0, axis=-1) if _should_return_any_of(["attack_cum_poss", "attack_poss_density", "defense_cum_poss", "defense_poss_density", "player_cum_poss", "player_poss_density"]) else None  # F x V0 x PHI x T
    int_sum_ar_def = scipy.integrate.cumulative_trapezoid(y=sum_ar_def, x=T_BALL_SIM[:, :, np.newaxis, :], initial=0, axis=-1) if _should_return_any_of(["attack_cum_poss", "attack_poss_density", "defense_cum_poss", "defense_poss_density", "player_cum_poss", "player_poss_density"]) else None  # F x V0 x PHI x T

    # 3.3 Cumulative probability P0 from integrals
    cum_p0 = np.exp(-int_sum_ar) if _should_return_any_of(["attack_prob_density", "defense_prob_density", "player_prob_density", "attack_cum_prob", "defense_cum_prob", "player_cum_prob", "p0_density", "cum_p0"]) else None  # F x V0 x PHI x T, cumulative probability that no one intercepted
    cum_p0_only_att = np.exp(-int_sum_ar_att) if _should_return_any_of(["attack_cum_poss", "attack_poss_density", "defense_cum_poss", "defense_poss_density", "player_cum_poss", "player_poss_density"]) else None  # F x V0 x PHI x T
    cum_p0_only_def = np.exp(-int_sum_ar_def) if _should_return_any_of(["attack_cum_poss", "attack_poss_density", "defense_cum_poss", "defense_poss_density", "player_cum_poss", "player_poss_density"]) else None  # F x V0 x PHI x T
    cum_p0_only_opp = np.where(
        player_is_attacking[:, :, np.newaxis, np.newaxis, np.newaxis],
        cum_p0_only_def[:, np.newaxis, :, :, :], cum_p0_only_att[:, np.newaxis, :, :, :]
    ) if _should_return_any_of(["attack_cum_poss", "attack_poss_density", "defense_cum_poss", "defense_poss_density", "player_cum_poss", "player_poss_density"]) else None  # F x P x V0 x PHI x T

    # 3.4 Probability density from P0
    dpr_over_dt = cum_p0[:, np.newaxis, :, :, :] * ar_time if _should_return_any_of(["attack_prob_density", "defense_prob_density", "player_prob_density", "attack_cum_prob", "defense_cum_prob", "player_cum_prob", "cum_p0"]) else None  # if "prob" in ptypes else None  # F x P x V0 x PHI x T
    dp0_over_dt = -cum_p0 * sum_ar if _should_return_any_of(["p0_density"]) else None  # F x V0 x PHI x T
    dpr_poss_over_dt = cum_p0_only_opp * ar_time if _should_return_any_of(["attack_cum_poss", "attack_poss_density", "defense_cum_poss", "defense_poss_density", "player_cum_poss", "player_poss_density"]) else None  # if "poss" in ptypes else None  # F x P x V0 x PHI x T

    # 3.5 Cumulative probability
    pr_cum_prob = scipy.integrate.cumulative_trapezoid(  # F x P x V0 x PHI x T, cumulative probability that player P intercepted
        y=dpr_over_dt,  # F x P x V0 x PHI x T
        x=T_BALL_SIM[:, np.newaxis, :, np.newaxis, :],  # F x V0 x T
        initial=0, axis=-1,
    ) if _should_return_any_of(["attack_cum_prob", "defense_cum_prob", "player_cum_prob", "cum_p0"]) else None

    # 3.6. Go from dt -> dx
    DX = radial_gridsize  # ok because we use an equally spaced grid
    dpr_over_dx = dpr_over_dt * DT[:, np.newaxis, :, np.newaxis, np.newaxis] / DX if _should_return_any_of(["attack_prob_density", "defense_prob_density", "player_prob_density"]) else None  # F x P x V0 x PHI x T
    dp0_over_dx = dp0_over_dt * DT[:, :, np.newaxis, np.newaxis] / DX if _should_return_any_of(["p0_density"]) else None  # F x V0 x PHI x T
    dpr_poss_over_dx = dpr_poss_over_dt * DT[:, np.newaxis, :, np.newaxis, np.newaxis] / DX if _should_return_any_of(["attack_cum_poss", "attack_poss_density", "defense_cum_poss", "defense_poss_density", "player_cum_poss", "player_poss_density"]) else None  # F x P x V0 x PHI x T

    # 3.7 Aggregate over v0
    player_prob_density = (np.mean(dpr_over_dx, axis=2) if v0_prob_aggregation_mode == "mean" else np.max(dpr_over_dx, axis=2)) if _should_return_any_of(["attack_prob_density", "defense_prob_density", "player_prob_density"]) else None  # F x P x PHI x T, Take the average over all V0 in v0_grid
    p0_density = (np.mean(dp0_over_dx, axis=1) if v0_prob_aggregation_mode == "mean" else np.min(dp0_over_dx, axis=1)) if _should_return_any_of(["p0_density"]) else None  # F x PHI x T
    player_poss_density = np.max(dpr_poss_over_dx, axis=2) if _should_return_any_of(["attack_cum_poss", "attack_poss_density", "defense_cum_poss", "defense_poss_density", "player_cum_poss", "player_poss_density"]) else None  # F x P x PHI x T, np.max not supported yet with numba using axis https://github.com/numba/numba/issues/1269

    cum_p0_vagg = (np.mean(cum_p0, axis=1) if v0_prob_aggregation_mode == "mean" else np.min(cum_p0, axis=1)) if _should_return_any_of(["cum_p0"]) or normalize and _should_return_any_of(["attack_cum_prob", "defense_cum_prob", "player_cum_prob"]) else None  # F x PHI x T
    pr_cum_prob_vagg = (np.mean(pr_cum_prob, axis=2) if v0_prob_aggregation_mode == "mean" else np.max(pr_cum_prob, axis=2)) if _should_return_any_of(["attack_cum_prob", "defense_cum_prob", "player_cum_prob", "cum_p0"]) else None  # F x P x PHI x T

    # 3.8 Normalize
    if normalize:
        # Normalize cumulative probability
        p_cum_sum = cum_p0_vagg + pr_cum_prob_vagg.sum(axis=1) if _should_return_any_of(["attack_cum_prob", "defense_cum_prob", "player_cum_prob", "cum_p0"]) else None  # F x PHI x T
        cum_p0_vagg = cum_p0_vagg / p_cum_sum if _should_return_any_of(["cum_p0"]) else None
        pr_cum_prob_vagg = pr_cum_prob_vagg / p_cum_sum[:, np.newaxis, :, :] if _should_return_any_of(["attack_cum_prob", "defense_cum_prob", "player_cum_prob"]) else None  # F x P x PHI x T

        # Normalize possibility density
        dpr_over_dx_vagg_poss_times_dx = player_poss_density * DX if _should_return_any_of(["attack_cum_poss", "attack_poss_density", "defense_cum_poss", "defense_poss_density", "player_cum_poss", "player_poss_density"]) else None  # F x P x PHI x T
        num_max = np.max(dpr_over_dx_vagg_poss_times_dx, axis=(1, 3)) if _should_return_any_of(["attack_cum_poss", "attack_poss_density", "defense_cum_poss", "defense_poss_density", "player_cum_poss", "player_poss_density"]) else None  # F x PHI
        player_poss_density = player_poss_density / num_max[:, np.newaxis, :, np.newaxis] if _should_return_any_of(["attack_cum_poss", "attack_poss_density", "defense_cum_poss", "defense_poss_density", "player_cum_poss", "player_poss_density"]) else None  # F x P x PHI x T

        # TODO: Normalization is hard because the prob-/possibilities are time-dependent AND need to be normalized w.r.t both the player- and the time-axis.

    # 3.9 Aggregate over players (Individual level -> Team level)
    attack_prob_density = np.nansum(np.where(player_is_attacking[:, :, np.newaxis, np.newaxis], player_prob_density, 0), axis=1) if _should_return_any_of(["attack_prob_density"]) else None  # F x PHI x T
    defense_prob_density = np.nansum(np.where(~player_is_attacking[:, :, np.newaxis, np.newaxis], player_prob_density, 0), axis=1) if _should_return_any_of(["defense_prob_density"]) else None  # F x PHI x T
    attack_poss_density = np.nanmax(np.where(player_is_attacking[:, :, np.newaxis, np.newaxis], player_poss_density, 0), axis=1) if _should_return_any_of(["attack_cum_poss", "attack_poss_density"]) else None  # F x PHI x T
    defense_poss_density = np.nanmax(np.where(~player_is_attacking[:, :, np.newaxis, np.newaxis], player_poss_density, 0), axis=1) if _should_return_any_of(["defense_cum_poss", "defense_poss_density"]) else None  # F x PHI x T
    attack_cum_prob = np.nansum(np.where(player_is_attacking[:, :, np.newaxis, np.newaxis], pr_cum_prob_vagg, 0), axis=1) if _should_return_any_of(["attack_cum_prob"]) else None  # F x PHI x T
    defense_cum_prob = np.nansum(np.where(~player_is_attacking[:, :, np.newaxis, np.newaxis], pr_cum_prob_vagg, 0), axis=1) if _should_return_any_of(["defense_cum_prob"]) else None  # F x PHI x T

    player_cum_poss = np.maximum.accumulate(player_poss_density, axis=-1) * radial_gridsize if _should_return_any_of(["player_cum_poss"]) else None  # TODO cleaner would be to move this earlier?
    attack_cum_poss = np.maximum.accumulate(attack_poss_density, axis=-1) * radial_gridsize if _should_return_any_of(["attack_cum_poss"]) else None  # possibility CDF uses cummax instead of cumsum to emerge from PDF
    defense_cum_poss = np.maximum.accumulate(defense_poss_density, axis=-1) * radial_gridsize if _should_return_any_of(["defense_cum_poss"]) else None

    result = SimulationResult(
        # Team-level prob-/possibilities (cumulative and densities) along simulated ball trajectories
        attack_cum_prob=attack_cum_prob,  # F x PHI x T
        attack_cum_poss=attack_cum_poss,  # F x PHI x T
        attack_prob_density=attack_prob_density,  # F x PHI x T
        attack_poss_density=attack_poss_density,  # F x PHI x T
        defense_cum_prob=defense_cum_prob,  # F x PHI x T
        defense_cum_poss=defense_cum_poss,  # F x PHI x T
        defense_prob_density=defense_prob_density,  # F x PHI x T
        defense_poss_density=defense_poss_density,  # F x PHI x T

        # Player-specific prob-/possibilities
        player_cum_prob=pr_cum_prob_vagg,  # F x P x PHI x T
        player_cum_poss=player_cum_poss,  # F x P x PHI x T
        player_prob_density=player_prob_density,  # F x P x PHI x T
        player_poss_density=player_poss_density,  # F x P x PHI x T

        # Complementary proability
        cum_p0=cum_p0_vagg,  # F x PHI x T
        p0_density=p0_density,  # F x PHI x T

        # Trajectory grids
        phi_grid=phi_grid,  # F x PHI
        r_grid=D_BALL_SIM,  # T
        x_grid=X_BALL_SIM,  # F x PHI x T
        y_grid=Y_BALL_SIM,  # F x PHI x T
    )

    # Set fields not to return to zero
    for field in _result_fields:
        if field in ["phi_grid", "r_grid", "x_grid", "y_grid"]:
            continue
        if field not in fields_to_return:
            result = result._replace(**{field: None})

    return result


def simulate_passes_chunked(
    # Input data
    PLAYER_POS,
    BALL_POS,
    phi_grid,
    v0_grid,
    passer_teams,
    player_teams,
    players=None,
    passers_to_exclude=None,
    x_pitch_min=-52.5, x_pitch_max=52.5, y_pitch_min=-34, y_pitch_max=34,

    # Options
    use_progress_bar=True,
    chunk_size=50,
    fields_to_return=(
        "attack_cum_prob",  # F x PHI x T
        "attack_cum_poss",  # F x PHI x T
        "attack_prob_density",  # F x PHI x T
        "attack_poss_density",  # F x PHI x T
        "defense_cum_prob",  # F x PHI x T
        "defense_cum_poss",  # F x PHI x T
        "defense_prob_density",  # F x PHI x T
        "defense_poss_density",  # F x PHI x T
        "cum_p0",  # F x PHI x T
        "p0_density",  # F x PHI x T
        "player_cum_prob",  # F x P x PHI x T
        "player_cum_poss",  # F x P x PHI x T
        "player_prob_density",  # F x P x PHI x T
        "player_poss_density",  # F x P x PHI x T
    ),

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
    v0_prob_aggregation_mode=_DEFAULT_V0_PROB_AGGREGATION_MODE,
    normalize=_DEFAULT_NORMALIZE,
    use_efficient_sigmoid=_DEFAULT_USE_EFFICIENT_SIGMOID,
) -> SimulationResult:
    """
    Execute pass simulation in chunks to avoid OOM.

    >>> res = simulate_passes_chunked(np.array([[[0, 0, 0, 0], [50, 0, 0, 0]]]), np.array([[0, 0]]), np.array([[0]]), np.array([[10]]), np.array([0]), np.array([0, 1]), players=np.array(["A", "B"]), passers_to_exclude=np.array(["A"]), radial_gridsize=15)
    >>> res.defense_poss_density.shape, res.defense_poss_density
    ((1, 1, 13), array([[[4.04061215e-05, 6.93665348e-05, 2.44880664e-04, 6.66617232e-02,
             6.66666667e-02, 6.63364931e-02, 4.10293354e-04, 1.45308093e-04,
             8.82879207e-05, 6.34066099e-05, 4.94660755e-05, 4.05506393e-05,
             3.43581543e-05]]]))
    """
    if chunk_size is None or chunk_size <= 0:
        chunk_size = PLAYER_POS.shape[0]
    _assert_matrix_consistency(PLAYER_POS, BALL_POS, phi_grid, v0_grid, passer_teams, player_teams, players, passers_to_exclude)

    F = PLAYER_POS.shape[0]

    i_chunks = list(range(0, F, chunk_size))

    full_result = None

    if use_progress_bar:
        i_chunks = progress_bar(i_chunks, desc="Simulating passes", total=len(i_chunks), unit="chunk")

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
            fields_to_return,
            x_pitch_min, x_pitch_max, y_pitch_min, y_pitch_max,
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
            v0_prob_aggregation_mode,
            normalize,
            use_efficient_sigmoid,
        )

        if full_result is None:
            full_result = result
        else:
            full_p_cum = np.concatenate([full_result.attack_cum_prob, result.attack_cum_prob], axis=0)
            full_poss_cum = np.concatenate([full_result.attack_cum_poss, result.attack_cum_poss], axis=0)
            full_p_density = np.concatenate([full_result.attack_poss_density, result.attack_poss_density], axis=0)
            full_prob_density = np.concatenate([full_result.attack_prob_density, result.attack_prob_density], axis=0)
            full_p_cum_def = np.concatenate([full_result.defense_cum_prob, result.defense_cum_prob], axis=0)
            full_defense_cum_poss = np.concatenate([full_result.defense_cum_poss, result.defense_cum_poss], axis=0)
            full_p_density_def = np.concatenate([full_result.defense_poss_density, result.defense_poss_density], axis=0)
            full_defense_prob_density = np.concatenate([full_result.defense_prob_density, result.defense_prob_density], axis=0)
            full_cum_p0 = np.concatenate([full_result.cum_p0, result.cum_p0], axis=0)
            full_p0_density = np.concatenate([full_result.p0_density, result.p0_density], axis=0)
            full_phi = np.concatenate([full_result.phi_grid, result.phi_grid], axis=0)
            full_x0 = np.concatenate([full_result.x_grid, result.x_grid], axis=0)
            full_y0 = np.concatenate([full_result.y_grid, result.y_grid], axis=0)
            full_player_prob_density = np.concatenate([full_result.player_prob_density, result.player_prob_density], axis=0)
            full_player_poss_density = np.concatenate([full_result.player_poss_density, result.player_poss_density], axis=0)
            full_player_cum_prob = np.concatenate([full_result.player_cum_prob, result.player_cum_prob], axis=0)
            full_player_cum_poss = np.concatenate([full_result.player_cum_poss, result.player_cum_poss], axis=0)
            full_result = SimulationResult(
                attack_cum_poss=full_poss_cum,
                attack_cum_prob=full_p_cum,
                attack_poss_density=full_p_density,
                attack_prob_density=full_prob_density,
                defense_cum_poss=full_defense_cum_poss,
                defense_cum_prob=full_p_cum_def,
                defense_poss_density=full_p_density_def,
                defense_prob_density=full_defense_prob_density,
                cum_p0=full_cum_p0,
                p0_density=full_p0_density,
                player_prob_density=full_player_prob_density,
                player_poss_density=full_player_poss_density,
                player_cum_prob=full_player_cum_prob,
                player_cum_poss=full_player_cum_poss,
                phi_grid=full_phi,
                r_grid=full_result.r_grid,
                x_grid=full_x0,
                y_grid=full_y0,
            )

    return full_result


def clip_simulation_result_to_pitch(
    simulation_result: SimulationResult, x_pitch_min=-52.5, x_pitch_max=52.5, y_pitch_min=-34, y_pitch_max=34,
) -> SimulationResult:
    """
    Set all data points that are outside the pitch to zero (e.g. for DAS computation)

    >>> res = simulate_passes(np.array([[[0, 0, 0, 0], [50, 0, 0, 0]]]), np.array([[0, 0]]), np.array([[0]]), np.array([[10]]), np.array([0]), np.array([0, 1]), players=np.array(["A", "B"]), passers_to_exclude=np.array(["A"]), radial_gridsize=15)
    >>> res.defense_poss_density
    array([[[4.04061215e-05, 6.93665348e-05, 2.44880664e-04, 6.66617232e-02,
             6.66666667e-02, 6.63364931e-02, 4.10293354e-04, 1.45308093e-04,
             8.82879207e-05, 6.34066099e-05, 4.94660755e-05, 4.05506393e-05,
             3.43581543e-05]]])
    >>> clip_simulation_result_to_pitch(res).defense_poss_density
    array([[[4.04061215e-05, 6.93665348e-05, 2.44880664e-04, 6.66617232e-02,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00]]])
    """
    x = simulation_result.x_grid
    y = simulation_result.y_grid

    if x_pitch_min > x_pitch_max:
        raise ValueError("x_pitch_min must be smaller than x_pitch_max")
    if y_pitch_min > y_pitch_max:
        raise ValueError("y_pitch_min must be smaller than y_pitch_max")

    on_pitch_mask = ((x >= x_pitch_min) & (x <= x_pitch_max) & (y >= y_pitch_min) & (y <= y_pitch_max))  # F x PHI x T

    for cumulative_field in ["attack_cum_prob", "attack_cum_poss", "defense_cum_prob", "defense_cum_poss", "cum_p0", "player_cum_prob", "player_cum_poss"]:
        cum_field_data = getattr(simulation_result, cumulative_field)
        if cum_field_data is None:
            continue

        on_pitch_mask_field = on_pitch_mask if len(cum_field_data.shape) == 3 else np.repeat(on_pitch_mask[:, np.newaxis, :, :], cum_field_data.shape[1], axis=1)

        start_value = 0 if cumulative_field != "cum_p0" else 1
        updated_cum = cum_field_data.copy()
        for i in range(cum_field_data.shape[-1]):
            updated_cum[..., i] = np.where(on_pitch_mask_field[..., i], cum_field_data[..., i], updated_cum[..., i-1] if i > 0 else start_value)
            if cumulative_field == "attack_cum_prob":
                if i > 0:
                    assert np.all(on_pitch_mask_field[..., i] | (updated_cum[..., i] == updated_cum[..., i-1]))
        # for i in range(cum_field_data.shape[-1]):
        #     if i > 0:
        #         assert np.all(on_pitch_mask_field[..., i] | (updated_cum[..., i] == updated_cum[..., i - 1]))

        # updated_cum = np.where(~on_pitch_mask_field, updated_cum, np.nan)
        # st.write("x", x.shape)
        # st.write(x[0, ...])
        # st.write("y", y.shape)
        # st.write(y[0, ...])
        # st.write("updated_cum", updated_cum.shape)
        # st.write(updated_cum[0, 0, ...])
        # assert ~np.isnan(updated_cum).any()
        # st.write("np.unique(updated_cum[0, 0, ...])")
        # st.write(np.unique(updated_cum[0, 0, ...]))
        # assert len(np.unique(updated_cum[0, 0, ...])) == 2
        #
        def array_to_list_of_1d_slices(array):
            # Reshape to a 2D array, where each row corresponds to the last dimension
            reshaped_array = array.reshape(-1, array.shape[-1])
            list_of_arrays = [row for row in reshaped_array]
            return list_of_arrays

        for time_slice in array_to_list_of_1d_slices(updated_cum):
            x = np.unique(time_slice[~np.isnan(time_slice)])
            # assert len(x) == 1

        simulation_result = simulation_result._replace(**{cumulative_field: updated_cum})

    density_replacement_value = 0
    simulation_result = simulation_result._replace(
        attack_prob_density=np.where(on_pitch_mask, simulation_result.attack_prob_density, density_replacement_value) if simulation_result.attack_prob_density is not None else None,
        attack_poss_density=np.where(on_pitch_mask, simulation_result.attack_poss_density, density_replacement_value) if simulation_result.attack_poss_density is not None else None,
        defense_prob_density=np.where(on_pitch_mask, simulation_result.defense_prob_density, density_replacement_value) if simulation_result.defense_prob_density is not None else None,
        defense_poss_density=np.where(on_pitch_mask, simulation_result.defense_poss_density, density_replacement_value) if simulation_result.defense_poss_density is not None else None,
        p0_density=np.where(on_pitch_mask, simulation_result.p0_density, density_replacement_value) if simulation_result.p0_density is not None else None,
        player_prob_density=np.where(on_pitch_mask[:, np.newaxis, :, :], simulation_result.player_prob_density, density_replacement_value) if simulation_result.player_prob_density is not None else None,
        player_poss_density=np.where(on_pitch_mask[:, np.newaxis, :, :], simulation_result.player_poss_density, density_replacement_value) if simulation_result.player_poss_density is not None else None,

        # attack_cum_prob=np.where(on_pitch_mask, simulation_result.attack_cum_prob, density_replacement_value) if simulation_result.attack_cum_prob is not None else None,
        # attack_cum_poss=np.where(on_pitch_mask, simulation_result.attack_cum_poss, density_replacement_value) if simulation_result.attack_cum_poss is not None else None,
        # defense_cum_prob=np.where(on_pitch_mask, simulation_result.defense_cum_prob, density_replacement_value) if simulation_result.defense_cum_prob is not None else None,
        # defense_cum_poss=np.where(on_pitch_mask, simulation_result.defense_cum_poss, density_replacement_value) if simulation_result.defense_cum_poss is not None else None,
        # cum_p0=np.where(on_pitch_mask, simulation_result.cum_p0, density_replacement_value) if simulation_result.cum_p0 is not None else None,
        # player_cum_prob=np.where(on_pitch_mask[:, np.newaxis, :, :], simulation_result.player_cum_prob, density_replacement_value) if simulation_result.player_cum_prob is not None else None,
        # player_cum_poss=np.where(on_pitch_mask[:, np.newaxis, :, :], simulation_result.player_cum_poss, density_replacement_value) if simulation_result.player_cum_poss is not None else None,
    )
    return simulation_result


def integrate_surfaces(result: SimulationResult, x_pitch_min=-52.5, x_pitch_max=52.5, y_pitch_min=-34, y_pitch_max=34):
    """
    Integrate attacking possibility density in result to obtain surface area (AS/DAS)

    >>> res = simulate_passes(np.array([[[0, 0, 0, 0], [50, 0, 0, 0]]]), np.array([[0, 0]]), np.array([[0, 1*np.pi/3, 2*np.pi/3]]), np.array([[10, 10, 10]]), np.array([0]), np.array([0, 1]), radial_gridsize=15)
    >>> res.attack_poss_density
    array([[[2.57353741e-03, 2.11363527e-04, 1.09952435e-04, 4.50749277e-05,
             1.25566766e-05, 3.72947419e-06, 1.89161761e-06, 1.61805922e-06,
             1.41556450e-06, 1.25840005e-06, 1.13271995e-06, 1.02988309e-06,
             9.44165893e-07],
            [6.66666667e-02, 5.47601482e-03, 2.85289793e-03, 1.92762472e-03,
             1.45486398e-03, 1.16798493e-03, 9.75438331e-04, 8.37299470e-04,
             7.33378193e-04, 6.52369824e-04, 5.87453284e-04, 5.34269519e-04,
             4.89902959e-04],
            [6.66666667e-02, 5.47661163e-03, 2.85412423e-03, 1.92937880e-03,
             1.45695096e-03, 1.17021957e-03, 9.77696205e-04, 8.39511551e-04,
             7.35510810e-04, 6.54409110e-04, 5.89395847e-04, 5.36117293e-04,
             4.91660429e-04]]])
    >>> integrate_surfaces(res)
    Areas(attack_prob=array([4.36786127]), attack_poss=array([85.18623857]), defense_prob=array([211.08322005]), defense_poss=array([378.55149779]), player_prob=array([[  4.36786127, 211.08322005]]), player_poss=array([[ 85.18623857, 378.55149779]]))
    """
    result = clip_simulation_result_to_pitch(result, x_pitch_min, x_pitch_max, y_pitch_min, y_pitch_max)

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
    Areas = collections.namedtuple("Areas", ["attack_prob", "attack_poss", "defense_prob", "defense_poss", "player_prob", "player_poss"])

    area_data = {}
    for attribute, team_field in [
        ("attack_prob", result.attack_prob_density),
        ("attack_poss", result.attack_poss_density),
        ("defense_prob", result.defense_prob_density),
        ("defense_poss", result.defense_poss_density),
    ]:
        if team_field is None:
            area_data[attribute] = None
        else:
            area_data[attribute] = np.sum(team_field * dr[np.newaxis, np.newaxis, :] * dA, axis=(1, 2))  # F x PHI x T

    for attribute, player_field in [
        ("player_prob", result.player_prob_density),
        ("player_poss", result.player_poss_density),
    ]:
        if player_field is None:
            area_data[attribute] = None
        else:
            probability_field = player_field * dr[np.newaxis, np.newaxis, np.newaxis, :] * dA[:, np.newaxis, :, :]
            area_data[attribute] = np.sum(probability_field, axis=(2, 3))  # F x P x PHI x T

    return Areas(**area_data)


def as_dangerous_result(result, danger, danger_weight=None):
    """
    Convert a simulation result to a dangerous simulation result by multiplying density with danger.

    >>> res = simulate_passes_chunked(np.array([[[0, 0, 0, 0], [50, 0, 0, 0]]]), np.array([[0, 0]]), np.array([[0]]), np.array([[10]]), np.array([0]), np.array([0, 1]), players=np.array(["A", "B"]), passers_to_exclude=np.array(["A"]), radial_gridsize=15)
    >>> res.defense_poss_density
    array([[[4.04061215e-05, 6.93665348e-05, 2.44880664e-04, 6.66617232e-02,
             6.66666667e-02, 6.63364931e-02, 4.10293354e-04, 1.45308093e-04,
             8.82879207e-05, 6.34066099e-05, 4.94660755e-05, 4.05506393e-05,
             3.43581543e-05]]])
    >>> danger = np.array([[[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0]]])
    >>> dangerous_res = as_dangerous_result(res, danger)
    >>> dangerous_res.defense_poss_density
    array([[[0.00000000e+00, 6.93665348e-06, 4.89761327e-05, 1.99985170e-02,
             2.66666667e-02, 3.31682466e-02, 2.46176012e-04, 1.01715665e-04,
             7.06303366e-05, 5.70659490e-05, 4.94660755e-05, 4.05506393e-05,
             3.43581543e-05]]])
    """
    def weighted_multiplication(_danger, _space, weight=danger_weight):
        return (_danger ** (1 / weight)) * _space if weight is not None else _danger * _space

    return result._replace(
        attack_cum_poss=None,
        attack_cum_prob=None,
        attack_poss_density=weighted_multiplication(danger, result.attack_poss_density) if result.attack_poss_density is not None else None,
        attack_prob_density=weighted_multiplication(danger, result.attack_prob_density) if result.attack_prob_density is not None else None,
        defense_cum_poss=None,
        defense_cum_prob=None,
        defense_poss_density=weighted_multiplication(danger, result.defense_poss_density) if result.defense_poss_density is not None else None,
        defense_prob_density=weighted_multiplication(danger, result.defense_prob_density) if result.defense_prob_density is not None else None,
        player_cum_prob=None,
        player_cum_poss=None,
        player_poss_density=weighted_multiplication(danger[:, np.newaxis, :, :], result.player_poss_density) if result.player_poss_density is not None else None,
        player_prob_density=weighted_multiplication(danger[:, np.newaxis, :, :], result.player_prob_density) if result.player_prob_density is not None else None,
    )
