import numpy as np


def distance(x, y, xt, yt):
    """
    >>> distance(0, 0, 10, 0)
    10.0
    """
    return np.sqrt((x - xt) ** 2 + (y - yt) ** 2)


def constant_velocity_time_to_arrive(x, y, x_target, y_target, player_velocity=9):
    """
    >>> constant_velocity_time_to_arrive(0, 0, 10, 0, 10)
    1.0
    """
    D = np.hypot(x_target - x, y_target - y)
    V = player_velocity
    return D / V


def approx_two_point_time_to_arrive(
    x, y, vx, vy, x_target, y_target, use_max=False, velocity=9, keep_inertial_velocity=True,
    v_max=9, a_max=10, inertial_seconds=0.5, tol_distance=None
):
    """
    Calculates the time to arrive at a target point from a current point by decomposing the path into two linear segments according to Renkin et al. (2022).

    >>> approx_two_point_time_to_arrive(0, 0, 0, 0, 10, 0, use_max=False, velocity=10, keep_inertial_velocity=True, v_max=10, a_max=10, inertial_seconds=0.5, tol_distance=None)
    1.5
    """
    def _velocity_seg1(vx, vy):
        if keep_inertial_velocity:
            v_inert_x = vx  # F
            v_inert_y = vy  # F
        else:
            v0_magnitude = np.linalg.norm(np.array([vx, vy]), axis=0)
            v0_magnitude = np.maximum(1e-3, v0_magnitude)  # avoid div by null
            v_inert_x = velocity * (vx / v0_magnitude)
            v_inert_y = velocity * (vy / v0_magnitude)
        return v_inert_x, v_inert_y

    def _midpoint(x, y, vx, vy):
        v_inert_x, v_inert_y = _velocity_seg1(vx, vy)
        x_mid, y_mid = x + v_inert_x * inertial_seconds, y + v_inert_y * inertial_seconds  # F
        return x_mid, y_mid

    if tol_distance is not None:
        tol_mask = distance(x, y, x_target, y_target) < tol_distance
        tol_T = constant_velocity_time_to_arrive(x, y, x_target, y_target, velocity)

    x_mid, y_mid = _midpoint(x, y, vx, vy)
    if not use_max:
        remaining_velocity = velocity  # 1
    else:
        v_inert_magnitude = np.linalg.norm(np.array([vx, vy]), axis=0) if keep_inertial_velocity else velocity
        v_limit = v_max if v_max is not None else np.inf
        remaining_velocity = np.minimum(v_inert_magnitude + a_max * inertial_seconds, v_limit)  # , v_max if v_max is not None else np.inf)

    # T2 = football1234.util.geometry.distance(x_mid, y_mid, x_target, y_target) / remaining_velocity
    T2 = distance(x_mid, y_mid, x_target, y_target) / remaining_velocity
    T_total = T2 + inertial_seconds
    if tol_distance is not None:
        T_total[tol_mask] = tol_T[tol_mask]
    return T_total


def constant_velocity_time_to_arrive_1d(x, v, x_target):
    """
    >>> constant_velocity_time_to_arrive_1d(0, 10, 10)
    1.0
    """
    return (x_target - x) / v
