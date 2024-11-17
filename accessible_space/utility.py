import colorsys
import numpy as np


def get_unused_column_name(df, prefix):
    """
    >>> import pandas as pd
    >>> df = pd.DataFrame({"Team": [1, 2], "Player": [3, 4]})
    >>> get_unused_column_name(df, "Stadium")
    'Stadium'
    >>> get_unused_column_name(df, "Team")
    'Team_1'
    """
    i = 1
    new_column_name = prefix
    while new_column_name in df.columns:
        new_column_name = f"{prefix}_{i}"
        i += 1
    return new_column_name


def _dist_to_opp_goal(x_norm, y_norm):
    """
    >>> _dist_to_opp_goal(0, 1)
    52.5
    """
    MAX_GOAL_POST_RADIUS = 0.06
    SEMI_GOAL_WIDTH_INNER_EDGE = 7.32 / 2
    SEMI_GOAL_WIDTH_CENTER = SEMI_GOAL_WIDTH_INNER_EDGE + MAX_GOAL_POST_RADIUS
    def _distance(x, y, x_target, y_target):
        return np.sqrt((x - x_target) ** 2 + (y - y_target) ** 2)
    x_goal = 52.5
    y_goal = np.clip(y_norm, -SEMI_GOAL_WIDTH_CENTER, SEMI_GOAL_WIDTH_CENTER)
    return _distance(x_norm, y_norm, x_goal, y_goal)


def _opening_angle_to_goal(x, y):
    """
    >>> _opening_angle_to_goal(np.array([52.499999]), np.array([0]))
    array([3.14159212])
    """
    MAX_GOAL_POST_RADIUS = 0.06
    SEMI_GOAL_WIDTH_INNER_EDGE = 7.32 / 2
    SEMI_GOAL_WIDTH_CENTER = SEMI_GOAL_WIDTH_INNER_EDGE + MAX_GOAL_POST_RADIUS

    def angle_between(u, v):
        divisor = np.linalg.norm(u, axis=0) * np.linalg.norm(v, axis=0)
        i_div_0 = divisor == 0
        divisor[i_div_0] = np.inf  # Avoid division by zero by setting divisor to inf
        dot_product = np.sum(u * v, axis=0)
        cosTh1 = dot_product / divisor
        angle = np.arccos(cosTh1)
        return angle

    x_goal = 52.5
    return np.abs(angle_between(np.array([x_goal - x, SEMI_GOAL_WIDTH_CENTER - y]), np.array([x_goal - x, -SEMI_GOAL_WIDTH_CENTER - y])))


def _adjust_saturation(color, saturation):
    """
    >>> _adjust_saturation((0.5, 0.5, 0.5), 0.5)
    (0.75, 0.25, 0.25)
    """
    h, l, s = colorsys.rgb_to_hls(*color)
    return colorsys.hls_to_rgb(h, l, saturation)
