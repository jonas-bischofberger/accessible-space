import colorsys
import numpy as np
import pandas as pd
import tqdm
import streamlit as st


class _Sentinel:
    def __eq__(self, other):
        return isinstance(other, _Sentinel)


_unset = _Sentinel()  # To explicitly differentiate between a default None and a user-set None


def progress_bar(iterable, update_interval=1, **kwargs):
    """
    >>> for i in progress_bar(range(100)):
    ...     pass
    """
    try:
        total = kwargs["total"]
        kwargs.pop("total")
    except KeyError:
        try:
            total = len(iterable)
        except TypeError:
            total = None

    def _get_progress_text_without_progress_bar(console_progress_bar):
        return str(console_progress_bar).replace("█", "").replace("▌", "").replace("▊", "").replace("▍", "").replace("▋", "").replace("▉", "").replace("▏", "").replace("▎", "")

    console_progress_bar = tqdm.tqdm(iterable, total=total, **kwargs)#CustomTqdm(**kwargs)
    st.empty()
    streamlit_progress_bar = st.progress(0)
    streamlit_progress_bar.progress(0, text=_get_progress_text_without_progress_bar(console_progress_bar))
    for i, item in enumerate(console_progress_bar):
        yield item
        if i % update_interval == 0:
            if total is not None:
                progress_value = (i + 1) / total
            else:
                progress_value = 0
            streamlit_progress_bar.progress(progress_value, text=_get_progress_text_without_progress_bar(console_progress_bar))
    streamlit_progress_bar.progress(0.999, text=_get_progress_text_without_progress_bar(console_progress_bar))


def get_unused_column_name(existing_columns, prefix):
    """
    >>> import pandas as pd
    >>> df = pd.DataFrame({"Team": [1, 2], "Player": [3, 4]})
    >>> get_unused_column_name(df.columns, "Stadium")
    'Stadium'
    >>> get_unused_column_name(df.columns, "Team")
    'Team_1'
    """
    i = 1
    new_column_name = prefix
    while new_column_name in existing_columns:
        new_column_name = f"{prefix}_{i}"
        i += 1
    return new_column_name


def _dist_to_opp_goal(x_norm, y_norm, x_opp_goal):
    """
    >>> _dist_to_opp_goal(0, 1, 52.5)
    52.5
    """
    MAX_GOAL_POST_RADIUS = 0.06
    SEMI_GOAL_WIDTH_INNER_EDGE = 7.32 / 2
    SEMI_GOAL_WIDTH_CENTER = SEMI_GOAL_WIDTH_INNER_EDGE + MAX_GOAL_POST_RADIUS

    def _distance(x, y, x_target, y_target):
        return np.sqrt((x - x_target) ** 2 + (y - y_target) ** 2)

    y_goal = np.clip(y_norm, -SEMI_GOAL_WIDTH_CENTER, SEMI_GOAL_WIDTH_CENTER)
    return _distance(x_norm, y_norm, x_opp_goal, y_goal)


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
    return np.abs(angle_between(np.array([x_goal - x, SEMI_GOAL_WIDTH_CENTER - y]),
                                np.array([x_goal - x, -SEMI_GOAL_WIDTH_CENTER - y])))


def _adjust_saturation(color, saturation):
    """
    >>> _adjust_saturation((0.5, 0.5, 0.5), 0.5)
    (0.75, 0.25, 0.25)
    """
    h, l, s = colorsys.rgb_to_hls(*color)
    return colorsys.hls_to_rgb(h, l, saturation)


def _replace_column_values_except_nans(df_target, target_key_col, target_col, df_source, source_key_col, source_col):
    """
    >>> df = pd.DataFrame({"key": [2, np.nan, 4, 5], "a": [np.nan, 2, np.nan, 4]})
    >>> df2 = pd.DataFrame({"another_key": [1, np.nan, 3, 4, 5], "b": [12, 13, np.nan, np.nan, 16]})
    >>> _replace_column_values_except_nans(df, "key", "a", df2, "another_key", "b")
       key     a
    0  2.0   NaN
    1  NaN   2.0
    2  4.0   NaN
    3  5.0  16.0
    """
    assert target_key_col in df_target.columns, f"{target_key_col} not in {df_target.columns}"
    assert target_col in df_target.columns, f"{target_col} not in {df_target.columns}"
    assert source_key_col in df_source.columns, f"{source_key_col} not in {df_source.columns}"
    assert source_col in df_source.columns, f"{source_col} not in {df_source.columns}"

    df_target = df_target.copy()

    key2source_value = df_source[~df_source[source_key_col].isna()].set_index(source_key_col)[source_col].to_dict()

    def _foo(row):
        target_key = row[target_key_col]
        return key2source_value[target_key] if target_key in key2source_value and not pd.isna(key2source_value[target_key]) else row[target_col]

    df_target[target_col] = df_target.apply(_foo, axis=1)
    return df_target
