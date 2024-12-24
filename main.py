import time

import matplotlib.pyplot as plt
import pandas as pd

import accessible_space
import numpy as np

import accessible_space.utility

# import streamlit as st


if __name__ == '__main__':
    import importlib
    import accessible_space.tests.test_model
    import accessible_space.interface
    import accessible_space.core
    # import accessible_space.validation
    from accessible_space.utility import progress_bar

    importlib.reload(accessible_space.tests.test_model)
    importlib.reload(accessible_space.interface)
    importlib.reload(accessible_space.core)
    # importlib.reload(accessible_space.validation)

    accessible_space.tests.test_model.test_surface_plot(accessible_space.tests.test_model._get_butterfly_data)
    # accessible_space.tests.test_model.test_infer_playing_direction(_get_data=accessible_space.tests.test_model._get_butterfly_data)

    # accessible_space.tests.test_model.test_xc_parameters(accessible_space.tests.test_model._get_butterfly_data, False, False, True, False)
    # accessible_space.tests.test_model.test_bad_data_das(pd.DataFrame({"frame_id": [1, 2], "player_id": ["a", "b"], "team_id": ["H", "A"], "x": [0, 0], "y": [0, 0], "vx": [0, 0], "vy": [0, 0], "team_in_possession": ["H", "H"]}), ValueError, "Ball flag ball_tracking_player_id='ball' does not exist in column ")
    # accessible_space.tests.test_model.test_real_world_data(1)
    # accessible_space.validation_dashboard()
    # accessible_space.tests.qualitative_profiling.profiling_dashboard()
    # accessible_space.tests.test_model.test_as_symmetry(_get_data=accessible_space.tests.test_model._get_butterfly_data)
    # readme()
