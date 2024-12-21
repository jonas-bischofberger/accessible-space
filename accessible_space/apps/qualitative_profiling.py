import importlib

import matplotlib.pyplot as plt
import pytest
import streamlit as st
import accessible_space.tests.test_model
import wfork_streamlit_profiler


def call_test_function_with_profiler(test_func, default_func):
    # Determine the function signature
    from inspect import signature

    # Get the parameter names of the function
    params = signature(test_func).parameters

    # Prepare arguments based on function signature
    args = {}
    if default_func:
        args['_get_data'] = default_func

    # Filter out any non-default positional arguments
    for param in params.values():
        if param.default is param.empty and param.name not in args:
            # If it's a required argument and not explicitly passed, we cannot call the function
            raise TypeError(f"Function {test_func.__name__} requires argument '{param.name}' which was not provided.")

    # Call the test function with the prepared arguments
    test_func(**args)


@pytest.mark.parametrize("a", [1, 2])
def test_example_function(a):
    st.write("a", a)
    I = 3
    assert a > 0



def extract_params_and_run(test_func, run_only_once=False):
    parametrize_mark = getattr(test_func, "pytestmark", None)
    if parametrize_mark:
        for mark in parametrize_mark:
            if mark.name == "parametrize":
                argnames = mark.args[0]  # "dataset_nr"
                values = mark.args[1]   # [1, 2]
                for value in values:
                    kwargs = {argnames: value}
                    test_func(**kwargs)
                    if run_only_once:
                        return


def profiling_dashboard():
    all_test_functions = dir(accessible_space.tests.test_model)
    # all_test_functions = [f for f in all_test_functions if f.startswith("test_")]
    selected_test_function = st.multiselect("Select test", all_test_functions, default=[])
    run_only_once = st.toggle("Run only once", value=False)

    profile = st.toggle("Profile", True)

    if profile:
        profiler = wfork_streamlit_profiler.Profiler()

    func = getattr(accessible_space.tests.test_model, selected_test_function[0])

    if profile:
        profiler.start()

    extract_params_and_run(func, run_only_once)

    if profile:
        profiler.stop()


def parameter_exploration_dashboard():
    data_source = st.selectbox("Data source", ["test", "metrica"])
    if data_source == "test":
        from accessible_space.tests.resources import df_passes, df_tracking
    else:
        from accessible_space.tests.resources import df_passes_metrica as df_passes, df_tracking_metrica as df_tracking

    st.write("df_passes")
    st.write(df_passes)
    st.write("df_tracking")
    st.write(df_tracking)

    ret = accessible_space.get_expected_pass_completion(df_passes, df_tracking)
    df_passes["xc"] = ret.xc
    df_passes["frame_index"] = ret.event_frame_index

    ret2 = accessible_space.get_das_gained(df_passes, df_tracking)
    df_passes["DAS Gained"] = ret2.das_gained

    st.write(df_passes)

    for _, p4ss in df_passes.iterrows():
        st.write("p4ss")
        st.write(p4ss)

        def plot_pass(df_tracking, p4ss, pass_frame_col="frame_id", tracking_frame_col="frame_id", tracking_team_col="team_id", tracking_x_col="x", tracking_y_col="y", event_x_col="x", event_y_col="y", event_target_x_col="x_target", event_target_y_col="y_target", tracking_vx_col="vx", tracking_vy_col="vy"):
            df_tracking_frame = df_tracking[df_tracking[tracking_frame_col] == p4ss[pass_frame_col]]

            colors = ["red", "blue", "black"]
            for i, (team, df_tracking_frame_team) in enumerate(df_tracking_frame.groupby(tracking_team_col)):
                plt.scatter(df_tracking_frame_team[tracking_x_col], df_tracking_frame_team[tracking_y_col], color=colors[i])
                plt.quiver(df_tracking_frame_team[tracking_x_col], df_tracking_frame_team[tracking_y_col], df_tracking_frame_team[tracking_vx_col], df_tracking_frame_team[tracking_vy_col], color=colors[i])

            plt.arrow(p4ss[event_x_col], p4ss[event_y_col], p4ss[event_target_x_col] - p4ss[event_target_x_col], p4ss[event_target_y_col] - p4ss[event_y_col], head_length=3, head_width=2)

            return plt.gcf()

        plt.figure()
        fig = plot_pass(df_tracking, p4ss)
        plt.title(f"xC={p4ss['xc']}, DAS_gained={p4ss['DAS Gained']}")

        importlib.reload(accessible_space.interface)

        accessible_space.plot_expected_completion_surface(ret2.simulation_result, p4ss["frame_index"])  # TODO delauny error -> catch!
        plt.xlim((-52.5, 52.5))
        plt.ylim((-34, 34))
        st.write(fig)


if __name__ == '__main__':
    parameter_exploration_dashboard()
