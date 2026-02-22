import pytest

try:
    import accessible_space.apps.readme
    import accessible_space.apps.validation
    import accessible_space.apps.qualitative_profiling
except ImportError as e:
    pytest.skip(f"Skipping tests because import failed: {e}", allow_module_level=True)


def test_readme_dashboard():
    accessible_space.apps.readme.main(run_as_streamlit_app=False)


def test_qualitative_profiling_dashboard():
    accessible_space.apps.qualitative_profiling.parameter_exploration_dashboard()


@pytest.mark.slow
def test_validation_dashboard():
    accessible_space.apps.validation.main(run_as_streamlit_app=False, dummy=False)
