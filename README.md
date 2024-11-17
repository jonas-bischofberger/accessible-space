# Accessible space

This package implements the Dangerous Accessible Space (DAS) model for football analytics.


### Install package

```
pip install accessible-space
```

### Usage example

```
# Declare example data
import accessible_space.tests.resources
df_passes = accessible_space.tests.resources.df_passes
df_tracking = accessible_space.tests.resources.df_tracking

# Add expected completion to passes
df_passes["xc"], df_passes["matrix_index"], simulation_result_xc = accessible_space.get_expected_pass_completion(df_passes, df_tracking)
print(df_passes)

# Add Dangerous Accessible Space to tracking frames
df_tracking["AS"], df_tracking["DAS"], df_tracking["matrix_index"], simulation_result_as, simulation_result_das = accessible_space.get_dangerous_accessible_space(df_tracking)
print(df_tracking)
```


### Run tests

```
python -m pytest --doctest-modules path/to/accessible_space/
```

