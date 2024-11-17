import accessible_space.tests.resources
df_passes = accessible_space.tests.resources.df_passes
df_tracking = accessible_space.tests.resources.df_tracking

df_passes["xc"], df_passes["matrix_index"], simulation_result_xc = accessible_space.get_expected_pass_completion(df_passes, df_tracking)
print(df_passes)

df_tracking["AS"], df_tracking["DAS"], df_tracking["matrix_index"], simulation_result_as, simulation_result_das = accessible_space.get_dangerous_accessible_space(df_tracking)
print(df_tracking)
