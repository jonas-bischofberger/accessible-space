# Declare example data
import accessible_space.tests.resources
df_passes = accessible_space.tests.resources.get_df_passes()
df_tracking = accessible_space.tests.resources.get_df_tracking()


# Functionality 1: Add expected completion to passes
df_passes["xC"], df_passes["simulation_index"], simulation_result_xc = accessible_space.get_expected_pass_completion(df_passes, df_tracking)

# Functionality 2: Add Dangerous Accessible Space to tracking frames
df_tracking["AS"], df_tracking["DAS"], df_tracking["simulation_index"], simulation_result_as, simulation_result_das = accessible_space.get_dangerous_accessible_space(df_tracking)


# Application example 1: Rank players by xC outperformance
df_passes["outcome-xc"] = df_passes["pass_outcome"].map({"successful": 1, "failed": 0}) - df_passes["xC"]
dfg = df_passes.groupby("player_id")["outcome-xc"].mean().sort_values(ascending=False)
print(dfg)

# Application example 2: Analyze DAS Gained per team or player
frame_to_DAS = df_tracking.groupby("frame_id")["DAS"].first()
df_passes["DAS"] = df_passes["frame_id"].map(frame_to_DAS)
df_passes["DAS_target"] = df_passes["target_frame_id"].map(frame_to_DAS)
i_success = df_passes["pass_outcome"] == "successful"
df_passes.loc[i_success, "DAS_gained"] = df_passes.loc[i_success, "DAS_target"] - df_passes.loc[i_success, "DAS"]
df_passes["DAS_gained"] = df_passes["DAS_gained"].fillna(0)  # unsuccessful passes lose all DAS
dfg = df_passes.groupby("player_id")["DAS_gained"].sum().sort_values(ascending=False)
print(dfg)
dfg_team = df_passes.groupby("team_id")["DAS_gained"].sum().sort_values(ascending=False)
print(dfg_team)
