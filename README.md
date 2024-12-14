# Accessible space

This package implements the Dangerous Accessible Space (DAS) model for football analytics. It includes a physical pass simulation that can be run for all locations on the pitch to obtain the area that a team can access by a pass.

Compatible with Python >= 3.7.

### Install package

```bash
pip install accessible-space
```

### Usage examples

The package has a simple pandas interface that you can use to add...
- xC (Expected completion): The expected probability that a pass is completed. Measures the risk of a pass.
- DAS (Dangerous accessible space) and AS (Accessible space): The (dangerous) area on the pitch that a team controls. DAS represents the value of a situation based on the amount of dangerous space that is accessible to the attacking team.
- DAS Gained: The increase in DAS through a pass. Measures the reward of a pass.

You only need to pass your dataframes and the schema of your data.

```python
import accessible_space
from accessible_space.tests.resources import df_passes, df_tracking  # Example data
import matplotlib.pyplot as plt

### Example 1. Add expected completion rate to passes
pass_result = accessible_space.get_expected_pass_completion(df_passes, df_tracking, event_frame_col="frame_id", event_player_col="player_id", event_team_col="team_id", event_start_x_col="x", event_start_y_col="y", event_end_x_col="x_target", event_end_y_col="y_target", tracking_frame_col="frame_id", tracking_player_col="player_id", tracking_team_col="team_id", tracking_team_in_possession_col="team_in_possession", tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy", ball_tracking_player_id="ball")
df_passes["xC"] = pass_result.xc  # Expected pass completion rate
print(df_passes[["event_string", "xC"]])

### Example 2. Add DAS Gained to passes
das_gained_result = accessible_space.get_das_gained(df_passes, df_tracking, event_frame_col="frame_id", event_success_col="pass_outcome", event_target_frame_col="target_frame_id", tracking_frame_col="frame_id", tracking_period_col="period_id", tracking_player_col="player_id", tracking_team_col="team_id", tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy", tracking_team_in_possession_col="team_in_possession", x_pitch_min=-52.5, x_pitch_max=52.5, y_pitch_min=-34, y_pitch_max=34)
df_passes["DAS_Gained"] = das_gained_result.das_gained
df_passes["AS_Gained"] = das_gained_result.as_gained
print(df_passes[["event_string", "DAS_Gained", "AS_Gained"]])

### Example 3. Add Dangerous Accessible Space to tracking frames
pitch_result = accessible_space.get_dangerous_accessible_space(df_tracking, frame_col="frame_id", period_col="period_id", player_col="player_id", team_col="team_id", x_col="x", y_col="y", vx_col="vx", vy_col="vy", team_in_possession_col="team_in_possession", x_pitch_min=-52.5, x_pitch_max=52.5, y_pitch_min=-34, y_pitch_max=34)
df_tracking["AS"] = pitch_result.acc_space  # Accessible space
df_tracking["DAS"] = pitch_result.das  # Dangerous accessible space
print(df_tracking[["frame_id", "team_in_possession", "AS", "DAS"]].drop_duplicates())
```

For even more advanced analytics, you can access the raw simulation results on team- and player-level.

```python
### Example 4. Access raw simulation results
# Example 4.1: Expected interception rate = last value of the cumulative interception probability of the defending team
pass_result = accessible_space.get_expected_pass_completion(df_passes, df_tracking)
pass_frame = 0  # We consider the pass at frame 0
df_passes["frame_index"] = pass_result.event_frame_index  # frame_index implements a mapping from original frame number to indexes of the numpy arrays in the raw simulation_result.
df_pass = df_passes[df_passes["frame_id"] == pass_frame]  # Consider the pass at frame 0
frame_index = int(df_pass["frame_index"].iloc[0])
expected_interception_rate = pass_result.simulation_result.defense_cum_prob[frame_index, 0, -1]  # Frame x Angle x Distance
print(f"Expected interception rate: {expected_interception_rate:.1%}")

# Example 4.2: Plot accessible space and dangerous accessible space
df_tracking["frame_index"] = pitch_result.frame_index

def plot_constellation(df_tracking_frame):
    plt.figure()
    plt.xlim([-52.5, 52.5])
    plt.ylim([-34, 34])
    plt.scatter(df_tracking_frame["x"], df_tracking_frame["y"], c=df_tracking_frame["team_id"].map({"Home": "red", "Away": "blue"}).fillna("black"), marker="o")
    for _, row in df_tracking_frame.iterrows():
        plt.text(row["x"], row["y"], row["player_id"] if row["player_id"] != "ball" else "")
    plt.gca().set_aspect('equal', adjustable='box')

df_tracking_frame = df_tracking[df_tracking["frame_id"] == 0]  # Plot frame 0
frame_index = df_tracking_frame["frame_index"].iloc[0]

plot_constellation(df_tracking_frame)
accessible_space.plot_expected_completion_surface(pitch_result.simulation_result, frame_index=frame_index)
plt.title(f"Accessible space: {df_tracking_frame['AS'].iloc[0]:.0f} m²")

plot_constellation(df_tracking_frame)
accessible_space.plot_expected_completion_surface(pitch_result.dangerous_result, frame_index=frame_index, color="red")
plt.title(f"Dangerous accessible space: {df_tracking_frame['DAS'].iloc[0]:.2f} m²")
plt.show()

# Example 4.3: Get (dangerous) accessible space of individual players
df_tracking["player_index"] = pitch_result.player_index  # Mapping from player to index in simulation_result
areas = accessible_space.integrate_surfaces(pitch_result.simulation_result)  # Calculate surface integrals
dangerous_areas = accessible_space.integrate_surfaces(pitch_result.dangerous_result)
for _, row in df_tracking[(df_tracking["frame_id"] == 0) & (df_tracking["player_id"] != "ball")].iterrows():  # Consider frame 0
    is_attacker = row["team_id"] == row["team_in_possession"]
    acc_space = areas.player_poss[int(frame_index), int(row["player_index"])]
    das = dangerous_areas.player_poss[int(frame_index), int(row["player_index"])]

    plot_constellation(df_tracking_frame)
    accessible_space.plot_expected_completion_surface(pitch_result.simulation_result, "player_poss_density", frame_index=frame_index, player_index=int(row["player_index"]))
    accessible_space.plot_expected_completion_surface(pitch_result.dangerous_result, "player_poss_density", frame_index=frame_index, player_index=int(row["player_index"]), color="red")
    plt.title(f"{row['player_id']} ({'attacker' if is_attacker else 'defender'}) {acc_space:.0f}m² AS and {das:.2f} m² DAS.")
    plt.show()
    # Note: Individual space is not exclusive within a team. This is intentional because your team mates do not take away space from you in the competitive way that your opponents do.
    print(f"Player {row['player_id']} ({'attacker' if is_attacker else 'defender'}) controls {acc_space:.0f}m² AS and {das:.2f} m² DAS.")
```


### Run tests

```bash
python -m pytest path/to/accessible_space/
```


### Known issues (feel free to improve upon them)

- Offside players should have an interception rate of 0 - this functionality is not implemented yet.
- This model doesn't simulate high passes, which is a significant limitation. If you have an idea how to add it, feel free to do so!
- Probabilities and possibilities are not fully normalized yet, i.e. probabilities generally do not sum to 1, possibilities may exceed 1, etc. This is because of numerical errors. Normalizing the prob-/possibilities is a difficult problem because it has to be done w.r.t two different axes (along the ball trajectory and across players) while maintaining temporal dependencies. Due to the difficulty, it is currently only partially implemented for the possibility density and cumulative possibility.
