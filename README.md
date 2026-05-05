# Accessible space

[![PyPI Version](https://img.shields.io/pypi/v/accessible-space)](https://pypi.org/project/accessible-space/)
[![Python Versions](https://img.shields.io/badge/Python-%3E=3.7-blue)](https://pypi.org/project/accessible-space/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jonas-bischofberger/accessible-space/HEAD?urlpath=%2Fdoc%2Ftree%2Faccessible_space%2Fapps%2Fdemo.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonas-bischofberger/accessible-space/blob/HEAD/accessible_space/apps/demo.ipynb)
[![License](https://img.shields.io/github/license/jonas-bischofberger/accessible-space)](https://github.com/jonas-bischofberger/accessible-space/blob/HEAD/LICENSE)

**Research paper:** Bischofberger, J., Baca, A. Dangerous accessible space: a unified model of space and value in team sports. J Big Data 13, 76 (2026). https://doi.org/10.1186/s40537-026-01387-8

This is a provider-agnostic implementation of the **Dangerous Accessible Space (DAS)** model for advanced football (soccer) analytics.
Based on physical pass simulations, DAS quantifies threatening areas that a team can access by a pass.
You can use DAS to analyze profound aspects of performance like:

- Strategic passing behaviour
- Timing and danger of attacking movement
- Defensive positioning to close down passing options
- Team defensive organization

To learn how to access DAS and related metrics, see the examples below.

If you are interested in understanding the model, there is [a demo notebook explaining its basic workings](https://github.com/jonas-bischofberger/accessible-space/blob/main/accessible_space/apps/demo.ipynb).
This notebook can also be accessed without installation through [Google Colab](https://colab.research.google.com/github/jonas-bischofberger/accessible-space/blob/main/accessible_space/apps/demo.ipynb) or [Binder](https://mybinder.org/v2/gh/jonas-bischofberger/accessible-space/HEAD?urlpath=%2Fdoc%2Ftree%2Faccessible_space%2Fapps%2Fdemo.ipynb).

### Installation 

```bash
pip install accessible-space
```

### Usage

``accessible-space`` exposes a simple pandas interface that you can use to add the following metrics to tracking and event data from any provider.
- xC (Expected completion): The expected probability that a pass is completed. Measures the risk of a pass.
- DAS (Dangerous accessible space) and AS (Accessible space): The (dangerous) area on the pitch that a team controls. DAS represents the value or danger of a situation based on the amount of dangerous space that is accessible to the attacking team.
- DAS Gained: The increase in DAS through a pass. Measures the reward and strategic impact of a pass by evaluating whether the pass opens up new dangerous passing opportunities or not.
- AS Gained: The increase in AS (Accessible space) through a pass. Measures the degree to which the pass opens up safe passing opportunities.

To obtain these metrics, you need to pass your dataframes and the schema of your data as follows:

```python
import accessible_space
from accessible_space.tests.resources import df_passes, df_tracking  # Example data
import matplotlib.pyplot as plt

print(df_passes[["frame_id", "player_id", "team_id", "x", "y", "x_target", "y_target", "pass_outcome", "target_frame_id"]])
#    frame_id player_id team_id     x     y  x_target  y_target  pass_outcome  target_frame_id
# 0         0         A    Home  -0.1   0.0        20        30             1                6
# 1         6         B    Home  25.0  30.0        15        30             0                9
# 2        14         C    Home -13.8  40.1        49        -1             0               16
print(df_tracking[["period_id", "frame_id", "player_id", "team_id", "team_in_possession", "x", "y", "vx", "vy"]])
#      period_id  frame_id player_id team_id team_in_possession    x     y   vx    vy
# 0            0         0         A    Home               Home -0.1  0.00  0.1  0.05
# 1            0         1         A    Home               Home  0.0  0.05  0.1  0.05
# ..         ...       ...       ...     ...                ...  ...   ...  ...   ...
# 117          0        18      ball    None               Away  2.8  0.00  0.1  0.00
# 118          0        19      ball    None               Away  2.9  0.00  0.1  0.00

chunk_size = 50  # Adjust according to your available RAM

### Example 1. Add expected completion rate to passes
pass_result = accessible_space.get_expected_pass_completion(df_passes, df_tracking, event_frame_col="frame_id", event_player_col="player_id", event_team_col="team_id", event_start_x_col="x", event_start_y_col="y", event_end_x_col="x_target", event_end_y_col="y_target", tracking_frame_col="frame_id", tracking_player_col="player_id", tracking_team_col="team_id", tracking_team_in_possession_col="team_in_possession", tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy", ball_tracking_player_id="ball", chunk_size=chunk_size)
df_passes["xC"] = pass_result.xc  # Expected pass completion rate
print(df_passes[["event_string", "xC"]])

### Example 2. Add DAS Gained to passes
das_gained_result = accessible_space.get_das_gained(df_passes, df_tracking, event_frame_col="frame_id", event_success_col="pass_outcome", event_target_frame_col="target_frame_id", tracking_frame_col="frame_id", tracking_period_col="period_id", tracking_player_col="player_id", tracking_team_col="team_id", tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy", tracking_team_in_possession_col="team_in_possession", x_pitch_min=-52.5, x_pitch_max=52.5, y_pitch_min=-34, y_pitch_max=34, chunk_size=chunk_size)
df_passes["DAS_Gained"] = das_gained_result.das_gained
df_passes["AS_Gained"] = das_gained_result.as_gained
print(df_passes[["event_string", "DAS_Gained", "AS_Gained"]])

### Example 3. Add Dangerous Accessible Space to tracking frames
pitch_result = accessible_space.get_dangerous_accessible_space(df_tracking, frame_col="frame_id", period_col="period_id", player_col="player_id", team_col="team_id", x_col="x", y_col="y", vx_col="vx", vy_col="vy", team_in_possession_col="team_in_possession", x_pitch_min=-52.5, x_pitch_max=52.5, y_pitch_min=-34, y_pitch_max=34, chunk_size=chunk_size)
df_tracking["AS"] = pitch_result.acc_space  # Accessible space
df_tracking["DAS"] = pitch_result.das  # Dangerous accessible space
print(df_tracking[["frame_id", "team_in_possession", "AS", "DAS"]].drop_duplicates())

### Example 4. Add individual DAS to tracking frames
individual_result = accessible_space.get_individual_dangerous_accessible_space(df_tracking, frame_col="frame_id", period_col="period_id", player_col="player_id", team_col="team_id", x_col="x", y_col="y", vx_col="vx", vy_col="vy", team_in_possession_col="team_in_possession", x_pitch_min=-52.5, x_pitch_max=52.5, y_pitch_min=-34, y_pitch_max=34, chunk_size=chunk_size)
df_tracking["AS_player"] = individual_result.player_acc_space
df_tracking["DAS_player"] = individual_result.player_das
print(df_tracking[["frame_id", "player_id", "team_id", "team_in_possession", "AS_player", "DAS_player"]].drop_duplicates())
```

For even more advanced analytics, you can also access the raw simulation results on both the team- and player-level.

```python
### Example 4. Access raw simulation results
# Example 4.1: Expected interception rate = last value of the cumulative interception probability of the defending team
pass_result = accessible_space.get_expected_pass_completion(df_passes, df_tracking, additional_fields_to_return=["defense_cum_prob"], chunk_size=chunk_size)
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

# Example 4.3: Visualize (dangerous) accessible space of individual players
individual_result = accessible_space.get_individual_dangerous_accessible_space(df_tracking, period_col=None, chunk_size=chunk_size)
df_tracking["player_index"] = individual_result.player_index  # Mapping from player to index in simulation_result
df_tracking["player_AS"] = individual_result.player_acc_space
df_tracking["player_DAS"] = individual_result.player_das
for _, row in df_tracking[(df_tracking["frame_id"] == 0) & (df_tracking["player_id"] != "ball")].iterrows():  # Consider frame 0
    is_attacker = row["team_id"] == row["team_in_possession"]
    plot_constellation(df_tracking_frame)
    accessible_space.plot_expected_completion_surface(individual_result.simulation_result, frame_index=frame_index, attribute="player_poss_density", player_index=int(row["player_index"]))
    accessible_space.plot_expected_completion_surface(individual_result.dangerous_result, frame_index=frame_index, attribute="player_poss_density", player_index=int(row["player_index"]), color="red")
    plt.title(f"{row['player_id']} ({'attacker' if is_attacker else 'defender'}) {row['player_AS']:.0f}m² AS and {row['player_DAS']:.2f} m² DAS.")
    plt.show()
    # Note: Individual space is not exclusive within a team. This is intentional because your team mates do not take away space from you.
    print(f"Player {row['player_id']} ({'attacker' if is_attacker else 'defender'}) controls {row['player_AS']:.0f}m² AS and {row['player_DAS']:.2f} m² DAS.")
```

The above examples can be visualized in a Streamlit dashboard using:

```bash
pip install accessible_space[full]  # additional dependencies for dashboards, such as Streamlit
python -m accessible_space demo
```

### Reproduce my validation

My validation can be reproduced with this command, which opens up a Streamlit dashboard. Feel free to explore the dashboard and code to understand the model and its predictive performance.

```bash
pip install accessible_space[full]==2.0.0  # exact reproduction
python -m accessible_space validation
```

### Run tests

```bash
pip install accessible_space[dev]
python -m accessible_space tests
```

### Known issues (feel free to improve upon them)

- This model doesn't simulate high passes, which is a significant limitation. If you have an idea how to add it, feel free to do so!
- Probabilities and possibilities are not fully normalized yet, i.e. probabilities generally do not sum to 1, possibilities may exceed 1, etc. This is because of numerical errors. Normalizing the prob-/possibilities is a non-trivial problem because it has to be done w.r.t two different axes (along the ball trajectory and across players) while maintaining temporal dependencies. Due to the difficulty, it is currently only partially implemented.

### Contact

Feel free to reach out!

E-Mail: <a href="mailto:jonas.bischofberger@univie.ac.at">jonas.bischofberger[at]univie.ac.at</a>

