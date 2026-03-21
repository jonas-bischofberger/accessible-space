# Preventable Defensive DAS

This readme only describes the usage of _defensive_ subfolder. To see the whole readme for **Dangerous Accessible Space (DAS)** look at the original repository: [Accessible Space](https://github.com/jonas-bischofberger/accessible-space)


### Run Test Script 

The test Skript can be found under `accessible_space/scripts/run_pdd.py`. The following arguments can be passed at the moment:
```
--format [wide, long] default=long
--player_id default=home_24
--output default=output
--frame_filter_rate default=1000
```
The Output argument controlls the output path of the results. Player Id needs to be a player that is in the dataset used. The dataset for the test script is the default from the databallpy library.


### Generell Usage

To make use of the full functionality of the preventable defensive DAS metric, the `compute_pdd` function can be used directly.

**Input**
- tracking_data: pandas Dataframe containing tracking data for a game - wide or long format
- (optional) config: PDDConfig: set of parameters for optimization
- (optional) column_schema: TrackingColumnSchema: describes format of tracking_data, needed if not in schema from databallpy
- (optional) player_id: can be list or single string of player_ids to be optimized, defaults to None = all players
- (optional) wide_format: True|False, determines of tracking_data in wide or long format

**Output**
- pdd_frame_level: List of PDD values on frame level
- pdd_player_level: List of PDD values aggregated for players

#### Config
PDDConfig and TrackingColumnSchema can be adapted by using the dataclasses in `accessible_space/scripts/config.py`.

```bash
@dataclass(frozen=True)
class PDDConfig:
    frame_filter_rate: int = 125
    max_radius: float = 5.0
    optimization_method: str = "all_positions"
    das_threshold: float = 0.1
    discretization_step: float = 1.0
    min_distance_to_teammates: float = 2.0
    frame_rate: int = 25
```
```bash
@dataclass(frozen=True)
class TrackingColumnSchema:
    x: str = "x"
    y: str = "y"
    vx: str = "vx"
    vy: str = "vy"
    ball_prefix: str = "ball"
    separator: str = "_"
    home_team: str = "home"
    away_team: str = "away"
    frame_column: str = "frame"
    team_column: str = "team_id"
    period_column: str = "period_id"
    team_possession_col: str = "team_possession"
    player_in_possession_col: str = "player_possession"
```


### Contact

Feel free to reach out!

E-Mail: <a href="mailto:a12230572@unet.univie.ac.at">a12230572@unet.univie.ac.at</a>

