import pandas as pd

def _generate_smooth_positions(start_x, start_y, vx, vy, n_frames):
    x_positions = [start_x + i * vx for i in range(n_frames)]
    y_positions = [start_y + i * vy for i in range(n_frames)]
    return x_positions, y_positions

# Create tracking data for all players and ball
_n_frames = 20
x_A, y_A = _generate_smooth_positions(start_x=-0.1, start_y=0, vx=0.1, vy=0.05, n_frames=_n_frames)
x_B, y_B = _generate_smooth_positions(start_x=-10, start_y=11, vx=0.2, vy=-0.1, n_frames=_n_frames)
x_C, y_C = _generate_smooth_positions(start_x=-15, start_y=-14, vx=0.3, vy=0.1, n_frames=_n_frames)
x_X, y_X = _generate_smooth_positions(start_x=15, start_y=30, vx=0.2, vy=0, n_frames=_n_frames)
x_Y, y_Y = _generate_smooth_positions(start_x=50, start_y=-1, vx=-0.2, vy=0, n_frames=_n_frames - 1)
x_ball, y_ball = _generate_smooth_positions(start_x=0, start_y=0, vx=0.1, vy=0, n_frames=_n_frames)

_df_tracking = pd.DataFrame({
    "frame_id": list(range(_n_frames)) * 4 + list(range(_n_frames - 1)) + list(range(_n_frames)),
    "player_id": ["A"] * _n_frames + ["B"] * _n_frames + ["C"] * _n_frames + ["X"] * _n_frames + ["Y"] * (_n_frames - 1) + ["ball"] * _n_frames,
    "team_id": [0] * _n_frames + [0] * _n_frames + [0] * _n_frames + [1] * _n_frames + [1] * (_n_frames - 1) + [None] * _n_frames,
    "x": x_A + x_B + x_C + x_X + x_Y + x_ball,
    "y": y_A + y_B + y_C + y_X + y_Y + y_ball,
    "vx": [0.1] * _n_frames + [0.2] * _n_frames + [0.3] * _n_frames + [0.2] * _n_frames + [-0.2] * (_n_frames - 1) + [0.1] * _n_frames,
    "vy": [0.05] * _n_frames + [-0.1] * _n_frames + [0.1] * _n_frames + [0] * _n_frames + [0] * (_n_frames - 1) + [0] * _n_frames,
})
_frame2controlling_team = {fr: 0 for fr in range(0, 14)}
_frame2controlling_team.update({fr: 1 for fr in range(14, 20)})
_df_tracking["ball_possession"] = _df_tracking["frame_id"].map(_frame2controlling_team)
_df_tracking["period_id"] = 0

_df_passes = pd.DataFrame({
    "frame_id": [0, 6, 14],
    "target_frame_id": [6, 10, 19],  # Frame ID where the pass is received
    "player_id": ["A", "B", "C"],  # Players making the passes
    "team_id": [0, 0, 0],  # Team of players making the passes
    "receiver_id": ["B", "X", "Y"],  # Intended receivers
    "pass_outcome": ["successful", "failed", "failed"],  # Correct pass outcomes
    "x": [-0.1, -9.6, -13.8],  # X coordinate where the pass is made
    "y": [0, 10.5, -12.9],  # Y coordinate where the pass is made
    "x_target": [-10, 15, 49],  # X target of the pass (location of receiver)
    "y_target": [11, 30, -1],  # Y target of the pass (location of receiver)
})


def get_df_tracking():
    return _df_tracking.copy()


def get_df_passes():
    return _df_passes.copy()

