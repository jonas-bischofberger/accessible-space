import pandas as pd


def generate_smooth_positions(start_x, start_y, vx, vy, n_frames):
    x_positions = [start_x + i * vx for i in range(n_frames)]
    y_positions = [start_y + i * vy for i in range(n_frames)]
    return x_positions, y_positions


# Create tracking data for all players and ball
n_frames = 20
x_A, y_A = generate_smooth_positions(start_x=-0.1, start_y=0, vx=0.1, vy=0.05, n_frames=n_frames)
x_B, y_B = generate_smooth_positions(start_x=25, start_y=31, vx=0.2, vy=-0.1, n_frames=n_frames)
x_C, y_C = generate_smooth_positions(start_x=-15, start_y=14, vx=0.3, vy=0.1, n_frames=n_frames)
x_X, y_X = generate_smooth_positions(start_x=0, start_y=-10, vx=0.2, vy=0, n_frames=n_frames)
x_Y, y_Y = generate_smooth_positions(start_x=15, start_y=10, vx=-0.2, vy=0, n_frames=n_frames - 1)
x_ball, y_ball = generate_smooth_positions(start_x=1, start_y=0, vx=0.1, vy=0, n_frames=n_frames)

df_tracking = pd.DataFrame({
    "frame_id": list(range(n_frames)) * 4 + list(range(n_frames - 1)) + list(range(n_frames)),
    "player_id": ["A"] * n_frames + ["B"] * n_frames + ["C"] * n_frames + ["X"] * n_frames + ["Y"] * (n_frames - 1) + ["ball"] * n_frames,
    "team_id": ["Home"] * n_frames * 3 + ["Away"] * (2 * n_frames - 1) + [None] * n_frames,
    "x": x_A + x_B + x_C + x_X + x_Y + x_ball,
    "y": y_A + y_B + y_C + y_X + y_Y + y_ball,
    "vx": [0.1] * n_frames + [0.2] * n_frames + [0.3] * n_frames + [0.2] * n_frames + [-0.2] * (n_frames - 1) + [0.1] * n_frames,
    "vy": [0.05] * n_frames + [-0.1] * n_frames + [0.1] * n_frames + [0] * n_frames + [0] * (n_frames - 1) + [0] * n_frames,
})
frame2controlling_team = {fr: "Home" for fr in range(0, 14)}
frame2controlling_team.update({fr: "Away" for fr in range(14, 20)})
df_tracking["team_in_possession"] = df_tracking["frame_id"].map(frame2controlling_team)
df_tracking["period_id"] = 0

df_passes = pd.DataFrame({
    "frame_id": [0, 6, 14],
    "target_frame_id": [6, 9, 16],
    "player_id": ["A", "B", "C"],  # Players making the passes
    "receiver_id": ["B", "X", "Y"],
    "team_id": ["Home", "Home", "Home"],
    "x": [-0.1, 25, -13.8],  # X coordinate where the pass is made
    "y": [0, 30, 40.1],  # Y coordinate where the pass is made
    "x_target": [20, 15, 49],  # X target of the pass (location of receiver)
    "y_target": [30, 30, -1],  # Y target of the pass (location of receiver)
    "pass_outcome": [1, 0, 0],  # Correct pass outcomes
    "receiver_team_id": ["Home", "Away", "Away"],
})
df_passes["event_string"] = df_passes.apply(lambda row: f"{row['frame_id']}: Pass {row['player_id']} -> {row['receiver_id']} ({row['pass_outcome']})", axis=1)
