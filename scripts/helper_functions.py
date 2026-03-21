from databallpy import get_open_game
from accessible_space import per_object_frameify_tracking_data


def load_game(game_id="J03WMX"):
    game = get_open_game(game_id=game_id)
    column_ids = game.get_column_ids() + ["ball"]
    game.tracking_data.add_velocity(column_ids=column_ids, max_velocity=37.5)
    game.synchronise_tracking_and_event_data()
    game.tracking_data.add_individual_player_possession()
    return game


def frameify_tracking_data(tracking_data, game):
    coordinate_cols = []
    player_to_team = {}
    players = game.get_column_ids()
    players.append("ball")
    for player in players:
        coordinate_cols.append(
            [f"{player}_x", f"{player}_y", f"{player}_vx", f"{player}_vy"]
        )
        player_to_team[str(player)] = player.split("_")[0]

    df_tracking = per_object_frameify_tracking_data(
        tracking_data,
        "frame",
        coordinate_cols,
        players,
        player_to_team,
        new_coordinate_cols=("player_x", "player_y", "player_vx", "player_vy"),
    )
    df_tracking = df_tracking[df_tracking["player_x"].notna()]
    return df_tracking
