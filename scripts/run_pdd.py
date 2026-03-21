from accessible_space import compute_pdd, PDDConfig, get_dangerous_accessible_space
from helper_functions import load_game, frameify_tracking_data
import pandas as pd
import joblib
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["wide", "long"], default="long")
    parser.add_argument("--output", default="output")
    parser.add_argument("--player_id", default="home_24")
    parser.add_argument("--frame_filter_rate", default=1000)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    config = PDDConfig(frame_filter_rate=args.frame_filter_rate)

    game = load_game()
    tracking_data = pd.DataFrame(game.tracking_data.copy())

    if args.format == "wide":

        pdd_frame, pdd_player = compute_pdd(
            tracking_data,
            config=config,
            player_id=args.player_id,
        )

        pdd_frame.to_csv(output_path / "framePDD.csv")
        pdd_player.to_csv(output_path / "playerPDD.csv")

    else:

        tracking_data_long = frameify_tracking_data(tracking_data, game)
        pitch_result = get_dangerous_accessible_space(
            tracking_data_long,
            "frame",
            "player_id",
            "ball",
            "team_id",
            "player_x",
            "player_y",
            "player_vx",
            "player_vy",
            None,
            "period_id",
            "team_possession",
            "player_possession",
        )
        tracking_data_long["AS"] = pitch_result.acc_space
        tracking_data_long["DAS"] = pitch_result.das

        pdd_frame, pdd_player = compute_pdd(
            tracking_data_long,
            config=config,
            player_id=args.player_id,
            wide_format=False,
        )

        pdd_frame.to_csv(output_path / "longFramePDD.csv")
        pdd_player.to_csv(output_path / "longPlayerPDD.csv")


if __name__ == "__main__":
    main()
