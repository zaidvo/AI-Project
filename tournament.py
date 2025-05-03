import os
import argparse
from tqdm import tqdm
from itertools import combinations

from agent import AlphaZeroAgent
from play import play_agent_vs_agent


def run_tournament(
    checkpoints_dir, board_size=5, win_length=4, num_simulations=800, games_per_match=10
):
    """Run a tournament between different checkpoints"""
    # Find all checkpoint files
    checkpoint_files = []
    for root, _, files in os.walk(checkpoints_dir):
        for file in files:
            if file.endswith(".pt"):
                checkpoint_files.append(os.path.join(root, file))

    if len(checkpoint_files) < 2:
        print("Need at least 2 checkpoints for a tournament")
        return

    print(f"Found {len(checkpoint_files)} checkpoints")
    for i, cp in enumerate(checkpoint_files):
        print(f"{i+1}: {os.path.basename(cp)}")

    # Store results
    results = {}
    for cp in checkpoint_files:
        results[cp] = {"wins": 0, "losses": 0, "draws": 0, "points": 0}

    # Play all combinations
    total_matches = len(list(combinations(checkpoint_files, 2)))
    match_num = 0

    for cp1, cp2 in tqdm(
        combinations(checkpoint_files, 2),
        total=total_matches,
        desc="Tournament matches",
    ):
        match_num += 1
        print(f"\nMatch {match_num}/{total_matches}")
        print(f"{os.path.basename(cp1)} vs {os.path.basename(cp2)}")

        # Create agents
        agent1 = AlphaZeroAgent(
            board_size=board_size,
            win_length=win_length,
            num_simulations=num_simulations,
        )
        agent2 = AlphaZeroAgent(
            board_size=board_size,
            win_length=win_length,
            num_simulations=num_simulations,
        )

        # Load checkpoints
        agent1.load_checkpoint(cp1)
        agent2.load_checkpoint(cp2)

        # Play match
        wins1, wins2, draws = play_agent_vs_agent(
            agent1, agent2, num_games=games_per_match, render=False
        )

        # Update results
        results[cp1]["wins"] += wins1
        results[cp1]["losses"] += wins2
        results[cp1]["draws"] += draws
        results[cp1]["points"] += wins1 + 0.5 * draws

        results[cp2]["wins"] += wins2
        results[cp2]["losses"] += wins1
        results[cp2]["draws"] += draws
        results[cp2]["points"] += wins2 + 0.5 * draws

        print(f"Results: {wins1} - {draws} - {wins2}")

    # Print final standings
    print("\nTournament Results:")
    print("-" * 80)
    print(
        f"{'Checkpoint':<40} {'Wins':<6} {'Draws':<6} {'Losses':<6} {'Points':<6} {'Win %':<6}"
    )
    print("-" * 80)

    # Sort by points
    sorted_results = sorted(results.items(), key=lambda x: x[1]["points"], reverse=True)

    for cp, stats in sorted_results:
        total_games = stats["wins"] + stats["draws"] + stats["losses"]
        win_percentage = (
            (stats["wins"] + 0.5 * stats["draws"]) / total_games * 100
            if total_games > 0
            else 0
        )
        print(
            f"{os.path.basename(cp):<40} {stats['wins']:<6} {stats['draws']:<6} {stats['losses']:<6} {stats['points']:<6.1f} {win_percentage:<6.1f}"
        )

    print("-" * 80)
    winner = sorted_results[0][0]
    print(f"Tournament winner: {os.path.basename(winner)}")

    return sorted_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a tournament between AlphaZero checkpoints"
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        required=True,
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--board_size", type=int, default=5, help="Board size (default: 5)"
    )
    parser.add_argument(
        "--win_length", type=int, default=4, help="Win condition length (default: 4)"
    )
    parser.add_argument(
        "--simulations", type=int, default=800, help="MCTS simulations per move"
    )
    parser.add_argument("--games", type=int, default=10, help="Games per match")

    args = parser.parse_args()

    run_tournament(
        args.checkpoints_dir,
        board_size=args.board_size,
        win_length=args.win_length,
        num_simulations=args.simulations,
        games_per_match=args.games,
    )
