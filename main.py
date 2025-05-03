import argparse
import os

from train import train_alphazero
from play import play_against_agent, find_best_checkpoint
from tournament import run_tournament
from agent import AlphaZeroAgent


def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero for 5x5 TicTacToe (4-in-a-row)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the AlphaZero agent")
    train_parser.add_argument(
        "--board_size", type=int, default=5, help="Board size (default: 5)"
    )
    train_parser.add_argument(
        "--win_length", type=int, default=4, help="Win condition length (default: 4)"
    )
    train_parser.add_argument(
        "--iterations", type=int, default=50, help="Number of training iterations"
    )
    train_parser.add_argument(
        "--games", type=int, default=50, help="Self-play games per iteration"
    )
    train_parser.add_argument(
        "--simulations", type=int, default=800, help="MCTS simulations per move"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=10, help="Training epochs per iteration"
    )
    train_parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1,
        help="Save checkpoint every N iterations",
    )
    train_parser.add_argument(
        "--no_resume",
        action="store_false",
        dest="resume",
        help="Don't resume from checkpoint",
    )
    train_parser.add_argument(
        "--evaluate_every",
        type=int,
        default=5,
        help="Evaluate against random player every N iterations",
    )
    train_parser.add_argument(
        "--eval_games", type=int, default=20, help="Number of evaluation games"
    )

    # Play command
    play_parser = subparsers.add_parser("play", help="Play against the trained agent")
    play_parser.add_argument(
        "--board_size", type=int, default=5, help="Board size (default: 5)"
    )
    play_parser.add_argument(
        "--win_length", type=int, default=4, help="Win condition length (default: 4)"
    )
    play_parser.add_argument(
        "--checkpoint_dir", type=str, default=None, help="Directory with checkpoints"
    )
    play_parser.add_argument(
        "--checkpoint", type=str, default=None, help="Specific checkpoint to load"
    )
    play_parser.add_argument(
        "--player",
        type=int,
        default=1,
        choices=[1, -1],
        help="Human player: 1=X (first), -1=O (second)",
    )
    play_parser.add_argument(
        "--simulations",
        type=int,
        default=800,
        help="Number of MCTS simulations per move",
    )

    # Tournament command
    tournament_parser = subparsers.add_parser(
        "tournament", help="Run tournament between checkpoints"
    )
    tournament_parser.add_argument(
        "--checkpoints_dir",
        type=str,
        required=True,
        help="Directory containing checkpoints",
    )
    tournament_parser.add_argument(
        "--board_size", type=int, default=5, help="Board size (default: 5)"
    )
    tournament_parser.add_argument(
        "--win_length", type=int, default=4, help="Win condition length (default: 4)"
    )
    tournament_parser.add_argument(
        "--simulations", type=int, default=800, help="MCTS simulations per move"
    )
    tournament_parser.add_argument(
        "--games", type=int, default=10, help="Games per match"
    )

    args = parser.parse_args()

    if args.command == "train":
        train_alphazero(
            board_size=args.board_size,
            win_length=args.win_length,
            num_iterations=args.iterations,
            games_per_iteration=args.games,
            num_simulations=args.simulations,
            training_epochs=args.epochs,
            checkpoint_interval=args.checkpoint_interval,
            resume_training=args.resume,
            evaluate_every=args.evaluate_every,
            eval_games=args.eval_games,
        )
    elif args.command == "play":
        # Set checkpoint directory if not specified
        if args.checkpoint_dir is None:
            args.checkpoint_dir = (
                f"./checkpoints_{args.board_size}x{args.board_size}_{args.win_length}in"
            )

        # Initialize agent
        agent = AlphaZeroAgent(
            board_size=args.board_size,
            win_length=args.win_length,
            num_simulations=args.simulations,
            checkpoint_dir=args.checkpoint_dir,
        )

        # Load checkpoint
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            checkpoint_path = find_best_checkpoint(args.checkpoint_dir)

        if checkpoint_path:
            agent.load_checkpoint(checkpoint_path)
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            print("No checkpoint found. Agent will play randomly.")

        # Play against the agent
        play_against_agent(
            agent=agent,
            board_size=args.board_size,
            win_length=args.win_length,
            human_player=args.player,
        )
    elif args.command == "tournament":
        # Run tournament between checkpoints
        run_tournament(
            checkpoints_dir=args.checkpoints_dir,
            board_size=args.board_size,
            win_length=args.win_length,
            num_simulations=args.simulations,
            games_per_match=args.games,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
