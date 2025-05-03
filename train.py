import os
import argparse
import time
from tqdm import tqdm

from agent import AlphaZeroAgent
from play import find_best_checkpoint


def train_alphazero(
    board_size=5,
    win_length=4,
    num_iterations=50,
    games_per_iteration=50,
    num_simulations=800,
    training_epochs=10,
    checkpoint_interval=1,
    resume_training=True,
    evaluate_every=5,
    eval_games=20,
):
    """Train AlphaZero for 5x5 TicTacToe with 4-in-a-row win condition"""
    checkpoint_dir = f"./checkpoints_{board_size}x{board_size}_{win_length}in"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize agent
    agent = AlphaZeroAgent(
        board_size=board_size,
        win_length=win_length,
        num_simulations=num_simulations,
        checkpoint_dir=checkpoint_dir,
    )

    # Resume from checkpoint if requested
    if resume_training:
        best_checkpoint = find_best_checkpoint(checkpoint_dir)
        if best_checkpoint:
            agent.load_checkpoint(best_checkpoint)
            print(f"Resumed training from checkpoint: {best_checkpoint}")

    # Training loop
    start_time = time.time()
    best_win_rate = 0

    for iteration in range(num_iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration+1}/{num_iterations}")
        print(f"{'='*50}")

        # Self-play phase
        print("\nSelf-play phase:")
        outcomes = agent.self_play(num_games=games_per_iteration)

        # Training phase
        print("\nTraining phase:")
        loss = agent.train(epochs=training_epochs)

        # Evaluation phase
        if (iteration + 1) % evaluate_every == 0:
            print("\nEvaluation phase:")
            win_rate = agent.evaluate_against_random(num_games=eval_games)

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                agent.save_checkpoint(
                    f"alphazero_tictactoe_{board_size}x{board_size}_{win_length}in_best.pt"
                )
                print(f"New best model with win rate: {win_rate:.1f}%")

        # Save checkpoint
        if (iteration + 1) % checkpoint_interval == 0:
            agent.save_checkpoint(
                f"alphazero_tictactoe_{board_size}x{board_size}_{win_length}in_iter{iteration+1}.pt"
            )

        # Print time elapsed
        elapsed = time.time() - start_time
        print(
            f"\nTime elapsed: {elapsed//3600:.0f}h {(elapsed%3600)//60:.0f}m {elapsed%60:.0f}s"
        )

    # Save final model
    agent.save_checkpoint(
        f"alphazero_tictactoe_{board_size}x{board_size}_{win_length}in_final.pt"
    )

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AlphaZero for TicTacToe")
    parser.add_argument(
        "--board_size", type=int, default=5, help="Board size (default: 5)"
    )
    parser.add_argument(
        "--win_length", type=int, default=4, help="Win condition length (default: 4)"
    )
    parser.add_argument(
        "--iterations", type=int, default=50, help="Number of training iterations"
    )
    parser.add_argument(
        "--games", type=int, default=50, help="Self-play games per iteration"
    )
    parser.add_argument(
        "--simulations", type=int, default=800, help="MCTS simulations per move"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Training epochs per iteration"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1,
        help="Save checkpoint every N iterations",
    )
    parser.add_argument(
        "--no_resume",
        action="store_false",
        dest="resume",
        help="Don't resume from checkpoint",
    )
    parser.add_argument(
        "--evaluate_every",
        type=int,
        default=5,
        help="Evaluate against random player every N iterations",
    )
    parser.add_argument(
        "--eval_games", type=int, default=20, help="Number of evaluation games"
    )

    args = parser.parse_args()

    # Train AlphaZero
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
