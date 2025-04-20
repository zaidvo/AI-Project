import argparse
import torch
import numpy as np

from tictactoe_env import TicTacToeEnv
from dqn_agent import DQNAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Play against trained DQN agent")
    parser.add_argument(
        "--model_path",
        type=str,
        default="model_final.pt",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--human_player",
        type=int,
        default=-1,
        choices=[1, -1],
        help="Human plays as X (1) or O (-1)",
    )
    parser.add_argument("--games", type=int, default=1, help="Number of games to play")
    return parser.parse_args()


def get_human_action(env):
    """Get action from human player"""
    while True:
        try:
            print("Enter row (0-6) and column (0-6) separated by space: ")
            row, col = map(int, input().split())

            if row < 0 or row >= env.size or col < 0 or col >= env.size:
                print(
                    f"Invalid input. Row and column must be between 0 and {env.size-1}"
                )
                continue

            action = row * env.size + col

            if action not in env.get_valid_moves():
                print("Invalid move. Cell is already occupied.")
                continue

            return action
        except ValueError:
            print("Invalid input. Please enter two integers separated by space.")


def display_board(board):
    """Display the board with coordinates"""
    size = board.shape[0]

    # Print column headers
    print("  ", end="")
    for j in range(size):
        print(f" {j} ", end="")
    print()

    # Print separator
    print("  ", end="")
    print("---" * size)

    # Print rows with headers
    symbols = {0: ".", 1: "X", -1: "O"}
    for i in range(size):
        print(f"{i}|", end="")
        for j in range(size):
            print(f" {symbols[board[i, j]]} ", end="")
        print()

    print()


def play_game(env, agent, human_player):
    """Play one game against the agent"""
    state = env.reset()
    print("Game starts!")
    display_board(state)

    while not env.done:
        if env.current_player == human_player:
            print("Your turn!")
            action = get_human_action(env)
        else:
            print("Agent is thinking...")
            valid_moves = env.get_valid_moves()
            action = agent.select_action(state, valid_moves)
            print(f"Agent chooses position: {action // env.size}, {action % env.size}")

        next_state, _, done, info = env.step(action)
        state = next_state

        display_board(state)

        if done:
            if env.winner == human_player:
                print("You win!")
            elif env.winner == -human_player:
                print("Agent wins!")
            else:
                print("It's a draw!")


def main():
    args = parse_args()

    # Initialize environment
    env = TicTacToeEnv()

    # Initialize agent
    agent = DQNAgent()

    # Load trained model
    try:
        agent.load_model(args.model_path)
        print(f"Loaded model from {args.model_path}")
    except:
        print(f"Failed to load model from {args.model_path}")
        return

    # Set agent to evaluation mode
    agent.policy_net.eval()

    human_player = args.human_player
    human_symbol = "X" if human_player == 1 else "O"
    agent_symbol = "O" if human_player == 1 else "X"

    print(f"You are playing as {human_symbol}")
    print(f"Agent is playing as {agent_symbol}")
    print(f"The goal is to get 4 in a row (horizontally, vertically, or diagonally)")

    # Play games
    for game in range(args.games):
        print(f"\nGame {game+1}/{args.games}")
        play_game(env, agent, human_player)

        if game < args.games - 1:
            # Ask if player wants to continue
            response = input("Play another game? (y/n): ")
            if response.lower() != "y":
                break


if __name__ == "__main__":
    main()
