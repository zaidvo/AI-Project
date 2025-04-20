import argparse
import numpy as np
from tqdm import tqdm

from tictactoe_env import TicTacToeEnv
from dqn_agent import DQNAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Test trained DQN agent performance")
    parser.add_argument(
        "--model_path",
        type=str,
        default="model_final.pt",
        help="Path to the trained model",
    )
    parser.add_argument("--games", type=int, default=100, help="Number of test games")
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "self"],
        help="Type of opponent (random or self)",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render games (slows down testing)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed game info"
    )
    return parser.parse_args()


def random_agent(env):
    """Simple random agent that selects randomly from valid moves"""
    valid_moves = env.get_valid_moves()
    if valid_moves:
        return np.random.choice(valid_moves)
    return None


def test_against_random(env, agent, num_games, render=False, verbose=False):
    """Test agent against random opponent"""
    wins = 0
    losses = 0
    draws = 0
    game_lengths = []

    for game in tqdm(range(num_games)):
        state = env.reset()
        moves = 0

        while not env.done:
            # Agent's turn (always plays as X)
            if env.current_player == 1:
                valid_moves = env.get_valid_moves()
                action = agent.select_action(state, valid_moves)
            else:
                # Random agent's turn
                action = random_agent(env)

            state, _, done, _ = env.step(action)
            moves += 1

            if render:
                env.render()

        # Record game result
        if env.winner == 1:
            wins += 1
            result = "Win"
        elif env.winner == -1:
            losses += 1
            result = "Loss"
        else:
            draws += 1
            result = "Draw"

        game_lengths.append(moves)

        if verbose:
            print(f"Game {game+1}: {result} in {moves} moves")

    # Calculate statistics
    win_rate = wins / num_games
    loss_rate = losses / num_games
    draw_rate = draws / num_games
    avg_game_length = np.mean(game_lengths)

    print("\nTest Results against Random Opponent:")
    print(f"Win Rate: {win_rate:.4f} ({wins}/{num_games})")
    print(f"Loss Rate: {loss_rate:.4f} ({losses}/{num_games})")
    print(f"Draw Rate: {draw_rate:.4f} ({draws}/{num_games})")
    print(f"Average Game Length: {avg_game_length:.2f} moves")

    return win_rate, loss_rate, draw_rate


def test_self_play(env, agent, num_games, render=False, verbose=False):
    """Test agent playing against itself"""
    player1_wins = 0  # X
    player2_wins = 0  # O
    draws = 0
    game_lengths = []

    for game in tqdm(range(num_games)):
        state = env.reset()
        moves = 0

        while not env.done:
            valid_moves = env.get_valid_moves()
            action = agent.select_action(state, valid_moves)
            state, _, done, _ = env.step(action)
            moves += 1

            if render:
                env.render()

        # Record game result
        if env.winner == 1:
            player1_wins += 1
            result = "Player 1 (X) Win"
        elif env.winner == -1:
            player2_wins += 1
            result = "Player 2 (O) Win"
        else:
            draws += 1
            result = "Draw"

        game_lengths.append(moves)

        if verbose:
            print(f"Game {game+1}: {result} in {moves} moves")

    # Calculate statistics
    player1_win_rate = player1_wins / num_games
    player2_win_rate = player2_wins / num_games
    draw_rate = draws / num_games
    avg_game_length = np.mean(game_lengths)

    print("\nTest Results for Self-Play:")
    print(f"Player 1 (X) Win Rate: {player1_win_rate:.4f} ({player1_wins}/{num_games})")
    print(f"Player 2 (O) Win Rate: {player2_win_rate:.4f} ({player2_wins}/{num_games})")
    print(f"Draw Rate: {draw_rate:.4f} ({draws}/{num_games})")
    print(f"Average Game Length: {avg_game_length:.2f} moves")

    return player1_win_rate, player2_win_rate, draw_rate


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

    # Set agent to evaluation mode (no exploration)
    agent.policy_net.eval()
    agent.epsilon_start = 0
    agent.epsilon_end = 0

    # Run tests
    if args.opponent == "random":
        test_against_random(env, agent, args.games, args.render, args.verbose)
    else:  # self-play
        test_self_play(env, agent, args.games, args.render, args.verbose)


if __name__ == "__main__":
    main()
