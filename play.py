import os
import argparse
from env import TicTacToeEnv
from agent import AlphaZeroAgent


def find_best_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint in the directory"""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not checkpoint_files:
        return None

    # Sort by modification time (newest first)
    checkpoint_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True
    )

    return os.path.join(checkpoint_dir, checkpoint_files[0])


def play_agent_vs_agent(agent1, agent2, num_games=1, render=True):
    """Have two agents play against each other"""
    wins_agent1 = 0
    wins_agent2 = 0
    draws = 0

    for game in range(num_games):
        env = TicTacToeEnv(board_size=agent1.board_size, win_length=agent1.win_length)

        if render:
            print(f"\nGame {game+1}/{num_games}")

        turn = 0
        while not env.done:
            if render:
                env.render()

            # Determine which agent's turn it is
            current_agent = agent1 if env.current_player == 1 else agent2

            # Get agent's move
            action, _ = current_agent.select_action(env, temperature=0)
            x, y = action // env.board_size, action % env.board_size

            if render:
                print(
                    f"Agent {'1' if env.current_player == 1 else '2'} plays: ({x}, {y})"
                )

            _, reward, done, info = env.step(action)
            turn += 1

        # Game over
        if render:
            env.render()

        if env.winner == 1:
            wins_agent1 += 1
            if render:
                print("Agent 1 wins!")
        elif env.winner == -1:
            wins_agent2 += 1
            if render:
                print("Agent 2 wins!")
        else:
            draws += 1
            if render:
                print("Game ended in a draw!")

    # Print summary
    print(f"\nResults over {num_games} games:")
    print(f"Agent 1 wins: {wins_agent1} ({wins_agent1/num_games*100:.1f}%)")
    print(f"Agent 2 wins: {wins_agent2} ({wins_agent2/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")

    return wins_agent1, wins_agent2, draws


def play_against_agent(agent, human_player=1):
    """Play a game against the agent"""
    env = TicTacToeEnv(board_size=agent.board_size, win_length=agent.win_length)
    env.reset()

    print(
        f"Playing {agent.board_size}x{agent.board_size} TicTacToe with {agent.win_length} in a row to win"
    )
    print(
        f"You are {'X' if human_player == 1 else 'O'}, and {'go first' if human_player == 1 else 'go second'}"
    )
    print("Enter your moves as 'row col' (0-indexed)")

    while not env.done:
        env.render()

        if env.current_player == human_player:
            # Human player's turn
            valid_moves = env.get_valid_moves()
            print("Valid moves (0-indexed):")
            valid_moves_list = []
            for i in range(env.board_size):
                for j in range(env.board_size):
                    if valid_moves[i, j]:
                        valid_moves_list.append(f"({i},{j})")

            # Print valid moves in a readable format
            for i in range(0, len(valid_moves_list), 5):
                print(" ".join(valid_moves_list[i : i + 5]))

            # Get human input
            while True:
                try:
                    move = input("\nYour move (row col): ")
                    if move.lower() == "q" or move.lower() == "quit":
                        print("Game aborted.")
                        return

                    x, y = map(int, move.split())
                    if (
                        0 <= x < env.board_size
                        and 0 <= y < env.board_size
                        and valid_moves[x, y]
                    ):
                        break
                    else:
                        print("Invalid move! Try again.")
                except ValueError:
                    print(
                        "Invalid input! Please enter row and column as two numbers separated by a space."
                    )

            _, reward, done, info = env.step((x, y))

        else:
            # Agent's turn
            print("Agent is thinking...")
            action, _ = agent.select_action(env, temperature=0)
            x, y = action // env.board_size, action % env.board_size
            print(f"Agent plays: ({x}, {y})")
            _, reward, done, info = env.step(action)

    # Game over
    env.render()
    winner = info["winner"]

    if winner == 0:
        print("Game ended in a draw!")
    elif winner == human_player:
        print("You win!")
    else:
        print("Agent wins!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Play TicTacToe against trained AlphaZero agent"
    )
    parser.add_argument(
        "--board_size", type=int, default=5, help="Board size (default: 5)"
    )
    parser.add_argument(
        "--win_length", type=int, default=4, help="Win condition length (default: 4)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory with checkpoints",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Specific checkpoint to load"
    )
    parser.add_argument(
        "--player",
        type=int,
        default=1,
        choices=[1, -1],
        help="Human player: 1=X (first), -1=O (second)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=800,
        help="Number of MCTS simulations per move",
    )

    args = parser.parse_args()

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
    else:
        print("No checkpoint found. Agent will play randomly.")

    # Play game
    play_against_agent(agent, human_player=args.player)
