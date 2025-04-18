import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Board
from game.constants import *
from ai.dqn_agent import DQNAgent

def train_dqn():
    """Train DQN agent for MegaTicTacToe"""
    # Training parameters
    episodes = 1000
    epsilon_decay = 0.995
    save_interval = 200
    print_interval = 100

    # Paths - now using .weights.h5 extension consistently
    model_path = os.path.join(os.path.dirname(__file__), "dqn_megatictactoe.weights.h5")
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize board and agent
    board = Board()
    agent = DQNAgent(
        state_size=BOARD_SIZE * BOARD_SIZE,
        action_size=BOARD_SIZE * BOARD_SIZE,
        hidden_layers=[128, 64],
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=epsilon_decay,
        epsilon_min=0.01,
        memory_size=10000,
        batch_size=64,
        target_update_freq=1000
    )

    # Try to load existing model (with .weights.h5 extension)
    if os.path.exists(model_path):
        print("Loading existing model...")
        agent.load(model_path)

    # Metrics tracking
    rewards_history = []
    win_history = []
    loss_history = []
    draw_history = []
    epsilons = []
    losses = []

    print(f"Starting training for {episodes} episodes")

    # Training loop
    for episode in tqdm(range(episodes)):
        board.reset()
        total_reward = 0
        game_over = False

        while not game_over:
            state = board.get_state()
            valid_moves = board.get_valid_moves()

            # Choose action
            action = agent.act(state, valid_moves)

            # Make move
            valid_move = board.make_move(action)

            # Get reward
            reward = 0
            if not valid_move:
                reward = REWARD_INVALID_MOVE
                game_over = True
            else:
                winner = board.check_win()
                if winner is not None:
                    if winner == PLAYER_X:
                        reward = REWARD_WIN
                    else:
                        reward = REWARD_LOSS
                    game_over = True
                elif board.is_draw():
                    reward = REWARD_DRAW
                    game_over = True

            next_state = board.get_state()

            # Store experience
            agent.remember(state, action, reward, next_state, game_over)
            total_reward += reward

            # Train
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                losses.append(loss)

        # Track metrics
        rewards_history.append(total_reward)
        winner = board.check_win()
        if winner == PLAYER_X:
            win_history.append(1)
            loss_history.append(0)
            draw_history.append(0)
        elif winner == PLAYER_O:
            win_history.append(0)
            loss_history.append(1)
            draw_history.append(0)
        else:
            win_history.append(0)
            loss_history.append(0)
            draw_history.append(1)

        epsilons.append(agent.epsilon)

        # Print progress
        if (episode + 1) % print_interval == 0:
            win_rate = np.mean(win_history[-print_interval:])
            loss_rate = np.mean(loss_history[-print_interval:])
            draw_rate = np.mean(draw_history[-print_interval:])
            avg_reward = np.mean(rewards_history[-print_interval:])

            print(f"Episode: {episode+1}, Win Rate: {win_rate:.3f}, Loss Rate: {loss_rate:.3f}, "
                  f"Draw Rate: {draw_rate:.3f}, Avg Reward: {avg_reward:.3f}, Epsilon: {agent.epsilon:.3f}")

        # Save checkpoint (now using .weights.h5 extension)
        if (episode + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"dqn_checkpoint_ep{episode+1}.weights.h5")
            agent.save(checkpoint_path)
            print(f"Checkpoint saved at: {checkpoint_path}")

    # Save final model
    agent.save(model_path)
    print("Training complete. Final model saved.")

    # Plot metrics
    plot_metrics(rewards_history, win_history, loss_history, draw_history, epsilons, losses)

def plot_metrics(rewards, wins, losses, draws, epsilons, training_losses):
    """Plot training metrics"""
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    window = 100
    rewards_avg = np.convolve(rewards, np.ones(window)/window, mode='valid') if len(rewards) >= window else rewards
    wins_avg = np.convolve(wins, np.ones(window)/window, mode='valid') if len(wins) >= window else wins
    losses_avg = np.convolve(losses, np.ones(window)/window, mode='valid') if len(losses) >= window else losses
    draws_avg = np.convolve(draws, np.ones(window)/window, mode='valid') if len(draws) >= window else draws

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(rewards)
    if isinstance(rewards_avg, np.ndarray): plt.plot(np.arange(window-1, len(rewards)), rewards_avg, 'r-')
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    if isinstance(wins_avg, np.ndarray): plt.plot(np.arange(window-1, len(wins)), wins_avg, 'g-', label='Win Rate')
    if isinstance(losses_avg, np.ndarray): plt.plot(np.arange(window-1, len(losses)), losses_avg, 'r-', label='Loss Rate')
    if isinstance(draws_avg, np.ndarray): plt.plot(np.arange(window-1, len(draws)), draws_avg, 'b-', label='Draw Rate')
    plt.title('Win/Loss/Draw Rates')
    plt.xlabel('Episode')
    plt.ylabel('Rate')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(epsilons)
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.grid(True)

    if len(training_losses) > 0:
        plt.subplot(2, 2, 4)
        plt.plot(training_losses)
        plt.title('Training Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "training_metrics.png"))
    plt.close()

    print("Training metrics saved to plots directory.")

if __name__ == "__main__":
    train_dqn()