import numpy as np
import torch
import torch.nn.functional as F

from env import TicTacToeEnv


class Node:
    def __init__(self, prior=0):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None
        self.reward = 0
        self.done = False

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature=1):
        """Select action based on visit counts and temperature"""
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = list(self.children.keys())

        if temperature == 0 or len(actions) == 1:
            # Greedy selection
            action = actions[np.argmax(visit_counts)]
        else:
            # Sample based on visit count distribution
            visit_count_distribution = visit_counts ** (1 / temperature)
            if np.sum(visit_count_distribution) > 0:  # Avoid division by zero
                visit_count_distribution = visit_count_distribution / np.sum(
                    visit_count_distribution
                )
                action = np.random.choice(actions, p=visit_count_distribution)
            else:
                action = np.random.choice(actions)  # Fallback to uniform random

        return action

    def get_visit_count_policy(self):
        """Convert visit counts to a policy distribution"""
        visit_counts = np.zeros(self.state.board_size * self.state.board_size)
        for action, child in self.children.items():
            visit_counts[action] = child.visit_count

        if np.sum(visit_counts) > 0:
            return visit_counts / np.sum(visit_counts)
        return visit_counts  # All zeros


class MCTS:
    def __init__(
        self,
        network,
        device,
        num_simulations=800,
        dirichlet_alpha=0.3,
        dirichlet_weight=0.25,
        exploration_constant=1.25,
        board_size=5,
        win_length=4,
    ):
        self.network = network
        self.device = device
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.exploration_constant = exploration_constant
        self.board_size = board_size
        self.win_length = win_length

    def search(self, env, add_exploration_noise=False):
        """Perform MCTS search starting from the given environment state"""
        # Create root node
        root = Node(0)
        root.state = self._clone_env(env)

        # Get valid moves and network prediction
        observation = root.state.get_observation()
        tensor_observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.network(tensor_observation)
            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

        valid_moves = root.state.get_valid_moves_flat()

        # Mask invalid moves and renormalize
        policy = policy * valid_moves
        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy /= policy_sum

        # Add Dirichlet noise to the root policy for exploration
        if add_exploration_noise:
            noise = np.random.dirichlet(
                [self.dirichlet_alpha] * np.count_nonzero(valid_moves)
            )
            noise_idx = 0
            noisy_policy = np.zeros_like(policy)
            for i in range(len(policy)):
                if valid_moves[i]:
                    noisy_policy[i] = (
                        policy[i] * (1 - self.dirichlet_weight)
                        + noise[noise_idx] * self.dirichlet_weight
                    )
                    noise_idx += 1
            policy = noisy_policy

            # Renormalize after adding noise
            policy_sum = np.sum(policy)
            if policy_sum > 0:
                policy /= policy_sum

        # Create children for valid moves
        for i in range(len(policy)):
            if valid_moves[i]:
                root.children[i] = Node(prior=policy[i])

        # Perform simulations
        for _ in range(self.num_simulations):
            self._simulate(root)

        return root

    def _simulate(self, node):
        """Run a single MCTS simulation"""
        # If node not expanded, expand it and return evaluation
        if not node.expanded():
            return self._expand_and_evaluate(node)

        # Select best child according to UCB score
        action, child = self._select_child(node)

        # If child state not yet set, apply the action
        if child.state is None:
            child.state = self._clone_env(node.state)
            observation, reward, done, info = child.state.step(action)
            child.reward = reward
            child.done = done

        # If terminal state, return the reward
        if child.done:
            value = child.reward
        else:
            # Otherwise, recursively simulate from child and negate (zero-sum game)
            value = -self._simulate(child)

        # Update statistics
        child.visit_count += 1
        child.value_sum += value

        return value

    def _expand_and_evaluate(self, node):
        """Expand node and evaluate it with the neural network"""
        valid_moves = node.state.get_valid_moves_flat()

        # Get network prediction
        observation = node.state.get_observation()
        tensor_observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.network(tensor_observation)
            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value = value.item()

        # Mask invalid moves and renormalize
        policy = policy * valid_moves
        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy /= policy_sum

        # Create children nodes
        for i in range(len(policy)):
            if valid_moves[i]:
                node.children[i] = Node(prior=policy[i])

        return value

    def _select_child(self, node):
        """Select child with highest UCB score"""
        best_ucb = -float("inf")
        best_action = -1
        best_child = None

        # Calculate UCB score for each child
        for action, child in node.children.items():
            # UCB formula: Q(s,a) + c*P(s,a)*sqrt(N(s))/(1+N(s,a))
            # Where Q is the mean value, P is prior probability, N(s) is parent visits, N(s,a) is child visits
            ucb = child.value() + self.exploration_constant * child.prior * np.sqrt(
                node.visit_count
            ) / (1 + child.visit_count)

            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action
                best_child = child

        return best_action, best_child

    def _clone_env(self, env):
        """Create a deep copy of the environment"""
        cloned = TicTacToeEnv(board_size=self.board_size, win_length=self.win_length)
        cloned.board = env.board.copy()
        cloned.current_player = env.current_player
        cloned.done = env.done
        cloned.winner = env.winner
        cloned.last_move = env.last_move
        return cloned
