# AI Tic Tac Toe - Project

## Group Members

- **Faizan Ali** (22I-2496)
- **Zaid Vohra** (22K-4195)
- **Hamail Rehman** (22K-4443)

---

## Project Overview

This project implements an AI agent that plays a variant of **Tic Tac Toe** on a **5x5 grid** where the goal is to get **4 in a row** (horizontally, vertically, or diagonally) instead of the traditional 3. The AI agent uses the **Minimax algorithm** with **alpha-beta pruning** and leverages various optimizations such as **heuristic evaluation**, **move ordering**, and **caching** to make decisions more efficiently.

---

## Features

### 1. **Minimax Algorithm with Alpha-Beta Pruning**

- The core of the AI decision-making process is the **Minimax algorithm**, which recursively explores all possible game states to select the best move.
- **Alpha-beta pruning** is used to optimize the Minimax search by pruning branches that do not need to be explored, thereby improving efficiency by eliminating unnecessary calculations.

### 2. **Heuristic Evaluation**

- The agent uses a custom heuristic to evaluate the quality of a game state. This includes a scoring system based on the number of consecutive marks (1-4) in rows, columns, and diagonals.
- The heuristic assigns:

  - **4 in a row** (winning move): 10,000 points.
  - **3 in a row**: 100 points.
  - **2 in a row**: 10 points.
  - **1 in a row**: 1 point.
  - **Block opponent’s winning move**: Weighted with a **BLOCK_WEIGHT** factor of **1.2**.

### 3. **Move Ordering**

- Moves are ordered based on their potential impact on the game. By evaluating each possible move with a shallow evaluation (just looking at the current state), the AI explores the most promising moves first.
- This optimization improves the performance of **alpha-beta pruning** by reducing the number of states that need to be explored, allowing the AI to make better decisions faster.

### 4. **Caching**

- The AI caches previously computed game states to avoid redundant calculations. This helps speed up the decision-making process for repeating game states.
- The **board_hash** function generates a unique key for each game state, and the results are stored in a cache to be reused later.

### 5. **State Exploration Tracking**

- The agent tracks the number of states it explores during each move. This provides insight into the performance and efficiency of the algorithm and can be printed for debugging or analysis.

---

## Project Files

### 1. **`config.py`**

- Contains all the configuration constants used throughout the project.
- Defines scoring constants for various line lengths (1-4 in a row), as well as the **BLOCK_WEIGHT** used to penalize the agent for blocking the opponent.

### 2. **`agent.py`**

- Defines the **MinimaxAgent** class, which is responsible for implementing the **Minimax algorithm with alpha-beta pruning**.
- Includes methods for:

  - `choose_move()`: Chooses the best move based on the current game state.
  - `minimax()`: The recursive minimax function that explores the game tree and returns the best move.
  - `evaluate()`: The heuristic evaluation function used to score different game states.
  - `cache`: Stores previously computed game states to avoid redundant evaluations.
  - **Move ordering**: Optimizes the search by evaluating moves in order of their impact on the game.

### 3. **`game.py`**

- Contains the game logic, including the **game loop**, **player interactions**, and **AI turn**.
- Manages the state of the 5x5 board and determines the winner or if the game ends in a draw.
- Handles input from players (human or AI) and prints the current state of the board after every move.

### 4. **`interface.py`**

- The entry point for running the game. This file allows the user to interact with the game via the console and runs the entire application.
- Handles user input, displays the board, and invokes the **MinimaxAgent** to play against the user or another AI.

---

## How It Works

### 1. **Minimax Algorithm with Alpha-Beta Pruning**

The agent uses a recursive search approach with **Minimax** to explore all possible moves:

- **Maximizing Player**: The agent tries to maximize its score by choosing the move that leads to the best outcome.
- **Minimizing Player**: The opponent tries to minimize the agent’s score by blocking winning moves.
- **Alpha-Beta Pruning**: As the search progresses, branches that cannot affect the final outcome are pruned, reducing the number of explored states.

### 2. **Move Selection Process**

1. **Evaluate Moves**: The agent evaluates all valid moves based on a heuristic evaluation of the resulting game state.
2. **Order Moves**: Moves are ordered based on the heuristic score, prioritizing more promising moves.
3. **Explore State Tree**: The agent recursively explores the game tree, using alpha-beta pruning to minimize unnecessary evaluations.
4. **Choose Optimal Move**: The agent selects the move with the highest score (for maximizing) or the lowest score (for minimizing), based on the current turn.

### 3. **Caching for Efficiency**

- A unique **board hash** is generated for each game state. If a state has been encountered before, the results are retrieved from the cache, reducing redundant calculations.
- This significantly speeds up the decision-making process, especially in games where states are repeated or symmetrical.

### 4. **Tracking State Exploration**

- During each move, the number of explored states is tracked and printed to the console. This helps in understanding the performance of the AI and how efficiently it explores the search space.

---

## Running the Game

1. **Clone the Repository**:

   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install Dependencies**:
   If any external dependencies are needed, install them using:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Game**:
   The game can be run using the `interface.py` file, which serves as the main entry point.

   ```bash
   python interface.py
   ```

   The player will be prompted to make moves in the console. The AI will automatically make its moves after the player’s turn.

---

## Performance Metrics

- **State Exploration Count**: The number of game states explored during the Minimax search is displayed after every move. This metric helps evaluate the performance and efficiency of the AI.
