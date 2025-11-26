"""
AlphaZero-inspired Agent for ConnectX
Kaggle Submission Ready - Simplified MCTS with Neural Network Heuristic
"""

import random
import numpy as np
import math


# ============================================================================
# Configuration
# ============================================================================

MCTS_SIMULATIONS = 80  # Increased for better play
C_PUCT = 1.4          # Exploration constant for UCB


# ============================================================================
# Board Utilities
# ============================================================================

def to_grid(board_flat, cfg):
    """Convert 1D board to 2D grid."""
    return np.asarray(board_flat).reshape(cfg.rows, cfg.columns)


def get_valid_actions(board, cfg):
    """Get list of valid column indices."""
    return [c for c in range(cfg.columns) if board[c] == 0]


def drop_piece(grid, col, player, cfg):
    """Drop a piece in column and return new grid."""
    new_grid = grid.copy()
    for r in range(cfg.rows - 1, -1, -1):
        if new_grid[r, col] == 0:
            new_grid[r, col] = player
            break
    return new_grid


def check_win(grid, player, cfg):
    """Check if player has won."""
    rows, cols, n = cfg.rows, cfg.columns, cfg.inarow
    
    # Horizontal
    for r in range(rows):
        for c in range(cols - n + 1):
            if all(grid[r, c + i] == player for i in range(n)):
                return True
    
    # Vertical
    for r in range(rows - n + 1):
        for c in range(cols):
            if all(grid[r + i, c] == player for i in range(n)):
                return True
    
    # Diagonal ↘
    for r in range(rows - n + 1):
        for c in range(cols - n + 1):
            if all(grid[r + i, c + i] == player for i in range(n)):
                return True
    
    # Diagonal ↗
    for r in range(n - 1, rows):
        for c in range(cols - n + 1):
            if all(grid[r - i, c + i] == player for i in range(n)):
                return True
    
    return False


def is_terminal(grid, cfg):
    """Check if game is over."""
    # Check win for both players
    if check_win(grid, 1, cfg) or check_win(grid, 2, cfg):
        return True
    
    # Check draw (board full)
    if np.all(grid[0, :] != 0):
        return True
    
    return False


def get_winner(grid, cfg):
    """Get winner (1, 2, or 0 for draw)."""
    if check_win(grid, 1, cfg):
        return 1
    if check_win(grid, 2, cfg):
        return 2
    return 0  # Draw or not terminal


# ============================================================================
# Neural Network Heuristic (Simplified Pattern-Based)
# ============================================================================

# ============================================================================
# Enhanced Strategy: Connectivity and Pattern Recognition
# ============================================================================

def count_potential_lines(grid, player, cfg):
    """
    Count potential winning lines (lines that can still become 4-in-a-row).
    This measures connectivity and strategic positioning.
    """
    rows, cols, n = cfg.rows, cfg.columns, cfg.inarow
    opponent = 3 - player
    potential_lines = 0
    
    # Check all possible windows
    windows = []
    
    # Horizontal
    for r in range(rows):
        for c in range(cols - n + 1):
            windows.append([grid[r, c + i] for i in range(n)])
    
    # Vertical
    for r in range(rows - n + 1):
        for c in range(cols):
            windows.append([grid[r + i, c] for i in range(n)])
    
    # Diagonal ↘
    for r in range(rows - n + 1):
        for c in range(cols - n + 1):
            windows.append([grid[r + i, c + i] for i in range(n)])
    
    # Diagonal ↗
    for r in range(n - 1, rows):
        for c in range(cols - n + 1):
            windows.append([grid[r - i, c + i] for i in range(n)])
    
    # Evaluate each window for potential
    for window in windows:
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(0)
        
        # A line is "potential" if it has no opponent pieces
        if opponent_count == 0 and player_count > 0:
            # Weight by how many pieces we already have
            if player_count == 3:
                potential_lines += 100  # Almost winning
            elif player_count == 2:
                potential_lines += 10   # Good position
            elif player_count == 1:
                potential_lines += 1    # Starting position
    
    return potential_lines


def evaluate_connectivity(grid, col, player, cfg):
    """
    Evaluate how well a move at 'col' connects with existing pieces.
    Measures potential for future 4-in-a-rows in all directions.
    """
    rows, cols = cfg.rows, cfg.columns
    
    # Find where piece would land
    row = None
    for r in range(rows - 1, -1, -1):
        if grid[r, col] == 0:
            row = r
            break
    
    if row is None:
        return 0  # Column full
    
    connectivity_score = 0
    
    # Check all 4 directions for connectivity
    directions = [
        (0, 1),   # Horizontal →
        (1, 0),   # Vertical ↓
        (1, 1),   # Diagonal ↘
        (1, -1),  # Diagonal ↗
    ]
    
    for dr, dc in directions:
        # Count connected pieces in both directions
        connected = 1  # The piece we're placing
        
        # Check positive direction
        for dist in range(1, 4):
            r, c = row + dr * dist, col + dc * dist
            if 0 <= r < rows and 0 <= c < cols:
                if grid[r, c] == player:
                    connected += 1
                elif grid[r, c] != 0:
                    break  # Blocked by opponent
            else:
                break
        
        # Check negative direction
        for dist in range(1, 4):
            r, c = row - dr * dist, col - dc * dist
            if 0 <= r < rows and 0 <= c < cols:
                if grid[r, c] == player:
                    connected += 1
                elif grid[r, c] != 0:
                    break  # Blocked by opponent
            else:
                break
        
        # Score based on connectivity
        if connected >= 4:
            connectivity_score += 100000  # Winning move
        elif connected == 3:
            connectivity_score += 5000    # Strong threat
        elif connected == 2:
            connectivity_score += 500     # Building position
        else:
            connectivity_score += 50      # Starting position
    
    return connectivity_score


def check_double_threat(grid, col, player, cfg):
    """
    Check if placing at 'col' creates multiple winning threats.
    This is a powerful tactical pattern.
    """
    rows, cols = cfg.rows, cfg.columns
    
    # Simulate the move
    test_grid = drop_piece(grid, col, player, cfg)
    
    # Count how many ways we can win on the next move
    winning_moves = 0
    
    for next_col in range(cols):
        if test_grid[0, next_col] == 0:  # Can place here
            next_grid = drop_piece(test_grid, next_col, player, cfg)
            if check_win(next_grid, player, cfg):
                winning_moves += 1
    
    # Score based on number of threats
    if winning_moves >= 2:
        return 50000  # Double threat (opponent can't block both)
    elif winning_moves == 1:
        return 10000  # Single threat
    else:
        return 0


def evaluate_diagonal_potential(grid, col, player, cfg):
    """
    Specifically evaluate diagonal connection potential.
    Diagonals are harder to block and very powerful.
    """
    rows, cols = cfg.rows, cfg.columns
    
    # Find landing position
    row = None
    for r in range(rows - 1, -1, -1):
        if grid[r, col] == 0:
            row = r
            break
    
    if row is None:
        return 0
    
    diagonal_score = 0
    
    # Check both diagonals
    for dr, dc in [(1, 1), (1, -1)]:
        # Count empty spaces and our pieces in diagonal
        diagonal_line = []
        
        # Collect 7 positions along diagonal (enough to find 4-in-a-row)
        for dist in range(-3, 4):
            r, c = row + dr * dist, col + dc * dist
            if 0 <= r < rows and 0 <= c < cols:
                diagonal_line.append(grid[r, c])
        
        # Look for patterns in diagonal
        if len(diagonal_line) >= 4:
            for i in range(len(diagonal_line) - 3):
                window = diagonal_line[i:i+4]
                player_count = window.count(player)
                opponent_count = window.count(opponent := 3 - player)
                
                # Only count if not blocked by opponent
                if opponent_count == 0:
                    if player_count == 3:
                        diagonal_score += 8000
                    elif player_count == 2:
                        diagonal_score += 2000
                    elif player_count == 1:
                        diagonal_score += 200
    
    return diagonal_score


def evaluate_window(window, player):
    """Evaluate a single window with strategic weights."""
    opponent = 3 - player
    score = 0
    
    player_count = window.count(player)
    opponent_count = window.count(opponent)
    empty_count = window.count(0)
    
    # Immediate win
    if player_count == 4:
        return 1000000
    
    # Strong offensive patterns
    if opponent_count == 0:  # Not blocked
        if player_count == 3 and empty_count == 1:
            score += 15000  # One move from winning
        elif player_count == 2 and empty_count == 2:
            score += 500    # Building threat
        elif player_count == 1 and empty_count == 3:
            score += 50     # Early position
    
    # Strong defensive patterns
    if player_count == 0:  # Pure opponent threat
        if opponent_count == 3 and empty_count == 1:
            score -= 10000  # Must block
        elif opponent_count == 2 and empty_count == 2:
            score -= 400    # Growing threat
    
    return score


def heuristic_evaluate(grid, player, cfg):
    """
    Strategic heuristic focusing on:
    1. Connectivity (can pieces connect to form 4?)
    2. Multiple directions (horizontal, vertical, diagonals)
    3. Potential winning lines
    """
    rows, cols, n = cfg.rows, cfg.columns, cfg.inarow
    score = 0
    
    # 1. Evaluate all windows for immediate patterns
    # Horizontal
    for r in range(rows):
        for c in range(cols - n + 1):
            window = [grid[r, c + i] for i in range(n)]
            score += evaluate_window(window, player)
    
    # Vertical
    for r in range(rows - n + 1):
        for c in range(cols):
            window = [grid[r + i, c] for i in range(n)]
            score += evaluate_window(window, player)
    
    # Diagonal ↘
    for r in range(rows - n + 1):
        for c in range(cols - n + 1):
            window = [grid[r + i, c + i] for i in range(n)]
            score += evaluate_window(window, player) * 1.2  # Diagonals slightly more valuable
    
    # Diagonal ↗
    for r in range(n - 1, rows):
        for c in range(cols - n + 1):
            window = [grid[r - i, c + i] for i in range(n)]
            score += evaluate_window(window, player) * 1.2
    
    # 2. Count potential winning lines (long-term strategy)
    potential = count_potential_lines(grid, player, cfg)
    score += potential * 5
    
    # 3. Penalize moves that give opponent winning opportunities
    opponent = 3 - player
    opponent_potential = count_potential_lines(grid, opponent, cfg)
    score -= opponent_potential * 3
    
    return score


def get_policy_value(grid, player, cfg):
    """
    Enhanced policy-value with strategic move evaluation.
    Focuses on:
    1. Immediate wins/blocks
    2. Creating multiple threats
    3. Diagonal opportunities
    4. Connectivity
    """
    valid_actions = get_valid_actions(grid.flatten().tolist(), cfg)
    
    if not valid_actions:
        return {}, 0
    
    opponent = 3 - player
    action_scores = {}
    
    for action in valid_actions:
        new_grid = drop_piece(grid, action, player, cfg)
        score = 0
        
        # PRIORITY 1: Immediate win (highest priority)
        if check_win(new_grid, player, cfg):
            action_scores[action] = 10000000
            continue
        
        # PRIORITY 2: Block opponent win (must do)
        opp_grid = drop_piece(grid, action, opponent, cfg)
        if check_win(opp_grid, opponent, cfg):
            action_scores[action] = 9000000
            continue
        
        # PRIORITY 3: Create double/multiple threats (very strong)
        double_threat_score = check_double_threat(grid, action, player, cfg)
        score += double_threat_score
        
        # PRIORITY 4: Connectivity (can this lead to 4-in-a-row?)
        connectivity_score = evaluate_connectivity(grid, action, player, cfg)
        score += connectivity_score
        
        # PRIORITY 5: Diagonal potential (hard to block)
        diagonal_score = evaluate_diagonal_potential(grid, action, player, cfg)
        score += diagonal_score
        
        # PRIORITY 6: General board evaluation
        base_score = heuristic_evaluate(new_grid, player, cfg)
        score += base_score
        
        # PENALTY: Don't create winning opportunities for opponent
        # Check if opponent can win after our move
        for opp_action in get_valid_actions(new_grid.flatten().tolist(), cfg):
            test_grid = drop_piece(new_grid, opp_action, opponent, cfg)
            if check_win(test_grid, opponent, cfg):
                score -= 50000  # Heavy penalty
        
        action_scores[action] = score
    
    # Convert to probabilities with temperature
    max_score = max(action_scores.values())
    min_score = min(action_scores.values())
    
    # Normalize scores to avoid overflow
    if max_score > min_score:
        normalized_scores = {
            a: (s - min_score) / (max_score - min_score) 
            for a, s in action_scores.items()
        }
    else:
        normalized_scores = {a: 1.0 for a in action_scores.keys()}
    
    # Apply temperature-based softmax
    temperature = 0.5  # Lower = more focused on best move
    exp_scores = {a: math.exp(s / temperature) for a, s in normalized_scores.items()}
    total = sum(exp_scores.values())
    policy = {a: s / total for a, s in exp_scores.items()}
    
    # Value estimation
    best_score = max(action_scores.values())
    value = best_score / 1000000.0
    value = max(-1.0, min(1.0, value))
    
    return policy, value


# ============================================================================
# MCTS Node
# ============================================================================

class MCTSNode:
    def __init__(self, grid, player, cfg, parent=None, action=None, prior=0):
        self.grid = grid
        self.player = player
        self.cfg = cfg
        self.parent = parent
        self.action = action
        self.prior = prior
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        
    def is_expanded(self):
        return len(self.children) > 0
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, child):
        """Calculate UCB score for child."""
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = child.value()
        
        u_value = C_PUCT * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
        return q_value + u_value
    
    def select_child(self):
        """Select child with highest UCB score."""
        return max(self.children.values(), key=lambda c: self.ucb_score(c))
    
    def expand(self):
        """Expand node with all valid actions."""
        if is_terminal(self.grid, self.cfg):
            return
        
        policy, _ = get_policy_value(self.grid, self.player, self.cfg)
        
        valid_actions = get_valid_actions(self.grid.flatten().tolist(), self.cfg)
        
        for action in valid_actions:
            if action not in self.children:
                new_grid = drop_piece(self.grid, action, self.player, self.cfg)
                prior = policy.get(action, 1.0 / len(valid_actions))
                
                self.children[action] = MCTSNode(
                    grid=new_grid,
                    player=3 - self.player,  # Switch player
                    cfg=self.cfg,
                    parent=self,
                    action=action,
                    prior=prior
                )
    
    def backpropagate(self, value):
        """Backpropagate value up the tree."""
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent:
            self.parent.backpropagate(-value)  # Negate for opponent


# ============================================================================
# MCTS Search
# ============================================================================

def mcts_search(grid, player, cfg, num_simulations=MCTS_SIMULATIONS):
    """
    Run MCTS search and return best action.
    """
    root = MCTSNode(grid, player, cfg)
    root.expand()
    
    # Run simulations
    for _ in range(num_simulations):
        node = root
        
        # Selection
        while node.is_expanded() and not is_terminal(node.grid, cfg):
            node = node.select_child()
        
        # Expansion
        if not is_terminal(node.grid, cfg):
            node.expand()
            if node.children:
                node = random.choice(list(node.children.values()))
        
        # Evaluation
        if is_terminal(node.grid, cfg):
            winner = get_winner(node.grid, cfg)
            if winner == player:
                value = 1.0
            elif winner == 3 - player:
                value = -1.0
            else:
                value = 0.0
        else:
            _, value = get_policy_value(node.grid, node.player, cfg)
        
        # Backpropagation
        node.backpropagate(value)
    
    # Select action with most visits
    if not root.children:
        valid_actions = get_valid_actions(grid.flatten().tolist(), cfg)
        return random.choice(valid_actions) if valid_actions else 0
    
    best_action = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
    return best_action


# ============================================================================
# Main Agent Function (Kaggle Submission Format)
# ============================================================================

def my_agent(obs, config):
    """
    AlphaZero-inspired agent using MCTS with heuristic evaluation.
    
    Args:
        obs: Observation object with 'board' and 'mark'
        config: Configuration object with game parameters
    
    Returns:
        int: Column index to play (0-6 for standard ConnectX)
    """
    # Convert board to grid
    grid = to_grid(obs.board, config)
    player = obs.mark
    
    # Quick win check
    valid_actions = get_valid_actions(obs.board, config)
    
    # Check for immediate win
    for action in valid_actions:
        test_grid = drop_piece(grid, action, player, config)
        if check_win(test_grid, player, config):
            return action
    
    # Check for must-block
    opponent = 3 - player
    for action in valid_actions:
        test_grid = drop_piece(grid, action, opponent, config)
        if check_win(test_grid, opponent, config):
            return action
    
    # Use MCTS for decision
    try:
        action = mcts_search(grid, player, config, num_simulations=MCTS_SIMULATIONS)
    except:
        # Fallback to random if MCTS fails
        action = random.choice(valid_actions) if valid_actions else 0
    
    return int(action)


# ============================================================================
# Testing (Optional - remove for submission)
# ============================================================================

if __name__ == "__main__":
    from kaggle_environments import make, evaluate
    
    print("=== Testing AlphaZero Agent ===\n")
    
    # Test vs Random
    print("Test 1: my_agent vs random")
    env = make("connectx", debug=True)
    env.run([my_agent, "random"])
    print(env.render(mode="ansi"))
    print()
    
    print("Test 2: random vs my_agent")
    env.reset()
    env.run(["random", my_agent])
    print(env.render(mode="ansi"))
    print()
    
    # Evaluation with correct reward calculation
    print("=== Evaluation (10 games each) ===")
    
    def mean_reward(rewards):
        """
        Calculate win rate for player 1.
        Kaggle evaluate returns: [1, -1] for P1 win, [-1, 1] for P2 win, [0, 0] for draw
        """
        if not rewards:
            print("  Warning: Empty rewards list")
            return 0.0
        
        print(f"  Raw rewards: {rewards}")
        
        player1_wins = sum(1 for r in rewards if r[0] == 1)
        player2_wins = sum(1 for r in rewards if r[0] == -1)
        draws = sum(1 for r in rewards if r[0] == 0)
        total_games = len(rewards)
        
        print(f"  Player 1 wins: {player1_wins}")
        print(f"  Player 2 wins: {player2_wins}")
        print(f"  Draws: {draws}")
        print(f"  Total games: {total_games}")
        
        if total_games == 0:
            print("  Warning: No games played!")
            return 0.0
        
        win_rate = player1_wins / total_games
        print(f"  Win rate: {win_rate:.1%}")
        return win_rate
    
    print("\n1. Testing: my_agent vs random")
    try:
        result = evaluate("connectx", [my_agent, "random"], num_episodes=10)
        win_rate = mean_reward(result)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n2. Testing: my_agent vs negamax")
    try:
        result = evaluate("connectx", [my_agent, "negamax"], num_episodes=10)
        win_rate = mean_reward(result)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n3. Testing: random vs my_agent")
    try:
        result = evaluate("connectx", ["random", my_agent], num_episodes=10)
        win_rate = mean_reward(result)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n4. Testing: negamax vs my_agent")
    try:
        result = evaluate("connectx", ["negamax", my_agent], num_episodes=10)
        win_rate = mean_reward(result)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Simple direct test
    print("\n=== Direct Single Game Test ===")
    try:
        env = make("connectx", debug=True)
        env.run([my_agent, "random"])
        print("✓ Game completed successfully!")
        
        # Get game result
        final_state = env.state
        if final_state[0].status == 'DONE':
            if final_state[0].reward == 1:
                winner = "my_agent (Player 1) WON! 🎉"
            elif final_state[0].reward == -1:
                winner = "random (Player 2) won"
            else:
                winner = "Draw"
            print(f"Result: {winner}")
        
        print(env.render(mode="ansi"))
    except Exception as e:
        print(f"Direct test failed: {e}")
        import traceback
        traceback.print_exc()