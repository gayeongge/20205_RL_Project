"""
AlphaZero-inspired Agent for ConnectX (Gap Threat Detection)
Kaggle Submission Ready - Window Based Defense
"""

import random
import numpy as np
import math


# ============================================================================
# Configuration & Utils
# ============================================================================

MCTS_SIMULATIONS = 100
C_PUCT = 1.4

def to_grid(board_flat, cfg):
    return np.asarray(board_flat).reshape(cfg.rows, cfg.columns)

def get_valid_actions(board, cfg):
    return [c for c in range(cfg.columns) if board[c] == 0]

def drop_piece(grid, col, player, cfg):
    new_grid = grid.copy()
    for r in range(cfg.rows - 1, -1, -1):
        if new_grid[r, col] == 0:
            new_grid[r, col] = player
            break
    return new_grid

def check_win(grid, player, cfg):
    rows, cols, n = cfg.rows, cfg.columns, cfg.inarow
    # Horizontal
    for r in range(rows):
        for c in range(cols - n + 1):
            if all(grid[r, c + i] == player for i in range(n)): return True
    # Vertical
    for r in range(rows - n + 1):
        for c in range(cols):
            if all(grid[r + i, c] == player for i in range(n)): return True
    # Diagonal ↘
    for r in range(rows - n + 1):
        for c in range(cols - n + 1):
            if all(grid[r + i, c + i] == player for i in range(n)): return True
    # Diagonal ↗
    for r in range(n - 1, rows):
        for c in range(cols - n + 1):
            if all(grid[r - i, c + i] == player for i in range(n)): return True
    return False

def is_terminal(grid, cfg):
    return check_win(grid, 1, cfg) or check_win(grid, 2, cfg) or np.all(grid[0, :] != 0)

def get_winner(grid, cfg):
    if check_win(grid, 1, cfg): return 1
    if check_win(grid, 2, cfg): return 2
    return 0


# ============================================================================
# Advanced Threat Detection (Window Based)
# ============================================================================

def analyze_threat_window(grid, col, opponent, cfg):
    """
    [개선된 로직]
    내가 'col'에 두지 않았을 때, 그 위치를 포함하여 상대가 승리할 수 있는 패턴이 있는지 확인.
    단순 연속이 아니라 '윈도우(4칸)' 내에 상대 돌 개수를 셈.
    
    감지 가능 패턴:
    1. O O O _ (연속 3개)
    2. O O _ O (징검다리: 2개 - 1개)
    3. O _ O O (징검다리: 1개 - 2개)
    
    Returns:
        4: 치명적 위협 (상대가 3개를 이미 둠, 여기 두면 상대 승리)
        3: 잠재적 위협 (상대가 2개를 이미 둠)
        0: 위협 없음
    """
    rows, cols, n = cfg.rows, cfg.columns, cfg.inarow
    
    # 돌이 놓일 위치 계산
    row = None
    for r in range(rows - 1, -1, -1):
        if grid[r, col] == 0:
            row = r
            break
            
    if row is None: return 0 # 꽉 찬 컬럼
    
    max_threat = 0
    
    # 4가지 방향에 대해 윈도우 스캔
    # (dr, dc): 가로, 세로, 대각선↘, 대각선↗
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for dr, dc in directions:
        # 현재 위치(row, col)를 포함하는 모든 4칸짜리 윈도우를 검사
        # 윈도우의 시작점은 (row, col)에서 -3칸 떨어진 곳부터 (row, col)까지 가능
        for offset in range(-3, 1):
            opponent_count = 0
            my_piece_count = 0 # 방해물(내 돌)
            
            # 윈도우 내 4개 셀 검사
            for i in range(4):
                r_check = row + (offset + i) * dr
                c_check = col + (offset + i) * dc
                
                # 보드 범위 밖이면 이 윈도우는 무효
                if not (0 <= r_check < rows and 0 <= c_check < cols):
                    my_piece_count = 99 # 무효 처리용
                    break
                
                cell_value = grid[r_check][c_check]
                
                if r_check == row and c_check == col:
                    # 지금 검사하고 있는 '빈칸' (상대가 여기 두면 어떻게 되나 보는 중)
                    pass
                elif cell_value == opponent:
                    opponent_count += 1
                elif cell_value != 0:
                    my_piece_count += 1 # 내 돌이나 다른 돌이 막고 있음
            
            # 평가
            if my_piece_count == 0: # 방해물이 없을 때만 위협 유효
                if opponent_count == 3:
                    return 4 # [CRITICAL] 상대 3개 + 빈칸 1개 = 즉시 승리 패턴
                elif opponent_count == 2:
                    max_threat = max(max_threat, 3) # [MAJOR] 상대 2개 + 빈칸 1개
                    
    return max_threat


def check_double_threat(grid, col, player, cfg):
    """두 곳 이상의 승리 루트를 만드는지 확인"""
    test_grid = drop_piece(grid, col, player, cfg)
    winning_moves = 0
    valid_cols = get_valid_actions(test_grid.flatten(), cfg)
    for next_col in valid_cols:
        next_grid = drop_piece(test_grid, next_col, player, cfg)
        if check_win(next_grid, player, cfg):
            winning_moves += 1
    if winning_moves >= 2: return 50000
    return 0

def evaluate_connectivity(grid, col, player, cfg):
    """공격적 연결성 평가"""
    rows, cols = cfg.rows, cfg.columns
    row = None
    for r in range(rows - 1, -1, -1):
        if grid[r, col] == 0:
            row = r
            break
    if row is None: return 0
    
    score = 0
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        connected = 1
        for dist in range(1, 4):
            r, c = row + dr * dist, col + dc * dist
            if 0 <= r < rows and 0 <= c < cols and grid[r, c] == player: connected += 1
            else: break
        for dist in range(1, 4):
            r, c = row - dr * dist, col - dc * dist
            if 0 <= r < rows and 0 <= c < cols and grid[r, c] == player: connected += 1
            else: break
        if connected >= 4: score += 100000
        elif connected == 3: score += 5000
        elif connected == 2: score += 500
        else: score += 50
    return score

def heuristic_evaluate(grid, player, cfg):
    rows, cols = cfg.rows, cfg.columns
    score = 0
    center_col = cols // 2
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == player:
                score += 10 - abs(center_col - c)
    return score


# ============================================================================
# Policy Value with Gap Detection
# ============================================================================

def get_policy_value(grid, player, cfg):
    valid_actions = get_valid_actions(grid.flatten().tolist(), cfg)
    if not valid_actions: return {}, 0
    
    opponent = 3 - player
    action_scores = {}
    
    for action in valid_actions:
        score = 0
        new_grid = drop_piece(grid, action, player, cfg)
        
        # 1. 공격: 즉시 승리 (최우선)
        if check_win(new_grid, player, cfg):
            action_scores[action] = 10000000
            continue 
        
        # 2. 방어: 윈도우 기반 위협 분석 (징검다리 포함)
        threat_level = analyze_threat_window(grid, action, opponent, cfg)
        
        if threat_level == 4:
            # 상대가 O O _ O 혹은 O O O _ 상태임. 여기 안 두면 짐.
            score += 9000000 
        elif threat_level == 3:
            # 상대가 O _ O 혹은 O O _ 상태임.
            score += 40000 
            
        # 3. 전략: 더블 위협
        score += check_double_threat(grid, action, player, cfg)
        
        # 4. 기본: 연결성
        score += evaluate_connectivity(grid, action, player, cfg)
        score += heuristic_evaluate(new_grid, player, cfg)
        
        # 5. 페널티: 자살수 방지
        opp_valid_actions = get_valid_actions(new_grid.flatten().tolist(), cfg)
        for opp_action in opp_valid_actions:
            opp_grid = drop_piece(new_grid, opp_action, opponent, cfg)
            if check_win(opp_grid, opponent, cfg):
                score -= 100000
                break
        
        action_scores[action] = score
    
    # Softmax
    max_score = max(action_scores.values())
    min_score = min(action_scores.values())
    if max_score > min_score:
        normalized = {k: (v - min_score) / (max_score - min_score) for k, v in action_scores.items()}
    else:
        normalized = {k: 1.0 for k in action_scores.keys()}
        
    temperature = 0.5
    exp_scores = {k: math.exp(v / temperature) for k, v in normalized.items()}
    total = sum(exp_scores.values())
    policy = {k: v / total for k, v in exp_scores.items()}
    
    value = max(-1.0, min(1.0, max_score / 10000000.0))
    return policy, value


# ============================================================================
# MCTS & Agent
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
        
    def is_expanded(self): return len(self.children) > 0
    
    def value(self):
        if self.visit_count == 0: return 0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, child):
        if child.visit_count == 0: q_value = 0
        else: q_value = -child.value()
        u_value = C_PUCT * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
        return q_value + u_value
    
    def select_child(self):
        return max(self.children.values(), key=lambda c: self.ucb_score(c))
    
    def expand(self):
        if is_terminal(self.grid, self.cfg): return
        policy, _ = get_policy_value(self.grid, self.player, self.cfg)
        valid_actions = get_valid_actions(self.grid.flatten().tolist(), self.cfg)
        for action in valid_actions:
            if action not in self.children:
                new_grid = drop_piece(self.grid, action, self.player, self.cfg)
                prior = policy.get(action, 1.0 / len(valid_actions))
                self.children[action] = MCTSNode(new_grid, 3 - self.player, self.cfg, self, action, prior)
    
    def backpropagate(self, value):
        self.visit_count += 1
        self.value_sum += value
        if self.parent: self.parent.backpropagate(-value)

def mcts_search(grid, player, cfg, num_simulations=MCTS_SIMULATIONS):
    root = MCTSNode(grid, player, cfg)
    root.expand()
    for _ in range(num_simulations):
        node = root
        while node.is_expanded() and not is_terminal(node.grid, cfg):
            node = node.select_child()
        if not is_terminal(node.grid, cfg):
            node.expand()
        if is_terminal(node.grid, cfg):
            winner = get_winner(node.grid, cfg)
            value = 1.0 if winner == node.player else -1.0 if winner == 3 - node.player else 0.0
        else:
            _, value = get_policy_value(node.grid, node.player, cfg)
        node.backpropagate(value)
    if not root.children: return random.choice(get_valid_actions(grid.flatten().tolist(), cfg))
    return max(root.children.items(), key=lambda x: x[1].visit_count)[0]

def my_agent(obs, config):
    grid = to_grid(obs.board, config)
    player = obs.mark
    
    # 0. 즉시 승리
    valid_actions = get_valid_actions(obs.board, config)
    for action in valid_actions:
        if check_win(drop_piece(grid, action, player, config), player, config):
            return action
    
    # 1. 즉시 패배 방어 (한번 더 안전장치)
    opponent = 3 - player
    for action in valid_actions:
        if check_win(drop_piece(grid, action, opponent, config), opponent, config):
            return action
            
    # 2. MCTS
    try:
        return int(mcts_search(grid, player, config))
    except:
        return int(random.choice(valid_actions))
    


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

# ================= Kaggle용 진입점 함수 =================
def agent(observation, configuration):
    """Kaggle이 호출하는 기본 에이전트 함수."""
    return my_agent(observation, configuration)
