"""
AlphaZero-Edu: MTD(f) & Negamax Edition (Full Strategic)
Features:
1. Height-optimized Winning (여유 있는 열로 승리)
2. Critical Blocking (상대 킬각 무조건 방어)
3. Suicide Prevention (상대에게 승리를 주는 수 배제)
"""

import numpy as np
import random
import time

# ============================================================================
# 1. Configuration
# ============================================================================

TIME_LIMIT = 1.6 

# Score Constants (Relative to Current Player)
SCORE_WIN = 100_000_000
SCORE_LOSS = -100_000_000
SCORE_CENTER = 100

# Transposition Table
TT = {}

# ============================================================================
# 2. Board Utilities
# ============================================================================

def to_grid(board_flat, cfg):
    return np.asarray(board_flat).reshape(cfg.rows, cfg.columns)

def get_valid_actions(grid, cfg):
    return [c for c in range(cfg.columns) if grid[0, c] == 0]

def drop_piece(grid, col, player, cfg):
    new_grid = grid.copy()
    for r in range(cfg.rows - 1, -1, -1):
        if new_grid[r, col] == 0:
            new_grid[r, col] = player
            return new_grid
    return new_grid

def check_win(grid, player, cfg):
    rows, cols, n = cfg.rows, cfg.columns, cfg.inarow
    # 가로
    for r in range(rows):
        for c in range(cols - n + 1):
            if all(grid[r, c+i] == player for i in range(n)): return True
    # 세로
    for r in range(rows - n + 1):
        for c in range(cols):
            if all(grid[r+i, c] == player for i in range(n)): return True
    # 대각선 ↘
    for r in range(rows - n + 1):
        for c in range(cols - n + 1):
            if all(grid[r+i, c+i] == player for i in range(n)): return True
    # 대각선 ↗
    for r in range(n - 1, rows):
        for c in range(cols - n + 1):
            if all(grid[r-i, c+i] == player for i in range(n)): return True
    return False

def get_column_height(grid, col, cfg):
    """해당 열에 쌓인 돌의 개수 반환 (0: 비어있음 ~ 6: 꽉참)"""
    rows = cfg.rows
    cnt = 0
    for r in range(rows):
        if grid[r, col] != 0:
            cnt += 1
    return cnt

# ============================================================================
# 3. Evaluation
# ============================================================================

def count_window(window, player, opponent):
    p_cnt = np.count_nonzero(window == player)
    o_cnt = np.count_nonzero(window == opponent)
    e_cnt = np.count_nonzero(window == 0)
    return p_cnt, o_cnt, e_cnt

def evaluate_board(grid, player, cfg):
    opponent = 3 - player
    score = 0
    rows, cols, n = cfg.rows, cfg.columns, cfg.inarow
    
    # [1] Center Preference
    center_col = cols // 2
    for r in range(rows):
        if grid[r, center_col] == player: score += SCORE_CENTER
        elif grid[r, center_col] == opponent: score -= SCORE_CENTER

    # [2] Window Analysis
    # Optimize: Collect windows once
    windows = []
    for r in range(rows):
        for c in range(cols - n + 1): windows.append(grid[r, c:c+n])
    for r in range(rows - n + 1):
        for c in range(cols): windows.append(grid[r:r+n, c])
    for r in range(rows - n + 1):
        for c in range(cols - n + 1): windows.append(np.array([grid[r+i, c+i] for i in range(n)]))
    for r in range(n - 1, rows):
        for c in range(cols - n + 1): windows.append(np.array([grid[r-i, c+i] for i in range(n)]))
            
    for w in windows:
        p_cnt, o_cnt, e_cnt = count_window(w, player, opponent)
        if p_cnt == 4: return SCORE_WIN
        if o_cnt == 4: return SCORE_LOSS
        
        # Scoring logic
        if p_cnt == 3 and e_cnt == 1: score += 10000
        if o_cnt == 3 and e_cnt == 1: score -= 80000 # Defensive penalty
        if p_cnt == 2 and e_cnt == 2: score += 500
        if o_cnt == 2 and e_cnt == 2: score -= 1000

    return score

# ============================================================================
# 4. Search Engine (Negamax + MTD-f)
# ============================================================================

def negamax(grid, depth, alpha, beta, player, cfg, start_time):
    if time.time() - start_time > TIME_LIMIT: raise TimeoutError

    board_bytes = grid.tobytes()
    if board_bytes in TT:
        entry = TT[board_bytes]
        if entry['depth'] >= depth:
            if entry['flag'] == 'EXACT': return entry['value']
            elif entry['flag'] == 'LOWER': alpha = max(alpha, entry['value'])
            elif entry['flag'] == 'UPPER': beta = min(beta, entry['value'])
            if alpha >= beta: return entry['value']

    if check_win(grid, player, cfg): return SCORE_WIN
    if check_win(grid, 3-player, cfg): return SCORE_LOSS
    
    valid_actions = get_valid_actions(grid, cfg)
    if not valid_actions: return 0
    if depth == 0: return evaluate_board(grid, player, cfg)

    # Move Ordering
    center = cfg.columns // 2
    valid_actions.sort(key=lambda x: abs(x - center))

    best_value = -float('inf')
    flag = 'UPPER'
    
    for col in valid_actions:
        new_grid = drop_piece(grid, col, player, cfg)
        try:
            val = -negamax(new_grid, depth - 1, -beta, -alpha, 3-player, cfg, start_time)
        except TimeoutError: raise

        if val > best_value: best_value = val
        alpha = max(alpha, best_value)
        if alpha >= beta:
            flag = 'LOWER'
            break
            
    if flag != 'LOWER':
        flag = 'EXACT' if best_value > alpha else 'UPPER'
        
    TT[board_bytes] = {'depth': depth, 'flag': flag, 'value': best_value}
    return best_value

def mtdf_search(grid, guess, depth, player, cfg, start_time):
    g = guess
    upper = float('inf')
    lower = -float('inf')
    while lower < upper:
        if time.time() - start_time > TIME_LIMIT: raise TimeoutError
        beta = g + 1 if g == lower else g
        g = negamax(grid, depth, beta - 1, beta, player, cfg, start_time)
        if g < beta: upper = g
        else: lower = g
    return g

# ============================================================================
# 5. Main Agent (Strategic Priority Updated)
# ============================================================================

def check_suicide_move(grid, col, player, cfg):
    """
    Check if placing a piece at 'col' immediately gives the opponent a win.
    (i.e., opponent can place right on top of my piece to win)
    """
    # 1. Simulate my move
    my_grid = drop_piece(grid, col, player, cfg)
    
    # 2. Check if the opponent can place in the SAME column immediately
    # (Since gravity exists, the spot directly above my new piece is now playable)
    # However, we must ensure the column isn't full after my move.
    if my_grid[0, col] == 0: # Still space in this column
        # Simulate opponent move on top
        opp_grid = drop_piece(my_grid, col, 3-player, cfg)
        if check_win(opp_grid, 3-player, cfg):
            return True # This is a suicide move
    return False

def my_agent(obs, config):
    global TT
    TT = {}
    
    start_time = time.time()
    grid = to_grid(obs.board, config)
    me = obs.mark
    opponent = 3 - me
    valid_actions = get_valid_actions(grid, config)
    
    # ---------------------------------------------------------------------
    # [Step 0] Immediate Win (Attack) - Height Optimized
    # ---------------------------------------------------------------------
    winning_moves = []
    for col in valid_actions:
        if check_win(drop_piece(grid, col, me, config), me, config):
            winning_moves.append(col)
            
    if winning_moves:
        # 이길 수 있다면, '가장 비어있는 열'을 선택하여 승리
        center = config.columns // 2
        winning_moves.sort(key=lambda col: (
            get_column_height(grid, col, config), # 1순위: 높이 낮은 순
            abs(col - center)                     # 2순위: 중앙 우선
        ))
        return int(winning_moves[0])

    # ---------------------------------------------------------------------
    # [Step 1] Immediate Defense (Block) - Absolute Priority
    # ---------------------------------------------------------------------
    # 내가 이기지 못한다면, 상대가 이길 수 있는 곳은 무조건 막아야 함
    for col in valid_actions:
        if check_win(drop_piece(grid, col, opponent, config), opponent, config):
            return int(col)

    # ---------------------------------------------------------------------
    # [Step 2] Filter "Suicide" Moves (Safety)
    # ---------------------------------------------------------------------
    # 상대에게 킬각을 내주는 수(Bad Move)를 후보에서 가급적 제외
    safe_actions = []
    for col in valid_actions:
        if not check_suicide_move(grid, col, me, config):
            safe_actions.append(col)
    
    # 만약 모든 수가 자살수라면 어쩔 수 없이 valid_actions 사용
    search_candidates = safe_actions if safe_actions else valid_actions

    # ---------------------------------------------------------------------
    # [Step 3] MTD(f) Search
    # ---------------------------------------------------------------------
    best_action = search_candidates[len(search_candidates)//2]
    guess = 0
    action_scores = {a: 0 for a in search_candidates}
    
    try:
        for depth in range(1, 20):
            current_best = None
            max_val = -float('inf')
            
            # Move Ordering
            sorted_candidates = sorted(search_candidates, key=lambda x: action_scores[x], reverse=True)
            
            for col in sorted_candidates:
                new_grid = drop_piece(grid, col, me, config)
                val = -mtdf_search(new_grid, guess, depth - 1, opponent, config, start_time)
                action_scores[col] = val
                
                if val > max_val:
                    max_val = val
                    current_best = col
            
            best_action = current_best
            guess = max_val
            
            if time.time() - start_time > TIME_LIMIT * 0.6:
                break
                
    except TimeoutError: pass
    except Exception: return int(best_action)

    return int(best_action)

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
