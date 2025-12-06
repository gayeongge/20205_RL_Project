"""
AlphaZero-Edu: MTD(f) & Negamax Edition
Fixes:
- Sign error in minimax logic (switched to pure Negamax)
- Strict Time Management (Safe-fail)
- Optimized Evaluation
"""

import numpy as np
import random
import time

# ============================================================================
# 1. Configuration
# ============================================================================

# [Time Limit]
# Kaggle 기본 2.0초지만, 오버헤드 고려하여 1.5~1.7초 내에 끊는 것이 안전
TIME_LIMIT = 1.6 

# [Scoring - Relative to Current Player]
# Negamax에서는 '나'에게 좋으면 +, 나쁘면 -
SCORE_WIN = 100_000_000      # 승리
SCORE_LOSS = -100_000_000    # 패배 (상대 승리)

# 전술 점수
SCORE_BLOCK_WIN = 50_000_000 # 상대 킬각 방어 (거의 승리급으로 중요)
SCORE_GAP_THREAT = 500_000   # 징검다리 공격/방어
SCORE_CENTER = 100           # 중앙 점유

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
    """빠른 승리 체크"""
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

# ============================================================================
# 3. Evaluation (Relative to Current Player)
# ============================================================================

def count_window(window, player, opponent):
    p_cnt = np.count_nonzero(window == player)
    o_cnt = np.count_nonzero(window == opponent)
    e_cnt = np.count_nonzero(window == 0)
    return p_cnt, o_cnt, e_cnt

def evaluate_board(grid, player, cfg):
    """
    현재 턴인 'player' 입장에서의 점수.
    내가 유리하면 양수, 불리하면 음수.
    """
    opponent = 3 - player
    score = 0
    rows, cols, n = cfg.rows, cfg.columns, cfg.inarow
    
    # [1] 중앙 점유 (초반 전략)
    center_col = cols // 2
    for r in range(rows):
        if grid[r, center_col] == player:
            score += SCORE_CENTER
        elif grid[r, center_col] == opponent:
            score -= SCORE_CENTER

    # [2] 윈도우 탐색
    # 모든 4칸짜리 윈도우를 검사
    windows = []
    
    # (최적화를 위해 제너레이터 대신 리스트 수집 후 일괄 처리 가능하나, 여기선 직관적으로)
    # 가로, 세로, 대각선 수집
    for r in range(rows):
        for c in range(cols - n + 1):
            windows.append(grid[r, c:c+n])
    for r in range(rows - n + 1):
        for c in range(cols):
            windows.append(grid[r:r+n, c])
    for r in range(rows - n + 1):
        for c in range(cols - n + 1):
            windows.append(np.array([grid[r+i, c+i] for i in range(n)]))
    for r in range(n - 1, rows):
        for c in range(cols - n + 1):
            windows.append(np.array([grid[r-i, c+i] for i in range(n)]))
            
    for w in windows:
        p_cnt, o_cnt, e_cnt = count_window(w, player, opponent)
        
        # 1. 승리/패배 확정
        if p_cnt == 4: return SCORE_WIN
        if o_cnt == 4: return SCORE_LOSS
        
        # 2. 치명적 위협 (3개 + 빈칸 1개)
        if p_cnt == 3 and e_cnt == 1:
            score += 10000 # 공격 기회
        if o_cnt == 3 and e_cnt == 1:
            score -= 80000 # 방어 실패 위기 (패널티 매우 큼)

        # 3. 징검다리/잠재력 (2개 + 빈칸 2개)
        if p_cnt == 2 and e_cnt == 2:
            score += 500
        if o_cnt == 2 and e_cnt == 2:
            score -= 1000 # 상대 견제 우선

    return score

# ============================================================================
# 4. Negamax & MTD(f) Engine
# ============================================================================

def negamax(grid, depth, alpha, beta, player, cfg, start_time):
    # 시간 체크 (노드 방문마다 체크하면 느리므로, depth가 일정 이상일 때만 하거나 그냥 둠)
    if time.time() - start_time > TIME_LIMIT:
        raise TimeoutError

    # TT 조회
    board_bytes = grid.tobytes() # 간단한 키 생성
    if board_bytes in TT:
        entry = TT[board_bytes]
        if entry['depth'] >= depth:
            if entry['flag'] == 'EXACT': return entry['value']
            elif entry['flag'] == 'LOWER': alpha = max(alpha, entry['value'])
            elif entry['flag'] == 'UPPER': beta = min(beta, entry['value'])
            if alpha >= beta: return entry['value']

    # 기저 조건: 게임 종료 or 깊이 도달
    if check_win(grid, player, cfg): return SCORE_WIN
    if check_win(grid, 3-player, cfg): return SCORE_LOSS
    
    valid_actions = get_valid_actions(grid, cfg)
    if not valid_actions: return 0 # 무승부
    if depth == 0:
        return evaluate_board(grid, player, cfg)

    # Move Ordering (중앙 -> 외곽)
    center = cfg.columns // 2
    valid_actions.sort(key=lambda x: abs(x - center))

    # Recursive Search
    best_value = -float('inf')
    flag = 'UPPER'
    
    for col in valid_actions:
        new_grid = drop_piece(grid, col, player, cfg)
        # Negamax의 핵심: -negamax(...) 
        # 상대방 턴에서의 점수를 뒤집어서 내 점수로 가져옴
        try:
            val = -negamax(new_grid, depth - 1, -beta, -alpha, 3-player, cfg, start_time)
        except TimeoutError:
            raise

        if val > best_value:
            best_value = val
        
        alpha = max(alpha, best_value)
        if alpha >= beta:
            flag = 'LOWER' # Beta Cutoff
            break
            
    # TT 저장
    if flag != 'LOWER': # Beta 컷이 안 났으면 Exact일 수도 있음 (간소화)
        if best_value <= alpha: flag = 'UPPER' # Fail Low
        else: flag = 'EXACT'
        
    TT[board_bytes] = {'depth': depth, 'flag': flag, 'value': best_value}
    return best_value

def mtdf_search(grid, guess, depth, player, cfg, start_time):
    g = guess
    upper = float('inf')
    lower = -float('inf')
    
    while lower < upper:
        if time.time() - start_time > TIME_LIMIT:
            raise TimeoutError
            
        beta = g + 1 if g == lower else g
        
        # Null Window Search
        g = negamax(grid, depth, beta - 1, beta, player, cfg, start_time)
        
        if g < beta: upper = g
        else: lower = g
        
    return g

# ============================================================================
# 5. Main Agent (Iterative Deepening)
# ============================================================================

def my_agent(obs, config):
    global TT
    TT = {} # 매 턴 TT 초기화 (메모리 관리)
    
    start_time = time.time()
    grid = to_grid(obs.board, config)
    me = obs.mark
    valid_actions = get_valid_actions(grid, config)
    
    # [0] 1수 킬각/방어 (필수)
    for col in valid_actions:
        if check_win(drop_piece(grid, col, me, config), me, config): return col
    for col in valid_actions:
        if check_win(drop_piece(grid, col, 3-me, config), 3-me, config): return col

    # [1] Iterative Deepening
    best_action = valid_actions[len(valid_actions)//2] # 기본값: 중앙
    guess = 0
    
    # Move Ordering을 위한 임시 점수판
    action_scores = {a: 0 for a in valid_actions}
    
    try:
        # 깊이 1부터 6~7 정도까지 (파이썬은 느려서 4~5가 한계일 수 있음)
        for depth in range(1, 10): 
            current_best_action = None
            max_score = -float('inf')
            
            # 이전 깊이에서 좋았던 순서대로 정렬하여 탐색 (Move Ordering 효과)
            sorted_actions = sorted(valid_actions, key=lambda x: action_scores[x], reverse=True)
            
            for col in sorted_actions:
                new_grid = drop_piece(grid, col, me, config)
                
                # MTD(f) 호출
                # 상대방 턴이므로 결과를 뒤집어야 내 점수 (-val)
                val = -mtdf_search(new_grid, guess, depth - 1, 3-me, config, start_time)
                
                action_scores[col] = val # 다음 정렬을 위해 점수 저장
                
                if val > max_score:
                    max_score = val
                    current_best_action = col
            
            # 이번 깊이가 무사히 끝났으면 Best Action 갱신
            best_action = current_best_action
            guess = max_score # 다음 깊이의 예상 점수
            
            # 시간이 절반 이상 지났으면 다음 깊이는 위험하므로 중단
            if time.time() - start_time > TIME_LIMIT * 0.6:
                break
                
    except TimeoutError:
        # 시간 초과 시, '지금까지 완료한 깊이'에서의 best_action 반환
        pass
    except Exception as e:
        # 에러 발생 시 랜덤 방지용 중앙 우선
        return int(best_action)

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