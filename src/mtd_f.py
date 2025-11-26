"""
AlphaZero-Edu: MTD(f) Edition for ConnectX
Features:
- MTD(f) Search Algorithm (Memory-enhanced Test Driver)
- Transposition Table with Zobrist Hashing
- Window-based Gap Detection (징검다리 위협 감지)
- Time-managed Iterative Deepening
"""

import numpy as np
import random
import time

# ============================================================================
# 1. Configuration & Scoring Constants
# ============================================================================

# 시간 관리 (Kaggle 제한시간 5초 기준 안전하게 설정)
TIME_LIMIT = 2.0  # 초 단위

# 점수 체계 (User Defined Strategy)
SCORE_WIN = 1_000_000_000        # 승리
SCORE_LOSS = -1_000_000_000      # 패배
SCORE_OPP_3 = -20_000_000        # [방어 1순위] 상대 3목 (무조건 막아야 함)
SCORE_MY_3 = 100_000             # [공격] 내 3목
SCORE_OPP_2 = -40_000            # [방어 2순위] 상대 2목 견제
SCORE_MY_2 = 1_000               # [기반] 내 2목
SCORE_CENTER = 10                # 중앙 점유 가산점

# Zobrist Hashing Table (Runtime Initialization)
ZOBRIST_TABLE = {}
TT = {}  # Transposition Table

# ============================================================================
# 2. Board & Zobrist Utils
# ============================================================================

def init_zobrist(cfg):
    """Zobrist Hashing을 위한 난수 테이블 생성"""
    global ZOBRIST_TABLE
    rows, cols = cfg.rows, cfg.columns
    # 0: Empty, 1: P1, 2: P2
    for r in range(rows):
        for c in range(cols):
            for p in range(1, 3):
                ZOBRIST_TABLE[(r, c, p)] = random.getrandbits(64)

def compute_hash(grid, cfg):
    """현재 보드의 해시값 계산"""
    h = 0
    rows, cols = cfg.rows, cfg.columns
    for r in range(rows):
        for c in range(cols):
            p = grid[r, c]
            if p != 0:
                h ^= ZOBRIST_TABLE[(r, c, p)]
    return h

def to_grid(board_flat, cfg):
    return np.asarray(board_flat).reshape(cfg.rows, cfg.columns)

def get_valid_actions(board_flat, cfg):
    return [c for c in range(cfg.columns) if board_flat[c] == 0]

def drop_piece(grid, col, player, cfg):
    """돌을 두고 새로운 그리드와 업데이트된 해시 반환 (속도 최적화 가능하지만 가독성 위주)"""
    new_grid = grid.copy()
    for r in range(cfg.rows - 1, -1, -1):
        if new_grid[r, col] == 0:
            new_grid[r, col] = player
            return new_grid
    return new_grid

def check_win(grid, player, cfg):
    """승리 확인 (빠른 종료용)"""
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
# 3. Static Evaluation (Window-based Gap Detection)
# ============================================================================

def evaluate_window(window, player, cfg):
    """
    단일 윈도우(4칸) 평가.
    징검다리(Gap) 패턴도 여기서 감지됨 (순서 상관없이 개수만 세므로).
    """
    opponent = 3 - player
    
    my_count = np.count_nonzero(window == player)
    opp_count = np.count_nonzero(window == opponent)
    empty_count = np.count_nonzero(window == 0)
    
    score = 0
    
    if my_count == 4:
        return SCORE_WIN
    if opp_count == 4:
        return SCORE_LOSS
        
    # [방어] 상대 3개 (O O _ O 포함)
    if opp_count == 3 and empty_count == 1:
        score += SCORE_OPP_3
        
    # [공격] 내 3개
    elif my_count == 3 and empty_count == 1:
        score += SCORE_MY_3
        
    # [방어] 상대 2개 (미리 끊기)
    elif opp_count == 2 and empty_count == 2:
        score += SCORE_OPP_2
        
    # [기반] 내 2개
    elif my_count == 2 and empty_count == 2:
        score += SCORE_MY_2
        
    return score

def evaluate_board(grid, player, cfg):
    """전체 보드 상태 평가"""
    score = 0
    rows, cols, n = cfg.rows, cfg.columns, cfg.inarow
    
    # 1. 중앙 점유 가산점
    center_col = cols // 2
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == player:
                score += SCORE_CENTER * (1 if c == center_col else 0)
            elif grid[r, c] == 3 - player:
                score -= SCORE_CENTER * (1 if c == center_col else 0)

    # 2. 윈도우 슬라이딩 평가
    # 가로
    for r in range(rows):
        for c in range(cols - n + 1):
            score += evaluate_window(grid[r, c:c+n], player, cfg)
    # 세로
    for r in range(rows - n + 1):
        for c in range(cols):
            score += evaluate_window(grid[r:r+n, c], player, cfg)
    # 대각선 ↘
    for r in range(rows - n + 1):
        for c in range(cols - n + 1):
            window = np.array([grid[r+i, c+i] for i in range(n)])
            score += evaluate_window(window, player, cfg)
    # 대각선 ↗
    for r in range(n - 1, rows):
        for c in range(cols - n + 1):
            window = np.array([grid[r-i, c+i] for i in range(n)])
            score += evaluate_window(window, player, cfg)
            
    return score

# ============================================================================
# 4. MTD(f) Engine
# ============================================================================

def alpha_beta_with_memory(grid, depth, alpha, beta, player, root_player, cfg, start_time):
    """Transposition Table을 사용하는 Alpha-Beta Pruning"""
    
    # 시간 초과 체크 (Time Soft-limit)
    if time.time() - start_time > TIME_LIMIT:
        raise TimeoutError
        
    # 해싱 및 TT 조회
    board_hash = compute_hash(grid, cfg)
    if board_hash in TT:
        entry = TT[board_hash]
        if entry['depth'] >= depth:
            if entry['flag'] == 'EXACT':
                return entry['value']
            elif entry['flag'] == 'LOWERBOUND':
                alpha = max(alpha, entry['value'])
            elif entry['flag'] == 'UPPERBOUND':
                beta = min(beta, entry['value'])
            
            if alpha >= beta:
                return entry['value']

    # 기저 조건 (Leaf Node)
    if depth == 0 or check_win(grid, 1, cfg) or check_win(grid, 2, cfg):
        # 정적 평가: 내(root_player) 기준으로 점수 계산
        # 현재 턴이 누구든 평가는 항상 '나(root_player)'의 관점에서
        eval_score = evaluate_board(grid, root_player, cfg)
        return eval_score

    # 탐색 (Recursion)
    best_value = -float('inf') if player == root_player else float('inf')
    valid_actions = get_valid_actions(grid.flatten(), cfg)
    
    # Move Ordering (중앙부터 탐색하면 가지치기 확률 높음)
    center = cfg.columns // 2
    valid_actions.sort(key=lambda x: abs(x - center))

    if player == root_player: # Maximize
        for col in valid_actions:
            new_grid = drop_piece(grid, col, player, cfg)
            val = alpha_beta_with_memory(new_grid, depth - 1, alpha, beta, 3-player, root_player, cfg, start_time)
            best_value = max(best_value, val)
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break
    else: # Minimize
        for col in valid_actions:
            new_grid = drop_piece(grid, col, player, cfg)
            val = alpha_beta_with_memory(new_grid, depth - 1, alpha, beta, 3-player, root_player, cfg, start_time)
            best_value = min(best_value, val)
            beta = min(beta, best_value)
            if alpha >= beta:
                break
                
    # TT 저장
    flag = 'EXACT'
    if best_value <= alpha: flag = 'UPPERBOUND'
    elif best_value >= beta: flag = 'LOWERBOUND'
    
    TT[board_hash] = {'depth': depth, 'flag': flag, 'value': best_value}
    
    return best_value

def mtdf(grid, guess, depth, player, cfg, start_time):
    """
    MTD(f) Driver
    Null-Window Search를 반복하여 실제 Minimax 값에 수렴
    """
    upper = float('inf')
    lower = -float('inf')
    g = guess
    
    while lower < upper:
        if g == lower: beta = g + 1
        else: beta = g
        
        # Null Window Search: [beta-1, beta]
        g = alpha_beta_with_memory(grid, depth, beta - 1, beta, player, player, cfg, start_time)
        
        if g < beta: upper = g
        else: lower = g
        
        # 시간 초과 시 루프 탈출
        if time.time() - start_time > TIME_LIMIT:
            break
            
    return g

# ============================================================================
# 5. Main Agent
# ============================================================================

def my_agent(obs, config):
    global ZOBRIST_TABLE, TT
    
    # 초기화
    if not ZOBRIST_TABLE:
        init_zobrist(config)
    
    grid = to_grid(obs.board, config)
    me = obs.mark
    valid_actions = get_valid_actions(obs.board, config)
    
    # 0. One-move Win/Loss Check (가장 빠른 방어)
    # 탐색 전에 확실한 수는 바로 둔다.
    for action in valid_actions:
        if check_win(drop_piece(grid, action, me, config), me, config):
            return action
    for action in valid_actions:
        if check_win(drop_piece(grid, action, 3-me, config), 3-me, config):
            return action

    # 1. Iterative Deepening with MTD(f)
    best_action = valid_actions[0]
    start_time = time.time()
    
    # 중앙 선호 정렬
    center = config.columns // 2
    valid_actions.sort(key=lambda x: abs(x - center))
    
    try:
        # 깊이를 1부터 점진적으로 늘려감
        for depth in range(1, 20): # 최대 깊이 20 (시간 되면 멈춤)
            current_best_move = None
            max_val = -float('inf')
            
            # Root Node에서 각 자식 노드에 대해 MTD(f) 수행
            for action in valid_actions:
                new_grid = drop_piece(grid, action, me, config)
                
                # 이전 단계의 값을 guess로 사용하면 좋지만, 여기선 0으로 시작
                val = mtdf(new_grid, 0, depth - 1, 3-me, config, start_time) # 다음 턴은 상대방
                
                # 상대방 턴에서 반환된 값은 '나의 관점' 점수여야 함 (evaluate_board가 그렇게 설계됨)
                
                if val > max_val:
                    max_val = val
                    current_best_move = action
            
            # 이번 깊이가 시간 내에 완료되었다면 결과 갱신
            best_action = current_best_move
            
            # 시간이 얼마 안 남았으면 다음 깊이는 포기
            if time.time() - start_time > TIME_LIMIT * 0.8:
                break
                
    except TimeoutError:
        pass # 시간 초과 시 지금까지 찾은 best_action 반환
    except Exception as e:
        # 만일의 에러 대비
        return int(random.choice(valid_actions))

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