import numpy as np
import random


# 몇 수 앞까지 볼지 (minimax depth)
LOOKAHEAD_DEPTH = 3

# 패턴별 가중치
PATTERN_SCORE = {
    "my_4": 1_000_000,
    "my_3": 10_000,
    "my_2": 100,
    "opp_3": -50_000,
    "opp_2": -200,
}


# 1) 보드 관련 유틸 함수들 ------------------------------------------------------
def to_grid(board, cfg):
    """
    1차원 보드를 2D 보드로 변환.
    """
    return np.array(board).reshape(cfg.rows, cfg.columns)

def find_playable_columns(board, cfg):
    """
    현재 둘 수 있는 모든 column 반환.
    """
    return [c for c in range(cfg.columns) if board[c] == 0]


def drop_token(grid, column, player_mark, cfg):
    """
    grid 복사본에 player 칩을 column에 투입한 상태 반환.
    """
    new_grid = grid.copy()
    # 아래 행부터 올라가면서 비어 있는 자리 찾기
    for r in range(cfg.rows - 1, -1, -1):
        if new_grid[r, column] == 0:
            new_grid[r, column] = player_mark
            return new_grid
    return new_grid 

# 2) 윈도우 생성 & 패턴 카운팅 ---------------------------------------------------
def iter_all_windows(grid, cfg):
    """가로/세로/양 대각선으로 길이 inarow 윈도우 생성."""
    R, C, N = cfg.rows, cfg.columns, cfg.inarow

    # 가로
    for r in range(R):
        for c in range(C - N + 1):
            yield grid[r, c:c+N]

    # 세로
    for r in range(R - N + 1):
        for c in range(C):
            yield grid[r:r+N, c]

    # 대각선 ↘
    for r in range(R - N + 1):
        for c in range(C - N + 1):
            yield np.array([grid[r+i, c+i] for i in range(N)])

    # 대각선 ↗
    for r in range(N-1, R):
        for c in range(C - N + 1):
            yield np.array([grid[r-i, c+i] for i in range(N)])


def evaluate(grid, me, cfg):
    """
    패턴 기반으로 보드의 점수 계산.
    """
    opp = 1 if me == 2 else 2

    score = 0
    for window in iter_all_windows(grid, cfg):
        if np.count_nonzero(window == me) == 4:
            score += PATTERN_SCORE["my_4"]

        elif np.count_nonzero(window == me) == 3 and np.count_nonzero(window == 0) == 1:
            score += PATTERN_SCORE["my_3"]

        elif np.count_nonzero(window == me) == 2 and np.count_nonzero(window == 0) == 2:
            score += PATTERN_SCORE["my_2"]

        if np.count_nonzero(window == opp) == 3 and np.count_nonzero(window == 0) == 1:
            score += PATTERN_SCORE["opp_3"]

        elif np.count_nonzero(window == opp) == 2 and np.count_nonzero(window == 0) == 2:
            score += PATTERN_SCORE["opp_2"]

    return score


# 3) N-step 미니맥스 탐색 --------------------------------------------------------
def minimax(grid, cur_player, me, depth, cfg):
    """
    N-step minimax 탐색 (alpha-beta pruning 없는 단순 버전).
    """
    playable = find_playable_columns(grid.flatten(), cfg)

    # 리프 조건
    if depth == 0 or not playable:
        return evaluate(grid, me, cfg)

    next_player = 1 if cur_player == 2 else 2

    # 내 차례: 최대화
    if cur_player == me:
        best_score = -float("inf")
        for c in playable:
            next_grid = drop_token(grid, c, cur_player, cfg)
            value = minimax(next_grid, next_player, me, depth - 1, cfg)
            best_score = max(best_score, value)
        return best_score

    # 상대 차례: 최소화
    else:
        worst_score = float("inf")
        for c in playable:
            next_grid = drop_token(grid, c, cur_player, cfg)
            value = minimax(next_grid, next_player, me, depth - 1, cfg)
            worst_score = min(worst_score, value)
        return worst_score


# 4) 최종 N-step Agent -------------------------------------------------------------
def my_agent(obs, cfg):
    """미니맥스 기반 N-step lookahead agent."""
    board, me = obs.board, obs.mark
    grid = to_grid(board, cfg)
    playable = find_playable_columns(board, cfg)

    if not playable:  # 둘 곳 없으면 그냥 0 리턴
        return 0

    opp = 1 if me == 2 else 2

    scores = {}
    for c in playable:
        next_grid = drop_token(grid, c, me, cfg)
        scores[c] = minimax(
            grid=next_grid,
            cur_player=opp,
            me=me,
            depth=LOOKAHEAD_DEPTH,
            cfg=cfg
        )

    # 최고 점수 선택
    best_score = max(scores.values())
    best_cols = [c for c, sc in scores.items() if sc == best_score]
    return random.choice(best_cols)


if __name__ == "__main__":
    # 간단한 테스트 코드
    from kaggle_environments import make

    env = make("connectx", debug=True)

    print("=== 테스트 1: 랜덤 에이전트 vs my_agent ===")
    game1 = env.run(["random", my_agent])
    print(env.render(mode="ansi")) 
    print()

    print("=== 테스트 2: my_agent vs 랜덤 에이전트 ===")
    game2 = env.run([my_agent, "random"])
    print(env.render(mode="ansi"))
    print()

# ================= Kaggle 제출용 엔트리 함수 =================
def agent(observation, configuration):
    """Kaggle에서 요구하는 기본 에이전트 함수."""
    return my_agent(observation, configuration)
