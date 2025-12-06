import numpy as np
import random

# 몇 수 앞까지 볼지 (Minimax 깊이)
MAX_DEPTH = 4  # 3~5 정도로 실험해보면서 조정 추천

# # 패턴별 가중치 (필요하면 aggressively 튜닝 가능)
PATTERN_SCORE = {
    "my_4": 1_000_000,
    "my_3": 10_000,
    "my_2": 100,
    "opp_3": -50_000,
    "opp_2": -200,
}


# 1) 보드 관련 유틸 함수들 ------------------------------------------------------
def to_grid(board, cfg):
    """1차원 보드를 2D 보드로 변환."""
    return np.array(board).reshape(cfg.rows, cfg.columns)


def drop_token(grid, column, player, cfg):
    """grid 복사본에 player의 돌을 column에 투입한 상태 반환."""
    copied = grid.copy()
    for r in range(cfg.rows - 1, -1, -1):
        if copied[r, column] == 0:
            copied[r, column] = player
            return copied
    return copied


def find_playable_columns(board_flat, cfg):
    """현재 둘 수 있는 column 목록 반환."""
    return [c for c in range(cfg.columns) if board_flat[c] == 0]


def _get_board_and_mark(obs):
    """obs가 dict든 객체든 안전하게 board, mark 꺼내기."""
    if isinstance(obs, dict):
        return obs["board"], obs["mark"]
    return obs.board, obs.mark


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
    for r in range(N - 1, R):
        for c in range(C - N + 1):
            yield np.array([grid[r-i, c+i] for i in range(N)])


def evaluate_board(grid, me, cfg):
    """
    패턴 기반 보드 평가 함수.
    - 내 2/3/4 연결
    - 상대 2/3 연결
    에 대해 점수를 부여한다.
    """
    opp = 1 if me == 2 else 2
    score = 0

    for window in iter_all_windows(grid, cfg):
        my_count = np.count_nonzero(window == me)
        opp_count = np.count_nonzero(window == opp)
        empty_count = np.count_nonzero(window == 0)

        # 내가 이긴 패턴
        if my_count == 4:
            score += PATTERN_SCORE["my_4"]
        # 공격 기회 (3 + 1 빈칸)
        elif my_count == 3 and empty_count == 1:
            score += PATTERN_SCORE["my_3"]
        # 가벼운 우위
        elif my_count == 2 and empty_count == 2:
            score += PATTERN_SCORE["my_2"]

        # 상대가 곧 이길 패턴
        if opp_count == 3 and empty_count == 1:
            score += PATTERN_SCORE["opp_3"]
        # 상대가 유리한 2-연결
        elif opp_count == 2 and empty_count == 2:
            score += PATTERN_SCORE["opp_2"]

    return int(score)

#  3) Minimax + Alpha-Beta Pruning --------------------------------------------------------
def minimax_alpha_beta(grid, cur_player, me, depth, alpha, beta, cfg):
    """
    Alpha-Beta Pruning이 적용된 Minimax 탐색.
    - cur_player: 현재 수를 둘 플레이어 (1 또는 2)
    - me: '나'의 마크
    - depth: 남은 깊이
    - alpha: 현재까지의 최선 MAX 값
    - beta: 현재까지의 최선 MIN 값
    """
    flat = grid.flatten()
    playable = find_playable_columns(flat, cfg)

    # 깊이 도달 or 둘 곳 없음 → 현재 보드를 평가하고 종료
    if depth == 0 or not playable:
        return evaluate_board(grid, me, cfg)

    next_player = 1 if cur_player == 2 else 2

    # 내 턴: 최대화
    if cur_player == me:
        value = -float("inf")
        for col in playable:
            child_grid = drop_token(grid, col, cur_player, cfg)
            child_value = minimax_alpha_beta(
                child_grid,
                next_player,
                me,
                depth - 1,
                alpha,
                beta,
                cfg,
            )
            value = max(value, child_value)
            alpha = max(alpha, value)

            # 가지치기 (더 볼 필요 없음)
            if alpha >= beta:
                break
        return value

    # 상대 턴: 최소화
    else:
        value = float("inf")
        for col in playable:
            child_grid = drop_token(grid, col, cur_player, cfg)
            child_value = minimax_alpha_beta(
                child_grid,
                next_player,
                me,
                depth - 1,
                alpha,
                beta,
                cfg,
            )
            value = min(value, child_value)
            beta = min(beta, value)

            # 가지치기
            if alpha >= beta:
                break
        return value


# 4) 최종 에이전트 N-step Alpha-Beta Agent -------------------------------------------------------------

def my_agent(obs, cfg):
    """
    N-step(깊이 MAX_DEPTH) Minimax + Alpha-Beta Pruning 기반 에이전트.
    - 가능한 모든 수에 대해 탐색을 수행하고
    - 최종 평가 점수가 가장 높은 column 중 하나를 선택한다.
    """
    board, me = _get_board_and_mark(obs)
    grid = to_grid(board, cfg)
    playable = find_playable_columns(board, cfg)

    # 둘 곳이 없으면 그냥 0 리턴 (거의 게임 끝 상황)
    if not playable:
        return 0

    opp = 1 if me == 2 else 2
    scores = {}

    for col in playable:
        # 1수 앞: 내가 col에 둔 후, 상대 차례부터 탐색 시작
        next_grid = drop_token(grid, col, me, cfg)
        value = minimax_alpha_beta(
            grid=next_grid,
            cur_player=opp,
            me=me,
            depth=MAX_DEPTH - 1,  # 이미 한 수 뒀으니 -1
            alpha=-float("inf"),
            beta=float("inf"),
            cfg=cfg,
        )
        scores[col] = value

    # 최고 점수 선택
    best_score = max(scores.values())
    best_cols = [c for c, v in scores.items() if v == best_score]

    if not best_cols:
        return playable[0]

    return random.choice(best_cols)


if __name__ == "__main__":
    # VSCode 로컬 테스트 용
    from kaggle_environments import make

    env = make("connectx", debug=True)

    print("=== 테스트 1: 랜덤 에이전트 vs my_agent ===")
    env.reset()
    env.run(["random", my_agent])
    print(env.render(mode="ansi"))
    print()

    print("=== 테스트 2: my_agent vs 랜덤 에이전트 ===")
    env.reset()
    env.run([my_agent, "random"])
    print(env.render(mode="ansi"))
    print()
