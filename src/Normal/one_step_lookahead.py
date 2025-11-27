import random
import numpy as np


# 패턴별 가중치
PATTERN_WEIGHTS = {
    "my_4": 1_000_000.0,   # 내가 4개 연결
    "my_3": 1_000.0,       # 내가 3개 연결
    "my_2": 10.0,          # 내가 2개 연결
    "opp_3": -50_000.0,    # 상대 3개 연결
    "opp_2": -100.0,       # 상대 2개 연결
}


# 1) 보드 관련 유틸 함수들 ------------------------------------------------------
def to_grid(board_flat, cfg):
    """1차원 보드를 (rows, columns) 2차원 배열로 변환."""
    return np.asarray(board_flat).reshape(cfg.rows, cfg.columns)


def find_playable_columns(board_flat, cfg):
    """
    현재 둘 수 있는 column 리스트 반환.
    - ConnectX에서는 맨 위(row=0) 셀이 비어 있으면 해당 열에 둘 수 있음.
    """
    playable = []
    for c in range(cfg.columns):
        if board_flat[c] == 0:
            playable.append(c)
    return playable


def drop_token(grid, column, player_mark, cfg):
    """
    grid에 column 위치로 player_mark 돌을 떨어뜨린 다음,
    새로운 grid를 반환 (원본은 건드리지 않음).
    """
    new_grid = grid.copy()
    # 아래 행부터 올라가면서 비어 있는 자리 찾기
    for r in range(cfg.rows - 1, -1, -1):
        if new_grid[r, column] == 0:
            new_grid[r, column] = player_mark
            break
    return new_grid


# 2) 윈도우 생성 & 패턴 카운팅 ---------------------------------------------------
def iter_all_windows(grid, cfg):
    """
    가로/세로/대각선 방향으로 길이 inarow짜리 '창(window)'들을 모두 생성.
    각 window는 리스트 형태 [cell1, cell2, ...] 로 yield.
    """
    rows, cols, n = cfg.rows, cfg.columns, cfg.inarow

    # 가로 방향
    for r in range(rows):
        for c in range(cols - n + 1):
            yield list(grid[r, c:c + n])

    # 세로 방향
    for r in range(rows - n + 1):
        for c in range(cols):
            yield list(grid[r:r + n, c])

    # ↘ 대각선
    for r in range(rows - n + 1):
        for c in range(cols - n + 1):
            yield [grid[r + i, c + i] for i in range(n)]

    # ↗ 대각선
    for r in range(n - 1, rows):
        for c in range(cols - n + 1):
            yield [grid[r - i, c + i] for i in range(n)]


def count_pattern(grid, num_stones, player_mark, cfg):
    """
    특정 플레이어(player_mark)의 돌이 num_stones개 들어 있고,
    나머지는 빈칸(0)인 window 개수를 센다.
    """
    count = 0
    needed_zeros = cfg.inarow - num_stones

    for window in iter_all_windows(grid, cfg):
        if window.count(player_mark) == num_stones and window.count(0) == needed_zeros:
            count += 1

    return count


# 3) heuristic 계산 로직 --------------------------------------------------------

def evaluate_board(grid, me, cfg):
    """
    현재 grid에 대해 '내 입장(me)'에서 점수를 계산.
    - 나 2/3/4 연결
    - 상대 2/3 연결
    를 패턴별 가중치로 합산.
    """
    opponent = 1 if me == 2 else 2

    my_two  = count_pattern(grid, 2, me, cfg)
    my_three = count_pattern(grid, 3, me, cfg)
    my_four = count_pattern(grid, 4, me, cfg)

    opp_two  = count_pattern(grid, 2, opponent, cfg)
    opp_three = count_pattern(grid, 3, opponent, cfg)

    w = PATTERN_WEIGHTS
    score = (
        w["my_4"]  * my_four +
        w["my_3"]  * my_three +
        w["my_2"]  * my_two +
        w["opp_2"] * opp_two +
        w["opp_3"] * opp_three
    )
    return float(score)


def score_single_move(grid, col, me, cfg):
    """
    col에 수를 뒀다고 가정했을 때, 한 수 이후 보드의 heuristic 점수 계산.
    """
    next_grid = drop_token(grid, col, me, cfg)
    return evaluate_board(next_grid, me, cfg)


# 4) 최종 에이전트 -------------------------------------------------------------

def my_agent(obs, config):
    """
    한 턴에 한 수만 미리 보는(one-step lookahead) 단순 에이전트.
    - 둘 수 있는 모든 열에 대해 점수를 계산하고
    - 가장 점수가 높은 열들 중 하나를 랜덤으로 선택.
    """
    # 1차원 보드 → 2D grid
    grid = to_grid(obs.board, config)
    my_mark = obs.mark

    # 현재 가능한 열들
    candidate_cols = find_playable_columns(obs.board, config)

    # 각 열에 대해 점수 계산
    scores = {}
    for col in candidate_cols:
        scores[col] = score_single_move(grid, col, my_mark, config)

    # 최고 점수 찾기
    best_score = max(scores.values())
    best_cols = [c for c, s in scores.items() if s == best_score]

    # 최고 점수 중 하나를 랜덤 선택
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