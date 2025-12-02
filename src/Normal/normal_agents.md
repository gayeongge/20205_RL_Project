# ConnectX Normal 에이전트 변형 비교

## 한눈 요약
| 파일 | 변경된 이름 | 개선 배경 | 핵심 특징 | 현재 한계 |
| --- | --- | --- | --- | --- |
| `simple_greedy_baseline.py` | (기존 `one_step_lookahead.py`) | 즉석 반응 기반 기본 휴리스틱으로 단순하고 빠른 에이전트 구현 | 1수 앞 평가만으로 즉시 최선수 선택, 패턴 기반 점수 계산 (승리 1M, 방어 -50K), 계산 복잡도 O(열수) | 전략적 깊이 부재로 함정에 빠지기 쉬움, 상대 다음수 예측 불가능으로 수비적 플레이 취약 |
| `minimax_basic_search.py` | (기존 `n_step_lookahead.py`) | 1수 예측의 한계 극복을 위해 기본 minimax로 다수 미래 시나리오 탐색 | 깊이 3 minimax 탐색, 재귀적 최적해 탐색, 강화된 패턴 점수 (내 3목 10K, 상대 3목 -50K) | Alpha-beta 가지치기 없어 탐색 비효율, 동일 상황 중복 계산으로 깊이 제한 |
| `minimax_optimized_search.py` | (기존 `minimax_alpha_beta.py`) | 기본 minimax의 탐색 비효율성 해결을 위한 alpha-beta 가지치기 최적화 | 깊이 4 탐색 + Alpha-beta 가지치기로 탐색 공간 대폭 절약, 동일 패턴 점수 체계 유지, 실용적 성능 확보 | 전이표 없어 동일 보드 상태 중복 평가, 정적 평가함수의 단순함으로 복잡한 전술 패턴 놓침 |

## 파일별 상세

### `simple_greedy_baseline.py` (기존 `one_step_lookahead.py`)
- **개선 배경.** ConnectX 기본 에이전트로서 즉석에서 빠른 판단을 내리는 단순 휴리스틱 기반 구현.
- **핵심 특징.**
  - **즉석 평가**: 현재 보드에서 각 열에 돌을 놓았을 때의 즉시 효과만 계산하여 O(열수) 복잡도로 빠른 의사결정
  - **패턴 기반 점수**: 내 4목 연결 1M점, 내 3목 1K점, 상대 3목 방어 -50K점 등 직관적 가중치 체계
  - **단순 그리디**: `count_pattern()`으로 현재 상태의 연결 패턴만 카운트하여 가장 높은 점수의 수 즉시 선택
  - **기본 fallback**: 계산된 점수가 동일할 경우 무작위 선택으로 예측 불가능성 확보
- **한계.** 1수 후 결과만 고려하여 상대의 반격이나 장기 전략 부재, 함정 상황(상대에게 승리 기회 제공)에 취약.

### `minimax_basic_search.py` (기존 `n_step_lookahead.py`)
- **개선 배경.** 1수 예측의 전략적 한계를 극복하기 위해 기본 minimax 알고리즘으로 다수 미래 시나리오를 체계적으로 탐색.
- **핵심 특징.**
  - **기본 Minimax**: `LOOKAHEAD_DEPTH=3`으로 3수 앞까지 모든 가능한 게임 트리를 재귀적으로 탐색
  - **Zero-sum 게임 모델링**: 내 차례에는 최대값, 상대 차례에는 최소값을 추구하는 고전적 minimax 구조
  - **강화된 패턴 점수**: 내 3목 10K점으로 공격성 강화, 상대 2목 -200점으로 조기 견제 체계
  - **완전 탐색**: 가지치기 없이 모든 노드를 방문하여 이론적으로 최적해 보장 (주어진 깊이 내에서)
- **한계.** Alpha-beta 가지치기 부재로 지수적 탐색 비용, 깊이 제한으로 여전히 장기 전략 부족, 동일 보드 상태 중복 계산.

### `minimax_optimized_search.py` (기존 `minimax_alpha_beta.py`)
- **개선 배경.** 기본 minimax의 탐색 비효율성을 Alpha-beta 가지치기로 해결하여 실용적 성능과 깊이 확보.
- **핵심 특징.**
  - **Alpha-beta 가지치기**: `MAX_DEPTH=4`로 깊이 증가하면서도 불필요한 분기 조기 차단으로 탐색 효율 대폭 개선
  - **최적화된 탐색**: `alpha`(지금까지 최대값), `beta`(지금까지 최소값) 추적으로 확실히 나쁜 분기 건너뛰기
  - **동일 평가 체계**: n_step_lookahead와 동일한 패턴 점수 유지하여 일관된 전략 판단 기준
  - **실용적 균형**: 이론적 최적성을 유지하면서도 현실적 시간 내 더 깊은 탐색 가능
- **한계.** 전이표(해시 테이블) 부재로 동일 보드 상태 재계산, 정적 평가함수 단순함으로 복잡한 전술적 뉘앙스 놓침.

## 📊 알고리즘 진화 과정

### 1단계: 즉석 반응 (One-step Lookahead)
```python
# 각 가능한 수에 대해 즉시 결과만 평가
for col in valid_actions:
    new_grid = drop_piece(grid, col, me, cfg)
    score = evaluate_board(new_grid, me, cfg)
    # 가장 좋은 점수의 수 선택
```

### 2단계: 기본 미래 예측 (Basic Minimax)
```python
def minimax(grid, cur_player, me, depth, cfg):
    if depth == 0:
        return evaluate(grid, me, cfg)
    
    if cur_player == me:  # 내 차례 - 최대화
        return max(minimax(...) for each action)
    else:  # 상대 차례 - 최소화
        return min(minimax(...) for each action)
```

### 3단계: 효율적 최적화 (Alpha-beta Pruning)
```python
def minimax_alpha_beta(grid, cur_player, me, depth, alpha, beta, cfg):
    # 기본 minimax + 가지치기
    if score >= beta:  # 베타 차단
        return score
    alpha = max(alpha, score)  # 알파 업데이트
```

## 🎯 패턴 점수 체계 비교

| 패턴 | One-step | N-step/Alpha-beta | 설명 |
|------|----------|------------------|------|
| 내 4목 | 1,000,000 | 1,000,000 | 즉시 승리 |
| 내 3목 | 1,000 | 10,000 | 공격 강화 (10배) |
| 내 2목 | 10 | 100 | 기반 구축 (10배) |
| 상대 3목 | -50,000 | -50,000 | 필수 방어 |
| 상대 2목 | -100 | -200 | 조기 견제 강화 |

**진화 포인트**: N-step부터 공격성 대폭 강화 (3목, 2목 점수 10배 증가)

## 실무 관점 메모
1. **빠른 프로토타입**이 필요하면 `simple_greedy_baseline.py`로 즉석 반응 확인
2. **기본 전략성** 확보가 목표면 `minimax_basic_search.py`로 미래 예측 도입
3. **실전 성능** 최적화가 필요하면 `minimax_optimized_search.py`로 효율적 깊이 탐색
4. **계산 복잡도**: O(열수) → O(브랜칭^깊이) → O(효율적브랜칭^깊이) 순으로 증가하지만 전략적 완성도도 향상

## 파일명 변경 완료 ✅
- `one_step_lookahead.py` → `simple_greedy_baseline.py` (즉석 반응 기준선)
- `n_step_lookahead.py` → `minimax_basic_search.py` (기본 minimax 탐색)
- `minimax_alpha_beta.py` → `minimax_optimized_search.py` (최적화된 탐색)