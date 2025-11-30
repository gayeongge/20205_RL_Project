# submission_files Python Agents – 보상 & 에이전트 설계 비교

## 전체 경향 요약
- **휴리스틱 단일/다중 스텝 탐색** (`submission_one_step_lookahead.py`, `submission_minimax_alpha_beta.py`)은 패턴 가중치로 보드를 즉시 평가하고, 탐색 깊이를 늘려가며 최적 수를 고르는 방식이다.
- **MTD(f) + Negamax 변형** (`submission_mtd_f_negamax*.py`)은 같은 패턴 기반 평가를 쓰되, 시간 제한, 트랜스포지션 테이블, MTD(f) 반복 탐색으로 연속적인 깊이 확장을 수행한다.
- **MCTS(AlphaZero 스타일)** (`submission_alphazero_edu_4.py`, `submission_alphazero_edu_5.py`)은 휴리스틱 정책/가치로 prior를 만들고, 시뮬레이션 횟수를 늘려가며 방문 수 기반 결정을 한다.
- **DQN 계열** (`submission_DQN.py`, `submission_drl_dqn.py`)은 CNN 기반 Q-network를 학습한 뒤, 내장된 가중치로 합법 수만 argmax 하는 RL 접근이다. 전자는 보상 shaping을, 후자는 Kaggle 원 보상을 그대로 사용한다.

## 파일별 상세 비교
### submission_one_step_lookahead.py
- **보상/평가 설계**: `PATTERN_WEIGHTS`로 내 돌 4/3/2 줄과 상대 줄을 각각 +/− 가중치로 평가하는 순수 휴리스틱이다 (`src/submission_files/submission_one_step_lookahead.py:6`). `evaluate_board`는 가능한 모든 4칸 창을 세어 가중 합을 계산해 수의 가치를 대신한다 (`src/submission_files/submission_one_step_lookahead.py:93`).
- **에이전트 설계**: 가능한 열마다 토큰을 드롭해 본 뒤 해당 heuristic 점수를 계산하고, 최고 점수 후보 중 무작위로 선택하는 원-스텝 룩어헤드이다 (`src/submission_files/submission_one_step_lookahead.py:130`).

### submission_minimax_alpha_beta.py
- **보상/평가 설계**: 동일한 패턴 기반 점수표를 쓰지만, 상대 줄 차단과 내 줄 확장 모두에 상대적으로 큰 가중치를 둬 공격·수비 균형을 맞춘다 (`src/submission_files/submission_minimax_alpha_beta.py:8`, `src/submission_files/submission_minimax_alpha_beta.py:71`).
- **에이전트 설계**: 깊이 `MAX_DEPTH`까지 Minimax를 전개하면서 알파-베타로 가지치기를 해 탐색 폭을 줄인다 (`src/submission_files/submission_minimax_alpha_beta.py:4`, `src/submission_files/submission_minimax_alpha_beta.py:106`). 각 열을 1수 적용한 후 상대 차례에서 재귀적으로 평가하고, 동점이면 후보 중 랜덤으로 출력한다 (`src/submission_files/submission_minimax_alpha_beta.py:171`).

### submission_mtd_f_negamax.py
- **보상/평가 설계**: 승/패는 ±1e8, 블록/갭 위협/중앙 선호 등의 상수를 포함한 상대적 점수 체계를 둬 현재 플레이어 관점에서의 우위만 계산한다 (`src/submission_files/submission_mtd_f_negamax.py:17`, `src/submission_files/submission_mtd_f_negamax.py:83`). 창 스캔 중 상대 4개가 보이면 즉시 패배 점수를 반환해 탐색이 빠르게 끊긴다.
- **에이전트 설계**: 트랜스포지션 테이블을 사용한 Negamax에 MTD(f) 반복을 덧씌워, 타임아웃 내에서 추정치를 수렴시키는 엔진이다 (`src/submission_files/submission_mtd_f_negamax.py:144`, `src/submission_files/submission_mtd_f_negamax.py:201`). 루트에서는 즉시 승리/차단 수를 먼저 검사하고, 이후 반복적 깊이 증가로 최고 추정 수를 갱신한다 (`src/submission_files/submission_mtd_f_negamax.py:224`).

### submission_mtd_f_negamax_nf.py
- **보상/평가 설계**: 단순화된 점수표지만 상대 3연속은 크게 벌점, 내 2·3연속은 보너스를 주는 식으로 동일한 상대적 평가를 제공한다 (`src/submission_files/submission_mtd_f_negamax_nf.py:20`, `src/submission_files/submission_mtd_f_negamax_nf.py:84`).
- **에이전트 설계**: 동일한 Negamax + MTD(f) 틀을 쓰되, 열 높이 계산과 무의미한 수 거르기 등 안전장치를 강조한다 (`src/submission_files/submission_mtd_f_negamax_nf.py:124`, `src/submission_files/submission_mtd_f_negamax_nf.py:168`, `src/submission_files/submission_mtd_f_negamax_nf.py:202`).

### submission_alphazero_edu_4.py
- **보상/평가 설계**: `count_potential_lines`, `evaluate_window`, `heuristic_evaluate`로 연결 가능성, 대각선 가중치, 상대 잠재력 패널티 등을 동시에 고려하는 전략형 휴리스틱을 만든다 (`src/submission_files/submission_alphazero_edu_4.py:104`, `src/submission_files/submission_alphazero_edu_4.py:298`, `src/submission_files/submission_alphazero_edu_4.py:330`). 이 점수는 정책 확률과 가치 추정 모두에 사용돼 MCTS prior/value 역할을 한다 (`src/submission_files/submission_alphazero_edu_4.py:377`).
- **에이전트 설계**: `MCTSNode`가 PUCT 공식을 이용해 자식을 선택하고 방문 수를 업데이트하며 (`src/submission_files/submission_alphazero_edu_4.py:465`), `mcts_search`는 80회 시뮬레이션 후 방문 수가 가장 높은 열을 반환한다 (`src/submission_files/submission_alphazero_edu_4.py:536`). 루트에서는 즉시 승리/차단 수를 먼저 처리한 뒤 MCTS 결과를 따른다 (`src/submission_files/submission_alphazero_edu_4.py:585`).

### submission_alphazero_edu_5.py
- **보상/평가 설계**: 창 단위 위협 분석(`analyze_threat_window`)으로 gap threat, 3연속 등급을 세분화하고 (`src/submission_files/submission_alphazero_edu_5.py:65`), 전반적 휴리스틱은 상대/자신 잠재 라인을 동시에 계산한다 (`src/submission_files/submission_alphazero_edu_5.py:175`). `get_policy_value`는 이러한 위협 점수를 기반으로 prior를 생성해 방어 우선순위를 높인다 (`src/submission_files/submission_alphazero_edu_5.py:190`).
- **에이전트 설계**: 구조는 4번과 동일하지만 시뮬레이션 횟수를 100으로 늘리고, gap 위협 탐지 결과를 child prior에 반영해 방어적 MCTS가 된다 (`src/submission_files/submission_alphazero_edu_5.py:254`, `src/submission_files/submission_alphazero_edu_5.py:296`). 루트 단계에서도 즉시 승/패 차단을 수행한 뒤 MCTS를 호출한다 (`src/submission_files/submission_alphazero_edu_5.py:314`).

### submission_drl_dqn.py
- **보상/평가 설계**: Kaggle 환경이 주는 보상을 그대로 사용하며, 추가 shaping 없이 step transition을 버퍼에 저장한다 (`src/submission_files/submission_drl_dqn.py:201`). 이는 `submission_DQN.py` 대비 간단하지만 보상 신호가 희소해 학습 안정성이 낮을 수 있다.
- **에이전트 설계**: CNN 아키텍처와 리플레이 버퍼는 동일하나, `train_dqn_agent`가 더 긴 에피소드 수와 Double-DQN 스타일 마스크를 포함하지 않는 기본 TD 업데이트를 사용한다 (`src/submission_files/submission_drl_dqn.py:128`, `src/submission_files/submission_drl_dqn.py:180`). 추론은 역시 내장 가중치 기반 `EmbeddedPolicy`로 합법 열에서 argmax를 수행하고, 준비된 가중치가 없을 경우 중앙/랜덤 휴리스틱으로 폴백한다 (`src/submission_files/submission_drl_dqn.py:234`, `src/submission_files/submission_drl_dqn.py:292`).

### submission_drl_double.py
- **보상/평가 설계**: Kaggle 환경 리워드를 그대로 사용하며 추가 shaping 없이 경험을 버퍼에 적재한다 (`src/submission_files/submission_drl_double.py:173`). 엔트리 보드가 잘못된 위치일 때만 즉시 종료되어 음수 보상을 받으므로, reward는 전적으로 승패/무승부에 의해 결정된다.
- **에이전트 설계**: 기본 CNN Q-network로 Q 값을 추론하지만, 타깃 계산 시 온라인 네트워크가 다음 행동을 선택하고 타깃 네트워크가 그 행동의 Q 값을 평가하는 Double DQN 공식을 쓴다 (`src/submission_files/submission_drl_double.py:118`, `src/submission_files/submission_drl_double.py:141`). 학습된 가중치는 NumPy `EmbeddedPolicy`로 내장되어 Kaggle 환경에서 합법 열만 남긴 뒤 argmax를 수행한다 (`src/submission_files/submission_drl_double.py:227`).

### submission_drl_dueling.py
- **보상/평가 설계**: Double DQN 변형과 동일하게 Kaggle 기본 보상만 활용하며, 별도의 shaping 없이 replay buffer에 저장한다 (`src/submission_files/submission_drl_dueling.py:180`). 따라서 승리/패배 신호에 민감하게 반응하도록 설계되었다.
- **에이전트 설계**: CNN 특징 추출 뒤 가치/어드밴티지 스트림을 분리한 Dueling 구조를 갖고, Double DQN 방식으로 타깃을 계산해 안정성을 높인다 (`src/submission_files/submission_drl_dueling.py:117`, `src/submission_files/submission_drl_dueling.py:148`). 추론 단계에서도 내장 `EmbeddedPolicy`가 두 스트림을 다시 결합해 합법 열의 argmax를 반환하며, 가중치가 없으면 중앙/랜덤 휴리스틱으로 폴백한다 (`src/submission_files/submission_drl_dueling.py:234`, `src/submission_files/submission_drl_dueling.py:303`).
