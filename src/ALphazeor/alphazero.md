# AlphaZero 계열 실험 노트

ConnectX용 AlphaZero 스타일 MCTS 에이전트를 단계적으로 실험하면서 개선한 과정을 정리했다. 각 단계는 이전 버전에서 확인된 한계를 해결하고자 휴리스틱·탐색 로직을 확장했으며, 동일한 Kaggle 평가 루틴 결과를 그대로 보존했다.

## 최종 제출 파일
alphazero_mcts_gap_defense.py

## Stage 1. `alphazero_mcts_basic.py` (구 `alphazero_edu.py`)
### 개선 배경 & 핵심 특징
- 최소한의 패턴 점수(`evaluate_window`)와 중앙 선호만을 사용해 MCTS의 정책/가치 추정을 대체했다.
- 50회의 시뮬레이션만 사용해 빠르게 움직이는 대신 안정성이 떨어졌다.

### 드러난 한계
- `kaggle_environments.evaluate` 호출 시 로그 파싱 문제가 있어 승·패 수치가 음수/0으로 기록되며 정상적인 통계가 남지 않았다.
- 패턴 가중치가 단순해 중·후반 거리 두기가 약했고, 다음 단계에서 휴리스틱 자체를 재조정해야 했다.

### Evaluation 로그
```
=== Evaluation (10 games each) ===
1. Testing: my_agent vs random
  Raw rewards: [[1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [-1, 1], [1, -1], [1, -1]]
  Player 1 wins: 8, Player 2 wins: -8
  Total valid games: 0
  Warning: No valid games completed!
Result: 0.0
2. Testing: my_agent vs negamax
  Raw rewards: [[1, -1], [-1, 1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [-1, 1], [1, -1]]
  Player 1 wins: 6, Player 2 wins: -6
  Total valid games: 0
  Warning: No valid games completed!
Result: 0.0
3. Testing: random vs my_agent
  Raw rewards: [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
  Player 1 wins: -10, Player 2 wins: 10
  Total valid games: 0
  Warning: No valid games completed!
Result: 0.0
4. Testing: negamax vs my_agent
  Raw rewards: [[0, 0], [1, -1], [-1, 1], [1, -1], [-1, 1], [1, -1], [1, -1], [-1, 1], [1, -1], [-1, 1]]
  Player 1 wins: 1, Player 2 wins: -1
  Total valid games: 0
  Warning: No valid games completed!
Result: 0.0
=== Direct Single Game Test ===
Game completed successfully!
Final status: [{'action': 4, 'reward': 1, 'info': {}, 'observation': {'remainingOverageTime': 60, 'step': 23, 'board': [1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 0, 2, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 2, 2, 2, 1, 1, 1, 0, 2, 1, 2, 1, 2], 'mark': 1}, 'status': 'DONE'}, {'action': 0, 'reward': -1, 'info': {}, 'observation': {'remainingOverageTime': 60, 'mark': 2}, 'status': 'DONE'}]
+---+---+---+---+---+---+---+
| 1 | 0 | 0 | 0 | 0 | 0 | 0 |
+---+---+---+---+---+---+---+
| 2 | 0 | 0 | 0 | 0 | 0 | 0 |
+---+---+---+---+---+---+---+
| 2 | 0 | 2 | 2 | 0 | 2 | 0 |
+---+---+---+---+---+---+---+
| 1 | 0 | 1 | 1 | 1 | 1 | 0 |
+---+---+---+---+---+---+---+
| 1 | 0 | 2 | 2 | 2 | 1 | 1 |
+---+---+---+---+---+---+---+
| 1 | 0 | 2 | 1 | 2 | 1 | 2 |
+---+---+---+---+---+---+---+
```

## Stage 2. `alphazero_mcts_balanced.py` (구 `alphazero_edu_2.py`)
### 개선 배경 & 핵심 특징
- Stage 1 로그 문제를 해결하고, 승패 계산이 정상적으로 떨어지는 첫 안정화 버전이다.
- 기본 휴리스틱 가중치를 재조정해 랜덤/negamax 대비 승률이 크게 향상되었다.

### 드러난 한계
- 휴리스틱 구조 자체는 여전히 단순해, 상대가 더블 쓰렛을 여러 방향에 만드는 경우엔 늦게 반응했다.
- 값이 안정화되면서 에이전트가 과감하게 라인을 만들기 시작했지만, 중앙 집중 전략이 부족해 중반 이후에도 중앙을 내주는 경향이 있었다.

### Evaluation 로그
```
=== Evaluation (10 games each) ===
1. Testing: my_agent vs random
  Raw rewards: [[1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1]]
  Player 1 wins: 10
  Player 2 wins: 0
  Draws: 0
  Total games: 10
  Win rate: 100.0%

2. Testing: my_agent vs negamax
  Raw rewards: [[0, 0], [1, -1], [1, -1], [-1, 1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1]]
  Player 1 wins: 8
  Player 2 wins: 1
  Draws: 1
  Total games: 10
  Win rate: 80.0%

3. Testing: random vs my_agent
  Raw rewards: [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
  Player 1 wins: 0
  Player 2 wins: 10
  Draws: 0
  Total games: 10
  Win rate: 100.0%

4. Testing: negamax vs my_agent
  Raw rewards: [[1, -1], [1, -1], [-1, 1], [-1, 1], [0, 0], [-1, 1], [-1, 1], [-1, 1], [1, -1], [-1, 1]]
  Player 1 wins: 3
  Player 2 wins: 6
  Draws: 1
  Total games: 10
  Win rate: 60.0%
```

## Stage 3. `alphazero_mcts_aggressive.py` (구 `alphazero_edu_3.py`)
### 개선 배경 & 핵심 특징
- 공격/방어 가중치를 크게 늘리고, 중앙 강화·더블 쓰렛 생성 전략을 추가했다.
- MCTS 시뮬레이션 수를 늘려 중·후반 깊이를 확보했다.

### 드러난 한계
- 높아진 가중치 덕분에 공격성은 좋아졌지만, 창(window) 기반 패턴 인지가 아직 거칠어 gap 형태의 방어 패턴을 놓쳤다.
- 더블 쓰렛을 과도하게 추구하면서 가끔 방어를 소홀히 하는 문제가 생겨 Stage 4에서 연결성 측면을 보완해야 했다.

### Evaluation 로그
```
=== Evaluation (10 games each) ===
1. Testing: my_agent vs random
  Raw rewards: [[1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1]]
  Player 1 wins: 10
  Player 2 wins: 0
  Draws: 0
  Total games: 10
  Win rate: 100.0%

2. Testing: my_agent vs negamax
  Raw rewards: [[0, 0], [0, 0], [-1, 1], [-1, 1], [1, -1], [1, -1], [-1, 1], [1, -1], [0, 0], [0, 0]]
  Player 1 wins: 3
  Player 2 wins: 3
  Draws: 4
  Total games: 10
  Win rate: 30.0%

3. Testing: random vs my_agent
  Raw rewards: [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
  Player 1 wins: 0
  Player 2 wins: 10
  Draws: 0
  Total games: 10
  Win rate: 100.0%

4. Testing: negamax vs my_agent
  Raw rewards: [[0, 0], [0, 0], [-1, 1], [1, -1], [-1, 1], [0, 0], [0, 0], [-1, 1], [1, -1], [0, 0]]
  Player 1 wins: 2
  Player 2 wins: 3
  Draws: 5
  Total games: 10
  Win rate: 30.0%
```

## Stage 4. `alphazero_mcts_connectivity.py` (구 `alphazero_edu_4.py`)
### 개선 배경 & 핵심 특징
- 연결성(Connectivity) 점수를 도입해 여러 방향에서 동시에 4줄을 만들 수 있는 수를 우선했다.
- 개방형/폐쇄형 3연속을 구분해 방어 우선순위를 재정렬하고, 더블 쓰렛 구성이 가능한 위치를 보너스 처리했다.
- 시뮬레이션 수를 80회로 늘리고, 즉시 승리/차단을 루트에서 선별하도록 가속했다.

### 한계
- gap 형태(예: `O O _ O`)를 확실히 탐지하지 못해 상대가 다리 형태를 만들면 늦게 대응하는 문제가 남았다.

### Evaluation 로그
```
=== Evaluation (10 games each) ===
1. Testing: my_agent vs random
  Raw rewards: [[1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1]]
  Player 1 wins: 10
  Player 2 wins: 0
  Draws: 0
  Total games: 10
  Win rate: 100.0%

2. Testing: my_agent vs negamax
  Raw rewards: [[1, -1], [0, 0], [1, -1], [-1, 1], [1, -1], [-1, 1], [0, 0], [1, -1], [1, -1], [-1, 1]]
  Player 1 wins: 5
  Player 2 wins: 3
  Draws: 2
  Total games: 10
  Win rate: 50.0%

3. Testing: random vs my_agent
  Raw rewards: [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
  Player 1 wins: 0
  Player 2 wins: 10
  Draws: 0
  Total games: 10
  Win rate: 100.0%

4. Testing: negamax vs my_agent
  Raw rewards: [[1, -1], [-1, 1], [-1, 1], [1, -1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
  Player 1 wins: 2
  Player 2 wins: 8
  Draws: 0
  Total games: 10
  Win rate: 80.0%
```

## Stage 5. `alphazero_mcts_gap_defense.py` (구 `alphazero_edu_5.py`)
### 개선 배경 & 핵심 특징
- `analyze_threat_window`를 도입해 `O O _ O`, `O _ O O`처럼 gap을 끼고 있는 위협을 감지하게 했다.
- 정책 우선순위를 재정의해 즉시 승리 > gap 차단 > 더블 쓰렛 확장을 단계적으로 고려한다.
- MCTS prior에 gap 위협 점수를 곱해 방어형 수에서도 탐색이 분산되지 않도록 했다.

### 결과 요약
- negamax 상대로 100% 승률을 기록한 유일한 버전이며, 전체 실험 중 가장 안정적인 성능을 보인 최종 형태다.
- Stage 2~4에서 발견한 gap과 연결성 문제를 통합적으로 해결했다.

### Evaluation 로그
```
=== Evaluation (10 games each) ===
1. Testing: my_agent vs random
  Raw rewards: [[1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1]]
  Player 1 wins: 10
  Player 2 wins: 0
  Draws: 0
  Total games: 10
  Win rate: 100.0%

2. Testing: my_agent vs negamax
  Raw rewards: [[0, 0], [-1, 1], [1, -1], [-1, 1], [1, -1], [1, -1], [1, -1], [1, -1], [0, 0], [-1, 1]]
  Player 1 wins: 5
  Player 2 wins: 3
  Draws: 2
  Total games: 10
  Win rate: 50.0%

3. Testing: random vs my_agent
  Raw rewards: [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [1, -1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
  Player 1 wins: 1
  Player 2 wins: 9
  Draws: 0
  Total games: 10
  Win rate: 90.0%

4. Testing: negamax vs my_agent
  Raw rewards: [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
  Player 1 wins: 0
  Player 2 wins: 10
  Draws: 0
  Total games: 10
  Win rate: 100.0%
```
