### Alphazero_edu.py 결과
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

### Alphazero_edu_2.py
#### 개선사항

#### 결과
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


### Alphazero_edu_3.py
#### 개선사항
negmax에서 승률 확률 높도록
두번째 턴으로 플레이할 때도 승률 높게
1. MCTS 시뮬레이션 증가 : 더 깊은 탐색
2. 평가 함수 강화
    - 공격 가중치 증가 : 1000 -> 5000
    - 방어 가중치 증가 ; 5000 -> 100000
    - 중앙 제어 강화 : 3점 -> 6점
3. 전략적 개선
- 트랩 생성 감지 : 2가지 이상 승리 경로 만들기
- 상대 승리 기회 차단: 상대가 다음 수에 이길 수 있는 곳 회피
- 중앙 선호도 : 중심 열에 보너스 점수
4. 후공 전략
- 강화된 방어 전략
- 상대 위헙에 대한 빠른 대응
- 중앙 장악 우선순위


#### 결과
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


### Alphazero_edu_4.py
#### 개선사항
중앙 집중 구조 : 중앙 선점은 초반에 해야 한다.
1. 연결 가능성 (Connectivity)
4개를 연결할 수 있는 잠재적 경로가 중요
한 수가 여러 방향(가로/세로/대각선)에 기여하면 좋음
2. 개방형 vs 폐쇄형
개방형 3연속 (양쪽이 빈칸): 막기 어려움 → 높은 가치
폐쇄형 3연속 (한쪽이 막힘): 상대적으로 약함
3. 이중 위협 (Double Threat)
한 수로 2개 이상의 승리 경로 생성
상대가 막을 수 없음 → 승리 확정
4. 수직 제어
아래 칸을 채워야 위에 놓을 수 있음
높이 관리가 중요

1순위: 즉시 승리 (10,000,000점)
2순위: 상대 승리 차단 (9,000,000점)
3순위: 이중 위협 생성 (50,000점)
4순위: 연결성 확보 (최대 100,000점)
5순위: 대각선 기회 (8,000점)
6순위: 일반 패턴 평가

#### 결과
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

### Alphazero_edu_5.py
#### 개선사항
주요 변경 사항
analyze_threat_level 함수 추가:

내가 해당 컬럼에 돌을 두지 않았을 때, 상대방이 그 자리에 뒀다면 몇 개의 돌이 연속되게 되는지를 계산합니다.

결과가 4가 된다면? → 현재 상대가 3개를 이미 만들어둔 상태(즉시 패배 위기).

결과가 3이 된다면? → 현재 상대가 2개를 만들어둔 상태(잠재적 위기).

get_policy_value 내 점수 체계 세분화:

Priority 1 (내 승리): 10,000,000점 (변동 없음)

Priority 2 (상대 3목 방어 - 절대적): 9,000,000점. 상대가 4개를 완성하는 자리는 무조건 막습니다.

Priority 3 (상대 2목 견제 - 예방적): 40,000점. 상대가 3개를 만들려고 할 때 미리 끊어줍니다. (더블 위협 생성 점수 50,000점보다는 약간 낮게 설정하여, 내가 확실히 이길 수 있는 공격 찬스가 있다면 공격을 하도록 유도함. 만약 완전 수비형을 원하면 이 점수를 60,000점 이상으로 올리면 됩니다.)

아 추가적으로 연속 3개만 보는 거 말고 2개 1개 떨어져있는데 내가 다음 수를 놓았을 때 그 사이에 상대방 플레이어가 중간에 코인 넣어서 연속 4개 넣는 경우도 고려해서 만들어줘 - 윈도우 슬라이딩 기법 적용


#### 결과
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