🏆 최종 보상 및 방어 체계 (Scoring System)
사용자님의 전략적 요청사항을 MTD(f)의 정적 평가 함수(Static Evaluation) 점수로 환산하면 다음과 같습니다. 이 점수들은 탐색 엔진이 "어떤 상태가 유리한지" 판단하는 기준이 됩니다.

승리/패배 (Game Over)

내 승리 (4목 완성): +1,000,000,000 (10억 점 / 절대적)

상대 승리 (4목 허용): -1,000,000,000 (-10억 점 / 절대적)

치명적 위협 (Critical Threat) - "무조건 막아라"

상대 3개 + 빈칸 1개 (연속/징검다리 포함): -20,000,000

상대가 다음 수에 끝낼 수 있는 상태입니다. 내가 이길 수 없다면, 내 공격 점수(100,000 등)보다 패배 방어 점수의 비중을 훨씬 높게 잡아 무조건 막도록 강제했습니다.

강력한 기회 (Major Opportunity)

내 3개 + 빈칸 1개: +100,000

상대를 압박하여 방어를 강요하거나 승리로 연결되는 기회입니다.

잠재적 위협/기회 (Potential) - "미리 끊어라"

상대 2개 + 빈칸 2개: -40,000

요청하신 대로, 상대가 2개일 때 미리 끊어주는 플레이에 높은 가중치를 주었습니다. (단순 연결 점수보다 훨씬 높음)

더블 위협 (Double Threat): MTD(f) 탐색 알고리즘은 수읽기를 통해 자동으로 이 상황(양수겸장)을 찾아내므로 별도 점수가 없어도 +100,000 이상의 가치로 판단하게 됩니다.

기초 점수 (Positional)

내 2개: +1,000

중앙 점령: +10 (중앙에 가까울수록 유리)

## 결과
=== Evaluation (10 games each) ===

1. Testing: my_agent vs random
  Raw rewards: [[1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1]]
  Player 1 wins: 10
  Player 2 wins: 0
  Draws: 0
  Total games: 10
  Win rate: 100.0%

2. Testing: my_agent vs negamax
  Raw rewards: [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
  Player 1 wins: 0
  Player 2 wins: 10
  Draws: 0
  Total games: 10
  Win rate: 0.0%

3. Testing: random vs my_agent
  Raw rewards: [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [1, -1]]
  Player 1 wins: 1
  Player 2 wins: 9
  Draws: 0
  Total games: 10
  Win rate: 90.0%

4. Testing: negamax vs my_agent
  Raw rewards: [[1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1]]
  Player 1 wins: 10
  Player 2 wins: 0
  Draws: 0
  Total games: 10
  Win rate: 0.0%

  # negamax 기반 mtdf
  💡 무엇이 바뀌었나요? (승리 포인트)
Negamax 전환:

기존: player와 root_player를 비교하며 점수를 계산 -> 실수하기 딱 좋음.

변경: negamax 함수는 항상 **"지금 돌을 두는 사람 입장에서의 점수"**를 계산하고, 재귀 호출할 때 -를 붙여 뒤집습니다. (Zero-sum 게임의 정석)

안전한 시간 관리 (Safe-fail):

try-except TimeoutError 구문을 사용하여, 깊이 4를 탐색하다가 시간이 초과되면 억지로 결과를 내지 않고 **"완벽하게 끝난 깊이 3의 결과"**를 best_action으로 사용합니다.

이전 코드에서는 타임아웃 시 방금 탐색하던(완성되지 않은) 쓰레기 값을 반환했을 가능성이 큽니다.

점수 체계 부호 정리:

SCORE_LOSS를 -1억으로 설정하고, 상대 3목 방어 실패 시 -8만 패널티를 주었습니다.

negamax는 -(-8만) = +8만으로 인식하여, 상대의 공격을 막는 수가 나에게 엄청난 이득이라고 정확히 판단하게 됩니다.

이 코드로 negamax 에이전트와 다시 붙어보세요. 최소한 어이없게 지는 일은 사라질 것입니다. 파이썬의 속도 한계 때문에 깊이는 3~5 정도가 한계일 텐데, 이 깊이 내에서는 실수하지 않을 것입니다.

## 결과
=== Evaluation (10 games each) ===

1. Testing: my_agent vs random
  Raw rewards: [[1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1]]
  Player 1 wins: 10
  Player 2 wins: 0
  Draws: 0
  Total games: 10
  Win rate: 100.0%

2. Testing: my_agent vs negamax
  Raw rewards: [[1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1]]
  Player 1 wins: 10
  Player 2 wins: 0
  Draws: 0
  Total games: 10
  Win rate: 100.0%

3. Testing: random vs my_agent
  Raw rewards: [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
  Player 1 wins: 0
  Player 2 wins: 10
  Draws: 0
  Total games: 10
  Win rate: 100.0%

4. Testing: negamax vs my_agent
  Raw rewards: [[-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
  Player 1 wins: 0
  Player 2 wins: 10
  Draws: 0
  Total games: 10
  Win rate: 100.0%

  ## 자살 수 배제
  