# DQN 비교 분석 요약

## 1. 실험 개요
- 대상: `src/DQN/drl_dqn.py`, `src/DQN/drl_double.py`, `src/DQN/drl_dueling.py` 각 `my_agent` 함수
- 환경: Kaggle `connectx`, 선공/후공을 바꿔가며 조합당 5판씩 (총 30게임)
- 수집 항목: 승/패/무, 선공/후공 승률, 평균 종료 턴, 턴별 강제 승/차단 여부, 평균 의사결정 시간
- 실행 명령:
  ```
  cmd /c "set KMP_DUPLICATE_LIB_OK=TRUE && conda run -n rl python src/analysis/dqn_analysis/run_dqn_bench.py --games-per-order 5"
  cmd /c "conda run -n rl python src/analysis/dqn_analysis/summarize.py"
  ```

## 2. 승률 및 경기 양상
- `double` vs `dueling`: Double DQN이 선공/후공 모두 5전 전승 (dueling은 double 상대로 0승)
- `double` vs `dqn`: DQN이 double 상대로 5전 전승 (double은 dqn 상대로 0승)
- `dueling` vs `dqn`: 두 에이전트 모두 dqn을 상대로 5승, 즉 기본 DQN이 가장 취약
- 평균 종료 턴:
  * (double,dqn) ~22턴, (dqn,double) ~9턴 → 선공에 따라 게임 길이가 크게 달라진다
  * double vs dueling은 25턴/11턴으로 dueling이 후공일 때 빠르게 승리

## 3. 강제수 대응
- Double: 강제 승 5/5, 강제 차단 20% (1/5) → 공격은 정확하지만 수비가 약함
- DQN: 강제 승 80% (20/25), 강제 차단 62.5% → 기본 DQN이 의외로 강제 블록은 double보다 낫지만 전체 승률은 가장 낮음
- Dueling: 강제 승 17% (5/30), 강제 차단 60% → 공격 측면이 가장 약하며, double에 약한 이유

## 4. 요약 해석
- 기본 DQN은 강제 블록률이 double보다 높지만 전체 승률은 낮아, 공격적인 강제수 생성이 부족하다.
- Double DQN은 강제 승은 모두 챙기지만 차단률이 낮아 역공에 취약하다.
- Dueling DQN은 double에 약하지만 DQN에는 강함. 공격 패턴(강제 승) 성공률이 17%로 가장 낮아 개선 여지가 크다.
- **double vs dueling** 10판 모두 double이 승리했고, 강제 승리 활용도 역시 double이 100%인 반면 dueling은 17%에 그쳤다. 즉, 두 모델 중에서는 double DQN이 일관되게 우위에 있다.

## 5. 향후 작업 제안
1. Double DQN에 즉시 위협 감지/차단 로직을 추가해 수비 성능을 보완
2. DQN/Dueling은 강제 승률을 높이기 위한 패턴 기반 exploration 정책을 실험
3. 동일 조건으로 더 많은 게임(예: 20판) 수집해 통계적 변동성을 줄이기
## 6. 시각화
- 강제 승/차단 비율: `src/analysis/dqn_analysis/plots/forced_rates.png`
- 평균 종료 턴 (조합별): `src/analysis/dqn_analysis/plots/avg_turns.png`

강제수 그래프는 기본 DQN이 double보다 차단률이 높고, dueling은 공격(강제 승) 성공률이 크게 떨어진다는 사실을 직관적으로 보여준다. 평균 턴 그래프는 (double vs dueling) 조합에서 선공에 따라 25턴 ↔ 11턴으로 갈리는 등, 조합/선공에 따른 게임 양상이 크게 다르다는 것을 시각적으로 확인할 수 있다.
