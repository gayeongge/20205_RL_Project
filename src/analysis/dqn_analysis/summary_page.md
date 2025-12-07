# DQN 비교 분석 요약

## 사용법 요약
1. `cd src/analysis/dqn_analysis`
2. 실험 실행  
   `python run_dqn_bench.py --games-per-order 5`
3. 요약 생성  
   `python summarize.py`
4. 시각화  
   `python plot_metrics.py`
5. 산출물 확인  \\
   - data/bench_results.jsonl, data/summary.json
   - plots/forced_rates.png, plots/avg_turns.png

---

## 1. 실험 개요
- 대상: src/DQN/drl_dqn.py, src/DQN/drl_double.py, src/DQN/drl_dueling.py
- 환경: Kaggle connectx, 선공/후공 조합당 5판씩 (총 30게임)
- 수집 항목: 승/패/무, 선공/후공 승률, 평균 종료 턴, 강제 승/차단 여부, 평균 의사결정 시간
- Conda 환경을 쓰면 conda run -n rl python ... 형태로 실행

## 2. 승률 및 경기 양상
- double vs dueling: Double DQN이 선공/후공 모두 5전 전승
- double vs dqn: 기본 DQN이 double 상대로 5전 전승 → double의 수비 취약 노출
- dueling vs dqn: 두 모델 모두 기본 DQN을 상대로 전승 → baseline DQN이 가장 취약
- 평균 종료 턴: (double,dqn) 22턴 vs (dqn,double) 9턴처럼 선공 여부에 따라 경기 길이가 크게 달라짐

## 3. 강제수 대응
- Double: 강제 승 5/5, 강제 차단 20% → 공격은 정확하지만 역공 차단이 약함
- DQN: 강제 승 80%(20/25), 강제 차단 62.5% → double보다 차단 성능은 낫지만 전체 승률은 최저
- Dueling: 강제 승 17%(5/30), 강제 차단 60% → 공격 패턴이 가장 약해 double 상대로 취약

## 4. 요약 해석
- Double DQN은 강제 승 활용도는 최고지만 방어가 약해 DQN에게 전패
- 기본 DQN은 차단률은 준수하나 강제 승 생성력이 낮아 전체 승률이 가장 나쁨
- Dueling은 double에게는 지지만 DQN은 이김. 공격 성공률 개선이 최우선
- **결론**: 현재 가중치 기준으로 Double > Dueling > DQN 순으로, Double DQN을 선택

## 5. 시각화
- plots/forced_rates.png: 강제 승/차단 비율 비교
- plots/avg_turns.png: 조합별 평균 종료 턴

그래프로 보면 "double=공격형, DQN=수비형, dueling=공격 실패" 구조가 선명하게 드러나 후속 개선 논의에 활용할 수 있다.

## 6. 결과 표

### 강제 수 대응 비율
| Player  | Forced Win Rate | Forced Block Rate |
|---------|-----------------|-------------------|
| double  | 100%            | 20%               |
| dqn     | 80%             | 62.5%             |
| dueling | 16.7%           | 60%               |

### 평균 종료 턴
| Matchup             | Avg Turns |
|---------------------|-----------|
| double vs dqn       | 22        |
| double vs dueling   | 25        |
| dqn vs double       | 9         |
| dqn vs dueling      | 21        |
| dueling vs double   | 11        |
| dueling vs dqn      | 22        |
