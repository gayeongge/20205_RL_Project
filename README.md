# 📘 **ConnectX Reinforcement Learning Agents — Project README**

> Kaggle ConnectX 환경을 기반으로 여러 강화학습 기법(One-step, N-step, DQN 등)을 구현하고 비교하여, 가장 우수한 전략을 찾는 프로젝트입니다.
> 
> 프로젝트 인원 : 이가영, 이현지

---

## ✨ **프로젝트 목표**

ConnectX 강화를 위한 실험 환경과 문서를 동시에 제공하기 위해 다음 항목에 집중한다:

* ConnectX 환경에서 동작하는 휴리스틱·탐색·신경망 기반 에이전트를 통합 관리하고 실험
* One-step, N-step, AlphaZero 스타일 MCTS, DQN 계열까지 발전 과정을 버전별로 문서화
* Kaggle 제출 규격을 만족하는 자동 submission 생성 파이프라인을 유지·개선
* VSCode + Local Python + Kaggle Environments 조합에서 재현 가능한 테스트/디버깅 환경 제공
* 동일 경기 조합을 반복 실행해 승률을 비교할 수 있는 실험 스크립트/리포트 템플릿 제공

---

## 📁 **프로젝트 구조 (.py 중심)**

```
📦 2025_RL_Project
 ┣ 📂 kaggle_commit
 ┣ 📂 src
 │   ┣ 📂 ALphazeor     # AlphaZero 계열
 │   │   ┣ alphazero_mcts_basic.py, alphazero_mcts_balanced.py, alphazero_mcts_aggressive.py, alphazero_mcts_connectivity.py, alphazero_mcts_gap_defense.py
 │   │   ┗ submission_alphazero_edu_*.py
 │   ┣ 📂 DQN           # DQN 계열
 │   │   ┣ DQN.py, drl_double.py, drl_dqn.py, drl_dueling.py
 │   │   ┗ submission_DQN.py, submission_drl_dqn.py
 │   ┣ 📂 MTDF          # MTD(f)/Negamax 계열
 │   │   ┣ mtd_f.py, mtd_f_negamax.py, mtd_f_negamax_nf.py
 │   │   ┗ submission_mtd_f_negamax*.py
 │   ┣ 📂 Normal        # 기본 탐색/헬퍼
 │   │   ┣ one_step_lookahead.py, n_step_lookahead.py
 │   │   ┣ minimax_alpha_beta.py
 │   │   ┗ submission_minimax_alpha_beta.py, submission.py
 │   ┣ agents.txt       # 자동 제출 생성용 파일 목록
 │   ┗ make_submission.py  # submission 생성 스크립트 (경로 검색)
 ┗ 📄 README.md
```

---

## 🧠 **에이전트 종류 (.py 기준)**

- One-Step / N-Step: `src/Normal/one_step_lookahead.py`, `src/Normal/n_step_lookahead.py`
- Minimax/Alpha-Beta: `src/Normal/minimax_alpha_beta.py`
- DQN 계열: `src/DQN/DQN.py`, `src/DQN/drl_double.py`, `src/DQN/drl_dqn.py`, `src/DQN/drl_dueling.py`
- AlphaZero 계열: `src/ALphazeor/alphazero_mcts_basic.py`, `src/ALphazeor/alphazero_mcts_balanced.py`, `src/ALphazeor/alphazero_mcts_aggressive.py`, `src/ALphazeor/alphazero_mcts_connectivity.py`, `src/ALphazeor/alphazero_mcts_gap_defense.py`
- MTD(f) 계열: `src/MTDF/mtd_f*.py`
- 제출용 생성본: 각 폴더의 `submission_*.py` (make_submission.py로 자동 생성)

각 에이전트는 독립된 `.py` 파일로 관리되어 한 파일에 한 Agent가 들어있습니다.

### 🔗 DQN 학습 가중치
- [`src/DQN/drl_dqn_weights.pth`](파일 링크)[!https://github.com/gayeongge/2025_RL_Project/blob/main/src/DQN/drl_dqn_weights.pth]
- [`src/DQN/drl_double_weights.pth`](파일 링크)[!https://github.com/gayeongge/2025_RL_Project/blob/main/src/DQN/drl_double_weights.pth]
- [`src/DQN/drl_dueling_weights.pth`](파일 링크)[!https://github.com/gayeongge/2025_RL_Project/blob/main/src/DQN/drl_dueling_weights.pth]

---

## ▶️ **로컬에서 에이전트 테스트하기**

### 1. ConnectX 환경 생성 후 테스트

```python
from kaggle_environments import make
from one_step_lookahead import my_agent

env = make("connectx", debug=True)
env.run(["random", my_agent])
print(env.render(mode="ansi"))
```

### 2. 내 에이전트가 1P인지 2P인지 확인하려면

```
env.run([my_agent, "random"])  # my_agent = Player 1
env.run(["random", my_agent"]) # my_agent = Player 2
```

ANSI 렌더링에서 `1` 또는 `2`가 내 돌입니다.

---

## 🛠️ Submission 파일 자동 생성

`src/make_submission.py`가 `agents.txt`에 적힌 파일을 찾아 같은 폴더에 `submission_<원본>.py`를 생성합니다. 폴더 구조가 변해도 파일명을 검색해 경로를 해석합니다.

1) `src/agents.txt` 작성  
   - 파일명만: `mtd_f_negamax_nf.py` (src 이하에서 첫 매칭 사용)  
   - 상대경로: `MTDF/mtd_f_negamax_nf.py`  
   - 절대경로도 허용  
   - 확장자가 없으면 `.py`로 처리

2) 실행 (리포지토리 루트에서)  
   ```bash
   python src/make_submission.py
   ```

3) 결과/로그  
   - 각 소스 파일과 같은 폴더에 `submission_<원본>.py`가 만들어집니다.  
   - 후보가 여러 개면 첫 경로를 사용하며 경고를 출력합니다.  
   - agents.txt가 비었거나 파일을 찾지 못하면 에러 메시지를 출력합니다.  
   - 생성된 파일은 Kaggle 제출 규격의 `agent` 함수 래퍼가 자동으로 덧붙습니다.

---

## 🚀 **Kaggle 제출 방법**

1. Kaggle ConnectX 대회 페이지로 이동
2. “Submit Agent” 클릭
3. 생성된 `submission_*.py` 업로드
4. Submit!

제출 후:

* replay 화면 확인
* reward 로그 분석
* ERROR 로그가 있을 경우, replay JSON을 기반으로 디버깅 가능

---

## ❗ Troubleshooting & 성능 개선 가이드

### 📋 **알고리즘별 주요 이슈 & 해결책**

#### 🎯 **MTD(f) 계열 (`src/MTDF/`)**
| 문제 | 원인 | 해결책 | 해당 파일 |
|------|------|--------|----------|
| **부호 오류로 승부 결과 뒤바뀜** | player/root_player 비교 방식의 혼동 | Negamax 알고리즘 도입으로 항상 현재 플레이어 관점에서 점수 계산 | `mtdf_negamax_stable.py` |
| **시간 초과시 쓰레기 값 반환** | 미완료 탐색 결과를 그대로 사용 | `try-except TimeoutError`로 완료된 이전 깊이 결과 사용 (safe-fail) | `mtdf_negamax_stable.py` |
| **동일 효과 승리수 중 비효율적 선택** | 높이 고려 없이 첫 번째 발견된 승리수 선택 | `get_column_height()`로 낮은 높이 우선 선택하여 여유있는 승리 | `mtdf_negamax_strategic.py` |
| **자살수로 상대에게 승리 기회 제공** | 내 수 후 상대 즉시 승리 가능 여부 미검사 | `check_suicide_move()`로 위험한 수 사전 배제 | `mtdf_negamax_strategic.py` |

**성능 개선 결과**: negamax vs basic MTD(f) 승률 0% → 100%

#### 🔄 **Normal 에이전트 계열 (`src/Normal/`)**
| 문제 | 원인 | 해결책 | 해당 파일 |
|------|------|--------|----------|
| **1수 앞만 보여 함정에 빠짐** | 즉석 평가만으로 전략적 깊이 부재 | Minimax 알고리즘으로 3수 앞 미래 시나리오 탐색 | `minimax_basic_search.py` |
| **탐색 속도 너무 느림** | Alpha-beta 가지치기 없이 모든 노드 방문 | Alpha-beta 가지치기로 불필요 분기 조기 차단 | `minimax_optimized_search.py` |
| **동일 보드 상태 중복 계산** | 전이표(해시테이블) 없어 같은 상황 재계산 | ⚠️ **추후 개선 필요**: Zobrist 해싱 + 전이표 도입 고려 | - |

**알고리즘 진화**: O(열수) → O(분기^깊이) → O(효율적분기^깊이)

#### 🧠 **DQN 계열 (`src/DQN/`)**
| 문제 | 원인 | 해결책 | 해당 파일 |
|------|------|--------|----------|
| **Q값 폭주로 학습 불안정** | 동일 네트워크로 행동·가치 동시 추정 | 온라인/타깃 네트워크 분리한 Double DQN 도입 | `drl_double.py` |
| **열별 가치 구분 못함** | 단일 Q값으로 미묘한 차이 표현 한계 | Value/Advantage 분리한 Dueling 아키텍처 | `drl_dueling.py` |
| **Kaggle 점수 저조** | 기본 보상 체계로 장기 전략 학습 부족 | 보상 셰이핑 + 다중 TD 업데이트 + 유효 열 마스킹 | `drl_dqn_optimized.py` |
| **제출 후 추가 학습 필요** | 가중치 미내장 상태 | `EMBEDDED_STATE_DICT` 내장으로 학습 없이 제출 가능 | `drl_dqn.py` |


---

## ✨ **향후 계획 (Roadmap)**

* ⚙ AlphaZero MCTS 계열 고도화: Balanced/Aggressive/Connectivity/Gap 버전 통합 및 self-play 학습 검토
* ⚙ DQN 파이프라인 확장: 더블/듀얼링/가중치 내장 자동화와 학습 로그 시각화 추가
* ⚙ Kaggle 운영 지원: 자동 벤치마크 스크립트와 submission 빌더 개선, Leaderboard 성능 추적 자동화
