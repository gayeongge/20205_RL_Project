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

## ❗ Troubleshooting Highlights

1. **Stage 1 (alphazero_mcts_basic)** — `kaggle_environments.evaluate` 로그 파싱 오류로 승·패 수치가 음수/0으로 기록되고 “No valid games completed” 경고만 남음. Stage 2에서 휴리스틱/로그 루틴 재구성.
2. **Stage 2 (alphazero_mcts_balanced)** — 단순 휴리스틱 때문에 여러 방향 더블 쓰렛 대응이 느리고 중앙 장악이 약함. Stage 3에서 공격/수비 가중치 재조정 및 중앙 전략 추가.
3. **Stage 3 (alphazero_mcts_aggressive)** — 공격성은 좋아졌지만 gap 패턴 감지가 부족하고 방어를 희생. Stage 4/5에서 연결성·gap 감지를 추가하며 개선.

---

## ✨ **향후 계획 (Roadmap)**

* ⚙ AlphaZero MCTS 계열 고도화: Balanced/Aggressive/Connectivity/Gap 버전 통합 및 self-play 학습 검토
* ⚙ DQN 파이프라인 확장: 더블/듀얼링/가중치 내장 자동화와 학습 로그 시각화 추가
* ⚙ Kaggle 운영 지원: 자동 벤치마크 스크립트와 submission 빌더 개선, Leaderboard 성능 추적 자동화
