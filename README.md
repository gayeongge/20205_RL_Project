# 📘 **ConnectX Reinforcement Learning Agents — Project README**

> Kaggle ConnectX 환경을 기반으로 여러 강화학습 기법(One-step, N-step, DQN 등)을 구현하고 비교하여, 가장 우수한 전략을 찾는 프로젝트입니다.
> 프로젝트 인원 : 이가영, 이현지

---

## ✨ **프로젝트 목표**

이 프로젝트는 다음을 목표로 합니다:

* ConnectX 게임 환경에서 동작하는 다양한 RL Agents 개발
* One-step Heuristic → N-step Lookahead → DQN → 강화형 Hybrid Agent까지 확장
* Kaggle 제출 형식에 맞춘 자동 submission 파일 생성 시스템 구축
* VSCode + Local Python 환경에서도 안정적으로 테스트 가능하도록 구성
* 여러 Agent 버전 간 A/B 테스트를 쉽게 수행할 수 있는 구조 제공

---

## 📁 **프로젝트 구조**

```
📦 20205_RL_Project
 ┣ 📂 src
 │   ┣ one_step_lookahead.py        # One-Step 에이전트 코드
 │   ┣ minimax_agent.py             # Minimax 또는 Hybrid Agent
 │   ┣ agents.txt                   # 자동 제출 생성용 파일 목록
 │   ┣ make_submission.py           # submission 생성 스크립트
 ┗ 📄 README.md                     # 프로젝트 문서
```

---

## 🧠 **에이전트 종류**

### 1. **One-Step Lookahead (완료)**

* heuristic 기반의 가장 단순한 평가 방식
* 빠르고 안정적이며 baseline으로 활용 가능
* 완전한 리팩터링 & Kaggle 호환 완료


각 에이전트는 독립된 `.py` 파일로 관리되어
한 파일에 한 Agent가 들어있습니다.

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

## ⚙️ **Submission 파일 자동 생성 시스템**

### 1. 에이전트 목록을 작성

`agents.txt` 사용

#### ✔ agents.txt 사용

```
one_step_lookahead.py
n_step_agent.py
dqn_agent.py
```

---

### 2. submission 파일 생성

VSCode 터미널에서:

```
cd src
python make_submission.py
```

그러면 다음이 자동 생성됩니다:

```
submission_one_step_lookahead.py
submission_n_step_agent.py
submission_dqn_agent.py
```

각 파일은 Kaggle 제출 규격에 맞는:

```
def agent(observation, configuration):
    return my_agent(observation, configuration)
```

래퍼가 자동으로 포함됩니다.

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

## 💡 **Troubleshooting**

정리중 ... 

---

## ✨ **향후 계획 (Roadmap)**

* ✔ One-step 리팩터링 / 개선 / Kaggle-safe 완료
* ⏳ N-step Lookahead
* ⏳ DQN Training Loop 구축
* ⏳ Leaderboard 기반 자동 제출 파이프라인