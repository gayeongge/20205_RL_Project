# drl_double.py 문서

## 1. 이론 요약 (Double DQN)
- Double DQN은 타깃 계산 시 `a* = argmax_a Q_online(s', a)`를 먼저 구한 뒤, 타깃 네트워크의 값을 `Q_target(s', a*)`로 샘플링한다.
- 이 방법은 DQN이 가지는 overestimation 문제를 줄여 보다 안정적인 학습을 돕는다.
- 네트워크 구조는 기본 DQN과 동일한 CNN이지만, 타깃 계산만 다르다.

## 2. 코드 구성
- **전처리 및 버퍼**: `state_from_board`, `ReplayBuffer` 등은 drl_dqn.py와 동일하게 notebook 코드를 정리한 것이다.
- **`optimize_double_dqn`**: 온라인 네트워크로 argmax 행동을 구하고, 타깃 네트워크 값으로 TD 타깃을 만든다.
- **`train_double_dqn_agent`**: Kaggle 환경에서 학습하며 ε-greedy, 타깃 동기화 등은 동일하다.
- **`EmbeddedPolicy`**: DQN과 같은 CNN 구조를 NumPy로 구현해 Kaggle 제출 시 torch 없이 동작한다.
- **`my_agent`**: 내장된 literal이 있으면 학습 모델을 이용하고, 없으면 중앙 열 우선 휴리스틱으로 플레이한다.

## 3. 사용 방법
1. PyTorch가 있는 환경에서 `python drl_double.py` 실행 → literal 생성.
2. 출력 literal을 `EMBEDDED_STATE_DICT`에 복사.
3. Kaggle 제출 파일에 포함(단독 제출 or `make_submission.py`로 래핑).
