# drl_dqn.py 문서

## 1. 이론 요약 (DQN)
- DQN은 CNN 기반 Q-네트워크로 보드를 평가하며, `max_a Q(s', a)`를 사용하는 전통적인 TD 타깃을 쓴다.
- 경험 재현(Replay Buffer)과 타깃 네트워크를 활용해 학습을 안정화한다.
- ε-greedy 정책으로 행동을 샘플링하면서 버퍼가 채워진 뒤 배치 학습을 수행한다.

## 2. 코드 구성
- **전처리 유틸**: `state_from_board`, `find_playable_columns`, `_extract_board` 등이 Kaggle 관측 포맷(dict/객체) 모두를 처리한다.
- **ReplayBuffer / QNetwork**: notebook에서 쓰던 CNN 구조(1→32→64 conv, FC 384→7)를 그대로 사용한다.
- **`train_dqn_agent`**: Kaggle 환경을 직접 불러와 random 에이전트와 싸우며 학습한다. 일정 스텝마다 타깃 네트워크를 동기화한다.
- **`EmbeddedPolicy`**: torch 없이도 추론할 수 있도록 CNN을 NumPy 연산으로 재작성했다. Kaggle 제출 시 torch가 없어도 동작한다.
- **`my_agent`**: `EMBEDDED_STATE_DICT`가 비어 있으면 중앙 열 위주 휴리스틱으로, 채워져 있으면 `EmbeddedPolicy`로 행동한다.

## 3. 사용 방법
1. 로컬(PyTorch 설치)에서 `python drl_dqn.py`를 실행해 학습과 가중치 literal을 생성한다.
2. 출력된 내용을 `EMBEDDED_STATE_DICT` 변수에 붙여 넣는다.
3. `make_submission.py`에 파일명을 추가하거나 단독으로 Kaggle에 업로드한다.
