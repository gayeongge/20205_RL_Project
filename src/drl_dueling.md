# drl_dueling.py 문서

## 1. 이론 요약 (Dueling Double DQN)
- Dueling DQN은 상태 가치 `V(s)`와 행동 별 이득 `A(s,a)`를 분리해 합치는 구조다. 보드 상태 자체의 가치를 추정하면서도 행동 간 차이를 학습해, 관찰별 데이터가 부족해도 안정적인 값을 추정한다.
- 본 구현은 Dueling 구조에 Double DQN 업데이트를 결합하여 overestimation을 억제한다.

## 2. 코드 구성
- **DuelingQNetwork**: CNN feature extractor 뒤에 Value/Advantage 두 개의 MLP를 배치하고, `Q = V + (A - mean(A))` 공식을 사용한다.
- **`optimize_double_dueling`**: 온라인 네트워크로 argmax 행동을 고르고, 타깃 네트워크 값으로 TD 타깃을 계산한다.
- **`EmbeddedPolicy`**: Value/Adv 스트림을 모두 NumPy로 재현해 Kaggle 환경(무-Torch)에서도 추론 가능하다.
- **`my_agent`**: literal 가중치가 없으면 휴리스틱으로 플레이, 있으면 Dueling 정책을 사용한다.

## 3. 사용 방법
1. torch 환경에서 `python drl_dueling.py` → literal 출력.
2. literal을 `EMBEDDED_STATE_DICT`에 붙여 넣는다.
3. 단일 파일 그대로 Kaggle에 제출하거나 `make_submission.py`로 래핑한다.
