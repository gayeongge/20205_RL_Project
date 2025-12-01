# ConnectX DQN 변형 비교

## 한눈 요약
| 파일 | 개선 배경 | 핵심 특징 | 현재 한계 |
| --- | --- | --- | --- |
| `drl_dqn.py` | 노트북(`drl.ipynb`) 코드를 Kaggle 제출 가능한 단일 파일로 옮기려는 초기 작업 | 가중치 내장, 전역 스텝 기반 입실론, 고전 TD 타깃 사용으로 구현이 단순 (src/DQN/drl_dqn.py:33, src/DQN/drl_dqn.py:146, src/DQN/drl_dqn.py:151) | Double/dueling 기법 부재로 과대추정과 보상 희소성 문제에 취약, 안전 장치 최소 수준 |
| `drl_double.py` | 기본 DQN이 과대추정으로 수렴이 불안정했던 문제 개선 | 온라인/타깃 네트워크를 분리한 Double DQN 손실만 교체, 하이퍼파라미터·추론 경로는 그대로 유지 (src/DQN/drl_double.py:35, src/DQN/drl_double.py:141, src/DQN/drl_double.py:173) | 모델 표현력·보상 설계 모두 기본과 동일해 학습 속도·적응력은 크게 향상되지 않음 |
| `drl_dueling.py` | 열 단위 가치 평가가 약해 중앙 집중적 전략만 반복되던 현상 개선 | 듀얼링 CNN 헤드로 value/advantage를 분리, Double DQN 손실과 결합해 안정성과 표현력을 동시에 확보 (src/DQN/drl_dueling.py:117, src/DQN/drl_dueling.py:148) | 추가 FC 계층으로 리터럴 크기 및 추론 지연 증가, 보상 설계는 여전히 기본형 |
| `drl_dqn_optimized.py` + `report_dqn_v1.md` | Kaggle 점수가 낮아 보상·탐험·폴백을 전면 재설계한 실험 로그 | ConnectFourTrainer 보상 셰이핑, 다중 TD 업데이트, 유효 열 마스킹, 강화된 예외 처리 등 보고서의 모든 실험을 코드화 (src/DQN/drl_dqn_optimized.py:240, src/DQN/drl_dqn_optimized.py:269, src/DQN/drl_dqn_optimized.py:369, src/DQN/report_dqn_v1.md:6) | 구현 복잡도가 높아 유지보수가 어렵고, 가중치 미내장 상태에서는 여전히 추가 학습·내보내기 과정이 필요 |

## 파일별 상세

### `drl_dqn.py`
- **개선 배경.** 노트북 안의 레퍼런스 DQN을 Kaggle에 그대로 올릴 수 있도록, 학습 코드와 추론 코드를 하나의 파일로 통합한 초기 버전 (src/DQN/drl_dqn.py:1).
- **핵심 특징.**
  - `EMBEDDED_STATE_DICT`가 기본으로 채워져 있어 학습 없이 제출 가능 (src/DQN/drl_dqn.py:33, src/DQN/drl_dqn.py:292).
  - `epsilon_by_step`으로 스텝 기반 탐험 조절, `optimize_dqn`에서 `max`를 직접 사용해 단순한 TD 타깃 구현 (src/DQN/drl_dqn.py:146, src/DQN/drl_dqn.py:151).
  - NumPy `EmbeddedPolicy`가 PyTorch CNN과 동일한 구조를 재현해 Kaggle 환경에서 torch 없이 동작 (src/DQN/drl_dqn.py:234).
- **한계.** Double DQN/듀얼링·보상 셰이핑이 없어 과대추정과 탐험 부족이 그대로 발생하며, 폴백은 단순 중앙열 선택에 그쳐 안전성이 제한적 (src/DQN/drl_dqn.py:99).

### `drl_double.py`
- **개선 배경.** 기본 DQN이 동일 네트워크로 행동·가치를 동시에 추정하면서 학습 후반에 Q 값이 폭주하던 문제를 완화하려는 시도.
- **핵심 특징.**
  - `optimize_double_dqn`이 온라인 네트워크로 argmax를 고르고 타깃 네트워크로 값을 평가해 편향을 줄임 (src/DQN/drl_double.py:141).
  - 나머지 구성(하이퍼파라미터, 리플레이 버퍼, 추론 방식)은 기본 파일을 그대로 복제해 전환 비용이 없음 (src/DQN/drl_double.py:35, src/DQN/drl_double.py:173, src/DQN/drl_double.py:227).
- **한계.** 모델 구조와 보상 체계가 그대로라 학습 표현력 향상이 제한적이고, 여전히 reward sparsity 때문에 수렴이 느릴 수 있음.

### `drl_dueling.py`
- **개선 배경.** 특정 열에 대한 상대적 가치를 제대로 구분하지 못해 균일한 정책만 반복되던 현상을 해결하기 위해 value/advantage 분리 도입.
- **핵심 특징.**
  - `DuelingQNetwork`가 합성곱 특징을 value/adv 스트림으로 나눠 `v + a - mean(a)`로 재조합, 열마다 세밀한 Q를 제공 (src/DQN/drl_dueling.py:117).
  - Double DQN 손실을 그대로 사용해 안정성과 표현력 모두 확보 (src/DQN/drl_dueling.py:148).
  - NumPy 추론기 역시 value/adv 경로 전체를 복원해 Kaggle 추론과 일관성 유지 (src/DQN/drl_dueling.py:234).
- **한계.** 추가 fully connected 계층으로 내보내야 할 파라미터가 늘고, 보상 셰이핑이 없어 장기 전략 학습에는 여전히 시간이 걸림.

### `report_dqn_v1.md` & `drl_dqn_optimized.py`
- **개선 배경.** `drl_dqn.py` 제출 버전이 Kaggle에서 낮은 점수를 기록하자, fallback 안전성·보상 셰이핑·탐험 스케줄 등을 단계적으로 실험한 로그 (`report_dqn_v1.md`)와 그 결과물을 반영한 코드 (`drl_dqn_optimized.py`).
- **핵심 특징.**
  - `ConnectFourTrainer`가 Kaggle 환경을 감싸 승패 외에도 작은 living reward, 열 높이 보너스/패널티, 가득 찬 열 선택 시 -10을 부여 (src/DQN/drl_dqn_optimized.py:259, src/DQN/drl_dqn_optimized.py:269, src/DQN/drl_dqn_optimized.py:286).
  - 에피소드 기반 `epsilon_by_episode`, 워밍업 이후 스텝당 2회 TD 업데이트, 유효 열 마스킹으로 학습 효율 향상 (src/DQN/drl_dqn_optimized.py:35, src/DQN/drl_dqn_optimized.py:363, src/DQN/drl_dqn_optimized.py:369, src/DQN/drl_dqn_optimized.py:389).
  - `fallback_move`와 `my_agent`의 예외 처리로 어떤 상황에서도 안전한 수를 반환하도록 강화 (src/DQN/drl_dqn_optimized.py:142, src/DQN/drl_dqn_optimized.py:588).
- **한계.** 코드가 복잡하고 파라미터가 많아 재현·튜닝 비용이 커졌으며, 기본 파일처럼 가중치를 내장해 두지 않아 제출 전 추가 학습 및 literal 내보내기가 필요.

## 실무 관점 메모
1. 빠른 제출/재사용이 목적이면 `drl_dqn.py`가 가장 가볍지만, 성능 지붕은 낮다.
2. 안정성만 보완하려면 `drl_double.py`, 표현력까지 늘리고 싶다면 `drl_dueling.py`가 자연스러운 업그레이드 순서다.
3. Kaggle 점수 최적화가 목표라면 `drl_dqn_optimized.py`가 보고서의 모든 개선안을 포함하므로 가장 강력한 출발점이지만, 학습/제출 파이프라인을 별도로 관리해야 한다.
