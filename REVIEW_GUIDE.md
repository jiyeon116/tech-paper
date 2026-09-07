# 코드 주석 검토 가이드 (보조 과제)

## 역할과 산출물

이 역할의 우선순위는 **secondary**이다. 그림·참고문헌 검토를 방해하지 않는 범위에서, 동봉된 commit `67c1a8c` 소스의 주석과 docstring만 검토한다. 결과는 아래 체크리스트와 “제안 위치 / 제안 문구 / 이유” 3열 표로 반환한다.

허용 범위는 주석·docstring의 정확성, 용어 통일, 오해 방지이다. 실행문, 함수 시그니처, 상수, 설정, 테스트, import, 공백 재포맷은 바꾸지 않는다. 즉 **comments-only이며 semantics changes는 금지**한다.

## 먼저 확인할 구현 계약

- [run.py — `run_episode`](source/dynamic_experiment/run.py#L54): 매 slot context를 계산하고, `switching` arm만 보정된 mapping과 dwell을 사용한다.
- [learner.py — `SpecialistBank`](source/dynamic_experiment/learner.py#L60): `fixed`, `adaptive`, `contextual`은 UE별 network·optimizer·replay·counter·RNG가 분리된 독립 bank다.
- [selector.py — `calibrate`, `DwellSelector`](source/dynamic_experiment/selector.py#L15): selector는 calibration 평균과 최소 표본 수로 만든 규칙 기반 mapping이다. 학습된 RL selector가 아니다.
- [environment.py — `TraceMEC`](source/dynamic_experiment/environment.py#L23): active mask는 새 task arrival만 제어한다. 이미 시작된 local/transmission/edge work는 사용자가 이후 inactive가 되어도 상속된 service discipline에서 계속 처리된다.
- [traces.py — `make_trace`](source/dynamic_experiment/traces.py#L184): `(seed, phase, episode, stream)`별 RNG로 policy-independent trace를 만든다.
- [config.py — `DynamicConfig`](source/dynamic_experiment/config.py#L8)와 [dynamic_pilot.json](source/configs/dynamic_pilot.json): 기본 pilot은 pool 150, load 30/90/150, persistent mode, 100 steps, train 30, calibration 6, evaluation 12 episode다.

## 반드시 보존할 의미

1. **Independent bank와 old adapter를 구분한다.** 이 pilot의 `fixed`/`adaptive` specialist는 각각 별도 학습 상태를 가진다. 기존 adapter 방식처럼 하나의 agent network·replay를 유지한 채 energy-weight/gate mode만 바꾸는 구조로 설명하지 않는다. `contextual`도 별도 학습된 no-gate 비교 arm이다.
2. **Heuristic selector와 learned RL을 구분한다.** context score를 3개 bin으로 나누고 calibration QoE 평균으로 `fixed` 또는 `adaptive`를 배정하며, `DwellSelector`가 최소 체류시간을 적용한다. selector 자체의 gradient/RL 학습은 없다.
3. **Action-time history를 보존한다.** transition은 action 시점의 `state`, `history_window`, `next_state`, `next_history_window`, `origin_done`을 복사해 둔다. completion 순서로 history를 재구성한다고 쓰지 않는다.
4. **보상은 origin에 귀속한다.** task 완료가 늦거나 중간에 bank가 바뀌어도 `origin_expert`의 replay에, action 시점의 `origin_energy_weight`로 계산한 training reward를 저장한다.
5. **Idle transition을 누락하지 않는다.** arrival이 없는 UE도 training 중 action 0(local/no-op), reward 0 transition을 매 slot 저장하며 terminal drain step도 포함한다. 다음 state의 task size가 0이면 bootstrap action도 0으로 제한한다.
6. **Pending은 inactive 이후에도 유지한다.** `(origin_step, ue)` pending entry는 완료 분류 전까지 삭제하지 않는다. 이후 active mask가 false가 되거나 selector가 다른 expert를 고르더라도 기존 in-flight work와 origin 귀속은 유지된다.
7. **Frozen paired evaluation을 정확히 쓴다.** 학습 후 snapshot을 고정하고 calibration을 거친 뒤 `fixed`, `adaptive`, `contextual`, `switching` 네 arm을 평가한다. 같은 evaluation episode index의 네 arm은 동일 trace hash를 사용하며 store/learn과 learner-state mutation이 없어야 한다.

위 3–6의 replay 처리는 `training=true` 경로의 계약이다. 본 pilot에서 `switching` arm은 frozen evaluation에만 등장하므로, 실제 평가 전환에서는 origin 정보를 기록하되 replay 저장·학습을 실행하지 않는다. 또한 세 학습 policy는 모두 같은 adaptive energy weight를 사용한다. `fixed`/`adaptive`는 gate scaling 방식의 차이이며, `contextual`은 같은 context 정의를 받는 UE별 no-gate single-policy bank이다. Arm 간 queue context 값까지 동일하다는 뜻은 아니다.

## 주석 검토 체크리스트

- [ ] “전문가 2개 + 독립 contextual 비교기 1개”와 “평가 arm 4개”를 혼동하지 않았다.
- [ ] `fixed`와 `adaptive`가 gate specialist 이름이며 절대적 sparse/dense 우열명이 아님을 유지했다.
- [ ] context가 normalized users, edge backlog, access pressure의 3-vector임을 정확히 설명했다.
- [ ] calibration 미충족 bin은 `fixed` default이고, 충분할 때만 adaptive 평균이 fixed보다 엄격히 클 경우 adaptive를 고른다고 썼다.
- [ ] 모든 bin이 한 policy로 귀결되는 degenerate mapping도 허용된다고 썼다.
- [ ] reward와 common evaluation QoE를 구분했다. 전자는 origin energy weight를 포함한 학습용이고 후자는 arm 간 공통 산식이다.
- [ ] idle transition, terminal drain, delayed completion, switch 후 origin 귀속을 빠뜨리지 않았다.
- [ ] persistent activity가 “inactive task 취소” 또는 “mobility”를 뜻한다고 쓰지 않았다.
- [ ] evaluation은 frozen selection 비교이며 continual online adaptation 실험이라고 쓰지 않았다.
- [ ] 결과가 없는 상태에서 QoE 우위, 수렴, 통계적 유의성, 일반적 우월성을 주장하지 않았다.

## 주요 제안 위치

| 위치 | 검토할 주석 의미 |
|---|---|
| [run.py — `training_reward`](source/dynamic_experiment/run.py#L37) | 완료 시점 값이 아니라 `origin_step` 결과와 저장된 origin energy weight를 사용한다. |
| [run.py — history/pending 생성](source/dynamic_experiment/run.py#L63) | recurrent window는 action 순서대로 갱신되고, 도착 task만 pending에 들어간다. |
| [run.py — completion 처리](source/dynamic_experiment/run.py#L130) | 현재 선택 policy가 아니라 `transition.origin_expert`에 delayed reward를 저장한다. |
| [learner.py — `Transition`](source/dynamic_experiment/learner.py#L22) | action-time snapshot이라는 표현을 유지한다. |
| [learner.py — `_learn_slot`](source/dynamic_experiment/learner.py#L302) | Double-DQN target과 idle next-action 0 제한을 과장 없이 설명한다. |
| [selector.py — `calibrate`](source/dynamic_experiment/selector.py#L15) | held-out evaluation을 보지 않는 calibration-only heuristic이다. |
| [traces.py — `_activity_masks`](source/dynamic_experiment/traces.py#L149) | persistent count 구간에서 membership을 유지하고 count 변경 시 가능한 사용자를 보존한다. |
| [run.py — frozen evaluation](source/dynamic_experiment/run.py#L241) | 네 arm 전후 snapshot 동일성과 episode별 paired trace hash를 검사한다. |

## 회귀 확인

원본 checkout에서 다음 명령을 사용한다. 동봉 snapshot에 주석만 반영한 경우에도 diff가 주석/docstring 외 실행 토큰을 바꾸지 않았는지 먼저 확인한다.

```bash
python3 -m unittest discover -s tests -p 'test_dynamic_*.py'
```

테스트 모듈에는 NumPy가 필요하고, 실제 `SpecialistBank` 및 `execute` 경로는 TensorFlow v1 compatibility runtime에도 의존한다. pinned 환경이 아닌 Python에서는 import가 실패하거나 일부 real-TensorFlow 검증이 skip될 수 있으므로, 그 경우 “테스트 통과”로 확대하지 말고 사용한 Python 경로, pass/fail/skip 수와 원문 오류를 함께 기록한다. 새 실험 실행은 이 역할 범위가 아니다.
