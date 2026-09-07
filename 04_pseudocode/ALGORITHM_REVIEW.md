# 동적 Specialist Pilot 의사코드 검토 (보조 과제)

## 검토 목표

이 역할은 **secondary pseudocode review**이다. 아래 알고리즘이 commit `67c1a8c`의 training → calibration → frozen evaluation 순서와 지연 보상 귀속을 정확히 나타내는지 검토한다. 수정 제안은 의사코드와 설명에 한정하며 소스 의미 변경을 제안하지 않는다.

## Algorithm 1 — 학습, 보정, 고정 평가

```text
INPUT:
  config C, seed s
  TRAIN_POLICIES = [fixed, adaptive, contextual]
  EVAL_ARMS      = [fixed, adaptive, contextual, switching]

BUILD independent SpecialistBank B
  for policy p in TRAIN_POLICIES:
    for UE u in 0 .. pool_users-1:
      create separate D3QN network, optimizer, replay, counters, RNGs
SNAPSHOT initial <- B.snapshot()

# 1) Independent training
for p in TRAIN_POLICIES:
  for episode e in 0 .. train_episodes-1:
    T <- MakeTrace(seed=s, phase="train", episode=e)
    epsilon <- linear_schedule(e, exploration_start, exploration_end)
    RunEpisode(B, T, arm=p, training=true, exploration=epsilon)

frozen <- B.snapshot()
assert each p has update_count > 0
assert each p's trainable_weight_digest differs from initial

# 2) Calibration-only heuristic (fixed/adaptive only)
records <- []
for p in [fixed, adaptive]:
  for episode e in 0 .. calibration_episodes-1:
    T <- MakeTrace(seed=s, phase="calibration", episode=e)
    records += RunEpisode(B, T, arm=p, training=false, exploration=0).tasks

for context bin b in [0, 1, 2]:
  F <- common-QoE samples whose origin bin=b and policy=fixed
  A <- common-QoE samples whose origin bin=b and policy=adaptive
  if |F| >= calibration_min_tasks and |A| >= calibration_min_tasks
     and mean(A) > mean(F) + 1e-12:
    mapping[b] <- adaptive
  else:
    mapping[b] <- fixed
# All bins may legitimately map to one policy.

# 3) Held-out, paired, frozen evaluation
for arm a in EVAL_ARMS:
  assert B.snapshot() == frozen
  for episode e in 0 .. eval_episodes-1:
    T <- MakeTrace(seed=s, phase="evaluation", episode=e)
    RunEpisode(B, T, arm=a, training=false, exploration=0, mapping=mapping)
  assert B.snapshot() == frozen

for episode e:
  assert trace_hash(e, fixed) == trace_hash(e, adaptive)
         == trace_hash(e, contextual) == trace_hash(e, switching)
REPORT four-arm mean metrics and limitations
```

중요한 해석은 다음과 같다. `MakeTrace`가 policy가 아닌 `(seed, phase, episode, stream)`에 의해 결정되므로 같은 evaluation episode의 네 arm이 외생 trace를 공유한다. evaluation과 calibration 모두 `training=false`라 replay 저장과 parameter update가 없다. 따라서 이 비교는 **학습된 policy/bank 선택의 frozen 평가**이지 continual online adaptation의 검증이 아니다.

## Algorithm 2 — 한 episode의 action-time transition과 delayed reward

```text
RunEpisode(B, trace T, arm, training, exploration, mapping):
  env <- TraceMEC(T)
  obs, recurrent <- env.reset_trace()
  context <- env.context()
  append context to every UE obs and recurrent row
  history[u] <- (history_steps-1 zero rows) + current recurrent[u]
  pending <- empty map keyed by (origin_step, UE)

  for step t in 0 .. total_steps-1:
    context_t <- env.context()
    bin_t <- ContextBin(context_t)
      # score = clip(0.60*users + 0.25*backlog + 0.15*pressure, 0, 1)
      # bin   = min(floor(3*score), 2)

    if arm == switching:
      desired <- mapping[bin_t]
      chosen <- DwellSelect(desired, selector_dwell)
    else:
      chosen <- arm

    active_count <- sum(T.active_mask[t])
    origin_weight <- EnergyWeight(active_count)
    window_t <- copy(history)
    actions[:] <- 0

    for UE u:
      if T.task_size[t,u] > 0:
        proposed <- B[chosen,u].choose(obs[u], window_t[u], exploration)
        actions[u] <- ApplyGate(chosen, proposed, obs[u])
      # no arrival keeps executable action 0

    next_obs, next_recurrent, done <- env.step(actions)
    append next context to next_obs and next_recurrent
    append each next_recurrent[u] to history[u]
    next_window <- copy(history)

    for UE u:
      tr <- copy(action-time fields:
        origin_expert=chosen,
        origin_energy_weight=origin_weight,
        state=obs[u], history_window=window_t[u], action=actions[u],
        next_state=next_obs[u], next_history_window=next_window[u],
        origin_done=done, context_bin=bin_t)

      if T.task_size[t,u] > 0:
        pending[(t,u)] <- tr
      else if training:
        B[chosen,u].store(tr, reward=0)       # idle/local-no-op transition

    for each ((origin,u), tr) in pending:
      if env.process_delay[origin,u] > 0:
        if training:
          r <- TrainingReward(
                 outcome at origin,
                 energy_weight=tr.origin_energy_weight)
          B[tr.origin_expert,u].store(tr, r) # reward follows origin expert
        record common QoE for the current arm
        # task records also supply calibration, not only evaluation
        delete pending[(origin,u)]

    if training and (t+1) mod learn_every == 0:
      B.learn(chosen)
    obs <- next_obs

  assert pending is empty
  assert classified task count == arrived task count
  assert common QoE recomputation matches environment QoE
```

### 비활성 사용자와 pending 해석

`active_mask[t,u] = false`는 그 slot의 **새 arrival가 없음**을 뜻한다. 이전 slot에 발생한 task의 pending entry와 실제 local/transmission/edge work를 취소한다는 뜻이 아니다. 그러므로 사용자가 이후 inactive여도 pending은 `process_delay[origin,u] > 0`으로 완료 분류될 때까지 남고, 그 사이 selector가 바뀌어도 reward는 저장된 `origin_expert`와 `origin_energy_weight`에 귀속된다. Training에서는 arrival이 없는 모든 UE의 action-0/reward-0 transition도 저장되며 마지막 drain slot의 terminal transition도 포함된다.

Replay 저장은 위 의사코드의 `if training` 조건에 한정된다. 본 pilot의 switching arm은 frozen evaluation에서만 실행되므로 전환 중 origin 정보와 common QoE는 측정·기록하지만 replay나 parameter를 갱신하지 않는다. `fixed`, `adaptive`, `contextual`의 학습용 energy weight는 동일한 adaptive 식이며, 두 specialist의 차이는 gate scaling이다. `contextual`은 gate를 적용하지 않는 별도 UE별 policy bank이다.

## 구조 비교에서 사용할 정확한 표현

| 구분 | 이 pilot | 혼동하면 안 되는 표현 |
|---|---|---|
| `fixed` / `adaptive` specialist | UE마다 network·optimizer·replay·RNG가 분리된 독립 policy bank | 하나의 old adapter가 mode flag만 실시간 변경 |
| `contextual` | context가 observation/history에 포함된 독립 no-gate single-policy 비교기 | switching selector 또는 세 번째 gate specialist |
| `switching` selector | calibration mean으로 미리 만든 3-bin mapping + minimum dwell heuristic | learned RL selector, meta-controller 학습 |
| evaluation | snapshot 고정, 네 arm에 episode별 동일 trace를 적용 | continual online adaptation, 평가 중 재학습 |

기존 adapter 계열 selector는 agent network·replay·observation/action space를 교체하지 않고 energy-weight/gate mode를 선택한다. 반면 이 dynamic pilot의 switching arm은 독립 학습된 `fixed`와 `adaptive` bank 중 하나의 action을 사용한다. 이 차이를 “더 우수한 구조”로 쓰지 말고 **상태 소유와 선택 단위가 다르다**고만 기술한다.

## 소스 대조 포인터

- 전체 phase와 4-arm pairing: [run.py — `execute`](../03_code_comments/source/dynamic_experiment/run.py#L178)
- episode transition/pending 처리: [run.py — `run_episode`](../03_code_comments/source/dynamic_experiment/run.py#L54)
- origin-weight training reward: [run.py — `training_reward`](../03_code_comments/source/dynamic_experiment/run.py#L37)
- 독립 bank 및 action-time record: [learner.py — `Transition`, `SpecialistBank`](../03_code_comments/source/dynamic_experiment/learner.py#L22)
- idle next-action 제한과 Double-DQN target: [learner.py — `_learn_slot`](../03_code_comments/source/dynamic_experiment/learner.py#L302)
- calibration과 dwell: [selector.py](../03_code_comments/source/dynamic_experiment/selector.py#L7)
- inactive arrival/persistent membership: [traces.py — `_activity_masks`, `make_trace`](../03_code_comments/source/dynamic_experiment/traces.py#L149)
- in-flight work 보존 계약: [environment.py — `TraceMEC`](../03_code_comments/source/dynamic_experiment/environment.py#L23)
- pilot 해석 한계: [dynamic_specialist_pilot.md](../03_code_comments/source/docs/dynamic_specialist_pilot.md)

## 검토 체크리스트

- [ ] training은 세 policy 각각 독립적으로 수행되고, calibration은 fixed/adaptive만 사용한다.
- [ ] evaluation 네 arm은 `fixed`, `adaptive`, `contextual`, `switching`이며 episode별 trace가 paired다.
- [ ] action-time history와 next-history가 transition에 복사되고 completion-order history를 만들지 않는다.
- [ ] delayed reward가 completion 당시 선택이 아닌 origin expert와 origin energy weight로 간다.
- [ ] idle action-0/reward-0 transition과 terminal drain transition이 포함된다.
- [ ] inactive는 새 arrival만 막고 persistent pending/in-flight service를 제거하지 않는다.
- [ ] selector는 calibration heuristic이며 learned RL selector라고 부르지 않는다.
- [ ] frozen evaluation을 continual online adaptation이라고 부르지 않는다.
- [ ] 결과 없이 QoE superiority, 수렴, 통계적 유의성을 주장하지 않는다.

## 회귀 명령

```bash
python -m unittest discover -s tests -p 'test_dynamic_*.py'
```

테스트 모듈에는 NumPy가 필요하며, real-network 경로에는 TensorFlow v1 compatibility runtime도 필요하다. pinned 환경이 아니면 import가 실패하거나 real-network test가 skip될 수 있다. 검토 보고에는 Python 경로, pass/fail/skip 수, 최초 오류를 그대로 적고, 이를 우회하려고 새 실험을 실행하거나 dependency를 변경하지 않는다.
