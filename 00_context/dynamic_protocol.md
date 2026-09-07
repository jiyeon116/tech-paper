# Dynamic Specialist CPU Pilot

## 목적과 해석 범위

이 pilot은 고정 사용자 수에서 조합별 차이를 확인한 예비 실험을 확장해, 한 episode 안에서 사용자 활동량이 변할 때 독립적으로 학습된 두 전문 정책을 선택하는 구조의 실행 가능성을 검증한다. 사용자 pool은 150명이며 persistent schedule이 30, 90, 150명의 상대 부하 수준을 오간다. 이는 절대적인 sparse, general, dense 분류가 아니다.

각 seed는 100 slot(90 arrival slot과 10 drain slot)을 사용한다. 두 specialist와 context-conditioned single learner는 각각 30 episode를 학습한다. Specialist calibration은 정책별 6 episode, held-out evaluation은 fixed specialist, adaptive specialist, switching bank, context-conditioned single learner의 네 arm에 각각 12 episode를 사용한다. Evaluation에서는 학습을 중단하므로, 결과는 학습 중 실시간 parameter update의 효과가 아니라 학습된 정책 선택의 효과를 측정한다. 아직 측정 결과나 QoE 우위를 주장하지 않는다.

기본 pilot은 seed 42, 77, 123을 한 worker에서 순차 실행한다. 서버 root disk의 확인 당시 여유 공간이 약 16 GiB였으므로 실행기는 각 cell 시작 전과 실행 중에 5 GiB disk floor를 검사한다. 전체 budget은 21,600초, seed cell timeout은 7,200초이다. 실패하거나 중단된 `attempt_NNN`은 보존하며, 재실행은 새 attempt를 만든다.

`steps.jsonl`은 실제 active users와 선택 정책을 기록한다. `action_seconds`는 추론·gate 반복 처리 시간이고, `context_selector_seconds`는 context·selector·입력 준비 시간이다. 두 값의 합인 `controller_seconds`를 step 및 실제 결정 수로 나눈 값을 함께 보고한다. `delayed_across_switch`는 작업 발생부터 완료까지 한 번 이상 정책이 바뀐 작업 수로, 원래 정책으로 돌아온 경우도 포함한다.

Calibration에서 모든 bin이 같은 정책을 고를 수 있으며, 이를 억지로 변경하지 않는다. Smoke의 강제 전환 진단은 실제 bank 전환과 지연 보상 처리를 검사하는 별도 시험으로, 실험 성능 결과에 포함하지 않는다. 짧은 학습과 3개 seed는 실행 가능성 및 기술적 비교를 위한 조건이며, 수렴·통계적 유의성·QoE 우위의 근거로 사용하지 않는다.

## tmux 실행

아래 변수는 줄을 분리하지 말고 각 대입문 한 줄로 설정한다.

```bash
ROOT="$HOME/qeco-adapt2-experiments/<local-commit>"
cd "$ROOT"

PYTHON="$HOME/QECO-Adapt/env_channel/bin/python"
CONFIG="$ROOT/configs/dynamic_pilot.json"
OUT="$ROOT/results/dynamic_specialist_pilot"
SESSION="qeco_adapt2_dynamic_cpu"

scripts/launch_dynamic_cpu.sh \
  --python "$PYTHON" \
  --config "$CONFIG" \
  --output-dir "$OUT" \
  --session-name "$SESSION" \
  --seeds 42 77 123 \
  --budget-seconds 21600 \
  --cell-timeout-seconds 7200 \
  --min-free-gib 5
```

런처는 CPU 실행 환경을 고정한다: `CUDA_VISIBLE_DEVICES=-1`, intra-op 2, inter-op 1, OMP 2, OpenBLAS 1, per-agent session mode, unbuffered output. 같은 이름의 tmux session이 이미 있으면 기존 session에 간섭하지 않고 종료한다.

## 추적

```bash
SESSION="qeco_adapt2_dynamic_cpu"
ROOT="$HOME/qeco-adapt2-experiments/<local-commit>"
OUT="$ROOT/results/dynamic_specialist_pilot"

tmux ls
tmux capture-pane -pt "$SESSION":0 -S -80
pgrep -af 'dynamic_experiment.run|run_dynamic_campaign.py'
find "$OUT" -maxdepth 3 -name status.json -print
tail -F "$OUT"/seed_*/attempt_*/run.log
df -h "$OUT"
```

SSH 연결이 끊겨도 tmux의 campaign process는 계속 실행된다. `campaign_manifest.json`의 campaign `status`와 seed별 `state`를 확인한다. 성공한 cell은 `attempt_NNN/result/summary.json`의 `status`, resolved config hash, source identity, seed 및 revision과 `checksums.json`에 기록된 모든 산출물의 checksum이 모두 맞을 때만 재사용된다.

## 복구

동일한 실행 명령을 다시 입력한다. 검증된 완료 cell은 건너뛰고 실패, timeout, 비정상 summary 또는 checksum 불일치 cell은 기존 파일을 덮어쓰지 않은 채 다음 `attempt_NNN`에서 seed 시작점부터 재실행된다. Optimizer 중간 상태에서 이어졌다고 해석하지 않는다.
