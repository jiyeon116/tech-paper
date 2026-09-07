# 그림 제작 작업지시서 (공저자 역할 2 우선)

작성일: 2026-09-06  
대상: 공동 연구자 중 그림 제작 담당자  
우선순위: **F01 → F02 → F03 → F04 → F05**

이 문서는 새 동적 specialist 실험을 중심으로 논문용 그림을 재작성하기 위한 운영 지시서다. 기존 PNG는 참고용이며 이 작업에서 덮어쓰거나 재생성하지 않는다. 그림 안의 모든 수치와 범주 표시는 아래 원자료에서 코드로 읽어 생성하고, Illustrator·PowerPoint·Inkscape 등에서 숫자를 손으로 입력하거나 수정하지 않는다.

## 1. 먼저 지켜야 할 증거 경계

서로 다른 세 버전을 한 그림이나 한 문장 안에서 같은 알고리즘으로 합치지 않는다.

| 버전 | 이 문서에서 쓸 이름 | 핵심 차이 | 허용되는 그림 용도 |
|---|---|---|---|
| Alpha | **Alpha (channel-free)** | AP-channel contention이 없는 초기 실험 | 역사적 배경만. channel-aware 결과와 절대값 비교 금지 |
| adapt1 | **QECO-ADAPT / adapt1 (channel-aware)** | 고정 사용자 수, channel-aware 환경, gate·energy-weight ablation | 공통 topology와 UE/server flow의 기반 |
| 기존 adapt2 | **legacy episode-variable adapt2** | 사용자 수가 episode마다 100–1500에서 바뀌며 공유 backbone의 reward/gate 구성을 episode 단위로 선택 | F04의 단일-seed 기술통계만 |
| 새 동적 실험 | **dynamic specialist bank** | 한 episode 안에서 active users가 30/90/150으로 변하고, 독립 학습된 fixed/adaptive specialist 중 하나를 calibration-frozen selector가 선택 | F01–F03 및 완료 후 F05 |

새 동적 실험의 selector는 RL agent나 meta-RL learner가 아니다. Calibration에서 context bin별 specialist label을 정한 뒤 evaluation 동안 mapping, network weight, replay state, update count를 모두 고정하는 규칙 기반 lookup/dwell selector다. 따라서 그림이나 캡션에 `reinforcement-learned selector`, `meta-RL`, `online adaptation`, `continual learning`을 쓰지 않는다.

해석 문구와 수치 사용 전 `../00_context/PROVENANCE_AND_LIMITS.md`를 확인한다. 해당 문서와 이 지시서가 다르면 provenance 문서의 더 보수적인 제한을 따른다.

## 2. 공통 제작 규격

### 2.1 지면·글자·제목

- 목표 배치 폭은 **1 column ≈ 148 mm**로 둔다. 현재 값은 provisional style이므로 최종 저널 template의 실제 column 폭을 확인한 뒤 비율을 유지해 조정한다.
- 최종 배치 크기에서 본문·축·범례·node label은 **10–11 pt**를 목표로 한다. PNG를 만들 때는 화면상 픽셀 크기가 아니라 148 mm로 축소된 뒤의 실제 글자 크기로 역산해 충분히 크게 렌더한다.
- 그림 내부에 논문 캡션과 같은 제목을 반복하지 않는다. Panel label `(a)`, `(b)`와 꼭 필요한 축·범례만 둔다. **캡션은 영어**로 별도 제공한다.
- 약어는 첫 등장 또는 범례에서 정의한다. `fixed`와 `adaptive`만 쓰지 말고 각각 `Fixed-gate specialist`, `Adaptive-gate specialist`로 한 번은 풀어 쓴다.

### 2.2 구조도 규칙

- 모든 labeled node는 둥근 모서리 사각형으로 만들고, 글자는 box의 시각적 정중앙에 둔다.
- connector는 수평·수직의 직교선으로 만들며 다른 node나 label을 통과하거나 서로 교차하지 않게 한다.
- 물리적 처리·전송 경로는 **solid line**, selector·calibration·weight freeze 같은 논리적 제어 관계는 **dotted line**으로 표시하고 두 선형을 설명하는 작은 legend를 넣는다.
- 색만으로 의미를 구분하지 않는다. line style, node label, marker를 함께 사용한다.
- 기존 `assets/topology_structure.png`, `assets/user_perspective_data_flow.png`, `assets/server_perspective_data_flow.png`는 topology와 용어 확인용 참고자료다. pixel trace나 부분 crop을 최종 벡터 그림으로 제출하지 않는다.

### 2.3 차트 규칙

- 시계열은 **raw episode/step trace를 가늘고 옅게** 남기고, 그 위에 명시한 smoothed curve를 굵게 올린다. 범례에는 알고리즘/정책 이름만 한 번 표시하며 `average`, `avg`, `mean`이라는 별도 series label이나 선 끝의 직접 label을 추가하지 않는다.
- smoothing이 있는 모든 캡션에는 방법과 window를 정확히 쓴다. 이 지시서에서 별도 지정하지 않은 smoothing은 만들지 않는다.
- 범례는 data를 가리지 않을 때만 plot 안에 둔다. 가리면 plot 밖 여백 또는 panel 사이의 공용 범례로 이동한다.
- 막대를 사용할 경우 막대 높이는 원자료에서 계산한 **산술평균**이어야 하며 y축 baseline은 0이다. 잘린 축으로 작은 차이를 과장하지 않는다. uncertainty가 없는 단일 seed 막대에는 error bar를 만들지 않는다.
- `completion_rate`는 원자료의 0–1 factor를 **100배** 해 percent로 그린다. y축 tick과 막대 값 label 모두 `%`를 붙인다.
- 단위가 다른 metric은 같은 y축에 겹치지 않는다. `active users`는 users, episode/step은 count, delay는 원자료의 slot 단위, energy와 QoE는 원자료 정의를 유지한다.
- 그림에 새 수치가 필요하면 변환 script에서 생성한 CSV에 그 수치와 계산식을 남긴다. 원본 JSONL/CSV를 수정하지 않는다.

### 2.4 각 그림의 납품 세트

각 F-ID마다 다음을 모두 반환한다.

1. `Fxx_<short_name>.svg` — vector master
2. `Fxx_<short_name>.pdf` — font embedding 또는 outline을 확인한 print version
3. `Fxx_<short_name>.png` — 최종 배치 148 mm 기준 최소 600 dpi preview
4. 편집 가능한 원본(`.svg`, `.ai`, `.pptx`, `.drawio` 중 하나)과 재생성 script
5. 그림에 실제 사용한 정규화 CSV와 그 CSV를 만든 script/command
6. 최종 영어 caption을 담은 UTF-8 text 또는 Markdown
7. 아래 acceptance check 결과

`editable/render_paper_figures_legacy.py`는 layout·색·기존 metric naming을 확인하는 **읽기 전용 참고**다. 특정 로컬 font 경로에 묶여 있어 그대로 실행 가능한 generator로 간주하지 않는다. 수정하거나 실행해 기존 PNG를 덮어쓰지 말고, 새 script는 package-relative input과 일반 배포 가능한 font fallback을 사용한다.

## 3. 그림별 작업지시

### F01. Dynamic specialist bank mechanism — 최우선

**목적**  
새 기여를 한 장에서 오해 없이 설명한다. 핵심은 하나의 learner가 실시간으로 변신하는 구조가 아니라, **독립적으로 학습된 두 specialist bank + calibration으로 고정된 selector**다.

**권장 구성**

- 왼쪽: 외생 입력인 `Dynamic MEC trace`와 환경의 현재 상태인 `Current MEC context`를 구분한다. Context에는 `Active-user fraction`, `Normalized edge backlog`, `Normalized channel-rate pressure`를 적는다. 마지막 항은 UE별 최선 AP-channel rate의 정규화 평균을 1에서 뺀 값이며 단순 occupancy나 packet-loss rate가 아니다. Queue context는 이전 action의 영향을 받는 내생 상태이다.
- 중앙 위: `Calibration-only evidence` → `Bin-to-specialist mapping` → `Frozen lookup + minimum dwell`을 dotted control path로 연결한다.
- 중앙 아래: selector가 `Fixed-gate specialist`와 `Adaptive-gate specialist` 중 하나를 선택하는 두 갈래 dotted path. 두 specialist node는 같은 크기로 병렬 배치하고 `independently trained`와 `frozen during evaluation`을 명시한다.
- 오른쪽: 선택된 specialist가 `Per-UE action proposal`을 만들고, 해당 specialist의 gate를 거쳐 `Local compute` 또는 `AP cluster → channel → edge queue`로 가는 physical path를 표시한다.
- 하단 feedback lane: completion/drop, delay, energy가 reward/transition으로 돌아가되 evaluation에서는 update가 없음을 `evaluation: no parameter/replay updates`로 표시한다.
- 그림 아래 작은 note: switching bank는 single policy보다 deployed parameter count가 크고, contextual no-gate single-policy bank는 별도 비교군이며 switching bank component가 아님을 적는다. 각 policy bank에는 UE별 network가 포함된다.
- solid physical path와 dotted logical/control path를 설명하는 legend를 반드시 둔다.

**포함하지 말 것**

- Alpha, adapt1, legacy adapt2의 성능 숫자
- brain icon, self-modifying loop, gradient가 selector로 들어가는 화살표
- `meta-learning`, `meta-RL`, `online adaptation`, `policy fusion` 표기

**근거 파일**

- `../00_context/PROVENANCE_AND_LIMITS.md`
- `assets/topology_structure.png`
- `assets/user_perspective_data_flow.png`
- `assets/server_perspective_data_flow.png`

**영어 캡션 초안**

> **Figure 1. Dynamic specialist-bank mechanism.** Independently trained fixed-gate and adaptive-gate specialists accept the same context schema; the selected specialist acts on the current environment state. All training policies use the same adaptive energy-weight formula. Calibration assigns one specialist label to each predeclared context bin, and the resulting lookup with a minimum dwell is frozen for held-out evaluation; neither the selector nor the specialist parameters are updated during evaluation. Solid connectors denote physical task-processing paths, whereas dotted connectors denote logical selection and freezing relations. The selector is a calibrated rule, not an RL or meta-RL policy.

**완료 판정**

- 두 specialist의 독립 학습과 evaluation freeze가 한눈에 구분된다.
- selector로 gradient/update가 들어가는 것으로 보이는 화살표가 없다.
- 모든 box와 connector가 겹치지 않고, line-style legend가 있다.

### F02. Train–calibration–evaluation paired-trace protocol

**목적**  
데이터 누수 없이 진행되는 세 phase와 paired comparison의 단위를 설명한다.

**권장 구성**

- 왼쪽에서 오른쪽으로 `Train → Freeze checkpoint → Calibration → Freeze mapping → Held-out evaluation`의 5단계 timeline을 만든다.
- Train block에는 seed당 `3 arms × 30 episodes`: fixed, adaptive, contextual을 적는다.
- Calibration block에는 `2 specialist arms × 6 episodes`, calibration-only QoE task records, three predeclared context bins, insufficient coverage 시 fixed default를 적는다. 모든 bin이 같은 specialist를 선택해도 허용됨을 note로 표시한다.
- Evaluation block에는 `4 arms × 12 held-out episodes`: fixed, adaptive, contextual, switching을 적는다.
- 같은 phase/episode index에서 모든 arm으로 갈라지는 공통 `trace_hash` lane을 그려 paired exogenous trace를 강조한다. Train, calibration, evaluation 사이에는 서로 다른 trace partition임을 separator로 보인다.
- 우측 하단에 `3 seeds: 42, 77, 123`, `100 slots/episode = 90 arrival + 10 drain`, `150 rollouts/seed; 450 total`을 compact note로 둔다.
- evaluation에서 network weights, replay, update counts가 전후 동일해야 한다는 check를 표시한다.

**근거 파일**

- `../00_context/PROVENANCE_AND_LIMITS.md`

**영어 캡션 초안**

> **Figure 2. Paired train–calibration–evaluation protocol for the full dynamic pilot.** For each seed (42, 77, and 123), three policies are trained for 30 episodes each, the two specialists are calibrated for 6 episodes each, and four frozen arms are evaluated on 12 held-out episodes each. Every episode contains 100 slots (90 arrival slots and 10 drain slots). Arms with the same phase and episode index share an identical exogenous trace, while the train, calibration, and evaluation partitions remain disjoint. The protocol comprises 150 rollouts per seed and 450 rollouts in total; outcome labels remain pending until all three seed cells and their checksums pass validation.

**완료 판정**

- paired 단위가 episode가 아니라 **같은 seed/phase/episode trace를 공유하는 arm 비교**임이 보인다.
- 학습·calibration·held-out evaluation의 trace partition이 섞여 보이지 않는다.
- caption과 그림에 seed, episode 수, slot 수, 총 rollout 수, pending 상태가 모두 있다.

### F03. Scale-check step trace: active users, selected expert, and queue context

**목적**  
일반 evaluation에서 실제 active-user 변화와 calibration 기반 specialist switching이 동작했음을 보여준다. 이는 **기능·규모 확인 그림이지 QoE 우위 그림이 아니다**.

**데이터 선택**

- `data/dynamic_scale/seed_42/attempt_001/result/steps.jsonl`
- `data/dynamic_scale/seed_42/attempt_001/result/episodes.jsonl`
- `phase == "evaluation"`, `arm == "switching"`, `episode == 0`의 정확히 100 step만 사용한다.
- `steps.jsonl`의 `active_users`, `policy`, `context[1]`을 각각 active users, selected expert, normalized edge backlog로 사용한다. `context[1]`은 raw queue length가 아니라 구현에서 0–1로 clip된 **normalized edge-backlog context**이므로 `Queue length`라고 부르지 않는다.
- `episodes.jsonl`은 같은 row의 `switch_count`, `trace_hash`, sample 확인에만 사용한다. QoE 값을 panel이나 caption의 결론으로 쓰지 않는다.

**권장 구성**

- 같은 x축 `Step (0–99)`을 공유하는 세 개의 수직 정렬 panel.
- (a) active users: 30/90/150 users의 step plot. y축 `Active users (users)`.
- (b) selected expert: `Adaptive-gate specialist`와 `Fixed-gate specialist`를 두 categorical band로 표시하고 실제 전환점에 얇은 vertical marker를 둔다. mapping label은 JSONL의 `policy` 값에서 읽는다.
- (c) queue context: `context[1]` raw trace, y축 `Normalized edge backlog (0–1)`. smoothing 없음.
- 세 panel의 background regime shading은 선택 사항이나, 쓰면 active-user 값에서 직접 생성하고 범례로 정의한다.
- figure note에 `scale check: train 2, calibration 1, evaluation 1 episode per arm; seed 42`와 `not confirmatory performance evidence`를 넣는다. 이번 기능 시험을 확증적 우위 검증으로 사용하지 않는다.

**영어 캡션 초안**

> **Figure 3. Functional step trace from the seed-42 scale check.** Active users, the calibration-selected specialist, and the normalized edge-backlog context are shown for the 100 slots of switching-arm evaluation episode 0. The workload moves among 30, 90, and 150 active users, and the plotted expert labels are read directly from the recorded policy field. No smoothing is applied. This scale check used 2 training episodes, 1 calibration episode, and 1 evaluation episode per arm; it demonstrates executable switching and queue-context recording, not QoE superiority or statistical evidence.

**완료 판정**

- source filter가 caption 또는 companion CSV metadata에 그대로 남아 있다.
- policy band가 원자료와 step-by-step 일치하고 전환 수가 `episodes.jsonl`의 `switch_count`와 일치한다.
- queue panel이 0–1 normalized context로 표시되고 raw queue로 오인될 표현이 없다.
- QoE 우위 또는 일반화 문구가 없다.

### F04. Legacy episode-variable 100–1500 comparison (single seed)

**목적**  
기존 adapt2가 episode 단위 사용자 변화에서 보인 단일-seed 기술통계를 요약한다. 새 dynamic specialist 결과와 같은 panel이나 같은 평균에 결합하지 않는다.

**근거 파일**

- `data/legacy_random/comparison_timeseries.csv`
- `data/legacy_random/comparison_by_regime.csv`
- `data/legacy_random/active_users_timeseries.csv`
- `../00_context/PROVENANCE_AND_LIMITS.md`

**권장 구성**

- 2×2 panel을 기본으로 한다.
  - (a) `Active users (users)` vs `Episode (1–500)`, raw step line, smoothing 없음.
  - (b) QoE vs episode: 네 legacy algorithm의 raw trace(얇고 낮은 alpha) + **25-episode trailing moving average**(굵은 선).
  - (c) Energy vs episode: 같은 raw + 25-episode trailing moving average.
  - (d) Completion vs episode: CSV의 factor를 100배한 percent, 같은 raw + 25-episode trailing moving average.
- 공용 범례는 `QECO`, `QECO-ADAPT general`, `QECO adaptive gate`, `legacy episode-variable adapt2`로 표기한다. 새 `dynamic specialist bank`라는 이름을 이 그림에 쓰지 않는다.
- 148 mm에서 네 시계열이 식별되지 않으면 F04a(active users)와 F04b(QoE/energy/completion)를 두 파일로 나누되 동일 F04 family로 납품한다.
- `comparison_by_regime.csv`는 regime별 episode count와 요약표/보조 grouped bar를 만들 때만 사용한다. 막대를 만들면 mean은 원자료 행을 그대로 사용하고 baseline 0을 유지한다. Completion은 100배와 `%` label을 적용한다.

**영어 캡션 초안**

> **Figure 4. Legacy episode-variable comparison under a shared 100–1500-user schedule.** The four legacy channel-aware algorithms share the same 500-episode active-user schedule sampled from 100, 300, 500, 800, 1,200, and 1,500 users. QoE, energy, and completion show each raw episode value as a faint line and a 25-episode trailing moving average as a solid line; completion factors are multiplied by 100 and reported as percentages. Results are descriptive statistics from seed 42 only and are not combined with the new within-episode dynamic specialist pilot.

**완료 판정**

- 500개 episode와 여섯 사용자 수가 companion CSV 검사로 확인된다.
- 네 algorithm의 episode index가 완전히 정렬되고, active-user schedule은 한 번만 그린다.
- smoothing window가 25이며 raw trace가 숨겨지지 않는다.
- single-seed 및 legacy/new 분리 문구가 caption에 있다.

### F05. Full dynamic pilot paired-seed outcome — 결과 완료 후 제작

**상태**: **PENDING — 세 seed의 검증된 full-pilot 결과가 모두 준비되기 전에는 수치 panel을 만들지 않는다.** 빈 막대, 예상값, scale-check 값, legacy 값을 placeholder로 넣지 않는다.

**목적**  
seed 42/77/123의 held-out evaluation에서 fixed, adaptive, contextual, switching 네 arm을 paired seed 단위로 비교한다.

**입력 승인 조건**

- seed 42, 77, 123 각각의 `status.json`과 `result/summary.json`이 complete여야 한다.
- 각 seed의 source identity, config hash, expected/completed rollout 수가 일치해야 한다.
- 각 `checksums.json`을 재계산해 모든 결과 파일이 일치해야 한다.
- 같은 seed와 evaluation episode index의 네 arm `trace_hash`가 동일해야 한다.
- evaluation 전후 frozen learner snapshot이 동일해야 한다.
- 위 조건은 `../00_context/PROVENANCE_AND_LIMITS.md`의 최신 판정과 함께 확인한다.

**승인 후 권장 구성**

- metric별 small multiple: QoE, delay (slots), energy, completion (%). 필요하면 switching cost는 별도 panel로 `switch count`와 `controller seconds per step`을 둔다.
- 각 metric에는 네 arm의 seed별 값을 점으로 표시하고 같은 seed끼리 가는 paired line을 연결한다. 중앙 요약을 막대로 추가할 경우 막대는 세 seed mean, y축 baseline 0, seed point overlay를 유지한다.
- Completion은 factor×100으로 변환하고 모든 tick/value label에 `%`를 붙인다.
- statistical interval을 추가할 경우 seed가 3개뿐임을 밝히고 계산법을 companion CSV와 caption에 적는다. episode를 독립 표본으로 취급한 error bar는 금지한다.
- 결과가 switching 우위를 보이지 않거나 calibration mapping이 한 specialist로 퇴화해도 그대로 표시한다. `best`, `improved`, `robust` 같은 outcome label은 사전 정의된 계산으로 확인되기 전 사용하지 않는다.

**영어 캡션 템플릿 — 수치와 판정은 자동 생성으로 채울 것**

> **Figure 5. Paired held-out outcomes of the full dynamic specialist pilot (results pending).** Fixed-gate, adaptive-gate, context-conditioned no-gate single-policy bank, and switching-bank arms are evaluated on the same 12 held-out 100-slot traces within each of seeds 42, 77, and 123 (36 paired evaluation traces per arm). Points denote seed-level means, and lines connect arms evaluated under the same seed; any bars denote the arithmetic mean across the three seeds and begin at zero. Completion factors are multiplied by 100 and reported as percentages. **[Replace “results pending” and add only checksum-verified numerical findings after all three seed cells complete.]**

**완료 판정**

- 세 seed point가 모두 보이고 paired connection이 seed별로 정확하다.
- sample size를 `3 independently retrained seeds`, `12 held-out episodes per seed and arm`으로 구분한다.
- episode를 독립 seed처럼 사용한 CI나 유의성 표기가 없다.
- 캡션의 pending label은 결과 승인 전 유지되고 승인 후 자동 생성된 판정으로만 교체된다.

## 4. 재생성·검수 기록 형식

각 그림 폴더에 다음 내용을 담은 `CHECKS.md`를 둔다.

```text
Figure ID:
Input files and SHA-256:
Row/filter contract:
Derived CSV and generation command:
Smoothing/aggregation:
Seed(s) and sample unit:
Output dimensions and embedded-font check:
SVG/PDF/PNG visual check:
No-overlap / no-crossing check (diagrams only):
Raw-vs-smoothed and legend-occlusion check (charts only):
Caption status: draft | pending | approved
Reviewer/date:
```

마지막 검수에서는 SVG/PDF를 148 mm 폭으로 실제 렌더링해 10–11 pt 글자 가독성, 중앙 정렬, connector 교차, 범례의 data 가림, clipping을 확인한다. PNG는 최종 산출물의 raster preview일 뿐 수치나 레이아웃의 편집 원본으로 사용하지 않는다.
