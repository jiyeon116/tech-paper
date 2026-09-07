# 사용자 밀도 적응형 정책 선택 QECO-ADAPT2의 채널 인지형 MEC 오프로딩 성능 분석: 확장 연구 초안

> **문서 성격:** 본 문서는 QECO-ADAPT 확장본(`docs/adapt1/paper_draft_channel_aware_qeco_adapt_kr_extended.md`, 이하 "adapt1 확장본")의 후속 실험군 QECO-ADAPT2의 설계·실험·결과를 정리한 분량 제한 없는 확장 연구 초안이다. 시스템 모델(AP-channel access resolution), D3QN/LSTM backbone, 공통 평가 QoE 정의(수식 (12)–(17))와 참고문헌 [1]–[41]은 adapt1 확장본을 그대로 승계하며 본문에서 같은 번호로 인용한다.
>
> **명칭 규칙 (2026-08-20 통일):** `qeco-adapt-general`(**general**) = adaptive energy weight + fixed gate(σ=1) = adapt1의 "adaptive weight/fixed gate"; `qeco-adaptive-gate`(**adaptive gate**) = adaptive energy weight + adaptive gate(σ=g(N)) = adapt1의 "Full"(`qeco-adapt`, `qeco-adapt-full`). 네 이름은 registry에 등록된 동일 구현의 별칭이다.
>
> **증거 경계:** 본 초안의 모든 정량 결과는 2026-08-20~22 bear 서버(dmlab-bear0, 16-thread CPU)에서 `CUDA_VISIBLE_DEVICES=-1`, `QECO_AGENT_SESSION_MODE=per-agent`, `TF_NUM_INTRAOP_THREADS=2`, `TF_NUM_INTEROP_THREADS=1`, seed 42, 500 episode로 생성한 단일 seed run이다. adapt1 확장본의 legacy 결과(대부분 `shared` 실행 배치)와 절대값을 직접 비교하거나 결합하지 않는다. 단일 seed이므로 inferential significance를 주장하지 않으며, 해석은 adapt1 확장본 §5.3의 통계 단위 규칙을 따른다.

나현우*, 이창우**

*국민대학교 소프트웨어학부, nahw0861@kookmin.ac.kr  
**국민대학교 소프트웨어학부, leecw05@kookmin.ac.kr (교신저자)

## 초록

QECO-ADAPT 계열의 채널 인지형 MEC 오프로딩 실험은 고정 사용자 수 조건에서 adaptive energy weight와 fixed gate를 결합한 구성(general)이 가장 강한 관측 구성임을 보였고, load-conditioned gate(adaptive gate)는 부하 증가에 따라 fixed gate에 수렴했다. 본 연구는 이 관측을 알고리즘 구조로 옮긴 QECO-ADAPT2를 제안한다. QECO-ADAPT2는 매 episode 시작 시 활성 사용자 밀도(edge당 사용자 수)를 sparse, general, dense 영역으로 판정하고, 영역별로 등록된 QECO-family 하위 정책의 (energy weight, gate) 구성으로 전환한다. 모든 하위 정책은 동일한 D3QN/LSTM network와 replay memory를 공유하므로 전환은 실행 전 gate와 reward weighting만 바꾼다. 사용자 밀도가 episode마다 {100–1500}에서 무작위로 변하는 랜덤 밀도 환경을 정의하고, 고정 밀도 실험(Phase A)으로 sparse 영역 정책을 데이터로 확정한 뒤(전 지표에서 general 최고), 랜덤 밀도 최종 실험(Phase B)에서 QECO-ADAPT2가 최고 단일 정책과 사실상 동률(전체 QoE −0.09%)로 영역 전환을 손해 없이 수행하며, adaptive gate 단독(+16.7%)과 QECO(+120.4%)를 크게 상회함을 보였다. 반면 dense 영역에서 adaptive gate로 전환하는 이득은 관측되지 않았으므로, 현 증거는 QECO-ADAPT2의 성능 우위가 아니라 밀도 영역별 최적 정책을 자동 선택하는 구조의 무손실성을 지지하는 것으로 제한 해석한다.

**주제어:** Mobile Edge Computing, Computation Offloading, QECO, QECO-ADAPT2, User Density, Adaptive Policy Selection, Deep Reinforcement Learning

## 1. 서론

adapt1 확장본은 QECO [7]의 local/edge 행동 공간을 유지한 채 AP cluster와 AP-channel resource를 환경 내부에서 해소하는 channel-aware MDP에서, access-aware gate와 load-adaptive energy weight의 조합을 분리 평가했다. 그 결론은 두 가지였다. 첫째, 시험한 모든 고정 부하(user=300–1500)에서 adaptive weight/fixed gate(통일 명칭: general)가 가장 강한 observed configuration이었다. 둘째, load-conditioned gate(통일 명칭: adaptive gate)는 부하 증가에 따라 fixed gate에 수렴하며($g(N_U)\to1$), 고정 부하 평균만으로는 그 필요성을 입증할 수 없어 부하가 변하는 조건의 검증(dynamic-load evaluation)이 남은 과제로 제시되었다(adapt1 확장본 §7.4).

본 연구는 이 두 결론을 하나의 후속 설계로 통합한다. 단일 고정 구성을 고르는 대신, **사용자 밀도를 계산해 밀도 영역별로 하위 정책을 선택하는 QECO-ADAPT2**를 정의하고, 밀도가 episode마다 무작위로 변하는 실험 환경에서 이를 평가한다. 기여는 다음과 같다.

1. QECO의 행동 규약(`0 = local`, `1..N_EDGE = edge`)과 D3QN/LSTM backbone을 유지한 채, 활성 사용자 밀도(users per edge)를 sparse/general/dense 영역으로 판정하여 영역별 (energy weight mode, gate mode)를 전환하는 밀도 적응형 선택기를 구성한다. network parameter는 추가되지 않는다.
2. episode별 활성 사용자 수를 무작위 표본화하는 랜덤 밀도 실험 환경을 정의한다. 스케줄은 run seed로 시드된 전용 RNG에서 생성되어 같은 seed의 모든 알고리즘이 동일한 밀도 시계열을 공유하며, 기존 고정 밀도 실험의 재현성은 바이트 수준에서 보존된다.
3. 고정 밀도 단계(Phase A)에서 sparse 영역 정책을 ablation으로 확정하고 선택기의 무손실성(고정 밀도에서 하위 정책과의 시계열 완전 일치)을 검증한 뒤, 랜덤 밀도 최종 단계(Phase B)에서 전체/영역별/전환 관점의 판정 기준을 적용한다.
4. dense 영역에서 adaptive gate 전환의 이득이 관측되지 않는다는 부정적 결과를 포함해, 현 단일 seed 증거가 지지하는 결론과 지지하지 않는 결론을 구분한다.

## 2. 배경: adapt1 결과와 명칭 통일

adapt1 확장본의 ablation(§6.3–6.4)에서 general은 user=300/500/800/1200/1500의 전 부하에서 QoE·energy 최고 또는 최고 근접이었고, adaptive gate(Full)와의 QoE 차이는 부하 증가에 따라 −1.6498(300)에서 −0.0035(1500)로 수렴했다. 이 수렴은 $\sigma=g(N_U)\to1$의 수학적 예측과 일치하지만(§7.2), adaptive gate가 필요한 조건 — 부하가 변하는 환경 — 은 고정 부하 실험으로 검증되지 않았다.

2026-08-20 명칭 통일에 따라 본 초안은 다음 canonical 명칭을 사용한다.

| Canonical name | Label | 구성 | adapt1 명칭 |
|---|---|---|---|
| `qeco` | QECO | weight 1.0, gate 없음 | 동일 |
| `qeco-adapt-general` | general | adaptive $w_E(N)$, fixed gate $\sigma=1$ | adaptive weight/fixed gate |
| `qeco-adaptive-gate` | adaptive gate | adaptive $w_E(N)$, adaptive gate $\sigma=g(N)$ | Full, QECO-ADAPT full |
| `qeco-adapt2` | QECO-ADAPT2 | 영역별 선택 | 신규 |

## 3. QECO-ADAPT2 설계

### 3.1 밀도 지표와 영역

Episode $k$의 활성 사용자 수를 $N^{\mathrm{act}}_k$라 할 때 밀도는 $d_k=N^{\mathrm{act}}_k/N_E$ (users per edge)로 정의한다. 영역 판정은 두 임계값 $d_{\mathrm{sp}}$, $d_{\mathrm{dn}}$을 사용한다.

$$
\mathrm{regime}(d_k)=
\begin{cases}
\text{sparse}, & d_k < d_{\mathrm{sp}}\\
\text{general}, & d_{\mathrm{sp}} \le d_k < d_{\mathrm{dn}}\\
\text{dense}, & d_k \ge d_{\mathrm{dn}}
\end{cases}
\tag{A1}
$$

본 실험은 $d_{\mathrm{sp}}=50$, $d_{\mathrm{dn}}=350$ (users per edge)을 사용한다. $N_E=3$에서 시험 사용자 수 집합 {100, 300, 500, 800, 1200, 1500}은 각각 sparse(100), general(300–800), dense(1200, 1500)로 분류된다. 임계값과 영역별 정책은 환경변수로 조정 가능한 설정값이며 closed-form optimum이 아니다.

### 3.2 영역별 정책과 선택 절차

영역별 하위 정책 매핑은 다음과 같다(기본값).

| 영역 | 정책 | (weight, gate) | 선정 근거 |
|---|---|---|---|
| sparse | `qeco-adapt-general` | (adaptive, fixed) | Phase A-1 ablation으로 확정 (§6.1) |
| general | `qeco-adapt-general` | (adaptive, fixed) | adapt1 §6.3 전 부하 최강 구성 |
| dense | `qeco-adaptive-gate` | (adaptive, adaptive) | 설계 의도: 초고부하에서 load-conditioned 완화 gate 사용. 검증 결과는 §7.5 |

Episode 시작 시 선택 절차는 다음과 같다.

```text
N_act  = density_schedule.active_count(episode)
regime = classify(N_act / N_E)                         # (A1)
policy = REGIME_POLICY[regime]
(energy_weight_mode, gate_mode) = modes_of(policy)
w_E    = w_E(N_act);  sigma = 1 (fixed) 또는 g(N_act) (adaptive)
```

adapt1 확장본 수식 (4)–(5)의 $g(\cdot)$, $w_E(\cdot)$를 그대로 사용하되 입력이 구성된 사용자 수 $N_U$가 아니라 **해당 episode의 활성 사용자 수 $N^{\mathrm{act}}_k$** 라는 점만 다르다. 고정 밀도에서는 $N^{\mathrm{act}}_k=N_U$이므로 두 정의가 일치한다. 시험 밀도별 값은 다음과 같다.

| $N^{\mathrm{act}}$ | 100 | 300 | 500 | 800 | 1200 | 1500 |
|---|---:|---:|---:|---:|---:|---:|
| $g(N^{\mathrm{act}})$ | 0.2405 | 0.4872 | 0.6129 | 0.7170 | 0.7917 | 0.8261 |
| $w_E(N^{\mathrm{act}})$ | 1.2940 | 1.3788 | 1.4185 | 1.4499 | 1.4717 | 1.4815 |

### 3.3 공유 backbone과 무손실성

QECO-ADAPT2는 새로운 network variant가 아니다. 모든 하위 정책은 하나의 per-UE D3QN/LSTM parameter set과 replay memory를 공유하며, 영역 전환은 (i) 실행 전 gate의 존재·scale과 (ii) 학습 reward의 energy weight만 바꾼다. 따라서 고정 밀도 run에서는 전 episode가 한 영역에 속하므로 QECO-ADAPT2는 해당 영역의 하위 정책과 **정확히 동일한 알고리즘으로 붕괴**해야 하며, 이 성질은 §6.2의 A-0 검증으로 확인한다. 이는 선택기 구현이 하위 정책 대비 추가 비용이나 잡음을 넣지 않음을 보증하는 구조적 안전장치다.

제안 action을 실행 전에 검사·치환하는 gate의 계열 분류(invalid action masking [24], shielding [25], safe RL 분류 [26], [27])와 reward 변경의 policy invariance 한계 [23]는 adapt1 확장본 §4.3–4.4의 논의를 그대로 승계한다. 환경 변화에 적은 갱신으로 적응하는 meta-RL [40], [41]은 본 선택기의 rule-based 전환을 학습형으로 확장할 때의 참고 구조다.

## 4. 랜덤 밀도 실험 환경

### 4.1 밀도 스케줄

랜덤 밀도 모드에서 구성된 사용자 수 $N_U$는 **agent pool 크기**로 재해석된다. Run 시작 시 전용 `numpy.random.default_rng(seed)`에서 episode별 활성 사용자 수 $N^{\mathrm{act}}_k$를 후보 집합에서 균등 표본화하고, 각 episode의 활성 UE 부분집합을 무작위로 추출한다(`subset=random`). 비활성 UE는 그 episode에 task arrival이 0이므로 항상 local action 0을 취하고 학습 transition을 만들지 않는다.

전용 RNG를 쓰는 이유는 두 가지다. 첫째, task arrival·channel gain·exploration이 사용하는 전역 RNG 흐름과 분리되어 **고정 밀도 모드의 기존 결과가 바이트 수준으로 보존**된다. 둘째, 같은 seed의 모든 알고리즘이 동일한 밀도 시계열과 활성 부분집합을 보므로 알고리즘 간 비교가 paired 구조가 된다.

### 4.2 기록과 산출물

각 run은 기존 metric series에 더해 `ActiveUsers.txt`(episode별 활성 사용자 수), `density_schedule.json`(스케줄 전체), QECO-ADAPT2의 경우 `regime_selection.json`(episode별 영역·정책·$\sigma$·$w_E$)을 기록한다. 비교 단계는 `comparison_by_regime.csv/json`(영역별·전체 평균)과 `active_users_timeseries.csv`, `ActiveUsers_Timeseries.png`를 추가 생성한다. 평가 지표 정의는 adapt1 확장본 수식 (12)–(17)과 동일하며, QoE·delay·energy는 arrived task당 평균이므로 밀도가 섞여도 episode 간 비교 가능하다. 반면 drop은 episode별 미완료 task 수이므로 밀도 혼합 평균은 dense episode가 지배한다 — 영역별 표(§7.2)로 해석한다.

## 5. 실험 설계

### 5.1 단계 구성

| 단계 | 밀도 | 목적 | 구성 |
|---|---|---|---|
| Phase A-1 | 고정 user=100 | sparse 영역 정책 확정 | `qeco`, `qeco-gate-only`, general, adaptive gate |
| Phase A-0 | 고정 user=300 | 선택기 무손실성 검증 | QECO-ADAPT2 vs general |
| Phase B | 랜덤, pool 1500, 후보 {100,300,500,800,1200,1500} | 최종 평가 | `qeco`, general, adaptive gate, QECO-ADAPT2 |

공통 조건: seed 42, 500 episode, slot 100(+$D_{\max}$), 3 edge, 6 AP, AP당 4 channel, cluster size 3 — adapt1 확장본 §5.1의 공통 설정과 동일. 실행 배치는 CPU-only `per-agent` mode(bear 서버, 알고리즘당 1 process, process당 intra-op 2/inter-op 1 thread)이며 Phase A 6 process, Phase B 4 process를 tmux로 병렬 실행했다. Phase B의 wall time은 약 30시간이었다.

### 5.2 판정 기준 (사전 정의)

Phase B에서 QECO-ADAPT2는 다음을 만족해야 유효한 것으로 본다.

1. 전체 평균 QoE·completion에서 단일 정책 최고값 이상.
2. 영역별 평균에서 각 영역 최고 정책과 근접.
3. 밀도 전환 직후 episode의 성능 하락(adaptation lag)이 단일 정책보다 작거나 같음.

## 6. Phase A 결과 (고정 밀도)

### 6.1 A-1: sparse ablation (user=100)

| Algorithm | QoE | Delay | Energy | Drop | Completion |
|---|---:|---:|---:|---:|---:|
| QECO | 7.5948 | 7.7360 | 36.8619 | 1211.954 | 0.5823 |
| QECO + gate only | 8.2409 | 7.6798 | 35.9036 | 1178.798 | 0.5944 |
| **general** | **12.8759** | **7.4437** | **24.8319** | **1054.360** | **0.6370** |
| adaptive gate | 8.5885 | 7.6742 | 34.2996 | 1182.322 | 0.5932 |

general이 전 지표에서 최고다(QoE는 adaptive gate 대비 +49.9%, energy −27.6%, drop −10.8%). adapt1의 3종 기본 비교(user=100)에서는 QECO-ADAPT(=adaptive gate)가 최고였으나 general과 직접 비교된 적이 없었고, 직접 비교되자 열세로 판정되었다. 이 결과로 sparse 영역 정책 기본값을 adaptive gate에서 **general로 확정**했다. 매우 희소한 조건(user=30–50, users per edge 10–17)의 추가 검증은 선택 과제로 남긴다.

### 6.2 A-0: 선택기 무손실성 (user=300)

같은 seed의 고정 밀도 user=300에서 QECO-ADAPT2(general 영역으로 판정)와 general을 독립 재학습한 결과, 500 episode × 5개 지표의 시계열이 **한 셀도 다르지 않았다**(mismatched cells = 0; 공통 평균 QoE 7.1618, delay 8.3796, energy 20.6053, drop 4519.544, completion 0.4809). §3.3의 구조적 붕괴 성질이 실측으로 확인되었으며, 이후 Phase B에서 관측되는 QECO-ADAPT2와 general의 차이는 전적으로 dense episode에서의 gate mode 차이와 그로 인한 학습 이력 차이에서 나온다.

## 7. Phase B 결과 (랜덤 밀도)

### 7.1 전체 평균

밀도 구성은 100×89, 300×80, 500×87, 800×77, 1200×94, 1500×73 episode(sparse 89 / general 244 / dense 167)였고, 네 알고리즘이 동일 스케줄을 공유했다. QECO-ADAPT2의 선택 기록은 sparse 89회 → general(fixed), general 244회 → general(fixed), dense 167회 → adaptive gate(adaptive)로 설계와 정확히 일치했다.

| Algorithm | QoE | Delay | Energy | Drop | Completion |
|---|---:|---:|---:|---:|---:|
| QECO | 3.6494 | 8.5452 | 17.3217 | 13286.054 | 0.4018 |
| **general** | **8.0496** | 8.3519 | **12.0890** | **11369.948** | **0.4792** |
| adaptive gate | 6.8902 | 8.3624 | 16.2059 | 11411.874 | 0.4721 |
| QECO-ADAPT2 | 8.0425 | **8.3513** | 12.1225 | 11374.782 | 0.4791 |

QECO-ADAPT2는 최고 단일 정책 general과 QoE 차이 −0.0071(−0.09%)로 사실상 동률이며, adaptive gate 단독보다 +16.7%, QECO보다 +120.4% 높다. Energy에서도 general(12.089)과 QECO-ADAPT2(12.123)가 adaptive gate(16.206)를 크게 앞선다 — adaptive gate는 sparse·general 밀도에서 완화된 gate로 불리한 offloading을 더 허용해 energy를 낭비한다.

### 7.2 영역별 평균

| 영역(ep) | 지표 | QECO | general | adaptive gate | QECO-ADAPT2 |
|---|---|---:|---:|---:|---:|
| sparse(89) | QoE | 7.3104 | 11.5023 | 7.6105 | **11.5159** |
| | Energy | 27.4304 | **11.2648** | 26.3567 | 11.2924 |
| | Completion | 0.5442 | 0.5662 | 0.5473 | **0.5664** |
| general(244) | QoE | 2.9474 | **7.4288** | 6.4700 | 7.4209 |
| | Energy | 18.8492 | **14.3215** | 17.2472 | 14.3498 |
| | Completion | 0.3856 | **0.4685** | 0.4608 | 0.4684 |
| dense(167) | QoE | 2.7240 | 7.1165 | **7.1202** | 7.0996 |
| | Energy | 9.7025 | **9.2666** | 9.2749 | 9.3108 |
| | Completion | 0.3494 | 0.4486 | **0.4487** | 0.4482 |

밀도별 세분화(QoE):

| $N^{\mathrm{act}}$ | 100 | 300 | 500 | 800 | 1200 | 1500 |
|---|---:|---:|---:|---:|---:|---:|
| QECO | 7.310 | 4.243 | 1.773 | 2.928 | 1.897 | 3.789 |
| general | 11.502 | **8.272** | **6.904** | **7.147** | **7.033** | 7.224 |
| adaptive gate | 7.610 | 6.143 | 6.173 | 7.145 | 7.031 | **7.235** |
| QECO-ADAPT2 | **11.516** | 8.255 | 6.903 | 7.140 | 7.018 | 7.204 |

QECO-ADAPT2는 sparse에서 최고, 나머지 영역에서 각 영역 최고 정책과 0.3% 이내다(판정 기준 2 충족). general과 adaptive gate의 차이는 100–500에서 크고 800 이상에서 소멸한다 — adapt1 §7.2의 gate 수렴($g\to1$) 예측이 랜덤 밀도 환경에서 밀도 축을 따라 재현된 것이다.

### 7.3 학습 구간 분해

| 구간 | QECO | general | adaptive gate | QECO-ADAPT2 |
|---|---:|---:|---:|---:|
| QoE 1–300 | 0.2603 | **7.4832** | 5.5120 | 7.4720 |
| Energy 1–300 | 21.8635 | **14.5830** | 20.8626 | 14.6065 |
| QoE 301–500 | 8.7330 | 8.8991 | **8.9575** | 8.8982 |
| Energy 301–500 | 10.5089 | **8.3482** | 9.2209 | 8.3965 |

초기 1–300 구간에서 QECO의 QoE는 0.2603인 반면 general과 QECO-ADAPT2는 7.4832와 7.4720이었고, energy도 21.8635에서 14.5830과 14.6065로 감소했다. 후반 301–500 구간에서는 네 정책의 QoE가 8.73–8.96으로 근접하며 adaptive gate가 근소 최고가 된다. 따라서 전체 평균 차이의 주된 출처는 후반 정책 상한의 큰 차이보다 초기 warm-up 구간의 누적 손실 감소다. 이는 수학적 수렴 속도의 증명이 아니라 adapt1 확장본 §6.5와 Alpha 단계에서 관측한 warm-up loss pattern이 랜덤 밀도 환경에서도 재현된 결과이며, 메커니즘과 일반화 범위는 §8에서 구분해 논의한다.

### 7.4 전환(adaptation lag) 분석

밀도가 episode마다 i.i.d.로 표본화되는 본 설계에서는 500 episode 중 419회가 직전과 다른 밀도, 324회가 직전과 다른 영역이었다. 전환 episode와 비전환 episode의 QoE 평균 차이는 general(8.085 vs 7.851)과 QECO-ADAPT2(8.080 vs 7.829)에서 모두 유의미한 lag 방향이 아니었고(전환 episode가 오히려 높음 — 밀도 구성 효과), 두 알고리즘 간 차이도 없었다. dense 영역에 진입한 117회 episode에서도 QoE는 adaptive gate 7.122 ≈ general 7.104 ≈ QECO-ADAPT2 7.091로 구분되지 않았다.

**판정 기준 3은 본 설계로는 판별 불가로 처리한다.** episode 단위 i.i.d. 표본화는 전환이 너무 잦아 전환-비전환 구분의 변별력이 없고, per-UE Q-network가 한 episode 안에서 적응해야 하는 상황이 아니라 매 episode 새로운 밀도에서 처음부터 상호작용하는 구조이기 때문이다. lag 검증에는 밀도가 구간 단위로 상승·하강하는 sustained-shift 시나리오가 별도로 필요하다(§9).

### 7.5 dense 정책 관측 (부정적 결과)

설계 의도는 dense 영역에서 adaptive gate로 전환하는 것이었으나, 관측된 dense QoE는 adaptive gate 7.120 ≈ general 7.116 > QECO-ADAPT2 7.100으로 전환의 이득이 없다(전 지표 0.3% 이내). 이는 (i) dense에서 $g(N^{\mathrm{act}})\ge0.79$로 두 gate가 이미 수렴해 있고, (ii) QECO-ADAPT2의 network가 sparse·general episode에서는 fixed gate로, dense episode에서만 adaptive gate로 학습해 어느 단일 정책과도 완전히 같지 않은 학습 이력을 갖기 때문으로 해석한다. 현 데이터 기준으로는 dense 정책도 general로 두는 구성(즉 QECO-ADAPT2 ≡ general)이 가장 단순하고 강한 구성이지만, 차이가 단일 seed 잡음 범위이므로 기본값은 설계 의도(dense = adaptive gate)를 유지하고 multi-seed·sustained-shift 검증 후 확정한다.

## 8. 논의

**QECO-ADAPT2가 입증한 것.** 밀도 영역 판정과 정책 전환이 (i) 고정 밀도에서 하위 정책으로 정확히 붕괴하고(A-0, 0 mismatch), (ii) 랜덤 밀도에서 세 영역 전환을 모두 수행하면서 최고 단일 정책과 동률 성능을 유지하며(전체 QoE −0.09%), (iii) 잘못된 단일 구성(adaptive gate 고정)을 쓸 때의 손실(−14.4% QoE, +33.7% energy)을 자동으로 회피한다는 점이다. 즉 현 증거는 "밀도별 최적 정책을 아는 oracle 대비 무손실"을 지지한다.

**QECO-ADAPT2가 입증하지 못한 것.** 최고 단일 정책(general)을 능가하는 성능이다. 시험한 밀도 범위(33–500 users per edge)에서 general이 사전 판정 기준(0.3%)상 모든 영역에서 최강 또는 동률이므로, 영역별로 다른 정책을 고르는 것 자체의 이득이 발생할 여지가 없었다. 선택 구조의 이득은 (a) 영역별 최적 정책이 실제로 다른 밀도 범위(예: users per edge 10 부근의 초희소 조건 — Alpha 단계 legacy 관측에서 QECO가 ADAPT를 앞섰던 영역), 또는 (b) sustained-shift 조건에서만 관측될 수 있으며, 이는 반증 가능한 후속 가설로 남긴다.

**에너지 중심 튜닝의 의의.** 본 연구의 에너지 중심 조정은 energy만을 독립적으로 최소화하거나 평가 QoE의 정의를 바꾸는 것이 아니다. 공통 평가는 모든 알고리즘에 QECO의 원래 energy weight 1을 적용하고, 학습 단계에서만 활성 사용자 수에 따라 UE computation·transmission energy의 penalty를 조정한다. 이는 AP-channel contention이 있는 환경에서 비효율적인 offloading이 단말 transmission energy뿐 아니라 access·edge resource 점유, queueing delay와 drop을 함께 증가시킬 수 있기 때문이다. Fixed gate도 낮은 energy state, 높은 edge backlog, 작은 task, local execution이 더 빠른 조건과 낮은 access score의 edge proposal을 local로 치환한다. 따라서 energy는 단독 목적이라기보다 불리한 offloading의 위험을 제어하는 학습 신호이며, 현재 결과는 energy-aware reward와 실행 전 action filtering의 결합 효과로 해석해야 한다.

**보수적 제어의 부하 가변성과 general의 지배.** 사용자 밀도가 달라지면 offloading의 조건부 net gain과 energy·access·queue cost의 상대적 크기도 달라지므로, 하나의 보수적 설정이 모든 부하 영역에서 항상 최적이라고 가정할 수 없다. Adaptive energy weight는 시험 범위에서 $w_E(N^{\mathrm{act}})=1.2940$–$1.4815$로 완만하게 증가해 부하별 energy penalty를 연속 조정한다. Gate scale은 다른 방식의 가변성을 제공한다. $g(N^{\mathrm{act}})<1$인 adaptive gate는 fixed gate($\sigma=1$)보다 완화된 허용 집합을 사용하고, 부하가 증가하면 $g\to1$로 fixed gate에 수렴한다. 그러나 랜덤 밀도 전체와 모든 영역에서 general이 사전 판정 기준(0.3%)상 최강 또는 동률이었던 결과는, 완화된 gate가 추가로 허용한 offloading 집합의 조건부 net gain이 시험 조건에서 평균적으로 양수가 아니었음을 시사한다(adapt1 §7.1의 mechanism hypothesis와 부합). 그러므로 현 단일-seed ablation이 상대적으로 지지하는 구성요소는 **load-adaptive energy weighting**이며, **load-adaptive gate**의 이득은 아직 검증되지 않은 구조적 가설이다. 여기서 확장성은 network parameter 수의 증가가 아니라 밀도 영역이 바뀔 때 보수적 제어 강도를 재조정할 수 있는 구성 가능성을 뜻한다.

**초기 warm-up 손실 감소에 대한 해석.** §7.3의 초기 차이를 “모든 에너지 보수적 모델의 수렴성 향상”으로 일반화할 수는 없다. adapt1 구성요소 ablation에서 reward-weight-only는 QECO와 가까웠지만 gate-only가 큰 QoE·drop 개선을 보였고, Phase B에서 초기 우위를 보인 세 energy-aware 변형은 모두 gate를 포함한다. 구현상 greedy probability $\epsilon$은 0에서 증가하며 random branch는 local action을 제외한 edge action만 표본화하므로, 충분히 학습되지 않은 초기 정책은 offloading proposal에 편향된다(adapt1 §4.5). Gate는 낮은 access quality, 높은 backlog 또는 불리한 local–transmission time 관계의 proposal을 실행 전에 local action으로 치환하고, 치환된 실행 행동과 reward가 replay memory에 저장된다. 이 구조는 Q-function이 안정되기 전에 큰 손실을 만드는 transition의 누적을 줄이는 domain prior로 작동할 수 있고, adaptive energy weight는 높은 UE-side energy cost의 학습 가치를 낮추는 보조 역할을 한다. 초기 1–300 구간의 QoE·energy와 후반 QoE 수렴은 이 설명과 일치하지만, 이는 aggregate 결과에 기반한 메커니즘 가설이다. 보편적 인과로 확정하려면 proposed/executed action, gate rejection reason, counterfactual local/offloading cost, TD error와 replay 분포를 기록하고 paired multi-seed로 검증해야 한다.

## 9. 한계와 향후 연구

1. **단일 seed.** 모든 결과가 seed 42 단일 run이다. adapt1 확장본 §5.3의 paired design(동일 seed 공유, 최소 5개, 가능하면 10개 이상 [29]–[31])을 Phase B에 적용해야 하며, 필요한 run 수는 순차 검정 [32]으로 적응적으로 결정할 수 있다.
2. **oracle 밀도.** 현 구현은 알려진 활성 사용자 수를 사용한다. 이는 기존 $g(N_U)$와 같은 조건이지만, 실환경에서는 최근 slot의 arrival 관측으로 밀도를 추정하는 변형이 필요하다.
3. **i.i.d. 표본화의 한계.** 판정 기준 3(adaptation lag)은 본 설계로 판별 불가였다. 밀도가 구간 단위로 변하는 sustained-shift 스케줄 모드(low→high, high→low, burst)를 추가해 전환 직후 구간의 누적 QoE·drop을 비교해야 한다.
4. **dense 정책 미확정.** §7.5의 부정적 결과에 따라 dense = adaptive gate의 기본값은 검증 전 설계 의도로만 유지된다.
5. **학습 빈도 비대칭.** subset=random에서 개별 UE agent는 활성 episode에서만 transition을 받는다. 활성 빈도가 학습 품질에 미치는 영향은 prefix subset 방식과의 비교로 분리해야 한다.
6. **영역 임계값.** $d_{\mathrm{sp}}=50$, $d_{\mathrm{dn}}=350$은 시험 밀도 집합을 분류하기 위한 초기값이며 경계 근처(150, 1000–1100명 등)의 민감도 실험이 없다.
7. **실행 배치.** 모든 adapt2 결과는 CPU per-agent 배치다. adapt1 legacy의 shared 배치 결과와 절대값 비교가 불가능한 한계를 승계하며, 공개 artifact에는 실행 배치 metadata와 launcher log를 함께 보존한다(본 실험은 `logs/adapt2/`에 보존함).
8. **초희소 영역 미탐색.** users per edge 10–17(user=30–50) 조건은 실행하지 않았다. Alpha 단계의 channel-free 관측(1× 밀도에서 QECO 우위)이 channel-aware 환경에서 재현되는지에 따라 sparse 정책이 달라질 수 있다.

## 10. 결론

QECO-ADAPT2는 QECO의 행동 규약과 D3QN/LSTM backbone을 유지한 채 사용자 밀도 영역별로 QECO-family 하위 정책을 전환하는 선택 구조다. 고정 밀도 검증에서 선택기는 하위 정책과 시계열이 완전히 일치하는 무손실성을 보였고, sparse ablation은 general(adaptive weight + fixed gate)이 희소 조건에서도 전 지표 최강임을 확인시켰다. episode별 무작위 밀도(100–1500) 환경의 최종 실험에서 QECO-ADAPT2는 세 영역 전환을 모두 수행하며 최고 단일 정책과 동률의 QoE·energy·completion을 유지했고, 부적합 단일 구성과 QECO를 크게 상회했다. 반면 dense 영역에서 adaptive gate로 전환하는 이득과 전환 직후 적응 우위는 관측되지 않았다. 따라서 현 단일 seed 증거는 QECO-ADAPT2를 "성능을 능가하는 새 정책"이 아니라 "밀도 영역별 최적 구성을 자동 선택하는 무손실 선택 구조"로 지지하며, 그 실질적 이득의 입증은 paired multi-seed 평가와 sustained-shift 동적 부하 시나리오에 달려 있다.

## Appendix A. Scenario Inventory (adapt2 실험군)

| Scenario | 밀도 | Algorithms | 위치 |
|---|---|---|---|
| `adapt2_user_100_seed_42_ep_500_edge_3_ap_6_channel_4` | 고정 100 | qeco, gate-only, general, adaptive gate | `results/adapt2/comparisons/fixed_density/` |
| `adapt2_user_300_seed_42_ep_500_edge_3_ap_6_channel_4` | 고정 300 | QECO-ADAPT2, general | `results/adapt2/comparisons/fixed_density/` |
| `adapt2_user_100-1500_random_seed_42_ep_500_edge_3_ap_6_channel_4` | 랜덤 pool 1500 | qeco, general, adaptive gate, QECO-ADAPT2 | `results/adapt2/comparisons/random_density/` |

Run 원본은 `results/adapt2/{fixed,random}_density/<algorithm>/run_*`에 있으며, 랜덤 run은 `ActiveUsers.txt`, `density_schedule.json`, (QECO-ADAPT2) `regime_selection.json`을 포함한다. 런처 로그는 `logs/adapt2/`에 보존된다.

주요 그림(랜덤 밀도 비교 디렉토리):

- `ActiveUsers_Timeseries.png` — episode별 활성 사용자 스케줄
- `selected4_QoE_Timeseries.png`, `selected4_QoE_Timeseries_0_300.png`, `selected4_QoE_Timeseries_300_500.png`
- `selected4_CompletionRate_Timeseries*.png`, 지표별 `*_Timeseries.png`/`*_chart.png`

## Appendix B. Interpretation Rules

- adapt2 결과(CPU per-agent)와 adapt1 legacy 결과(대부분 shared)의 절대값을 결합하지 않는다.
- 랜덤 밀도의 drop 전체 평균은 dense episode가 지배하므로 영역별 표로 해석한다.
- 고정 밀도에서 QECO-ADAPT2는 영역 정책과 동일 알고리즘이다(A-0). 랜덤 밀도에서는 학습 이력이 달라 동일하지 않다.
- `qeco-adapt-general` ≡ `qeco-adaptive-weight-fixed-gate`, `qeco-adaptive-gate` ≡ `qeco-adapt` ≡ `qeco-adapt-full` — 동일 구현의 별칭이며 같은 배치·seed·시나리오에서 수치를 상호 인용할 수 있다.
- 통계적 독립 표본은 episode가 아니라 seed별 재학습 run이다. 본 초안의 모든 표는 단일 seed 기술통계다.

## Appendix C. Reproduction

```bash
cd /Users/nahw/Documents/QECO-Adapt

# Phase A (고정 밀도, 6 워커 tmux)
scripts/run_adapt2_phase_a.sh start
scripts/run_adapt2_phase_a.sh status
scripts/run_adapt2_phase_a.sh compare

# Phase B (랜덤 밀도, 4 워커 tmux)
scripts/run_adapt2_phase_b.sh start
scripts/run_adapt2_phase_b.sh compare
```

개별 run은 다음 형식으로 재현한다(서버 실행 프로파일 포함).

```bash
QECO_EXPERIMENT_FAMILY=adapt2 \
QECO_DENSITY_MODE=random \
QECO_DENSITY_CHOICES=100,300,500,800,1200,1500 \
QECO_NUM_USERS=1500 \
QECO_RANDOM_SEED=42 \
CUDA_VISIBLE_DEVICES=-1 \
QECO_AGENT_SESSION_MODE=per-agent \
TF_NUM_INTRAOP_THREADS=2 \
TF_NUM_INTEROP_THREADS=1 \
./env_channel/bin/python channel_common_eval.py qeco-adapt2 --episodes 500
```

검증 항목: `experiment_config.json`의 `density` 블록, `density_schedule.json`의 seed·choices, `regime_selection.json`의 영역별 선택 횟수(sparse 89 / general 244 / dense 167), `comparison_by_regime.csv`의 regime counts 일치 여부.

## References

본 초안은 adapt1 확장본 `docs/adapt1/paper_draft_channel_aware_qeco_adapt_kr_extended.md`의 References [1]–[41]을 승계하며 동일 번호로 인용한다. 서지 문자열과 검증 기록은 해당 문서와 `journals/notion_reference_review_registry.md`를 참조한다.
