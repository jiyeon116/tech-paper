# QECO-ADAPT2 초희소 고정부하 탐색 실험: 부분 복구 결과 및 해석

> **문서 상태:** 부분 완료 실험의 탐색적 분석 자료
>
> **ORX run:** `a73d418f-8ed9-452a-86b5-37c6d13088df`
>
> **Campaign:** `qeco-adapt2-ultrasparse-4a6c70dac16b-20260901T050744Z-1619116`
>
> **실행 commit:** `4a6c70dac16bba5577615e50b98524fe33dfcca8`
>
> **완료도:** 계획한 60개 cell 중 54개 검증 완료(90.0%)
>
> **공식 판정:** 사전등록한 네 개 QoE 대조의 Holm 보정 최종 판정은 계산할 수 없음

## 1. 요약

이 실험은 기존 100–1,500명 조건에서 `qeco-adapt-general`이 대체로 가장 높은 성능을 보인 상황에서, 더 낮은 부하에서는 다른 정책이 재현성 있게 우세해지는지를 확인하기 위해 수행되었다. 부하 구간마다 우세한 정책이 달라야만 QECO-ADAPT2의 밀도 기반 정책 선택에 실질적인 역할을 부여할 수 있기 때문이다. 이에 기존에 검증되지 않은 고정 30명 및 50명 조건에서 `qeco`, `qeco-adapt-general`, `qeco-adaptive-gate`를 seed-matched 방식으로 비교하였다.

30명 조건은 계획한 10개 seed와 세 알고리즘이 모두 완료되었다. 이 조건에서 `qeco-adaptive-gate`의 평균 QoE는 20.7845로 가장 높았으며, `qeco-adapt-general`보다 seed-paired 평균 6.9729% 높고 10/10 seed에서 사전 정의한 +1% SESOI를 초과했다. 다만 energy는 general보다 9.3928 높았다. 이는 adaptive gate가 completion과 QoE를 회복하는 대신 general의 강한 energy 절감 일부를 포기하는 절충과 일치한다.

50명 조건은 10개 중 8개 seed만 완료되었다. 확보된 8개 seed의 전체 500-episode 평균에서는 `qeco-adapt-general`의 QoE가 18.3667로 가장 높았고, `qeco-adaptive-gate`는 16.1054였다. 반면 episodes 301–500에서는 adaptive gate가 general보다 seed-paired 평균 4.7628% 높았고 7/8 seed에서 +1%를 초과했다. 이 후반 역전은 정책별 초기 수렴 특성과 안정 구간 성능이 다를 가능성을 보여주지만, 사전등록된 primary endpoint가 아니며 두 seed가 누락되어 진단적 결과로만 해석해야 한다.

따라서 현재 결과는 초희소 부하를 하나의 정책으로 포괄하기 어렵다는 탐색적 신호를 제공한다. 그러나 전체 캠페인이 완결되지 않았고 모든 run이 fixed-density로 수행되었으므로, QECO-ADAPT2 selector의 실시간 전환 우수성이나 사전등록 가설의 최종 통계적 지지를 주장할 수는 없다.

## 2. 실험 목적과 연구 질문

QECO-ADAPT2는 활성 사용자 밀도에 따라 하위 정책을 선택한다. 그러나 기존 channel-aware 결과에서는 `qeco-adapt-general`이 100–1,500명 범위에서 대부분 최상 또는 실질적으로 동등했다. 단일 정책이 모든 부하에서 계속 우세하다면 selector는 복잡성만 증가시키고 성능 이득을 제공하지 못한다.

본 실험의 연구 질문은 다음과 같다.

> 고정 30명 또는 50명 조건에서 `qeco`나 `qeco-adaptive-gate`가 `qeco-adapt-general`보다 재현성 있게 높은 QoE를 보이는가?

사전등록 주가설은 두 사용자 조건 중 적어도 하나에서 대안 정책의 seed별 500-episode 평균 QoE가 general보다 1.0% 이상 높다는 것이다. Seed별 효과량은 다음과 같이 정의되었다.

$$
d_s=100\frac{\overline{QoE}_{\mathrm{alt},s}-\overline{QoE}_{\mathrm{general},s}}
{\left|\overline{QoE}_{\mathrm{general},s}\right|}
$$

1.0%는 새 결과를 본 뒤 선택한 문턱값이 아니라, 기존 0.3% practical-tie 범위를 넘고 selector 복잡성을 정당화하기 위해 실험 전에 정한 smallest effect size of interest(SESOI)이다.

## 3. 비교 정책과 실험 설계

| 정책 | 구성 | 실험 내 역할 |
|---|---|---|
| `qeco` | 기존 QECO | 기준 정책 |
| `qeco-adapt-general` | adaptive energy weight + fixed gate | 기존 부하 범위의 강한 단일 정책이자 primary baseline |
| `qeco-adaptive-gate` | adaptive energy weight + load-conditioned gate | 부하별 gate 조절 후보 정책 |

| 항목 | 사전 고정 조건 |
|---|---|
| 활성 사용자 수 | 고정 30명, 고정 50명 |
| Seed | 42, 77, 123, 202, 314, 509, 777, 1024, 2026, 4099 |
| Episode | cell당 500 |
| 계획된 cell | 2 사용자 조건 × 3 정책 × 10 seed = 60 |
| 통계 단위 | 독립적으로 재학습된 seed run |
| Primary endpoint | seed별 500-episode 평균 QoE |
| Primary SESOI | general 대비 상대 QoE +1.0% |
| Secondary endpoint | CompletionRate, Energy, DropRate |
| Topology | edge server 3개, AP 6개, AP당 channel 4개, AP cluster size 3 |
| Runtime | CPU-only, per-agent TensorFlow session, intra-op 2, inter-op 1 |
| Density mode | fixed |

같은 사용자 조건과 seed label을 세 정책에 적용했지만, 각 정책은 독립적으로 재학습되었다. 또한 정책별 random-number consumption과 task/channel random stream이 완전히 분리되어 있지 않으므로, 이는 seed-matched block design이지 세 정책이 byte-identical 외생 trajectory를 관측했다는 의미는 아니다. Episode를 독립 표본으로 간주하지 않으며 seed run을 통계 단위로 사용한다.

## 4. 실행 완료도와 중단 원인

| 사용자 조건 | 계획 | 검증 완료 | 상태 |
|---:|---:|---:|---|
| 30 | 30 cells | 30 cells | 10 seed × 3 정책 완전 층 |
| 50 | 30 cells | 24 cells | 8 seed × 3 정책 부분 층 |
| 합계 | 60 cells | 54 cells | 90.0% 완료 |

50명·seed 2026의 세 정책은 전체 campaign hard deadline 때문에 약 episode 220에서 status 124로 종료되었다. 직전 시점에는 12시간 예산 중 1,599초만 남아 worker timeout이 1,239초로 축소된 상태였다. Seed 4099의 세 정책은 시작되지 않았다. 이는 알고리즘 내부 예외가 아니라 campaign 시간 예산 부족으로 발생한 기술적 중단이다.

중단된 wave 때문에 정확한 60-run manifest 검증과 최종 archive/checksum 생성 단계가 실행되지 않았다. 따라서 run wrapper의 종료 코드는 1이며, 사전등록한 네 개 primary contrast의 Holm 보정 familywise 판정은 `not_computable`이다.

## 5. 복구 및 자료 무결성

원격 ORX run은 로컬 OpenResearch artifact 영역으로 회수되었다. 복구 시 원격·로컬의 파일 수, 디렉터리 수, 전체 tree digest, campaign subtree digest를 대조했으며 `rsync --checksum` 차이가 없었다.

- 복구 파일: 1,458개
- 복구 디렉터리: 152개
- 완료 run manifest: 54개, 중복 없음
- 정상 status: 54개 `0`
- Timeout status: 3개 `124`
- 완료된 각 run: QoE, Delay, Energy, Drop, CompletionRate, DropRate 및 ActiveUsers 각각 500개 finite 값
- 완료 로그: progress 500/500 및 `[EXIT STATUS] 0`
- 원시자료 재계산 평균: `validated_run_means.csv`와 일치
- 분석 산출물: `SHA256SUMS`에 포함된 모든 파일 검증 통과

이 검증은 현재 확보된 54개 cell이 분석 가능한 정상 결과임을 의미한다. 다만 누락된 여섯 cell을 보완하거나 전체 캠페인의 확인적 성공을 대신하지는 않는다.

## 6. 전체 500-episode 수치 결과

아래 값은 각 seed run의 500-episode 평균을 구한 뒤 seed 간 평균 ± 표본 표준편차로 집계한 것이다.

| Users | Policy | Seeds | QoE | Delay | Energy | Drop | CompletionRate | DropRate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 30 | QECO | 10 | 19.2255 ± 0.6133 | 4.7223 ± 0.0283 | 39.0384 ± 1.5183 | 67.6354 ± 3.2435 | 0.9226 ± 0.0037 | 0.0774 ± 0.0037 |
| 30 | General | 10 | 19.4627 ± 1.2332 | 5.5261 ± 0.3519 | 20.5380 ± 2.4939 | 198.7112 ± 37.0043 | 0.7723 ± 0.0423 | 0.2277 ± 0.0423 |
| 30 | Adaptive gate | 10 | **20.7845 ± 0.7359** | 4.7430 ± 0.0902 | 29.9308 ± 0.9737 | 79.9442 ± 7.8315 | 0.9084 ± 0.0089 | 0.0916 ± 0.0089 |
| 50 | QECO | 8 | 14.9889 ± 0.2288 | 6.2517 ± 0.0195 | 38.0591 ± 0.6462 | 297.1278 ± 3.0100 | 0.7955 ± 0.0020 | 0.2045 ± 0.0020 |
| 50 | General | 8 | **18.3667 ± 0.3458** | 6.1036 ± 0.0588 | **22.4069 ± 1.6639** | 351.5818 ± 22.0856 | 0.7581 ± 0.0152 | 0.2419 ± 0.0152 |
| 50 | Adaptive gate | 8 | 16.1054 ± 0.1296 | 6.1546 ± 0.0112 | 34.2189 ± 0.6704 | **286.2403 ± 2.5081** | **0.8030 ± 0.0017** | **0.1970 ± 0.0017** |

### 6.1 활성 사용자 30명: 완전한 10-seed 층

Adaptive gate는 general보다 QoE가 seed-paired 평균 6.9729% 높았으며, 10/10 seed에서 사전등록 SESOI인 +1.0%를 초과했다. 탐색적 one-sided exact sign test는 $p=0.0009766$이고 win-probability exact 95% confidence interval은 0.6915–1.0000이었다. 그러나 전체 네 contrast가 완결되지 않아 이 값을 Holm 보정된 사전등록 최종 판정으로 사용할 수는 없다.

General 대비 adaptive gate의 평균 차이는 다음과 같다.

| 지표 | Adaptive gate − General | 방향 해석 |
|---|---:|---|
| QoE | +6.9729% | adaptive gate 우세 |
| Delay | −0.7831 | adaptive gate 우세 |
| Energy | +9.3928 | general 우세 |
| Drop | −118.7670 | adaptive gate 우세 |
| CompletionRate | +0.1361 | adaptive gate 우세 |
| DropRate | −0.1361 | adaptive gate 우세 |

Adaptive gate는 QECO와 비교해 QoE가 seed-paired 평균 8.1037% 높고 energy는 9.1076 낮았다. 반면 CompletionRate는 QECO보다 0.0142 낮고 DropRate는 0.0142 높았다. 따라서 이 조건에서 adaptive gate는 세 정책 중 가장 높은 QoE를 제공했지만, 모든 개별 지표에서 동시에 최적인 정책은 아니었다.

### 6.2 활성 사용자 50명: 불완전한 8-seed 층

확보된 8개 seed에서 general은 가장 높은 전체 평균 QoE와 가장 낮은 energy를 기록했다. Adaptive gate는 general보다 QoE가 seed-paired 평균 12.2954% 낮았고 +1.0%를 넘은 seed는 0/8이었다. 반면 CompletionRate는 0.0449 높고 DropRate는 0.0449 낮아, QoE·energy와 task completion 사이의 순위가 일치하지 않았다.

Adaptive gate는 QECO보다 QoE가 seed-paired 평균 7.4666% 높았고 8/8 seed에서 +1.0%를 초과했다. Delay는 0.0972 낮고 energy는 3.8402 낮았으며, CompletionRate는 0.0075 높고 DropRate는 0.0075 낮았다. 현재 확보된 8개 seed만 보면 adaptive gate는 QECO를 전 지표에서 상회했지만, 누락된 두 seed 때문에 50명 조건의 확정적 순위로 일반화해서는 안 된다.

## 7. 초기 구간과 후반 구간의 진단적 분석

구간 분석은 사전등록 primary endpoint가 아니며, 관측된 수렴 패턴을 설명하기 위한 사후 진단이다.

| Users | 구간 | QECO QoE | General QoE | Adaptive gate QoE |
|---:|---|---:|---:|---:|
| 30 | Episodes 1–300 | 17.8077 ± 0.6422 | 19.2362 ± 1.1901 | **19.6994 ± 0.7685** |
| 30 | Episodes 301–500 | 21.3524 ± 0.5885 | 19.8026 ± 1.3017 | **22.4120 ± 0.6920** |
| 50 | Episodes 1–300 | 11.9119 ± 0.2706 | **17.7562 ± 0.2849** | 13.3824 ± 0.1754 |
| 50 | Episodes 301–500 | 19.6045 ± 0.2449 | 19.2824 ± 0.4964 | **20.1899 ± 0.1377** |

30명 조건에서 adaptive gate는 general보다 episodes 1–300의 paired QoE가 평균 2.5480% 높고, episodes 301–500에서는 13.4280% 높았다. 전체 500 episodes의 우위는 특히 후반 구간에서 확대되었다.

50명 조건에서는 반대되는 시간 패턴이 나타났다. Adaptive gate는 general보다 episodes 1–300에서 24.6258% 낮았지만, episodes 301–500에서는 4.7628% 높았고 7/8 seed가 +1.0%를 초과했다. 즉 general의 초기 수렴 우위가 전체 500-episode 평균을 지배했지만 adaptive gate가 후반 안정 구간에서 역전하는 신호가 관측되었다.

이 결과는 “어떤 정책이 우수한가”가 활성 사용자 수뿐 아니라 평가 시간 구간에도 의존할 수 있음을 시사한다. 그러나 후반 구간을 새 primary endpoint로 바꾸거나 이 결과만으로 adaptive gate의 장기 우위를 확정해서는 안 된다.

## 8. 해석과 QECO-ADAPT2에 대한 의미

### 8.1 초희소 부하 내 정책 이질성 신호

30명 조건에서는 adaptive gate가 전체 QoE에서 general을 일관되게 상회했지만, 50명의 현재 8개 seed에서는 general이 전체 평균 QoE에서 우세했다. 이는 기존에 하나의 sparse 영역으로 묶었던 낮은 부하 내부에서도 정책 순위가 달라질 수 있음을 보여주는 탐색적 신호다. QECO-ADAPT2의 sparse threshold와 하위 정책을 단일 기준으로 고정하기보다, 더 세밀한 부하 구분 또는 상태 의존 선택 규칙을 검토할 근거가 된다.

### 8.2 Energy와 service completion의 절충

두 ADAPT 변형은 같은 adaptive energy weight를 사용하며 gate 구성만 다르다. 30명에서 general은 energy를 가장 적게 사용했지만 CompletionRate가 0.7723으로 낮고 DropRate가 0.2277로 높았다. Adaptive gate는 energy를 추가로 사용하면서 CompletionRate를 0.9084로 회복하고 가장 높은 QoE를 얻었다. 50명에서도 general은 energy가 가장 낮았지만 adaptive gate보다 CompletionRate가 낮았다.

이 패턴은 load-conditioned gate가 fixed gate보다 더 많은 offloading 기회를 허용하여 completion을 높이는 대신 energy를 더 소비할 수 있다는 설계 메커니즘과 일치한다. 다만 gate rejection, proposed action, executed action의 세부 로그가 없으므로 gate가 수치 변화의 직접 원인이라고 인과적으로 확정할 수는 없다.

### 8.3 수렴 구간에 따른 정책 선택 가능성

50명의 부분 결과에서 general은 초기 300 episodes 동안 크게 우세했지만 adaptive gate는 후반 200 episodes에서 역전했다. 이는 고정된 사용자 수만으로 정책을 선택하는 방식 외에도 학습 단계, queue 상태, access 상태 또는 최근 성능 추세를 함께 사용하는 state-dependent selector의 가능성을 제기한다. 다만 이를 검증하려면 후반 구간을 사전등록한 독립 실험과 action-level 기록이 필요하다.

## 9. 해석 한계와 허용되는 주장

### 9.1 반드시 유지해야 할 한계

1. 50명 조건에서 두 seed가 누락되어 전체 60-cell 캠페인이 완결되지 않았다.
2. 사전등록한 네 primary contrast의 Holm 보정 최종 판정은 계산할 수 없다.
3. Episodes 301–500 분석은 사후 진단이며 primary endpoint가 아니다.
4. 모든 run은 fixed-density다. Episode별 active-user 변화나 sparse/general/dense selector의 실시간 전환 효과를 검증하지 않았다.
5. 동일 seed label은 사용했지만 정책 간 byte-identical 외생 trajectory를 보장하지 않는다.
6. Gate rejection과 action-level log가 없어 관측된 절충의 인과 메커니즘을 확정할 수 없다.

### 9.2 현재 자료로 허용되는 주장

- 30명 조건의 완전한 10-seed 층에서 adaptive gate의 QoE 우위와 energy–completion trade-off가 관측되었다.
- 50명 조건의 확보된 8개 seed에서는 general의 전체 평균 QoE 우위와 adaptive gate의 후반 역전 신호가 관측되었다.
- 초희소 부하를 하나의 고정 sparse 정책으로 묶는 현재 경계는 추가 검증이 필요하다.
- 현재 자료는 후속 실험 가설, sparse threshold 재검토, 내부 초안 및 보완 보고서의 근거로 사용할 수 있다.

### 9.3 현재 자료로 허용되지 않는 주장

- 사전등록한 전체 캠페인이 policy heterogeneity를 통계적으로 입증했다.
- QECO-ADAPT2의 실시간 density selector가 단일 정책보다 우수하다.
- 50명 조건에서 특정 정책이 확정적으로 최적이다.
- 사후 구간 분석이 사전등록 primary endpoint를 대체한다.
- 누락된 여섯 cell만 추가하면 기존 캠페인의 확인적 성공 판정이 복원된다.

## 10. 후속 검증 설계

확인적 결론이 필요하면 새로운 계약으로 60개 cell 전체를 독립 재실행해야 한다. 기존과 같은 12시간 단일 campaign은 다시 timeout될 위험이 있으므로 사용자 조건 또는 seed block별로 독립적으로 finalization 가능한 campaign을 구성하거나, 관측 runtime에 충분한 시간 예산을 설정해야 한다.

후속 실험에서는 다음 항목을 사전에 고정하는 것이 필요하다.

1. Primary endpoint를 전체 500-episode 평균과 후반 안정 구간 중 하나로 명시한다.
2. Fixed-density 확인 실험과 random-density selector 실험을 분리한다.
3. Random-density 비교에서는 정책별로 동일한 active-user schedule을 사용한다.
4. Episode별 active users, selected regime, gate threshold, gate rejection, proposed action, executed action을 기록한다.
5. Sustained low-to-high 및 high-to-low shift를 counterbalanced 순서로 실행하여 전환 직후 QoE와 cumulative regret을 측정한다.
6. 분석은 seed run을 통계 단위로 유지하고 사전등록된 다중비교 보정을 적용한다.

## 11. 산출물과 provenance

### 실험 계약

- [Experiment contract](/Users/nahw/Documents/.omx/specs/autoresearch-qeco-adapt2-ultrasparse/experiment_contract.md)
- [Mission and preregistered hypothesis](/Users/nahw/Documents/.omx/specs/autoresearch-qeco-adapt2-ultrasparse/mission.md)

### 복구 원시자료

- [Recovered ORX run](/Users/nahw/.local/share/openresearch/files/qeco-adapt2-ultra-sparse-screen/recovered-a73d418f/)
- [Recovery provenance](/Users/nahw/.local/share/openresearch/files/qeco-adapt2-ultra-sparse-screen/recovered-a73d418f.provenance.md)

### 분석 자료

- [Partial analysis summary](/Users/nahw/.local/share/openresearch/files/qeco-adapt2-ultra-sparse-screen/analysis-a73d418f/partial_analysis.md)
- [Machine-readable analysis](/Users/nahw/.local/share/openresearch/files/qeco-adapt2-ultra-sparse-screen/analysis-a73d418f/partial_analysis.json)
- [Validated run means](/Users/nahw/.local/share/openresearch/files/qeco-adapt2-ultra-sparse-screen/analysis-a73d418f/validated_run_means.csv)
- [Group descriptives](/Users/nahw/.local/share/openresearch/files/qeco-adapt2-ultra-sparse-screen/analysis-a73d418f/group_descriptives.csv)
- [Paired effects](/Users/nahw/.local/share/openresearch/files/qeco-adapt2-ultra-sparse-screen/analysis-a73d418f/paired_effects.csv)
- [Section QoE descriptives](/Users/nahw/.local/share/openresearch/files/qeco-adapt2-ultra-sparse-screen/analysis-a73d418f/section_qoe_descriptives.csv)
- [Section QoE paired effects](/Users/nahw/.local/share/openresearch/files/qeco-adapt2-ultra-sparse-screen/analysis-a73d418f/section_qoe_paired_effects.csv)
- [Analysis checksums](/Users/nahw/.local/share/openresearch/files/qeco-adapt2-ultra-sparse-screen/analysis-a73d418f/SHA256SUMS)
