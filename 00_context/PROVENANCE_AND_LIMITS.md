# 연구 목적과 근거의 경계

## 이번 후속 연구가 묻는 것

활동 사용자와 queue/access 상태가 변할 때, 사전에 학습한 전문 정책을 선택하면 강한 단일 정책보다 유리한 QoE–energy 절충을 얻을 수 있는가? 정책을 바꿀 수 있다는 사실과 바꾸는 것이 성능상 유리하다는 주장은 별도로 검증해야 한다.

새 구현은 `fixed`, `adaptive`, `contextual`별로 UE마다 독립 D3QN·replay·optimizer 상태를 둔다. 전환 arm은 앞의 두 정책만 사용하며 `contextual`은 공통 context 정의를 사용하는 no-gate 단일 정책 비교군이다. 현재 selector는 calibration에서 정한 context-bin별 label을 선택하는 규칙 기반 장치로, 별도의 RL 학습기나 미래 부하 예측기는 아니다.

세 학습 정책은 모두 같은 활동 사용자 수 기반 adaptive energy weight를 사용한다. `fixed`는 고정 gate scaling, `adaptive`는 부하 적응형 gate scaling, `contextual`은 gate 미적용이라는 차이가 있다. `fixed`를 fixed energy weight로 읽으면 안 된다. 각 policy bank는 UE별 network를 포함하므로 `contextual`도 전체 사용자용 network 하나가 아니다. 공통 context **정의**를 쓰지만, 비교 arm의 이전 action이 달라지면 queue 등 내생 context의 실제 값은 달라진다. 동일성을 보장하는 대상은 외생 trace이다.

## 버전을 섞지 않는 기준

| 구분 | 의미 | 이번 전달본에서의 사용 |
|---|---|---|
| Alpha QECO-Adapt | 이전 channel-free/단일 edge 연구 계열 | 배경만; 새 채널 환경의 수치와 합산하지 않음 |
| Channel-aware Adapt1 | AP cluster·AP-channel을 포함한 기존 gate/energy 연구 | topology·data-flow의 시각적 참고 |
| 이전 Adapt2 | episode 시작마다 reward/gate 정책 구성을 선택하는 공유 학습기 | `adapt2_extended_snapshot.md` 및 `legacy_random/`; 최신 bank 모델과 구분 |
| 동적 전문 학습기 pilot | 한 episode 내부에서 30/90/150 활동 사용자가 변하고 독립 전문 학습기를 선택 | 최신 구현 검토의 기준; commit `67c1a8c40c6b94de4add2f6d0a6d2dadbdddd3e7` |

`adapt2_extended_snapshot.md`는 기존 사용자 수정분을 그대로 보존한 **구버전 구현 중심 초안**이다. 특히 공유 backbone, episode별 선택, 실험 단계 서술을 새 구현 설명으로 그대로 옮기면 안 된다. 최신 코드와 [dynamic_protocol.md](dynamic_protocol.md), [의사코드](../04_pseudocode/ALGORITHM_REVIEW.md)가 이번 신규 그림의 기준이다.

## 실험 자료 상태

| 자료 | 확인 가능한 범위 | 말하면 안 되는 결론 |
|---|---|---|
| `legacy_random` | seed42, 500episode; 기존 QECO/General/Adaptive Gate/Adapt2 비교 | 독립 bank 전환 성능 또는 다중 seed 유의성 |
| 초희소 ORX 부분 보고서 | 원 보고서에 54/60cell 복구 검증이 기록됨; 이 패키지는 그 보고서 사본만 포함 | 이 패키지에서 원격 run/log 전체를 새로 검증했다는 주장; 누락 cell 포함 최종 통계 판정 |
| `dynamic_smoke` | 16/16 rollout 완료된 기능 시험 산출물 | 일반 평가가 전환 이득을 입증했다는 주장 |
| `dynamic_scale` | 150-user 규모 12/12 rollout 자원·기능 시험; 평가 episode 1개 | 수렴·통계적 유의성·제안 모델 우위 |
| 본 3-seed pilot | 이번 전달본에 결과 미포함; 서버 live 상태는 이번 자료 정리에서 조회하지 않음 | 현재 실행/완료 상태를 이 snapshot만으로 단정 |

Scale 시험 평가에서 switching의 QoE는 약14.727, fixed는 약15.794이다. 이 한 episode에서는 switching이 높지 않았다. 그림은 기능과 변동을 정직하게 보여야 하며 유리한 구간만 선택해 우위를 만들지 않는다. 실제 정책 전환 횟수는 `steps.jsonl`로 재계산한다.

## 공정성·재현성 검토점

- 새 pilot은 train/calibration/evaluation을 분리하고 같은 phase/episode의 외생 사용자·작업·채널 trace를 비교군 사이에 동일하게 제공한다. 이전 실험의 같은 seed label을 byte-identical trace와 혼동하지 않는다.
- 비활성화는 신규 arrival을 막는다. 기존 작업과 queue는 계속 처리되며 사용자 수가 바뀔 때 reset하지 않는다. Episode 사이까지 queue를 유지하는 실험은 아니다.
- 작업에는 action 시점의 origin expert와 energy weight를 보존한다. Replay 저장은 `training=true` 경로에만 적용한다. 본 pilot의 switching은 frozen evaluation에서 실행되므로 전환 뒤 완료된 작업도 측정·귀속 기록만 남기며 replay 저장이나 재학습을 하지 않는다.
- 평가 중 weights/replay/update는 고정한다. 학습된 정책의 실시간 선택이지 실시간 parameter 재학습의 효과 측정은 아니다.
- 2-expert bank와 contextual single bank의 전체 parameter 수가 다르다. 결과 그림에는 필요 시 계산량·메모리·선택 비용을 함께 제시해야 한다.
- 500episode를 500개의 독립 학습 seed처럼 세지 않는다. 다중 seed 평균/분산/paired 차이는 본 pilot 회수 후 seed 단위로 계산한다.
- 데이터는 시뮬레이션 생성이며 실제 사용자 이동 데이터셋이라고 표기하지 않는다.

## 사본 보존

`SOURCE_MANIFEST.json`의 원본 경로는 로컬 Documents 기준 상대 경로다. 모든 사본은 원본과 동일한 SHA-256을 기록한다. 원본의 날짜·절 번호·과거 검증 기록은 현재 확인 결과로 승격하지 않는다. 이번 검증의 범위는 루트 `VALIDATION.md`를 따른다.
