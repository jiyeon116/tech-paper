# 전달본 검증 기록

검증일: 2026-09-06. 범위는 이 전달본의 정합성과 검토 가능성이다. 전체 연구의 성능 우위 또는 41편 모두의 인용 타당성 판정이 아니다.

## 완료한 검사

| 항목 | 확인 결과 |
|---|---|
| 원본 보존 | 사본 110개 각각의 크기·SHA-256이 로컬 원본 및 SOURCE_MANIFEST와 일치 |
| 코드 기준 | qeco-adapt2 clean commit `67c1a8c40c6b94de4add2f6d0a6d2dadbdddd3e7`; 원본 코드 변경 없음 |
| 참고문헌 | 번호 1–41 연속, 검토 상태 모두 PENDING; 원문 PDF 33개 동봉 |
| PDF 구조 | qpdf: 32개 경고 없이 통과, [15] 1개 linearization hint-table 경고; 구조 검사 실패 0개 |
| PNG 구조 | 기존 그림/차트 6개의 Pillow decode 검증 통과 |
| 시각 표본 점검 | topology·UE flow·server flow와 Completion Rate 차트 4개를 화면에서 확인. PNG 전체 6개·PDF 전체 페이지 시각 검증은 아님 |
| Legacy raw 비교 | 500행·연속 episode·finite value 확인; 전체 평균 20개, 구간 평균 40개, regime/all 평균 80개 재계산 |
| 평균 오차 | 최대 절대오차 `4.943821352298983e-09`; CSV 소수점 저장 정밀도를 고려한 허용치 `1e-8` 이내 |
| Dynamic smoke | 16/16 rollouts, 결과 checksum 8개 일치, 평가 trace pairing 2세트, 일반 switching 평가 전환 0/0회 |
| Dynamic scale | 12/12 rollouts, 결과 checksum 7개 일치, 평가 trace pairing 1세트, switching 전환 6회 재계산 일치 |
| 간편 데이터 | timeline 100행, evaluation 4행; 원본 필드 동일, completion percentage만 factor×100 변환 |
| 회귀 테스트 | 코드 사본에서 dynamic 테스트 50개 실행: 49개 통과, TensorFlow-dependent 1개 skip, 실패 0개 |
| 문법·링크 | Python AST 문법 검사 및 새 지침 Markdown 상대 링크 존재 검사 통과. 보존용 기존 Markdown의 workspace 외부 링크는 제외 |

테스트 runtime: `/Users/nahw/Documents/.translation-runtime/venv/bin/python`, NumPy 2.5.1. pinned TensorFlow 실험 runtime이 아니므로 TensorFlow 기반 학습 전체 검증을 주장하지 않는다. 테스트 중 표시되는 campaign complete/failed/timeout은 임시 디렉터리의 시험 fixture이며 bear 서버 새 실험이 아니다.

## 독립 내용 검토에서 반영한 점

- 고정·적응 gate 및 contextual 비교군의 adaptive energy-weight 통제조건을 명시했다.
- `contextual single learner`를 UE별 network를 포함하는 `single-policy bank`로 정정했다.
- 동일 외생 trace와 action에 따라 달라지는 내생 queue context를 구분했다.
- Calibration에서도 common QoE task record를 사용함을 의사코드에 보완했다.
- Training replay 귀속과 frozen switching 평가의 측정·기록을 분리했다.

## 남아 있는 제한

- 로컬 PDF 미포함: [1], [2], [3], [4], [5], [21], [33], [34]. 매핑된 경로에서 미발견이며 논문 자체 부재를 뜻하지 않는다.
- 추가 PLOS PDF 다운로드 명령은 `approval required by policy, but AskForApproval is set to Never`로 차단됐다. 신규 PDF 다운로드 0건, transfer 전문 번역 미수행. 공식 원문·코드 링크와 활용 지침만 제공한다.
- 41편 전부의 DOI/게재판/본문 인용 적합성/정정·철회 상태는 검증하지 않았다. 개별 외부 점검 범위는 reference_review.csv에 명시했다.
- [15] 원문 PDF 경고는 자동 복구하지 않았다. 원본과 같은 bytes를 보존한다.
- 기존 PNG는 제작 참고물이다. Legacy Completion Rate 차트의 큰 내부 제목·겹친 곡선·낮은 대비 색 등은 새 제작에서 개선할 대상이지 최종 양식 승인 결과가 아니다. 구조도도 새 specialist 구조로 오인하지 않는다.
- 본 3-seed pilot 결과 및 원격 live 상태는 이번 전달본에서 확인하지 않았다. Smoke/scale 또는 과거 54/60 보고서로 본 pilot 완료·우월성을 주장하지 않는다.
- 기존 논문/HWPX/실험 수치, 사용자 dirty 파일을 변경하지 않았고 Git commit/push·외부 전달을 실행하지 않았다.
- 테스트가 생성한 bytecode 22개(373,950 bytes)는 재생성 가능·원본 보존·열린 handle 없음까지 확인했으나 삭제 명령이 실행 정책에 차단됐다. 재시도나 우회 삭제 없이 로컬 캐시를 보존하며 전달 ZIP에서는 제외한다. 다른 사용자 캐시·runtime·실험 자료는 정리 대상이 아니다.

## 재검증

패키지 루트에서 실행한다. `verify_package.py`에는 Pillow와 qpdf가 필요하다.

```bash
python3 tools/export_review.py
python3 tools/verify_package.py
```

원본과도 대조하려면 마지막 명령 뒤에 로컬 Documents 절대 경로를 인자로 추가한다. 코드 회귀 테스트는 source 폴더에서 NumPy가 있는 interpreter로 실행한다.

```bash
cd 03_code_comments/source
python3 -m unittest discover -s tests -p 'test_dynamic_*.py'
```

압축본에는 `PACKAGE_MANIFEST.json`의 파일별 SHA-256을 포함하며 압축 생성 시 전체 member read-back·CRC·해시를 검사한다. ZIP 자체 SHA-256은 같은 폴더의 `.zip.sha256` 파일에 제공한다. 압축은 전달용 포장이지 외부 전송이 아니다.
