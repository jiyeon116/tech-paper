# QECO-ADAPT 공동 검토 자료

작성일: 2026-09-06 · 공동 연구 검토용 · 외부 전송/공개 게시 미수행

현재 후속 연구인 **QECO-ADAPT2의 동적 전문 학습기 전환**을 중심으로 정리했다. 기존 episode별 정책 선택 초안과 channel-aware 결과는 배경 자료이며, 최신 구현과 동일한 실험이 아니다. 원본 논문·HWPX·코드·실험 수치는 수정하지 않았다.

## 먼저 할 일

1. [연구 범위와 결과 해석](00_context/PROVENANCE_AND_LIMITS.md)을 읽는다.
2. **그림 제작:** [그림별 요청서](01_figures/FIGURE_BRIEFS.md)의 F01·F02를 먼저 구체화하고, F03은 제공된 시험 데이터로 표현을 검토한다.
3. **참고문헌 검증:** [검토 지침](02_references/REVIEW_GUIDE.md)에 따라 [41편 검토표](02_references/reference_review.csv)의 P1부터 검토한다. [원문 PDF](02_references/papers/) 33편이 포함되어 있다.
4. 여력이 있을 때 [코드 주석 검토](03_code_comments/REVIEW_GUIDE.md)와 [의사코드 검토](04_pseudocode/ALGORITHM_REVIEW.md)를 진행한다.

## 역할과 반환물

| 역할 | 우선도 | 넘겨드리는 자료 | 반환을 요청하는 자료 |
|---|---|---|---|
| 2. Figure 제작 | 최우선 | 기존 그림 3개, 생성 코드, 새 그림 요청서, CSV/JSONL | 편집 가능한 원본 + SVG/PDF + PNG + 영문 caption + 데이터/수정 내역 |
| 4. References 유효성 검증 | 최우선 | 기존 번호 41개, 로컬 PDF 33편, DOI/공식 링크/공개 코드 확인 범위 | 번호별 판정, 근거 페이지·절, 정정 서지문, 인용 적합성 메모 |
| 1. Code 정리·주석 | 보조 | commit `67c1a8c`의 코드 사본과 주석 점검 지침 | comments/docstring 위주 diff 및 테스트 결과; 동작 변경은 별도 제안 |
| 3. Pseudocode 검토 | 보조 | 코드와 연결된 단계별 의사코드 | 잘못되거나 누락된 단계, 변수 정의, 코드와의 일치 여부 |

일정은 임의로 정하지 않았다. 첫 반환은 F01·F02와 참고문헌 P1 검토를 권장한다. [피드백 양식](05_feedback/REVIEW_RETURN.md)에 수정 위치와 근거를 남기면 원저자가 통합한다. 그림의 내용 변경·새 성능 주장·실험 조건 변경은 자동 반영 대상이 아니다.

## 폴더 안내

- `00_context/`: 연구 목적, 버전 경계, 기존 확장 초안과 부분 실험 보고서 사본
- `01_figures/`: 그림 요청서, 기존 PNG, 생성 코드, 숫자를 재현할 수 있는 데이터
- `02_references/`: 원문 PDF, 기존 참고문헌 문자열, 검토용 CSV/Markdown, 공식 출처 점검
- `03_code_comments/`: 실험 당시 소스 사본과 코드 주석 작업 범위
- `04_pseudocode/`: 최신 구현에 대응하는 의사코드 초안과 검토 항목
- `05_feedback/`: 공동 검토 결과 반환 양식
- `SOURCE_MANIFEST.json`: 사본별 원본 상대 경로·크기·SHA-256
- `VALIDATION.md`: 이번 전달본 검증 결과와 미완료 사항

## 사용 시 주의

- 최신 실험 소스는 GitHub에 push하지 않은 로컬 commit이다. 전달본 `03_code_comments/source/`가 이번 검토 기준이며 GitHub `main`과 같다고 가정하지 않는다.
- 본 pilot 전체 결과는 이 전달본에 포함하지 않았다. smoke/scale 시험으로 QoE 우위나 통계적 유의성을 주장하지 않는다.
- 추가 자료 **Ten Simple Rules for Better Figures**의 공식 원문·코드 링크는 [추가 자료 메모](02_references/ADDITIONAL_READING.md)에 있다. 다운로드 명령이 실행 정책에 차단되어 PDF 확보·transfer 번역은 미완료다.
- 기존 PDF는 연구 검토용 사본이다. 각 출판사/원문의 이용 조건을 확인하고 공용 저장소에 일괄 재배포하지 않는다. 코드 사본에는 LICENSE와 THIRD_PARTY_NOTICES를 보존했다.
- HWPX는 사용자가 직접 편집한 그림·수식을 보호하기 위해 변경하지 않았으며 이 전달본에도 포함하지 않았다.
- 새 검토 지침의 링크는 전달본 내부 기준이다. 기존 초안·프로토콜·소스 README는 원문 보존 사본이므로 일부 링크는 원래 연구 workspace를 전제로 한다. 사본 경로는 `SOURCE_MANIFEST.json`에서 확인한다.
- 그림용 [간편 CSV 설명](01_figures/data/DERIVED_DATA.md)에 100-step 전환 trace와 4-arm 시험 결과의 단위·변환식을 정리했다. 이는 본 pilot 결과의 대체물이 아니다.
