# 003: 전처리부터 전체 Task 제출까지

## 1. 모델 선택

| Task | 선택 | 이유 |
|---|---|---|
| 1: 결측 상태 복원 | Interpolation + STAEformer residual | 기존 2월 로컬 평가에서 보간 0.906532 → 0.933377 |
| 2: 정체 예측 | CatBoostClassifier | 짧은 이력의 최근값·추세·인접 링크·시간 정보를 함께 사용 |
| 3: 물리 정합성 | Task 1의 speed·flow를 그대로 제출 | 별도 출력 없이 동일 state 행을 평가 |
| 4: 경로 유량 추정 | prior 정규화 + 비음수 최소제곱 | 공개 링크 교통량을 만족하도록 weak prior 보정 |

Task 1은 미래 예측용 원본 STAEformer의 출력층을 그대로 사용하는 모델이 아니다. 시공간 attention과 embedding을 결측 재구성에 맞게 적용한 residual 변형이다.

## 2. 공통 전처리

원본을 수정하지 않고 `cache/preprocessed_v1/`에 저장했다.

- **시간:** UTC 기준 5분 간격, 하루 288개 시점.
- **공간:** 패널별 mainline 링크 순서를 고정. 10개 방향 패널, 총 1,583개 링크.
- **단위:** 모델에서는 flow를 차로 수로 나눔. 제출할 때 다시 곱해 `flow_vph`로 복원.
- **결측:** 원래 결측과 공식 마스크를 보존. 결측을 교통량 0으로 취급하지 않음.
- **품질:** `pct_observed`, `is_score_eligible`, `is_imputed`, 실제 관측 여부를 보존.
- **타깃:** 공식 `sample_submission_state.csv` 좌표만 Task 1의 예측 대상으로 사용.
- **검사:** 중복 키, 시간 간격, station/link 매핑, 차로 수, 음수·무한대, 타깃 누락 확인.

전체 캐시는 **6,070개 일별 격자**다. train 원본 2,730개 + train masked 2,730개 + validation/private masked 610개이며, 서로 다른 날 6,070개라는 뜻은 아니다. Task 2의 공식 window는 세 split 합계 240개다.

보간 전 데이터에는 smoothing이나 임의 이상치 제거를 하지 않았다. Ramp도 별도 테이블로 보존하며 결측 유량을 0으로 채우지 않았다. 이번 모델의 입력에는 ramp를 추가하지 않았다.

## 3. Task 1: 복원용 입력

```text
train 원본의 eligible·유효 관측
→ 20% / 30% / 50% 무작위 셀 가림
→ 남아 있는 관측만으로 하루 단위 보간
→ 24시점 × 링크 수 × 5채널
→ STAEformer가 보간 오차를 예측
→ 보간값 + residual
```

5개 채널은 관측 speed, 관측 flow/차로, 관측 mask, 보간 speed, 보간 flow/차로다. 입력 speed는 120, flow/차로는 2,000으로 나눈다. 이전에 선택한 모델과 같은 고정 스케일을 사용하며 validation/private 통계를 fit하지 않는다.

무작위로 가린 셀에서만 loss를 계산한다. 원래부터 값이 없는 셀은 정답으로 쓰지 않는다. 공식 R1/R2/R3도 셀 단위 마스크지만, 실제 masked 입력에는 원래 결측과 Task 2 보호 구간의 추가 공백이 함께 있으므로 학습 입력과 완전히 같지는 않다.

최종 출력은 이전 로컬 실험과 동일하게 speed를 `5 ~ min(135, 1.05 × free_speed)`로 제한하고 flow는 0 이상으로 둔다. 유량을 임의로 50 이상으로 올리지 않는다.

### 학습과 평가 구분

| 기간 | 기존 개발 실험 | 이번 제출용 학습 |
|---|---|---|
| 2030-06 ~ 2030-12 | 학습 | 학습 |
| 2031-01 | epoch·residual 비율 선택 | 학습 |
| 2031-02 | 독립 로컬 평가 | 학습 |
| 2031-03, 2031-04 | 숨겨진 평가 입력 | 예측만 |

기존 1월 결과에서 정한 **5 epoch, residual 비율 1.0**을 고정하고 train 9개월 전체로 재학습했다. 2,730개 패널·일을 epoch마다 사용했고, 학습 시간은 합계 **1,239.3초(20분 39초)**였다. 학습 loss는 0.008969 → 0.007728 → 0.006797 → 0.006010 → 0.005442로 감소했다. 이 값은 masked cell의 정규화 MSE이며 공식 점수가 아니다.

기존 2월 점수 0.933377은 이전 개발 모델의 결과이며, 이번 9개월 학습 모델의 독립 테스트 점수로 재사용하지 않는다.

## 4. Task 2: 예측용 입력

기준 시각을 T라고 하면 입력은 `T−60분 ~ T−5분`의 12개 시점이다. 출력은 `T+5분 ~ T+30분`의 6개 시점이다. Task 1처럼 하루 전체를 보간하면 미래 정보가 섞일 수 있으므로, Task 2 특징은 전달받은 12개 과거 시점에서만 계산한다.

사용한 특징:

- speed/free_speed, flow/차로/capacity_per_lane, occupancy의 최근값과 lag
- 과거 평균·최솟값·표준편차·기울기
- 관측 비율, 마지막 관측까지의 시간, 관측 품질
- 상·하류의 최근 속도와 변화량, 패널·링크 ID
- 시간대·요일, 예측할 horizon

공식 정체 라벨은 비공개다. train 원본의 `speed ≤ 0.6 × free_speed`를 **대체 라벨**로 사용한다. 공개 속도에는 관측 오차가 있으므로 주최측의 잠재 교통 상태 라벨과 동일하지 않다.

6~12월로 학습하고 1월에서 tree 수와 분류 임계값을 선택했다. 2월은 선택 후 한 번 대체 평가했다. 이후 같은 설정으로 9개월 전체를 재학습했다. 학습 표본은 패널·날짜별 onset/ongoing 조건에서 최대 한 window씩 추출했다. 원본에서 만든 표본은 공식 window의 추출 간격을 완전히 재현하지 않는다.

| 항목 | 결과 |
|---|---:|
| 선택한 tree 수 / 임계값 | 400 / 0.40 |
| 1월 대체 IoU | 0.772287 |
| 2월 마지막 관측 유지 IoU | 0.383404 |
| 2월 CatBoost IoU | **0.757789** |
| 2월 평가 | 396개 window, 유효 대체 라벨 340,933개 |
| 최종 재학습 | 3,911개 window, 3,363,920개 행 |

위 IoU는 공식 Task 2 점수가 아니다. 각 window에서 유효 대체 라벨에 대해 IoU를 계산하고, 패널 내 onset/ongoing을 같은 비중으로 평균했다.

## 5. Task 4: 경로별 유량

`path_link_incidence`를 경로-링크 연결행렬 A로 만든다. 경로·링크 ID를 명시적으로 맞춘 뒤 공개 count c와 weak prior b를 사용한다.

```text
minimize ||A f − c||² + 0.05 × ||f − b||²
subject to f ≥ 0
```

λ=0.05는 [주최측 baseline 설정](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/BASELINES.md)을 고정해 사용했다.

경로 수가 많아 처음의 반복법은 일부 패널에서 수렴하지 않았다. 동일한 목적함수를 링크 수 차원의 dual Newton 방식으로 풀어, 모든 validation/private 패널에서 수렴과 비음수 최적성 조건을 확인했다.

| 공개 링크 적합도 | 10개 패널 평균 S_link |
|---|---:|
| validation | 0.999939 |
| private | 0.999923 |

S_link는 공개 링크 교통량으로 확인 가능한 부분이다. OD 오차·prior 대비 변화·도착지 분포 평가는 숨겨진 reference가 필요하므로 이 값을 전체 S_ODME로 해석하지 않는다.

## 6. 제출과 기록

state·queue·odme 예측을 `submission_key.csv`로 매칭한다. split, panel, 시간, 링크, regime, window, path, zone 중 각 task에 필요한 키를 모두 사용한다.

1. 각 task에서 공식 타깃 대비 누락·추가·중복이 없는지 확인.
2. 해당 task의 예측값이 모두 준비된 뒤 관련 없는 출력 열만 0으로 작성.
3. 공식 `submission_id`, task, 행 순서와 값 범위를 확인.
4. CSV를 다시 읽어 메모리의 예측값과 일치하는지 확인.
5. 코드·설정·모델·입력·제출 파일 해시를 저장하고 업로드.

검증된 제출 파일은 **6,980,503행**이다. state 6,735,795행, queue 174,000행, odme 70,708행이며, 각 task의 누락·추가·중복은 0개다. 파일 크기는 약 378 MB다. 입력 누수·캐시·타깃 매칭·최적화 관련 8개 테스트도 통과했다. 자세한 검증 항목은 [verification.json](../submissions/003_multitask/verification.json)에 있다.

최종 점수와 Kaggle 제출 ID는 [result.json](../submissions/003_multitask/result.json)에 기록한다. 전체 재실행 명령과 파일 위치는 [README](../README.md)에 있다.

총점은 `0.35 × Task 1 + 0.30 × Task 2 + 0.15 × Task 3 + 0.20 × Task 4`다. [공식 scoring specification](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/SCORING_SPEC.md)

## 7. 실제 Kaggle 결과

2026-10-21 제출, ID **57461328**, 상태 **COMPLETE**.

| 비교 | Public 점수 |
|---|---:|
| 이번 제출 직전 계정의 최고 기록 | 0.76089 |
| 003: STAEformer + CatBoost queue + NNLS | **0.76826** |
| 차이 | **+0.00737** |

이번 실행에서 실제 업로드는 한 번 했다. Private 점수는 아직 공개되지 않았다. 반환된 결과는 종합 점수이므로, 어느 task가 얼마나 개선됐는지는 이 제출 하나로 구분할 수 없다.

- 제출 CSV: `submissions/003_multitask/submission.csv`
- 점수·접수 기록: [result.json](../submissions/003_multitask/result.json), [upload_receipt.json](../submissions/003_multitask/upload_receipt.json)
- 모델·소스·입력 해시: [metadata.json](../submissions/003_multitask/metadata.json)
- CSV SHA-256: `260905d04c12519d2468348b019377715d6c27f1b648cbb871f396f15564f061`
