# Task 1 모델 비교: 2031년 2월 로컬 평가

## 공통 평가 조건

10개 방향 패널(5개 corridor × 양방향)의 공식 Task 1 타깃만 채점한다. 두 완료 실험은 모두 2030년 6~12월의 `corridors/*/train/mainline_states`에서 합성 mask로 학습하고, 2031년 1월의 공식 masked 입력과 `task1/*/train/sample_submission_state.csv` 좌표로 모델·혼합 비율을 선택했다. 2031년 2월에는 같은 종류의 공식 masked 입력으로 예측한 뒤, 원본 값을 **점수 계산에만** 사용했다. 2월 정답은 학습과 선택에 쓰지 않았다. 이 점수는 정답이 공개된 train 구간의 로컬 평가이며 숨겨진 validation/private 점수는 아니다.

패널·regime별로 공식 타깃 좌표에서 speed RMSE와 flow/차로 RMSE를 구한다. 점수는 `0.54 × max(0, 1 − RMSE_speed/25) + 0.46 × max(0, 1 − RMSE_flow_per_lane/600)`이다. R1/R2/R3 → 양방향 → 5개 corridor family를 각각 동일 가중 평균하여 전체 `S_state`를 얻는다. 두 완료 실험은 동일한 **3,443,944개 타깃**, 30개 패널·regime 그룹을 모두 채점했다. 아래 R1/R2/R3 값은 해당 regime의 **10개 패널 점수 산술평균**이며, 타깃 셀 가중 평균이 아니다.

## 확인된 결과

| 방법 | 1월 선택 `S_state` | 2월 `S_state` | 보간 대비 상승 | R1 | R2 | R3 | 2월 타깃 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 공통 보간 기준점¹ | — | 0.906532 | — | 0.910363 | 0.909499 | 0.899733 | 3,443,944 |
| [STAEformer residual](../runs/staeformer_residual_001/test_result.json) | 0.931830 | **0.933377** | +0.026845 | 0.936964 | 0.935679 | 0.927489 | 3,443,944 |
| [CatBoost residual](../runs/catboost_residual_001/test_result.json) | 0.928985 | 0.930256 | +0.023724 | 0.933701 | 0.932533 | 0.924535 | 3,443,944 |
| ImputeFormer | **미확정** | **미확정** | **미확정** | **미확정** | **미확정** | **미확정** | **미확정** |
| SPIN-H | **미확정** | **미확정** | **미확정** | **미확정** | **미확정** | **미확정** | **미확정** |

¹ 각 모델 결과의 residual 혼합 비율 `0.0`으로 계산한 동일 보간 기준점이다. 두 실행 사이의 수치 차이는 1e-11 미만이다. 별도 [2월 보간 스크립트](../runs/staeformer_baseline_feb.json)는 fallback 구성에 차이가 있어 이 표의 비교 기준으로 사용하지 않는다. 완료된 두 모델의 1월 선택 혼합 비율은 모두 `1.0`이다. 2월 전체 점수에서 STAEformer가 CatBoost보다 **0.003121** 높았다.

## 학습량과 기록된 실행 시간

| 방법 | 학습 표본 범위 | 기록된 시간 | 자원·측정 범위 |
|---|---|---:|---|
| STAEformer residual | 2,140개 패널·일/epoch × 5 epoch, 합성 mask 20/30/50%; 전체 epoch에서 감독된 masked cell 누계 **124,224,414개** | **1,144.7초** (19분 4.7초) | GPU 9(A6000), **학습 시간만** 합계. 전처리·1월 선택·2월 평가는 미포함 |
| CatBoost residual | 같은 2,140개 패널·일에서 하루 최대 512개, 총 **1,095,680개** masked cell 표본; speed와 flow/차로 각각 400-tree 회귀기 | **203.9초** (3분 23.9초) | CPU 8 threads, 준비 82.9초·두 모델 학습 75.8초·1월 선택 22.4초·2월 평가 19.7초 등 전체 실행 시간 |
| ImputeFormer | **미확정** | **미확정** | **미확정** |
| SPIN-H | **미확정** | **미확정** | **미확정** |

STAEformer의 masked cell 수는 반복 epoch의 **누계**이고 CatBoost 수는 추출한 학습 행 수다. 표본 예산과 시간 측정 범위가 달라 이 표만으로 모델 간 학습 속도를 직접 비교할 수 없다. 현재 점수 차이에도 학습 표본량 차이가 포함된다.

## 후속 모델 결과 기입 기준

ImputeFormer와 SPIN-H의 **미확정** 칸은 실행 결과 파일을 확인한 뒤 채운다. 두 모델도 위의 6~12월 학습, 1월 선택, 2월 단회 평가 분할과 공식 masked 입력·타깃 좌표를 사용해야 한다. 결과마다 2월 타깃 3,443,944개와 30개 그룹의 누락 여부, `S_state`, R1/R2/R3, 보간 기준점, 학습 표본량, 측정 시간 범위를 함께 기록한다. 결과가 나오기 전에는 점수나 상대 순위를 추정하지 않는다.

근거 파일: [STAEformer 설정](../runs/staeformer_residual_001/config.json)·[학습 기록](../runs/staeformer_residual_001/history.json)·[2월 결과](../runs/staeformer_residual_001/test_result.json), [CatBoost 설정](../runs/catboost_residual_001/config.json)·[표본 기록](../runs/catboost_residual_001/sample_report.json)·[2월 결과](../runs/catboost_residual_001/test_result.json).
