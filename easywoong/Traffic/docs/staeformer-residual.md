# Task 1: Interpolation + STAEformer residual

## 실험 범위

- **대상:** 10개 방향 corridor의 mainline 링크 1,583개. 하나의 모델을 모든 corridor에 공유한다.
- **입력:** 5분 간격으로 정렬한 하루 `288 × 링크 수` 배열. 관측 speed, 관측 flow/차로 수, 관측 여부, 보간 speed, 보간 flow/차로 수의 5개 채널을 사용한다.
- **출력:** 보간값에 더할 speed와 flow/차로 수의 residual. 모델은 24개 시점(2시간) 창을 입력받는다. 보간은 그 전에 하루 전체의 **관측 가능한 값만** 이용한다.
- **학습 타깃:** 원본 train에서 `is_score_eligible=1`이고 두 값이 있는 셀에 20%, 30%, 50% 확률로 무작위 mask를 생성한다. 가린 셀에서만 residual 오차를 학습한다.

모델은 STAEformer 논문의 시공간 attention과 시간/노드 embedding을 이 문제에 맞게 구현한 **재구성용 변형**이다. 원 논문의 미래 예측용 출력층을 그대로 재현한 모델은 아니다. 모델 구현: `src/task1/staeformer_residual.py`.

## 누수 없는 로컬 평가

| 기간 | 용도 | 파일 |
|---|---|---|
| 2030-06 ~ 2030-12 | 학습 | `corridors/*/train/mainline_states` |
| 2031-01 | epoch와 residual 혼합 비율 선택 | `corridors/*/train/mainline_states_masked` + `task1/*/train/sample_submission_state.csv` |
| 2031-02 | 마지막 한 번 로컬 테스트 | 동일한 종류의 공식 masked 파일과 Task 1 타깃 CSV |

1월과 2월 예측은 masked 파일에서 먼저 만든다. 원본 `mainline_states`는 예측을 만든 **뒤에 점수를 계산할 때만** 읽는다. Task 1 타깃은 공식 CSV에 있는 좌표만 사용하며, 빈 셀 전체를 타깃으로 간주하지 않는다. 이 점수는 공개 정답이 있는 train 구간의 로컬 테스트 점수다. validation/private 숨겨진 정답 점수가 아니다.

각 패널·mask regime에서 speed RMSE와 flow/차로 RMSE를 계산한다. `0.54 × max(0, 1 − RMSE_speed/25) + 0.46 × max(0, 1 − RMSE_flow_per_lane/600)`으로 regime 점수를 만들고, regime → 방향 → corridor 순서로 동일 가중 평균한다. 공식 타깃 수만큼 전부 예측됐는지도 검증한다.

## 실행

`Traffic` 디렉터리에서, 학교 서버의 **GPU 9**와 `easypj` 환경을 사용한다. 스크립트도 torch import 전에 GPU 9만 보이도록 설정한다.
실행에 사용한 주요 패키지 버전은 `requirements.txt`에 고정했다.

```bash
conda run -n easypj python -m src.task1.train_staeformer --smoke
conda run -n easypj python -m src.task1.train_staeformer
```

기본 학습은 **5 epoch**, seed 42, AdamW, BF16이다. `--smoke`는 각 corridor의 하루만 학습하고 regime별 하루씩 평가하여 실행 경로를 확인한다. 실제 실험 결과는 `runs/staeformer_residual_001/`에 저장한다.

- `config.json`: 기간, 입력, 모델, 학습 설정
- `history.json`: epoch별 학습 loss와 1월 선택 점수
- `selection_best.json`: 1월에서 선택한 checkpoint와 residual 비율
- `test_result.json`: 2월 공식 타깃 전체의 보간 단독/선택 모델 점수와 패널·regime별 오차
- `best.pt`, `last.pt`: 로컬 checkpoint. Git에서는 제외한다.

비교용 보간 기준점은 `python -m src.task1.compare_baseline_feb`로 실행하며 `runs/staeformer_baseline_feb.json`에 저장된다. 이 기준점은 2월의 masked 관측으로 만든 프로필 fallback을 사용하므로, 모델의 residual 효과만 보려면 `test_result.json` 안의 `0.0` 혼합 비율과 비교한다.

## 이번 5 epoch 결과

1월 선택 점수는 5 epoch checkpoint에서 **0.931830**이었고 residual 혼합 비율은 **1.0**으로 선택됐다. 이후 2월 공식 Task 1 타깃 **3,443,944개**를 모두 채점했다.

| 2월 로컬 Task 1 | 보간만 | 보간 + residual |
|---|---:|---:|
| 전체 `S_state` | 0.906532 | **0.933377** |
| R1, 패널 평균 | 0.910363 | 0.936964 |
| R2, 패널 평균 | 0.909499 | 0.935679 |
| R3, 패널 평균 | 0.899733 | 0.927489 |

동일한 입력·평가 코드에서 residual이 전체 점수를 **0.026845** 높였다. 다섯 corridor family 모두 점수가 올랐다. 자세한 패널·regime별 RMSE는 `runs/staeformer_residual_001/test_result.json`에 있다.

## 해석 시 주의

무작위 독립 mask는 공식 R1/R2/R3의 연속·공간 결측 패턴과 다르다. 특히 긴 결측 구간에서 성능 차이가 날 수 있다. 로컬 테스트가 끝난 뒤 9개월 전체로 재학습한 모델을 만들 수 있지만, 그 모델의 2월 점수는 더 이상 독립 테스트 점수가 아니다.

참고: [STAEformer 논문](https://arxiv.org/abs/2308.10425), [원저자 공개 구현](https://github.com/XDZhelheim/STAEformer).
