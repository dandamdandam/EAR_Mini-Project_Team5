# TrafficFlowBench 2026

10개 방향 corridor의 교통 관측으로 결측 상태, 미래 정체, 경로 교통량을 예측한다.

## 기준 제출 구성 (003)

| Task | 모델 | 입력 → 출력 |
|---|---|---|
| 1 | 보간 + STAEformer residual, 5 epoch | masked speed·flow → 지정된 위치의 speed·flow |
| 2 | CatBoost, 400 trees | 과거 60분 → 5~30분 뒤 정체 여부 |
| 3 | Task 1 예측을 함께 평가 | 별도 제출 행 없음 |
| 4 | prior를 반영한 비음수 최소제곱, λ=0.05 | 링크 교통량·경로 연결·prior → 경로별 교통량 |

**003 제출 완료:** Kaggle Public **0.76826**. 이전 최고 0.76089 대비 **+0.00737**. 제출 ID `57461328`, 상태 `COMPLETE`.

- [전처리·학습·평가 설명](docs/multitask-003.md)
- [실행 설정](configs/multitask_003.json)
- [기존 Task 1 모델 비교](docs/task1-model-comparison.md)
- [제출 기록](submissions/003_multitask/metadata.json) / [Kaggle 결과](submissions/003_multitask/result.json)

## 추가 실험 (004–006)

**3건 모두 Kaggle 제출·채점 완료.** 물리 GPU 9에서 학습하고, 1월에서 선택한 뒤 전체 9개월로 재학습했다. Task 2·4 예측은 003과 정확히 동일하게 유지했다.

| 제출 | Task 1 모델 | Kaggle Public | 003 대비 |
|---|---|---:|---:|
| 004 | 개선 STAEformer, 누적 20 epoch | 0.77803 | +0.00977 |
| 005 | ImputeFormer, 20 epoch | 0.78138 | +0.01312 |
| 006 | STAEformer 25% + ImputeFormer 75% | **0.78255** | **+0.01429** |

최고 점수 제출은 **006**, Kaggle 제출 번호는 `57468140`이다.

- [실험 설정·기간 분할·재현 방법](docs/task1-upgrade-004-006.md)
- [구체적인 설정](configs/task1_upgrade_004_006.json)

```bash
conda run --no-capture-output -n easypj python -m src.submit.run_task1_upgrade
conda run --no-capture-output -n easypj python -m src.submit.run_task1_upgrade --stages submit
conda run --no-capture-output -n easypj python -m src.submit.run_task1_upgrade --stages status
```

세 CSV를 모두 확정하고 검증한 다음 업로드한다. 이미 시도한 파일은 중복 제출하지 않는다.

## 실행

작업 디렉터리는 `Traffic`, 환경은 `easypj`, 딥러닝은 **물리 GPU 9**만 사용한다.
원본 데이터는 `data/kaggle_public/`에 둔다. 패키지 버전은 `requirements.txt`에 있다.

```bash
# 기존 003 결과를 보존하고 새 번호로 전체 재실행
conda run --no-capture-output -n easypj python -m src.submit.run_multitask --id 007

# 동일 명령에 --submit을 추가하면 검증 완료 후 Kaggle에 한 번 업로드
# Kaggle 인증은 서버에 설정된 계정을 사용한다.
```

저장된 모델로 예측과 CSV만 다시 만들려면 기존 제출 CSV를 보존할 별도 출력 경로를 지정한다.

```bash
conda run --no-capture-output -n easypj python -m src.submit.run_multitask --id 003 --stages state queue --predict-only
conda run --no-capture-output -n easypj python -m src.submit.multitask --out submissions/003_recreated
```

제출과 상태 확인:

```bash
conda run --no-capture-output -n easypj python -m src.submit.kaggle_submit --out submissions/003_multitask
conda run --no-capture-output -n easypj python -m src.submit.kaggle_submit --out submissions/003_multitask --status-only
```

이미 업로드를 시도한 폴더에서는 중복 업로드를 막는다. 상태 확인은 제출 횟수를 사용하지 않는다.

## 파일 위치

| 경로 | 내용 |
|---|---|
| `src/preprocess/run.py` | 날짜·링크 정렬, 데이터 검증, 결측 보존 캐시 |
| `src/task1/final_staeformer.py` | 9개월 전체 학습 및 공식 Task 1 타깃 예측 |
| `src/task2/catboost_queue.py` | 과거 시점만 사용하는 정체 특징·학습·예측 |
| `src/task4/regularized_odme.py` | 경로 연결행렬과 공개 교통량으로 최적화 |
| `src/submit/` | 전체 타깃 결합, 검증, 코드 스냅샷, 업로드 |
| `cache/preprocessed_v1/` | 전처리 데이터·품질 검사 결과 |
| `runs/staeformer_final_003/` | Task 1 모델·학습 기록·예측 |
| `runs/queue_catboost_003/` | Task 2 모델·로컬 대체 평가·예측 |
| `runs/odme_003/` | Task 4 예측·수렴 및 링크 적합도 |
| `submissions/003_multitask/` | 제출 CSV·설정·소스와 입력 해시·Kaggle 결과 |

원본 데이터, 캐시, 모델 가중치, 예측 parquet, 제출 CSV는 Git에서 제외한다.

## 검증

```bash
conda run --no-capture-output -n easypj python -m src.preprocess.run --smoke --out cache/preprocessed_smoke
conda run --no-capture-output -n easypj python -m unittest discover -s tests -v
```

가린 정답이 입력에 들어가지 않는지, 캐시가 원본과 일치하는지, 제출 타깃 누락·중복·잘못된 task 매핑을 차단하는지 확인한다.
