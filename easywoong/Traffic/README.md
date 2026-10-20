# TrafficFlowBench

파일별 설명과 결과: [../read.me](../read.me)

실행 경로는 이 `Traffic/` 폴더입니다. 원본 데이터는 `data/kaggle_public/`에 두거나 `TFB_DATA_ROOT` 환경 변수로 지정합니다. 주요 패키지 버전은 `requirements.txt`에 있습니다.

```bash
# Task 1: 보간 + STAEformer residual
conda run -n easypj python -m src.task1.train_staeformer --smoke
conda run -n easypj python -m src.task1.train_staeformer

# 기본 보간 방식으로 전체 제출 CSV 생성
conda run -n easypj python -m src.submit.build_submission --out submissions/local_interp --task1 interpolate
```

GPU 학습 코드는 학교 서버의 물리 GPU 9 기준으로 작성됐습니다. 노트북의 그림과 요약표는 `eda_assets/`에도 저장돼 있습니다.
