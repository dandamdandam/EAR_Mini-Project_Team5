# TrafficFlowBench 2026 — 대회 사양 (Overview / Data / Rules 통합)

> Kaggle: `2026-ieee-big-data-traffic-flow-bench`
> GitHub: https://github.com/jacky850/trafficflowbench-public
> 이 문서는 참조용 상세 사양이다. 작업 중 지켜야 할 규칙 요약은 `CLAUDE.md`에 있다.

---

## 1. 대회 개요

고속도로 회랑 데이터를 **복원 → 예측 → 물리검증 → 수요추정** 순서로 다루는 멀티태스크 벤치마크.
단일 예측 오차 최소화가 아니라 "교통 시스템을 재구성하고, 예측하고, 설명할 수 있는가"를 묻는다.

네 태스크는 서로를 필요로 한다. 모든 관측에 맞는 재구성도 차량이 검지기 사이에서 생기거나 사라지면
물리적으로 불가능하고, 제곱오차만으로는 이를 잡아내지 못한다. 큐 예측은 출발 상태만큼만 정확하고,
수요 추정은 올바른 링크 카운트를 요구한다. 이 네 단계가 실제 교통관리센터의 작업 순서다.

### 점수

```
S_total = 0.35·S_state + 0.30·S_queue + 0.15·S_physics + 0.20·S_ODME
```

- 각 성분은 [0, 1], 1이 완벽
- 태스크 출력이 없거나 invalid → 해당 태스크 0점, 나머지는 정상 채점
- **태스크의 필수 행이 하나라도 없으면 그 태스크 전체 0점** (빠진 행만 0 처리가 아님)

### 집계

10개 방향 패널별 계산 → 같은 corridor의 2방향 평균 → 5개 corridor 균등 평균.
양방향이 있다고 corridor가 두 번 카운트되지 않는다.

### 채점 자격 (eligibility)

```python
is_imputed        = pct_observed < 100
is_score_eligible = (pct_observed >= 75) and (필요 값 존재)
```

- mainline·ramp 모두 약 **76%**가 eligible, 10개 패널에 걸쳐 거의 균일
- ineligible 셀도 릴리스되지만 어느 태스크에서도 ground truth로 쓰이지 않는다
- **quality rule을 통과 못한 ramp 관측은 "0 veh/h 측정"이 아니라 "증거 없음"**

---

## 2. Task 1 — State Reconstruction (0.35)

### 내용

blank 처리된 셀의 `speed_kmh`, `flow_vph`를 복원한다.
density는 제출·채점 대상이 아니며, 필요 시 평가자가 `k = q/v`로 유도한다.

### 채점식

```
S_v(r) = max(0, 1 − RMSE_speed(r) / 25)            # km/h
S_q(r) = max(0, 1 − RMSE_flow_per_lane(r) / 600)   # veh/h/lane
S_state(r) = 0.54·S_v(r) + 0.46·S_q(r)
S_state    = mean(S_state(R1), S_state(R2), S_state(R3))
```

flow 오차는 **차로당**으로 계산한다. 같은 절대 오차가 3차로와 6차로 corridor에서 다른 의미를 갖기 때문.

### 목표 수치 감각

| RMSE_speed (km/h) | S_v | RMSE_flow (veh/h/lane) | S_q |
|---|---|---|---|
| 25 | 0.00 | 600 | 0.00 |
| 15 | 0.40 | 360 | 0.40 |
| 10 | 0.60 | 240 | 0.60 |
| 5 | 0.80 | 120 | 0.80 |
| 2.5 | 0.90 | 60 | 0.90 |

S_state 0.1 상승 = 총점 0.035 상승.

### 마스킹 메커니즘

- `mainline_states_masked`는 `mask_regime`으로 파티셔닝: **R1=20% / R2=30% / R3=50%** 제거
- **각 캘린더 날짜는 정확히 하나의 regime에만 속한다.** 같은 날이 세 뷰로 중복 공개되지 않는다
  (한 뷰에서 가린 셀을 다른 뷰에서 읽는 것을 차단)
- blank 시 **모든 측정 채널이 함께 제거**: `speed_kmh`, `flow_vph`, `occupancy`, occupancy 유래 `density`.
  루프 검지기가 같은 측정에서 전부 산출하므로 장애는 전부를 가져간다
- **quality 필드는 남는다** → eligible 타깃과 ineligible 셀을 구분할 수 있다

### 타깃 목록

`task1/<PANEL>/<split>/sample_submission_state.csv`가 리더보드가 쓰는 목록이다. 반드시 이 파일을 사용한다.

published hash로 마스크를 직접 유도하면 **약간 더 큰 집합**이 나온다 — Task 2 window 안에 들어가는 셀은
Task 1에서 제외되기 때문. 그 window들은 raw 관측을 공개하므로, 그 안의 Task 1 타깃은 답이 노출된 상태가 된다.

> **역이용**: validation/private에서도 `task2/<PANEL>/<split>/window_history.parquet`에는
> 마스킹되지 않은 실관측 60분이 들어 있다. 평가월의 수요 수준·노이즈 스케일 추정에 쓸 수 있는
> 유일하게 열린 창구이며, 규정상 정상적인 모델링이다.

### R3가 어려운 이유

R3는 특정 날의 eligible 셀 절반이 통째로 비는 상황이라, 그 날의 시간축 이웃도 함께 비어 있을 확률이 높다.
같은 날 내 국소 보간이 약해지고 **공간(link 간) 정보 + 과거 동일 요일·시간대 패턴**의 가치가 커진다.
3레짐 단순 평균이므로 R3가 병목이다.

---

## 3. Task 2 — Short-term Queue Prediction (0.30)

### 내용

유일한 online 태스크. forecast origin T 이전 **60분 관측**이 주어지고,
**T+5, +10, +15, +20, +25, +30** 6스텝에 대해 전 link의 큐 상태를 이진 예측한다. T 자체는 horizon에 포함되지 않는다.

### 라벨 정의

**speed ≤ 0.60 × free-flow speed**일 때 queued.
라벨은 released observation이 아니라 **underlying state**에서 계산되므로, 임계 부근 측정 노이즈가 라벨을 바꾸지 않는다.

### window 구성

corridor·split당 10개 = `queue_onset` 5 + `queue_ongoing` 5

- **onset**: history 전체에 queue 없음 + horizon에 최소 1개 발생
- **ongoing**: T 시점에 최소 1개 link가 queued

모든 window의 horizon에 queue 셀이 최소 1개 존재하므로, "전부 0" 전략은 어떤 window도 얻지 못한다.

### 채점식

```
IoU = |predicted ∩ true| / |predicted ∪ true|
S_queue = 전체 window의 평균 IoU (window·조건 균등 가중)
```

양쪽 집합이 모두 비면 그 window는 1점.

### 제약

- horizon 셀 + **그 이후 60분**까지 state 파일에서 blank → 읽거나 보간해서 답을 얻을 수 없다
- **D12_I405_N, D12_I405_S는 Task 2 window가 없다.** Tasks 1·3·4로만 채점된다

### 구현 노트

- IoU는 precision/recall 비대칭이라 임계값을 그대로 `0.6·v_f`로 쓰는 게 IoU 최적이 아닐 수 있다.
  확률 컷오프를 train에서 IoU 기준으로 튜닝할 것
- loss는 BCE보다 **soft-IoU / Dice** 계열이 목적함수와 정렬된다
- 라벨이 모든 split에서 비공개이므로, train unmasked 레이어로 자체 라벨링해서 window를 구성한다.
  단 train unmasked는 관측(노이즈 포함)이고 공식 라벨은 underlying state 기준이라 임계 부근에서 어긋난다
  → 해당 구간 가중치를 낮추거나 soft label로 처리

---

## 4. Task 3 — Physics Consistency (0.15)

### 내용

**제출 행이 없다.** 평가자가 내 Task 1의 speed/flow를 읽어 나머지를 전부 유도한다:

- density = `q / v`
- link 위 차량 대수 = density × link length
- 경계 유량 = released topology
- ramp 유량 = released ramp observations

즉 **점수를 올리는 유일한 방법은 Task 1 답을 바꾸는 것**이다.

### 채점식

```
S_physics = (1/3)·S_FD + (2/3)·S_LWR
```

**Fundamental Diagram** — sensored mainline link, per-lane, 삼각 FD

```
q_FD(k) = v_f · k              (k ≤ k_crit)
q_FD(k) = w · (k_jam − k)      (k >  k_crit)
```

내 flow와 같은 density에서의 FD-정합 flow 간 정규화 절대차로 계산.
"내 speed와 flow가 실제 도로에서 동시에 일어날 수 있는 조합인가"를 본다.
물리적으로 그럴듯한 답이면 대체로 만족하므로 가중치가 작다.

**Conservation (LWR)** — 5분 스텝 차량 보존

```
N(t + dt) − N(t) = dt · (q_in + q_on_ramp − q_out − q_off_ramp)
```

위반 크기에 따라 연속적으로 감소하며, **혼잡을 지워버린 재구성에서 0에 도달**한다.
개별 셀 오차가 모두 작아도 위반할 수 있다 — 위반은 단일 셀이 아니라 **이웃 link 쌍의 성질**이기 때문.

### Coverage mode

| mode | 조건 | 처리 |
|---|---|---|
| A | 높은 ramp coverage | quality rule을 통과한 on/off-ramp 관측이 보존식을 고정 |
| B | 부분 coverage | ramp 항은 quality rule 통과 셀에서만 포함 |
| C | 낮은 coverage | ramp 부착 transition 제외, ramp-free mainline 보존만 검사 |

**이번 릴리스는 10개 corridor 전부 Mode A, ramp coverage 0.756.** 대회 기간 고정, GitHub에 공개.
참가자가 선택할 수 없다.

### 로컬 채점이 불가능한 이유

`score_task3.py`는 실행되고 숫자를 출력하지만 공개 패키지에서는 **의미가 없다.**
조직자 경계 유량이 없으면 평가자가 내 제출 flow로부터 경계 유량을 추정하는데, 이 추정이 너무 거칠다.

- 보존 신호 크기 ≈ link 위 차량의 **0.75%**
- 로컬 추정 오차 ≈ **0.22%**
- → 신호의 **1/3이 노이즈**. S_LWR이 모두에게 0으로 바닥

**D12_I5_N 실측**

| | 로컬 `score_task3.py` | 실제 채점 |
|---|---|---|
| 완벽한 답 | 0.3305 | **0.9604** |
| naive baseline | 0.3242 | 0.3242 |

baseline이 양쪽에서 같은 숫자인 것은 S_LWR이 어느 쪽이든 0이라 FD 항만 남기 때문이고,
로컬 추정이 구분하지 못하는 쪽은 **완벽한 답**이다.

→ **Task 1로 개발하고, Task 3은 리더보드에게 물어본다.**

---

## 5. Task 4 — Path-flow / OD Estimation (0.20)

### 내용

- 입력: `path_set.csv`, `path_link_incidence.csv`(행렬 A), `synthetic_link_counts.csv`, `synthetic_weak_prior.csv`
- 출력: `path_flow` (finite, ≥ 0)
- 적재 링크 유량: `c_hat = A · f`

### 채점식

```
S_ODME = 0.45·S_od + 0.25·S_link + 0.15·S_dev + 0.15·S_attr
```

| 항 | 내용 |
|---|---|
| `S_od` | 내 path flow ↔ 비공개 reference path flow |
| `S_link` | `c_hat` ↔ released link counts |
| `S_dev` | 내가 prior에서 이동한 거리 ↔ reference가 prior에서 이동한 거리 |
| `S_attr` | destination-attraction 분포 ↔ reference |

`S_dev`는 두 가지 축퇴 해를 막기 위한 항이다 — ① prior를 그대로 제출 ② link count는 완벽히 맞추지만 수요 패턴이 비현실적.

### 구현 노트

기본형은 비음수 제약 최소자승:

```
min ‖A f − c‖²  +  λ‖f − f_prior‖²    s.t.  f ≥ 0
```

λ는 **prior로부터의 이동량이 reference 수준이 되도록** train에서 캘리브레이션한다.

제출의 모든 `path_id`, `origin_zone`, `destination_zone`, link 참조는 released network 파일에 존재해야 한다.

---

## 6. 데이터 패키지

### 패널

```
D7_I10_E    D7_I10_W
D7_I210_E   D7_I210_W
D7_I405_N   D7_I405_S
D12_I5_N    D12_I5_S
D12_I405_N  D12_I405_S    ← Task 2 제외
```

### 디렉토리

```
config/
  synthetic_release_v1.json      캘린더 + 릴리스 계약
  corridors.json                 10개 방향 패널

corridors/<PANEL>/
  network/
    links.csv                    length, lanes, free speed, capacity
    fd_parameters.csv            link별 FD 파라미터 + 차로수
    lwr_mainline_topology.csv    어느 link가 어느 link로 흘러드는지
    ramp_attachment_map.csv      ramp 부착 위치
    path_set.csv                 path + origin/destination zone
    path_link_incidence.csv      path별 사용 link
  train/
    mainline_states/             ★ unmasked (train 전용)
    mainline_states_masked/      Task1 타깃 + Task2 horizon blank
    ramp_states/
  validation/ , private/
    mainline_states_masked/
    ramp_states/

task1/<PANEL>/<split>/   sample_submission_state.csv
task2/<PANEL>/<split>/   window_index.csv, window_history.parquet, sample_submission_queue.csv
task4/<PANEL>/<split>/   synthetic_link_counts.csv, synthetic_weak_prior.csv,
                         sample_submission_path_flow.csv

sample_submission.csv    152 MB  ← 업로드 대상
submission_key.csv       643 MB  ← 읽기 전용
README.md
```

전체 **12.39 GB / 9,688 files**. Task 3은 디렉토리가 없다.

per-corridor 템플릿(`task1/`, `task2/`, `task4/`)은 **train도 커버**한다 — 업로드 템플릿은 train을 포함하지 않는다.
GitHub baseline이 읽는 것도 이 형태이므로 로컬 개발은 여기서 한다.

### 스플릿

| split | 기간 | 일수 | masked 입력 | unmasked 관측 | incident |
|---|---|---|---|---|---|
| train | 2030-06-01 ~ 2031-02-28 | 273 | O | **O** | — |
| validation | 2031-03 | 31 | O | X | 5건 |
| private | 2031-04 | 30 | O | X | 7건 |

캘린더는 합성이며 실제 기간과 일치하지 않는다. 연도에 의미를 부여하지 말 것.

**train 정답 복원**: `mainline_states_masked ⋈ mainline_states` on `(timestamp, station_id, link_id)`.
`score_task1.py`가 정확히 이 조인을 한다.

**validation과 private은 독립 생성**되었다. 서로 다른 수요 draw, 서로 다른 incident 스케줄.
어느 쪽도 다른 쪽의 홀드아웃 표본이 아니다.

### 레코드 스키마

**mainline**: `speed_kmh`, `flow_vph`, `occupancy`, occupancy 유래 `density`, `station_id`, `link_id`, milepost, quality 필드

**ramp**: 검지기 count, quality 필드, 식별자. **speed 컬럼 없음** — ramp 루프는 램프 통과 차량을 셀 뿐 옆 본선을 재지 않는다

> 주의: 릴리스의 `density`는 occupancy 유래인데, Task 3 평가자는 `k = q/v`로 재계산한다.
> 물리 정규화를 걸 때는 평가자와 동일하게 `q/v`를 써야 점수와 정렬된다.

### 로컬 채점 가능 여부

| Task | 가능 | 이유 |
|---|---|---|
| 1 | **예 (train)** | unmasked 관측이 공개됨 |
| 2 | 아니오 | queue 라벨이 모든 split에서 비공개 |
| 3 | 아니오 | 보존식이 참 경계 유량 기준인데, 그게 바로 Task 1이 추정하라는 값 |
| 4 | 부분적 | reference를 로컬에서 유도 → solver 검증용, 순위 예측 아님 |

---

## 7. 제출

### 파일 포맷

```
submission_id,task,speed_kmh,flow_vph,queue_pred,path_flow
1,state,0,0,0,0
```

| task | 채울 것 | 0으로 둘 것 |
|---|---|---|
| `state` | `speed_kmh` (km/h), `flow_vph` (veh/h) | 나머지 |
| `queue` | `queue_pred` ∈ {0, 1} | 나머지 |
| `odme` | `path_flow` (finite, ≥ 0) | 나머지 |

### 평가자가 강제하는 것

- 빈 셀 있으면 채점 전 거부. 해당 없는 컬럼은 0 유지
- `submission_id` 변경 금지, 행 추가 금지 (템플릿에 없는 id → **전체 제출 0점**), 중복 id 금지
- **태스크의 필수 행이 하나라도 빠지면 그 태스크 전체 0점.** 두 태스크만 제출하고 세 번째를 0점 받는 것은 허용
- 파일은 validation·private 두 달치를 모두 포함한다. 어느 행이 어느 split인지 몰라도 되며, 전부 채우면 된다

### 식별자 매핑

`submission_key.csv`:

```
submission_id, task, split, panel, timestamp, station_id, link_id, mask_regime,
window_id, departure_time, path_id, origin_zone, destination_zone
```

읽기 전용. 절대 업로드하지 않는다.

### 로컬 워크플로우

```bash
python src/merge_submissions.py \
  --state state_submission.csv \
  --queue queue_submission.csv \
  --odme  path_flow_submission.csv \
  --output submission.csv
```

Task 3은 파일이 필요 없다.

---

## 8. 규정

### 팀·계정

- 1인 1계정, 1인 1팀, 팀 최대 **4명**
- 팀 병합 마감은 competition settings에 표시
- 팀 외부 비공개 공유 금지. 포럼 공개 공유는 허용 (공개 시 OSI 승인 라이선스로 배포된 것으로 간주)

### 외부 데이터·모델

- 공개 외부 데이터셋, 사전학습 모델, AutoML, API 사용 가능 — 합법·라이선스 적법·타 참가자도 무/저비용 접근 가능
- 결선 진출자는 출처·라이선스와 함께 전부 신고
- 합성 데이터이므로 정답이 들어 있는 외부 소스는 존재하지 않는다

### 금지

- 모델링 외의 방법으로 withheld 라벨을 획득·추론·복원
- 조직자 전용 파일 / 평가자 자산 / reference 라벨 접근·공유
- 타인 계정 사용, 계정·팀 간 제출 조율
- withheld 셀을 외부 소스로 수기 라벨링
- 일반화되는 방법이 아니라 특정 정답 식별이 목적인 리소스 사용

> **데이터를 생성한 교통 물리를 모델링하는 것은 금지가 아니라 벤치마크의 취지다.**

### 재현성

수상 확정 전 조직자가 요구:

- 제출 파일을 만든 코드
- 모델·의존성 버전
- 전처리 단계
- random seed
- 외부 데이터·라이선스 신고
- 재현 명령어, 런타임, 필요 하드웨어

제출 방식: 조직자 계정과 공유한 Kaggle Notebook, 또는 clone 가능한 repo.
어느 쪽이든 **릴리스 데이터로 end-to-end 실행되어 선택한 제출 파일을 재생성**해야 한다.

- 기한 내 미제출 → 수상 자격 상실
- 코드가 제출 파일을 재현하지 못해도 자격 상실
- **팀 간 코드·예측값을 실제로 대조한다.** 760만 셀에서 거의 동일한 예측이 나오는 독립 방법은 없다는 전제
  → 유사 제출은 양쪽 모두 실격

### 데이터 재빌드 (2026-09-11)

Task 2 정답의 일부를 예측이 아니라 릴리스에서 읽어낼 수 있는 경로가 두 개 발견되었다.
그 자리에서 가리는 것만으로는 불충분했다 — 이미 받아간 사본은 되돌릴 수 없기 때문.

- validation·private을 새 시드로 재생성. 데이터·솔루션 파일·메트릭을 함께 교체
- **재빌드 이후 제출만 수상 자격.** 9/10부터 재빌드 완료까지 제출이 닫혀 있어 경계에 걸치는 제출은 없다
- 재빌드 이전 점수는 존재하지 않는 데이터로 계산된 것이며 비교 불가. Kaggle은 제출 후 재채점하지 않는다
- 코드는 바꿀 필요 없다 — 태스크·채점식·가중치·제출 포맷 모두 동일

### Kaggle 기본 규정

- 만 18세 이상 (또는 거주지 성년 연령 중 더 높은 쪽)
- **동점 시 먼저 제출한 쪽이 승리**
- 최종 순위는 private 리더보드가 단독 결정

---

## 9. 일정과 상금

| 항목 | 날짜 |
|---|---|
| 데이터 릴리스 | 2026-06-01 |
| 데이터 재빌드 | 2026-09-11 |
| **최종 제출 마감** | **2026-11-06** |
| IEEE BigData 2026 컨퍼런스 | 2026-12-14 ~ 17, Phoenix, Arizona |

| 상 | 금액 |
|---|---|
| Gold | USD 1,500 |
| Silver | USD 1,000 |
| Bronze | USD 500 |
| Student / Newcomer | USD 500 |

리더보드 하나, 상 한 세트. 수상은 private 리더보드로 결정.

---

## 10. 어느 페이지가 우선하는가

| 항목 | 근거 |
|---|---|
| 마감일, 일일 제출 한도, 최종 제출 개수, 팀 크기 | competition settings (Overview / Timeline) |
| 채점 | Overview 페이지의 Evaluation 섹션 |
| 파일 내용·스키마 | Data 페이지 |
| 그 외 | Kaggle General Competition Rules |

Rules 페이지와 Evaluation 섹션이 충돌하면 **Evaluation이 우선**한다.
Kaggle Foundational Rules는 대회별 규정과 충돌 시 항상 우선한다.