<div align="center">

# 🌱 콩잎 질병 분류 · Fine-tuning부터 동기 · 비동기 서빙까지

**ResNet-18 · FastAPI · Redis · Celery · PostgreSQL · Docker Compose**

</div>

콩잎 사진으로 `microsoft/resnet-18`을 **각진잎반점병 · 녹병 · 정상** 3클래스로 Fine-tuning하고, 동기 API · Redis 캐시 · Celery 비동기 작업 · PostgreSQL 작업 기록을 Docker Compose로 서비스했습니다.
제공 코드의 약점 4가지를 하나씩 실험으로 확인해 고쳤고, 모델은 **Validation으로 고른 뒤 Test에 한 번** 평가했습니다.

## 한눈에 보기

<p align="center"><img src="reports/figures/architecture.png" alt="시스템 아키텍처" width="100%"></p>

<table>
  <tr><th width="12%"></th><th width="44%">① 모델</th><th width="44%">② 서비스</th></tr>
  <tr>
    <td align="center"><b>한 일</b></td>
    <td>데이터 점검 → Head-only 기준선 → 약점 4가지 실험 (E2 · E3 · E1 · E4) → Val 선택 → Test 1회 → 강건성 평가</td>
    <td>동기 API + Redis 캐시 · Celery 비동기 · PostgreSQL 기록 · 모델 교체(E0 → E4) · worker 정지 테스트</td>
  </tr>
  <tr>
    <td align="center"><b>핵심 숫자</b></td>
    <td>Test <b>0.7188 → 0.9844</b> (126/128) · 변형 8종 평균 <b>0.9570</b> · 병든 잎을 정상으로 놓침 <b>13 → 0</b>장</td>
    <td>캐시 HIT <b>−93%</b> (27.4 → 1.9 ms) · 비동기 접수 <b>8.6 ms</b> · 실환경 테스트 <b>23 / 23 PASS</b></td>
  </tr>
  <tr>
    <td align="center"><b>기록</b></td>
    <td><a href="reports/01_FineTuning_결과보고서.md">산출물 1</a> · <a href="reports/09_성능개선_실험보고서.md">산출물 9</a></td>
    <td><a href="#산출물">산출물 2~7</a> · <a href="reports/08_최종기술보고서.md">최종 기술보고서</a></td>
  </tr>
</table>

### 실험 — 무엇을 바꿨나

| 실험 | 바꾼 것 | 학습 범위 | 학습률 | epoch | Validation | Test |
|---|---|---|---:|---:|---:|---:|
| **E0** 기준선 (필수) | 제공 명령 그대로 (`--freeze-backbone`) | 분류층만 (1,539개) | 0.0002 | 3 | 0.7669 | 0.7188 |
| **E2** | 학습률 · epoch ↑ (약점 1: 덜 학습됨) | 분류층만 | 0.001 | 10 | 0.9398 | — |
| **E3** | Backbone 고정만 해제 (약점 2) | 전체 (약 1,170만 개) | 0.0002 | 3 | 0.9624 | — |
| **E1** | E3 + 학습률 ↓ (평가가이드 선택 실험) | 전체 | 0.00005 | 3 | 0.9699 | — |
| **E4** ⭐ | LP-FT + 계층별 학습률 + 증강 + 코사인 (약점 4: 과적합) | 2 epoch 분류층 → 전체 | 0.00005 (분류층 ×20) | 15 (저장 12) | **0.9850** | **0.9844** |

<p align="center"><img src="reports/figures/improvement_steps.png" alt="개선 단계" width="90%"></p>

- **약점 1 · 2:** 학습량만 늘려도 +17.29%p, Backbone 고정만 풀어도 +19.55%p로 두 원인이 비슷하게 컸다. 약점 3(BatchNorm 통계 흔들림)은 코드 테스트로 확인하고 `--freeze-bn` 옵션만 추가했다.
- **약점 4:** 증강 없는 Full 학습(E3 · E1)은 epoch 1 이후 Val이 계속 떨어졌다. E4는 후반까지 유지했고, Train–Val 차이가 2.62 → 1.11%p로 줄었다. 증강은 크롭 · 반전 · 회전 · 밝기만 주고, 병반 색이 단서라 **색조는 거의 바꾸지 않았다.**
- **Test는 E4와 비교용 E0만 1회** 평가했다(여러 모델의 Test를 보고 고르지 않기 위해). 제공 코드는 수정하지 않고 `training/finetune_v2.py` 등을 따로 추가했다(md5로 원본 일치 확인).

## 산출물

| # | 문서 | 핵심 |
|---:|---|---|
| 1 | [Fine-tuning 결과보고서](reports/01_FineTuning_결과보고서.md) | 데이터 점검 · E0 / E1 · 출력층 교체 · 재현성 |
| 2 | [시스템 아키텍처](reports/02_시스템_아키텍처.md) | Docker Compose 4서비스 · Redis DB 1/2/3 역할 |
| 3 | [API 명세서](reports/03_API_명세서.md) | 4개 API · 실제 응답 JSON · 400/404/413/503 |
| 4 | [DB 구조 · 작업이력](reports/04_DB_구조_작업이력.md) | ERD · 상태 전이 · 실제 행 |
| 5 | [AI 버튼 처리흐름](reports/05_AI버튼_처리흐름.md) | 동기 · 비동기 시퀀스도 · 코드 대응표 |
| 6 | [성능 비교](reports/06_성능비교_보고서.md) | `timings.csv` · 캐시 · 비동기 해석 |
| 7 | [기능 테스트 · 오류](reports/07_기능테스트_오류분석.md) | 실환경 23건 · 모의 테스트 · 오류 9건 |
| 8 | [**최종 기술보고서**](reports/08_최종기술보고서.md) | 종합 · 개선 제안 · 한계 · 실행 방법 · 트러블슈팅 |
| 9 | [성능 개선 실험 보고서](reports/09_성능개선_실험보고서.md) (추가) | 약점 4가지 → 원리 → 실험 · 최종 모델 평가 · 강건성 · 논문 |

<details>
<summary><b>빠른 실행</b> — 데이터 배치 → 학습 → <code>docker compose up</code> (데이터 · 모델 · <code>.env</code>는 저장소에 없음)</summary>

```bash
# 1. 환경 (WSL2 Ubuntu 22.04 · Python 3.11 · Docker Desktop WSL Integration)
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. 데이터: 원본은 보존하고, 제공 코드가 읽는 dataset/train · val · test 로 복사 (validation → val)
mkdir -p dataset/{train,val,test}
cp -r ~/beans_source/train/* dataset/train/ && cp -r ~/beans_source/validation/* dataset/val/ && cp -r ~/beans_source/test/* dataset/test/

# 3. 학습 (GPU) — E0 기준선, E4 서비스 모델
python -m training.finetune --data dataset --output artifacts/beans_resnet18 --epochs 3 --batch-size 8 --freeze-backbone
python -u -m training.finetune_v2 --output artifacts/e4_lpft_aug_cosine --epochs 15 --batch-size 8 \
  --lp-epochs 2 --lr 0.00005 --head-lr-mult 20 --augment --scheduler cosine --warmup-epochs 1 --patience 5
python -m training.evaluate --model artifacts/e4_lpft_aug_cosine --split test --out reports/data/eval_test_e4.json

# 4. 서비스
cp .env.example .env                                   # 기본: MODEL_PATH=artifacts/beans_resnet18 (E0)
sed -i 's#^MODEL_PATH=.*#MODEL_PATH=artifacts/e4_lpft_aug_cosine#' .env
mkdir -p media/jobs && docker compose up -d --build
curl -s http://127.0.0.1:8000/health                   # 화면: /vision/sync/ · /vision/async/ · /docs
docker compose stop                                    # 정지 (down -v 는 DB · 캐시를 지우므로 사용 안 함)

# 5. 테스트 · 측정
docker compose exec web python -m pytest -q tests/test_api.py           # 4 passed
python -m pytest -q tests/test_training.py tests/test_robustness.py    # 10 passed
python scripts/measure_timings.py                                      # → reports/data/timings.csv
```

전체 명령 · 트러블슈팅 → [최종 기술보고서 §10](reports/08_최종기술보고서.md#10-실행-방법--기술-스택--트러블슈팅)

</details>

<details>
<summary><b>추가한 도구</b> — 제공 코드는 수정하지 않고 파일만 추가</summary>

| 파일 | 하는 일 |
|---|---|
| `training/finetune_v2.py` | 증강 · LP-FT · 계층별 학습률 · 코사인 · freeze-bn · 조기 종료 (E4) |
| `training/evaluate.py` | 정확도 · 혼동행렬 · 클래스별 P/R/F1 · 틀린 이미지 목록 |
| `training/robustness.py` | 밝기 · 흐림 · 노이즈 · JPEG · 저해상도 등 8가지 변형 평가 |
| `scripts/measure_timings.py` | 동기 / 비동기 × MISS / HIT 반복 측정, 워밍업 분리 |
| `scripts/make_final_eval_figures.py` | `reports/data` 기록만 읽어 혼동행렬 · 비교 그래프 생성 |
| `tests/test_training.py` · `tests/test_robustness.py` | 학습 · 평가 · 강건성 코드 단위 테스트 (10개) |

</details>

<details>
<summary><b>폴더 구조</b></summary>

```text
├── ai/ · main.py          # 제공 코드: 모델 로드 · 검증 · 캐시 · Celery · DB · 라우터
├── training/              # 제공 finetune.py + 추가 finetune_v2 · evaluate · robustness
├── scripts/ · tests/      # 측정 · 그림 스크립트 / 모의 · 단위 테스트
├── templates/ · static/   # 제공 웹 화면
├── compose.yaml · Dockerfile · requirements.txt · .env.example
└── reports/
    ├── 01_*.md ~ 09_*.md  # 산출물
    ├── data/              # 원본 기록 (학습 리포트 · 평가 · 강건성 · 응답 시간)
    ├── figures/           # 구성도 · 시퀀스도 · ERD · 그래프
    └── evidence/          # 실행 캡처 · 학습 로그
```

</details>

<details>
<summary><b>환경 · 보안 · 한계</b></summary>

- **환경:** WSL2 Ubuntu 22.04 · Python 3.11 · torch 2.5.1 · transformers 4.57.6 · RTX 3060(학습만) · Docker 29.9.2 / Compose v5.6.1 · 컨테이너 추론은 CPU
- **보안:** `.env` · `dataset/` · `artifacts/` · `media/jobs/`는 Git 제외. `compose.yaml`의 DB 비밀번호는 실습용 값이고 포트는 `127.0.0.1`에만 연다.
- **한계:** 실험당 seed 1개 · Test 128장 · 단일 출처 데이터 · E4 기법별 기여도 미분리 · 저해상도 0.9141 / 흐림 0.9375에 약함 · 응답 시간은 E0 기준 · 동시 부하 미측정
- **출처:** 데이터 [AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans) (MIT) · 모델 [microsoft/resnet-18](https://huggingface.co/microsoft/resnet-18) (Apache-2.0) · 참고 논문 → [산출물 9](reports/09_성능개선_실험보고서.md#참고-논문)

</details>
