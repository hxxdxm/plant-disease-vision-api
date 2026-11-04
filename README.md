<div align="center">

# 🌱 Plant Disease Vision API

**콩잎 사진 한 장으로 잎의 질병을 분류하는 AI 웹 서비스**

ResNet18을 콩잎 데이터로 Fine-tuning하고, FastAPI로 **동기 · 비동기** 두 방식으로 서빙합니다.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_Transformers-ResNet18-FFD21E)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![Celery](https://img.shields.io/badge/Celery-5.6-37814A?logo=celery&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-7.4-DC382D?logo=redis&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-4169E1?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker_Compose-4_services-2496ED?logo=docker&logoColor=white)

[빠른 시작](#-빠른-시작) · [결과](#-결과-한눈에-보기) · [API](#-api) · [보고서](#-문서산출물)

</div>

---

<details>
<summary><b>📑 목차</b></summary>

1. [프로젝트 소개](#-프로젝트-소개)
2. [주요 기능](#-주요-기능)
3. [데모](#-데모)
4. [결과 한눈에 보기](#-결과-한눈에-보기)
5. [시스템 아키텍처](#-시스템-아키텍처)
6. [기술 스택](#-기술-스택)
7. [빠른 시작](#-빠른-시작)
8. [API](#-api)
9. [테스트와 성능 측정](#-테스트와-성능-측정)
10. [프로젝트 구조](#-프로젝트-구조)
11. [문서(산출물)](#-문서산출물)
12. [트러블슈팅](#-트러블슈팅)
13. [한계와 향후 과제](#-한계와-향후-과제)
14. [출처와 라이선스](#-출처와-라이선스)

</details>

## 📌 프로젝트 소개

콩잎 사진을 올리면 아래 **3개 클래스** 중 하나로 분류합니다(이미지 분류. 병반의 위치를 찾는 객체 검출은 범위에 넣지 않았습니다).

| 클래스 | 의미 |
|---|---|
| `angular_leaf_spot` | 각진잎반점병 |
| `bean_rust` | 콩 녹병 |
| `healthy` | 정상 |

**핵심 내용**

- ImageNet 1,000클래스로 사전학습된 `microsoft/resnet-18`의 출력층을 **3클래스로 교체**하고 콩잎 데이터로 추가 학습했습니다.
- 같은 모델을 **동기 API**(결과까지 바로 응답)와 **비동기 API**(접수 후 Celery worker가 처리)로 서빙하고, 두 방식의 응답 시간을 **실측으로 비교**했습니다.
- **Redis 캐시**로 같은 이미지에 대한 추론을 생략하고, **PostgreSQL**에 비동기 작업 이력을 남깁니다.

## ✨ 주요 기능

- 🧠 **Fine-tuning** — Head-only(Backbone 고정)와 Full Fine-tuning을 같은 Validation 데이터로 비교
- ⚡ **동기 분류** — `POST /api/sync/classify/` → 200 + 결과
- 📨 **비동기 분류** — `POST /api/async/classify/` → 202 + `job_id` → 상태 조회로 결과 확인
- 🗃️ **Redis 캐시** — 이미지 내용(sha256)과 모델 경로 기준으로 600초 동안 결과 재사용, 캐시 · 브로커 · 결과를 DB 번호로 분리
- 🧾 **작업 이력** — `image_jobs` 테이블에 `queued → running → done / failed` 상태 기록
- 🛡️ **입력 검증** — JPG · PNG · WEBP만 허용, 5MB 초과는 413, 잘못된 파일은 400
- 🐳 **한 번에 실행** — `docker compose up` 하나로 web · worker · Redis · PostgreSQL 기동

## 🎬 데모

| 동기 분류 (`/vision/sync/`) | 비동기 분류 (`/vision/async/`) |
|:---:|:---:|
| <img src="reports/evidence/01_sync_ui.png" width="420"> | <img src="reports/evidence/02_async_ui.png" width="420"> |
| `bean_rust` 사진 → **bean_rust** 0.6051<br>결과까지 **35ms** | `bean_rust` 사진 → **bean_rust** 0.5148<br>접수 **22ms** → 1초 간격 조회로 done 확인 |

## 📊 결과 한눈에 보기

### 모델

| 학습 방식 | Best Validation Accuracy | 비고 |
|---|:---:|---|
| Head-only Fine-tuning | **0.7669** (102/133) | 필수 실험 · **서비스에 사용한 모델** |
| Full Fine-tuning | **0.9699** (129/133) | 선택 실험 · 같은 Validation에서 +20.30%p |

<img src="reports/figures/finetune_compare.png" width="760">

> Full Fine-tuning은 학습률도 함께 바꿨고(0.0002 → 0.00005), epoch가 늘수록 Val Acc가 조금씩 떨어지는 과적합 신호가 있었습니다. Test Accuracy는 두 모델 모두 측정하지 않았습니다. 자세한 내용 → [산출물 1](reports/01_FineTuning_결과보고서.md)

### 응답 시간 (5회 중앙값, CPU 추론)

| 경로 | 캐시 | HTTP 응답 | 결과 확인까지 | 모델 추론 |
|---|:---:|---:|---:|---:|
| 동기 | MISS | 27.41 ms | 27.41 ms | 21.56 ms |
| 동기 | HIT | **1.90 ms** | 1.90 ms | 생략 |
| 비동기 | MISS | **8.63 ms** (202 접수) | 63.26 ms | 20.94 ms |
| 비동기 | HIT | 8.16 ms (202 접수) | 63.14 ms | 생략 |

- **캐시 HIT → 응답 93.07% 감소**: 같은 이미지는 추론을 건너뜁니다.
- **비동기 접수 → 동기보다 68.52% 빠른 응답**: 추론이 빨라진 게 아니라, 접수와 처리를 분리한 효과입니다.
- **결과를 받기까지는 비동기가 35.85ms 더 느립니다**: 큐 · DB · 상태 조회 비용 때문입니다.

<img src="reports/figures/timings_chart.png" width="640">

자세한 내용 → [산출물 6](reports/06_성능비교_보고서.md)

## 🏗️ 시스템 아키텍처

![architecture](reports/figures/architecture.png)

| 흐름 | 경로 |
|---|---|
| **동기** | 브라우저 → FastAPI → Redis 캐시 확인 → (MISS면) ResNet18 → 200 JSON |
| **비동기** | 브라우저 → FastAPI(검증 · 파일 저장 · DB `queued`) → Redis 브로커 → Celery worker → 캐시/ResNet18 → DB `done` → 브라우저가 상태 조회 |

| Redis DB | 역할 |
|:---:|---|
| 1 | 분류 결과 캐시 (TTL 600초) |
| 2 | Celery 브로커 |
| 3 | Celery 결과 백엔드 |

모델은 web과 worker **각 프로세스에서 따로 로드**됩니다(첫 요청 약 2.3초, 이후 약 20ms). 처리 순서 → [산출물 5](reports/05_AI버튼_처리흐름.md)

## 🧰 기술 스택

| 구분 | 사용 기술 |
|---|---|
| 모델 · 학습 | Hugging Face Transformers, PyTorch, `microsoft/resnet-18` |
| API 서버 | FastAPI, Uvicorn, Jinja2 |
| 비동기 처리 | Celery(solo, concurrency=1), Redis |
| 데이터베이스 | PostgreSQL 17, SQLAlchemy 2.0 |
| 인프라 | Docker Compose (web · worker · redis · postgres) |
| 테스트 | pytest, FastAPI TestClient |
| 개발 환경 | Windows + WSL2 Ubuntu 22.04 · i7-12700 · RTX 3060(학습만 사용) |

## 🚀 빠른 시작

**준비물:** Docker Desktop(WSL Integration 켜기), Python 3.11, 학습용 GPU(선택)

### 1. 데이터 받기

```bash
mkdir -p ~/beans_source && cd ~/beans_source
for f in train validation test; do
  wget -O $f.zip https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/$f.zip
done
unzip -q train.zip && unzip -q validation.zip && unzip -q test.zip

cd ~/fastapi_resnet18
mkdir -p dataset/{train,val,test}
cp -r ~/beans_source/train/*      dataset/train/
cp -r ~/beans_source/validation/* dataset/val/     # validation → val
cp -r ~/beans_source/test/*       dataset/test/
```

### 2. 모델 학습

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python -m training.finetune --data dataset --output artifacts/beans_resnet18 \
  --epochs 3 --batch-size 8 --freeze-backbone

cp .env.example .env        # MODEL_PATH=artifacts/beans_resnet18
```

<details>
<summary>선택 실험: Full Fine-tuning · GPU 없이 학습하기</summary>

```bash
# Backbone까지 학습 (별도 폴더에 저장, 서비스 모델은 그대로)
python -m training.finetune --data dataset --output artifacts/beans_resnet18_full \
  --epochs 3 --batch-size 8 --lr 0.00005

# GPU가 없으면 컨테이너에서 학습 (CPU)
docker compose run --rm trainer
```
</details>

### 3. 서비스 실행

```bash
mkdir -p media/jobs
docker compose up -d --build
docker compose ps                    # 4개 Up, redis · postgres · web (healthy)
curl http://127.0.0.1:8000/health    # {"status":"ok"}
```

| 주소 | 내용 |
|---|---|
| http://127.0.0.1:8000/vision/sync/ | 동기 분류 화면 |
| http://127.0.0.1:8000/vision/async/ | 비동기 분류 화면 |
| http://127.0.0.1:8000/docs | Swagger API 문서 |

<details>
<summary>재시작 · 종료 · 모델 교체</summary>

```bash
docker compose logs -f --tail=100 web worker      # 로그
docker compose up -d --force-recreate web worker  # .env의 MODEL_PATH나 모델을 바꿨을 때
docker compose stop                               # 정지 (데이터 보존)
docker compose up -d                              # 다시 시작
```
`docker compose down -v`는 DB · Redis 데이터를 지우니 쓰지 마세요.
</details>

## 🔌 API

| Method | URL | 설명 | 성공 | 오류 |
|---|---|---|:---:|---|
| `GET` | `/health` | 서버 상태 | 200 | — |
| `POST` | `/api/sync/classify/` | 동기 분류 (`image` 파일) | 200 | 400 · 413 · 422 |
| `POST` | `/api/async/classify/` | 비동기 작업 접수 | **202** | 400 · 413 · 422 · 503 |
| `GET` | `/api/async/jobs/{job_id}/` | 작업 상태 · 결과 조회 | 200 | 404 |

```bash
IMG=dataset/test/healthy/healthy_test.39.jpg
curl -s -X POST -F "image=@$IMG" http://127.0.0.1:8000/api/sync/classify/
```

```json
{
  "model_id": "artifacts/beans_resnet18",
  "top1": {"label": "healthy", "score": 0.8066},
  "all": [
    {"label": "healthy", "score": 0.8066},
    {"label": "bean_rust", "score": 0.1229},
    {"label": "angular_leaf_spot", "score": 0.0705}
  ],
  "inference_ms": 2282.39,
  "cache_hit": false
}
```

<details>
<summary>비동기 요청 예시</summary>

```bash
curl -s -X POST -F "image=@$IMG" http://127.0.0.1:8000/api/async/classify/
# {"job_id":1,"status":"queued","check_url":"/api/async/jobs/1/"}   ← HTTP 202 (접수만)

curl -s http://127.0.0.1:8000/api/async/jobs/1/
# {"job_id":1,"status":"done","top1":{"label":"bean_rust","score":0.5152}, ... ,"cache_hit":false}
```
`202`는 **접수**, 상태 조회의 `"status": "done"`이 **완료**입니다. 캐시 HIT이면 `inference_ms`가 `null`입니다(0ms가 아니라 추론 생략).
</details>

전체 명세 → [산출물 3](reports/03_API_명세서.md) · DB 구조 → [산출물 4](reports/04_DB_구조_작업이력.md)

## 🧪 테스트와 성능 측정

```bash
# 모의 통합 테스트 (가짜 모델 · SQLite) → 4 passed
docker compose exec web python -m pytest -q tests/test_api.py

# 동기 · 비동기 · 캐시 응답 시간 측정 → reports/data/timings.csv
python scripts/measure_timings.py
```

| 구분 | 결과 |
|---|---|
| 실환경 통합 테스트 | **22건 PASS** — 3클래스 분류, 202 → done, 400 · 404 · 413, 캐시 MISS → HIT, worker 정지 시 `queued` 유지 → 재시작 후 처리, Redis DB 역할, DB 스키마, Swagger |
| 모의 테스트 | 01 `1 passed` · 02 `4 passed` (503 · `failed` 기록 포함) |
| 오류 기록 | 실제로 겪은 오류 8건의 원인 · 조치 · 재검증 |

자세한 내용 → [산출물 7](reports/07_기능테스트_오류분석.md)

## 📁 프로젝트 구조

```text
plant-disease-vision-api/
├── ai/                        # 서비스 코드
│   ├── config.py              #   환경 변수 · 경로 · 업로드 제한
│   ├── hf_model.py            #   ResNet18 로드 (프로세스당 1회)
│   ├── services.py            #   이미지 검증 · 추론 · 캐시 키
│   ├── cache_service.py       #   Redis 캐시 HIT/MISS
│   ├── database.py · models.py#   PostgreSQL 연결 · image_jobs 테이블
│   ├── celery_app.py · tasks.py   # Celery 설정 · 비동기 분류 작업
│   └── router_sync.py · router_async.py · router_web.py
├── training/finetune.py       # Fine-tuning 학습
├── templates/ · static/       # 동기 · 비동기 화면
├── tests/test_api.py          # 통합 모의 테스트 (02)
├── test_api.py                # 동기 API 모의 테스트 (01)
├── scripts/measure_timings.py # 성능 측정
├── reports/                   # 산출물 1~8 · 증빙 · 그림 · 측정 원본
├── main.py                    # FastAPI 앱
├── Dockerfile · compose.yaml  # web · worker · redis · postgres (+ trainer)
└── requirements.txt · .env.example
```

`dataset/`, `artifacts/`(모델 가중치), `media/jobs/`(업로드 이미지), `.env`는 `.gitignore`로 저장소에서 제외했습니다.

## 📚 문서(산출물)

| # | 문서 | 내용 |
|:---:|---|---|
| 1 | [Fine-tuning 결과보고서](reports/01_FineTuning_결과보고서.md) | 데이터 구성 · 학습 설정 · Head-only vs Full 비교 |
| 2 | [시스템 아키텍처](reports/02_시스템_아키텍처.md) | 구성도 · 컴포넌트 역할 · 실행 환경 |
| 3 | [API 명세서](reports/03_API_명세서.md) | 요청 · 응답 · 상태코드 · 실행 근거 |
| 4 | [DB 구조 · 작업이력](reports/04_DB_구조_작업이력.md) | `image_jobs` ERD · 상태 전이 · 실제 행 |
| 5 | [AI 버튼 처리흐름](reports/05_AI버튼_처리흐름.md) | 동기 · 비동기 시퀀스도 · 코드 대응표 |
| 6 | [성능 비교 보고서](reports/06_성능비교_보고서.md) | 응답 시간 · 캐시 효과 · 측정 한계 |
| 7 | [기능 테스트 · 오류 분석](reports/07_기능테스트_오류분석.md) | 테스트 22건 · 오류 8건 |
| 8 | [최종 기술보고서](reports/08_최종기술보고서.md) | 전체 요약 · OpenCV 개선 제안 · 한계 |

증빙 캡처 `reports/evidence/` · 그림 `reports/figures/` · 측정 원본 `reports/data/`

## 🔧 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `Duplicate dispatch rule for <built-in function intern>` | Ubuntu 22.04의 Python 3.11.0rc1 + 최신 torch | `torch==2.5.1`, `torchvision==0.20.1` 설치 |
| `pytest`에서 `No module named 'ai'` | `pytest`가 현재 폴더를 import 경로에 넣지 않음 | **`python -m pytest`**로 실행 |
| 비동기 작업이 계속 `queued` | worker 미실행 | `docker compose logs worker` 확인 후 `docker compose start worker` |
| 1,000개 ImageNet 라벨이 나옴 | `.env`가 원본 모델을 가리킴 | `MODEL_PATH=artifacts/beans_resnet18` 후 web · worker 재생성 |
| `address already in use` | 8000 · 6379 · 5432 포트를 다른 프로그램이 사용 중 | 로컬 uvicorn 등 기존 프로그램 종료 |

## 🧭 한계와 향후 과제

**한계**

- 서비스 모델(Head-only)의 Validation Accuracy는 0.7669입니다. 실제로 `bean_rust` 사진을 `healthy`로 틀린 사례가 있습니다.
- Test Accuracy와 클래스별 지표(Precision · Recall · 혼동행렬)는 측정하지 않았습니다.
- 단일 사용자의 순차 요청만 측정했고, 동시 요청 처리량은 측정하지 않았습니다.
- worker가 root 권한으로 실행되고, DB 비밀번호가 `compose.yaml`에 그대로 들어 있습니다. 로컬 실습 환경 기준입니다.

**향후 과제**

- [ ] 학습률을 같게 두고 Backbone 고정 여부만 바꿔서 Full Fine-tuning 효과를 분리해 확인
- [ ] 최종 모델을 Test 데이터로 1회 평가 (정확도 · 혼동행렬)
- [ ] Full Fine-tuning 모델을 서비스에 적용하고 다시 검증
- [ ] Grad-CAM으로 오분류 원인 분석 → OpenCV HSV · 잎 ROI 규칙 실험 (설계만 완료, [산출물 8](reports/08_최종기술보고서.md) §8)
- [ ] 동시 요청 부하 테스트

## 📄 출처와 라이선스

| 항목 | 출처 | 라이선스(각 페이지 표기) |
|---|---|---|
| 모델 | [microsoft/resnet-18](https://huggingface.co/microsoft/resnet-18) | Apache-2.0 |
| 데이터 | [AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans) | MIT |

데이터셋과 학습된 가중치는 저장소에 포함하지 않았습니다. 사용하거나 공유하기 전에 각 저장소의 이용 조건을 확인하세요.
