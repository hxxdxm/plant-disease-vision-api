# 🌱 Plant Disease Vision API — 콩잎 질병 분류 서비스

콩잎 사진 한 장을 올리면 **각진잎반점병 · 콩 녹병 · 정상** 중 하나로 분류하는 AI 웹 서비스입니다.
Hugging Face `microsoft/resnet-18`을 콩잎 데이터로 Fine-tuning하고, FastAPI로 **동기·비동기** 두 방식으로 서빙합니다.

| | |
|---|---|
| 모델 | `microsoft/resnet-18` → Head-only Fine-tuning (3클래스) |
| 데이터 | [AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans) |
| 서빙 | FastAPI · Celery · Redis · PostgreSQL · Docker Compose |
| 결과 | **Validation Accuracy 0.7669** · 캐시 HIT 시 응답 **93% 감소** · 비동기 접수 응답 **68% 단축** |

![아키텍처](reports/figures/architecture.png)

---

## 1. 환경

- Windows + WSL2 Ubuntu 22.04, Python 3.11 가상환경(학습용)
- Docker Desktop / Docker Compose (WSL Integration 켜기)
- Docker 서비스: `web`(FastAPI :8000) · `worker`(Celery) · `redis`(:6379) · `postgres`(:5432)
- 확인한 사양: i7-12700 · RAM 15GiB · RTX 3060 12GB(학습에만 사용, 서빙은 CPU)

## 2. 데이터 준비

```bash
mkdir -p ~/beans_source && cd ~/beans_source
for f in train validation test; do
  wget -O $f.zip https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/$f.zip
done
sudo apt install -y unzip
unzip -q train.zip && unzip -q validation.zip && unzip -q test.zip

cd ~/fastapi_resnet18
mkdir -p dataset/{train,val,test}
cp -r ~/beans_source/train/*      dataset/train/
cp -r ~/beans_source/validation/* dataset/val/     # validation → val (코드가 val을 읽음)
cp -r ~/beans_source/test/*       dataset/test/
```

- `dataset/train`, `dataset/val`, `dataset/test` 모두 `angular_leaf_spot` · `bean_rust` · `healthy` 3개 폴더
- 확인한 수: Train 1,035 · Val 133 · Test 128 (Train의 `healthy_train.120tore` 1개는 확장자 문제로 학습에서 제외됨)
- `dataset/`은 용량 때문에 저장소에 포함하지 않습니다. 공유 전 데이터셋 이용 조건(페이지 표기 MIT)을 확인하세요.

## 3. 모델 준비 (Fine-tuning)

```bash
cd ~/fastapi_resnet18
sudo apt install -y python3.11 python3.11-venv
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python -m training.finetune --data dataset --output artifacts/beans_resnet18 \
  --epochs 3 --batch-size 8 --freeze-backbone

echo "MODEL_PATH=artifacts/beans_resnet18" > .env     # 또는 cp .env.example .env
```

> ⚠️ Ubuntu 22.04 기본 패키지의 Python 3.11.0rc1에 최신 torch(2.15)를 설치하면 `Duplicate dispatch rule for <built-in function intern>` 오류가 납니다. 위처럼 **torch 2.5.1**(Dockerfile과 같은 버전)을 먼저 설치하세요.

- 결과: `artifacts/beans_resnet18/` (`config.json`, `model.safetensors`, `preprocessor_config.json`, `training_report.json`)
- GPU가 없으면 `docker compose run --rm trainer`로 컨테이너에서 학습할 수도 있습니다(CPU).
- 학습된 가중치는 저장소에 포함하지 않습니다. 위 명령으로 다시 만들거나 허용된 경로로 따로 전달합니다.

## 4. 실행

```bash
cd ~/fastapi_resnet18
mkdir -p media/jobs
docker compose up -d --build        # redis · postgres · web · worker 한 번에 기동
docker compose ps                   # 4개 Up, redis·postgres·web (healthy)
docker compose logs --tail=40 web worker
curl http://127.0.0.1:8000/health   # {"status":"ok"}
```

| 주소 | 용도 |
|---|---|
| http://127.0.0.1:8000/vision/sync/ | 동기 분류 화면 |
| http://127.0.0.1:8000/vision/async/ | 비동기 분류 화면 |
| http://127.0.0.1:8000/docs | Swagger API 문서 |

```bash
# 모델 파일이나 .env의 MODEL_PATH를 바꿨을 때
docker compose up -d --force-recreate web worker
# 정지(데이터 보존) / 재시작
docker compose stop
docker compose up -d
```

`docker compose down -v`는 DB·Redis 볼륨을 지우니 사용하지 마세요. 01 단계에서 로컬 uvicorn을 켜 두었다면 8000번 포트가 겹치니 먼저 끄세요.

## 5. 확인할 응답

```bash
IMG="$(find dataset/test/healthy -maxdepth 1 -type f -iname '*.jpg' -print -quit)"

# 동기: 결과까지 바로 (같은 파일 2번째는 cache_hit=true, inference_ms=null)
curl -s -X POST -F "image=@${IMG}" http://127.0.0.1:8000/api/sync/classify/

# 비동기: 202 + job_id → 상태 조회
curl -i -X POST -F "image=@${IMG}" http://127.0.0.1:8000/api/async/classify/
curl -s http://127.0.0.1:8000/api/async/jobs/1/          # 실제 job_id로 교체

# PostgreSQL 작업 이력 (비동기만 기록)
docker compose exec postgres psql -U vision_user -d vision_db -c \
  'SELECT id, status, model_id, top1_label, cache_hit, inference_ms FROM image_jobs ORDER BY id DESC LIMIT 10;'
```

## 6. 테스트 · 측정

```bash
# 모의 통합 테스트 (가짜 모델·SQLite) → 4 passed
docker compose exec web python -m pytest -q tests/test_api.py
```

> `pytest -q ...`로 실행하면 `ModuleNotFoundError: No module named 'ai'`가 납니다. **`python -m pytest`**로 실행하세요(현재 폴더가 import 경로에 들어갑니다).

```bash
# 동기·비동기·캐시 응답시간 5회 측정 → reports/data/timings.csv
source .venv/bin/activate
python scripts/measure_timings.py
```

## 7. 결과물 위치

| 경로 | 내용 |
|---|---|
| `reports/01_FineTuning_결과보고서.md` | 데이터 · 학습 설정 · Validation 결과 |
| `reports/02_시스템_아키텍처.md` | 구성도 · 컴포넌트 · 실행 환경 |
| `reports/03_API_명세서.md` | API 요청 · 응답 · 상태코드 |
| `reports/04_DB_구조_작업이력.md` | `image_jobs` 구조 · 실제 행 |
| `reports/05_AI버튼_처리흐름.md` | 동기 · 비동기 시퀀스 |
| `reports/06_성능비교_보고서.md` | 응답시간 · 캐시 효과 비교 |
| `reports/07_기능테스트_오류분석.md` | 테스트 17건 · 오류 8건 |
| `reports/08_최종기술보고서.md` | 종합 보고서 · 개선 제안 |
| `reports/data/` | `training_report.json`, `timings.csv` |
| `reports/evidence/` | 실행 캡처 · 학습 로그 |
| `reports/figures/` | 아키텍처 · 시퀀스 · ERD · 성능 그래프 (Mermaid 원본 `.mmd` 포함) |

## 8. 프로젝트 구조

```text
fastapi_resnet18/
├── ai/              # config · hf_model · services · cache_service · database · models
│                    # celery_app · tasks · router_sync · router_async · router_web
├── training/        # finetune.py
├── templates/       # base.html · sync.html · async.html
├── static/          # style.css
├── tests/           # test_api.py (02 모의 통합 테스트)
├── scripts/         # measure_timings.py (성능 측정)
├── reports/         # 산출물 1~8 · data · evidence · figures
├── main.py · test_api.py(01 모의 테스트)
├── Dockerfile · compose.yaml · .dockerignore
└── requirements.txt · .env.example · .gitignore
```

## 9. 제출 및 보안

- `.env`, `.venv/`, `dataset/`, `artifacts/*`(모델 가중치), `media/jobs/*`는 `.gitignore`로 저장소에서 제외합니다.
- `compose.yaml`의 DB 비밀번호(`vision_pass`)는 로컬 실습용입니다. 공개 환경에서는 바꾸세요.
- 데이터셋과 가중치는 허용된 제출 경로로 따로 전달하고, 받은 뒤 위 2·3번 위치에 배치합니다.
