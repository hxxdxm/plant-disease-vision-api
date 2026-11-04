# 산출물 3. API 명세서

- Base URL: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- 모든 응답 예시는 **실제 실행 결과**이다(2026-11-04, `MODEL_PATH=artifacts/beans_resnet18`). 긴 소수는 그대로 옮겼다.

## 1. API 목록

| API ID | Method | URL | 요청 형태 | 성공 HTTP | 주요 응답 필드 | 실패 HTTP | 실제 확인 |
|---|---|---|---|---:|---|---|:---:|
| API-01 | GET | `/health` | 없음 | 200 | `status` | 서버 미실행 시 응답 없음 | O |
| API-02 | POST | `/api/sync/classify/` | `multipart/form-data`, 필드 `image` | 200 | `model_id`, `top1`, `all`, `inference_ms`, `cache_hit` | 400, 413 | O |
| API-03 | POST | `/api/async/classify/` | `multipart/form-data`, 필드 `image` | **202** | `job_id`, `status`, `check_url` | 400, 413, 503 | O (503은 모의) |
| API-04 | GET | `/api/async/jobs/{job_id}/` | 경로 정수 `job_id` | 200 | `job_id`, `status`, `model_id`, `top1`, `all`, `inference_ms`, `cache_hit`, `error_message` | 404 | O |
| WEB-00 | GET | `/` | 없음 | 307 | `/vision/sync/`로 이동 | — | O |
| WEB-01 | GET | `/vision/sync/` | 없음 | 200 | HTML | — | O |
| WEB-02 | GET | `/vision/async/` | 없음 | 200 | HTML | — | O |
| STATIC | GET | `/static/style.css` | 없음 | 200 / 304 | CSS | — | O |

---

## 2. API-01 `GET /health`

```bash
curl http://127.0.0.1:8000/health
```
```json
{"status":"ok"}
```
Docker web 컨테이너의 healthcheck도 이 주소를 10초마다 호출한다.

---

## 3. API-02 `POST /api/sync/classify/` — 동기 분류

| 항목 | 내용 |
|---|---|
| Content-Type | `multipart/form-data` |
| 요청 필드 | `image`: JPEG / PNG / WEBP, 1바이트 이상 5MB 이하, 1,200만 픽셀 이하 |
| 처리 | Redis 캐시 확인 → MISS면 ResNet18 추론 → 캐시에 600초 저장 |
| 성공 | 200 OK (분류가 끝난 뒤 응답) |
| 오류 | 400 잘못된 이미지 · 형식 · 해상도 / 413 5MB 초과 |

### 응답 필드

| 필드 | 타입 | 설명 |
|---|---|---|
| `model_id` | string | 추론에 쓴 모델 경로 (`MODEL_PATH`) |
| `top1` | object | 가장 높은 점수의 `{label, score}` |
| `all` | array | 점수 높은 순 `[{label, score}, ...]` (3클래스) |
| `inference_ms` | number \| null | 모델 추론 시간. **캐시 HIT이면 `null`**(0ms가 아니라 추론 생략) |
| `cache_hit` | boolean | Redis 캐시 재사용 여부 |

### 성공 응답 — 캐시 MISS (실측, `healthy_test.39.jpg`, web 첫 요청)
```json
{
  "model_id": "artifacts/beans_resnet18",
  "top1": {"label": "healthy", "score": 0.8065634369850159},
  "all": [
    {"label": "healthy", "score": 0.8065634369850159},
    {"label": "bean_rust", "score": 0.12289518117904663},
    {"label": "angular_leaf_spot", "score": 0.07054135203361511}
  ],
  "inference_ms": 2282.39,
  "cache_hit": false
}
```

### 성공 응답 — 캐시 HIT (같은 파일 재요청)
```json
{
  "model_id": "artifacts/beans_resnet18",
  "top1": {"label": "healthy", "score": 0.8065634369850159},
  "all": [
    {"label": "healthy", "score": 0.8065634369850159},
    {"label": "bean_rust", "score": 0.12289518117904663},
    {"label": "angular_leaf_spot", "score": 0.07054135203361511}
  ],
  "inference_ms": null,
  "cache_hit": true
}
```

### 오류 응답 (실측)
```jsonc
// 400 — 텍스트 파일 업로드
{"detail": "올바른 이미지 파일이 아닙니다."}
// 413 — 5.3MB 파일 업로드
{"detail": "5MB 이하 파일만 허용합니다."}
```

코드상 400 메시지는 상황에 따라 `"이미지 파일은 1바이트 이상, 5MB 이하이어야 합니다."`, `"JPG, PNG, WEBP 이미지만 허용합니다."`, `"이미지 해상도가 너무 큽니다(최대 1200만 픽셀)."`도 있다(실측은 위 1종).

---

## 4. API-03 `POST /api/async/classify/` — 비동기 작업 접수

| 항목 | 내용 |
|---|---|
| Content-Type | `multipart/form-data` |
| 요청 필드 | `image` (API-02와 같은 검증을 **큐 등록 전에** 수행) |
| 처리 | 검증 → `media/jobs/<uuid>.bin` 저장(사용자 파일명은 쓰지 않음) → DB `queued` 행 생성 → `classify_job.delay(job_id)` |
| 성공 | **202 Accepted** — 작업 *접수*만 뜻하고 분석 완료가 아니다 |
| 오류 | 400 / 413 / 503 큐 등록 실패(Redis 연결 불가, DB 행은 `failed`로 기록) |

### 접수 응답 (실측, `bean_rust_test.14.jpg`)
```http
HTTP/1.1 202 Accepted
server: uvicorn
content-length: 63
content-type: application/json

{"job_id":1,"status":"queued","check_url":"/api/async/jobs/1/"}
```

### 503 응답 (모의 테스트 `test_broker_failure`에서 확인)
```json
{"detail": "작업 큐 등록 실패: Redis 연결을 확인하세요."}
```

---

## 5. API-04 `GET /api/async/jobs/{job_id}/` — 상태 조회

| status | 의미 | `top1`·`all` | `error_message` |
|---|---|---|---|
| `queued` | 접수됨, worker 대기 | null | null |
| `running` | worker 처리 중 | null | null |
| `done` | 완료 | 값 있음 | null |
| `failed` | 실패 | null | 오류 내용 |

`inference_ms`와 `cache_hit`은 상태와 관계없이 DB 값을 그대로 돌려준다.

### 완료 응답 (실측, job 1)
```json
{
  "job_id": 1,
  "status": "done",
  "model_id": "artifacts/beans_resnet18",
  "top1": {"label": "bean_rust", "score": 0.5152392387390137},
  "all": [
    {"label": "bean_rust", "score": 0.5152392387390137},
    {"label": "angular_leaf_spot", "score": 0.2754869759082794},
    {"label": "healthy", "score": 0.20927377045154572}
  ],
  "inference_ms": 2269.19,
  "cache_hit": false,
  "error_message": null
}
```

### 대기 응답 (실측, worker 정지 중 job 17)
```json
{"job_id":17,"status":"queued","model_id":"artifacts/beans_resnet18","top1":null,"all":null,"inference_ms":null,"cache_hit":null,"error_message":null}
```

### 오류 응답 (실측)
```jsonc
// 404 — 없는 job_id 999999
{"detail": "해당 작업이 없습니다."}
```

---

## 6. 필드명 주의

| 위치 | 이름 |
|---|---|
| API 응답 | **`all`** |
| DB 컬럼 | **`all_scores`** |

같은 데이터지만 이름이 다르다. 응답 필드를 `all_scores`로 바꾸지 않았다.

---

## 7. 실행 근거표

| 검증 ID | 테스트 입력 | 실제 상태코드 | 결과 | 증빙 |
|---|---|---:|---|---|
| API-01-T01 | GET `/health` | 200 | `{"status":"ok"}` | `evidence/02_health_redis_pg.png` |
| API-02-T01 | `healthy_test.39.jpg` 1회 | 200 | healthy, MISS | `evidence/02_sync_cache_miss_hit.png` |
| API-02-T02 | 같은 파일 2회 | 200 | healthy, HIT, `inference_ms:null` | `evidence/02_sync_cache_miss_hit.png` |
| API-02-T03 | 3클래스 test 이미지 (01 로컬) | 200 ×3 | 3개 모두 폴더 라벨과 일치 | `evidence/01_sync_curl.png` |
| API-02-T04 | `/tmp/bad.txt` | 400 | 올바른 이미지 아님 | `evidence/07_error_400_404_413.png` |
| API-02-T05 | 5.3MB 무작위 바이트 | 413 | 5MB 이하만 허용 | `evidence/07_error_400_404_413.png` |
| API-03-T01 | `bean_rust_test.14.jpg` | 202 | job_id 1, queued | `evidence/02_async_202.png` |
| API-04-T01 | job_id 1 | 200 | done, bean_rust | `evidence/02_async_job_done.png` |
| API-04-T02 | job_id 999999 | 404 | 해당 작업 없음 | `evidence/07_error_400_404_413.png` |
| API-04-T03 | worker 정지 중 job 17 | 200 | queued 유지 → 재시작 후 done | `evidence/07_worker_stop_queued_done.png` |
| WEB-01 | `/vision/sync/` 업로드 | 200 | bean_rust_test.0 → bean_rust, 35ms | `evidence/01_sync_ui.png` |
| WEB-02 | `/vision/async/` 업로드 | 202 → 200 | bean_rust_test.1 → bean_rust, 접수 22ms / 전체 1041ms | `evidence/02_async_ui.png` |
| API-03-T02 | Redis 큐 장애(모의) | 503 | `test_broker_failure` PASS | `evidence/02_pytest_fail_then_pass.png` |

> `202`는 작업 접수, 상태 조회 `200`의 `status:"done"`이 작업 완료다.
