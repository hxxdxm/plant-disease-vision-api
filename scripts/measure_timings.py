"""성능 측정 스크립트 (산출물 6용) — 제공 코드는 수정하지 않고 HTTP로만 측정한다.

실행: python scripts/measure_timings.py   (docker compose up 상태, 프로젝트 루트에서)
출력: reports/data/timings.csv + 화면에 조건별 중앙값

측정 조건
- 워밍업: 측정 전에 동기·비동기 각 1회 실행(web·worker 모델 로드), CSV에는 run_id=0, note=warmup 으로 기록하고 비교에서 제외
- Sync MISS : 처음 보내는 서로 다른 이미지 5장 → http_ms = 결과까지 받은 시간
- Sync HIT  : 위 5장을 다시 전송 → Redis 캐시 HIT
- Async MISS: 처음 보내는 서로 다른 이미지 5장 → http_ms = 202 접수 시간, final_ms = done 확인까지
- Async HIT : Async MISS 5장을 다시 전송 → worker가 캐시 HIT
- 비동기 상태 조회 간격: 0.05초 (브라우저 화면은 1초 간격이라 별도 기록)
- 캐시 키는 이미지 내용 기준이므로 동기·비동기 MISS 이미지 집합을 서로 겹치지 않게 고른다
"""
import csv
import hashlib
import statistics
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
POLL_SEC = 0.05
REPEAT = 5
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
# 앞선 수동 검증에서 이미 보낸 이미지(캐시에 남아 있을 수 있음)는 제외
USED = {"healthy_test.39.jpg", "bean_rust_test.14.jpg", "bean_rust_test.1.jpg",
        "bean_rust_test.0.jpg", "angular_leaf_spot_test.39.jpg"}
OUT = Path("reports/data/timings.csv")


def pick_images():
    files = []
    for cls in ["angular_leaf_spot", "bean_rust", "healthy"]:
        folder = Path("dataset/test") / cls
        files += [p for p in sorted(folder.iterdir())
                  if p.suffix.lower() in EXTS and p.name not in USED]
    # 같은 내용의 파일이 섞이지 않도록 해시로 한 번 더 거른다
    seen, unique = set(), []
    for p in files:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(p)
    # 클래스가 섞이도록 앞쪽에서 번갈아 선택
    need = 2 + REPEAT * 2  # warmup 2 + sync 5 + async 5
    by_cls = {}
    for p in unique:
        by_cls.setdefault(p.parent.name, []).append(p)
    picked, i = [], 0
    while len(picked) < need:
        for cls in sorted(by_cls):
            if i < len(by_cls[cls]) and len(picked) < need:
                picked.append(by_cls[cls][i])
        i += 1
    return picked[:2], picked[2:2 + REPEAT], picked[2 + REPEAT:]


def post(client, url, path):
    data = path.read_bytes()
    t0 = time.perf_counter()
    r = client.post(url, files={"image": (path.name, data, "image/jpeg")})
    ms = (time.perf_counter() - t0) * 1000
    return r, ms, t0


def run_sync(client, path):
    r, ms, _ = post(client, f"{BASE}/api/sync/classify/", path)
    r.raise_for_status()
    j = r.json()
    return {"http_status": r.status_code, "http_ms": round(ms, 2), "final_ms": round(ms, 2),
            "inference_ms": j["inference_ms"], "cache_hit": j["cache_hit"],
            "label": j["top1"]["label"], "score": round(j["top1"]["score"], 4), "job_id": ""}


def run_async(client, path):
    r, ms, t0 = post(client, f"{BASE}/api/async/classify/", path)
    r.raise_for_status()
    acc = r.json()
    polls = 0
    while True:
        time.sleep(POLL_SEC)
        polls += 1
        j = client.get(BASE + acc["check_url"]).json()
        if j["status"] in ("done", "failed"):
            break
        if polls > 1200:
            raise RuntimeError(f"timeout job {acc['job_id']}")
    final = (time.perf_counter() - t0) * 1000
    return {"http_status": r.status_code, "http_ms": round(ms, 2), "final_ms": round(final, 2),
            "inference_ms": j["inference_ms"], "cache_hit": j["cache_hit"],
            "label": (j["top1"] or {}).get("label", j["status"]),
            "score": round((j["top1"] or {}).get("score", 0.0), 4), "job_id": acc["job_id"],
            "polls": polls}


def main():
    warm, sync_imgs, async_imgs = pick_images()
    rows = []
    fields = ["run_id", "path", "cache_status", "http_status", "http_ms", "final_ms",
              "inference_ms", "cache_hit", "job_id", "image", "true_label", "pred_label",
              "score", "note"]

    def add(run_id, path_name, status, res, img, note=""):
        rows.append({"run_id": run_id, "path": path_name, "cache_status": status,
                     "http_status": res["http_status"], "http_ms": res["http_ms"],
                     "final_ms": res["final_ms"],
                     "inference_ms": "" if res["inference_ms"] is None else res["inference_ms"],
                     "cache_hit": res["cache_hit"], "job_id": res["job_id"], "image": img.name,
                     "true_label": img.parent.name, "pred_label": res["label"],
                     "score": res["score"], "note": note})
        print(f"{path_name:5} {status:4} run{run_id} http={res['http_ms']:>8}ms "
              f"final={res['final_ms']:>8}ms inf={res['inference_ms']} hit={res['cache_hit']} "
              f"{img.parent.name}->{res['label']} {note}")

    with httpx.Client(timeout=120) as client:
        assert client.get(f"{BASE}/health").json() == {"status": "ok"}
        add(0, "sync", "MISS", run_sync(client, warm[0]), warm[0], "warmup")
        add(0, "async", "MISS", run_async(client, warm[1]), warm[1], "warmup")
        for i, img in enumerate(sync_imgs, 1):
            add(i, "sync", "MISS", run_sync(client, img), img)
        for i, img in enumerate(sync_imgs, 1):
            add(i, "sync", "HIT", run_sync(client, img), img)
        for i, img in enumerate(async_imgs, 1):
            res = run_async(client, img)
            add(i, "async", "MISS", res, img, f"polls={res['polls']}")
        for i, img in enumerate(async_imgs, 1):
            res = run_async(client, img)
            add(i, "async", "HIT", res, img, f"polls={res['polls']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\n=== 중앙값 (warmup 제외, 각 5회, ms) ===")
    print(f"{'path':6}{'cache':6}{'http_ms':>10}{'final_ms':>10}{'inference_ms':>14}  hit확인")
    for p in ["sync", "async"]:
        for c in ["MISS", "HIT"]:
            sel = [r for r in rows if r["path"] == p and r["cache_status"] == c and r["note"] != "warmup"]
            inf = [r["inference_ms"] for r in sel if r["inference_ms"] != ""]
            hits = sum(1 for r in sel if r["cache_hit"] is True)
            print(f"{p:6}{c:6}{statistics.median(r['http_ms'] for r in sel):>10.2f}"
                  f"{statistics.median(r['final_ms'] for r in sel):>10.2f}"
                  f"{(statistics.median(inf) if inf else float('nan')):>14.2f}  {hits}/{len(sel)}")
    correct = sum(1 for r in rows if r["true_label"] == r["pred_label"])
    print(f"\n예측 일치(폴더 라벨 기준, 참고용): {correct}/{len(rows)}")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()