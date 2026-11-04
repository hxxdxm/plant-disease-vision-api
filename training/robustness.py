"""강건성 평가: Test 이미지에 실제 촬영 환경에서 생길 수 있는 변형을 주고 정확도를 잰다.

- 색조(hue) 변형은 넣지 않는다. 병반의 색이 진단 근거라서 색을 바꾸면 정답 자체가 달라질 수 있다.
- 같은 이미지에는 항상 같은 노이즈가 들어가도록 파일 경로로 seed를 고정한다(재현 가능).
- 학습 증강(finetune_v2 --augment)과 비슷한 변형인지 여부를 함께 기록해, 공정하게 비교할 수 있게 한다.

예:
python -m training.robustness --model artifacts/e4_lpft_aug_cosine --split test \
  --out reports/data/robustness_e4.json
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from transformers import AutoImageProcessor, AutoModelForImageClassification

from training.evaluate import EXTS, compute_metrics


def _seed_of(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def dark(img, key=""):
    return ImageEnhance.Brightness(img).enhance(0.6)


def bright(img, key=""):
    return ImageEnhance.Brightness(img).enhance(1.4)


def low_contrast(img, key=""):
    return ImageEnhance.Contrast(img).enhance(0.6)


def rotate30(img, key=""):
    return img.rotate(30, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))


def blur(img, key=""):
    return img.filter(ImageFilter.GaussianBlur(radius=2))


def noise(img, key=""):
    rng = np.random.default_rng(_seed_of(key))
    arr = np.asarray(img).astype(np.float32)
    arr = arr + rng.normal(0.0, 15.0, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def jpeg20(img, key=""):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=20)
    buf.seek(0)
    with Image.open(buf) as j:
        return j.convert("RGB")


def low_res(img, key=""):
    w, h = img.size
    small = img.resize((max(1, w // 4), max(1, h // 4)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


# (이름, 함수, 한국어 설명, 학습 증강에 비슷한 변형이 있었는지)
CORRUPTIONS = [
    ("dark", dark, "어둡게 (밝기 x0.6)", True),
    ("bright", bright, "밝게 (밝기 x1.4)", True),
    ("low_contrast", low_contrast, "저대비 (대비 x0.6)", True),
    ("rotate30", rotate30, "회전 30도 (학습은 ±20도)", True),
    ("blur", blur, "흐림 (Gaussian r=2)", False),
    ("noise", noise, "노이즈 (sigma=15)", False),
    ("jpeg20", jpeg20, "JPEG 압축 (품질 20)", False),
    ("low_res", low_res, "저해상도 (1/4 축소 후 복원)", False),
]


def load_records(root: Path, classes: list[str]):
    records = []
    for idx, name in enumerate(classes):
        folder = root / name
        if not folder.is_dir():
            raise ValueError(f"평가 폴더가 없습니다: {folder}")
        records += [(p, idx) for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in EXTS]
    if not records:
        raise ValueError(f"이미지가 없습니다: {root}")
    return records


@torch.no_grad()
def predict(model, processor, images, device, batch_size):
    preds = []
    for s in range(0, len(images), batch_size):
        px = processor(images=images[s:s + batch_size], return_tensors="pt")["pixel_values"].to(device)
        preds += model(pixel_values=px).logits.argmax(-1).cpu().tolist()
    return preds


def run(model_dir, data="dataset", split="test", batch_size=32, cpu=False):
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    model.to(device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    classes = [id2label[i] for i in range(len(id2label))]
    records = load_records(Path(data) / split, classes)

    originals, labels, keys = [], [], []
    for path, label in records:
        with Image.open(path) as raw:
            originals.append(raw.convert("RGB"))
        labels.append(label)
        keys.append(str(path))

    results = {}
    clean = compute_metrics(labels, predict(model, processor, originals, device, batch_size), classes)
    results["clean"] = {"desc": "원본", "seen_in_training": None,
                        "accuracy": clean["accuracy"], "macro_f1": clean["macro_f1"], "correct": clean["correct"]}
    for name, fn, desc, seen in CORRUPTIONS:
        imgs = [fn(im, k) for im, k in zip(originals, keys)]
        m = compute_metrics(labels, predict(model, processor, imgs, device, batch_size), classes)
        results[name] = {"desc": desc, "seen_in_training": seen, "accuracy": m["accuracy"],
                         "macro_f1": m["macro_f1"], "correct": m["correct"],
                         "drop_pp": round((clean["accuracy"] - m["accuracy"]) * 100, 2)}

    def mean(sel):
        vals = [v["accuracy"] for k, v in results.items() if k != "clean" and sel(v)]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {"model": model_dir, "split": split, "total": len(records),
               "clean_accuracy": clean["accuracy"],
               "robust_mean_all": mean(lambda v: True),
               "robust_mean_seen": mean(lambda v: v["seen_in_training"] is True),
               "robust_mean_unseen": mean(lambda v: v["seen_in_training"] is False),
               "worst": min(((k, v["accuracy"]) for k, v in results.items() if k != "clean"), key=lambda x: x[1]),
               "per_corruption": results}
    return summary


def markdown_report(s: dict) -> str:
    lines = [f"모델 `{s['model']}` · {s['split']} {s['total']}장", "",
             "| 변형 | 학습 증강과 비슷? | Accuracy | 정답 수 | Clean 대비 변화 |", "|---|:---:|---:|---:|---:|"]
    for k, v in s["per_corruption"].items():
        seen = "—" if v["seen_in_training"] is None else ("✅" if v["seen_in_training"] else "❌")
        drop = "" if k == "clean" else ("±0.00%p" if v["drop_pp"] == 0 else f"{-v['drop_pp']:+.2f}%p")
        lines.append(f"| {v['desc']} | {seen} | {v['accuracy']:.4f} | {v['correct']}/{s['total']} | {drop} |")
    lines += ["", f"- 강건성 평균(8개 변형): **{s['robust_mean_all']:.4f}**",
              f"- 학습 증강과 비슷한 변형 평균(✅): {s['robust_mean_seen']:.4f}",
              f"- 학습 때 못 본 변형 평균(❌): {s['robust_mean_unseen']:.4f}",
              f"- 가장 약한 변형: {s['worst'][0]} ({s['worst'][1]:.4f})"]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="변형 이미지로 강건성 평가")
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="dataset")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save-examples", help="변형 예시 이미지를 저장할 폴더(첫 번째 이미지 기준)")
    a = p.parse_args()
    s = run(a.model, a.data, a.split, a.batch_size, a.cpu)
    print(markdown_report(s))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
        Path(a.out).with_suffix(".md").write_text(f"# 강건성 평가\n\n{markdown_report(s)}\n", encoding="utf-8")
        print(f"저장: {a.out}")
    if a.save_examples:
        out = Path(a.save_examples)
        out.mkdir(parents=True, exist_ok=True)
        first = sorted((Path(a.data) / a.split).rglob("*.jpg"))[0]
        with Image.open(first) as raw:
            img = raw.convert("RGB")
        img.save(out / "00_clean.jpg")
        for i, (name, fn, _, _) in enumerate(CORRUPTIONS, 1):
            fn(img, str(first)).save(out / f"{i:02d}_{name}.jpg")
        print(f"예시 저장: {out}")


if __name__ == "__main__":
    main()
