"""학습된 모델을 폴더 데이터(val 또는 test)로 평가한다.

서비스 추론(ai/services.py)과 같은 전처리(AutoImageProcessor)를 사용한다.
출력: 정확도, 클래스별 Precision · Recall · F1, macro F1, 혼동행렬, 틀린 이미지 목록.

예:
python -m training.evaluate --model artifacts/beans_resnet18 --split test \
  --out reports/data/eval_test_beans_resnet18.json
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def compute_metrics(y_true: list[int], y_pred: list[int], classes: list[str]) -> dict:
    """순수 파이썬으로 계산(sklearn 불필요). 행=정답, 열=예측."""
    n = len(classes)
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    per_class = {}
    for i, name in enumerate(classes):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n)) - tp
        fn = sum(cm[i]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": round(precision, 4), "recall": round(recall, 4),
                           "f1": round(f1, 4), "support": sum(cm[i])}
    total = len(y_true)
    correct = sum(cm[i][i] for i in range(n))
    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "correct": correct,
        "total": total,
        "macro_f1": round(sum(v["f1"] for v in per_class.values()) / n, 4),
        "per_class": per_class,
        "confusion_matrix": {"labels": classes, "rows_true_cols_pred": cm},
    }


def markdown_report(m: dict) -> str:
    classes = m["confusion_matrix"]["labels"]
    lines = [f"Accuracy **{m['accuracy']:.4f}** ({m['correct']}/{m['total']}) · Macro F1 **{m['macro_f1']:.4f}**", "",
             "| 클래스 | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for c in classes:
        v = m["per_class"][c]
        lines.append(f"| {c} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['support']} |")
    lines += ["", "| 정답 \\ 예측 | " + " | ".join(classes) + " |", "|---|" + "---:|" * len(classes)]
    for c, row in zip(classes, m["confusion_matrix"]["rows_true_cols_pred"]):
        lines.append(f"| **{c}** | " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


@torch.no_grad()
def evaluate(model_dir: str, data: str = "dataset", split: str = "test", batch_size: int = 32,
             cpu: bool = False) -> dict:
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    model.to(device).eval()

    # 모델 config의 라벨 순서를 기준으로 폴더 라벨을 맞춘다.
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    classes = [id2label[i] for i in range(len(id2label))]
    root = Path(data) / split
    records = []
    for idx, name in enumerate(classes):
        folder = root / name
        if not folder.is_dir():
            raise ValueError(f"평가 폴더가 없습니다: {folder}")
        records += [(p, idx) for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in EXTS]
    if not records:
        raise ValueError(f"이미지가 없습니다: {root}")

    y_true, y_pred, wrong = [], [], []
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        images = []
        for path, _ in chunk:
            with Image.open(path) as raw:
                images.append(raw.convert("RGB"))
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        probs = torch.softmax(model(pixel_values=pixels).logits, dim=-1).cpu()
        for (path, label), prob in zip(chunk, probs):
            pred = int(prob.argmax())
            y_true.append(label)
            y_pred.append(pred)
            if pred != label:
                wrong.append({"file": str(path), "true": classes[label], "pred": classes[pred],
                              "score": round(float(prob[pred]), 4)})

    metrics = compute_metrics(y_true, y_pred, classes)
    metrics.update({"model": model_dir, "split": split, "data": data, "misclassified": wrong})
    return metrics


def main():
    p = argparse.ArgumentParser(description="폴더 데이터셋으로 모델 평가")
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="dataset")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", help="결과 JSON 저장 경로")
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    m = evaluate(a.model, a.data, a.split, a.batch_size, a.cpu)
    print(markdown_report(m))
    print(f"\n틀린 이미지 {len(m['misclassified'])}장")
    for w in m["misclassified"]:
        print(f"  {w['file']}  정답={w['true']}  예측={w['pred']} ({w['score']})")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        md = Path(a.out).with_suffix(".md")
        md.write_text(f"# 평가: {a.model} · {a.split}\n\n" + markdown_report(m) + "\n", encoding="utf-8")
        print(f"저장: {a.out}, {md}")


if __name__ == "__main__":
    main()
