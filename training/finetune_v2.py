"""개선 학습 스크립트 (원본 training/finetune.py는 그대로 두고 별도로 추가).

원본과 같은 데이터 폴더 · 같은 출력 형식(save_pretrained + training_report.json)을 쓰므로
결과 폴더를 그대로 .env의 MODEL_PATH로 지정해 FastAPI · Celery에서 사용할 수 있다.

추가한 기법과 근거
- 데이터 증강(--augment)            : 학습 데이터에만 적용. 원본 코드는 증강이 없어 Full 학습에서 과적합 신호가 있었다.
- LP-FT(--lp-epochs N)              : 처음 N epoch는 헤드만 학습(Linear Probing) 후 전체 미세조정(Kumar et al., ICLR 2022).
- 계층별 학습률(--head-lr-mult)      : 새로 만든 헤드는 Backbone보다 큰 학습률로 학습.
- 코사인 학습률 스케줄(--scheduler)  : warmup 후 cosine 감소(Loshchilov & Hutter, ICLR 2017).
- Label smoothing(--label-smoothing): 과신 방지 · 일반화(Szegedy 2016, Müller et al. 2019).
- MixUp(--mixup-alpha)              : 두 이미지를 섞어 학습(Zhang et al., ICLR 2018).
- BatchNorm 고정(--freeze-bn)        : 고정된 Backbone의 BN 통계가 작은 배치로 흔들리지 않게 eval 모드 유지.
- best 저장 기준 보완               : Val Acc가 같으면 Val Loss가 더 낮은 epoch를 저장(원본은 같은 Acc면 저장 안 함).
- 조기 종료(--patience)

평가(Validation)는 서비스 추론과 같은 전처리(AutoImageProcessor)를 그대로 사용한다.

예:
python -m training.finetune_v2 --output artifacts/e6_lpft_aug --epochs 15 \
  --lp-epochs 3 --lr 0.0001 --head-lr-mult 10 --augment --label-smoothing 0.1 --scheduler cosine
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

DEFAULT_BASE_MODEL = "microsoft/resnet-18"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_records(root: Path, classes: list[str]):
    records = []
    for idx, name in enumerate(classes):
        folder = root / name
        if not folder.is_dir():
            raise ValueError(f"필요한 클래스 폴더가 없습니다: {folder}")
        records.extend((p, idx) for p in sorted(folder.iterdir())
                       if p.is_file() and p.suffix.lower() in EXTS)
    if not records:
        raise ValueError(f"이미지가 없습니다: {root}")
    return records


def crop_size_of(processor) -> int:
    """processor 설정에서 모델 입력 크기(224 등)를 읽는다."""
    size = getattr(processor, "crop_size", None) or getattr(processor, "size", None) or {}
    if isinstance(size, dict):
        return int(size.get("height") or size.get("shortest_edge") or 224)
    return int(size)


def build_train_transform(processor):
    """학습 전용 증강. 정규화 값은 사전학습 processor와 같게 맞춘다."""
    from torchvision import transforms as T
    size = crop_size_of(processor)
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        # 병반 색이 진단 근거이므로 hue(색상)는 아주 조금만 바꾼다.
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])


class FolderImages(Dataset):
    def __init__(self, records, processor, transform=None):
        self.records = records
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        with Image.open(path) as raw:
            image = raw.convert("RGB")
            if self.transform is not None:
                tensor = self.transform(image)
            else:  # 평가: 서비스 추론과 같은 전처리
                tensor = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": tensor, "labels": torch.tensor(label, dtype=torch.long)}


def set_backbone_trainable(model, trainable: bool):
    for name, p in model.named_parameters():
        p.requires_grad = True if name.startswith("classifier.") else trainable


def keep_frozen_bn_eval(model):
    """requires_grad=False인 BatchNorm의 running mean/var가 학습 중 바뀌지 않게 한다."""
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()


def make_optimizer(model, lr, head_lr_mult, weight_decay):
    head = [p for n, p in model.named_parameters() if n.startswith("classifier.")]
    body = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]
    return torch.optim.AdamW(
        [{"params": body, "lr": lr, "base_lr": lr},
         {"params": head, "lr": lr * head_lr_mult, "base_lr": lr * head_lr_mult}],
        weight_decay=weight_decay,
    )


def lr_factor(step, total_steps, warmup_steps, kind):
    if kind == "none":
        return 1.0
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def mixup_batch(x, y, alpha, num_classes):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    y1 = torch.nn.functional.one_hot(y, num_classes).float()
    return lam * x + (1 - lam) * x[idx], lam * y1 + (1 - lam) * y1[idx]


def soft_cross_entropy(logits, soft_targets, label_smoothing, num_classes):
    if label_smoothing > 0:
        soft_targets = soft_targets * (1 - label_smoothing) + label_smoothing / num_classes
    return torch.sum(-soft_targets * torch.log_softmax(logits, dim=-1), dim=-1).mean()


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for batch in loader:
        x = batch["pixel_values"].to(device)
        y = batch["labels"].to(device)
        logits = model(pixel_values=x).logits
        loss_sum += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / total, loss_sum / total


def run(args):
    set_seed(args.seed)
    data = Path(args.data)
    train_root = data / "train"
    classes = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise ValueError("클래스 폴더가 2개 이상 필요합니다.")

    processor = AutoImageProcessor.from_pretrained(args.base_model)
    train_records = list_records(train_root, classes)
    val_records = list_records(data / "val", classes)
    train_tf = build_train_transform(processor) if args.augment else None
    train_set = FolderImages(train_records, processor, train_tf)
    val_set = FolderImages(val_records, processor, None)
    print(f"클래스: {classes} / Train: {len(train_set)}, Val: {len(val_set)}")

    model = AutoModelForImageClassification.from_pretrained(
        args.base_model,
        num_labels=len(classes),
        id2label={i: n for i, n in enumerate(classes)},
        label2id={n: i for i, n in enumerate(classes)},
        ignore_mismatched_sizes=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, generator=g)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    optim = make_optimizer(model, args.lr, args.head_lr_mult, args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history, best = [], {"val_accuracy": -1.0, "val_loss": float("inf"), "epoch": 0}
    bad_epochs, step = 0, 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        head_only = args.freeze_backbone or epoch <= args.lp_epochs
        set_backbone_trainable(model, not head_only)
        model.train()
        if args.freeze_bn:
            keep_frozen_bn_eval(model)

        losses = []
        for batch in train_loader:
            x = batch["pixel_values"].to(device)
            y = batch["labels"].to(device)
            if args.mixup_alpha > 0:
                x, targets = mixup_batch(x, y, args.mixup_alpha, len(classes))
            else:
                targets = torch.nn.functional.one_hot(y, len(classes)).float()
            factor = lr_factor(step, total_steps, warmup_steps, args.scheduler)
            for group in optim.param_groups:
                group["lr"] = group["base_lr"] * factor
            optim.zero_grad()
            logits = model(pixel_values=x).logits
            loss = soft_cross_entropy(logits, targets, args.label_smoothing, len(classes))
            loss.backward()
            optim.step()
            losses.append(loss.item())
            step += 1

        val_acc, val_loss = evaluate_loader(model, val_loader, device)
        row = {"epoch": epoch, "phase": "head-only" if head_only else "full",
               "loss": round(sum(losses) / len(losses), 4),
               "val_accuracy": round(val_acc, 4), "val_loss": round(val_loss, 4),
               "lr_head_end": round(optim.param_groups[1]["lr"], 8)}
        history.append(row)
        print(row)

        improved = (val_acc > best["val_accuracy"]) or (
            val_acc == best["val_accuracy"] and val_loss < best["val_loss"])
        if improved:
            best = {"val_accuracy": val_acc, "val_loss": val_loss, "epoch": epoch}
            model.save_pretrained(str(output), safe_serialization=True)
            processor.save_pretrained(str(output))
            bad_epochs = 0
        else:
            bad_epochs += 1
            if args.patience and bad_epochs >= args.patience:
                print(f"조기 종료: {args.patience} epoch 동안 개선 없음")
                break

    report = {
        "script": "training/finetune_v2.py",
        "base_model": args.base_model,
        "classes": classes,
        "train_count": len(train_set),
        "val_count": len(val_set),
        "best_val_accuracy": best["val_accuracy"],
        "best_val_loss": round(best["val_loss"], 4),
        "best_epoch": best["epoch"],
        "train_seconds": round(time.perf_counter() - started, 1),
        "device": str(device),
        "config": {k: v for k, v in vars(args).items()},
        "history": history,
    }
    (output / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장 완료: {output} / best epoch={best['epoch']} "
          f"val accuracy={best['val_accuracy']:.4f} val loss={best['val_loss']:.4f}")
    return report


def build_parser():
    p = argparse.ArgumentParser(description="ResNet18 Beans 개선 학습")
    p.add_argument("--data", default="dataset")
    p.add_argument("--output", required=True)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4, help="Backbone 학습률")
    p.add_argument("--head-lr-mult", type=float, default=1.0, help="헤드 학습률 = lr x 이 값")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    p.add_argument("--warmup-epochs", type=float, default=0.0)
    p.add_argument("--freeze-backbone", action="store_true", help="모든 epoch 헤드만 학습")
    p.add_argument("--lp-epochs", type=int, default=0, help="LP-FT: 처음 N epoch 헤드만 학습")
    p.add_argument("--freeze-bn", action="store_true", help="고정된 BN을 eval 모드로 유지")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--mixup-alpha", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=0, help="0이면 조기 종료 안 함")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
