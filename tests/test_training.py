"""학습(finetune_v2) · 평가(evaluate) 스크립트 단위 테스트.

인터넷 다운로드 없이 돌도록, 아주 작은 ResNet을 임시 폴더에 만들어 base model로 쓴다.
실행: python -m pytest -q tests/test_training.py
"""
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import ConvNextImageProcessor, ResNetConfig, ResNetForImageClassification, pipeline

from training import evaluate as ev
from training import finetune_v2 as ft

CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
COLORS = {"angular_leaf_spot": (120, 80, 30), "bean_rust": (200, 110, 40), "healthy": (40, 150, 50)}


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory) -> Path:
    """ImageNet 대신 2클래스짜리 작은 ResNet (출력층 교체 경로까지 확인하려고 클래스 수를 다르게 함)."""
    d = tmp_path_factory.mktemp("tiny_base")
    torch.manual_seed(0)
    cfg = ResNetConfig(num_channels=3, embedding_size=8, hidden_sizes=[8, 8, 8, 8], depths=[1, 1, 1, 1],
                       layer_type="basic", num_labels=2)
    ResNetForImageClassification(cfg).save_pretrained(d)
    ConvNextImageProcessor(size={"shortest_edge": 32}, crop_pct=0.875).save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("dataset")
    counts = {"train": 4, "val": 2, "test": 2}
    for split, n in counts.items():
        for c in CLASSES:
            (root / split / c).mkdir(parents=True)
            for i in range(n):
                r, g, b = COLORS[c]
                Image.new("RGB", (40, 40), (r + i, g, b)).save(root / split / c / f"{c}_{i}.jpg")
    (root / "train" / "healthy" / "note.txt").write_text("not an image")  # 확장자 필터 확인용
    return root


def train(tiny_base, tiny_data, out, *extra):
    argv = ["--data", str(tiny_data), "--output", str(out), "--base-model", str(tiny_base),
            "--batch-size", "4", "--cpu", *extra]
    return ft.run(ft.build_parser().parse_args(argv))


def test_compute_metrics_known_values():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    m = ev.compute_metrics(y_true, y_pred, CLASSES)
    assert m["accuracy"] == round(4 / 6, 4)
    assert m["confusion_matrix"]["rows_true_cols_pred"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert m["per_class"]["bean_rust"] == {"precision": round(2 / 3, 4), "recall": 1.0,
                                           "f1": 0.8, "support": 2}
    assert m["per_class"]["angular_leaf_spot"]["precision"] == 0.5


def test_finetune_v2_all_options_saves_service_ready_model(tiny_base, tiny_data, tmp_path):
    out = tmp_path / "model"
    report = train(tiny_base, tiny_data, out, "--epochs", "2", "--lp-epochs", "1", "--augment",
                   "--mixup-alpha", "0.2", "--label-smoothing", "0.1", "--scheduler", "cosine",
                   "--warmup-epochs", "0.5", "--head-lr-mult", "10", "--freeze-bn")
    for f in ["config.json", "model.safetensors", "preprocessor_config.json", "training_report.json"]:
        assert (out / f).exists(), f
    saved = json.loads((out / "training_report.json").read_text(encoding="utf-8"))
    assert saved["train_count"] == 12 and saved["val_count"] == 6  # note.txt 제외
    assert [h["phase"] for h in saved["history"]] == ["head-only", "full"]
    assert saved["classes"] == CLASSES
    assert 1 <= saved["best_epoch"] <= 2
    assert report["best_val_accuracy"] == saved["best_val_accuracy"]

    # 서비스(ai/hf_model.py)와 같은 방식으로 로드해 3클래스 라벨이 나오는지 확인
    clf = pipeline("image-classification", model=str(out), device=-1)
    result = clf(Image.new("RGB", (40, 40), (40, 150, 50)), top_k=5)
    assert {r["label"] for r in result} == set(CLASSES)


def test_freeze_backbone_keeps_backbone_and_bn_stats(tiny_base, tiny_data, tmp_path):
    out = tmp_path / "head_only"
    train(tiny_base, tiny_data, out, "--epochs", "1", "--freeze-backbone", "--freeze-bn", "--lr", "0.01")
    before = ResNetForImageClassification.from_pretrained(tiny_base).state_dict()
    after = ResNetForImageClassification.from_pretrained(out).state_dict()
    for k, v in before.items():
        if k.startswith("classifier."):
            continue  # 헤드는 2 → 3클래스로 교체되어 크기가 다름
        assert torch.equal(v, after[k]), f"Backbone이 바뀜: {k}"
    assert after["classifier.1.weight"].shape == (3, 8)


def test_without_freeze_bn_running_stats_change(tiny_base, tiny_data, tmp_path):
    """원본 finetune.py처럼 BN을 고정하지 않으면 가중치를 얼려도 running stats는 바뀐다(가설 확인)."""
    out = tmp_path / "head_only_bn_moves"
    train(tiny_base, tiny_data, out, "--epochs", "1", "--freeze-backbone")
    before = ResNetForImageClassification.from_pretrained(tiny_base).state_dict()
    after = ResNetForImageClassification.from_pretrained(out).state_dict()
    bn_keys = [k for k in before if k.endswith("running_mean")]
    assert any(not torch.equal(before[k], after[k]) for k in bn_keys)
    conv_keys = [k for k in before if k.endswith("convolution.weight")]
    assert all(torch.equal(before[k], after[k]) for k in conv_keys)


def test_evaluate_matches_training_validation(tiny_base, tiny_data, tmp_path):
    out = tmp_path / "eval_model"
    report = train(tiny_base, tiny_data, out, "--epochs", "2")
    m = ev.evaluate(str(out), data=str(tiny_data), split="val", cpu=True)
    assert m["total"] == 6
    assert m["accuracy"] == round(report["best_val_accuracy"], 4)
    assert len(m["misclassified"]) == m["total"] - m["correct"]
    t = ev.evaluate(str(out), data=str(tiny_data), split="test", cpu=True)
    assert t["total"] == 6 and sum(v["support"] for v in t["per_class"].values()) == 6
    assert "| 정답 \\ 예측 |" in ev.markdown_report(t)


def test_missing_split_folder_raises(tiny_base, tiny_data, tmp_path):
    out = tmp_path / "m"
    train(tiny_base, tiny_data, out, "--epochs", "1")
    with pytest.raises(ValueError):
        ev.evaluate(str(out), data=str(tmp_path / "nope"), split="test", cpu=True)
