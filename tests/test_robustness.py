"""강건성 평가 스크립트(training/robustness.py) 테스트. 작은 가짜 모델 사용, 인터넷 불필요.
실행: python -m pytest -q tests/test_robustness.py
"""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import ConvNextImageProcessor, ResNetConfig, ResNetForImageClassification

from training import robustness as rb

CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("tiny_model")
    torch.manual_seed(0)
    cfg = ResNetConfig(num_channels=3, embedding_size=8, hidden_sizes=[8, 8, 8, 8], depths=[1, 1, 1, 1],
                       layer_type="basic", num_labels=3,
                       id2label={i: c for i, c in enumerate(CLASSES)}, label2id={c: i for i, c in enumerate(CLASSES)})
    ResNetForImageClassification(cfg).save_pretrained(d)
    ConvNextImageProcessor(size={"shortest_edge": 32}, crop_pct=0.875).save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def tiny_test(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(1)
    for c in CLASSES:
        (root / "test" / c).mkdir(parents=True)
        for i in range(3):
            arr = (rng.random((48, 48, 3)) * 255).astype(np.uint8)
            Image.fromarray(arr).save(root / "test" / c / f"{c}_{i}.jpg")
    return root


def test_corruptions_keep_size_and_change_pixels():
    img = Image.fromarray((np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8))
    base = np.asarray(img).astype(float)
    for name, fn, _, _ in rb.CORRUPTIONS:
        out = fn(img, "k")
        assert out.size == img.size and out.mode == "RGB", name
        assert np.abs(np.asarray(out).astype(float) - base).mean() > 0, f"{name}가 이미지를 바꾸지 않음"
    assert np.asarray(rb.dark(img)).mean() < base.mean() < np.asarray(rb.bright(img)).mean()


def test_noise_is_reproducible_per_image():
    img = Image.new("RGB", (32, 32), (100, 120, 90))
    a, b, c = (np.asarray(rb.noise(img, k)) for k in ["x.jpg", "x.jpg", "y.jpg"])
    assert (a == b).all() and (a != c).any()


def test_no_hue_corruption():
    """병반 색이 진단 근거이므로 색조 변형은 넣지 않는다는 설계를 고정."""
    assert not any("hue" in name for name, *_ in rb.CORRUPTIONS)
    assert len(rb.CORRUPTIONS) == 8


def test_run_reports_all_corruptions_and_means(tiny_model, tiny_test):
    s = rb.run(str(tiny_model), data=str(tiny_test), split="test", cpu=True)
    pc = s["per_corruption"]
    assert s["total"] == 9 and set(pc) == {"clean"} | {n for n, *_ in rb.CORRUPTIONS}
    accs = [v["accuracy"] for k, v in pc.items() if k != "clean"]
    assert s["robust_mean_all"] == round(sum(accs) / 8, 4)
    for k, v in pc.items():
        assert 0.0 <= v["accuracy"] <= 1.0
        if k != "clean":
            assert v["drop_pp"] == round((pc["clean"]["accuracy"] - v["accuracy"]) * 100, 2)
    assert "강건성 평균" in rb.markdown_report(s)
