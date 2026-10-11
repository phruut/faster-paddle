"""OpenVINO backend tests (opt-in).

Skipped unless the extension was built with `--features openvino-backend`
and IR weights are reachable (`{OV_IR_DIR or model_dir}/{size}/det.xml` etc,
or downloadable from the release host).

The OV backend trades bit-parity for speed: detector masks agree to ~1px
and recognizer numerics drift on CTC-fragile crops, so these tests assert
the output contract, determinism, and known-text tokens — not byte-identical
parity with the ORT backend.
"""
import os

import pytest

faster_paddle = pytest.importorskip("faster_paddle")

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "document.png")


def _small():
    with open(FIXTURE, "rb") as f:
        return f.read()


def _ov_engine(**kw):
    try:
        return faster_paddle.OcrEngine(backend="ov", **kw)
    except Exception as e:
        msg = str(e)
        if "openvino-backend" in msg or "IR weights" in msg or "failed to download" in msg:
            pytest.skip(f"OV backend unavailable: {msg}")
        raise


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        faster_paddle.OcrEngine(backend="nope")


def test_ov_config():
    e = _ov_engine(model_size="tiny")
    cfg = e.config
    assert cfg["backend"] == "ov"
    assert cfg["rec_batch"] == 1  # static bucket shapes force singleton batches
    assert cfg["rec_buckets"] >= 1
    assert "ov_runtime" in cfg


def test_ov_result_shape():
    r = _ov_engine(model_size="tiny").ocr(_small())
    assert set(r.keys()) == {"text", "structured_text", "bounds"}
    assert len(r["bounds"]) > 50
    b = next(iter(r["bounds"].values()))
    assert set(b.keys()) == {"topLeftCoord", "bottomRightCoord", "text", "confidence"}


def test_ov_known_text_detected():
    for size in ("tiny", "small"):
        r = _ov_engine(model_size=size).ocr(_small())
        norm = "".join(ch for ch in r["text"].upper() if ch.isalnum())
        assert "INFRAESTRUTURA" in norm
        assert "IPCMAPSMUNICIPIO" in norm


def test_ov_deterministic():
    data = _small()
    e = _ov_engine(model_size="tiny")
    first = e.ocr(data)["text"]
    assert e.ocr(data)["text"] == first
