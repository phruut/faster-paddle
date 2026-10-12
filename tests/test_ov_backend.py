"""OpenVINO backend tests (opt-in wheel).

Skipped unless the extension was built with the OpenVINO backend
(`--no-default-features --features openvino-backend`) AND local IR weights are
present under `OV_IR_DIR` (`{OV_IR_DIR}/{size}/det.xml` etc). CI converts them
with `scripts/convert_ir.py`; nothing here touches the network.

Capability is decided from wheel metadata (`__backends__`) and the IR files on
disk — never by catching an error substring — so a construction failure is a
test failure, not a silent skip.

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
BACKENDS = tuple(getattr(faster_paddle, "__backends__", ("ort",)))
IR_DIR = os.environ.get("OV_IR_DIR")
HAS_LOCAL_IR = bool(IR_DIR) and os.path.isfile(os.path.join(IR_DIR, "tiny", "det.xml"))
requires_ov_weights = pytest.mark.skipif(
    "ov" not in BACKENDS or not HAS_LOCAL_IR,
    reason="needs the OpenVINO backend and local OV_IR_DIR IR weights",
)


def _small():
    with open(FIXTURE, "rb") as f:
        return f.read()


def _ov_engine(**kw):
    # no error-substring skipping: availability is enforced by the marker
    return faster_paddle.OcrEngine(backend="ov", **kw)


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        faster_paddle.OcrEngine(backend="nope")


@pytest.mark.skipif("ov" in BACKENDS, reason="build includes the OpenVINO backend")
def test_ov_request_on_ort_only_build_errors():
    with pytest.raises(RuntimeError, match="openvino-backend"):
        faster_paddle.OcrEngine(backend="ov")


@requires_ov_weights
def test_ov_config():
    e = _ov_engine(model_size="tiny")
    cfg = e.config
    assert cfg["backend"] == "ov"
    assert cfg["rec_batch"] == 1  # static bucket shapes force singleton batches
    assert cfg["rec_buckets"] >= 1
    assert "ov_runtime" in cfg


@requires_ov_weights
def test_ov_result_shape():
    r = _ov_engine(model_size="tiny").ocr(_small())
    assert set(r.keys()) == {"text", "structured_text", "bounds"}
    assert len(r["bounds"]) > 50
    b = next(iter(r["bounds"].values()))
    assert set(b.keys()) == {"topLeftCoord", "bottomRightCoord", "text", "confidence"}


@requires_ov_weights
def test_ov_known_text_detected():
    for size in ("tiny", "small"):
        r = _ov_engine(model_size=size).ocr(_small())
        norm = "".join(ch for ch in r["text"].upper() if ch.isalnum())
        assert "INFRAESTRUTURA" in norm
        assert "IPCMAPSMUNICIPIO" in norm


@requires_ov_weights
def test_ov_deterministic():
    data = _small()
    e = _ov_engine(model_size="tiny")
    first = e.ocr(data)["text"]
    assert e.ocr(data)["text"] == first


@requires_ov_weights
def test_ov_ir_dir_is_the_default_local_dir(monkeypatch, tmp_path):
    # OV_IR_DIR beats the FASTER_PADDLE_MODEL_DIR fallback: with both set, the
    # IR directory is the resolved model_dir and no download happens
    monkeypatch.setenv("OV_IR_DIR", IR_DIR)
    monkeypatch.setenv("FASTER_PADDLE_MODEL_DIR", str(tmp_path))
    eng = _ov_engine(model_size="tiny")
    assert eng.config["model_dir"] == IR_DIR
    assert (eng.config["det_source"], eng.config["rec_source"]) == ("local", "local")


@requires_ov_weights
def test_explicit_model_dir_beats_ov_ir_dir(tmp_path, monkeypatch):
    # constructor beats env: an explicit model_dir wins over OV_IR_DIR
    monkeypatch.setenv("OV_IR_DIR", str(tmp_path))  # empty; must not be used
    eng = _ov_engine(model_size="tiny", model_dir=IR_DIR)
    assert eng.config["model_dir"] == IR_DIR
    assert (eng.config["det_source"], eng.config["rec_source"]) == ("local", "local")


def test_ov_only_wheel_has_no_ort():
    """The OpenVINO wheel ships OpenVINO only; ORT builds never report "ov" as
    their default backend."""
    if "ort" in BACKENDS:
        pytest.skip("build also carries ONNX Runtime (local A/B build)")
    with pytest.raises(RuntimeError, match="OpenVINO-only"):
        faster_paddle.OcrEngine(backend="ort")
    assert "OpenVINO" in faster_paddle.__runtime_build__
    assert faster_paddle.OcrEngine().config["backend"] == "ov"
