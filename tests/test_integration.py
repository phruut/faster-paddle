"""Integration tests for faster_paddle.

Self-contained — uses the bundled fixture image under tests/fixtures/.

    pip install pytest pillow && pytest faster_paddle/tests -q
or directly:
    python faster_paddle/tests/test_integration.py
"""
import io
import os
import time

import faster_paddle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "document.png")  # two-pane UI screenshot


def _small():
    with open(FIXTURE, "rb") as f:
        return f.read()


def test_bundled_runtime():
    # Verify the linked native runtime, not just the Rust crate/package version.
    assert "1.28.0" in faster_paddle.__runtime_build__


def _big():
    """An image larger than the 2100x3000 OCR canvas, so resize=True triggers."""
    im = Image.open(FIXTURE).convert("RGB").resize((2400, 3200))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def test_result_shape():
    r = faster_paddle.OcrEngine().ocr(_small())
    assert set(r.keys()) == {"text", "structured_text", "bounds"}
    assert len(r["bounds"]) > 50
    b = next(iter(r["bounds"].values()))
    assert set(b.keys()) == {"topLeftCoord", "bottomRightCoord", "text", "confidence"}


def test_known_text_detected():
    r = faster_paddle.OcrEngine().ocr(_small())
    norm = "".join(ch for ch in r["text"].upper() if ch.isalnum())
    assert "INFRAESTRUTURA" in norm
    assert "IPCMAPSMUNICIPIO" in norm


def test_rec_pool_is_deterministic():
    """Output must be identical regardless of the rec session-pool size."""
    data = _big()

    def run(pool):
        os.environ["REC_POOL"] = str(pool)
        r = faster_paddle.OcrEngine().ocr(data)
        return [(b["topLeftCoord"], b["bottomRightCoord"], b["text"]) for b in r["bounds"].values()]

    try:
        a, b = run(1), run(4)
    finally:
        os.environ.pop("REC_POOL", None)
    assert a == b, "rec pool changed the result (must be deterministic)"


def test_bounds_in_original_coords_after_resize():
    """With resize=True the engine works on a smaller image, but bounds must be
    mapped back to the ORIGINAL image coordinate space (not the resized one)."""
    data = _big()
    eng = faster_paddle.OcrEngine()

    def max_xy(**kw):
        r = eng.ocr(data, **kw)
        xs = [c for b in r["bounds"].values() for c in (b["topLeftCoord"][0], b["bottomRightCoord"][0])]
        ys = [c for b in r["bounds"].values() for c in (b["topLeftCoord"][1], b["bottomRightCoord"][1])]
        return max(xs), max(ys)

    base_x, base_y = max_xy()                 # no resize -> original coords
    res_x, res_y = max_xy(resize=True)        # resize -> must still be original coords
    # if bounds were left in the resized (<=2100x3000) space they'd be much smaller
    assert res_x > 0.8 * base_x, f"x looks resized: {res_x} vs {base_x}"
    assert res_y > 0.8 * base_y, f"y looks resized: {res_y} vs {base_y}"


def test_preprocessing_options_run():
    data = _small()
    eng = faster_paddle.OcrEngine()
    for kw in (
        {"resize": True},
        {"denoise": True},
        {"deskew": True},
        {"binarize": True},
        {"resize": True, "denoise": True, "deskew": True, "binarize": True},
    ):
        r = eng.ocr(data, **kw)
        assert len(r["bounds"]) > 30, f"{kw} produced too few boxes"


def test_prepare_returns_image_bytes():
    data = _small()
    eng = faster_paddle.OcrEngine()

    # all options off -> original bytes returned unchanged
    assert eng.prepare(data) == data
    assert faster_paddle.prepare(data) == data

    # resize -> a decodable (color) PNG
    p = eng.prepare(data, resize=True)
    assert isinstance(p, bytes)
    Image.open(io.BytesIO(p)).verify()

    # binarize -> grayscale PNG with only black/white pixels
    pb = eng.prepare(data, binarize=True)
    im = Image.open(io.BytesIO(pb))
    assert im.mode == "L"
    assert set(im.convert("L").tobytes()) <= {0, 255}

    # deskew expands the canvas of a rotated input
    rot = Image.open(FIXTURE).convert("RGB").rotate(8, expand=True, fillcolor=(255, 255, 255))
    buf = io.BytesIO(); rot.save(buf, format="PNG")
    pd = Image.open(io.BytesIO(eng.prepare(buf.getvalue(), deskew=True)))
    assert pd.size[0] >= rot.size[0] and pd.size[1] >= rot.size[1]

    # a prepared image still OCRs
    r = eng.ocr(pb)
    norm = "".join(c for c in r["text"].upper() if c.isalnum())
    assert "INFRAESTRUTURA" in norm


def test_latency_smoke():
    # Gross smoke limit only; compare benchmark JSONL on the same hardware for
    # meaningful performance regression measurements.
    data = _big()
    eng = faster_paddle.OcrEngine()
    eng.ocr(data)  # warm up
    times = []
    for _ in range(3):
        t = time.time()
        eng.ocr(data)
        times.append(time.time() - t)
    median = sorted(times)[len(times) // 2]
    assert median < 6.0, f"OCR too slow: {median:.2f}s"



def test_batch_matches_single_and_preserves_order():
    blank = io.BytesIO()
    Image.new("RGB", (300, 180), "white").save(blank, format="PNG")
    crop = Image.open(FIXTURE).crop((0, 0, 320, 240))
    cropped = io.BytesIO()
    crop.save(cropped, format="PNG")
    inputs = [_small(), blank.getvalue(), cropped.getvalue(), _small(), blank.getvalue()]
    for size in ("tiny", "small"):
        eng = faster_paddle.OcrEngine(model_size=size, threads=2)
        expected = [eng.ocr(x) for x in inputs]
        assert eng.ocr_batch(inputs, batch_size=3) == expected
        assert eng.ocr_batch(inputs[:2], batch_size=1) == expected[:2]
        assert eng.ocr_batch([]) == []
        # Exercise reused input buffers across shapes and then a single call.
        assert eng.ocr(inputs[0]) == expected[0]
    assert faster_paddle.ocr_batch([]) == []


def test_batch_preprocessing_maps_each_image_separately():
    eng = faster_paddle.OcrEngine(threads=2)
    data = [_small(), _big()]
    expected = [eng.ocr(b, resize=True) for b in data]
    assert eng.ocr_batch(data, resize=True) == expected


def test_invalid_options_and_batch_errors():
    import pytest
    for kw in ({"threads": 0}, {"rec_batch": 0}, {"det_max_side": 16},
               {"det_min_side": -1}, {"rec_min_width": 32}):
        with pytest.raises(ValueError):
            faster_paddle.OcrEngine(**kw)
    eng = faster_paddle.OcrEngine(threads=2)
    with pytest.raises(ValueError):
        eng.ocr_batch([], batch_size=0)
    with pytest.raises(ValueError):
        faster_paddle.ocr_batch([], batch_size=0)
    with pytest.raises(RuntimeError, match="image 1"):
        eng.ocr_batch([_small(), b"invalid"], batch_size=1)
    assert eng.ocr(_small())["bounds"]  # an error must not poison the engine


def test_resolved_cpu_budget_and_batch_cap():
    from unittest.mock import patch
    with patch.dict(os.environ, {"REC_POOL": "100", "OCR_DET_THREADS": "100", "RAYON_NUM_THREADS": "100"}):
        eng = faster_paddle.OcrEngine(threads=2, rec_batch=3)
        cfg = eng.config
        assert cfg["threads"] == 2
        assert cfg["det_threads"] <= 2
        assert cfg["rec_workers"] * cfg["rec_threads"] <= 2
        assert cfg["rayon_threads"] <= 2
        assert cfg["rec_batch"] == 3
    with patch.dict(os.environ, {"OCR_THREADS": "1"}):
        assert faster_paddle.OcrEngine().config["threads"] == 1


def test_affinity_is_respected():
    if not hasattr(os, "sched_setaffinity"):
        return
    import subprocess
    import sys
    code = """
import os
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import faster_paddle
c = faster_paddle.OcrEngine().config
assert c['threads'] == c['det_threads'] == c['rec_workers'] == c['rayon_threads'] == 1, c
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OCR_", "REC_", "RAYON_"))}
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=60)


def test_concurrent_calls_are_safe():
    from concurrent.futures import ThreadPoolExecutor
    eng = faster_paddle.OcrEngine(threads=2)
    image = _small()
    expected = eng.ocr(image)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(eng.ocr, image), pool.submit(eng.ocr_batch, [image, image]),
                   pool.submit(lambda: eng.config)]
        assert futures[0].result(timeout=60) == expected
        assert futures[1].result(timeout=60) == [expected, expected]
        assert futures[2].result(timeout=60)["threads"] == 2


def test_short_labels_with_model_specific_padding():
    from collections import Counter
    data = open(os.path.join(HERE, "fixtures", "labels.png"), "rb").read()
    labels = "A B C X Y Z 1 2 3 4 5 6 7 8 9 0 RF RE OK ID Yes No SIM NAO CEP ZIP ABC 123 USD EUR R$ 42"
    for size, width in (("tiny", 64), ("small", 96)):
        eng = faster_paddle.OcrEngine(model_size=size, threads=2)
        assert eng.config["rec_min_width"] == width
        assert Counter(eng.ocr(data)["text"].split()) == Counter(labels.split())


def test_sparse_wide_jobs_match_batch_text_and_boxes():
    data = open(os.path.join(HERE, "fixtures", "long-line.png"), "rb").read()
    signature = lambda r: [(b["topLeftCoord"], b["bottomRightCoord"], b["text"]) for b in r["bounds"].values()]
    for size in ("tiny", "small"):
        eng = faster_paddle.OcrEngine(model_size=size, threads=4)
        expected = signature(eng.ocr(data))
        assert expected
        assert list(map(signature, eng.ocr_batch([data] * 3))) == [expected] * 3
        cfg = eng.config
        assert cfg["wide_rec_workers"] * cfg["wide_rec_threads"] <= cfg["threads"]


def _signature(r):
    return [(b["topLeftCoord"], b["bottomRightCoord"], b["text"]) for b in r["bounds"].values()]


def test_det_defaults_match_explicit():
    data = _small()
    for size, box in (("tiny", 0.40), ("small", 0.45)):
        default = _signature(faster_paddle.OcrEngine(model_size=size, threads=2).ocr(data))
        explicit = _signature(
            faster_paddle.OcrEngine(
                model_size=size, threads=2,
                det_thresh=0.2, det_box_thresh=box, det_unclip_ratio=1.4,
                det_max_candidates=3000, text_score=0.0,
            ).ocr(data)
        )
        assert default == explicit
        assert len(default) > 50


def test_det_knobs_shift_output():
    data = _small()
    eng_default = faster_paddle.OcrEngine(model_size="tiny", threads=2)
    eng_custom = faster_paddle.OcrEngine(
        model_size="tiny", threads=2,
        det_thresh=0.3, det_box_thresh=0.5, det_unclip_ratio=1.6,
        det_max_candidates=1000, text_score=0.5,
    )
    assert _signature(eng_default.ocr(data)) != _signature(eng_custom.ocr(data))


def test_det_config_round_trips():
    import pytest
    eng = faster_paddle.OcrEngine(
        model_size="tiny", threads=2,
        det_thresh=0.3, det_box_thresh=0.5, det_unclip_ratio=1.6,
        det_max_candidates=1000, text_score=0.5,
    )
    cfg = eng.config
    assert cfg["det_thresh"] == pytest.approx(0.3)
    assert cfg["det_box_thresh"] == pytest.approx(0.5)
    assert cfg["det_unclip_ratio"] == pytest.approx(1.6)
    assert cfg["det_max_candidates"] == 1000
    assert cfg["text_score"] == pytest.approx(0.5)
    defaults = faster_paddle.OcrEngine(model_size="tiny", threads=2).config
    assert defaults["det_thresh"] == pytest.approx(0.2)
    assert defaults["det_box_thresh"] == pytest.approx(0.40)
    assert defaults["det_unclip_ratio"] == pytest.approx(1.4)
    assert defaults["det_max_candidates"] == 3000
    assert defaults["text_score"] == pytest.approx(0.0)


def test_det_invalid_values():
    import pytest
    for kw in ({"det_thresh": -0.1}, {"det_thresh": 1.5},
               {"det_box_thresh": -0.1}, {"det_box_thresh": 1.5},
               {"det_unclip_ratio": -1.0}, {"det_unclip_ratio": 5.5},
               {"det_max_candidates": 0}, {"det_max_candidates": 10001},
               {"text_score": -0.1}, {"text_score": 1.5}):
        with pytest.raises(ValueError):
            faster_paddle.OcrEngine(**kw)


def test_det_batch_respects_knobs():
    data = _small()
    eng = faster_paddle.OcrEngine(
        model_size="tiny", threads=2,
        det_thresh=0.3, det_box_thresh=0.5, det_unclip_ratio=1.6,
        det_max_candidates=1000, text_score=0.5,
    )
    expected = eng.ocr(data)
    assert eng.ocr_batch([data, data]) == [expected, expected]
    default = faster_paddle.OcrEngine(model_size="tiny", threads=2).ocr(data)
    assert _signature(expected) != _signature(default)


def test_det_max_candidates_limits_boxes():
    data = _small()
    full = faster_paddle.OcrEngine(model_size="tiny", threads=2).ocr(data)
    tiny = faster_paddle.OcrEngine(
        model_size="tiny", threads=2, det_max_candidates=10,
    ).ocr(data)
    assert len(tiny["bounds"]) <= 10
    assert len(tiny["bounds"]) < len(full["bounds"])


def test_text_score_filters_low_conf():
    data = _small()
    full = faster_paddle.OcrEngine(model_size="tiny", threads=2).ocr(data)
    filtered = faster_paddle.OcrEngine(
        model_size="tiny", threads=2, text_score=0.5,
    ).ocr(data)
    assert len(filtered["bounds"]) <= len(full["bounds"])
    assert _signature(filtered) != _signature(full)
    for b in filtered["bounds"].values():
        assert b["confidence"] >= 0.5 - 1e-6
        assert b["text"].strip() != ""


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        t = time.time()
        fn()
        print(f"  PASS {fn.__name__} ({time.time()-t:.2f}s)")
    print(f"All {len(fns)} integration tests passed.")
