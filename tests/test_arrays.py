"""Raw pixel (numpy array) input tests for faster_paddle.

Covers ocr/det/rec/ocr_batch/prepare with uint8 arrays (RGB, BGR, gray,
RGBA), array-vs-bytes equality, and error paths.
"""
import io
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from PIL import Image

import faster_paddle

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "document.png")
MODELS_DIR = os.path.abspath(os.path.join(HERE, "..", "models"))

_ENG = None


def eng():
    global _ENG
    if _ENG is None:
        _ENG = faster_paddle.OcrEngine(model_size="tiny", threads=2, model_dir=MODELS_DIR)
    return _ENG


def _bytes():
    with open(FIXTURE, "rb") as f:
        return f.read()


def _rgb():
    return np.asarray(Image.open(FIXTURE).convert("RGB"))


def _sig(r):
    return [(b["topLeftCoord"], b["bottomRightCoord"], b["text"]) for b in r["bounds"].values()]


def _norm(r):
    return "".join(c for c in r["text"].upper() if c.isalnum())


def test_array_matches_bytes():
    e = eng()
    assert _sig(e.ocr(_rgb())) == _sig(e.ocr(_bytes()))


def test_array_matches_bytes_with_preprocessing():
    e = eng()
    arr = _rgb()
    data = _bytes()
    for kw in ({"resize": True}, {"denoise": True}, {"deskew": True}, {"binarize": True}):
        assert _sig(e.ocr(arr, **kw)) == _sig(e.ocr(data, **kw)), kw


def test_bgr_colorspace():
    e = eng()
    rgb = _rgb()
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    assert _sig(e.ocr(bgr, colorspace="bgr")) == _sig(e.ocr(rgb))
    # without the flag the swapped channels must not silently match
    assert _sig(e.ocr(bgr)) != _sig(e.ocr(rgb))


def test_rgba_and_bgra():
    e = eng()
    rgb = _rgb()
    h, w, _ = rgb.shape
    rgba = np.dstack([rgb, np.full((h, w), 255, dtype=np.uint8)])
    assert _sig(e.ocr(rgba)) == _sig(e.ocr(rgb))
    bgra = np.ascontiguousarray(rgba[:, :, [2, 1, 0, 3]])
    assert _sig(e.ocr(bgra, colorspace="bgr")) == _sig(e.ocr(rgb))


def test_gray_array():
    e = eng()
    gray = np.asarray(Image.open(FIXTURE).convert("L"))
    assert gray.ndim == 2
    r = e.ocr(gray)
    assert len(r["bounds"]) > 30
    assert "INFRAESTRUTURA" in _norm(r)


def test_det_array_matches_bytes():
    e = eng()
    assert e.det(_rgb()) == e.det(_bytes())
    assert len(e.det(_rgb())) > 50


def test_rec_array_matches_bytes():
    e = eng()
    data = _bytes()
    items = list(e.ocr(data)["bounds"].values())[:3]
    src = Image.open(io.BytesIO(data)).convert("RGB")
    crop_bytes, crop_arrays = [], []
    for b in items:
        x1, y1 = b["topLeftCoord"]
        x2, y2 = b["bottomRightCoord"]
        crop = src.crop((max(0, x1), max(0, y1), max(x1 + 1, x2), max(y1 + 1, y2)))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        crop_bytes.append(buf.getvalue())
        crop_arrays.append(np.asarray(crop))
    assert e.rec(crop_arrays) == e.rec(crop_bytes)


def test_batch_mixed_and_ordered():
    e = eng()
    data, arr = _bytes(), _rgb()
    inputs = [data, arr, data]
    expected = [e.ocr(data), e.ocr(arr), e.ocr(data)]
    assert e.ocr_batch(inputs, batch_size=2) == expected
    assert e.ocr_batch(inputs, batch_size=1) == expected


def test_prepare_return_array():
    e = eng()
    arr = _rgb()
    out = e.prepare(arr, binarize=True, return_array=True)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 3
    assert set(np.unique(out).tolist()) <= {0, 255}
    # default still returns PNG bytes for bytes input
    raw = e.prepare(_bytes(), binarize=True)
    assert isinstance(raw, bytes)
    Image.open(io.BytesIO(raw)).verify()
    # bytes input can also request an array
    assert isinstance(e.prepare(_bytes(), binarize=True, return_array=True), np.ndarray)
    # a prepared array still OCRs
    r = e.ocr(out)
    assert "INFRAESTRUTURA" in _norm(r)


def test_invalid_arrays_and_colorspace():
    e = eng()
    rgb = _rgb()
    with pytest.raises(ValueError):
        e.ocr(rgb.astype(np.float32))
    with pytest.raises(ValueError):
        e.ocr(np.zeros((rgb.shape[0], rgb.shape[1], 2), dtype=np.uint8))
    with pytest.raises(ValueError):
        e.ocr(rgb[:, ::2])  # strided view, not C-contiguous
    with pytest.raises(ValueError):
        e.ocr(rgb, colorspace="bgr2")
    with pytest.raises(ValueError):
        e.det("not an image")
    with pytest.raises(ValueError, match="image 1"):
        e.ocr_batch([_bytes(), rgb.astype(np.float32)])
    with pytest.raises(RuntimeError, match="image 1"):
        e.ocr_batch([_bytes(), b"invalid"], batch_size=1)
    with pytest.raises(ValueError, match="crop 0"):
        e.rec([rgb.astype(np.float32)])
    assert e.ocr(_bytes())["bounds"]  # errors must not poison the engine


def test_array_shape_dtype_and_contiguity_errors():
    """Edge shapes/dtypes/layouts must raise the documented messages."""
    e = eng()
    rgb = _rgb()
    h, w, _ = rgb.shape
    for bad in (rgb.astype(np.float64), rgb.astype(np.int32)):
        with pytest.raises(ValueError, match="uint8"):
            e.ocr(bad)
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        e.ocr(rgb[:, :, :1])  # (H, W, 1) is not a supported layout
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        e.ocr(np.zeros((h, w, 3, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="nonzero"):
        e.ocr(np.zeros((0, w, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="nonzero"):
        e.ocr(np.zeros((h, 0), dtype=np.uint8))
    with pytest.raises(ValueError, match="C-contiguous"):
        e.ocr(np.asfortranarray(rgb))


def test_read_only_array_matches_writable():
    e = eng()
    frozen = _rgb().copy()
    frozen.setflags(write=False)
    assert _sig(e.ocr(frozen)) == _sig(e.ocr(_rgb()))


def test_array_error_strings_and_chunk_indices():
    """Error messages name the offending input, and batch indices are global
    across chunk boundaries (the decode error happens inside run_ocr_batch)."""
    e = eng()
    bad = _rgb().astype(np.float32)
    with pytest.raises(ValueError, match="uint8"):
        e.det(bad)
    with pytest.raises(ValueError, match="uint8"):
        e.prepare(bad)
    with pytest.raises(ValueError, match="crop 0"):
        e.rec([bad])
    with pytest.raises(ValueError, match="image 1"):
        e.ocr_batch([_bytes(), np.zeros((0, 4, 3), dtype=np.uint8)], batch_size=2)
    with pytest.raises(RuntimeError, match="image 3"):
        e.ocr_batch([_bytes(), _bytes(), _bytes(), b"invalid"], batch_size=2)
    assert e.ocr(_bytes())["bounds"]  # errors must not poison the engine


def test_bytes_only_without_numpy():
    """Bytes inputs must work on a bare install with no numpy.

    pyo3-numpy's array downcasts panic when the numpy capsule is missing, so
    the sources only probe arrays when numpy is importable; return_array=True
    raises a clean RuntimeError instead. Runs in a subprocess because the
    capability check is cached per process.
    """
    code = textwrap.dedent(f"""
        import sys

        class BlockNumpy:
            def find_spec(self, name, path=None, target=None):
                if name == "numpy" or name.startswith("numpy."):
                    raise ModuleNotFoundError("No module named 'numpy'")

        sys.meta_path.insert(0, BlockNumpy())
        import faster_paddle

        e = faster_paddle.OcrEngine(model_size="tiny", threads=2, model_dir={MODELS_DIR!r})
        data = open({FIXTURE!r}, "rb").read()
        r = e.ocr(data)
        assert r["text"], r
        try:
            e.prepare(data, return_array=True)
        except RuntimeError as err:
            assert "numpy" in str(err), err
        else:
            raise AssertionError("return_array=True should require numpy")
        print("no-numpy OK")
    """)
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "no-numpy OK" in out.stdout
