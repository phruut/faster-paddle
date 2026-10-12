# faster-paddle fork

**Fast, CPU-only OCR in Rust with Python bindings**: a self-contained
reimplementation of PaddleOCR's PP-OCRv6 detection + recognition pipeline with
[ONNX Runtime](https://onnxruntime.ai/) or
[OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
inference. One runtime per wheel, no Python inference packages.

- CPU latency optimizations: fused image transforms, compact DB components,
  dynamic recognition scheduling, and model-specific input widths.
- Multi-image OCR with `ocr_batch`: ordered results and shared recognition work
  across pages. [Batch usage](docs/usage.md#multiple-images).
- Code-only default: the wheel bundles no model weights. Each size is loaded
  from `model_dir`, the local cache, or a one-time download from the GitHub
  release. Build with `--features bundled-tiny/small/medium/all` for a
  self-contained wheel with the weights embedded.
- Three model sizes: `tiny` (default, fastest), `small`, and `medium` (higher
  accuracy). Split-stage entry points too: boxes only with `det`, recognition
  only with `rec`, and mixed per-stage sizes
  (`OcrEngine("tiny", rec_model_size="small")`).
- Pure-Rust pre/post-processing (detection DB decode, `minAreaRect`,
  perspective crop, CTC decode, reading-order text reconstruction). No OpenCV.
- Prebuilt wheels for Linux (x86-64, arm64), Windows (x64), and macOS (arm64).

Defaults respect available physical cores, Linux CPU affinity, and container
CPU limits. Reuse an engine to amortize model/session loading. Performance
results and benchmark harnesses are linked at the
[bottom](#performance).

---

## Install

There are no PyPI releases. Wheels are attached to each GitHub release; pin
the exact release you validated (the URLs are version-pinned, and release
assets stay in place) and install its file directly:

```bash
uv pip install https://github.com/phruut/faster-paddle/releases/download/v1.2.1/<wheel-file>
# or
pip install https://github.com/phruut/faster-paddle/releases/download/v1.2.1/<wheel-file>
```

Replace `v1.2.1` with the release you want; the [releases page](https://github.com/phruut/faster-paddle/releases)
lists the available wheel files. Source builds are the fallback for platforms
without a wheel; releases before v1.1.1 have no prebuilt wheels at all.

Each release ships up to two wheel variants (same API, same models): the ONNX
Runtime wheel and the OpenVINO wheel. They are separate distributions with one
inference runtime each, so the filenames differ:

| wheel file | inference | download | pick it when |
|---|---|---|---|
| `faster_paddle-1.2.1-...-win_amd64.whl` | ONNX Runtime, linked in | ~9 MB | you want the minimal install |
| `faster_paddle_openvino-1.2.1-...-win_amd64.whl` | OpenVINO, DLLs bundled | ~20 MB (~65 MB installed) | you want lower latency (not always) |

Wheel coverage per platform:

| platform | ONNX Runtime wheel | OpenVINO wheel |
|---|---|---|
| Linux x86-64, glibc 2.34+ (`manylinux_2_34`) | yes | yes |
| Linux arm64, glibc 2.28+ (`manylinux_2_28`) | yes | no (no PyPI OpenVINO wheel) |
| Windows x64 | yes | yes |
| macOS arm64 | yes | yes |

There are no macOS x86-64 or Windows arm64 wheels. The manylinux tags are
glibc floors: x86-64 needs glibc 2.34+ (ONNX Runtime), arm64 needs 2.28+.

Both are self-contained: no Python `onnxruntime`/`openvino` package, no
`PATH` setup, no second install. Model weights download on first use either
way: ONNX weights for `"ort"`, IR weights (`ppocrv6_{size}_{det,rec}.xml`/`.bin`
release assets) for `"ov"`. The backend defaults to the wheel's runtime and
can be passed explicitly when the build offers more than one:

```python
engine = OcrEngine(model_size="small")                 # ONNX Runtime (default wheel)
engine = OcrEngine(model_size="small", backend="ov")   # OpenVINO (OpenVINO wheel)
```

Local A/B probes measured `"ov"` 20-70% faster end to end on small/medium (tiny
roughly tied), at the cost of minor numeric drift: ~1px box shifts and
occasional CTC-fragile glyph flips, the same class of difference as ONNX
Runtime vs PaddlePaddle; labeled-corpus recall was unchanged in those probes.
These OV numbers are **unverified in-repo**: the raw probes were not retained
and the committed benchmark harnesses run ONNX Runtime only, so treat them as
a direction, not a guarantee. `engine.config["backend"]` reports which one is
running.

The native ONNX Runtime (1.28.0) is statically linked into the Rust
extension; inspect it with `faster_paddle.__runtime_build__`. See the
[upgrade measurements](docs/PERFORMANCE_CHANGES.md#v104-native-runtime-upgrade)
for what that bought.

## Usage

```python
import faster_paddle

# one-shot, using a shared default engine (lazily initialized):
with open("document.jpg", "rb") as f:
    result = faster_paddle.ocr(f.read())

print(result["text"])              # reading-order reconstructed text
for idx, b in result["bounds"].items():
    print(idx, b["text"], b["confidence"], b["topLeftCoord"], b["bottomRightCoord"])
```

For anything beyond a one-off, construct an engine once and reuse it (this
amortizes model loading; recommended for servers):

```python
from faster_paddle import OcrEngine

# model_size: "tiny" (default), "small", or "medium"
engine = OcrEngine(model_size="tiny", threads=None, det_max_side=1600)

result = engine.ocr(image_bytes)                 # raw jpeg/png/webp/bmp/tiff/gif bytes
result = engine.ocr(np_array)                    # uint8 numpy array, no encode/decode
result = engine.ocr_base64(b64_string)           # base64-encoded image
results = engine.ocr_batch(images, batch_size=4) # list in, list out, order preserved
```

Every OCR entry point also takes optional preprocessing flags
(`resize`, `denoise`, `deskew`, `binarize`): they run in fast parallel Rust
and report boxes back in your original coordinates
([details](docs/usage.md#optional-preprocessing)).

Every result carries both `text` and `structured_text`, a layout-preserving
variant of the text (see [below](#text-vs-structured_text)).

### Model sizes

| size     | det+rec  | notes                |
|----------|----------|----------------------|
| `tiny`   | ~6 MB    | default, fastest     |
| `small`  | ~31 MB   | better accuracy      |
| `medium` | ~138 MB  | best accuracy        |

A size that isn't built into your wheel is downloaded once on first use (that
one time needs network) and cached under your user cache dir
(`.../faster_paddle/v<VERSION>/{tiny,small,medium}/`). Override the cache root
with `cache_dir` (or `FASTER_PADDLE_CACHE_DIR`); an override stores
`{cache_dir}/{tiny,small,medium}/` unversioned so upgrades reuse the files.
Point the engine at local weights to stay fully offline:

```python
engine = OcrEngine(model_size="tiny", model_dir="/path/to/models")
# or:  FASTER_PADDLE_MODEL_DIR=/path/to/models
```

Layout mirrors this repo: `{model_dir}/{tiny,small,medium}/{det.onnx,rec.onnx}`,
plus an optional `char_dict.json` per size for custom recognizers. Local files
beat bundled, cached, and downloaded weights. `engine.config` reports
`det_source` / `rec_source` (`local`, `bundled`, `cached`, `downloaded`),
the resolved `model_dir`, and the resolved `cache_dir`.

To embed weights instead of downloading, build from source with cargo
features (`bundled-tiny`, `bundled-small`, `bundled-medium`, `bundled-all`;
see [docs/development.md](docs/development.md#building-from-source)).

### Result shape

```python
{
  "text": "full reconstructed text...",
  "structured_text": "layout-preserving text (see below)",
  "bounds": {
     0: {
        "topLeftCoord":     (x1, y1),
        "bottomRightCoord": (x2, y2),
        "text":             "line text",
        "confidence":       0.97,
     },
     1: { ... },
  }
}
```

`text` and `bounds` match the JSON contract of the original `paddle-ocr-api`
service, so it is a drop-in replacement.

### text vs structured_text

Both are always returned, so pick per use case:

- `structured_text` reads left-to-right, top-to-bottom while keeping the visual
  layout: whitespace gaps split the page into columns/panes, and rows inside
  each pane keep their indentation and aligned sub-columns. Use it for
  screenshots, forms, table/tree UIs, and code.
- `text` also separates page blocks, but then flattens each one into plain
  reading-order lines without indentation or sub-column alignment. Use it for
  dense multi-column prose, where `structured_text`'s pixel-accurate spacing
  produces very wide lines.

A worked example is in [docs/usage.md](docs/usage.md#structured_text).

---

## Performance

- The [performance changelog](docs/PERFORMANCE_CHANGES.md) records what was
  measured and changed (runtime upgrade, scheduling, memory); its "unverified"
  section marks numbers whose raw artifacts were not retained.
- Reproducible harnesses live in `benchmarks/` (`cpu_latency.py`,
  `batch_latency.py`, graph and memory probes). They currently target the ONNX
  Runtime backend; no committed harness reproduces the OpenVINO figures above.

## Docs

| file | contents |
|---|---|
| [docs/api.md](docs/api.md) | full function reference, constructor args, env overrides, parallelism |
| [docs/usage.md](docs/usage.md) | numpy input, batching, preprocessing, `prepare`, `structured_text` |
| [docs/development.md](docs/development.md) | pipeline internals, building from source, tests |
| [docs/PERFORMANCE_CHANGES.md](docs/PERFORMANCE_CHANGES.md) | measured changes and their evidence |

## License

MIT
