# API reference

Signatures and defaults verified against the compiled extension. Usage
recipes live in [usage.md](usage.md).

## Module functions

| | |
|---|---|
| `faster_paddle.ocr(image, resize=False, denoise=False, deskew=False, binarize=False, colorspace="rgb") -> dict` | OCR encoded image bytes or a uint8 numpy array (shared default engine). |
| `faster_paddle.ocr_batch(images, *, batch_size=4, resize=False, denoise=False, deskew=False, binarize=False, colorspace="rgb") -> list[dict]` | OCR multiple encoded images/arrays; results preserve input order. |
| `faster_paddle.ocr_base64(image_base64, resize=False, denoise=False, deskew=False, binarize=False) -> dict` | OCR a base64 image string. |
| `faster_paddle.prepare(image, resize=False, denoise=False, deskew=False, binarize=False, colorspace="rgb", return_array=False) -> bytes` | Preprocess only; PNG bytes by default, `(H, W, 3)` numpy array with `return_array=True`. |

## OcrEngine

| | |
|---|---|
| `OcrEngine(...)` | Construct a reusable engine (see below for every argument). |
| `OcrEngine.ocr(image, resize=False, denoise=False, deskew=False, binarize=False, colorspace="rgb") -> dict` | OCR encoded image bytes or a uint8 numpy array. |
| `OcrEngine.ocr_batch(images, *, batch_size=4, resize=False, denoise=False, deskew=False, binarize=False, colorspace="rgb") -> list[dict]` | OCR multiple encoded images/arrays; results preserve input order. |
| `OcrEngine.rec(crops, colorspace="rgb") -> list[{"text": str, "confidence": float}]` | Recognize pre-cropped line images (bytes or arrays), skipping detection; input order preserved. |
| `OcrEngine.det(image, colorspace="rgb") -> list[{"topLeftCoord": (x1, y1), "bottomRightCoord": (x2, y2)}]` | Detect text boxes without recognizing; needs only the detection model. |
| `OcrEngine.ocr_base64(image_base64, resize=False, denoise=False, deskew=False, binarize=False) -> dict` | OCR a base64 image string. |
| `OcrEngine.prepare(image, resize=False, denoise=False, deskew=False, binarize=False, colorspace="rgb", return_array=False) -> bytes` | Preprocess only; PNG bytes by default, `(H, W, 3)` numpy array with `return_array=True`. |
| `OcrEngine.config` | Read-only dict of resolved settings (see Diagnostics). |

## Inputs

`image`/`images`/`crops` accept encoded `bytes`
(jpeg/png/webp/bmp/tiff/gif) or a C-contiguous `uint8` numpy array:
`(H, W, 3)` RGB/BGR, `(H, W)` gray, or `(H, W, 4)` RGBA/BGRA
(see [usage.md](usage.md#raw-pixel-input-numpy)).

`resize`/`denoise`/`deskew`/`binarize` are the optional preprocessing flags;
details and the execution order are in
[usage.md](usage.md#optional-preprocessing).

## Constructor

```python
OcrEngine(
    model_size="tiny", threads=None, rec_batch=None, det_max_side=None,
    *,
    det_model_size=None, rec_model_size=None, model_dir=None,
    det_min_side=None, rec_min_width=None,
    det_thresh=None, det_box_thresh=None, det_unclip_ratio=None,
    det_max_candidates=None, text_score=None, det_use_dilation=None,
    backend=None,
)
```

| argument | default | meaning |
|---|---|---|
| `model_size` | `"tiny"` | `"tiny"`, `"small"`, or `"medium"`. Sizes not built into this wheel are downloaded once and cached (see the README). |
| `threads` | auto | Total CPU budget: physical cores, bounded by process affinity and OS/container quota. A positive value overrides discovery. |
| `rec_batch` | `1` | Actual maximum crops per recognition tensor. Independent crops share the worker pool. Larger caps are available for tuning. The `ov` backend always uses 1. |
| `det_max_side` | `1600` | Detector long-side limit; recognition crops still come from the original source. Lower values can miss small text. Values <= 0 restore the reference 4000. |
| `det_model_size` | `model_size` | Detection model size. |
| `rec_model_size` | `model_size` | Recognition model size. Mixing sizes (e.g. `OcrEngine("tiny", rec_model_size="small")`) loads a fast detector with a more accurate recognizer in one engine. |
| `model_dir` | `FASTER_PADDLE_MODEL_DIR` | Directory with local weights. Each needed size falls back to bundled weights (if built in), then cache, then a one-time download. Local files win, so a code-only build stays fully offline. |
| `det_min_side` | `736` | Minimum detector short side. Set `0` to disable minimum-side upscaling. Dimensions are still rounded to multiples of 32. |
| `rec_min_width` | 64 tiny, 96 small, 320 medium | Recognition padding floor. Set `320` for the reference padding behavior. Shorter padding changes context and can change text/confidence; validate your languages and documents when migrating. Valid range 64-3200. |
| `det_thresh` | `0.2` | Detection probability-map threshold. |
| `det_box_thresh` | 0.40 tiny, 0.45 small/medium | Minimum box score for a detected box. |
| `det_unclip_ratio` | `1.4` | Box expansion ratio (`unclip`). |
| `det_max_candidates` | `3000` | Maximum detection boxes kept before scoring. |
| `text_score` | `0.0` (keep all) | Drop reads whose confidence is below this. |
| `det_use_dilation` | `False` | 2x2 OR-dilate the detection mask before connected components. Merges components with small gaps between them. |
| `backend` | wheel's runtime | `"ort"` (default wheel) or `"ov"` (OpenVINO wheel). Passing a backend the wheel wasn't built with raises a clear error. `"ov"` reads IR weights (`OV_IR_DIR` or `model_dir`) and always uses singleton `rec_batch`. |

## Environment overrides

Read once at engine construction. Constructor arguments take precedence over
the matching variable; detector/worker/Rayon counts are capped by the total
CPU budget. The CPU-count defaults are measured heuristics, not online
autotuning; benchmark other CPU architectures with
`benchmarks/cpu_latency.py` before overriding them.

| variable | overrides |
|---|---|
| `OCR_THREADS` | total CPU budget (`threads`) |
| `OCR_DET_THREADS` | detector threads (up to 8 within the budget) |
| `REC_POOL` | recognition worker pool size |
| `REC_BUDGET` | recognition scheduling budget |
| `RAYON_NUM_THREADS` | engine-local image/geometry pool |
| `OCR_REC_MIN_WIDTH` | padding floor (`rec_min_width`) |
| `OCR_DET_MIN_SIDE` | minimum detector short side (`det_min_side`) |
| `OCR_DET_THRESH` | probability-map threshold (`det_thresh`) |
| `OCR_DET_BOX_THRESH` | box score threshold (`det_box_thresh`) |
| `OCR_DET_UNCLIP_RATIO` | unclip ratio (`det_unclip_ratio`) |
| `OCR_DET_MAX_CANDIDATES` | candidate box cap (`det_max_candidates`) |
| `OCR_TEXT_SCORE` | confidence floor (`text_score`) |
| `OCR_DET_USE_DILATION` | 2x2 mask dilation (`det_use_dilation`), set `0`/`1` |
| `OCR_MEMPAT` | memory allocation pattern probe |
| `OCR_PREPACK` | weight prepacking |
| `OCR_DET_SPIN` / `OCR_REC_SPIN` | worker spin during inference |
| `OCR_APPROX_GELU` | approximate GELU. Stays off: its measured end-to-end benefit was inconsistent and it changed a detection. |
| `OCR_DEBUG` | `[dbg]` timing lines on stderr |
| `FASTER_PADDLE_MODEL_DIR` | default `model_dir` |
| `FASTER_PADDLE_RELEASE_BASE` | base URL for model downloads (local mirror) |
| `OV_IR_DIR` | fallback-free IR weight directory for `backend="ov"` |

## Parallelism

- The detector uses up to **8** threads within the CPU budget.
- Recognition uses up to one worker per available physical core, capped at 32
  (8 for medium) to limit model memory. Jobs are scheduled dynamically, largest
  estimated jobs first. Sparse long-line jobs use a separate pool of up to four
  sessions with up to four threads each, within the same CPU budget. Session
  construction is parallel in bounded groups to reduce startup latency.
- An engine-local Rayon pool handles image and geometry work within the same
  CPU budget. Workers spin during inference to reduce wakeup latency, and stop
  spinning immediately when their inference call finishes.
- Input tensor buffers are reused. Recognition budgets include actual padding.

Calls release the GIL. Inference on a shared engine is serialized; decode and
layout can overlap. Independent engines have independent CPU budgets, so set
`threads` explicitly when running several engines concurrently.

## Diagnostics

`engine.config` reports the resolved settings: thread counts, pool sizes and
input limits, the detection knob values in effect, `backend` (plus
`ov_runtime` on the OpenVINO wheel), `det_model_size`/`rec_model_size`,
`det_source`/`rec_source` (`local`, `bundled`, `cached`, `downloaded`), and
the resolved `model_dir`. Useful for diagnostics and reproducible benchmarks.

Module-level: `faster_paddle.__backends__` lists the compiled runtimes and
`faster_paddle.__runtime_build__` the linked runtime version.
