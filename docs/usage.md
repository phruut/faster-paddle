# Usage details

Recipes that go beyond the README quickstart. Full signatures live in
[api.md](api.md).

## Raw pixel input (numpy)

`ocr`, `ocr_batch`, `det`, `rec`, and `prepare` also accept uint8 numpy arrays
directly. Useful for OpenCV frames, PIL images, and video loops, where encoding
to PNG/JPG just to decode it again is wasted work:

```python
import numpy as np
from PIL import Image

arr = np.asarray(Image.open("document.jpg").convert("RGB"))  # (H, W, 3) RGB
result = engine.ocr(arr)

bgr = cv2.imread("document.jpg")                 # (H, W, 3) BGR from OpenCV
result = engine.ocr(bgr, colorspace="bgr")

gray = np.asarray(Image.open("document.jpg").convert("L"))  # (H, W) grayscale
result = engine.ocr(gray)  # (H, W, 4) RGBA also works; 4-channel uses BGRA order with colorspace="bgr"
```

Arrays must be `uint8` and C-contiguous (`np.ascontiguousarray` if needed).
`colorspace="rgb"` (default) or `"bgr"` applies to 3-channel (and 4-channel)
arrays; encoded bytes are always decoded as RGB. `prepare(...,
return_array=True)` returns a `(H, W, 3)` array instead of PNG bytes, so its
output can go straight back into `ocr` with no codec round-trip.

## Multiple images

`ocr_batch` takes a list of encoded image bytes or numpy arrays (see
[raw pixel input](#raw-pixel-input-numpy)) and returns one result dictionary per
image, in the same order:

```python
from pathlib import Path
from faster_paddle import OcrEngine

paths = [Path("page1.png"), Path("page2.jpg"), Path("page3.png")]
images = [path.read_bytes() for path in paths]

engine = OcrEngine(model_size="small")
results = engine.ocr_batch(images, batch_size=4)
for path, result in zip(paths, results):
    print(path.name, result["text"])
```

The shared default tiny engine also supports batches:

```python
import faster_paddle

results = faster_paddle.ocr_batch(images, batch_size=4, resize=True)
```

`ocr_batch(images, *, batch_size=4, resize=False, denoise=False, deskew=False,
binarize=False)` processes bounded windows of **images**. It decodes and
preprocesses in parallel and schedules recognition crops across pages. Each
page keeps its own detector resolution and coordinate transform. `batch_size`
controls the number of decoded pages resident in a window; it is independent
of `rec_batch`. Input bytes are borrowed when possible. An empty list returns
`[]`; invalid images raise an error with their zero-based input index.

Batching helps fill recognition workers on sparse pages and can improve
throughput. It is not guaranteed to accelerate dense pages, and the caller
waits for the complete result list. Use `ocr` for minimum time to the first
page's result. Minor floating-point confidence differences can occur between
the single-line and multi-page execution paths.

## Optional preprocessing

`ocr`, `ocr_base64`, and `ocr_batch` take four optional flags (all default
`False`), applied, when enabled, in the optimal order, all in fast parallel
Rust:

```python
result = engine.ocr(
    image_bytes,
    resize=True,     # 1. downscale source to <= 2100x3000 (aspect preserved) if larger
    denoise=True,    # 2. fast Non-Local-Means denoise (grayscale)
    deskew=True,     # 3. detect skew (Canny + Hough) and rotate to straighten
    binarize=True,   # 4. Sauvola adaptive thresholding (clean black/white)
)
```

Order rationale: resize first (everything downstream is then faster), denoise
before angle detection and thresholding, deskew on the cleaned image, binarize
last to produce the final B/W. `resize` can reduce source/crop work, but may
add an extra resampling pass when detection already hits its size cap;
benchmark it on your input. Any of `denoise`/`deskew`/`binarize` converts the
image to grayscale.

Returned `bounds` are always in the **original image's coordinate space**. Even
when `resize` or `deskew` changes the working image, the boxes are mapped back
so they line up with your input.

## Preprocess only (no OCR)

`prepare` runs the same preprocessing in one pass and returns the prepared
image as **PNG bytes** (grayscale once any of denoise/deskew/binarize is on,
else color). If every option is `False` the original bytes are returned
unchanged.

```python
prepared = engine.prepare(image_bytes, resize=True, denoise=True, deskew=True, binarize=False)
# or module-level:  faster_paddle.prepare(image_bytes, resize=True, ...)

with open("prepared.png", "wb") as f:
    f.write(prepared)
# you can also feed it straight back in:
result = engine.ocr(prepared)
```

## structured_text

A spatial reconstruction that reads **left-to-right, top-to-bottom** while
preserving the visual layout: vertical whitespace gaps split the page into
columns/panes (each read fully before the next), and within each one the rows
are laid out as a monospace grid, so indentation (tree nesting) and aligned
sub-columns (key/value tables) are kept. Single-glyph UI icon noise is
dropped.

Use **`structured_text`** for screenshots, forms, table/tree UIs, and code:
anything where spatial structure carries meaning. Use **`text`** for dense
multi-column prose: there the absolute pixel spacing of `structured_text`
produces very wide lines, so the column-merging `text` reconstruction reads
better. Both are always returned, so you can pick per use case.

Example `structured_text` for a two-pane file-tree + settings UI:

```
Project
 src (14)
   main.rs
   parser.rs
   utils.rs
 tests
 docs

Setting                                            Value
        max_connections                            128
        request_timeout_seconds                    30
        cache_size_mb                              512
```
