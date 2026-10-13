# Development

Pipeline internals, building from source, tests, and benchmarks. For measured
optimization history see [PERFORMANCE_CHANGES.md](PERFORMANCE_CHANGES.md).

## How it works

The pipeline faithfully mirrors PaddleOCR's lightweight path:

1. **Detection**: resize (min-side 736, cap the longer side at
   `det_max_side` = 1600 by default vs PaddleOCR's 4000, round to x32),
   normalize (BGR mean/std), run the DB detector. Lower detector resolution
   reduces work but can miss small text; recognition still crops from the
   full-res image.
2. **DB post-process**: threshold 0.2 (configurable via `det_thresh`),
   connected components, `minAreaRect`, box score `>= det_box_thresh`,
   `unclip` ratio 1.4, rescale to source coordinates.
3. **Sort** boxes top-to-bottom / left-to-right; **crop** each via perspective
   warp.
4. **Recognition**: resize each crop to H=48, normalize, batch, run the CTC
   recognizer (6,906 output classes for tiny, 18,710 for small), greedy CTC
   decode.
5. **Reconstruct** reading-order text with dynamic column/line detection.

Detection matches PaddlePaddle at **96 % IoU>0.5** with **0.93 character-level
similarity** on the recognized text; the residual difference is ONNX Runtime
vs PaddlePaddle floating-point numerics, not the algorithm.

The PP-OCRv6 ONNX models are exported with `paddle2onnx`. The OpenVINO wheel's
IR weights are converted from those same files with the pinned OpenVINO
version (see the `Models` workflow), so both runtimes read equivalent
weights.

## Building from source

```bash
pip install maturin
maturin develop --release      # build + install into the current environment
# or
maturin build --release        # produce a wheel in target/wheels/
```

Add `--features bundled-tiny`, `bundled-small`, `bundled-medium`, or
`bundled-all` to embed weights. The default wheel bundles nothing (code-only;
sizes download on first use, see the README).

Requires a Rust toolchain. ONNX Runtime is fetched automatically by the `ort`
crate at build time and bundled into the wheel (statically linked into the
extension on Windows).

### Building the OpenVINO wheel

The OpenVINO wheel ships under a separate distribution name
(`faster-paddle-openvino`) with `backend="ov"`, so both wheels can coexist as
release assets. It is built with `--no-default-features` (which drops the
`ort-backend` default) and `--features openvino-backend`, with the OpenVINO
native libs available at build time:

```bash
# one-time: pip install "openvino==2026.4.1" libclang  (pins in CI: OV_VERSION)
# then point the build at its libs (here: the pip layout):
#   $env:OV_LIB_DIR='<python>/Lib/site-packages/openvino/libs'
sed -i 's/^name = "faster-paddle"$/name = "faster-paddle-openvino"/' pyproject.toml
maturin build --release --no-default-features --features openvino-backend
```

(Windows dev shells: same rename via any string replace; CI uses bash on all
platforms, so the `sed` line works there verbatim.)

`build.rs` vendors the CPU-only OpenVINO runtime into the wheel, so the result
installs anywhere with no extra setup. IR weights are converted from the
`.onnx` files with the same pinned OpenVINO (`ov.convert_model` +
`ov.save_model`) and resolve from `OV_IR_DIR`/`model_dir`, cache, or release
download. Enabling both `ort-backend` and `openvino-backend` is possible for
local A/B builds; released wheels carry exactly one runtime.

## Tests

```bash
cargo test --release                 # Rust unit tests (geometry, resize, CTC)
maturin develop --release            # then the Python integration tests:
python -m pytest tests -q
```

The tests cover geometry equivalence, fused resize/padding, recognition caps,
known text, both small models, batch ordering and transforms, buffer reuse,
concurrent calls, error recovery, CPU affinity/budgets, detection knobs, and
preprocessing. CI also installs and runs the suite against each built wheel
(per platform, code-only and OpenVINO variants) before publishing. Install
`pytest`, `pillow`, and `numpy` for the Python suite.

The small bundled test corpus is not a comprehensive OCR accuracy benchmark.

## Benchmarks

Harnesses live in `benchmarks/` (`cpu_latency.py`, `batch_latency.py`, plus
graph and memory probes) and raw results land in `benchmarks/results/`. They
are the intended tool for judging overrides on a new CPU architecture; see
[api.md](api.md#environment-overrides) for the knobs. `scripts/convert_ir.py`
converts ONNX weights to OpenVINO IR for local OV testing (CI uses the same
script for its IR assets).
