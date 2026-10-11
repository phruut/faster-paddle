//! Opt-in OpenVINO inference backend (`openvino-backend` cargo feature).
//!
//! Build requirements: OpenVINO native libs plus libclang for bindgen.
//! Two knobs, different jobs:
//!   PATH (or OV_INSTALL_DIR) — where openvino-sys finds libs to LINK
//!     against at compile time;
//!   OV_LIB_DIR — where build.rs copies the vendored runtime DLLs/so's from
//!     into the wheel (self-contained install, no user-side setup).
//! On Windows a `pip install openvino` covers both:
//!   $env:LIBCLANG_PATH='<python>/Lib/site-packages/clang/native'
//!   $env:PATH='<python>/Lib/site-packages/openvino/libs;' + $env:PATH
//!   $env:OV_LIB_DIR='<python>/Lib/site-packages/openvino/libs'
//! Default builds never compile this module; the `*-ov` CI jobs build and
//! test the OpenVINO wheels for linux x86_64, windows x64 and macOS arm64.
//!
//! Model format note: openvino-rs 0.11 reads IR (`.xml` + `.bin`), not ONNX.
//! IR assets are converted offline from the `.onnx` files
//! (`ov.convert_model` + `ov.save_model`) and pinned per OpenVINO version.
//! Layout mirrors the ONNX layout: `{dir}/{size}/det.xml`, `det.bin`,
//! `rec.xml`, `rec.bin`.
//!
//! Recognition uses one static-shape compile per width bucket (see
//! `REC_BUCKETS`). Dynamic shapes cost OV its whole lead over ORT
//! (benchmarked ~3% faster dynamic vs ~35% static on tiny crops), so every
//! crop runs at its bucket width with zero padding on the right (trailing
//! blanks under CTC).
//! Batches are always singletons (`rec_batch` is clamped to 1): batching
//! would need a compile per (batch, width) combination.

use crate::cv;
use crate::ocr::{
    crop_quad, ctc_decode, db_postprocess, det_resize_dims_with_min, plan_rec_batches,
    quad_to_box4, rec_width, resize_to_tensor, sort_boxes, ImageRgb, OcrResult, DET_MEAN,
    DET_STD, LIMIT_SIDE_LEN, MAX_SIDE_LIMIT, REC_H, REC_MAX_W,
};
use openvino::{
    CompiledModel, Core, DeviceType, ElementType, InferRequest, PartialShape, Shape, Tensor,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

/// Static rec width buckets. Cover the full `rec_width` range (floor 64 ..
/// `REC_MAX_W` 3200) so no crop is ever downsampled into a smaller tensor —
/// tensor widths always match what the ORT backend would compute.
pub const REC_BUCKETS: [usize; 7] = [320, 640, 960, 1280, 1600, 2240, 3200];

/// Bucket index for a recognition width (ceiling into `REC_BUCKETS`).
pub fn rec_bucket_idx(w: usize) -> usize {
    for (i, &b) in REC_BUCKETS.iter().enumerate() {
        if w <= b {
            return i;
        }
    }
    REC_BUCKETS.len() - 1
}

/// Bucket width for a recognition width.
pub fn rec_bucket(w: usize) -> usize {
    REC_BUCKETS[rec_bucket_idx(w)]
}

/// Loaded backend probe: OpenVINO runtime version string.
pub fn runtime_version() -> String {
    let v = openvino::version();
    format!("{} ({})", v.description, v.build_number)
}

fn ov_err(ctx: &str, e: impl std::fmt::Debug) -> String {
    format!("openvino {ctx}: {e:?}")
}

/// Set CPU properties through the C API's non-variadic
/// `ov_core_set_properties(core, device_name, n, *const ov_property_t)`.
///
/// `openvino` 0.11's `Core::set_property`/`set_properties` route through an
/// `openvino-sys` binding that declares the *variadic*
/// `ov_core_set_property` with fixed arguments. That works on x86-64 by
/// luck (a variadic callee reads the register save area, which the fixed-arg
/// call still fills), but it is wrong on Darwin arm64, where variadic
/// arguments travel on the stack and the callee dereferences garbage
/// (observed as a segfault in CI). The `_properties` entry point exists
/// precisely for FFI callers: an ABI-stable struct array. The crate still
/// exposes no `Core` accessor, so recover the raw pointer from its
/// single-pointer representation. Drop once `openvino` wraps the
/// non-variadic call.
fn set_core_properties(core: &Core, device: &str, props: &[(&str, String)]) -> Result<(), String> {
    use std::ffi::CString;

    let ptr: *const openvino_sys::ov_core_t = unsafe { std::mem::transmute_copy(core) };
    let c_device = CString::new(device).map_err(|e| e.to_string())?;
    let owned: Vec<(CString, CString)> = props
        .iter()
        .map(|(k, v)| {
            Ok((
                CString::new(*k).map_err(|e| e.to_string())?,
                CString::new(v.as_str()).map_err(|e| e.to_string())?,
            ))
        })
        .collect::<Result<_, String>>()?;
    let ov_props: Vec<openvino_sys::ov_property_t> = owned
        .iter()
        .map(|(k, v)| openvino_sys::ov_property_t {
            key: k.as_ptr(),
            value: v.as_ptr().cast(),
        })
        .collect();
    let status = unsafe {
        openvino_sys::ov_core_set_properties(
            ptr,
            c_device.as_ptr(),
            ov_props.len(),
            ov_props.as_ptr(),
        )
    };
    if status == openvino_sys::ov_status_e::OK {
        Ok(())
    } else {
        Err(format!("ov_core_set_properties on {device} failed: {status:?}"))
    }
}

fn new_core(threads: usize, streams: Option<usize>) -> Result<Core, String> {
    let core = Core::new().map_err(|e| ov_err("Core::new", e))?;
    let mut props: Vec<(&str, String)> = vec![
        ("PERFORMANCE_HINT", "LATENCY".to_string()),
        ("INFERENCE_NUM_THREADS", threads.max(1).to_string()),
    ];
    // Per-model stream budget. AUTO gives every bucket compile its own full
    // stream set, which oversubscribes badly with a worker pool (benchmarked:
    // 7 models x AUTO streams at pool=12 tripled rec latency vs pool=2).
    // Default is one stream per pool worker; OV_NUM_STREAMS overrides.
    let streams = std::env::var("OV_NUM_STREAMS")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| streams.map(|s| s.max(1).to_string()));
    if let Some(s) = streams {
        props.push(("NUM_STREAMS", s));
    }
    set_core_properties(&core, "CPU", &props).map_err(|e| ov_err("set CPU properties", e))?;
    Ok(core)
}

struct OvRecWorker {
    /// One infer request per bucket; index parallels `REC_BUCKETS`.
    requests: Vec<InferRequest>,
    /// Reused fp32 tensor staging (sized per job to its bucket).
    input: Vec<f32>,
}

/// OpenVINO twin of `ocr::Engine`: same pipeline stages, thresholds, and
/// output contract; only the inference calls differ. See module docs for the
/// bucketing and singleton-batch constraints.
pub struct OvEngine {
    det_model: String,
    rec_model: String,
    det_request: InferRequest,
    /// Reused detector tensor + its dims; recreated when the shape changes.
    det_tensor: Option<Tensor>,
    det_tw: usize,
    det_th: usize,
    det_input: Vec<f32>,
    rec: Vec<OvRecWorker>,
    // Compiled models own the weights and must outlive every request.
    // Declared AFTER the workers so drops run requests-first.
    _det_compiled: CompiledModel,
    _rec_compiled: Vec<CompiledModel>,
    pub workers: Arc<rayon::ThreadPool>,
    rec_min_width: usize,
    det_min_side: i64,
    det_threads: usize,
    rec_threads: usize,
    threads: usize,
    chars: Vec<String>,
    box_thresh: f32,
    det_thresh: f32,
    det_unclip_ratio: f64,
    det_max_candidates: usize,
    text_score: f32,
    det_use_dilation: bool,
    det_max_side: i64,
}

impl OvEngine {
    /// Build from IR files (`{xml,bin}` pairs converted offline from ONNX).
    /// `rec_batch` is always clamped to singleton batches (see module docs).
    #[allow(clippy::too_many_arguments)]
    pub fn from_files(
        det_model: &str,
        rec_model: &str,
        det_xml: &str,
        det_bin: &str,
        rec_xml: &str,
        rec_bin: &str,
        char_dict: Vec<String>,
        threads: usize,
        det_threads: usize,
        box_thresh: f32,
        det_thresh: f32,
        det_unclip_ratio: f64,
        det_max_candidates: usize,
        text_score: f32,
        det_use_dilation: bool,
        rec_pool: usize,
        det_max_side: i64,
    ) -> Result<Self, String> {
        let pool = rec_pool.clamp(1, threads.max(1));
        let per = (threads / pool).max(1);
        let workers = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(
                    crate::env_usize("RAYON_NUM_THREADS")
                        .unwrap_or(threads)
                        .min(threads)
                        .max(1),
                )
                .build()
                .map_err(|e| format!("rayon pool: {e}"))?,
        );
        // Detector: one dynamic compile (per-image shapes vary; benchmarked
        // still far below ORT). Single request, so a single stream.
        let mut core_det = new_core(det_threads.max(1), Some(1))?;
        let det_ir = core_det
            .read_model_from_file(det_xml, det_bin)
            .map_err(|e| ov_err("read det IR", e))?;
        let mut det_compiled = core_det
            .compile_model(&det_ir, DeviceType::CPU)
            .map_err(|e| ov_err("compile det", e))?;
        let det_request = det_compiled
            .create_infer_request()
            .map_err(|e| ov_err("det request", e))?;
        // Recognizer: one static compile per bucket, then a request per
        // (worker, bucket) so dynamic job claiming never blocks on shape.
        // One stream per worker; the pool (not per-request threading) is the
        // parallelism mechanism, mirroring the ORT backend's design.
        let mut core_rec = new_core(per, Some(pool.min(4)))?;
        let mut rec_compiled = Vec::with_capacity(REC_BUCKETS.len());
        for &b in &REC_BUCKETS {
            let mut ir = core_rec
                .read_model_from_file(rec_xml, rec_bin)
                .map_err(|e| ov_err("read rec IR", e))?;
            ir.reshape_single_input(
                &PartialShape::new_static(4, &[1, 3, REC_H as i64, b as i64])
                    .map_err(|e| ov_err("bucket shape", e))?,
            )
            .map_err(|e| ov_err("reshape rec", e))?;
            rec_compiled.push(
                core_rec
                    .compile_model(&ir, DeviceType::CPU)
                    .map_err(|e| ov_err("compile rec", e))?,
            );
        }
        let mut rec = Vec::with_capacity(pool);
        for _ in 0..pool {
            let mut requests = Vec::with_capacity(REC_BUCKETS.len());
            for compiled in rec_compiled.iter_mut() {
                requests.push(
                    compiled
                        .create_infer_request()
                        .map_err(|e| ov_err("rec request", e))?,
                );
            }
            rec.push(OvRecWorker {
                requests,
                input: Vec::new(),
            });
        }
        // CHARS = ["blank"] + dict + [" "]
        let mut chars = Vec::with_capacity(char_dict.len() + 2);
        chars.push("blank".to_string());
        chars.extend(char_dict);
        chars.push(" ".to_string());
        Ok(OvEngine {
            det_model: det_model.to_string(),
            rec_model: rec_model.to_string(),
            det_request,
            det_tensor: None,
            det_tw: 0,
            det_th: 0,
            det_input: Vec::new(),
            rec,
            _det_compiled: det_compiled,
            _rec_compiled: rec_compiled,
            workers,
            rec_min_width: crate::env_usize("OCR_REC_MIN_WIDTH")
                .unwrap_or(320)
                .clamp(64, REC_MAX_W),
            det_min_side: std::env::var("OCR_DET_MIN_SIDE")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&v| v >= 0)
                .unwrap_or(LIMIT_SIDE_LEN),
            det_threads,
            rec_threads: per,
            threads,
            chars,
            box_thresh,
            det_thresh,
            det_unclip_ratio,
            det_max_candidates,
            text_score,
            det_use_dilation,
            det_max_side: if det_max_side <= 0 {
                MAX_SIDE_LIMIT
            } else {
                det_max_side
            },
        })
    }

    pub fn set_input_limits(&mut self, det_min: Option<i64>, rec_min: Option<usize>) {
        if let Some(v) = det_min {
            self.det_min_side = v;
        }
        if let Some(v) = rec_min {
            self.rec_min_width = v;
        }
    }

    pub fn settings(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("threads", self.threads),
            ("backend_ov", 1),
            ("det_threads", self.det_threads),
            ("rec_workers", self.rec.len()),
            ("rec_threads", self.rec_threads),
            ("rayon_threads", self.workers.current_num_threads()),
            ("rec_batch", 1),
            ("rec_buckets", REC_BUCKETS.len()),
            ("rec_min_width", self.rec_min_width),
            ("det_min_side", self.det_min_side as usize),
            ("det_max_side", self.det_max_side as usize),
            ("det_max_candidates", self.det_max_candidates),
            ("det_use_dilation", usize::from(self.det_use_dilation)),
        ]
    }

    pub fn model_names(&self) -> Vec<(&'static str, String)> {
        vec![
            ("det_model_size", self.det_model.clone()),
            ("rec_model_size", self.rec_model.clone()),
        ]
    }

    pub fn det_settings(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("det_thresh", self.det_thresh as f64),
            ("det_box_thresh", self.box_thresh as f64),
            ("det_unclip_ratio", self.det_unclip_ratio),
            ("text_score", self.text_score as f64),
        ]
    }

    pub fn run(&mut self, img: &ImageRgb) -> Result<Vec<OcrResult>, String> {
        let workers = Arc::clone(&self.workers);
        workers.install(|| {
            let (crops, boxes) = self.detect_crops(img)?;
            self.recognize(&crops, &boxes)
        })
    }

    pub fn run_many(&mut self, images: &[ImageRgb]) -> Result<Vec<Vec<OcrResult>>, String> {
        let workers = Arc::clone(&self.workers);
        workers.install(|| {
            let mut crops = Vec::new();
            let mut boxes = Vec::new();
            let mut counts = Vec::with_capacity(images.len());
            for img in images {
                let (cs, bs) = self.detect_crops(img)?;
                counts.push(cs.len());
                crops.extend(cs);
                boxes.extend(bs);
            }
            let mut recognized = self.recognize(&crops, &boxes)?.into_iter();
            Ok(counts
                .into_iter()
                .map(|n| recognized.by_ref().take(n).collect())
                .collect())
        })
    }

    /// Read pre-cropped lines without detecting; `(text, confidence)` per crop, in order.
    pub fn rec(&mut self, crops: &[ImageRgb]) -> Result<Vec<(String, f32)>, String> {
        let workers = Arc::clone(&self.workers);
        workers.install(|| self.recognize_texts(crops))
    }

    /// Find text boxes without reading; axis-aligned `[l, t, r, b]`, source coords.
    pub fn det(&mut self, img: &ImageRgb) -> Result<Vec<[i32; 4]>, String> {
        let workers = Arc::clone(&self.workers);
        workers.install(|| {
            let (_crops, boxes) = self.detect_crops(img)?;
            Ok(boxes.iter().map(quad_to_box4).collect())
        })
    }

    fn detect_crops(
        &mut self,
        img: &ImageRgb,
    ) -> Result<(Vec<ImageRgb>, Vec<[cv::Pt; 4]>), String> {
        let dbg = std::env::var("OCR_DEBUG").is_ok();
        let t0 = std::time::Instant::now();
        use rayon::prelude::*;
        let (rw, rh) = det_resize_dims_with_min(img.w, img.h, self.det_max_side, self.det_min_side);
        let tpn = std::time::Instant::now();
        self.det_input.resize(3 * rw * rh, 0.0);
        let input = &mut self.det_input;
        let alpha = [
            1.0 / (255.0 * DET_STD[0]),
            1.0 / (255.0 * DET_STD[1]),
            1.0 / (255.0 * DET_STD[2]),
        ];
        let beta = [
            -DET_MEAN[0] / DET_STD[0],
            -DET_MEAN[1] / DET_STD[1],
            -DET_MEAN[2] / DET_STD[2],
        ];
        resize_to_tensor(
            img,
            rw,
            rh,
            rw,
            input,
            |v, c| v as f32 * alpha[c] + beta[c],
            true,
        );
        if dbg {
            eprintln!(
                "[dbg]   det resize+normalize: {:.3}s",
                tpn.elapsed().as_secs_f64()
            );
        }
        let tinf = std::time::Instant::now();
        // (Re)create the tensor only when the shape changes; fill it in place.
        let shape_changed = self.det_tensor.is_none() || self.det_tw != rw || self.det_th != rh;
        if shape_changed {
            let shape =
                Shape::new(&[1, 3, rh as i64, rw as i64]).map_err(|e| ov_err("det shape", e))?;
            self.det_tensor = Some(
                Tensor::new(ElementType::F32, &shape).map_err(|e| ov_err("det tensor", e))?,
            );
            self.det_tw = rw;
            self.det_th = rh;
        }
        let tensor = self.det_tensor.as_mut().expect("det tensor set above");
        tensor
            .get_data_mut::<f32>()
            .map_err(|e| ov_err("det fill", e))?
            .copy_from_slice(input);
        let box_thresh = self.box_thresh;
        let det_thresh = self.det_thresh;
        let det_unclip_ratio = self.det_unclip_ratio;
        let det_max_candidates = self.det_max_candidates;
        let det_use_dilation = self.det_use_dilation;
        let t1;
        let boxes = {
            self.det_request
                .set_input_tensor(tensor)
                .map_err(|e| ov_err("det set_input", e))?;
            self.det_request
                .infer()
                .map_err(|e| ov_err("det infer", e))?;
            let out = self
                .det_request
                .get_output_tensor()
                .map_err(|e| ov_err("det output", e))?;
            let dims = out.get_shape().map_err(|e| ov_err("det out shape", e))?;
            let dims = dims.get_dimensions();
            let (ph, pw) = (dims[2] as usize, dims[3] as usize);
            if dbg {
                eprintln!("[dbg]   det OV infer: {:.3}s", tinf.elapsed().as_secs_f64());
                eprintln!(
                    "[dbg] det total ({}x{}): {:.3}s",
                    rw,
                    rh,
                    t0.elapsed().as_secs_f64()
                );
            }
            t1 = std::time::Instant::now();
            let data = out
                .get_data::<f32>()
                .map_err(|e| ov_err("det read", e))?;
            db_postprocess(
                &data,
                pw,
                ph,
                img.w,
                img.h,
                box_thresh,
                det_thresh,
                det_unclip_ratio,
                det_max_candidates,
                det_use_dilation,
            )
        };
        let boxes = sort_boxes(boxes);
        if dbg {
            eprintln!(
                "[dbg] db_postprocess ({} boxes): {:.3}s",
                boxes.len(),
                t1.elapsed().as_secs_f64()
            );
        }
        let t2 = std::time::Instant::now();
        let cropped: Vec<Option<(ImageRgb, [cv::Pt; 4])>> = boxes
            .par_iter()
            .map(|b| {
                crop_quad(img, b)
                    .filter(|c| c.w > 0 && c.h > 0)
                    .map(|c| (c, *b))
            })
            .collect();
        let mut crops: Vec<ImageRgb> = Vec::with_capacity(boxes.len());
        let mut kept_boxes: Vec<[cv::Pt; 4]> = Vec::with_capacity(boxes.len());
        for item in cropped.into_iter().flatten() {
            crops.push(item.0);
            kept_boxes.push(item.1);
        }
        if dbg {
            eprintln!(
                "[dbg] crops ({}): {:.3}s",
                crops.len(),
                t2.elapsed().as_secs_f64()
            );
        }
        Ok((crops, kept_boxes))
    }

    fn recognize(
        &mut self,
        crops: &[ImageRgb],
        kept_boxes: &[[cv::Pt; 4]],
    ) -> Result<Vec<OcrResult>, String> {
        let texts = self.recognize_texts(crops)?;
        let mut out = Vec::with_capacity(crops.len());
        let thresh = self.text_score;
        for ((t, sc), q) in texts.into_iter().zip(kept_boxes) {
            if sc < thresh {
                continue;
            }
            if thresh > 0.0 && t.trim().is_empty() {
                continue;
            }
            out.push(OcrResult {
                text: t,
                score: sc,
                box4: quad_to_box4(q),
            });
        }
        Ok(out)
    }

    fn recognize_texts(&mut self, crops: &[ImageRgb]) -> Result<Vec<(String, f32)>, String> {
        use rayon::prelude::*;
        if crops.is_empty() {
            return Ok(Vec::new());
        }
        let dbg = std::env::var("OCR_DEBUG").is_ok();
        let t3 = std::time::Instant::now();
        let min_width = self.rec_min_width;
        let rec_widths: Vec<usize> = crops
            .iter()
            .map(|c| rec_width(c.w, c.h, min_width))
            .collect();
        // Singleton batches only: static bucket shapes make multi-crop
        // batches a (batch, width) compile matrix, so every crop runs alone
        // and the worker pool supplies the parallelism.
        let mut order: Vec<usize> = (0..crops.len()).collect();
        order.sort_by_key(|&i| rec_widths[i]);
        let batches = plan_rec_batches(&order, &rec_widths, usize::MAX, 1);
        let mut jobs: Vec<usize> = (0..batches.len()).collect();
        jobs.sort_by_key(|&i| std::cmp::Reverse(rec_widths[batches[i][0]]));
        let next = AtomicUsize::new(0);
        let chars = &self.chars;
        let partial: Result<Vec<_>, String> = self
            .rec
            .par_iter_mut()
            .take(jobs.len())
            .map(|worker| {
                let mut local = Vec::new();
                loop {
                    let job = next.fetch_add(1, Ordering::Relaxed);
                    if job >= jobs.len() {
                        break;
                    }
                    let bi = jobs[job];
                    debug_assert_eq!(batches[bi].len(), 1);
                    let idx = batches[bi][0];
                    let decoded = rec_single_run(worker, crops, idx, rec_widths[idx], chars)?;
                    local.push((bi, decoded));
                }
                Ok(local)
            })
            .collect();
        let mut texts = vec![(String::new(), 0.0); crops.len()];
        for shard in partial? {
            for (bi, decoded) in shard {
                debug_assert_eq!(batches[bi].len(), 1);
                texts[batches[bi][0]] = decoded;
            }
        }
        if dbg {
            eprintln!(
                "[dbg] rec ({} crops): {:.3}s",
                crops.len(),
                t3.elapsed().as_secs_f64()
            );
        }
        Ok(texts)
    }
}

/// Run one singleton recognition crop on a worker's bucket request and CTC-decode it.
fn rec_single_run(
    worker: &mut OvRecWorker,
    crops: &[ImageRgb],
    idx: usize,
    w: usize,
    chars: &[String],
) -> Result<(String, f32), String> {
    let dbg = std::env::var("OCR_DEBUG").is_ok();
    let tp = std::time::Instant::now();
    let bucket = rec_bucket(w);
    let bi = rec_bucket_idx(w);
    let c = &crops[idx];
    // resized width mirrors the ORT runner (ceil ratio, clamped to the tensor
    // width); buckets cover the full range so `bucket >= w` always holds.
    let ratio = c.w as f64 / c.h as f64;
    let resized_w = ((REC_H as f64 * ratio).ceil() as usize)
        .min(bucket)
        .max(1);
    let plane = REC_H * bucket;
    worker.input.resize(3 * plane, 0.0);
    let data = &mut worker.input;
    resize_to_tensor(
        c,
        resized_w,
        REC_H,
        bucket,
        data,
        |v, _| (v as f32 / 255.0 - 0.5) / 0.5,
        false,
    );
    let prep_s = tp.elapsed().as_secs_f64();
    let ti = std::time::Instant::now();
    let shape = Shape::new(&[1, 3, REC_H as i64, bucket as i64])
        .map_err(|e| ov_err("rec shape", e))?;
    let mut tensor =
        Tensor::new(ElementType::F32, &shape).map_err(|e| ov_err("rec tensor", e))?;
    tensor
        .get_data_mut::<f32>()
        .map_err(|e| ov_err("rec fill", e))?
        .copy_from_slice(data);
    let req = &mut worker.requests[bi];
    req.set_input_tensor(&tensor)
        .map_err(|e| ov_err("rec set_input", e))?;
    req.infer().map_err(|e| ov_err("rec infer", e))?;
    let out = req.get_output_tensor().map_err(|e| ov_err("rec output", e))?;
    let dims = out.get_shape().map_err(|e| ov_err("rec out shape", e))?;
    let dims = dims.get_dimensions();
    let (t, cls) = (dims[1] as usize, dims[2] as usize);
    let infer_s = ti.elapsed().as_secs_f64();
    let tc = std::time::Instant::now();
    let data = out.get_data::<f32>().map_err(|e| ov_err("rec read", e))?;
    let res = ctc_decode(chars, &data, t, cls);
    if dbg {
        eprintln!(
            "[dbg]     rec crop w={w} bucket={bucket}: prep {prep_s:.3}s infer {infer_s:.3}s ctc {:.3}s",
            tc.elapsed().as_secs_f64()
        );
    }
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_cover_full_rec_range() {
        assert_eq!(rec_bucket(64), 320);
        assert_eq!(rec_bucket(320), 320);
        assert_eq!(rec_bucket(321), 640);
        assert_eq!(rec_bucket(1261), 1280);
        assert_eq!(rec_bucket(1280), 1280);
        assert_eq!(rec_bucket(1281), 1600);
        assert_eq!(rec_bucket(3200), 3200);
        // widest bucket always fits the widest tensor the ORT path computes
        assert!(REC_BUCKETS.contains(&REC_MAX_W));
    }
}
