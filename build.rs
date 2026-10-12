use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=ORT_CXX_STDLIB");
    // On Linux wheel builds we statically link the C++ runtime (libstdc++) so the
    // wheel doesn't depend on the host's libstdc++.so.6 version: built on a newer
    // toolchain (manylinux GCC 11+) it would otherwise reference symbols like
    // `std::__throw_bad_array_new_length` (GLIBCXX_3.4.29) that are missing on
    // older systems (GCC 9/10).
    //
    // The build sets `ORT_CXX_STDLIB=` (empty) so ort-sys skips its own *dynamic*
    // stdc++ link; we then link it statically here. When the var is not set (a
    // normal `cargo build`/dev build), we leave the default dynamic link so the
    // build still works without a static libstdc++.a in the toolchain.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let static_cxx = matches!(std::env::var("ORT_CXX_STDLIB").as_deref(), Ok(""));
    if target_os == "linux" && static_cxx {
        // Link libstdc++ statically at the *end* of the link line (as a link-arg,
        // which the linker places after all rlibs/archives) so it resolves the
        // C++ symbols referenced by the ONNX Runtime archive and pulls in all the
        // members it needs. `cargo:rustc-link-lib=static=stdc++` placed it too
        // early in the order, leaving the C++ runtime unresolved.
        println!("cargo:rustc-link-arg=-Wl,--push-state,-Bstatic");
        println!("cargo:rustc-link-arg=-lstdc++");
        println!("cargo:rustc-link-arg=-Wl,--pop-state");
    }
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_os == "linux" && target_arch == "aarch64" && target_env == "gnu" {
        // ORT's ARM64 archives call GCC's outlined atomic helpers. These are
        // provided by libgcc.a, not libgcc_s.so. Rust normally links only the
        // latter, leaving e.g. __aarch64_swp4_acq_rel unresolved at Python import.
        // Resolve them after the ORT and C++ archives have introduced references.
        println!("cargo:rustc-link-arg=-Wl,--push-state,-Bstatic");
        println!("cargo:rustc-link-arg=-lgcc");
        println!("cargo:rustc-link-arg=-Wl,--pop-state");
    }
    // OpenVINO wheel (`--features openvino-backend`): vendor the native
    // runtime into the wheel so it is self-contained — no system OpenVINO,
    // no PATH setup, no second package for the user. Default builds stage
    // nothing, so the minimal ORT wheel stays minimal.
    if std::env::var("CARGO_FEATURE_OPENVINO_BACKEND").is_ok() {
        // The extension dir is not on the loader's default search path on
        // ELF/Mach-O, so bake it in: installed beside the .so, the vendored
        // deps resolve without any env. (Windows searches the extension dir
        // first; no rpath needed there.)
        if target_os == "linux" {
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
            println!("cargo:rustc-link-arg=-Wl,-z,origin");
        } else if target_os == "macos" {
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
        }
        stage_ov_libs(&target_os, &target_arch);
    } else {
        // A feature-off build must not inherit files another OV build left in
        // OUT_DIR (the maturin include below packs the whole dir), so prune
        // defensively. Cargo hashes OUT_DIR per feature set, so this is
        // normally a no-op.
        prune_ov_staging(&target_os, &[]);
    }
}

/// One vendored runtime lib: a filename prefix whose most specific match wins
/// (Linux/macOS ship versioned names like `libopenvino.so.2641`, where `2641`
/// tracks the OV release; Windows names are unversioned).
struct OvLib {
    prefix: &'static str,
    /// The core library, probed for the `ov_core_set_properties` symbol.
    core: bool,
}

fn is_arm(target_arch: &str) -> bool {
    matches!(target_arch, "aarch64" | "arm64")
}

/// Filename prefixes of the vendored runtime for a target OS: the common
/// libraries, then the Intel and ARM CPU plugin names. `None` for target OSes
/// without a vendoring strategy (feature-off cleanup must not panic).
fn ov_staging_names(target_os: &str) -> Option<(&'static [&'static str], [&'static str; 2])> {
    match target_os {
        // Windows names are unversioned and verified against the 2026.4.1 libs.
        "windows" => Some((
            &[
                "openvino.dll",
                "openvino_c.dll",
                "openvino_ir_frontend.dll",
                "tbb12.dll",
                "tbbbind_2_5.dll",
            ],
            [
                "openvino_intel_cpu_plugin.dll",
                "openvino_arm_cpu_plugin.dll",
            ],
        )),
        // Linux/macOS names below are verified against the pinned
        // `pip install openvino==OV_VERSION` layout (see CI): versioned core
        // libs (`libopenvino.so.2641`), unversioned plugins, tbb + hwloc.
        // Only linux x86_64 and macOS arm64 are verified — other targets need
        // a CI wheel run before release.
        "linux" => Some((
            &[
                "libopenvino.so.",
                "libopenvino_c.so.",
                "libopenvino_ir_frontend.so.",
                "libtbb.so.12",
                "libtbbbind_2_5.so.",
                "libhwloc.so.",
            ],
            [
                "libopenvino_intel_cpu_plugin.so",
                "libopenvino_arm_cpu_plugin.so",
            ],
        )),
        "macos" => Some((
            &[
                "libopenvino.",
                "libopenvino_c.",
                "libopenvino_ir_frontend.",
                "libtbb.12",
                "libtbbbind_2_5.",
                "libhwloc.",
            ],
            [
                "libopenvino_intel_cpu_plugin.so",
                "libopenvino_arm_cpu_plugin.so",
            ],
        )),
        _ => None,
    }
}

/// The files the current target requires in the wheel (the CPU plugin is
/// per-arch). `None` for unsupported target OSes.
fn ov_lib_patterns(target_os: &str, target_arch: &str) -> Option<Vec<OvLib>> {
    let (common, plugins) = ov_staging_names(target_os)?;
    let plugin = if is_arm(target_arch) {
        plugins[1]
    } else {
        plugins[0]
    };
    let mut libs: Vec<OvLib> = common
        .iter()
        .copied()
        .enumerate()
        .map(|(i, prefix)| OvLib {
            prefix,
            core: i == 0,
        })
        .collect();
    libs.push(OvLib {
        prefix: plugin,
        core: false,
    });
    Some(libs)
}

/// The runtime name a shared library records for itself: ELF `SONAME`
/// (`readelf -d`), or the Mach-O install name (`otool -D`). `None` when the
/// file records none or the platform tool is unavailable.
fn recorded_library_name(path: &Path, target_os: &str) -> Option<String> {
    let out = match target_os {
        "linux" => Command::new("readelf").arg("-d").arg(path).output(),
        "macos" => Command::new("otool").arg("-D").arg(path).output(),
        _ => return None,
    }
    .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let name = match target_os {
        "linux" => text
            .lines()
            .find_map(|line| line.split("(SONAME)").nth(1))
            .and_then(|rest| rest.split('[').nth(1))
            .and_then(|rest| rest.split(']').next())?
            .trim(),
        // otool -D prints the file path, then the install name.
        "macos" => text.lines().nth(1)?.trim(),
        _ => return None,
    };
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

fn library_basename(raw: &str) -> &str {
    raw.rsplit('/').next().unwrap_or(raw)
}

/// True when `path` is a symlink whose canonical target stays inside `src`
/// and matches the same prefix — the archive-layout SONAME symlink.
fn is_matching_symlink(path: &Path, prefix: &str, src: &Path) -> bool {
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return false;
    };
    if !meta.file_type().is_symlink() {
        return false;
    }
    let Ok(target) = std::fs::canonicalize(path) else {
        return false;
    };
    target.starts_with(src)
        && target
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with(prefix))
}

/// Choose the file to vendor for `prefix`, most loader-correct first:
/// 1. a file already named like its recorded SONAME/install name;
/// 2. the shortest matching symlink (archive layouts name it after the SONAME);
/// 3. the longest matching filename (the pip layout ships only the real file).
fn pick_ov_file(available: &[String], prefix: &str, src: &Path, target_os: &str) -> Option<String> {
    let mut hits: Vec<&String> = available.iter().filter(|f| f.starts_with(prefix)).collect();
    hits.sort();
    if let Some(name) = hits.iter().find(|f| {
        recorded_library_name(&src.join(f.as_str()), target_os)
            .as_deref()
            .map(library_basename)
            == Some(f.as_str())
    }) {
        return Some((*name).clone());
    }
    let mut links: Vec<&String> = hits
        .iter()
        .copied()
        .filter(|f| is_matching_symlink(&src.join(f.as_str()), prefix, src))
        .collect();
    links.sort_by_key(|f| f.len());
    if let Some(name) = links.first() {
        return Some((*name).clone());
    }
    hits.sort_by_key(|f| std::cmp::Reverse(f.len()));
    hits.first().map(|f| (*f).clone())
}

/// Remove stale staged runtime files from `OUT_DIR`: anything matching the
/// vendored prefixes for this target, or `plugins.xml`, that is not in `keep`.
fn prune_ov_staging(target_os: &str, keep: &[String]) {
    let Some((common, plugins)) = ov_staging_names(target_os) else {
        return;
    };
    let dest = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let Ok(entries) = std::fs::read_dir(&dest) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let staged = name == "plugins.xml"
            || common
                .iter()
                .chain(plugins.iter())
                .any(|p| name.starts_with(p));
        if staged && !keep.contains(&name) {
            match std::fs::remove_file(entry.path()) {
                Ok(()) => println!("cargo:warning=openvino-backend: removed stale {name}"),
                Err(e) => {
                    println!("cargo:warning=openvino-backend: cannot remove stale {name}: {e}")
                }
            }
        }
    }
}

/// Copy the minimal CPU-only OpenVINO shared libs into `$OUT_DIR` (which the
/// build script owns entirely; maturin's out-dir include and the wheel's
/// package dir both need them flat, no subdirectory). pyproject's
/// `include = [{ path = "*", from = "out-dir", to = "faster_paddle/" }]`
/// packs them beside the extension, where the OS loader finds them without
/// search-path configuration. Staging in OUT_DIR instead of the source tree
/// keeps rebuilds clean and the repo free of ~90MB artifacts. Source is
/// `OV_LIB_DIR`: a `pip install openvino==OV_VERSION`'s
/// `.../site-packages/openvino/libs`, or an official archive's runtime dir.
fn stage_ov_libs(target_os: &str, target_arch: &str) {
    println!("cargo:rerun-if-env-changed=OV_LIB_DIR");
    let src = std::env::var("OV_LIB_DIR")
        .map(PathBuf::from)
        .ok()
        .filter(|p| p.is_dir())
        .unwrap_or_else(|| {
            panic!(
                "openvino-backend needs OV_LIB_DIR pointing at the OpenVINO native libs \
                 (`pip install openvino==<OV_VERSION>`, then `<python>/site-packages/openvino/libs`)"
            )
        });
    let src_canon = std::fs::canonicalize(&src).unwrap_or_else(|_| src.clone());
    let available: Vec<String> = std::fs::read_dir(&src)
        .unwrap_or_else(|e| panic!("openvino-backend: cannot list {}: {e}", src.display()))
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    let patterns = ov_lib_patterns(target_os, target_arch)
        .unwrap_or_else(|| panic!("openvino-backend: unsupported target OS {target_os:?}"));

    // Resolve every filename before touching OUT_DIR: this set drives the
    // stale-file pruning below, and a renamed file (SONAME) must be staged
    // under the name the loader will look for.
    let mut staged: Vec<(String, String)> = Vec::new();
    let mut staged_plugin: Option<String> = None;
    let mut core_name: Option<String> = None;
    for pat in &patterns {
        let Some(source_name) = pick_ov_file(&available, pat.prefix, &src_canon, target_os) else {
            panic!(
                "openvino-backend: nothing matching {:?} in OV_LIB_DIR={} (has: {}); \
                 expected the official OpenVINO runtime libs",
                pat.prefix,
                src.display(),
                available.join(", ")
            );
        };
        let staged_name = recorded_library_name(&src.join(&source_name), target_os)
            .map(|raw| library_basename(&raw).to_string())
            .filter(|n| !n.contains(|c| c == '/' || c == '\\'))
            .unwrap_or_else(|| source_name.clone());
        if staged_name.contains("cpu_plugin") {
            staged_plugin = Some(staged_name.clone());
        }
        if pat.core {
            core_name = Some(staged_name.clone());
        }
        staged.push((source_name, staged_name));
    }

    let dest = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    std::fs::create_dir_all(&dest).expect("create OV staging dir");
    let mut keep: Vec<String> = staged.iter().map(|(_, to)| to.clone()).collect();
    keep.push("plugins.xml".to_string());
    prune_ov_staging(target_os, &keep);

    for (from_name, to_name) in &staged {
        let from = src.join(from_name);
        let to = dest.join(to_name);
        // Skip unchanged copies so rebuilds stay cheap.
        let fresh = to
            .metadata()
            .map(|m| m.len() != from.metadata().map(|m| m.len()).unwrap_or(u64::MAX))
            .unwrap_or(true);
        if fresh {
            std::fs::copy(&from, &to).unwrap_or_else(|e| {
                panic!("openvino-backend: copy {} failed: {e}", from.display())
            });
            println!("cargo:warning=openvino-backend: staged {to_name}");
        } else {
            println!("cargo:warning=openvino-backend: reused {to_name}");
        }
    }
    stage_plugins_xml(&dest, staged_plugin);

    // Build-time proof that the vendored runtime is new enough for this code:
    // `set_core_properties` calls the non-variadic `ov_core_set_properties`,
    // which only exists from OpenVINO 2026.x.
    if let Some(name) = &core_name {
        verify_core_symbol(&dest.join(name), target_os);
    }
    if target_os == "macos" {
        verify_macos_vendoring(&dest, &staged, target_os);
    }
}

/// Fail the build if the staged core runtime does not export the
/// non-variadic `ov_core_set_properties` entry point (OpenVINO 2026.x API).
fn verify_core_symbol(path: &Path, target_os: &str) {
    let (args, symbol): (&[&str], &str) = match target_os {
        "linux" => (&["-D", "--defined-only"], "ov_core_set_properties"),
        "macos" => (&["-gU"], "_ov_core_set_properties"),
        // Windows: openvino-sys links the runtime's import lib at build time,
        // which fails already when the symbol is absent.
        _ => return,
    };
    let Ok(out) = Command::new("nm").args(args).arg(path).output() else {
        println!(
            "cargo:warning=openvino-backend: nm unavailable; cannot verify {symbol} in {}",
            path.display()
        );
        return;
    };
    if !out.status.success() {
        println!(
            "cargo:warning=openvino-backend: nm failed on {}; skipping runtime symbol probe",
            path.display()
        );
        return;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let found = text.lines().any(|line| {
        line.split_whitespace()
            .last()
            .map(|sym| sym.split('@').next().unwrap_or(sym) == symbol)
            .unwrap_or(false)
    });
    if !found {
        panic!(
            "openvino-backend: {} does not export {symbol}; the OV_LIB_DIR runtime predates \
             OpenVINO 2026.x, which this build's memory-safe `set_core_properties` requires. \
             Point OV_LIB_DIR at OpenVINO >= 2026.0 (CI pins 2026.4.1).",
            path.display()
        );
    }
    println!(
        "cargo:warning=openvino-backend: verified {symbol} in {}",
        path.display()
    );
}

/// Verify the staged dylibs are relocatable: their own install names and
/// every reference to a sibling vendored lib must be `@rpath/...` or
/// `@loader_path/...`, never an absolute build-machine path or a bare
/// filename (both break after wheel install). Signature/strip issues are left
/// to the CI import test and its crash diagnostics.
fn verify_macos_vendoring(dest: &Path, staged: &[(String, String)], target_os: &str) {
    let Some((common, plugins)) = ov_staging_names(target_os) else {
        return;
    };
    let is_ours = |name: &str| {
        common
            .iter()
            .chain(plugins.iter())
            .any(|p| name.starts_with(p))
    };
    let mut otool_missing = false;
    for (_, staged_name) in staged {
        let path = dest.join(staged_name);
        if let Some(id) = recorded_library_name(&path, target_os) {
            if id.starts_with('/') {
                panic!(
                    "openvino-backend: {} has absolute install name {id:?}; the vendored wheel \
                     would not load after install",
                    path.display()
                );
            }
        }
        match Command::new("otool").arg("-L").arg(&path).output() {
            Ok(out) if out.status.success() => {
                let text = String::from_utf8_lossy(&out.stdout);
                for line in text.lines().skip(1) {
                    let dep = line.trim().split(" (").next().unwrap_or("").trim();
                    if dep.is_empty() || !is_ours(library_basename(dep)) {
                        continue;
                    }
                    if !(dep.starts_with("@rpath/") || dep.starts_with("@loader_path/")) {
                        panic!(
                            "openvino-backend: {} references {dep:?}; vendored sibling libs must \
                             resolve via @rpath/@loader_path",
                            path.display()
                        );
                    }
                }
            }
            Ok(out) => println!(
                "cargo:warning=openvino-backend: otool -L failed on {} (status {:?})",
                path.display(),
                out.status.code()
            ),
            Err(_) => {
                if !otool_missing {
                    otool_missing = true;
                    println!(
                        "cargo:warning=openvino-backend: otool unavailable; skipping macOS rpath \
                         verification"
                    );
                }
            }
        }
    }
}

/// Write a minimal `plugins.xml` beside the staged libs registering the CPU
/// plugin. Without it, `ov::Core` falls back to `register_compile_time_plugins`,
/// whose baked-in candidate names only resolve when they literally match the
/// filenames (plus CWD luck) — on Linux this yields `Device with "CPU" name is
/// not registered`. With the xml present, `find_plugins_xml` discovers it next
/// to `libopenvino` and registers the absolute plugin path; the `./` prefix
/// resolves relative to the xml dir with no filename mangling on any OS.
/// Already-registered names are skipped, so platforms whose compile-time
/// fallback works (Windows) are unaffected.
fn stage_plugins_xml(dest: &std::path::Path, staged_plugin: Option<String>) {
    let plugin = staged_plugin.unwrap_or_else(|| {
        panic!("openvino-backend: staged no cpu plugin file; cannot write plugins.xml")
    });
    let xml = format!(
        "<ie>\n    <plugins>\n        <plugin name=\"CPU\" location=\"./{plugin}\"/>\n    </plugins>\n</ie>\n"
    );
    let to = dest.join("plugins.xml");
    let same = std::fs::read_to_string(&to).is_ok_and(|s| s == xml);
    if !same {
        std::fs::write(&to, &xml)
            .unwrap_or_else(|e| panic!("openvino-backend: write {} failed: {e}", to.display()));
        println!("cargo:warning=openvino-backend: staged plugins.xml (CPU -> ./{plugin})");
    } else {
        println!("cargo:warning=openvino-backend: reused plugins.xml");
    }
}
