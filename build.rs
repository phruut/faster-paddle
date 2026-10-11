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
    }
}

/// One vendored runtime lib: exact filename (Windows) or a filename prefix
/// whose most-specific match wins (Linux/macOS ship versioned names like
/// `libopenvino.so.2641`, where `2641` tracks the OV release).
struct OvLib {
    prefix: &'static str,
}

fn ov_lib_patterns(target_os: &str, target_arch: &str) -> Vec<OvLib> {
    // x86_64 uses the Intel CPU plugin; ARM64 the ARM CPU plugin.
    let cpu_plugin: &'static str = if target_arch == "aarch64" || target_arch == "arm64" {
        "openvino_arm_cpu_plugin"
    } else {
        "openvino_intel_cpu_plugin"
    };
    match target_os {
        // Windows names are unversioned and verified against the 2026.4.1 libs.
        "windows" => [
            "openvino.dll",
            "openvino_c.dll",
            "openvino_ir_frontend.dll",
            "tbb12.dll",
            "tbbbind_2_5.dll",
        ]
        .iter()
        .map(|s| OvLib { prefix: s })
        .chain(std::iter::once(OvLib {
            prefix: if cpu_plugin == "openvino_arm_cpu_plugin" {
                "openvino_arm_cpu_plugin.dll"
            } else {
                "openvino_intel_cpu_plugin.dll"
            },
        }))
        .collect(),
        // Linux/macOS names below are verified against the pinned
        // `pip install openvino==OV_VERSION` layout (see CI): versioned core
        // libs (`libopenvino.so.2641`), unversioned plugins, tbb + hwloc.
        // Only linux x86_64 and macOS arm64 are verified — other targets need
        // a CI wheel run before release.
        "linux" => [
            "libopenvino.so.",
            "libopenvino_c.so.",
            "libopenvino_ir_frontend.so.",
            "libtbb.so.12",
            "libtbbbind_2_5.so.",
            "libhwloc.so.",
        ]
        .iter()
        .map(|s| OvLib { prefix: s })
        .chain(std::iter::once(OvLib {
            prefix: if cpu_plugin == "openvino_arm_cpu_plugin" {
                "libopenvino_arm_cpu_plugin.so"
            } else {
                "libopenvino_intel_cpu_plugin.so"
            },
        }))
        .collect(),
        "macos" => [
            "libopenvino.",
            "libopenvino_c.",
            "libopenvino_ir_frontend.",
            "libtbb.12",
            "libtbbbind_2_5.",
            "libhwloc.",
        ]
        .iter()
        .map(|s| OvLib { prefix: s })
        .chain(std::iter::once(OvLib {
            prefix: if cpu_plugin == "openvino_arm_cpu_plugin" {
                "libopenvino_arm_cpu_plugin.so"
            } else {
                "libopenvino_intel_cpu_plugin.so"
            },
        }))
        .collect(),
        other => panic!("openvino-backend: unsupported target OS {other:?}"),
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
        .map(std::path::PathBuf::from)
        .ok()
        .filter(|p| p.is_dir())
        .unwrap_or_else(|| {
            panic!(
                "openvino-backend needs OV_LIB_DIR pointing at the OpenVINO native libs \
                 (`pip install openvino==<OV_VERSION>`, then `<python>/site-packages/openvino/libs`)"
            )
        });
    let available: Vec<String> = std::fs::read_dir(&src)
        .unwrap_or_else(|e| panic!("openvino-backend: cannot list {}: {e}", src.display()))
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    let dest = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    std::fs::create_dir_all(&dest).expect("create OV staging dir");
    // The staged CPU plugin filename, needed below for plugins.xml.
    let mut staged_plugin: Option<String> = None;
    for pat in ov_lib_patterns(target_os, target_arch) {
        // Most-specific match wins (e.g. versioned `libopenvino.so.2641`
        // over an unversioned symlink, when archives ship both).
        let mut hits: Vec<&String> = available
            .iter()
            .filter(|f| *f == pat.prefix || f.starts_with(pat.prefix))
            .collect();
        hits.sort_by_key(|f| std::cmp::Reverse(f.len()));
        let Some(name) = hits.into_iter().next() else {
            panic!(
                "openvino-backend: nothing matching {:?} in OV_LIB_DIR={} (has: {}); \
                 expected the official OpenVINO runtime libs",
                pat.prefix,
                src.display(),
                available.join(", ")
            );
        };
        // Stage under the source filename, plugin included; pyproject's
        // out-dir include packs this whole dir into the wheel.
        let (from, to) = (src.join(name), dest.join(name));
        // Skip unchanged copies so rebuilds stay cheap.
        let fresh = to
            .metadata()
            .map(|m| m.len() != from.metadata().map(|m| m.len()).unwrap_or(u64::MAX))
            .unwrap_or(true);
        if fresh {
            std::fs::copy(&from, &to).unwrap_or_else(|e| {
                panic!("openvino-backend: copy {} failed: {e}", from.display())
            });
            println!("cargo:warning=openvino-backend: staged {name}");
        } else {
            println!("cargo:warning=openvino-backend: reused {name}");
        }
        if name.contains("cpu_plugin") {
            staged_plugin = Some(name.to_string());
        }
    }
    stage_plugins_xml(&dest, staged_plugin);
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
