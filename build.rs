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
    /// The C API library, probed for the `ov_core_set_properties` symbol. The
    /// symbol lives in `openvino_c` (the C API), not the C++ `openvino` core.
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
        .map(|prefix| OvLib {
            prefix,
            core: prefix.contains("openvino_c"),
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
            .filter(|n| !n.contains(['/', '\\']))
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
        patch_macos_deps(&dest, &staged, target_os);
        verify_macos_vendoring(&dest, &staged, target_os);
    }
}

/// Fail the build if the staged C API runtime does not export the
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

/// True when `name` is one of the OpenVINO runtime files this build vendors
/// for `target_os` (the staged sibling set).
fn is_vendored_ov_lib(target_os: &str, name: &str) -> bool {
    let Some((common, plugins)) = ov_staging_names(target_os) else {
        return false;
    };
    common
        .iter()
        .chain(plugins.iter())
        .any(|p| name.starts_with(p))
}

/// Run a tool that edits `path`, panicking with its stderr on failure. A
/// partly applied edit or an un-re-signed dylib would ship a wheel that cannot
/// load, so after a needed edit there is no safe "warn and continue".
fn run_macos_edit_tool(cmd: &mut Command, path: &Path, what: &str) {
    let out = cmd.arg(path).output().unwrap_or_else(|e| {
        panic!(
            "openvino-backend: cannot run {what} on {}: {e}; macOS command line tools are needed \
             to make the vendored runtime relocatable",
            path.display()
        )
    });
    if !out.status.success() {
        panic!(
            "openvino-backend: {what} failed on {} (status {:?}): {}",
            path.display(),
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
}

/// `(file offset, name)` for every dylib install-name string in a thin 64-bit
/// little-endian Mach-O: `LC_ID_DYLIB` and the `LC_*_DYLIB` load commands
/// (`LC_REQ_DYLD` bit masked off).
fn macho_dylib_names(data: &[u8]) -> Result<Vec<(usize, String)>, String> {
    const DYLIB_CMDS: [u32; 6] = [0xc, 0xd, 0x18, 0x1f, 0x20, 0x23];
    fn word(data: &[u8], at: usize) -> Result<u32, String> {
        data.get(at..at + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .ok_or_else(|| format!("file ends inside the Mach-O header at offset {at}"))
    }
    let magic = word(data, 0)?;
    if magic != 0xfeed_facf {
        return Err(format!(
            "not a thin 64-bit little-endian Mach-O (magic {magic:#010x})"
        ));
    }
    let ncmds = word(data, 16)? as usize;
    let sizeofcmds = word(data, 20)? as usize;
    let lc_end = 32usize
        .checked_add(sizeofcmds)
        .filter(|end| *end <= data.len())
        .ok_or("load commands extend past end of file")?;
    let mut names = Vec::new();
    let mut off = 32usize;
    for _ in 0..ncmds {
        let cmd = word(data, off)?;
        let cmdsize = word(data, off + 4)? as usize;
        let end = off
            .checked_add(cmdsize)
            .filter(|end| cmdsize >= 8 && *end <= lc_end)
            .ok_or_else(|| format!("invalid load command size {cmdsize} at offset {off}"))?;
        if DYLIB_CMDS.contains(&(cmd & 0x7fff_ffff)) {
            let name_off = word(data, off + 8)? as usize;
            let start = off
                .checked_add(name_off)
                .filter(|start| *start < end)
                .ok_or("dylib command name offset is out of bounds")?;
            let nul = data[start..end]
                .iter()
                .position(|b| *b == 0)
                .ok_or("dylib command name is not NUL-terminated")?;
            names.push((
                start,
                String::from_utf8_lossy(&data[start..start + nul]).into_owned(),
            ));
        }
        off = end;
    }
    Ok(names)
}

/// Rewrite matching dylib install-name strings inside a thin 64-bit Mach-O.
/// Apple's `install_name_tool` refuses these pip-wheel dylibs ("link edit
/// information does not fill the __LINKEDIT segment"), so the swap happens
/// directly: a replacement must not outgrow the original NUL-terminated
/// string (`@loader_path/` is shorter than the absolute paths here), which
/// lets it be overwritten in place with the tail NUL-padded — sizes, offsets,
/// and every other load command stay byte-identical. The caller re-signs the
/// file afterwards. Returns the applied `old -> new` descriptions.
fn rewrite_macho_dylib_names(
    path: &Path,
    rewrites: &[(String, String)],
) -> Result<Vec<String>, String> {
    let mut data =
        std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let applied = rewrite_macho_dylib_names_in(&mut data, rewrites)?;
    if !applied.is_empty() {
        std::fs::write(path, &data).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    }
    Ok(applied)
}

/// The in-memory half of `rewrite_macho_dylib_names` (unit-tested).
fn rewrite_macho_dylib_names_in(
    data: &mut [u8],
    rewrites: &[(String, String)],
) -> Result<Vec<String>, String> {
    let names = macho_dylib_names(data)?;
    let mut applied = Vec::new();
    for (offset, name) in names {
        let Some((_, new)) = rewrites.iter().find(|(old, _)| *old == name) else {
            continue;
        };
        if new.len() > name.len() {
            return Err(format!(
                "replacement {new:?} is longer than the original load-command string {name:?}"
            ));
        }
        data[offset..offset + name.len()].fill(0);
        data[offset..offset + new.len()].copy_from_slice(new.as_bytes());
        applied.push(format!("{name} -> {new}"));
    }
    Ok(applied)
}

/// Remove the code-signature payload from a thin 64-bit little-endian Mach-O.
/// Apple's `codesign --remove-signature` refuses the pip wheel's dylibs with
/// "internal error in Code Signing subsystem" (cctools reports the same files
/// as "link edit information does not fill the __LINKEDIT segment"), so the
/// strip happens byte-wise like the name rewrite: delete the
/// `LC_CODE_SIGNATURE` load command, truncate the file at the old signature
/// start, and shrink __LINKEDIT to the truncated end. This is what a
/// successful `--remove-signature` leaves behind; zeroing the command instead
/// makes codesign complain about a ghost ("invalid or unsupported format for
/// signature"), so the command is removed outright. The result is a canonical
/// unsigned file that `codesign --force --sign -` can sign fresh. Returns
/// whether a signature was stripped.
fn strip_macho_signature(path: &Path) -> Result<bool, String> {
    let mut data =
        std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let stripped = strip_macho_signature_in(&mut data)?;
    if stripped {
        std::fs::write(path, &data).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    }
    Ok(stripped)
}

/// The in-memory half of `strip_macho_signature` (unit-tested).
fn strip_macho_signature_in(data: &mut Vec<u8>) -> Result<bool, String> {
    const LC_SYMTAB: u32 = 0x2;
    const LC_DYSYMTAB: u32 = 0xb;
    const LC_SEGMENT_64: u32 = 0x19;
    const LC_CODE_SIGNATURE: u32 = 0x1d;
    const LC_DYLD_INFO: u32 = 0x22;
    const LC_FUNCTION_STARTS: u32 = 0x26;
    const LC_DATA_IN_CODE: u32 = 0x29;
    const LC_DYLIB_CODE_SIGN_DRS: u32 = 0x2b;
    const LC_LINKER_OPTIMIZATION_HINT: u32 = 0x2e;
    const LC_DYLD_CHAINED_FIXUPS: u32 = 0x34;

    fn word(data: &[u8], at: usize) -> Result<u32, String> {
        data.get(at..at + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .ok_or_else(|| format!("file ends inside the Mach-O header at offset {at}"))
    }
    fn data_end(data: &[u8], off: usize) -> Result<u64, String> {
        Ok(u64::from(word(data, off)?) + u64::from(word(data, off + 4)?))
    }

    let magic = word(data, 0)?;
    if magic != 0xfeed_facf {
        return Err(format!(
            "not a thin 64-bit little-endian Mach-O (magic {magic:#010x})"
        ));
    }
    let ncmds = word(data, 16)? as usize;
    let sizeofcmds = word(data, 20)? as usize;
    let lc_end = 32usize
        .checked_add(sizeofcmds)
        .filter(|end| *end <= data.len())
        .ok_or("load commands extend past end of file")?;

    let mut cs_off: Option<usize> = None;
    let mut cs_cmdsize: usize = 0;
    let mut linkedit_fileoff: Option<u64> = None;
    let mut linkedit_filesize_at: usize = 0;
    // Furthest byte any file-backed __LINKEDIT structure occupies. Truncating
    // at the signature start is only safe when everything else ends before it.
    let mut content_end = 0u64;

    let mut off = 32usize;
    for _ in 0..ncmds {
        let cmd = word(data, off)?;
        let cmdsize = word(data, off + 4)? as usize;
        let end = off
            .checked_add(cmdsize)
            .filter(|end| cmdsize >= 8 && *end <= lc_end)
            .ok_or_else(|| format!("invalid load command size {cmdsize} at offset {off}"))?;
        match cmd & 0x7fff_ffff {
            LC_SEGMENT_64 => {
                let name = data.get(off + 8..off + 24).ok_or("segment ends inside header")?;
                if name.starts_with(b"__LINKEDIT")
                    && name[10..].iter().all(|b| *b == 0 || *b == b' ')
                {
                    let lo = u64::from(word(data, off + 40)?)
                        | (u64::from(word(data, off + 44)?) << 32);
                    linkedit_fileoff = Some(lo);
                    linkedit_filesize_at = off + 48;
                }
            }
            LC_SYMTAB => {
                let symoff = u64::from(word(data, off + 8)?);
                let nsyms = u64::from(word(data, off + 12)?);
                let stroff = u64::from(word(data, off + 16)?);
                let strsize = u64::from(word(data, off + 20)?);
                content_end = content_end.max(symoff + 16 * nsyms).max(stroff + strsize);
            }
            LC_DYSYMTAB => {
                let w = |i: usize| -> Result<u64, String> { Ok(u64::from(word(data, off + 8 + 4 * i)?)) };
                content_end = content_end
                    .max(w(6)? + 8 * w(7)?) // tocoff, ntoc
                    .max(w(8)? + 56 * w(9)?) // modtaboff, nmodtab
                    .max(w(10)? + 4 * w(11)?) // extrefsymoff, nextrefsyms
                    .max(w(12)? + 4 * w(13)?) // indirectsymoff, nindirectsyms
                    .max(w(14)? + 8 * w(15)?) // extreloff, nextrel
                    .max(w(16)? + 8 * w(17)?); // locreloff, nlocrel
            }
            LC_DYLD_INFO => {
                for i in 0..5 {
                    content_end = content_end
                        .max(data_end(data, off + 8 + 8 * i)?);
                }
            }
            LC_FUNCTION_STARTS | LC_DATA_IN_CODE | LC_DYLIB_CODE_SIGN_DRS
            | LC_LINKER_OPTIMIZATION_HINT | LC_DYLD_CHAINED_FIXUPS => {
                let start = u64::from(word(data, off + 8)?);
                if start > 0 {
                    content_end = content_end.max(start + u64::from(word(data, off + 12)?));
                }
            }
            LC_CODE_SIGNATURE => {
                cs_off = Some(off);
                cs_cmdsize = cmdsize;
            }
            _ => {}
        }
        off = end;
    }

    let Some(cs_off) = cs_off else {
        return Ok(false);
    };
    let cs_dataoff = u64::from(word(data, cs_off + 8)?);
    let cs_datasize = u64::from(word(data, cs_off + 12)?);
    if cs_datasize == 0 {
        return Ok(false);
    }
    let cs_end = cs_dataoff + cs_datasize;
    if cs_end > data.len() as u64 {
        return Err(format!(
            "signature region [0x{cs_dataoff:x}, 0x{cs_end:x}) extends past end of file"
        ));
    }
    let Some(linkedit_fileoff) = linkedit_fileoff else {
        return Err("no __LINKEDIT segment to shrink".to_string());
    };
    if cs_dataoff < linkedit_fileoff {
        return Err(format!(
            "signature starts at 0x{cs_dataoff:x}, before __LINKEDIT at 0x{linkedit_fileoff:x}"
        ));
    }
    if content_end > cs_dataoff {
        return Err(format!(
            "a __LINKEDIT structure ends at 0x{content_end:x}, inside the signature at \
             0x{cs_dataoff:x}; refusing to truncate"
        ));
    }
    if cs_cmdsize != 16 {
        return Err(format!(
            "LC_CODE_SIGNATURE has cmdsize {cs_cmdsize}, expected 16; refusing to delete it"
        ));
    }
    // Delete the command: shift everything after it left by 16 bytes. The 16
    // leftover bytes at the end of the command region are zeroed so no stale
    // copy of the signature command lingers for sloppy readers; codesign
    // overwrites that slot when it adds its own entry.
    let tail_end = off;
    data.copy_within(cs_off + cs_cmdsize..tail_end, cs_off);
    data[tail_end - cs_cmdsize..tail_end].fill(0);
    let ncmds = u64::from(word(data, 16)?);
    let sizeofcmds = u64::from(word(data, 20)?);
    let ncmds = ncmds
        .checked_sub(1)
        .ok_or("load command count underflows")?;
    let sizeofcmds = sizeofcmds
        .checked_sub(cs_cmdsize as u64)
        .ok_or("load command size underflows")?;
    data[16..20].copy_from_slice(&(ncmds as u32).to_le_bytes());
    data[20..24].copy_from_slice(&(sizeofcmds as u32).to_le_bytes());
    let new_size = cs_dataoff - linkedit_fileoff;
    // The __LINKEDIT command may have moved left by 16 bytes when it sat
    // behind the deleted one (in practice it always precedes it).
    let filesize_at = if linkedit_filesize_at > cs_off {
        linkedit_filesize_at - cs_cmdsize
    } else {
        linkedit_filesize_at
    };
    data[filesize_at..filesize_at + 4]
        .copy_from_slice(&(new_size as u32).to_le_bytes());
    data[filesize_at + 4..filesize_at + 8]
        .copy_from_slice(&((new_size >> 32) as u32).to_le_bytes());
    // Now drop the signature payload and the trailing gap after it: the file
    // ends exactly where the unsigned content does, like a never-signed file.
    data.truncate(cs_dataoff as usize);
    Ok(true)
}

/// Rewrite macOS dylib metadata so the vendored runtime is relocatable: a
/// dependency on a staged sibling that records an absolute build-machine path
/// (the pinned pip wheel ships e.g. `libtbbbind` referencing Homebrew's
/// `/usr/local/lib/libhwloc.15.dylib`) becomes `@loader_path/<name>`, and an
/// absolute install name becomes `@rpath/<name>`. The edit invalidates the
/// Mach-O signature, which codesign cannot remove or replace on these files
/// ("internal error in Code Signing subsystem"), so each modified file is
/// stripped byte-wise (`strip_macho_signature`) and signed ad-hoc fresh
/// (arm64 refuses to load stale-signed dylibs). `@loader_path` is safe
/// because every staged file lands flat beside the extension.
/// `verify_macos_vendoring` below is the post-condition proof of this rewrite.
fn patch_macos_deps(dest: &Path, staged: &[(String, String)], target_os: &str) {
    for (_, staged_name) in staged {
        let path = dest.join(staged_name);
        let Ok(out) = Command::new("otool").arg("-L").arg(&path).output() else {
            println!(
                "cargo:warning=openvino-backend: otool unavailable; cannot rewrite macOS \
                 dependency paths (the rpath verification below will report)"
            );
            return;
        };
        if !out.status.success() {
            println!(
                "cargo:warning=openvino-backend: otool -L failed on {} (status {:?})",
                path.display(),
                out.status.code()
            );
            continue;
        }
        let text = String::from_utf8_lossy(&out.stdout);
        let mut rewrites: Vec<(String, String)> = Vec::new();
        for line in text.lines().skip(1) {
            let dep = line.trim().split(" (").next().unwrap_or("").trim();
            if dep.is_empty() || dep.starts_with("@rpath/") || dep.starts_with("@loader_path/") {
                continue;
            }
            let base = library_basename(dep);
            if is_vendored_ov_lib(target_os, base) && !rewrites.iter().any(|(old, _)| old == dep) {
                rewrites.push((dep.to_string(), format!("@loader_path/{base}")));
            }
        }
        if let Some(id) = recorded_library_name(&path, target_os).filter(|id| id.starts_with('/')) {
            rewrites.push((id, format!("@rpath/{staged_name}")));
        }
        if rewrites.is_empty() {
            continue;
        }
        let applied = match rewrite_macho_dylib_names(&path, &rewrites) {
            Ok(applied) => applied,
            Err(e) => {
                println!(
                    "cargo:warning=openvino-backend: cannot rewrite {staged_name}: {e} \
                     (the rpath verification below will report)"
                );
                continue;
            }
        };
        if applied.is_empty() {
            println!(
                "cargo:warning=openvino-backend: no load-command string matched the rewrites \
                 for {staged_name}; leaving it to the rpath verification"
            );
            continue;
        }
        // The pip wheel's dylib signatures are non-canonical: cctools refuses
        // them ("link edit information does not fill the __LINKEDIT segment"),
        // and codesign's remove and replace paths both die with "internal
        // error in Code Signing subsystem". Strip byte-wise instead (see
        // `strip_macho_signature`), then sign the now-unsigned file fresh —
        // arm64 refuses to load stale-signed dylibs. A file that cannot be
        // stripped or signed is a hard error.
        match strip_macho_signature(&path) {
            Ok(true) => println!(
                "cargo:warning=openvino-backend: stripped the malformed code signature from \
                 {staged_name}"
            ),
            Ok(false) => {}
            Err(e) => panic!(
                "openvino-backend: cannot strip the code signature from {}: {e}",
                path.display()
            ),
        }
        let mut cmd = Command::new("codesign");
        cmd.args(["--force", "--sign", "-"]);
        run_macos_edit_tool(&mut cmd, &path, "codesign");
        println!(
            "cargo:warning=openvino-backend: patched {staged_name} ({}) and re-signed ad-hoc",
            applied.join(", ")
        );
    }
}

/// Verify the staged dylibs are relocatable after `patch_macos_deps`: their
/// own install names and every reference to a sibling vendored lib must be
/// `@rpath/...` or `@loader_path/...`, never an absolute build-machine path
/// or a bare filename (both break after wheel install). This is the
/// post-condition proof of the rewrite; signature issues are left to the CI
/// import test and its crash diagnostics.
fn verify_macos_vendoring(dest: &Path, staged: &[(String, String)], target_os: &str) {
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
                    if dep.is_empty() || !is_vendored_ov_lib(target_os, library_basename(dep)) {
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

// `cargo test` does not run build-script tests; run them with:
//   rustc --edition 2021 --test build.rs -o build-rs-tests && ./build-rs-tests
#[cfg(test)]
mod tests {
    use super::*;

    fn dylib_command(cmd: u32, name: &str) -> Vec<u8> {
        let size = (24 + name.len() + 1 + 7) & !7;
        let mut lc = vec![0u8; size];
        lc[0..4].copy_from_slice(&cmd.to_le_bytes());
        lc[4..8].copy_from_slice(&(size as u32).to_le_bytes());
        lc[8..12].copy_from_slice(&24u32.to_le_bytes());
        lc[24..24 + name.len()].copy_from_slice(name.as_bytes());
        lc
    }

    /// Minimal thin 64-bit little-endian Mach-O holding only load commands.
    fn thin_macho(commands: &[(u32, &str)]) -> Vec<u8> {
        let mut lcs = Vec::new();
        for (cmd, name) in commands {
            lcs.extend_from_slice(&dylib_command(*cmd, name));
        }
        let mut out = vec![0u8; 32];
        out[0..4].copy_from_slice(&0xfeed_facfu32.to_le_bytes());
        out[16..20].copy_from_slice(&(commands.len() as u32).to_le_bytes());
        out[20..24].copy_from_slice(&(lcs.len() as u32).to_le_bytes());
        out.extend_from_slice(&lcs);
        out
    }

    fn lc(cmd: u32, payload: &[u8]) -> Vec<u8> {
        let size = (8 + payload.len() + 7) & !7;
        let mut v = vec![0u8; size];
        v[0..4].copy_from_slice(&cmd.to_le_bytes());
        v[4..8].copy_from_slice(&(size as u32).to_le_bytes());
        v[8..8 + payload.len()].copy_from_slice(payload);
        v
    }

    fn segment_command(name: &str, fileoff: u64, filesize: u64) -> Vec<u8> {
        let mut p = vec![0u8; 64];
        p[..name.len()].copy_from_slice(name.as_bytes());
        p[32..40].copy_from_slice(&fileoff.to_le_bytes());
        p[40..48].copy_from_slice(&filesize.to_le_bytes());
        lc(0x19, &p)
    }

    fn symtab_command(symoff: u32, nsyms: u32, stroff: u32, strsize: u32) -> Vec<u8> {
        let mut p = vec![0u8; 16];
        p[0..4].copy_from_slice(&symoff.to_le_bytes());
        p[4..8].copy_from_slice(&nsyms.to_le_bytes());
        p[8..12].copy_from_slice(&stroff.to_le_bytes());
        p[12..16].copy_from_slice(&strsize.to_le_bytes());
        lc(0x2, &p)
    }

    fn code_signature_command(dataoff: u32, datasize: u32) -> Vec<u8> {
        let mut p = vec![0u8; 8];
        p[0..4].copy_from_slice(&dataoff.to_le_bytes());
        p[4..8].copy_from_slice(&datasize.to_le_bytes());
        lc(0x1d, &p)
    }

    /// Mach-O with the given load commands plus zero fill to `len` bytes.
    fn macho_with(commands: &[Vec<u8>], len: usize) -> Vec<u8> {
        let mut lcs = Vec::new();
        for c in commands {
            lcs.extend_from_slice(c);
        }
        let mut out = vec![0u8; len];
        out[0..4].copy_from_slice(&0xfeed_facfu32.to_le_bytes());
        out[16..20].copy_from_slice(&(commands.len() as u32).to_le_bytes());
        out[20..24].copy_from_slice(&(lcs.len() as u32).to_le_bytes());
        out[32..32 + lcs.len()].copy_from_slice(&lcs);
        out
    }

    #[test]
    fn macho_names_are_located() {
        let data = thin_macho(&[
            (0xc, "/usr/local/lib/libhwloc.15.dylib"),
            (0x8000_0018, "@rpath/libtbbbind_2_5.3.dylib"),
        ]);
        let names = macho_dylib_names(&data).unwrap();
        assert_eq!(names.len(), 2);
        assert_eq!(names[0].1, "/usr/local/lib/libhwloc.15.dylib");
        assert_eq!(names[1].1, "@rpath/libtbbbind_2_5.3.dylib");
    }

    #[test]
    fn rewrite_shrinks_in_place() {
        let old = "/usr/local/lib/libhwloc.15.dylib";
        let new = "@loader_path/libhwloc.15.dylib";
        let mut data = thin_macho(&[(0xc, old), (0x8000_0018, "@rpath/libtbbbind_2_5.3.dylib")]);
        let len = data.len();
        let applied =
            rewrite_macho_dylib_names_in(&mut data, &[(old.to_string(), new.to_string())]).unwrap();
        assert_eq!(applied, vec![format!("{old} -> {new}")]);
        assert_eq!(data.len(), len, "in-place edit must not change file size");
        let names = macho_dylib_names(&data).unwrap();
        assert_eq!(names[0].1, new);
        assert_eq!(names[1].1, "@rpath/libtbbbind_2_5.3.dylib");
    }

    #[test]
    fn rewrite_rejects_growth() {
        let mut data = thin_macho(&[(0xc, "/u/l.dylib")]);
        let err = rewrite_macho_dylib_names_in(
            &mut data,
            &[(
                "/u/l.dylib".to_string(),
                "@loader_path/much-longer.dylib".to_string(),
            )],
        )
        .unwrap_err();
        assert!(err.contains("longer"), "{err}");
    }

    #[test]
    fn parse_rejects_non_macho() {
        assert!(macho_dylib_names(b"definitely not a Mach-O file").is_err());
    }

    #[test]
    fn strip_deletes_command_truncates_and_shrinks_linkedit() {
        // layout: segment __LINKEDIT [0x100, 0x1c0), symtab content to 0x114,
        // signature [0x120, 0x130), file ends at 0x140 with 16 bytes after the
        // declared signature end (the pip wheel's off-by-16 layout).
        let data = [
            segment_command("__LINKEDIT", 0x100, 0xc0),
            symtab_command(0x100, 1, 0x110, 4),
            code_signature_command(0x120, 0x10),
        ];
        let mut out = macho_with(&data, 0x140);
        let cs_off = 32 + 72 + 24;
        assert!(strip_macho_signature_in(&mut out).unwrap());
        assert_eq!(out.len(), 0x120, "file must be truncated at the signature start");
        // The command region shrank by one 16-byte LC.
        let ncmds = u32::from_le_bytes(out[16..20].try_into().unwrap());
        let sizeofcmds = u32::from_le_bytes(out[20..24].try_into().unwrap());
        assert_eq!((ncmds, sizeofcmds), (2, 96));
        let linkedit_filesize = u64::from_le_bytes(out[32 + 48..32 + 56].try_into().unwrap());
        assert_eq!(linkedit_filesize, 0x20, "__LINKEDIT shrunk to the new end");
        assert_eq!(
            &out[cs_off..cs_off + 16],
            &[0u8; 16],
            "the deleted command's stale bytes must be zeroed"
        );
    }

    #[test]
    fn strip_deletes_non_trailing_signature() {
        // A trailing LC after the signature must shift left by 16 bytes.
        let tail = lc(0x1b, &[0x54u8; 16]);
        assert_eq!(tail.len(), 24);
        let data = [
            segment_command("__LINKEDIT", 0x100, 0x1c0),
            code_signature_command(0x200, 0x10),
            tail.clone(),
        ];
        let mut out = macho_with(&data, 0x220);
        assert!(strip_macho_signature_in(&mut out).unwrap());
        assert_eq!(out.len(), 0x200);
        let ncmds = u32::from_le_bytes(out[16..20].try_into().unwrap());
        assert_eq!(ncmds, 2);
        // cs at 104, tail was at 120: it moved to 104.
        assert_eq!(&out[104..128], &tail[..]);
        assert_eq!(&out[128..144], &[0u8; 16], "stale tail bytes zeroed");
        let linkedit_filesize = u64::from_le_bytes(out[32 + 48..32 + 56].try_into().unwrap());
        assert_eq!(linkedit_filesize, 0x100);
    }

    #[test]
    fn strip_refuses_foreign_cmdsize() {
        // A code-signature command that is not 16 bytes cannot be deleted by
        // shifting; refuse explicitly instead of corrupting the layout.
        let mut p = vec![0u8; 12];
        p[0..4].copy_from_slice(&0x200u32.to_le_bytes());
        p[4..8].copy_from_slice(&0x10u32.to_le_bytes());
        let data = [segment_command("__LINKEDIT", 0x100, 0x1c0), lc(0x1d, &p)];
        let mut out = macho_with(&data, 0x220);
        let err = strip_macho_signature_in(&mut out).unwrap_err();
        assert!(err.contains("cmdsize"), "{err}");
    }

    #[test]
    fn strip_without_signature_is_noop() {
        let mut data = thin_macho(&[(0xc, "/usr/local/lib/libhwloc.15.dylib")]);
        let len = data.len();
        assert!(!strip_macho_signature_in(&mut data).unwrap());
        assert_eq!(data.len(), len);
    }

    #[test]
    fn strip_refuses_when_data_reaches_into_signature() {
        // symtab: symoff 0x128 + 2 symbols * 16 = 0x148 > signature start 0x120
        let data = [
            segment_command("__LINKEDIT", 0x100, 0xc0),
            symtab_command(0x128, 2, 0x110, 4),
            code_signature_command(0x120, 0x10),
        ];
        let mut out = macho_with(&data, 0x130);
        let err = strip_macho_signature_in(&mut out).unwrap_err();
        assert!(err.contains("inside the signature"), "{err}");
    }

    #[test]
    fn strip_refuses_signature_past_eof() {
        let data = [
            segment_command("__LINKEDIT", 0x0, 0x130),
            code_signature_command(0x120, 0x20),
        ];
        let mut out = macho_with(&data, 0x130);
        let err = strip_macho_signature_in(&mut out).unwrap_err();
        assert!(err.contains("past end of file"), "{err}");
    }
}
