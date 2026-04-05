use crate::*;

// ── Write helper ────────────────────────────────────────────────────────────

pub(crate) fn write_voxelle_file_to_path(
    progress: Option<&AppHandle>,
    state: &ViewerState,
    path: &std::path::Path,
) -> Result<(), String> {
    let wp = match progress {
        Some(app) => {
            let mut g = WorkProgressGuard::new(app);
            g.arm();
            emit_work_progress(app, 0.1, "Saving…");
            Some(g)
        }
        None => None,
    };
    let file = {
        let g = state.current_file.lock();
        g.as_ref()
            .ok_or_else(|| "no model loaded".to_string())?
            .clone()
    };
    if let Some(app) = progress {
        emit_work_progress(app, 0.35, "Saving — encoding…");
    }
    let bytes = encode_payload_v5(&file).map_err(|e| e.to_string())?;
    if let Some(app) = progress {
        emit_work_progress(app, 0.7, "Saving — writing file…");
    }
    std::fs::write(path, bytes).map_err(|e| e.to_string())?;
    drop(wp);
    Ok(())
}

// ── Session state persistence ───────────────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize)]
struct LastSessionFile {
    last_document_path: String,
}

pub(crate) fn session_state_path(app: &AppHandle) -> Result<PathBuf, String> {
    let mut p = app.path().app_data_dir().map_err(|e| e.to_string())?;
    p.push("last_session.json");
    Ok(p)
}

pub(crate) fn persist_last_document_path(app: &AppHandle, document_path: &str) {
    if !document_path.ends_with(".voxelle") {
        return;
    }
    let Ok(path) = session_state_path(app) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let data = LastSessionFile {
        last_document_path: document_path.to_string(),
    };
    if let Ok(s) = serde_json::to_string_pretty(&data) {
        let _ = std::fs::write(path, s);
    }
}

fn read_last_document_path(app: &AppHandle) -> Option<String> {
    let path = session_state_path(app).ok()?;
    let bytes = std::fs::read(path).ok()?;
    let f: LastSessionFile = serde_json::from_slice(&bytes).ok()?;
    Some(f.last_document_path)
}

// ── Recent files list ───────────────────────────────────────────────────────

const MAX_RECENT_FILES: usize = 10;

#[derive(serde::Serialize, serde::Deserialize)]
struct RecentFiles {
    paths: Vec<String>,
}

fn recent_files_path(app: &AppHandle) -> Result<PathBuf, String> {
    let mut p = app.path().app_data_dir().map_err(|e| e.to_string())?;
    p.push("recent_files.json");
    Ok(p)
}

pub(crate) fn read_recent_files(app: &AppHandle) -> Vec<String> {
    let Ok(path) = recent_files_path(app) else {
        return Vec::new();
    };
    let Ok(bytes) = std::fs::read(&path) else {
        return Vec::new();
    };
    serde_json::from_slice::<RecentFiles>(&bytes)
        .map(|r| r.paths)
        .unwrap_or_default()
}

pub(crate) fn persist_recent_file(app: &AppHandle, document_path: &str) {
    if !document_path.ends_with(".voxelle") {
        return;
    }
    let Ok(path) = recent_files_path(app) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut paths = read_recent_files(app);
    // Remove if already present so we can move it to the front.
    paths.retain(|p| p != document_path);
    paths.insert(0, document_path.to_string());
    paths.truncate(MAX_RECENT_FILES);
    let data = RecentFiles { paths };
    if let Ok(s) = serde_json::to_string_pretty(&data) {
        let _ = std::fs::write(path, s);
    }
}

pub(crate) fn clear_recent_files(app: &AppHandle) {
    let Ok(path) = recent_files_path(app) else {
        return;
    };
    let _ = std::fs::remove_file(path);
}

// ── Autosave helpers ────────────────────────────────────────────────────────

pub(crate) fn autosave_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let mut d = app.path().app_data_dir().map_err(|e| e.to_string())?;
    d.push("autosaves");
    std::fs::create_dir_all(&d).map_err(|e| e.to_string())?;
    Ok(d)
}

fn stable_path_key(path: &Path) -> String {
    let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let key = canon.to_string_lossy();
    format!("{:016x}", crc32fast::hash(key.as_bytes()))
}

/// Legacy single backup before per-slot rotation (`{hash}.voxelle`).
fn legacy_autosave_path(app: &AppHandle, document_path: &Path) -> Result<PathBuf, String> {
    let h = stable_path_key(document_path);
    let mut p = autosave_dir(app)?;
    p.push(format!("{h}.voxelle"));
    Ok(p)
}

fn collect_autosave_paths_for_document(
    app: &AppHandle,
    state: &ViewerState,
    document_path: &Path,
) -> Result<Vec<PathBuf>, String> {
    let keep = *state.autosave_keep_count.lock();
    let keep = keep.max(1);
    let mut out = Vec::new();
    let leg = legacy_autosave_path(app, document_path)?;
    if leg.exists() {
        out.push(leg);
    }
    let h = stable_path_key(document_path);
    let dir = autosave_dir(app)?;
    for i in 0..keep {
        let p = dir.join(format!("{h}.{i}.voxelle"));
        if p.exists() {
            out.push(p);
        }
    }
    Ok(out)
}

fn newest_autosave_path(
    app: &AppHandle,
    state: &ViewerState,
    document_path: &Path,
) -> Option<PathBuf> {
    let paths = collect_autosave_paths_for_document(app, state, document_path).ok()?;
    let epoch = std::time::UNIX_EPOCH;
    paths
        .into_iter()
        .max_by_key(|p| file_mtime(p).unwrap_or(epoch))
}

pub(crate) fn next_rotating_autosave_path(
    app: &AppHandle,
    state: &ViewerState,
    document_path: &Path,
) -> Result<PathBuf, String> {
    let h = stable_path_key(document_path);
    let keep = *state.autosave_keep_count.lock();
    let k = (keep.max(1)) as u64;
    let idx = {
        let mut map = state.autosave_slot.lock();
        let n = map.entry(h.clone()).or_insert(0);
        let slot = (*n % k) as u32;
        *n = n.wrapping_add(1);
        slot
    };
    let mut dir = autosave_dir(app)?;
    dir.push(format!("{h}.{idx}.voxelle"));
    Ok(dir)
}

fn unsaved_autosave_anchor_path(app: &AppHandle) -> Result<PathBuf, String> {
    let mut p = app.path().app_data_dir().map_err(|e| e.to_string())?;
    p.push("unsaved_autosave_anchor.voxelle");
    Ok(p)
}

/// Logical document path for autosave keys and rotation. Saved projects use the real file path;
/// unsaved labels (e.g. `New project (…)`) use a stable app-local anchor so backups work before
/// "Save As…".
pub(crate) fn autosave_document_path_for_label(
    app: &AppHandle,
    label: &str,
) -> Result<PathBuf, String> {
    if label.ends_with(".voxelle") {
        Ok(PathBuf::from(label))
    } else {
        unsaved_autosave_anchor_path(app)
    }
}

/// `file_label` after restoring from the unsaved-work autosave bucket (not a real on-disk project path).
const ONGOING_UNSAVED_PROJECT_LABEL: &str = "An unsaved project";

pub(crate) fn try_initial_autosave_after_new_project(
    app: &AppHandle,
    state: &Arc<ViewerState>,
    label: &str,
) {
    let enabled = *state.autosave_enabled.lock();
    let interval = *state.autosave_interval_secs.lock();
    if !enabled || interval == 0 {
        return;
    }
    let (collab_on, is_host) = {
        let c = state.collab.lock();
        (c.is_active(), c.is_host())
    };
    if collab_on && !is_host {
        return;
    }
    if !state.active_project.load(Ordering::Relaxed) {
        return;
    }
    let Ok(doc) = autosave_document_path_for_label(app, label) else {
        return;
    };
    let Ok(dest) = next_rotating_autosave_path(app, Arc::as_ref(state), &doc) else {
        return;
    };
    if write_voxelle_file_to_path(None, Arc::as_ref(state), &dest).is_ok() {
        *state.last_autosave.lock() = Some(Instant::now());
    }
}

fn file_mtime(path: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).ok()?.modified().ok()
}

// ── Tauri commands: session info ────────────────────────────────────────────

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LastSessionInfo {
    last_document_path: Option<String>,
    document_basename: Option<String>,
    autosave_path: Option<String>,
    document_exists: bool,
    autosave_exists: bool,
    autosave_newer_than_document: bool,
}

#[tauri::command]
pub(crate) fn get_last_session_info(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
) -> Result<LastSessionInfo, String> {
    let doc_str_opt = read_last_document_path(&app);
    let anchor = unsaved_autosave_anchor_path(&app)?;
    let st = state.inner().as_ref();

    let doc_newest = doc_str_opt
        .as_ref()
        .and_then(|s| newest_autosave_path(&app, st, Path::new(s.as_str())));
    let anchor_newest = newest_autosave_path(&app, st, &anchor);

    let use_anchor_recovery = match (&doc_newest, &anchor_newest) {
        (Some(d_path), Some(a_path)) => match (file_mtime(d_path), file_mtime(a_path)) {
            (Some(dm), Some(am)) => am > dm,
            (None, Some(_)) => true,
            _ => false,
        },
        (None, Some(_)) => true,
        _ => false,
    };

    if use_anchor_recovery {
        let Some(ap) = anchor_newest else {
            return Ok(LastSessionInfo {
                last_document_path: None,
                document_basename: None,
                autosave_path: None,
                document_exists: false,
                autosave_exists: false,
                autosave_newer_than_document: false,
            });
        };
        let aex = ap.exists();
        return Ok(LastSessionInfo {
            last_document_path: Some(ONGOING_UNSAVED_PROJECT_LABEL.to_string()),
            document_basename: Some(ONGOING_UNSAVED_PROJECT_LABEL.to_string()),
            autosave_path: Some(ap.to_string_lossy().into_owned()),
            document_exists: false,
            autosave_exists: aex,
            autosave_newer_than_document: true,
        });
    }

    let Some(doc_str) = doc_str_opt else {
        return Ok(LastSessionInfo {
            last_document_path: None,
            document_basename: None,
            autosave_path: None,
            document_exists: false,
            autosave_exists: false,
            autosave_newer_than_document: false,
        });
    };

    let doc_path = PathBuf::from(&doc_str);
    let basename = doc_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned());
    let document_exists = doc_path.exists();
    let (autosave_str, autosave_exists, newer) = match doc_newest {
        Some(ap) => {
            let aex = ap.exists();
            let s = ap.to_string_lossy().into_owned();
            let newer = match (document_exists, aex) {
                (true, true) => match (file_mtime(&doc_path), file_mtime(&ap)) {
                    (Some(dm), Some(am)) => am > dm,
                    (None, Some(_)) => true,
                    _ => false,
                },
                (false, true) => true,
                _ => false,
            };
            (Some(s), aex, newer)
        }
        None => (None, false, false),
    };
    Ok(LastSessionInfo {
        last_document_path: Some(doc_str),
        document_basename: basename,
        autosave_path: autosave_str,
        document_exists,
        autosave_exists,
        autosave_newer_than_document: newer,
    })
}

// ── Tauri commands: load / recovery ─────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct LoadVoxelleRecoveryArgs {
    document_path: String,
    autosave_path: String,
}

#[tauri::command]
pub(crate) fn load_voxelle_recovery(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: LoadVoxelleRecoveryArgs,
) -> Result<(), String> {
    state
        .start_screen_logo_transparent
        .store(false, Ordering::Release);
    let read_from = PathBuf::from(&args.autosave_path);
    if !read_from.is_file() {
        return Err("Autosave file not found.".into());
    }
    *state.file_label.lock() = args.document_path.clone();
    let _ = app.emit("voxelle-load-start", args.document_path.clone());
    spawn_decode_and_mesh_with_label(Arc::clone(&*state), app, read_from, args.document_path);
    Ok(())
}

#[tauri::command]
pub(crate) fn load_voxelle_path(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    path: String,
) -> Result<(), String> {
    state
        .start_screen_logo_transparent
        .store(false, Ordering::Release);
    let p = std::path::PathBuf::from(&path);
    *state.file_label.lock() = path.clone();
    let _ = app.emit("voxelle-load-start", path.clone());
    spawn_decode_and_mesh(Arc::clone(&*state), app, p);
    Ok(())
}

// ── Tauri commands: save ────────────────────────────────────────────────────

#[tauri::command]
pub(crate) fn save_voxelle(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
) -> Result<(), String> {
    let label = state.file_label.lock();
    if label.starts_with("New project") || !label.ends_with(".voxelle") {
        return Err("Use \u{201c}Save As\u{2026}\u{201d} for new or unsaved projects.".into());
    }
    let s = label.clone();
    drop(label);
    write_voxelle_file_to_path(Some(&app), &state, Path::new(s.as_str()))?;
    persist_last_document_path(&app, s.as_str());
    persist_recent_file(&app, s.as_str());
    #[cfg(desktop)]
    if let Some(rm) = app.try_state::<RecentMenuState>() {
        rebuild_recent_submenu(&app, &rm.submenu);
    }
    Ok(())
}

#[tauri::command]
pub(crate) fn save_voxelle_as(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
) -> Result<(), String> {
    let state_c = Arc::clone(&*state);
    let app_c = app.clone();
    let mut builder = app
        .dialog()
        .file()
        .add_filter("Voxelle", &["voxelle"])
        .set_file_name("untitled.voxelle");
    if let Some(window) = app.get_webview_window("main") {
        builder = builder.set_parent(&window);
    }
    builder.save_file(move |file_path| {
        let Some(file_path) = file_path else {
            return;
        };
        let Ok(path) = file_path.into_path() else {
            let _ = app_c.emit("voxelle-load-error", "could not resolve save path");
            return;
        };
        if let Err(e) = write_voxelle_file_to_path(Some(&app_c), &state_c, &path) {
            let _ = app_c.emit("voxelle-load-error", e);
            return;
        }
        let s = path.to_string_lossy().to_string();
        *state_c.file_label.lock() = s.clone();
        persist_last_document_path(&app_c, &s);
        persist_recent_file(&app_c, &s);
        #[cfg(desktop)]
        if let Some(rm) = app_c.try_state::<RecentMenuState>() {
            rebuild_recent_submenu(&app_c, &rm.submenu);
        }
        emit_voxelle_loaded(&app_c, s, &state_c);
    });
    Ok(())
}

// ── Tauri commands: export ──────────────────────────────────────────────────

fn mesh_for_export(state: &Arc<ViewerState>) -> Result<greedy_mesh::MeshBuffers, String> {
    let fg = state.current_file.lock();
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let rm = *state.rendering_mode.lock();
    let mesh = match rm {
        RenderingMode::Greedy | RenderingMode::Ray => {
            greedy_mesh::build_greedy_mesh(&file.voxels, &file.objects).0
        }
        RenderingMode::MarchingCubes => {
            crate::smooth_mesh::build_marching_cubes_merged(&file.voxels)
        }
        RenderingMode::DualContour => crate::smooth_mesh::build_dual_contour_merged(&file.voxels),
    };
    Ok(mesh)
}

#[tauri::command]
pub(crate) fn export_mesh_glb(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
) -> Result<(), String> {
    let state_c = Arc::clone(&*state);
    let app_c = app.clone();
    let mut builder = app
        .dialog()
        .file()
        .add_filter("glTF Binary", &["glb"])
        .set_file_name("export.glb");
    if let Some(window) = app.get_webview_window("main") {
        builder = builder.set_parent(&window);
    }
    builder.save_file(move |file_path| {
        let Some(file_path) = file_path else {
            return;
        };
        let Ok(path) = file_path.into_path() else {
            let _ = app_c.emit("voxelle-load-error", "could not resolve export path");
            return;
        };
        let mesh = match mesh_for_export(&state_c) {
            Ok(m) => m,
            Err(e) => {
                let _ = app_c.emit("voxelle-load-error", e);
                return;
            }
        };
        let glb = match export_glb::mesh_buffers_to_glb(&mesh) {
            Ok(b) => b,
            Err(e) => {
                let _ = app_c.emit("voxelle-load-error", e);
                return;
            }
        };
        if let Err(e) = std::fs::write(&path, glb) {
            let _ = app_c.emit("voxelle-load-error", e.to_string());
        }
    });
    Ok(())
}

// ── Tauri commands: open dialog ─────────────────────────────────────────────

/// Non-blocking `pick_file` — `blocking_pick_file` stalls the wry event loop and freezes the
/// window (spinner) on macOS while the sheet is open.
pub(crate) fn open_voxelle_file_dialog(app: AppHandle, state: Arc<ViewerState>) {
    let state = Arc::clone(&state);
    let is_guest = state.collab.lock().is_client();

    if is_guest {
        // Warn the guest that opening a file will disconnect them from the session.
        let app_confirm = app.clone();
        let state_confirm = Arc::clone(&state);
        app.dialog()
            .message("Opening a file will disconnect you from the current collaboration session.")
            .title("Leave session?")
            .kind(MessageDialogKind::Warning)
            .buttons(MessageDialogButtons::OkCancelCustom(
                "Open File".into(),
                "Cancel".into(),
            ))
            .show(move |confirmed| {
                if confirmed {
                    leave_collab_guest(&state_confirm, &app_confirm);
                    show_file_picker(app_confirm, state_confirm);
                }
            });
    } else {
        show_file_picker(app, state);
    }
}

/// Disconnect a guest from the current collab session.
fn leave_collab_guest(state: &Arc<ViewerState>, app: &AppHandle) {
    let mut c = state.collab.lock();
    if c.is_client() {
        if let Some(tx) = &c.client_tx {
            let msg = serde_json::to_string(&collab::ClientToHost::Leave).unwrap();
            let _ = tx.try_send(collab::ClientOutgoing::Text(msg));
        }
        c.leave();
        drop(c);
        *state.ping_flash.lock() = None;
        let _ = app.emit("collab-ended", "You left the collaboration session.");
    }
}

fn show_file_picker(app: AppHandle, state: Arc<ViewerState>) {
    let app_cb = app.clone();
    let mut builder = app.dialog().file().add_filter("Voxelle", &["voxelle"]);
    if let Some(window) = app.get_webview_window("main") {
        builder = builder.set_parent(&window);
    }
    builder.pick_file(move |file_path| {
        let Some(file_path) = file_path else {
            return;
        };
        let Ok(path) = file_path.into_path() else {
            let _ = app_cb.emit("voxelle-load-error", "could not resolve file path");
            return;
        };
        let label = path.to_string_lossy().to_string();
        *state.file_label.lock() = label.clone();
        let _ = app_cb.emit("voxelle-load-start", label);
        spawn_decode_and_mesh(state, app_cb, path);
    });
}

#[tauri::command]
pub(crate) fn open_voxelle_dialog(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
) -> Result<(), String> {
    open_voxelle_file_dialog(app, Arc::clone(&*state));
    Ok(())
}

// ── Close project (return to start screen) ────────────────────────────────

/// Performs the unload + emit so the frontend returns to the start screen.
fn finish_close_project(state: &Arc<ViewerState>, app: &AppHandle) {
    *state.file_label.lock() = String::new();
    if let Err(e) = run_unload_on_main_thread(state, app) {
        log::error!(target: "voxelle_load", "close_project unload failed: {e}");
    }
    let _ = app.emit("voxelle-project-closed", ());
}

/// Show a "Save your changes?" dialog, then unload and return to the start screen.
/// Called from the File → Close Project menu item.
pub(crate) fn close_project_dialog(app: AppHandle, state: Arc<ViewerState>) {
    use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

    // Nothing to close — already on the start screen.
    if !state
        .active_project
        .load(std::sync::atomic::Ordering::Relaxed)
    {
        return;
    }

    let label = state.file_label.lock().clone();
    let is_named = !label.starts_with("New project") && label.ends_with(".voxelle");

    let app_d = app.clone();
    let state_d = Arc::clone(&state);
    let mut builder = app
        .dialog()
        .message("Do you want to save changes before closing?")
        .title("Close Project")
        .kind(MessageDialogKind::Warning)
        .buttons(MessageDialogButtons::OkCancelCustom(
            "Save".into(),
            "Don\u{2019}t Save".into(),
        ));
    if let Some(window) = app.get_webview_window("main") {
        builder = builder.parent(&window);
    }
    builder.show(move |save| {
        if save {
            // User chose "Save".
            if is_named {
                // Named file — save in place, then close.
                let path = std::path::Path::new(label.as_str());
                if let Err(e) = write_voxelle_file_to_path(Some(&app_d), &state_d, path) {
                    let _ = app_d.emit("voxelle-load-error", e);
                    return;
                }
                persist_last_document_path(&app_d, &label);
                finish_close_project(&state_d, &app_d);
            } else {
                // Unsaved / new project — show Save As dialog first.
                let app_sa = app_d.clone();
                let state_sa = Arc::clone(&state_d);
                let mut sa_builder = app_d
                    .dialog()
                    .file()
                    .add_filter("Voxelle", &["voxelle"])
                    .set_file_name("untitled.voxelle");
                if let Some(window) = app_d.get_webview_window("main") {
                    sa_builder = sa_builder.set_parent(&window);
                }
                sa_builder.save_file(move |file_path| {
                    let Some(file_path) = file_path else {
                        // User cancelled the Save As dialog — abort close.
                        return;
                    };
                    let Ok(path) = file_path.into_path() else {
                        let _ = app_sa.emit("voxelle-load-error", "could not resolve save path");
                        return;
                    };
                    if let Err(e) = write_voxelle_file_to_path(Some(&app_sa), &state_sa, &path) {
                        let _ = app_sa.emit("voxelle-load-error", e);
                        return;
                    }
                    let s = path.to_string_lossy().to_string();
                    persist_last_document_path(&app_sa, &s);
                    persist_recent_file(&app_sa, &s);
                    #[cfg(desktop)]
                    if let Some(rm) = app_sa.try_state::<RecentMenuState>() {
                        rebuild_recent_submenu(&app_sa, &rm.submenu);
                    }
                    finish_close_project(&state_sa, &app_sa);
                });
            }
        } else {
            // User chose "Don't Save" — close without saving.
            finish_close_project(&state_d, &app_d);
        }
    });
}

// ── Tauri commands: new project ─────────────────────────────────────────────

pub(crate) const MAX_GRID_SIZE: u32 = 256;

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct NewProjectArgs {
    grid_size: u32,
    shape: StartShape,
}

#[tauri::command]
pub(crate) fn create_new_project(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: NewProjectArgs,
) -> Result<(), String> {
    let grid_size = args.grid_size.clamp(1, MAX_GRID_SIZE);
    let shape_l = start_shape_label(args.shape);
    let label = format!("New project ({grid_size}³, {shape_l})");
    *state.file_label.lock() = label.clone();
    let _ = app.emit("voxelle-load-start", label);
    spawn_new_project(Arc::clone(&*state), app, grid_size, args.shape);
    Ok(())
}

// ── Tauri commands: autosave settings ───────────────────────────────────────

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AutosaveSettings {
    enabled: bool,
    interval_secs: u64,
    keep_count: u32,
}

#[tauri::command]
pub(crate) fn get_autosave_settings(
    state: State<'_, Arc<ViewerState>>,
) -> Result<AutosaveSettings, String> {
    Ok(AutosaveSettings {
        enabled: *state.autosave_enabled.lock(),
        interval_secs: *state.autosave_interval_secs.lock(),
        keep_count: *state.autosave_keep_count.lock(),
    })
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct AutosaveSettingsArgs {
    enabled: bool,
    interval_secs: u64,
    keep_count: u32,
}

#[tauri::command]
pub(crate) fn set_autosave_settings(
    state: State<'_, Arc<ViewerState>>,
    args: AutosaveSettingsArgs,
) -> Result<(), String> {
    *state.autosave_enabled.lock() = args.enabled;
    *state.autosave_interval_secs.lock() = args.interval_secs;
    let k = args.keep_count.clamp(1, 64);
    *state.autosave_keep_count.lock() = k;
    Ok(())
}

pub(crate) fn clear_autosaves_and_session(app: &AppHandle) -> Result<(), String> {
    let dir = autosave_dir(app)?;
    let mut deleted = 0u32;
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("voxelle")
                && std::fs::remove_file(&path).is_ok()
            {
                deleted += 1;
            }
        }
    }
    if let Ok(session_path) = session_state_path(app) {
        let _ = std::fs::remove_file(&session_path);
    }
    log::info!(
        "debug_clear_autosaves: deleted {deleted} autosave file(s) and cleared last_session.json"
    );
    Ok(())
}

#[tauri::command]
pub(crate) fn debug_clear_autosaves(app: AppHandle) -> Result<(), String> {
    clear_autosaves_and_session(&app)
}
