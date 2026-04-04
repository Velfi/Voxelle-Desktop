use crate::*;

/// Build a mesh from a decoded voxelle file, respecting the current rendering mode.
/// `is_stale` is polled during expensive smooth-mesh builds; returns `None` if cancelled.
fn build_mesh_for_mode<C: Fn() -> bool>(
    file: &voxelle::VoxelleFile,
    mode: RenderingMode,
    is_stale: C,
) -> Option<(greedy_mesh::MeshBuffers, greedy_mesh::MeshBounds)> {
    match mode {
        RenderingMode::Greedy | RenderingMode::Ray => {
            Some(greedy_mesh::build_greedy_mesh(&file.voxels, &file.objects))
        }
        RenderingMode::MarchingCubes => {
            let mesh = crate::smooth_mesh::build_marching_cubes_merged_cancellable(
                &file.voxels,
                |_, _, _| {},
                &is_stale,
            );
            if is_stale() {
                return None;
            }
            let bounds = greedy_mesh::mesh_bounds_from_voxels_world(&file.voxels, &file.objects)
                .or_else(|| greedy_mesh::mesh_bounds_from_voxels(&file.voxels))
                .unwrap_or(greedy_mesh::mesh_bounds_for_cube_side(file.grid_size));
            Some((mesh, bounds))
        }
        RenderingMode::DualContour => {
            let mesh = crate::smooth_mesh::build_dual_contour_merged_cancellable(
                &file.voxels,
                |_, _, _| {},
                &is_stale,
            );
            if is_stale() {
                return None;
            }
            let bounds = greedy_mesh::mesh_bounds_from_voxels_world(&file.voxels, &file.objects)
                .or_else(|| greedy_mesh::mesh_bounds_from_voxels(&file.voxels))
                .unwrap_or(greedy_mesh::mesh_bounds_for_cube_side(file.grid_size));
            Some((mesh, bounds))
        }
    }
}

// ── Scene object commands ───────────────────────────────────────────────────

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SceneObjectsPayload {
    objects: Vec<voxelle::SceneObject>,
    active_object_id: u32,
}

#[tauri::command]
pub(crate) fn get_scene_objects(state: State<'_, Arc<ViewerState>>) -> Result<SceneObjectsPayload, String> {
    let fg = state.current_file.lock();
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    Ok(SceneObjectsPayload {
        objects: file.objects.clone(),
        active_object_id: file.active_object_id,
    })
}

#[tauri::command]
pub(crate) fn set_active_object(state: State<'_, Arc<ViewerState>>, id: u32) -> Result<(), String> {
    let mut fg = state.current_file.lock();
    let Some(file) = fg.as_mut() else {
        return Err("no model loaded".into());
    };
    if !file.objects.iter().any(|o| o.id == id) {
        return Err("unknown object".into());
    }
    file.active_object_id = id;
    Ok(())
}

#[tauri::command]
pub(crate) fn set_object_visible(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    id: u32,
    visible: bool,
) -> Result<(), String> {
    {
        let mut fg = state.current_file.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(obj) = file.objects.iter_mut().find(|o| o.id == id) else {
            return Err("unknown object".into());
        };
        obj.visible = visible;
    }
    schedule_opaque_mesh_refresh(state.inner(), &app);
    Ok(())
}

#[tauri::command]
pub(crate) fn create_scene_object(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    name: String,
) -> Result<u32, String> {
    let next_id = {
        let mut fg = state.current_file.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let next_id = file
            .objects
            .iter()
            .map(|o| o.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let sort_order = file
            .objects
            .iter()
            .map(|o| o.sort_order)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        file.objects.push(voxelle::SceneObject {
            id: next_id,
            parent_id: None,
            name: if name.is_empty() {
                format!("Object {next_id}")
            } else {
                name
            },
            visible: true,
            sort_order,
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        });
        file.active_object_id = next_id;
        next_id
    };
    refresh_opaque_mesh(state.inner(), Some(&app))?;
    Ok(next_id)
}

// ── Selection menu sync ─────────────────────────────────────────────────────

/// Keeps the native **Match Material** menu checkbox in sync with app state.
#[tauri::command]
pub(crate) fn selection_menu_sync_match_material(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    checked: bool,
) -> Result<(), String> {
    *state.selection_match_material.lock() = checked;
    #[cfg(desktop)]
    {
        if let Some(menu) = app.try_state::<SelectionMenuState>() {
            menu.match_material
                .set_checked(checked)
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

// ── Performance report ──────────────────────────────────────────────────────

#[cfg(desktop)]
pub(crate) fn performance_report_text(state: &ViewerState) -> String {
    let fps = state.fps.lock().last_fps;
    let file_label = state.file_label.lock().clone();
    let (vw, vh, idx_count, vtx_buf_verts) = state
        .viewer
        .lock()
        .as_ref()
        .map(|viewer| {
            let (vw, vh) = viewer.viewport_size();
            (
                vw,
                vh,
                viewer.opaque_index_count(),
                viewer.opaque_vertex_buffer_vertices(),
            )
        })
        .unwrap_or((0, 0, 0, 0));
    let (voxel_n, grid_size) = state
        .current_file
        .lock()
        .as_ref()
        .map(|f| (f.voxels.len(), f.grid_size))
        .unwrap_or((0, 0));
    let unix_s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let edit_block = state
        .last_edit_perf
        .lock()
        .clone()
        .map(|e| {
            format!(
                "\nLast voxel edit (ms):\n\
                 \tapply_edit (ray + data): {:.2}\n\
                 \tprepare (scene bounds world + brick patch args): {:.2}\n\
                 \tviewer lock wait: {:.2}\n\
                 \tbrick (upload_scene_data): {:.2}\n\
                 \tmesh (total): {:.2}\n\
                 \t  spatial cache delta: {:.2}\n\
                 \t  spatial cache cold init: {:.2}\n\
                 \t  greedy (dirty chunks): {:.2}\n\
                 \t  greedy GPU (dirty chunks): {:.2}\n\
                 \t  greedy CPU (dirty chunks): {:.2}\n\
                 \t  chunk GPU buffers: {:.2}\n\
                 \t  full chunked rebuild: {:.2}\n\
                 \t  pipeline (rebuild_mesh_gpu_greedy): {:.2}\n\
                 \tpreview clear: {:.2}\n\
                 \tmesh route: {}\n\
                 \ttotal: {:.2}\n",
                e.apply_edit_ms,
                e.prepare_ms,
                e.viewer_lock_wait_ms,
                e.brick_ms,
                e.mesh_ms,
                e.mesh_voxel_map_ms,
                e.mesh_buckets_ms,
                e.mesh_greedy_ms,
                e.mesh_greedy_gpu_ms,
                e.mesh_greedy_cpu_ms,
                e.mesh_chunk_buffers_ms,
                e.mesh_full_chunked_rebuild_ms,
                e.mesh_pipeline_ms,
                e.preview_clear_ms,
                e.mesh_route,
                e.total_ms,
            )
        })
        .unwrap_or_else(|| "\nLast voxel edit (ms): (none yet this session)\n".to_string());
    format!(
        "Voxelle Desktop — performance snapshot\n\
         \n\
         Timestamp (UTC, Unix s): {unix_s}\n\
         Viewport FPS (last 1s avg): {fps}\n\
         Viewport size (physical px): {vw}×{vh}\n\
         Opaque mesh: index count = {idx_count}, vertex buffer slots ≈ {vtx_buf_verts}\n\
         Scene: voxel count = {voxel_n}, grid_size = {grid_size}\n\
         File label: {file_label}\n\
         Platform: {} / {}{edit_block}",
        std::env::consts::OS,
        std::env::consts::ARCH,
    )
}

#[cfg(desktop)]
pub(crate) fn copy_performance_data_to_clipboard(state: &Arc<ViewerState>) -> Result<(), String> {
    let text = performance_report_text(state);
    arboard::Clipboard::new()
        .map_err(|e| e.to_string())?
        .set_text(text)
        .map_err(|e| e.to_string())
}

// ── App update dialog ───────────────────────────────────────────────────────

/// Ok/Cancel prompt **without** parenting to the webview window. The JS `confirm` API always
/// attaches to the main window, which on macOS uses a sheet; keyboard/focus churn after the
/// app menu can activate the default OK before the user intends to.
#[tauri::command]
pub(crate) fn confirm_app_update_dialog(app: AppHandle, message: String, title: String) -> bool {
    app.dialog()
        .message(message)
        .title(title)
        .kind(MessageDialogKind::Info)
        .buttons(MessageDialogButtons::OkCancel)
        .blocking_show()
}

// ── Bundled assets ──────────────────────────────────────────────────────────

/// The start-screen logo, embedded at compile time.
static START_SCREEN_LOGO: &[u8] = include_bytes!("../../Logo.voxelle");

/// Bundled mascot models.  Key strings are used in the `mascot_load_embedded` command.
static MASCOT_SEAGULL: &[u8] = include_bytes!("../../mascots/Seagull.voxelle");

// ── Bundled avatar models (auto-generated from avatars/ at compile time) ─────
include!(concat!(env!("OUT_DIR"), "/avatars_generated.rs"));

/// Average of voxel center positions — used to pivot avatar rotation at the visual center
/// of mass rather than the bounding-box midpoint.
fn avatar_voxel_centroid(voxels: &[voxelle::Voxel]) -> glam::Vec3 {
    if voxels.is_empty() {
        return glam::Vec3::ZERO;
    }
    // Use voxel center positions (corner + 0.5) so the centroid matches the
    // visual center of mass rather than the corner-based grid origin.
    let sum: glam::Vec3 = voxels
        .iter()
        .map(|v| glam::Vec3::new(v.x as f32 + 0.5, v.y as f32 + 0.5, v.z as f32 + 0.5))
        .sum();
    sum / voxels.len() as f32
}

/// Build a single-voxel white glow mesh and register it under the key `""` (the default avatar).
pub(crate) fn init_default_avatar_mesh(viewer: &mut WgpuViewer) {
    use voxelle::{MaterialId, Voxel};
    let voxels = vec![Voxel {
        x: 0,
        y: 0,
        z: 0,
        color: 0xffffff,
        material: MaterialId::Glow,
        object_id: 0,
    }];
    let centroid = avatar_voxel_centroid(&voxels);
    let (mesh, _bounds) = greedy_mesh::build_greedy_mesh(&voxels, &[] as &[voxelle::SceneObject]);
    // scale=1.0 → exactly one voxel in world space.
    viewer.cache_avatar_mesh(String::new(), &mesh, centroid, 1.0);
}


/// Spawn a background thread that decodes an embedded avatar and uploads its mesh
/// to the shared avatar cache. No-op if the name is unknown.
pub(crate) fn spawn_load_avatar_mesh(state: Arc<ViewerState>, name: &str) {
    let Some(bytes) = embedded_avatar_bytes(name) else {
        return;
    };
    let name = name.to_owned();
    std::thread::Builder::new()
        .name("avatar-load".into())
        .spawn(move || {
            let file = match decode_payload(bytes) {
                Ok(f) => f,
                Err(_) => return,
            };
            let centroid = avatar_voxel_centroid(&file.voxels);
            let (mesh, bounds) = greedy_mesh::build_greedy_mesh(&file.voxels, &file.objects);
            let extent = (bounds.max - bounds.min).max_element().max(0.001);
            if let Some(viewer) = state.viewer.lock().as_mut() {
                viewer.cache_avatar_mesh(name, &mesh, centroid, 1.5 / extent);
            }
        })
        .ok();
}

/// Spawn a background thread that decodes a custom avatar from raw bytes and uploads
/// its mesh to the shared avatar cache.  Used for peer-supplied avatar files received
/// over collab.
pub(crate) fn spawn_load_avatar_from_bytes(state: Arc<ViewerState>, name: String, bytes: Vec<u8>) {
    std::thread::Builder::new()
        .name("avatar-load-collab".into())
        .spawn(move || {
            let file = match decode_payload(&bytes) {
                Ok(f) => f,
                Err(_) => return,
            };
            let centroid = avatar_voxel_centroid(&file.voxels);
            let (mesh, bounds) = greedy_mesh::build_greedy_mesh(&file.voxels, &file.objects);
            let extent = (bounds.max - bounds.min).max_element().max(0.001);
            if let Some(viewer) = state.viewer.lock().as_mut() {
                viewer.cache_avatar_mesh(name, &mesh, centroid, 1.5 / extent);
            }
        })
        .ok();
}

// ── Logo / Mascot commands ──────────────────────────────────────────────────

/// Loads bundled `Logo.voxelle` as a GPU overlay (does NOT use the project load pipeline).
#[tauri::command]
pub(crate) fn load_start_screen_logo(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
) -> Result<(), String> {
    let state = Arc::clone(&*state);
    let app_err = app.clone();
    let token = state.overlay_mesh_generation.fetch_add(1, Ordering::SeqCst) + 1;
    std::thread::Builder::new()
        .name("logo-load".into())
        .spawn(move || {
            let file = match decode_payload(START_SCREEN_LOGO) {
                Ok(f) => f,
                Err(e) => {
                    log::error!("logo load failed: {e}");
                    return;
                }
            };
            let mode = *state.rendering_mode.lock();
            let is_stale = {
                let st = Arc::clone(&state);
                move || st.overlay_mesh_generation.load(Ordering::Relaxed) != token
            };
            let Some((mesh, bounds)) = build_mesh_for_mode(&file, mode, is_stale) else {
                log::info!(target: "voxelle_load", "logo mesh build cancelled (stale)");
                return;
            };
            let app_main = app_err.clone();
            let state_up = Arc::clone(&state);
            let _ = app_err.run_on_main_thread(move || {
                if state_up.overlay_mesh_generation.load(Ordering::Relaxed) != token {
                    return;
                }
                let mut v = state_up.viewer.lock();
                if let Some(viewer) = v.as_mut() {
                    viewer.load_logo_mesh(&mesh, bounds);
                    if let Some(logo) = viewer.logo_overlay.as_mut() {
                        logo.visible = true;
                    }
                    drop(v);
                    let _ = app_main.emit("logo-loaded", ());
                    wake_viewport_loop(&app_main);
                }
            });
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Load a `.voxelle` file as a mascot model.
/// `path` should be a full filesystem path (the frontend resolves bundled assets
/// via Tauri's resource path API). `id` is a caller-chosen integer key (0–3).
#[tauri::command]
pub(crate) fn mascot_load(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    id: u32,
    path: String,
) -> Result<(), String> {
    let state = Arc::clone(&*state);
    let app_err = app.clone();
    let token = state.overlay_mesh_generation.load(Ordering::SeqCst);
    std::thread::Builder::new()
        .name("mascot-load".into())
        .spawn(move || {
            let bytes = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    let _ = app_err.emit("mascot-load-error", format!("id={id}: {e}"));
                    return;
                }
            };
            let file = match decode_payload(&bytes) {
                Ok(f) => f,
                Err(e) => {
                    let _ = app_err.emit("mascot-load-error", format!("id={id}: {e}"));
                    return;
                }
            };
            let mode = *state.rendering_mode.lock();
            let is_stale = {
                let st = Arc::clone(&state);
                move || st.overlay_mesh_generation.load(Ordering::Relaxed) != token
            };
            let Some((mesh, bounds)) = build_mesh_for_mode(&file, mode, is_stale) else {
                return;
            };
            let state_up = Arc::clone(&state);
            let _ = app_err.run_on_main_thread(move || {
                if state_up.overlay_mesh_generation.load(Ordering::Relaxed) != token {
                    return;
                }
                let mut v = state_up.viewer.lock();
                if let Some(viewer) = v.as_mut() {
                    viewer.load_mascot_mesh(id, &mesh, bounds);
                }
            });
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Set the viewport-relative screen rect for a mascot (physical pixels).
#[tauri::command]
pub(crate) fn mascot_set_screen_rect(
    state: State<'_, Arc<ViewerState>>,
    id: u32,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
) -> Result<(), String> {
    let mut v = state.viewer.lock();
    if let Some(viewer) = v.as_mut() {
        viewer.set_mascot_screen_rect(id, x, y, w, h);
    }
    Ok(())
}

/// Load a bundled (compile-time embedded) mascot by name.
/// Supported names: "seagull"
#[tauri::command]
pub(crate) fn mascot_load_embedded(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    id: u32,
    name: String,
) -> Result<(), String> {
    let bytes: &'static [u8] = match name.as_str() {
        "seagull" => MASCOT_SEAGULL,
        other => return Err(format!("unknown mascot: {other}")),
    };
    let state = Arc::clone(&*state);
    let app_err = app.clone();
    let token = state.overlay_mesh_generation.load(Ordering::SeqCst);
    std::thread::Builder::new()
        .name("mascot-load-embedded".into())
        .spawn(move || {
            let file = match decode_payload(bytes) {
                Ok(f) => f,
                Err(e) => {
                    let _ = app_err.emit("mascot-load-error", format!("id={id}: {e}"));
                    return;
                }
            };
            let mode = *state.rendering_mode.lock();
            let is_stale = {
                let st = Arc::clone(&state);
                move || st.overlay_mesh_generation.load(Ordering::Relaxed) != token
            };
            let Some((mesh, bounds)) = build_mesh_for_mode(&file, mode, is_stale) else {
                return;
            };
            let state_up = Arc::clone(&state);
            let app_main = app_err.clone();
            let _ = app_err.run_on_main_thread(move || {
                if state_up.overlay_mesh_generation.load(Ordering::Relaxed) != token {
                    return;
                }
                let mut v = state_up.viewer.lock();
                if let Some(viewer) = v.as_mut() {
                    viewer.load_mascot_mesh(id, &mesh, bounds);
                    drop(v);
                    let _ = app_main.emit("mascot-loaded", id);
                    wake_viewport_loop(&app_main);
                } else {
                    log::warn!("mascot_load_embedded: viewer was None when uploading mascot id={id}, mesh not uploaded");
                }
            });
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Show or hide a mascot.
#[tauri::command]
pub(crate) fn mascot_set_visible(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    id: u32,
    visible: bool,
) -> Result<(), String> {
    let mut v = state.viewer.lock();
    if let Some(viewer) = v.as_mut() {
        viewer.set_mascot_visible(id, visible);
    }
    drop(v);
    wake_viewport_loop(&app);
    Ok(())
}

// ── Logo overlay commands ───────────────────────────────────────────────────

/// Set the logo overlay camera angle by azimuth + elevation (degrees).
#[tauri::command]
pub(crate) fn logo_set_camera_angle(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    azimuth: f32,
    elevation: f32,
) -> Result<(), String> {
    let theta = azimuth.to_radians();
    let phi = (90.0 - elevation).to_radians().clamp(0.01, std::f32::consts::PI - 0.01);
    let mut v = state.viewer.lock();
    if let Some(viewer) = v.as_mut() {
        if let Some(logo) = viewer.logo_overlay.as_mut() {
            logo.theta = theta;
            logo.phi = phi;
            logo.rest_theta = theta;
            logo.rest_phi = phi;
        }
    }
    drop(v);
    wake_viewport_loop(&app);
    Ok(())
}

/// Set the logo overlay camera distance.
#[tauri::command]
pub(crate) fn logo_set_camera_dist(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    dist: f32,
) -> Result<(), String> {
    let mut v = state.viewer.lock();
    if let Some(viewer) = v.as_mut() {
        if let Some(logo) = viewer.logo_overlay.as_mut() {
            logo.cam_dist = dist;
        }
    }
    drop(v);
    wake_viewport_loop(&app);
    Ok(())
}

/// Set the logo overlay light direction by azimuth + elevation (degrees).
#[tauri::command]
pub(crate) fn logo_set_light_dir(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    azimuth: f32,
    elevation: f32,
) -> Result<(), String> {
    use crate::render::light_dir_from_azimuth_elevation_deg;
    let dir = light_dir_from_azimuth_elevation_deg(azimuth, elevation);
    let mut v = state.viewer.lock();
    if let Some(viewer) = v.as_mut() {
        if let Some(logo) = viewer.logo_overlay.as_mut() {
            logo.light_dir = dir.to_array();
            logo.light_azimuth_deg = azimuth;
            logo.light_elevation_deg = elevation;
        }
    }
    drop(v);
    wake_viewport_loop(&app);
    Ok(())
}

// ── Avatar commands ─────────────────────────────────────────────────────────

/// List the names of all bundled (compile-time embedded) avatars.
/// The empty string `""` is NOT included; it represents the default glow dot.
#[tauri::command]
pub(crate) fn avatar_list_embedded() -> Vec<String> {
    avatar_list_embedded_names()
}

/// Set this client's avatar choice, broadcast it to all collab peers, and kick off a
/// background mesh load so it appears on-screen immediately.
#[tauri::command]
pub(crate) fn set_local_avatar(
    state: State<'_, Arc<ViewerState>>,
    avatar_name: String,
) -> Result<(), String> {
    // If this is a custom (non-embedded) avatar, grab the raw bytes to send alongside
    // the AvatarChoice so peers can decode the mesh.
    let custom_bytes = if !avatar_name.is_empty() && embedded_avatar_bytes(&avatar_name).is_none() {
        state.local_avatar_data.lock().get(&avatar_name).cloned()
    } else {
        None
    };

    // Send to host (or, if we are the host, record locally and broadcast).
    {
        let c = state.collab.lock();
        if c.is_active() {
            let msg = serde_json::to_string(&collab::ClientToHost::AvatarChoice {
                avatar_name: avatar_name.clone(),
            })
            .map_err(|e| e.to_string())?;
            if let Some(tx) = &c.client_tx {
                let _ = tx.try_send(msg.clone());
            }
            // If there are custom bytes, also send AvatarData so peers can render it.
            if let (Some(bytes), Some(tx)) = (&custom_bytes, &c.client_tx) {
                if let Ok(data_msg) = serde_json::to_string(&collab::ClientToHost::AvatarData {
                    name: avatar_name.clone(),
                    bytes: bytes.clone(),
                }) {
                    let _ = tx.try_send(data_msg);
                }
            }
            // If we are the host, also record locally and broadcast to guests.
            if c.is_host() {
                let local_id = c.local_peer_id;
                drop(c);
                let mut c2 = state.collab.lock();
                c2.avatar_names.insert(local_id, avatar_name.clone());
                if let Some(bytes) = &custom_bytes {
                    c2.avatar_data.insert(avatar_name.clone(), bytes.clone());
                }
                let ev = serde_json::to_string(&collab::HostToClient::AvatarChoice {
                    peer_id: local_id,
                    avatar_name: avatar_name.clone(),
                })
                .unwrap_or_default();
                if let Some(tx) = &c2.host_broadcast {
                    let _ = tx.send(tokio_tungstenite::tungstenite::Message::Text(ev));
                }
                // Broadcast the raw bytes to guests as well.
                if let Some(bytes) = custom_bytes {
                    if let Ok(data_ev) = serde_json::to_string(&collab::HostToClient::AvatarData {
                        peer_id: local_id,
                        name: avatar_name.clone(),
                        bytes,
                    }) {
                        if let Some(tx) = &c2.host_broadcast {
                            let _ = tx.send(tokio_tungstenite::tungstenite::Message::Text(data_ev));
                        }
                    }
                }
            }
        }
    }
    // Ensure the mesh is cached so it renders on our own screen too.
    if !avatar_name.is_empty() {
        spawn_load_avatar_mesh(Arc::clone(&*state), &avatar_name);
    }
    Ok(())
}

/// Load a custom `.voxelle` file as a named avatar and cache its mesh.
#[tauri::command]
pub(crate) fn avatar_load_file(
    state: State<'_, Arc<ViewerState>>,
    path: String,
    name: String,
) -> Result<(), String> {
    let state = Arc::clone(&*state);
    std::thread::Builder::new()
        .name("avatar-load-file".into())
        .spawn(move || {
            let bytes = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    log::warn!("avatar_load_file: read {path}: {e}");
                    return;
                }
            };
            if bytes.len() > collab::MAX_AVATAR_FILE_BYTES {
                log::warn!("avatar_load_file: {path} exceeds MAX_AVATAR_FILE_BYTES, ignoring");
                return;
            }
            let file = match decode_payload(&bytes) {
                Ok(f) => f,
                Err(e) => {
                    log::warn!("avatar_load_file: decode {path}: {e}");
                    return;
                }
            };
            // Store raw bytes so they can be sent to collab peers.
            state.local_avatar_data.lock().insert(name.clone(), bytes);
            let centroid = avatar_voxel_centroid(&file.voxels);
            let (mesh, bounds) = greedy_mesh::build_greedy_mesh(&file.voxels, &file.objects);
            let extent = (bounds.max - bounds.min).max_element().max(0.001);
            if let Some(viewer) = state.viewer.lock().as_mut() {
                viewer.cache_avatar_mesh(name, &mesh, centroid, 1.5 / extent);
            }
        })
        .ok();
    Ok(())
}

/// Return the names of all `.voxelle` files found in the user's avatars folder
/// (`{app_data}/avatars/`).  Each valid file is decoded and its mesh cached so it
/// can be selected immediately.  Files that are too large or fail to decode are
/// silently skipped (a warning is logged).
#[tauri::command]
pub(crate) fn avatar_list_user(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
) -> Vec<String> {
    let avatars_dir = match app.path().app_data_dir() {
        Ok(mut d) => { d.push("avatars"); d }
        Err(_) => return vec![],
    };
    let _ = std::fs::create_dir_all(&avatars_dir);
    let entries = match std::fs::read_dir(&avatars_dir) {
        Ok(e) => e,
        Err(_) => return vec![],
    };
    let mut names = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("voxelle") {
            continue;
        }
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => { log::warn!("avatar_list_user: read {:?}: {e}", path); continue; }
        };
        if bytes.len() > collab::MAX_AVATAR_FILE_BYTES {
            log::warn!("avatar_list_user: {:?} exceeds MAX_AVATAR_FILE_BYTES, skipping", path);
            continue;
        }
        let file = match decode_payload(&bytes) {
            Ok(f) => f,
            Err(e) => { log::warn!("avatar_list_user: decode {:?}: {e}", path); continue; }
        };
        // Cache raw bytes for collab peer distribution.
        state.local_avatar_data.lock().insert(stem.clone(), bytes);
        // Build and upload mesh.
        let centroid = avatar_voxel_centroid(&file.voxels);
        let (mesh, bounds) = greedy_mesh::build_greedy_mesh(&file.voxels, &file.objects);
        let extent = (bounds.max - bounds.min).max_element().max(0.001);
        if let Some(viewer) = state.viewer.lock().as_mut() {
            viewer.cache_avatar_mesh(stem.clone(), &mesh, centroid, 1.5 / extent);
        }
        names.push(stem);
    }
    names.sort();
    names
}

/// Open (and create if needed) the user avatars folder in the OS file manager.
/// Drop `.voxelle` files here to make them appear in the Avatar picker.
#[tauri::command]
pub(crate) fn avatar_open_user_folder(app: AppHandle) -> Result<(), String> {
    let avatars_dir = app.path().app_data_dir().map_err(|e| e.to_string()).map(|mut d| { d.push("avatars"); d })?;
    std::fs::create_dir_all(&avatars_dir).map_err(|e| e.to_string())?;
    #[cfg(target_os = "macos")]
    std::process::Command::new("open").arg(&avatars_dir).spawn().map_err(|e| e.to_string())?;
    #[cfg(target_os = "windows")]
    std::process::Command::new("explorer").arg(&avatars_dir).spawn().map_err(|e| e.to_string())?;
    #[cfg(target_os = "linux")]
    std::process::Command::new("xdg-open").arg(&avatars_dir).spawn().map_err(|e| e.to_string())?;
    Ok(())
}

// ── Speech bubble commands ──────────────────────────────────────────────────

/// Show (or replace) a speech bubble.
/// `rx`, `ry`, `rw`, `rh` — bubble rect in viewport-relative physical pixels.
/// `tx`, `ty` — tail tip in viewport-relative physical pixels (anchor point toward subject).
/// `pages` — ordered list of text strings; click advances through them.
#[tauri::command]
pub(crate) fn speech_bubble_show(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    id: u32,
    pages: Vec<String>,
    rx: f32,
    ry: f32,
    rw: f32,
    rh: f32,
    tx: f32,
    ty: f32,
) -> Result<f32, String> {
    let mut v = state.viewer.lock();
    let computed_rh = if let Some(viewer) = v.as_mut() {
        viewer.show_speech_bubble(id, pages, [rx, ry, rw, rh], [tx, ty])
    } else {
        rh
    };
    drop(v);
    wake_viewport_loop(&app);
    Ok(computed_rh)
}

/// Register a click on bubble `id`.
/// Advances to the next page, or begins a shake-then-dismiss sequence on the last page.
/// Emits `"speech-bubble-dismissed"` with `id` when the bubble finally closes.
#[tauri::command]
pub(crate) fn speech_bubble_click(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    id: u32,
) -> Result<(), String> {
    let mut v = state.viewer.lock();
    let changed = if let Some(viewer) = v.as_mut() {
        viewer.click_speech_bubble(id)
    } else {
        false
    };
    drop(v);
    if changed {
        wake_viewport_loop(&app);
    }
    Ok(())
}

/// Immediately dismiss a speech bubble without the shake animation.
/// Emits `"speech-bubble-dismissed"` with `id`.
#[tauri::command]
pub(crate) fn speech_bubble_dismiss(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    id: u32,
) -> Result<(), String> {
    let mut v = state.viewer.lock();
    if let Some(viewer) = v.as_mut() {
        viewer.dismiss_speech_bubble(id);
    }
    drop(v);
    let _ = app.emit("speech-bubble-dismissed", id);
    wake_viewport_loop(&app);
    Ok(())
}

/// Move an existing bubble to a new screen rect + tail tip without resetting its page or state.
/// Used to keep bubbles anchored to their subject after a window resize.
#[tauri::command]
pub(crate) fn speech_bubble_reposition(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    id: u32,
    rx: f32,
    ry: f32,
    rw: f32,
    rh: f32,
    tx: f32,
    ty: f32,
) -> Result<(), String> {
    let mut v = state.viewer.lock();
    if let Some(viewer) = v.as_mut() {
        viewer.reposition_speech_bubble(id, [rx, ry, rw, rh], [tx, ty]);
    }
    drop(v);
    wake_viewport_loop(&app);
    Ok(())
}
