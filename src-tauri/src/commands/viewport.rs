use crate::*;

// ── Viewport size structs ──────────────────────────────────────────────────────

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ViewportPixelSize {
    pub width: u32,
    pub height: u32,
    pub surface_width: u32,
    pub surface_height: u32,
}

/// Authoritative swapchain size in physical pixels (from the viewer; matches `frame.texture` after render).
#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SurfacePixelSize {
    pub width: u32,
    pub height: u32,
}

// ── Viewport commands ──────────────────────────────────────────────────────────

/// Last known `.viewport` size and swapchain size in physical pixels (matches projection / picking / blit).
#[tauri::command]
pub(crate) fn get_viewport_pixel_size(
    state: State<'_, Arc<ViewerState>>,
) -> Result<ViewportPixelSize, String> {
    let v = state.viewer.lock();
    let Some(viewer) = v.as_ref() else {
        return Err("viewer not ready".into());
    };
    let (w, h) = viewer.viewport_size();
    let (sw, sh) = viewer.surface_pixel_size();
    Ok(ViewportPixelSize {
        width: w,
        height: h,
        surface_width: sw,
        surface_height: sh,
    })
}

#[tauri::command]
pub(crate) fn get_surface_pixel_size(state: State<'_, Arc<ViewerState>>) -> Result<SurfacePixelSize, String> {
    let v = state.viewer.lock();
    let Some(viewer) = v.as_ref() else {
        return Err("viewer not ready".into());
    };
    let (sw, sh) = viewer.surface_pixel_size();
    Ok(SurfacePixelSize {
        width: sw,
        height: sh,
    })
}

#[tauri::command]
pub(crate) fn viewer_resize(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    surface_width: u32,
    surface_height: u32,
    viewport_x: u32,
    viewport_y: u32,
    viewport_width: u32,
    viewport_height: u32,
) -> Result<(), String> {
    let sw = surface_width.max(1);
    let sh = surface_height.max(1);

    let mut g = state.viewer.lock();
    if let Some(v) = g.as_mut() {
        v.resize(
            sw,
            sh,
            viewport_x,
            viewport_y,
            viewport_width,
            viewport_height,
        );
        let (vw, vh) = v.viewport_size();
        let (sur_w, sur_h) = v.surface_pixel_size();
        let _ = app.emit_to(
            EventTarget::webview_window("main"),
            "viewport-pixel-size",
            ViewportPixelSize {
                width: vw,
                height: vh,
                surface_width: sur_w,
                surface_height: sur_h,
            },
        );
    }
    Ok(())
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
pub(crate) struct PointerEvent {
    kind: String,
    nx: f32,
    ny: f32,
    dx: f32,
    dy: f32,
    button: i32,
    buttons: u16,
    /// Left-drag pans when true (Three.js-style); otherwise left-drag orbits.
    #[serde(default, rename = "shiftKey")]
    shift_key: bool,
}

#[tauri::command]
pub(crate) fn viewport_pointer(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    ev: PointerEvent,
) -> Result<(), String> {
    if *state.fly_mode.lock() || *state.walk_mode.lock() {
        return Ok(());
    }
    // Read size without holding `camera` — the run loop locks `viewer` then `camera`; taking
    // `camera` then `viewer` here deadlocks with the render tick and freezes orbit input.
    let (vw, vh) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(());
        };
        let (w, h) = viewer.viewport_size();
        let w = w as f32;
        let h = h as f32;
        (w, h.max(1.0))
    };

    let (x, y) = viewport_texels_from_norm(ev.nx, ev.ny, vw, vh);

    // Check if logo overlay is active (use viewer lock, not camera lock).
    let logo_active = {
        let v = state.viewer.lock();
        v.as_ref().is_some_and(|viewer| viewer.logo_overlay_visible())
    };

    if logo_active {
        // Route pointer events to the logo overlay instead of the camera.
        let mut v = state.viewer.lock();
        if let Some(viewer) = v.as_mut() {
            if let Some(logo) = viewer.logo_overlay.as_mut() {
                match ev.kind.as_str() {
                    "down" | "move" => {
                        state.camera_dragging.store(ev.buttons != 0, Ordering::Relaxed);
                        logo.update_mouse_ndc(x, y, vw, vh);
                        if ev.buttons & 1 != 0 && !ev.shift_key {
                            logo.rotate_drag(ev.dx, ev.dy, vh);
                        } else if ev.kind == "move" && ev.buttons & 1 == 0 {
                            logo.hover_parallax(x, y, vw, vh);
                        }
                    }
                    "up" => {
                        state.camera_dragging.store(false, Ordering::Relaxed);
                        logo.reset_orbit();
                    }
                    "leave" => {
                        state.camera_dragging.store(false, Ordering::Relaxed);
                        logo.clear_mouse_ndc();
                        logo.hover_parallax(vw * 0.5, vh * 0.5, vw, vh);
                    }
                    _ => {}
                }
            }
        }
    } else {
        let mut cam = state.camera.lock();
        match ev.kind.as_str() {
            "down" | "move" => {
                state.camera_dragging.store(ev.buttons != 0, Ordering::Relaxed);
                // bitmask: 1=left orbit (or shift+left pan), 2=right pan, 4=middle dolly (Three.js OrbitControls defaults)
                if ev.buttons & 1 != 0 {
                    if ev.shift_key {
                        cam.pan_screen(ev.dx, ev.dy, vw, vh);
                    } else {
                        cam.rotate_screen(ev.dx, ev.dy, vh);
                    }
                } else if ev.buttons & 4 != 0 {
                    cam.dolly_delta(ev.dy);
                } else if ev.buttons & 2 != 0 {
                    cam.pan_screen(ev.dx, ev.dy, vw, vh);
                }
            }
            "up" => {
                state.camera_dragging.store(false, Ordering::Relaxed);
            }
            "leave" => {
                state.camera_dragging.store(false, Ordering::Relaxed);
            }
            _ => {}
        }
    }
    wake_viewport_loop(&app);
    Ok(())
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
pub(crate) struct WheelEvent {
    delta_x: f32,
    delta_y: f32,
}

#[tauri::command]
pub(crate) fn viewport_wheel(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    ev: WheelEvent,
) -> Result<(), String> {
    if *state.fly_mode.lock() || *state.walk_mode.lock() {
        return Ok(());
    }
    // Ignore scroll when logo overlay is active.
    {
        let v = state.viewer.lock();
        if v.as_ref().is_some_and(|viewer| viewer.logo_overlay_visible()) {
            return Ok(());
        }
    }
    let mut cam = state.camera.lock();
    // Same `deltaY` semantics as the browser / Three.js `onMouseWheel`.
    cam.dolly_delta(ev.delta_y);
    wake_viewport_loop(&app);
    Ok(())
}

pub(crate) fn scene_bounds_min_max_grid(state: &ViewerState) -> (glam::Vec3, glam::Vec3, i32) {
    let guard = state.last_scene_bounds.lock();
    if let Some(b) = guard.as_ref() {
        let grid = state
            .current_file
            .lock()
            .as_ref()
            .map(|file| file.grid_size)
            .unwrap_or(64);
        return (b.min, b.max, grid);
    }
    let fg = state.current_file.lock();
    if let Some(ref file) = *fg {
        let b = greedy_mesh::mesh_bounds_from_voxels_world(&file.voxels, &file.objects)
            .or_else(|| greedy_mesh::mesh_bounds_from_voxels(&file.voxels))
            .unwrap_or_else(|| greedy_mesh::mesh_bounds_for_cube_side(file.grid_size));
        return (b.min, b.max, file.grid_size);
    }
    let grid = 64_i32;
    let b = greedy_mesh::mesh_bounds_for_cube_side(grid);
    (b.min, b.max, grid)
}

fn perspective_zoom_base_dist(min: glam::Vec3, max: glam::Vec3, grid: i32) -> f32 {
    if (max - min).length() > 1e-3 {
        let dx = max.x - min.x;
        let dy = max.y - min.y;
        let dz = max.z - min.z;
        dx.max(dy).max(dz) * 1.5 + 10.0
    } else {
        grid as f32 * 2.5
    }
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct OrbitGizmoProjectionItem {
    sx: f32,
    sy: f32,
    depth: f32,
}

#[tauri::command]
pub(crate) fn get_orbit_gizmo_projection(
    state: State<'_, Arc<ViewerState>>,
) -> Result<Vec<OrbitGizmoProjectionItem>, String> {
    let cam = state.camera.lock();
    let axes = cam.gizmo_axis_projections();
    const R: f32 = 40.0;
    Ok(axes
        .into_iter()
        .map(|a| OrbitGizmoProjectionItem {
            sx: a[0] * R,
            sy: -a[1] * R,
            depth: a[2],
        })
        .collect())
}

#[tauri::command]
pub(crate) fn get_camera_zoom_percent(state: State<'_, Arc<ViewerState>>) -> Result<i32, String> {
    let (min, max, grid) = scene_bounds_min_max_grid(state.inner());
    let cam = state.camera.lock();
    let base = perspective_zoom_base_dist(min, max, grid);
    let r = (max - min).length() * 0.5;
    let ortho_ref = if r > 1e-3 {
        r * 1.1
    } else {
        (grid as f32) * 1.1
    };
    Ok(cam.zoom_percent_for_display(base, ortho_ref))
}

#[tauri::command]
pub(crate) fn camera_fit_to_scene(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    if *state.fly_mode.lock() || *state.walk_mode.lock() {
        return Ok(());
    }
    let (min, max, _) = scene_bounds_min_max_grid(state.inner());
    let (w, h) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(());
        };
        viewer.viewport_size()
    };
    let mut cam = state.camera.lock();
    cam.fit_to_aabb_preserving_view(min, max, w as f32, h as f32);
    drop(cam);
    wake_viewport_loop(&app);
    Ok(())
}

#[tauri::command]
pub(crate) fn camera_reset_view(app: AppHandle, state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    if *state.fly_mode.lock() || *state.walk_mode.lock() {
        return Ok(());
    }
    let (min, max, grid) = scene_bounds_min_max_grid(state.inner());
    let mut cam = state.camera.lock();
    cam.reset_view_to_bounds(min, max, grid as f32);
    drop(cam);
    wake_viewport_loop(&app);
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct OrbitGizmoDragArgs {
    dx: f32,
    dy: f32,
    theta_only: bool,
}

#[tauri::command]
pub(crate) fn camera_orbit_gizmo_drag(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: OrbitGizmoDragArgs,
) -> Result<(), String> {
    if *state.fly_mode.lock() || *state.walk_mode.lock() {
        return Ok(());
    }
    let mut cam = state.camera.lock();
    cam.orbit_gizmo_drag(args.dx, args.dy, args.theta_only);
    drop(cam);
    wake_viewport_loop(&app);
    Ok(())
}

// ── View settings commands ─────────────────────────────────────────────────────

#[tauri::command]
pub(crate) fn set_start_screen_light(state: State<'_, Arc<ViewerState>>, light: bool) -> Result<(), String> {
    state.start_screen_light.store(light, Ordering::Relaxed);
    Ok(())
}

#[tauri::command]
pub(crate) fn camera_snap_orbit_axis(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    axis: u8,
) -> Result<(), String> {
    if *state.fly_mode.lock() || *state.walk_mode.lock() {
        return Ok(());
    }
    let mut cam = state.camera.lock();
    cam.snap_to_axis(axis);
    drop(cam);
    wake_viewport_loop(&app);
    Ok(())
}

#[tauri::command]
pub(crate) fn camera_zoom_step(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    inward: bool,
) -> Result<(), String> {
    if *state.fly_mode.lock() || *state.walk_mode.lock() {
        return Ok(());
    }
    let mut cam = state.camera.lock();
    cam.zoom_step(inward);
    drop(cam);
    wake_viewport_loop(&app);
    Ok(())
}

pub(crate) fn apply_rendering_mode(
    state: &Arc<ViewerState>,
    app: &AppHandle,
    mode: RenderingMode,
) -> Result<(), String> {
    *state.rendering_mode.lock() = mode;
    // On the start screen, bump the overlay generation so in-flight smooth-mesh
    // builds are cancelled, then tell the frontend to reload logo + mascots.
    if !state.active_project.load(std::sync::atomic::Ordering::Relaxed) {
        state
            .overlay_mesh_generation
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let _ = app.emit("voxelle-reload-start-screen-overlays", ());
    }
    if mode.uses_smooth_surface() {
        // DC/MC mesh build can take many seconds; run it on a side thread so the
        // main thread stays responsive.  `schedule_opaque_mesh_refresh` handles
        // background compute + main-thread GPU upload with the stale-token guard.
        schedule_opaque_mesh_refresh(state, app);
        return Ok(());
    }
    refresh_opaque_mesh(state, Some(app))
}

pub(crate) fn apply_orthographic(state: &Arc<ViewerState>, orthographic: bool) -> Result<(), String> {
    {
        let mut cam = state.camera.lock();
        cam.perspective = !orthographic;
        if orthographic {
            let g = state.last_scene_bounds.lock();
            if let Some(b) = g.as_ref() {
                let r = b.radius().max(1.0);
                cam.ortho_half_height = r * 1.1;
            }
        }
    }
    {
        let mut fg = state.current_file.lock();
        if let Some(ref mut file) = *fg {
            file.scene.orthographic = orthographic;
        }
    }
    Ok(())
}

#[tauri::command]
pub(crate) fn get_rendering_mode(state: State<'_, Arc<ViewerState>>) -> Result<RenderingMode, String> {
    Ok(*state.rendering_mode.lock())
}

#[tauri::command]
pub(crate) fn set_rendering_mode(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    mode: RenderingMode,
) -> Result<(), String> {
    apply_rendering_mode(state.inner(), &app, mode)?;
    wake_viewport_loop(&app);
    #[cfg(desktop)]
    if let Some(sel) = app.try_state::<SelectionMenuState>() {
        let _ = sel
            .render_greedy
            .set_checked(matches!(mode, RenderingMode::Greedy));
        let _ = sel
            .render_marching
            .set_checked(matches!(mode, RenderingMode::MarchingCubes));
        let _ = sel
            .render_dual
            .set_checked(matches!(mode, RenderingMode::DualContour));
    }
    Ok(())
}

#[tauri::command]
pub(crate) fn get_raytrace_mode(
    state: State<'_, Arc<ViewerState>>,
) -> Result<bool, String> {
    let v = state.viewer.lock();
    Ok(v.as_ref().is_some_and(|viewer| viewer.raytrace_enabled))
}

#[tauri::command]
pub(crate) fn set_raytrace_mode(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    enabled: bool,
) -> Result<(), String> {
    let mut v = state.viewer.lock();
    let viewer = v.as_mut().ok_or("viewer not ready")?;
    viewer.set_raytrace_mode(enabled);
    drop(v);
    wake_viewport_loop(&app);
    #[cfg(desktop)]
    if let Some(sel) = app.try_state::<SelectionMenuState>() {
        let _ = sel.render_ray.set_checked(enabled);
    }
    let _ = app.emit_to(
        tauri::EventTarget::webview_window("main"),
        "voxelle-raytrace-changed",
        enabled,
    );
    Ok(())
}

#[tauri::command]
pub(crate) fn benchmark_raytrace(
    state: State<'_, Arc<ViewerState>>,
    frame_count: Option<u32>,
) -> Result<crate::render::RaytraceBenchmarkResult, String> {
    let mut v = state.viewer.lock();
    let viewer = v.as_mut().ok_or("viewer not ready")?;
    Ok(viewer.run_raytrace_benchmark(frame_count.unwrap_or(50)))
}

#[tauri::command]
pub(crate) fn get_orthographic(state: State<'_, Arc<ViewerState>>) -> Result<bool, String> {
    Ok(!state.camera.lock().perspective)
}

#[tauri::command]
pub(crate) fn set_orthographic(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    orthographic: bool,
) -> Result<(), String> {
    apply_orthographic(state.inner(), orthographic)?;
    wake_viewport_loop(&app);
    Ok(())
}

#[tauri::command]
pub(crate) fn get_show_grid_borders(state: State<'_, Arc<ViewerState>>) -> Result<bool, String> {
    Ok(state.show_grid_borders.load(Ordering::Relaxed))
}

/// Keeps **View -> Show borders** in sync with webview (e.g. after restoring preferences).
#[tauri::command]
pub(crate) fn view_menu_sync_show_borders(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    show: bool,
) -> Result<(), String> {
    state.show_grid_borders.store(show, Ordering::Relaxed);
    #[cfg(desktop)]
    {
        if let Some(menu) = app.try_state::<SelectionMenuState>() {
            menu.view_show_borders
                .set_checked(show)
                .map_err(|e| e.to_string())?;
        }
        wake_viewport_loop(&app);
    }
    #[cfg(not(desktop))]
    {
        let _ = app;
    }
    Ok(())
}

/// Keeps **View -> Hide UI** in sync with webview state.
#[tauri::command]
pub(crate) fn view_menu_sync_hide_ui(app: AppHandle, hidden: bool) -> Result<(), String> {
    #[cfg(desktop)]
    {
        if let Some(menu) = app.try_state::<SelectionMenuState>() {
            menu.view_hide_ui
                .set_checked(hidden)
                .map_err(|e| e.to_string())?;
        }
    }
    #[cfg(not(desktop))]
    {
        let _ = (app, hidden);
    }
    Ok(())
}

/// Keeps **Debug -> Viewport cursor debug overlay** in sync with webview / `localStorage`.
#[tauri::command]
pub(crate) fn debug_menu_sync_viewport_cursor_overlay(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    enabled: bool,
) -> Result<(), String> {
    state
        .viewport_cursor_debug_overlay
        .store(enabled, Ordering::Relaxed);
    #[cfg(desktop)]
    {
        if let Some(menu) = app.try_state::<SelectionMenuState>() {
            menu.viewport_cursor_debug
                .set_checked(enabled)
                .map_err(|e| e.to_string())?;
        }
    }
    #[cfg(not(desktop))]
    {
        let _ = app;
    }
    Ok(())
}

#[tauri::command]
pub(crate) fn set_soft_shadows(state: State<'_, Arc<ViewerState>>, enabled: bool) -> Result<(), String> {
    if let Some(viewer) = state.viewer.lock().as_mut() {
        viewer.soft_shadows = enabled;
    }
    Ok(())
}

#[tauri::command]
pub(crate) fn set_gizmo_on_top(state: State<'_, Arc<ViewerState>>, enabled: bool) -> Result<(), String> {
    if let Some(viewer) = state.viewer.lock().as_mut() {
        viewer.set_gizmo_on_top(enabled);
    }
    Ok(())
}

#[tauri::command]
pub(crate) fn set_soft_sunshafts(state: State<'_, Arc<ViewerState>>, enabled: bool) -> Result<(), String> {
    if let Some(viewer) = state.viewer.lock().as_mut() {
        viewer.set_soft_sunshafts(enabled);
    }
    Ok(())
}

#[tauri::command]
pub(crate) fn set_emission_lighting(state: State<'_, Arc<ViewerState>>, enabled: bool) -> Result<(), String> {
    greedy_mesh::EMISSION_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
    // Invalidate the mesh cache so the next frame triggers a full remesh with the new setting.
    if let Some(viewer) = state.viewer.lock().as_mut() {
        viewer.invalidate_spatial_mesh_cache();
    }
    state
        .mesh_refresh_generation
        .fetch_add(1, std::sync::atomic::Ordering::Release);
    Ok(())
}

#[tauri::command]
pub(crate) fn set_tone_mapping(state: State<'_, Arc<ViewerState>>, mode: u32) -> Result<(), String> {
    let mut v = state.viewer.lock();
    let Some(viewer) = v.as_mut() else {
        return Err("viewer not ready".into());
    };
    viewer.set_tone_mapping_mode(mode);
    Ok(())
}

#[tauri::command]
pub(crate) fn is_hdr_available(state: State<'_, Arc<ViewerState>>) -> Result<bool, String> {
    let v = state.viewer.lock();
    let Some(viewer) = v.as_ref() else {
        return Err("viewer not ready".into());
    };
    Ok(viewer.hdr_available())
}

#[tauri::command]
pub(crate) fn set_hdr_output(state: State<'_, Arc<ViewerState>>, enabled: bool) -> Result<(), String> {
    let mut v = state.viewer.lock();
    let Some(viewer) = v.as_mut() else {
        return Err("viewer not ready".into());
    };
    viewer.set_hdr_output(enabled);
    Ok(())
}

/// Convert `MoodParams` (from frontend) -> file-format `MoodSettings`.
fn mood_params_to_settings(p: &MoodParams) -> voxelle::MoodSettings {
    voxelle::MoodSettings {
        vignette: p.vignette,
        grain_enabled: p.grain_enabled,
        grain_strength: p.grain_strength,
        grain_animated: p.grain_animated,
        grain_speed: p.grain_speed,
        grain_colorful: p.grain_colorful,
        atm_enabled: p.atm_enabled,
        atm_color: p.atm_color.clone(),
        atm_thickness: p.atm_thickness,
        atm_density: p.atm_density,
        atm_aerial: p.atm_aerial,
        atm_positive_side: p.atm_positive_side,
        atm_plane_nx: p.atm_plane_nx,
        atm_plane_ny: p.atm_plane_ny,
        atm_plane_nz: p.atm_plane_nz,
        atm_plane_c: p.atm_plane_c,
        atm_height_bias: p.atm_height_bias,
        atm_height_falloff: p.atm_height_falloff,
        atm_drift_enabled: p.atm_drift_enabled,
        atm_drift_amount: p.atm_drift_amount,
        atm_drift_scale: p.atm_drift_scale,
        atm_drift_speed: p.atm_drift_speed,
        dt_enabled: p.dt_enabled,
        dt_near_color: p.dt_near_color.clone(),
        dt_mid_color: p.dt_mid_color.clone(),
        dt_far_color: p.dt_far_color.clone(),
        dt_near_dist: p.dt_near_dist,
        dt_far_dist: p.dt_far_dist,
        dt_strength: p.dt_strength,
        ss_enabled: p.ss_enabled,
        ss_strength: p.ss_strength,
        ss_decay: p.ss_decay,
        ss_density: p.ss_density,
        ss_weight: p.ss_weight,
        ss_samples: p.ss_samples,
        ssr_enabled: p.ssr_enabled,
        ssr_strength: p.ssr_strength,
        bloom_strength: p.bloom_strength,
    }
}

#[tauri::command]
pub(crate) fn set_mood_params(state: State<'_, Arc<ViewerState>>, args: MoodParams) -> Result<(), String> {
    let mut v = state.viewer.lock();
    let Some(viewer) = v.as_mut() else {
        return Err("viewer not ready".into());
    };
    viewer.set_mood_params(&args);
    drop(v);
    {
        let mut cf = state.current_file.lock();
        if let Some(f) = cf.as_mut() {
            f.mood = Some(mood_params_to_settings(&args));
        }
    }
    Ok(())
}

#[tauri::command]
pub(crate) fn set_scene_lighting(
    state: State<'_, Arc<ViewerState>>,
    args: voxelle::LightingSettings,
) -> Result<(), String> {
    let mut v = state.viewer.lock();
    let Some(viewer) = v.as_mut() else {
        return Err("viewer not ready".into());
    };
    viewer.apply_lighting_settings(&args);
    drop(v);
    {
        let mut cf = state.current_file.lock();
        if let Some(f) = cf.as_mut() {
            f.lighting = Some(args);
        }
    }
    Ok(())
}

#[tauri::command]
pub(crate) fn get_scene_lighting(
    state: State<'_, Arc<ViewerState>>,
) -> Result<voxelle::LightingSettings, String> {
    let g = state.current_file.lock();
    let Some(f) = g.as_ref() else {
        return Ok(voxelle::LightingSettings::default());
    };
    Ok(f.lighting.clone().unwrap_or_default())
}

#[tauri::command]
pub(crate) fn set_focal_length_mm(state: State<'_, Arc<ViewerState>>, mm: f32) -> Result<(), String> {
    let mm = mm.clamp(15.0, 200.0);
    let mut cam = state.camera.lock();
    if !cam.perspective {
        return Ok(());
    }
    cam.fov_y = focal_length_to_fov_y_radians(mm);
    {
        let mut cf = state.current_file.lock();
        if let Some(f) = cf.as_mut() {
            f.scene.focal_length_mm = Some(mm);
        }
    }
    Ok(())
}

#[tauri::command]
pub(crate) fn get_focal_length_mm(state: State<'_, Arc<ViewerState>>) -> Result<f32, String> {
    let g = state.current_file.lock();
    let Some(f) = g.as_ref() else {
        return Ok(29.0);
    };
    Ok(f.scene.focal_length_mm.unwrap_or(29.0))
}

// ── Fly / walk mode commands ───────────────────────────────────────────────────

#[tauri::command]
pub(crate) fn set_fly_mode(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    enabled: bool,
) -> Result<(), String> {
    *state.fly_mode.lock() = enabled;
    let mut cam = state.camera.lock();
    cam.is_fly_mode = enabled;
    if enabled {
        *state.fly_last_physics.lock() = None;
        drop(cam);
        wake_viewport_loop(&app);
    }
    Ok(())
}

#[tauri::command]
pub(crate) fn get_fly_mode(state: State<'_, Arc<ViewerState>>) -> Result<bool, String> {
    Ok(*state.fly_mode.lock())
}

#[tauri::command]
pub(crate) fn set_walk_mode(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    enabled: bool,
) -> Result<(), String> {
    if enabled {
        // Disable fly mode when entering walk mode.
        *state.fly_mode.lock() = false;
        state.camera.lock().is_fly_mode = false;
    }
    *state.walk_mode.lock() = enabled;
    let mut cam = state.camera.lock();
    cam.is_walk_mode = enabled;
    if enabled {
        // Initialize walk physics from current camera position.
        let eye = cam.target + cam.spherical.to_offset();
        let feet = glam::Vec3::new(eye.x, eye.y - camera::WALK_EYE_HEIGHT, eye.z);
        *state.walk_physics.lock() = camera::WalkPhysicsState {
            feet_pos: feet,
            vel_y: 0.0,
            on_ground: false,
        };
        *state.walk_last_physics.lock() = None;
        drop(cam);
        wake_viewport_loop(&app);
    }
    Ok(())
}

/// Check if a voxel coordinate is occupied.
#[inline]
fn walk_is_solid(vm: &AHashMap<greedy_mesh::VoxelCoord, usize>, x: i32, y: i32, z: i32) -> bool {
    vm.contains_key(&(x, y, z))
}

/// Resolve walk-mode collision against the voxel grid. Returns corrected feet position.
pub(crate) fn resolve_walk_collision(
    old_feet: glam::Vec3,
    mut new_feet: glam::Vec3,
    vm: &AHashMap<greedy_mesh::VoxelCoord, usize>,
    wp: &mut camera::WalkPhysicsState,
) -> glam::Vec3 {
    // --- Vertical collision (process Y first) ---
    let fc = voxel_edit::world_to_voxel(new_feet);

    // Check voxel AT feet level: are we inside a solid block?
    if walk_is_solid(vm, fc.0, fc.1, fc.2) {
        let ground_top_y = fc.1 as f32 + 0.5;
        new_feet.y = ground_top_y;
        wp.vel_y = 0.0;
        wp.on_ground = true;
    } else {
        // Check voxel directly below feet
        if walk_is_solid(vm, fc.0, fc.1 - 1, fc.2) {
            let ground_top_y = (fc.1 - 1) as f32 + 0.5;
            if new_feet.y <= ground_top_y + 0.05 {
                new_feet.y = ground_top_y;
                wp.vel_y = 0.0;
                wp.on_ground = true;
            }
        } else {
            wp.on_ground = false;
        }
    }

    // Ceiling collision: check at head height
    let head_pos = new_feet + glam::Vec3::Y * camera::WALK_EYE_HEIGHT;
    let hc = voxel_edit::world_to_voxel(head_pos);
    if walk_is_solid(vm, hc.0, hc.1, hc.2) && wp.vel_y > 0.0 {
        wp.vel_y = 0.0;
        let ceiling_bottom_y = hc.1 as f32 - 0.5;
        new_feet.y = ceiling_bottom_y - camera::WALK_EYE_HEIGHT;
    }

    // --- Horizontal collision + auto step-up ---
    // Check body voxels at the new horizontal position
    let body_low = voxel_edit::world_to_voxel(new_feet + glam::Vec3::Y * 0.1);
    let body_high = voxel_edit::world_to_voxel(new_feet + glam::Vec3::Y * 1.0);

    let blocked_low = walk_is_solid(vm, body_low.0, body_low.1, body_low.2);
    let blocked_high = walk_is_solid(vm, body_high.0, body_high.1, body_high.2);

    if blocked_low && !blocked_high {
        // Step-up candidate: blocked at feet but clear at torso
        let step_top = body_low.1 as f32 + 0.5;
        let step_height = step_top - new_feet.y;
        // Check head clearance above the step
        let clearance_ok = !walk_is_solid(vm, body_low.0, body_low.1 + 2, body_low.2);
        if clearance_ok && step_height <= camera::WALK_STEP_HEIGHT {
            new_feet.y = step_top;
            wp.vel_y = 0.0;
            wp.on_ground = true;
        } else {
            // Can't step up -- try wall sliding
            new_feet = walk_slide(old_feet, new_feet, vm);
        }
    } else if blocked_low || blocked_high {
        // Full wall block -- try wall sliding
        new_feet = walk_slide(old_feet, new_feet, vm);
    }

    new_feet
}

/// Wall sliding: try X-only, then Z-only, then full revert.
fn walk_slide(
    old_feet: glam::Vec3,
    new_feet: glam::Vec3,
    vm: &AHashMap<greedy_mesh::VoxelCoord, usize>,
) -> glam::Vec3 {
    // Try sliding along X only (revert Z)
    let try_x = glam::Vec3::new(new_feet.x, new_feet.y, old_feet.z);
    let bx_low = voxel_edit::world_to_voxel(try_x + glam::Vec3::Y * 0.1);
    let bx_high = voxel_edit::world_to_voxel(try_x + glam::Vec3::Y * 1.0);
    if !walk_is_solid(vm, bx_low.0, bx_low.1, bx_low.2)
        && !walk_is_solid(vm, bx_high.0, bx_high.1, bx_high.2)
    {
        return try_x;
    }

    // Try sliding along Z only (revert X)
    let try_z = glam::Vec3::new(old_feet.x, new_feet.y, new_feet.z);
    let bz_low = voxel_edit::world_to_voxel(try_z + glam::Vec3::Y * 0.1);
    let bz_high = voxel_edit::world_to_voxel(try_z + glam::Vec3::Y * 1.0);
    if !walk_is_solid(vm, bz_low.0, bz_low.1, bz_low.2)
        && !walk_is_solid(vm, bz_high.0, bz_high.1, bz_high.2)
    {
        return try_z;
    }

    // Fully blocked: revert horizontal
    glam::Vec3::new(old_feet.x, new_feet.y, old_feet.z)
}

fn fly_speed_scale_default() -> f32 {
    1.0
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SyncFlyInputArgs {
    forward: f32,
    right: f32,
    up: f32,
    #[serde(default = "fly_speed_scale_default")]
    speed_scale: f32,
    #[serde(default)]
    jump: bool,
}

/// WASD / shift state only. Translation integrates on the native event loop with real elapsed time.
#[tauri::command]
pub(crate) fn sync_fly_input(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: SyncFlyInputArgs,
) -> Result<(), String> {
    let fly = *state.fly_mode.lock();
    let walk = *state.walk_mode.lock();
    if !fly && !walk {
        return Ok(());
    }
    let scale = args.speed_scale;
    let speed_scale = if scale.is_finite() {
        scale.clamp(0.0, 1e6)
    } else {
        1.0
    };
    let has_movement = args.forward != 0.0 || args.right != 0.0 || args.up != 0.0 || args.jump;
    *state.fly_input.lock() = FlyInputState {
        forward: args.forward,
        right: args.right,
        up: args.up,
        speed_scale,
        jump: args.jump,
    };
    if has_movement {
        wake_viewport_loop(&app);
    }
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FlyLookArgs {
    dx: f32,
    dy: f32,
}

#[tauri::command]
pub(crate) fn camera_fly_look(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: FlyLookArgs,
) -> Result<(), String> {
    if !*state.fly_mode.lock() && !*state.walk_mode.lock() {
        return Ok(());
    }
    let vh = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(());
        };
        let (_, h) = viewer.viewport_size();
        h as f32
    };
    let mut cam = state.camera.lock();
    cam.fly_look_rotate_screen(args.dx, args.dy, vh.max(1.0));
    wake_viewport_loop(&app);
    Ok(())
}

// ── Input / query commands ─────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PickAtScreen {
    pub(crate) nx: f32,
    pub(crate) ny: f32,
}

/// Whether the camera ray from this screen point hits solid geometry (voxel) -- used to choose camera vs edit.
#[tauri::command]
pub(crate) fn voxel_pick_probe(
    state: State<'_, Arc<ViewerState>>,
    args: PickAtScreen,
) -> Result<bool, String> {
    let (w, h) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(false);
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let fg = state.current_file.lock();
    let vm = state.voxel_map.lock();
    let Some(file) = fg.as_ref() else {
        return Ok(false);
    };
    let Some(vmap) = vm.as_ref() else {
        return Ok(false);
    };
    let cam = state.camera.lock();
    Ok(voxel_edit::probe_solid_hit(file, vmap, &cam, w, h, sx, sy))
}

/// Returns the surface Y (topmost voxel) at the given screen position, for the terrain hover display.
#[tauri::command]
pub(crate) fn terrain_surface_y_at_screen(
    state: State<'_, Arc<ViewerState>>,
    nx: f32,
    ny: f32,
) -> Result<Option<i32>, String> {
    let (w, h) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(None);
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let fg = state.current_file.lock();
    let Some(file) = fg.as_ref() else {
        return Ok(None);
    };
    let vm = state.voxel_map.lock();
    let Some(vmap) = vm.as_ref() else {
        return Ok(None);
    };
    let cam = state.camera.lock();
    let (sx, sy) = viewport_texels_from_norm(nx, ny, w, h);
    let c = voxel_edit::anchor_for_stroke_edit(
        voxel_edit::EditTool::Remove,
        true,
        file,
        vmap,
        &cam,
        w,
        h,
        sx,
        sy,
    );
    Ok(c.map(|(_, y, _)| y))
}

fn pick_cell_for_ping(
    mode: PreviewMode,
    file: &voxelle::VoxelleFile,
    vmap: &AHashMap<greedy_mesh::VoxelCoord, usize>,
    cam: &camera::OrbitCamera,
    w: f32,
    h: f32,
    sx: f32,
    sy: f32,
) -> Option<(i32, i32, i32)> {
    match mode {
        PreviewMode::Add => {
            voxel_edit::preview_add_cell(file, vmap, cam, w, h, sx, sy).map(|(c, _)| c)
        }
        PreviewMode::Remove | PreviewMode::Paint | PreviewMode::Select => {
            voxel_edit::preview_remove_cell(file, vmap, cam, w, h, sx, sy).map(|(c, _)| c)
        }
        PreviewMode::Navigate
        | PreviewMode::Fly
        | PreviewMode::Squishy
        | PreviewMode::Bone
        | PreviewMode::SelectExtrude => {
            voxel_edit::preview_remove_cell(file, vmap, cam, w, h, sx, sy)
                .map(|(c, _)| c)
                .or_else(|| {
                    voxel_edit::preview_add_cell(file, vmap, cam, w, h, sx, sy).map(|(c, _)| c)
                })
        }
        PreviewMode::Stamp => {
            voxel_edit::preview_add_cell(file, vmap, cam, w, h, sx, sy).map(|(c, _)| c)
        }
        PreviewMode::Punch => {
            voxel_edit::preview_remove_cell(file, vmap, cam, w, h, sx, sy).map(|(c, _)| c)
        }
    }
}

fn local_accent_ping_color(state: &ViewerState) -> u32 {
    let c = state.collab.lock();
    c.roster
        .iter()
        .find(|r| r.peer_id == c.local_peer_id)
        .map(|r| r.color_rgb)
        .unwrap_or(0x66ccff)
}

fn local_accent_ping_display_name(state: &ViewerState) -> String {
    let c = state.collab.lock();
    c.roster
        .iter()
        .find(|r| r.peer_id == c.local_peer_id)
        .map(|r| {
            if r.display_name.trim().is_empty() {
                "You".to_string()
            } else {
                r.display_name.clone()
            }
        })
        .unwrap_or_else(|| "You".to_string())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PingCursorPickArgs {
    nx: f32,
    ny: f32,
    #[serde(default)]
    display_name: String,
    #[serde(default)]
    emoji: String,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PingCursorPickResult {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    x: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    y: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    z: Option<i32>,
}

/// Brief highlight at the voxel cell under the cursor ray (add / remove / navigate semantics).
#[tauri::command]
pub(crate) fn ping_cursor_pick(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: PingCursorPickArgs,
) -> Result<PingCursorPickResult, String> {
    let (w, h) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(PingCursorPickResult {
                ok: false,
                x: None,
                y: None,
                z: None,
            });
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let mode = *state.preview_mode.lock();
    let coords = {
        let fg = state.current_file.lock();
        let vm = state.voxel_map.lock();
        let Some(file) = fg.as_ref() else {
            return Ok(PingCursorPickResult {
                ok: false,
                x: None,
                y: None,
                z: None,
            });
        };
        let Some(vmap) = vm.as_ref() else {
            return Ok(PingCursorPickResult {
                ok: false,
                x: None,
                y: None,
                z: None,
            });
        };
        let cam = state.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        pick_cell_for_ping(mode, file, vmap, &cam, w, h, sx, sy)
    };
    let Some((x, y, z)) = coords else {
        return Ok(PingCursorPickResult {
            ok: false,
            x: None,
            y: None,
            z: None,
        });
    };
    let color = local_accent_ping_color(&state);
    let label = if !args.display_name.trim().is_empty() {
        args.display_name.trim().to_string()
    } else {
        local_accent_ping_display_name(&state)
    };
    collab::record_ping_flash_colored(Arc::as_ref(&*state), x, y, z, color, label, args.emoji);
    wake_viewport_loop(&app);
    Ok(PingCursorPickResult {
        ok: true,
        x: Some(x),
        y: Some(y),
        z: Some(z),
    })
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WorldPointArgs {
    x: f32,
    y: f32,
    z: f32,
}

#[tauri::command]
pub(crate) fn world_to_viewport_pixels(
    state: State<'_, Arc<ViewerState>>,
    args: WorldPointArgs,
) -> Result<Option<(f32, f32)>, String> {
    let (w, h) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(None);
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let cam = state.camera.lock();
    Ok(voxel_edit::world_to_viewport_pixels(
        &cam, w, h, args.x, args.y, args.z,
    ))
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ProjectWorldPointResult {
    on_screen: bool,
    sx: f32,
    sy: f32,
    vw: f32,
    vh: f32,
}

/// Project a world point to screen, always returning coords even when off-screen.
#[tauri::command]
pub(crate) fn project_world_point(
    state: State<'_, Arc<ViewerState>>,
    args: WorldPointArgs,
) -> Result<ProjectWorldPointResult, String> {
    let (w, h) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(ProjectWorldPointResult {
                on_screen: false,
                sx: 0.0,
                sy: 0.0,
                vw: 0.0,
                vh: 0.0,
            });
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let cam = state.camera.lock();
    let view = cam.view_matrix();
    let proj = cam.proj_matrix(w.max(1.0), h.max(1.0));
    let vp = proj * view;
    let clip = vp * glam::Vec4::new(args.x, args.y, args.z, 1.0);

    if clip.w.abs() < 1e-5 {
        return Ok(ProjectWorldPointResult {
            on_screen: false,
            sx: w * 0.5,
            sy: h * 0.5,
            vw: w,
            vh: h,
        });
    }

    let mut ndc_x = clip.x / clip.w;
    let mut ndc_y = clip.y / clip.w;

    // Behind camera: flip so the arrow points the right way
    if clip.w < 0.0 {
        ndc_x = -ndc_x;
        ndc_y = -ndc_y;
    }

    let sx_raw = (ndc_x + 1.0) * 0.5 * w - 0.5;
    let sy_raw = (1.0 - ndc_y) * 0.5 * h - 0.5;

    let on_screen = clip.w > 0.0 && ndc_x.abs() <= 1.0 && ndc_y.abs() <= 1.0;
    let sx = sx_raw.clamp(0.0, w);
    let sy = sy_raw.clamp(0.0, h);

    Ok(ProjectWorldPointResult {
        on_screen,
        sx,
        sy,
        vw: w,
        vh: h,
    })
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PeerLabel {
    name: String,
    color_rgb: u32,
    left_pct: f32,
    top_pct: f32,
}

#[tauri::command]
pub(crate) fn collab_peer_labels(state: State<'_, Arc<ViewerState>>) -> Result<Vec<PeerLabel>, String> {
    let (w, h) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(vec![]);
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    if w <= 0.0 || h <= 0.0 {
        return Ok(vec![]);
    }
    let cam = state.camera.lock();
    let c = state.collab.lock();
    if !c.is_active() {
        return Ok(vec![]);
    }
    let local_id = c.local_peer_id;
    let smooth = state.smooth_presence.lock();
    let mut labels = Vec::new();
    for (pid, pr) in smooth.iter() {
        if *pid == local_id {
            continue;
        }
        let eye = collab::presence_eye(pr);
        let Some((sx, sy)) = voxel_edit::world_to_viewport_pixels(&cam, w, h, eye.x, eye.y, eye.z)
        else {
            continue;
        };
        let entry = c.roster.iter().find(|r| r.peer_id == *pid);
        let name = entry.map(|r| r.display_name.clone()).unwrap_or_default();
        let color_rgb = entry.map(|r| r.color_rgb).unwrap_or(0x888888);
        labels.push(PeerLabel {
            name,
            color_rgb,
            left_pct: (sx / w) * 100.0,
            top_pct: (sy / h) * 100.0,
        });
    }
    Ok(labels)
}
