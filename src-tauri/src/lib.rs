mod camera;
mod collab;
mod commands;
pub mod crash_guard;
mod edit_pipeline;
mod export_glb;
mod frame_loop;
mod generators;
mod gpu_brick;
/// Greedy CPU meshing (public for `cargo bench`).
pub mod greedy_mesh;
#[cfg(desktop)]
mod headless_server;
mod load_pipeline;
#[cfg(target_os = "macos")]
mod macos_titlebar;
#[cfg(target_os = "macos")]
mod macos_undo;
mod marching_tables;
#[cfg(desktop)]
mod native_menu;
mod paint_color_distrib;
mod preview;
mod render;
mod render_constants;
mod sculpt_mesh_smooth;
mod smooth_mesh;
mod state;
mod stroke_modes;
mod voxel_edit;
/// Voxel format / types (public for `cargo bench` and tests).
pub mod voxelle;
use commands::avatar::*;
use commands::collab::*;
use commands::edit::*;
use commands::file_io::*;
use commands::generators::*;
use commands::sculpt::*;
use commands::selection::*;
use commands::viewport::*;
pub(crate) use commands::SculptStrokeAtScreenArgs;
use edit_pipeline::*;
use frame_loop::*;
use load_pipeline::*;
#[cfg(desktop)]
use native_menu::*;
use preview::*;
use state::*;

use camera::OrbitCamera;
use gpu_brick::BrickCellWrite;
use render::{compute_greedy_rebuild_cpu, MoodParams, PreparedGreedyRebuild, WgpuViewer};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use std::time::Instant;
use tauri::{AppHandle, Emitter, EventTarget, Manager, RunEvent, Runtime, State};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

use ahash::{AHashMap, AHashSet};
use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::path::{Path, PathBuf};

use voxelle::scene::object_world_matrix;
use voxelle::{
    decode_payload, encode_payload_v4, focal_length_to_fov_y_radians, start_shape::StartShape,
};

/// Convert file-format `MoodSettings` → GPU-ready `MoodParams`.
pub(crate) fn mood_settings_to_params(m: &voxelle::MoodSettings) -> MoodParams {
    MoodParams {
        vignette: m.vignette,
        grain_enabled: m.grain_enabled,
        grain_strength: m.grain_strength,
        grain_animated: m.grain_animated,
        grain_speed: m.grain_speed,
        grain_colorful: m.grain_colorful,
        atm_enabled: m.atm_enabled,
        atm_color: m.atm_color.clone(),
        atm_thickness: m.atm_thickness,
        atm_density: m.atm_density,
        atm_aerial: m.atm_aerial,
        atm_positive_side: m.atm_positive_side,
        atm_plane_nx: m.atm_plane_nx,
        atm_plane_ny: m.atm_plane_ny,
        atm_plane_nz: m.atm_plane_nz,
        atm_plane_c: m.atm_plane_c,
        atm_height_bias: m.atm_height_bias,
        atm_height_falloff: m.atm_height_falloff,
        atm_drift_enabled: m.atm_drift_enabled,
        atm_drift_amount: m.atm_drift_amount,
        atm_drift_scale: m.atm_drift_scale,
        atm_drift_speed: m.atm_drift_speed,
        dt_enabled: m.dt_enabled,
        dt_near_color: m.dt_near_color.clone(),
        dt_mid_color: m.dt_mid_color.clone(),
        dt_far_color: m.dt_far_color.clone(),
        dt_near_dist: m.dt_near_dist,
        dt_far_dist: m.dt_far_dist,
        dt_strength: m.dt_strength,
        ss_enabled: m.ss_enabled,
        ss_strength: m.ss_strength,
        ss_decay: m.ss_decay,
        ss_density: m.ss_density,
        ss_weight: m.ss_weight,
        ss_samples: m.ss_samples,
        ssr_enabled: m.ssr_enabled,
        ssr_strength: m.ssr_strength,
        bloom_strength: m.bloom_strength,
    }
}

/// Hover preview uses the same brush/stroke inputs as [`voxel_edit_at_screen`] / [`voxel_stroke_preview_at_screen`].
#[derive(Clone, Debug)]
pub(crate) struct PreviewHoverContext {
    brush_radius: u32,
    brush_shape: voxel_edit::BrushShape,
    spray_density: f32,
    stroke_mode: stroke_modes::DrawStrokeMode,
    plane_axis: stroke_modes::PlaneAxis,
    stroke_aux: stroke_modes::StrokeAux,
    color: u32,
    palette: Vec<u32>,
    paint_color_distrib: Option<paint_color_distrib::PaintColorDistrib>,
    material: String,
    match_material: bool,
    /// When false (e.g. sculpt), hover uses the legacy single-cell preview.
    use_brush_preview: bool,
    /// `Some("rope" | "cloth" | "rocks" | "grass")` when the generator tool is active (webview sync).
    generator_kind: Option<String>,
    generator_rope_first_nx: Option<f32>,
    generator_rope_first_ny: Option<f32>,
    generator_rope_sag: f32,
    generator_rope_tension: f32,
    generator_rope_gravity_direction: String,
    generator_cloth_pins: Vec<[i32; 3]>,
    generator_cloth_tension: f32,
    generator_cloth_gravity_direction: String,
    generator_cloth_gravity_scale: f64,
    generator_cloth_stiffness_scale: f64,
    generator_cloth_iterations: u32,
    generator_cloth_constraint_passes: u32,
    generator_rock_size: i32,
    generator_rock_roughness: f32,
    generator_rock_seed: i32,
    generator_rock_count: i32,
    generator_rock_cluster_radius: i32,
    generator_rock_sink_direction: i32,
    generator_rock_sink_amount: i32,
    generator_grass_radius: i32,
    generator_grass_density: f32,
    generator_grass_max_height: i32,
    generator_grass_seed: i32,
    generator_roof_pins: Vec<[i32; 3]>,
    generator_roof_style: String,
    generator_roof_height: i32,
    generator_roof_thickness: i32,
    generator_roof_break_ratio: f32,
    generator_roof_wall_height: i32,
    generator_roof_parapet_height: i32,
    generator_roof_salt_skew: f32,
    generator_roof_hollow: bool,
    generator_ashlar_size: i32,
    generator_ashlar_roughness: f32,
    generator_ashlar_seed: i32,
    generator_ashlar_thickness: i32,
    // Flora
    generator_flora_seed: i32,
    generator_flora_height: i32,
    generator_flora_girth: i32,
    generator_flora_wobble: f32,
    generator_flora_taper: f32,
    generator_flora_stem_count: i32,
    generator_flora_cluster_radius: i32,
    generator_flora_branch_count: i32,
    generator_flora_branch_depth: i32,
    generator_flora_branch_start: f32,
    generator_flora_branch_spread: f32,
    generator_flora_braid_strands: i32,
    generator_flora_braid_twist: f32,
    generator_flora_canopy: f32,
    // Insecta
    generator_insecta_species: String,
    generator_insecta_total_length: i32,
    generator_insecta_head_ratio: f32,
    generator_insecta_thorax_ratio: f32,
    generator_insecta_abdomen_ratio: f32,
    generator_insecta_body_half_width: i32,
    generator_insecta_body_half_height: i32,
    generator_insecta_abdomen_taper: f32,
    generator_insecta_head_shape: i32,
    generator_insecta_anchor_offset_u: i32,
    generator_insecta_anchor_offset_v: i32,
    generator_insecta_body_yaw: f32,
    generator_insecta_body_arch: f32,
    generator_insecta_antenna_length: i32,
    generator_insecta_antenna_spread: f32,
    generator_insecta_antenna_pitch: f32,
    generator_insecta_antenna_root: i32,
    generator_insecta_mandible_length: i32,
    generator_insecta_mandible_spread: f32,
    generator_insecta_mandible_forward: i32,
    generator_insecta_wing_shape: i32,
    generator_insecta_show_wing_fore: bool,
    generator_insecta_wing_fore_length: i32,
    generator_insecta_wing_fore_width: i32,
    generator_insecta_wing_fore_spread: f32,
    generator_insecta_wing_fore_pitch: f32,
    generator_insecta_wing_fore_offset: i32,
    generator_insecta_wing_fore_forward_cant: f32,
    generator_insecta_show_wing_hind: bool,
    generator_insecta_wing_hind_length: i32,
    generator_insecta_wing_hind_width: i32,
    generator_insecta_wing_hind_spread: f32,
    generator_insecta_wing_hind_pitch: f32,
    generator_insecta_wing_hind_offset: i32,
    // Fauna
    generator_fauna_stance: String,
    generator_fauna_archetype: String,
    generator_fauna_anchor_offset_u: i32,
    generator_fauna_anchor_offset_v: i32,
    generator_fauna_body_yaw: f32,
    generator_fauna_body_arch: f32,
    generator_fauna_spine_segments: i32,
    generator_fauna_body_length: i32,
    generator_fauna_body_half_width: i32,
    generator_fauna_body_half_height: i32,
    generator_fauna_neck_length: i32,
    generator_fauna_neck_half_width: i32,
    generator_fauna_neck_half_height: i32,
    generator_fauna_head_length: i32,
    generator_fauna_head_half_width: i32,
    generator_fauna_head_half_height: i32,
    generator_fauna_tail_length: i32,
    generator_fauna_shoulder_offset_forward: i32,
    generator_fauna_hip_offset_forward: i32,
    generator_fauna_front_upper_length: i32,
    generator_fauna_front_lower_length: i32,
    generator_fauna_hind_upper_length: i32,
    generator_fauna_hind_lower_length: i32,
    generator_fauna_auto_foot_placement: bool,
    // Piscina
    generator_piscina_seed: i32,
    generator_piscina_species: String,
    generator_piscina_length: i32,
    generator_piscina_width: i32,
    generator_piscina_thickness: i32,
    generator_piscina_spine_bend: f32,
    generator_piscina_spine_s_curve: f32,
    generator_piscina_fin_dorsal: i32,
    generator_piscina_fin_anal: i32,
    generator_piscina_fin_caudal: i32,
    generator_piscina_fin_pectoral: i32,
    generator_piscina_fin_pelvic: i32,
    generator_piscina_fin_adipose: i32,
    generator_piscina_show_fin_dorsal: bool,
    generator_piscina_show_fin_anal: bool,
    generator_piscina_show_fin_caudal: bool,
    generator_piscina_show_fin_pectoral: bool,
    generator_piscina_show_fin_pelvic: bool,
    generator_piscina_show_fin_adipose: bool,
    generator_piscina_anchor_offset_u: i32,
    generator_piscina_anchor_offset_v: i32,
    /// Stamp placement origin X: 0 = min edge, 1 = center, 2 = max edge.
    stamp_origin_x: i32,
    /// Stamp placement origin Z: 0 = min edge, 1 = center, 2 = max edge.
    stamp_origin_z: i32,
}

impl Default for PreviewHoverContext {
    fn default() -> Self {
        Self {
            brush_radius: 0,
            brush_shape: voxel_edit::BrushShape::default(),
            spray_density: 0.0,
            stroke_mode: stroke_modes::DrawStrokeMode::default(),
            plane_axis: stroke_modes::PlaneAxis::default(),
            stroke_aux: stroke_modes::StrokeAux::default(),
            color: 0,
            palette: Vec::new(),
            paint_color_distrib: None,
            material: String::new(),
            match_material: false,
            use_brush_preview: true,
            generator_kind: None,
            generator_rope_first_nx: None,
            generator_rope_first_ny: None,
            generator_rope_sag: 2.5,
            generator_rope_tension: 0.5,
            generator_rope_gravity_direction: "down".into(),
            generator_cloth_pins: Vec::new(),
            generator_cloth_tension: 0.5,
            generator_cloth_gravity_direction: "down".into(),
            generator_cloth_gravity_scale: 1.0,
            generator_cloth_stiffness_scale: 1.0,
            generator_cloth_iterations: 0,
            generator_cloth_constraint_passes: 2,
            generator_rock_size: 4,
            generator_rock_roughness: 0.4,
            generator_rock_seed: 42,
            generator_rock_count: 1,
            generator_rock_cluster_radius: 1,
            generator_rock_sink_direction: 0,
            generator_rock_sink_amount: 0,
            generator_grass_radius: 4,
            generator_grass_density: 0.6,
            generator_grass_max_height: 3,
            generator_grass_seed: 42,
            generator_roof_pins: Vec::new(),
            generator_roof_style: "gable".into(),
            generator_roof_height: 6,
            generator_roof_thickness: 1,
            generator_roof_break_ratio: 0.5,
            generator_roof_wall_height: 3,
            generator_roof_parapet_height: 2,
            generator_roof_salt_skew: 0.0,
            generator_roof_hollow: false,
            generator_ashlar_size: 4,
            generator_ashlar_roughness: 0.3,
            generator_ashlar_seed: 42,
            generator_ashlar_thickness: 3,
            // Flora
            generator_flora_seed: 42,
            generator_flora_height: 10,
            generator_flora_girth: 2,
            generator_flora_wobble: 0.3,
            generator_flora_taper: 0.5,
            generator_flora_stem_count: 1,
            generator_flora_cluster_radius: 0,
            generator_flora_branch_count: 4,
            generator_flora_branch_depth: 2,
            generator_flora_branch_start: 0.3,
            generator_flora_branch_spread: 0.5,
            generator_flora_braid_strands: 0,
            generator_flora_braid_twist: 0.5,
            generator_flora_canopy: 2.0,
            // Insecta
            generator_insecta_species: "beetle".into(),
            generator_insecta_total_length: 12,
            generator_insecta_head_ratio: 1.0,
            generator_insecta_thorax_ratio: 1.0,
            generator_insecta_abdomen_ratio: 2.0,
            generator_insecta_body_half_width: 2,
            generator_insecta_body_half_height: 2,
            generator_insecta_abdomen_taper: 0.5,
            generator_insecta_head_shape: 0,
            generator_insecta_anchor_offset_u: 0,
            generator_insecta_anchor_offset_v: 0,
            generator_insecta_body_yaw: 0.0,
            generator_insecta_body_arch: 0.0,
            generator_insecta_antenna_length: 4,
            generator_insecta_antenna_spread: 0.4,
            generator_insecta_antenna_pitch: 0.3,
            generator_insecta_antenna_root: 1,
            generator_insecta_mandible_length: 2,
            generator_insecta_mandible_spread: 0.3,
            generator_insecta_mandible_forward: 1,
            generator_insecta_wing_shape: 0,
            generator_insecta_show_wing_fore: true,
            generator_insecta_wing_fore_length: 8,
            generator_insecta_wing_fore_width: 4,
            generator_insecta_wing_fore_spread: 0.5,
            generator_insecta_wing_fore_pitch: 0.1,
            generator_insecta_wing_fore_offset: 0,
            generator_insecta_wing_fore_forward_cant: 0.0,
            generator_insecta_show_wing_hind: true,
            generator_insecta_wing_hind_length: 6,
            generator_insecta_wing_hind_width: 4,
            generator_insecta_wing_hind_spread: 0.6,
            generator_insecta_wing_hind_pitch: 0.2,
            generator_insecta_wing_hind_offset: 0,
            // Fauna
            generator_fauna_stance: "quadruped".into(),
            generator_fauna_archetype: "mammal".into(),
            generator_fauna_anchor_offset_u: 0,
            generator_fauna_anchor_offset_v: 0,
            generator_fauna_body_yaw: 0.0,
            generator_fauna_body_arch: 0.0,
            generator_fauna_spine_segments: 5,
            generator_fauna_body_length: 10,
            generator_fauna_body_half_width: 2,
            generator_fauna_body_half_height: 2,
            generator_fauna_neck_length: 3,
            generator_fauna_neck_half_width: 1,
            generator_fauna_neck_half_height: 1,
            generator_fauna_head_length: 3,
            generator_fauna_head_half_width: 2,
            generator_fauna_head_half_height: 2,
            generator_fauna_tail_length: 4,
            generator_fauna_shoulder_offset_forward: 3,
            generator_fauna_hip_offset_forward: -3,
            generator_fauna_front_upper_length: 4,
            generator_fauna_front_lower_length: 4,
            generator_fauna_hind_upper_length: 4,
            generator_fauna_hind_lower_length: 4,
            generator_fauna_auto_foot_placement: true,
            // Piscina
            generator_piscina_seed: 42,
            generator_piscina_species: "bass".into(),
            generator_piscina_length: 14,
            generator_piscina_width: 4,
            generator_piscina_thickness: 3,
            generator_piscina_spine_bend: 0.1,
            generator_piscina_spine_s_curve: 0.0,
            generator_piscina_fin_dorsal: 4,
            generator_piscina_fin_anal: 4,
            generator_piscina_fin_caudal: 4,
            generator_piscina_fin_pectoral: 4,
            generator_piscina_fin_pelvic: 4,
            generator_piscina_fin_adipose: 4,
            generator_piscina_show_fin_dorsal: true,
            generator_piscina_show_fin_anal: true,
            generator_piscina_show_fin_caudal: true,
            generator_piscina_show_fin_pectoral: true,
            generator_piscina_show_fin_pelvic: true,
            generator_piscina_show_fin_adipose: false,
            generator_piscina_anchor_offset_u: 0,
            generator_piscina_anchor_offset_v: 0,
            stamp_origin_x: 0,
            stamp_origin_z: 0,
        }
    }
}

/// Debug: last `sync_preview_input` cursor (normalized) and matching texels (same as picking `screen_to_world_ray`).
#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ViewportCursorDebug {
    viewport_width: u32,
    viewport_height: u32,
    /// Swapchain drawable size (physical px); see [`WgpuViewer::surface_pixel_size`].
    surface_width: u32,
    surface_height: u32,
    /// Top-left of the viewport texture in surface pixel space (`copy_texture_to_texture` dest origin).
    viewport_origin_x: u32,
    viewport_origin_y: u32,
    preview_nx: Option<f32>,
    preview_ny: Option<f32>,
    texel_sx: Option<f32>,
    texel_sy: Option<f32>,
    /// [`voxel_edit::screen_to_world_ray`] at `texel_s*` (same as picking).
    ray_origin_x: Option<f32>,
    ray_origin_y: Option<f32>,
    ray_origin_z: Option<f32>,
    ray_dir_x: Option<f32>,
    ray_dir_y: Option<f32>,
    ray_dir_z: Option<f32>,
    /// Preview cube world center projected back to normalized viewport coords.
    proj_cube_nx: Option<f32>,
    proj_cube_ny: Option<f32>,
    /// Same projection path as `proj_cube_*`, but voxel **center** in world space (matches hover mesh).
    proj_center_nx: Option<f32>,
    proj_center_ny: Option<f32>,
}

/// #region agent log
fn debug_agent_ndjson_log(payload: serde_json::Value) {
    const PATH: &str = "/Users/zelda/Documents/digital-garden/.cursor/debug-0e537f.log";
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(PATH)
    {
        if let Ok(s) = serde_json::to_string(&payload) {
            let _ = writeln!(f, "{}", s);
        }
    }
}
/// #endregion

#[tauri::command]
fn get_viewport_cursor_debug(
    state: State<'_, Arc<ViewerState>>,
) -> Result<ViewportCursorDebug, String> {
    let cam = state.camera.lock();
    let (vw, vh, wf, hf, viewport_x, viewport_y, surface_w, surface_h) = {
        let v = state.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (vw, vh) = viewer.viewport_size();
        let (sw, sh) = viewer.surface_pixel_size();
        (
            vw,
            vh,
            vw as f32,
            vh as f32,
            viewer.viewport_x,
            viewer.viewport_y,
            sw,
            sh,
        )
    };
    let pc = state.preview_cursor.lock();
    let (
        preview_nx,
        preview_ny,
        texel_sx,
        texel_sy,
        ray_origin_x,
        ray_origin_y,
        ray_origin_z,
        ray_dir_x,
        ray_dir_y,
        ray_dir_z,
    ) = match *pc {
        Some((nx, ny)) => {
            let (sx, sy) = viewport_texels_from_norm(nx, ny, wf, hf);
            let (o, d) = voxel_edit::screen_to_world_ray(&cam, wf, hf, sx, sy);
            (
                Some(nx),
                Some(ny),
                Some(sx),
                Some(sy),
                Some(o.x),
                Some(o.y),
                Some(o.z),
                Some(d.x),
                Some(d.y),
                Some(d.z),
            )
        }
        None => (None, None, None, None, None, None, None, None, None, None),
    };
    // #region agent log
    debug_agent_ndjson_log(serde_json::json!({
        "sessionId": "0e537f",
        "hypothesisId": "H_rust_surface",
        "location": "lib.rs:get_viewport_cursor_debug",
        "message": "gpu viewport + texels",
        "data": {
            "viewportWidth": vw,
            "viewportHeight": vh,
            "viewportX": viewport_x,
            "viewportY": viewport_y,
            "surfaceW": surface_w,
            "surfaceH": surface_h,
            "previewNx": preview_nx,
            "previewNy": preview_ny,
            "texelSx": texel_sx,
            "texelSy": texel_sy,
            "aspectWh": (vw as f64 / vh.max(1) as f64),
        },
        "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0),
    }));
    // #endregion
    let (proj_cube_nx, proj_cube_ny, proj_center_nx, proj_center_ny) = match (texel_sx, texel_sy) {
        (Some(sx), Some(sy)) => {
            let file_guard = state.current_file.lock();
            let vmap_guard = state.voxel_map.lock();
            match (file_guard.as_ref(), vmap_guard.as_ref()) {
                (Some(file), Some(vmap)) if !file.voxels.is_empty() => {
                    let grid_size = voxel_edit::effective_ray_grid_size(file);
                    let (o, d) = voxel_edit::screen_to_world_ray(&cam, wf, hf, sx, sy);
                    match voxel_edit::ray_first_solid_scene(o, d, file, vmap, grid_size) {
                        Some(((cx, cy, cz), _prev, oid)) => {
                            let m = object_world_matrix(&file.objects, oid);
                            let wp_hit =
                                voxel_edit::world_ray_entry_on_voxel_cell(o, d, cx, cy, cz, m)
                                    .unwrap_or_else(|| {
                                        m.transform_point3(glam::Vec3::new(
                                            cx as f32, cy as f32, cz as f32,
                                        ))
                                    });
                            let wc = m
                                .transform_point3(glam::Vec3::new(cx as f32, cy as f32, cz as f32));
                            let denom_x = (wf - 1.0).max(1.0);
                            let denom_y = (hf - 1.0).max(1.0);
                            let hit_norm = voxel_edit::world_to_viewport_pixels(
                                &cam, wf, hf, wp_hit.x, wp_hit.y, wp_hit.z,
                            )
                            .map(|(px, py)| (px / denom_x, py / denom_y));
                            let center_norm = voxel_edit::world_to_viewport_pixels(
                                &cam, wf, hf, wc.x, wc.y, wc.z,
                            )
                            .map(|(px, py)| (px / denom_x, py / denom_y));
                            // #region agent log
                            if let (Some((hnx, hny)), Some((cnx, cny))) = (hit_norm, center_norm) {
                                debug_agent_ndjson_log(serde_json::json!({
                                        "sessionId": "0e537f",
                                        "runId": "post-fix",
                                        "hypothesisId": "H1_center_vs_hit",
                                        "location": "lib.rs:get_viewport_cursor_debug",
                                        "message": "proj hit vs voxel center (hover mesh uses center)",
                                    "data": {
                                        "cx": cx, "cy": cy, "cz": cz,
                                        "projHitNx": hnx, "projHitNy": hny,
                                        "projCenterNx": cnx, "projCenterNy": cny,
                                        "deltaCenterMinusHitNx": cnx - hnx,
                                        "deltaCenterMinusHitNy": cny - hny,
                                        "previewNx": preview_nx,
                                        "previewNy": preview_ny,
                                    },
                                    "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0),
                                }));
                            }
                            // #endregion
                            match (hit_norm, center_norm) {
                                (Some((hnx, hny)), Some((cnx, cny))) => {
                                    (Some(hnx), Some(hny), Some(cnx), Some(cny))
                                }
                                (Some((hnx, hny)), _) => (Some(hnx), Some(hny), None, None),
                                _ => (None, None, None, None),
                            }
                        }
                        None => (None, None, None, None),
                    }
                }
                _ => (None, None, None, None),
            }
        }
        _ => (None, None, None, None),
    };
    Ok(ViewportCursorDebug {
        viewport_width: vw,
        viewport_height: vh,
        surface_width: surface_w,
        surface_height: surface_h,
        viewport_origin_x: viewport_x,
        viewport_origin_y: viewport_y,
        preview_nx,
        preview_ny,
        texel_sx,
        texel_sy,
        ray_origin_x,
        ray_origin_y,
        ray_origin_z,
        ray_dir_x,
        ray_dir_y,
        ray_dir_z,
        proj_cube_nx,
        proj_cube_ny,
        proj_center_nx,
        proj_center_ny,
    })
}

// Remaining commands extracted into commands/edit.rs, commands/avatar.rs, commands/collab.rs.

/// stderr logger: debug builds default to `warn` + `voxelle_load=info`. Override with `RUST_LOG`, e.g. `RUST_LOG=voxelle_load=debug`.
fn init_load_logging() {
    let default_filter = if cfg!(debug_assertions) {
        "warn,voxelle_load=info,cosmic_text::font::system=error"
    } else {
        "warn"
    };
    let _ =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_filter))
            .format_timestamp_millis()
            .try_init();
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    init_load_logging();
    #[cfg(desktop)]
    let headless_server_port: Option<u16> = headless_server::parse_config();
    #[cfg(not(desktop))]
    let headless_server_port: Option<u16> = None;

    let viewer_state = Arc::new(ViewerState {
        viewer: Mutex::new(None),
        camera: Mutex::new(OrbitCamera::new()),
        file_label: Mutex::new(String::new()),
        current_file: Mutex::new(None),
        voxel_map: Mutex::new(None),
        preview_cursor: Mutex::new(None),
        camera_dragging: AtomicBool::new(false),
        preview_mode: Mutex::new(PreviewMode::Navigate),
        preview_hover: Mutex::new(PreviewHoverContext::default()),
        rendering_mode: Mutex::new(RenderingMode::Greedy),
        fps: Mutex::new(FpsCounter {
            period_start: None,
            accum_frames: 0,
            last_fps: 0,
        }),
        last_edit_perf: Mutex::new(None),
        last_scene_bounds: Mutex::new(None),
        mesh_refresh_generation: AtomicU64::new(0),
        load_generation: AtomicU64::new(0),
        chunk_mesh_inbox: Mutex::new(VecDeque::new()),
        collab_edit_inbox: Mutex::new(VecDeque::new()),
        deferred_spatial_cache: Mutex::new(None),
        voxel_edit_stats_cache: Mutex::new(None),
        solo_undo: Mutex::new(Vec::new()),
        solo_redo: Mutex::new(Vec::new()),
        stroke_active: Mutex::new(false),
        stroke_buffer: Mutex::new(Vec::new()),
        stroke_preview_union: Mutex::new(AHashSet::new()),
        stroke_preview_last_args: Mutex::new(None),
        stroke_preview_suppresses_hover: AtomicBool::new(false),
        sculpt_stroke_replay: Mutex::new(Vec::new()),
        extrude_ray_spine: Mutex::new(None),
        collab: Arc::new(Mutex::new(collab::CollabRuntime::default())),
        local_avatar_data: Mutex::new(HashMap::new()),
        smooth_presence: Mutex::new(HashMap::new()),
        ping_flash: Mutex::new(None),
        autosave_interval_secs: Mutex::new(120),
        last_autosave: Mutex::new(None),
        autosave_enabled: Mutex::new(true),
        autosave_keep_count: Mutex::new(5),
        autosave_slot: Mutex::new(HashMap::new()),
        active_project: AtomicBool::new(false),
        fly_mode: Mutex::new(false),
        fly_input: Mutex::new(FlyInputState::default()),
        fly_last_physics: Mutex::new(None),
        walk_mode: Mutex::new(false),
        walk_physics: Mutex::new(camera::WalkPhysicsState::default()),
        walk_last_physics: Mutex::new(None),
        selection_cells: Mutex::new(AHashSet::new()),
        selection_stroke_before: Mutex::new(None),
        selection_stroke_accum: Mutex::new(None),
        selection_combine_mode: Mutex::new(SelectionCombineMode::Replace),
        selection_match_material: Mutex::new(false),
        stamp_clipboard: Mutex::new(None),
        squishy_session: Mutex::new(generators::SquishySession::new()),
        squishy_gizmo_drag: Mutex::new(None),
        selection_gizmo_drag: Mutex::new(SelectionGizmoDrag::None),
        extrude_gizmo_drag: Mutex::new(ExtrudeGizmoDrag::None),
        extrude_gizmo_base_depth: Mutex::new(0),
        hovered_extrude_axis: AtomicU8::new(255),
        start_screen_logo_transparent: std::sync::atomic::AtomicBool::new(true),
        start_screen_light: std::sync::atomic::AtomicBool::new(false),
        overlay_mesh_generation: AtomicU64::new(0),
        viewport_cursor_debug_overlay: AtomicBool::new(false),
        show_grid_borders: AtomicBool::new(false),
        hovered_gizmo_axis: AtomicU8::new(255),
        grid_overlay_cache_key: Mutex::new(None),
        selection_overlay_cache_key: Mutex::new(None),
        preview_overlay_cache_key: Mutex::new(None),
        generator_preview_locked_camera: Mutex::new(None),
        fill_operation_cancel: Arc::new(AtomicBool::new(false)),
        spray_constraint_plane: Mutex::new(None),
        wall_stroke_face_snapped: Mutex::new(None),
        terrain_accum: Mutex::new(AHashMap::new()),
    });
    let vs = viewer_state.clone();

    #[cfg(all(desktop, unix))]
    {
        let st = viewer_state.clone();
        let _ = std::thread::Builder::new()
            .name("voxelle-sigusr1-perf".into())
            .spawn(move || {
                use signal_hook::consts::SIGUSR1;
                use signal_hook::iterator::Signals;
                let Ok(mut signals) = Signals::new([SIGUSR1]) else {
                    return;
                };
                for _ in signals.forever() {
                    let text = performance_report_text(st.as_ref());
                    eprintln!(
                        "--- voxelle SIGUSR1 performance dump (paste for bugs) ---\n{text}\n--- end dump ---"
                    );
                }
            });
    }

    let mut builder = tauri::Builder::default();
    #[cfg(desktop)]
    {
        builder = builder
            .plugin(tauri_plugin_process::init())
            .plugin(tauri_plugin_updater::Builder::new().build());
    }
    builder
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(viewer_state.clone())
        .on_menu_event(|app, event| {
            if event.id() == "open_voxelle" {
                let state = app.state::<Arc<ViewerState>>();
                open_voxelle_file_dialog(app.clone(), state.inner().clone());
            } else if event.id() == "new_project" {
                let _ = app.emit("voxelle-open-new-project", ());
            } else if event.id() == "menu_undo" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = voxel_undo(state, app.clone());
            } else if event.id() == "menu_redo" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = voxel_redo(state, app.clone());
            } else if event.id() == "menu_save" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                if save_voxelle(app.clone(), state).is_err() {
                    let state: State<'_, Arc<ViewerState>> = app.state();
                    let _ = save_voxelle_as(state, app.clone());
                }
            } else if event.id() == "menu_save_as" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = save_voxelle_as(state, app.clone());
            } else if event.id() == "menu_close_project" {
                let state = app.state::<Arc<ViewerState>>();
                close_project_dialog(app.clone(), state.inner().clone());
            } else if event.id() == "menu_export_glb" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = export_mesh_glb(state, app.clone());
            } else if event.id() == "menu_collab_start" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-collab-start-session",
                    (),
                );
            } else if event.id() == "menu_collab_join" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-collab-join-session",
                    (),
                );
            } else if event.id() == "menu_collab_leave" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-collab-leave-session",
                    (),
                );
            } else if event.id() == "menu_chat_panel" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-show-chat-panel",
                    true,
                );
            } else if event.id() == "menu_check_updates" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-check-updates",
                    (),
                );
            } else if event.id() == "menu_preferences" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-open-preferences",
                    (),
                );
            } else if event.id() == "debug_copy_performance" {
                let state = app.state::<Arc<ViewerState>>();
                eprintln!(
                    "--- Debug → Copy performance info (terminal backup) ---\n{}\n--- end ---",
                    performance_report_text(state.inner())
                );
                if let Err(e) = copy_performance_data_to_clipboard(state.inner()) {
                    eprintln!("copy performance data: {e}");
                }
            } else if event.id() == "debug_raytrace_benchmark" {
                let state = app.state::<Arc<ViewerState>>();
                let result = state.viewer.lock().as_mut().map(|viewer| viewer.run_raytrace_benchmark(50));
                if let Some(result) = result {
                    eprintln!(
                        "[raytrace bench] {}×{}  {} frames  avg {:.1} ms  σ {:.1}  p50 {:.1}  p95 {:.1}  p99 {:.1}  max {:.1}  {:.1} Mpix/s",
                        result.viewport_width, result.viewport_height,
                        result.frame_count,
                        result.avg_ms, result.stddev_ms,
                        result.p50_ms, result.p95_ms, result.p99_ms, result.max_ms,
                        result.mpix_per_sec,
                    );
                    let _ = app.emit_to(
                        EventTarget::webview_window("main"),
                        "voxelle-debug-raytrace-benchmark",
                        &result,
                    );
                }
            } else if event.id() == "debug_viewport_cursor_overlay" {
                #[cfg(desktop)]
                if let Some(sel) = app.try_state::<SelectionMenuState>() {
                    if let Ok(enabled) = sel.viewport_cursor_debug.is_checked() {
                        let state = app.state::<Arc<ViewerState>>();
                        state
                            .viewport_cursor_debug_overlay
                            .store(enabled, Ordering::Relaxed);
                        let _ = app.emit_to(
                            EventTarget::webview_window("main"),
                            "voxelle-debug-viewport-cursor-overlay",
                            enabled,
                        );
                    }
                }
            } else if event.id() == "debug_logo_light_controls" {
                #[cfg(desktop)]
                if let Some(sel) = app.try_state::<SelectionMenuState>() {
                    if let Ok(enabled) = sel.logo_light_controls.is_checked() {
                        let _ = app.emit_to(
                            EventTarget::webview_window("main"),
                            "voxelle-debug-logo-light-controls",
                            enabled,
                        );
                    }
                }
            } else if event.id() == "debug_clear_autosaves" {
                let _ = clear_autosaves_and_session(&app);
            } else if event.id() == "debug_test_crash" {
                panic!("Test crash triggered from Debug menu");
            } else if event.id() == "view_render_greedy"
                || event.id() == "view_render_marching"
                || event.id() == "view_render_dual"
                || event.id() == "menu_view_render_ray"
            {
                let (mode, label) = match event.id().0.as_ref() {
                    "view_render_greedy" => (RenderingMode::Greedy, "greedy"),
                    "view_render_marching" => (RenderingMode::MarchingCubes, "marchingCubes"),
                    "view_render_dual" => (RenderingMode::DualContour, "dualContour"),
                    _ => (RenderingMode::Ray, "ray"),
                };
                let state = app.state::<Arc<ViewerState>>();
                let _ = apply_rendering_mode(&state, &app, mode);
                wake_viewport_loop(&app);
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-rendering-mode-changed",
                    label,
                );
                // Enforce radio-button style: exactly one checked at a time.
                #[cfg(desktop)]
                if let Some(sel) = app.try_state::<SelectionMenuState>() {
                    let _ = sel.render_greedy.set_checked(matches!(mode, RenderingMode::Greedy));
                    let _ = sel.render_marching.set_checked(matches!(mode, RenderingMode::MarchingCubes));
                    let _ = sel.render_dual.set_checked(matches!(mode, RenderingMode::DualContour));
                    let _ = sel.render_ray.set_checked(matches!(mode, RenderingMode::Ray));
                }
            } else if event.id() == "menu_view_ortho" {
                #[cfg(desktop)]
                if let Some(sel) = app.try_state::<SelectionMenuState>() {
                    if let Ok(checked) = sel.ortho_toggle.is_checked() {
                        let state = app.state::<Arc<ViewerState>>();
                        let _ = apply_orthographic(&state, checked);
                        wake_viewport_loop(&app);
                    }
                }
            } else if event.id() == "menu_view_show_borders" {
                #[cfg(desktop)]
                if let Some(sel) = app.try_state::<SelectionMenuState>() {
                    if let Ok(checked) = sel.view_show_borders.is_checked() {
                        let state = app.state::<Arc<ViewerState>>();
                        state
                            .show_grid_borders
                            .store(checked, Ordering::Relaxed);
                        let _ = app.emit_to(
                            EventTarget::webview_window("main"),
                            "voxelle-show-grid-borders",
                            checked,
                        );
                        wake_viewport_loop(&app);
                    }
                }
            } else if event.id() == "menu_view_hide_ui" {
                #[cfg(desktop)]
                if let Some(sel) = app.try_state::<SelectionMenuState>() {
                    if let Ok(checked) = sel.view_hide_ui.is_checked() {
                        let _ = app.emit_to(
                            EventTarget::webview_window("main"),
                            "voxelle-hide-ui",
                            checked,
                        );
                    }
                }
            } else if event.id() == "menu_view_stamp_book" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-menu-stamp-book",
                    (),
                );
            } else if event.id() == "menu_voxel_mirror_x" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_mirror(state, app.clone(), 0);
            } else if event.id() == "menu_voxel_mirror_y" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_mirror(state, app.clone(), 1);
            } else if event.id() == "menu_voxel_mirror_z" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_mirror(state, app.clone(), 2);
            } else if event.id() == "menu_voxel_rotate" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-menu-rotate-selection",
                    (),
                );
            } else if event.id() == "menu_voxel_scale" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-menu-scale-selection",
                    (),
                );
            } else if event.id() == "menu_voxel_hide_selected"
                || event.id() == "menu_voxel_unhide_all"
                || event.id() == "menu_voxel_hollow"
            {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-menu-not-implemented",
                    "This voxel transform is not wired up in the desktop build yet.",
                );
            } else if event.id() == "menu_sel_all" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_select_all(app.clone(), state);
            } else if event.id() == "menu_sel_connected" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_add_connected_at_cursor(app.clone(), state);
            } else if event.id() == "menu_sel_grow" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_grow(app.clone(), state);
            } else if event.id() == "menu_sel_shrink" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_shrink(app.clone(), state);
            } else if event.id() == "menu_sel_invert" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_invert(app.clone(), state);
            } else if event.id() == "menu_sel_deselect_all" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_clear(app.clone(), state);
            } else if event.id() == "menu_sel_deselect_inner" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_deselect_inner_voxels(app.clone(), state);
            } else if event.id() == "menu_sel_deselect_voxels" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_retain_empty_only(app.clone(), state);
            } else if event.id() == "menu_sel_deselect_empty" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_retain_solid_only(app.clone(), state);
            } else if event.id() == "menu_sel_mode_replace" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_set_combine_mode(app.clone(), state, SelectionCombineMode::Replace);
            } else if event.id() == "menu_sel_mode_add" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_set_combine_mode(app.clone(), state, SelectionCombineMode::Add);
            } else if event.id() == "menu_sel_mode_subtract" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_set_combine_mode(app.clone(), state, SelectionCombineMode::Subtract);
            } else if event.id() == "menu_sel_mode_intersect" {
                let state: State<'_, Arc<ViewerState>> = app.state();
                let _ = selection_set_combine_mode(
                    app.clone(),
                    state,
                    SelectionCombineMode::Intersect,
                );
            } else if event.id() == "menu_sel_by_color" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-menu-selection-mode",
                    "selectByColor",
                );
            } else if event.id() == "menu_sel_coplanar" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-menu-selection-mode",
                    "selectCoplanar",
                );
            } else if event.id() == "menu_sel_coplanar_empty" {
                let _ = app.emit_to(
                    EventTarget::webview_window("main"),
                    "voxelle-menu-selection-mode",
                    "selectCoplanarEmpty",
                );
            } else if event.id() == "menu_sel_match_material" {
                #[cfg(desktop)]
                if let Some(sel) = app.try_state::<SelectionMenuState>() {
                    if let Ok(checked) = sel.match_material.is_checked() {
                        let state = app.state::<Arc<ViewerState>>();
                        *state.selection_match_material.lock() = checked;
                        let _ = app.emit_to(
                            EventTarget::webview_window("main"),
                            "voxelle-menu-match-material",
                            checked,
                        );
                    }
                }
            } else if event.id() == "recent_clear" {
                clear_recent_files(app);
                #[cfg(desktop)]
                if let Some(rm) = app.try_state::<RecentMenuState>() {
                    rebuild_recent_submenu(app, &rm.submenu);
                }
            } else if event.id().0.starts_with("recent_file_") {
                let id_str = event.id().0.to_string();
                if let Some(idx_str) = id_str.strip_prefix("recent_file_") {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        let recent = read_recent_files(app);
                        if let Some(path_str) = recent.get(idx) {
                            let path = PathBuf::from(path_str);
                            if path.exists() {
                                let state = app.state::<Arc<ViewerState>>();
                                let label = path.to_string_lossy().to_string();
                                *state.file_label.lock() = label.clone();
                                let _ = app.emit("voxelle-load-start", label);
                                spawn_decode_and_mesh(
                                    state.inner().clone(),
                                    app.clone(),
                                    path,
                                );
                            } else {
                                let _ = app.emit(
                                    "voxelle-load-error",
                                    format!("File not found: {path_str}"),
                                );
                            }
                        }
                    }
                }
            }
        })
        .setup(move |app| {
            #[cfg(desktop)]
            {
                let (selection_menu_state, recent_menu_state) = install_app_menu(app.handle())?;
                app.manage(selection_menu_state);
                app.manage(recent_menu_state);
                let (has_voxels, has_selection) = scene_menu_flags(vs.as_ref());
                selection_menu_sync_enabled_for_scene(app.handle(), has_voxels, has_selection);
            }

            let window = app.get_webview_window("main").expect("main window");
            #[cfg(target_os = "macos")]
            if let Err(e) = macos_titlebar::apply_transparent_titlebar(&window) {
                eprintln!("macos_titlebar: {e}");
            }
            if headless_server_port.is_some() {
                let _ = window.hide();
            }
            let viewer = {
                let w = window.clone();
                tauri::async_runtime::block_on(async move { WgpuViewer::new(w).await })
                    .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?
            };
            // Do not resize to `inner_size()` here: the 3D view matches the `.viewport` div (below
            // toolbar / beside sidebar), not the full window. Wrong dimensions break screen→world
            // raycasts until the frontend sends `viewer_resize`.
            {
                let mut vl = vs.viewer.lock();
                let v = vl.insert(viewer);
                // Pre-cache the default avatar: a single white glowing voxel, tinted per-peer at runtime.
                init_default_avatar_mesh(v);
            }

            #[cfg(desktop)]
            if let Some(port) = headless_server_port {
                let listener = tauri::async_runtime::block_on(tokio::net::TcpListener::bind(
                    ("127.0.0.1", port),
                ))
                .map_err(|e| format!("headless server bind 127.0.0.1:{port}: {e}"))?;
                headless_server::start(listener)?;
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            viewer_resize,
            get_viewport_pixel_size,
            get_viewport_cursor_debug,
            get_surface_pixel_size,
            set_start_screen_light,
            viewport_pointer,
            viewport_wheel,
            get_orbit_gizmo_projection,
            get_camera_zoom_percent,
            camera_fit_to_scene,
            camera_reset_view,
            camera_orbit_gizmo_drag,
            camera_snap_orbit_axis,
            camera_zoom_step,
            open_voxelle_dialog,
            confirm_app_update_dialog,
            load_voxelle_path,
            load_start_screen_logo,
            load_voxelle_recovery,
            get_last_session_info,
            create_new_project,
            voxel_pick_probe,
            voxel_stroke_anchor_coord_at_screen,
            terrain_surface_y_at_screen,
            ping_cursor_pick,
            world_to_viewport_pixels,
            project_world_point,
            collab_peer_labels,
            sync_preview_input,
            lock_generator_preview_camera,
            unlock_generator_preview_camera,
            voxel_stroke_begin,
            voxel_stroke_preview_reset,
            voxel_stroke_preview_at_screen,
            query_cuboid_plane_geometry,
            voxel_stroke_end,
            voxel_pick_color_at_screen,
            voxel_edit_at_screen,
            voxel_fill_cancel,
            voxel_undo,
            voxel_redo,
            save_voxelle,
            save_voxelle_as,
            collab_host_start,
            collab_join,
            collab_cancel_join,
            collab_leave,
            collab_local_peer_id,
            collab_kick_peer,
            collab_update_profile,
            collab_set_can_edit,
            collab_push_camera,
            collab_snap_camera,
            collab_send_chat,
            collab_send_ping,
            get_autosave_settings,
            set_autosave_settings,
            debug_clear_autosaves,
            get_rendering_mode,
            set_rendering_mode,
            set_raytrace_mode,
            benchmark_raytrace,
            get_orthographic,
            set_orthographic,
            get_show_grid_borders,
            view_menu_sync_show_borders,
            view_menu_sync_hide_ui,
            selection_menu_sync_match_material,
            debug_menu_sync_viewport_cursor_overlay,
            set_soft_shadows,
            set_soft_sunshafts,
            set_emission_lighting,
            set_tone_mapping,
            is_hdr_available,
            set_hdr_output,
            set_mood_params,
            set_scene_lighting,
            get_scene_lighting,
            set_focal_length_mm,
            get_focal_length_mm,
            set_fly_mode,
            get_fly_mode,
            set_walk_mode,
            sync_fly_input,
            camera_fly_look,
            selection_toggle_at_screen,
            get_selection_gizmo_projected,
            gizmo_pointer_down,
            gizmo_pointer_move,
            gizmo_pointer_up,
            gizmo_hit_test,
            extrude_gizmo_pointer_down,
            extrude_gizmo_pointer_move,
            extrude_gizmo_pointer_up,
            extrude_gizmo_hit_test,
            set_gizmo_on_top,
            selection_translate,
            selection_rotate,
            selection_scale,
            selection_mirror,
            selection_clear,
            selection_delete_selected_voxels,
            selection_get_count,
            paint_selection,
            selection_add_by_color_at_screen,
            selection_add_coplanar_at_screen,
            selection_add_coplanar_empty_at_screen,
            selection_select_all,
            selection_invert,
            selection_grow,
            selection_shrink,
            selection_deselect_inner_voxels,
            selection_retain_empty_only,
            selection_retain_solid_only,
            selection_add_connected_at_screen,
            selection_add_connected_at_cursor,
            selection_set_combine_mode,
            get_selection_combine_mode,
            selection_stroke_begin,
            selection_stroke_end,
            selection_stroke_at_screen,
            voxel_fill_at_screen,
            clipboard_copy_selection,
            clipboard_stamp_at_screen,
            clipboard_punch_at_screen,
            stamp_face_normal_at_screen,
            get_selection_as_stamp_entries,
            stamp_book_load_entries,
            voxel_sculpt_raise_at_screen,
            voxel_sculpt_stroke_at_screen,
            voxel_sculpt_stroke_preview_at_screen,
            extrude_ray_preview,
            selection_extrude_preview,
            extrude_recompute_preview,
            generator_rocks_at_screen,
            generator_grass_at_screen,
            generator_rope_at_screen,
            generator_cloth_from_pins_cmd,
            generator_ashlar_at_screen,
            generator_flora_at_screen,
            generator_roof_from_pins_cmd,
            generator_piscina_at_screen,
            generator_insecta_at_screen,
            generator_fauna_at_screen,
            generator_squishy_metaball_at_screen,
            squishy_session_get,
            squishy_session_set_mode,
            squishy_session_set_flags,
            squishy_metaball_add_at_screen,
            squishy_metaball_remove,
            squishy_metaball_select,
            squishy_session_clear,
            squishy_session_commit,
            squishy_pick_at_screen,
            squishy_gizmo_pointer_down,
            squishy_gizmo_pointer_move,
            squishy_gizmo_pointer_up,
            export_mesh_glb,
            get_scene_objects,
            set_active_object,
            set_object_visible,
            create_scene_object,
            mascot_load,
            mascot_load_embedded,
            mascot_set_screen_rect,
            mascot_set_visible,
            logo_set_light_dir,
            logo_set_camera_angle,
            logo_set_camera_dist,
            avatar_list_embedded,
            avatar_list_user,
            avatar_open_user_folder,
            set_local_avatar,
            avatar_load_file,
            speech_bubble_show,
            speech_bubble_click,
            speech_bubble_dismiss,
            speech_bubble_reposition,
        ])
        .build(tauri::generate_context!())
        .expect("error building app")
        .run(move |app, event| {
            if let RunEvent::MainEventsCleared = event {
                let app_wake = app.clone();
                let state = app.state::<Arc<ViewerState>>();
                {
                    let mut cam = state.camera.lock();
                    cam.update_damping();
                }
                // Fly WASD: integrate here with wall-clock dt between native iterations (not webview RAF).
                if *state.fly_mode.lock() {
                    let now = Instant::now();
                    let dt = {
                        let mut last = state.fly_last_physics.lock();
                        match *last {
                            None => {
                                *last = Some(now);
                                0.0
                            }
                            Some(t) => {
                                let d = (now - t).as_secs_f32();
                                *last = Some(now);
                                d.max(0.0)
                            }
                        }
                    };
                    let input = *state.fly_input.lock();
                    let scale = if input.speed_scale.is_finite() {
                        input.speed_scale.clamp(0.0, 1e6)
                    } else {
                        1.0
                    };
                    if dt > 0.0
                        && (input.forward != 0.0
                            || input.right != 0.0
                            || input.up != 0.0)
                    {
                        const SPEED: f32 = 26.0;
                        let mut cam = state.camera.lock();
                        cam.fly_move(
                            input.forward,
                            input.right,
                            input.up,
                            dt,
                            SPEED * scale,
                        );
                    }
                }
                // Walk mode physics: gravity, collision, jumping.
                if *state.walk_mode.lock() {
                    let now = Instant::now();
                    let dt = {
                        let mut last = state.walk_last_physics.lock();
                        match *last {
                            None => {
                                *last = Some(now);
                                0.0
                            }
                            Some(t) => {
                                let d = (now - t).as_secs_f32();
                                *last = Some(now);
                                d.clamp(0.0, 0.05)
                            }
                        }
                    };
                    if dt > 0.0 {
                        let input = *state.fly_input.lock();
                        let scale = if input.speed_scale.is_finite() {
                            input.speed_scale.clamp(0.0, 1e6)
                        } else {
                            1.0
                        };

                        let mut wp = state.walk_physics.lock();
                        let h_delta = {
                            let cam = state.camera.lock();
                            cam.walk_horizontal_delta(
                                input.forward,
                                input.right,
                                dt,
                                camera::WALK_MOVE_SPEED * scale,
                            )
                        };

                        // Gravity
                        if !wp.on_ground {
                            wp.vel_y += camera::WALK_GRAVITY * dt;
                        }

                        // Jump
                        if input.jump && wp.on_ground {
                            wp.vel_y = camera::WALK_JUMP_VEL;
                            wp.on_ground = false;
                        }

                        // Candidate position
                        let mut new_feet = wp.feet_pos + h_delta + glam::Vec3::Y * (wp.vel_y * dt);

                        // Collision against voxel_map
                        {
                            let vm_guard = state.voxel_map.lock();
                            if let Some(ref vm) = *vm_guard {
                                new_feet = resolve_walk_collision(wp.feet_pos, new_feet, vm, &mut wp);
                            }
                        }

                        // Void floor safety
                        if new_feet.y < -100.0 {
                            new_feet.y = -100.0;
                            wp.vel_y = 0.0;
                            wp.on_ground = true;
                        }

                        wp.feet_pos = new_feet;
                        let mut cam = state.camera.lock();
                        cam.walk_set_eye_from_feet(new_feet, camera::WALK_EYE_HEIGHT);
                    }
                }
                // Prepare overlays without holding the viewer mutex so `current_file` can be locked
                // while IPC may be waiting on `viewer` + `camera` (see `finish_voxel_edit_gpu_deltas`).
                let frame_prep = {
                    let wh = {
                        let v = state.viewer.lock();
                        v.as_ref().map(|viewer| viewer.viewport_size())
                    };
                    match wh {
                        Some((viewport_w, viewport_h)) => {
                            let cam_snap = state.camera.lock().clone();
                            let grid_p = prepare_grid_border_overlay(Arc::as_ref(&state));
                            let sel_p = prepare_selection_overlay(Arc::as_ref(&state));
                            let prev_p = prepare_preview_mesh(
                                Arc::as_ref(&state),
                                &cam_snap,
                                viewport_w,
                                viewport_h,
                            );
                            Some((grid_p, sel_p, prev_p))
                        }
                        None => None,
                    }
                };
                // Drain collab edit inbox — apply queued guest edits/undo/redo
                // on the main thread, before we hold the viewer lock for the frame.
                {
                    let items: Vec<collab::CollabInboxItem> =
                        state.collab_edit_inbox.lock().drain(..).collect();
                    collab::process_inbox_items_batched(app, &state, &state.collab, items);
                }
                let mut v = state.viewer.lock();
                if let Some(viewer) = v.as_mut() {
                    let cam = state.camera.lock();
                    viewer.update_uniforms(&cam);
                    if let Some((grid_p, sel_p, prev_p)) = frame_prep {
                        apply_grid_border_overlay(viewer, Arc::as_ref(&state), grid_p);
                        apply_selection_overlay(viewer, Arc::as_ref(&state), sel_p);
                        apply_preview_mesh(viewer, Arc::as_ref(&state), prev_p);
                    }
                    sync_collab_peer_avatars(viewer, &state, &cam);
                    sync_collab_peer_labels(viewer, Arc::as_ref(&state), &cam);
                    sync_ping_flash(viewer, Arc::as_ref(&state), &cam);
                    sync_gizmo_gpu(viewer, Arc::as_ref(&state), &cam);
                    let transparent = state
                        .start_screen_logo_transparent
                        .load(Ordering::Relaxed);
                    viewer.set_start_screen_transparent(transparent);
                    let start_light = state.start_screen_light.load(Ordering::Relaxed);
                    viewer.set_start_screen_appearance(if start_light {
                        1.0
                    } else {
                        0.0
                    });
                    // Progressive loading: move chunks from background-thread inbox to viewer queue.
                    {
                        let mut inbox = state.chunk_mesh_inbox.lock();
                        if !inbox.is_empty() {
                            viewer.enqueue_chunk_uploads(&mut inbox);
                        }
                    }
                    // Drip-feed queued mesh chunks to GPU each frame.
                    if viewer.has_pending_chunk_uploads() {
                        viewer.drain_pending_chunk_uploads(std::time::Duration::from_millis(4));
                    }
                    // Once all chunks are uploaded, apply deferred spatial cache for editing.
                    if !viewer.has_pending_chunk_uploads() && !viewer.has_spatial_mesh_cache() {
                        let mut deferred = state.deferred_spatial_cache.lock();
                        if let Some(cache) = deferred.take() {
                            viewer.set_spatial_mesh_cache(cache);
                        }
                    }
                    let sz_before = viewer.surface_size;
                    let _ = viewer.render();
                    let (vw, vh) = viewer.viewport_size();
                    if viewer.surface_size != sz_before {
                        let (sur_w, sur_h) = viewer.surface_pixel_size();
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
                    sample_fps_and_emit(app, &state.fps);

                    // Drain speech bubbles that completed their shake-dismiss animation.
                    // app.emit is non-blocking; holding viewer lock here is safe.
                    for id in viewer.pending_dismissed_bubble_ids.drain(..) {
                        let _ = app.emit("speech-bubble-dismissed", id);
                    }

                    let enabled = *state.autosave_enabled.lock();
                    let interval = *state.autosave_interval_secs.lock();
                    let (collab_on, is_host) = {
                        let c = state.collab.lock();
                        (c.is_active(), c.is_host())
                    };
                    if enabled
                        && interval > 0
                        && (!collab_on || is_host)
                        && state.active_project.load(Ordering::Relaxed)
                    {
                        let label = state.file_label.lock().clone();
                        if !label.is_empty() {
                            if let Ok(doc) = autosave_document_path_for_label(&app, &label) {
                                let now = Instant::now();
                                let last = state.last_autosave.lock();
                                let do_save = last
                                    .map(|t| now.duration_since(t).as_secs() >= interval)
                                    .unwrap_or(true);
                                if do_save {
                                    drop(last);
                                    if let Ok(dest) =
                                        next_rotating_autosave_path(&app, Arc::as_ref(&state), &doc)
                                    {
                                        if write_voxelle_file_to_path(None, &state, &dest).is_ok() {
                                            *state.last_autosave.lock() = Some(now);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // While orbit damping runs, no pointer IPC wakes the Wry loop (`ControlFlow::Wait`).
                // Queue a no-op on the main thread from a background context so the proxy wakes
                // another iteration at display rate (see `send_user_message` vs main thread).
                // Fly mode: camera look snaps smooth state so `needs_redraw` is often false; WebKit
                // may throttle RAF; fly movement uses native loop dt, not the webview clock.
                // Keep spinning while fly is on so the viewport and FPS/status UI stay live.
                // Raytrace mode: accumulation requires one sample per frame, so spin continuously.
                let rt_active = v.as_ref().map_or(false, |viewer| viewer.raytrace_enabled);
                let mascots_active = v
                    .as_ref()
                    .map_or(false, |viewer| viewer.any_mascot_visible());
                let logo_active = v
                    .as_ref()
                    .map_or(false, |viewer| viewer.logo_overlay_visible());
                let bubbles_active = v
                    .as_ref()
                    .map_or(false, |viewer| viewer.has_visible_speech_bubbles());
                drop(v);
                let fly_on = *state.fly_mode.lock();
                let walk_on = *state.walk_mode.lock();
                let has_fly_movement = if fly_on || walk_on {
                    let input = *state.fly_input.lock();
                    input.forward != 0.0 || input.right != 0.0 || input.up != 0.0 || input.jump
                } else {
                    false
                };
                // Walk mode always spins (gravity may be in progress even with no input).
                let needs_next = state.camera.lock().needs_redraw()
                    || fly_on
                    || walk_on
                    || has_fly_movement
                    || rt_active
                    || mascots_active
                    || logo_active
                    || bubbles_active;
                if needs_next {
                    tauri::async_runtime::spawn(async move {
                        let _ = app_wake.run_on_main_thread(|| {});
                    });
                }
            }
        });
}

#[cfg(test)]
pub(crate) fn minimal_viewer_state_for_collab_tests() -> Arc<ViewerState> {
    Arc::new(ViewerState {
        viewer: Mutex::new(None),
        camera: Mutex::new(OrbitCamera::new()),
        file_label: Mutex::new(String::new()),
        current_file: Mutex::new(None),
        voxel_map: Mutex::new(None),
        preview_cursor: Mutex::new(None),
        camera_dragging: AtomicBool::new(false),
        preview_mode: Mutex::new(PreviewMode::Navigate),
        preview_hover: Mutex::new(PreviewHoverContext::default()),
        rendering_mode: Mutex::new(RenderingMode::Greedy),
        fps: Mutex::new(FpsCounter {
            period_start: None,
            accum_frames: 0,
            last_fps: 0,
        }),
        last_edit_perf: Mutex::new(None),
        last_scene_bounds: Mutex::new(None),
        mesh_refresh_generation: AtomicU64::new(0),
        load_generation: AtomicU64::new(0),
        chunk_mesh_inbox: Mutex::new(VecDeque::new()),
        collab_edit_inbox: Mutex::new(VecDeque::new()),
        deferred_spatial_cache: Mutex::new(None),
        voxel_edit_stats_cache: Mutex::new(None),
        solo_undo: Mutex::new(Vec::new()),
        solo_redo: Mutex::new(Vec::new()),
        terrain_accum: Mutex::new(AHashMap::new()),
        stroke_active: Mutex::new(false),
        stroke_buffer: Mutex::new(Vec::new()),
        stroke_preview_union: Mutex::new(AHashSet::new()),
        stroke_preview_last_args: Mutex::new(None),
        stroke_preview_suppresses_hover: AtomicBool::new(false),
        sculpt_stroke_replay: Mutex::new(Vec::new()),
        extrude_ray_spine: Mutex::new(None),
        collab: Arc::new(Mutex::new(collab::CollabRuntime::default())),
        local_avatar_data: Mutex::new(HashMap::new()),
        smooth_presence: Mutex::new(HashMap::new()),
        ping_flash: Mutex::new(None),
        autosave_interval_secs: Mutex::new(120),
        last_autosave: Mutex::new(None),
        autosave_enabled: Mutex::new(true),
        autosave_keep_count: Mutex::new(5),
        autosave_slot: Mutex::new(HashMap::new()),
        active_project: AtomicBool::new(false),
        fly_mode: Mutex::new(false),
        fly_input: Mutex::new(FlyInputState::default()),
        fly_last_physics: Mutex::new(None),
        walk_mode: Mutex::new(false),
        walk_physics: Mutex::new(camera::WalkPhysicsState::default()),
        walk_last_physics: Mutex::new(None),
        selection_cells: Mutex::new(AHashSet::new()),
        selection_stroke_before: Mutex::new(None),
        selection_stroke_accum: Mutex::new(None),
        selection_combine_mode: Mutex::new(SelectionCombineMode::Replace),
        selection_match_material: Mutex::new(false),
        stamp_clipboard: Mutex::new(None),
        squishy_session: Mutex::new(generators::SquishySession::new()),
        squishy_gizmo_drag: Mutex::new(None),
        selection_gizmo_drag: Mutex::new(SelectionGizmoDrag::None),
        extrude_gizmo_drag: Mutex::new(ExtrudeGizmoDrag::None),
        extrude_gizmo_base_depth: Mutex::new(0),
        hovered_extrude_axis: AtomicU8::new(255),
        start_screen_logo_transparent: std::sync::atomic::AtomicBool::new(true),
        start_screen_light: std::sync::atomic::AtomicBool::new(false),
        overlay_mesh_generation: AtomicU64::new(0),
        viewport_cursor_debug_overlay: AtomicBool::new(false),
        show_grid_borders: AtomicBool::new(false),
        hovered_gizmo_axis: AtomicU8::new(255),
        grid_overlay_cache_key: Mutex::new(None),
        selection_overlay_cache_key: Mutex::new(None),
        preview_overlay_cache_key: Mutex::new(None),
        generator_preview_locked_camera: Mutex::new(None),
        fill_operation_cancel: Arc::new(AtomicBool::new(false)),
        spray_constraint_plane: Mutex::new(None),
        wall_stroke_face_snapped: Mutex::new(None),
    })
}

#[cfg(test)]
mod edit_perf_tests {
    use super::*;
    use voxelle::{MaterialId, Voxel};

    fn voxel_at(x: i32, y: i32, z: i32, object_id: u32) -> Voxel {
        Voxel {
            x,
            y,
            z,
            color: 1,
            material: MaterialId::Plastic,
            object_id,
        }
    }

    #[test]
    fn resolve_stats_incremental_add_shrinks_aabb_min() {
        let cache = Some(VoxelEditStatsCache {
            aabb_min: (5, 5, 5),
            common_object_id: Some(0),
        });
        let added = voxel_at(2, 5, 5, 0);
        let delta = voxel_edit::VoxelEditDelta::Added(added);
        let voxels = vec![voxel_at(5, 5, 5, 0), added];
        let s = resolve_voxel_edit_stats(&voxels, &delta, cache);
        assert_eq!(s.aabb_min, (2, 5, 5));
        assert_eq!(s.common_object_id, Some(0));
    }

    #[test]
    fn resolve_stats_remove_interior_preserves_cache() {
        let cache = Some(VoxelEditStatsCache {
            aabb_min: (0, 0, 0),
            common_object_id: Some(0),
        });
        let voxels = vec![voxel_at(0, 0, 0, 0), voxel_at(5, 5, 5, 0)];
        let delta = voxel_edit::VoxelEditDelta::Removed {
            voxel: voxel_at(5, 5, 5, 0),
        };
        let s = resolve_voxel_edit_stats(&voxels, &delta, cache);
        assert_eq!(s.aabb_min, (0, 0, 0));
        assert_eq!(s.common_object_id, Some(0));
    }

    #[test]
    fn resolve_stats_remove_on_min_face_rescans() {
        let cache = Some(VoxelEditStatsCache {
            aabb_min: (0, 0, 0),
            common_object_id: Some(0),
        });
        let voxels = vec![voxel_at(5, 5, 5, 0)];
        let delta = voxel_edit::VoxelEditDelta::Removed {
            voxel: voxel_at(0, 0, 0, 0),
        };
        let s = resolve_voxel_edit_stats(&voxels, &delta, cache);
        assert_eq!(s.aabb_min, (5, 5, 5));
        assert_eq!(s.common_object_id, Some(0));
    }

    #[test]
    fn resolve_stats_add_second_object_id_clears_common() {
        let cache = Some(VoxelEditStatsCache {
            aabb_min: (0, 0, 0),
            common_object_id: Some(0),
        });
        let added = voxel_at(1, 1, 1, 1);
        let delta = voxel_edit::VoxelEditDelta::Added(added);
        let voxels = vec![voxel_at(0, 0, 0, 0), added];
        let s = resolve_voxel_edit_stats(&voxels, &delta, cache);
        assert_eq!(s.common_object_id, None);
    }

    // ── merge_coords_into_selection ─────────────────────────────────

    #[test]
    fn merge_replace_clears_and_sets() {
        let mut sel: AHashSet<greedy_mesh::VoxelCoord> =
            [(0, 0, 0), (1, 1, 1)].into_iter().collect();
        merge_coords_into_selection(&mut sel, vec![(2, 2, 2)], SelectionCombineMode::Replace);
        assert_eq!(sel, [(2, 2, 2)].into_iter().collect());
    }

    #[test]
    fn merge_add_unions() {
        let mut sel: AHashSet<greedy_mesh::VoxelCoord> = [(0, 0, 0)].into_iter().collect();
        merge_coords_into_selection(&mut sel, vec![(1, 1, 1)], SelectionCombineMode::Add);
        assert_eq!(sel, [(0, 0, 0), (1, 1, 1)].into_iter().collect());
    }

    #[test]
    fn merge_subtract_removes() {
        let mut sel: AHashSet<greedy_mesh::VoxelCoord> =
            [(0, 0, 0), (1, 1, 1)].into_iter().collect();
        merge_coords_into_selection(&mut sel, vec![(1, 1, 1)], SelectionCombineMode::Subtract);
        assert_eq!(sel, [(0, 0, 0)].into_iter().collect());
    }

    #[test]
    fn merge_intersect_keeps_overlap() {
        let mut sel: AHashSet<greedy_mesh::VoxelCoord> =
            [(0, 0, 0), (1, 1, 1), (2, 2, 2)].into_iter().collect();
        merge_coords_into_selection(
            &mut sel,
            vec![(1, 1, 1), (2, 2, 2), (3, 3, 3)],
            SelectionCombineMode::Intersect,
        );
        assert_eq!(sel, [(1, 1, 1), (2, 2, 2)].into_iter().collect());
    }

    // ── apply_selection_stroke_sample (accumulator) ─────────────────

    /// Simulates a full stroke: begin → N samples → verify selection.
    /// With intersect mode, successive samples should union their coords
    /// against the original `before` snapshot rather than shrinking.
    #[test]
    fn intersect_stroke_accumulates_across_samples() {
        // Original selection: A B C D
        let a = (0, 0, 0);
        let b = (1, 0, 0);
        let c = (2, 0, 0);
        let d = (3, 0, 0);
        let before: AHashSet<greedy_mesh::VoxelCoord> = [a, b, c, d].into_iter().collect();

        // stroke_begin: snapshot before, create empty accumulator
        let mut sel = before.clone();
        let mut accum: Option<AHashSet<greedy_mesh::VoxelCoord>> = Some(AHashSet::new());
        let before_snap = Some(before);

        // Sample 1: spray hits only A
        let r = apply_selection_stroke_sample(
            &mut sel,
            vec![a],
            SelectionCombineMode::Intersect,
            &mut accum,
            &before_snap,
        );
        assert!(r.is_some());
        assert_eq!(sel, [a].into_iter().collect());
        // Accum should contain A
        assert!(accum.as_ref().unwrap().contains(&a));

        // Sample 2: spray hits only C
        let r = apply_selection_stroke_sample(
            &mut sel,
            vec![c],
            SelectionCombineMode::Intersect,
            &mut accum,
            &before_snap,
        );
        assert!(r.is_some());
        // Selection should be before ∩ {A, C} = {A, C}
        assert_eq!(sel, [a, c].into_iter().collect());

        // Sample 3: spray hits D and B
        let r = apply_selection_stroke_sample(
            &mut sel,
            vec![d, b],
            SelectionCombineMode::Intersect,
            &mut accum,
            &before_snap,
        );
        assert!(r.is_some());
        // Selection should be before ∩ {A, B, C, D} = {A, B, C, D}
        assert_eq!(sel, [a, b, c, d].into_iter().collect());
    }

    /// Without an accumulator (no active stroke), intersect should work
    /// directly on the current selection (single-click fallthrough).
    #[test]
    fn intersect_no_stroke_falls_through_to_direct_merge() {
        let a = (0, 0, 0);
        let b = (1, 0, 0);
        let c = (2, 0, 0);
        let mut sel: AHashSet<greedy_mesh::VoxelCoord> = [a, b, c].into_iter().collect();
        let mut accum: Option<AHashSet<greedy_mesh::VoxelCoord>> = None;
        let before: Option<AHashSet<greedy_mesh::VoxelCoord>> = None;

        let r = apply_selection_stroke_sample(
            &mut sel,
            vec![b],
            SelectionCombineMode::Intersect,
            &mut accum,
            &before,
        );
        assert!(r.is_some());
        assert_eq!(sel, [b].into_iter().collect());
    }

    /// Empty coords with active accumulator should still recompute
    /// selection (accum unchanged, but selection is re-derived).
    #[test]
    fn intersect_empty_sample_preserves_accum_state() {
        let a = (0, 0, 0);
        let b = (1, 0, 0);
        let before: AHashSet<greedy_mesh::VoxelCoord> = [a, b].into_iter().collect();
        let mut sel = before.clone();
        let mut accum: Option<AHashSet<greedy_mesh::VoxelCoord>> = Some([a].into_iter().collect());
        let before_snap = Some(before);

        // Empty sample — accum stays {A}, sel = before ∩ {A} = {A}
        let r = apply_selection_stroke_sample(
            &mut sel,
            vec![],
            SelectionCombineMode::Intersect,
            &mut accum,
            &before_snap,
        );
        assert!(r.is_some());
        assert_eq!(sel, [a].into_iter().collect());
    }

    /// Non-intersect modes should ignore the accumulator entirely.
    #[test]
    fn add_mode_ignores_accumulator() {
        let a = (0, 0, 0);
        let b = (1, 0, 0);
        let mut sel: AHashSet<greedy_mesh::VoxelCoord> = [a].into_iter().collect();
        let mut accum: Option<AHashSet<greedy_mesh::VoxelCoord>> = Some(AHashSet::new());
        let before = Some([a].into_iter().collect());

        let r = apply_selection_stroke_sample(
            &mut sel,
            vec![b],
            SelectionCombineMode::Add,
            &mut accum,
            &before,
        );
        assert!(r.is_some());
        assert_eq!(sel, [a, b].into_iter().collect());
    }
}
