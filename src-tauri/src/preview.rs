//! Preview mesh preparation, caching, and sync-input plumbing.
//!
//! Extracted from `lib.rs` to keep the main module smaller.

use crate::camera::OrbitCamera;
use crate::generators;
use crate::greedy_mesh;
use crate::paint_color_distrib;
use crate::render::WgpuViewer;
use crate::stroke_modes;
use crate::voxel_edit;
use crate::voxelle;
use crate::voxelle::start_shape::StartShape;
use crate::{
    append_polygon_vertex_marker_meshes, build_color_resolver, build_raw_voxel_upload,
    preview_single_cell_world, preview_tool_colors, stroke_preview_meshes_for_union,
    viewport_texels_from_norm, wake_viewport_loop, PreviewHoverContext, PreviewMode, ViewerState,
};
// Default-value functions referenced by `#[serde(default = "...")]` on `SyncPreviewInput`.
// They live in lib.rs (shared with the generator command structs).
use crate::{
    default_fauna_archetype, default_fauna_body_length, default_fauna_spine_segments,
    default_fauna_stance, default_flora_braid_twist, default_flora_branch_start,
    default_flora_height, default_flora_taper, default_flora_wobble, default_grass_density,
    default_grass_radius, default_insecta_abdomen_ratio, default_insecta_abdomen_taper,
    default_insecta_antenna_length, default_insecta_antenna_pitch, default_insecta_antenna_spread,
    default_insecta_species, default_insecta_wing_fore_length, default_insecta_wing_hind_length,
    default_one_f32, default_one_i32, default_piscina_length, default_piscina_species,
    default_rock_cluster_radius, default_rock_count, default_rock_size, default_roof_break_ratio,
    default_roof_height, default_roof_parapet_height, default_roof_style, default_roof_wall_height,
    default_rope_sag, default_rope_tension,
};

use ahash::{AHashMap, AHashSet, AHasher};
use std::hash::{Hash, Hasher};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tauri::{AppHandle, State};

/// Cache key for single-cell preview.  Keyed on the discrete voxel cell
/// `(cx, cy, cz)` so the cache stays valid while the cursor remains in the
/// same cell — sub-pixel `texel_s*` coordinates are intentionally excluded
/// because the preview mesh is grid-snapped and does not vary within a cell.
pub(crate) fn hash_single_cell_preview(
    mode: PreviewMode,
    cx: i32,
    cy: i32,
    cz: i32,
    tag: u8,
    debug_overlay: bool,
    palette_color: u32,
    object_id: u32,
) -> u64 {
    let mut h = AHasher::default();
    mode.hash(&mut h);
    cx.hash(&mut h);
    cy.hash(&mut h);
    cz.hash(&mut h);
    tag.hash(&mut h);
    debug_overlay.hash(&mut h);
    palette_color.hash(&mut h);
    object_id.hash(&mut h);
    h.finish()
}

pub(crate) fn hash_preview_miss(mode: PreviewMode, debug_overlay: bool) -> u64 {
    let mut h = AHasher::default();
    mode.hash(&mut h);
    0x7Fu8.hash(&mut h);
    debug_overlay.hash(&mut h);
    h.finish()
}

pub(crate) fn hash_squishy_preview(
    session: &generators::SquishySession,
    sx: f32,
    sy: f32,
    add_anchor: Option<(i32, i32, i32)>,
    gizmo_drag: bool,
    delete_hover_id: Option<u32>,
    debug_overlay: bool,
    palette_color: u32,
) -> u64 {
    let mut h = AHasher::default();
    PreviewMode::Squishy.hash(&mut h);
    debug_overlay.hash(&mut h);
    palette_color.hash(&mut h);
    sx.to_bits().hash(&mut h);
    sy.to_bits().hash(&mut h);
    gizmo_drag.hash(&mut h);
    delete_hover_id.hash(&mut h);
    (session.mode as u8).hash(&mut h);
    session.hollow.hash(&mut h);
    session.wall_thickness.hash(&mut h);
    session.add_snap_to_surface.hash(&mut h);
    session.selected_id.hash(&mut h);
    for b in &session.balls {
        b.id.hash(&mut h);
        b.x.hash(&mut h);
        b.y.hash(&mut h);
        b.z.hash(&mut h);
        b.radius.to_bits().hash(&mut h);
    }
    if let Some((ax, ay, az)) = add_anchor {
        ax.hash(&mut h);
        ay.hash(&mut h);
        az.hash(&mut h);
    } else {
        0x5Eu8.hash(&mut h);
    }
    h.finish()
}

fn hash_bone_preview(
    session: &generators::BoneSession,
    sx: f32,
    sy: f32,
    gizmo_drag: bool,
    debug_overlay: bool,
    palette_color: u32,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = AHasher::default();
    PreviewMode::Bone.hash(&mut h);
    debug_overlay.hash(&mut h);
    palette_color.hash(&mut h);
    sx.to_bits().hash(&mut h);
    sy.to_bits().hash(&mut h);
    gizmo_drag.hash(&mut h);
    for j in &session.joints {
        j.id.hash(&mut h);
        j.x.to_bits().hash(&mut h);
        j.y.to_bits().hash(&mut h);
        j.z.to_bits().hash(&mut h);
        j.radius.to_bits().hash(&mut h);
    }
    for b in &session.bones {
        b.id.hash(&mut h);
        b.joint_a.hash(&mut h);
        b.joint_b.hash(&mut h);
    }
    session.selected.hash(&mut h);
    session.pending_joint.hash(&mut h);
    h.finish()
}

pub(crate) fn hash_brush_hover_targets(
    mode: PreviewMode,
    ctx: &PreviewHoverContext,
    targets: &[greedy_mesh::VoxelCoord],
    voxel_map: &AHashMap<greedy_mesh::VoxelCoord, usize>,
    debug_overlay: bool,
) -> u64 {
    let mut sorted: Vec<_> = targets.to_vec();
    sorted.sort_unstable();
    let mut h = AHasher::default();
    mode.hash(&mut h);
    debug_overlay.hash(&mut h);
    ctx.use_brush_preview.hash(&mut h);
    ctx.brush_radius.hash(&mut h);
    (ctx.brush_shape as u8).hash(&mut h);
    ctx.spray_density.to_bits().hash(&mut h);
    (ctx.stroke_mode as u8).hash(&mut h);
    (ctx.plane_axis as u8).hash(&mut h);
    ctx.color.hash(&mut h);
    ctx.material.hash(&mut h);
    ctx.match_material.hash(&mut h);
    sorted.hash(&mut h);
    for c in &sorted {
        voxel_map.contains_key(c).hash(&mut h);
    }
    if let Ok(s) = serde_json::to_string(&ctx.stroke_aux) {
        s.hash(&mut h);
    }
    h.finish()
}

pub(crate) fn default_true() -> bool {
    true
}

/// Cursor + mode + brush/stroke state for hover preview (mesh work runs on the viewport thread).
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SyncPreviewInput {
    nx: f32,
    ny: f32,
    mode: String,
    #[serde(default)]
    brush_radius: u32,
    #[serde(default)]
    brush_shape: voxel_edit::BrushShape,
    #[serde(default)]
    spray_density: f32,
    #[serde(default)]
    stroke_mode: stroke_modes::DrawStrokeMode,
    #[serde(default)]
    plane_axis: stroke_modes::PlaneAxis,
    #[serde(default)]
    stroke_aux: stroke_modes::StrokeAux,
    #[serde(default)]
    color: u32,
    #[serde(default)]
    palette: Vec<u32>,
    #[serde(default)]
    paint_color_distrib: Option<paint_color_distrib::PaintColorDistrib>,
    #[serde(default)]
    material: String,
    #[serde(default)]
    match_material: bool,
    #[serde(default = "default_true")]
    use_brush_preview: bool,
    #[serde(default)]
    generator_kind: Option<String>,
    #[serde(default)]
    generator_rope_first_voxel: Option<[i32; 3]>,
    #[serde(default = "default_rope_sag")]
    generator_rope_sag: f32,
    #[serde(default = "default_rope_tension")]
    generator_rope_tension: f32,
    #[serde(default = "default_cloth_gravity_direction_str")]
    generator_rope_gravity_direction: String,
    #[serde(default)]
    generator_cloth_pins: Vec<[i32; 3]>,
    #[serde(default = "default_cloth_tension_preview")]
    generator_cloth_tension: f32,
    #[serde(default = "default_cloth_gravity_direction_str")]
    generator_cloth_gravity_direction: String,
    #[serde(default = "default_one_f64")]
    generator_cloth_gravity_scale: f64,
    #[serde(default = "default_one_f64")]
    generator_cloth_stiffness_scale: f64,
    #[serde(default)]
    generator_cloth_iterations: u32,
    #[serde(default = "default_cloth_constraint_passes_u32")]
    generator_cloth_constraint_passes: u32,
    #[serde(default = "default_rock_size")]
    generator_rock_size: i32,
    #[serde(default = "default_rock_roughness")]
    generator_rock_roughness: f32,
    #[serde(default = "default_rock_seed")]
    generator_rock_seed: i32,
    #[serde(default = "default_rock_count")]
    generator_rock_count: i32,
    #[serde(default = "default_rock_cluster_radius")]
    generator_rock_cluster_radius: i32,
    #[serde(default)]
    generator_rock_sink_direction: i32,
    #[serde(default)]
    generator_rock_sink_amount: i32,
    #[serde(default = "default_grass_radius")]
    generator_grass_radius: i32,
    #[serde(default = "default_grass_density")]
    generator_grass_density: f32,
    #[serde(default = "default_grass_max_height")]
    generator_grass_max_height: i32,
    #[serde(default = "default_grass_seed")]
    generator_grass_seed: i32,
    #[serde(default)]
    generator_roof_pins: Vec<[i32; 3]>,
    #[serde(default = "default_roof_style")]
    generator_roof_style: String,
    #[serde(default = "default_roof_height")]
    generator_roof_height: i32,
    #[serde(default = "default_one_i32")]
    generator_roof_thickness: i32,
    #[serde(default = "default_roof_break_ratio")]
    generator_roof_break_ratio: f32,
    #[serde(default = "default_roof_wall_height")]
    generator_roof_wall_height: i32,
    #[serde(default = "default_roof_parapet_height")]
    generator_roof_parapet_height: i32,
    #[serde(default)]
    generator_roof_salt_skew: f32,
    #[serde(default)]
    generator_roof_hollow: bool,
    #[serde(default = "default_rock_size")]
    generator_ashlar_size: i32,
    #[serde(default = "default_ashlar_roughness")]
    generator_ashlar_roughness: f32,
    #[serde(default = "default_ashlar_seed")]
    generator_ashlar_seed: i32,
    #[serde(default = "default_ashlar_thickness")]
    generator_ashlar_thickness: i32,
    // Flora
    #[serde(default = "default_flora_seed")]
    generator_flora_seed: i32,
    #[serde(default = "default_flora_height")]
    generator_flora_height: i32,
    #[serde(default = "default_flora_girth")]
    generator_flora_girth: i32,
    #[serde(default = "default_flora_wobble")]
    generator_flora_wobble: f32,
    #[serde(default = "default_flora_taper")]
    generator_flora_taper: f32,
    #[serde(default = "default_one_i32")]
    generator_flora_stem_count: i32,
    #[serde(default)]
    generator_flora_cluster_radius: i32,
    #[serde(default = "default_flora_branch_count")]
    generator_flora_branch_count: i32,
    #[serde(default = "default_two_i32")]
    generator_flora_branch_depth: i32,
    #[serde(default = "default_flora_branch_start")]
    generator_flora_branch_start: f32,
    #[serde(default = "default_flora_branch_spread")]
    generator_flora_branch_spread: f32,
    #[serde(default)]
    generator_flora_braid_strands: i32,
    #[serde(default = "default_flora_braid_twist")]
    generator_flora_braid_twist: f32,
    #[serde(default = "default_flora_canopy")]
    generator_flora_canopy: f32,
    // Insecta
    #[serde(default = "default_insecta_species")]
    generator_insecta_species: String,
    #[serde(default = "default_insecta_total_length")]
    generator_insecta_total_length: i32,
    #[serde(default = "default_one_f32")]
    generator_insecta_head_ratio: f32,
    #[serde(default = "default_one_f32")]
    generator_insecta_thorax_ratio: f32,
    #[serde(default = "default_insecta_abdomen_ratio")]
    generator_insecta_abdomen_ratio: f32,
    #[serde(default = "default_two_i32")]
    generator_insecta_body_half_width: i32,
    #[serde(default = "default_two_i32")]
    generator_insecta_body_half_height: i32,
    #[serde(default = "default_insecta_abdomen_taper")]
    generator_insecta_abdomen_taper: f32,
    #[serde(default)]
    generator_insecta_head_shape: i32,
    #[serde(default)]
    generator_insecta_anchor_offset_u: i32,
    #[serde(default)]
    generator_insecta_anchor_offset_v: i32,
    #[serde(default)]
    generator_insecta_body_yaw: f32,
    #[serde(default)]
    generator_insecta_body_arch: f32,
    #[serde(default = "default_insecta_antenna_length")]
    generator_insecta_antenna_length: i32,
    #[serde(default = "default_insecta_antenna_spread")]
    generator_insecta_antenna_spread: f32,
    #[serde(default = "default_insecta_antenna_pitch")]
    generator_insecta_antenna_pitch: f32,
    #[serde(default = "default_one_i32")]
    generator_insecta_antenna_root: i32,
    #[serde(default = "default_two_i32")]
    generator_insecta_mandible_length: i32,
    #[serde(default = "default_insecta_mandible_spread")]
    generator_insecta_mandible_spread: f32,
    #[serde(default = "default_one_i32")]
    generator_insecta_mandible_forward: i32,
    #[serde(default)]
    generator_insecta_wing_shape: i32,
    #[serde(default = "default_true")]
    generator_insecta_show_wing_fore: bool,
    #[serde(default = "default_insecta_wing_fore_length")]
    generator_insecta_wing_fore_length: i32,
    #[serde(default = "default_four_i32")]
    generator_insecta_wing_fore_width: i32,
    #[serde(default = "default_insecta_wing_fore_spread")]
    generator_insecta_wing_fore_spread: f32,
    #[serde(default = "default_insecta_wing_fore_pitch")]
    generator_insecta_wing_fore_pitch: f32,
    #[serde(default)]
    generator_insecta_wing_fore_offset: i32,
    #[serde(default)]
    generator_insecta_wing_fore_forward_cant: f32,
    #[serde(default = "default_true")]
    generator_insecta_show_wing_hind: bool,
    #[serde(default = "default_insecta_wing_hind_length")]
    generator_insecta_wing_hind_length: i32,
    #[serde(default = "default_four_i32")]
    generator_insecta_wing_hind_width: i32,
    #[serde(default = "default_insecta_wing_hind_spread")]
    generator_insecta_wing_hind_spread: f32,
    #[serde(default = "default_insecta_wing_hind_pitch")]
    generator_insecta_wing_hind_pitch: f32,
    #[serde(default)]
    generator_insecta_wing_hind_offset: i32,
    // Fauna
    #[serde(default = "default_fauna_stance")]
    generator_fauna_stance: String,
    #[serde(default = "default_fauna_archetype")]
    generator_fauna_archetype: String,
    #[serde(default)]
    generator_fauna_anchor_offset_u: i32,
    #[serde(default)]
    generator_fauna_anchor_offset_v: i32,
    #[serde(default)]
    generator_fauna_body_yaw: f32,
    #[serde(default)]
    generator_fauna_body_arch: f32,
    #[serde(default = "default_fauna_spine_segments")]
    generator_fauna_spine_segments: i32,
    #[serde(default = "default_fauna_body_length")]
    generator_fauna_body_length: i32,
    #[serde(default = "default_two_i32")]
    generator_fauna_body_half_width: i32,
    #[serde(default = "default_two_i32")]
    generator_fauna_body_half_height: i32,
    #[serde(default = "default_three_i32")]
    generator_fauna_neck_length: i32,
    #[serde(default = "default_one_i32")]
    generator_fauna_neck_half_width: i32,
    #[serde(default = "default_one_i32")]
    generator_fauna_neck_half_height: i32,
    #[serde(default = "default_three_i32")]
    generator_fauna_head_length: i32,
    #[serde(default = "default_two_i32")]
    generator_fauna_head_half_width: i32,
    #[serde(default = "default_two_i32")]
    generator_fauna_head_half_height: i32,
    #[serde(default = "default_four_i32")]
    generator_fauna_tail_length: i32,
    #[serde(default = "default_three_i32")]
    generator_fauna_shoulder_offset_forward: i32,
    #[serde(default = "default_fauna_hip_offset_forward")]
    generator_fauna_hip_offset_forward: i32,
    #[serde(default = "default_four_i32")]
    generator_fauna_front_upper_length: i32,
    #[serde(default = "default_four_i32")]
    generator_fauna_front_lower_length: i32,
    #[serde(default = "default_four_i32")]
    generator_fauna_hind_upper_length: i32,
    #[serde(default = "default_four_i32")]
    generator_fauna_hind_lower_length: i32,
    #[serde(default = "default_true")]
    generator_fauna_auto_foot_placement: bool,
    // Piscina
    #[serde(default = "default_piscina_seed")]
    generator_piscina_seed: i32,
    #[serde(default = "default_piscina_species")]
    generator_piscina_species: String,
    #[serde(default = "default_piscina_length")]
    generator_piscina_length: i32,
    #[serde(default = "default_four_i32")]
    generator_piscina_width: i32,
    #[serde(default = "default_three_i32")]
    generator_piscina_thickness: i32,
    #[serde(default = "default_piscina_spine_bend")]
    generator_piscina_spine_bend: f32,
    #[serde(default)]
    generator_piscina_spine_s_curve: f32,
    #[serde(default = "default_four_i32")]
    generator_piscina_fin_dorsal: i32,
    #[serde(default = "default_four_i32")]
    generator_piscina_fin_anal: i32,
    #[serde(default = "default_four_i32")]
    generator_piscina_fin_caudal: i32,
    #[serde(default = "default_four_i32")]
    generator_piscina_fin_pectoral: i32,
    #[serde(default = "default_four_i32")]
    generator_piscina_fin_pelvic: i32,
    #[serde(default = "default_four_i32")]
    generator_piscina_fin_adipose: i32,
    #[serde(default = "default_true")]
    generator_piscina_show_fin_dorsal: bool,
    #[serde(default = "default_true")]
    generator_piscina_show_fin_anal: bool,
    #[serde(default = "default_true")]
    generator_piscina_show_fin_caudal: bool,
    #[serde(default = "default_true")]
    generator_piscina_show_fin_pectoral: bool,
    #[serde(default = "default_true")]
    generator_piscina_show_fin_pelvic: bool,
    #[serde(default)]
    generator_piscina_show_fin_adipose: bool,
    #[serde(default)]
    generator_piscina_anchor_offset_u: i32,
    #[serde(default)]
    generator_piscina_anchor_offset_v: i32,
    #[serde(default)]
    stamp_origin_x: i32,
    #[serde(default)]
    stamp_origin_z: i32,
    /// Symmetry bitmask: bit 0 = X, bit 1 = Y, bit 2 = Z. 0 = no mirroring.
    #[serde(default)]
    mirror_axes: u8,
    // Shape
    #[serde(default = "default_shape_kind")]
    generator_shape_kind: String,
    #[serde(default = "default_shape_size")]
    generator_shape_size: i32,
    #[serde(default)]
    generator_shape_rot_x: f32,
    #[serde(default)]
    generator_shape_rot_y: f32,
    #[serde(default)]
    generator_shape_rot_z: f32,
    #[serde(default = "default_true")]
    generator_shape_overwrite: bool,
}

pub(crate) fn default_ashlar_roughness() -> f32 {
    0.3
}
pub(crate) fn default_ashlar_seed() -> i32 {
    42
}
pub(crate) fn default_ashlar_thickness() -> i32 {
    3
}

pub(crate) fn default_rock_roughness() -> f32 {
    0.4
}
pub(crate) fn default_rock_seed() -> i32 {
    42
}
pub(crate) fn default_grass_max_height() -> i32 {
    3
}
pub(crate) fn default_grass_seed() -> i32 {
    42
}

// New defaults for bio-generator preview fields (SyncPreviewInput)
pub(crate) fn default_flora_seed() -> i32 {
    42
}
pub(crate) fn default_flora_girth() -> i32 {
    2
}
pub(crate) fn default_flora_branch_count() -> i32 {
    4
}
pub(crate) fn default_flora_branch_spread() -> f32 {
    0.5
}
pub(crate) fn default_flora_canopy() -> f32 {
    2.0
}
pub(crate) fn default_insecta_total_length() -> i32 {
    12
}
pub(crate) fn default_insecta_mandible_spread() -> f32 {
    0.3
}
pub(crate) fn default_insecta_wing_fore_spread() -> f32 {
    0.5
}
pub(crate) fn default_insecta_wing_fore_pitch() -> f32 {
    0.1
}
pub(crate) fn default_insecta_wing_hind_spread() -> f32 {
    0.6
}
pub(crate) fn default_insecta_wing_hind_pitch() -> f32 {
    0.2
}
pub(crate) fn default_fauna_hip_offset_forward() -> i32 {
    -3
}
pub(crate) fn default_piscina_seed() -> i32 {
    42
}
pub(crate) fn default_piscina_spine_bend() -> f32 {
    0.1
}
pub(crate) fn default_two_i32() -> i32 {
    2
}
pub(crate) fn default_three_i32() -> i32 {
    3
}
pub(crate) fn default_four_i32() -> i32 {
    4
}

pub(crate) fn default_cloth_tension_preview() -> f32 {
    0.5
}

pub(crate) fn default_cloth_gravity_direction_str() -> String {
    "down".into()
}

pub(crate) fn default_one_f64() -> f64 {
    1.0
}

pub(crate) fn default_cloth_constraint_passes_u32() -> u32 {
    2
}

pub(crate) fn default_shape_kind() -> String {
    "cube".into()
}

pub(crate) fn default_shape_size() -> i32 {
    8
}

#[tauri::command]
pub(crate) fn sync_preview_input(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: SyncPreviewInput,
) -> Result<(), String> {
    let new_mode = PreviewMode::parse(&args.mode);
    {
        let mut pm = state.preview.preview_mode.lock();
        let changed = *pm != new_mode;
        *pm = new_mode;
        if changed {
            wake_viewport_loop(&app);
        }
    }
    {
        let mut ph = state.preview.preview_hover.lock();
        ph.brush_radius = args.brush_radius;
        ph.brush_shape = args.brush_shape;
        ph.spray_density = args.spray_density;
        ph.stroke_mode = args.stroke_mode;
        ph.plane_axis = args.plane_axis;
        ph.stroke_aux = args.stroke_aux;
        ph.color = args.color;
        ph.palette = args.palette.clone();
        ph.paint_color_distrib = args.paint_color_distrib.clone();
        ph.material = args.material;
        ph.match_material = args.match_material;
        ph.use_brush_preview = args.use_brush_preview;
        ph.generator_kind = args.generator_kind.clone();
        ph.generator_rope_first_voxel = args.generator_rope_first_voxel;
        ph.generator_rope_sag = args.generator_rope_sag;
        ph.generator_rope_tension = args.generator_rope_tension;
        ph.generator_rope_gravity_direction = args.generator_rope_gravity_direction.clone();
        ph.generator_cloth_pins
            .clone_from(&args.generator_cloth_pins);
        ph.generator_cloth_tension = args.generator_cloth_tension;
        ph.generator_cloth_gravity_direction = args.generator_cloth_gravity_direction.clone();
        ph.generator_cloth_gravity_scale = args.generator_cloth_gravity_scale;
        ph.generator_cloth_stiffness_scale = args.generator_cloth_stiffness_scale;
        ph.generator_cloth_iterations = args.generator_cloth_iterations;
        ph.generator_cloth_constraint_passes = args.generator_cloth_constraint_passes;
        ph.generator_rock_size = args.generator_rock_size;
        ph.generator_rock_roughness = args.generator_rock_roughness;
        ph.generator_rock_seed = args.generator_rock_seed;
        ph.generator_rock_count = args.generator_rock_count;
        ph.generator_rock_cluster_radius = args.generator_rock_cluster_radius;
        ph.generator_rock_sink_direction = args.generator_rock_sink_direction;
        ph.generator_rock_sink_amount = args.generator_rock_sink_amount;
        ph.generator_grass_radius = args.generator_grass_radius;
        ph.generator_grass_density = args.generator_grass_density;
        ph.generator_grass_max_height = args.generator_grass_max_height;
        ph.generator_grass_seed = args.generator_grass_seed;
        ph.generator_roof_pins = args.generator_roof_pins.clone();
        ph.generator_roof_style = args.generator_roof_style.clone();
        ph.generator_roof_height = args.generator_roof_height;
        ph.generator_roof_thickness = args.generator_roof_thickness;
        ph.generator_roof_break_ratio = args.generator_roof_break_ratio;
        ph.generator_roof_wall_height = args.generator_roof_wall_height;
        ph.generator_roof_parapet_height = args.generator_roof_parapet_height;
        ph.generator_roof_salt_skew = args.generator_roof_salt_skew;
        ph.generator_roof_hollow = args.generator_roof_hollow;
        ph.generator_ashlar_size = args.generator_ashlar_size;
        ph.generator_ashlar_roughness = args.generator_ashlar_roughness;
        ph.generator_ashlar_seed = args.generator_ashlar_seed;
        ph.generator_ashlar_thickness = args.generator_ashlar_thickness;
        // Flora
        ph.generator_flora_seed = args.generator_flora_seed;
        ph.generator_flora_height = args.generator_flora_height;
        ph.generator_flora_girth = args.generator_flora_girth;
        ph.generator_flora_wobble = args.generator_flora_wobble;
        ph.generator_flora_taper = args.generator_flora_taper;
        ph.generator_flora_stem_count = args.generator_flora_stem_count;
        ph.generator_flora_cluster_radius = args.generator_flora_cluster_radius;
        ph.generator_flora_branch_count = args.generator_flora_branch_count;
        ph.generator_flora_branch_depth = args.generator_flora_branch_depth;
        ph.generator_flora_branch_start = args.generator_flora_branch_start;
        ph.generator_flora_branch_spread = args.generator_flora_branch_spread;
        ph.generator_flora_braid_strands = args.generator_flora_braid_strands;
        ph.generator_flora_braid_twist = args.generator_flora_braid_twist;
        ph.generator_flora_canopy = args.generator_flora_canopy;
        // Insecta
        ph.generator_insecta_species = args.generator_insecta_species.clone();
        ph.generator_insecta_total_length = args.generator_insecta_total_length;
        ph.generator_insecta_head_ratio = args.generator_insecta_head_ratio;
        ph.generator_insecta_thorax_ratio = args.generator_insecta_thorax_ratio;
        ph.generator_insecta_abdomen_ratio = args.generator_insecta_abdomen_ratio;
        ph.generator_insecta_body_half_width = args.generator_insecta_body_half_width;
        ph.generator_insecta_body_half_height = args.generator_insecta_body_half_height;
        ph.generator_insecta_abdomen_taper = args.generator_insecta_abdomen_taper;
        ph.generator_insecta_head_shape = args.generator_insecta_head_shape;
        ph.generator_insecta_anchor_offset_u = args.generator_insecta_anchor_offset_u;
        ph.generator_insecta_anchor_offset_v = args.generator_insecta_anchor_offset_v;
        ph.generator_insecta_body_yaw = args.generator_insecta_body_yaw;
        ph.generator_insecta_body_arch = args.generator_insecta_body_arch;
        ph.generator_insecta_antenna_length = args.generator_insecta_antenna_length;
        ph.generator_insecta_antenna_spread = args.generator_insecta_antenna_spread;
        ph.generator_insecta_antenna_pitch = args.generator_insecta_antenna_pitch;
        ph.generator_insecta_antenna_root = args.generator_insecta_antenna_root;
        ph.generator_insecta_mandible_length = args.generator_insecta_mandible_length;
        ph.generator_insecta_mandible_spread = args.generator_insecta_mandible_spread;
        ph.generator_insecta_mandible_forward = args.generator_insecta_mandible_forward;
        ph.generator_insecta_wing_shape = args.generator_insecta_wing_shape;
        ph.generator_insecta_show_wing_fore = args.generator_insecta_show_wing_fore;
        ph.generator_insecta_wing_fore_length = args.generator_insecta_wing_fore_length;
        ph.generator_insecta_wing_fore_width = args.generator_insecta_wing_fore_width;
        ph.generator_insecta_wing_fore_spread = args.generator_insecta_wing_fore_spread;
        ph.generator_insecta_wing_fore_pitch = args.generator_insecta_wing_fore_pitch;
        ph.generator_insecta_wing_fore_offset = args.generator_insecta_wing_fore_offset;
        ph.generator_insecta_wing_fore_forward_cant = args.generator_insecta_wing_fore_forward_cant;
        ph.generator_insecta_show_wing_hind = args.generator_insecta_show_wing_hind;
        ph.generator_insecta_wing_hind_length = args.generator_insecta_wing_hind_length;
        ph.generator_insecta_wing_hind_width = args.generator_insecta_wing_hind_width;
        ph.generator_insecta_wing_hind_spread = args.generator_insecta_wing_hind_spread;
        ph.generator_insecta_wing_hind_pitch = args.generator_insecta_wing_hind_pitch;
        ph.generator_insecta_wing_hind_offset = args.generator_insecta_wing_hind_offset;
        // Fauna
        ph.generator_fauna_stance = args.generator_fauna_stance.clone();
        ph.generator_fauna_archetype = args.generator_fauna_archetype.clone();
        ph.generator_fauna_anchor_offset_u = args.generator_fauna_anchor_offset_u;
        ph.generator_fauna_anchor_offset_v = args.generator_fauna_anchor_offset_v;
        ph.generator_fauna_body_yaw = args.generator_fauna_body_yaw;
        ph.generator_fauna_body_arch = args.generator_fauna_body_arch;
        ph.generator_fauna_spine_segments = args.generator_fauna_spine_segments;
        ph.generator_fauna_body_length = args.generator_fauna_body_length;
        ph.generator_fauna_body_half_width = args.generator_fauna_body_half_width;
        ph.generator_fauna_body_half_height = args.generator_fauna_body_half_height;
        ph.generator_fauna_neck_length = args.generator_fauna_neck_length;
        ph.generator_fauna_neck_half_width = args.generator_fauna_neck_half_width;
        ph.generator_fauna_neck_half_height = args.generator_fauna_neck_half_height;
        ph.generator_fauna_head_length = args.generator_fauna_head_length;
        ph.generator_fauna_head_half_width = args.generator_fauna_head_half_width;
        ph.generator_fauna_head_half_height = args.generator_fauna_head_half_height;
        ph.generator_fauna_tail_length = args.generator_fauna_tail_length;
        ph.generator_fauna_shoulder_offset_forward = args.generator_fauna_shoulder_offset_forward;
        ph.generator_fauna_hip_offset_forward = args.generator_fauna_hip_offset_forward;
        ph.generator_fauna_front_upper_length = args.generator_fauna_front_upper_length;
        ph.generator_fauna_front_lower_length = args.generator_fauna_front_lower_length;
        ph.generator_fauna_hind_upper_length = args.generator_fauna_hind_upper_length;
        ph.generator_fauna_hind_lower_length = args.generator_fauna_hind_lower_length;
        ph.generator_fauna_auto_foot_placement = args.generator_fauna_auto_foot_placement;
        // Piscina
        ph.generator_piscina_seed = args.generator_piscina_seed;
        ph.generator_piscina_species = args.generator_piscina_species.clone();
        ph.generator_piscina_length = args.generator_piscina_length;
        ph.generator_piscina_width = args.generator_piscina_width;
        ph.generator_piscina_thickness = args.generator_piscina_thickness;
        ph.generator_piscina_spine_bend = args.generator_piscina_spine_bend;
        ph.generator_piscina_spine_s_curve = args.generator_piscina_spine_s_curve;
        ph.generator_piscina_fin_dorsal = args.generator_piscina_fin_dorsal;
        ph.generator_piscina_fin_anal = args.generator_piscina_fin_anal;
        ph.generator_piscina_fin_caudal = args.generator_piscina_fin_caudal;
        ph.generator_piscina_fin_pectoral = args.generator_piscina_fin_pectoral;
        ph.generator_piscina_fin_pelvic = args.generator_piscina_fin_pelvic;
        ph.generator_piscina_fin_adipose = args.generator_piscina_fin_adipose;
        ph.generator_piscina_show_fin_dorsal = args.generator_piscina_show_fin_dorsal;
        ph.generator_piscina_show_fin_anal = args.generator_piscina_show_fin_anal;
        ph.generator_piscina_show_fin_caudal = args.generator_piscina_show_fin_caudal;
        ph.generator_piscina_show_fin_pectoral = args.generator_piscina_show_fin_pectoral;
        ph.generator_piscina_show_fin_pelvic = args.generator_piscina_show_fin_pelvic;
        ph.generator_piscina_show_fin_adipose = args.generator_piscina_show_fin_adipose;
        ph.generator_piscina_anchor_offset_u = args.generator_piscina_anchor_offset_u;
        ph.generator_piscina_anchor_offset_v = args.generator_piscina_anchor_offset_v;
        ph.stamp_origin_x = args.stamp_origin_x;
        ph.stamp_origin_z = args.stamp_origin_z;
        ph.mirror_axes = args.mirror_axes;
        // Shape
        ph.generator_shape_kind = args.generator_shape_kind.clone();
        ph.generator_shape_size = args.generator_shape_size;
        ph.generator_shape_rot_x = args.generator_shape_rot_x;
        ph.generator_shape_rot_y = args.generator_shape_rot_y;
        ph.generator_shape_rot_z = args.generator_shape_rot_z;
        ph.generator_shape_overwrite = args.generator_shape_overwrite;
    }
    if args.nx < 0.0 {
        *state.preview.preview_cursor.lock() = None;
    } else {
        *state.preview.preview_cursor.lock() = Some((args.nx, args.ny));
    }
    Ok(())
}

/// Snapshot the current camera for generator-placement confirm phases.
///
/// Called by the frontend immediately after a single-click generator (rocks, grass, ...) enters
/// its settings phase.  While the snapshot is held, `prepare_preview_mesh` raycasts using this
/// camera rather than the live one, so that orbiting / panning the viewport does **not** move the
/// preview voxels to a different world position.
#[tauri::command]
pub(crate) fn lock_generator_preview_camera(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
) -> Result<(), String> {
    let cam = state.cam.camera.lock().clone();
    *state.gizmos.generator_preview_locked_camera.lock() = Some(cam);
    wake_viewport_loop(&app);
    Ok(())
}

/// Release the generator-placement camera snapshot.
///
/// Called when a single-click generator phase is cancelled or committed so that the
/// regular hover preview resumes using the live camera.
#[tauri::command]
pub(crate) fn unlock_generator_preview_camera(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
) -> Result<(), String> {
    *state.gizmos.generator_preview_locked_camera.lock() = None;
    wake_viewport_loop(&app);
    Ok(())
}

#[derive(Clone)]
pub(crate) enum PreviewMeshPrepared {
    Noop,
    Clear,
    Upload {
        cache_key: u64,
        instanced: greedy_mesh::PreviewInstancedResult,
    },
    /// Generator preview: lit, opaque, self-shadowing. Uses gen_preview GPU buffers.
    GenUpload {
        cache_key: u64,
        instanced: greedy_mesh::PreviewInstancedResult,
    },
    /// Large-stroke preview processed entirely on the GPU (compute shell filter).
    /// Replaces `Upload` when the union exceeds [`greedy_mesh::PREVIEW_COMPUTE_THRESHOLD`]
    /// and all other GPU-path requirements are met.
    RawVoxelUpload {
        cache_key: u64,
        raw: greedy_mesh::RawVoxelUpload,
    },
}

#[inline]
pub(crate) fn preview_overlay_cache_key_get(state: &ViewerState) -> Option<u64> {
    *state.gpu.preview_overlay_cache_key.lock()
}

pub(crate) fn brush_shape_tag(s: voxel_edit::BrushShape) -> u8 {
    match s {
        voxel_edit::BrushShape::Sphere => 0,
        voxel_edit::BrushShape::Cube => 1,
        voxel_edit::BrushShape::Pyramid => 2,
        voxel_edit::BrushShape::Square => 3,
        voxel_edit::BrushShape::Circle => 4,
    }
}

pub(crate) fn hash_generator_rope_hover(
    h1: [i32; 3],
    sx2: f32,
    sy2: f32,
    tension: f32,
    gravity_dir: &str,
    brush_radius: u32,
    brush_shape: voxel_edit::BrushShape,
    color: u32,
    dbg: bool,
    mesh_gen: u64,
) -> u64 {
    let mut h = AHasher::default();
    0x52u8.hash(&mut h);
    h1.hash(&mut h);
    sx2.to_bits().hash(&mut h);
    sy2.to_bits().hash(&mut h);
    tension.to_bits().hash(&mut h);
    gravity_dir.hash(&mut h);
    brush_radius.hash(&mut h);
    brush_shape_tag(brush_shape).hash(&mut h);
    color.hash(&mut h);
    dbg.hash(&mut h);
    mesh_gen.hash(&mut h);
    h.finish()
}

pub(crate) fn hash_generator_cloth_hover(
    pins: &[[i32; 3]],
    tension: f32,
    gravity_dir: &str,
    gravity_scale: f64,
    stiffness_scale: f64,
    iterations: u32,
    passes: u32,
    brush_radius: u32,
    brush_shape: voxel_edit::BrushShape,
    color: u32,
    dbg: bool,
    mesh_gen: u64,
) -> u64 {
    let mut h = AHasher::default();
    0x43u8.hash(&mut h);
    pins.len().hash(&mut h);
    for p in pins {
        p[0].hash(&mut h);
        p[1].hash(&mut h);
        p[2].hash(&mut h);
    }
    tension.to_bits().hash(&mut h);
    gravity_dir.hash(&mut h);
    gravity_scale.to_bits().hash(&mut h);
    stiffness_scale.to_bits().hash(&mut h);
    iterations.hash(&mut h);
    passes.hash(&mut h);
    brush_radius.hash(&mut h);
    brush_shape_tag(brush_shape).hash(&mut h);
    color.hash(&mut h);
    dbg.hash(&mut h);
    mesh_gen.hash(&mut h);
    h.finish()
}

pub(crate) fn hash_generator_rock_hover(
    sx: f32,
    sy: f32,
    size: i32,
    roughness: f32,
    seed: i32,
    color: u32,
    dbg: bool,
    mesh_gen: u64,
    count: i32,
    cluster_radius: i32,
    sink_direction: i32,
    sink_amount: i32,
) -> u64 {
    let mut h = AHasher::default();
    0x72u8.hash(&mut h); // 'r' for rock
    sx.to_bits().hash(&mut h);
    sy.to_bits().hash(&mut h);
    size.hash(&mut h);
    roughness.to_bits().hash(&mut h);
    seed.hash(&mut h);
    color.hash(&mut h);
    dbg.hash(&mut h);
    mesh_gen.hash(&mut h);
    count.hash(&mut h);
    cluster_radius.hash(&mut h);
    sink_direction.hash(&mut h);
    sink_amount.hash(&mut h);
    h.finish()
}

pub(crate) fn hash_generator_shape_hover(
    sx: f32,
    sy: f32,
    shape_kind: &str,
    size: i32,
    rot_x: f32,
    rot_y: f32,
    rot_z: f32,
    overwrite: bool,
    color: u32,
    dbg: bool,
    mesh_gen: u64,
) -> u64 {
    let mut h = AHasher::default();
    0x73u8.hash(&mut h); // 's' for shape
    sx.to_bits().hash(&mut h);
    sy.to_bits().hash(&mut h);
    shape_kind.hash(&mut h);
    size.hash(&mut h);
    rot_x.to_bits().hash(&mut h);
    rot_y.to_bits().hash(&mut h);
    rot_z.to_bits().hash(&mut h);
    overwrite.hash(&mut h);
    color.hash(&mut h);
    dbg.hash(&mut h);
    mesh_gen.hash(&mut h);
    h.finish()
}

pub(crate) fn hash_generator_grass_hover(
    sx: f32,
    sy: f32,
    radius: i32,
    density: f32,
    max_height: i32,
    seed: i32,
    color: u32,
    dbg: bool,
    mesh_gen: u64,
) -> u64 {
    let mut h = AHasher::default();
    0x67u8.hash(&mut h); // 'g' for grass
    sx.to_bits().hash(&mut h);
    sy.to_bits().hash(&mut h);
    radius.hash(&mut h);
    density.to_bits().hash(&mut h);
    max_height.hash(&mut h);
    seed.hash(&mut h);
    color.hash(&mut h);
    dbg.hash(&mut h);
    mesh_gen.hash(&mut h);
    h.finish()
}

pub(crate) fn hash_generator_ashlar_hover(
    sx: f32,
    sy: f32,
    size: i32,
    roughness: f32,
    seed: i32,
    thickness: i32,
    color: u32,
    dbg: bool,
    mesh_gen: u64,
) -> u64 {
    let mut h = AHasher::default();
    0x61u8.hash(&mut h); // 'a' for ashlar
    sx.to_bits().hash(&mut h);
    sy.to_bits().hash(&mut h);
    size.hash(&mut h);
    roughness.to_bits().hash(&mut h);
    seed.hash(&mut h);
    thickness.hash(&mut h);
    color.hash(&mut h);
    dbg.hash(&mut h);
    mesh_gen.hash(&mut h);
    h.finish()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn hash_generator_flora_hover(
    sx: f32,
    sy: f32,
    seed: i32,
    height: i32,
    girth: i32,
    wobble: f32,
    taper: f32,
    stem_count: i32,
    cluster_radius: i32,
    branch_count: i32,
    branch_depth: i32,
    branch_start: f32,
    branch_spread: f32,
    braid_strands: i32,
    braid_twist: f32,
    canopy: f32,
    color: u32,
    dbg: bool,
    mesh_gen: u64,
) -> u64 {
    let mut h = AHasher::default();
    0x46u8.hash(&mut h); // 'F' for flora
    sx.to_bits().hash(&mut h);
    sy.to_bits().hash(&mut h);
    seed.hash(&mut h);
    height.hash(&mut h);
    girth.hash(&mut h);
    wobble.to_bits().hash(&mut h);
    taper.to_bits().hash(&mut h);
    stem_count.hash(&mut h);
    cluster_radius.hash(&mut h);
    branch_count.hash(&mut h);
    branch_depth.hash(&mut h);
    branch_start.to_bits().hash(&mut h);
    branch_spread.to_bits().hash(&mut h);
    braid_strands.hash(&mut h);
    braid_twist.to_bits().hash(&mut h);
    canopy.to_bits().hash(&mut h);
    color.hash(&mut h);
    dbg.hash(&mut h);
    mesh_gen.hash(&mut h);
    h.finish()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn hash_generator_insecta_hover(
    sx: f32,
    sy: f32,
    species: &str,
    total_length: i32,
    head_ratio: f32,
    thorax_ratio: f32,
    abdomen_ratio: f32,
    body_half_width: i32,
    body_half_height: i32,
    abdomen_taper: f32,
    head_shape: i32,
    anchor_offset_u: i32,
    anchor_offset_v: i32,
    body_yaw: f32,
    body_arch: f32,
    antenna_length: i32,
    antenna_spread: f32,
    antenna_pitch: f32,
    antenna_root: i32,
    mandible_length: i32,
    mandible_spread: f32,
    mandible_forward: i32,
    wing_shape: i32,
    show_wing_fore: bool,
    wing_fore_length: i32,
    wing_fore_width: i32,
    wing_fore_spread: f32,
    wing_fore_pitch: f32,
    wing_fore_offset: i32,
    wing_fore_forward_cant: f32,
    show_wing_hind: bool,
    wing_hind_length: i32,
    wing_hind_width: i32,
    wing_hind_spread: f32,
    wing_hind_pitch: f32,
    wing_hind_offset: i32,
    color: u32,
    dbg: bool,
    mesh_gen: u64,
) -> u64 {
    let mut h = AHasher::default();
    0x49u8.hash(&mut h); // 'I' for insecta
    sx.to_bits().hash(&mut h);
    sy.to_bits().hash(&mut h);
    species.hash(&mut h);
    total_length.hash(&mut h);
    head_ratio.to_bits().hash(&mut h);
    thorax_ratio.to_bits().hash(&mut h);
    abdomen_ratio.to_bits().hash(&mut h);
    body_half_width.hash(&mut h);
    body_half_height.hash(&mut h);
    abdomen_taper.to_bits().hash(&mut h);
    head_shape.hash(&mut h);
    anchor_offset_u.hash(&mut h);
    anchor_offset_v.hash(&mut h);
    body_yaw.to_bits().hash(&mut h);
    body_arch.to_bits().hash(&mut h);
    antenna_length.hash(&mut h);
    antenna_spread.to_bits().hash(&mut h);
    antenna_pitch.to_bits().hash(&mut h);
    antenna_root.hash(&mut h);
    mandible_length.hash(&mut h);
    mandible_spread.to_bits().hash(&mut h);
    mandible_forward.hash(&mut h);
    wing_shape.hash(&mut h);
    show_wing_fore.hash(&mut h);
    wing_fore_length.hash(&mut h);
    wing_fore_width.hash(&mut h);
    wing_fore_spread.to_bits().hash(&mut h);
    wing_fore_pitch.to_bits().hash(&mut h);
    wing_fore_offset.hash(&mut h);
    wing_fore_forward_cant.to_bits().hash(&mut h);
    show_wing_hind.hash(&mut h);
    wing_hind_length.hash(&mut h);
    wing_hind_width.hash(&mut h);
    wing_hind_spread.to_bits().hash(&mut h);
    wing_hind_pitch.to_bits().hash(&mut h);
    wing_hind_offset.hash(&mut h);
    color.hash(&mut h);
    dbg.hash(&mut h);
    mesh_gen.hash(&mut h);
    h.finish()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn hash_generator_fauna_hover(
    sx: f32,
    sy: f32,
    stance: &str,
    archetype: &str,
    anchor_offset_u: i32,
    anchor_offset_v: i32,
    body_yaw: f32,
    body_arch: f32,
    spine_segments: i32,
    body_length: i32,
    body_half_width: i32,
    body_half_height: i32,
    neck_length: i32,
    neck_half_width: i32,
    neck_half_height: i32,
    head_length: i32,
    head_half_width: i32,
    head_half_height: i32,
    tail_length: i32,
    shoulder_offset_forward: i32,
    hip_offset_forward: i32,
    front_upper_length: i32,
    front_lower_length: i32,
    hind_upper_length: i32,
    hind_lower_length: i32,
    auto_foot_placement: bool,
    color: u32,
    dbg: bool,
    mesh_gen: u64,
) -> u64 {
    let mut h = AHasher::default();
    0x41u8.hash(&mut h); // 'A' for fAuna
    sx.to_bits().hash(&mut h);
    sy.to_bits().hash(&mut h);
    stance.hash(&mut h);
    archetype.hash(&mut h);
    anchor_offset_u.hash(&mut h);
    anchor_offset_v.hash(&mut h);
    body_yaw.to_bits().hash(&mut h);
    body_arch.to_bits().hash(&mut h);
    spine_segments.hash(&mut h);
    body_length.hash(&mut h);
    body_half_width.hash(&mut h);
    body_half_height.hash(&mut h);
    neck_length.hash(&mut h);
    neck_half_width.hash(&mut h);
    neck_half_height.hash(&mut h);
    head_length.hash(&mut h);
    head_half_width.hash(&mut h);
    head_half_height.hash(&mut h);
    tail_length.hash(&mut h);
    shoulder_offset_forward.hash(&mut h);
    hip_offset_forward.hash(&mut h);
    front_upper_length.hash(&mut h);
    front_lower_length.hash(&mut h);
    hind_upper_length.hash(&mut h);
    hind_lower_length.hash(&mut h);
    auto_foot_placement.hash(&mut h);
    color.hash(&mut h);
    dbg.hash(&mut h);
    mesh_gen.hash(&mut h);
    h.finish()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn hash_generator_piscina_hover(
    sx: f32,
    sy: f32,
    seed: i32,
    species: &str,
    length: i32,
    width_param: i32,
    thickness: i32,
    spine_bend: f32,
    spine_s_curve: f32,
    fin_dorsal: i32,
    fin_anal: i32,
    fin_caudal: i32,
    fin_pectoral: i32,
    fin_pelvic: i32,
    fin_adipose: i32,
    show_fin_dorsal: bool,
    show_fin_anal: bool,
    show_fin_caudal: bool,
    show_fin_pectoral: bool,
    show_fin_pelvic: bool,
    show_fin_adipose: bool,
    anchor_offset_u: i32,
    anchor_offset_v: i32,
    color: u32,
    dbg: bool,
    mesh_gen: u64,
) -> u64 {
    let mut h = AHasher::default();
    0x50u8.hash(&mut h); // 'P' for piscina
    sx.to_bits().hash(&mut h);
    sy.to_bits().hash(&mut h);
    seed.hash(&mut h);
    species.hash(&mut h);
    length.hash(&mut h);
    width_param.hash(&mut h);
    thickness.hash(&mut h);
    spine_bend.to_bits().hash(&mut h);
    spine_s_curve.to_bits().hash(&mut h);
    fin_dorsal.hash(&mut h);
    fin_anal.hash(&mut h);
    fin_caudal.hash(&mut h);
    fin_pectoral.hash(&mut h);
    fin_pelvic.hash(&mut h);
    fin_adipose.hash(&mut h);
    show_fin_dorsal.hash(&mut h);
    show_fin_anal.hash(&mut h);
    show_fin_caudal.hash(&mut h);
    show_fin_pectoral.hash(&mut h);
    show_fin_pelvic.hash(&mut h);
    show_fin_adipose.hash(&mut h);
    anchor_offset_u.hash(&mut h);
    anchor_offset_v.hash(&mut h);
    color.hash(&mut h);
    dbg.hash(&mut h);
    mesh_gen.hash(&mut h);
    h.finish()
}

pub(crate) fn hash_generator_roof_hover(
    pins: &[[i32; 3]],
    style: &str,
    height: i32,
    thickness: i32,
    break_ratio: f32,
    wall_height: i32,
    parapet_height: i32,
    salt_skew: f32,
    hollow: bool,
    color: u32,
    dbg: bool,
    mesh_gen: u64,
) -> u64 {
    let mut h = AHasher::default();
    0x52u8.hash(&mut h); // 'R' for roof
    for p in pins {
        p.hash(&mut h);
    }
    style.hash(&mut h);
    height.hash(&mut h);
    thickness.hash(&mut h);
    break_ratio.to_bits().hash(&mut h);
    wall_height.hash(&mut h);
    parapet_height.hash(&mut h);
    salt_skew.to_bits().hash(&mut h);
    hollow.hash(&mut h);
    color.hash(&mut h);
    dbg.hash(&mut h);
    mesh_gen.hash(&mut h);
    h.finish()
}

pub(crate) fn prepare_preview_mesh(
    state: &ViewerState,
    cam: &OrbitCamera,
    viewport_w: u32,
    viewport_h: u32,
) -> PreviewMeshPrepared {
    if state
        .file
        .stroke_preview_suppresses_hover
        .load(Ordering::Relaxed)
    {
        return PreviewMeshPrepared::Noop;
    }
    let dbg = state
        .gpu
        .viewport_cursor_debug_overlay
        .load(Ordering::Relaxed);
    let (cursor, mode) = {
        let c = state.preview.preview_cursor.lock();
        let m = state.preview.preview_mode.lock();
        (*c, *m)
    };

    if matches!(mode, PreviewMode::Navigate | PreviewMode::Fly) {
        return PreviewMeshPrepared::Clear;
    }

    // Pin-based generators (cloth, roof) don't need a cursor position — run
    // them even when the mouse is outside the viewport so the preview persists.
    if cursor.is_none() && matches!(mode, PreviewMode::Add) {
        let file_guard = state.file.current_file.lock();
        let map_guard = state.file.voxel_map.lock();
        if let (Some(file), Some(vmap)) = (file_guard.as_ref(), map_guard.as_ref()) {
            let hover = state.preview.preview_hover.lock();
            let ctx = &*hover;
            let mesh_gen = state.gpu.mesh_refresh_generation.load(Ordering::Relaxed);
            if let Some(ref gk) = ctx.generator_kind {
                match gk.as_str() {
                    "cloth" => {
                        if ctx.generator_cloth_pins.len() >= 3 {
                            let sim = crate::generators::ClothSimOptions {
                                gravity_scale: ctx.generator_cloth_gravity_scale.max(0.0),
                                stiffness_scale: ctx
                                    .generator_cloth_stiffness_scale
                                    .clamp(0.05, 2.0),
                                iterations: if ctx.generator_cloth_iterations > 0 {
                                    Some(ctx.generator_cloth_iterations.clamp(4, 96))
                                } else {
                                    None
                                },
                                constraint_passes: ctx
                                    .generator_cloth_constraint_passes
                                    .clamp(1, 6),
                            };
                            let mut cells = crate::generators::preview_cloth_voxels(
                                &ctx.generator_cloth_pins,
                                ctx.generator_cloth_tension,
                                ctx.generator_cloth_gravity_direction.as_str(),
                                ctx.brush_radius,
                                ctx.brush_shape,
                                &sim,
                            );
                            voxel_edit::extend_with_mirror_targets(&mut cells, ctx.mirror_axes);
                            if !cells.is_empty() {
                                let key = hash_generator_cloth_hover(
                                    &ctx.generator_cloth_pins,
                                    ctx.generator_cloth_tension,
                                    ctx.generator_cloth_gravity_direction.as_str(),
                                    ctx.generator_cloth_gravity_scale,
                                    ctx.generator_cloth_stiffness_scale,
                                    ctx.generator_cloth_iterations,
                                    ctx.generator_cloth_constraint_passes,
                                    ctx.brush_radius,
                                    ctx.brush_shape,
                                    ctx.color,
                                    dbg,
                                    mesh_gen,
                                );
                                if preview_overlay_cache_key_get(state) == Some(key) {
                                    return PreviewMeshPrepared::Noop;
                                }
                                let set: AHashSet<_> = cells.iter().copied().collect();
                                let instanced = stroke_preview_meshes_for_union(
                                    voxel_edit::EditTool::Add,
                                    &set,
                                    vmap,
                                    file,
                                    dbg,
                                    ctx.color,
                                    None,
                                );
                                return PreviewMeshPrepared::Upload {
                                    cache_key: key,
                                    instanced,
                                };
                            }
                        }
                    }
                    "roof" => {
                        if !ctx.generator_roof_pins.is_empty() {
                            let mut instanced = if ctx.generator_roof_pins.len() >= 3 {
                                let mut cells = crate::generators::preview_roof_voxels(
                                    &ctx.generator_roof_pins,
                                    &ctx.generator_roof_style,
                                    ctx.generator_roof_height,
                                    ctx.generator_roof_thickness,
                                    0,
                                    0,
                                    ctx.generator_roof_break_ratio,
                                    ctx.generator_roof_wall_height,
                                    ctx.generator_roof_parapet_height,
                                    ctx.generator_roof_salt_skew,
                                    ctx.generator_roof_hollow,
                                );
                                voxel_edit::extend_with_mirror_targets(&mut cells, ctx.mirror_axes);
                                if !cells.is_empty() {
                                    let set: AHashSet<_> = cells.iter().copied().collect();
                                    stroke_preview_meshes_for_union(
                                        voxel_edit::EditTool::Add,
                                        &set,
                                        vmap,
                                        file,
                                        dbg,
                                        ctx.color,
                                        None,
                                    )
                                } else {
                                    greedy_mesh::PreviewInstancedResult::empty()
                                }
                            } else {
                                greedy_mesh::PreviewInstancedResult::empty()
                            };
                            // Yellow markers at each pin position.
                            append_polygon_vertex_marker_meshes(
                                &mut instanced.extra_solid,
                                &mut instanced.extra_wire,
                                &ctx.generator_roof_pins,
                                vmap,
                                file,
                                dbg,
                            );
                            let key = hash_generator_roof_hover(
                                &ctx.generator_roof_pins,
                                &ctx.generator_roof_style,
                                ctx.generator_roof_height,
                                ctx.generator_roof_thickness,
                                ctx.generator_roof_break_ratio,
                                ctx.generator_roof_wall_height,
                                ctx.generator_roof_parapet_height,
                                ctx.generator_roof_salt_skew,
                                ctx.generator_roof_hollow,
                                ctx.color,
                                dbg,
                                mesh_gen,
                            );
                            if preview_overlay_cache_key_get(state) == Some(key) {
                                return PreviewMeshPrepared::Noop;
                            }
                            return PreviewMeshPrepared::Upload {
                                cache_key: key,
                                instanced,
                            };
                        }
                    }
                    "shape" => {
                        // Shape with gizmo center — cursor is None but we can
                        // still render at the gizmo position.
                        let gen_center = *state.gizmos.generator_gizmo_center.lock();
                        if let Some([gx, gy, gz]) = gen_center {
                            let shape = StartShape::from_str_id(&ctx.generator_shape_kind);
                            let (pdx, pdy, pdz) = crate::frame_loop::pending_gizmo_translate(state);
                            let origin = (gx as i32 + pdx, gy as i32 + pdy, gz as i32 + pdz);
                            let all = crate::generators::compute_shape_positions(
                                shape,
                                ctx.generator_shape_size,
                                origin,
                                (
                                    ctx.generator_shape_rot_x,
                                    ctx.generator_shape_rot_y,
                                    ctx.generator_shape_rot_z,
                                ),
                            );
                            let mut cells: Vec<_> = if ctx.generator_shape_overwrite {
                                all
                            } else {
                                all.into_iter().filter(|c| !vmap.contains_key(c)).collect()
                            };
                            voxel_edit::extend_with_mirror_targets(&mut cells, ctx.mirror_axes);
                            if !cells.is_empty() {
                                let key = hash_generator_shape_hover(
                                    f32::from_bits((origin.0) as u32),
                                    f32::from_bits((origin.1 as u32).wrapping_add(origin.2 as u32)),
                                    &ctx.generator_shape_kind,
                                    ctx.generator_shape_size,
                                    ctx.generator_shape_rot_x,
                                    ctx.generator_shape_rot_y,
                                    ctx.generator_shape_rot_z,
                                    ctx.generator_shape_overwrite,
                                    ctx.color,
                                    dbg,
                                    mesh_gen,
                                );
                                if preview_overlay_cache_key_get(state) == Some(key) {
                                    return PreviewMeshPrepared::Noop;
                                }
                                const NBRS: [(i32, i32, i32); 6] = [
                                    (1, 0, 0),
                                    (-1, 0, 0),
                                    (0, 1, 0),
                                    (0, -1, 0),
                                    (0, 0, 1),
                                    (0, 0, -1),
                                ];
                                let set: AHashSet<_> = cells.iter().copied().collect();
                                let visible: AHashSet<_> = set
                                    .iter()
                                    .filter(|&&(x, y, z)| {
                                        NBRS.iter().any(|&(dx, dy, dz)| {
                                            !set.contains(&(x + dx, y + dy, z + dz))
                                        })
                                    })
                                    .copied()
                                    .collect();
                                let instanced = stroke_preview_meshes_for_union(
                                    voxel_edit::EditTool::Add,
                                    &visible,
                                    vmap,
                                    file,
                                    dbg,
                                    ctx.color,
                                    None,
                                );
                                return PreviewMeshPrepared::GenUpload {
                                    cache_key: key,
                                    instanced,
                                };
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        return PreviewMeshPrepared::Clear;
    }

    // Bone preview: session-based, doesn't need cursor position.
    if matches!(mode, PreviewMode::Bone) {
        let gizmo_drag = state.gizmos.bone_gizmo_drag.lock().is_some();
        let max_v = if gizmo_drag { 12_000 } else { 24_000 };
        let session_snap = state.gizmos.bone_session.lock().clone();
        let hover = state.preview.preview_hover.lock();
        let (csx, csy) = cursor
            .map(|(nx, ny)| viewport_texels_from_norm(nx, ny, viewport_w as f32, viewport_h as f32))
            .unwrap_or((-1.0, -1.0));
        let key = hash_bone_preview(&session_snap, csx, csy, gizmo_drag, dbg, hover.color);
        if preview_overlay_cache_key_get(state) == Some(key) {
            return PreviewMeshPrepared::Noop;
        }
        if session_snap.joints.is_empty() {
            return PreviewMeshPrepared::Clear;
        }
        let file_guard = state.file.current_file.lock();
        let map_guard = state.file.voxel_map.lock();
        let cam = state.cam.camera.lock();
        if let (Some(file), Some(vmap)) = (file_guard.as_ref(), map_guard.as_ref()) {
            let coords = generators::bone_voxel_coords_for_session(
                &session_snap,
                file.grid_size.max(1),
                max_v,
            );
            let set: AHashSet<_> = coords.iter().copied().collect();
            let mut instanced = stroke_preview_meshes_for_union(
                voxel_edit::EditTool::Add,
                &set,
                vmap,
                file,
                dbg,
                hover.color,
                None,
            );
            generators::append_bone_skeleton_wire(&session_snap, &cam, &mut instanced.extra_wire);
            // Move gizmo is rendered by the shared gizmo system (generator_gizmo_center).
            return PreviewMeshPrepared::Upload {
                cache_key: key,
                instanced,
            };
        }
        return PreviewMeshPrepared::Clear;
    }

    let Some((nx, ny)) = cursor else {
        return PreviewMeshPrepared::Clear;
    };

    let file_guard = state.file.current_file.lock();
    let map_guard = state.file.voxel_map.lock();
    let Some(file) = file_guard.as_ref() else {
        return PreviewMeshPrepared::Clear;
    };
    let Some(vmap) = map_guard.as_ref() else {
        return PreviewMeshPrepared::Clear;
    };

    let w = viewport_w as f32;
    let h = viewport_h as f32;
    let (sx, sy) = viewport_texels_from_norm(nx, ny, w, h);

    if matches!(mode, PreviewMode::Squishy) {
        let hover = state.preview.preview_hover.lock();
        let gizmo_drag = state.gizmos.squishy_gizmo_drag.lock().is_some();

        let session_snap = state.gizmos.squishy_session.lock().clone();

        let add_anchor = if session_snap.mode == generators::SquishyMode::Add {
            if session_snap.add_snap_to_surface {
                voxel_edit::preview_add_cell(file, vmap, cam, w, h, sx, sy).map(|(c, _)| c)
            } else {
                voxel_edit::pick_solid_coord_at_screen(file, vmap, cam, w, h, sx, sy)
            }
        } else {
            None
        };

        let delete_hover_id = if session_snap.mode == generators::SquishyMode::Delete {
            generators::pick_metaball_at_screen(&session_snap, cam, w, h, sx, sy)
        } else {
            None
        };

        let key = hash_squishy_preview(
            &session_snap,
            sx,
            sy,
            add_anchor,
            gizmo_drag,
            delete_hover_id,
            dbg,
            hover.color,
        );
        if preview_overlay_cache_key_get(state) == Some(key) {
            return PreviewMeshPrepared::Noop;
        }

        let mut temp_session = session_snap.clone();
        temp_session.hollow = false;

        let mut coords = generators::voxel_coords_for_session_with_limit(
            &temp_session,
            file.grid_size.max(1),
            usize::MAX,
        );

        // In add mode show a single cursor voxel instead of a full metaball blob.
        if session_snap.mode == generators::SquishyMode::Add {
            if let Some((ax, ay, az)) = add_anchor {
                coords.push((ax, ay, az));
            }
        }

        let has_pick_chrome = !session_snap.balls.is_empty()
            || (session_snap.mode == generators::SquishyMode::Add && add_anchor.is_some());

        if coords.is_empty() && !has_pick_chrome {
            return PreviewMeshPrepared::Clear;
        }

        let set: AHashSet<_> = coords.iter().copied().collect();
        let mut instanced = stroke_preview_meshes_for_union(
            voxel_edit::EditTool::Add,
            &set,
            vmap,
            file,
            dbg,
            hover.color,
            None,
        );

        if has_pick_chrome {
            generators::append_squishy_metaball_pick_rings(
                &mut instanced.extra_wire,
                &session_snap,
                delete_hover_id,
            );
        }

        return PreviewMeshPrepared::Upload {
            cache_key: key,
            instanced,
        };
    }

    let hover = state.preview.preview_hover.lock();
    let ctx = &*hover;

    if matches!(mode, PreviewMode::Add) {
        if let Some(ref gk) = ctx.generator_kind {
            let mesh_gen = state.gpu.mesh_refresh_generation.load(Ordering::Relaxed);
            match gk.as_str() {
                "rope" => {
                    if let Some([vx1, vy1, vz1]) = ctx.generator_rope_first_voxel {
                        let h1 = (vx1, vy1, vz1);
                        let mut cells = crate::generators::preview_rope_voxels_between_screens(
                            file,
                            vmap,
                            cam,
                            w,
                            h,
                            h1,
                            sx,
                            sy,
                            ctx.generator_rope_tension,
                            ctx.brush_radius,
                            ctx.brush_shape,
                            &ctx.generator_rope_gravity_direction,
                        );
                        voxel_edit::extend_with_mirror_targets(&mut cells, ctx.mirror_axes);
                        if !cells.is_empty() {
                            let key = hash_generator_rope_hover(
                                [vx1, vy1, vz1],
                                sx,
                                sy,
                                ctx.generator_rope_tension,
                                &ctx.generator_rope_gravity_direction,
                                ctx.brush_radius,
                                ctx.brush_shape,
                                ctx.color,
                                dbg,
                                mesh_gen,
                            );
                            if preview_overlay_cache_key_get(state) == Some(key) {
                                return PreviewMeshPrepared::Noop;
                            }
                            let set: AHashSet<_> = cells.iter().copied().collect();
                            let instanced = stroke_preview_meshes_for_union(
                                voxel_edit::EditTool::Add,
                                &set,
                                vmap,
                                file,
                                dbg,
                                ctx.color,
                                None,
                            );
                            return PreviewMeshPrepared::Upload {
                                cache_key: key,
                                instanced,
                            };
                        }
                    }
                }
                "cloth" => {
                    if ctx.generator_cloth_pins.len() >= 3 {
                        let sim = crate::generators::ClothSimOptions {
                            gravity_scale: ctx.generator_cloth_gravity_scale.max(0.0),
                            stiffness_scale: ctx.generator_cloth_stiffness_scale.clamp(0.05, 2.0),
                            iterations: if ctx.generator_cloth_iterations > 0 {
                                Some(ctx.generator_cloth_iterations.clamp(4, 96))
                            } else {
                                None
                            },
                            constraint_passes: ctx.generator_cloth_constraint_passes.clamp(1, 6),
                        };
                        let mut cells = crate::generators::preview_cloth_voxels(
                            &ctx.generator_cloth_pins,
                            ctx.generator_cloth_tension,
                            ctx.generator_cloth_gravity_direction.as_str(),
                            ctx.brush_radius,
                            ctx.brush_shape,
                            &sim,
                        );
                        voxel_edit::extend_with_mirror_targets(&mut cells, ctx.mirror_axes);
                        if !cells.is_empty() {
                            let key = hash_generator_cloth_hover(
                                &ctx.generator_cloth_pins,
                                ctx.generator_cloth_tension,
                                ctx.generator_cloth_gravity_direction.as_str(),
                                ctx.generator_cloth_gravity_scale,
                                ctx.generator_cloth_stiffness_scale,
                                ctx.generator_cloth_iterations,
                                ctx.generator_cloth_constraint_passes,
                                ctx.brush_radius,
                                ctx.brush_shape,
                                ctx.color,
                                dbg,
                                mesh_gen,
                            );
                            if preview_overlay_cache_key_get(state) == Some(key) {
                                return PreviewMeshPrepared::Noop;
                            }
                            let set: AHashSet<_> = cells.iter().copied().collect();
                            let instanced = stroke_preview_meshes_for_union(
                                voxel_edit::EditTool::Add,
                                &set,
                                vmap,
                                file,
                                dbg,
                                ctx.color,
                                None,
                            );
                            return PreviewMeshPrepared::Upload {
                                cache_key: key,
                                instanced,
                            };
                        }
                    }
                }
                "rocks" => {
                    let mut cells = crate::generators::preview_rock_at_screen(
                        file,
                        vmap,
                        cam,
                        w,
                        h,
                        sx,
                        sy,
                        ctx.generator_rock_seed,
                        ctx.generator_rock_size,
                        ctx.generator_rock_roughness,
                        ctx.generator_rock_count,
                        ctx.generator_rock_cluster_radius,
                        ctx.generator_rock_sink_direction,
                        ctx.generator_rock_sink_amount,
                    );
                    voxel_edit::extend_with_mirror_targets(&mut cells, ctx.mirror_axes);
                    if !cells.is_empty() {
                        let key = hash_generator_rock_hover(
                            sx,
                            sy,
                            ctx.generator_rock_size,
                            ctx.generator_rock_roughness,
                            ctx.generator_rock_seed,
                            ctx.color,
                            dbg,
                            mesh_gen,
                            ctx.generator_rock_count,
                            ctx.generator_rock_cluster_radius,
                            ctx.generator_rock_sink_direction,
                            ctx.generator_rock_sink_amount,
                        );
                        if preview_overlay_cache_key_get(state) == Some(key) {
                            return PreviewMeshPrepared::Noop;
                        }
                        const NBRS: [(i32, i32, i32); 6] = [
                            (1, 0, 0),
                            (-1, 0, 0),
                            (0, 1, 0),
                            (0, -1, 0),
                            (0, 0, 1),
                            (0, 0, -1),
                        ];
                        let set: AHashSet<_> = cells.iter().copied().collect();
                        let visible: AHashSet<_> = set
                            .iter()
                            .filter(|&&(x, y, z)| {
                                NBRS.iter()
                                    .any(|&(dx, dy, dz)| !set.contains(&(x + dx, y + dy, z + dz)))
                            })
                            .copied()
                            .collect();
                        let instanced = stroke_preview_meshes_for_union(
                            voxel_edit::EditTool::Add,
                            &visible,
                            vmap,
                            file,
                            dbg,
                            ctx.color,
                            None,
                        );
                        return PreviewMeshPrepared::GenUpload {
                            cache_key: key,
                            instanced,
                        };
                    }
                }
                "grass" => {
                    let mut cells = crate::generators::preview_grass_at_screen(
                        file,
                        vmap,
                        cam,
                        w,
                        h,
                        sx,
                        sy,
                        ctx.generator_grass_seed,
                        ctx.generator_grass_radius,
                        ctx.generator_grass_density,
                        ctx.generator_grass_max_height,
                    );
                    voxel_edit::extend_with_mirror_targets(&mut cells, ctx.mirror_axes);
                    if !cells.is_empty() {
                        let key = hash_generator_grass_hover(
                            sx,
                            sy,
                            ctx.generator_grass_radius,
                            ctx.generator_grass_density,
                            ctx.generator_grass_max_height,
                            ctx.generator_grass_seed,
                            ctx.color,
                            dbg,
                            mesh_gen,
                        );
                        if preview_overlay_cache_key_get(state) == Some(key) {
                            return PreviewMeshPrepared::Noop;
                        }
                        const NBRS: [(i32, i32, i32); 6] = [
                            (1, 0, 0),
                            (-1, 0, 0),
                            (0, 1, 0),
                            (0, -1, 0),
                            (0, 0, 1),
                            (0, 0, -1),
                        ];
                        let set: AHashSet<_> = cells.iter().copied().collect();
                        let visible: AHashSet<_> = set
                            .iter()
                            .filter(|&&(x, y, z)| {
                                NBRS.iter()
                                    .any(|&(dx, dy, dz)| !set.contains(&(x + dx, y + dy, z + dz)))
                            })
                            .copied()
                            .collect();
                        let instanced = stroke_preview_meshes_for_union(
                            voxel_edit::EditTool::Add,
                            &visible,
                            vmap,
                            file,
                            dbg,
                            ctx.color,
                            None,
                        );
                        return PreviewMeshPrepared::GenUpload {
                            cache_key: key,
                            instanced,
                        };
                    }
                }
                "ashlar" => {
                    let mut cells = crate::generators::preview_ashlar_at_screen(
                        file,
                        vmap,
                        cam,
                        w,
                        h,
                        sx,
                        sy,
                        ctx.generator_ashlar_seed,
                        ctx.generator_ashlar_size,
                        ctx.generator_ashlar_roughness,
                        ctx.generator_ashlar_thickness,
                    );
                    voxel_edit::extend_with_mirror_targets(&mut cells, ctx.mirror_axes);
                    if !cells.is_empty() {
                        let key = hash_generator_ashlar_hover(
                            sx,
                            sy,
                            ctx.generator_ashlar_size,
                            ctx.generator_ashlar_roughness,
                            ctx.generator_ashlar_seed,
                            ctx.generator_ashlar_thickness,
                            ctx.color,
                            dbg,
                            mesh_gen,
                        );
                        if preview_overlay_cache_key_get(state) == Some(key) {
                            return PreviewMeshPrepared::Noop;
                        }
                        const NBRS: [(i32, i32, i32); 6] = [
                            (1, 0, 0),
                            (-1, 0, 0),
                            (0, 1, 0),
                            (0, -1, 0),
                            (0, 0, 1),
                            (0, 0, -1),
                        ];
                        let set: AHashSet<_> = cells.iter().copied().collect();
                        let visible: AHashSet<_> = set
                            .iter()
                            .filter(|&&(x, y, z)| {
                                NBRS.iter()
                                    .any(|&(dx, dy, dz)| !set.contains(&(x + dx, y + dy, z + dz)))
                            })
                            .copied()
                            .collect();
                        let instanced = stroke_preview_meshes_for_union(
                            voxel_edit::EditTool::Add,
                            &visible,
                            vmap,
                            file,
                            dbg,
                            ctx.color,
                            None,
                        );
                        return PreviewMeshPrepared::GenUpload {
                            cache_key: key,
                            instanced,
                        };
                    }
                }
                "flora" => {
                    let material = voxelle::MaterialId::from_str_id(&ctx.material);
                    let mut cells = crate::generators::preview_flora_at_screen(
                        file,
                        vmap,
                        cam,
                        w,
                        h,
                        sx,
                        sy,
                        ctx.generator_flora_seed,
                        ctx.generator_flora_height,
                        ctx.generator_flora_girth,
                        ctx.generator_flora_wobble,
                        ctx.generator_flora_taper,
                        ctx.generator_flora_stem_count,
                        ctx.generator_flora_cluster_radius,
                        ctx.generator_flora_branch_count,
                        ctx.generator_flora_branch_depth,
                        ctx.generator_flora_branch_start,
                        ctx.generator_flora_branch_spread,
                        ctx.generator_flora_braid_strands,
                        ctx.generator_flora_braid_twist,
                        ctx.generator_flora_canopy,
                        ctx.color,
                        material,
                    );
                    voxel_edit::extend_with_mirror_targets_colored(&mut cells, ctx.mirror_axes);
                    if !cells.is_empty() {
                        let key = hash_generator_flora_hover(
                            sx,
                            sy,
                            ctx.generator_flora_seed,
                            ctx.generator_flora_height,
                            ctx.generator_flora_girth,
                            ctx.generator_flora_wobble,
                            ctx.generator_flora_taper,
                            ctx.generator_flora_stem_count,
                            ctx.generator_flora_cluster_radius,
                            ctx.generator_flora_branch_count,
                            ctx.generator_flora_branch_depth,
                            ctx.generator_flora_branch_start,
                            ctx.generator_flora_branch_spread,
                            ctx.generator_flora_braid_strands,
                            ctx.generator_flora_braid_twist,
                            ctx.generator_flora_canopy,
                            ctx.color,
                            dbg,
                            mesh_gen,
                        );
                        if preview_overlay_cache_key_get(state) == Some(key) {
                            return PreviewMeshPrepared::Noop;
                        }
                        const NBRS: [(i32, i32, i32); 6] = [
                            (1, 0, 0),
                            (-1, 0, 0),
                            (0, 1, 0),
                            (0, -1, 0),
                            (0, 0, 1),
                            (0, 0, -1),
                        ];
                        let color_map: AHashMap<(i32, i32, i32), u32> =
                            cells.iter().cloned().collect();
                        let visible: AHashSet<_> = color_map
                            .keys()
                            .filter(|&&(x, y, z)| {
                                NBRS.iter().any(|&(dx, dy, dz)| {
                                    !color_map.contains_key(&(x + dx, y + dy, z + dz))
                                })
                            })
                            .copied()
                            .collect();
                        let fallback = ctx.color;
                        let instanced = stroke_preview_meshes_for_union(
                            voxel_edit::EditTool::Add,
                            &visible,
                            vmap,
                            file,
                            dbg,
                            fallback,
                            Some(&|x, y, z| *color_map.get(&(x, y, z)).unwrap_or(&fallback)),
                        );
                        return PreviewMeshPrepared::GenUpload {
                            cache_key: key,
                            instanced,
                        };
                    }
                }
                "insecta" => {
                    let material = voxelle::MaterialId::from_str_id(&ctx.material);
                    let mut cells = crate::generators::preview_insecta_at_screen(
                        file,
                        vmap,
                        cam,
                        w,
                        h,
                        sx,
                        sy,
                        &ctx.generator_insecta_species,
                        ctx.generator_insecta_total_length,
                        ctx.generator_insecta_head_ratio,
                        ctx.generator_insecta_thorax_ratio,
                        ctx.generator_insecta_abdomen_ratio,
                        ctx.generator_insecta_body_half_width,
                        ctx.generator_insecta_body_half_height,
                        ctx.generator_insecta_abdomen_taper,
                        ctx.generator_insecta_head_shape,
                        ctx.generator_insecta_anchor_offset_u,
                        ctx.generator_insecta_anchor_offset_v,
                        ctx.generator_insecta_body_yaw,
                        ctx.generator_insecta_body_arch,
                        ctx.generator_insecta_antenna_length,
                        ctx.generator_insecta_antenna_spread,
                        ctx.generator_insecta_antenna_pitch,
                        ctx.generator_insecta_antenna_root,
                        ctx.generator_insecta_mandible_length,
                        ctx.generator_insecta_mandible_spread,
                        ctx.generator_insecta_mandible_forward,
                        ctx.generator_insecta_wing_shape,
                        ctx.generator_insecta_show_wing_fore,
                        ctx.generator_insecta_wing_fore_length,
                        ctx.generator_insecta_wing_fore_width,
                        ctx.generator_insecta_wing_fore_spread,
                        ctx.generator_insecta_wing_fore_pitch,
                        ctx.generator_insecta_wing_fore_offset,
                        ctx.generator_insecta_wing_fore_forward_cant,
                        ctx.generator_insecta_show_wing_hind,
                        ctx.generator_insecta_wing_hind_length,
                        ctx.generator_insecta_wing_hind_width,
                        ctx.generator_insecta_wing_hind_spread,
                        ctx.generator_insecta_wing_hind_pitch,
                        ctx.generator_insecta_wing_hind_offset,
                        ctx.color,
                        material,
                    );
                    voxel_edit::extend_with_mirror_targets_colored(&mut cells, ctx.mirror_axes);
                    if !cells.is_empty() {
                        let key = hash_generator_insecta_hover(
                            sx,
                            sy,
                            &ctx.generator_insecta_species,
                            ctx.generator_insecta_total_length,
                            ctx.generator_insecta_head_ratio,
                            ctx.generator_insecta_thorax_ratio,
                            ctx.generator_insecta_abdomen_ratio,
                            ctx.generator_insecta_body_half_width,
                            ctx.generator_insecta_body_half_height,
                            ctx.generator_insecta_abdomen_taper,
                            ctx.generator_insecta_head_shape,
                            ctx.generator_insecta_anchor_offset_u,
                            ctx.generator_insecta_anchor_offset_v,
                            ctx.generator_insecta_body_yaw,
                            ctx.generator_insecta_body_arch,
                            ctx.generator_insecta_antenna_length,
                            ctx.generator_insecta_antenna_spread,
                            ctx.generator_insecta_antenna_pitch,
                            ctx.generator_insecta_antenna_root,
                            ctx.generator_insecta_mandible_length,
                            ctx.generator_insecta_mandible_spread,
                            ctx.generator_insecta_mandible_forward,
                            ctx.generator_insecta_wing_shape,
                            ctx.generator_insecta_show_wing_fore,
                            ctx.generator_insecta_wing_fore_length,
                            ctx.generator_insecta_wing_fore_width,
                            ctx.generator_insecta_wing_fore_spread,
                            ctx.generator_insecta_wing_fore_pitch,
                            ctx.generator_insecta_wing_fore_offset,
                            ctx.generator_insecta_wing_fore_forward_cant,
                            ctx.generator_insecta_show_wing_hind,
                            ctx.generator_insecta_wing_hind_length,
                            ctx.generator_insecta_wing_hind_width,
                            ctx.generator_insecta_wing_hind_spread,
                            ctx.generator_insecta_wing_hind_pitch,
                            ctx.generator_insecta_wing_hind_offset,
                            ctx.color,
                            dbg,
                            mesh_gen,
                        );
                        if preview_overlay_cache_key_get(state) == Some(key) {
                            return PreviewMeshPrepared::Noop;
                        }
                        const NBRS: [(i32, i32, i32); 6] = [
                            (1, 0, 0),
                            (-1, 0, 0),
                            (0, 1, 0),
                            (0, -1, 0),
                            (0, 0, 1),
                            (0, 0, -1),
                        ];
                        let color_map: AHashMap<(i32, i32, i32), u32> =
                            cells.iter().cloned().collect();
                        let visible: AHashSet<_> = color_map
                            .keys()
                            .filter(|&&(x, y, z)| {
                                NBRS.iter().any(|&(dx, dy, dz)| {
                                    !color_map.contains_key(&(x + dx, y + dy, z + dz))
                                })
                            })
                            .copied()
                            .collect();
                        let fallback = ctx.color;
                        let instanced = stroke_preview_meshes_for_union(
                            voxel_edit::EditTool::Add,
                            &visible,
                            vmap,
                            file,
                            dbg,
                            fallback,
                            Some(&|x, y, z| *color_map.get(&(x, y, z)).unwrap_or(&fallback)),
                        );
                        return PreviewMeshPrepared::GenUpload {
                            cache_key: key,
                            instanced,
                        };
                    }
                }
                "fauna" => {
                    let material = voxelle::MaterialId::from_str_id(&ctx.material);
                    let mut cells = crate::generators::preview_fauna_at_screen(
                        file,
                        vmap,
                        cam,
                        w,
                        h,
                        sx,
                        sy,
                        &ctx.generator_fauna_stance,
                        &ctx.generator_fauna_archetype,
                        ctx.generator_fauna_anchor_offset_u,
                        ctx.generator_fauna_anchor_offset_v,
                        ctx.generator_fauna_body_yaw,
                        ctx.generator_fauna_body_arch,
                        ctx.generator_fauna_spine_segments,
                        ctx.generator_fauna_body_length,
                        ctx.generator_fauna_body_half_width,
                        ctx.generator_fauna_body_half_height,
                        ctx.generator_fauna_neck_length,
                        ctx.generator_fauna_neck_half_width,
                        ctx.generator_fauna_neck_half_height,
                        ctx.generator_fauna_head_length,
                        ctx.generator_fauna_head_half_width,
                        ctx.generator_fauna_head_half_height,
                        ctx.generator_fauna_tail_length,
                        ctx.generator_fauna_shoulder_offset_forward,
                        ctx.generator_fauna_hip_offset_forward,
                        ctx.generator_fauna_front_upper_length,
                        ctx.generator_fauna_front_lower_length,
                        ctx.generator_fauna_hind_upper_length,
                        ctx.generator_fauna_hind_lower_length,
                        ctx.generator_fauna_auto_foot_placement,
                        ctx.color,
                        material,
                    );
                    voxel_edit::extend_with_mirror_targets_colored(&mut cells, ctx.mirror_axes);
                    if !cells.is_empty() {
                        let key = hash_generator_fauna_hover(
                            sx,
                            sy,
                            &ctx.generator_fauna_stance,
                            &ctx.generator_fauna_archetype,
                            ctx.generator_fauna_anchor_offset_u,
                            ctx.generator_fauna_anchor_offset_v,
                            ctx.generator_fauna_body_yaw,
                            ctx.generator_fauna_body_arch,
                            ctx.generator_fauna_spine_segments,
                            ctx.generator_fauna_body_length,
                            ctx.generator_fauna_body_half_width,
                            ctx.generator_fauna_body_half_height,
                            ctx.generator_fauna_neck_length,
                            ctx.generator_fauna_neck_half_width,
                            ctx.generator_fauna_neck_half_height,
                            ctx.generator_fauna_head_length,
                            ctx.generator_fauna_head_half_width,
                            ctx.generator_fauna_head_half_height,
                            ctx.generator_fauna_tail_length,
                            ctx.generator_fauna_shoulder_offset_forward,
                            ctx.generator_fauna_hip_offset_forward,
                            ctx.generator_fauna_front_upper_length,
                            ctx.generator_fauna_front_lower_length,
                            ctx.generator_fauna_hind_upper_length,
                            ctx.generator_fauna_hind_lower_length,
                            ctx.generator_fauna_auto_foot_placement,
                            ctx.color,
                            dbg,
                            mesh_gen,
                        );
                        if preview_overlay_cache_key_get(state) == Some(key) {
                            return PreviewMeshPrepared::Noop;
                        }
                        const NBRS: [(i32, i32, i32); 6] = [
                            (1, 0, 0),
                            (-1, 0, 0),
                            (0, 1, 0),
                            (0, -1, 0),
                            (0, 0, 1),
                            (0, 0, -1),
                        ];
                        let color_map: AHashMap<(i32, i32, i32), u32> =
                            cells.iter().cloned().collect();
                        let visible: AHashSet<_> = color_map
                            .keys()
                            .filter(|&&(x, y, z)| {
                                NBRS.iter().any(|&(dx, dy, dz)| {
                                    !color_map.contains_key(&(x + dx, y + dy, z + dz))
                                })
                            })
                            .copied()
                            .collect();
                        let fallback = ctx.color;
                        let instanced = stroke_preview_meshes_for_union(
                            voxel_edit::EditTool::Add,
                            &visible,
                            vmap,
                            file,
                            dbg,
                            fallback,
                            Some(&|x, y, z| *color_map.get(&(x, y, z)).unwrap_or(&fallback)),
                        );
                        return PreviewMeshPrepared::GenUpload {
                            cache_key: key,
                            instanced,
                        };
                    }
                }
                "piscina" => {
                    let material = voxelle::MaterialId::from_str_id(&ctx.material);
                    let mut cells = crate::generators::preview_piscina_at_screen(
                        file,
                        vmap,
                        cam,
                        w,
                        h,
                        sx,
                        sy,
                        ctx.generator_piscina_seed,
                        &ctx.generator_piscina_species,
                        ctx.generator_piscina_length,
                        ctx.generator_piscina_width,
                        ctx.generator_piscina_thickness,
                        ctx.generator_piscina_spine_bend,
                        ctx.generator_piscina_spine_s_curve,
                        ctx.generator_piscina_fin_dorsal,
                        ctx.generator_piscina_fin_anal,
                        ctx.generator_piscina_fin_caudal,
                        ctx.generator_piscina_fin_pectoral,
                        ctx.generator_piscina_fin_pelvic,
                        ctx.generator_piscina_fin_adipose,
                        ctx.generator_piscina_show_fin_dorsal,
                        ctx.generator_piscina_show_fin_anal,
                        ctx.generator_piscina_show_fin_caudal,
                        ctx.generator_piscina_show_fin_pectoral,
                        ctx.generator_piscina_show_fin_pelvic,
                        ctx.generator_piscina_show_fin_adipose,
                        ctx.generator_piscina_anchor_offset_u,
                        ctx.generator_piscina_anchor_offset_v,
                        ctx.color,
                        material,
                    );
                    voxel_edit::extend_with_mirror_targets_colored(&mut cells, ctx.mirror_axes);
                    if !cells.is_empty() {
                        let key = hash_generator_piscina_hover(
                            sx,
                            sy,
                            ctx.generator_piscina_seed,
                            &ctx.generator_piscina_species,
                            ctx.generator_piscina_length,
                            ctx.generator_piscina_width,
                            ctx.generator_piscina_thickness,
                            ctx.generator_piscina_spine_bend,
                            ctx.generator_piscina_spine_s_curve,
                            ctx.generator_piscina_fin_dorsal,
                            ctx.generator_piscina_fin_anal,
                            ctx.generator_piscina_fin_caudal,
                            ctx.generator_piscina_fin_pectoral,
                            ctx.generator_piscina_fin_pelvic,
                            ctx.generator_piscina_fin_adipose,
                            ctx.generator_piscina_show_fin_dorsal,
                            ctx.generator_piscina_show_fin_anal,
                            ctx.generator_piscina_show_fin_caudal,
                            ctx.generator_piscina_show_fin_pectoral,
                            ctx.generator_piscina_show_fin_pelvic,
                            ctx.generator_piscina_show_fin_adipose,
                            ctx.generator_piscina_anchor_offset_u,
                            ctx.generator_piscina_anchor_offset_v,
                            ctx.color,
                            dbg,
                            mesh_gen,
                        );
                        if preview_overlay_cache_key_get(state) == Some(key) {
                            return PreviewMeshPrepared::Noop;
                        }
                        const NBRS: [(i32, i32, i32); 6] = [
                            (1, 0, 0),
                            (-1, 0, 0),
                            (0, 1, 0),
                            (0, -1, 0),
                            (0, 0, 1),
                            (0, 0, -1),
                        ];
                        let color_map: AHashMap<(i32, i32, i32), u32> =
                            cells.iter().cloned().collect();
                        let visible: AHashSet<_> = color_map
                            .keys()
                            .filter(|&&(x, y, z)| {
                                NBRS.iter().any(|&(dx, dy, dz)| {
                                    !color_map.contains_key(&(x + dx, y + dy, z + dz))
                                })
                            })
                            .copied()
                            .collect();
                        let fallback = ctx.color;
                        let instanced = stroke_preview_meshes_for_union(
                            voxel_edit::EditTool::Add,
                            &visible,
                            vmap,
                            file,
                            dbg,
                            fallback,
                            Some(&|x, y, z| *color_map.get(&(x, y, z)).unwrap_or(&fallback)),
                        );
                        return PreviewMeshPrepared::GenUpload {
                            cache_key: key,
                            instanced,
                        };
                    }
                }
                "roof" => {
                    if !ctx.generator_roof_pins.is_empty() {
                        let mut instanced = if ctx.generator_roof_pins.len() >= 3 {
                            let mut cells = crate::generators::preview_roof_voxels(
                                &ctx.generator_roof_pins,
                                &ctx.generator_roof_style,
                                ctx.generator_roof_height,
                                ctx.generator_roof_thickness,
                                0, // shed_edge_index
                                0, // gable_orientation
                                ctx.generator_roof_break_ratio,
                                ctx.generator_roof_wall_height,
                                ctx.generator_roof_parapet_height,
                                ctx.generator_roof_salt_skew,
                                ctx.generator_roof_hollow,
                            );
                            voxel_edit::extend_with_mirror_targets(&mut cells, ctx.mirror_axes);
                            if !cells.is_empty() {
                                let set: AHashSet<_> = cells.iter().copied().collect();
                                stroke_preview_meshes_for_union(
                                    voxel_edit::EditTool::Add,
                                    &set,
                                    vmap,
                                    file,
                                    dbg,
                                    ctx.color,
                                    None,
                                )
                            } else {
                                greedy_mesh::PreviewInstancedResult::empty()
                            }
                        } else {
                            greedy_mesh::PreviewInstancedResult::empty()
                        };
                        // Yellow markers at each pin position.
                        append_polygon_vertex_marker_meshes(
                            &mut instanced.extra_solid,
                            &mut instanced.extra_wire,
                            &ctx.generator_roof_pins,
                            vmap,
                            file,
                            dbg,
                        );
                        let key = hash_generator_roof_hover(
                            &ctx.generator_roof_pins,
                            &ctx.generator_roof_style,
                            ctx.generator_roof_height,
                            ctx.generator_roof_thickness,
                            ctx.generator_roof_break_ratio,
                            ctx.generator_roof_wall_height,
                            ctx.generator_roof_parapet_height,
                            ctx.generator_roof_salt_skew,
                            ctx.generator_roof_hollow,
                            ctx.color,
                            dbg,
                            mesh_gen,
                        );
                        if preview_overlay_cache_key_get(state) == Some(key) {
                            return PreviewMeshPrepared::Noop;
                        }
                        return PreviewMeshPrepared::Upload {
                            cache_key: key,
                            instanced,
                        };
                    }
                }
                "shape" => {
                    let shape = StartShape::from_str_id(&ctx.generator_shape_kind);
                    // When the gizmo center is set (settings phase), place the shape
                    // at that center rather than raycasting from the cursor.  Also
                    // account for any in-flight gizmo drag offset so the preview
                    // moves in real-time while dragging the movement arrows.
                    let gen_center = *state.gizmos.generator_gizmo_center.lock();
                    let mut cells = if let Some([gx, gy, gz]) = gen_center {
                        let (pdx, pdy, pdz) = crate::frame_loop::pending_gizmo_translate(state);
                        let origin = (gx as i32 + pdx, gy as i32 + pdy, gz as i32 + pdz);
                        let all = crate::generators::compute_shape_positions(
                            shape,
                            ctx.generator_shape_size,
                            origin,
                            (
                                ctx.generator_shape_rot_x,
                                ctx.generator_shape_rot_y,
                                ctx.generator_shape_rot_z,
                            ),
                        );
                        if ctx.generator_shape_overwrite {
                            all
                        } else {
                            all.into_iter().filter(|c| !vmap.contains_key(c)).collect()
                        }
                    } else {
                        crate::generators::preview_shape_at_screen(
                            file,
                            vmap,
                            cam,
                            w,
                            h,
                            sx,
                            sy,
                            shape,
                            ctx.generator_shape_size,
                            ctx.generator_shape_rot_x,
                            ctx.generator_shape_rot_y,
                            ctx.generator_shape_rot_z,
                            ctx.generator_shape_overwrite,
                        )
                    };
                    voxel_edit::extend_with_mirror_targets(&mut cells, ctx.mirror_axes);
                    if !cells.is_empty() {
                        // Include gizmo center + drag offset in the cache key so the
                        // preview rebuilds when the gizmo moves.
                        let (hash_x, hash_y) = if let Some(gc) = gen_center {
                            let (pdx, pdy, pdz) = crate::frame_loop::pending_gizmo_translate(state);
                            // Pack center + pending into two f32s for the hash.
                            (
                                f32::from_bits((gc[0] as i32 + pdx) as u32),
                                f32::from_bits(
                                    ((gc[1] as i32 + pdy) as u32)
                                        .wrapping_add((gc[2] as i32 + pdz) as u32),
                                ),
                            )
                        } else {
                            (sx, sy)
                        };
                        let key = hash_generator_shape_hover(
                            hash_x,
                            hash_y,
                            &ctx.generator_shape_kind,
                            ctx.generator_shape_size,
                            ctx.generator_shape_rot_x,
                            ctx.generator_shape_rot_y,
                            ctx.generator_shape_rot_z,
                            ctx.generator_shape_overwrite,
                            ctx.color,
                            dbg,
                            mesh_gen,
                        );
                        if preview_overlay_cache_key_get(state) == Some(key) {
                            return PreviewMeshPrepared::Noop;
                        }
                        const NBRS: [(i32, i32, i32); 6] = [
                            (1, 0, 0),
                            (-1, 0, 0),
                            (0, 1, 0),
                            (0, -1, 0),
                            (0, 0, 1),
                            (0, 0, -1),
                        ];
                        let set: AHashSet<_> = cells.iter().copied().collect();
                        let visible: AHashSet<_> = set
                            .iter()
                            .filter(|&&(x, y, z)| {
                                NBRS.iter()
                                    .any(|&(dx, dy, dz)| !set.contains(&(x + dx, y + dy, z + dz)))
                            })
                            .copied()
                            .collect();
                        let instanced = stroke_preview_meshes_for_union(
                            voxel_edit::EditTool::Add,
                            &visible,
                            vmap,
                            file,
                            dbg,
                            ctx.color,
                            None,
                        );
                        return PreviewMeshPrepared::GenUpload {
                            cache_key: key,
                            instanced,
                        };
                    }
                }
                _ => {}
            }
        }
    }

    if matches!(mode, PreviewMode::SelectExtrude) {
        return PreviewMeshPrepared::Clear;
    }

    if matches!(mode, PreviewMode::Select) {
        let poly_placing = matches!(
            ctx.stroke_mode,
            stroke_modes::DrawStrokeMode::Polygon | stroke_modes::DrawStrokeMode::PolygonHull
        ) && !ctx.stroke_aux.polygon_vertices.is_empty()
            && ctx.use_brush_preview;
        if poly_placing {
            let material = voxelle::MaterialId::from_str_id(&ctx.material);
            let spray_cp = *state.file.spray_constraint_plane.lock();
            let targets = voxel_edit::collect_stroke_preview_targets(
                file,
                vmap,
                cam,
                w,
                h,
                sx,
                sy,
                voxel_edit::EditTool::Remove,
                ctx.color,
                material,
                ctx.brush_radius,
                ctx.brush_shape,
                ctx.spray_density,
                None,
                None,
                ctx.stroke_mode,
                ctx.plane_axis,
                &ctx.stroke_aux,
                spray_cp,
            );
            let key = hash_brush_hover_targets(mode, ctx, &targets, vmap, dbg);
            if preview_overlay_cache_key_get(state) == Some(key) {
                return PreviewMeshPrepared::Noop;
            }
            let set: AHashSet<_> = targets.iter().copied().collect();
            let mut instanced = if targets.is_empty() {
                greedy_mesh::PreviewInstancedResult::empty()
            } else {
                stroke_preview_meshes_for_union(
                    voxel_edit::EditTool::Remove,
                    &set,
                    vmap,
                    file,
                    dbg,
                    ctx.color,
                    None,
                )
            };
            append_polygon_vertex_marker_meshes(
                &mut instanced.extra_solid,
                &mut instanced.extra_wire,
                &ctx.stroke_aux.polygon_vertices,
                vmap,
                file,
                dbg,
            );
            if instanced.solid_instances.is_empty() && instanced.extra_solid.positions.is_empty() {
                return PreviewMeshPrepared::Clear;
            }
            return PreviewMeshPrepared::Upload {
                cache_key: key,
                instanced,
            };
        }
        let key_cell = voxel_edit::preview_remove_cell(file, vmap, cam, w, h, sx, sy);
        let key = match key_cell {
            Some(((cx, cy, cz), oid)) => hash_single_cell_preview(mode, cx, cy, cz, 3, dbg, 0, oid),
            None => hash_preview_miss(mode, dbg),
        };
        if preview_overlay_cache_key_get(state) == Some(key) {
            return PreviewMeshPrepared::Noop;
        }
        if let Some(((cx, cy, cz), oid)) = key_cell {
            let (sr, sg, sb, wr, wg, wb, size, wem) = if dbg {
                (1.0f32, 0.12, 0.1, 0.55, 0.0, 0.0, 0.56f32, 3.5f32)
            } else {
                // Fixed blue for selection hover — not the active palette.
                (0.35, 0.55, 0.98, 0.05, 0.08, 0.2, 0.5, 2.0)
            };
            // Grid-snap: render at integer cell center (same as brush preview)
            // instead of the face-hit float, so the highlight locks to the voxel.
            let instanced = preview_single_cell_world(
                file, cx as f32, cy as f32, cz as f32, oid, sr, sg, sb, wr, wg, wb, size, wem,
            );
            return PreviewMeshPrepared::Upload {
                cache_key: key,
                instanced,
            };
        }
        return PreviewMeshPrepared::Clear;
    }

    if matches!(mode, PreviewMode::Stamp | PreviewMode::Punch) {
        let clip = state.selection.stamp_clipboard.lock().clone();
        let Some(clip) = clip else {
            return PreviewMeshPrepared::Clear;
        };
        if clip.entries.is_empty() {
            return PreviewMeshPrepared::Clear;
        }
        let anchor = if matches!(mode, PreviewMode::Stamp) {
            // Stamp places at the empty cell in front of the first solid.
            voxel_edit::preview_add_cell(file, vmap, cam, w, h, sx, sy).map(|(c, _)| c)
        } else {
            // Punch removes starting at the hit solid cell.
            voxel_edit::preview_remove_cell(file, vmap, cam, w, h, sx, sy).map(|(c, _)| c)
        };
        let (origin_x, origin_z) = (ctx.stamp_origin_x, ctx.stamp_origin_z);
        let (off_x, off_z) =
            voxel_edit::stamp_origin_offsets_pub(&clip.entries, origin_x, origin_z);
        let key = {
            let mut hasher = AHasher::default();
            mode.hash(&mut hasher);
            anchor.hash(&mut hasher);
            for &(dx, dy, dz, color, _mat) in &clip.entries {
                dx.hash(&mut hasher);
                dy.hash(&mut hasher);
                dz.hash(&mut hasher);
                color.hash(&mut hasher);
            }
            origin_x.hash(&mut hasher);
            origin_z.hash(&mut hasher);
            dbg.hash(&mut hasher);
            hasher.finish()
        };
        if preview_overlay_cache_key_get(state) == Some(key) {
            return PreviewMeshPrepared::Noop;
        }
        let Some((ax, ay, az)) = anchor else {
            return PreviewMeshPrepared::Clear;
        };
        let tool = if matches!(mode, PreviewMode::Stamp) {
            voxel_edit::EditTool::Add
        } else {
            voxel_edit::EditTool::Remove
        };
        // Build coord→color map for stamp so each ghost voxel shows its source color.
        let color_map: AHashMap<greedy_mesh::VoxelCoord, u32> = clip
            .entries
            .iter()
            .map(|&(dx, dy, dz, src_color, _)| {
                ((ax + dx - off_x, ay + dy, az + dz - off_z), src_color)
            })
            .collect();
        let cells: AHashSet<greedy_mesh::VoxelCoord> = color_map.keys().copied().collect();
        let color_resolver =
            |x: i32, y: i32, z: i32| color_map.get(&(x, y, z)).copied().unwrap_or(ctx.color);
        let instanced = stroke_preview_meshes_for_union(
            tool,
            &cells,
            vmap,
            file,
            dbg,
            ctx.color,
            if matches!(mode, PreviewMode::Stamp) {
                Some(&color_resolver as &dyn Fn(i32, i32, i32) -> u32)
            } else {
                None
            },
        );
        if instanced.solid_instances.is_empty() && instanced.extra_solid.positions.is_empty() {
            return PreviewMeshPrepared::Clear;
        }
        return PreviewMeshPrepared::Upload {
            cache_key: key,
            instanced,
        };
    }

    let tool = match mode {
        PreviewMode::Add => voxel_edit::EditTool::Add,
        PreviewMode::Remove => voxel_edit::EditTool::Remove,
        PreviewMode::Paint => voxel_edit::EditTool::Paint,
        PreviewMode::Navigate
        | PreviewMode::Fly
        | PreviewMode::Select
        | PreviewMode::SelectExtrude
        | PreviewMode::Squishy
        | PreviewMode::Bone
        | PreviewMode::Stamp
        | PreviewMode::Punch => {
            unreachable!()
        }
    };
    let mode_tag: u8 = match mode {
        PreviewMode::Add => 0,
        PreviewMode::Remove => 1,
        PreviewMode::Paint => 2,
        _ => 0,
    };

    if !ctx.use_brush_preview {
        let key_cell = match mode {
            PreviewMode::Add => voxel_edit::preview_add_cell(file, vmap, cam, w, h, sx, sy),
            PreviewMode::Remove | PreviewMode::Paint => {
                voxel_edit::preview_remove_cell(file, vmap, cam, w, h, sx, sy)
            }
            _ => None,
        };
        let key = match key_cell {
            Some(((cx, cy, cz), oid)) => {
                hash_single_cell_preview(mode, cx, cy, cz, mode_tag, dbg, ctx.color, oid)
            }
            None => hash_preview_miss(mode, dbg),
        };
        if preview_overlay_cache_key_get(state) == Some(key) {
            return PreviewMeshPrepared::Noop;
        }
        match key_cell {
            Some(((cx, cy, cz), oid)) => {
                let (sr, sg, sb, wr, wg, wb, size, wem) = preview_tool_colors(tool, dbg, ctx.color);
                // Grid-snap: render at integer cell center so the preview locks
                // to the voxel grid instead of the floating-point face-hit.
                let instanced = preview_single_cell_world(
                    file, cx as f32, cy as f32, cz as f32, oid, sr, sg, sb, wr, wg, wb, size, wem,
                );
                return PreviewMeshPrepared::Upload {
                    cache_key: key,
                    instanced,
                };
            }
            None => return PreviewMeshPrepared::Clear,
        }
    }

    let material = voxelle::MaterialId::from_str_id(&ctx.material);
    let spray_cp = *state.file.spray_constraint_plane.lock();
    let targets = voxel_edit::collect_stroke_preview_targets(
        file,
        vmap,
        cam,
        w,
        h,
        sx,
        sy,
        tool,
        ctx.color,
        material,
        ctx.brush_radius,
        ctx.brush_shape,
        ctx.spray_density,
        None,
        None,
        ctx.stroke_mode,
        ctx.plane_axis,
        &ctx.stroke_aux,
        spray_cp,
    );
    let key = hash_brush_hover_targets(mode, ctx, &targets, vmap, dbg);
    if preview_overlay_cache_key_get(state) == Some(key) {
        return PreviewMeshPrepared::Noop;
    }
    let poly_corners = matches!(
        ctx.stroke_mode,
        stroke_modes::DrawStrokeMode::Polygon | stroke_modes::DrawStrokeMode::PolygonHull
    ) && !ctx.stroke_aux.polygon_vertices.is_empty();
    if targets.is_empty() && !poly_corners {
        return PreviewMeshPrepared::Clear;
    }
    let set: AHashSet<_> = targets.iter().copied().collect();
    let hover_resolver_owned = if ctx.palette.len() > 1 {
        Some(build_color_resolver(
            ctx.color,
            ctx.palette.clone(),
            ctx.paint_color_distrib.clone(),
            0, // fixed seed for consistent hover preview
        ))
    } else {
        None
    };
    let hover_resolver_ref: Option<&dyn Fn(i32, i32, i32) -> u32> = hover_resolver_owned
        .as_ref()
        .map(|f| f as &dyn Fn(i32, i32, i32) -> u32);
    // For large uniform-colour strokes with no polygon-corner extras, use the
    // GPU compute shell-filter path.  Falls back to CPU instanced otherwise.
    if !targets.is_empty() && !poly_corners && hover_resolver_ref.is_none() {
        if let Some(raw) = build_raw_voxel_upload(tool, &set, vmap, file, dbg, ctx.color, None) {
            return PreviewMeshPrepared::RawVoxelUpload {
                cache_key: key,
                raw,
            };
        }
    }

    let mut instanced = if targets.is_empty() {
        greedy_mesh::PreviewInstancedResult::empty()
    } else {
        stroke_preview_meshes_for_union(tool, &set, vmap, file, dbg, ctx.color, hover_resolver_ref)
    };
    if poly_corners {
        append_polygon_vertex_marker_meshes(
            &mut instanced.extra_solid,
            &mut instanced.extra_wire,
            &ctx.stroke_aux.polygon_vertices,
            vmap,
            file,
            dbg,
        );
    }
    if instanced.solid_instances.is_empty() && instanced.extra_solid.positions.is_empty() {
        PreviewMeshPrepared::Clear
    } else {
        PreviewMeshPrepared::Upload {
            cache_key: key,
            instanced,
        }
    }
}

pub(crate) fn clear_preview_mesh_sync_cache(viewer: &mut WgpuViewer, state: &ViewerState) {
    viewer.clear_preview_mesh();
    *state.gpu.preview_overlay_cache_key.lock() = None;
}

pub(crate) fn apply_preview_mesh(
    viewer: &mut WgpuViewer,
    state: &ViewerState,
    prep: PreviewMeshPrepared,
) {
    match prep {
        PreviewMeshPrepared::Noop => {}
        PreviewMeshPrepared::Clear => {
            clear_preview_mesh_sync_cache(viewer, state);
        }
        PreviewMeshPrepared::Upload {
            cache_key,
            instanced,
        } => {
            viewer.upload_preview_mesh_instanced(&instanced);
            viewer.preview_cache_key = Some(cache_key);
            *state.gpu.preview_overlay_cache_key.lock() = Some(cache_key);
        }
        PreviewMeshPrepared::GenUpload {
            cache_key,
            instanced,
        } => {
            viewer.upload_gen_preview_mesh_instanced(&instanced);
            viewer.preview_cache_key = Some(cache_key);
            *state.gpu.preview_overlay_cache_key.lock() = Some(cache_key);
        }
        PreviewMeshPrepared::RawVoxelUpload { cache_key, raw } => {
            viewer.upload_preview_raw_voxels(&raw);
            viewer.preview_cache_key = Some(cache_key);
            *state.gpu.preview_overlay_cache_key.lock() = Some(cache_key);
        }
    }
}
