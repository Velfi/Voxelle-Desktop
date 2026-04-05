use crate::edit_pipeline::wake_viewport_loop;
use crate::preview::default_true;
use crate::state::viewport_texels_from_norm;
use crate::*;

// ── Generator arg structs & defaults ─────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeneratorRocksArgs {
    nx: f32,
    ny: f32,
    #[serde(default)]
    seed: i32,
    #[serde(default = "default_rock_size")]
    size: i32,
    #[serde(default = "default_roughness")]
    roughness: f32,
    color: u32,
    material: String,
    #[serde(default = "default_rock_count")]
    count: i32,
    #[serde(default = "default_rock_cluster_radius")]
    cluster_radius: i32,
    #[serde(default)]
    sink_direction: i32,
    #[serde(default)]
    sink_amount: i32,
}

pub(crate) fn default_rock_size() -> i32 {
    4
}

fn default_roughness() -> f32 {
    0.4
}

pub(crate) fn default_rock_count() -> i32 {
    1
}

pub(crate) fn default_rock_cluster_radius() -> i32 {
    1
}

#[tauri::command]
pub(crate) fn generator_rocks_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorRocksArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.gpu.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.file.current_file.lock();
        let mut vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.cam.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        crate::generators::generator_rocks_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            args.seed,
            args.size,
            args.roughness,
            args.color,
            material,
            args.count,
            args.cluster_radius,
            args.sink_direction,
            args.sink_amount,
        )?
    };
    commit_generator_edits(&state, &app, deltas)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeneratorGrassArgs {
    nx: f32,
    ny: f32,
    #[serde(default)]
    seed: i32,
    #[serde(default = "default_grass_radius")]
    radius: i32,
    #[serde(default = "default_grass_density")]
    density: f32,
    #[serde(default = "default_grass_height")]
    max_height: i32,
    color: u32,
    material: String,
}

pub(crate) fn default_grass_radius() -> i32 {
    4
}

pub(crate) fn default_grass_density() -> f32 {
    0.6
}

fn default_grass_height() -> i32 {
    3
}

#[tauri::command]
pub(crate) fn generator_grass_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorGrassArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.gpu.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.file.current_file.lock();
        let mut vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.cam.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        crate::generators::generator_grass_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            args.seed,
            args.radius,
            args.density,
            args.max_height,
            args.color,
            material,
        )?
    };
    commit_generator_edits(&state, &app, deltas)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeneratorRopeArgs {
    nx1: f32,
    ny1: f32,
    nx2: f32,
    ny2: f32,
    /// 0 = loose, 1 = taut (scales sag; web `ropeTension`).
    #[serde(default = "default_rope_tension")]
    tension: f32,
    /// Web `ropeBrushRadius` index (same mapping as sculpt brush index).
    #[serde(default = "default_rope_brush_radius_index")]
    brush_radius: u32,
    #[serde(default)]
    brush_shape: voxel_edit::BrushShape,
    color: u32,
    material: String,
    /// Web `ropeGravityDirection`: down | up | left | right | forward | back.
    #[serde(default = "default_cloth_gravity_direction")]
    gravity_direction: String,
}

pub(crate) fn default_rope_sag() -> f32 {
    2.5
}

pub(crate) fn default_rope_tension() -> f32 {
    0.5
}

fn default_rope_brush_radius_index() -> u32 {
    2
}

#[tauri::command]
pub(crate) fn generator_rope_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorRopeArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.gpu.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.file.current_file.lock();
        let mut vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.cam.camera.lock();
        let (sx1, sy1) = viewport_texels_from_norm(args.nx1, args.ny1, w, h);
        let (sx2, sy2) = viewport_texels_from_norm(args.nx2, args.ny2, w, h);
        crate::generators::generator_rope_between_screens(
            file,
            vmap,
            &cam,
            w,
            h,
            sx1,
            sy1,
            sx2,
            sy2,
            args.tension,
            args.brush_radius,
            args.brush_shape,
            args.color,
            material,
            &args.gravity_direction,
        )?
    };
    commit_generator_edits(&state, &app, deltas)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeneratorClothArgs {
    /// At least three distinct corner voxels (surface picks).
    pins: Vec<[i32; 3]>,
    #[serde(default = "default_rope_tension")]
    tension: f32,
    /// Web `ropeGravityDirection`: down | up | left | right | forward | back.
    #[serde(default = "default_cloth_gravity_direction")]
    gravity_direction: String,
    #[serde(default = "default_rope_brush_radius_index")]
    brush_radius: u32,
    #[serde(default)]
    brush_shape: voxel_edit::BrushShape,
    color: u32,
    material: String,
    /// Web `clothSimGravityPct / 100`.
    #[serde(default = "default_cloth_gravity_stiffness_scale")]
    gravity_scale: f64,
    /// Web `clothSimStiffnessPct / 100`.
    #[serde(default = "default_cloth_gravity_stiffness_scale")]
    stiffness_scale: f64,
    /// 0 = automatic iteration count from tension.
    #[serde(default)]
    cloth_iterations: u32,
    #[serde(default = "default_cloth_constraint_passes")]
    cloth_constraint_passes: u32,
}

fn default_cloth_gravity_direction() -> String {
    "down".into()
}

fn default_cloth_gravity_stiffness_scale() -> f64 {
    1.0
}

fn default_cloth_constraint_passes() -> u32 {
    2
}

#[tauri::command]
pub(crate) fn generator_cloth_from_pins_cmd(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorClothArgs,
) -> Result<bool, String> {
    if args.pins.len() < 3 {
        return Err("cloth needs at least three pin points".into());
    }
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let sim = crate::generators::ClothSimOptions {
        gravity_scale: args.gravity_scale.max(0.0),
        stiffness_scale: args.stiffness_scale.clamp(0.05, 2.0),
        iterations: if args.cloth_iterations > 0 {
            Some(args.cloth_iterations.clamp(4, 96))
        } else {
            None
        },
        constraint_passes: args.cloth_constraint_passes.clamp(1, 6),
    };
    let deltas = {
        let mut fg = state.file.current_file.lock();
        let mut vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        crate::generators::generator_cloth_from_pins(
            file,
            vmap,
            &args.pins,
            args.tension,
            args.gravity_direction.as_str(),
            args.brush_radius,
            args.brush_shape,
            args.color,
            material,
            sim,
        )?
    };
    commit_generator_edits(&state, &app, deltas)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeneratorSquishyArgs {
    nx: f32,
    ny: f32,
    #[serde(default = "default_squishy_radius")]
    radius: i32,
    color: u32,
    material: String,
}

fn default_squishy_radius() -> i32 {
    5
}

#[tauri::command]
pub(crate) fn generator_squishy_metaball_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorSquishyArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.gpu.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.file.current_file.lock();
        let mut vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.cam.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        crate::generators::squishy_metaball_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            args.radius,
            args.color,
            material,
        )?
    };
    commit_generator_edits(&state, &app, deltas)
}

// ── Ashlar generator ──────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeneratorAshlarArgs {
    nx: f32,
    ny: f32,
    #[serde(default)]
    seed: i32,
    #[serde(default = "default_rock_size")]
    size: i32,
    #[serde(default = "default_roughness")]
    roughness: f32,
    color: u32,
    material: String,
    #[serde(default)]
    thickness: Option<i32>,
    #[serde(default)]
    thickness_axis: Option<i32>,
}

#[tauri::command]
pub(crate) fn generator_ashlar_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorAshlarArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.gpu.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.file.current_file.lock();
        let mut vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.cam.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        crate::generators::generator_ashlar_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            args.seed,
            args.size,
            args.roughness,
            args.color,
            material,
            args.thickness,
            args.thickness_axis,
        )?
    };
    commit_generator_edits(&state, &app, deltas)
}

// ── Flora generator ───────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeneratorFloraArgs {
    nx: f32,
    ny: f32,
    #[serde(default)]
    seed: i32,
    #[serde(default = "default_flora_height")]
    height: i32,
    #[serde(default)]
    girth: i32,
    #[serde(default = "default_flora_wobble")]
    wobble: f32,
    #[serde(default = "default_flora_taper")]
    taper: f32,
    #[serde(default = "default_one_i32")]
    stem_count: i32,
    #[serde(default)]
    cluster_radius: i32,
    #[serde(default)]
    branch_count: i32,
    #[serde(default = "default_one_i32")]
    branch_depth: i32,
    #[serde(default = "default_flora_branch_start")]
    branch_start: f32,
    #[serde(default = "default_one_f32_flora")]
    branch_spread: f32,
    #[serde(default = "default_one_i32")]
    braid_strands: i32,
    #[serde(default = "default_flora_braid_twist")]
    braid_twist: f32,
    #[serde(default)]
    canopy: f32,
    color: u32,
    material: String,
}

pub(crate) fn default_flora_height() -> i32 {
    14
}
pub(crate) fn default_flora_wobble() -> f32 {
    0.12
}
pub(crate) fn default_flora_taper() -> f32 {
    0.12
}
pub(crate) fn default_one_i32() -> i32 {
    1
}
pub(crate) fn default_flora_branch_start() -> f32 {
    0.5
}
pub(crate) fn default_flora_braid_twist() -> f32 {
    0.35
}
fn default_one_f32_flora() -> f32 {
    1.0
}

#[tauri::command]
pub(crate) fn generator_flora_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorFloraArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.gpu.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.file.current_file.lock();
        let mut vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.cam.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        crate::generators::generator_flora_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            args.seed,
            args.height,
            args.girth,
            args.wobble,
            args.taper,
            args.stem_count,
            args.cluster_radius,
            args.branch_count,
            args.branch_depth,
            args.branch_start,
            args.branch_spread,
            args.braid_strands,
            args.braid_twist,
            args.canopy,
            args.color,
            material,
        )?
    };
    commit_generator_edits(&state, &app, deltas)
}

// ── Roof generator ────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeneratorRoofArgs {
    pins: Vec<[i32; 3]>,
    #[serde(default = "default_roof_style")]
    style: String,
    #[serde(default = "default_roof_height")]
    height: i32,
    #[serde(default = "default_one_i32")]
    thickness: i32,
    #[serde(default)]
    shed_edge_index: i32,
    #[serde(default)]
    gable_orientation: i32,
    #[serde(default = "default_roof_break_ratio")]
    break_ratio: f32,
    #[serde(default = "default_roof_wall_height")]
    wall_height: i32,
    #[serde(default = "default_roof_parapet_height")]
    parapet_height: i32,
    #[serde(default)]
    salt_skew: f32,
    #[serde(default)]
    hollow: bool,
    color: u32,
    material: String,
}

pub(crate) fn default_roof_style() -> String {
    "gable".into()
}
pub(crate) fn default_roof_height() -> i32 {
    6
}
pub(crate) fn default_roof_break_ratio() -> f32 {
    0.5
}
pub(crate) fn default_roof_wall_height() -> i32 {
    3
}
pub(crate) fn default_roof_parapet_height() -> i32 {
    2
}

#[tauri::command]
pub(crate) fn generator_roof_from_pins_cmd(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorRoofArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    if args.pins.len() < 3 {
        return Err("roof needs at least 3 pins".into());
    }
    let deltas = {
        let mut fg = state.file.current_file.lock();
        let mut vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        crate::generators::generate_roof_from_pins(
            file,
            vmap,
            &args.pins,
            &args.style,
            args.height,
            args.thickness,
            args.shed_edge_index,
            args.gable_orientation,
            args.break_ratio,
            args.wall_height,
            args.parapet_height,
            args.salt_skew,
            args.hollow,
            args.color,
            material,
        )
    };
    commit_generator_edits(&state, &app, deltas)
}

// ── Piscina (fish) generator ──────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeneratorPiscinaArgs {
    nx: f32,
    ny: f32,
    #[serde(default)]
    seed: i32,
    #[serde(default = "default_piscina_species")]
    species: String,
    #[serde(default = "default_piscina_length")]
    length: i32,
    #[serde(default = "default_piscina_width")]
    width_param: i32,
    #[serde(default = "default_piscina_thickness")]
    thickness: i32,
    #[serde(default)]
    spine_bend: f32,
    #[serde(default)]
    spine_s_curve: f32,
    #[serde(default = "default_piscina_fin")]
    fin_dorsal: i32,
    #[serde(default = "default_piscina_fin")]
    fin_anal: i32,
    #[serde(default = "default_piscina_fin")]
    fin_caudal: i32,
    #[serde(default = "default_piscina_fin")]
    fin_pectoral: i32,
    #[serde(default = "default_piscina_fin")]
    fin_pelvic: i32,
    #[serde(default = "default_piscina_fin")]
    fin_adipose: i32,
    #[serde(default = "default_true")]
    show_fin_dorsal: bool,
    #[serde(default = "default_true")]
    show_fin_anal: bool,
    #[serde(default = "default_true")]
    show_fin_caudal: bool,
    #[serde(default = "default_true")]
    show_fin_pectoral: bool,
    #[serde(default = "default_true")]
    show_fin_pelvic: bool,
    #[serde(default = "default_true")]
    show_fin_adipose: bool,
    #[serde(default)]
    anchor_offset_u: i32,
    #[serde(default)]
    anchor_offset_v: i32,
    color: u32,
    material: String,
}

pub(crate) fn default_piscina_species() -> String {
    "trout".into()
}
pub(crate) fn default_piscina_length() -> i32 {
    16
}
fn default_piscina_width() -> i32 {
    4
}
fn default_piscina_thickness() -> i32 {
    3
}
fn default_piscina_fin() -> i32 {
    3
}

#[tauri::command]
pub(crate) fn generator_piscina_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorPiscinaArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.gpu.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.file.current_file.lock();
        let mut vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.cam.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        crate::generators::generator_piscina_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            args.seed,
            &args.species,
            args.length,
            args.width_param,
            args.thickness,
            args.spine_bend,
            args.spine_s_curve,
            args.fin_dorsal,
            args.fin_anal,
            args.fin_caudal,
            args.fin_pectoral,
            args.fin_pelvic,
            args.fin_adipose,
            args.show_fin_dorsal,
            args.show_fin_anal,
            args.show_fin_caudal,
            args.show_fin_pectoral,
            args.show_fin_pelvic,
            args.show_fin_adipose,
            args.anchor_offset_u,
            args.anchor_offset_v,
            args.color,
            material,
        )?
    };
    commit_generator_edits(&state, &app, deltas)
}

// ── Insecta (insect) generator ────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeneratorInsectaArgs {
    nx: f32,
    ny: f32,
    #[serde(default = "default_insecta_species")]
    species: String,
    #[serde(default = "default_insecta_length")]
    total_length: i32,
    #[serde(default = "default_one_f32")]
    head_ratio: f32,
    #[serde(default = "default_insecta_thorax_ratio")]
    thorax_ratio: f32,
    #[serde(default = "default_insecta_abdomen_ratio")]
    abdomen_ratio: f32,
    #[serde(default = "default_insecta_body_half_width")]
    body_half_width: i32,
    #[serde(default = "default_insecta_body_half_height")]
    body_half_height: i32,
    #[serde(default = "default_insecta_abdomen_taper")]
    abdomen_taper: f32,
    #[serde(default = "default_insecta_head_shape")]
    head_shape: i32,
    #[serde(default)]
    anchor_offset_u: i32,
    #[serde(default)]
    anchor_offset_v: i32,
    #[serde(default)]
    body_yaw: f32,
    #[serde(default)]
    body_arch: f32,
    #[serde(default = "default_insecta_antenna_length")]
    antenna_length: i32,
    #[serde(default = "default_insecta_antenna_spread")]
    antenna_spread: f32,
    #[serde(default = "default_insecta_antenna_pitch")]
    antenna_pitch: f32,
    #[serde(default)]
    antenna_root: i32,
    #[serde(default)]
    mandible_length: i32,
    #[serde(default)]
    mandible_spread: f32,
    #[serde(default)]
    mandible_forward: i32,
    #[serde(default = "default_insecta_wing_shape")]
    wing_shape: i32,
    #[serde(default = "default_true")]
    show_wing_fore: bool,
    #[serde(default = "default_insecta_wing_fore_length")]
    wing_fore_length: i32,
    #[serde(default = "default_insecta_wing_fore_width")]
    wing_fore_width: i32,
    #[serde(default = "default_insecta_wing_spread")]
    wing_fore_spread: f32,
    #[serde(default)]
    wing_fore_pitch: f32,
    #[serde(default)]
    wing_fore_offset: i32,
    #[serde(default)]
    wing_fore_forward_cant: f32,
    #[serde(default)]
    show_wing_hind: bool,
    #[serde(default = "default_insecta_wing_hind_length")]
    wing_hind_length: i32,
    #[serde(default = "default_insecta_wing_hind_width")]
    wing_hind_width: i32,
    #[serde(default = "default_insecta_wing_spread")]
    wing_hind_spread: f32,
    #[serde(default)]
    wing_hind_pitch: f32,
    #[serde(default)]
    wing_hind_offset: i32,
    color: u32,
    material: String,
}

pub(crate) fn default_insecta_species() -> String {
    "bee".into()
}
fn default_insecta_length() -> i32 {
    24
}
pub(crate) fn default_one_f32() -> f32 {
    1.0
}
fn default_insecta_thorax_ratio() -> f32 {
    1.2
}
pub(crate) fn default_insecta_abdomen_ratio() -> f32 {
    2.0
}
fn default_insecta_body_half_width() -> i32 {
    3
}
fn default_insecta_body_half_height() -> i32 {
    3
}
pub(crate) fn default_insecta_abdomen_taper() -> f32 {
    0.6
}
fn default_insecta_head_shape() -> i32 {
    60
}
pub(crate) fn default_insecta_antenna_length() -> i32 {
    6
}
pub(crate) fn default_insecta_antenna_spread() -> f32 {
    20.0
}
pub(crate) fn default_insecta_antenna_pitch() -> f32 {
    30.0
}
fn default_insecta_wing_shape() -> i32 {
    85
}
pub(crate) fn default_insecta_wing_fore_length() -> i32 {
    12
}
fn default_insecta_wing_fore_width() -> i32 {
    3
}
fn default_insecta_wing_spread() -> f32 {
    15.0
}
pub(crate) fn default_insecta_wing_hind_length() -> i32 {
    8
}
fn default_insecta_wing_hind_width() -> i32 {
    2
}

#[tauri::command]
pub(crate) fn generator_insecta_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorInsectaArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.gpu.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.file.current_file.lock();
        let mut vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.cam.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        crate::generators::generator_insecta_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            &args.species,
            args.total_length,
            args.head_ratio,
            args.thorax_ratio,
            args.abdomen_ratio,
            args.body_half_width,
            args.body_half_height,
            args.abdomen_taper,
            args.head_shape,
            args.anchor_offset_u,
            args.anchor_offset_v,
            args.body_yaw,
            args.body_arch,
            args.antenna_length,
            args.antenna_spread,
            args.antenna_pitch,
            args.antenna_root,
            args.mandible_length,
            args.mandible_spread,
            args.mandible_forward,
            args.wing_shape,
            args.show_wing_fore,
            args.wing_fore_length,
            args.wing_fore_width,
            args.wing_fore_spread,
            args.wing_fore_pitch,
            args.wing_fore_offset,
            args.wing_fore_forward_cant,
            args.show_wing_hind,
            args.wing_hind_length,
            args.wing_hind_width,
            args.wing_hind_spread,
            args.wing_hind_pitch,
            args.wing_hind_offset,
            args.color,
            material,
        )?
    };
    commit_generator_edits(&state, &app, deltas)
}

// ── Fauna (creature) generator ────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeneratorFaunaArgs {
    nx: f32,
    ny: f32,
    #[serde(default = "default_fauna_stance")]
    stance: String,
    #[serde(default = "default_fauna_archetype")]
    archetype: String,
    #[serde(default)]
    anchor_offset_u: i32,
    #[serde(default)]
    anchor_offset_v: i32,
    #[serde(default)]
    body_yaw: f32,
    #[serde(default)]
    body_arch: f32,
    #[serde(default = "default_fauna_spine_segments")]
    spine_segments: i32,
    #[serde(default = "default_fauna_body_length")]
    body_length: i32,
    #[serde(default = "default_fauna_body_half")]
    body_half_width: i32,
    #[serde(default = "default_fauna_body_half_height")]
    body_half_height: i32,
    #[serde(default = "default_fauna_neck_length")]
    neck_length: i32,
    #[serde(default = "default_fauna_neck_half")]
    neck_half_width: i32,
    #[serde(default = "default_fauna_neck_half")]
    neck_half_height: i32,
    #[serde(default = "default_fauna_head_length")]
    head_length: i32,
    #[serde(default = "default_fauna_head_half")]
    head_half_width: i32,
    #[serde(default = "default_fauna_head_half")]
    head_half_height: i32,
    #[serde(default = "default_one_i32")]
    tail_length: i32,
    #[serde(default = "default_fauna_shoulder_offset")]
    shoulder_offset_forward: i32,
    #[serde(default = "default_fauna_hip_offset")]
    hip_offset_forward: i32,
    #[serde(default = "default_fauna_upper_length")]
    front_upper_length: i32,
    #[serde(default = "default_fauna_upper_length")]
    front_lower_length: i32,
    #[serde(default = "default_fauna_hind_upper")]
    hind_upper_length: i32,
    #[serde(default = "default_fauna_hind_upper")]
    hind_lower_length: i32,
    #[serde(default = "default_fauna_limb_targets")]
    limb_targets: [[f32; 3]; 4],
    #[serde(default = "default_fauna_limb_poles")]
    limb_poles: [[f32; 3]; 4],
    #[serde(default)]
    spine_pose_chest: [f32; 3],
    #[serde(default)]
    spine_pose_neck: [f32; 3],
    #[serde(default)]
    spine_pose_head: [f32; 3],
    #[serde(default)]
    auto_foot_placement: bool,
    color: u32,
    material: String,
}

pub(crate) fn default_fauna_stance() -> String {
    "quadruped".into()
}
pub(crate) fn default_fauna_archetype() -> String {
    "ungulate".into()
}
pub(crate) fn default_fauna_spine_segments() -> i32 {
    7
}
pub(crate) fn default_fauna_body_length() -> i32 {
    17
}
fn default_fauna_body_half() -> i32 {
    2
}
fn default_fauna_body_half_height() -> i32 {
    3
}
fn default_fauna_neck_length() -> i32 {
    8
}
fn default_fauna_neck_half() -> i32 {
    2
}
fn default_fauna_head_length() -> i32 {
    6
}
fn default_fauna_head_half() -> i32 {
    2
}
fn default_fauna_shoulder_offset() -> i32 {
    3
}
fn default_fauna_hip_offset() -> i32 {
    -3
}
fn default_fauna_upper_length() -> i32 {
    7
}
fn default_fauna_hind_upper() -> i32 {
    8
}
fn default_fauna_limb_targets() -> [[f32; 3]; 4] {
    [
        [20.0, -2.1, -19.0],
        [20.0, 2.1, -19.0],
        [-3.5, -2.2, -20.0],
        [-3.5, 2.2, -20.0],
    ]
}
fn default_fauna_limb_poles() -> [[f32; 3]; 4] {
    [
        [20.0, -2.4, 0.6],
        [20.0, 2.4, 0.6],
        [1.8, -2.8, 1.2],
        [1.8, 2.8, 1.2],
    ]
}

#[tauri::command]
pub(crate) fn generator_fauna_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorFaunaArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let (w, h) = {
            let v = state.gpu.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.file.current_file.lock();
        let mut vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.cam.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        crate::generators::generator_fauna_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            &args.stance,
            &args.archetype,
            args.anchor_offset_u,
            args.anchor_offset_v,
            args.body_yaw,
            args.body_arch,
            args.spine_segments,
            args.body_length,
            args.body_half_width,
            args.body_half_height,
            args.neck_length,
            args.neck_half_width,
            args.neck_half_height,
            args.head_length,
            args.head_half_width,
            args.head_half_height,
            args.tail_length,
            args.shoulder_offset_forward,
            args.hip_offset_forward,
            args.front_upper_length,
            args.front_lower_length,
            args.hind_upper_length,
            args.hind_lower_length,
            &args.limb_targets,
            &args.limb_poles,
            args.spine_pose_chest,
            args.spine_pose_neck,
            args.spine_pose_head,
            args.auto_foot_placement,
            args.color,
            material,
        )?
    };
    commit_generator_edits(&state, &app, deltas)
}

// ── Squishy session commands ─────────────────────────────────────────────────

#[tauri::command]
pub(crate) fn squishy_session_get(
    state: State<'_, Arc<ViewerState>>,
) -> Result<generators::SquishySession, String> {
    Ok(state.gizmos.squishy_session.lock().clone())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SquishySetModeArgs {
    mode: String,
}

#[tauri::command]
pub(crate) fn squishy_session_set_mode(
    state: State<'_, Arc<ViewerState>>,
    args: SquishySetModeArgs,
) -> Result<(), String> {
    let mut g = state.gizmos.squishy_session.lock();
    g.mode = match args.mode.as_str() {
        "edit" => generators::SquishyMode::Edit,
        "delete" => generators::SquishyMode::Delete,
        _ => generators::SquishyMode::Add,
    };
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SquishySessionFlagsArgs {
    #[serde(default)]
    hollow: Option<bool>,
    #[serde(default)]
    wall_thickness: Option<i32>,
    #[serde(default)]
    add_snap_to_surface: Option<bool>,
}

#[tauri::command]
pub(crate) fn squishy_session_set_flags(
    state: State<'_, Arc<ViewerState>>,
    args: SquishySessionFlagsArgs,
) -> Result<(), String> {
    let mut g = state.gizmos.squishy_session.lock();
    if let Some(h) = args.hollow {
        g.hollow = h;
    }
    if let Some(w) = args.wall_thickness {
        g.wall_thickness = w.max(1);
    }
    if let Some(a) = args.add_snap_to_surface {
        g.add_snap_to_surface = a;
    }
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SquishyMetaballAddArgs {
    nx: f32,
    ny: f32,
    #[serde(default = "default_squishy_radius")]
    radius: i32,
}

#[tauri::command]
pub(crate) fn squishy_metaball_add_at_screen(
    state: State<'_, Arc<ViewerState>>,
    args: SquishyMetaballAddArgs,
) -> Result<Option<u32>, String> {
    let (w, h) = {
        let v = state.gpu.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let mut sg = state.gizmos.squishy_session.lock();
    let fg = state.file.current_file.lock();
    let vm = state.file.voxel_map.lock();
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let cam = state.cam.camera.lock();
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let id = generators::squishy_add_ball_at_screen(
        &mut sg,
        file,
        vmap,
        &cam,
        w,
        h,
        sx,
        sy,
        args.radius,
    );
    Ok(id)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SquishyMetaballIdArgs {
    id: u32,
}

#[tauri::command]
pub(crate) fn squishy_metaball_remove(
    state: State<'_, Arc<ViewerState>>,
    args: SquishyMetaballIdArgs,
) -> Result<bool, String> {
    let mut g = state.gizmos.squishy_session.lock();
    Ok(g.remove_ball(args.id))
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SquishySelectArgs {
    id: Option<u32>,
}

#[tauri::command]
pub(crate) fn squishy_metaball_select(
    state: State<'_, Arc<ViewerState>>,
    args: SquishySelectArgs,
) -> Result<(), String> {
    let mut g = state.gizmos.squishy_session.lock();
    g.selected_id = args.id;
    Ok(())
}

#[tauri::command]
pub(crate) fn squishy_session_clear(state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    let mut g = state.gizmos.squishy_session.lock();
    g.clear();
    *state.gizmos.squishy_gizmo_drag.lock() = None;
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SquishyCommitArgs {
    color: u32,
    material: String,
}

#[tauri::command]
pub(crate) fn squishy_session_commit(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: SquishyCommitArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let sg = state.gizmos.squishy_session.lock();
        let mut fg = state.file.current_file.lock();
        let mut vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        generators::squishy_commit_session(&sg, file, vmap, args.color, material)?
    };
    if deltas.is_empty() {
        return Ok(false);
    }
    commit_voxel_edits(&state, &app, deltas)?;
    let mut g = state.gizmos.squishy_session.lock();
    g.clear();
    *state.gizmos.squishy_gizmo_drag.lock() = None;
    Ok(true)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SquishyPickArgs {
    nx: f32,
    ny: f32,
}

#[tauri::command]
pub(crate) fn squishy_pick_at_screen(
    state: State<'_, Arc<ViewerState>>,
    args: SquishyPickArgs,
) -> Result<Option<u32>, String> {
    let (w, h) = {
        let v = state.gpu.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let sg = state.gizmos.squishy_session.lock();
    let cam = state.cam.camera.lock();
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    Ok(generators::pick_metaball_at_screen(&sg, &cam, w, h, sx, sy))
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SquishyGizmoPointerArgs {
    nx: f32,
    ny: f32,
}

#[tauri::command]
pub(crate) fn squishy_gizmo_pointer_down(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: SquishyGizmoPointerArgs,
) -> Result<bool, String> {
    let (w, h) = {
        let v = state.gpu.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let cam = state.cam.camera.lock();
    let sg = state.gizmos.squishy_session.lock();
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let Some(handle) = generators::pick_squishy_gizmo_handle(&sg, &cam, w, h, sx, sy) else {
        return Ok(false);
    };
    let Some(drag) = generators::squishy_gizmo_begin_drag(&sg, &cam, w, h, sx, sy, handle) else {
        return Ok(false);
    };
    drop(sg);
    drop(cam);
    *state.gizmos.squishy_gizmo_drag.lock() = Some(drag);
    wake_viewport_loop(&app);
    Ok(true)
}

#[tauri::command]
pub(crate) fn squishy_gizmo_pointer_move(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: SquishyGizmoPointerArgs,
) -> Result<(), String> {
    let drag = state.gizmos.squishy_gizmo_drag.lock().clone();
    let Some(drag) = drag else {
        return Ok(());
    };
    let (w, h) = {
        let v = state.gpu.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Ok(());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let cam = state.cam.camera.lock();
    let mut sg = state.gizmos.squishy_session.lock();
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    generators::squishy_gizmo_apply_drag(&mut sg, &cam, w, h, sx, sy, &drag);
    wake_viewport_loop(&app);
    Ok(())
}

#[tauri::command]
pub(crate) fn squishy_gizmo_pointer_up(state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    *state.gizmos.squishy_gizmo_drag.lock() = None;
    Ok(())
}

// ── Bone session commands ───────────────────────────────────────────────────

#[tauri::command]
pub(crate) fn bone_session_get(
    state: State<'_, Arc<ViewerState>>,
) -> Result<generators::BoneSession, String> {
    Ok(state.gizmos.bone_session.lock().clone())
}

#[tauri::command]
pub(crate) fn bone_session_clear(state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    state.gizmos.bone_session.lock().clear();
    *state.gizmos.bone_gizmo_drag.lock() = None;
    *state.gizmos.bone_ik_drag.lock() = None;
    *state.gizmos.generator_gizmo_center.lock() = None;
    *state.gizmos.generator_gizmo_ring_radius.lock() = None;
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BoneCommitArgs {
    color: u32,
    material: String,
}

#[tauri::command]
pub(crate) fn bone_session_commit(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: BoneCommitArgs,
) -> Result<bool, String> {
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let deltas = {
        let bs = state.gizmos.bone_session.lock();
        let mut fg = state.file.current_file.lock();
        let mut vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        generators::bone_commit_session(&bs, file, vmap, args.color, material)?
    };
    if deltas.is_empty() {
        return Ok(false);
    }
    commit_voxel_edits(&state, &app, deltas)?;
    state.gizmos.bone_session.lock().clear();
    *state.gizmos.bone_gizmo_drag.lock() = None;
    *state.gizmos.bone_ik_drag.lock() = None;
    Ok(true)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BoneAddJointArgs {
    nx: f32,
    ny: f32,
    #[serde(default = "default_bone_radius")]
    radius: f32,
}

fn default_bone_radius() -> f32 {
    3.0
}

#[tauri::command]
pub(crate) fn bone_add_joint_at_screen(
    state: State<'_, Arc<ViewerState>>,
    args: BoneAddJointArgs,
) -> Result<Option<u32>, String> {
    let (w, h) = {
        let v = state.gpu.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let fg = state.file.current_file.lock();
    let vm = state.file.voxel_map.lock();
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let cam = state.cam.camera.lock();
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let Some(pos) = generators::bone_screen_to_world_pos(file, vmap, &cam, w, h, sx, sy) else {
        return Ok(None);
    };
    let mut bs = state.gizmos.bone_session.lock();
    let id = bs.add_joint(pos.x, pos.y, pos.z, args.radius);
    Ok(Some(id))
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BoneMoveJointArgs {
    id: u32,
    nx: f32,
    ny: f32,
}

#[tauri::command]
pub(crate) fn bone_move_joint_to_screen(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: BoneMoveJointArgs,
) -> Result<bool, String> {
    let (w, h) = {
        let v = state.gpu.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let fg = state.file.current_file.lock();
    let vm = state.file.voxel_map.lock();
    let Some(file) = fg.as_ref() else {
        return Err("no model loaded".into());
    };
    let Some(vmap) = vm.as_ref() else {
        return Err("voxel index not ready".into());
    };
    let cam = state.cam.camera.lock();
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let Some(pos) = generators::bone_screen_to_world_pos(file, vmap, &cam, w, h, sx, sy) else {
        return Ok(false);
    };
    let mut bs = state.gizmos.bone_session.lock();
    let ok = bs.set_joint_position(args.id, pos.x, pos.y, pos.z);
    if ok {
        wake_viewport_loop(&app);
    }
    Ok(ok)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BoneConnectArgs {
    joint_a: u32,
    joint_b: u32,
}

#[tauri::command]
pub(crate) fn bone_connect_joints(
    state: State<'_, Arc<ViewerState>>,
    args: BoneConnectArgs,
) -> Result<Option<u32>, String> {
    let mut bs = state.gizmos.bone_session.lock();
    Ok(bs.add_bone(args.joint_a, args.joint_b))
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BonePickArgs {
    nx: f32,
    ny: f32,
}

#[tauri::command]
pub(crate) fn bone_pick_at_screen(
    state: State<'_, Arc<ViewerState>>,
    args: BonePickArgs,
) -> Result<Option<generators::BoneSelection>, String> {
    let (w, h) = {
        let v = state.gpu.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let bs = state.gizmos.bone_session.lock();
    let cam = state.cam.camera.lock();
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    Ok(generators::bone_pick_at_screen(&bs, &cam, w, h, sx, sy))
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BoneSelectArgs {
    selection: Option<generators::BoneSelection>,
}

#[tauri::command]
pub(crate) fn bone_select(
    state: State<'_, Arc<ViewerState>>,
    args: BoneSelectArgs,
) -> Result<(), String> {
    let mut bs = state.gizmos.bone_session.lock();
    bs.selected = args.selection;
    // Show the shared move gizmo at the selected joint's position.
    let (center, ring_radius) = match args.selection {
        Some(generators::BoneSelection::Joint(id)) => {
            let j = bs.find_joint(id);
            (j.map(|j| [j.x, j.y, j.z]), j.map(|j| j.radius))
        }
        Some(generators::BoneSelection::Bone(bone_id)) => {
            let c = bs.bones.iter().find(|b| b.id == bone_id).and_then(|bone| {
                let ja = bs.find_joint(bone.joint_a)?;
                let jb = bs.find_joint(bone.joint_b)?;
                Some([
                    (ja.x + jb.x) * 0.5,
                    (ja.y + jb.y) * 0.5,
                    (ja.z + jb.z) * 0.5,
                ])
            });
            (c, None) // no radius ring for bone selection
        }
        None => (None, None),
    };
    drop(bs);
    *state.gizmos.generator_gizmo_center.lock() = center;
    *state.gizmos.generator_gizmo_ring_radius.lock() = ring_radius;
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BoneRemoveArgs {
    selection: generators::BoneSelection,
}

#[tauri::command]
pub(crate) fn bone_remove(
    state: State<'_, Arc<ViewerState>>,
    args: BoneRemoveArgs,
) -> Result<bool, String> {
    let mut bs = state.gizmos.bone_session.lock();
    let ok = match args.selection {
        generators::BoneSelection::Joint(id) => bs.remove_joint(id),
        generators::BoneSelection::Bone(bone_id) => {
            // Find the joints at each end before removing the bone.
            let endpoints: Option<(u32, u32)> = bs
                .bones
                .iter()
                .find(|b| b.id == bone_id)
                .map(|b| (b.joint_a, b.joint_b));
            let removed = bs.remove_bone(bone_id);
            if removed {
                // Remove joints that are now orphaned (no remaining bones).
                if let Some((ja, jb)) = endpoints {
                    if bs.connected_bones(ja).is_empty() {
                        bs.remove_joint(ja);
                    }
                    if bs.connected_bones(jb).is_empty() {
                        bs.remove_joint(jb);
                    }
                }
            }
            removed
        }
    };
    if ok {
        *state.gizmos.generator_gizmo_center.lock() = None;
    }
    Ok(ok)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BoneSetJointRadiusArgs {
    id: u32,
    radius: f32,
}

#[tauri::command]
pub(crate) fn bone_set_joint_radius(
    state: State<'_, Arc<ViewerState>>,
    args: BoneSetJointRadiusArgs,
) -> Result<bool, String> {
    let mut bs = state.gizmos.bone_session.lock();
    Ok(bs.set_joint_radius(args.id, args.radius))
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BoneSetJointPositionArgs {
    id: u32,
    x: f32,
    y: f32,
    z: f32,
}

#[tauri::command]
pub(crate) fn bone_set_joint_position(
    state: State<'_, Arc<ViewerState>>,
    args: BoneSetJointPositionArgs,
) -> Result<bool, String> {
    let mut bs = state.gizmos.bone_session.lock();
    let ok = bs.set_joint_position(args.id, args.x, args.y, args.z);
    if ok {
        // Keep the shared gizmo centered on the joint as it moves.
        *state.gizmos.generator_gizmo_center.lock() = Some([args.x, args.y, args.z]);
    }
    Ok(ok)
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BoneGizmoPointerArgs {
    nx: f32,
    ny: f32,
}

#[tauri::command]
pub(crate) fn bone_gizmo_pointer_down(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: BoneGizmoPointerArgs,
) -> Result<bool, String> {
    let (w, h) = {
        let v = state.gpu.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let cam = state.cam.camera.lock();
    let bs = state.gizmos.bone_session.lock();
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let Some((handle, joint_id)) = generators::pick_bone_gizmo_handle(&bs, &cam, w, h, sx, sy)
    else {
        return Ok(false);
    };
    let Some(drag) = generators::bone_gizmo_begin_drag(&bs, &cam, w, h, sx, sy, handle, joint_id)
    else {
        return Ok(false);
    };
    drop(bs);
    drop(cam);
    *state.gizmos.bone_gizmo_drag.lock() = Some(drag);
    wake_viewport_loop(&app);
    Ok(true)
}

#[tauri::command]
pub(crate) fn bone_gizmo_pointer_move(
    app: AppHandle,
    state: State<'_, Arc<ViewerState>>,
    args: BoneGizmoPointerArgs,
) -> Result<(), String> {
    let gizmo_drag = state.gizmos.bone_gizmo_drag.lock().clone();
    if let Some(drag) = gizmo_drag {
        let (w, h) = {
            let v = state.gpu.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Ok(());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let cam = state.cam.camera.lock();
        let mut bs = state.gizmos.bone_session.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        generators::bone_gizmo_apply_drag(&mut bs, &cam, w, h, sx, sy, &drag);
        wake_viewport_loop(&app);
        return Ok(());
    }
    let ik_drag = state.gizmos.bone_ik_drag.lock().clone();
    if let Some(drag) = ik_drag {
        let (w, h) = {
            let v = state.gpu.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Ok(());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let cam = state.cam.camera.lock();
        let mut bs = state.gizmos.bone_session.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        generators::ik_drag_update(&mut bs, &cam, w, h, sx, sy, &drag);
        wake_viewport_loop(&app);
    }
    Ok(())
}

#[tauri::command]
pub(crate) fn bone_gizmo_pointer_up(state: State<'_, Arc<ViewerState>>) -> Result<(), String> {
    *state.gizmos.bone_gizmo_drag.lock() = None;
    *state.gizmos.bone_ik_drag.lock() = None;
    Ok(())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BoneIkDragStartArgs {
    joint_id: u32,
    nx: f32,
    ny: f32,
}

#[tauri::command]
pub(crate) fn bone_ik_drag_start(
    state: State<'_, Arc<ViewerState>>,
    args: BoneIkDragStartArgs,
) -> Result<bool, String> {
    let (w, h) = {
        let v = state.gpu.viewer.lock();
        let Some(viewer) = v.as_ref() else {
            return Err("viewer not ready".into());
        };
        let (w, h) = viewer.viewport_size();
        (w as f32, h as f32)
    };
    let cam = state.cam.camera.lock();
    let bs = state.gizmos.bone_session.lock();
    let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
    let Some(drag) = generators::ik_drag_begin(&bs, &cam, w, h, sx, sy, args.joint_id) else {
        return Ok(false);
    };
    drop(bs);
    drop(cam);
    *state.gizmos.bone_ik_drag.lock() = Some(drag);
    Ok(true)
}

// ── Shape generator ─────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GeneratorShapeArgs {
    nx: f32,
    ny: f32,
    #[serde(default = "default_shape_kind")]
    shape: String,
    #[serde(default = "default_shape_size")]
    size: i32,
    #[serde(default)]
    rot_x: f32,
    #[serde(default)]
    rot_y: f32,
    #[serde(default)]
    rot_z: f32,
    color: u32,
    material: String,
    #[serde(default = "crate::preview::default_true")]
    overwrite: bool,
    /// Explicit gizmo center passed from the frontend to avoid race conditions.
    #[serde(default)]
    gizmo_center: Option<[f32; 3]>,
}

fn default_shape_kind() -> String {
    "cube".into()
}

pub(crate) fn default_shape_size() -> i32 {
    8
}

#[tauri::command]
pub(crate) fn generator_shape_at_screen(
    state: State<'_, Arc<ViewerState>>,
    app: AppHandle,
    args: GeneratorShapeArgs,
) -> Result<bool, String> {
    use crate::voxel_edit::{ensure_grid_fits_coords, VoxelEditDelta};
    use crate::voxelle::start_shape::StartShape;
    let material = voxelle::MaterialId::from_str_id(&args.material);
    let shape = StartShape::from_str_id(&args.shape);
    // Use the explicit gizmo center from the frontend args (avoids race with
    // clear_generator_gizmo_center). Fall back to the state if not provided.
    let gen_center = args
        .gizmo_center
        .or_else(|| *state.gizmos.generator_gizmo_center.lock());
    let deltas = if let Some([gx, gy, gz]) = gen_center {
        let mut fg = state.file.current_file.lock();
        let mut vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let origin = (gx as i32, gy as i32, gz as i32);
        let positions = crate::generators::compute_shape_positions(
            shape,
            args.size,
            origin,
            (args.rot_x, args.rot_y, args.rot_z),
        );
        if positions.is_empty() {
            Vec::new()
        } else {
            ensure_grid_fits_coords(file, positions.iter().copied());
            let mut deltas = Vec::with_capacity(positions.len());
            for (x, y, z) in positions {
                if !args.overwrite && vmap.contains_key(&(x, y, z)) {
                    continue;
                }
                deltas.push(VoxelEditDelta::Added(voxelle::Voxel {
                    x,
                    y,
                    z,
                    color: args.color,
                    material,
                    object_id: 0,
                }));
            }
            deltas
        }
    } else {
        let (w, h) = {
            let v = state.gpu.viewer.lock();
            let Some(viewer) = v.as_ref() else {
                return Err("viewer not ready".into());
            };
            let (w, h) = viewer.viewport_size();
            (w as f32, h as f32)
        };
        let mut fg = state.file.current_file.lock();
        let mut vm = state.file.voxel_map.lock();
        let Some(file) = fg.as_mut() else {
            return Err("no model loaded".into());
        };
        let Some(vmap) = vm.as_mut() else {
            return Err("voxel index not ready".into());
        };
        let cam = state.cam.camera.lock();
        let (sx, sy) = viewport_texels_from_norm(args.nx, args.ny, w, h);
        crate::generators::generator_shape_at_screen(
            file,
            vmap,
            &cam,
            w,
            h,
            sx,
            sy,
            shape,
            args.size,
            args.rot_x,
            args.rot_y,
            args.rot_z,
            args.color,
            material,
            args.overwrite,
        )?
    };
    commit_generator_edits(&state, &app, deltas)
}

// ── Generator gizmo override ────────────────────────────────────────────

#[tauri::command]
pub(crate) fn set_generator_gizmo_center(state: State<'_, Arc<ViewerState>>, center: [f32; 3]) {
    *state.gizmos.generator_gizmo_center.lock() = Some(center);
}

#[tauri::command]
pub(crate) fn clear_generator_gizmo_center(state: State<'_, Arc<ViewerState>>) {
    *state.gizmos.generator_gizmo_center.lock() = None;
}
