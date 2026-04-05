use super::common::{hash3, smoothstep, v3_add, v3_cross, v3_len, v3_normalize, v3_scale, v3_sub};
use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::{
    effective_ray_grid_size, ensure_grid_fits_coord, ray_first_solid, screen_to_world_ray,
    VoxelEditDelta,
};
use crate::voxelle::{MaterialId, Scene, Voxel, VoxelleFile};
use ahash::AHashMap;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Fin multiplier: finMul(scale) = 1 + ((scale-1)/7) * 6.25 for scale in [1,8]
// ---------------------------------------------------------------------------

fn fin_mul(scale: i32) -> f32 {
    let s = (scale as f32).clamp(1.0, 8.0);
    1.0 + ((s - 1.0) / 7.0) * 6.25
}

// ---------------------------------------------------------------------------
// Voxel cap
// ---------------------------------------------------------------------------

fn compute_piscina_voxel_cap(length: i32, width: i32, thickness: i32) -> usize {
    let l = length as f32;
    let w = width as f32;
    let t = thickness as f32;
    let raw = (std::f32::consts::PI * w * t * l * 0.88 + (w + t) * l * 0.38 + 900.0) * 1.12;
    (raw.ceil() as usize).clamp(2200, 52000)
}

// ---------------------------------------------------------------------------
// Species outline: returns (half_side, half_dorsal, section_power) at a
// parametric station t in [0,1] (0=nose, 1=tail).
// ---------------------------------------------------------------------------

struct SpeciesOutline {
    half_side: f32,
    half_dorsal: f32,
    section_power: f32,
}

fn species_outline(species: &str, t: f32, w: f32, th: f32) -> SpeciesOutline {
    match species {
        "bass" => {
            // Deep body, wider mid-section
            let bell = smoothstep(0.0, 0.25, t) * smoothstep(1.0, 0.6, t);
            let belly_boost = 1.0 + 0.25 * smoothstep(0.2, 0.45, t) * smoothstep(0.7, 0.45, t);
            SpeciesOutline {
                half_side: w * bell * belly_boost,
                half_dorsal: th * bell * 1.15,
                section_power: 2.0,
            }
        }
        "goldfish" => {
            // Very round, compressed
            let bell = smoothstep(0.0, 0.2, t) * smoothstep(1.0, 0.55, t);
            SpeciesOutline {
                half_side: w * bell * 1.2,
                half_dorsal: th * bell * 1.3,
                section_power: 2.5,
            }
        }
        "tuna" => {
            // Torpedo, narrow peduncle
            let bell = smoothstep(0.0, 0.15, t) * smoothstep(1.0, 0.65, t);
            let peduncle_narrow = 1.0 - 0.5 * smoothstep(0.7, 0.95, t);
            SpeciesOutline {
                half_side: w * bell * peduncle_narrow,
                half_dorsal: th * bell * peduncle_narrow * 0.9,
                section_power: 2.8,
            }
        }
        "eel" => {
            // Elongated, uniform width
            let rise = smoothstep(0.0, 0.08, t);
            let fall = smoothstep(1.0, 0.92, t);
            SpeciesOutline {
                half_side: w * 0.45 * rise * fall,
                half_dorsal: th * 0.45 * rise * fall,
                section_power: 2.0,
            }
        }
        // Default: trout -- fusiform, moderate taper
        _ => {
            let bell = smoothstep(0.0, 0.2, t) * smoothstep(1.0, 0.7, t);
            SpeciesOutline {
                half_side: w * bell,
                half_dorsal: th * bell,
                section_power: 2.2,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Spine construction
// ---------------------------------------------------------------------------

struct SpineStation {
    pos: [f32; 3],
    tangent: [f32; 3],
    side: [f32; 3],
    up: [f32; 3],
    t: f32,
}

fn build_spine(
    origin: [f32; 3],
    forward: [f32; 3],
    side_base: [f32; 3],
    up_base: [f32; 3],
    length: i32,
    bend: f32,
    s_curve: f32,
) -> Vec<SpineStation> {
    let n = length.max(4);
    let mut points: Vec<[f32; 3]> = Vec::with_capacity(n as usize + 1);

    for i in 0..=n {
        let t = i as f32 / n as f32;
        // Lateral bend: gentle arc toward the side
        let bend_offset = bend * (std::f32::consts::PI * t).sin();
        // S-curve: sinusoidal lateral undulation
        let s_offset = s_curve * (2.0 * std::f32::consts::PI * t).sin();
        let lateral = bend_offset + s_offset;

        let p = v3_add(
            v3_add(origin, v3_scale(forward, i as f32)),
            v3_scale(side_base, lateral),
        );
        points.push(p);
    }

    // Build Frenet frames using central differences with phantom ends
    let count = points.len();
    let mut stations: Vec<SpineStation> = Vec::with_capacity(count);

    for i in 0..count {
        let t = i as f32 / (count - 1) as f32;
        // Central difference for tangent (phantom endpoints)
        let prev = if i == 0 {
            v3_sub(v3_scale(points[0], 2.0), points[1])
        } else {
            points[i - 1]
        };
        let next = if i == count - 1 {
            v3_sub(v3_scale(points[count - 1], 2.0), points[count - 2])
        } else {
            points[i + 1]
        };
        let tangent = v3_normalize(v3_sub(next, prev));

        // Side = cross(tangent, up_base), then re-derive up
        let mut side = v3_normalize(v3_cross(tangent, up_base));
        if v3_len(v3_cross(tangent, up_base)) < 1e-6 {
            side = side_base;
        }
        let up = v3_normalize(v3_cross(side, tangent));

        stations.push(SpineStation {
            pos: points[i],
            tangent,
            side,
            up,
            t,
        });
    }
    stations
}

// ---------------------------------------------------------------------------
// Frame derivation from face hit
// ---------------------------------------------------------------------------

fn derive_frame(
    face_empty: VoxelCoord,
    solid: VoxelCoord,
    anchor_u: i32,
    anchor_v: i32,
) -> ([f32; 3], [f32; 3], [f32; 3], [f32; 3]) {
    let nx = (face_empty.0 - solid.0) as f32;
    let ny = (face_empty.1 - solid.1) as f32;
    let nz = (face_empty.2 - solid.2) as f32;
    let normal = v3_normalize([nx, ny, nz]);

    // Forward = some tangent perpendicular to normal
    let candidate = if normal[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let forward = v3_normalize(v3_cross(normal, candidate));
    let side = v3_normalize(v3_cross(normal, forward));
    let up = normal;

    // Origin at face_empty, shifted by anchor offsets along forward/side
    let origin = [
        face_empty.0 as f32 + forward[0] * anchor_u as f32 + side[0] * anchor_v as f32,
        face_empty.1 as f32 + forward[1] * anchor_u as f32 + side[1] * anchor_v as f32,
        face_empty.2 as f32 + forward[2] * anchor_u as f32 + side[2] * anchor_v as f32,
    ];

    (origin, forward, side, up)
}

// ---------------------------------------------------------------------------
// Voxel placement helper
// ---------------------------------------------------------------------------

fn place_voxel(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    seen: &mut HashSet<VoxelCoord>,
    out: &mut Vec<VoxelEditDelta>,
    x: i32,
    y: i32,
    z: i32,
    color: u32,
    material: MaterialId,
) {
    ensure_grid_fits_coord(file, x, y, z);
    if !seen.insert((x, y, z)) {
        return;
    }
    if voxel_map.contains_key(&(x, y, z)) {
        return;
    }
    let nv = Voxel {
        x,
        y,
        z,
        color,
        material,
        object_id: file.active_object_id,
    };
    let idx = file.voxels.len();
    file.voxels.push(nv);
    voxel_map.insert((x, y, z), idx);
    out.push(VoxelEditDelta::Added(nv));
}

// ---------------------------------------------------------------------------
// Median fin (dorsal / anal) generation
// ---------------------------------------------------------------------------

fn generate_median_fin(
    stations: &[SpineStation],
    species: &str,
    w_half: f32,
    t_half: f32,
    fin_scale: i32,
    dorsal: bool, // true=dorsal (up), false=anal (down)
    t_start: f32,
    t_end: f32,
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    seen: &mut HashSet<VoxelCoord>,
    out: &mut Vec<VoxelEditDelta>,
    color: u32,
    material: MaterialId,
) {
    let mul = fin_mul(fin_scale);
    let max_h = (mul * t_half * 0.7).ceil() as i32;
    if max_h < 1 {
        return;
    }

    for st in stations.iter() {
        if st.t < t_start || st.t > t_end {
            continue;
        }
        // Height bell within fin range
        let fin_t = (st.t - t_start) / (t_end - t_start);
        let height_scale = smoothstep(0.0, 0.2, fin_t) * smoothstep(1.0, 0.8, fin_t);
        let h = (max_h as f32 * height_scale).ceil() as i32;

        let outline = species_outline(species, st.t, w_half, t_half);
        let base_offset = outline.half_dorsal;

        let dir = if dorsal { 1.0 } else { -1.0 };

        for hi in 0..h {
            let offset = base_offset + dir * (hi as f32 + 1.0);
            let p = v3_add(st.pos, v3_scale(st.up, offset));
            let x = p[0].round() as i32;
            let y = p[1].round() as i32;
            let z = p[2].round() as i32;
            place_voxel(file, voxel_map, seen, out, x, y, z, color, material);
        }
    }
}

// ---------------------------------------------------------------------------
// Caudal (tail) fin
// ---------------------------------------------------------------------------

fn generate_caudal_fin(
    stations: &[SpineStation],
    species: &str,
    w_half: f32,
    t_half: f32,
    fin_scale: i32,
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    seen: &mut HashSet<VoxelCoord>,
    out: &mut Vec<VoxelEditDelta>,
    color: u32,
    material: MaterialId,
) {
    if stations.is_empty() {
        return;
    }
    let mul = fin_mul(fin_scale);
    let tail_span = (mul * t_half * 1.1).ceil() as i32;
    let tail_depth = (mul * w_half * 0.5).ceil().max(1.0) as i32;

    // Fork angle depends on species
    let fork_factor: f32 = match species {
        "tuna" => 0.9,
        "goldfish" => 0.3,
        "eel" => 0.15,
        "bass" => 0.55,
        _ => 0.6, // trout
    };

    let last = &stations[stations.len() - 1];

    for d in 0..tail_depth {
        let fwd_p = v3_add(last.pos, v3_scale(last.tangent, d as f32 + 1.0));
        let spread = (d as f32 + 1.0) / tail_depth as f32;

        for hi in -tail_span..=tail_span {
            if hi == 0 && d > 0 {
                continue; // leave center gap for fork
            }
            // Fork: fan out from center
            let fan = hi as f32 + hi.signum() as f32 * spread * fork_factor * tail_span as f32;
            let p = v3_add(fwd_p, v3_scale(last.up, fan));
            let x = p[0].round() as i32;
            let y = p[1].round() as i32;
            let z = p[2].round() as i32;
            place_voxel(file, voxel_map, seen, out, x, y, z, color, material);
        }
    }
}

// ---------------------------------------------------------------------------
// Paired fin (pectoral / pelvic)
// ---------------------------------------------------------------------------

fn generate_paired_fin(
    stations: &[SpineStation],
    species: &str,
    w_half: f32,
    t_half: f32,
    fin_scale: i32,
    t_center: f32,
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    seen: &mut HashSet<VoxelCoord>,
    out: &mut Vec<VoxelEditDelta>,
    color: u32,
    material: MaterialId,
) {
    let mul = fin_mul(fin_scale);
    let fan_len = (mul * w_half * 0.6).ceil().max(1.0) as i32;
    let fan_width = (mul * 0.4 * t_half).ceil().max(1.0) as i32;

    // Find nearest station to t_center
    let station = stations.iter().min_by(|a, b| {
        ((a.t - t_center).abs())
            .partial_cmp(&(b.t - t_center).abs())
            .unwrap()
    });
    let Some(st) = station else { return };

    let outline = species_outline(species, st.t, w_half, t_half);
    let base_lat = outline.half_side;

    // Place on both sides
    for sign in [-1.0_f32, 1.0] {
        let down_dir = v3_scale(st.up, -0.3); // fins angle slightly downward

        for fi in 0..fan_len {
            let spread = (fi as f32 + 1.0) / fan_len as f32;
            for fw in -fan_width..=fan_width {
                let taper = 1.0 - spread * 0.5;
                if (fw.abs() as f32) > fan_width as f32 * taper {
                    continue;
                }
                let p = v3_add(
                    v3_add(
                        st.pos,
                        v3_add(
                            v3_scale(st.side, sign * (base_lat + fi as f32 + 1.0)),
                            v3_scale(down_dir, fi as f32),
                        ),
                    ),
                    v3_scale(st.tangent, fw as f32),
                );
                let x = p[0].round() as i32;
                let y = p[1].round() as i32;
                let z = p[2].round() as i32;
                place_voxel(file, voxel_map, seen, out, x, y, z, color, material);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Adipose fin (small bump dorsal, posterior)
// ---------------------------------------------------------------------------

fn generate_adipose_fin(
    stations: &[SpineStation],
    species: &str,
    w_half: f32,
    t_half: f32,
    fin_scale: i32,
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    seen: &mut HashSet<VoxelCoord>,
    out: &mut Vec<VoxelEditDelta>,
    color: u32,
    material: MaterialId,
) {
    let mul = fin_mul(fin_scale);
    let h = (mul * t_half * 0.3).ceil().max(1.0) as i32;

    for st in stations.iter() {
        if st.t < 0.65 || st.t > 0.78 {
            continue;
        }
        let fin_t = (st.t - 0.65) / 0.13;
        let height_scale = smoothstep(0.0, 0.3, fin_t) * smoothstep(1.0, 0.7, fin_t);
        let hi = (h as f32 * height_scale).ceil() as i32;

        let outline = species_outline(species, st.t, w_half, t_half);

        for j in 0..hi {
            let p = v3_add(
                st.pos,
                v3_scale(st.up, outline.half_dorsal + j as f32 + 1.0),
            );
            let x = p[0].round() as i32;
            let y = p[1].round() as i32;
            let z = p[2].round() as i32;
            place_voxel(file, voxel_map, seen, out, x, y, z, color, material);
        }
    }
}

// ---------------------------------------------------------------------------
// Core body generation: superellipse cross-sections along spine
// ---------------------------------------------------------------------------

fn generate_body(
    stations: &[SpineStation],
    species: &str,
    w_half: f32,
    t_half: f32,
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    seen: &mut HashSet<VoxelCoord>,
    out: &mut Vec<VoxelEditDelta>,
    color: u32,
    material: MaterialId,
    cap: usize,
) {
    for st in stations.iter() {
        let outline = species_outline(species, st.t, w_half, t_half);
        let h_lat = outline.half_side;
        let h_dw = outline.half_dorsal;
        let p = outline.section_power;

        if h_lat < 0.5 || h_dw < 0.5 {
            continue;
        }

        let ri = h_lat.ceil() as i32;
        let rj = h_dw.ceil() as i32;

        for dv in -ri..=ri {
            for dw in -rj..=rj {
                // Superellipse test: |dv/hLat|^p + |dw/hDw|^p <= 1
                let uv = (dv as f32).abs() / h_lat;
                let uw = (dw as f32).abs() / h_dw;
                if uv.powf(p) + uw.powf(p) > 1.0 {
                    continue;
                }

                let world = v3_add(
                    st.pos,
                    v3_add(v3_scale(st.side, dv as f32), v3_scale(st.up, dw as f32)),
                );
                let x = world[0].round() as i32;
                let y = world[1].round() as i32;
                let z = world[2].round() as i32;

                if out.len() >= cap {
                    return;
                }
                place_voxel(file, voxel_map, seen, out, x, y, z, color, material);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main generation logic
// ---------------------------------------------------------------------------

pub fn generate_piscina_deltas(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    face_empty: VoxelCoord,
    solid: VoxelCoord,
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
    material: MaterialId,
) -> Vec<VoxelEditDelta> {
    let l = length.clamp(4, 80);
    let w = width_param.clamp(1, 40);
    let t = thickness.clamp(1, 40);
    let w_half = w as f32 / 2.0;
    let t_half = t as f32 / 2.0;

    let cap = compute_piscina_voxel_cap(l, w, t);

    let (origin, forward, side_base, up_base) =
        derive_frame(face_empty, solid, anchor_offset_u, anchor_offset_v);

    // Add a tiny seed-based jitter to bend so each fish is unique
    let jitter = (hash3(seed, 0, 0, seed) - 0.5) * 0.15;
    let bend = spine_bend + jitter;

    let stations = build_spine(origin, forward, side_base, up_base, l, bend, spine_s_curve);

    let mut out = Vec::new();
    let mut seen: HashSet<VoxelCoord> = HashSet::new();

    // 1. Body
    generate_body(
        &stations, species, w_half, t_half, file, voxel_map, &mut seen, &mut out, color, material,
        cap,
    );

    // 2. Dorsal fin
    if show_fin_dorsal {
        let (ts, te) = match species {
            "eel" => (0.15, 0.85),
            "bass" => (0.20, 0.65),
            _ => (0.25, 0.55),
        };
        generate_median_fin(
            &stations, species, w_half, t_half, fin_dorsal, true, ts, te, file, voxel_map,
            &mut seen, &mut out, color, material,
        );
    }

    // 3. Anal fin
    if show_fin_anal {
        let (ts, te) = match species {
            "eel" => (0.40, 0.85),
            _ => (0.55, 0.75),
        };
        generate_median_fin(
            &stations, species, w_half, t_half, fin_anal, false, ts, te, file, voxel_map,
            &mut seen, &mut out, color, material,
        );
    }

    // 4. Caudal (tail) fin
    if show_fin_caudal {
        generate_caudal_fin(
            &stations, species, w_half, t_half, fin_caudal, file, voxel_map, &mut seen, &mut out,
            color, material,
        );
    }

    // 5. Pectoral fins
    if show_fin_pectoral {
        let tc = match species {
            "eel" => 0.12,
            _ => 0.22,
        };
        generate_paired_fin(
            &stations,
            species,
            w_half,
            t_half,
            fin_pectoral,
            tc,
            file,
            voxel_map,
            &mut seen,
            &mut out,
            color,
            material,
        );
    }

    // 6. Pelvic fins
    if show_fin_pelvic {
        let tc = match species {
            "eel" => 0.35,
            _ => 0.42,
        };
        generate_paired_fin(
            &stations, species, w_half, t_half, fin_pelvic, tc, file, voxel_map, &mut seen,
            &mut out, color, material,
        );
    }

    // 7. Adipose fin (trout, salmon-family)
    if show_fin_adipose {
        generate_adipose_fin(
            &stations,
            species,
            w_half,
            t_half,
            fin_adipose,
            file,
            voxel_map,
            &mut seen,
            &mut out,
            color,
            material,
        );
    }

    // Enforce voxel cap
    out.truncate(cap);
    out
}

// ---------------------------------------------------------------------------
// Public face-click entry point (web parity)
// ---------------------------------------------------------------------------

/// Face-click piscina (fish) generator. Spawns a parametric fish from the
/// clicked face, oriented along a tangent perpendicular to the face normal.
pub fn generator_piscina_at_screen(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
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
    material: MaterialId,
) -> Result<Vec<VoxelEditDelta>, String> {
    let grid_size = effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((solid, prev)) = ray_first_solid(origin, dir, voxel_map, grid_size) else {
        return Ok(Vec::new());
    };
    let Some(face_empty) = prev else {
        return Ok(Vec::new());
    };
    Ok(generate_piscina_deltas(
        file,
        voxel_map,
        face_empty,
        solid,
        seed,
        species,
        length,
        width_param,
        thickness,
        spine_bend,
        spine_s_curve,
        fin_dorsal,
        fin_anal,
        fin_caudal,
        fin_pectoral,
        fin_pelvic,
        fin_adipose,
        show_fin_dorsal,
        show_fin_anal,
        show_fin_caudal,
        show_fin_pectoral,
        show_fin_pelvic,
        show_fin_adipose,
        anchor_offset_u,
        anchor_offset_v,
        color,
        material,
    ))
}

/// Preview-only: compute the set of voxel coords a fish would occupy,
/// without mutating the real file. Used for hover preview.
#[allow(clippy::too_many_arguments)]
pub fn preview_piscina_at_screen(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
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
    material: MaterialId,
) -> Vec<(VoxelCoord, u32)> {
    let grid_size = effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let Some((solid, prev)) = ray_first_solid(origin, dir, voxel_map, grid_size) else {
        return Vec::new();
    };
    let Some(face_empty) = prev else {
        return Vec::new();
    };
    let mut stub_file = VoxelleFile {
        version: 0,
        grid_size: file.grid_size,
        scene: Scene::default(),
        scene_extra: None,
        mood: None,
        lighting: None,
        voxels: Vec::new(),
        objects: Vec::new(),
        active_object_id: 0,
    };
    let mut stub_map: AHashMap<VoxelCoord, usize> = AHashMap::new();
    generate_piscina_deltas(
        &mut stub_file,
        &mut stub_map,
        face_empty,
        solid,
        seed,
        species,
        length,
        width_param,
        thickness,
        spine_bend,
        spine_s_curve,
        fin_dorsal,
        fin_anal,
        fin_caudal,
        fin_pectoral,
        fin_pelvic,
        fin_adipose,
        show_fin_dorsal,
        show_fin_anal,
        show_fin_caudal,
        show_fin_pectoral,
        show_fin_pelvic,
        show_fin_adipose,
        anchor_offset_u,
        anchor_offset_v,
        color,
        material,
    )
    .into_iter()
    .filter_map(|d| {
        if let VoxelEditDelta::Added(v) = d {
            if !voxel_map.contains_key(&(v.x, v.y, v.z)) {
                return Some(((v.x, v.y, v.z), v.color));
            }
        }
        None
    })
    .collect()
}
