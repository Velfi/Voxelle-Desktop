use super::common::{
    v3_add, v3_normalize, v3_round, v3_scale, PlacementFrame, V3,
};
use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
use crate::voxel_edit::{
    effective_ray_grid_size, ensure_grid_fits_coord, ray_first_solid, screen_to_world_ray,
    VoxelEditDelta,
};
use crate::voxelle::{MaterialId, Scene, Voxel, VoxelleFile};
use ahash::AHashMap;
use std::collections::HashSet;

const VOXEL_CAP: usize = 10_000;

// ---------------------------------------------------------------------------
// Body segment radii
// ---------------------------------------------------------------------------

/// Returns (half_width, half_height) for a given distance from the nose.
fn segment_radii(
    dist: f32,
    head_len: f32,
    thorax_len: f32,
    _abdomen_len: f32,
    total_len: f32,
    body_hw: f32,
    body_hh: f32,
    abdomen_taper: f32,
    head_shape: i32,
) -> (f32, f32) {
    if dist < 0.0 || dist > total_len {
        return (0.0, 0.0);
    }
    let thorax_start = head_len;
    let abdomen_start = head_len + thorax_len;

    if dist < thorax_start {
        // Head segment
        let t = dist / head_len.max(1.0); // 0=nose, 1=back-of-head
                                          // Head shape: 0=round, 1=pointed, 2=flat
        let profile = match head_shape {
            1 => {
                // Pointed: narrow at nose, wider at back
                t.powf(1.5)
            }
            2 => {
                // Flat: nearly full width immediately
                if t < 0.15 {
                    t / 0.15
                } else {
                    1.0
                }
            }
            _ => {
                // Round: elliptical profile
                (1.0 - (1.0 - t).powi(2)).sqrt().min(1.0)
            }
        };
        let w = body_hw * 0.85 * profile;
        let h = body_hh * 0.8 * profile;
        (w, h)
    } else if dist < abdomen_start {
        // Thorax segment – bulges mid-segment, pinches at ends (petiole)
        let t = (dist - thorax_start) / thorax_len.max(1.0);
        // Bell curve
        let bulge = (-(t - 0.5).powi(2) * 8.0).exp();
        let pinch = 0.55 + 0.45 * bulge;
        let w = body_hw * pinch;
        let h = body_hh * pinch;
        (w, h)
    } else {
        // Abdomen segment – wider early, tapers toward rear
        let t = (dist - abdomen_start) / (total_len - abdomen_start).max(1.0);
        // Wide band early, taper toward tail
        let swell = (1.0 - (t - 0.3).powi(2) * 2.0).max(0.0).sqrt();
        let taper_factor = 1.0 - abdomen_taper * t;
        let profile = swell * taper_factor.max(0.0);
        let w = body_hw * 1.1 * profile;
        let h = body_hh * profile;
        (w, h)
    }
}

// ---------------------------------------------------------------------------
// Bresenham 3D line: emit voxels along segment
// ---------------------------------------------------------------------------

fn bresenham_line(a: (i32, i32, i32), b: (i32, i32, i32)) -> Vec<(i32, i32, i32)> {
    let mut out = Vec::new();
    let dx = (b.0 - a.0).abs();
    let dy = (b.1 - a.1).abs();
    let dz = (b.2 - a.2).abs();
    let sx = if b.0 > a.0 { 1 } else { -1 };
    let sy = if b.1 > a.1 { 1 } else { -1 };
    let sz = if b.2 > a.2 { 1 } else { -1 };
    let dm = dx.max(dy).max(dz);
    let mut x = a.0;
    let mut y = a.1;
    let mut z = a.2;
    let mut ex = dm / 2;
    let mut ey = dm / 2;
    let mut ez = dm / 2;
    for _ in 0..=dm {
        out.push((x, y, z));
        // Advance along the dominant axis
        ex -= dx;
        ey -= dy;
        ez -= dz;
        if ex < 0 {
            ex += dm;
            x += sx;
        }
        if ey < 0 {
            ey += dm;
            y += sy;
        }
        if ez < 0 {
            ez += dm;
            z += sz;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Species-default leg parameters
// ---------------------------------------------------------------------------

/// Per-pair leg offsets as (forward_frac, side_offset, up_offset) for hip,
/// knee, and foot. `forward_frac` is fraction along thorax length.
struct LegPairDef {
    hip_fwd_frac: f32,
    hip_side: f32,
    hip_up: f32,
    knee_side: f32,
    knee_down: f32,
    foot_side: f32,
    foot_down: f32,
}

fn species_leg_defs(species: &str) -> [LegPairDef; 3] {
    match species {
        "dragonfly" => [
            // Front – clustered at thorax front
            LegPairDef {
                hip_fwd_frac: 0.15,
                hip_side: 1.0,
                hip_up: -0.3,
                knee_side: 1.8,
                knee_down: 1.5,
                foot_side: 2.0,
                foot_down: 3.0,
            },
            // Mid
            LegPairDef {
                hip_fwd_frac: 0.3,
                hip_side: 1.0,
                hip_up: -0.3,
                knee_side: 1.8,
                knee_down: 1.5,
                foot_side: 2.0,
                foot_down: 3.0,
            },
            // Hind
            LegPairDef {
                hip_fwd_frac: 0.45,
                hip_side: 1.0,
                hip_up: -0.3,
                knee_side: 2.0,
                knee_down: 1.5,
                foot_side: 2.5,
                foot_down: 3.5,
            },
        ],
        "grasshopper" => [
            LegPairDef {
                hip_fwd_frac: 0.1,
                hip_side: 0.8,
                hip_up: -0.2,
                knee_side: 1.2,
                knee_down: 1.0,
                foot_side: 1.5,
                foot_down: 2.5,
            },
            LegPairDef {
                hip_fwd_frac: 0.4,
                hip_side: 0.9,
                hip_up: -0.2,
                knee_side: 1.4,
                knee_down: 1.2,
                foot_side: 1.6,
                foot_down: 3.0,
            },
            // Long hind legs
            LegPairDef {
                hip_fwd_frac: 0.75,
                hip_side: 1.0,
                hip_up: 0.2,
                knee_side: 2.5,
                knee_down: -1.5,
                foot_side: 3.5,
                foot_down: 4.5,
            },
        ],
        "junebug" => [
            LegPairDef {
                hip_fwd_frac: 0.15,
                hip_side: 1.0,
                hip_up: -0.4,
                knee_side: 1.4,
                knee_down: 1.0,
                foot_side: 1.5,
                foot_down: 1.8,
            },
            LegPairDef {
                hip_fwd_frac: 0.45,
                hip_side: 1.1,
                hip_up: -0.4,
                knee_side: 1.5,
                knee_down: 1.0,
                foot_side: 1.6,
                foot_down: 1.8,
            },
            LegPairDef {
                hip_fwd_frac: 0.75,
                hip_side: 1.0,
                hip_up: -0.4,
                knee_side: 1.5,
                knee_down: 1.0,
                foot_side: 1.6,
                foot_down: 2.0,
            },
        ],
        "fly" => [
            LegPairDef {
                hip_fwd_frac: 0.15,
                hip_side: 0.9,
                hip_up: -0.3,
                knee_side: 1.5,
                knee_down: 1.2,
                foot_side: 2.0,
                foot_down: 2.8,
            },
            LegPairDef {
                hip_fwd_frac: 0.45,
                hip_side: 1.0,
                hip_up: -0.3,
                knee_side: 1.6,
                knee_down: 1.3,
                foot_side: 2.2,
                foot_down: 3.0,
            },
            LegPairDef {
                hip_fwd_frac: 0.75,
                hip_side: 0.9,
                hip_up: -0.3,
                knee_side: 1.8,
                knee_down: 1.5,
                foot_side: 2.5,
                foot_down: 3.5,
            },
        ],
        // "bee" and default
        _ => [
            LegPairDef {
                hip_fwd_frac: 0.1,
                hip_side: 0.9,
                hip_up: -0.4,
                knee_side: 1.3,
                knee_down: 1.2,
                foot_side: 1.5,
                foot_down: 2.2,
            },
            LegPairDef {
                hip_fwd_frac: 0.4,
                hip_side: 1.0,
                hip_up: -0.4,
                knee_side: 1.4,
                knee_down: 1.2,
                foot_side: 1.6,
                foot_down: 2.5,
            },
            LegPairDef {
                hip_fwd_frac: 0.7,
                hip_side: 1.0,
                hip_up: -0.4,
                knee_side: 1.5,
                knee_down: 1.3,
                foot_side: 1.8,
                foot_down: 2.8,
            },
        ],
    }
}

// ---------------------------------------------------------------------------
// Wing sheet generation
// ---------------------------------------------------------------------------

/// Generate a flat wing sheet as a list of voxel coords.
/// `hinge` is the dorsal-lateral attachment point.
/// `spread` tilts from lateral toward +forward, `pitch` lifts tips.
/// `wing_shape`: 0=elliptical, 1=pointed, 2=rounded rectangle.
/// `forward_cant` adds forward sweep (dragonfly style).
fn wing_sheet(
    hinge: V3,
    forward: V3,
    side: V3,
    up: V3,
    length: i32,
    width: i32,
    spread: f32,
    pitch: f32,
    wing_shape: i32,
    sign: f32, // +1 right, -1 left
    forward_cant: f32,
) -> Vec<(i32, i32, i32)> {
    let mut out = Vec::new();
    if length <= 0 || width <= 0 {
        return out;
    }
    // Wing span direction: side rotated toward forward by spread, lifted by pitch
    let span_base = v3_add(v3_scale(side, sign), v3_scale(forward, spread * sign));
    let span_dir = v3_normalize(v3_add(span_base, v3_scale(up, pitch)));
    // Chord direction: forward + forward_cant contribution
    let chord_dir = v3_normalize(v3_add(forward, v3_scale(side, forward_cant * sign)));
    let fl = length as f32;
    let fw = width as f32;
    for li in 0..=length {
        let t_span = li as f32 / fl;
        // Taper width along span based on wing_shape
        let local_half_w = match wing_shape {
            1 => {
                // Pointed: linear taper
                fw * 0.5 * (1.0 - t_span)
            }
            2 => {
                // Rounded rectangle: nearly constant then drops
                let edge = 1.0 - (t_span - 0.85).max(0.0) / 0.15;
                fw * 0.5 * edge.clamp(0.0, 1.0)
            }
            _ => {
                // Elliptical taper
                fw * 0.5 * (1.0 - t_span * t_span).max(0.0).sqrt()
            }
        };
        let w_int = local_half_w.ceil() as i32;
        for ci in -w_int..=w_int {
            if (ci as f32).abs() > local_half_w + 0.5 {
                continue;
            }
            let p = v3_add(
                hinge,
                v3_add(
                    v3_scale(span_dir, li as f32),
                    v3_scale(chord_dir, ci as f32),
                ),
            );
            out.push(v3_round(p));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Emit helper: add voxel if not already occupied
// ---------------------------------------------------------------------------

fn emit_voxel(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    seen: &mut HashSet<VoxelCoord>,
    out: &mut Vec<VoxelEditDelta>,
    x: i32,
    y: i32,
    z: i32,
    color: u32,
    material: MaterialId,
) -> bool {
    if out.len() >= VOXEL_CAP {
        return false;
    }
    ensure_grid_fits_coord(file, x, y, z);
    if !seen.insert((x, y, z)) {
        return true;
    }
    if voxel_map.contains_key(&(x, y, z)) {
        return true;
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
    true
}

// ---------------------------------------------------------------------------
// Core insecta generation
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn generate_insecta_deltas(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    face_empty: VoxelCoord,
    solid: VoxelCoord,
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
    material: MaterialId,
) -> Vec<VoxelEditDelta> {
    let tl = total_length.max(3).min(80) as f32;
    let bhw = body_half_width.max(1).min(20) as f32;
    let bhh = body_half_height.max(1).min(20) as f32;
    let abdomen_taper = abdomen_taper.clamp(0.0, 1.0);

    // Normalize ratios
    let rsum = (head_ratio + thorax_ratio + abdomen_ratio).max(0.01);
    let head_len = (tl * head_ratio / rsum).round().max(1.0);
    let thorax_len = (tl * thorax_ratio / rsum).round().max(1.0);
    let abdomen_len = (tl - head_len - thorax_len).max(1.0);

    // Face normal
    let nx = face_empty.0 - solid.0;
    let ny = face_empty.1 - solid.1;
    let nz = face_empty.2 - solid.2;
    if nx.abs() + ny.abs() + nz.abs() != 1 {
        return Vec::new();
    }

    // Build body frame using common PlacementFrame
    let frame = PlacementFrame::from_normal(
        (face_empty.0, face_empty.1, face_empty.2),
        nx, ny, nz,
    )
    .with_anchor_offset(anchor_offset_u as f32, anchor_offset_v as f32)
    .with_yaw(body_yaw);
    let forward = frame.forward;
    let side = frame.side;
    let up = frame.up;

    // Nose position = frame origin (bug faces outward from click point)
    let nose = frame.origin;

    let mut out = Vec::new();
    let mut seen: HashSet<VoxelCoord> = HashSet::new();

    // ------------------------------------------------------------------
    // 1. Body voxelization: iterate slices along forward axis
    // ------------------------------------------------------------------
    let total_slices = tl as i32;
    for i in 0..=total_slices {
        let dist = i as f32;
        let (rw, rh) = segment_radii(
            dist,
            head_len,
            thorax_len,
            abdomen_len,
            tl,
            bhw,
            bhh,
            abdomen_taper,
            head_shape,
        );
        if rw < 0.3 && rh < 0.3 {
            continue;
        }
        // Body arch: curve the body upward toward the tail
        let arch_offset = body_arch * (dist / tl).powi(2);
        let slice_center = v3_add(
            nose,
            v3_add(v3_scale(forward, dist), v3_scale(up, arch_offset)),
        );
        // Fill elliptical cross-section
        let iw = rw.ceil() as i32;
        let ih = rh.ceil() as i32;
        for du in -iw..=iw {
            for dv in -ih..=ih {
                let eu = du as f32 / rw.max(0.5);
                let ev = dv as f32 / rh.max(0.5);
                if eu * eu + ev * ev > 1.0 {
                    continue;
                }
                let p = v3_add(
                    slice_center,
                    v3_add(v3_scale(side, du as f32), v3_scale(up, dv as f32)),
                );
                let (x, y, z) = v3_round(p);
                if !emit_voxel(
                    file, voxel_map, &mut seen, &mut out, x, y, z, color, material,
                ) {
                    return out;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 2. Nose cap: extra rounded slices in front of head
    // ------------------------------------------------------------------
    let cap_slices = (head_len * 0.3).ceil() as i32;
    for i in 1..=cap_slices {
        let t = i as f32 / (cap_slices as f32 + 1.0);
        let r = bhw * 0.5 * (1.0 - t * t).max(0.0).sqrt();
        if r < 0.3 {
            continue;
        }
        let p = v3_add(nose, v3_scale(forward, -(i as f32)));
        let ir = r.ceil() as i32;
        for du in -ir..=ir {
            for dv in -ir..=ir {
                let eu = du as f32 / r;
                let ev = dv as f32 / r;
                if eu * eu + ev * ev > 1.0 {
                    continue;
                }
                let vp = v3_add(
                    p,
                    v3_add(v3_scale(side, du as f32), v3_scale(up, dv as f32)),
                );
                let (x, y, z) = v3_round(vp);
                if !emit_voxel(
                    file, voxel_map, &mut seen, &mut out, x, y, z, color, material,
                ) {
                    return out;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 3. Compound eyes (fly species): dorsolateral mass
    // ------------------------------------------------------------------
    if species == "fly" {
        let eye_center_dist = head_len * 0.4;
        for &sign in &[1.0_f32, -1.0] {
            let eye_center = v3_add(
                nose,
                v3_add(
                    v3_scale(forward, eye_center_dist),
                    v3_add(v3_scale(side, sign * bhw * 0.9), v3_scale(up, bhh * 0.5)),
                ),
            );
            let er = (bhw * 0.45).max(1.0);
            let ier = er.ceil() as i32;
            for dx in -ier..=ier {
                for dy in -ier..=ier {
                    for dz in -ier..=ier {
                        let d = ((dx * dx + dy * dy + dz * dz) as f32).sqrt();
                        if d > er {
                            continue;
                        }
                        let p = v3_add(eye_center, [dx as f32, dy as f32, dz as f32]);
                        let (x, y, z) = v3_round(p);
                        if !emit_voxel(
                            file, voxel_map, &mut seen, &mut out, x, y, z, color, material,
                        ) {
                            return out;
                        }
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 4. Legs: three pairs with hip-knee-foot articulation
    // ------------------------------------------------------------------
    let leg_defs = species_leg_defs(species);
    let thorax_start_dist = head_len;
    for (pair_idx, def) in leg_defs.iter().enumerate() {
        let hip_fwd = thorax_start_dist + def.hip_fwd_frac * thorax_len;
        let leg_scale = tl / 20.0; // scale leg lengths to body size
        for &sign in &[1.0_f32, -1.0] {
            let hip = v3_add(
                nose,
                v3_add(
                    v3_scale(forward, hip_fwd),
                    v3_add(
                        v3_scale(side, sign * (bhw + def.hip_side) * leg_scale.min(2.0)),
                        v3_scale(up, def.hip_up * leg_scale),
                    ),
                ),
            );
            let knee = v3_add(
                hip,
                v3_add(
                    v3_scale(side, sign * def.knee_side * leg_scale),
                    v3_scale(up, -def.knee_down * leg_scale),
                ),
            );
            // Grasshopper hind legs: knee goes UP then foot goes down
            let foot_down = if species == "grasshopper" && pair_idx == 2 {
                def.foot_down * leg_scale * 1.3
            } else {
                def.foot_down * leg_scale
            };
            let foot = v3_add(
                knee,
                v3_add(
                    v3_scale(side, sign * (def.foot_side - def.knee_side) * leg_scale),
                    v3_scale(up, -foot_down),
                ),
            );
            let hip_v = v3_round(hip);
            let knee_v = v3_round(knee);
            let foot_v = v3_round(foot);
            // hip → knee
            for (x, y, z) in bresenham_line(hip_v, knee_v) {
                if !emit_voxel(
                    file, voxel_map, &mut seen, &mut out, x, y, z, color, material,
                ) {
                    return out;
                }
            }
            // knee → foot
            for (x, y, z) in bresenham_line(knee_v, foot_v) {
                if !emit_voxel(
                    file, voxel_map, &mut seen, &mut out, x, y, z, color, material,
                ) {
                    return out;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 5. Antennae: directed walk from head top
    // ------------------------------------------------------------------
    if antenna_length > 0 {
        let root_dist = antenna_root.max(0) as f32;
        let root_pos = v3_add(
            nose,
            v3_add(
                v3_scale(forward, root_dist.min(head_len)),
                v3_scale(up, bhh * 0.9),
            ),
        );
        let al = antenna_length.min(40) as f32;
        for &sign in &[1.0_f32, -1.0] {
            let ant_dir = v3_normalize(v3_add(
                v3_add(
                    v3_scale(forward, -1.0), // project forward (away from body)
                    v3_scale(side, sign * antenna_spread),
                ),
                v3_scale(up, antenna_pitch),
            ));
            let tip = v3_add(root_pos, v3_scale(ant_dir, al));
            let rv = v3_round(root_pos);
            let tv = v3_round(tip);
            for (x, y, z) in bresenham_line(rv, tv) {
                if !emit_voxel(
                    file, voxel_map, &mut seen, &mut out, x, y, z, color, material,
                ) {
                    return out;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 6. Mandibles: directed walk from head front
    // ------------------------------------------------------------------
    if mandible_length > 0 {
        let mand_root = v3_add(nose, v3_scale(forward, -(mandible_forward.max(0) as f32)));
        let ml = mandible_length.min(20) as f32;
        for &sign in &[1.0_f32, -1.0] {
            let mand_dir = v3_normalize(v3_add(
                v3_scale(forward, -1.0),
                v3_scale(side, sign * mandible_spread),
            ));
            let tip = v3_add(mand_root, v3_scale(mand_dir, ml));
            let rv = v3_round(mand_root);
            let tv = v3_round(tip);
            for (x, y, z) in bresenham_line(rv, tv) {
                if !emit_voxel(
                    file, voxel_map, &mut seen, &mut out, x, y, z, color, material,
                ) {
                    return out;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 7. Wings
    // ------------------------------------------------------------------
    // Fore wings
    if show_wing_fore && wing_fore_length > 0 {
        let hinge_fwd = head_len + wing_fore_offset.max(0) as f32;
        for &sign in &[1.0_f32, -1.0] {
            let hinge = v3_add(
                nose,
                v3_add(
                    v3_scale(forward, hinge_fwd),
                    v3_add(v3_scale(side, sign * bhw * 0.7), v3_scale(up, bhh * 0.8)),
                ),
            );
            let cells = wing_sheet(
                hinge,
                forward,
                side,
                up,
                wing_fore_length.min(40),
                wing_fore_width.min(20),
                wing_fore_spread,
                wing_fore_pitch,
                wing_shape,
                sign,
                wing_fore_forward_cant,
            );
            for (x, y, z) in cells {
                if !emit_voxel(
                    file, voxel_map, &mut seen, &mut out, x, y, z, color, material,
                ) {
                    return out;
                }
            }
        }
    }

    // Hind wings
    if show_wing_hind && wing_hind_length > 0 {
        let hinge_fwd = head_len + thorax_len * 0.5 + wing_hind_offset.max(0) as f32;
        for &sign in &[1.0_f32, -1.0] {
            let hinge = v3_add(
                nose,
                v3_add(
                    v3_scale(forward, hinge_fwd),
                    v3_add(v3_scale(side, sign * bhw * 0.6), v3_scale(up, bhh * 0.7)),
                ),
            );
            let cells = wing_sheet(
                hinge,
                forward,
                side,
                up,
                wing_hind_length.min(40),
                wing_hind_width.min(20),
                wing_hind_spread,
                wing_hind_pitch,
                wing_shape,
                sign,
                0.0, // hind wings have no forward cant
            );
            for (x, y, z) in cells {
                if !emit_voxel(
                    file, voxel_map, &mut seen, &mut out, x, y, z, color, material,
                ) {
                    return out;
                }
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Public face-click entry point
// ---------------------------------------------------------------------------

/// Face-click insecta generator (web parity).
#[allow(clippy::too_many_arguments)]
pub fn generator_insecta_at_screen(
    file: &mut VoxelleFile,
    voxel_map: &mut AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
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
    Ok(generate_insecta_deltas(
        file,
        voxel_map,
        face_empty,
        solid,
        species,
        total_length,
        head_ratio,
        thorax_ratio,
        abdomen_ratio,
        body_half_width,
        body_half_height,
        abdomen_taper,
        head_shape,
        anchor_offset_u,
        anchor_offset_v,
        body_yaw,
        body_arch,
        antenna_length,
        antenna_spread,
        antenna_pitch,
        antenna_root,
        mandible_length,
        mandible_spread,
        mandible_forward,
        wing_shape,
        show_wing_fore,
        wing_fore_length,
        wing_fore_width,
        wing_fore_spread,
        wing_fore_pitch,
        wing_fore_offset,
        wing_fore_forward_cant,
        show_wing_hind,
        wing_hind_length,
        wing_hind_width,
        wing_hind_spread,
        wing_hind_pitch,
        wing_hind_offset,
        color,
        material,
    ))
}

/// Preview-only: compute the set of voxel coords an insect would occupy,
/// without mutating the real file. Used for hover preview.
#[allow(clippy::too_many_arguments)]
pub fn preview_insecta_at_screen(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
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
    generate_insecta_deltas(
        &mut stub_file,
        &mut stub_map,
        face_empty,
        solid,
        species,
        total_length,
        head_ratio,
        thorax_ratio,
        abdomen_ratio,
        body_half_width,
        body_half_height,
        abdomen_taper,
        head_shape,
        anchor_offset_u,
        anchor_offset_v,
        body_yaw,
        body_arch,
        antenna_length,
        antenna_spread,
        antenna_pitch,
        antenna_root,
        mandible_length,
        mandible_spread,
        mandible_forward,
        wing_shape,
        show_wing_fore,
        wing_fore_length,
        wing_fore_width,
        wing_fore_spread,
        wing_fore_pitch,
        wing_fore_offset,
        wing_fore_forward_cant,
        show_wing_hind,
        wing_hind_length,
        wing_hind_width,
        wing_hind_spread,
        wing_hind_pitch,
        wing_hind_offset,
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
