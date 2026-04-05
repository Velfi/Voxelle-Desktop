//! Screen→world ray computation, DDA grid traversal, pick result types, and ray-voxel intersection.

use crate::camera::OrbitCamera;
use crate::greedy_mesh::VoxelCoord;
use crate::voxelle::scene::{
    is_object_visible, object_world_matrix, scene_objects_identity_for_bounds_fast_path,
};
use crate::voxelle::VoxelleFile;
use ahash::{AHashMap, AHashSet};
use glam::{Mat4, Vec3, Vec4};

pub fn screen_to_world_ray(
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> (Vec3, Vec3) {
    let w = width.max(1.0);
    let h = height.max(1.0);
    // Pixel centers: map (sx,sy) through viewport so the ray matches fragment centers (GPU).
    let ndc_x = ((sx + 0.5) / w) * 2.0 - 1.0;
    let ndc_y = 1.0 - ((sy + 0.5) / h) * 2.0;
    let proj = camera.proj_matrix(width, height);
    let view = camera.view_matrix();
    let inv_vp = (proj * view).inverse();
    // Glam `perspective_rh` uses NDC z in [0,1] (WebGPU); unproject with inverse(view*proj).
    let mut near_h = inv_vp * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let mut far_h = inv_vp * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    near_h /= near_h.w;
    far_h /= far_h.w;
    let o = near_h.truncate();
    let d = (far_h.truncate() - o).normalize();
    (o, d)
}

/// World-space point → physical viewport pixels `(sx, sy)` with top-left origin (+Y down), matching [`screen_to_world_ray`].
pub fn world_to_viewport_pixels(
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    wx: f32,
    wy: f32,
    wz: f32,
) -> Option<(f32, f32)> {
    let w = width.max(1.0);
    let h = height.max(1.0);
    let view = camera.view_matrix();
    let proj = camera.proj_matrix(w, h);
    let vp = proj * view;
    let p = Vec4::new(wx, wy, wz, 1.0);
    let clip = vp * p;
    if clip.w.abs() < 1e-5 {
        return None;
    }
    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;
    if ndc_x.abs() > 1.55 || ndc_y.abs() > 1.55 {
        return None;
    }
    let sx = (ndc_x + 1.0) * 0.5 * w - 0.5;
    let sy = (1.0 - ndc_y) * 0.5 * h - 0.5;
    Some((sx, sy))
}

fn ray_aabb_intersect(origin: Vec3, dir: Vec3, bmin: Vec3, bmax: Vec3) -> Option<(f32, f32)> {
    let mut tmin = f32::NEG_INFINITY;
    let mut tmax = f32::INFINITY;
    for i in 0..3 {
        let o = origin[i];
        let d = dir[i];
        let bn = bmin[i];
        let bx = bmax[i];
        if d.abs() < 1e-8 {
            if o < bn || o > bx {
                return None;
            }
            continue;
        }
        let inv_d = 1.0 / d;
        let mut t0 = (bn - o) * inv_d;
        let mut t1 = (bx - o) * inv_d;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        tmin = tmin.max(t0);
        tmax = tmax.min(t1);
        if tmin > tmax {
            return None;
        }
    }
    Some((tmin, tmax))
}

/// Local-space entry point on the voxel cell AABB (same convention as [`world_ray_entry_on_voxel_cell`]).
/// Callers that build **local** geometry (e.g. hover mesh) should use this — not `inverse(M) * world_hit`,
/// which reintroduces floating-point drift vs the slab math.
pub fn local_ray_entry_on_voxel_cell(
    origin_world: Vec3,
    dir_world: Vec3,
    cx: i32,
    cy: i32,
    cz: i32,
    world_from_local: Mat4,
) -> Option<Vec3> {
    let inv = world_from_local.inverse();
    let o_l = inv.transform_point3(origin_world);
    let d_l = inv.transform_vector3(dir_world);
    if d_l.length_squared() < 1e-18 {
        return None;
    }
    let bmin = Vec3::new(cx as f32 - 0.5, cy as f32 - 0.5, cz as f32 - 0.5);
    let bmax = Vec3::new(cx as f32 + 0.5, cy as f32 + 0.5, cz as f32 + 0.5);
    let (t_enter, t_exit) = ray_aabb_intersect(o_l, d_l, bmin, bmax)?;
    if t_exit < 0.0 {
        return None;
    }
    let t_hit = if t_enter >= 0.0 {
        t_enter
    } else if t_exit >= 0.0 {
        0.0
    } else {
        return None;
    };
    Some(o_l + d_l * t_hit)
}

/// World-space point where the ray first enters the axis-aligned voxel cell `(cx,cy,cz)` in **local**
/// object space (cell is `[c-0.5,c+0.5]` per axis), transformed by `world_from_local`.
/// Matches the face the DDA/marcher crosses when entering that cell from outside; falls back usefully
/// when the ray origin is already inside the cell (`t_hit = 0` in local space).
pub fn world_ray_entry_on_voxel_cell(
    origin_world: Vec3,
    dir_world: Vec3,
    cx: i32,
    cy: i32,
    cz: i32,
    world_from_local: Mat4,
) -> Option<Vec3> {
    local_ray_entry_on_voxel_cell(origin_world, dir_world, cx, cy, cz, world_from_local)
        .map(|p_l| world_from_local.transform_point3(p_l))
}

fn exit_t_axis(ox: f32, dx: f32, c: i32, t_min: f32) -> f32 {
    const EPS: f32 = 1e-5;
    if dx.abs() < EPS {
        return f32::INFINITY;
    }
    let t = if dx > 0.0 {
        ((c as f32 + 0.5) - ox) / dx
    } else {
        ((c as f32 - 0.5) - ox) / dx
    };
    if t > t_min + EPS {
        t
    } else {
        f32::INFINITY
    }
}

/// First solid voxel along the ray within the grid bounding box, and the previous empty cell visited (adjacent along the ray).
pub(crate) fn ray_first_solid(
    origin: Vec3,
    dir: Vec3,
    occupied: &AHashMap<VoxelCoord, usize>,
    grid_size: i32,
) -> Option<((i32, i32, i32), Option<(i32, i32, i32)>)> {
    let (lo, hi) = super::grid_valid_range(grid_size);
    let bmin = Vec3::new(lo as f32 - 0.5, lo as f32 - 0.5, lo as f32 - 0.5);
    let bmax = Vec3::new(hi as f32 + 0.5, hi as f32 + 0.5, hi as f32 + 0.5);
    let (t_enter, t_exit) = ray_aabb_intersect(origin, dir, bmin, bmax)?;
    let mut t = t_enter.max(0.0) + 1e-4;
    let mut prev: Option<(i32, i32, i32)> = None;
    let ox = origin.x;
    let oy = origin.y;
    let oz = origin.z;
    let dx = dir.x;
    let dy = dir.y;
    let dz = dir.z;

    for _ in 0..200_000 {
        if t > t_exit {
            break;
        }
        let p = origin + dir * t;
        let c = super::world_to_voxel(p);
        if !super::in_grid(c.0, c.1, c.2, grid_size) {
            break;
        }
        if occupied.contains_key(&c) {
            return Some((c, prev));
        }
        prev = Some(c);
        let t_next = exit_t_axis(ox, dx, c.0, t)
            .min(exit_t_axis(oy, dy, c.1, t))
            .min(exit_t_axis(oz, dz, c.2, t));
        if !t_next.is_finite() {
            break;
        }
        t = t_next + 1e-4;
    }
    None
}

/// Like [`ray_first_solid`], but voxel indices are **object-local** while `origin` / `dir` are in **world**
/// space — matching GPU meshing ([`crate::greedy_mesh::build_greedy_mesh`] applies [`object_world_matrix`] per object).
///
/// The third tuple element is the **object id** for the winning hit (for transforming local preview geometry).
pub(crate) fn ray_first_solid_scene(
    origin: Vec3,
    dir: Vec3,
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    grid_size: i32,
) -> Option<((i32, i32, i32), Option<(i32, i32, i32)>, u32)> {
    let objs = &file.objects;
    if objs.is_empty() || scene_objects_identity_for_bounds_fast_path(objs) {
        return ray_first_solid(origin, dir, voxel_map, grid_size).map(|(c, prev)| {
            let oid = voxel_map
                .get(&c)
                .map(|&vi| file.voxels[vi].object_id)
                .unwrap_or(0);
            (c, prev, oid)
        });
    }

    let mut oids = AHashSet::with_capacity(voxel_map.len().min(256));
    for &vi in voxel_map.values() {
        oids.insert(file.voxels[vi].object_id);
    }

    let mut best_t = f32::INFINITY;
    let mut best: Option<((i32, i32, i32), Option<(i32, i32, i32)>, u32)> = None;

    for oid in oids {
        if !is_object_visible(objs, oid) {
            continue;
        }
        let mut sub_map = AHashMap::with_capacity(voxel_map.len().min(4096));
        for (&k, &vi) in voxel_map.iter() {
            if file.voxels[vi].object_id == oid {
                sub_map.insert(k, vi);
            }
        }
        if sub_map.is_empty() {
            continue;
        }
        let m = object_world_matrix(objs, oid);
        let inv = m.inverse();
        let o_l = inv.transform_point3(origin);
        let d_l = inv.transform_vector3(dir);
        if d_l.length_squared() < 1e-18 {
            continue;
        }
        let d_l = d_l.normalize();
        let Some((hit, prev)) = ray_first_solid(o_l, d_l, &sub_map, grid_size) else {
            continue;
        };
        let wc = m.transform_point3(Vec3::new(hit.0 as f32, hit.1 as f32, hit.2 as f32));
        let t = (wc - origin).dot(dir);
        if t >= 0.0 && t < best_t {
            best_t = t;
            best = Some((hit, prev, oid));
        }
    }
    best
}

/// `true` if the ray from the screen point hits any solid voxel before exiting the grid (same test as edit/remove).
pub fn probe_solid_hit(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> bool {
    if file.voxels.is_empty() {
        return false;
    }
    let grid_size = super::effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    ray_first_solid_scene(origin, dir, file, voxel_map, grid_size).is_some()
}

/// First solid voxel along the ray (for selection toggle).
pub fn pick_solid_coord_at_screen(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Option<VoxelCoord> {
    if file.voxels.is_empty() {
        return None;
    }
    let grid_size = super::effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    ray_first_solid_scene(origin, dir, file, voxel_map, grid_size).map(|(c, _, _)| c)
}

/// Cell where an add would place (empty cell in front of first solid along the ray), if valid.
/// Second tuple element is the **object id** for that cell (same object as the ray hit).
pub fn preview_add_cell(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Option<((i32, i32, i32), u32)> {
    if file.voxels.is_empty() {
        return None;
    }
    let grid_size = super::effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let (_hit, prev, oid) = ray_first_solid_scene(origin, dir, file, voxel_map, grid_size)?;
    let (px, py, pz) = prev?;
    if !voxel_map.contains_key(&(px, py, pz)) {
        Some(((px, py, pz), oid))
    } else {
        None
    }
}

/// Same as [`anchor_for_edit`]. `stroke_snap_to_surface` only affects add brush alignment
/// ([`adjust_add_centers_for_surface_snap_brush`]), not this ray anchor.
pub(crate) fn anchor_for_stroke_edit(
    tool: super::EditTool,
    _snap_to_surface: bool,
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Option<(i32, i32, i32)> {
    anchor_for_edit(tool, file, voxel_map, camera, width, height, sx, sy)
}

/// Intersect a screen ray with an infinite plane defined by a point and normal.
/// Returns the voxel coordinate at the intersection (floored to grid), or None if the ray is
/// parallel / behind the camera.
pub fn anchor_on_plane(
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
    plane_point: Vec3,
    plane_normal: Vec3,
) -> Option<VoxelCoord> {
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let denom = plane_normal.dot(dir);
    if denom.abs() < 1e-7 {
        return None; // ray parallel to plane
    }
    let t = (plane_point - origin).dot(plane_normal) / denom;
    if t < 0.0 {
        return None; // behind camera
    }
    let hit = origin + dir * t;
    Some((
        hit.x.round() as i32,
        hit.y.round() as i32,
        hit.z.round() as i32,
    ))
}

/// Compute the constraint plane normal from a `constrain_to_plane_ref` string and camera.
/// Returns `None` if the reference is not recognized or "auto" with no face normal available.
pub fn constrain_plane_normal(
    plane_ref: &str,
    camera: &OrbitCamera,
    face_normal: Option<(i32, i32, i32)>,
) -> Option<Vec3> {
    match plane_ref {
        "camera" => {
            let view = camera.view_matrix();
            // Camera forward is -Z in view space; extract from view matrix 3rd row.
            Some(Vec3::new(-view.col(2).x, -view.col(2).y, -view.col(2).z).normalize())
        }
        "auto" => {
            let (nx, ny, nz) = face_normal?;
            Some(Vec3::new(nx as f32, ny as f32, nz as f32).normalize())
        }
        "x" => Some(Vec3::X),
        "y" => Some(Vec3::Y),
        "z" => Some(Vec3::Z),
        _ => None,
    }
}

/// Solid voxel the ray would remove, if any.
/// Second tuple element is the hit voxel's **object id** (for world-space preview).
pub fn preview_remove_cell(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Option<((i32, i32, i32), u32)> {
    if file.voxels.is_empty() {
        return None;
    }
    let grid_size = super::effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    ray_first_solid_scene(origin, dir, file, voxel_map, grid_size).map(|(h, _, oid)| (h, oid))
}

#[inline]
pub(crate) fn anchor_for_edit(
    tool: super::EditTool,
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Option<(i32, i32, i32)> {
    match tool {
        super::EditTool::Add => {
            preview_add_cell(file, voxel_map, camera, width, height, sx, sy).map(|(c, _)| c)
        }
        super::EditTool::Remove | super::EditTool::Paint => {
            preview_remove_cell(file, voxel_map, camera, width, height, sx, sy).map(|(c, _)| c)
        }
    }
}

/// Outward face normal (empty cell before solid along the ray).
pub fn outward_face_normal_from_screen_ray(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Option<(i32, i32, i32)> {
    let grid_size = super::effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let (hit, prev, _oid) = ray_first_solid_scene(origin, dir, file, voxel_map, grid_size)?;
    let prev = prev?;
    Some((prev.0 - hit.0, prev.1 - hit.1, prev.2 - hit.2))
}

pub fn pick_voxel_at_screen(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Option<crate::voxelle::Voxel> {
    if file.voxels.is_empty() {
        return None;
    }
    let grid_size = super::effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let (hit, _, _oid) = ray_first_solid_scene(origin, dir, file, voxel_map, grid_size)?;
    let idx = *voxel_map.get(&hit)?;
    Some(file.voxels[idx])
}

/// Pick the add-position and outward face normal at a screen point (for extrude ray start).
pub fn pick_extrude_start(
    file: &VoxelleFile,
    voxel_map: &AHashMap<VoxelCoord, usize>,
    camera: &OrbitCamera,
    width: f32,
    height: f32,
    sx: f32,
    sy: f32,
) -> Option<(VoxelCoord, Option<(i32, i32, i32)>)> {
    if file.voxels.is_empty() {
        return None;
    }
    let grid_size = super::effective_ray_grid_size(file);
    let (origin, dir) = screen_to_world_ray(camera, width, height, sx, sy);
    let (hit, prev, _oid) = ray_first_solid_scene(origin, dir, file, voxel_map, grid_size)?;
    // Add-position: the empty cell just before the solid hit (or hit itself if no prev)
    let add_pos = prev.unwrap_or(hit);
    // Face normal: difference between add-position and the solid hit
    let face_n = prev.map(|p| (p.0 - hit.0, p.1 - hit.1, p.2 - hit.2));
    Some((add_pos, face_n))
}

/// Voxel centers along a 3D line (inclusive endpoints).
pub(crate) fn voxel_line_dda(a: (i32, i32, i32), b: (i32, i32, i32)) -> Vec<(i32, i32, i32)> {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let dz = b.2 - a.2;
    let steps = dx.abs().max(dy.abs()).max(dz.abs()).max(1);
    let mut pts = Vec::with_capacity((steps + 1) as usize);
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = (a.0 as f32 + dx as f32 * t).round() as i32;
        let y = (a.1 as f32 + dy as f32 * t).round() as i32;
        let z = (a.2 as f32 + dz as f32 * t).round() as i32;
        pts.push((x, y, z));
    }
    pts.sort_unstable();
    pts.dedup();
    pts
}
