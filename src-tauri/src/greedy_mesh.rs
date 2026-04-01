//! Greedy meshing with Minecraft-style per-corner vertex AO — face visibility rules from Voxelle `greedyMeshCore`.

use crate::gpu_brick::{pack_cell, pack_empty};
use crate::voxelle::{MaterialId, SceneObject, Voxel};
use ahash::{AHashMap, AHashSet};
use glam::{Mat4, Vec3};
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Integer voxel coordinate key for maps and meshing.
pub type VoxelCoord = (i32, i32, i32);

type IVec3 = VoxelCoord;

fn coord_key(x: i32, y: i32, z: i32) -> IVec3 {
    (x, y, z)
}

#[inline]
fn grid_pos(axis: usize, depth: i32, u: i32, v: i32) -> IVec3 {
    match axis {
        0 => (depth, u, v),
        1 => (u, depth, v),
        _ => (u, v, depth),
    }
}

fn is_transmissive(m: MaterialId) -> bool {
    matches!(m, MaterialId::Glass | MaterialId::Water)
}

/// `mat_kind` per vertex: 0 plastic/rubber, 0.5 metal, 1 glow, 2 glass, 2.5 water.
fn mat_kind_f32(m: MaterialId) -> f32 {
    match m {
        MaterialId::Metal => 0.5,
        MaterialId::Glow => 1.0,
        MaterialId::Glass => 2.0,
        MaterialId::Water => 2.5,
        _ => 0.0, // Plastic, Rubber
    }
}

/// Mirrors `isFaceOccludedByNeighbor` in greedyMeshCore.ts — `true` = face not emitted.
fn face_occluded(source: MaterialId, neighbor: MaterialId) -> bool {
    if is_transmissive(source) {
        source == neighbor
    } else {
        !is_transmissive(neighbor)
    }
}

fn neighbor_occludes_face(
    map: &AHashMap<IVec3, Voxel>,
    pos: IVec3,
    axis: usize,
    sign: i32,
    src: MaterialId,
    src_object_id: u32,
) -> bool {
    let (x, y, z) = pos;
    let (nx, ny, nz) = match axis {
        0 => (x + sign, y, z),
        1 => (x, y + sign, z),
        _ => (x, y, z + sign),
    };
    let Some(neigh) = map.get(&coord_key(nx, ny, nz)) else {
        return false;
    };
    if neigh.object_id != src_object_id {
        return false;
    }
    face_occluded(src, neigh.material)
}

fn material_tag(m: MaterialId) -> u8 {
    match m {
        MaterialId::Plastic => 0,
        MaterialId::Metal => 1,
        MaterialId::Rubber => 2,
        MaterialId::Glass => 3,
        MaterialId::Water => 4,
        MaterialId::Glow => 5,
    }
}

#[inline]
fn bucket_key_parts(v: &Voxel) -> (u32, u8) {
    (v.color, material_tag(v.material))
}

/// Coplanar grouping: axis (0..3), sign (±1), depth along that axis.
type GreedySliceKey = (usize, i32, i32);

fn face_normal(axis: usize, sign: i32) -> glam::Vec3 {
    let mut v = glam::Vec3::ZERO;
    v[axis] = sign as f32;
    v
}

fn quad_corner(axis: usize, sign: i32, depth: i32, u: i32, v: i32) -> glam::Vec3 {
    let fo = 0.5 * sign as f32;
    match axis {
        0 => glam::Vec3::new(depth as f32 + fo, u as f32 - 0.5, v as f32 - 0.5),
        1 => glam::Vec3::new(u as f32 - 0.5, depth as f32 + fo, v as f32 - 0.5),
        _ => glam::Vec3::new(u as f32 - 0.5, v as f32 - 0.5, depth as f32 + fo),
    }
}

/// Matches `greedyMeshCore.ts` `AO_PRESETS` strength **2** (`aoStrength` default).
const AO_STRONG_PRESET: [f32; 4] = [0.55, 0.72, 0.88, 1.0];

/// Matches `getAOState` in `greedyMeshCore.ts`.
#[inline]
fn ao_state(side1: u32, side2: u32, corner: u32) -> usize {
    if side1 != 0 && side2 != 0 {
        return 0;
    }
    (3 - (side1 + side2 + corner)) as usize
}

#[inline]
fn ao_state_to_multiplier(state: usize) -> f32 {
    AO_STRONG_PRESET[state.min(3)]
}

/// Matches `aoCellOccludesForOwner` in `greedyMeshCore.ts`: same object, non-transmissive.
#[inline]
fn ao_cell_occludes_for_owner(map: &AHashMap<VoxelCoord, Voxel>, pos: IVec3, source_object_id: u32) -> bool {
    map.get(&pos)
        .map(|v| v.object_id == source_object_id && !is_transmissive(v.material))
        .unwrap_or(false)
}

/// Neighbor (du, dv) triplets per corner index — `AO_NEIGHBORS` in `greedyMeshCore.ts`.
const AO_DU_DV: [[(i32, i32); 3]; 4] = [
    [(-1, 0), (0, -1), (-1, -1)],
    [(1, 0), (0, -1), (1, -1)],
    [(1, 0), (0, 1), (1, 1)],
    [(-1, 0), (0, 1), (-1, 1)],
];

/// Matches `getAONeighborCoords` / `getCornerAO` in `greedyMeshCore.ts` (strong preset).
pub(super) fn corner_ao_factor(
    map: &AHashMap<VoxelCoord, Voxel>,
    axis: usize,
    sign: i32,
    depth: i32,
    cu: i32,
    cv: i32,
    corner_index: usize,
) -> f32 {
    let Some(face_voxel) = map.get(&grid_pos(axis, depth, cu, cv)) else {
        return 1.0;
    };
    let source_object_id = face_voxel.object_id;
    let d_idx = axis;
    let (u_idx, v_idx) = match axis {
        0 => (1usize, 2usize),
        1 => (0usize, 2usize),
        _ => (0usize, 1usize),
    };
    let du_dv = AO_DU_DV[corner_index.min(3)];
    let mut p = [0i32; 3];
    let mut bits = [0u32; 3];
    for i in 0..3 {
        let (du, dv) = du_dv[i];
        p[d_idx] = depth + sign;
        p[u_idx] = cu + du;
        p[v_idx] = cv + dv;
        let pos = (p[0], p[1], p[2]);
        bits[i] = u32::from(ao_cell_occludes_for_owner(map, pos, source_object_id));
    }
    let st = ao_state(bits[0], bits[1], bits[2]);
    ao_state_to_multiplier(st)
}

fn greedy_merge(cells: &[(i32, i32)]) -> Vec<(i32, i32, i32, i32)> {
    let n = cells.len().max(1);
    let set: AHashSet<(i32, i32)> = cells.iter().copied().collect();
    let mut consumed = AHashSet::with_capacity(n);
    let mut quads = Vec::new();
    for &(u, v) in cells {
        if consumed.contains(&(u, v)) {
            continue;
        }
        let mut w = 1_i32;
        while set.contains(&(u + w, v)) && !consumed.contains(&(u + w, v)) {
            w += 1;
        }
        let mut h = 1_i32;
        'rows: loop {
            for i in 0..w {
                let cell = (u + i, v + h);
                if !set.contains(&cell) || consumed.contains(&cell) {
                    break 'rows;
                }
            }
            h += 1;
        }
        for dv in 0..h {
            for du in 0..w {
                consumed.insert((u + du, v + dv));
            }
        }
        quads.push((u, v, w, h));
    }
    quads
}

fn color_rgb(color: u32) -> glam::Vec3 {
    let r = ((color >> 16) & 0xff) as f32 / 255.0;
    let g = ((color >> 8) & 0xff) as f32 / 255.0;
    let b = (color & 0xff) as f32 / 255.0;
    glam::Vec3::new(r, g, b)
}

#[derive(Clone, Copy)]
pub struct MeshBounds {
    pub min: glam::Vec3,
    pub max: glam::Vec3,
}

impl MeshBounds {
    pub fn center(&self) -> glam::Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn radius(&self) -> f32 {
        (self.max - self.min).length() * 0.5
    }
}

/// Expand bounds to include `v` (after an add). Voxel centers are integers as `f32`.
pub fn mesh_bounds_expand_with_voxel(previous: &MeshBounds, v: &Voxel) -> MeshBounds {
    let p = glam::Vec3::new(v.x as f32, v.y as f32, v.z as f32);
    MeshBounds {
        min: previous.min.min(p),
        max: previous.max.max(p),
    }
}

/// True if removing `(x,y,z)` cannot change the scene AABB (strictly interior to current bounds).
pub fn mesh_bounds_remove_is_strict_interior(bounds: &MeshBounds, x: i32, y: i32, z: i32) -> bool {
    let min_x = bounds.min.x as i32;
    let max_x = bounds.max.x as i32;
    let min_y = bounds.min.y as i32;
    let max_y = bounds.max.y as i32;
    let min_z = bounds.min.z as i32;
    let max_z = bounds.max.z as i32;
    x > min_x && x < max_x && y > min_y && y < max_y && z > min_z && z < max_z
}

#[cfg(test)]
mod bounds_edit_tests {
    use super::*;
    use crate::voxelle::MaterialId;

    #[test]
    fn expand_with_voxel_grows_aabb() {
        let b = MeshBounds {
            min: glam::Vec3::new(0.0, 0.0, 0.0),
            max: glam::Vec3::new(1.0, 1.0, 1.0),
        };
        let v = Voxel {
            x: 5,
            y: 0,
            z: 0,
            color: 1,
            material: MaterialId::Plastic,
            object_id: 0,
        };
        let e = mesh_bounds_expand_with_voxel(&b, &v);
        assert_eq!(e.max.x, 5.0);
        assert_eq!(e.min, b.min);
    }

    #[test]
    fn strict_interior_vs_boundary() {
        let b = MeshBounds {
            min: glam::Vec3::new(0.0, 0.0, 0.0),
            max: glam::Vec3::new(10.0, 10.0, 10.0),
        };
        assert!(mesh_bounds_remove_is_strict_interior(&b, 5, 5, 5));
        assert!(!mesh_bounds_remove_is_strict_interior(&b, 0, 5, 5));
        assert!(!mesh_bounds_remove_is_strict_interior(&b, 10, 5, 5));
    }

    #[test]
    fn mesh_bounds_world_matches_local_for_identity_objects() {
        let voxels = vec![
            Voxel {
                x: 1,
                y: 2,
                z: 3,
                color: 4,
                material: MaterialId::Plastic,
                object_id: 0,
            },
            Voxel {
                x: 10,
                y: 0,
                z: 0,
                color: 1,
                material: MaterialId::Plastic,
                object_id: 0,
            },
        ];
        let objs = crate::voxelle::default_scene_objects();
        let w = mesh_bounds_from_voxels_world(&voxels, &objs).expect("world");
        let l = mesh_bounds_from_voxels(&voxels).expect("local");
        assert!((w.min - l.min).length() < 1e-4);
        assert!((w.max - l.max).length() < 1e-4);
    }
}

#[derive(Clone, Debug, Default)]
pub struct MeshBuffers {
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    pub colors: Vec<f32>,
    pub mat_kind: Vec<f32>,
    /// Per-vertex ambient factor from corner AO (`AO_STRONG_PRESET` / `greedyMeshCore` strength 2); hemisphere only in shader.
    pub ao: Vec<f32>,
    pub indices: Vec<u32>,
}

/// Padded brick (+1 voxel halo) for GPU mesh AO: same row-major layout as [`crate::gpu_brick::GpuVoxelBrick`],
/// with origin shifted by −1 and dims +2, filled from `map` so in-plane neighbor checks see voxels outside the tight brick.
pub fn pack_brick_halo_cells(
    map: &AHashMap<VoxelCoord, Voxel>,
    origin: (i32, i32, i32),
    dims: (u32, u32, u32),
) -> Option<((i32, i32, i32), (u32, u32, u32), Vec<u32>)> {
    let (ox, oy, oz) = origin;
    let ho = (ox - 1, oy - 1, oz - 1);
    let hd = (
        dims.0.checked_add(2)?,
        dims.1.checked_add(2)?,
        dims.2.checked_add(2)?,
    );
    let n = (hd.0 as usize)
        .checked_mul(hd.1 as usize)?
        .checked_mul(hd.2 as usize)?;
    let mut cells = vec![pack_empty(); n];
    for ((x, y, z), v) in map {
        let rx = x - ho.0;
        let ry = y - ho.1;
        let rz = z - ho.2;
        if rx < 0 || ry < 0 || rz < 0 {
            continue;
        }
        let ux = rx as u32;
        let uy = ry as u32;
        let uz = rz as u32;
        if ux >= hd.0 || uy >= hd.1 || uz >= hd.2 {
            continue;
        }
        let idx = (ux as usize)
            + (uy as usize) * (hd.0 as usize)
            + (uz as usize) * (hd.0 as usize) * (hd.1 as usize);
        cells[idx] = pack_cell(v.color, v.material);
    }
    Some((ho, hd, cells))
}

pub fn voxel_map(voxels: &[Voxel]) -> AHashMap<VoxelCoord, Voxel> {
    let mut map = AHashMap::with_capacity(voxels.len());
    for v in voxels {
        map.insert(coord_key(v.x, v.y, v.z), *v);
    }
    map
}

/// Spatial index for raycasts / swap-remove: coord → index in `VoxelleFile::voxels`.
pub fn voxel_map_indices(voxels: &[Voxel]) -> AHashMap<VoxelCoord, usize> {
    let mut map = AHashMap::with_capacity(voxels.len());
    for (i, v) in voxels.iter().enumerate() {
        map.insert(coord_key(v.x, v.y, v.z), i);
    }
    map
}

/// Packed slice for GPU greedy mesh (see `render/gpu/mesh_greedy.wgsl` `SliceHeader`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSliceHeader {
    pub axis: u32,
    pub sign: i32,
    pub depth: i32,
    pub color: u32,
    pub mat_kind: f32,
    pub u0: i32,
    pub v0: i32,
    pub width: u32,
    pub height: u32,
    pub bit_start: u32,
    pub bit_word_count: u32,
}

/// Pack coplanar face cells into 2D bitmaps for GPU greedy meshing (each sub-slice ≤64×64).
pub fn pack_gpu_greedy_slices(
    map: &AHashMap<VoxelCoord, Voxel>,
    emit: &[Voxel],
) -> Result<(Vec<GpuSliceHeader>, Vec<u32>), ()> {
    let mut buckets: AHashMap<(u32, u8), Vec<IVec3>> = AHashMap::new();
    for v in emit {
        buckets
            .entry(bucket_key_parts(v))
            .or_default()
            .push(coord_key(v.x, v.y, v.z));
    }

    let mut headers: Vec<GpuSliceHeader> = Vec::new();
    let mut all_bits: Vec<u32> = Vec::new();

    for cell_positions in buckets.values() {
        let Some(&first_pos) = cell_positions.first() else {
            continue;
        };
        let vx = map[&first_pos];
        let mat_k = mat_kind_f32(vx.material);

        let mut faces: Vec<(IVec3, usize, i32)> = Vec::with_capacity(cell_positions.len() * 4);
        for &pos in cell_positions {
            let source = map[&pos];
            for i in 0..6usize {
                let axis = i / 2;
                let sign = if i % 2 == 0 { 1 } else { -1 };
                if !neighbor_occludes_face(
                    map,
                    pos,
                    axis,
                    sign,
                    source.material,
                    source.object_id,
                ) {
                    faces.push((pos, axis, sign));
                }
            }
        }

        let mut slices: AHashMap<GreedySliceKey, Vec<(i32, i32)>> = AHashMap::new();
        for (pos, axis, sign) in faces {
            let (x, y, z) = pos;
            let depth = match axis {
                0 => x,
                1 => y,
                _ => z,
            };
            let u = if axis == 0 { y } else { x };
            let v = if axis == 2 { y } else { z };
            slices.entry((axis, sign, depth)).or_default().push((u, v));
        }

        for ((axis, sign, depth), cells) in slices {
            let axis_u32 = axis as u32;

            let mut min_u = i32::MAX;
            let mut max_u = i32::MIN;
            let mut min_v = i32::MAX;
            let mut max_v = i32::MIN;
            for &(u, v) in &cells {
                min_u = min_u.min(u);
                max_u = max_u.max(u);
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }
            let width = (max_u - min_u + 1) as u32;
            let height = (max_v - min_v + 1) as u32;

            // Tile slices larger than 64×64 into GPU-sized sub-bitmaps (same plane).
            let tile_u = (width + 63) / 64;
            let tile_v = (height + 63) / 64;
            for tu in 0..tile_u {
                for tv in 0..tile_v {
                    let u0 = min_u + (tu * 64) as i32;
                    let v0 = min_v + (tv * 64) as i32;
                    let u1 = (u0 + 63).min(max_u);
                    let v1 = (v0 + 63).min(max_v);
                    let w = (u1 - u0 + 1) as u32;
                    let h = (v1 - v0 + 1) as u32;
                    debug_assert!(w <= 64 && h <= 64);

                    let ncells = w as usize * h as usize;
                    let bit_word_count = (ncells + 31) / 32;
                    let bit_start = all_bits.len() as u32;
                    all_bits.resize(all_bits.len() + bit_word_count, 0u32);

                    for &(u, v) in &cells {
                        if u < u0 || u > u1 || v < v0 || v > v1 {
                            continue;
                        }
                        let lu = (u - u0) as u32;
                        let lv = (v - v0) as u32;
                        let idx = lu + lv * w;
                        let wi = (idx / 32) as usize;
                        let bi = idx % 32;
                        all_bits[bit_start as usize + wi] |= 1u32 << bi;
                    }

                    headers.push(GpuSliceHeader {
                        axis: axis_u32,
                        sign,
                        depth,
                        color: vx.color,
                        mat_kind: mat_k,
                        u0,
                        v0,
                        width: w,
                        height: h,
                        bit_start,
                        bit_word_count: bit_word_count as u32,
                    });
                }
            }
        }
    }

    Ok((headers, all_bits))
}

#[allow(dead_code)]
pub fn append_mesh_buffers(dst: &mut MeshBuffers, mut src: MeshBuffers) {
    let base = (dst.positions.len() / 3) as u32;
    for i in &mut src.indices {
        *i += base;
    }
    dst.positions.append(&mut src.positions);
    dst.normals.append(&mut src.normals);
    dst.colors.append(&mut src.colors);
    dst.mat_kind.append(&mut src.mat_kind);
    dst.ao.append(&mut src.ao);
    dst.indices.append(&mut src.indices);
}

/// Axis-aligned bounds for a solid cube with voxels in `[−side/2, side/2)` (same convention as `.voxelle` / new project).
pub fn mesh_bounds_for_cube_side(side: i32) -> MeshBounds {
    let start = -(side / 2);
    let end = start + side;
    let lo = start as f32;
    let hi = (end - 1) as f32;
    MeshBounds {
        min: glam::Vec3::new(lo, lo, lo),
        max: glam::Vec3::new(hi, hi, hi),
    }
}

/// Default chunk edge length for v3 / spatially chunked meshing (retained for tools / future chunking).
pub const SPATIAL_CHUNK_SIZE: i32 = 48;

/// Use [`build_greedy_mesh_chunked`] on CPU fallback when voxel count exceeds this (cache-friendly on huge scenes).
pub const CHUNKED_CPU_MESH_MIN_VOXELS: usize = 200_000;

#[allow(dead_code)]
pub const VOXEL_CHUNK_SIZE: i32 = SPATIAL_CHUNK_SIZE;

/// Tight axis-aligned bounds of voxel centers (same convention as [`build_greedy_mesh`]).
pub fn mesh_bounds_from_voxels(voxels: &[Voxel]) -> Option<MeshBounds> {
    if voxels.is_empty() {
        return None;
    }
    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
    for v in voxels {
        let pf = glam::Vec3::new(v.x as f32, v.y as f32, v.z as f32);
        min = min.min(pf);
        max = max.max(pf);
    }
    Some(MeshBounds { min, max })
}

/// Spatial chunk index for incremental meshing (aligned with [`voxels_by_spatial_chunks`] bucketing).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ChunkKey {
    pub ix: i32,
    pub iy: i32,
    pub iz: i32,
}

/// Minimum integer coordinates of all voxels (chunk grid origin for [`chunk_key_from_world`]).
pub fn voxel_aabb_min_int(voxels: &[Voxel]) -> Option<(i32, i32, i32)> {
    if voxels.is_empty() {
        return None;
    }
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut min_z = i32::MAX;
    for v in voxels {
        min_x = min_x.min(v.x);
        min_y = min_y.min(v.y);
        min_z = min_z.min(v.z);
    }
    Some((min_x, min_y, min_z))
}

/// Chunk key for a world cell, using the same origin and `cs` as [`voxel_buckets_by_chunk`].
pub fn chunk_key_from_world(x: i32, y: i32, z: i32, origin: (i32, i32, i32), cs: i32) -> ChunkKey {
    let cs = cs.max(1);
    let (ox, oy, oz) = origin;
    ChunkKey {
        ix: (x - ox).div_euclid(cs),
        iy: (y - oy).div_euclid(cs),
        iz: (z - oz).div_euclid(cs),
    }
}

/// Dirty chunk keys for a voxel edit: only the center chunk plus neighbors where the voxel
/// sits on a chunk boundary (local coord 0 or cs-1 on that axis).
pub fn dirty_chunk_keys_for_voxel(x: i32, y: i32, z: i32, origin: (i32, i32, i32), cs: i32) -> Vec<ChunkKey> {
    let cs = cs.max(1);
    let (ox, oy, oz) = origin;
    let center = chunk_key_from_world(x, y, z, origin, cs);
    let lx = (x - ox).rem_euclid(cs);
    let ly = (y - oy).rem_euclid(cs);
    let lz = (z - oz).rem_euclid(cs);
    // Which axis directions need neighbor chunks?
    let dx: &[i32] = if lx == 0 && lx == cs - 1 { &[-1, 0, 1] }
        else if lx == 0 { &[-1, 0] }
        else if lx == cs - 1 { &[0, 1] }
        else { &[0] };
    let dy: &[i32] = if ly == 0 && ly == cs - 1 { &[-1, 0, 1] }
        else if ly == 0 { &[-1, 0] }
        else if ly == cs - 1 { &[0, 1] }
        else { &[0] };
    let dz: &[i32] = if lz == 0 && lz == cs - 1 { &[-1, 0, 1] }
        else if lz == 0 { &[-1, 0] }
        else if lz == cs - 1 { &[0, 1] }
        else { &[0] };
    let mut v = Vec::with_capacity(dx.len() * dy.len() * dz.len());
    for &dix in dx {
        for &diy in dy {
            for &diz in dz {
                v.push(ChunkKey {
                    ix: center.ix + dix,
                    iy: center.iy + diy,
                    iz: center.iz + diz,
                });
            }
        }
    }
    v
}

/// The 3×3×3 neighbor chunk keys in index space (for remeshing after one cell changes).
pub fn dirty_chunk_keys_3x3(center: ChunkKey) -> Vec<ChunkKey> {
    let mut v = Vec::with_capacity(27);
    for dix in -1i32..=1 {
        for diy in -1i32..=1 {
            for diz in -1i32..=1 {
                v.push(ChunkKey {
                    ix: center.ix + dix,
                    iy: center.iy + diy,
                    iz: center.iz + diz,
                });
            }
        }
    }
    v
}

/// Bucket voxels by spatial chunk (same layout as [`voxels_by_spatial_chunks`]).
pub fn voxel_buckets_by_chunk(
    voxels: &[Voxel],
    cs: i32,
) -> Option<((i32, i32, i32), AHashMap<ChunkKey, Vec<Voxel>>)> {
    let origin = voxel_aabb_min_int(voxels)?;
    let cs = cs.max(1);
    let mut buckets: AHashMap<ChunkKey, Vec<Voxel>> = AHashMap::new();
    for v in voxels {
        let k = chunk_key_from_world(v.x, v.y, v.z, origin, cs);
        buckets.entry(k).or_default().push(*v);
    }
    Some((origin, buckets))
}

/// Greedy mesh for one chunk’s **core** voxels, with neighbor occlusion from `map` (full scene).
pub fn mesh_buffers_for_chunk_key(
    buckets: &AHashMap<ChunkKey, AHashMap<VoxelCoord, Voxel>>,
    map: &AHashMap<VoxelCoord, Voxel>,
    key: ChunkKey,
) -> MeshBuffers {
    let bucket = match buckets.get(&key) {
        Some(b) if !b.is_empty() => b,
        _ => return MeshBuffers::default(),
    };
    let mut core: Vec<Voxel> = bucket.values().copied().collect();
    core.sort_unstable_by_key(|v| (v.x, v.y, v.z));
    build_greedy_mesh_mapped(&core, map)
}

/// Single pass: [`SpatialMeshCache`] plus per-chunk greedy meshes (one `voxel_map` + bucketing).
/// Chunk meshes build in parallel across chunks.
///
/// `progress` is called from worker threads: `fraction` in \([0, 1]\), and `completed` / `total_chunks`
/// count spatial buckets processed (including empty buckets). Throttled to avoid excessive UI updates.
pub fn build_chunk_meshes_and_spatial_cache<F>(voxels: &[Voxel], cs: i32, progress: F) -> Option<(
    (i32, i32, i32),
    BTreeMap<ChunkKey, MeshBuffers>,
    SpatialMeshCache,
)>
where
    F: Fn(f32, u32, u32) + Sync,
{
    use std::sync::atomic::{AtomicU32, Ordering};

    let cache = SpatialMeshCache::from_voxels(voxels, cs)?;
    let origin = cache.origin;
    let keys: Vec<ChunkKey> = cache.buckets.keys().copied().collect();
    let total = keys.len().max(1) as u32;
    let last_permille = AtomicU32::new(0);
    let done = AtomicU32::new(0);
    let parts: Vec<(ChunkKey, MeshBuffers)> = keys
        .par_iter()
        .filter_map(|&key| {
            let mesh = mesh_buffers_for_chunk_key(&cache.buckets, &cache.occupancy, key);
            let d = done.fetch_add(1, Ordering::Relaxed) + 1;
            let frac = d as f32 / total as f32;
            let permille = (frac * 1000.0).min(1000.0) as u32;
            let prev = last_permille.load(Ordering::Relaxed);
            if permille.saturating_sub(prev) >= 40 || d == total {
                last_permille.store(permille, Ordering::Relaxed);
                progress(frac, d, total);
            }
            (!mesh.indices.is_empty()).then_some((key, mesh))
        })
        .collect();
    progress(1.0, total, total);
    let meshes: BTreeMap<ChunkKey, MeshBuffers> = parts.into_iter().collect();
    Some((origin, meshes, cache))
}

/// Build per-chunk meshes (for GPU upload / incremental updates). Skips empty outputs.
pub fn build_all_chunk_meshes_btree(
    voxels: &[Voxel],
    cs: i32,
) -> Option<((i32, i32, i32), BTreeMap<ChunkKey, MeshBuffers>)> {
    let (origin, meshes, _) = build_chunk_meshes_and_spatial_cache(voxels, cs, |_, _, _| {})?;
    Some((origin, meshes))
}

/// Full occupancy map + spatial buckets for incremental edits (O(1) add/remove vs full rescans).
#[derive(Clone, Debug)]
pub struct SpatialMeshCache {
    pub origin: (i32, i32, i32),
    pub occupancy: AHashMap<VoxelCoord, Voxel>,
    pub buckets: AHashMap<ChunkKey, AHashMap<VoxelCoord, Voxel>>,
}

impl SpatialMeshCache {
    pub fn from_voxels(voxels: &[Voxel], cs: i32) -> Option<Self> {
        let origin = voxel_aabb_min_int(voxels)?;
        let cs = cs.max(1);
        const YIELD_EVERY: usize = 16_384;
        let mut occupancy = AHashMap::with_capacity(voxels.len());
        for (i, v) in voxels.iter().enumerate() {
            if i != 0 && i % YIELD_EVERY == 0 {
                std::thread::yield_now();
            }
            occupancy.insert(coord_key(v.x, v.y, v.z), *v);
        }
        let mut buckets: AHashMap<ChunkKey, AHashMap<VoxelCoord, Voxel>> = AHashMap::new();
        for (i, v) in voxels.iter().enumerate() {
            if i != 0 && i % YIELD_EVERY == 0 {
                std::thread::yield_now();
            }
            let k = chunk_key_from_world(v.x, v.y, v.z, origin, cs);
            buckets.entry(k).or_default().insert(coord_key(v.x, v.y, v.z), *v);
        }
        Some(Self {
            origin,
            occupancy,
            buckets,
        })
    }

    pub fn apply_add(&mut self, v: Voxel, cs: i32) {
        let cs = cs.max(1);
        let coord = (v.x, v.y, v.z);
        self.occupancy.insert(coord, v);
        let k = chunk_key_from_world(v.x, v.y, v.z, self.origin, cs);
        self.buckets.entry(k).or_default().insert(coord, v);
    }

    pub fn apply_remove(&mut self, x: i32, y: i32, z: i32, cs: i32) {
        let cs = cs.max(1);
        let coord = (x, y, z);
        self.occupancy.remove(&coord);
        let k = chunk_key_from_world(x, y, z, self.origin, cs);
        if let Some(map) = self.buckets.get_mut(&k) {
            map.remove(&coord);
            if map.is_empty() {
                self.buckets.remove(&k);
            }
        }
    }

    /// In-place color/material change (same grid cell).
    pub fn apply_paint(&mut self, after: Voxel, cs: i32) {
        let cs = cs.max(1);
        let coord = (after.x, after.y, after.z);
        self.occupancy.insert(coord, after);
        let k = chunk_key_from_world(after.x, after.y, after.z, self.origin, cs);
        if let Some(map) = self.buckets.get_mut(&k) {
            map.insert(coord, after);
        }
    }
}

/// Partitions arbitrary voxels into spatial chunks of edge length `cs` (world axes, aligned to the model AABB).
/// Each element is `(halo_voxels, core_voxels)` where halo includes the 26 neighbor chunks for correct greedy faces.
#[allow(dead_code)]
pub fn voxels_by_spatial_chunks(voxels: &[Voxel], cs: i32) -> Vec<(Vec<Voxel>, Vec<Voxel>)> {
    let cs = cs.max(1);
    if voxels.is_empty() {
        return Vec::new();
    }
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut min_z = i32::MAX;
    for v in voxels {
        min_x = min_x.min(v.x);
        min_y = min_y.min(v.y);
        min_z = min_z.min(v.z);
    }

    let mut buckets: AHashMap<(i32, i32, i32), Vec<Voxel>> = AHashMap::new();
    for v in voxels {
        let ix = (v.x - min_x).div_euclid(cs);
        let iy = (v.y - min_y).div_euclid(cs);
        let iz = (v.z - min_z).div_euclid(cs);
        buckets.entry((ix, iy, iz)).or_default().push(*v);
    }

    let mut keys: Vec<(i32, i32, i32)> = buckets.keys().copied().collect();
    keys.sort_unstable();

    let mut out = Vec::with_capacity(keys.len());
    for (ix, iy, iz) in keys {
        let core = buckets[&(ix, iy, iz)].clone();
        let mut halo = Vec::new();
        for dix in -1i32..=1 {
            for diy in -1i32..=1 {
                for diz in -1i32..=1 {
                    if let Some(vs) = buckets.get(&(ix + dix, iy + diy, iz + diz)) {
                        halo.extend(vs.iter().copied());
                    }
                }
            }
        }
        out.push((halo, core));
    }
    out
}

#[cfg(test)]
mod chunk_tests {
    use super::*;
    use crate::voxelle::{MaterialId, SceneObject};

    /// Baseline: one spatial cache, sequential per-chunk mesh (must match fused+parallel output).
    fn sequential_chunk_meshes_and_spatial_cache(
        voxels: &[Voxel],
        cs: i32,
    ) -> Option<(
        (i32, i32, i32),
        BTreeMap<ChunkKey, MeshBuffers>,
        SpatialMeshCache,
    )> {
        let cache = SpatialMeshCache::from_voxels(voxels, cs)?;
        let origin = cache.origin;
        let mut out = BTreeMap::new();
        for &key in cache.buckets.keys() {
            let mesh = mesh_buffers_for_chunk_key(&cache.buckets, &cache.occupancy, key);
            if !mesh.indices.is_empty() {
                out.insert(key, mesh);
            }
        }
        Some((origin, out, cache))
    }

    #[test]
    fn fused_chunk_meshes_match_sequential() {
        let mut voxels = Vec::new();
        for z in 0..10 {
            for y in 0..10 {
                for x in 0..10 {
                    voxels.push(Voxel {
                        x,
                        y,
                        z,
                        color: 0x112233,
                        material: MaterialId::Plastic,
                        object_id: 0,
                    });
                }
            }
        }
        let cs = SPATIAL_CHUNK_SIZE;
        let fused = build_chunk_meshes_and_spatial_cache(&voxels, cs, |_, _, _| {}).expect("fused");
        let seq = sequential_chunk_meshes_and_spatial_cache(&voxels, cs).expect("sequential");
        assert_eq!(fused.0, seq.0, "chunk origin");
        assert_eq!(fused.1.len(), seq.1.len(), "chunk count");
        for (k, m) in &fused.1 {
            let m2 = seq.1.get(k).expect("missing chunk");
            assert_eq!(sorted_triangle_set(m), sorted_triangle_set(m2), "triangles {:?}", k);
        }
    }

    #[test]
    fn spatial_chunks_split_distant_voxels() {
        let voxels = vec![
            Voxel {
                x: 0,
                y: 0,
                z: 0,
                color: 1,
                material: MaterialId::Plastic,
                object_id: 0,
            },
            Voxel {
                x: 100,
                y: 0,
                z: 0,
                color: 2,
                material: MaterialId::Plastic,
                object_id: 0,
            },
        ];
        let ch = voxels_by_spatial_chunks(&voxels, 48);
        assert_eq!(ch.len(), 2);
        assert_eq!(ch[0].1.len(), 1);
        assert_eq!(ch[1].1.len(), 1);
    }

    #[test]
    fn cross_object_adjacent_keeps_shared_faces() {
        let same = vec![
            Voxel {
                x: 0,
                y: 0,
                z: 0,
                color: 1,
                material: MaterialId::Plastic,
                object_id: 0,
            },
            Voxel {
                x: 1,
                y: 0,
                z: 0,
                color: 2,
                material: MaterialId::Plastic,
                object_id: 0,
            },
        ];
        let split = vec![
            Voxel {
                x: 0,
                y: 0,
                z: 0,
                color: 1,
                material: MaterialId::Plastic,
                object_id: 0,
            },
            Voxel {
                x: 1,
                y: 0,
                z: 0,
                color: 2,
                material: MaterialId::Plastic,
                object_id: 1,
            },
        ];
        let objs = vec![
            SceneObject::default(),
            SceneObject {
                id: 1,
                name: "B".into(),
                sort_order: 1,
                ..Default::default()
            },
        ];
        let (m_same, _) = build_greedy_mesh(&same, &[]);
        let (m_split, _) = build_greedy_mesh(&split, &objs);
        assert!(
            m_split.indices.len() > m_same.indices.len(),
            "expected extra faces between objects: same={}, split={}",
            m_same.indices.len(),
            m_split.indices.len(),
        );
    }

    /// Per-triangle multiset of quantized vertex positions (order-independent).
    fn sorted_triangle_set(mesh: &MeshBuffers) -> Vec<[[u32; 3]; 3]> {
        let pos = &mesh.positions;
        let mut tris: Vec<[[u32; 3]; 3]> = Vec::new();
        for chunk in mesh.indices.chunks(3) {
            let i0 = chunk[0] as usize * 3;
            let i1 = chunk[1] as usize * 3;
            let i2 = chunk[2] as usize * 3;
            let mut verts = [
                [
                    pos[i0].to_bits(),
                    pos[i0 + 1].to_bits(),
                    pos[i0 + 2].to_bits(),
                ],
                [
                    pos[i1].to_bits(),
                    pos[i1 + 1].to_bits(),
                    pos[i1 + 2].to_bits(),
                ],
                [
                    pos[i2].to_bits(),
                    pos[i2 + 1].to_bits(),
                    pos[i2 + 2].to_bits(),
                ],
            ];
            verts.sort();
            tris.push(verts);
        }
        tris.sort();
        tris
    }

    #[test]
    fn full_mesh_matches_stitched_chunk_meshes_fixture() {
        let voxels = vec![
            Voxel {
                x: 0,
                y: 0,
                z: 0,
                color: 0xff0000,
                material: MaterialId::Plastic,
                object_id: 0,
            },
            Voxel {
                x: 1,
                y: 0,
                z: 0,
                color: 0x00ff00,
                material: MaterialId::Plastic,
                object_id: 0,
            },
            Voxel {
                x: 0,
                y: 1,
                z: 0,
                color: 0x0000ff,
                material: MaterialId::Plastic,
                object_id: 0,
            },
        ];
        let (full, _) = build_greedy_mesh(&voxels, &[]);
        let Some((_origin, btree)) = build_all_chunk_meshes_btree(&voxels, SPATIAL_CHUNK_SIZE)
        else {
            panic!("expected buckets");
        };
        let mut stitched = MeshBuffers::default();
        for (_, m) in btree {
            append_mesh_buffers(&mut stitched, m);
        }
        assert_eq!(
            full.indices.len(),
            stitched.indices.len(),
            "index count parity"
        );
        assert_eq!(
            sorted_triangle_set(&full),
            sorted_triangle_set(&stitched),
            "triangle multiset parity"
        );
    }

    #[test]
    fn spatial_cache_incremental_matches_from_voxels() {
        let voxels = vec![
            Voxel {
                x: 0,
                y: 0,
                z: 0,
                color: 1,
                material: MaterialId::Plastic,
                object_id: 0,
            },
            Voxel {
                x: 5,
                y: 0,
                z: 0,
                color: 2,
                material: MaterialId::Plastic,
                object_id: 0,
            },
        ];
        let cs = SPATIAL_CHUNK_SIZE;
        let mut cache = SpatialMeshCache::from_voxels(&voxels, cs).unwrap();
        cache.apply_remove(5, 0, 0, cs);
        let after_remove: Vec<Voxel> = vec![voxels[0]];
        let expected = SpatialMeshCache::from_voxels(&after_remove, cs).unwrap();
        assert_eq!(cache.occupancy, expected.occupancy);
        assert_eq!(cache.buckets, expected.buckets);
        let add = Voxel {
            x: 1,
            y: 1,
            z: 1,
            color: 3,
            material: MaterialId::Plastic,
            object_id: 0,
        };
        cache.apply_add(add, cs);
        let mut after_add = after_remove;
        after_add.push(add);
        let expected2 = SpatialMeshCache::from_voxels(&after_add, cs).unwrap();
        assert_eq!(cache.occupancy, expected2.occupancy);
    }

    #[test]
    fn full_mesh_matches_chunked_concat() {
        let voxels = vec![
            Voxel {
                x: -2,
                y: 0,
                z: 0,
                color: 1,
                material: MaterialId::Plastic,
                object_id: 0,
            },
            Voxel {
                x: 50,
                y: 0,
                z: 0,
                color: 2,
                material: MaterialId::Plastic,
                object_id: 0,
            },
        ];
        let (full, _) = build_greedy_mesh(&voxels, &[]);
        let (chunked, _) = build_greedy_mesh_chunked(&voxels, SPATIAL_CHUNK_SIZE, &[]);
        assert_eq!(full.indices.len(), chunked.indices.len());
        assert_eq!(sorted_triangle_set(&full), sorted_triangle_set(&chunked));
    }
}

/// Greedy mesh for `emit` voxels only, using `map` for neighbor occlusion (include a 1-voxel halo around each chunk).
/// `mat_kind` per vertex: 0 plastic/rubber, 0.5 metal, 1 glow, 2 glass, 2.5 water.
pub fn build_greedy_mesh_mapped(emit: &[Voxel], map: &AHashMap<VoxelCoord, Voxel>) -> MeshBuffers {
    let mut buckets: AHashMap<(u32, u8), Vec<IVec3>> = AHashMap::new();
    for v in emit {
        buckets
            .entry(bucket_key_parts(v))
            .or_default()
            .push(coord_key(v.x, v.y, v.z));
    }

    let mut out = MeshBuffers {
        positions: Vec::new(),
        normals: Vec::new(),
        colors: Vec::new(),
        mat_kind: Vec::new(),
        ao: Vec::new(),
        indices: Vec::new(),
    };

    for cell_positions in buckets.values() {
        let Some(&first_pos) = cell_positions.first() else {
            continue;
        };
        let vx = map[&first_pos];
        let col = color_rgb(vx.color);
        let mat_k = mat_kind_f32(vx.material);

        let mut faces: Vec<(IVec3, usize, i32)> = Vec::with_capacity(cell_positions.len() * 4);
        for &pos in cell_positions {
            let source = map[&pos];
            for i in 0..6usize {
                let axis = i / 2;
                let sign = if i % 2 == 0 { 1 } else { -1 };
                if !neighbor_occludes_face(
                    map,
                    pos,
                    axis,
                    sign,
                    source.material,
                    source.object_id,
                ) {
                    faces.push((pos, axis, sign));
                }
            }
        }

        let mut slices: AHashMap<GreedySliceKey, Vec<(i32, i32)>> = AHashMap::new();
        for (pos, axis, sign) in faces {
            let (x, y, z) = pos;
            let depth = match axis {
                0 => x,
                1 => y,
                _ => z,
            };
            let u = if axis == 0 { y } else { x };
            let v = if axis == 2 { y } else { z };
            slices.entry((axis, sign, depth)).or_default().push((u, v));
        }

        for ((axis, sign, depth), cells) in slices {
            let merged = greedy_merge(&cells);
            let n = face_normal(axis, sign);

            for (u, v, w, h) in merged {
                let p00 = quad_corner(axis, sign, depth, u, v);
                let p10 = quad_corner(axis, sign, depth, u + w, v);
                let p11 = quad_corner(axis, sign, depth, u + w, v + h);
                let p01 = quad_corner(axis, sign, depth, u, v + h);

                let ao00 = corner_ao_factor(map, axis, sign, depth, u, v, 0);
                let ao10 = corner_ao_factor(map, axis, sign, depth, u + w - 1, v, 1);
                let ao11 = corner_ao_factor(map, axis, sign, depth, u + w - 1, v + h - 1, 2);
                let ao01 = corner_ao_factor(map, axis, sign, depth, u, v + h - 1, 3);

                let base = (out.positions.len() / 3) as u32;
                for (p, ao_v) in [
                    (p00, ao00),
                    (p10, ao10),
                    (p11, ao11),
                    (p01, ao01),
                ] {
                    out.positions.extend_from_slice(&p.to_array());
                    out.normals.extend_from_slice(&n.to_array());
                    out.colors.extend_from_slice(&col.to_array());
                    out.mat_kind.push(mat_k);
                    out.ao.push(ao_v);
                }

                let ccw = if n.x != 0.0 {
                    n.x > 0.0
                } else if n.y != 0.0 {
                    n.y < 0.0
                } else {
                    n.z > 0.0
                };
                if ccw {
                    out.indices.extend_from_slice(&[
                        base,
                        base + 1,
                        base + 2,
                        base,
                        base + 2,
                        base + 3,
                    ]);
                } else {
                    out.indices.extend_from_slice(&[
                        base,
                        base + 2,
                        base + 1,
                        base,
                        base + 3,
                        base + 2,
                    ]);
                }
            }
        }
    }

    out
}

/// Single axis-aligned cube for add/remove hover preview. `half` is half-edge length (0.5 = unit voxel).
pub fn preview_cube_mesh(
    cx: f32,
    cy: f32,
    cz: f32,
    half: f32,
    color: [f32; 3],
    mat_k: f32,
) -> MeshBuffers {
    let h = half;
    let mut positions: Vec<f32> = Vec::with_capacity(24 * 3);
    let mut normals: Vec<f32> = Vec::with_capacity(24 * 3);
    let mut colors: Vec<f32> = Vec::with_capacity(24 * 3);
    let mut mat_kind: Vec<f32> = Vec::with_capacity(24);
    let mut ao: Vec<f32> = Vec::with_capacity(24);
    let mut indices: Vec<u32> = Vec::with_capacity(36);

    let mut face = |nx: f32, ny: f32, nz: f32, corners: [[f32; 3]; 4]| {
        let base = (positions.len() / 3) as u32;
        for p in corners {
            positions.extend_from_slice(&p);
            normals.extend_from_slice(&[nx, ny, nz]);
            colors.extend_from_slice(&color);
            mat_kind.push(mat_k);
            ao.push(1.0);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    // +X
    face(
        1.0,
        0.0,
        0.0,
        [
            [cx + h, cy - h, cz - h],
            [cx + h, cy + h, cz - h],
            [cx + h, cy + h, cz + h],
            [cx + h, cy - h, cz + h],
        ],
    );
    // -X
    face(
        -1.0,
        0.0,
        0.0,
        [
            [cx - h, cy - h, cz + h],
            [cx - h, cy + h, cz + h],
            [cx - h, cy + h, cz - h],
            [cx - h, cy - h, cz - h],
        ],
    );
    // +Y (CCW from +Y so back-face cull keeps the outside)
    face(
        0.0,
        1.0,
        0.0,
        [
            [cx - h, cy + h, cz + h],
            [cx + h, cy + h, cz + h],
            [cx + h, cy + h, cz - h],
            [cx - h, cy + h, cz - h],
        ],
    );
    // -Y
    face(
        0.0,
        -1.0,
        0.0,
        [
            [cx - h, cy - h, cz - h],
            [cx + h, cy - h, cz - h],
            [cx + h, cy - h, cz + h],
            [cx - h, cy - h, cz + h],
        ],
    );
    // +Z
    face(
        0.0,
        0.0,
        1.0,
        [
            [cx - h, cy - h, cz + h],
            [cx + h, cy - h, cz + h],
            [cx + h, cy + h, cz + h],
            [cx - h, cy + h, cz + h],
        ],
    );
    // -Z
    face(
        0.0,
        0.0,
        -1.0,
        [
            [cx - h, cy - h, cz - h],
            [cx - h, cy + h, cz - h],
            [cx + h, cy + h, cz - h],
            [cx + h, cy - h, cz - h],
        ],
    );

    MeshBuffers {
        positions,
        normals,
        colors,
        mat_kind,
        ao,
        indices,
    }
}

/// 12 edges as thin axis-aligned boxes (triangles). `LineList` is ~1px on many GPUs and disappears on HiDPI.
pub fn preview_cube_wireframe_mesh(
    cx: f32,
    cy: f32,
    cz: f32,
    half: f32,
    color: [f32; 3],
    mat_k: f32,
) -> MeshBuffers {
    let h = half;
    // Thin beams — large `t` read as a second solid shell, not lines.
    let t = (half * 0.048).clamp(0.014, 0.036);
    let mut positions: Vec<f32> = Vec::with_capacity(72 * 6 * 3);
    let mut normals: Vec<f32> = Vec::with_capacity(72 * 6 * 3);
    let mut colors: Vec<f32> = Vec::with_capacity(72 * 6 * 3);
    let mut mat_kind: Vec<f32> = Vec::with_capacity(72 * 6);
    let mut ao: Vec<f32> = Vec::with_capacity(72 * 6);
    let mut indices: Vec<u32> = Vec::with_capacity(72 * 6);

    let mut face = |nx: f32, ny: f32, nz: f32, corners: [[f32; 3]; 4]| {
        let base = (positions.len() / 3) as u32;
        for p in corners {
            positions.extend_from_slice(&p);
            normals.extend_from_slice(&[nx, ny, nz]);
            colors.extend_from_slice(&color);
            mat_kind.push(mat_k);
            ao.push(1.0);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    let mut push_box = |xmin: f32, xmax: f32, ymin: f32, ymax: f32, zmin: f32, zmax: f32| {
        // Same face winding as `preview_cube_mesh` / greedy mesh (CCW outside).
        face(
            1.0,
            0.0,
            0.0,
            [
                [xmax, ymin, zmin],
                [xmax, ymax, zmin],
                [xmax, ymax, zmax],
                [xmax, ymin, zmax],
            ],
        );
        face(
            -1.0,
            0.0,
            0.0,
            [
                [xmin, ymin, zmax],
                [xmin, ymax, zmax],
                [xmin, ymax, zmin],
                [xmin, ymin, zmin],
            ],
        );
        face(
            0.0,
            1.0,
            0.0,
            [
                [xmin, ymax, zmax],
                [xmax, ymax, zmax],
                [xmax, ymax, zmin],
                [xmin, ymax, zmin],
            ],
        );
        face(
            0.0,
            -1.0,
            0.0,
            [
                [xmin, ymin, zmin],
                [xmax, ymin, zmin],
                [xmax, ymin, zmax],
                [xmin, ymin, zmax],
            ],
        );
        face(
            0.0,
            0.0,
            1.0,
            [
                [xmin, ymin, zmax],
                [xmax, ymin, zmax],
                [xmax, ymax, zmax],
                [xmin, ymax, zmax],
            ],
        );
        face(
            0.0,
            0.0,
            -1.0,
            [
                [xmin, ymin, zmin],
                [xmin, ymax, zmin],
                [xmax, ymax, zmin],
                [xmax, ymin, zmin],
            ],
        );
    };

    // Bottom face (z = cz - h): edges 0-1, 1-2, 2-3, 3-0
    push_box(
        cx - h,
        cx + h,
        cy - h - t,
        cy - h + t,
        cz - h - t,
        cz - h + t,
    );
    push_box(
        cx + h - t,
        cx + h + t,
        cy - h,
        cy + h,
        cz - h - t,
        cz - h + t,
    );
    push_box(
        cx - h,
        cx + h,
        cy + h - t,
        cy + h + t,
        cz - h - t,
        cz - h + t,
    );
    push_box(
        cx - h - t,
        cx - h + t,
        cy - h,
        cy + h,
        cz - h - t,
        cz - h + t,
    );

    // Top face (z = cz + h): 4-5, 5-6, 6-7, 7-4
    push_box(
        cx - h,
        cx + h,
        cy - h - t,
        cy - h + t,
        cz + h - t,
        cz + h + t,
    );
    push_box(
        cx + h - t,
        cx + h + t,
        cy - h,
        cy + h,
        cz + h - t,
        cz + h + t,
    );
    push_box(
        cx - h,
        cx + h,
        cy + h - t,
        cy + h + t,
        cz + h - t,
        cz + h + t,
    );
    push_box(
        cx - h - t,
        cx - h + t,
        cy - h,
        cy + h,
        cz + h - t,
        cz + h + t,
    );

    // Verticals: 0-4, 1-5, 2-6, 3-7
    push_box(
        cx - h - t,
        cx - h + t,
        cy - h - t,
        cy - h + t,
        cz - h,
        cz + h,
    );
    push_box(
        cx + h - t,
        cx + h + t,
        cy - h - t,
        cy - h + t,
        cz - h,
        cz + h,
    );
    push_box(
        cx + h - t,
        cx + h + t,
        cy + h - t,
        cy + h + t,
        cz - h,
        cz + h,
    );
    push_box(
        cx - h - t,
        cx - h + t,
        cy + h - t,
        cy + h + t,
        cz - h,
        cz + h,
    );

    MeshBuffers {
        positions,
        normals,
        colors,
        mat_kind,
        ao,
        indices,
    }
}

// ---------------------------------------------------------------------------
// GPU-instanced preview: prototype meshes (position + normal only, at origin)
// ---------------------------------------------------------------------------

/// Lightweight mesh with only positions and normals (no per-vertex color/material).
/// Used as the shared prototype geometry for instanced preview drawing.
pub struct PreviewPrototype {
    /// Flat `[x,y,z, …]` positions.
    pub positions: Vec<f32>,
    /// Flat `[nx,ny,nz, …]` normals.
    pub normals: Vec<f32>,
    pub indices: Vec<u32>,
}

/// Per-instance data for GPU-instanced preview cubes.
///
/// Layout must match the vertex buffer attributes in `preview_instance_layout()`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PreviewInstance {
    /// Column 0 of the model matrix (object_world * translate(cell)).
    pub model_c0: [f32; 4],
    pub model_c1: [f32; 4],
    pub model_c2: [f32; 4],
    pub model_c3: [f32; 4],
    /// Vertex color for this instance (solid fill or wire tint).
    pub color: [f32; 3],
    /// Material-kind tag read by the fragment shader.
    pub mat_kind: f32,
}

/// Result of [`crate::stroke_preview_meshes_for_union`] — instanced bulk voxels plus optional
/// small non-instanced extras (gizmos, polygon markers, etc.).
#[derive(Clone)]
pub struct PreviewInstancedResult {
    pub solid_instances: Vec<PreviewInstance>,
    pub wire_instances: Vec<PreviewInstance>,
    /// Cube half-extent used to build the prototypes (constant per call).
    pub cube_half: f32,
    /// Extra non-instanced solid geometry (gizmos, polygon markers, …).
    pub extra_solid: MeshBuffers,
    /// Extra non-instanced wire geometry.
    pub extra_wire: MeshBuffers,
}

impl PreviewInstancedResult {
    pub fn empty() -> Self {
        Self {
            solid_instances: Vec::new(),
            wire_instances: Vec::new(),
            cube_half: 0.53,
            extra_solid: MeshBuffers::default(),
            extra_wire: MeshBuffers::default(),
        }
    }
}

/// Unit solid cube prototype at the origin with the given half-extent.
/// 24 vertices (4 per face × 6 faces), 36 indices.
pub fn preview_cube_prototype(half: f32) -> PreviewPrototype {
    let h = half;
    let mut positions: Vec<f32> = Vec::with_capacity(24 * 3);
    let mut normals: Vec<f32> = Vec::with_capacity(24 * 3);
    let mut indices: Vec<u32> = Vec::with_capacity(36);

    let mut face = |nx: f32, ny: f32, nz: f32, corners: [[f32; 3]; 4]| {
        let base = (positions.len() / 3) as u32;
        for p in corners {
            positions.extend_from_slice(&p);
            normals.extend_from_slice(&[nx, ny, nz]);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    face(1.0, 0.0, 0.0, [[h, -h, -h], [h, h, -h], [h, h, h], [h, -h, h]]);
    face(-1.0, 0.0, 0.0, [[-h, -h, h], [-h, h, h], [-h, h, -h], [-h, -h, -h]]);
    face(0.0, 1.0, 0.0, [[-h, h, h], [h, h, h], [h, h, -h], [-h, h, -h]]);
    face(0.0, -1.0, 0.0, [[-h, -h, -h], [h, -h, -h], [h, -h, h], [-h, -h, h]]);
    face(0.0, 0.0, 1.0, [[-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]]);
    face(0.0, 0.0, -1.0, [[-h, -h, -h], [-h, h, -h], [h, h, -h], [h, -h, -h]]);

    PreviewPrototype { positions, normals, indices }
}

/// Unit wireframe prototype at the origin (12 edges as thin boxes, same winding as
/// [`preview_cube_wireframe_mesh`]).
pub fn preview_wireframe_prototype(half: f32) -> PreviewPrototype {
    let h = half;
    let t = (half * 0.048).clamp(0.014, 0.036);
    let mut positions: Vec<f32> = Vec::with_capacity(72 * 6 * 3);
    let mut normals: Vec<f32> = Vec::with_capacity(72 * 6 * 3);
    let mut indices: Vec<u32> = Vec::with_capacity(72 * 6);

    let mut face = |nx: f32, ny: f32, nz: f32, corners: [[f32; 3]; 4]| {
        let base = (positions.len() / 3) as u32;
        for p in corners {
            positions.extend_from_slice(&p);
            normals.extend_from_slice(&[nx, ny, nz]);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    let mut push_box = |xmin: f32, xmax: f32, ymin: f32, ymax: f32, zmin: f32, zmax: f32| {
        face(1.0, 0.0, 0.0, [[xmax, ymin, zmin], [xmax, ymax, zmin], [xmax, ymax, zmax], [xmax, ymin, zmax]]);
        face(-1.0, 0.0, 0.0, [[xmin, ymin, zmax], [xmin, ymax, zmax], [xmin, ymax, zmin], [xmin, ymin, zmin]]);
        face(0.0, 1.0, 0.0, [[xmin, ymax, zmax], [xmax, ymax, zmax], [xmax, ymax, zmin], [xmin, ymax, zmin]]);
        face(0.0, -1.0, 0.0, [[xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymin, zmax], [xmin, ymin, zmax]]);
        face(0.0, 0.0, 1.0, [[xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax]]);
        face(0.0, 0.0, -1.0, [[xmin, ymin, zmin], [xmin, ymax, zmin], [xmax, ymax, zmin], [xmax, ymin, zmin]]);
    };

    // Bottom face edges (z = -h)
    push_box(-h, h, -h - t, -h + t, -h - t, -h + t);
    push_box(h - t, h + t, -h, h, -h - t, -h + t);
    push_box(-h, h, h - t, h + t, -h - t, -h + t);
    push_box(-h - t, -h + t, -h, h, -h - t, -h + t);
    // Top face edges (z = +h)
    push_box(-h, h, -h - t, -h + t, h - t, h + t);
    push_box(h - t, h + t, -h, h, h - t, h + t);
    push_box(-h, h, h - t, h + t, h - t, h + t);
    push_box(-h - t, -h + t, -h, h, h - t, h + t);
    // Vertical edges
    push_box(-h - t, -h + t, -h - t, -h + t, -h, h);
    push_box(h - t, h + t, -h - t, -h + t, -h, h);
    push_box(h - t, h + t, h - t, h + t, -h, h);
    push_box(-h - t, -h + t, h - t, h + t, -h, h);

    PreviewPrototype { positions, normals, indices }
}

/// Thin solid beam between two points (used for squishy metaball wire spheres).
pub fn beam_segment_mesh(
    p0: [f32; 3],
    p1: [f32; 3],
    thickness: f32,
    color: [f32; 3],
    mat_k: f32,
) -> MeshBuffers {
    let a = Vec3::from_array(p0);
    let b = Vec3::from_array(p1);
    let d = b - a;
    let len = d.length();
    if len < 1e-5 {
        return MeshBuffers::default();
    }
    let dir = d / len;
    let mid = (a + b) * 0.5;
    let up = if dir.y.abs() < 0.9 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let mut u_ax = dir.cross(up);
    if u_ax.length_squared() < 1e-10 {
        u_ax = dir.cross(Vec3::Z);
    }
    u_ax = u_ax.normalize();
    let v_ax = dir.cross(u_ax);
    let hx = thickness * 0.5;
    let hy = thickness * 0.5;
    let hz = len * 0.5;

    let corner = |sx: f32, sy: f32, sz: f32| -> [f32; 3] {
        (mid + u_ax * (sx * hx) + v_ax * (sy * hy) + dir * (sz * hz)).to_array()
    };

    let c000 = corner(-1.0, -1.0, -1.0);
    let c100 = corner(1.0, -1.0, -1.0);
    let c110 = corner(1.0, 1.0, -1.0);
    let c010 = corner(-1.0, 1.0, -1.0);
    let c001 = corner(-1.0, -1.0, 1.0);
    let c101 = corner(1.0, -1.0, 1.0);
    let c111 = corner(1.0, 1.0, 1.0);
    let c011 = corner(-1.0, 1.0, 1.0);

    let mut positions: Vec<f32> = Vec::with_capacity(72);
    let mut normals: Vec<f32> = Vec::with_capacity(72);
    let mut colors: Vec<f32> = Vec::with_capacity(72);
    let mut mat_kind: Vec<f32> = Vec::with_capacity(24);
    let mut ao: Vec<f32> = Vec::with_capacity(24);
    let mut indices: Vec<u32> = Vec::with_capacity(36);

    let mut face = |nx: f32, ny: f32, nz: f32, corners: [[f32; 3]; 4]| {
        let base = (positions.len() / 3) as u32;
        for p in corners {
            positions.extend_from_slice(&p);
            normals.extend_from_slice(&[nx, ny, nz]);
            colors.extend_from_slice(&color);
            mat_kind.push(mat_k);
            ao.push(1.0);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    // Winding CCW when viewed from outward normal (matches `preview_cube_mesh`).
    face(
        u_ax.x,
        u_ax.y,
        u_ax.z,
        [c100, c110, c111, c101],
    );
    face(
        -u_ax.x,
        -u_ax.y,
        -u_ax.z,
        [c000, c001, c011, c010],
    );
    face(
        v_ax.x,
        v_ax.y,
        v_ax.z,
        [c010, c011, c111, c110],
    );
    face(
        -v_ax.x,
        -v_ax.y,
        -v_ax.z,
        [c000, c100, c101, c001],
    );
    face(
        dir.x,
        dir.y,
        dir.z,
        [c001, c101, c111, c011],
    );
    face(
        -dir.x,
        -dir.y,
        -dir.z,
        [c000, c010, c110, c100],
    );

    MeshBuffers {
        positions,
        normals,
        colors,
        mat_kind,
        ao,
        indices,
    }
}

/// Three orthogonal wire rings (web `SphereGeometry` wireframe parity for pick shells).
pub fn append_sphere_pick_rings(
    dst: &mut MeshBuffers,
    cx: f32,
    cy: f32,
    cz: f32,
    radius: f32,
    color: [f32; 3],
    mat_k: f32,
    segments: u32,
) {
    let r = radius.max(0.2);
    let t = (r * 0.025).clamp(0.012, 0.045);
    let n = segments.max(8) as usize;
    let tau = std::f32::consts::TAU;
    for i in 0..n {
        let a0 = i as f32 / n as f32 * tau;
        let a1 = (i + 1) as f32 / n as f32 * tau;
        let p0 = [cx + r * a0.cos(), cy + r * a0.sin(), cz];
        let p1 = [cx + r * a1.cos(), cy + r * a1.sin(), cz];
        append_mesh_buffers(dst, beam_segment_mesh(p0, p1, t, color, mat_k));
    }
    for i in 0..n {
        let a0 = i as f32 / n as f32 * tau;
        let a1 = (i + 1) as f32 / n as f32 * tau;
        let p0 = [cx + r * a0.cos(), cy, cz + r * a0.sin()];
        let p1 = [cx + r * a1.cos(), cy, cz + r * a1.sin()];
        append_mesh_buffers(dst, beam_segment_mesh(p0, p1, t, color, mat_k));
    }
    for i in 0..n {
        let a0 = i as f32 / n as f32 * tau;
        let a1 = (i + 1) as f32 / n as f32 * tau;
        let p0 = [cx, cy + r * a0.cos(), cz + r * a0.sin()];
        let p1 = [cx, cy + r * a1.cos(), cz + r * a1.sin()];
        append_mesh_buffers(dst, beam_segment_mesh(p0, p1, t, color, mat_k));
    }
}

pub fn transform_mesh_buffers(mesh: &mut MeshBuffers, model: glam::Mat4) {
    let det = model.determinant();
    let inv_t = if det.is_finite() && det.abs() > 1e-20 {
        model.inverse().transpose()
    } else {
        glam::Mat4::IDENTITY
    };
    for i in (0..mesh.positions.len()).step_by(3) {
        let p = glam::Vec3::new(mesh.positions[i], mesh.positions[i + 1], mesh.positions[i + 2]);
        let pw = model.transform_point3(p);
        mesh.positions[i] = pw.x;
        mesh.positions[i + 1] = pw.y;
        mesh.positions[i + 2] = pw.z;
    }
    for i in (0..mesh.normals.len()).step_by(3) {
        let n = glam::Vec3::new(mesh.normals[i], mesh.normals[i + 1], mesh.normals[i + 2]);
        let nw = inv_t.transform_vector3(n).normalize();
        mesh.normals[i] = nw.x;
        mesh.normals[i + 1] = nw.y;
        mesh.normals[i + 2] = nw.z;
    }
}

/// Bounds of voxel centers transformed into world space (respects object visibility).
pub fn mesh_bounds_from_voxels_world(voxels: &[Voxel], objects: &[SceneObject]) -> Option<MeshBounds> {
    if voxels.is_empty() {
        return None;
    }
    let default_objs = crate::voxelle::default_scene_objects();
    let objs: &[SceneObject] = if objects.is_empty() {
        default_objs.as_slice()
    } else {
        objects
    };
    let mats = crate::voxelle::scene::object_world_matrices_by_id(objs);
    let vis = crate::voxelle::scene::object_visibility_by_id(objs);
    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
    let mut any = false;
    for v in voxels {
        if !vis.get(&v.object_id).copied().unwrap_or(true) {
            continue;
        }
        any = true;
        let m = mats.get(&v.object_id).copied().unwrap_or(Mat4::IDENTITY);
        let pf = m.transform_point3(glam::Vec3::new(v.x as f32, v.y as f32, v.z as f32));
        min = min.min(pf);
        max = max.max(pf);
    }
    if !any {
        return None;
    }
    Some(MeshBounds { min, max })
}

fn append_object_meshes_sorted(
    voxels: &[Voxel],
    map: &AHashMap<VoxelCoord, Voxel>,
    objects: &[SceneObject],
    dst: &mut MeshBuffers,
) {
    let mut order: Vec<(i32, u32)> = objects.iter().map(|o| (o.sort_order, o.id)).collect();
    order.sort_by_key(|(s, id)| (*s, *id));
    for (_, oid) in order {
        if !crate::voxelle::scene::is_object_visible(objects, oid) {
            continue;
        }
        let emit: Vec<Voxel> = voxels.iter().filter(|v| v.object_id == oid).cloned().collect();
        if emit.is_empty() {
            continue;
        }
        let mut part = build_greedy_mesh_mapped(&emit, map);
        let m = crate::voxelle::object_world_matrix(objects, oid);
        transform_mesh_buffers(&mut part, m);
        append_mesh_buffers(dst, part);
    }
}

/// `mat_kind` per vertex: 0 plastic/rubber, 0.5 metal, 1 glow, 2 glass, 2.5 water.
pub fn build_greedy_mesh(voxels: &[Voxel], objects: &[SceneObject]) -> (MeshBuffers, MeshBounds) {
    let default_objs = crate::voxelle::default_scene_objects();
    let objs: &[SceneObject] = if objects.is_empty() {
        default_objs.as_slice()
    } else {
        objects
    };
    let map = voxel_map(voxels);
    let mut combined = MeshBuffers::default();
    append_object_meshes_sorted(voxels, &map, objs, &mut combined);
    let bounds = mesh_bounds_from_voxels_world(voxels, objs)
        .or_else(|| mesh_bounds_from_voxels(voxels))
        .unwrap_or_else(|| MeshBounds {
            min: glam::Vec3::ZERO,
            max: glam::Vec3::ZERO,
        });
    (combined, bounds)
}

/// Same output as [`build_greedy_mesh`], but builds per spatial chunk (with global occlusion map) for large scenes.
#[allow(dead_code)] // Retained for tests and callers; runtime uses [`build_all_chunk_meshes_btree`].
pub fn build_greedy_mesh_chunked(
    voxels: &[Voxel],
    chunk_size: i32,
    objects: &[SceneObject],
) -> (MeshBuffers, MeshBounds) {
    if voxels.is_empty() {
        return (
            MeshBuffers::default(),
            MeshBounds {
                min: glam::Vec3::ZERO,
                max: glam::Vec3::ZERO,
            },
        );
    }
    let default_objs = crate::voxelle::default_scene_objects();
    let objs: &[SceneObject] = if objects.is_empty() {
        default_objs.as_slice()
    } else {
        objects
    };
    let map = voxel_map(voxels);
    let bounds = mesh_bounds_from_voxels_world(voxels, objs)
        .or_else(|| mesh_bounds_from_voxels(voxels))
        .unwrap();
    let chunks = voxels_by_spatial_chunks(voxels, chunk_size);
    let mut acc = MeshBuffers::default();
    for (_halo, core) in chunks {
        if core.is_empty() {
            continue;
        }
        append_object_meshes_sorted(&core, &map, objs, &mut acc);
    }
    (acc, bounds)
}

/// Line-list vertices for collab line shader (`position` + `color` per vertex): expanding ripples in XZ at voxel center Y.
pub fn ping_ripple_line_vertices(
    vx: i32,
    vy: i32,
    vz: i32,
    elapsed: f32,
    color: [f32; 3],
) -> Vec<f32> {
    const SEGMENTS: usize = 44;
    const WAVE_SPEED: f32 = 2.5;
    const WAVE_GAP: f32 = 0.55;
    const MAX_R: f32 = 3.6;
    let cx = vx as f32 + 0.5;
    let cy = vy as f32 + 0.5;
    let cz = vz as f32 + 0.5;
    let mut out: Vec<f32> = Vec::with_capacity(SEGMENTS * 6 * 6 * 24);
    let tau = std::f32::consts::TAU;
    for wave_idx in 0..5 {
        let t = elapsed * WAVE_SPEED - (wave_idx as f32) * WAVE_GAP;
        let r = t.rem_euclid(MAX_R);
        if r < 0.05 || r > MAX_R - 0.03 {
            continue;
        }
        let fade = (1.0 - r / MAX_R).clamp(0.2, 1.0);
        let c = [
            color[0] * fade,
            color[1] * fade,
            color[2] * fade,
        ];
        for i in 0..SEGMENTS {
            let a0 = (i as f32) / (SEGMENTS as f32) * tau;
            let a1 = ((i + 1) as f32) / (SEGMENTS as f32) * tau;
            let x0 = cx + r * a0.cos();
            let z0 = cz + r * a0.sin();
            let x1 = cx + r * a1.cos();
            let z1 = cz + r * a1.sin();
            out.extend_from_slice(&[x0, cy, z0, c[0], c[1], c[2]]);
            out.extend_from_slice(&[x1, cy, z1, c[0], c[1], c[2]]);
        }
    }
    out
}

/// Web `SELECTION_OVERLAY_HEX` — greedy selection overlay tint.
pub const SELECTION_OVERLAY_COLOR: u32 = 0x3399ff;

/// Above this count, use an axis-aligned box + bbox wireframe (web parity).
pub const SELECTION_OVERLAY_MESH_THRESHOLD: usize = 20_000;

/// Face-adjacent offsets (6-connected).
const FACE_NEIGHBOR_OFFSETS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// Keep only voxels that are not **strictly interior** to `set` (every face neighbor lies in `set`).
/// Used for preview and selection overlay meshing so large solid regions only pay for a surface shell.
/// Single-cell sets are returned unchanged.
pub fn filter_voxel_set_to_shell(set: &AHashSet<VoxelCoord>) -> AHashSet<VoxelCoord> {
    let n = set.len();
    if n <= 1 {
        return set.clone();
    }
    let mut out = AHashSet::with_capacity(n.min(65_536));
    for &c in set.iter() {
        let mut has_outside = false;
        for &(dx, dy, dz) in &FACE_NEIGHBOR_OFFSETS {
            let nb = (c.0 + dx, c.1 + dy, c.2 + dz);
            if !set.contains(&nb) {
                has_outside = true;
                break;
            }
        }
        if has_outside {
            out.insert(c);
        }
    }
    out
}

pub fn selection_bounds(sel: &AHashSet<VoxelCoord>) -> Option<(i32, i32, i32, i32, i32, i32)> {
    let mut it = sel.iter();
    let first = *it.next()?;
    let (mut min_x, mut min_y, mut min_z) = first;
    let (mut max_x, mut max_y, mut max_z) = first;
    for &(x, y, z) in it {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        min_z = min_z.min(z);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
        max_z = max_z.max(z);
    }
    Some((min_x, min_y, min_z, max_x, max_y, max_z))
}

/// Axis-aligned box centered at `(cx,cy,cz)` with half-extents `(hx,hy,hz)` — same winding as [`preview_cube_mesh`].
pub fn axis_aligned_box_mesh(
    cx: f32,
    cy: f32,
    cz: f32,
    hx: f32,
    hy: f32,
    hz: f32,
    color: [f32; 3],
    mat_k: f32,
) -> MeshBuffers {
    let mut positions: Vec<f32> = Vec::with_capacity(24 * 3);
    let mut normals: Vec<f32> = Vec::with_capacity(24 * 3);
    let mut colors: Vec<f32> = Vec::with_capacity(24 * 3);
    let mut mat_kind: Vec<f32> = Vec::with_capacity(24);
    let mut ao: Vec<f32> = Vec::with_capacity(24);
    let mut indices: Vec<u32> = Vec::with_capacity(36);

    let mut face = |nx: f32, ny: f32, nz: f32, corners: [[f32; 3]; 4]| {
        let base = (positions.len() / 3) as u32;
        for p in corners {
            positions.extend_from_slice(&p);
            normals.extend_from_slice(&[nx, ny, nz]);
            colors.extend_from_slice(&color);
            mat_kind.push(mat_k);
            ao.push(1.0);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    face(
        1.0,
        0.0,
        0.0,
        [
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx + hx, cy + hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
        ],
    );
    face(
        -1.0,
        0.0,
        0.0,
        [
            [cx - hx, cy - hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz - hz],
        ],
    );
    face(
        0.0,
        1.0,
        0.0,
        [
            [cx - hx, cy + hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
        ],
    );
    face(
        0.0,
        -1.0,
        0.0,
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz + hz],
            [cx - hx, cy - hy, cz + hz],
        ],
    );
    face(
        0.0,
        0.0,
        1.0,
        [
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
    );
    face(
        0.0,
        0.0,
        -1.0,
        [
            [cx - hx, cy - hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
        ],
    );

    MeshBuffers {
        positions,
        normals,
        colors,
        mat_kind,
        ao,
        indices,
    }
}

/// Solid selection overlay: greedy mesh for modest selections, AABB box when huge (matches web).
pub fn mesh_buffers_selection_overlay_solid(
    sel: &AHashSet<VoxelCoord>,
    world: &AHashMap<VoxelCoord, Voxel>,
) -> MeshBuffers {
    if sel.is_empty() {
        return MeshBuffers::default();
    }
    let Some((min_x, min_y, min_z, max_x, max_y, max_z)) = selection_bounds(sel) else {
        return MeshBuffers::default();
    };
    if sel.len() >= SELECTION_OVERLAY_MESH_THRESHOLD {
        let c = color_rgb(SELECTION_OVERLAY_COLOR);
        let col = [c.x, c.y, c.z];
        let hx = (max_x - min_x + 1) as f32 * 0.5;
        let hy = (max_y - min_y + 1) as f32 * 0.5;
        let hz = (max_z - min_z + 1) as f32 * 0.5;
        let cx = (min_x + max_x) as f32 * 0.5;
        let cy = (min_y + max_y) as f32 * 0.5;
        let cz = (min_z + max_z) as f32 * 0.5;
        return axis_aligned_box_mesh(cx, cy, cz, hx, hy, hz, col, 1.0);
    }
    let sel_mesh = filter_voxel_set_to_shell(sel);
    if sel_mesh.is_empty() {
        return MeshBuffers::default();
    }
    let mut combined: AHashMap<VoxelCoord, Voxel> = AHashMap::with_capacity(world.len() + sel_mesh.len());
    for (k, v) in world.iter() {
        combined.insert(*k, *v);
    }
    for &c in sel_mesh.iter() {
        let oid = world.get(&c).map(|v| v.object_id).unwrap_or(0);
        combined.insert(
            c,
            Voxel {
                x: c.0,
                y: c.1,
                z: c.2,
                color: SELECTION_OVERLAY_COLOR,
                material: MaterialId::Plastic,
                object_id: oid,
            },
        );
    }
    let mut emit: Vec<Voxel> = Vec::with_capacity(sel_mesh.len());
    for &c in sel_mesh.iter() {
        if let Some(v) = combined.get(&c) {
            emit.push(*v);
        }
    }
    if emit.is_empty() {
        return MeshBuffers::default();
    }
    build_greedy_mesh_mapped(&emit, &combined)
}

/// Line-list vertices for selection AABB wireframe (`position` + `color` per vertex); matches web `selectionAabbWireframePositions` tint.
pub fn selection_aabb_line_vertices(
    min_x: i32,
    min_y: i32,
    min_z: i32,
    max_x: i32,
    max_y: i32,
    max_z: i32,
) -> Vec<f32> {
    let x0 = min_x as f32 - 0.5;
    let y0 = min_y as f32 - 0.5;
    let z0 = min_z as f32 - 0.5;
    let x1 = max_x as f32 + 0.5;
    let y1 = max_y as f32 + 0.5;
    let z1 = max_z as f32 + 0.5;
    let r = 0x9f as f32 / 255.0;
    let g = 0xd8 as f32 / 255.0;
    let b = 0xff as f32 / 255.0;
    let pos: [f32; 72] = [
        x0, y0, z0, x1, y0, z0, x1, y0, z0, x1, y0, z1, x1, y0, z1, x0, y0, z1, x0, y0, z1, x0, y0, z0,
        x0, y1, z0, x1, y1, z0, x1, y1, z0, x1, y1, z1, x1, y1, z1, x0, y1, z1, x0, y1, z1, x0, y1, z0,
        x0, y0, z0, x0, y1, z0, x1, y0, z0, x1, y1, z0, x1, y0, z1, x1, y1, z1, x0, y0, z1, x0, y1, z1,
    ];
    let mut out = Vec::with_capacity(144);
    for chunk in pos.chunks(3) {
        out.extend_from_slice(&[chunk[0], chunk[1], chunk[2], r, g, b]);
    }
    out
}

// Matches web `gridLines.ts` (surface lift reduces depth fighting).
const GRID_SURFACE_LIFT: f32 = 0.01;

const VOXEL_GRID_CUBE_EDGES: [[f32; 6]; 12] = [
    [-0.5, -0.5, -0.5, 0.5, -0.5, -0.5],
    [-0.5, -0.5, -0.5, -0.5, 0.5, -0.5],
    [-0.5, -0.5, -0.5, -0.5, -0.5, 0.5],
    [0.5, -0.5, -0.5, 0.5, 0.5, -0.5],
    [0.5, -0.5, -0.5, 0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5, 0.5, 0.5, -0.5],
    [-0.5, 0.5, -0.5, -0.5, 0.5, 0.5],
    [-0.5, -0.5, 0.5, 0.5, -0.5, 0.5],
    [-0.5, -0.5, 0.5, -0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5, 0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5, 0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
];

const VOXEL_GRID_EDGE_NEIGHBORS: [[(i32, i32, i32); 2]; 12] = [
    [(0, -1, 0), (0, 0, -1)],
    [(-1, 0, 0), (0, 0, -1)],
    [(-1, 0, 0), (0, -1, 0)],
    [(1, 0, 0), (0, 0, -1)],
    [(1, 0, 0), (0, -1, 0)],
    [(0, 1, 0), (0, 0, -1)],
    [(-1, 0, 0), (0, 1, 0)],
    [(0, -1, 0), (0, 0, 1)],
    [(-1, 0, 0), (0, 0, 1)],
    [(1, 0, 0), (0, 1, 0)],
    [(1, 0, 0), (0, 0, 1)],
    [(0, 1, 0), (0, 0, 1)],
];

/// Indexed line-list for per-voxel surface borders.
///
/// Returns `(vertices, indices)` where vertices are `[x,y,z,r,g,b]` and indices
/// form a line-list. Edges shared by adjacent surface voxels are emitted only
/// once, and endpoint vertices are reused via the index buffer.
pub fn voxel_surface_grid_line_vertices(occupancy: &AHashMap<VoxelCoord, Voxel>) -> (Vec<f32>, Vec<u32>) {
    if occupancy.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let has = |x: i32, y: i32, z: i32| occupancy.contains_key(&coord_key(x, y, z));
    let r = 0x9f as f32 / 255.0;
    let g = 0xd8 as f32 / 255.0;
    let b = 0xff as f32 / 255.0;

    // Vertex dedup: map quantised position → vertex index.
    // Positions are on a half-integer grid offset by a tiny lift, so we quantise
    // to 1/1024 units which is far below visual precision.
    let mut vert_map: AHashMap<(i32, i32, i32), u32> = AHashMap::new();
    let mut verts: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Edge dedup: canonical (min,max) endpoint pair.
    let mut seen_edges: AHashSet<(u64, u64)> = AHashSet::new();

    let quantise = |v: f32| -> i32 { (v * 1024.0).round() as i32 };
    let pack_key = |x: i32, y: i32, z: i32| -> u64 {
        // Shift into positive range before packing to avoid sign issues.
        let xu = (x as i64 + 0x100000) as u64;
        let yu = (y as i64 + 0x100000) as u64;
        let zu = (z as i64 + 0x100000) as u64;
        xu | (yu << 21) | (zu << 42)
    };

    let push_vert = |px: f32, py: f32, pz: f32,
                          vert_map: &mut AHashMap<(i32, i32, i32), u32>,
                          verts: &mut Vec<f32>| -> u32 {
        let qk = (quantise(px), quantise(py), quantise(pz));
        if let Some(&idx) = vert_map.get(&qk) {
            return idx;
        }
        let idx = (verts.len() / 6) as u32;
        verts.extend_from_slice(&[px, py, pz, r, g, b]);
        vert_map.insert(qk, idx);
        idx
    };

    for &(x, y, z) in occupancy.keys() {
        for i in 0..12 {
            let [(dx1, dy1, dz1), (dx2, dy2, dz2)] = VOXEL_GRID_EDGE_NEIGHBORS[i];
            let n1 = has(x + dx1, y + dy1, z + dz1);
            let n2 = has(x + dx2, y + dy2, z + dz2);
            if n1 && n2 {
                continue;
            }
            let edge = VOXEL_GRID_CUBE_EDGES[i];
            let mut ox = 0.0f32;
            let mut oy = 0.0f32;
            let mut oz = 0.0f32;
            if GRID_SURFACE_LIFT > 0.0 {
                if !n1 {
                    ox += dx1 as f32;
                    oy += dy1 as f32;
                    oz += dz1 as f32;
                }
                if !n2 {
                    ox += dx2 as f32;
                    oy += dy2 as f32;
                    oz += dz2 as f32;
                }
                let len = (ox * ox + oy * oy + oz * oz).sqrt();
                if len > 0.0 {
                    let k = GRID_SURFACE_LIFT / len;
                    ox *= k;
                    oy *= k;
                    oz *= k;
                }
            }
            let xa = x as f32 + edge[0] + ox;
            let ya = y as f32 + edge[1] + oy;
            let za = z as f32 + edge[2] + oz;
            let xb = x as f32 + edge[3] + ox;
            let yb = y as f32 + edge[4] + oy;
            let zb = z as f32 + edge[5] + oz;

            // Deduplicate: canonical edge key (smaller endpoint first).
            let ka = pack_key(quantise(xa), quantise(ya), quantise(za));
            let kb = pack_key(quantise(xb), quantise(yb), quantise(zb));
            let edge_key = if ka <= kb { (ka, kb) } else { (kb, ka) };
            if !seen_edges.insert(edge_key) {
                continue;
            }

            let ia = push_vert(xa, ya, za, &mut vert_map, &mut verts);
            let ib = push_vert(xb, yb, zb, &mut vert_map, &mut verts);
            indices.push(ia);
            indices.push(ib);
        }
    }
    (verts, indices)
}

#[cfg(test)]
mod gpu_pack_tests {
    use super::*;
    use crate::voxelle::MaterialId;

    #[test]
    fn pack_gpu_greedy_single_voxel_ok() {
        let voxels = vec![Voxel {
            x: 0,
            y: 0,
            z: 0,
            color: 0xff00ff,
            material: MaterialId::Plastic,
            object_id: 0,
        }];
        let map = voxel_map(&voxels);
        let (headers, bits) = pack_gpu_greedy_slices(&map, &voxels).expect("pack");
        assert!(!headers.is_empty());
        assert!(!bits.is_empty());
        let (cpu, _) = build_greedy_mesh(&voxels, &[]);
        assert!(!cpu.indices.is_empty());
    }

    /// CPU chunk mesh vs [`pack_gpu_greedy_slices`] on the same core slice: both see faces or both empty (single-object incremental path).
    #[test]
    fn incremental_chunk_pack_nonempty_matches_cpu_mesh() {
        let voxels: Vec<Voxel> = (0..3)
            .flat_map(|x| {
                (0..3).flat_map(move |y| {
                    (0..3).map(move |z| Voxel {
                        x,
                        y,
                        z,
                        color: 0x00_88_cc,
                        material: MaterialId::Plastic,
                        object_id: 0,
                    })
                })
            })
            .collect();
        let cs = SPATIAL_CHUNK_SIZE;
        let cache = SpatialMeshCache::from_voxels(&voxels, cs).expect("cache");
        for &key in cache.buckets.keys() {
            let mut core_vec: Vec<Voxel> = cache
                .buckets
                .get(&key)
                .map(|b| b.values().copied().collect())
                .unwrap_or_default();
            core_vec.sort_unstable_by_key(|v| (v.x, v.y, v.z));
            let cpu = mesh_buffers_for_chunk_key(&cache.buckets, &cache.occupancy, key);
            let (headers, _) = pack_gpu_greedy_slices(&cache.occupancy, &core_vec).expect("pack");
            assert_eq!(
                cpu.indices.is_empty(),
                headers.is_empty(),
                "chunk key {:?}",
                key
            );
        }
    }

    #[test]
    fn voxel_surface_grid_line_single_cube_all_twelve_edges() {
        let v = Voxel {
            x: 0,
            y: 0,
            z: 0,
            color: 0,
            material: MaterialId::Plastic,
            object_id: 0,
        };
        let mut m = AHashMap::new();
        m.insert((0, 0, 0), v);
        let (verts, indices) = voxel_surface_grid_line_vertices(&m);
        // Single isolated cube: surface lift pushes each edge in a different
        // direction, so no vertex sharing within one voxel.
        // 12 edges × 2 endpoints = 24 unique vertices, 24 indices.
        assert_eq!(verts.len(), 12 * 2 * 6);
        assert_eq!(indices.len(), 12 * 2);
    }

    /// 2×1 merged +Y top: voxel at `(-1,1,0)` occludes only the `(cu,cv)=(0,0)` corner’s outward samples (see `AO_NEIGHBORS`), not `(1,0)` / corner 1.
    #[test]
    fn merged_quad_per_corner_ao_not_uniform() {
        let voxels = vec![
            Voxel {
                x: 0,
                y: 0,
                z: 0,
                color: 0xcc_cc_cc,
                material: MaterialId::Plastic,
                object_id: 0,
            },
            Voxel {
                x: 1,
                y: 0,
                z: 0,
                color: 0xcc_cc_cc,
                material: MaterialId::Plastic,
                object_id: 0,
            },
            Voxel {
                x: -1,
                y: 1,
                z: 0,
                color: 0xcc_cc_cc,
                material: MaterialId::Plastic,
                object_id: 0,
            },
        ];
        let map = voxel_map(&voxels);
        let (mesh, _) = build_greedy_mesh(&voxels, &[]);
        assert!(!mesh.ao.is_empty());
        let mut set = AHashSet::new();
        for &a in &mesh.ao {
            set.insert((a * 1000.0).round() as i32);
        }
        assert!(
            set.len() >= 2,
            "expected varying AO after per-corner fix, got unique {:?}",
            set
        );

        let axis = 1usize;
        let sign = 1i32;
        let depth = 0i32;
        let ao00 = corner_ao_factor(&map, axis, sign, depth, 0, 0, 0);
        let ao10 = corner_ao_factor(&map, axis, sign, depth, 1, 0, 1);
        assert!(
            (ao00 - ao10).abs() > 1e-4,
            "expected different corner AO: ao00={} ao10={}",
            ao00,
            ao10
        );
    }

    /// Glass in an AO neighbor slot does not occlude (unlike same-object plastic).
    /// +X face on `(5,5,5)`: corner 0 samples `(6,4,5)` among others — occluder there is plastic vs glass.
    #[test]
    fn corner_ao_transmissive_neighbor_does_not_occlude() {
        fn plastic_at(x: i32, y: i32, z: i32) -> Voxel {
            Voxel {
                x,
                y,
                z,
                color: 1,
                material: MaterialId::Plastic,
                object_id: 0,
            }
        }
        let mut map_plastic = AHashMap::new();
        map_plastic.insert((5, 5, 5), plastic_at(5, 5, 5));
        map_plastic.insert((6, 4, 5), plastic_at(6, 4, 5));

        let mut map_glass = AHashMap::new();
        map_glass.insert((5, 5, 5), plastic_at(5, 5, 5));
        map_glass.insert(
            (6, 4, 5),
            Voxel {
                x: 6,
                y: 4,
                z: 5,
                color: 1,
                material: MaterialId::Glass,
                object_id: 0,
            },
        );

        let axis = 0usize;
        let sign = 1i32;
        let depth = 5i32;
        let cu = 5i32;
        let cv = 5i32;
        let ao_p = corner_ao_factor(&map_plastic, axis, sign, depth, cu, cv, 0);
        let ao_g = corner_ao_factor(&map_glass, axis, sign, depth, cu, cv, 0);
        assert!(
            ao_g > ao_p,
            "glass should occlude less than plastic: ao_g={} ao_p={}",
            ao_g,
            ao_p
        );
    }
}

#[cfg(test)]
mod shell_tests {
    use super::*;
    use ahash::AHashSet;

    #[test]
    fn filter_shell_drops_3x3x3_center() {
        let mut s = AHashSet::new();
        for x in 0..3 {
            for y in 0..3 {
                for z in 0..3 {
                    s.insert((x, y, z));
                }
            }
        }
        let sh = filter_voxel_set_to_shell(&s);
        assert_eq!(sh.len(), 26);
        assert!(!sh.contains(&(1, 1, 1)));
    }

    #[test]
    fn filter_shell_keeps_single() {
        let mut s = AHashSet::new();
        s.insert((0, 0, 0));
        let sh = filter_voxel_set_to_shell(&s);
        assert_eq!(sh.len(), 1);
    }
}
