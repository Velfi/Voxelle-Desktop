//! Greedy meshing with Minecraft-style per-corner vertex AO — face visibility rules from Voxelle `greedyMeshCore`.

use crate::gpu_brick::{pack_cell, pack_empty};
use crate::voxelle::{MaterialId, SceneObject, Voxel};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};

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

/// Mirrors `isFaceOccludedByNeighbor` in greedyMeshCore.ts — `true` = face not emitted.
fn face_occluded(source: MaterialId, neighbor: MaterialId) -> bool {
    if is_transmissive(source) {
        source == neighbor
    } else {
        !is_transmissive(neighbor)
    }
}

fn neighbor_occludes_face(
    map: &HashMap<IVec3, Voxel>,
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

#[inline]
fn cell_same_object(map: &HashMap<VoxelCoord, Voxel>, object_id: u32, pos: IVec3) -> bool {
    map.get(&pos)
        .map(|v| v.object_id == object_id)
        .unwrap_or(false)
}

/// Minecraft-style corner AO: count solid neighbors among the three voxels meeting at this face corner.
fn corner_ao_factor(
    map: &HashMap<VoxelCoord, Voxel>,
    object_id: u32,
    axis: usize,
    depth: i32,
    cu: i32,
    cv: i32,
) -> f32 {
    let occ = match axis {
        0 => {
            let a = cell_same_object(map, object_id, (depth, cu - 1, cv));
            let b = cell_same_object(map, object_id, (depth, cu, cv - 1));
            let c = cell_same_object(map, object_id, (depth, cu - 1, cv - 1));
            u32::from(a) + u32::from(b) + u32::from(c)
        }
        1 => {
            let a = cell_same_object(map, object_id, (cu - 1, depth, cv));
            let b = cell_same_object(map, object_id, (cu, depth, cv - 1));
            let c = cell_same_object(map, object_id, (cu - 1, depth, cv - 1));
            u32::from(a) + u32::from(b) + u32::from(c)
        }
        _ => {
            let a = cell_same_object(map, object_id, (cu - 1, cv, depth));
            let b = cell_same_object(map, object_id, (cu, cv - 1, depth));
            let c = cell_same_object(map, object_id, (cu - 1, cv - 1, depth));
            u32::from(a) + u32::from(b) + u32::from(c)
        }
    };
    (1.0 - 0.2 * occ as f32).clamp(0.4, 1.0)
}

fn greedy_merge(cells: &[(i32, i32)]) -> Vec<(i32, i32, i32, i32)> {
    let n = cells.len().max(1);
    let set: HashSet<(i32, i32)> = cells.iter().copied().collect();
    let mut consumed = HashSet::with_capacity(n);
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
}

#[derive(Clone, Debug, Default)]
pub struct MeshBuffers {
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    pub colors: Vec<f32>,
    pub mat_kind: Vec<f32>,
    /// Per-vertex ambient factor [~0.4, 1.0] from 3-neighbor corner occlusion (hemisphere term only in shader).
    pub ao: Vec<f32>,
    pub indices: Vec<u32>,
}

/// Padded brick (+1 voxel halo) for GPU mesh AO: same row-major layout as [`crate::gpu_brick::GpuVoxelBrick`],
/// with origin shifted by −1 and dims +2, filled from `map` so in-plane neighbor checks see voxels outside the tight brick.
pub fn pack_brick_halo_cells(
    map: &HashMap<VoxelCoord, Voxel>,
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

pub fn voxel_map(voxels: &[Voxel]) -> HashMap<VoxelCoord, Voxel> {
    let mut map = HashMap::with_capacity(voxels.len());
    for v in voxels {
        map.insert(coord_key(v.x, v.y, v.z), *v);
    }
    map
}

/// Spatial index for raycasts / swap-remove: coord → index in `VoxelleFile::voxels`.
pub fn voxel_map_indices(voxels: &[Voxel]) -> HashMap<VoxelCoord, usize> {
    let mut map = HashMap::with_capacity(voxels.len());
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
    map: &HashMap<VoxelCoord, Voxel>,
    emit: &[Voxel],
) -> Result<(Vec<GpuSliceHeader>, Vec<u32>), ()> {
    let mut buckets: HashMap<(u32, u8), Vec<IVec3>> = HashMap::new();
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
        let mat_k = match vx.material {
            MaterialId::Glow => 1.0,
            MaterialId::Glass | MaterialId::Water => 2.0,
            _ => 0.0,
        };

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

        let mut slices: HashMap<GreedySliceKey, Vec<(i32, i32)>> = HashMap::new();
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
) -> Option<((i32, i32, i32), HashMap<ChunkKey, Vec<Voxel>>)> {
    let origin = voxel_aabb_min_int(voxels)?;
    let cs = cs.max(1);
    let mut buckets: HashMap<ChunkKey, Vec<Voxel>> = HashMap::new();
    for v in voxels {
        let k = chunk_key_from_world(v.x, v.y, v.z, origin, cs);
        buckets.entry(k).or_default().push(*v);
    }
    Some((origin, buckets))
}

/// Greedy mesh for one chunk’s **core** voxels, with neighbor occlusion from `map` (full scene).
pub fn mesh_buffers_for_chunk_key(
    buckets: &HashMap<ChunkKey, Vec<Voxel>>,
    map: &HashMap<VoxelCoord, Voxel>,
    key: ChunkKey,
) -> MeshBuffers {
    let core = buckets.get(&key).cloned().unwrap_or_default();
    if core.is_empty() {
        return MeshBuffers::default();
    }
    build_greedy_mesh_mapped(&core, map)
}

/// Single pass: [`SpatialMeshCache`] plus per-chunk greedy meshes (one `voxel_map` + bucketing).
/// Chunk meshes build in parallel across chunks.
///
/// `progress` is called from worker threads with values in \([0, 1]\) as chunks complete (throttled).
pub fn build_chunk_meshes_and_spatial_cache<F>(voxels: &[Voxel], cs: i32, progress: F) -> Option<(
    (i32, i32, i32),
    BTreeMap<ChunkKey, MeshBuffers>,
    SpatialMeshCache,
)>
where
    F: Fn(f32) + Sync,
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
                progress(frac);
            }
            (!mesh.indices.is_empty()).then_some((key, mesh))
        })
        .collect();
    progress(1.0);
    let meshes: BTreeMap<ChunkKey, MeshBuffers> = parts.into_iter().collect();
    Some((origin, meshes, cache))
}

/// Build per-chunk meshes (for GPU upload / incremental updates). Skips empty outputs.
pub fn build_all_chunk_meshes_btree(
    voxels: &[Voxel],
    cs: i32,
) -> Option<((i32, i32, i32), BTreeMap<ChunkKey, MeshBuffers>)> {
    let (origin, meshes, _) = build_chunk_meshes_and_spatial_cache(voxels, cs, |_| {})?;
    Some((origin, meshes))
}

/// Full occupancy map + spatial buckets for incremental edits (O(1) add/remove vs full rescans).
#[derive(Clone, Debug)]
pub struct SpatialMeshCache {
    pub origin: (i32, i32, i32),
    pub occupancy: HashMap<VoxelCoord, Voxel>,
    pub buckets: HashMap<ChunkKey, Vec<Voxel>>,
}

impl SpatialMeshCache {
    pub fn from_voxels(voxels: &[Voxel], cs: i32) -> Option<Self> {
        let origin = voxel_aabb_min_int(voxels)?;
        let cs = cs.max(1);
        let occupancy = voxel_map(voxels);
        let mut buckets: HashMap<ChunkKey, Vec<Voxel>> = HashMap::new();
        for v in voxels {
            let k = chunk_key_from_world(v.x, v.y, v.z, origin, cs);
            buckets.entry(k).or_default().push(*v);
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
        self.buckets.entry(k).or_default().push(v);
    }

    pub fn apply_remove(&mut self, x: i32, y: i32, z: i32, cs: i32) {
        let cs = cs.max(1);
        let coord = (x, y, z);
        self.occupancy.remove(&coord);
        let k = chunk_key_from_world(x, y, z, self.origin, cs);
        if let Some(vec) = self.buckets.get_mut(&k) {
            if let Some(i) = vec.iter().position(|v| v.x == x && v.y == y && v.z == z) {
                vec.swap_remove(i);
            }
            if vec.is_empty() {
                self.buckets.remove(&k);
            }
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

    let mut buckets: HashMap<(i32, i32, i32), Vec<Voxel>> = HashMap::new();
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
        let fused = build_chunk_meshes_and_spatial_cache(&voxels, cs, |_| {}).expect("fused");
        let seq = sequential_chunk_meshes_and_spatial_cache(&voxels, cs).expect("sequential");
        assert_eq!(fused.0, seq.0, "chunk origin");
        assert_eq!(fused.1.len(), seq.1.len(), "chunk count");
        for (k, m) in &fused.1 {
            let m2 = seq.1.get(k).expect("missing chunk");
            assert_eq!(m.indices.len(), m2.indices.len(), "indices {:?}", k);
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
        assert_eq!(cache.buckets.len(), expected.buckets.len());
        for k in expected.buckets.keys() {
            let mut a = cache.buckets.get(k).cloned().unwrap_or_default();
            let mut b = expected.buckets.get(k).cloned().unwrap_or_default();
            a.sort_by_key(|v| (v.x, v.y, v.z));
            b.sort_by_key(|v| (v.x, v.y, v.z));
            assert_eq!(a, b, "bucket {k:?}");
        }
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
/// `mat_kind` per vertex: 0 solid, 1 glow, 2 glass/water (shader uses for emissive / spec).
pub fn build_greedy_mesh_mapped(emit: &[Voxel], map: &HashMap<VoxelCoord, Voxel>) -> MeshBuffers {
    let mut buckets: HashMap<(u32, u8), Vec<IVec3>> = HashMap::new();
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
        let mat_k = match vx.material {
            MaterialId::Glow => 1.0,
            MaterialId::Glass | MaterialId::Water => 2.0,
            _ => 0.0,
        };

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

        let mut slices: HashMap<GreedySliceKey, Vec<(i32, i32)>> = HashMap::new();
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

                let id00 = map[&grid_pos(axis, depth, u, v)].object_id;
                let id10 = map[&grid_pos(axis, depth, u + w - 1, v)].object_id;
                let id11 = map[&grid_pos(axis, depth, u + w - 1, v + h - 1)].object_id;
                let id01 = map[&grid_pos(axis, depth, u, v + h - 1)].object_id;
                let ao00 = corner_ao_factor(map, id00, axis, depth, u, v);
                let ao10 = corner_ao_factor(map, id10, axis, depth, u + w - 1, v);
                let ao11 = corner_ao_factor(map, id11, axis, depth, u + w - 1, v + h - 1);
                let ao01 = corner_ao_factor(map, id01, axis, depth, u, v + h - 1);
                let ao_face = (ao00 + ao10 + ao11 + ao01) * 0.25;

                let base = (out.positions.len() / 3) as u32;
                for (p, ao_v) in [
                    (p00, ao_face),
                    (p10, ao_face),
                    (p11, ao_face),
                    (p01, ao_face),
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

pub fn transform_mesh_buffers(mesh: &mut MeshBuffers, model: glam::Mat4) {
    let inv_t = model.inverse().transpose();
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
    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
    let mut any = false;
    for v in voxels {
        if !crate::voxelle::scene::is_object_visible(objects, v.object_id) {
            continue;
        }
        any = true;
        let m = crate::voxelle::object_world_matrix(objects, v.object_id);
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
    map: &HashMap<VoxelCoord, Voxel>,
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

/// `mat_kind` per vertex: 0 solid, 1 glow, 2 glass/water (shader uses for emissive / spec).
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
}
