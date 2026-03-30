//! Greedy meshing (no AO) — face visibility rules from Voxelle `greedyMeshCore`.

use crate::voxelle::{MaterialId, Voxel};
use std::collections::{HashMap, HashSet};

/// Integer voxel coordinate key for maps and meshing.
pub type VoxelCoord = (i32, i32, i32);

type IVec3 = VoxelCoord;

fn coord_key(x: i32, y: i32, z: i32) -> IVec3 {
    (x, y, z)
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

fn neighbor_occludes_face(map: &HashMap<IVec3, Voxel>, pos: IVec3, axis: usize, sign: i32, src: MaterialId) -> bool {
    let (x, y, z) = pos;
    let (nx, ny, nz) = match axis {
        0 => (x + sign, y, z),
        1 => (x, y + sign, z),
        _ => (x, y, z + sign),
    };
    let Some(neigh) = map.get(&coord_key(nx, ny, nz)) else {
        return false;
    };
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

fn bucket_key(v: &Voxel) -> String {
    format!("{}|{}", v.color, material_tag(v.material))
}

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

fn greedy_merge(cells: &[(i32, i32)]) -> Vec<(i32, i32, i32, i32)> {
    let set: HashSet<String> = cells.iter().map(|(u, v)| format!("{u},{v}")).collect();
    let mut consumed = HashSet::new();
    let mut quads = Vec::new();
    for &(u, v) in cells {
        let k = format!("{u},{v}");
        if consumed.contains(&k) {
            continue;
        }
        let mut w = 1_i32;
        while set.contains(&format!("{},{}", u + w, v)) && !consumed.contains(&format!("{},{}", u + w, v)) {
            w += 1;
        }
        let mut h = 1_i32;
        'rows: loop {
            for i in 0..w {
                let kk = format!("{},{}", u + i, v + h);
                if !set.contains(&kk) || consumed.contains(&kk) {
                    break 'rows;
                }
            }
            h += 1;
        }
        for dv in 0..h {
            for du in 0..w {
                consumed.insert(format!("{},{}", u + du, v + dv));
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

#[derive(Clone, Debug, Default)]
pub struct MeshBuffers {
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    pub colors: Vec<f32>,
    pub mat_kind: Vec<f32>,
    pub indices: Vec<u32>,
}

pub fn voxel_map(voxels: &[Voxel]) -> HashMap<VoxelCoord, Voxel> {
    let mut map = HashMap::with_capacity(voxels.len());
    for v in voxels {
        map.insert(coord_key(v.x, v.y, v.z), *v);
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

/// Pack coplanar face cells into 2D bitmaps for GPU greedy meshing (max 64×64 per slice).
/// Returns `Err` if any slice exceeds the GPU limit (caller should use CPU [`build_greedy_mesh`]).
pub fn pack_gpu_greedy_slices(map: &HashMap<VoxelCoord, Voxel>, emit: &[Voxel]) -> Result<(Vec<GpuSliceHeader>, Vec<u32>), ()> {
    let mut buckets: HashMap<String, Vec<IVec3>> = HashMap::new();
    for v in emit {
        buckets.entry(bucket_key(v)).or_default().push(coord_key(v.x, v.y, v.z));
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

        let mut faces: Vec<(IVec3, usize, i32)> = Vec::new();
        for &pos in cell_positions {
            let source = map[&pos];
            for i in 0..6usize {
                let axis = i / 2;
                let sign = if i % 2 == 0 { 1 } else { -1 };
                if !neighbor_occludes_face(map, pos, axis, sign, source.material) {
                    faces.push((pos, axis, sign));
                }
            }
        }

        let mut slices: HashMap<String, Vec<(i32, i32)>> = HashMap::new();
        for (pos, axis, sign) in faces {
            let (x, y, z) = pos;
            let depth = match axis {
                0 => x,
                1 => y,
                _ => z,
            };
            let u = if axis == 0 { y } else { x };
            let v = if axis == 2 { y } else { z };
            let sk = format!("{axis},{sign},{depth}");
            slices.entry(sk).or_default().push((u, v));
        }

        for (sk, cells) in slices {
            let parts: Vec<&str> = sk.split(',').collect();
            let axis: u32 = parts[0].parse().unwrap();
            let sign: i32 = parts[1].parse().unwrap();
            let depth: i32 = parts[2].parse().unwrap();

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
            if width > 64 || height > 64 {
                return Err(());
            }
            let ncells = width as usize * height as usize;
            let bit_word_count = (ncells + 31) / 32;
            let bit_start = all_bits.len() as u32;
            all_bits.resize(all_bits.len() + bit_word_count, 0u32);

            for &(u, v) in &cells {
                let lu = (u - min_u) as u32;
                let lv = (v - min_v) as u32;
                let idx = lu + lv * width;
                let wi = (idx / 32) as usize;
                let bi = idx % 32;
                all_bits[bit_start as usize + wi] |= 1u32 << bi;
            }

            headers.push(GpuSliceHeader {
                axis,
                sign,
                depth,
                color: vx.color,
                mat_kind: mat_k,
                u0: min_u,
                v0: min_v,
                width,
                height,
                bit_start,
                bit_word_count: bit_word_count as u32,
            });
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
#[allow(dead_code)]
pub const VOXEL_CHUNK_SIZE: i32 = 48;

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
    use crate::voxelle::MaterialId;

    #[test]
    fn spatial_chunks_split_distant_voxels() {
        let voxels = vec![
            Voxel {
                x: 0,
                y: 0,
                z: 0,
                color: 1,
                material: MaterialId::Plastic,
            },
            Voxel {
                x: 100,
                y: 0,
                z: 0,
                color: 2,
                material: MaterialId::Plastic,
            },
        ];
        let ch = voxels_by_spatial_chunks(&voxels, 48);
        assert_eq!(ch.len(), 2);
        assert_eq!(ch[0].1.len(), 1);
        assert_eq!(ch[1].1.len(), 1);
    }
}

/// Greedy mesh for `emit` voxels only, using `map` for neighbor occlusion (include a 1-voxel halo around each chunk).
/// `mat_kind` per vertex: 0 solid, 1 glow, 2 glass/water (shader uses for emissive / spec).
pub fn build_greedy_mesh_mapped(emit: &[Voxel], map: &HashMap<VoxelCoord, Voxel>) -> MeshBuffers {
    let mut buckets: HashMap<String, Vec<IVec3>> = HashMap::new();
    for v in emit {
        buckets.entry(bucket_key(v)).or_default().push(coord_key(v.x, v.y, v.z));
    }

    let mut out = MeshBuffers {
        positions: Vec::new(),
        normals: Vec::new(),
        colors: Vec::new(),
        mat_kind: Vec::new(),
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

        let mut faces: Vec<(IVec3, usize, i32)> = Vec::new();
        for &pos in cell_positions {
            let source = map[&pos];
            for i in 0..6usize {
                let axis = i / 2;
                let sign = if i % 2 == 0 { 1 } else { -1 };
                if !neighbor_occludes_face(&map, pos, axis, sign, source.material) {
                    faces.push((pos, axis, sign));
                }
            }
        }

        let mut slices: HashMap<String, Vec<(i32, i32)>> = HashMap::new();
        for (pos, axis, sign) in faces {
            let (x, y, z) = pos;
            let depth = match axis {
                0 => x,
                1 => y,
                _ => z,
            };
            let u = if axis == 0 { y } else { x };
            let v = if axis == 2 { y } else { z };
            let sk = format!("{axis},{sign},{depth}");
            slices.entry(sk).or_default().push((u, v));
        }

        for (sk, cells) in slices {
            let parts: Vec<&str> = sk.split(',').collect();
            let axis: usize = parts[0].parse().unwrap();
            let sign: i32 = parts[1].parse().unwrap();
            let depth: i32 = parts[2].parse().unwrap();
            let merged = greedy_merge(&cells);
            let n = face_normal(axis, sign);

            for (u, v, w, h) in merged {
                let p00 = quad_corner(axis, sign, depth, u, v);
                let p10 = quad_corner(axis, sign, depth, u + w, v);
                let p11 = quad_corner(axis, sign, depth, u + w, v + h);
                let p01 = quad_corner(axis, sign, depth, u, v + h);

                let base = (out.positions.len() / 3) as u32;
                for p in [p00, p10, p11, p01] {
                    out.positions.extend_from_slice(&p.to_array());
                    out.normals.extend_from_slice(&n.to_array());
                    out.colors.extend_from_slice(&col.to_array());
                    out.mat_kind.push(mat_k);
                }

                let ccw = if n.x != 0.0 {
                    n.x > 0.0
                } else if n.y != 0.0 {
                    n.y < 0.0
                } else {
                    n.z > 0.0
                };
                if ccw {
                    out.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                } else {
                    out.indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
                }
            }
        }
    }

    out
}

/// Single axis-aligned cube for add/remove hover preview. `half` is half-edge length (0.5 = unit voxel).
pub fn preview_cube_mesh(cx: f32, cy: f32, cz: f32, half: f32, color: [f32; 3], mat_k: f32) -> MeshBuffers {
    let h = half;
    let mut positions: Vec<f32> = Vec::with_capacity(24 * 3);
    let mut normals: Vec<f32> = Vec::with_capacity(24 * 3);
    let mut colors: Vec<f32> = Vec::with_capacity(24 * 3);
    let mut mat_kind: Vec<f32> = Vec::with_capacity(24);
    let mut indices: Vec<u32> = Vec::with_capacity(36);

    let mut face = |nx: f32, ny: f32, nz: f32, corners: [[f32; 3]; 4]| {
        let base = (positions.len() / 3) as u32;
        for p in corners {
            positions.extend_from_slice(&p);
            normals.extend_from_slice(&[nx, ny, nz]);
            colors.extend_from_slice(&color);
            mat_kind.push(mat_k);
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
        indices,
    }
}

/// `mat_kind` per vertex: 0 solid, 1 glow, 2 glass/water (shader uses for emissive / spec).
pub fn build_greedy_mesh(voxels: &[Voxel]) -> (MeshBuffers, MeshBounds) {
    let map = voxel_map(voxels);
    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
    for v in voxels {
        let pf = glam::Vec3::new(v.x as f32, v.y as f32, v.z as f32);
        min = min.min(pf);
        max = max.max(pf);
    }
    let bounds = MeshBounds { min, max };
    let mesh = build_greedy_mesh_mapped(voxels, &map);
    (mesh, bounds)
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
        }];
        let map = voxel_map(&voxels);
        let (headers, bits) = pack_gpu_greedy_slices(&map, &voxels).expect("pack");
        assert!(!headers.is_empty());
        assert!(!bits.is_empty());
        let (cpu, _) = build_greedy_mesh(&voxels);
        assert!(!cpu.indices.is_empty());
    }
}
