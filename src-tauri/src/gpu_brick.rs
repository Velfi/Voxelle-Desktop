//! Compact GPU voxel brick: one `u32` per cell (RGB8 + material + occupied bit).

use crate::voxelle::{MaterialId, Voxel};
use glam::IVec3;

/// Bit layout: occupied (31), reserved (27-30), mat (24-26), B (16-23), G (8-15), R (0-7).
#[inline]
pub fn pack_cell(rgb: u32, material: MaterialId) -> u32 {
    let r = (rgb >> 16) & 0xff;
    let g = (rgb >> 8) & 0xff;
    let b = rgb & 0xff;
    let mat = material_to_u3(material);
    r | (g << 8) | (b << 16) | (mat << 24) | (1u32 << 31)
}

#[inline]
pub fn pack_empty() -> u32 {
    0u32
}

#[inline]
fn material_to_u3(m: MaterialId) -> u32 {
    match m {
        MaterialId::Plastic => 0,
        MaterialId::Metal => 1,
        MaterialId::Rubber => 2,
        MaterialId::Glass => 3,
        MaterialId::Water => 4,
        MaterialId::Glow => 5,
    }
}

/// Axis-aligned dense brick for shader `textureLoad`-style indexing.
pub struct GpuVoxelBrick {
    pub origin: IVec3,
    pub dims: (u32, u32, u32),
    /// Row-major: x fastest, then y, then z.
    pub cells: Vec<u32>,
}

/// Tight brick layout (origin + dimensions) without allocating `cells`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrickLayout {
    pub origin: IVec3,
    pub dims: (u32, u32, u32),
}

/// Single-cell GPU brick update (`packed` from [`pack_cell`] / [`pack_empty`]).
#[derive(Clone, Copy, Debug)]
pub struct BrickCellWrite {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub packed: u32,
}

impl BrickLayout {
    /// Linear index into row-major `cells` for world integer coords, or `None` if outside dims.
    #[inline]
    pub fn index_of_world(self, x: i32, y: i32, z: i32) -> Option<usize> {
        let ix = x - self.origin.x;
        let iy = y - self.origin.y;
        let iz = z - self.origin.z;
        if ix < 0 || iy < 0 || iz < 0 {
            return None;
        }
        let (sx, sy, sz) = self.dims;
        let ux = ix as u32;
        let uy = iy as u32;
        let uz = iz as u32;
        if ux >= sx || uy >= sy || uz >= sz {
            return None;
        }
        Some(
            (ux as usize)
                + (uy as usize) * (sx as usize)
                + (uz as usize) * (sx as usize) * (sy as usize),
        )
    }
}

impl GpuVoxelBrick {
    /// Min/max and capped dimensions (same as [`Self::from_voxels`] but no allocation).
    pub fn layout_from_voxels(voxels: &[Voxel], max_axis: u32) -> Option<BrickLayout> {
        if voxels.is_empty() {
            return None;
        }
        let mut min = IVec3::splat(i32::MAX);
        let mut max = IVec3::splat(i32::MIN);
        for v in voxels {
            let p = IVec3::new(v.x, v.y, v.z);
            min = min.min(p);
            max = max.max(p);
        }
        let mut sx = (max.x - min.x + 1).max(1) as u32;
        let mut sy = (max.y - min.y + 1).max(1) as u32;
        let mut sz = (max.z - min.z + 1).max(1) as u32;
        sx = sx.min(max_axis);
        sy = sy.min(max_axis);
        sz = sz.min(max_axis);
        Some(BrickLayout {
            origin: min,
            dims: (sx, sy, sz),
        })
    }

    /// Build from voxel list. Returns `None` if empty. Caps each axis to `max_axis` to bound VRAM.
    pub fn from_voxels(voxels: &[Voxel], max_axis: u32) -> Option<Self> {
        let layout = Self::layout_from_voxels(voxels, max_axis)?;
        let (sx, sy, sz) = layout.dims;
        let min = layout.origin;

        let n = (sx as usize)
            .saturating_mul(sy as usize)
            .saturating_mul(sz as usize);
        let mut cells = vec![pack_empty(); n];

        for v in voxels {
            let ix = (v.x - min.x) as i64;
            let iy = (v.y - min.y) as i64;
            let iz = (v.z - min.z) as i64;
            if ix < 0 || iy < 0 || iz < 0 {
                continue;
            }
            let ux = ix as u32;
            let uy = iy as u32;
            let uz = iz as u32;
            if ux >= sx || uy >= sy || uz >= sz {
                continue;
            }
            let idx = (ux as usize)
                + (uy as usize) * (sx as usize)
                + (uz as usize) * (sx as usize) * (sy as usize);
            if idx < cells.len() {
                cells[idx] = pack_cell(v.color, v.material);
            }
        }

        Some(Self {
            origin: min,
            dims: (sx, sy, sz),
            cells,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxelle::Voxel;

    #[test]
    fn roundtrip_pack_non_empty() {
        // 0xRRGGBB → r in bits 0–7, g in 8–15, b in 16–23 (matches WGSL unpack).
        let c = pack_cell(0xff8040, MaterialId::Glass);
        assert!(c & (1 << 31) != 0);
        assert_eq!(c & 0xff, 0xff);
        assert_eq!((c >> 8) & 0xff, 0x80);
        assert_eq!((c >> 16) & 0xff, 0x40);
        assert_eq!((c >> 24) & 7, 3);
    }

    #[test]
    fn brick_single_voxel() {
        let v = Voxel {
            x: 0,
            y: 0,
            z: 0,
            color: 0x112233,
            material: MaterialId::Plastic,
        };
        let b = GpuVoxelBrick::from_voxels(&[v], 256).unwrap();
        assert_eq!(b.origin, IVec3::ZERO);
        assert_eq!(b.dims, (1, 1, 1));
        assert_ne!(b.cells[0], 0);
    }
}
