//! Starting voxel layouts — matches `store/shapes.ts` `initShape` (web New project).

use super::format::{MaterialId, Voxel};

/// Upper bound on voxel count when creating a new project (memory / mesh budget).
pub const MAX_NEW_PROJECT_VOXELS: u64 = 50_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StartShape {
    Cube,
    Orb,
    Cylinder,
    HollowCube,
    Plane,
    Circle,
    Empty,
}

fn grid_bounds(size: i32) -> (i32, i32) {
    let lo = -size / 2;
    let hi = (size - 1) / 2;
    (lo, hi)
}

/// Same inclusion rules as `initShape` in `shapes.ts` (default plastic, color `0x888888`).
pub fn voxels_for_start_shape(size: i32, shape: StartShape) -> Result<Vec<Voxel>, String> {
    if size < 1 {
        return Ok(Vec::new());
    }
    if shape == StartShape::Empty {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    let (lo, hi) = grid_bounds(size);
    let r = (size - 1) as f64 * 0.5;
    let r_sq = r * r;

    let color = 0x888888_u32;
    let material = MaterialId::Plastic;
    let vx = |x: i32, y: i32, z: i32| Voxel {
        x,
        y,
        z,
        color,
        material,
        object_id: 0,
    };

    for x in lo..=hi {
        for y in lo..=hi {
            for z in lo..=hi {
                let include = match shape {
                    StartShape::Cube => true,
                    StartShape::Orb => {
                        let xf = x as f64;
                        let yf = y as f64;
                        let zf = z as f64;
                        xf * xf + yf * yf + zf * zf <= r_sq
                    }
                    StartShape::Cylinder => {
                        let xf = x as f64;
                        let zf = z as f64;
                        xf * xf + zf * zf <= r_sq
                    }
                    StartShape::HollowCube => {
                        x == lo || x == hi || y == lo || y == hi || z == lo || z == hi
                    }
                    StartShape::Plane => y == 0,
                    StartShape::Circle => {
                        y == 0 && {
                            let xf = x as f64;
                            let zf = z as f64;
                            xf * xf + zf * zf <= r_sq
                        }
                    }
                    StartShape::Empty => unreachable!("handled above"),
                };
                if include {
                    if out.len() as u64 >= MAX_NEW_PROJECT_VOXELS {
                        return Err(format!(
                            "new project would exceed {} voxels (try a smaller grid or a sparser starting shape).",
                            MAX_NEW_PROJECT_VOXELS
                        ));
                    }
                    out.push(vx(x, y, z));
                }
            }
        }
    }

    Ok(out)
}
