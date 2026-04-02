//! Shared math primitives for all face-placement generators.
//!
//! Standardises on `V3 = [f32; 3]` arrays, a Mulberry32 RNG, and a
//! `PlacementFrame` that every generator needs to build from a face normal.

// Several items here are part of the API for generators to adopt incrementally;
// suppress dead_code warnings until all generators are migrated.
#![allow(dead_code)]

// ---------------------------------------------------------------------------
// V3 type and vector helpers
// ---------------------------------------------------------------------------

pub type V3 = [f32; 3];

#[inline(always)]
pub fn v3_add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline(always)]
pub fn v3_sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline(always)]
pub fn v3_scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline(always)]
pub fn v3_dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline(always)]
pub fn v3_cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline(always)]
pub fn v3_len(a: V3) -> f32 {
    v3_dot(a, a).sqrt()
}

#[inline(always)]
pub fn v3_normalize(a: V3) -> V3 {
    let l = v3_len(a);
    if l < 1e-9 {
        [0.0, 1.0, 0.0]
    } else {
        v3_scale(a, 1.0 / l)
    }
}

#[inline(always)]
pub fn v3_lerp(a: V3, b: V3, t: f32) -> V3 {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

#[inline(always)]
pub fn v3_round(a: V3) -> (i32, i32, i32) {
    (
        a[0].round() as i32,
        a[1].round() as i32,
        a[2].round() as i32,
    )
}

/// Rodrigues rotation: rotate `v` around unit `axis` by `angle` radians.
#[inline(always)]
pub fn v3_rotate_around(v: V3, axis: V3, angle: f32) -> V3 {
    let c = angle.cos();
    let s = angle.sin();
    let d = v3_dot(axis, v);
    let cr = v3_cross(axis, v);
    [
        v[0] * c + cr[0] * s + axis[0] * d * (1.0 - c),
        v[1] * c + cr[1] * s + axis[1] * d * (1.0 - c),
        v[2] * c + cr[2] * s + axis[2] * d * (1.0 - c),
    ]
}

// ---------------------------------------------------------------------------
// Scalar helpers
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline(always)]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Golden angle in radians: PI * (3 − √5).  Used for evenly-distributed
/// spiral distributions (branches, canopy scatter, etc.).
pub const GOLDEN_ANGLE_RAD: f32 = std::f32::consts::PI * (3.0 - 2.2360679_f32);

// ---------------------------------------------------------------------------
// Deterministic spatial hash
// ---------------------------------------------------------------------------

/// Returns a pseudo-random float in [0, 1) from three integer coordinates and
/// a seed.  Used for per-voxel jitter without an RNG state.
#[inline]
pub fn hash3(x: i32, y: i32, z: i32, seed: i32) -> f32 {
    let h = (x.wrapping_mul(73_856_093) as i64)
        ^ (y.wrapping_mul(19_349_663) as i64)
        ^ (z.wrapping_mul(83_492_791) as i64)
        ^ ((seed as i64) << 20);
    let u = (h as u64).wrapping_mul(6_364_136_223_846_793_005);
    (u as f32) / (u64::MAX as f32)
}

// ---------------------------------------------------------------------------
// Mulberry32 seeded RNG
// ---------------------------------------------------------------------------

pub struct Rng {
    state: u32,
}

impl Rng {
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    /// Returns a value in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x6D2B79F5);
        let mut t = (self.state as u64).wrapping_mul((self.state ^ (self.state >> 15)) as u64);
        t = (t & 0xFFFF_FFFF) ^ (t >> 16);
        (t as u32 as f32) / (u32::MAX as f32)
    }

    /// Returns a value in [0, 1) as f64.  Flora uses this for high-precision
    /// braid / wobble paths.
    pub fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x6D2B79F5);
        let mut t = (self.state as u64).wrapping_mul((self.state ^ (self.state >> 15)) as u64);
        t = (t & 0xFFFF_FFFF) ^ (t >> 16);
        (t as u32 as f64) / (u32::MAX as f64)
    }

    /// Returns a value in [-1, 1).
    pub fn next_signed_f32(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }

    /// Returns a value in [-1, 1) as f64.
    pub fn next_signed_f64(&mut self) -> f64 {
        self.next_f64() * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// PlacementFrame — orthonormal frame built from a face normal
// ---------------------------------------------------------------------------

/// A right-handed orthonormal frame anchored at a face-empty voxel center.
///
/// - `up`      — outward face normal (the direction things grow / stand).
/// - `forward` — in-plane, primary orientation direction (head→tail for
///               creatures, stem growth direction for flora).
/// - `side`    — in-plane, lateral (right-hand side).
/// - `origin`  — world-space anchor in float coordinates.
#[derive(Clone, Copy, Debug)]
pub struct PlacementFrame {
    pub origin: V3,
    pub up: V3,
    pub forward: V3,
    pub side: V3,
}

impl PlacementFrame {
    /// Build from an integer face normal `(nx, ny, nz)` (components in {-1, 0, 1}).
    /// `origin` is typically the `face_empty` voxel coordinate cast to f32.
    pub fn from_normal(origin: (i32, i32, i32), nx: i32, ny: i32, nz: i32) -> Self {
        let up = v3_normalize([nx as f32, ny as f32, nz as f32]);

        // Choose a reference vector not parallel to `up`.
        let ref_vec: V3 = if up[1].abs() < 0.9 {
            [0.0, 1.0, 0.0]
        } else {
            [1.0, 0.0, 0.0]
        };

        let side = v3_normalize(v3_cross(up, ref_vec));
        let forward = v3_normalize(v3_cross(side, up));

        Self {
            origin: [origin.0 as f32, origin.1 as f32, origin.2 as f32],
            up,
            forward,
            side,
        }
    }

    /// Rotate `forward` and `side` in the face plane by `yaw_rad`.
    pub fn with_yaw(self, yaw_rad: f32) -> Self {
        Self {
            forward: v3_rotate_around(self.forward, self.up, yaw_rad),
            side: v3_rotate_around(self.side, self.up, yaw_rad),
            ..self
        }
    }

    /// Shift `origin` by `u` along `forward` and `v` along `side`.
    pub fn with_anchor_offset(self, u: f32, v: f32) -> Self {
        Self {
            origin: v3_add(
                self.origin,
                v3_add(v3_scale(self.forward, u), v3_scale(self.side, v)),
            ),
            ..self
        }
    }

    /// Express a local `[forward, side, up]` offset as a world-space position.
    #[inline(always)]
    pub fn local_to_world(&self, local: V3) -> V3 {
        v3_add(
            self.origin,
            v3_add(
                v3_scale(self.forward, local[0]),
                v3_add(v3_scale(self.side, local[1]), v3_scale(self.up, local[2])),
            ),
        )
    }
}
