//! Shared math primitives for all face-placement generators.
//!
//! Standardises on `V3 = [f32; 3]` arrays, a Mulberry32 RNG, and a
//! `PlacementFrame` that every generator needs to build from a face normal.

// Several items here are part of the API for generators to adopt incrementally;
// suppress dead_code warnings until all generators are migrated.
#![allow(dead_code)]

use glam::Vec3;

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

#[inline(always)]
pub fn ray_plane_intersect(ro: Vec3, rd: Vec3, plane_n: Vec3, plane_p: Vec3) -> Option<Vec3> {
    let denom = rd.dot(plane_n);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (plane_p - ro).dot(plane_n) / denom;
    if t < 0.0 {
        return None;
    }
    Some(ro + rd * t)
}

/// Returns the scalar `t` on the infinite line `axis_origin + axis_dir * t`
/// that best matches the cursor ray. Falls back to projecting the cursor hit
/// on a camera-facing plane onto the axis when the ray and axis are nearly
/// parallel.
#[inline(always)]
pub fn axis_drag_scalar_from_ray(
    axis_origin: Vec3,
    axis_dir: Vec3,
    ray_origin: Vec3,
    ray_dir: Vec3,
    fallback_plane_n: Vec3,
    fallback_plane_p: Vec3,
) -> Option<f32> {
    let axis_dir = axis_dir.normalize_or_zero();
    let ray_dir = ray_dir.normalize_or_zero();
    if axis_dir.length_squared() <= 1e-12 || ray_dir.length_squared() <= 1e-12 {
        return None;
    }

    let w0 = axis_origin - ray_origin;
    let b = axis_dir.dot(ray_dir);
    let d = axis_dir.dot(w0);
    let e = ray_dir.dot(w0);
    let denom = 1.0 - b * b;

    if denom.abs() > 1e-5 {
        let ray_t = (e - b * d) / denom;
        if ray_t >= 0.0 {
            return Some((b * e - d) / denom);
        }
    }

    let hit = ray_plane_intersect(ray_origin, ray_dir, fallback_plane_n, fallback_plane_p)?;
    Some((hit - axis_origin).dot(axis_dir))
}

/// Golden angle in radians: PI * (3 − √5).  Used for evenly-distributed
/// spiral distributions (branches, canopy scatter, etc.).
pub const GOLDEN_ANGLE_RAD: f32 = std::f32::consts::PI * (3.0 - 2.236_068_f32);

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI};

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn v3_approx_eq(a: V3, b: V3) -> bool {
        approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2])
    }

    // ── v3 arithmetic ──────────────────────────────────────────────────

    #[test]
    fn v3_add_basic() {
        assert!(v3_approx_eq(
            v3_add([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
            [5.0, 7.0, 9.0]
        ));
    }

    #[test]
    fn v3_sub_basic() {
        assert!(v3_approx_eq(
            v3_sub([4.0, 5.0, 6.0], [1.0, 2.0, 3.0]),
            [3.0, 3.0, 3.0]
        ));
    }

    #[test]
    fn v3_scale_basic() {
        assert!(v3_approx_eq(
            v3_scale([1.0, 2.0, 3.0], 2.0),
            [2.0, 4.0, 6.0]
        ));
    }

    #[test]
    fn v3_dot_perpendicular_is_zero() {
        assert!(approx_eq(v3_dot([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), 0.0));
    }

    #[test]
    fn v3_dot_parallel_is_length_product() {
        assert!(approx_eq(v3_dot([3.0, 0.0, 0.0], [5.0, 0.0, 0.0]), 15.0));
    }

    #[test]
    fn v3_cross_unit_axes() {
        // X × Y = Z
        assert!(v3_approx_eq(
            v3_cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            [0.0, 0.0, 1.0]
        ));
        // Y × Z = X
        assert!(v3_approx_eq(
            v3_cross([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            [1.0, 0.0, 0.0]
        ));
    }

    #[test]
    fn v3_cross_parallel_is_zero() {
        let z = v3_cross([2.0, 0.0, 0.0], [3.0, 0.0, 0.0]);
        assert!(v3_approx_eq(z, [0.0, 0.0, 0.0]));
    }

    #[test]
    fn v3_len_pythagorean() {
        assert!(approx_eq(v3_len([3.0, 4.0, 0.0]), 5.0));
    }

    #[test]
    fn v3_normalize_produces_unit_vector() {
        let n = v3_normalize([3.0, 4.0, 0.0]);
        assert!(approx_eq(v3_len(n), 1.0));
        assert!(approx_eq(n[0], 0.6));
        assert!(approx_eq(n[1], 0.8));
    }

    #[test]
    fn v3_normalize_zero_returns_fallback() {
        let n = v3_normalize([0.0, 0.0, 0.0]);
        // Fallback is [0, 1, 0]
        assert!(v3_approx_eq(n, [0.0, 1.0, 0.0]));
    }

    #[test]
    fn v3_lerp_midpoint() {
        let mid = v3_lerp([0.0, 0.0, 0.0], [2.0, 4.0, 6.0], 0.5);
        assert!(v3_approx_eq(mid, [1.0, 2.0, 3.0]));
    }

    #[test]
    fn v3_lerp_endpoints() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert!(v3_approx_eq(v3_lerp(a, b, 0.0), a));
        assert!(v3_approx_eq(v3_lerp(a, b, 1.0), b));
    }

    #[test]
    fn v3_round_basic() {
        // -0.4 rounds toward zero → 0; 2.6 rounds up → 3
        assert_eq!(v3_round([1.4, 2.6, -0.4]), (1, 3, 0));
    }

    #[test]
    fn v3_rotate_around_90_degrees() {
        // Rotate [1, 0, 0] around Z by 90° → should be near [0, 1, 0].
        let result = v3_rotate_around([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], FRAC_PI_2);
        assert!(v3_approx_eq(result, [0.0, 1.0, 0.0]));
    }

    #[test]
    fn v3_rotate_around_preserves_length() {
        let v = [1.0, 2.0, 3.0];
        let axis = v3_normalize([1.0, 1.0, 0.0]);
        let rotated = v3_rotate_around(v, axis, PI / 3.0);
        assert!(approx_eq(v3_len(rotated), v3_len(v)));
    }

    // ── Scalar helpers ─────────────────────────────────────────────────

    #[test]
    fn smoothstep_at_edges_and_midpoint() {
        assert!(approx_eq(smoothstep(0.0, 1.0, 0.0), 0.0));
        assert!(approx_eq(smoothstep(0.0, 1.0, 1.0), 1.0));
        // At t=0.5: smoothstep = 3*0.25 - 2*0.125 = 0.75 - 0.25 = 0.5
        assert!(approx_eq(smoothstep(0.0, 1.0, 0.5), 0.5));
    }

    #[test]
    fn smoothstep_clamps_outside_range() {
        assert!(approx_eq(smoothstep(0.0, 1.0, -1.0), 0.0));
        assert!(approx_eq(smoothstep(0.0, 1.0, 2.0), 1.0));
    }

    #[test]
    fn lerp_basic() {
        assert!(approx_eq(lerp(0.0, 10.0, 0.5), 5.0));
        assert!(approx_eq(lerp(0.0, 10.0, 0.0), 0.0));
        assert!(approx_eq(lerp(0.0, 10.0, 1.0), 10.0));
    }

    // ── hash3 ──────────────────────────────────────────────────────────

    #[test]
    fn hash3_is_deterministic() {
        assert_eq!(hash3(1, 2, 3, 42), hash3(1, 2, 3, 42));
    }

    #[test]
    fn hash3_different_coords_differ() {
        assert_ne!(hash3(0, 0, 0, 0), hash3(1, 0, 0, 0));
        assert_ne!(hash3(0, 0, 0, 0), hash3(0, 1, 0, 0));
    }

    #[test]
    fn hash3_different_seeds_differ() {
        assert_ne!(hash3(0, 0, 0, 0), hash3(0, 0, 0, 1));
    }

    // ── Rng ───────────────────────────────────────────────────────────

    #[test]
    fn rng_is_deterministic() {
        let v1 = Rng::new(42).next_f32();
        let v2 = Rng::new(42).next_f32();
        assert_eq!(v1, v2);
    }

    #[test]
    fn rng_different_seeds_differ() {
        assert_ne!(Rng::new(0).next_f32(), Rng::new(1).next_f32());
    }

    #[test]
    fn rng_f32_in_range() {
        let mut rng = Rng::new(99);
        for _ in 0..100 {
            let v = rng.next_f32();
            assert!(v >= 0.0 && v < 1.0, "next_f32 out of [0,1): {v}");
        }
    }

    #[test]
    fn rng_signed_f32_in_range() {
        let mut rng = Rng::new(7);
        for _ in 0..100 {
            let v = rng.next_signed_f32();
            assert!(v >= -1.0 && v < 1.0, "next_signed_f32 out of [-1,1): {v}");
        }
    }

    // ── PlacementFrame ────────────────────────────────────────────────

    #[test]
    fn placement_frame_axes_are_orthonormal() {
        let f = PlacementFrame::from_normal((0, 0, 0), 0, 1, 0);
        assert!(approx_eq(v3_len(f.up), 1.0));
        assert!(approx_eq(v3_len(f.forward), 1.0));
        assert!(approx_eq(v3_len(f.side), 1.0));
        // Mutual orthogonality
        assert!(approx_eq(v3_dot(f.up, f.forward), 0.0));
        assert!(approx_eq(v3_dot(f.up, f.side), 0.0));
        assert!(approx_eq(v3_dot(f.forward, f.side), 0.0));
    }

    #[test]
    fn placement_frame_x_normal_orthonormal() {
        let f = PlacementFrame::from_normal((5, 5, 5), 1, 0, 0);
        assert!(approx_eq(v3_dot(f.up, f.forward), 0.0));
        assert!(approx_eq(v3_dot(f.up, f.side), 0.0));
    }

    #[test]
    fn placement_frame_local_to_world_origin_is_origin() {
        let f = PlacementFrame::from_normal((3, 7, 2), 0, 1, 0);
        let world = f.local_to_world([0.0, 0.0, 0.0]);
        assert!(v3_approx_eq(world, f.origin));
    }

    #[test]
    fn placement_frame_local_to_world_forward_offset() {
        let f = PlacementFrame::from_normal((0, 0, 0), 0, 1, 0);
        let world = f.local_to_world([1.0, 0.0, 0.0]);
        let expected = v3_add(f.origin, f.forward);
        assert!(v3_approx_eq(world, expected));
    }
}

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
///   creatures, stem growth direction for flora).
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
