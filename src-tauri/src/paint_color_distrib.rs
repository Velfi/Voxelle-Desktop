//! Per-voxel paint color distribution (multi-color modes).
//! Mirrors web `paintColorDistributionMath.ts` and `valueNoise3d.ts`.

use serde::Deserialize;

// ── Bayer matrices ──────────────────────────────────────────────────────────

const BAYER_2: [u8; 4] = [0, 2, 3, 1];
const BAYER_4: [u8; 16] = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
const BAYER_8: [u8; 64] = [
    0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26, 12, 44, 4, 36, 14, 46, 6, 38, 60,
    28, 52, 20, 62, 30, 54, 22, 3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25, 15,
    47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21,
];

// ── Hash / noise primitives ─────────────────────────────────────────────────

/// Deterministic spatial hash → float in [0, 1). Matches `hash3()` in `valueNoise3d.ts`.
fn hash3_f(seed: u32, x: i32, y: i32, z: i32) -> f32 {
    let mut h = seed
        ^ (x as u32).wrapping_mul(73_856_093)
        ^ (y as u32).wrapping_mul(19_349_663)
        ^ (z as u32).wrapping_mul(83_492_791);
    h = (h ^ (h >> 16)).wrapping_mul(0x85eb_ca6b);
    h = (h ^ (h >> 13)).wrapping_mul(0xc2b2_ae35);
    (h ^ (h >> 16)) as f32 / 4_294_967_296.0
}

/// Deterministic palette index for white-noise mode. Matches `paintColorIndexForCoord()`.
pub fn paint_color_index(x: i32, y: i32, z: i32, palette_size: usize) -> usize {
    if palette_size <= 1 {
        return 0;
    }
    let mut h = (x as u32).wrapping_mul(0x9e37_79b1)
        ^ (y as u32).wrapping_mul(0x85eb_ca6b)
        ^ (z as u32).wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    (h as usize) % palette_size
}

/// Smooth trilinear value noise in [0, 1]. Matches `valueNoise3()`.
fn value_noise3(seed: u32, x: f32, y: f32, z: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let z0 = z.floor() as i32;
    let fx = x - x.floor();
    let fy = y - y.floor();
    let fz = z - z.floor();
    let u = fx * fx * (3.0 - 2.0 * fx);
    let v = fy * fy * (3.0 - 2.0 * fy);
    let w = fz * fz * (3.0 - 2.0 * fz);
    let n000 = hash3_f(seed, x0, y0, z0);
    let n100 = hash3_f(seed, x0 + 1, y0, z0);
    let n010 = hash3_f(seed, x0, y0 + 1, z0);
    let n110 = hash3_f(seed, x0 + 1, y0 + 1, z0);
    let n001 = hash3_f(seed, x0, y0, z0 + 1);
    let n101 = hash3_f(seed, x0 + 1, y0, z0 + 1);
    let n011 = hash3_f(seed, x0, y0 + 1, z0 + 1);
    let n111 = hash3_f(seed, x0 + 1, y0 + 1, z0 + 1);
    let nx00 = n000 * (1.0 - u) + n100 * u;
    let nx10 = n010 * (1.0 - u) + n110 * u;
    let nx01 = n001 * (1.0 - u) + n101 * u;
    let nx11 = n011 * (1.0 - u) + n111 * u;
    let nxy0 = nx00 * (1.0 - v) + nx10 * v;
    let nxy1 = nx01 * (1.0 - v) + nx11 * v;
    nxy0 * (1.0 - w) + nxy1 * w
}

/// Fractional Brownian motion in 3D → [0, 1]. Matches `fbmValue3()`.
fn fbm_value3(
    seed: u32,
    x: f32,
    y: f32,
    z: f32,
    octaves: u32,
    lacunarity: f32,
    persistence: f32,
    frequency: f32,
) -> f32 {
    let n_oct = octaves.clamp(1, 12);
    let mut amp = 1.0_f32;
    let mut freq = frequency.max(1e-6);
    let mut sum = 0.0_f32;
    let mut norm = 0.0_f32;
    for o in 0..n_oct {
        let s = seed.wrapping_add(o.wrapping_mul(0x9e37_79b1));
        sum += amp * value_noise3(s, x * freq, y * freq, z * freq);
        norm += amp;
        amp *= persistence;
        freq *= lacunarity;
    }
    if norm > 0.0 {
        sum / norm
    } else {
        0.0
    }
}

// ── Palette helpers ─────────────────────────────────────────────────────────

/// Linear RGB interpolation along ordered palette; t in [0, 1].
fn lerp_palette_rgb(colors: &[u32], t: f32) -> u32 {
    let n = colors.len();
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return colors[0] & 0x00ff_ffff;
    }
    let t = t.clamp(0.0, 1.0);
    let u = t * (n - 1) as f32;
    let i0 = (u as usize).min(n - 2);
    let f = u - i0 as f32;
    let a = colors[i0] & 0x00ff_ffff;
    let b = colors[i0 + 1] & 0x00ff_ffff;
    let ar = (a >> 16) & 0xff;
    let ag = (a >> 8) & 0xff;
    let ab = a & 0xff;
    let br = (b >> 16) & 0xff;
    let bg = (b >> 8) & 0xff;
    let bb = b & 0xff;
    let r = (ar as f32 + (br as f32 - ar as f32) * f).round() as u32;
    let g = (ag as f32 + (bg as f32 - ag as f32) * f).round() as u32;
    let bl = (ab as f32 + (bb as f32 - ab as f32) * f).round() as u32;
    ((r & 0xff) << 16) | ((g & 0xff) << 8) | (bl & 0xff)
}

/// Quantized palette step: floor(t * n) clamped to n-1.
fn quantize_palette(colors: &[u32], t: f32) -> u32 {
    let n = colors.len();
    if n == 0 {
        return 0;
    }
    let t = t.clamp(0.0, 1.0);
    let idx = ((t * n as f32) as usize).min(n - 1);
    colors[idx] & 0x00ff_ffff
}

fn color_from_t(colors: &[u32], t: f32, quantized: bool) -> u32 {
    if quantized {
        quantize_palette(colors, t)
    } else {
        lerp_palette_rgb(colors, t)
    }
}

// ── Bayer ordered dither ────────────────────────────────────────────────────

fn bayer_threshold(size: u32, x: i32, y: i32) -> f32 {
    let xi = (x.unsigned_abs()) % size;
    let yi = (y.unsigned_abs()) % size;
    let idx = (yi * size + xi) as usize;
    match size {
        2 => BAYER_2[idx] as f32 / 4.0,
        4 => BAYER_4[idx] as f32 / 16.0,
        8 => BAYER_8[idx] as f32 / 64.0,
        _ => 0.0,
    }
}

// ── Serde types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub enum PaintColorMode {
    #[serde(rename = "whiteNoise")]
    WhiteNoise,
    #[serde(rename = "randomSingle")]
    RandomSingle,
    #[serde(rename = "fbmNoise")]
    FbmNoise,
    #[serde(rename = "gradient")]
    Gradient,
    #[serde(rename = "dither")]
    Dither,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FbmParams {
    #[serde(default = "default_octaves")]
    pub octaves: u32,
    #[serde(default = "default_lacunarity")]
    pub lacunarity: f32,
    #[serde(default = "default_persistence")]
    pub persistence: f32,
    #[serde(default = "default_frequency")]
    pub frequency: f32,
    #[serde(default = "default_noise_seed")]
    pub noise_seed: u32,
    #[serde(default)]
    pub quantized: bool,
}

fn default_octaves() -> u32 {
    4
}
fn default_lacunarity() -> f32 {
    2.0
}
fn default_persistence() -> f32 {
    0.5
}
fn default_frequency() -> f32 {
    0.15
}
fn default_noise_seed() -> u32 {
    0x1234_5678
}

impl Default for FbmParams {
    fn default() -> Self {
        Self {
            octaves: default_octaves(),
            lacunarity: default_lacunarity(),
            persistence: default_persistence(),
            frequency: default_frequency(),
            noise_seed: default_noise_seed(),
            quantized: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub enum GradientKind {
    #[serde(rename = "linear")]
    Linear,
    #[serde(rename = "radial")]
    Radial,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GradientParams {
    #[serde(default = "default_gradient_kind")]
    pub kind: GradientKind,
    /// 0 = X, 1 = Y, 2 = Z
    #[serde(default = "default_linear_axis")]
    pub linear_axis: u32,
    #[serde(default = "default_gradient_scale")]
    pub scale: f32,
    #[serde(default)]
    pub phase: f32,
    #[serde(default)]
    pub radial_center: [f32; 3],
    #[serde(default)]
    pub quantized: bool,
}

fn default_gradient_kind() -> GradientKind {
    GradientKind::Linear
}
fn default_linear_axis() -> u32 {
    1
}
fn default_gradient_scale() -> f32 {
    0.08
}

impl Default for GradientParams {
    fn default() -> Self {
        Self {
            kind: default_gradient_kind(),
            linear_axis: default_linear_axis(),
            scale: default_gradient_scale(),
            phase: 0.0,
            radial_center: [0.0, 0.0, 0.0],
            quantized: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DitherParams {
    #[serde(default = "default_ordered_size")]
    pub ordered_size: u32,
    #[serde(default = "default_ordered_strength")]
    pub ordered_strength: f32,
}

fn default_ordered_size() -> u32 {
    4
}
fn default_ordered_strength() -> f32 {
    0.35
}

impl Default for DitherParams {
    fn default() -> Self {
        Self {
            ordered_size: default_ordered_size(),
            ordered_strength: default_ordered_strength(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PaintColorDistrib {
    pub mode: PaintColorMode,
    #[serde(default)]
    pub fbm: FbmParams,
    #[serde(default)]
    pub gradient: GradientParams,
    #[serde(default)]
    pub dither: DitherParams,
}

// ── Resolver ────────────────────────────────────────────────────────────────

impl PaintColorDistrib {
    /// Resolve color for a voxel at `(x, y, z)`. `stroke_seed` is generated once per stroke.
    pub fn resolve(&self, palette: &[u32], stroke_seed: u32, x: i32, y: i32, z: i32) -> u32 {
        let n = palette.len();
        if n == 0 {
            return 0;
        }
        if n == 1 {
            return palette[0] & 0x00ff_ffff;
        }
        match self.mode {
            PaintColorMode::WhiteNoise => {
                let idx = paint_color_index(x, y, z, n);
                palette[idx] & 0x00ff_ffff
            }
            PaintColorMode::RandomSingle => {
                let idx = mix_seed_to_index(stroke_seed, n);
                palette[idx] & 0x00ff_ffff
            }
            PaintColorMode::FbmNoise => {
                let t = fbm_value3(
                    self.fbm.noise_seed,
                    x as f32,
                    y as f32,
                    z as f32,
                    self.fbm.octaves,
                    self.fbm.lacunarity,
                    self.fbm.persistence,
                    self.fbm.frequency,
                );
                color_from_t(palette, t, self.fbm.quantized)
            }
            PaintColorMode::Gradient => {
                let t = self.gradient_t(x as f32, y as f32, z as f32);
                color_from_t(palette, t, self.gradient.quantized)
            }
            PaintColorMode::Dither => {
                let size = self.dither.ordered_size.clamp(2, 8);
                let bayer_t = bayer_threshold(size, x, y);
                let t = if self.dither.ordered_strength > 0.0 {
                    let seed = self.fbm.noise_seed ^ 0xabc;
                    let noise = hash3_f(seed, x, y, z);
                    (bayer_t + (noise - 0.5) * self.dither.ordered_strength).clamp(0.0, 1.0)
                } else {
                    bayer_t
                };
                color_from_t(palette, t, true)
            }
        }
    }

    fn gradient_t(&self, x: f32, y: f32, z: f32) -> f32 {
        match self.gradient.kind {
            GradientKind::Linear => {
                let p = match self.gradient.linear_axis {
                    0 => x,
                    1 => y,
                    _ => z,
                };
                fract(p * self.gradient.scale + self.gradient.phase)
            }
            GradientKind::Radial => {
                let [cx, cy, cz] = self.gradient.radial_center;
                let dx = x - cx;
                let dy = y - cy;
                let dz = z - cz;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                fract(dist * self.gradient.scale + self.gradient.phase)
            }
        }
    }
}

fn mix_seed_to_index(seed: u32, n: usize) -> usize {
    let mut h = seed;
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    (h as usize) % n
}

fn fract(v: f32) -> f32 {
    v - v.floor()
}
