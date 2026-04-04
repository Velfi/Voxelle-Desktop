//! Numeric parity with Voxelle web (`voxelMaterial.ts`, `glassShadowConstants.ts`, `webgpuBloom.ts`).
#![allow(dead_code)]

/// [`VOXEL_GLASS_PHYSICAL`](https://github.com/...) — transmission, thickness, IOR, attenuation.
pub const GLASS_TRANSMISSION: f32 = 0.96;
pub const GLASS_THICKNESS: f32 = 0.65;
pub const GLASS_IOR: f32 = 1.5;
pub const GLASS_ATTENUATION_DISTANCE: f32 = 2.5;
pub const GLASS_ROUGHNESS: f32 = 0.06;

pub const WATER_TRANSMISSION: f32 = 0.998;
pub const WATER_THICKNESS: f32 = 0.9;
pub const WATER_IOR: f32 = 1.333;
pub const WATER_ATTENUATION_DISTANCE: f32 = 32.0;

// Wax (subsurface scattering)
pub const WAX_IOR: f32 = 1.46;
pub const WAX_SSS_ATTENUATION: f32 = 1.8;
pub const WAX_SSS_STRENGTH: f32 = 0.65;
pub const WAX_WRAP: f32 = 0.55;

// Holographic (thin-film diffraction grating)
/// Thin-film optical thickness in nanometers — controls which wavelengths interfere.
/// 550 nm ≈ center of visible spectrum; range ~380–780 nm sweeps through all hues.
pub const HOLO_FILM_THICKNESS_NM: f32 = 550.0;
/// Index of refraction of the thin film layer (MgF₂ / oil-film range).
pub const HOLO_FILM_IOR: f32 = 1.38;
/// Base reflectance of the metallic substrate underneath the film.
pub const HOLO_BASE_REFLECTANCE: f32 = 0.75;
/// Strength of the spectral iridescence (0 = pure metal, 1 = full rainbow).
pub const HOLO_IRIDESCENCE_STRENGTH: f32 = 0.85;

/// Shadow depth bias tuning (matches `glassShadowConstants.ts`).
pub const GLASS_SHADOW_VERTEX_AO_POW: f32 = 1.65;
pub const GLASS_SHADOW_VERTEX_AO_SCALE: f32 = 1.0;
pub const GLASS_SHADOW_SLAB_ABSORPTION: f32 = 0.16;
pub const GLASS_SHADOW_SLAB_MIN_TRANSMITTANCE: f32 = 0.35;
pub const GLASS_SHADOW_DEPTH_PUSH_MAX: f32 = 0.02;

/// Web `sceneSetup` directional shadow map size.
pub const SHADOW_MAP_SIZE: u32 = 8192;
/// NDC depth subtracted in `textureSampleCompare` (base term; see `scene.wgsl` slope term).
/// World-space shadow bias (voxel units). Converted to NDC in shader via `light_view_proj` Z gradient.
pub const SHADOW_BIAS_WORLD_BASE: f32 = 0.04;
/// Extra world-space bias for surfaces at grazing angles (`(1 - N·L)` scaling).
pub const SHADOW_BIAS_WORLD_SLOPE: f32 = 0.15;

/// `webgpuBloom.ts`
pub const BLOOM_STRENGTH: f32 = 0.88;
pub const BLOOM_RADIUS: f32 = 0.42;
pub const BLOOM_THRESHOLD: f32 = 0.15;
