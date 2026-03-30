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

/// Shadow depth bias tuning (matches `glassShadowConstants.ts`).
pub const GLASS_SHADOW_VERTEX_AO_POW: f32 = 1.65;
pub const GLASS_SHADOW_VERTEX_AO_SCALE: f32 = 1.0;
pub const GLASS_SHADOW_SLAB_ABSORPTION: f32 = 0.16;
pub const GLASS_SHADOW_SLAB_MIN_TRANSMITTANCE: f32 = 0.35;
pub const GLASS_SHADOW_DEPTH_PUSH_MAX: f32 = 0.02;

/// Web `sceneSetup` directional shadow map size.
pub const SHADOW_MAP_SIZE: u32 = 4096;
pub const SHADOW_BIAS: f32 = -0.0002;
pub const SHADOW_NORMAL_BIAS: f32 = 0.02;

/// `webgpuBloom.ts`
pub const BLOOM_STRENGTH: f32 = 0.88;
pub const BLOOM_RADIUS: f32 = 0.42;
pub const BLOOM_THRESHOLD: f32 = 0.15;

/// SSAO (tunable toward web `aoStrength` tiers).
pub const SSAO_RADIUS: f32 = 0.35;
pub const SSAO_STRENGTH: f32 = 0.85;
pub const SSAO_BIAS: f32 = 0.025;
