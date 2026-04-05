use bson::raw::{RawArray, RawBsonRef, RawDocument};
use bson::{Bson, Document};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use rayon::prelude::*;
use std::io::{Read, Write};
use thiserror::Error;

pub const V3_MAGIC: [u8; 4] = [0x56, 0x58, 0x33, 0x1a];
pub const V4_MAGIC: [u8; 4] = [0x56, 0x58, 0x34, 0x1a];
/// V5 container: same inner payload as V4 but compressed with zstd instead of gzip.
pub const V5_MAGIC: [u8; 4] = [0x56, 0x58, 0x35, 0x1a];
pub const V3_RECORD_SIZE: usize = 20;
/// Dense **VX3 wire version 4** body: `object_id` `u32` after the legacy 20-byte fields (24 bytes total).
pub const V4_WIRE_RECORD_SIZE: usize = 24;
/// Always use dense wire format for the voxel array (faster and more compact than BSON-per-voxel).
pub const V3_WIRE_VOXEL_THRESHOLD: usize = 0;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("empty payload")]
    Empty,
    #[error("gzip decompress: {0}")]
    Gzip(std::io::Error),
    #[error("invalid v3 wire")]
    InvalidV3,
    #[error("invalid v4 container")]
    InvalidV4,
    #[error("v4 crc mismatch")]
    V4CrcMismatch,
    #[error("zstd decompress: {0}")]
    Zstd(std::io::Error),
    #[error("bson: {0}")]
    Bson(bson::de::Error),
    #[error("raw bson: {0}")]
    RawBson(String),
    #[error("missing required fields")]
    InvalidDocument,
}

#[derive(Debug, Error)]
pub enum EncodeError {
    #[error("bson encode: {0}")]
    Bson(bson::ser::Error),
    #[error("io: {0}")]
    Io(std::io::Error),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MaterialId {
    Plastic,
    Metal,
    Rubber,
    Glass,
    Water,
    Glow,
    Velvet,
    Wax,
    Holographic,
}

impl MaterialId {
    pub fn from_index(i: u8) -> Self {
        match i {
            0 => MaterialId::Plastic,
            1 => MaterialId::Metal,
            2 => MaterialId::Rubber,
            3 => MaterialId::Glass,
            4 => MaterialId::Water,
            5 => MaterialId::Glow,
            6 => MaterialId::Velvet,
            7 => MaterialId::Wax,
            8 => MaterialId::Holographic,
            _ => MaterialId::Plastic,
        }
    }

    pub fn material_index(self) -> u8 {
        match self {
            MaterialId::Plastic => 0,
            MaterialId::Metal => 1,
            MaterialId::Rubber => 2,
            MaterialId::Glass => 3,
            MaterialId::Water => 4,
            MaterialId::Glow => 5,
            MaterialId::Velvet => 6,
            MaterialId::Wax => 7,
            MaterialId::Holographic => 8,
        }
    }

    pub fn as_str_id(self) -> &'static str {
        match self {
            MaterialId::Plastic => "plastic",
            MaterialId::Metal => "metal",
            MaterialId::Rubber => "rubber",
            MaterialId::Glass => "glass",
            MaterialId::Water => "water",
            MaterialId::Glow => "glow",
            MaterialId::Velvet => "velvet",
            MaterialId::Wax => "wax",
            MaterialId::Holographic => "holographic",
        }
    }

    pub fn from_str_id(s: &str) -> Self {
        match s {
            "metal" => MaterialId::Metal,
            "rubber" => MaterialId::Rubber,
            "glass" => MaterialId::Glass,
            "water" => MaterialId::Water,
            "glow" => MaterialId::Glow,
            "velvet" => MaterialId::Velvet,
            "wax" => MaterialId::Wax,
            "holographic" => MaterialId::Holographic,
            _ => MaterialId::Plastic,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Voxel {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub color: u32,
    pub material: MaterialId,
    /// Scene object that owns this voxel (local integer coordinates relative to object).
    pub object_id: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneObject {
    pub id: u32,
    pub parent_id: Option<u32>,
    pub name: String,
    pub visible: bool,
    pub sort_order: i32,
    pub translation: [f32; 3],
    /// xyzw quaternion (identity = 0,0,0,1).
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

impl Default for SceneObject {
    fn default() -> Self {
        Self {
            id: 0,
            parent_id: None,
            name: "Scene".to_string(),
            visible: true,
            sort_order: 0,
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

/// Single implicit object for legacy files or empty `objects` in BSON.
pub fn default_scene_objects() -> Vec<SceneObject> {
    vec![SceneObject::default()]
}

#[derive(Clone, Debug, Default)]
pub struct Scene {
    pub focal_length_mm: Option<f32>,
    pub orthographic: bool,
}

/// Post-process mood stored under `scene.mood` in BSON.
/// Rich per-effect settings matching the web Voxelle implementation.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MoodSettings {
    // vignette (desktop-only)
    #[serde(default)]
    pub vignette: f32,
    // grain
    #[serde(default)]
    pub grain_enabled: bool,
    #[serde(default)]
    pub grain_strength: f32,
    #[serde(default = "default_true")]
    pub grain_animated: bool,
    #[serde(default = "default_one")]
    pub grain_speed: f32,
    #[serde(default = "default_true")]
    pub grain_colorful: bool,
    // atmosphere
    #[serde(default)]
    pub atm_enabled: bool,
    #[serde(default = "default_atm_color")]
    pub atm_color: String,
    #[serde(default = "default_28")]
    pub atm_thickness: f32,
    #[serde(default = "default_085")]
    pub atm_density: f32,
    #[serde(default = "default_true")]
    pub atm_aerial: bool,
    #[serde(default)]
    pub atm_positive_side: bool,
    #[serde(default)]
    pub atm_plane_nx: f32,
    #[serde(default)]
    pub atm_plane_ny: f32,
    #[serde(default)]
    pub atm_plane_nz: f32,
    #[serde(default)]
    pub atm_plane_c: f32,
    #[serde(default)]
    pub atm_height_bias: f32,
    #[serde(default = "default_120")]
    pub atm_height_falloff: f32,
    #[serde(default)]
    pub atm_drift_enabled: bool,
    #[serde(default = "default_02")]
    pub atm_drift_amount: f32,
    #[serde(default = "default_002")]
    pub atm_drift_scale: f32,
    #[serde(default = "default_02")]
    pub atm_drift_speed: f32,
    // distance tint
    #[serde(default)]
    pub dt_enabled: bool,
    #[serde(default = "default_dt_near")]
    pub dt_near_color: String,
    #[serde(default = "default_dt_mid")]
    pub dt_mid_color: String,
    #[serde(default = "default_dt_far")]
    pub dt_far_color: String,
    #[serde(default = "default_16")]
    pub dt_near_dist: f32,
    #[serde(default = "default_140")]
    pub dt_far_dist: f32,
    #[serde(default = "default_06")]
    pub dt_strength: f32,
    // sun shafts
    #[serde(default)]
    pub ss_enabled: bool,
    #[serde(default = "default_07")]
    pub ss_strength: f32,
    #[serde(default = "default_092")]
    pub ss_decay: f32,
    #[serde(default = "default_08")]
    pub ss_density: f32,
    #[serde(default = "default_06")]
    pub ss_weight: f32,
    #[serde(default = "default_32")]
    pub ss_samples: f32,
    // screen-space reflections
    #[serde(default)]
    pub ssr_enabled: bool,
    #[serde(default = "default_08")]
    pub ssr_strength: f32,
    // bloom
    #[serde(default = "default_09")]
    pub bloom_strength: f32,
}

fn default_true() -> bool {
    true
}
fn default_one() -> f32 {
    1.0
}
fn default_atm_color() -> String {
    "#c8d4e0".into()
}
fn default_28() -> f32 {
    28.0
}
fn default_085() -> f32 {
    0.85
}
fn default_120() -> f32 {
    120.0
}
fn default_02() -> f32 {
    0.2
}
fn default_002() -> f32 {
    0.02
}
fn default_dt_near() -> String {
    "#ffffff".into()
}
fn default_dt_mid() -> String {
    "#c8d4e0".into()
}
fn default_dt_far() -> String {
    "#8fa3bf".into()
}
fn default_16() -> f32 {
    16.0
}
fn default_140() -> f32 {
    140.0
}
fn default_06() -> f32 {
    0.6
}
fn default_07() -> f32 {
    0.7
}
fn default_092() -> f32 {
    0.92
}
fn default_08() -> f32 {
    0.8
}
fn default_32() -> f32 {
    32.0
}
fn default_09() -> f32 {
    0.1
}

impl Default for MoodSettings {
    fn default() -> Self {
        Self {
            vignette: 0.0,
            grain_enabled: false,
            grain_strength: 0.12,
            grain_animated: true,
            grain_speed: 1.0,
            grain_colorful: true,
            atm_enabled: false,
            atm_color: default_atm_color(),
            atm_thickness: 28.0,
            atm_density: 0.85,
            atm_aerial: true,
            atm_positive_side: false,
            atm_plane_nx: 0.0,
            atm_plane_ny: 0.0,
            atm_plane_nz: 0.0,
            atm_plane_c: 0.0,
            atm_height_bias: 0.0,
            atm_height_falloff: 120.0,
            atm_drift_enabled: false,
            atm_drift_amount: 0.2,
            atm_drift_scale: 0.02,
            atm_drift_speed: 0.2,
            dt_enabled: false,
            dt_near_color: default_dt_near(),
            dt_mid_color: default_dt_mid(),
            dt_far_color: default_dt_far(),
            dt_near_dist: 16.0,
            dt_far_dist: 140.0,
            dt_strength: 0.6,
            ss_enabled: false,
            ss_strength: 0.7,
            ss_decay: 0.92,
            ss_density: 0.8,
            ss_weight: 0.6,
            ss_samples: 32.0,
            ssr_enabled: false,
            ssr_strength: 0.8,
            bloom_strength: 0.1,
        }
    }
}

/// Scene lighting / viewport defaults under `scene.lighting` in BSON (matches web Voxelle sidebar).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LightingSettings {
    pub ambient_intensity: f32,
    pub sunlight_intensity: f32,
    pub light_color: String,
    /// Degrees in XZ (web `lightAngle`).
    #[serde(rename = "lightAngle")]
    pub light_angle_deg: f32,
    /// Degrees above horizon (web `lightElevation`).
    #[serde(rename = "lightElevation")]
    pub light_elevation_deg: f32,
    pub enable_shadows: bool,
    pub enable_sky: bool,
    pub background_color: String,
    pub exposure_ev: f32,
    pub auto_exposure: bool,
}

impl Default for LightingSettings {
    fn default() -> Self {
        Self {
            ambient_intensity: 1.0,
            sunlight_intensity: 2.0,
            light_color: "#ffffff".to_string(),
            light_angle_deg: 45.0,
            light_elevation_deg: 45.0,
            enable_shadows: true,
            enable_sky: true,
            background_color: "#0a0b0e".to_string(),
            exposure_ev: 0.0,
            auto_exposure: false,
        }
    }
}

fn lighting_to_bson_document(l: &LightingSettings) -> Document {
    bson::doc! {
        "ambientIntensity": l.ambient_intensity as f64,
        "sunlightIntensity": l.sunlight_intensity as f64,
        "lightColor": &l.light_color,
        "lightAngle": l.light_angle_deg as f64,
        "lightElevation": l.light_elevation_deg as f64,
        "enableShadows": l.enable_shadows,
        "enableSky": l.enable_sky,
        "backgroundColor": &l.background_color,
        "exposureEv": l.exposure_ev as f64,
        "autoExposure": l.auto_exposure,
    }
}

fn parse_lighting_from_scene_optional(scene: &Document) -> Option<LightingSettings> {
    let m = scene.get_document("lighting").ok()?;
    Some(LightingSettings {
        ambient_intensity: m
            .get("ambientIntensity")
            .and_then(|b| bson_f32(b))
            .unwrap_or(1.0),
        sunlight_intensity: m
            .get("sunlightIntensity")
            .and_then(|b| bson_f32(b))
            .unwrap_or(1.0),
        light_color: m
            .get_str("lightColor")
            .ok()
            .map(|s| s.to_string())
            .filter(|s| s.starts_with('#'))
            .unwrap_or_else(|| "#ffffff".to_string()),
        light_angle_deg: m
            .get("lightAngle")
            .and_then(|b| bson_f32(b))
            .unwrap_or(45.0),
        light_elevation_deg: m
            .get("lightElevation")
            .and_then(|b| bson_f32(b))
            .unwrap_or(45.0),
        enable_shadows: m.get_bool("enableShadows").unwrap_or(true),
        enable_sky: m.get_bool("enableSky").unwrap_or(true),
        background_color: m
            .get_str("backgroundColor")
            .ok()
            .map(|s| s.to_string())
            .filter(|s| s.starts_with('#'))
            .unwrap_or_else(|| "#0a0b0e".to_string()),
        exposure_ev: m.get("exposureEv").and_then(|b| bson_f32(b)).unwrap_or(0.0),
        auto_exposure: m.get_bool("autoExposure").unwrap_or(false),
    })
}

fn parse_lighting_from_raw_file_bytes(bytes: &[u8]) -> Option<LightingSettings> {
    let doc = RawDocument::from_bytes(bytes).ok()?;
    let scene = doc.get_document("scene").ok()?;
    let m = scene.get_document("lighting").ok()?;
    Some(LightingSettings {
        ambient_intensity: m
            .get("ambientIntensity")
            .ok()
            .flatten()
            .and_then(raw_bson_to_f32)
            .unwrap_or(1.0),
        sunlight_intensity: m
            .get("sunlightIntensity")
            .ok()
            .flatten()
            .and_then(raw_bson_to_f32)
            .unwrap_or(1.0),
        light_color: m
            .get("lightColor")
            .ok()
            .flatten()
            .and_then(|b| match b {
                RawBsonRef::String(s) if s.starts_with('#') => Some(s.to_string()),
                _ => None,
            })
            .unwrap_or_else(|| "#ffffff".to_string()),
        light_angle_deg: m
            .get("lightAngle")
            .ok()
            .flatten()
            .and_then(raw_bson_to_f32)
            .unwrap_or(45.0),
        light_elevation_deg: m
            .get("lightElevation")
            .ok()
            .flatten()
            .and_then(raw_bson_to_f32)
            .unwrap_or(45.0),
        enable_shadows: m
            .get("enableShadows")
            .ok()
            .flatten()
            .and_then(|b| match b {
                RawBsonRef::Boolean(x) => Some(x),
                _ => None,
            })
            .unwrap_or(true),
        enable_sky: m
            .get("enableSky")
            .ok()
            .flatten()
            .and_then(|b| match b {
                RawBsonRef::Boolean(x) => Some(x),
                _ => None,
            })
            .unwrap_or(true),
        background_color: m
            .get("backgroundColor")
            .ok()
            .flatten()
            .and_then(|b| match b {
                RawBsonRef::String(s) if s.starts_with('#') => Some(s.to_string()),
                _ => None,
            })
            .unwrap_or_else(|| "#0a0b0e".to_string()),
        exposure_ev: m
            .get("exposureEv")
            .ok()
            .flatten()
            .and_then(raw_bson_to_f32)
            .unwrap_or(0.0),
        auto_exposure: m
            .get("autoExposure")
            .ok()
            .flatten()
            .and_then(|b| match b {
                RawBsonRef::Boolean(x) => Some(x),
                _ => None,
            })
            .unwrap_or(false),
    })
}

fn parse_lighting_from_file_bytes(bytes: &[u8]) -> Option<LightingSettings> {
    if bytes.len() <= 8 * 1024 * 1024 {
        let doc = bson::from_slice::<Document>(bytes).ok()?;
        let scene = doc.get_document("scene").ok()?;
        parse_lighting_from_scene_optional(scene)
    } else {
        parse_lighting_from_raw_file_bytes(bytes)
    }
}

fn bson_f32(b: &Bson) -> Option<f32> {
    match b {
        Bson::Double(d) if d.is_finite() => Some(*d as f32),
        Bson::Int32(i) => Some(*i as f32),
        Bson::Int64(i) => Some(*i as f32),
        _ => None,
    }
}

/// Read `scene.mood` from a full scene document. Returns `None` if the `mood` key is absent.
/// Handles both old (bare-float) and new (rich nested) formats.
pub fn parse_mood_from_scene_optional(scene: &Document) -> Option<MoodSettings> {
    let m = scene.get_document("mood").ok()?;
    // Detect old format: bare `grain` float at top level with no `grainEnabled` key.
    let is_old = m.get("grainEnabled").is_none() && m.get("grain").is_some();
    if is_old {
        let grain_v = m.get("grain").and_then(|b| bson_f32(b)).unwrap_or(0.0);
        let vig_v = m.get("vignette").and_then(|b| bson_f32(b)).unwrap_or(0.0);
        let dt_v = m
            .get("distanceTint")
            .and_then(|b| bson_f32(b))
            .unwrap_or(0.0);
        let atm_v = m.get("atmosphere").and_then(|b| bson_f32(b)).unwrap_or(0.0);
        let ss_v = m.get("sunShafts").and_then(|b| bson_f32(b)).unwrap_or(0.0);
        let mut ms = MoodSettings::default();
        ms.vignette = vig_v;
        if grain_v > 0.001 {
            ms.grain_enabled = true;
            ms.grain_strength = grain_v.clamp(0.0, 0.5);
        }
        if atm_v > 0.001 {
            ms.atm_enabled = true;
            ms.atm_density = atm_v;
        }
        if dt_v > 0.001 {
            ms.dt_enabled = true;
            ms.dt_strength = dt_v;
        }
        if ss_v > 0.001 {
            ms.ss_enabled = true;
            ms.ss_strength = ss_v * 10.0; // old range was 0–1, new is 0–10
        }
        return Some(ms);
    }
    // New format: full serde deserialisation via BSON → serde.
    bson::from_document(m.clone()).ok()
}

fn mood_to_bson_document(m: &MoodSettings) -> Document {
    bson::to_document(m).unwrap_or_default()
}

fn raw_bson_to_f32(b: RawBsonRef<'_>) -> Option<f32> {
    match b {
        RawBsonRef::Double(d) if d.is_finite() => Some(d as f32),
        RawBsonRef::Int32(i) => Some(i as f32),
        RawBsonRef::Int64(i) => Some(i as f32),
        _ => None,
    }
}

fn parse_mood_from_raw_file_bytes(bytes: &[u8]) -> Option<MoodSettings> {
    let doc = RawDocument::from_bytes(bytes).ok()?;
    let scene = doc.get_document("scene").ok()?;
    let mood = scene.get_document("mood").ok()?;
    // Detect old format: bare `grain` float with no `grainEnabled`.
    let has_grain_enabled = mood.get("grainEnabled").ok().flatten().is_some();
    if !has_grain_enabled {
        // Old bare-float format
        let grain_v = mood
            .get("grain")
            .ok()
            .flatten()
            .and_then(raw_bson_to_f32)
            .unwrap_or(0.0);
        let vig_v = mood
            .get("vignette")
            .ok()
            .flatten()
            .and_then(raw_bson_to_f32)
            .unwrap_or(0.0);
        let dt_v = mood
            .get("distanceTint")
            .ok()
            .flatten()
            .and_then(raw_bson_to_f32)
            .unwrap_or(0.0);
        let atm_v = mood
            .get("atmosphere")
            .ok()
            .flatten()
            .and_then(raw_bson_to_f32)
            .unwrap_or(0.0);
        let ss_v = mood
            .get("sunShafts")
            .ok()
            .flatten()
            .and_then(raw_bson_to_f32)
            .unwrap_or(0.0);
        let mut ms = MoodSettings::default();
        ms.vignette = vig_v;
        if grain_v > 0.001 {
            ms.grain_enabled = true;
            ms.grain_strength = grain_v.clamp(0.0, 0.5);
        }
        if atm_v > 0.001 {
            ms.atm_enabled = true;
            ms.atm_density = atm_v;
        }
        if dt_v > 0.001 {
            ms.dt_enabled = true;
            ms.dt_strength = dt_v;
        }
        if ss_v > 0.001 {
            ms.ss_enabled = true;
            ms.ss_strength = ss_v * 10.0;
        }
        return Some(ms);
    }
    // New format: convert raw to owned Document, then serde
    let owned = bson::from_slice::<Document>(mood.as_bytes()).ok()?;
    bson::from_document(owned).ok()
}

fn parse_mood_from_file_bytes(bytes: &[u8]) -> Option<MoodSettings> {
    if bytes.len() <= 8 * 1024 * 1024 {
        let doc = bson::from_slice::<Document>(bytes).ok()?;
        let scene = doc.get_document("scene").ok()?;
        parse_mood_from_scene_optional(scene)
    } else {
        parse_mood_from_raw_file_bytes(bytes)
    }
}

#[derive(Clone, Debug)]
pub struct VoxelleFile {
    #[allow(dead_code)]
    pub version: i32,
    #[allow(dead_code)]
    pub grid_size: i32,
    pub scene: Scene,
    /// Full `scene` subdocument when loaded (preserves `atmosphere` etc.). If `Some`, encode prefers this over [`Scene`] alone.
    pub scene_extra: Option<Document>,
    /// Parsed from `scene.mood` on load; merged into `scene` on save when `Some`.
    pub mood: Option<MoodSettings>,
    /// Parsed from `scene.lighting` on load; merged into `scene` on save when `Some`.
    pub lighting: Option<LightingSettings>,
    pub voxels: Vec<Voxel>,
    pub objects: Vec<SceneObject>,
    /// Target object for new voxel edits (persisted in BSON; default 0).
    pub active_object_id: u32,
}

/// Match `focalLengthToFov` in Voxelle `sceneSetup.ts`.
pub fn focal_length_to_fov_y_radians(mm: f32) -> f32 {
    2.0 * (12.0_f32 / mm).atan()
}

fn srgb_byte_to_linear_u8(b: u8) -> f32 {
    let c = b as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// `#rgb` or `#rrggbb` (CSS-style) → linear RGB in \([0,1]\) for GPU.
pub fn hex_srgb_to_linear_rgb3(hex: &str) -> Option<[f32; 3]> {
    let t = hex.trim();
    let s = t.strip_prefix('#')?;
    let (r, g, b) = match s.len() {
        3 => {
            let r = u8::from_str_radix(&format!("{}{}", &s[0..1], &s[0..1]), 16).ok()?;
            let g = u8::from_str_radix(&format!("{}{}", &s[1..2], &s[1..2]), 16).ok()?;
            let b = u8::from_str_radix(&format!("{}{}", &s[2..3], &s[2..3]), 16).ok()?;
            (r, g, b)
        }
        6 => {
            let r = u8::from_str_radix(&s[0..2], 16).ok()?;
            let g = u8::from_str_radix(&s[2..4], 16).ok()?;
            let b = u8::from_str_radix(&s[4..6], 16).ok()?;
            (r, g, b)
        }
        _ => return None,
    };
    Some([
        srgb_byte_to_linear_u8(r),
        srgb_byte_to_linear_u8(g),
        srgb_byte_to_linear_u8(b),
    ])
}

/// Packed `0xRRGGBB` voxel color (same as UI) → linear RGB in \([0,1]\) for GPU preview meshes.
#[inline]
pub fn rgb24_u32_to_linear_rgb3(rgb: u32) -> [f32; 3] {
    let r = ((rgb >> 16) & 0xff) as u8;
    let g = ((rgb >> 8) & 0xff) as u8;
    let b = (rgb & 0xff) as u8;
    [
        srgb_byte_to_linear_u8(r),
        srgb_byte_to_linear_u8(g),
        srgb_byte_to_linear_u8(b),
    ]
}

fn decompress_if_gzipped(bytes: &[u8]) -> Result<Vec<u8>, ParseError> {
    if bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b {
        let mut decoder = GzDecoder::new(bytes);
        let mut out = Vec::new();
        decoder.read_to_end(&mut out).map_err(ParseError::Gzip)?;
        Ok(out)
    } else {
        Ok(bytes.to_vec())
    }
}

fn is_v3_wire(bytes: &[u8]) -> bool {
    bytes.len() >= 12
        && bytes[0] == V3_MAGIC[0]
        && bytes[1] == V3_MAGIC[1]
        && bytes[2] == V3_MAGIC[2]
        && bytes[3] == V3_MAGIC[3]
}

/// Infer bytes per voxel from total body length. Wire **4** may be legacy **20**-byte (pre–object-id
/// writers) or current **24**-byte. Wire **5** was a short-lived label for 24-byte dense; still accepted.
fn infer_v3_wire_record_size(
    wire_ver: u32,
    body_byte_len: usize,
    voxel_count: i32,
    hidden_count: i32,
) -> Result<usize, ParseError> {
    let total = (voxel_count + hidden_count) as usize;
    if total == 0 {
        return if body_byte_len == 0 {
            Ok(V3_RECORD_SIZE)
        } else {
            Err(ParseError::InvalidV3)
        };
    }
    if body_byte_len % total != 0 {
        return Err(ParseError::InvalidV3);
    }
    let per = body_byte_len / total;
    match wire_ver {
        3 => {
            if per != V3_RECORD_SIZE {
                return Err(ParseError::InvalidV3);
            }
            Ok(V3_RECORD_SIZE)
        }
        4 => match per {
            V4_WIRE_RECORD_SIZE => Ok(V4_WIRE_RECORD_SIZE),
            V3_RECORD_SIZE => Ok(V3_RECORD_SIZE),
            _ => Err(ParseError::InvalidV3),
        },
        5 => {
            if per != V4_WIRE_RECORD_SIZE {
                return Err(ParseError::InvalidV3);
            }
            Ok(V4_WIRE_RECORD_SIZE)
        }
        _ => Err(ParseError::InvalidV3),
    }
}

fn parse_objects_from_document(doc: &Document) -> Option<Vec<SceneObject>> {
    let arr = doc.get_array("objects").ok()?;
    let mut out = Vec::with_capacity(arr.len());
    for b in arr {
        let sub = b.as_document()?;
        let id = doc_u32(sub, "id")?;
        let parent_id = match sub.get("parent") {
            None | Some(Bson::Null) => None,
            Some(Bson::Int32(i)) if *i >= 0 => Some(*i as u32),
            Some(Bson::Int64(i)) if *i >= 0 && *i <= u32::MAX as i64 => Some(*i as u32),
            Some(Bson::Double(d)) if *d >= 0.0 && d.is_finite() => Some(*d as u32),
            _ => None,
        };
        let name = sub
            .get_str("name")
            .ok()
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("Object {id}"));
        let visible = sub.get_bool("visible").unwrap_or(true);
        let sort_order = doc_i32(sub, "sortOrder").unwrap_or(0);
        let translation = parse_f32_array3(sub, "t").unwrap_or([0.0; 3]);
        let rotation = parse_f32_array4(sub, "r").unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let scale = parse_f32_array3(sub, "s").unwrap_or([1.0, 1.0, 1.0]);
        out.push(SceneObject {
            id,
            parent_id,
            name,
            visible,
            sort_order,
            translation,
            rotation,
            scale,
        });
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn doc_u32(doc: &Document, key: &str) -> Option<u32> {
    doc.get(key).and_then(|b| match b {
        Bson::Int32(i) if *i >= 0 => Some(*i as u32),
        Bson::Int64(i) if *i >= 0 && *i <= i64::from(u32::MAX) => Some(*i as u32),
        Bson::Double(d) if *d >= 0.0 && d.is_finite() => Some(*d as u32),
        _ => None,
    })
}

fn parse_f32_array3(doc: &Document, key: &str) -> Option<[f32; 3]> {
    let arr = doc.get_array(key).ok()?;
    if arr.len() < 3 {
        return None;
    }
    let mut o = [0.0_f32; 3];
    for i in 0..3 {
        o[i] = match &arr[i] {
            Bson::Double(d) => *d as f32,
            Bson::Int32(v) => *v as f32,
            Bson::Int64(v) => *v as f32,
            _ => return None,
        };
    }
    Some(o)
}

fn parse_f32_array4(doc: &Document, key: &str) -> Option<[f32; 4]> {
    let arr = doc.get_array(key).ok()?;
    if arr.len() < 4 {
        return None;
    }
    let mut o = [0.0_f32; 4];
    for i in 0..4 {
        o[i] = match &arr[i] {
            Bson::Double(d) => *d as f32,
            Bson::Int32(v) => *v as f32,
            Bson::Int64(v) => *v as f32,
            _ => return None,
        };
    }
    Some(o)
}

fn parse_v3(bytes: &[u8]) -> Result<VoxelleFile, ParseError> {
    if bytes.len() < 16 {
        return Err(ParseError::InvalidV3);
    }
    let wire_ver = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let header_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if header_len < 8 || 12 + header_len > bytes.len() {
        return Err(ParseError::InvalidV3);
    }
    let header_slice = &bytes[12..12 + header_len];
    let doc = bson::from_slice::<Document>(header_slice).map_err(ParseError::Bson)?;
    let grid_size = doc_i32(&doc, "gridSize").ok_or(ParseError::InvalidV3)?;
    if grid_size < 1 {
        return Err(ParseError::InvalidV3);
    }
    let voxel_count = doc_i32(&doc, "voxelCount").ok_or(ParseError::InvalidV3)?;
    let hidden_count = doc_i32(&doc, "hiddenCount").ok_or(ParseError::InvalidV3)?;
    if voxel_count < 0 || hidden_count < 0 {
        return Err(ParseError::InvalidV3);
    }
    let body_byte_len = bytes.len().saturating_sub(12 + header_len);
    let rec_size = infer_v3_wire_record_size(wire_ver, body_byte_len, voxel_count, hidden_count)?;

    let scene = parse_scene_bson(&doc);
    let scene_extra = doc.get_document("scene").ok().cloned();
    let file_version = doc_i32(&doc, "version").unwrap_or(if wire_ver >= 4 { 4 } else { 3 });
    let objects = parse_objects_from_document(&doc).unwrap_or_else(default_scene_objects);
    let active_object_id = doc_u32(&doc, "activeObjectId").unwrap_or(0);

    let body_start = 12 + header_len;
    let body = &bytes[body_start..];
    let total = voxel_count as usize;
    let voxels: Vec<Voxel> = if rec_size == V4_WIRE_RECORD_SIZE && total >= 4096 {
        // Parallel parse: records are fixed-size and record-aligned.
        body[..total * rec_size]
            .par_chunks_exact(rec_size)
            .map(|chunk| {
                let x = i32::from_le_bytes(chunk[0..4].try_into().unwrap());
                let y = i32::from_le_bytes(chunk[4..8].try_into().unwrap());
                let z = i32::from_le_bytes(chunk[8..12].try_into().unwrap());
                let color = u32::from_le_bytes(chunk[12..16].try_into().unwrap()) & 0xffffff;
                let mi = chunk[16];
                let object_id = u32::from_le_bytes(chunk[20..24].try_into().unwrap());
                Voxel {
                    x,
                    y,
                    z,
                    color,
                    material: MaterialId::from_index(mi),
                    object_id,
                }
            })
            .collect()
    } else {
        // Sequential fallback for V3 records or small files.
        let mut voxels = Vec::with_capacity(total);
        let mut o = body_start;
        for i in 0..voxel_count {
            let v = if rec_size == V4_WIRE_RECORD_SIZE {
                read_v4_wire_record(bytes, o)?
            } else {
                read_v3_record(bytes, o)?
            };
            voxels.push(v);
            o += rec_size;
            if i & 0x7fff == 0x7fff {
                std::thread::yield_now();
            }
        }
        voxels
    };
    let mood = scene_extra
        .as_ref()
        .and_then(parse_mood_from_scene_optional);
    let lighting = scene_extra
        .as_ref()
        .and_then(parse_lighting_from_scene_optional);
    // Skip hidden voxels (viewer policy: visible only)
    Ok(VoxelleFile {
        version: file_version,
        grid_size,
        scene,
        scene_extra,
        mood,
        lighting,
        voxels,
        objects,
        active_object_id,
    })
}

fn read_v3_record(bytes: &[u8], o: usize) -> Result<Voxel, ParseError> {
    if o + V3_RECORD_SIZE > bytes.len() {
        return Err(ParseError::InvalidV3);
    }
    let x = i32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let y = i32::from_le_bytes(bytes[o + 4..o + 8].try_into().unwrap());
    let z = i32::from_le_bytes(bytes[o + 8..o + 12].try_into().unwrap());
    let color = u32::from_le_bytes(bytes[o + 12..o + 16].try_into().unwrap()) & 0xffffff;
    let mi = bytes[o + 16];
    Ok(Voxel {
        x,
        y,
        z,
        color,
        material: MaterialId::from_index(mi),
        object_id: 0,
    })
}

fn read_v4_wire_record(bytes: &[u8], o: usize) -> Result<Voxel, ParseError> {
    if o + V4_WIRE_RECORD_SIZE > bytes.len() {
        return Err(ParseError::InvalidV3);
    }
    let mut v = read_v3_record(bytes, o)?;
    v.object_id = u32::from_le_bytes(bytes[o + 20..o + 24].try_into().unwrap());
    Ok(v)
}

fn parse_scene_bson(doc: &Document) -> Scene {
    let mut scene = Scene::default();
    if let Ok(b) = doc.get_document("scene") {
        if let Ok(f) = b.get_f64("focalLength") {
            let ff = f as f32;
            if (15.0..=200.0).contains(&ff) {
                scene.focal_length_mm = Some(ff);
            }
        }
        if let Ok(o) = b.get_bool("orthographic") {
            scene.orthographic = o;
        }
    }
    scene
}

fn parse_scene_raw(doc: &RawDocument) -> Scene {
    let mut scene = Scene::default();
    if let Ok(s) = doc.get_document("scene") {
        if let Ok(f) = s.get_f64("focalLength") {
            let ff = f as f32;
            if (15.0..=200.0).contains(&ff) {
                scene.focal_length_mm = Some(ff);
            }
        }
        if let Ok(o) = s.get_bool("orthographic") {
            scene.orthographic = o;
        }
    }
    scene
}

fn raw_bson_to_i32(b: RawBsonRef<'_>) -> Option<i32> {
    match b {
        RawBsonRef::Int32(i) => Some(i),
        RawBsonRef::Int64(i) => i32::try_from(i).ok(),
        RawBsonRef::Double(d) if d.is_finite() => Some(d as i32),
        _ => None,
    }
}

fn raw_bson_to_u32(b: RawBsonRef<'_>) -> Option<u32> {
    match b {
        RawBsonRef::Int32(i) if i >= 0 => Some(i as u32),
        RawBsonRef::Int64(i) if i >= 0 && i <= i64::from(u32::MAX) => Some(i as u32),
        RawBsonRef::Double(d) if d.is_finite() && d >= 0.0 => Some(d as u32),
        _ => None,
    }
}

fn raw_doc_i32(doc: &RawDocument, key: &str) -> Result<i32, ParseError> {
    let Some(v) = doc
        .get(key)
        .map_err(|e| ParseError::RawBson(e.to_string()))?
    else {
        return Err(ParseError::InvalidDocument);
    };
    raw_bson_to_i32(v).ok_or(ParseError::InvalidDocument)
}

fn raw_bson_color(b: RawBsonRef<'_>) -> Option<u32> {
    let v = match b {
        RawBsonRef::Int32(i) => i as i64,
        RawBsonRef::Int64(i) => i,
        RawBsonRef::Double(d) if d.is_finite() => d as i64,
        _ => return None,
    };
    Some((v as u32) & 0xffffff)
}

fn parse_voxel_row_raw(row: &RawArray) -> Option<Voxel> {
    let x = raw_bson_to_i32(row.get(0).ok().flatten()?)?;
    let y = raw_bson_to_i32(row.get(1).ok().flatten()?)?;
    let z = raw_bson_to_i32(row.get(2).ok().flatten()?)?;
    let color = raw_bson_color(row.get(3).ok().flatten()?)?;
    let material = match row.get(4).ok().flatten() {
        Some(RawBsonRef::String(s)) => MaterialId::from_str_id(s),
        Some(b) => MaterialId::from_index(raw_bson_to_i32(b).unwrap_or(0).clamp(0, 6) as u8),
        None => MaterialId::Plastic,
    };
    let object_id = match row.get(5).ok().flatten() {
        Some(b) => raw_bson_to_u32(b).unwrap_or(0),
        None => 0,
    };
    Some(Voxel {
        x,
        y,
        z,
        color,
        material,
        object_id,
    })
}

fn doc_i32(doc: &Document, key: &str) -> Option<i32> {
    doc.get(key).and_then(|b| {
        use bson::Bson;
        match b {
            Bson::Int32(i) => Some(*i),
            Bson::Int64(i) => i32::try_from(*i).ok(),
            Bson::Double(d) if d.is_finite() => Some(*d as i32),
            _ => None,
        }
    })
}

fn parse_objects_bson_full(bytes: &[u8]) -> (Vec<SceneObject>, u32) {
    if bytes.len() > 32 * 1024 * 1024 {
        return (default_scene_objects(), 0);
    }
    let Ok(doc) = bson::from_slice::<Document>(bytes) else {
        return (default_scene_objects(), 0);
    };
    let objects = parse_objects_from_document(&doc).unwrap_or_else(default_scene_objects);
    let active = doc_u32(&doc, "activeObjectId").unwrap_or(0);
    (objects, active)
}

/// Stream voxels from BSON without deserializing the full document into `Document` / `Bson`.
fn parse_bson_full_raw(bytes: &[u8]) -> Result<VoxelleFile, ParseError> {
    let doc = RawDocument::from_bytes(bytes).map_err(|e| ParseError::RawBson(e.to_string()))?;
    let version = raw_doc_i32(doc, "version")?;
    let grid_size = raw_doc_i32(doc, "gridSize")?;
    if grid_size < 1 {
        return Err(ParseError::InvalidDocument);
    }
    let scene = parse_scene_raw(doc);
    let voxels_arr = doc
        .get_array("voxels")
        .map_err(|e| ParseError::RawBson(e.to_string()))?;
    let mut voxels = Vec::new();
    for (i, item) in voxels_arr.into_iter().enumerate() {
        let raw = item.map_err(|e| ParseError::RawBson(e.to_string()))?;
        if let Some(row) = raw.as_array() {
            if let Some(parsed) = parse_voxel_row_raw(row) {
                voxels.push(parsed);
            }
        }
        if i & 0x7fff == 0x7fff {
            std::thread::yield_now();
        }
    }
    let scene_extra = if bytes.len() <= 8 * 1024 * 1024 {
        bson::from_slice::<Document>(bytes)
            .ok()
            .and_then(|d| d.get_document("scene").ok().cloned())
    } else {
        None
    };

    let mood = parse_mood_from_file_bytes(bytes);
    let lighting = parse_lighting_from_file_bytes(bytes);

    let (objects, active_object_id) = parse_objects_bson_full(bytes);

    Ok(VoxelleFile {
        version,
        grid_size,
        scene,
        scene_extra,
        mood,
        lighting,
        voxels,
        objects,
        active_object_id,
    })
}

fn is_v4_file(bytes: &[u8]) -> bool {
    bytes.len() >= 4
        && bytes[0] == V4_MAGIC[0]
        && bytes[1] == V4_MAGIC[1]
        && bytes[2] == V4_MAGIC[2]
        && bytes[3] == V4_MAGIC[3]
}

fn is_v5_file(bytes: &[u8]) -> bool {
    bytes.len() >= 4
        && bytes[0] == V5_MAGIC[0]
        && bytes[1] == V5_MAGIC[1]
        && bytes[2] == V5_MAGIC[2]
        && bytes[3] == V5_MAGIC[3]
}

/// Decompress the inner payload of a V4/V5 container, then parse.
fn parse_v4v5_container(bytes: &[u8], use_zstd: bool) -> Result<VoxelleFile, ParseError> {
    if bytes.len() < 12 {
        return Err(ParseError::InvalidV4);
    }
    let ulen = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let crc_exp = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let tail = &bytes[12..];
    let inner = if use_zstd {
        zstd::decode_all(tail).map_err(ParseError::Zstd)?
    } else {
        decompress_if_gzipped(tail)?
    };
    if inner.len() != ulen {
        return Err(ParseError::InvalidV4);
    }
    let crc = crc32fast::hash(&inner);
    if crc != crc_exp {
        return Err(ParseError::V4CrcMismatch);
    }
    let slice = inner.as_slice();
    if is_v3_wire(slice) {
        parse_v3(slice)
    } else {
        parse_bson_full_raw(slice)
    }
}

fn scene_document_for_encode(file: &VoxelleFile) -> Document {
    let mut d = if let Some(ref ext) = file.scene_extra {
        ext.clone()
    } else {
        let mut d = Document::new();
        if let Some(fl) = file.scene.focal_length_mm {
            d.insert("focalLength", Bson::Double(fl as f64));
        }
        d.insert("orthographic", Bson::Boolean(file.scene.orthographic));
        d
    };
    if let Some(ref mood) = file.mood {
        d.insert("mood", Bson::Document(mood_to_bson_document(mood)));
    }
    if let Some(ref lighting) = file.lighting {
        d.insert(
            "lighting",
            Bson::Document(lighting_to_bson_document(lighting)),
        );
    }
    d
}

fn grid_size_for_encode(file: &VoxelleFile) -> i32 {
    if file.voxels.is_empty() {
        return file.grid_size.max(1);
    }
    let mut max_a = 0i32;
    for v in &file.voxels {
        max_a = max_a.max(v.x.abs()).max(v.y.abs()).max(v.z.abs());
    }
    let extent = max_a * 2 + 1;
    file.grid_size.max(1).max(extent)
}

fn scene_object_to_bson(o: &SceneObject) -> Bson {
    let mut d = Document::new();
    d.insert("id", Bson::Int32(o.id as i32));
    match o.parent_id {
        Some(p) => d.insert("parent", Bson::Int32(p as i32)),
        None => d.insert("parent", Bson::Null),
    };
    d.insert("name", Bson::String(o.name.clone()));
    d.insert("visible", Bson::Boolean(o.visible));
    d.insert("sortOrder", Bson::Int32(o.sort_order));
    d.insert(
        "t",
        Bson::Array(
            o.translation
                .map(|x| Bson::Double(f64::from(x)))
                .into_iter()
                .collect(),
        ),
    );
    d.insert(
        "r",
        Bson::Array(
            o.rotation
                .map(|x| Bson::Double(f64::from(x)))
                .into_iter()
                .collect(),
        ),
    );
    d.insert(
        "s",
        Bson::Array(
            o.scale
                .map(|x| Bson::Double(f64::from(x)))
                .into_iter()
                .collect(),
        ),
    );
    Bson::Document(d)
}

fn objects_array_for_encode(file: &VoxelleFile) -> bson::Array {
    let objs = if file.objects.is_empty() {
        default_scene_objects()
    } else {
        file.objects.clone()
    };
    objs.iter().map(scene_object_to_bson).collect()
}

/// Dense VX3 wire **version 4**: BSON header includes `objects` + `activeObjectId`; each voxel is 24 bytes (`object_id` last).
fn build_v3_wire_payload(file: &VoxelleFile) -> Result<Vec<u8>, EncodeError> {
    const WIRE_VERSION: u32 = 4;
    let grid_size = grid_size_for_encode(file);
    let voxel_count = file.voxels.len() as i32;
    let hidden_count = 0_i32;
    let scene = scene_document_for_encode(file);
    let objects_bson = objects_array_for_encode(file);

    let header = bson::doc! {
        "version": 4_i32,
        "gridSize": grid_size,
        "scene": scene,
        "voxelCount": voxel_count,
        "hiddenCount": hidden_count,
        "objects": objects_bson,
        "activeObjectId": Bson::Int32(file.active_object_id as i32),
    };

    let mut header_bytes = Vec::new();
    header
        .to_writer(&mut header_bytes)
        .map_err(EncodeError::Bson)?;
    let header_len = header_bytes.len() as u32;

    let rec_size = V4_WIRE_RECORD_SIZE;
    let mut out = Vec::with_capacity(12 + header_bytes.len() + file.voxels.len() * rec_size);
    out.extend_from_slice(&V3_MAGIC);
    out.extend_from_slice(&WIRE_VERSION.to_le_bytes());
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(&header_bytes);
    for v in &file.voxels {
        out.extend_from_slice(&v.x.to_le_bytes());
        out.extend_from_slice(&v.y.to_le_bytes());
        out.extend_from_slice(&v.z.to_le_bytes());
        out.extend_from_slice(&(v.color & 0xffffff).to_le_bytes());
        let pad = [v.material.material_index(), 0, 0, 0];
        out.extend_from_slice(&pad);
        out.extend_from_slice(&v.object_id.to_le_bytes());
    }
    Ok(out)
}

fn build_bson_v4_payload(file: &VoxelleFile) -> Result<Vec<u8>, EncodeError> {
    let grid_size = grid_size_for_encode(file);
    let scene = scene_document_for_encode(file);
    let mut voxels_bson = bson::Array::new();
    for v in &file.voxels {
        let mut row = vec![
            Bson::Int32(v.x),
            Bson::Int32(v.y),
            Bson::Int32(v.z),
            Bson::Int32((v.color & 0xffffff) as i32),
            Bson::String(v.material.as_str_id().to_string()),
        ];
        if v.object_id != 0 {
            row.push(Bson::Int32(v.object_id as i32));
        }
        voxels_bson.push(Bson::Array(row));
    }
    let file_meta = bson::doc! {
        "savedAt": chrono::Utc::now().to_rfc3339(),
        "generator": concat!("voxelle-desktop/", env!("CARGO_PKG_VERSION")),
        "documentId": uuid::Uuid::new_v4().to_string(),
    };
    let objects_bson = objects_array_for_encode(file);
    let doc = bson::doc! {
        "version": 4_i32,
        "gridSize": grid_size,
        "voxels": voxels_bson,
        "scene": scene,
        "fileMeta": file_meta,
        "objects": objects_bson,
        "activeObjectId": Bson::Int32(file.active_object_id as i32),
    };
    let mut buf = Vec::new();
    doc.to_writer(&mut buf).map_err(EncodeError::Bson)?;
    Ok(buf)
}

/// Empty scene used for collab welcome when the host has no file open yet (lobby).
pub fn empty_collab_placeholder() -> VoxelleFile {
    VoxelleFile {
        version: 4,
        grid_size: 64,
        scene: Scene {
            focal_length_mm: Some(29.0),
            orthographic: false,
        },
        scene_extra: None,
        mood: None,
        lighting: None,
        voxels: Vec::new(),
        objects: default_scene_objects(),
        active_object_id: 0,
    }
}

/// Encode as **v4 container** (VX4 magic + gzip + CRC32 of uncompressed inner).
/// Kept for backward-compatible exports and collab snapshots; prefer [`encode_payload_v5`] for normal saves.
pub fn encode_payload_v4(file: &VoxelleFile) -> Result<Vec<u8>, EncodeError> {
    let inner = if file.voxels.len() >= V3_WIRE_VOXEL_THRESHOLD {
        build_v3_wire_payload(file)?
    } else {
        build_bson_v4_payload(file)?
    };
    let crc = crc32fast::hash(&inner);
    let ulen = inner.len() as u32;
    let mut gz = GzEncoder::new(Vec::new(), Compression::default());
    gz.write_all(&inner).map_err(EncodeError::Io)?;
    let compressed = gz.finish().map_err(EncodeError::Io)?;

    let mut out = Vec::with_capacity(12 + compressed.len());
    out.extend_from_slice(&V4_MAGIC);
    out.extend_from_slice(&ulen.to_le_bytes());
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&compressed);
    Ok(out)
}

/// Encode as **v5 container** (VX5 magic + zstd + CRC32 of uncompressed inner).
/// Same inner payload as V4 but with zstd compression for 3-5x faster load/save.
pub fn encode_payload_v5(file: &VoxelleFile) -> Result<Vec<u8>, EncodeError> {
    let inner = if file.voxels.len() >= V3_WIRE_VOXEL_THRESHOLD {
        build_v3_wire_payload(file)?
    } else {
        build_bson_v4_payload(file)?
    };
    let crc = crc32fast::hash(&inner);
    let ulen = inner.len() as u32;
    let compressed = zstd::encode_all(inner.as_slice(), 3).map_err(EncodeError::Io)?;

    let mut out = Vec::with_capacity(12 + compressed.len());
    out.extend_from_slice(&V5_MAGIC);
    out.extend_from_slice(&ulen.to_le_bytes());
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&compressed);
    Ok(out)
}

/// After optional gzip/zstd: BSON or v3 wire, or **v4/v5 container** at outer level.
pub fn decode_payload(bytes: &[u8]) -> Result<VoxelleFile, ParseError> {
    if bytes.is_empty() {
        return Err(ParseError::Empty);
    }
    if is_v5_file(bytes) {
        return parse_v4v5_container(bytes, true);
    }
    if is_v4_file(bytes) {
        return parse_v4v5_container(bytes, false);
    }
    let payload = decompress_if_gzipped(bytes)?;
    let slice = payload.as_slice();
    if is_v3_wire(slice) {
        parse_v3(slice)
    } else {
        parse_bson_full_raw(slice)
    }
}
