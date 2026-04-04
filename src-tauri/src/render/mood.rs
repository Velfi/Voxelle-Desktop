//! Mood effect parameters (vignette, grain, atmosphere, distance tint, sun shafts, SSR, bloom).

/// sRGB hex (#rrggbb or rrggbb) → linear RGB (0–1).
pub(crate) fn hex_to_linear_rgb(hex: &str) -> (f32, f32, f32) {
    let s = hex.trim_start_matches('#');
    let n = u32::from_str_radix(s, 16).unwrap_or(0xffffff);
    let srgb = |c: u32| -> f32 {
        let v = (c & 0xff) as f32 / 255.0;
        v.powf(2.2)
    };
    (srgb(n >> 16), srgb(n >> 8), srgb(n))
}

/// All mood effect parameters sent from the React frontend.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MoodParams {
    // vignette (desktop-only)
    #[serde(default)]
    pub vignette: f32,
    // grain
    #[serde(default)]
    pub grain_enabled: bool,
    #[serde(default = "default_grain_strength")]
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
    #[serde(default = "default_atm_thickness")]
    pub atm_thickness: f32,
    #[serde(default = "default_atm_density")]
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
    #[serde(default = "default_atm_height_falloff")]
    pub atm_height_falloff: f32,
    #[serde(default)]
    pub atm_drift_enabled: bool,
    #[serde(default = "default_drift_amount")]
    pub atm_drift_amount: f32,
    #[serde(default = "default_drift_scale")]
    pub atm_drift_scale: f32,
    #[serde(default = "default_drift_speed")]
    pub atm_drift_speed: f32,
    // distance tint
    #[serde(default)]
    pub dt_enabled: bool,
    #[serde(default = "default_dt_near_color")]
    pub dt_near_color: String,
    #[serde(default = "default_dt_mid_color")]
    pub dt_mid_color: String,
    #[serde(default = "default_dt_far_color")]
    pub dt_far_color: String,
    #[serde(default = "default_dt_near_dist")]
    pub dt_near_dist: f32,
    #[serde(default = "default_dt_far_dist")]
    pub dt_far_dist: f32,
    #[serde(default = "default_dt_strength")]
    pub dt_strength: f32,
    // sun shafts
    #[serde(default)]
    pub ss_enabled: bool,
    #[serde(default = "default_ss_strength")]
    pub ss_strength: f32,
    #[serde(default = "default_ss_decay")]
    pub ss_decay: f32,
    #[serde(default = "default_ss_density")]
    pub ss_density: f32,
    #[serde(default = "default_ss_weight")]
    pub ss_weight: f32,
    #[serde(default = "default_ss_samples")]
    pub ss_samples: f32,
    // screen-space reflections
    #[serde(default)]
    pub ssr_enabled: bool,
    #[serde(default = "default_ssr_strength")]
    pub ssr_strength: f32,
    // bloom
    #[serde(default = "default_bloom_strength")]
    pub bloom_strength: f32,
}

fn default_true() -> bool {
    true
}
fn default_one() -> f32 {
    1.0
}
fn default_grain_strength() -> f32 {
    0.12
}
fn default_atm_color() -> String {
    "#c8d4e0".into()
}
fn default_atm_thickness() -> f32 {
    28.0
}
fn default_atm_density() -> f32 {
    0.85
}
fn default_atm_height_falloff() -> f32 {
    120.0
}
fn default_drift_amount() -> f32 {
    0.2
}
fn default_drift_scale() -> f32 {
    0.02
}
fn default_drift_speed() -> f32 {
    0.2
}
fn default_dt_near_color() -> String {
    "#ffffff".into()
}
fn default_dt_mid_color() -> String {
    "#c8d4e0".into()
}
fn default_dt_far_color() -> String {
    "#8fa3bf".into()
}
fn default_dt_near_dist() -> f32 {
    16.0
}
fn default_dt_far_dist() -> f32 {
    140.0
}
fn default_dt_strength() -> f32 {
    0.6
}
fn default_ss_strength() -> f32 {
    0.7
}
fn default_ss_decay() -> f32 {
    0.92
}
fn default_ss_density() -> f32 {
    0.8
}
fn default_ss_weight() -> f32 {
    0.6
}
fn default_ss_samples() -> f32 {
    32.0
}
fn default_ssr_strength() -> f32 {
    0.8
}
fn default_bloom_strength() -> f32 {
    0.1
}

impl Default for MoodParams {
    fn default() -> Self {
        Self {
            vignette: 0.0,
            grain_enabled: false,
            grain_strength: default_grain_strength(),
            grain_animated: true,
            grain_speed: 1.0,
            grain_colorful: true,
            atm_enabled: false,
            atm_color: default_atm_color(),
            atm_thickness: default_atm_thickness(),
            atm_density: default_atm_density(),
            atm_aerial: true,
            atm_positive_side: false,
            atm_plane_nx: 0.0,
            atm_plane_ny: 0.0,
            atm_plane_nz: 0.0,
            atm_plane_c: 0.0,
            atm_height_bias: 0.0,
            atm_height_falloff: default_atm_height_falloff(),
            atm_drift_enabled: false,
            atm_drift_amount: default_drift_amount(),
            atm_drift_scale: default_drift_scale(),
            atm_drift_speed: default_drift_speed(),
            dt_enabled: false,
            dt_near_color: default_dt_near_color(),
            dt_mid_color: default_dt_mid_color(),
            dt_far_color: default_dt_far_color(),
            dt_near_dist: default_dt_near_dist(),
            dt_far_dist: default_dt_far_dist(),
            dt_strength: default_dt_strength(),
            ss_enabled: false,
            ss_strength: default_ss_strength(),
            ss_decay: default_ss_decay(),
            ss_density: default_ss_density(),
            ss_weight: default_ss_weight(),
            ss_samples: default_ss_samples(),
            ssr_enabled: false,
            ssr_strength: default_ssr_strength(),
            bloom_strength: default_bloom_strength(),
        }
    }
}
