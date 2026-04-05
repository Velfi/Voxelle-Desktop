//! Voxelle `.voxelle` parsing: optional gzip, BSON v1/v2, or v3 wire.

mod format;
pub mod scene;
pub mod start_shape;

pub use format::parse_mood_from_scene_optional;
pub use format::{
    decode_payload, default_scene_objects, empty_collab_placeholder, encode_payload_v4,
    encode_payload_v5, focal_length_to_fov_y_radians, hex_srgb_to_linear_rgb3,
    rgb24_u32_to_linear_rgb3, EncodeError, LightingSettings, MaterialId, MoodSettings, Scene,
    SceneObject, Voxel, VoxelleFile,
};
pub use scene::object_world_matrix;

#[cfg(test)]
mod format_tests;

#[cfg(test)]
mod fixture_tests;
