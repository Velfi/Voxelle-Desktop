//! Voxelle `.voxelle` parsing: optional gzip, BSON v1/v2, or v3 wire.

mod format;
pub mod start_shape;

pub use format::{
    decode_payload, empty_collab_placeholder, encode_payload_v4, focal_length_to_fov_y_radians,
    EncodeError, MaterialId, Voxel, VoxelleFile,
};

#[cfg(test)]
mod format_tests;

#[cfg(test)]
mod fixture_tests;
