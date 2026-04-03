use super::format::{
    decode_payload, encode_payload_v4, focal_length_to_fov_y_radians, MaterialId, V3_MAGIC,
};

#[test]
fn focal_matches_ts() {
    let mm = 29.0_f32;
    let rad = focal_length_to_fov_y_radians(mm);
    let ts_deg =
        (2.0_f64 * (12.0_f64 / f64::from(mm)).atan() * 180.0 / std::f64::consts::PI) as f32;
    let rust_deg = rad.to_degrees();
    assert!((ts_deg - rust_deg).abs() < 1e-3);
}

#[test]
fn bson_roundtrip_minimal() {
    let doc = bson::doc! {
        "version": 2_i32,
        "gridSize": 8_i32,
        "voxels": [
            [0_i32, 0_i32, 0_i32, 0x00ff00_i32, "plastic"],
            [1_i32, 0_i32, 0_i32, 0xff0000_i32, "metal"],
        ],
        "scene": { "focalLength": 35_f64, "orthographic": false },
    };
    let mut buf = Vec::new();
    doc.to_writer(&mut buf).unwrap();
    let file = decode_payload(&buf).unwrap();
    assert_eq!(file.version, 2);
    assert_eq!(file.grid_size, 8);
    assert_eq!(file.voxels.len(), 2);
    assert_eq!(file.voxels[0].material, MaterialId::Plastic);
    assert_eq!(file.voxels[1].material, MaterialId::Metal);
    assert_eq!(file.scene.focal_length_mm, Some(35.0));
}

#[test]
fn gzip_bson() {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    let doc = bson::doc! {
        "version": 2_i32,
        "gridSize": 4_i32,
        "voxels": [[0_i32, 0_i32, 0_i32, 0xffffff_i32]],
    };
    let mut raw = Vec::new();
    doc.to_writer(&mut raw).unwrap();
    let mut gz = GzEncoder::new(Vec::new(), Compression::default());
    gz.write_all(&raw).unwrap();
    let compressed = gz.finish().unwrap();
    let file = decode_payload(&compressed).unwrap();
    assert_eq!(file.voxels.len(), 1);
}

#[test]
fn mood_scene_roundtrip() {
    use super::format::{MoodSettings, VoxelleFile};
    let file = VoxelleFile {
        version: 4,
        grid_size: 8,
        scene: Default::default(),
        scene_extra: None,
        mood: Some(MoodSettings {
            grain_strength: 0.1,
            vignette: 0.2,
            ..Default::default()
        }),
        lighting: None,
        voxels: vec![],
        objects: super::format::default_scene_objects(),
        active_object_id: 0,
    };
    let bytes = encode_payload_v4(&file).unwrap();
    let back = decode_payload(&bytes).unwrap();
    assert_eq!(back.mood, file.mood);
}

#[test]
fn v4_roundtrip_small() {
    let file = super::format::VoxelleFile {
        version: 4,
        grid_size: 16,
        scene: Default::default(),
        scene_extra: None,
        mood: None,
        lighting: None,
        voxels: vec![super::format::Voxel {
            x: 0,
            y: 0,
            z: 0,
            color: 0x112233,
            material: MaterialId::Metal,
            object_id: 0,
        }],
        objects: super::format::default_scene_objects(),
        active_object_id: 0,
    };
    let bytes = encode_payload_v4(&file).unwrap();
    let back = decode_payload(&bytes).unwrap();
    assert_eq!(back.voxels.len(), 1);
    assert_eq!(back.voxels[0].material, MaterialId::Metal);
}

#[test]
fn v4_objects_roundtrip_bson() {
    use super::format::{SceneObject, VoxelleFile};
    let file = VoxelleFile {
        version: 4,
        grid_size: 8,
        scene: Default::default(),
        scene_extra: None,
        mood: None,
        lighting: None,
        voxels: vec![
            super::format::Voxel {
                x: 0,
                y: 0,
                z: 0,
                color: 0xff0000,
                material: MaterialId::Plastic,
                object_id: 0,
            },
            super::format::Voxel {
                x: 1,
                y: 0,
                z: 0,
                color: 0x00ff00,
                material: MaterialId::Plastic,
                object_id: 1,
            },
        ],
        objects: vec![
            SceneObject {
                id: 0,
                parent_id: None,
                name: "A".into(),
                visible: true,
                sort_order: 0,
                translation: [0.0; 3],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0; 3],
            },
            SceneObject {
                id: 1,
                parent_id: None,
                name: "B".into(),
                visible: true,
                sort_order: 1,
                translation: [2.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0; 3],
            },
        ],
        active_object_id: 1,
    };
    let bytes = encode_payload_v4(&file).unwrap();
    let back = decode_payload(&bytes).unwrap();
    assert_eq!(back.objects.len(), 2);
    assert_eq!(back.active_object_id, 1);
    assert_eq!(back.voxels[1].object_id, 1);
}

/// Pre-object-id desktop builds wrote `wire_version` **4** with **20**-byte records (same as v3 body).
#[test]
fn dense_wire_version_4_legacy_20_byte_record_loads() {
    let header = bson::doc! {
        "version": 4_i32,
        "gridSize": 32_i32,
        "scene": bson::doc! {},
        "voxelCount": 1_i32,
        "hiddenCount": 0_i32,
    };
    let mut header_bytes = Vec::new();
    header.to_writer(&mut header_bytes).unwrap();
    let mut raw = Vec::new();
    raw.extend_from_slice(&V3_MAGIC);
    raw.extend_from_slice(&4u32.to_le_bytes());
    raw.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
    raw.extend_from_slice(&header_bytes);
    raw.extend_from_slice(&0i32.to_le_bytes());
    raw.extend_from_slice(&0i32.to_le_bytes());
    raw.extend_from_slice(&0i32.to_le_bytes());
    raw.extend_from_slice(&0xff0000u32.to_le_bytes());
    raw.extend_from_slice(&[0u8, 0, 0, 0]);

    let file = decode_payload(&raw).unwrap();
    assert_eq!(file.voxels.len(), 1);
    assert_eq!(file.voxels[0].color, 0xff0000);
    assert_eq!(file.voxels[0].object_id, 0);
}

/// A fully-filled 15×15×15 cube (3 375 voxels) must encode below `MAX_AVATAR_FILE_BYTES`
/// (512 KB).  This guards against the collab avatar transfer limit being too tight for
/// dense voxel art at a realistic avatar scale.
#[test]
fn full_15_cube_fits_in_avatar_size_limit() {
    use super::format::{MaterialId, Voxel, VoxelleFile};
    use crate::collab::MAX_AVATAR_FILE_BYTES;

    let voxels: Vec<Voxel> = (0..15)
        .flat_map(|x| (0..15).flat_map(move |y| (0..15).map(move |z| Voxel {
            x,
            y,
            z,
            color: 0xff8800,
            material: MaterialId::Plastic,
            object_id: 0,
        })))
        .collect();
    assert_eq!(voxels.len(), 15 * 15 * 15);

    let file = VoxelleFile {
        version: 4,
        grid_size: 15,
        scene: Default::default(),
        scene_extra: None,
        mood: None,
        lighting: None,
        voxels,
        objects: super::format::default_scene_objects(),
        active_object_id: 0,
    };
    let bytes = encode_payload_v4(&file).unwrap();
    assert!(
        bytes.len() < MAX_AVATAR_FILE_BYTES,
        "encoded size {} bytes exceeds MAX_AVATAR_FILE_BYTES ({})",
        bytes.len(),
        MAX_AVATAR_FILE_BYTES,
    );
}
