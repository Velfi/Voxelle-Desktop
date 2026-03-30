use super::format::{decode_payload, encode_payload_v4, focal_length_to_fov_y_radians, MaterialId};

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
fn v4_roundtrip_small() {
    let file = super::format::VoxelleFile {
        version: 4,
        grid_size: 16,
        scene: Default::default(),
        scene_extra: None,
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
