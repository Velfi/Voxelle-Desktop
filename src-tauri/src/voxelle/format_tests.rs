use super::format::{decode_payload, focal_length_to_fov_y_radians, MaterialId};

#[test]
fn focal_matches_ts() {
    let mm = 29.0_f32;
    let rad = focal_length_to_fov_y_radians(mm);
    let ts_deg = (2.0_f64 * (12.0_f64 / f64::from(mm)).atan() * 180.0 / std::f64::consts::PI) as f32;
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
