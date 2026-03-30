//! Integration tests against real `.voxelle` fixtures in `../.test-files/`.

use super::format::decode_payload;
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../.test-files")
        .join(name)
}

#[test]
fn cemetery_voxelle_decodes() {
    let p = fixture("Cemetery.voxelle");
    assert!(
        p.exists(),
        "missing fixture {} — add .test-files/Cemetery.voxelle",
        p.display()
    );
    let bytes = std::fs::read(&p).expect("read Cemetery.voxelle");
    let file = decode_payload(&bytes).expect("decode Cemetery.voxelle");
    assert!(file.grid_size >= 1, "gridSize");
    assert!(
        file.voxels.len() >= 10_000,
        "expected large visible voxel set, got {}",
        file.voxels.len()
    );
}

#[test]
fn plains_landscape_voxelle_decodes() {
    let p = fixture("PlainsLandscape.voxelle");
    assert!(
        p.exists(),
        "missing fixture {} — add .test-files/PlainsLandscape.voxelle",
        p.display()
    );
    let bytes = std::fs::read(&p).expect("read PlainsLandscape.voxelle");
    let file = decode_payload(&bytes).expect("decode PlainsLandscape.voxelle");
    assert!(file.grid_size >= 1, "gridSize");
    assert!(
        file.voxels.len() >= 1_000,
        "expected substantial voxel count, got {}",
        file.voxels.len()
    );
}
