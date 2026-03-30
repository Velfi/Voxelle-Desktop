use bson::raw::{RawArray, RawBsonRef, RawDocument};
use bson::Document;
use flate2::read::GzDecoder;
use std::io::Read;
use thiserror::Error;

pub const V3_MAGIC: [u8; 4] = [0x56, 0x58, 0x33, 0x1a];
pub const V3_RECORD_SIZE: usize = 20;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("empty payload")]
    Empty,
    #[error("gzip decompress: {0}")]
    Gzip(std::io::Error),
    #[error("invalid v3 wire")]
    InvalidV3,
    #[error("bson: {0}")]
    Bson(bson::de::Error),
    #[error("raw bson: {0}")]
    RawBson(String),
    #[error("missing required fields")]
    InvalidDocument,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialId {
    Plastic,
    Metal,
    Rubber,
    Glass,
    Water,
    Glow,
}

impl MaterialId {
    pub fn from_index(i: u8) -> Self {
        match i {
            0 => MaterialId::Plastic,
            1 => MaterialId::Metal,
            2 => MaterialId::Rubber,
            3 => MaterialId::Glass,
            4 => MaterialId::Water,
            _ => MaterialId::Glow,
        }
    }

    pub fn from_str_id(s: &str) -> Self {
        match s {
            "metal" => MaterialId::Metal,
            "rubber" => MaterialId::Rubber,
            "glass" => MaterialId::Glass,
            "water" => MaterialId::Water,
            "glow" => MaterialId::Glow,
            _ => MaterialId::Plastic,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Voxel {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub color: u32,
    pub material: MaterialId,
}

#[derive(Clone, Debug, Default)]
pub struct Scene {
    pub focal_length_mm: Option<f32>,
    pub orthographic: bool,
}

#[derive(Clone, Debug)]
pub struct VoxelleFile {
    #[allow(dead_code)]
    pub version: i32,
    #[allow(dead_code)]
    pub grid_size: i32,
    pub scene: Scene,
    pub voxels: Vec<Voxel>,
}

/// Match `focalLengthToFov` in Voxelle `sceneSetup.ts`.
pub fn focal_length_to_fov_y_radians(mm: f32) -> f32 {
    2.0 * (12.0_f32 / mm).atan()
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

fn parse_v3(bytes: &[u8]) -> Result<VoxelleFile, ParseError> {
    if bytes.len() < 16 {
        return Err(ParseError::InvalidV3);
    }
    let wire_ver = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if wire_ver != 3 {
        return Err(ParseError::InvalidV3);
    }
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
    let body_len = (voxel_count + hidden_count) as usize * V3_RECORD_SIZE;
    if 12 + header_len + body_len != bytes.len() {
        return Err(ParseError::InvalidV3);
    }

    let scene = parse_scene_bson(&doc);
    let mut voxels = Vec::with_capacity(voxel_count as usize);
    let mut o = 12 + header_len;
    for i in 0..voxel_count {
        let v = read_v3_record(bytes, o)?;
        voxels.push(v);
        o += V3_RECORD_SIZE;
        if i & 0x7fff == 0x7fff {
            std::thread::yield_now();
        }
    }
    // Skip hidden voxels (viewer policy: visible only)
    Ok(VoxelleFile {
        version: 3,
        grid_size,
        scene,
        voxels,
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
    })
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

fn raw_doc_i32(doc: &RawDocument, key: &str) -> Result<i32, ParseError> {
    let Some(v) = doc.get(key).map_err(|e| ParseError::RawBson(e.to_string()))? else {
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
    Some(Voxel {
        x,
        y,
        z,
        color,
        material,
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
    Ok(VoxelleFile {
        version,
        grid_size,
        scene,
        voxels,
    })
}

/// After optional gzip: BSON or v3 wire.
pub fn decode_payload(bytes: &[u8]) -> Result<VoxelleFile, ParseError> {
    if bytes.is_empty() {
        return Err(ParseError::Empty);
    }
    let payload = decompress_if_gzipped(bytes)?;
    let slice = payload.as_slice();
    if is_v3_wire(slice) {
        parse_v3(slice)
    } else {
        parse_bson_full_raw(slice)
    }
}
