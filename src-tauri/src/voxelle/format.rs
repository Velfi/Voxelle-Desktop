use bson::raw::{RawArray, RawBsonRef, RawDocument};
use bson::{Bson, Document};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::io::{Read, Write};
use thiserror::Error;

pub const V3_MAGIC: [u8; 4] = [0x56, 0x58, 0x33, 0x1a];
pub const V4_MAGIC: [u8; 4] = [0x56, 0x58, 0x34, 0x1a];
pub const V3_RECORD_SIZE: usize = 20;
/// Dense **VX3 wire version 4** body: `object_id` `u32` after the legacy 20-byte fields (24 bytes total).
pub const V4_WIRE_RECORD_SIZE: usize = 24;
/// Use dense v3-style body when at least this many voxels (matches web / prior tooling).
pub const V3_WIRE_VOXEL_THRESHOLD: usize = 50_000;

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

    pub fn material_index(self) -> u8 {
        match self {
            MaterialId::Plastic => 0,
            MaterialId::Metal => 1,
            MaterialId::Rubber => 2,
            MaterialId::Glass => 3,
            MaterialId::Water => 4,
            MaterialId::Glow => 5,
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

#[derive(Clone, Debug)]
pub struct VoxelleFile {
    #[allow(dead_code)]
    pub version: i32,
    #[allow(dead_code)]
    pub grid_size: i32,
    pub scene: Scene,
    /// Full `scene` subdocument when loaded (preserves `atmosphere` etc.). If `Some`, encode prefers this over [`Scene`] alone.
    pub scene_extra: Option<Document>,
    pub voxels: Vec<Voxel>,
    pub objects: Vec<SceneObject>,
    /// Target object for new voxel edits (persisted in BSON; default 0).
    pub active_object_id: u32,
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

    let mut voxels = Vec::with_capacity(voxel_count as usize);
    let mut o = 12 + header_len;
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
    // Skip hidden voxels (viewer policy: visible only)
    Ok(VoxelleFile {
        version: file_version,
        grid_size,
        scene,
        scene_extra,
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

    let (objects, active_object_id) = parse_objects_bson_full(bytes);

    Ok(VoxelleFile {
        version,
        grid_size,
        scene,
        scene_extra,
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

fn parse_v4_container(bytes: &[u8]) -> Result<VoxelleFile, ParseError> {
    if bytes.len() < 12 {
        return Err(ParseError::InvalidV4);
    }
    let ulen = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let crc_exp = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let tail = &bytes[12..];
    let inner = decompress_if_gzipped(tail)?;
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
    if let Some(ref ext) = file.scene_extra {
        return ext.clone();
    }
    let mut d = Document::new();
    if let Some(fl) = file.scene.focal_length_mm {
        d.insert("focalLength", Bson::Double(fl as f64));
    }
    d.insert("orthographic", Bson::Boolean(file.scene.orthographic));
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
        voxels: Vec::new(),
        objects: default_scene_objects(),
        active_object_id: 0,
    }
}

/// Encode as **v4 container** (VX4 magic + gzip + CRC32 of uncompressed inner). Inner is BSON or VX3 dense wire v4.
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

/// After optional gzip: BSON or v3 wire, or **v4 container** at outer level.
pub fn decode_payload(bytes: &[u8]) -> Result<VoxelleFile, ParseError> {
    if bytes.is_empty() {
        return Err(ParseError::Empty);
    }
    if is_v4_file(bytes) {
        return parse_v4_container(bytes);
    }
    let payload = decompress_if_gzipped(bytes)?;
    let slice = payload.as_slice();
    if is_v3_wire(slice) {
        parse_v3(slice)
    } else {
        parse_bson_full_raw(slice)
    }
}
