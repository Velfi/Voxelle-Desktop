//! Binary GLB export from [`crate::greedy_mesh::MeshBuffers`] (positions, optional normals, indices).

use crate::greedy_mesh::MeshBuffers;

fn pad_json(mut v: Vec<u8>) -> Vec<u8> {
    while v.len() % 4 != 0 {
        v.push(b' ');
    }
    v
}

fn position_bounds(pos: &[f32]) -> Option<([f32; 3], [f32; 3])> {
    if pos.len() < 3 {
        return None;
    }
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for p in pos.chunks_exact(3) {
        for i in 0..3 {
            mn[i] = mn[i].min(p[i]);
            mx[i] = mx[i].max(p[i]);
        }
    }
    Some((mn, mx))
}

fn max_bounds(norm: &[f32]) -> Option<([f32; 3], [f32; 3])> {
    if norm.len() < 3 {
        return None;
    }
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for p in norm.chunks_exact(3) {
        for i in 0..3 {
            mn[i] = mn[i].min(p[i]);
            mx[i] = mx[i].max(p[i]);
        }
    }
    Some((mn, mx))
}

/// Writes GLB 2.0 with a single primitive: POSITION, optional NORMAL, indices.
pub fn mesh_buffers_to_glb(mesh: &MeshBuffers) -> Result<Vec<u8>, String> {
    if mesh.indices.is_empty() || mesh.positions.len() < 9 {
        return Err("empty mesh".into());
    }
    let vertex_count = mesh.positions.len() / 3;
    let Some((pmin, pmax)) = position_bounds(&mesh.positions) else {
        return Err("invalid positions".into());
    };

    let pos_bytes: &[u8] = bytemuck::cast_slice(&mesh.positions);
    let idx_u32: Vec<u32> = mesh.indices.iter().map(|&i| i as u32).collect();
    let idx_bytes: &[u8] = bytemuck::cast_slice(&idx_u32);

    let has_normals = mesh.normals.len() == mesh.positions.len() && mesh.normals.len() >= 9;
    let norm_bytes: &[u8] = if has_normals {
        bytemuck::cast_slice(&mesh.normals)
    } else {
        &[]
    };

    let pos_len = pos_bytes.len();
    let norm_len = if has_normals { norm_bytes.len() } else { 0 };
    let pos_pad = (4 - (pos_len % 4)) % 4;
    let norm_pad = if has_normals {
        (4 - (norm_len % 4)) % 4
    } else {
        0
    };

    let idx_offset = pos_len + pos_pad + norm_len + norm_pad;
    let idx_total_len = idx_bytes.len();
    let idx_pad = (4 - (idx_total_len % 4)) % 4;
    let bin_total = idx_offset + idx_total_len + idx_pad;

    let json = if has_normals {
        let Some((nmin, nmax)) = max_bounds(&mesh.normals) else {
            return Err("invalid normals".into());
        };
        serde_json::json!({
            "asset": { "version": "2.0", "generator": "Voxelle Desktop" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{
                "primitives": [{
                    "attributes": { "POSITION": 0, "NORMAL": 1 },
                    "indices": 2
                }]
            }],
            "accessors": [
                {
                    "bufferView": 0,
                    "byteOffset": 0,
                    "componentType": 5126,
                    "count": vertex_count,
                    "type": "VEC3",
                    "min": [pmin[0], pmin[1], pmin[2]],
                    "max": [pmax[0], pmax[1], pmax[2]]
                },
                {
                    "bufferView": 0,
                    "byteOffset": pos_len + pos_pad,
                    "componentType": 5126,
                    "count": vertex_count,
                    "type": "VEC3",
                    "min": [nmin[0], nmin[1], nmin[2]],
                    "max": [nmax[0], nmax[1], nmax[2]]
                },
                {
                    "bufferView": 0,
                    "byteOffset": idx_offset,
                    "componentType": 5125,
                    "count": idx_u32.len(),
                    "type": "SCALAR"
                }
            ],
            "bufferViews": [{ "buffer": 0, "byteOffset": 0, "byteLength": bin_total }],
            "buffers": [{ "byteLength": bin_total }]
        })
    } else {
        serde_json::json!({
            "asset": { "version": "2.0", "generator": "Voxelle Desktop" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{
                "primitives": [{
                    "attributes": { "POSITION": 0 },
                    "indices": 1
                }]
            }],
            "accessors": [
                {
                    "bufferView": 0,
                    "byteOffset": 0,
                    "componentType": 5126,
                    "count": vertex_count,
                    "type": "VEC3",
                    "min": [pmin[0], pmin[1], pmin[2]],
                    "max": [pmax[0], pmax[1], pmax[2]]
                },
                {
                    "bufferView": 0,
                    "byteOffset": pos_len + pos_pad,
                    "componentType": 5125,
                    "count": idx_u32.len(),
                    "type": "SCALAR"
                }
            ],
            "bufferViews": [{ "buffer": 0, "byteOffset": 0, "byteLength": bin_total }],
            "buffers": [{ "byteLength": bin_total }]
        })
    };

    let mut json_bytes = serde_json::to_vec(&json).map_err(|e| e.to_string())?;
    json_bytes = pad_json(json_bytes);

    let json_chunk_len = json_bytes.len();
    let total_len = 12 + 8 + json_chunk_len + 8 + bin_total;

    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total_len as u32).to_le_bytes());

    out.extend_from_slice(&(json_chunk_len as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json_bytes);

    out.extend_from_slice(&(bin_total as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(pos_bytes);
    for _ in 0..pos_pad {
        out.push(0);
    }
    if has_normals {
        out.extend_from_slice(norm_bytes);
        for _ in 0..norm_pad {
            out.push(0);
        }
    }
    out.extend_from_slice(idx_bytes);
    for _ in 0..idx_pad {
        out.push(0);
    }

    Ok(out)
}
