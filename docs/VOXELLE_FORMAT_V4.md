# Voxelle format v4 (Voxelle Desktop)

This document describes **how Voxelle Desktop writes `.voxelle` files today** and how `decode_payload` reads them. The canonical implementation lives in [`src-tauri/src/voxelle/format.rs`](../src-tauri/src/voxelle/format.rs).

## Role of v4

- **Writers** (Save / Save As / collab snapshot / autosave) use **`encode_payload_v4`**, which wraps an inner payload in a small **VX4 container** with integrity metadata.
- **Readers** use **`decode_payload`**, which accepts:
  - **v4 container** files (VX4 outer),
  - **legacy gzip + BSON** (v1/v2-style),
  - **gzip + v3 wire** (VX3 inner, no outer VX4),
  - uncompressed BSON where applicable.

Legacy **VX3 wire version 3** (20-byte records) and gzip BSON still load. Older **VX3 wire version 4** payloads that used **20-byte** records (an unreleased experiment) are **not** supported; current **wire version 4** means **24-byte** records as below.

## Outer container (VX4)

All v4 **disk** files from Desktop begin with:

| Offset | Size | Content |
|--------|------|---------|
| 0 | 4 | Magic: `0x56 0x58 0x34 0x1a` — ASCII **`VX4`** + `0x1a` |
| 4 | 4 | `inner_len`: `u32` LE — length in bytes of the **uncompressed** inner payload |
| 8 | 4 | `crc32`: `u32` LE — **CRC32** (algorithm: `crc32fast::hash`) over the **uncompressed** inner bytes |
| 12 | * | **gzip** stream of the inner payload |

Decoding steps:

1. Verify magic.
2. Decompress the tail with **gzip** (same path as `decompress_if_gzipped` for inner).
3. Check `inner.len() == inner_len` and `crc32fast::hash(inner) == crc32`.

> **Note:** The container does **not** currently encode a compression enum (e.g. zstd vs gzip). The outer layer is always **gzip** of the inner blob as implemented.

## Inner payload (after gzip decompress)

The uncompressed inner is either:

### A. BSON document (small models)

Used when visible voxel count **&lt;** `V3_WIRE_VOXEL_THRESHOLD` (**50_000**).

Top-level fields (see `build_bson_v4_payload`):

| Field | Type | Meaning |
|-------|------|---------|
| `version` | `i32` | **4** |
| `gridSize` | `i32` | Grid extent; encoder uses `max(|coords|)` and `file.grid_size` (see `grid_size_for_encode`). |
| `voxels` | array of rows | Each row: `[x, y, z, color, material]` — `color` is 24-bit in an `i32`; `material` is a **string** id (see [Materials](#materials)). |
| `scene` | document | Camera-related subset and/or full subtree (see [Scene](#scene)). |
| `fileMeta` | document | Metadata (see [fileMeta](#filemeta)). |

### B. VX3-style wire (large models)

Used when voxel count **≥** 50_000. Inner layout:

| Offset | Size | Content |
|--------|------|---------|
| 0 | 4 | Magic **`VX3`** + `0x1a` |
| 4 | 4 | `wire_version`: `u32` LE — **3** (legacy 20-byte records) or **4** (v4 dense: 24-byte records + objects in header) |
| 8 | 4 | `header_len`: `u32` LE |
| 12 | `header_len` | BSON document (UTF-8 BSON bytes) |
| … | `voxel_count × record_size` | Dense voxel records (see below) |

**Wire version 3** (legacy): `record_size = 20`. Header BSON includes at least: `version`, `gridSize`, `scene`, `voxelCount`, `hiddenCount` — no `objects` / `activeObjectId` in the stripped writer path.

**Wire version 4** (dense): `record_size` is **24** (`V4_WIRE_RECORD_SIZE`) for current writers (per-voxel `object_id`). Header BSON includes **`version`** (4), **`gridSize`**, **`scene`**, **`voxelCount`**, **`hiddenCount`**, **`objects`**, **`activeObjectId`**. Some **unreleased** builds wrote wire version **4** with a **20**-byte body (same layout as wire version 3); **`decode_payload`** infers 20 vs 24 from total body length ÷ (`voxelCount` + `hiddenCount`). Desktop writes **`hiddenCount`: 0** for the dense body; hidden geometry is not round-tripped on save.

**Legacy V3 record** (`V3_RECORD_SIZE = 20` bytes), wire version **3** only:

| Bytes | Field |
|-------|--------|
| 0–3 | `x` `i32` |
| 4–7 | `y` `i32` |
| 8–11 | `z` `i32` |
| 12–15 | `color` `u32` (lower 24 bits used) |
| 16 | `material` index `u8` (see [Materials](#materials)) |
| 17–19 | padding `0` |

**V4 dense record** (`V4_WIRE_RECORD_SIZE = 24` bytes), wire version **4**:

| Bytes | Field |
|-------|--------|
| 0–19 | Same as legacy V3 record above |
| 20–23 | `object_id` `u32` LE |

## fileMeta

Present on **BSON inner** path only (`build_bson_v4_payload`):

| Field | Meaning |
|-------|---------|
| `savedAt` | RFC3339 UTC timestamp (`chrono::Utc::now().to_rfc3339()`). |
| `generator` | `voxelle-desktop/<CARGO_PKG_VERSION>`. |
| `documentId` | New **UUID v4** string on each save (not stable across saves). |

Optional fields from the broader spec (e.g. `collabSessionId`) are not written by Desktop today.

## Scene

- **`Scene`** in memory holds `focal_length_mm`, `orthographic`, and optionally **`scene_extra`**: a full BSON `scene` subdocument preserved from load (e.g. web-authored **`atmosphere`**).
- On encode, if `scene_extra` is set, it is written as the `scene` document; otherwise a minimal document is built from `Scene` (`focalLength`, `orthographic`).
- Readers populate `scene_extra` when parsing BSON from smaller files; large wire paths take `scene` from the v3 header document when present.

## Materials

String ids in BSON rows and index in v3 wire must match:

`plastic`, `metal`, `rubber`, `glass`, `water`, `glow`

(See `MaterialId::as_str_id` / `from_str_id` / `material_index` in `format.rs`.)

## Integrity

- **Outer CRC32** (v4 container): detects truncation or corruption after decompress (matches logical inner bytes before gzip).
- **gzip** has its own CRC; both apply when the outer segment is gzipped.

## Compatibility summary

| Source | Detection |
|--------|-----------|
| v4 file | First 4 bytes = VX4 magic → parse container → inner BSON or VX3 wire. |
| Legacy | Not VX4 → optional gzip decompress → VX3 magic at start of payload, else BSON. |

Wire version **3** (20-byte records) and **4** (24-byte records + object ids) are accepted (`parse_v3`).

## Related

- A broader ecosystem spec may live alongside the web app (e.g. `VOXELLE_FORMAT.md`); Desktop behavior is defined by this file and `format.rs`.
- Implementation: `encode_payload_v4`, `decode_payload`, `parse_v4_container`, `build_bson_v4_payload`, `build_v3_wire_payload` in [`src-tauri/src/voxelle/format.rs`](../src-tauri/src/voxelle/format.rs).
