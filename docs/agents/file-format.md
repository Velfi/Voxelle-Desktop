# `.voxelle` file format (v4)

Canonical spec: [`docs/VOXELLE_FORMAT_V4.md`](../VOXELLE_FORMAT_V4.md). Implementation: [`encode_payload_v4`](../../src-tauri/src/voxelle/format.rs) / [`decode_payload`](../../src-tauri/src/voxelle/format.rs).

- **Outer:** `VX4` magic + gzip + CRC32 of the uncompressed inner.
- **Inner (small):** BSON with `version: 4`, voxel rows, `scene`, `objects`, `activeObjectId`, `fileMeta`, etc.
- **Inner (large, ≥ `V3_WIRE_VOXEL_THRESHOLD` voxels):** `VX3` magic + BSON header + dense body. **`wire_version` 3** = 20-byte records. **`wire_version` 4** = 20- or **24**-byte records (decoder picks by body length); 24-byte = 20-byte prefix + `object_id` u32, header includes `objects` and `activeObjectId`. **`wire_version` 5** (briefly used) = treat like 24-byte v4. There is **no** separate wire "v5" in current writers.
