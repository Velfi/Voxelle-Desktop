// Pass 1 of GPU preview shell filter.
//
// Each thread reads one packed voxel and sets the corresponding bit in the
// flat occupancy bitfield.  The bitfield is a dense u32 array covering the
// stroke bounding box:  index = dx * (bbox_size.y * bbox_size.z) + dy * bbox_size.z + dz
//
// Packed voxel layout (u32):
//   bits  0– 8  dx   (relative x in [0, bbox_size.x-1])
//   bits  9–17  dy
//   bits 18–26  dz
//   bits 27–30  object index  (unused in this pass)
//   bit    31   is_ghost      (unused in this pass)

struct PreviewUniforms {
    bbox_min:          vec3<i32>,
    voxel_count:       u32,
    bbox_size:         vec3<u32>,
    _pad0:             u32,
    solid_color:       vec4<f32>,
    solid_ghost_color: vec4<f32>,
    wire_color:        vec4<f32>,
    wire_ghost_color:  vec4<f32>,
    cube_half:         f32,
    max_instances:     u32,   // solid/wire instance buffer capacity (shader bounds-check)
    skip_wire:         u32,   // 1 = suppress wireframe emit in shell_emit pass
    _pad3:             u32,
}

@group(0) @binding(0) var<uniform>              uniforms:   PreviewUniforms;
@group(0) @binding(1) var<storage, read>        raw_voxels: array<u32>;
@group(0) @binding(2) var<storage, read_write>  occupancy:  array<atomic<u32>>;

@compute @workgroup_size(64)
fn fill_occ(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= uniforms.voxel_count { return; }

    let packed = raw_voxels[idx];
    let dx = (packed      ) & 0x1FFu;
    let dy = (packed >>  9) & 0x1FFu;
    let dz = (packed >> 18) & 0x1FFu;

    let flat = dx * uniforms.bbox_size.y * uniforms.bbox_size.z
             + dy * uniforms.bbox_size.z
             + dz;
    let word = flat / 32u;
    let bit  = flat % 32u;
    atomicOr(&occupancy[word], 1u << bit);
}
