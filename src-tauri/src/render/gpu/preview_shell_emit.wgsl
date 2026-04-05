// Pass 2 of GPU preview shell filter.
//
// Each thread reads one packed voxel, checks its 6 face neighbours in the
// occupancy bitfield, and emits a PreviewInstance pair (solid + wire) into
// the output storage buffers if the voxel is on the surface shell (≥1
// neighbour absent from the set).
//
// The output instance count is written via atomicAdd into the DrawIndexedIndirect
// buffer:  indirect[1]  = solid instance count  (byte offset  4)
//          indirect[6]  = wire  instance count  (byte offset 24)
//
// DrawIndexedIndirect layout (5 × u32 = 20 bytes, two structs back-to-back):
//   [0]  index_count     (solid: written by CPU before dispatch, = 36)
//   [1]  instance_count  (written by this shader via atomicAdd)
//   [2]  first_index     (= 0)
//   [3]  base_vertex     (= 0)
//   [4]  first_instance  (= 0)
//   [5]  index_count     (wire: = 24)
//   [6]  instance_count  (wire, written by this shader)
//   [7..9] first_index / base_vertex / first_instance (= 0)
//
// PreviewInstance layout (must match Rust repr(C) greedy_mesh::PreviewInstance, 80 bytes):
//   model_c0 : vec4<f32>   offset  0
//   model_c1 : vec4<f32>   offset 16
//   model_c2 : vec4<f32>   offset 32
//   model_c3 : vec4<f32>   offset 48
//   color_r  : f32         offset 64
//   color_g  : f32         offset 68
//   color_b  : f32         offset 72
//   mat_kind : f32         offset 76

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
    max_instances:     u32,   // solid/wire instance buffer capacity (bounds-check)
    skip_wire:         u32,   // 1 = suppress wireframe emit
    _pad3:             u32,
}

// Matches greedy_mesh::PreviewInstance (80 bytes, stride 80, align 16).
struct PreviewInstance {
    model_c0: vec4<f32>,
    model_c1: vec4<f32>,
    model_c2: vec4<f32>,
    model_c3: vec4<f32>,
    color_r:  f32,
    color_g:  f32,
    color_b:  f32,
    mat_kind: f32,
}

@group(0) @binding(0) var<uniform>             uniforms:        PreviewUniforms;
@group(0) @binding(1) var<storage, read>       raw_voxels:      array<u32>;
@group(0) @binding(2) var<storage, read>       occupancy:       array<u32>;
@group(0) @binding(3) var<storage, read>       obj_matrices:    array<mat4x4<f32>>;
@group(0) @binding(4) var<storage, read_write> solid_instances: array<PreviewInstance>;
@group(0) @binding(5) var<storage, read_write> wire_instances:  array<PreviewInstance>;
@group(0) @binding(6) var<storage, read_write> indirect:        array<atomic<u32>>;

fn occ_contains(dx: u32, dy: u32, dz: u32) -> bool {
    // Out-of-bbox neighbours are treated as absent (shell).
    if dx >= uniforms.bbox_size.x
    || dy >= uniforms.bbox_size.y
    || dz >= uniforms.bbox_size.z {
        return false;
    }
    let flat = dx * uniforms.bbox_size.y * uniforms.bbox_size.z
             + dy * uniforms.bbox_size.z
             + dz;
    let word = flat / 32u;
    let bit  = flat % 32u;
    return (occupancy[word] >> bit & 1u) != 0u;
}

@compute @workgroup_size(64)
fn shell_emit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= uniforms.voxel_count { return; }

    let packed   = raw_voxels[idx];
    let dx       = (packed      ) & 0x1FFu;
    let dy       = (packed >>  9) & 0x1FFu;
    let dz       = (packed >> 18) & 0x1FFu;
    let obj_idx  = (packed >> 27) & 0xFu;
    let is_ghost = (packed >> 31) & 1u;

    // Shell test: keep only voxels with at least one absent face neighbour.
    // Unsigned subtraction wraps to large values → occ_contains bounds-check catches them.
    let is_shell =
        !occ_contains(dx + 1u, dy,      dz     )
     || !occ_contains(dx - 1u, dy,      dz     )   // wraps to 0xFFFFFFFF if dx==0
     || !occ_contains(dx,      dy + 1u, dz     )
     || !occ_contains(dx,      dy - 1u, dz     )
     || !occ_contains(dx,      dy,      dz + 1u)
     || !occ_contains(dx,      dy,      dz - 1u);

    if !is_shell { return; }

    // Absolute voxel position in world grid.
    let abs_x = f32(uniforms.bbox_min.x + i32(dx));
    let abs_y = f32(uniforms.bbox_min.y + i32(dy));
    let abs_z = f32(uniforms.bbox_min.z + i32(dz));

    // Object-space → world-space model matrix.
    let m = obj_matrices[obj_idx];
    // model = obj_m * translate(abs_x, abs_y, abs_z)
    // c0/1/2 are the rotation+scale columns of obj_m (unchanged by translation).
    // c3 = obj_m * vec4(abs_x, abs_y, abs_z, 1.0)
    let c0 = m[0];
    let c1 = m[1];
    let c2 = m[2];
    let c3 = m * vec4<f32>(abs_x, abs_y, abs_z, 1.0);

    // Choose colour based on ghost flag.
    let sc = select(uniforms.solid_color,       uniforms.solid_ghost_color, is_ghost != 0u);
    let wc = select(uniforms.wire_color,        uniforms.wire_ghost_color,  is_ghost != 0u);

    // Allocate a slot in the solid instance buffer.
    // atomicAdd returns the old value, so if it reaches max_instances the slot
    // is out-of-bounds and we discard the instance rather than writing OOB.
    let solid_slot = atomicAdd(&indirect[1], 1u);
    if solid_slot < uniforms.max_instances {
        solid_instances[solid_slot] = PreviewInstance(c0, c1, c2, c3, sc.x, sc.y, sc.z, sc.w);
    }

    // Wireframe is suppressed for large previews (skip_wire != 0).
    if uniforms.skip_wire == 0u {
        let wire_slot = atomicAdd(&indirect[6], 1u);
        if wire_slot < uniforms.max_instances {
            wire_instances[wire_slot] = PreviewInstance(c0, c1, c2, c3, wc.x, wc.y, wc.z, wc.w);
        }
    }
}
