//! Criterion benchmarks for preview mesh generation (`cargo bench -p voxelle-desktop --bench preview_generation`).
//!
//! Tests the CPU loop that generates 25k individual cube meshes for stroke previews.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use voxelle_desktop_lib::greedy_mesh::{self, MeshBuffers, VoxelCoord};

/// Generate a sparse set of voxel coordinates (mimics brush footprint).
fn sparse_voxel_coords(count: usize) -> Vec<VoxelCoord> {
    let mut coords = Vec::with_capacity(count);
    let mut idx = 0usize;
    while coords.len() < count {
        let x = (idx / 10) as i32;
        let y = ((idx / 100) % 10) as i32;
        let z = (idx % 10) as i32;
        coords.push((x, y, z));
        idx += 1;
    }
    coords
}

/// Simulate the core loop: generate individual cube meshes, transform, and append.
/// This mirrors `stroke_preview_meshes_for_union` behavior.
fn generate_preview_cubes(coords: &[VoxelCoord]) -> (MeshBuffers, MeshBuffers) {
    let mut solid = MeshBuffers::default();
    let mut wire = MeshBuffers::default();

    for &(cx, cy, cz) in coords {
        let s = greedy_mesh::preview_cube_mesh(
            cx as f32,
            cy as f32,
            cz as f32,
            0.53,
            [1.0, 0.5, 0.3],
            1.0,
        );
        let w = greedy_mesh::preview_cube_wireframe_mesh(
            cx as f32,
            cy as f32,
            cz as f32,
            0.53,
            [0.8, 0.4, 0.2],
            2.0,
        );
        greedy_mesh::append_mesh_buffers(&mut solid, s);
        greedy_mesh::append_mesh_buffers(&mut wire, w);
    }

    (solid, wire)
}

fn bench_preview_cube_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("preview_cube_generation");

    // Baseline: small preview (1k cubes)
    let coords_1k = sparse_voxel_coords(1_000);
    group.bench_function("1_000 cubes", |b| {
        b.iter(|| generate_preview_cubes(black_box(&coords_1k)))
    });

    // Medium: typical large brush (10k cubes)
    let coords_10k = sparse_voxel_coords(10_000);
    group.bench_function("10_000 cubes", |b| {
        b.iter(|| generate_preview_cubes(black_box(&coords_10k)))
    });

    // Max: current limit (25k cubes)
    let coords_25k = sparse_voxel_coords(25_000);
    group.bench_function("25_000 cubes (MAX)", |b| {
        b.iter(|| generate_preview_cubes(black_box(&coords_25k)))
    });

    group.finish();
}

fn bench_single_cube_generation(c: &mut Criterion) {
    c.bench_function("single preview_cube_mesh + wireframe", |b| {
        b.iter(|| {
            let _s = greedy_mesh::preview_cube_mesh(5.0, 5.0, 5.0, 0.53, [1.0, 0.5, 0.3], 1.0);
            let _w =
                greedy_mesh::preview_cube_wireframe_mesh(5.0, 5.0, 5.0, 0.53, [0.8, 0.4, 0.2], 2.0);
        })
    });
}

/// Simulate the new GPU-instanced path: only generate instance data (no per-voxel meshes).
fn generate_preview_instances(
    coords: &[VoxelCoord],
) -> (
    Vec<greedy_mesh::PreviewInstance>,
    Vec<greedy_mesh::PreviewInstance>,
) {
    let _model_cols = glam::Mat4::IDENTITY.to_cols_array_2d();
    let mut solid = Vec::with_capacity(coords.len());
    let mut wire = Vec::with_capacity(coords.len());
    for &(cx, cy, cz) in coords {
        let translate =
            glam::Mat4::from_translation(glam::Vec3::new(cx as f32, cy as f32, cz as f32));
        let cols = translate.to_cols_array_2d();
        solid.push(greedy_mesh::PreviewInstance {
            model_c0: cols[0],
            model_c1: cols[1],
            model_c2: cols[2],
            model_c3: cols[3],
            color: [1.0, 0.5, 0.3],
            mat_kind: 1.0,
        });
        wire.push(greedy_mesh::PreviewInstance {
            model_c0: cols[0],
            model_c1: cols[1],
            model_c2: cols[2],
            model_c3: cols[3],
            color: [0.8, 0.4, 0.2],
            mat_kind: 2.0,
        });
    }
    (solid, wire)
}

fn bench_preview_instanced_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("preview_instanced_generation");

    let coords_1k = sparse_voxel_coords(1_000);
    group.bench_function("1_000 instances", |b| {
        b.iter(|| generate_preview_instances(black_box(&coords_1k)))
    });

    let coords_10k = sparse_voxel_coords(10_000);
    group.bench_function("10_000 instances", |b| {
        b.iter(|| generate_preview_instances(black_box(&coords_10k)))
    });

    let coords_25k = sparse_voxel_coords(25_000);
    group.bench_function("25_000 instances (MAX)", |b| {
        b.iter(|| generate_preview_instances(black_box(&coords_25k)))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_cube_generation,
    bench_preview_cube_generation,
    bench_preview_instanced_generation,
);
criterion_main!(benches);
