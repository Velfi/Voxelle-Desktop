//! Criterion benchmarks for CPU smooth meshing (marching cubes + dual contour).
//!
//! Run with: `cargo bench -p voxelle-desktop --bench smooth_mesh`
//!
//! Sphere sizes chosen to bracket the off-thread smooth-mesh threshold:
//!   OFF_THREAD_SMOOTH_MESH_MIN_VOXELS = 4_000 (current), proposed = 1_500.
//!   r=5  → ~524 voxels  (below proposed threshold)
//!   r=7  → ~1437 voxels (near proposed threshold)
//!   r=10 → ~4189 voxels (above current threshold)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use voxelle_desktop_lib::smooth_mesh;
use voxelle_desktop_lib::voxelle::{MaterialId, Voxel};

/// Generate a filled voxel sphere centred at origin with integer radius.
fn voxel_sphere(radius: i32, color: u32) -> Vec<Voxel> {
    let r2 = radius * radius;
    let mut voxels = Vec::new();
    for z in -radius..=radius {
        for y in -radius..=radius {
            for x in -radius..=radius {
                if x * x + y * y + z * z <= r2 {
                    voxels.push(Voxel {
                        x,
                        y,
                        z,
                        color,
                        material: MaterialId::Plastic,
                        object_id: 0,
                    });
                }
            }
        }
    }
    voxels
}

fn bench_marching_cubes(c: &mut Criterion) {
    let mut group = c.benchmark_group("marching_cubes threshold sizes");

    for (radius, label) in [
        (5, "r=5 (~524v)"),
        (7, "r=7 (~1437v)"),
        (10, "r=10 (~4189v)"),
    ] {
        let voxels = voxel_sphere(radius, 0x8899aa);
        group.bench_with_input(BenchmarkId::new("sphere", label), &voxels, |b, v| {
            b.iter(|| smooth_mesh::build_marching_cubes_merged(black_box(v)))
        });
    }
    group.finish();
}

fn bench_dual_contour(c: &mut Criterion) {
    let mut group = c.benchmark_group("dual_contour threshold sizes");
    group.sample_size(20);

    for (radius, label) in [
        (5, "r=5 (~524v)"),
        (7, "r=7 (~1437v)"),
        (10, "r=10 (~4189v)"),
    ] {
        let voxels = voxel_sphere(radius, 0x8899aa);
        group.bench_with_input(BenchmarkId::new("sphere", label), &voxels, |b, v| {
            b.iter(|| smooth_mesh::build_dual_contour_merged(black_box(v)))
        });
    }
    group.finish();
}

/// Multicolor spheres stress the per-bucket split path (more realistic for painted models).
fn bench_marching_cubes_multicolor(c: &mut Criterion) {
    let mut group = c.benchmark_group("marching_cubes multicolor threshold sizes");

    for (radius, label) in [
        (5, "r=5 (~524v)"),
        (7, "r=7 (~1437v)"),
        (10, "r=10 (~4189v)"),
    ] {
        // Alternate colors by octant so each octant is a separate bucket.
        let r2 = radius * radius;
        let voxels: Vec<Voxel> = {
            let mut v = Vec::new();
            let palette = [
                0xff4444u32,
                0x44ff44,
                0x4444ff,
                0xffff44,
                0xff44ff,
                0x44ffff,
                0xffffff,
                0x888888,
            ];
            for z in -radius..=radius {
                for y in -radius..=radius {
                    for x in -radius..=radius {
                        if x * x + y * y + z * z <= r2 {
                            let octant = ((x >= 0) as usize)
                                | (((y >= 0) as usize) << 1)
                                | (((z >= 0) as usize) << 2);
                            v.push(Voxel {
                                x,
                                y,
                                z,
                                color: palette[octant],
                                material: MaterialId::Plastic,
                                object_id: 0,
                            });
                        }
                    }
                }
            }
            v
        };
        group.bench_with_input(BenchmarkId::new("sphere", label), &voxels, |b, v| {
            b.iter(|| smooth_mesh::build_marching_cubes_merged(black_box(v)))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_marching_cubes,
    bench_dual_contour,
    bench_marching_cubes_multicolor,
);
criterion_main!(benches);
