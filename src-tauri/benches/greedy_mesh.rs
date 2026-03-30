//! Criterion benchmarks for CPU greedy meshing (`cargo bench -p voxelle-desktop --bench greedy_mesh`).
//!
//! Large cases can take noticeable time; Criterion will shorten iterations automatically.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use voxelle_desktop_lib::greedy_mesh::{self, ChunkKey, SpatialMeshCache};
use voxelle_desktop_lib::voxelle::{MaterialId, Voxel};

fn solid_box(origin: (i32, i32, i32), edge: i32, color: u32) -> Vec<Voxel> {
    let (ox, oy, oz) = origin;
    let e = edge.max(1);
    let mut voxels = Vec::with_capacity((e as usize).pow(3));
    for dz in 0..e {
        for dy in 0..e {
            for dx in 0..e {
                voxels.push(Voxel {
                    x: ox + dx,
                    y: oy + dy,
                    z: oz + dz,
                    color,
                    material: MaterialId::Plastic,
                });
            }
        }
    }
    voxels
}

fn bench_full_mesh(c: &mut Criterion) {
    let v16 = solid_box((0, 0, 0), 16, 0x8899aa);
    let v24 = solid_box((0, 0, 0), 24, 0x8899aa);

    c.bench_function("build_greedy_mesh solid 16³", |b| {
        b.iter(|| greedy_mesh::build_greedy_mesh(black_box(&v16)))
    });
    c.bench_function("build_greedy_mesh solid 24³", |b| {
        b.iter(|| greedy_mesh::build_greedy_mesh(black_box(&v24)))
    });
}

fn bench_mapped_and_chunked(c: &mut Criterion) {
    let voxels = solid_box((0, 0, 0), 20, 0x445566);
    let map = greedy_mesh::voxel_map(&voxels);
    let cs = greedy_mesh::SPATIAL_CHUNK_SIZE;

    c.bench_function("build_greedy_mesh_mapped solid 20³", |b| {
        b.iter(|| greedy_mesh::build_greedy_mesh_mapped(black_box(&voxels), black_box(&map)))
    });

    c.bench_function("build_greedy_mesh_chunked solid 20³", |b| {
        b.iter(|| greedy_mesh::build_greedy_mesh_chunked(black_box(&voxels), cs))
    });
}

fn bench_spatial_cache(c: &mut Criterion) {
    let voxels = solid_box((0, 0, 0), 32, 0xaabbcc);
    let cs = greedy_mesh::SPATIAL_CHUNK_SIZE;

    c.bench_function("SpatialMeshCache::from_voxels solid 32³", |b| {
        b.iter(|| SpatialMeshCache::from_voxels(black_box(&voxels), cs))
    });

    let mut cache = SpatialMeshCache::from_voxels(&voxels, cs).unwrap();
    let add = Voxel {
        x: -1,
        y: 0,
        z: 0,
        color: 0xff0000,
        material: MaterialId::Plastic,
    };
    c.bench_function("SpatialMeshCache apply_add + apply_remove", |b| {
        b.iter(|| {
            cache.apply_add(add, cs);
            cache.apply_remove(add.x, add.y, add.z, cs);
        })
    });
}

fn bench_bucket_scans(c: &mut Criterion) {
    let voxels = solid_box((0, 0, 0), 28, 0x112233);
    let cs = greedy_mesh::SPATIAL_CHUNK_SIZE;

    c.bench_function("voxel_map 28³", |b| {
        b.iter(|| greedy_mesh::voxel_map(black_box(&voxels)))
    });

    c.bench_function("voxel_buckets_by_chunk 28³", |b| {
        b.iter(|| greedy_mesh::voxel_buckets_by_chunk(black_box(&voxels), cs))
    });
}

fn bench_dirty_chunk_remesh(c: &mut Criterion) {
    let voxels = solid_box((0, 0, 0), 32, 0x334455);
    let map = greedy_mesh::voxel_map(&voxels);
    let cs = greedy_mesh::SPATIAL_CHUNK_SIZE;
    let Some((origin, buckets)) = greedy_mesh::voxel_buckets_by_chunk(&voxels, cs) else {
        panic!("buckets");
    };
    let center = greedy_mesh::chunk_key_from_world(16, 16, 16, origin, cs);
    let dirty: Vec<ChunkKey> = greedy_mesh::dirty_chunk_keys_3x3(center);

    c.bench_function("mesh_buffers_for_chunk_key ×27 (dirty 3³)", |b| {
        b.iter(|| {
            for key in &dirty {
                let _ = greedy_mesh::mesh_buffers_for_chunk_key(
                    black_box(&buckets),
                    black_box(&map),
                    *key,
                );
            }
        })
    });
}

criterion_group!(
    benches,
    bench_full_mesh,
    bench_mapped_and_chunked,
    bench_spatial_cache,
    bench_bucket_scans,
    bench_dirty_chunk_remesh,
);
criterion_main!(benches);
