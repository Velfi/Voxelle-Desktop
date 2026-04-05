//! Criterion benchmarks for CPU greedy meshing (`cargo bench -p voxelle-desktop --bench greedy_mesh`).
//!
//! Large cases can take noticeable time; Criterion will shorten iterations automatically.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::collections::BTreeMap;
use voxelle_desktop_lib::greedy_mesh::{self, ChunkKey, MeshBuffers, SpatialMeshCache};
use voxelle_desktop_lib::voxelle::{default_scene_objects, MaterialId, SceneObject, Voxel};

/// Empty slice: [`greedy_mesh::build_greedy_mesh`] uses default scene objects when `objects` is empty.
const NO_OBJECTS: &[SceneObject] = &[];

/// Same spatial cache as the fused loader, then **sequential** per-chunk greedy mesh (baseline before rayon).
fn sequential_chunk_meshes_after_cache(
    voxels: &[Voxel],
    cs: i32,
) -> Option<(
    (i32, i32, i32),
    BTreeMap<ChunkKey, MeshBuffers>,
    SpatialMeshCache,
)> {
    let cache = SpatialMeshCache::from_voxels(voxels, cs)?;
    let origin = cache.origin;
    let mut out = BTreeMap::new();
    for &key in cache.buckets.keys() {
        let mesh = greedy_mesh::mesh_buffers_for_chunk_key(&cache.buckets, &cache.occupancy, key);
        if !mesh.indices.is_empty() {
            out.insert(key, mesh);
        }
    }
    Some((origin, out, cache))
}

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
                    object_id: 0,
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
        b.iter(|| greedy_mesh::build_greedy_mesh(black_box(&v16), black_box(NO_OBJECTS)))
    });
    c.bench_function("build_greedy_mesh solid 24³", |b| {
        b.iter(|| greedy_mesh::build_greedy_mesh(black_box(&v24), black_box(NO_OBJECTS)))
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
        b.iter(|| {
            greedy_mesh::build_greedy_mesh_chunked(black_box(&voxels), cs, black_box(NO_OBJECTS))
        })
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
        object_id: 0,
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

/// Fused `SpatialMeshCache` + parallel chunk meshes vs the same cache + sequential chunk meshes.
fn bench_load_chunk_meshes_fused_vs_sequential(c: &mut Criterion) {
    // 64³ ≈ 262k voxels; multiple spatial chunks so parallel meshing has work to split.
    let voxels = solid_box((0, 0, 0), 64, 0x8899aa);
    let cs = greedy_mesh::SPATIAL_CHUNK_SIZE;

    let mut group = c.benchmark_group("load_chunk_meshes 64³ solid");
    group.sample_size(15);
    group.bench_function("fused_parallel", |b| {
        b.iter(|| {
            greedy_mesh::build_chunk_meshes_and_spatial_cache(black_box(&voxels), cs, |_, _, _| {})
        })
    });
    group.bench_function("sequential_chunk_meshes", |b| {
        b.iter(|| sequential_chunk_meshes_after_cache(black_box(&voxels), cs))
    });
    group.finish();
}

fn bench_dirty_chunk_remesh(c: &mut Criterion) {
    let voxels = solid_box((0, 0, 0), 32, 0x334455);
    let cs = greedy_mesh::SPATIAL_CHUNK_SIZE;
    let cache = SpatialMeshCache::from_voxels(&voxels, cs).expect("cache");
    let center = greedy_mesh::chunk_key_from_world(16, 16, 16, cache.origin, cs);
    let dirty: Vec<ChunkKey> = greedy_mesh::dirty_chunk_keys_3x3(center);

    c.bench_function("mesh_buffers_for_chunk_key ×27 (dirty 3³)", |b| {
        b.iter(|| {
            for key in &dirty {
                let _ = greedy_mesh::mesh_buffers_for_chunk_key(
                    black_box(&cache.buckets),
                    black_box(&cache.occupancy),
                    *key,
                );
            }
        })
    });
}

fn bench_mesh_bounds(c: &mut Criterion) {
    let voxels = solid_box((0, 0, 0), 64, 0x8899aa);
    let objs = default_scene_objects();
    c.bench_function("mesh_bounds_from_voxels 64³ solid", |b| {
        b.iter(|| greedy_mesh::mesh_bounds_from_voxels(black_box(&voxels)))
    });
    c.bench_function(
        "mesh_bounds_from_voxels_world 64³ solid (identity objects)",
        |b| {
            b.iter(|| {
                greedy_mesh::mesh_bounds_from_voxels_world(
                    black_box(&voxels),
                    black_box(objs.as_slice()),
                )
            })
        },
    );
}

/// Build the `AHashMap<VoxelCoord, Voxel>` occupancy map that
/// `voxel_surface_grid_line_vertices` expects (mirrors `prepare_grid_border_overlay`).
fn occupancy_from_voxels(voxels: &[Voxel]) -> ahash::AHashMap<greedy_mesh::VoxelCoord, Voxel> {
    let mut m = ahash::AHashMap::with_capacity(voxels.len());
    for v in voxels {
        m.insert((v.x, v.y, v.z), *v);
    }
    m
}

fn bench_grid_border_lines(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_border_lines");

    // 16³ = 4 096 voxels — small model
    let v16 = solid_box((0, 0, 0), 16, 0x8899aa);
    let occ16 = occupancy_from_voxels(&v16);
    group.bench_function("solid 16³ (4k voxels)", |b| {
        b.iter(|| greedy_mesh::voxel_surface_grid_line_vertices(black_box(&occ16)))
    });

    // 32³ = 32 768 voxels — medium model
    let v32 = solid_box((0, 0, 0), 32, 0x8899aa);
    let occ32 = occupancy_from_voxels(&v32);
    group.bench_function("solid 32³ (32k voxels)", |b| {
        b.iter(|| greedy_mesh::voxel_surface_grid_line_vertices(black_box(&occ32)))
    });

    // 64³ = 262 144 voxels — large model
    let v64 = solid_box((0, 0, 0), 64, 0x8899aa);
    let occ64 = occupancy_from_voxels(&v64);
    group.sample_size(15);
    group.bench_function("solid 64³ (262k voxels)", |b| {
        b.iter(|| greedy_mesh::voxel_surface_grid_line_vertices(black_box(&occ64)))
    });

    group.finish();
}

/// Build a hollow box (1-voxel-thick shell) — more realistic than a solid box
/// for typical user models in the 500–2000 voxel range.
fn hollow_box(edge: i32, color: u32) -> Vec<Voxel> {
    let e = edge.max(3);
    let mut voxels = Vec::new();
    for z in 0..e {
        for y in 0..e {
            for x in 0..e {
                let on_surface =
                    x == 0 || x == e - 1 || y == 0 || y == e - 1 || z == 0 || z == e - 1;
                if on_surface {
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

/// Build a hollow box with alternating colors per face to stress per-bucket logic
/// (every face bucket is different, defeating greedy merge across colors).
fn hollow_box_multicolor(edge: i32) -> Vec<Voxel> {
    let e = edge.max(3);
    let mut voxels = Vec::new();
    let colors = [
        0xff4444u32,
        0x44ff44,
        0x4444ff,
        0xffff44,
        0xff44ff,
        0x44ffff,
    ];
    for z in 0..e {
        for y in 0..e {
            for x in 0..e {
                let on_surface =
                    x == 0 || x == e - 1 || y == 0 || y == e - 1 || z == 0 || z == e - 1;
                if on_surface {
                    let face = if z == 0 {
                        0
                    } else if z == e - 1 {
                        1
                    } else if y == 0 {
                        2
                    } else if y == e - 1 {
                        3
                    } else if x == 0 {
                        4
                    } else {
                        5
                    };
                    voxels.push(Voxel {
                        x,
                        y,
                        z,
                        color: colors[face],
                        material: MaterialId::Plastic,
                        object_id: 0,
                    });
                }
            }
        }
    }
    voxels
}

/// Benchmarks at sizes that straddle the off-thread meshing thresholds
/// (OFF_THREAD_GREEDY_MESH_MIN_VOXELS = 2_000, proposed new value = 500).
/// These establish the baseline for validating the threshold change.
fn bench_threshold_sizes(c: &mut Criterion) {
    // 10³ outer - 8³ inner = 488 voxels  (below old threshold, above proposed 500 — edge case)
    let v_490 = hollow_box(10, 0x8899aa);
    // 14³ outer - 12³ inner = 1016 voxels (between proposed 500 and old 2000)
    let v_1k = hollow_box(14, 0x8899aa);
    // 20³ outer - 18³ inner = 2168 voxels (just above old threshold)
    let v_2k = hollow_box(20, 0x8899aa);
    // Multicolor variants (worst-case per-bucket overhead)
    let v_1k_mc = hollow_box_multicolor(14);
    let v_2k_mc = hollow_box_multicolor(20);

    let mut group = c.benchmark_group("greedy_mesh threshold sizes");
    group.bench_function("~490 voxels hollow (mono)", |b| {
        b.iter(|| greedy_mesh::build_greedy_mesh(black_box(&v_490), black_box(NO_OBJECTS)))
    });
    group.bench_function("~1k voxels hollow (mono)", |b| {
        b.iter(|| greedy_mesh::build_greedy_mesh(black_box(&v_1k), black_box(NO_OBJECTS)))
    });
    group.bench_function("~2k voxels hollow (mono)", |b| {
        b.iter(|| greedy_mesh::build_greedy_mesh(black_box(&v_2k), black_box(NO_OBJECTS)))
    });
    group.bench_function("~1k voxels hollow (multicolor)", |b| {
        b.iter(|| greedy_mesh::build_greedy_mesh(black_box(&v_1k_mc), black_box(NO_OBJECTS)))
    });
    group.bench_function("~2k voxels hollow (multicolor)", |b| {
        b.iter(|| greedy_mesh::build_greedy_mesh(black_box(&v_2k_mc), black_box(NO_OBJECTS)))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_full_mesh,
    bench_mapped_and_chunked,
    bench_spatial_cache,
    bench_bucket_scans,
    bench_load_chunk_meshes_fused_vs_sequential,
    bench_dirty_chunk_remesh,
    bench_mesh_bounds,
    bench_grid_border_lines,
    bench_threshold_sizes,
);
criterion_main!(benches);
