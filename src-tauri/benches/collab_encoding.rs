//! Criterion benchmarks for collab edit encoding (`cargo bench -p voxelle-desktop --bench collab_encoding`).
//!
//! Compares JSON (old path) vs bincode (new path) for serializing/deserializing
//! edit deltas over the WebSocket collab protocol.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use voxelle_desktop_lib::collab::{encode_client_edit_binary, ClientToHost, HostToClient};
use voxelle_desktop_lib::voxel_edit::VoxelEditDelta;
use voxelle_desktop_lib::voxelle::{MaterialId, Voxel};

fn make_deltas(n: usize) -> Vec<VoxelEditDelta> {
    (0..n)
        .map(|i| {
            let i = i as i32;
            VoxelEditDelta::Added(Voxel {
                x: i % 64,
                y: i / 64,
                z: -(i % 32),
                color: 0xFF8800 + (i as u32 & 0xFF),
                material: MaterialId::Plastic,
                object_id: 0,
            })
        })
        .collect()
}

fn bench_client_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("client_edit_encode");
    for count in [10, 100, 500] {
        let deltas = make_deltas(count);
        group.bench_with_input(BenchmarkId::new("json", count), &deltas, |b, deltas| {
            b.iter(|| {
                let msg = serde_json::to_string(&ClientToHost::Edit {
                    deltas: deltas.clone(),
                })
                .unwrap();
                black_box(msg);
            });
        });
        group.bench_with_input(BenchmarkId::new("bincode", count), &deltas, |b, deltas| {
            b.iter(|| {
                let bin = encode_client_edit_binary(deltas);
                black_box(bin);
            });
        });
    }
    group.finish();
}

fn bench_client_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("client_edit_decode");
    for count in [10, 100, 500] {
        let deltas = make_deltas(count);
        let json = serde_json::to_string(&ClientToHost::Edit {
            deltas: deltas.clone(),
        })
        .unwrap();
        let bin = encode_client_edit_binary(&deltas);

        group.bench_with_input(BenchmarkId::new("json", count), &json, |b, json| {
            b.iter(|| {
                let parsed: ClientToHost = serde_json::from_str(json).unwrap();
                black_box(parsed);
            });
        });
        group.bench_with_input(BenchmarkId::new("bincode", count), &bin, |b, bin| {
            b.iter(|| {
                if bin.len() > 1 {
                    let decoded: Vec<VoxelEditDelta> = bincode::deserialize(&bin[1..]).unwrap();
                    black_box(decoded);
                }
            });
        });
    }
    group.finish();
}

fn bench_host_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("host_edit_encode");
    for count in [10, 100, 500] {
        let deltas = make_deltas(count);
        group.bench_with_input(BenchmarkId::new("json", count), &deltas, |b, deltas| {
            b.iter(|| {
                let msg = serde_json::to_string(&HostToClient::Edit {
                    seq: 42,
                    peer_id: 3,
                    deltas: deltas.clone(),
                })
                .unwrap();
                black_box(msg);
            });
        });
        group.bench_with_input(BenchmarkId::new("bincode", count), &deltas, |b, deltas| {
            b.iter(|| {
                let payload = bincode::serialize(&(42u64, 3u32, deltas)).unwrap();
                black_box(payload);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_client_encode,
    bench_client_decode,
    bench_host_encode
);
criterion_main!(benches);
