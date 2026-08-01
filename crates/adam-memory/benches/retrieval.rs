use adam_memory::{MemoryKind, MemoryRecord, MemoryStore, Provenance};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn seeded_store(count: usize) -> MemoryStore {
    let store = MemoryStore::open(":memory:").unwrap();
    for i in 0..count {
        let embedding: Vec<f32> = (0..64).map(|d| ((i + d) % 7) as f32).collect();
        let record = MemoryRecord::new(
            MemoryKind::Episodic,
            format!("memory number {i}"),
            embedding,
            0.8,
            Provenance {
                origin: "bench".to_string(),
                evidence: vec![],
            },
            0.0,
        );
        store.store(&record).unwrap();
    }
    store
}

fn bench_query_similar(c: &mut Criterion) {
    let store = seeded_store(500);
    let query: Vec<f32> = (0..64).map(|d| (d % 7) as f32).collect();

    c.bench_function("memory_query_similar_500", |b| {
        b.iter(|| black_box(store.query_similar(&query, None, 10).unwrap()))
    });
}

fn bench_store(c: &mut Criterion) {
    let store = MemoryStore::open(":memory:").unwrap();
    let mut i = 0usize;
    c.bench_function("memory_store_single_record", |b| {
        b.iter(|| {
            let embedding: Vec<f32> = (0..64).map(|d| ((i + d) % 7) as f32).collect();
            let record = MemoryRecord::new(
                MemoryKind::Episodic,
                format!("memory number {i}"),
                embedding,
                0.8,
                Provenance {
                    origin: "bench".to_string(),
                    evidence: vec![],
                },
                0.0,
            );
            store.store(&record).unwrap();
            i += 1;
        })
    });
}

criterion_group!(benches, bench_query_similar, bench_store);
criterion_main!(benches);
