use adam_kernel::{Genome, GenomeHistory};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_content_hash(c: &mut Criterion) {
    let mut genome = Genome::new("ADAM", "benchmark organism");
    for i in 0..50 {
        genome.capabilities.push(format!("capability-{i}"));
        genome
            .preferences
            .insert(format!("pref-{i}"), format!("value-{i}"));
    }
    c.bench_function("genome_content_hash", |b| {
        b.iter(|| black_box(genome.content_hash()))
    });
}

fn bench_commit_and_rollback(c: &mut Criterion) {
    c.bench_function("genome_history_commit", |b| {
        b.iter(|| {
            let mut history = GenomeHistory::init(Genome::new("ADAM", "bench"), "genesis");
            for i in 0..20 {
                let mut next = history.head().genome.clone();
                next.goals.push(format!("goal-{i}"));
                history.commit(next, "bench commit");
            }
            black_box(history.head_id())
        })
    });
}

criterion_group!(benches, bench_content_hash, bench_commit_and_rollback);
criterion_main!(benches);
