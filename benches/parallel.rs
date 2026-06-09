//! Parallel chunking: data-parallelism to get past the single-core latency wall.
//!
//! The gear hash is a serial dependency chain, so one core caps at ~2 GiB/s. The
//! way past that is *data* parallelism, not faster instructions. Two questions:
//!
//!   A. Fidelity. If you split a buffer and chunk each region independently
//!      (each region starts a fresh chunker), do the interior cut points still
//!      match the serial chunker? FastCDC resets its hash to 0 after every cut
//!      and starts the next search at cut+min_size, so two chunkers that ever
//!      cut at the same absolute offset are identical forever after. The only
//!      question is how fast an out-of-phase chunker re-synchronizes. We measure
//!      the resync distance and the fraction of serial boundaries reproduced.
//!
//!   B. Throughput. Split into N regions, chunk them on N threads, measure MiB/s
//!      and speedup vs 1 thread.
//!
//! Run: `cargo bench --bench parallel`
//! Deterministic input, std threads only, nothing to clean up.

use std::hint::black_box;
use std::thread;
use std::time::Instant;

use fastcdc::v2020::Chunker;

struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}
fn gen_random(len: usize, seed: u64) -> Vec<u8> {
    let mut rng = SplitMix64(seed);
    let mut out = Vec::with_capacity(len);
    while out.len() + 8 <= len {
        out.extend_from_slice(&rng.next_u64().to_le_bytes());
    }
    while out.len() < len {
        out.push(rng.next_u64() as u8);
    }
    out
}

/// Absolute chunk-start offsets for `&data[from..]`, offset back to absolute.
fn starts_from(c: &Chunker, data: &[u8], from: usize) -> Vec<usize> {
    let mut v = vec![from];
    c.for_each_boundary(&data[from..], |o, l, _h| {
        let end = from + o + l;
        if end < data.len() {
            v.push(end);
        }
    });
    v
}

fn time_ns(mut f: impl FnMut()) -> u128 {
    let t = Instant::now();
    f();
    t.elapsed().as_nanos()
}
fn mib_s(bytes: usize, ns: u128) -> f64 {
    (bytes as f64 / (1024.0 * 1024.0)) / (ns as f64 / 1e9)
}

fn main() {
    let mib = 1024 * 1024;
    let len = 64 * mib;
    let avg = 16 * 1024;
    let (min, max) = (avg / 4, avg * 4);
    let data = gen_random(len, 99);
    let c = Chunker::new(min, avg, max);

    // ---- A. Fidelity / self-synchronization -------------------------------
    let serial = starts_from(&c, &data, 0);
    let serial_set: std::collections::BTreeSet<usize> = serial.iter().copied().collect();

    // start an independent chunker at the midpoint (worst case: arbitrary phase)
    let p = len / 2;
    let indep = starts_from(&c, &data, p);

    // first absolute offset >= p that both agree is a chunk start
    let resync = indep
        .iter()
        .find(|&&o| o > p && serial_set.contains(&o))
        .copied();
    let resync_dist = resync.map(|r| r - p);

    // beyond the resync point, what fraction of independent starts match serial?
    let (mut matched, mut total) = (0usize, 0usize);
    if let Some(r) = resync {
        for &o in indep.iter().filter(|&&o| o >= r) {
            total += 1;
            if serial_set.contains(&o) {
                matched += 1;
            }
        }
    }

    println!("=== A. fidelity (64MiB random, avg 16KiB, split at midpoint) ===");
    println!("  serial chunks: {}", serial.len());
    match resync_dist {
        Some(d) => println!(
            "  resync distance past split: {d} bytes ({:.1} avg-chunks)",
            d as f64 / avg as f64
        ),
        None => println!("  never resynced (!)"),
    }
    println!(
        "  interior boundaries reproduced after resync: {matched}/{total} ({:.4}%)",
        100.0 * matched as f64 / total.max(1) as f64
    );
    println!(
        "  => only chunks straddling the split differ; everything after resync is identical.\n"
    );

    // ---- B. throughput scaling --------------------------------------------
    // count boundaries in a region (the work), return an accumulator
    fn chunk_region(c: &Chunker, region: &[u8]) -> u64 {
        let mut acc = 0u64;
        c.for_each_boundary(region, |_o, l, h| acc ^= (l as u64) ^ h);
        acc
    }

    fn run_parallel(c: &Chunker, data: &[u8], nthreads: usize) -> u64 {
        if nthreads == 1 {
            return chunk_region(c, data);
        }
        let chunk = data.len() / nthreads;
        let mut acc = 0u64;
        thread::scope(|s| {
            let mut handles = Vec::new();
            for t in 0..nthreads {
                let start = t * chunk;
                let end = if t == nthreads - 1 { data.len() } else { start + chunk };
                let region = &data[start..end];
                handles.push(s.spawn(move || chunk_region(c, region)));
            }
            for h in handles {
                acc ^= h.join().unwrap();
            }
        });
        acc
    }

    println!("=== B. throughput scaling (64MiB random, avg 16KiB) ===");
    let avail = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("  available_parallelism: {avail}");
    let mut base_ns = u128::MAX;
    for &n in &[1usize, 2, 4, 8] {
        if n > avail.max(1) * 2 {
            continue;
        }
        // warmup
        black_box(run_parallel(&c, &data, n));
        let mut best = u128::MAX;
        for _ in 0..15 {
            let ns = time_ns(|| {
                black_box(run_parallel(&c, black_box(&data), n));
            });
            best = best.min(ns);
        }
        if n == 1 {
            base_ns = best;
        }
        println!(
            "  {n:>2} threads: {:>8.1} MiB/s   speedup {:.2}x",
            mib_s(len, best),
            base_ns as f64 / best as f64
        );
    }
    println!("\n(min-of-15; speedup vs 1 thread. fidelity = how identical the output stays.)");
}
