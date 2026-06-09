//! Interleaved A/B benchmark: original 4.0.1 `cut_gear` (slice indexing, 8
//! bounds checks in the hot loop) vs the patched `cut_gear` (fixed-size GEAR
//! arrays + narrowed source window, 0 bounds checks).
//!
//! Both run in ONE process, alternating round-by-round on identical data, so
//! thermal drift and scheduler noise hit both equally and cancel in the ratio.
//! We report the MINIMUM time over many rounds (the cleanest estimator for a
//! deterministic CPU-bound loop: noise only adds time) plus the median.
//!
//! Run: `cargo bench --features internal-bench --bench ab`
//!
//! Deterministic inputs, no temp files, nothing to clean up.

use std::hint::black_box;
use std::time::Instant;

use fastcdc::v2020::{self, MASKS, cut_gear, cut_gear_legacy, get_gear_with_seed};

// --- deterministic data (mirrors benches/chunking.rs) ----------------------

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

fn gen_text(len: usize, seed: u64) -> Vec<u8> {
    const WORDS: &[&str] = &[
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "lorem", "ipsum", "dolor",
        "sit", "amet", "fn", "let", "mut", "return", "struct", "impl", "self",
    ];
    let mut rng = SplitMix64(seed);
    let mut out = Vec::with_capacity(len + 16);
    let mut col = 0;
    while out.len() < len {
        let w = WORDS[(rng.next_u64() as usize) % WORDS.len()];
        out.extend_from_slice(w.as_bytes());
        col += w.len();
        if col > 64 {
            out.push(b'\n');
            col = 0;
        } else {
            out.push(b' ');
        }
    }
    out.truncate(len);
    out
}

// --- config -----------------------------------------------------------------

#[derive(Clone, Copy)]
struct Cfg {
    min: usize,
    avg: usize,
    max: usize,
    mask_s: u64,
    mask_l: u64,
    mask_s_ls: u64,
    mask_l_ls: u64,
}

fn cfg(avg: usize) -> Cfg {
    // avg is a power of two here, so plain ilog2 matches the crate's rounded
    // logarithm2 and mask selection is unambiguous.
    let bits = avg.ilog2() as usize;
    let mask_s = MASKS[bits + 1];
    let mask_l = MASKS[bits - 1];
    Cfg {
        min: avg / 4,
        avg,
        max: avg * 4,
        mask_s,
        mask_l,
        mask_s_ls: mask_s << 1,
        mask_l_ls: mask_l << 1,
    }
}

// --- drivers: chunk the whole buffer via repeated cut() --------------------

#[inline]
fn drive_new(data: &[u8], c: Cfg, gear: &[u64], gear_ls: &[u64]) -> usize {
    let mut pos = 0usize;
    let mut acc = 0usize;
    while pos < data.len() {
        let (hash, count) = cut_gear(
            &data[pos..],
            c.min,
            c.avg,
            c.max,
            c.mask_s,
            c.mask_l,
            c.mask_s_ls,
            c.mask_l_ls,
            gear,
            gear_ls,
        );
        if count == 0 {
            break;
        }
        acc ^= count ^ (hash as usize);
        pos += count;
    }
    acc
}

#[inline]
fn drive_old(data: &[u8], c: Cfg, gear: &[u64], gear_ls: &[u64]) -> usize {
    let mut pos = 0usize;
    let mut acc = 0usize;
    while pos < data.len() {
        let (hash, count) = cut_gear_legacy(
            &data[pos..],
            c.min,
            c.avg,
            c.max,
            c.mask_s,
            c.mask_l,
            c.mask_s_ls,
            c.mask_l_ls,
            gear,
            gear_ls,
        );
        if count == 0 {
            break;
        }
        acc ^= count ^ (hash as usize);
        pos += count;
    }
    acc
}

fn time_ns(mut f: impl FnMut() -> usize) -> (u128, usize) {
    let t = Instant::now();
    let acc = f();
    (t.elapsed().as_nanos(), acc)
}

fn median(v: &mut [u128]) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn mib_s(bytes: usize, ns: u128) -> f64 {
    (bytes as f64 / (1024.0 * 1024.0)) / (ns as f64 / 1e9)
}

fn main() {
    let (gear, gear_ls) = get_gear_with_seed(0);
    let gear: &[u64] = &gear;
    let gear_ls: &[u64] = &gear_ls;

    const ROUNDS: usize = 41;
    const WARMUP: usize = 5;

    // (label, data, avg_chunk)
    let mut cases: Vec<(String, Vec<u8>, usize)> = Vec::new();
    let mib = 1024 * 1024;
    cases.push(("random 16MiB avg16KiB".into(), gen_random(16 * mib, 1), 16 * 1024));
    cases.push(("text   16MiB avg16KiB".into(), gen_text(16 * mib, 2), 16 * 1024));
    cases.push(("zeros  16MiB avg16KiB".into(), vec![0u8; 16 * mib], 16 * 1024));
    cases.push(("random 32MiB avg1MiB".into(), gen_random(32 * mib, 3), mib));
    cases.push(("random 32MiB avg2MiB".into(), gen_random(32 * mib, 4), 2 * mib));

    // sanity: confirm old and new produce identical accumulators (=> identical
    // cut points + hashes) before reporting any timing.
    println!("{:<26} {:>12} {:>12} {:>9} {:>9} {:>8}", "case", "old MiB/s", "new MiB/s", "old ms", "new ms", "speedup");
    println!("{}", "-".repeat(82));

    for (label, data, avg) in &cases {
        let c = cfg(*avg);

        // correctness gate
        let a_old = drive_old(data, c, gear, gear_ls);
        let a_new = drive_new(data, c, gear, gear_ls);
        assert_eq!(a_old, a_new, "A/B mismatch for {label}: cut points diverged");

        let mut old_ns = Vec::with_capacity(ROUNDS);
        let mut new_ns = Vec::with_capacity(ROUNDS);

        for r in 0..(ROUNDS + WARMUP) {
            // alternate order each round so neither side is systematically first
            let (o, n) = if r % 2 == 0 {
                let (o, ao) = time_ns(|| drive_old(black_box(data), c, gear, gear_ls));
                let (n, an) = time_ns(|| drive_new(black_box(data), c, gear, gear_ls));
                black_box(ao ^ an);
                (o, n)
            } else {
                let (n, an) = time_ns(|| drive_new(black_box(data), c, gear, gear_ls));
                let (o, ao) = time_ns(|| drive_old(black_box(data), c, gear, gear_ls));
                black_box(ao ^ an);
                (o, n)
            };
            if r >= WARMUP {
                old_ns.push(o);
                new_ns.push(n);
            }
        }

        let old_min = *old_ns.iter().min().unwrap();
        let new_min = *new_ns.iter().min().unwrap();
        let old_med = median(&mut old_ns.clone());
        let new_med = median(&mut new_ns.clone());

        // report MIN-based throughput (cleanest), median ms for context
        let speedup = old_min as f64 / new_min as f64;
        println!(
            "{:<26} {:>12.1} {:>12.1} {:>9.2} {:>9.2} {:>7.3}x",
            label,
            mib_s(data.len(), old_min),
            mib_s(data.len(), new_min),
            old_med as f64 / 1e6,
            new_med as f64 / 1e6,
            speedup,
        );
    }

    println!();
    println!("min = best of {ROUNDS} interleaved rounds (noise only adds time);");
    println!("ms columns are medians for context. speedup = old_min / new_min.");
    // keep v2020 namespace referenced so unused-import lints stay quiet if drivers change
    let _ = v2020::AVERAGE_MIN;
}
