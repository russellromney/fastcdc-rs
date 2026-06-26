//! Interleaved A/B/C benchmark of three `cut` implementations:
//!   - `old`  : original 4.0.1 `cut_gear` (slice indexing, 8 bounds checks)
//!   - `new`  : patched `cut_gear` (fixed-size GEAR arrays, 0 table checks)
//!   - `gear` : the `gearhash` crate's `next_match` fed *fastcdc's* GEAR table
//!              (issue #42 — SIMD gear). On x86_64 with AVX2 this is the AVX2
//!              path; on ARM, gearhash 0.1 has no NEON path so it is gearhash's
//!              scalar fallback.
//!
//! All three run in ONE process, alternating order round-by-round on identical
//! data, so thermal drift and scheduler noise hit each equally and cancel in
//! the ratio. We report the MINIMUM time over many rounds (cleanest estimator
//! for a deterministic CPU-bound loop: noise only adds time) plus the median.
//!
//! gearhash's recurrence `hash = (hash << 1) + table[b]` is bit-identical to
//! v2020's two-byte roll (GEAR_LS == GEAR << 1), so with fastcdc's GEAR table
//! it must produce the same cut points. The correctness gate asserts that
//! before any timing is reported.
//!
//! Run: `cargo bench --features internal-bench --bench ab`
//!
//! Deterministic inputs, no temp files, nothing to clean up.

use std::hint::black_box;
use std::time::Instant;

use fastcdc::v2020::{self, MASKS, cut_gear, cut_gear_legacy, get_gear_with_seed};
use gearhash::Hasher;

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

/// Replicates v2020 `cut_gear` cut points using gearhash's `next_match`:
/// region 1 [min, center) under mask_s, then region 2 [center, remaining)
/// under mask_l with the hash state carried across. Returns `(hash, count)`
/// matching the scalar contract (`count` = absolute index of the match byte).
#[inline]
fn cut_gearhash(source: &[u8], c: Cfg, table: &[u64; 256]) -> (u64, usize) {
    let mut remaining = source.len();
    if remaining <= c.min {
        return (0, remaining);
    }
    let mut center = c.avg;
    if remaining > c.max {
        remaining = c.max;
    } else if remaining < center {
        center = remaining;
    }
    let mut h = Hasher::new(table);
    // gearhash hashes from 0 — same start state as v2020 (sub-minimum bytes
    // never enter the hash). n = bytes processed; match byte is at local index
    // n-1, absolute index (base + n - 1), which is the v2020 `count`.
    if let Some(n) = h.next_match(&source[c.min..center], c.mask_s) {
        return (h.get_hash(), c.min + n - 1);
    }
    if let Some(n) = h.next_match(&source[center..remaining], c.mask_l) {
        return (h.get_hash(), center + n - 1);
    }
    (h.get_hash(), remaining)
}

#[inline]
fn drive_gear(data: &[u8], c: Cfg, table: &[u64; 256]) -> usize {
    let mut pos = 0usize;
    let mut acc = 0usize;
    while pos < data.len() {
        let (hash, count) = cut_gearhash(&data[pos..], c, table);
        if count == 0 {
            break;
        }
        acc ^= count ^ (hash as usize);
        pos += count;
    }
    acc
}

// --- cut-point collectors (for the correctness gate) ------------------------
//
// We compare the SEQUENCE OF CUT POINTS (chunk lengths), not the returned hash.
// v2020's `cut` returns a hash that is doubled at even-position matches (an
// artifact of the two-byte roll), whereas gearhash returns the true single-byte
// gear hash; the cut points are identical but the hash *fields* are not, so the
// timing accumulators (which fold in the hash) legitimately differ. Identical
// cut points are the property that matters for chunking/dedup.

fn cuts_v2020(data: &[u8], c: Cfg, gear: &[u64], gear_ls: &[u64]) -> Vec<usize> {
    let mut pos = 0usize;
    let mut out = Vec::new();
    while pos < data.len() {
        let (_h, count) = cut_gear(
            &data[pos..],
            c.min, c.avg, c.max, c.mask_s, c.mask_l, c.mask_s_ls, c.mask_l_ls, gear, gear_ls,
        );
        if count == 0 {
            break;
        }
        out.push(count);
        pos += count;
    }
    out
}

fn cuts_gearhash(data: &[u8], c: Cfg, table: &[u64; 256]) -> Vec<usize> {
    let mut pos = 0usize;
    let mut out = Vec::new();
    while pos < data.len() {
        let (_h, count) = cut_gearhash(&data[pos..], c, table);
        if count == 0 {
            break;
        }
        out.push(count);
        pos += count;
    }
    out
}

fn time_ns(mut f: impl FnMut() -> usize) -> (u128, usize) {
    let t = Instant::now();
    let acc = f();
    (t.elapsed().as_nanos(), acc)
}

fn mib_s(bytes: usize, ns: u128) -> f64 {
    (bytes as f64 / (1024.0 * 1024.0)) / (ns as f64 / 1e9)
}

fn main() {
    let (gear, gear_ls) = get_gear_with_seed(0);
    let table: &[u64; 256] = (&*gear).try_into().unwrap();
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

    // Throughput columns are min-based; the two speedup columns are new/old and
    // new/gear (>1 => new is faster than that contender).
    println!(
        "{:<24} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "case", "old MiB/s", "new MiB/s", "gear MiB/s", "n/old", "n/gear"
    );
    println!("{}", "-".repeat(76));

    for (label, data, avg) in &cases {
        let c = cfg(*avg);

        // correctness gate: all three must agree on the SEQUENCE OF CUT POINTS.
        // (old vs new also agree on hash; gearhash agrees on cut points but not
        // the hash field — see note above the collectors.)
        let a_old = drive_old(data, c, gear, gear_ls);
        let a_new = drive_new(data, c, gear, gear_ls);
        assert_eq!(a_old, a_new, "old/new mismatch for {label}: cut points diverged");
        let cuts_ref = cuts_v2020(data, c, gear, gear_ls);
        let cuts_g = cuts_gearhash(data, c, table);
        assert_eq!(
            cuts_ref, cuts_g,
            "new/gear mismatch for {label}: gearhash cut points diverged from v2020 \
             ({} vs {} chunks)",
            cuts_ref.len(),
            cuts_g.len()
        );

        let mut old_ns = Vec::with_capacity(ROUNDS);
        let mut new_ns = Vec::with_capacity(ROUNDS);
        let mut gear_ns = Vec::with_capacity(ROUNDS);

        for r in 0..(ROUNDS + WARMUP) {
            // rotate which driver runs first each round so none is systematically
            // advantaged by cache/thermal ordering
            let mut o = 0u128;
            let mut n = 0u128;
            let mut g = 0u128;
            let order = r % 3;
            for step in 0..3 {
                match (order + step) % 3 {
                    0 => {
                        let (t, a) = time_ns(|| drive_old(black_box(data), c, gear, gear_ls));
                        black_box(a);
                        o = t;
                    }
                    1 => {
                        let (t, a) = time_ns(|| drive_new(black_box(data), c, gear, gear_ls));
                        black_box(a);
                        n = t;
                    }
                    _ => {
                        let (t, a) = time_ns(|| drive_gear(black_box(data), c, table));
                        black_box(a);
                        g = t;
                    }
                }
            }
            if r >= WARMUP {
                old_ns.push(o);
                new_ns.push(n);
                gear_ns.push(g);
            }
        }

        let old_min = *old_ns.iter().min().unwrap();
        let new_min = *new_ns.iter().min().unwrap();
        let gear_min = *gear_ns.iter().min().unwrap();

        println!(
            "{:<24} {:>9.1} {:>9.1} {:>9.1} {:>8.3}x {:>8.3}x",
            label,
            mib_s(data.len(), old_min),
            mib_s(data.len(), new_min),
            mib_s(data.len(), gear_min),
            old_min as f64 / new_min as f64,
            gear_min as f64 / new_min as f64,
        );
    }

    println!();
    println!("min = best of {ROUNDS} interleaved rounds (noise only adds time).");
    println!("n/old  = old_min / new_min  (>1 => new faster than old).");
    println!("n/gear = gear_min / new_min (>1 => new scalar faster than gearhash).");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    println!(
        "gearhash SIMD: AVX2={}, SSE4.2={}",
        std::is_x86_feature_detected!("avx2"),
        std::is_x86_feature_detected!("sse4.2")
    );
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    println!("gearhash on this arch: scalar fallback (no NEON path in gearhash 0.1).");
    // keep v2020 namespace referenced so unused-import lints stay quiet if drivers change
    let _ = v2020::AVERAGE_MIN;
}
