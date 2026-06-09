//! Fused chunk+hash vs two-pass.
//!
//! Content-addressed stores hash every chunk. The question: is it
//! faster to (A) scan the whole buffer collecting boundaries, then loop again
//! hashing each chunk ("two-pass", bytes evicted from cache between passes), or
//! (B) hash each chunk the instant its boundary is found, while it is still hot
//! in L1 ("fused", via `Chunker::for_each_chunk`)?
//!
//! Also reports chunk-only and hash-only throughput so we can see how much of
//! the combined cost is hashing vs chunking.
//!
//! Interleaved, min-of-N, one process (same discipline as benches/ab.rs).
//! Deterministic input, nothing to clean up.
//!
//! Run: `cargo bench --bench fuse`

use std::hint::black_box;
use std::time::Instant;

use blake3::Hasher as Blake3;
use fastcdc::v2020::Chunker;
use sha2::{Digest, Sha256};

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

// fold a 32-byte digest into the accumulator so the optimizer can't drop it
#[inline]
fn fold(acc: &mut u64, d: &[u8]) {
    for c in d.chunks(8) {
        let mut b = [0u8; 8];
        b[..c.len()].copy_from_slice(c);
        *acc ^= u64::from_le_bytes(b);
    }
}

// --- chunk only -------------------------------------------------------------
fn chunk_only(c: &Chunker, data: &[u8]) -> u64 {
    let mut acc = 0u64;
    c.for_each_boundary(data, |_o, l, h| acc ^= (l as u64) ^ h);
    acc
}

// --- hash only (whole buffer, one digest) -----------------------------------
fn hash_only_blake3(data: &[u8]) -> u64 {
    let mut h = Blake3::new();
    h.update(data);
    let mut acc = 0u64;
    fold(&mut acc, h.finalize().as_bytes());
    acc
}
fn hash_only_sha256(data: &[u8]) -> u64 {
    let mut h = Sha256::new();
    h.update(data);
    let mut acc = 0u64;
    fold(&mut acc, &h.finalize());
    acc
}

// --- two-pass: collect boundaries, then hash each chunk ---------------------
fn two_pass_blake3(c: &Chunker, data: &[u8], spans: &mut Vec<(usize, usize)>) -> u64 {
    spans.clear();
    c.for_each_boundary(data, |o, l, _h| spans.push((o, l)));
    let mut acc = 0u64;
    for &(o, l) in spans.iter() {
        let mut h = Blake3::new();
        h.update(&data[o..o + l]);
        fold(&mut acc, h.finalize().as_bytes());
    }
    acc
}
fn two_pass_sha256(c: &Chunker, data: &[u8], spans: &mut Vec<(usize, usize)>) -> u64 {
    spans.clear();
    c.for_each_boundary(data, |o, l, _h| spans.push((o, l)));
    let mut acc = 0u64;
    for &(o, l) in spans.iter() {
        let mut h = Sha256::new();
        h.update(&data[o..o + l]);
        fold(&mut acc, &h.finalize());
    }
    acc
}

// --- fused: hash each chunk in the chunker callback (while hot) -------------
fn fused_blake3(c: &Chunker, data: &[u8]) -> u64 {
    let mut acc = 0u64;
    c.for_each_chunk(data, |_o, _l, _h, bytes| {
        let mut h = Blake3::new();
        h.update(bytes);
        fold(&mut acc, h.finalize().as_bytes());
    });
    acc
}
fn fused_sha256(c: &Chunker, data: &[u8]) -> u64 {
    let mut acc = 0u64;
    c.for_each_chunk(data, |_o, _l, _h, bytes| {
        let mut h = Sha256::new();
        h.update(bytes);
        fold(&mut acc, &h.finalize());
    });
    acc
}

fn time_ns(mut f: impl FnMut() -> u64) -> (u128, u64) {
    let t = Instant::now();
    let acc = f();
    (t.elapsed().as_nanos(), acc)
}
fn mib_s(bytes: usize, ns: u128) -> f64 {
    (bytes as f64 / (1024.0 * 1024.0)) / (ns as f64 / 1e9)
}

fn main() {
    const ROUNDS: usize = 31;
    const WARMUP: usize = 4;
    let mib = 1024 * 1024;

    for (label, len, avg) in [
        ("16KiB chunks", 32 * mib, 16 * 1024usize),
        ("1MiB chunks", 32 * mib, mib),
    ] {
        let data = gen_random(len, 1234);
        let c = Chunker::new(avg / 4, avg, avg * 4);
        let mut spans: Vec<(usize, usize)> = Vec::new();

        // single-shot reference throughputs (median of a few)
        let mut t_chunk = Vec::new();
        let mut t_hb = Vec::new();
        let mut t_hs = Vec::new();
        for _ in 0..7 {
            t_chunk.push(time_ns(|| chunk_only(&c, black_box(&data))).0);
            t_hb.push(time_ns(|| hash_only_blake3(black_box(&data))).0);
            t_hs.push(time_ns(|| hash_only_sha256(black_box(&data))).0);
        }
        t_chunk.sort_unstable();
        t_hb.sort_unstable();
        t_hs.sort_unstable();

        // correctness: fused and two-pass must produce the same digest fold
        assert_eq!(
            two_pass_blake3(&c, &data, &mut spans),
            fused_blake3(&c, &data),
            "blake3 fused/two-pass mismatch"
        );
        assert_eq!(
            two_pass_sha256(&c, &data, &mut spans),
            fused_sha256(&c, &data),
            "sha256 fused/two-pass mismatch"
        );

        // interleaved A/B: two-pass vs fused, per hash
        let mut run = |which: u8| -> (u128, u128) {
            let mut tp = Vec::with_capacity(ROUNDS);
            let mut fu = Vec::with_capacity(ROUNDS);
            for r in 0..(ROUNDS + WARMUP) {
                let (a, b) = if r % 2 == 0 {
                    let a = time_ns(|| match which {
                        0 => two_pass_blake3(&c, black_box(&data), &mut spans),
                        _ => two_pass_sha256(&c, black_box(&data), &mut spans),
                    })
                    .0;
                    let b = time_ns(|| match which {
                        0 => fused_blake3(&c, black_box(&data)),
                        _ => fused_sha256(&c, black_box(&data)),
                    })
                    .0;
                    (a, b)
                } else {
                    let b = time_ns(|| match which {
                        0 => fused_blake3(&c, black_box(&data)),
                        _ => fused_sha256(&c, black_box(&data)),
                    })
                    .0;
                    let a = time_ns(|| match which {
                        0 => two_pass_blake3(&c, black_box(&data), &mut spans),
                        _ => two_pass_sha256(&c, black_box(&data), &mut spans),
                    })
                    .0;
                    (a, b)
                };
                if r >= WARMUP {
                    tp.push(a);
                    fu.push(b);
                }
            }
            tp.sort_unstable();
            fu.sort_unstable();
            (tp[0], fu[0])
        };

        let (tp_b, fu_b) = run(0);
        let (tp_s, fu_s) = run(1);

        println!("\n=== {label} ({}MiB random) ===", len / mib);
        println!(
            "  chunk-only:        {:>8.1} MiB/s",
            mib_s(len, t_chunk[t_chunk.len() / 2])
        );
        println!(
            "  hash-only BLAKE3:  {:>8.1} MiB/s    hash-only SHA-256: {:>8.1} MiB/s",
            mib_s(len, t_hb[t_hb.len() / 2]),
            mib_s(len, t_hs[t_hs.len() / 2])
        );
        println!(
            "  BLAKE3  two-pass:  {:>8.1} MiB/s    fused: {:>8.1} MiB/s    speedup {:.3}x",
            mib_s(len, tp_b),
            mib_s(len, fu_b),
            tp_b as f64 / fu_b as f64
        );
        println!(
            "  SHA-256 two-pass:  {:>8.1} MiB/s    fused: {:>8.1} MiB/s    speedup {:.3}x",
            mib_s(len, tp_s),
            mib_s(len, fu_s),
            tp_s as f64 / fu_s as f64
        );
    }
    println!("\n(min-of-{ROUNDS} interleaved; combined-pipeline MiB/s = input bytes / time)");
}
