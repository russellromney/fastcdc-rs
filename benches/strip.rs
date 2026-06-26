//! Experiment: can we beat the scalar 2-byte roll by breaking the hash
//! recurrence's dependency chain with N *independent* strips, run interleaved
//! so a wide OOO core overlaps them (ILP)? Pure scalar — no SIMD, no unsafe.
//!
//! Contenders (all must agree on cut points; gated before timing):
//!   roll     - current v2020 2-byte roll (baseline)
//!   gearhash - gearhash crate (AVX2 on x86, scalar fallback on ARM)
//!   strip4   - 4 independent scalar strips, interleaved
//!   strip8   - 8 independent scalar strips, interleaved
//!
//! Run: cargo bench --bench strip
//!
//! Strip correctness: the gear hash is a 64-bit rolling value, so h at position
//! p depends only on the last 64 bytes once p >= start+64. A strip beginning at
//! offset `start` warms up from max(min, start-64) with seed 0; by `start` its
//! hash equals the reference (which seeds 0 at `min`). Strips tile the scan
//! region contiguously, so the earliest match overall is the lowest-indexed
//! strip with a match — identical cut points to the serial scan.

use std::hint::black_box;
use std::time::Instant;

use fastcdc::v2020::{MASKS, get_gear_with_seed};
use gearhash::Hasher;

// --- deterministic data -----------------------------------------------------

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
    let bits = avg.ilog2() as usize;
    let mask_s = MASKS[bits + 1];
    let mask_l = MASKS[bits - 1];
    Cfg { min: avg / 4, avg, max: avg * 4, mask_s, mask_l, mask_s_ls: mask_s << 1, mask_l_ls: mask_l << 1 }
}

// --- baseline: current 2-byte roll ------------------------------------------

#[inline]
fn cut_roll(source: &[u8], c: Cfg, gear: &[u64; 256], gear_ls: &[u64; 256]) -> (u64, usize) {
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
    let src = &source[..remaining];
    let limit1 = center / 2;
    let limit2 = remaining / 2;
    let mut index = c.min / 2;
    let mut hash: u64 = 0;
    while index < limit1 {
        let a = index * 2;
        hash = (hash << 2).wrapping_add(gear_ls[src[a] as usize]);
        if (hash & c.mask_s_ls) == 0 {
            return (hash, a);
        }
        hash = hash.wrapping_add(gear[src[a + 1] as usize]);
        if (hash & c.mask_s) == 0 {
            return (hash, a + 1);
        }
        index += 1;
    }
    while index < limit2 {
        let a = index * 2;
        hash = (hash << 2).wrapping_add(gear_ls[src[a] as usize]);
        if (hash & c.mask_l_ls) == 0 {
            return (hash, a);
        }
        hash = hash.wrapping_add(gear[src[a + 1] as usize]);
        if (hash & c.mask_l) == 0 {
            return (hash, a + 1);
        }
        index += 1;
    }
    (hash, remaining)
}

// --- N-wide interleaved scalar strips ---------------------------------------

const STRIDE: usize = 1024;

#[inline(always)]
fn mask_at(pos: usize, center: usize, c: &Cfg) -> u64 {
    if pos < center { c.mask_s } else { c.mask_l }
}

/// Single-byte gear scalar scan of [from, to); returns first match (pos, hash)
/// or None. `hash` is the running value seeded at the caller.
#[inline(always)]
fn scan_scalar(
    src: &[u8],
    from: usize,
    to: usize,
    mut hash: u64,
    center: usize,
    c: &Cfg,
    gear: &[u64; 256],
) -> Option<(usize, u64)> {
    let mut p = from;
    while p < to {
        hash = (hash << 1).wrapping_add(gear[src[p] as usize]);
        if (hash & mask_at(p, center, c)) == 0 {
            return Some((p, hash));
        }
        p += 1;
    }
    None
}

#[inline(always)]
fn warmup(src: &[u8], start: usize, min: usize, gear: &[u64; 256]) -> u64 {
    let w = if start >= min + 64 { start - 64 } else { min };
    let mut h = 0u64;
    let mut i = w;
    while i < start {
        h = (h << 1).wrapping_add(gear[src[i] as usize]);
        i += 1;
    }
    h
}

fn cut_strip<const N: usize>(source: &[u8], c: Cfg, gear: &[u64; 256]) -> (u64, usize) {
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
    let src = &source[..remaining];

    let mut base = c.min;
    while base + N * STRIDE <= remaining {
        let block_end = base + N * STRIDE;
        // Blocks that straddle the mask switch are rare (once per chunk): handle
        // serially so the hot loop can use a single uniform mask.
        if base < center && block_end > center {
            let warm = warmup(src, base, c.min, gear);
            if let Some((cut, hh)) = scan_scalar(src, base, block_end, warm, center, &c, gear) {
                return (hh, cut);
            }
            base = block_end;
            continue;
        }
        let mask = if base >= center { c.mask_l } else { c.mask_s };

        // warm up the N strips to their start positions
        let mut h = [0u64; N];
        for k in 0..N {
            h[k] = warmup(src, base + k * STRIDE, c.min, gear);
        }
        // branchless detector: bit k = strip k contains a cut. Locate only the
        // lowest matching strip (1/N of the block), not the whole block.
        let mut matched = 0u32;
        for t in 0..STRIDE {
            for k in 0..N {
                h[k] = (h[k] << 1).wrapping_add(gear[src[base + k * STRIDE + t] as usize]);
                matched |= (((h[k] & mask) == 0) as u32) << k;
            }
        }
        if matched != 0 {
            let k = matched.trailing_zeros() as usize;
            let ks = base + k * STRIDE;
            let warm = warmup(src, ks, c.min, gear);
            if let Some((cut, hh)) = scan_scalar(src, ks, ks + STRIDE, warm, center, &c, gear) {
                return (hh, cut);
            }
        }
        base += N * STRIDE;
    }
    // scalar tail
    let warm = warmup(src, base, c.min, gear);
    if let Some((cut, hh)) = scan_scalar(src, base, remaining, warm, center, &c, gear) {
        return (hh, cut);
    }
    (0, remaining)
}

// --- AVX2 in-lane striped detector (the "improve gearhash" experiment) ------
//
// Same striping framework as cut_strip, but the per-block "any cut here?"
// detector runs the N independent strip-hashes in AVX2 lanes (4 per __m256i).
// gearhash is 4-wide; this tries 4-wide and 8-wide. The exact cut is still
// located by the proven scalar `scan_scalar`, so correctness is identical to
// the scalar strip (gated). On non-x86 / no-AVX2 it falls back to the scalar
// detector, so the bench still builds and runs on ARM.

/// Returns a bitmask: bit k set if strip k contains a cut in this block.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn detect_avx2(
    src: &[u8],
    base: usize,
    stride: usize,
    mask: u64,
    gear: &[u64; 256],
    h: &[u64],
) -> u32 {
    use std::arch::x86_64::*;
    let maskv = _mm256_set1_epi64x(mask as i64);
    let zero = _mm256_setzero_si256();
    let tbl = |b: u8| gear[b as usize] as i64;
    let g = |k: usize, t: usize| tbl(*src.get_unchecked(base + k * stride + t));
    if h.len() == 8 {
        let mut h0 = _mm256_setr_epi64x(h[0] as i64, h[1] as i64, h[2] as i64, h[3] as i64);
        let mut h1 = _mm256_setr_epi64x(h[4] as i64, h[5] as i64, h[6] as i64, h[7] as i64);
        let mut a0 = zero;
        let mut a1 = zero;
        for t in 0..stride {
            let gv0 = _mm256_setr_epi64x(g(0, t), g(1, t), g(2, t), g(3, t));
            let gv1 = _mm256_setr_epi64x(g(4, t), g(5, t), g(6, t), g(7, t));
            h0 = _mm256_add_epi64(_mm256_slli_epi64(h0, 1), gv0);
            h1 = _mm256_add_epi64(_mm256_slli_epi64(h1, 1), gv1);
            // per-lane "ever matched": OR the cmpeq results across t
            a0 = _mm256_or_si256(a0, _mm256_cmpeq_epi64(_mm256_and_si256(h0, maskv), zero));
            a1 = _mm256_or_si256(a1, _mm256_cmpeq_epi64(_mm256_and_si256(h1, maskv), zero));
        }
        // movemask_pd: 1 bit per 64-bit lane (its sign bit; all-ones if matched)
        let m0 = _mm256_movemask_pd(_mm256_castsi256_pd(a0)) as u32;
        let m1 = _mm256_movemask_pd(_mm256_castsi256_pd(a1)) as u32;
        m0 | (m1 << 4)
    } else {
        let mut h0 = _mm256_setr_epi64x(h[0] as i64, h[1] as i64, h[2] as i64, h[3] as i64);
        let mut a0 = zero;
        for t in 0..stride {
            let gv0 = _mm256_setr_epi64x(g(0, t), g(1, t), g(2, t), g(3, t));
            h0 = _mm256_add_epi64(_mm256_slli_epi64(h0, 1), gv0);
            a0 = _mm256_or_si256(a0, _mm256_cmpeq_epi64(_mm256_and_si256(h0, maskv), zero));
        }
        _mm256_movemask_pd(_mm256_castsi256_pd(a0)) as u32
    }
}

/// Bitmask: bit k set if strip k contains a cut in this block.
#[inline]
fn block_detect<const N: usize>(
    src: &[u8],
    base: usize,
    mask: u64,
    gear: &[u64; 256],
    h: &[u64; N],
) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { detect_avx2(src, base, STRIDE, mask, gear, h) };
        }
    }
    // scalar fallback detector
    let mut hh = *h;
    let mut matched = 0u32;
    for t in 0..STRIDE {
        for k in 0..N {
            hh[k] = (hh[k] << 1).wrapping_add(gear[src[base + k * STRIDE + t] as usize]);
            matched |= ((hh[k] & mask == 0) as u32) << k;
        }
    }
    matched
}

fn cut_strip_avx<const N: usize>(source: &[u8], c: Cfg, gear: &[u64; 256]) -> (u64, usize) {
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
    let src = &source[..remaining];
    let mut base = c.min;
    while base + N * STRIDE <= remaining {
        let block_end = base + N * STRIDE;
        if base < center && block_end > center {
            let warm = warmup(src, base, c.min, gear);
            if let Some((cut, hh)) = scan_scalar(src, base, block_end, warm, center, &c, gear) {
                return (hh, cut);
            }
            base = block_end;
            continue;
        }
        let mask = if base >= center { c.mask_l } else { c.mask_s };
        let mut h = [0u64; N];
        for k in 0..N {
            h[k] = warmup(src, base + k * STRIDE, c.min, gear);
        }
        let matched = block_detect::<N>(src, base, mask, gear, &h);
        if matched != 0 {
            let k = matched.trailing_zeros() as usize;
            let ks = base + k * STRIDE;
            let warm = warmup(src, ks, c.min, gear);
            if let Some((cut, hh)) = scan_scalar(src, ks, ks + STRIDE, warm, center, &c, gear) {
                return (hh, cut);
            }
        }
        base += N * STRIDE;
    }
    let warm = warmup(src, base, c.min, gear);
    if let Some((cut, hh)) = scan_scalar(src, base, remaining, warm, center, &c, gear) {
        return (hh, cut);
    }
    (0, remaining)
}

// --- gearhash (issue #42 candidate) -----------------------------------------

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
    if let Some(n) = h.next_match(&source[c.min..center], c.mask_s) {
        return (h.get_hash(), c.min + n - 1);
    }
    if let Some(n) = h.next_match(&source[center..remaining], c.mask_l) {
        return (h.get_hash(), center + n - 1);
    }
    (h.get_hash(), remaining)
}

// --- drivers + cut-point collectors -----------------------------------------

#[inline]
fn drive(data: &[u8], c: Cfg, mut cut: impl FnMut(&[u8], Cfg) -> (u64, usize)) -> usize {
    let mut pos = 0usize;
    let mut acc = 0usize;
    while pos < data.len() {
        let (hash, count) = cut(&data[pos..], c);
        if count == 0 {
            break;
        }
        acc ^= count ^ (hash as usize);
        pos += count;
    }
    acc
}

fn cuts(data: &[u8], c: Cfg, mut cut: impl FnMut(&[u8], Cfg) -> (u64, usize)) -> Vec<usize> {
    let (mut pos, mut out) = (0usize, Vec::new());
    while pos < data.len() {
        let (_h, n) = cut(&data[pos..], c);
        if n == 0 {
            break;
        }
        out.push(n);
        pos += n;
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
    let g: &[u64; 256] = (&*gear).try_into().unwrap();
    let gl: &[u64; 256] = (&*gear_ls).try_into().unwrap();

    // closures capturing the tables, all with signature (&[u8], Cfg)->(u64,usize)
    let roll = |s: &[u8], c: Cfg| cut_roll(s, c, g, gl);
    let gh = |s: &[u8], c: Cfg| cut_gearhash(s, c, g);
    let s4 = |s: &[u8], c: Cfg| cut_strip_avx::<4>(s, c, g);
    let s8 = |s: &[u8], c: Cfg| cut_strip_avx::<8>(s, c, g);

    const ROUNDS: usize = 41;
    const WARMUP: usize = 5;
    let mib = 1024 * 1024;
    let cases: Vec<(String, Vec<u8>, usize)> = vec![
        ("random 16MiB avg16KiB".into(), gen_random(16 * mib, 1), 16 * 1024),
        ("random 16MiB avg32KiB".into(), gen_random(16 * mib, 6), 32 * 1024),
        ("random 16MiB avg64KiB".into(), gen_random(16 * mib, 7), 64 * 1024),
        ("random 16MiB avg128KiB".into(), gen_random(16 * mib, 8), 128 * 1024),
        ("random 32MiB avg256KiB".into(), gen_random(32 * mib, 5), 256 * 1024),
        ("random 32MiB avg1MiB".into(), gen_random(32 * mib, 3), mib),
        ("random 32MiB avg2MiB".into(), gen_random(32 * mib, 4), 2 * mib),
    ];

    println!("STRIDE={STRIDE} (strip4/strip8 = AVX2 in-lane on x86, scalar fallback on ARM)");
    println!(
        "{:<22} {:>7} {:>7} {:>7} {:>7} | {:>6} {:>6} {:>6} {:>7}",
        "case", "roll", "gear", "avx4", "avx8", "g/roll", "a8/roll", "a8/g", ""
    );
    println!("{}", "-".repeat(92));

    for (label, data, avg) in &cases {
        let c = cfg(*avg);
        // correctness gate vs roll
        let cr = cuts(data, c, roll);
        for (nm, f) in [
            ("gear", Box::new(gh) as Box<dyn Fn(&[u8], Cfg) -> (u64, usize)>),
            ("strip4", Box::new(s4)),
            ("strip8", Box::new(s8)),
        ] {
            let cc = cuts(data, c, |s, cc| f(s, cc));
            if cc != cr {
                println!("{label}: *** {nm} DIVERGED ({} vs roll {} chunks) ***", cc.len(), cr.len());
            }
        }

        let mut t = [[0u128; 4]; ROUNDS];
        for r in 0..(ROUNDS + WARMUP) {
            let mut row = [0u128; 4];
            for step in 0..4 {
                match (r + step) % 4 {
                    0 => row[0] = time_ns(|| drive(black_box(data), c, roll)).0,
                    1 => row[1] = time_ns(|| drive(black_box(data), c, gh)).0,
                    2 => row[2] = time_ns(|| drive(black_box(data), c, s4)).0,
                    _ => row[3] = time_ns(|| drive(black_box(data), c, s8)).0,
                }
            }
            if r >= WARMUP {
                t[r - WARMUP] = row;
            }
        }
        let mn = |i: usize| t.iter().map(|r| r[i]).min().unwrap();
        let (roll_m, gh_m, s4_m, s8_m) = (mn(0), mn(1), mn(2), mn(3));
        println!(
            "{:<22} {:>7.0} {:>7.0} {:>7.0} {:>7.0} | {:>5.2}x {:>5.2}x {:>5.2}x",
            label,
            mib_s(data.len(), roll_m),
            mib_s(data.len(), gh_m),
            mib_s(data.len(), s4_m),
            mib_s(data.len(), s8_m),
            roll_m as f64 / gh_m as f64,
            roll_m as f64 / s8_m as f64,
            gh_m as f64 / s8_m as f64,
        );
    }
    println!();
    println!("MiB/s = min-of-{ROUNDS}. g/roll,a8/roll >1 => faster than roll. a8/g >1 => avx8 BEATS gearhash.");
}
