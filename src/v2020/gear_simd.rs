//! SIMD-accelerated gear-hash boundary scan, vendored from the `gearhash` crate.
//!
//! Source: https://github.com/srijs/rust-gearhash (`src/scalar.rs`,
//! `src/simd/avx2.rs`), at commit master. Adapted only by inlining the `Table`
//! type and removing the crate's own tests/benches; the hot-loop logic is
//! unchanged. fastcdc supplies its *own* GEAR table, so cut points are identical
//! to the scalar path.
//!
//! Copyright (c) 2019 Sam Rijs and contributors. Licensed MIT OR Apache-2.0
//! (see the gearhash repository). Re-distributed here under fastcdc's MIT
//! license, which is compatible.
//!
//! `next_match(hash, table, buf, mask)` advances the rolling gear hash over
//! `buf` and returns `Some(n)` where `n` is the number of bytes consumed up to
//! and including the first byte at which `hash & mask == 0`, or `None`.

#![allow(dead_code)]

type Table = [u64; 256];

/// Portable scalar fallback (also used to resolve sub-strip ordering in AVX2).
#[inline]
pub(crate) fn scalar_next_match(
    hash: &mut u64,
    table: &Table,
    buf: &[u8],
    mask: u64,
) -> Option<usize> {
    for (i, b) in buf.iter().enumerate() {
        *hash = (*hash << 1).wrapping_add(table[*b as usize]);
        if *hash & mask == 0 {
            return Some(i + 1);
        }
    }
    None
}

/// Dispatch: AVX2 when available at runtime, else scalar. Identical results.
#[inline]
pub(crate) fn next_match(hash: &mut u64, table: &Table, buf: &[u8], mask: u64) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2_next_match(hash, table, buf, mask) };
        }
    }
    scalar_next_match(hash, table, buf, mask)
}

// --- AVX2, vendored verbatim from gearhash src/simd/avx2.rs -----------------

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
const CHUNK_SIZE: usize = 1024;
#[cfg(target_arch = "x86_64")]
const STRIP_SIZE: usize = CHUNK_SIZE / 4;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_next_match(
    hash: &mut u64,
    table: &Table,
    buf: &[u8],
    mask: u64,
) -> Option<usize> {
    for (ic, chunk) in buf.chunks(CHUNK_SIZE).enumerate() {
        if chunk.len() != CHUNK_SIZE {
            return scalar_next_match(hash, table, chunk, mask).map(|off| off + ic * CHUNK_SIZE);
        }

        let mut h = _mm256_setzero_si256();

        for i in 0..64 {
            let b1 = *chunk.get_unchecked((STRIP_SIZE - 64) + i);
            let b2 = *chunk.get_unchecked((STRIP_SIZE * 2 - 64) + i);
            let b3 = *chunk.get_unchecked((STRIP_SIZE * 3 - 64) + i);

            let g = _mm256_set_epi64x(
                0,
                table[b1 as usize] as i64,
                table[b2 as usize] as i64,
                table[b3 as usize] as i64,
            );

            h = _mm256_add_epi64(_mm256_slli_epi64(h, 1), g);
        }

        h = _mm256_insert_epi64(h, *hash as i64, 3);

        let mut pre_off = usize::MAX;
        let mut pre_hash = 0u64;

        for i in 0..STRIP_SIZE {
            let b0 = *chunk.get_unchecked(i);
            let b1 = *chunk.get_unchecked(STRIP_SIZE + i);
            let b2 = *chunk.get_unchecked(STRIP_SIZE * 2 + i);
            let b3 = *chunk.get_unchecked(STRIP_SIZE * 3 + i);

            let g = _mm256_set_epi64x(
                table[b0 as usize] as i64,
                table[b1 as usize] as i64,
                table[b2 as usize] as i64,
                table[b3 as usize] as i64,
            );

            h = _mm256_add_epi64(_mm256_slli_epi64(h, 1), g);

            let m = _mm256_and_si256(h, _mm256_set1_epi64x(mask as i64));
            let c = _mm256_cmpeq_epi64(m, _mm256_setzero_si256());
            let z = _mm256_movemask_epi8(c) as u32;

            if z == 0 {
                continue;
            }

            if z & (1u32 << 24) != 0 {
                *hash = _mm256_extract_epi64(h, 3) as u64;
                return Some(ic * CHUNK_SIZE + i + 1);
            }

            // Match in the second strip: fall back to scalar over the rest of
            // the first strip to see if there is an earlier match there.
            if z & (1u32 << 16) != 0 {
                let rest = &chunk[i + 1..STRIP_SIZE];
                *hash = _mm256_extract_epi64(h, 3) as u64;
                if let Some(off) = scalar_next_match(hash, table, rest, mask) {
                    return Some(ic * CHUNK_SIZE + i + 1 + off);
                } else {
                    *hash = _mm256_extract_epi64(h, 2) as u64;
                    return Some(ic * CHUNK_SIZE + STRIP_SIZE + i + 1);
                }
            }

            if z & (1u32 << 8) != 0 {
                let off = STRIP_SIZE * 2 + i;
                if off < pre_off {
                    pre_off = off;
                    pre_hash = _mm256_extract_epi64(h, 1) as u64;
                }
            }

            if z & 1u32 != 0 {
                let off = STRIP_SIZE * 3 + i;
                if off < pre_off {
                    pre_off = off;
                    pre_hash = _mm256_extract_epi64(h, 0) as u64;
                }
            }
        }

        if pre_off != usize::MAX {
            *hash = pre_hash;
            return Some(ic * CHUNK_SIZE + pre_off + 1);
        }

        *hash = _mm256_extract_epi64(h, 0) as u64;
    }

    None
}
