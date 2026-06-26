# FastCDC v2020 — performance notes

Notes from the work behind #44 (restore array-typed GEAR lookups in `cut_gear`;
add `Chunker`). Scope is the `v2020` cut path only. Measured on two
architectures: an Apple M1 Pro (ARM), and a dedicated-CPU x86_64 (AMD EPYC)
VM on [Fly.io](https://fly.io) (`fly machine` / `performance` size, so a pinned
core with no noisy neighbours). rustc 1.95, release/bench profile
(`opt-level=3`, thin LTO, 1 codegen unit).

## 1. The change in #44, and why it helps

4.0.0 changed the GEAR tables from `&[u64; 256]` to `&[u64]`/`Cow`. Indexing a
fixed `[u64; 256]` by a `u8`-derived value is provably in range, so the compiler
emits no check; indexing a `&[u64]` is not, so it reinserts a
`panic_bounds_check` on **every** table lookup — twice per loop iteration, in
each of the two scan loops.

The fix converts the tables back to `&[u64; 256]` once at the top of `cut_gear`
via `try_into` (the tables are 256 entries by construction), then runs the loop
through a small inner `cut_gear_arr`. The public `cut_gear(&[u64], &[u64])`
signature is unchanged.

Evidence:

- **Cut points identical** — the existing fixture tests (exact cut points + BLAKE3
  digests across NC levels, seeds, and sizes) pass unchanged.
- **asm:** `panic_bounds_check` sites inside `cut_gear` drop from **8 → 4** (the 4
  GEAR-table checks go; the 4 source-index checks remain — see §2).
- **Timing:** an interleaved A/B (old `cut_gear` vs new in one process,
  alternating order each round, minimum over 41 rounds so thermal noise cancels)
  measured **~7–14% throughput** on random / text / zeros at 16 KiB / 1 MiB /
  2 MiB average chunk sizes, on both the M1 and the x86 VM.

## 2. Why only 4 checks went, and why the other 4 don't matter

The patch also narrows the source to `&source[..remaining]` with hoisted loop
bounds. That was *intended* to also drop the `src[a]`/`src[a+1]` checks, but the
compiler does not prove `2*index + 1 < remaining` through the loop, so 4
source-index checks remain. That turns out not to matter:

`llvm-mca` (znver3) on the actual inner loop, patched (4 checks) vs a
hand-stripped copy with those checks removed (0 checks):

```
                       Total cycles (1000 iters)
patched (4 checks)            6010
stripped (0 checks)           6010
```

Identical. The loop is **latency-bound on the gear-hash recurrence**
(`hash = (hash << 2) + GEAR_LS[b0]`, then `+ GEAR[b1]` — each iteration's `hash`
depends on the previous one). The bounds-check compares are independent of that
chain and execute in the shadow of its latency, so they cost ≈0. That is also
why removing the GEAR-table checks helped at all: it was not about the compares
themselves but about not gating the table *loads* behind them.

Practical takeaways:

- I did not chase the remaining 4 checks with `unsafe get_unchecked` — the safe
  array fix captures the win, and the recurrence (not the checks) is the real
  ceiling, ~2 GiB/s/core on these machines.
- The same reasoning says micro-optimizing the loop body further is unlikely to
  pay; the lever is the dependency chain, not instruction count.

## 3. Two properties worth knowing (no code change)

### `cut_gear` is self-synchronizing → region-parallel chunking is nearly free

Because `cut_gear` resets the hash to 0 after each cut and restarts the search
at `cut + min_size`, two chunkers that ever cut at the same absolute offset are
identical from then on. Measured: an independent chunker started at the midpoint
of a 64 MiB buffer re-synchronizes within **~1 average chunk** and then
reproduces **100%** of the interior boundaries a single-pass chunker finds.

So a large input can be chunked across N threads by splitting into regions; only
the 1–2 chunks straddling each seam differ from the serial result. Throughput
scaled near-linearly (≈4× on 4 cores). This might be worth a documentation note,
or a helper, for users chunking very large files.

### Hashing usually dominates a chunk-then-hash pipeline

Content-addressed callers hash every chunk. Hashing each chunk *immediately*
after its boundary is found — while the bytes are still hot in L1 — beat a
separate hashing pass by up to ~1.6× on the combined chunk+hash, for small
(~16 KiB) chunks with a fast SIMD hash (BLAKE3). The new `Chunker::for_each_chunk`
hands the callback the borrowed chunk slice for exactly this, with no allocation.
(With an unaccelerated hash, the hash so dominates that chunking speed barely
registers — worth checking SHA hardware acceleration is actually engaged before
optimizing the chunker.)

## 4. A note on measurement quality

The laptop is a poor microbenchmark host. A single full-suite `criterion` pass
reported the *same* operation at 450, 1070, and 1380 MiB/s across three groups —
cold-start / thermal ramp / memory pressure. So none of the ~12% rests on a
single criterion delta. It rests on, in decreasing order of trust:

- the asm fact (the work provably shrank — independent of any timer);
- the interleaved A/B (old and new run microseconds apart, alternating order, so
  drift divides out of the ratio), min-of-N (loop noise is one-sided);
- re-running on the isolated x86 Fly VM (dedicated core), where criterion CIs
  tightened to ±2–3% and the win reproduced on a different architecture.

Residual caveats: the inputs are synthetic (a seeded SplitMix64 fill), and
`llvm-mca` is only a model — it predicts even the GEAR-check removal should be
near-free, yet hardware shows ~12%. Where the model and the silicon disagree at
the margin, the interleaved on-hardware A/B wins.

---

# Appendix: SIMD / parallel-chunking investigation (issue #42)

Issue [#42](https://github.com/nlfiedler/fastcdc-rs/issues/42) asks for a SIMD
gear hash. We evaluated every credible option on real hardware — an **Apple M1
Pro (ARM/NEON)** and a **dedicated AMD EPYC core (x86/AVX2)** on Fly.io — with
the interleaved-A/B harness (min-of-41, alternating order). **Every variant
below produces byte-identical cut points; only speed differs.** Numbers are
MiB/s for the `v2020` cut path on random data; ratios are vs the scalar 2-byte
roll baseline.

## What we measured

**1. The `gearhash` crate** (the crate named in #42; srijs). Fed *fastcdc's*
GEAR table it reproduces our cut points exactly. Its AVX2 path keeps the hash
state in vector lanes (genuine 4-wide parallelism).

| | M1 (no NEON path → scalar fallback) | EPYC AVX2 |
|---|---|---|
| gearhash vs roll | **0.78× (−22%)** | **1.2–1.6× (+20–60%)** |

Real win on x86; regression on ARM. x86-only, and adds an `unsafe` dependency to
a pure-Rust crate.

**2. `johnrichardrinehart`'s SIMD commit** (bc8813a, the actual PR-in-progress
for #42; AVX2/SSE4.1/NEON). Correct, but a regression on **both** arches:

| | M1 NEON | EPYC AVX2 |
|---|---|---|
| vs roll | **0.91× (−9%)** | **0.62× (−38%)** |

Root cause: it computes the hashes in *scalar* registers and only moves them
into vectors for the mask compare — keeping the sequential dependency *and*
paying GPR↔vector domain-crossing. It vectorizes the part that was already free
(the roll is latency-bound on the recurrence; the mask check hides in its
shadow). **Do not merge as-is.**

**3. Pure-scalar parallel-strip ILP (no SIMD, no `unsafe`).** Break the
recurrence's dependency chain by running N independent strip-hashes interleaved,
so a wide OOO core overlaps them. A tight branchless detector finds the block
with a cut; the exact point/hash is then located with the real 2-byte roll
(hence byte-identical). Size-dependent:

| avg size | M1 strip8/roll | EPYC strip8/roll |
|---|---|---|
| 16 KiB | 0.76× | 0.64× |
| 64 KiB | 1.01× | — |
| 128 KiB | **1.10×** | — |
| 1–2 MiB | **1.19–1.20×** | 0.77–0.82× |

**Wins on M1 for large chunks (up to +20%); loses on x86.** The warm-up +
locate overhead dominates for small chunks (block ≈ chunk). Crossover ~64 KiB on
M1. Wide ARM integer cores extract the ILP; Zen3 does not.

**4. Our own wider AVX2 (8-wide, two `__m256i`).** Tried to beat gearhash's
4-wide on x86. It does not: `avx8 < avx4 < gearhash` (e.g. at 2 MiB, 0.84× of
gearhash). The vector ALU saturates at 4 lanes for this op mix, and a
detector+rescan structure can't beat gearhash's locate-in-lane. **4-wide AVX2 is
the practical x86 ceiling; don't widen.**

## Conclusion — the two architectures want opposite techniques

| | small chunks (≤64 KiB) | large chunks (≥128 KiB) |
|---|---|---|
| **x86 / AVX2** | gearhash 4-wide | gearhash 4-wide (ceiling) |
| **ARM / wide OOO** | scalar roll | **scalar ILP striping** |
| **either** | scalar roll | — |

No single technique is universally best. gearhash loses on ARM; striping loses
on x86; both lose to the plain roll at 16 KiB.

## What shipped here (experimental, off by default)

`cut_strip` — pure-scalar parallel-strip chunking (`STRIP_N = 8`,
`STRIP_STRIDE = 1024`), byte-identical to `cut_gear_arr` (proven by the
`strip_matches_scalar` test: hash + cut point, fixture + synthetic, all sizes).
Wired into `cut_gear` behind `#[cfg(all(target_arch = "aarch64", feature =
"strip-experimental"))]` and only for `avg_size >= STRIP_THRESHOLD` (128 KiB).
Default builds are unchanged and bit-for-bit identical; the full suite passes
with the feature both on and off.

This is recorded for evaluation, not yet proposed upstream. An eventual #42 PR
would likely: keep the scalar roll as the cross-platform default; offer an
opt-in `simd` feature dispatching to vendored gearhash-style 4-wide AVX2 on x86
and this scalar striping on aarch64 for large averages; never change the
default. Benches: `benches/strip.rs` (roll / gearhash / strip / AVX2),
`benches/ab.rs` (adds a gearhash column).

---

*These notes came out of a downstream performance spike; happy to share the
benchmark harness (the interleaved A/B + the `llvm-mca` loop) if useful.
Developed with Claude Code.*
