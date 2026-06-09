# FastCDC v2020 — performance notes

Notes from the work behind #44 (restore array-typed GEAR lookups in `cut_gear`;
add `Chunker`). Scope is the `v2020` cut path only. Measured on an Apple M1 Pro
and a dedicated-CPU x86_64 (AMD EPYC) VM, rustc 1.95, release/bench profile
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
- re-running on the isolated x86 VM, where criterion CIs tightened to ±2–3% and
  the win reproduced on a different architecture.

Residual caveats: the inputs are synthetic (a seeded SplitMix64 fill), and
`llvm-mca` is only a model — it predicts even the GEAR-check removal should be
near-free, yet hardware shows ~12%. Where the model and the silicon disagree at
the margin, the interleaved on-hardware A/B wins.

---

*These notes came out of a downstream performance spike; happy to share the
benchmark harness (the interleaved A/B + the `llvm-mca` loop) if useful.
Developed with Claude Code.*
