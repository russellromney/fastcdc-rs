# FastCDC vendored spike — research note, benchmarks, review

Scope: improve the vendored FastCDC *implementation* (`third_party/fastcdc`),
not CinchFS integration. Vendored from crates.io `fastcdc` 4.0.1 (MIT). This is
a research/performance spike. Algorithm compatibility is preserved; the one
optimization is bounds-check-only and the one new API is additive.

Machine for all timings: **Apple M1 Pro, 8 cores, macOS (darwin 25.3)**, rustc
1.95.0, bench profile `opt-level=3 + lto="thin" + codegen-units=1`.

---

## 1. What the papers imply for this implementation

**FastCDC 2016** ("a Fast and Efficient Content-Defined Chunking Approach…",
Xia et al., USENIX ATC '16). Three ideas, all present in `src/v2016`:

- **Gear hash.** A rolling fingerprint `h = (h << 1) + GEAR[byte]`, one table
  load + shift + add per byte. Cheaper than Rabin because there is no removal of
  the outgoing byte — the left shift ages old bytes out of the window
  implicitly.
- **Cut-point skipping.** Start scanning at `min_size`, never declaring a cut
  before it. Removes tiny chunks and skips a chunk's worth of hashing per chunk.
  `v2016::cut` starts `index = min_size`.
- **Normalized chunking (NC).** Two masks. Before the "center" (≈ average size)
  use a mask with *more* 1-bits (`mask_s`, harder to satisfy → suppresses small
  chunks); after center use *fewer* 1-bits (`mask_l`, easier → forces a cut
  sooner). Masks are picked as `MASKS[log2(avg) ± level]`. NC trades dedup ratio
  against chunk-size variance; it does not change per-byte cost.

**FastCDC 2020** ("The Design of Fast Content-Defined Chunking…", Xia et al.,
IEEE TPDS 2020). The relevant addition is **"rolling two bytes each time"**
(§3.7): precompute a second table `GEAR_LS` whose entries are the gear values
shifted left by one, so two bytes can be folded per iteration —
`h = (h << 2) + GEAR_LS[b0]`, test, `h += GEAR[b1]`, test — halving loop
bookkeeping and branch count for the same fingerprints. The authors report
30–40% over 2016 for the same cut points. `src/v2020` implements exactly this:
`GEAR_LS`, the `<<2` fold, paired mask tests (`mask_s_ls`/`mask_s`), and indices
that run over *halves* of the byte range.

**The implication for performance work.** The 2020 algorithm has already
squeezed the algorithmic constant: per *pair* of bytes the work is 2 loads, 1
shift, 2 adds, 2 mask-tests, 2 predictable branches. There is no cheaper way to
gear-hash. So the only remaining lever in the Rust port is **work the compiler
adds that the algorithm does not call for** — specifically array bounds checks
on the GEAR-table lookup and the source read. The papers describe a fixed
256-entry table indexed by a byte; that index *cannot* be out of range, and an
implementation that still pays for a bounds check is leaving the paper's
constant on the floor. That is the whole opportunity, and it is exactly what the
4.0.0 refactor reintroduced (see §3).

A second, smaller implication: the scan cost is **content-independent per byte**
— every byte is hashed and tested regardless of value. Content only changes
*how many* cut points are found (i.e. how many times the inner loop restarts).
The pathological case is all-zeros: no content cut ever fires, so every chunk
runs the full scan to `max_size`. That makes "zeros" the cleanest worst-case to
bound throughput, which is why it is a benchmark case.

---

## 2. Current implementation: where the time goes

The hot path for every API (`FastCDC` iterator, `cut()` loop, `StreamCDC`) is
`v2020::cut_gear` → the two `while` loops. Everything else (iterator state,
mask/log setup, `Chunk` construction) is per-*chunk*, i.e. amortized over
thousands of bytes and irrelevant to throughput.

`StreamCDC` is the exception that *does* spend real time outside the scan: it
owns a `max_size` buffer, allocates a fresh `Vec` per chunk (`drain_bytes`), and
`copy_within`s the tail back to the front after every chunk. For in-memory
callers that is pure overhead versus the borrowing iterator — which is the
motivation for the additive zero-copy API in §4.

---

## 3. The optimization (bounds-check-only, cut points preserved)

### Root cause

4.0.0 changed the GEAR tables from `&[u64; 256]` (fixed array) to
`Cow<'static, [u64]>` / `&[u64]` (slice). Indexing a `[u64; 256]` by a value
derived from a `u8` is provably in range, so the compiler emits no check.
Indexing a `&[u64]` is **not** provable — the slice length is a runtime value —
so the compiler reinserts a `panic_bounds_check` on *every* table lookup, twice
per loop iteration. The source reads `source[a]` / `source[a + 1]` were
similarly unprovable.

Noise-free evidence (emitted asm, `--emit asm -C opt-level=3`):

```
panic_bounds_check sites inside cut_gear:   8   (baseline 4.0.1)
panic_bounds_check sites inside cut_gear:   0   (patched)
```

8 = {GEAR_LS, GEAR, source[a], source[a+1]} × {pre-center loop, post-center loop}.

### The change

`cut_gear` now, once per call (not per byte):

1. `let gear: &[u64; 256] = gear.try_into().expect(...)` (same for `gear_ls`).
   The gear hash table is 256 entries by construction; non-256 input is a
   programming error and panics at the boundary instead of silently. The inner
   loop then indexes arrays → no table bounds check.
2. `let src = &source[..remaining];` and hoists `limit1 = center/2`,
   `limit2 = remaining/2`. Since `center ≤ remaining`, `2*index+1 < remaining`
   holds throughout, so `src[a]` / `src[a+1]` need no check.

The arithmetic is byte-for-byte identical; only the *provenance* of the indices
changed. The original loop body is retained verbatim as `cut_gear_legacy`
(behind the `internal-bench` feature) for the A/B harness.

### Proof that cut points are unchanged

- **Existing upstream tests** pin exact cut points, exact 64-bit hashes, and
  exact BLAKE3 digests for the `SekienAkashita.jpg` fixture across NC levels 0/1/3,
  seeds 0 and 666, and 16K/32K/64K parameters — for `cut()`, the iterator, and
  `StreamCDC`. All pass unchanged (51 default-feature tests).
- **A/B correctness gate.** `benches/ab.rs` chunks 16–32 MiB of random, text, and
  zeros with both implementations and `assert_eq!`s the XOR-accumulated
  (length ⊕ hash) stream before timing. Identical for every case.

---

## 4. Additive API: `v2020::Chunker`

`FastCDC` borrows its source for its whole lifetime, so chunking many in-memory
buffers re-pays `new()` each time and forces the caller to use the iterator's
`Chunk` structs. `Chunker` holds only the precomputed config (masks + gear
tables) and applies it to any slice:

- `for_each_chunk(source, |offset, length, hash, bytes|)` — **zero allocation**;
  `bytes` borrows directly from `source`, so a caller doing per-segment SHA-256
  neither allocates nor copies (the `StreamCDC` per-chunk `Vec` is avoided
  entirely for in-memory inputs).
- `for_each_boundary(source, |offset, length, hash|)` — boundary-only, for
  building an extent map without touching chunk bytes.

Both reuse the optimized `cut_gear` and produce **identical cut points** to
`FastCDC` (regression test `test_chunker_matches_iterator` compares offset,
length, hash, *and* the bytes against the iterator across three configs on the
fixture and on zeros; `test_chunker_seed_matches_iterator` covers a seeded gear).

This API is the natural fit for CinchFS's eventual per-chunk hashing path
(roadmap item #14), but nothing in CinchFS is touched by this spike.

---

## 5. Benchmarks

### 5a. Headline: interleaved A/B (old `cut_gear` vs new), one process

`benches/ab.rs` alternates old/new round-by-round on identical data and reports
the **minimum** over 41 timed rounds (for a deterministic CPU loop, noise can
only *add* time, so the min is the least-contaminated estimate). Two independent
runs:

| case                    | old MiB/s | new MiB/s | speedup (run 1 / run 2) |
|-------------------------|-----------|-----------|-------------------------|
| random 16 MiB, avg 16K  | 1650 / 1665 | 1875 / 1895 | **1.136× / 1.138×** |
| text   16 MiB, avg 16K  | 1654 / 1663 | 1880 / 1891 | **1.137× / 1.137×** |
| zeros  16 MiB, avg 16K  | 1418 / 1414 | 1604 / 1597 | **1.131× / 1.129×** |
| random 32 MiB, avg 1 MiB| 1578 / 1575 | 1790 / 1792 | **1.134× / 1.137×** |
| random 32 MiB, avg 2 MiB| 1741 / 1737 | 1971 / 1975 | **1.133× / 1.137×** |

Consistent **~1.13× (≈13% faster)** everywhere, reproducible to the third
decimal across runs and across five independent inputs. CPU time per 16 MiB
random scan drops from ~9.94 ms to ~8.68 ms (median).

### 5b. Descriptive throughput (criterion, "after")

Useful for shape, **not** for the before/after delta (see §6). Isolated runs,
3 s measurement, M1 Pro:

```
v2020_paths/iterator   ~1.0–1.3 GiB/s     (8 MiB random, avg 16K)
v2020_paths/cut_loop   ~ same as iterator (iterator IS a cut() loop)
v2020_paths/stream     ~0.8–0.9 GiB/s     (per-chunk Vec alloc + copy_within)
content/zeros          worst case (full scan to max_size every chunk)
versions/v2020         ~3× versions/v2016, ~1.5–2× versions/ronomon
new_api/for_each_chunk ~ on par with iterator, zero-allocation
```

The `versions` group is directionally consistent with the 2020 paper's claim
that v2020 ≫ v2016, and then some: v2016 still carries the per-byte source
bounds check *and* hashes one byte per iteration, so on this microarchitecture
the gap is larger than the paper's 30–40%. (These are noisy single-pass numbers;
treat the ratio as "v2020 is clearly the one to use," not a precise multiple.)

### 5c. Neutral environment (Fly dedicated-CPU VM, Linux/x86_64)

The laptop numbers were re-run on a **Fly `performance-2x` machine (2 dedicated
vCPU, x86_64, IAD)** — an isolated box with no thermal throttling and no
shared-core neighbors. This both confirms the win on a different architecture
than my ARM laptop and gives much tighter measurements. Build was remote (so the
binary is native amd64); the VM was destroyed right after.

Criterion CIs collapsed from ±10–20% (laptop) to **±2–3%**, and the cross-group
3× discrepancy vanished — identical work now reads identically:

```
v2020_paths/iterator   2.22 GiB/s   [3.45 3.53 3.62] ms
v2020_paths/cut_loop   2.20 GiB/s   (= iterator, as it should be)
v2020_paths/stream     1.83 GiB/s
content/random         2.21 GiB/s   content/text 2.16   content/zeros 2.01   content/mixed 2.18
avg_size 16K/1M/2M     2.07 / 2.32 / 2.28 GiB/s
```

Interleaved A/B on the neutral box (old `cut_gear` vs patched):

| case                     | old MiB/s | new MiB/s | speedup |
|--------------------------|-----------|-----------|---------|
| random 16 MiB, avg 16K   | 2195      | 2447      | 1.115×  |
| text   16 MiB, avg 16K   | 2247      | 2561      | 1.140×  |
| zeros  16 MiB, avg 16K   | 1961      | 2216      | 1.130×  |
| random 32 MiB, avg 1 MiB | 2149      | 2308      | 1.074×  |
| random 32 MiB, avg 2 MiB | 2306      | 2587      | 1.122×  |

The win holds on x86_64: **+7% to +14%, clustered around +12%**, with the same
identical-cut-points correctness gate passing. (The single 1.074× outlier is the
1 MiB-average case, where fewer-but-longer scans shift the loop balance; still a
clear gain.) Absolute throughput is higher than the laptop's min-of-N figures
because this is a different CPU at a higher sustained clock — the *ratio* is the
portable result.

Repro on Fly: `Dockerfile.bench` + `fly.toml` build a throwaway dedicated-CPU
app that runs both benches and prints to `fly logs`; destroy with
`fly apps destroy`. The image excludes `target/` via `.dockerignore`.

### Reproduce

```
cd third_party/fastcdc
cargo bench --features internal-bench --bench ab          # headline A/B
cargo bench --bench chunking -- new_api                   # new API vs iterator
cargo bench --bench chunking                              # full descriptive suite
```

All inputs are generated in-process from a fixed SplitMix64 seed. No temp files,
nothing to clean up. `target/` is git-ignored.

---

## 6. Hostile review of measurement quality

> **Update:** the laptop critique below stands, but the result was since
> re-run on an isolated Fly dedicated-CPU x86_64 VM (§5c) where CIs are ±2–3%
> and the cross-group discrepancy is gone. The ~12% win reproduces there, on a
> different architecture, with the same correctness gate. That closes the two
> biggest holes (machine noise + single-arch). The remaining caveats —
> synthetic data, no `perf` counters, profile coupling, `cut_gear_legacy` as a
> stand-in symbol — still apply.

**The environment is bad for microbenchmarks, and the first numbers proved it.**
A single full-suite criterion pass reported the *same* operation (v2020 iterator
over 8 MiB random, avg 16K) at 450, 1070, and 1380 MiB/s in three different
groups — a 3× spread. Re-running each in isolation collapsed them to ~1.0–1.3
GiB/s. Diagnosis: cold-start/DVFS ramp on the first group plus memory pressure
from many live multi-MiB buffers. **Conclusion: no single-pass criterion delta
on this machine is trustworthy; CIs run ±10–20%.** That is why the headline
result does *not* rely on criterion.

How the headline result defends against that noise, and the residual holes:

- **Interleaving cancels slow drift.** Old and new run microseconds apart, in
  alternating order, in one process. Thermal state, frequency, and background
  load are ~equal for both, so they divide out of the ratio. *Hole:* a fast
  periodic interferer phase-locked to the alternation could bias one side — but
  it would have to survive order-swapping every round and produce the *same*
  1.13× across five different inputs and two runs. Implausible.
- **Min-of-N, not mean.** For a deterministic loop, noise is one-sided (adds
  time), so the min is the closest thing to the true cost. *Hole:* if the new
  code's min is reached more often by luck it could flatter slightly — but the
  medians move by the same ~13%, so it is not a min artifact.
- **Asm is the ground truth.** 8→0 bounds checks is a fact about the binary, not
  a timing. The 13% is the *consequence*; even if every wall-clock number were
  thrown out, the work provably shrank. *Hole:* fewer instructions need not mean
  faster on an out-of-order core where the bounds-check branch is perfectly
  predicted — which is exactly why the wall-clock A/B exists to confirm the asm
  win is real and not just cosmetic.
- **`cut_gear_legacy` is a separate symbol** from the original inlined loop, so
  its codegen could differ from "true 4.0.1." Mitigation: its source is the
  original loop body verbatim, and its measured throughput matches the §5b
  criterion baseline of the unmodified crate. It is a faithful stand-in, not the
  literal original call site.
- **Synthetic data.** SplitMix64 fill, not real corpora. Defensible because scan
  cost is content-independent per byte; content only sets the cut-point count,
  which the zeros (max cuts skipped → longest scans) and random (typical) cases
  bracket. Real CinchFS data could differ in *chunk count*, not in per-byte cost.
- **No instruction/branch counters.** macOS has no easy `perf stat`; there are
  no cycles-per-byte or branch-miss numbers here, which would have made the asm
  argument quantitative rather than a count.
- **Profile coupling.** Numbers are for `opt-level=3 + thin-LTO + 1 CGU`. A
  consumer building differently will see different absolute throughput; the
  bounds-check *removal* holds regardless, but the *13%* is specific to this
  build and this CPU.

**What would make this airtight (out of scope for a laptop spike):** an isolated
Linux box, `taskset` pinning, `cpupower` perf governor / turbo disabled,
`perf stat` for IPC and branch-misses, `hyperfine` for process-level CIs, and a
real mixed corpus. With those, I would report cycles/byte before/after rather
than a wall-clock ratio.

---

## 7. Recommendation

Adopt the patch. It is bounds-check-only, preserves every existing cut point
(proven by asm + the upstream digest tests + the A/B gate), and buys a
reproducible ~13% on the hottest loop in the chunker. The additive `Chunker`
API gives CinchFS a zero-allocation per-segment hashing path when roadmap item
#14 (content-defined chunking) lands, without committing to it now.

No `unsafe` was added. The source-read bounds checks were removed by narrowing
the slice and hoisting the loop bound — the safe way — rather than
`get_unchecked`. A further `unsafe` pass over the inner loop was considered and
rejected: the safe change already reaches 0 bounds checks, so `unsafe` would add
risk for no measured gain.

---

## 8. Appendix: I/O vs CPU, and where io_uring fits (forward-looking)

Framed by Enberg, Rao & Tarkoma, *"I/O Is Faster Than the CPU — Let's Partition
Resources and Eliminate (Most) OS Abstractions"* (HotOS '19).

**The paper's thesis.** I/O has caught up to the CPU. 400 GbE NICs and NVM near
DRAM speed, while single-thread CPU has stagnated. Their sharpest number: a
40 GbE NIC delivers a cache-line every **~5 ns**, but one **LLC access is ~15 ns**
— a single cache miss already makes the CPU miss the next packet. The
consequence: per-operation CPU cost and cache pollution (syscalls, context
switches, shadow copies, the page cache) now dominate, not I/O wait. Their fix:
**partition hardware per core** (NIC/NVMe queues, DRAM), keep the kernel out of
the data plane, thread-per-core, app-managed buffers, no kernel page cache.

**What this means for chunking.** FastCDC's scan touches every byte once and
reuses nothing, so for large inputs it streams through cache and runs near
memory bandwidth — on the Fly box ~2.2 GiB/s/core. That puts the chunker
squarely in the paper's regime: **the CPU/memory system is the scarce resource,
not the disk.** Two consequences:

1. The naive "io_uring overlaps slow I/O with compute" framing is only half
   right. For **cold tier data (Tigris, ~250 ms)** I/O latency genuinely
   dominates and overlap/prefetch wins big. For **warm local NVMe** (several
   GiB/s) the chunker is at or near the bottleneck — there io_uring's value is
   *not* hiding latency but **not stealing CPU/cache from the scan**: no
   per-read syscall, no context switch, and zero-copy via **registered (fixed)
   buffers** so bytes land in the app buffer the scan already owns. My zero-alloc
   `Chunker::for_each_chunk` is the application-side half of that; today's
   `StreamCDC` does the opposite (per-chunk `Vec` + `copy_within`).

2. The real scaling lever in this regime is **parallelism**, exactly as the
   paper argues — one chunker per core, each with its own ring and file/NVMe
   queue, shared-nothing. This already matches cinch's "one engine owns each
   tenant, no fan-out" model and the CLAUDE.md `spawn_blocking` rule (don't let
   one tenant's CPU-bound scan stall a Tokio worker multiplexing others).

**Concrete recommendation (when chunking is actually built — roadmap #14):**
keep the library pure and in-memory (the `Chunker` API), and do the Linux I/O
trick *outside* it: a thread-per-core reader using `io_uring` with O_DIRECT +
registered buffers, double-buffered so the next block DMAs in while the current
one is chunked, then hand each ready block to `Chunker::for_each_chunk`. Prefer
this over `mmap`, which the paper flags as effectively a blocking interface
(surprise page-fault stalls, eviction outside app control). Avoid putting
chunking behind a shared async worker pool. And measure against the ~2.2 GiB/s
ceiling: if a single core can't keep the device busy, add cores, don't add async.

**Honest caveat.** This is forward-looking. FastCDC is not integrated yet, and
the largest io_uring payoff in cinch is almost certainly the *storage/sync data
plane* (page fetches, WAL shipping — many concurrent kernel-mediated I/Os),
where the paper's syscall/copy critique bites hardest. Chunking is a small
CPU-bound consumer at the end of that pipe.
