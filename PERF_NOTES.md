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

Noise-free evidence (emitted asm, x86-64-v3, counting `panic_bounds_check`
within the `cut_gear` function body):

```
panic_bounds_check sites inside cut_gear:   8   (baseline / cut_gear_legacy)
panic_bounds_check sites inside cut_gear:   4   (patched)
```

The original 8 = {GEAR_LS, GEAR, source[a], source[a+1]} × {pre-center loop,
post-center loop}. The patch removes the **4 GEAR-table checks**; the **4
source-index checks remain**.

> **Correction.** An earlier draft claimed "8 → 0". That was a measurement bug:
> an `awk` range keyed on the *inlined* `cut_gear_arr` name matched nothing, so
> `grep -c` reported 0 on an empty range. Counting within the real `cut_gear`
> symbol gives 8 → 4. The measured speedup (§5) is unaffected — it comes from
> the interleaved A/B with a correctness gate, not from the check count. And per
> §6a the remaining 4 checks are free anyway, so removing them is not worth
> doing.

### The change

`cut_gear` now, once per call (not per byte):

1. `let gear: &[u64; 256] = gear.try_into().expect(...)` (same for `gear_ls`).
   The gear hash table is 256 entries by construction; non-256 input is a
   programming error and panics at the boundary instead of silently. The inner
   loop then indexes arrays → no table bounds check. **This is the win** (4
   checks gone).
2. `let src = &source[..remaining];` and hoists `limit1 = center/2`,
   `limit2 = remaining/2`. Intent was to also drop the `src[a]` / `src[a+1]`
   checks, but the compiler does **not** prove `2*index+1 < remaining` here, so
   those 4 checks remain. Kept anyway: harmless, and (§6a) free in practice.

The arithmetic is byte-for-byte identical; only the *provenance* of the indices
changed. The original loop body is retained verbatim as `cut_gear_legacy`
(behind the `internal-bench` feature) for the A/B harness.

### Proof that cut points are unchanged

- **Existing upstream tests** pin exact cut points, exact 64-bit hashes, and
  exact BLAKE3 digests for the `SekienAkashita.jpg` fixture across NC levels 0/1/3,
  seeds 0 and 666, and 16K/32K/64K parameters — for `cut()`, the iterator, and
  `StreamCDC`. All pass unchanged (54 default-feature tests in the fork).
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

## 6a. Why the loop is latency-bound (llvm-mca), and what that implies

To decide whether removing the remaining 4 source checks was worth it, I ran
`llvm-mca` (static pipeline model) on the actual inner loop for `znver3` (AMD,
the likely Fly uarch), comparing the patched loop (4 source checks) against a
hand-stripped loop with those checks removed:

```
                       Total Cycles (1000 iters)   uOps/cycle
patched (4 checks)            6010                    2.83
stripped (0 checks)           6010                    2.00
```

**Identical cycle counts.** Removing the checks drops uOps/cycle (less work per
cycle) but not total time — the freed slots were idle anyway. The loop is
**bound by the hash dependency chain** (`hash = (hash<<2) + G[b0]`, then
`+ G[b1]` — each iteration's `hash` needs the previous one), not by instruction
throughput. The bounds-check compares are independent of that chain, so they
execute in the shadow of the latency and cost ≈0.

Implications, which steer all future work:
- **Do not chase the remaining 4 source checks** — proven 0 cycle benefit.
- Micro-optimizing the loop body in general is a dead end; we are latency-bound,
  not throughput- or instruction-count-bound.
- The only ways past the ~2.2 GiB/s/core wall are to **break the chain**
  (SIMD-parallel gear hashing, SS-CDC style — start hashing from many offsets at
  once, since the gear hash only "remembers" the last few dozen bytes) or to
  **stop re-reading the bytes** (fuse per-chunk hashing into the scan so the
  bytes are hashed while still hot in L1), or to **scale across cores**.
- Caveat: `llvm-mca` is a model. It also predicts the GEAR-check removal should
  be near-free, yet the hardware A/B measured ~12%. Treat the *relative* "checks
  are free" result as robust and the absolute cycle figure as approximate; where
  model and hardware disagree, the interleaved A/B on real silicon wins.

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
- **Asm is the ground truth.** 8→4 bounds checks is a fact about the binary, not
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

No `unsafe` was added. The 4 GEAR-table checks were removed safely via fixed
array typing. The 4 source-index checks remain, and `unsafe` `get_unchecked`
was considered and rejected to remove them: §6a shows they cost ~0 cycles
(latency-bound loop), so `unsafe` would add risk for no measured gain.

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

---

## 9. Experiment: fuse chunking with per-chunk hashing (`benches/fuse.rs`)

A content-addressed store hashes every chunk. Question: hash each chunk the
instant its boundary is found (while hot in L1, via `Chunker::for_each_chunk`),
or scan first and hash in a second pass? Interleaved min-of-31, laptop, 32 MiB
random (ratios are trustworthy; the single-shot hash-only figures are rough):

```
16 KiB chunks:   BLAKE3  two-pass 451 -> fused 715 MiB/s   1.58x
                 SHA-256 two-pass 220 -> fused 221 MiB/s   1.01x
1 MiB chunks:    BLAKE3  two-pass 788 -> fused 791 MiB/s   1.00x
                 SHA-256                                    1.00x
reference:       chunk-only 957 (16K) / 1944 (1M);  hash-only BLAKE3 ~770-1316
```

Findings:
- **Fusing wins big (1.58x) only when both hold:** the hash is fast
  (BLAKE3, memory-bound) *and* the chunk fits in L1 (~16 KiB). Then the second
  pass would re-read the chunk from L2/L3; fusing hashes it while still in L1.
- **No benefit for 1 MiB chunks** — a 1 MiB chunk is far bigger than L1, so its
  front is already evicted by the time the boundary is found; both paths re-read.
- **No benefit when the hash is the bottleneck** (SHA-256 here). Which surfaces
  the real headline:
- **SHA-256 is running unaccelerated at ~134 MiB/s.** The `sha2` crate fell back
  to the portable implementation (no aarch64 crypto / x86 SHA-NI on this build).
  In any SHA-256 content-addressing pipeline that 134 MiB/s dwarfs everything —
  the chunker's ~2 GiB/s and even BLAKE3's ~1 GiB/s are irrelevant next to it.
  **Enabling SHA hardware (sha2 `asm`/intrinsics) is a >10x lever and matters far
  more than any chunk-loop tweak.** This is a cinch-integration concern (the
  library doesn't hash), but it's the single most important thing this whole
  spike found about the *end-to-end* cost.
- **Takeaway for cinch:** in a chunk+hash pipeline, **hashing dominates**, not
  chunking. Pick a SIMD hash (BLAKE3, or hardware SHA), use small (~16 KiB)
  chunks, and fuse hashing into `for_each_chunk`. The ~12% chunk-loop win is real
  but small next to getting the hash right.

(Numbers are directional laptop figures; confirm on the Fly dedicated-CPU box,
and re-check SHA-256 with the `asm` feature enabled.)

---

## 10. Experiment: parallel chunking (`benches/parallel.rs`)

The single core is latency-bound (§6a). Data parallelism is the way past it. Two
questions, both answered:

**A. Fidelity — does region-split chunking match serial?** FastCDC resets its
hash to 0 after every cut and restarts the search at `cut+min_size`, so two
chunkers that ever cut at the same absolute offset are *identical forever after*.
An independent chunker started at an arbitrary midpoint of a 64 MiB buffer:

```
resync distance past the split:  16,917 bytes  (= 1.0 average chunk)
interior boundaries reproduced after resync:  1691/1691  (100.0000%)
```

So FastCDC is **self-synchronizing**: split it anywhere and the only chunks that
differ from serial are the 1–2 straddling each seam. An N-way split has N−1
seams → ~N−1 divergent chunks. For 64 MiB / 16 KiB (~4000 chunks), an 8-way
split perturbs ~7 chunks = **0.17%** — a negligible dedup hit. Identical on M1
and x86 (it's deterministic).

> This refines a common warning (e.g. elemeng/chunkrs docs: "do not parallelize
> within a single file — this destroys deduplication ratios"). With *controlled*
> region splits that is too strong: the measured loss is the handful of seam
> chunks, not the ratio wholesale. The warning holds for *uncontrolled* batching
> where split points are arbitrary and unbounded.

**B. Throughput scaling** (chunk a 64 MiB buffer split across N threads, min-of-15):

```
              M1 Pro (8 core)        Fly performance-2x (2 dedicated vCPU)
 1 thread     1879 MiB/s  1.00x      2275 MiB/s  1.00x
 2 threads    3772 MiB/s  2.01x      4147 MiB/s  1.82x
 4 threads    7494 MiB/s  3.99x      (only 2 cores)
 8 threads    8294 MiB/s  4.41x
```

Near-linear per available performance core (M1: perfect to 4, tapers at 8 over
its 2 efficiency cores + memory bandwidth; Fly: 1.82x on its 2 cores). Combined
with the per-core ~12%, this is the real path past the 2.2 GiB/s/core wall —
and it matches cinch's shared-nothing, one-owner-per-tenant model.

## 11. x86 confirmation (Fly dedicated CPU) — the dominance flips

Re-running `fuse` on the Fly x86 box, where the `sha2` crate auto-uses **SHA-NI**:

```
                       M1 laptop (portable)     Fly x86 (SHA-NI / AVX2)
 hash-only SHA-256          134 MiB/s                1514 MiB/s   (11.3x)
 hash-only BLAKE3       ~770-1316 MiB/s              ~2700 MiB/s
 chunk-only             ~960-1944 MiB/s              ~2000-2256 MiB/s
 BLAKE3 fuse vs 2-pass       1.58x (16K)              1.07x
```

Two conclusions:
1. **SHA-256 was an artifact, not a law.** Unaccelerated portable SHA on the M1
   (no `asm` feature for aarch64) read as "hashing dominates." On any x86 server
   (SHA-NI is in every Zen and recent Intel) it's ~1.5 GiB/s out of the box.
2. **Which stage dominates depends on the hardware.** On x86 with hardware
   hashing, BLAKE3 (~2.7) and SHA-256 (~1.5) are as fast as or faster than the
   gear-hash scan (~2.0), so **the chunk loop becomes the bottleneck** — the
   opposite of the laptop. That *raises* the value of the per-core chunk win and
   parallel chunking on real servers, and shrinks the fuse win (faster caches +
   hashing hide the re-read; 1.58x → 1.07x).

The honest, hardware-aware summary: on a modern x86 server the chunk scan and the
hash are within ~1.5x of each other, so a content-addressing pipeline wants *both*
a fast (SIMD/parallel) chunker and hardware hashing — neither alone is enough.

---

## 12. Related work / algorithm landscape (and a premise check)

This spike set out to *optimize FastCDC*. Surveying the field suggests the bigger
question is *whether to use FastCDC at all*. Cinch has **no deployed chunks**
(content-defined chunking is roadmap #14, not built), so there is **no
cut-point-compatibility constraint** — we are free to pick the best algorithm,
not bound to FastCDC's.

| Project | Idea | Speed | Dedup | Notes for cinch |
|---|---|---|---|---|
| **fastcdc** (this crate) | gear hash, dual-mask NC | ~2 GiB/s/core (latency-bound) | baseline | what we patched; serial chain is the wall (§6a) |
| **[orlp/mincdc](https://github.com/orlp/mincdc)** | **sliding-window minimum** cut points (SIMD, 4-byte window) | **41 GB/s** (9950X) vs FastCDC 6.6; 23.8 vs 4.1 on M2 | **61% vs 54%** on Linux kernel | **strong candidate.** Faster *and* better dedup *and* near-uniform chunk sizes (no long tail). Zlib license. |
| **[srijs/rust-gearhash](https://github.com/srijs/rust-gearhash)** | gear hash + `next_match()`, SSE4.2/AVX2/NEON | multi-GiB/s | same family as FastCDC | the SIMD gear primitive; single-mask, different formulation → *different cut points*. Apache/MIT, `unsafe`+fuzzed. |
| **[elemeng/chunkrs](https://github.com/elemeng/chunkrs)** | FastCDC, modern API, `#![forbid(unsafe)]`, `Bytes` zero-copy | ~3–5 GB/s target | FastCDC-equivalent | clean base if we stay on FastCDC; its roadmap (SIMD, HW hash) = our findings. |
| **[QuickCDC](https://joshleeb.com/posts/quickcdc.html)** | feature-vector (front/end 3 bytes + len) skip table | very fast (skips hashing) | up to 2.2x in cases | **disqualified for cinch:** treats same-features+length as duplicate → can mis-dedup altered middles → **data loss**. CLAUDE.md: data loss is unacceptable. |
| **[chonkie-inc/chunk](https://github.com/chonkie-inc/chunk)** | **semantic *text* chunking** (periods/newlines, SIMD) | ~1 TB/s | n/a | **different domain.** This is RAG/LLM text splitting, *not* dedup byte-CDC. Relevant to cinch's *agent/RAG* side, not storage dedup. Don't conflate the two "chunking"s. |

**The key connection to §6a.** FastCDC's gear hash is a *serial dependency
chain*, which is exactly why one core caps at ~2 GiB/s and SIMD can't easily help
it. **mincdc sidesteps the whole problem**: a sliding-window *minimum* over a
4-byte window is a vectorizable min-reduction, not a recurrence, so it hits
40+ GB/s on one core. It doesn't *break* FastCDC's chain — it *avoids having
one*. That, plus the better dedup ratio and the near-uniform chunk-size
distribution (which is also nicer for cinch's per-GB tiering/billing and for
bounded extents), makes mincdc the most interesting lead by a wide margin.

**Revised recommendation.**
1. **Evaluate mincdc head-to-head** against this patched FastCDC on cinch-shaped
   data (agent files: overwrite/append, text+binary), measuring throughput,
   dedup ratio, and chunk-size distribution. If it holds up, it likely beats
   anything we could do to FastCDC.
2. **Keep the patched FastCDC + `Chunker` as the safe default** in the meantime —
   it's correct, tested, and shipped.
3. **Reject QuickCDC** for any durability-critical path (data-loss risk).
4. **Hardware hashing is mandatory regardless of chunker** (§9, §11): ensure
   SHA-NI / BLAKE3-SIMD is on; it's an 11x lever and orthogonal to the chunker.
5. **chonkie is a separate tool** for the (separate) text/RAG chunking need.

This turns the "make FastCDC faster" task into a clearer strategic picture:
the per-core FastCDC ceiling is ~2 GiB/s and latency-bound; mincdc reports ~20x
that with better dedup; so the highest-value next step is a rigorous mincdc
bake-off, not more FastCDC micro-optimization.

## 13. The authoritative survey, and the methodology for choosing

Gregoriadis, Balduf, Scheuermann & Pouwelse, *"A Thorough Investigation of
Content-Defined Chunking Algorithms for Data Deduplication"* (submitted to IEEE
TCC, Sept 2024, arXiv:2409.06066) is exactly the impartial comparison this
decision needs. It re-implements and benchmarks the whole field on **four real
datasets** across **four metrics** (throughput, dedup ratio, average chunk size,
chunk-size variance), and explicitly calls out the bias in prior single-algorithm
papers.

Its taxonomy (which places everything above):
- **BSW / rolling-hash + mask:** Rabin (1981), Buzhash (1997), **Gear (2014)**,
  PCI (2020). **FastCDC = Gear + normalized chunking + cut-point skipping.** This
  is the family we patched — and the one with the serial-hash latency wall.
- **Local-extrema (no hash):** **AE (2015), RAM (2017), MII (2019)** — cut on a
  local max/min over a window. The paper notes these "achieve higher throughput
  than BSW algorithms" with "significantly lower chunk-size variance." **This is
  the family mincdc belongs to** — and it's the throughput + low-variance winner.
- **Statistical:** BFBC (2020) — byte-pair frequency table lookup.

Concrete takeaways for cinch:
- The paper independently corroborates our findings: it footnotes the
  `gearhash` SIMD crate and notes Gear "is easy to SIMD-parallelize" with
  data-parallelism (our §10), and it confirms extremum methods (mincdc's family)
  are faster with lower variance.
- It warns that prior throughput numbers are often **skewed by bundling the SHA
  fingerprint into the measurement** (our §9/§11: hashing can dominate, and is
  hardware-dependent) and by **artificial datasets** (zeros + random insertions)
  that ignore low-entropy strings. Our own spike used synthetic SplitMix64 data —
  same caveat applies; a real bake-off must use cinch-shaped data.
- It derives corrected expected-chunk-size formulas for AE/RAM/MII/BFBC — useful
  if we tune an extremum-based chunker to a target average for cinch's tiering.

### Bottom line of the whole spike

1. **Shipped, safe, done:** FastCDC patch (8→4 bounds checks, ~12% per core,
   identical cut points) + zero-alloc `Chunker` API + a reproducible benchmark
   suite (A/B, criterion, fuse, parallel) + neutral-env x86 validation.
2. **Profiling verdict:** one FastCDC core is latency-bound at ~2 GiB/s; you get
   past it with **data parallelism** (§10: near-linear, 100% fidelity) and
   **hardware hashing** (§9/§11: 11x), *not* loop micro-opts (§6a).
3. **Strategic lead:** the per-core ceiling is an artifact of the gear-hash
   *recurrence*. **Extremum-based CDC (mincdc / AE / RAM / MII) avoids the
   recurrence entirely** — mincdc reports ~20x throughput *and* better dedup *and*
   lower size variance. Since cinch has no deployed chunks, it's free to choose.
4. **Next step:** run the 2024-survey methodology — FastCDC vs mincdc (and
   maybe RAM/MII) on real cinch data, scoring throughput + dedup ratio + chunk
   size variance — before committing chunking for roadmap #14. Reject QuickCDC
   (data-loss risk). Keep chonkie in mind only for the separate text/RAG need.

## 14. The bake-off: FastCDC vs MinCDC (done)

Built the §13 bake-off (`third_party/cdc-bakeoff` in cinch-cloud; depends on this
fork + `mincdc`). Deterministic ~40 MiB corpus with realistic redundancy (10
text bases × 6 edited versions — append/insert/overwrite — + 4 binary blobs),
all algorithms tuned to **equal mean chunk size** (the only fair dedup test;
FastCDC's quantized mean sets the reference). Metrics: throughput (chunk-only),
dedup% (BLAKE3 fingerprints), and chunk-size distribution. M1 Pro / NEON:

```
mean ~10.3 KiB        MiB/s   dedup%    CV    p99
FastCDC (patched)      1651   74.48%   0.52   29205   (baseline)
MinCDC4               10155   74.87%   0.47   18924   (6.2x faster, tighter tail)
MinCDCHash4            4992   74.56%   0.57   29392   (3.0x faster, robust default)
fixed-size control        —   59.88%   0.09       —   (dedup collapses: boundary shift)
```

**Verdict: MinCDC matches FastCDC's dedup while running 3–6x faster** with
equal-or-more-uniform sizes. The 16 KiB target gives the same shape. The speed
ratio matches mincdc's published ~6x; on dedup it's a tie here (mincdc's README
shows an *advantage* on real Linux-kernel data — data-dependent, but never
worse). **Recommend MinCDCHash4** (the author's robust default — MinCDC4 can
skew sizes on adversarial input; cinch's data-integrity bar wants the robust
one, still 3x faster than FastCDC at equal dedup).

**x86 confirmation (Fly, AMD EPYC, AVX2):** the lead *grows* on a real server
CPU — MinCDCHash4 **3.3–3.8x** faster than FastCDC, MinCDC4 **6.6–7.7x**, and at
64 KiB chunks MinCDC also *wins* dedup (+1–3 pts) with lower CV. A size sweep
(256 KiB→2 GiB) shows FastCDC flat at ~2.2 GiB/s (latency-bound, never reaches
memory bandwidth) while MinCDC peaks at ~4 MiB (L3) then declines toward 2 GiB
(memory-bound) — the faster algorithm hits the RAM wall first. **Profiling
mincdc found no FastCDC-style easy win:** its AVX2 path is already expertly
optimized (32 B/step, 2x unroll, prefetch, vectorized argmin, no bounds checks);
the only lever is parallelism across cores (it's self-synchronizing too). Full
method + numbers: `cinch-cloud/third_party/cdc-bakeoff/BAKEOFF.md`. Still worth a
real cinch-data sample before locking the choice.

**Bottom line of the whole effort:** the FastCDC patch is a fine, safe local win
(~12%/core, identical cut points, shipped). But the strategically correct move
for cinch is to **adopt MinCDCHash4 for roadmap #14** — 3x the throughput at the
same dedup and similar uniformity — rather than invest further in FastCDC.
