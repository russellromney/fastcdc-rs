# Profiling the `v2020::cut_gear` hot loop

Reproduces the two profiling results behind `PERF_NOTES.md`: the bounds-check
count in the emitted assembly, and the `llvm-mca` finding that the remaining
source-index checks are free because the loop is latency-bound on the gear-hash
recurrence.

## 1. Count `panic_bounds_check` in `cut_gear`

```sh
# emit assembly for the library
cargo rustc --release --lib -- --emit asm -C opt-level=3
ASM=$(ls -t target/release/deps/fastcdc-*.s | head -1)

# count the bounds-check sites inside the cut_gear function body
awk '/8cut_gear17h.*E:$/{f=1} f && /\.cfi_endproc/{print c+0; exit} f && /panic_bounds_check/{c++}' "$ASM"
```

Expect **4** on the current code (the 4 source-index checks; the 4 GEAR-table
checks were removed by #44). Build the `internal-bench` feature to also get
`cut_gear_legacy`, the pre-#44 body, which shows **8**:

```sh
cargo rustc --release --lib --features internal-bench -- --emit asm -C opt-level=3
```

## 2. `llvm-mca`: the remaining checks are free

`loop_checks.s` is the current inner loop (4 source-index checks);
`loop_clean.s` is the same loop with those compares hand-removed. Both are
wrapped in `# LLVM-MCA-BEGIN/END` markers.

```sh
for f in loop_checks loop_clean; do
  echo "=== $f ==="
  llvm-mca -mcpu=znver3 -mtriple=x86_64-unknown-linux-gnu -iterations=1000 profiling/$f.s \
    | grep -E "Total Cycles|uOps Per Cycle|Block RThroughput"
done
```

Both report the **same total cycles** (≈6010 for 1000 iterations). Removing the
checks lowers uOps/cycle but not wall-clock: the loop is bound by the
`hash = (hash << 2) + GEAR_LS[b0]; hash += GEAR[b1]` dependency chain, and the
bounds-check compares execute in the shadow of that latency.

(`llvm-mca` ships with LLVM; on macOS it is at `$(brew --prefix llvm)/bin/llvm-mca`.)

## 3. The A/B throughput benchmark

`benches/ab.rs` times the pre-#44 `cut_gear` (`cut_gear_legacy`, behind the
`internal-bench` feature) against the current one, interleaved in one process so
thermal drift cancels, reporting the minimum over many rounds. It also asserts
both produce identical cut points before timing.

```sh
cargo bench --features internal-bench --bench ab
```

`benches/chunking.rs` (criterion) gives descriptive throughput; `benches/fuse.rs`
and `benches/parallel.rs` back the two observations in `PERF_NOTES.md` §3
(hash-while-chunking, and self-synchronizing region-parallel chunking).
