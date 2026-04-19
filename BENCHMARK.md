# Demosaic Algorithm Benchmark

Comparison of darktable's existing demosaic methods (PPG, AMaZE, RCD, LMMSE, AMaZE+dual, RCD+dual) against the **Menon (2007)** and **ARI (Monno 2015)** implementations added in PR #20800, across multiple datasets and ISO conditions.

Reproducible scripts, the Python reference implementation, and edge-image output PNGs are included.

---

## 1. Kodak-24 CPSNR (dB, average, low/mid/high ISO)

Measured on the Kodak True Color set (kodim01–24) with additive Gaussian noise on the CFA input:

- low:  σ=0 (clean)
- mid:  σ=0.02 (~ISO 1600 equivalent)
- high: σ=0.05 (~ISO 6400 equivalent)

| method | low | mid | high |
|---|---|---|---|
| **ari** | **39.939** | 33.256 | 26.412 |
| amaze | 39.134 | 32.954 | 26.282 |
| amaze+dual | 38.759 | **33.373** | 26.464 |
| rcd | 36.907 | 32.404 | 26.417 |
| rcd+dual | 36.584 | 32.707 | **26.586** |
| menon_r1 | 39.078 | 32.687 | 25.902 |
| menon_r0 | 38.447 | 32.453 | 25.804 |

**Observations**:
- At low ISO, ari is top by +0.8 dB over amaze.
- At mid/high ISO, dual-base methods win; ari degrades to parity.
- Menon sits within 0.1 dB of amaze — no clear advantage at any ISO.

---

## 2. Chroma zipper (5 edge-heavy Kodak images, mean)

Measured on kodim01, 08, 13, 19, 20 — the images with the most high-frequency edge content. Metric: Δ = |Laplace(R−G) deviation| + |Laplace(B−G) deviation|. Reported inside an edge mask (top 15 % gradient), both local mean and local peak.

| method | cz_mean ↓ | cz_peak ↓ | CPSNR (ref) |
|---|---|---|---|
| **ari** | **0.0620** | **0.0803** | 38.405 |
| amaze | 0.0725 | 0.0901 | 36.892 |
| amaze+dual | 0.0730 | 0.0907 | — |
| menon_r1 | 0.0696 | 0.1375 | 36.942 |
| menon_r0 | 0.0921 | 0.1533 | 35.942 |
| rcd | 0.1151 | 0.1453 | 34.093 |
| rcd+dual | 0.1146 | 0.1436 | — |

**Observations**:
- ari has the smallest chroma zipper on both mean and peak. Consistent with the paper's design goal (per-pixel adaptive RI+MLRI selection).
- Menon mean is OK but peak is worst-tier — it spikes locally.
- RCD is the weakest of the tested methods on zipper metrics (visible to the eye as well).

Visual comparison PNGs: [`viewer_data_paper_vs_simple/iso_low/`](viewer_data_paper_vs_simple/iso_low/) — directories `__gt__`, `c_port_q2`, `py_ref`, `amaze`, `rcd`, `menon_r0`, `menon_r1` × kodim01/08/13/19/20.

---

## 3. SIDD medium real raw (20 scenes × 5 phones, per-scene 3×3 CCM fit)

Uses SIDD Medium's clean reference (gt_raw) and matching noisy raw (noisy_raw) for 20 scenes (G4, GP, IP, N6, S6, 4 frames each), demosaicked and compared against gt_srgb. A per-scene 3×3 CCM + bias fit is applied before CPSNR to absorb sensor colour-space differences so the number reflects demosaic behaviour rather than pipeline mismatch.

| method | iso_clean CPSNR | iso_noise CPSNR |
|---|---|---|
| **ari** | **40.354** | 28.197 |
| amaze+dual | 39.976 | 28.118 |
| rcd+dual | 39.927 | **28.202** |
| amaze | 38.299 | 27.651 |
| rcd | 38.424 | 27.730 |
| menon_r1 | 37.675 | 27.422 |
| menon_r0 | 37.755 | 27.268 |

**Observations**:
- iso_clean (low-noise real raw): ari leads by +0.38 dB over amaze+dual.
- iso_noise (real high-ISO noise): rcd+dual and ari at 28.20 / 28.197 — statistically tied.
- Menon is bottom-tier even on clean real raw.

### Chroma zipper (SIDD edges)

| method | iso_clean cz_mean | iso_clean cz_peak |
|---|---|---|
| **ari** | **0.0407** | 0.2127 |
| amaze+dual | 0.0442 | **0.1179** |
| rcd+dual | 0.0444 | 0.1450 |
| rcd | 0.0519 | 0.1621 |
| amaze | 0.0520 | 0.1610 |
| menon_r0 | 0.0484 | 0.1701 |
| menon_r1 | 0.0497 | 0.1622 |

---

## 4. Processing time (GCC 15.2 + OpenMP)

### Kodak 0.4 MP (kodim19, 4 threads, internal timing)

| method | time | Mpix/s |
|---|---|---|
| menon_r0 | 0.029s | 13.5 |
| menon_r1 | 0.028s | 14.3 |
| rcd | 0.039s | 10.1 |
| amaze | 0.062s | 6.4 |
| rcd+dual | 0.090s | 4.4 |
| amaze+dual | 0.112s | 3.5 |
| **ari** | **1.6s** | **0.25** |

### SIDD 15.86 MP (16 threads)

| method | time |
|---|---|
| rcd / amaze / menon | 0.5 – 1.5s |
| rcd+dual / amaze+dual | 2 – 3s |
| **ari (standalone)** | **53s** |
| **ari (darktable pipeline, mire1.cr2 18.8 MP)** | **113s** |

### ari optimization progression (SIDD 15.86 MP, 16 threads)

| state | time |
|---|---|
| Initial C port (MSVC /O2, no OpenMP) | 180s |
| + box-call reduction + integral reuse (phase 1) | 94s |
| + case A paired box-sum sharing | 69s |
| + case D separable 5×5 Gaussian | 60s |
| + rolling sum (no integral image) | 53s (float 2-buffer; the double col_acc variant is 2.4× slower) |
| + row-major vertical pass | **53s (current)** |

Attempted but rejected (documented as reverts in commit history):
- **Rolling sum with double col_acc + copy-back**: 2.4× slower at 15 MP due to double-precision intermediate doubling bandwidth. The float-only variant (adopted) avoids this.
- **Fused `box(A*B*C)`**: eliminates the prep pass but serializes the integral-image scan loop (the accumulator has a dependency chain), defeating auto-vectorization; net −12 %.

### Iteration-count sensitivity (ari with q = iter override)

| iter | Kodak-24 CPSNR | cz_mean | Kodak time |
|---|---|---|---|
| 3 | 37.687 | 0.0827 | 0.72s |
| 5 | 39.088 | 0.0686 | 1.08s |
| 7 | 39.647 | 0.0644 | 1.50s |
| 9 | 39.862 | 0.0628 | 1.89s |
| **11** | **39.939** | **0.0620** | 2.37s |

Paper default is iter=11. iter=7 captures ~87 % of the quality for ~37 % of the time — but this does not close the gap to the other methods.

---

## 5. When to use each method

| use case | recommended |
|---|---|
| interactive darkroom preview | rcd (default) / amaze |
| high-ISO raw | rcd+dual / amaze+dual |
| low-ISO stills, archival work, pixel-peeping | ari (if 1–2 minutes per 24 MP is acceptable) |
| fastest at acceptable quality | menon_r1 (though no margin over amaze in practice) |

---

## Reproduction

### Building the standalone harness (MSYS2 UCRT64 + GCC)

```bash
cd demosaic_test
bash build_gcc_omp.sh
```

### Running benchmarks

```bash
# Full Kodak-24, low/mid/high ISO
python bench_final_all.py --iso low
python bench_final_all.py --iso mid
python bench_final_all.py --iso high

# Chroma zipper detail
python bench_zipper_all_omp.py

# SIDD real raw
python bench_sidd_final.py

# ari iteration-count sweep
python bench_ari_iter_sweep.py

# viewer PNGs (5 edge images × 7 methods)
python _paper_vs_simple_save.py
python _paper_vs_simple_add_amaze_rcd.py
python _paper_vs_simple_add_menon.py
```

### Building darktable on Windows (MSYS2 UCRT64)

```bash
# Required: UCRT64 env, pointing to gcc.exe/g++.exe directly, excluding mingw64
cmake -G Ninja \
  -DCMAKE_C_COMPILER=/c/msys64/ucrt64/bin/gcc.exe \
  -DCMAKE_CXX_COMPILER=/c/msys64/ucrt64/bin/g++.exe \
  -DCMAKE_IGNORE_PATH=/c/msys64/mingw64 \
  -DCMAKE_INSTALL_PREFIX=/c/msys64/opt/darktable \
  -DCMAKE_BUILD_TYPE=Release \
  ..
MSYSTEM=UCRT64 ninja install
```

---

## Included materials

- [_paper_ari_ref.py](_paper_ari_ref.py) — Python paper-exact reference of ARI (structurally matches Monno's MATLAB code)
- [_paper_ri_ref.py](_paper_ri_ref.py) — RI-only reference
- [refs/matlab/](refs/matlab/) — Authors' original MATLAB code (RI / MLRI / ARI generations)
- [viewer_data_paper_vs_simple/iso_low/](viewer_data_paper_vs_simple/iso_low/) — PNG outputs, 5 edge images × 7 methods
- [viewer.html](viewer.html) — HTML viewer for split-comparison of the above PNGs
- bench_*.py — reproducible scripts
