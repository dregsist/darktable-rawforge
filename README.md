# darktable Demosaic Benchmark

Benchmark data and reproducible scripts for PR [darktable-org/darktable#20800](https://github.com/darktable-org/darktable/pull/20800) which adds Menon (2007) and ARI (Monno 2015) demosaic methods to darktable.

- **[BENCHMARK.md](BENCHMARK.md)** — full writeup (English)
- **[BENCHMARK_JA.md](BENCHMARK_JA.md)** — full writeup (Japanese / 日本語)

## Quick summary

| method | Kodak-24 low-ISO CPSNR | Chroma zipper mean | Speed per 24 MP (est.) |
|---|---|---|---|
| rcd (default) | 36.91 | 0.115 | ~1.5s |
| amaze | 39.13 | 0.073 | ~2s |
| amaze+dual | 38.76 | 0.073 | ~4s |
| rcd+dual | 36.58 | 0.115 | ~3s |
| menon_r1 | 39.08 | 0.070 | ~1.5s |
| **ari** | **39.94** | **0.062** | **~80s (!)** |

See `BENCHMARK.md` for complete data across Kodak-24 low/mid/high ISO, SIDD medium real-raw iso_clean/iso_noise, and chroma-zipper metrics.

## Repository layout

```
BENCHMARK.md                     # English writeup
BENCHMARK_JA.md                  # Japanese writeup
_paper_ari_ref.py                # Python paper-exact ARI reference (matches Monno MATLAB)
_paper_ri_ref.py                 # RI-only Python reference
bench_*.py                       # benchmark scripts (Kodak, SIDD, zipper, iter sweep, ...)
_paper_vs_simple_*.py            # edge-image comparison generators
test_ari.c / test_menon.c / ...  # standalone C test harnesses for each demosaic method
stubs.h                          # darktable API stubs for the test harnesses
build_gcc_omp.sh                 # build script (MSYS2 UCRT64 + GCC)
viewer.html                      # split-comparison HTML viewer for the PNG outputs
viewer_data_paper_vs_simple/     # 5 Kodak edge images × 7 methods, PNG outputs
refs/matlab/                     # original MATLAB source of RI (TIP) / MLRI (EI2014) / ARI (ICIP2015)
dt_tests_dtbuild/                # end-to-end mire1.cr2 JPEG outputs from darktable-cli
```

## Reproducing the benchmarks

See `BENCHMARK.md` section "Reproduction" for commands. TL;DR:

```bash
bash build_gcc_omp.sh
python bench_final_all.py --iso low
python bench_final_all.py --iso mid
python bench_final_all.py --iso high
python bench_zipper_all_omp.py
python bench_sidd_final.py
python bench_ari_iter_sweep.py
```

Kodak-24 images expected at `C:/Users/dregsist/datasets/kodak/kodim{01..24}.png` (adjust paths inside scripts for other locations). SIDD medium at `C:/Users/dregsist/datasets/sidd/medium_bench/*.npy` (see `bench_sidd_final.py` for the expected format).
