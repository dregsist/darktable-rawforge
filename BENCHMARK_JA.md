# デモザイクアルゴリズム ベンチマーク結果

darktable 既存のデモザイク手法 (PPG, AMaZE, RCD, LMMSE, AMaZE+dual, RCD+dual) と PR #20800 で追加した **Menon (2007)** / **ARI (Monno 2015)** を、複数データセット・複数 ISO 条件で比較した結果集。

再現用スクリプト・Python 参照実装・Kodak エッジ画像出力を同梱。

---

## 1. Kodak-24 CPSNR (dB, 平均、低/中/高 ISO)

Kodak True Color Kodim01-24 に合成 Gaussian ノイズを加えた CFA で計測。ISO の対応:

- low: σ=0 (クリーン)
- mid: σ=0.02 (約 ISO 1600 相当)
- high: σ=0.05 (約 ISO 6400 相当)

| method | low | mid | high |
|---|---|---|---|
| **ari** | **39.939** | 33.256 | 26.412 |
| amaze | 39.134 | 32.954 | 26.282 |
| amaze+dual | 38.759 | **33.373** | 26.464 |
| rcd | 36.907 | 32.404 | 26.417 |
| rcd+dual | 36.584 | 32.707 | **26.586** |
| menon_r1 | 39.078 | 32.687 | 25.902 |
| menon_r0 | 38.447 | 32.453 | 25.804 |

**観察**:
- 低 ISO で ari が最高、+0.8 dB vs amaze
- 中・高 ISO では dual 系が最高、ari は競争力あるが首位は取らない
- menon は amaze と 0.1 dB 差、明確な優位性なし

---

## 2. Chroma zipper (エッジ 5 画像平均)

エッジ画像 (kodim01, 08, 13, 19, 20) でグラデ付近の色モアレ/ジッパー強度を計測。
Δ = |Laplace(R−G) の偏差| + |Laplace(B−G) の偏差|、エッジマスク内平均と局所ピーク。

| method | cz_mean ↓ | cz_peak ↓ | CPSNR (参考) |
|---|---|---|---|
| **ari** | **0.0620** | **0.0803** | 38.405 |
| amaze | 0.0725 | 0.0901 | 36.892 |
| amaze+dual | 0.0730 | 0.0907 | — |
| menon_r1 | 0.0696 | 0.1375 | 36.942 |
| menon_r0 | 0.0921 | 0.1533 | 35.942 |
| rcd | 0.1151 | 0.1453 | 34.093 |
| rcd+dual | 0.1146 | 0.1436 | — |

**観察**:
- ari は chroma zipper が最小 (mean, peak 共に)。論文設計 (per-pixel adaptive RI+MLRI) を数値で裏付け
- menon は mean 良いが peak が悪い (局所的にスパイク)
- rcd は zipper 指標で最悪 (目視でも確認可能)

目視比較用 PNG: [`viewer_data_paper_vs_simple/iso_low/`](viewer_data_paper_vs_simple/iso_low/) (`__gt__`, `c_port_q2`, `py_ref`, `amaze`, `rcd`, `menon_r0`, `menon_r1` × kodim01/08/13/19/20)

---

## 3. SIDD medium 実 raw (20 scenes × 5 phones、per-scene 3×3 CCM fit)

SIDD Medium のクリーン参照 (gt_raw) とノイズ入り (noisy_raw) 両方で、20 シーン (G4, GP, IP, N6, S6 各 4枚) をデモザイク処理し、gt_srgb と比較。
センサー色空間差を埋める per-scene 3×3 CCM + bias fit を適用してから CPSNR 計測。

| method | iso_clean CPSNR | iso_noise CPSNR |
|---|---|---|
| **ari** | **40.354** | 28.197 |
| amaze+dual | 39.976 | 28.118 |
| rcd+dual | 39.927 | **28.202** |
| amaze | 38.299 | 27.651 |
| rcd | 38.424 | 27.730 |
| menon_r1 | 37.675 | 27.422 |
| menon_r0 | 37.755 | 27.268 |

**観察**:
- iso_clean (低ノイズ実 raw): ari 最高、+0.38 dB vs 2位 amaze+dual
- iso_noise (高 ISO 実ノイズ): rcd+dual 最高、ari と 0.005 dB 差の実質同格
- menon は iso_clean でも最下位クラス

### Chroma zipper (SIDD エッジ)

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

## 4. 処理時間 (GCC 15.2 + OpenMP)

### Kodak 0.4 MP (kodim19、4 threads、内部タイミング)

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
| **ari (darktable pipeline、mire1.cr2 18.8 MP)** | **113s** |

### ari 最適化の推移 (SIDD 15.86 MP、16 threads)

| state | time |
|---|---|
| 初期 C port (MSVC /O2、no OpenMP) | 180s |
| + box 削減 + integral 再利用 (phase 1) | 94s |
| + 案A ペア box 共有 | 69s |
| + 案D Gaussian 5x5 分離 | 60s |
| + rolling sum (integral 廃止) | 53s (float2バッファ、double2バッファ版は 2.4× 遅い) |
| + row-major vertical pass | **53s (現状)** |

試した & 却下:
- 案B double col_acc 版 rolling sum: 2.4x 遅い
- 案C `box(A*B*C)` 融合: -12% 退行 (integral scan の vectorize が退化)

### iteration 数の感度 (SIDD 15.86 MP ari q=iter)

| iter | CPSNR (Kodak-24) | cz_mean | time |
|---|---|---|---|
| 3 | 37.687 | 0.0827 | 0.72s (Kodak) |
| 5 | 39.088 | 0.0686 | 1.08s |
| 7 | 39.647 | 0.0644 | 1.50s |
| 9 | 39.862 | 0.0628 | 1.89s |
| **11** | **39.939** | **0.0620** | 2.37s |

paper 標準 iter=11。iter=7 で 87%の品質到達するが、他手法との速度差を埋めるには足りない。

---

## 5. 使い分けガイド

| 用途 | 推奨 |
|---|---|
| インタラクティブプレビュー | rcd (デフォルト) / amaze |
| 高 ISO raw | rcd+dual / amaze+dual |
| 低 ISO 静止画、静止画書き出し、ピクセルピーピング | ari (1 枚 1-2 分許容なら) |
| 高速処理優先 | menon_r1 (ただし amaze との優位性は僅差) |

---

## 再現方法

### standalone ハーネスビルド (MSYS2 UCRT64 + GCC)

```bash
cd demosaic_test
bash build_gcc_omp.sh
```

### ベンチ実行

```bash
# Kodak-24 全メソッド、ISO 低/中/高
python bench_final_all.py --iso low
python bench_final_all.py --iso mid
python bench_final_all.py --iso high

# Chroma zipper 詳細
python bench_zipper_all_omp.py

# SIDD 実 raw
python bench_sidd_final.py

# ari iteration 数スイープ
python bench_ari_iter_sweep.py

# viewer PNG 生成 (エッジ画像 7 methods)
python _paper_vs_simple_save.py
python _paper_vs_simple_add_amaze_rcd.py
python _paper_vs_simple_add_menon.py
```

### darktable 本体ビルド (Windows MSYS2 UCRT64 の場合)

```bash
# 必須: UCRT64 環境で gcc.exe/g++.exe 直接指定、mingw64 lib 混入回避
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

## 同梱物

- [_paper_ari_ref.py](_paper_ari_ref.py) — Python で書いた ARI の paper-exact リファレンス (Monno の MATLAB コードと構造一致)
- [_paper_ri_ref.py](_paper_ri_ref.py) — RI 単体リファレンス
- [refs/matlab/](refs/matlab/) — 著者オリジナル MATLAB コード (ARI/MLRI/RI 各世代)
- [viewer_data_paper_vs_simple/iso_low/](viewer_data_paper_vs_simple/iso_low/) — エッジ 5 画像 × 7 methods 出力
- [viewer.html](viewer.html) — 上記 PNG を split 比較表示する HTML viewer
- bench_*.py — 再現用スクリプト
