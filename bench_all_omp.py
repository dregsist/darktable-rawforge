"""Benchmark all demosaic methods (OpenMP-enabled): ari-q2, rcd, amaze,
menon_r0, menon_r1. Report CPSNR avg + time avg on Kodak-24.
"""
import subprocess
import time
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from colour_demosaicing import mosaicing_CFA_Bayer

KD = Path(r'C:/Users/dregsist/datasets/kodak')
tmp = Path('results/tmp_bench'); tmp.mkdir(parents=True, exist_ok=True)


def cpsnr(gt, pred, border=12):
    h, w = gt.shape[:2]
    g = gt[border:h-border, border:w-border].astype(np.float64)
    p = pred[border:h-border, border:w-border].astype(np.float64)
    mse = np.mean((g-p)**2)
    return 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')


METHODS = [
    ('ari-q2',   './test_ari.exe',   ['2']),
    ('rcd',      './test_rcd.exe',   []),
    ('amaze',    './test_amaze.exe', []),
    ('menon_r0', './test_menon.exe', ['0']),
    ('menon_r1', './test_menon.exe', ['1']),
]

# Prep CFA inputs once per image
inputs = []
for i in range(1, 25):
    fn = f'kodim{i:02d}.png'
    img = iio.imread(str(KD / fn)).astype(np.float32) / 255.0
    img = img[:, :, :3]
    h, w = img.shape[:2]; h &= ~1; w &= ~1; img = img[:h, :w]
    cfa = mosaicing_CFA_Bayer(img, 'RGGB').astype(np.float32)
    inputs.append((fn, img, cfa, h, w))

print(f'{"method":<10} {"CPSNR":>8}  {"time/img":>10}  {"Mpix/s":>8}  {"total":>8}')
print('-' * 55)
for (name, exe, args) in METHODS:
    cps = []
    times = []
    for (fn, img, cfa, h, w) in inputs:
        (tmp / 'in.bin').write_bytes(np.ascontiguousarray(cfa).tobytes())
        t0 = time.time()
        subprocess.run([exe, str(w), str(h), '0x94949494', str(tmp / 'in.bin'), str(tmp / 'out.bin'), *args], capture_output=True)
        t = time.time() - t0
        times.append(t)
        out = np.frombuffer((tmp / 'out.bin').read_bytes(), dtype=np.float32).reshape(h, w, 4)[:, :, :3]
        cps.append(cpsnr(img, out))
    avg_t = np.mean(times)
    mpix = (h * w) / 1e6
    print(f'{name:<10} {np.mean(cps):>8.3f}  {avg_t:>8.3f}s   {mpix/avg_t:>7.3f}  {sum(times):>6.1f}s')
