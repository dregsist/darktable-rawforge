"""Chroma zipper metric benchmark: all demosaic methods on Kodak edge subset.

Metric: chroma-zipper = mean(|Laplacian(dR)|+|Laplacian(dB)|) on edge mask,
        where dR = (R-G)_pred - (R-G)_gt, similarly dB.

Lower = less zipper. Measured on edge-heavy pixels only (top 15% gradient).
"""
import subprocess
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from colour_demosaicing import mosaicing_CFA_Bayer
from scipy import ndimage

KD = Path(r'C:/Users/dregsist/datasets/kodak')
tmp = Path('results/tmp_zipper'); tmp.mkdir(parents=True, exist_ok=True)
BORDER = 32
EDGE_IMAGES = ['kodim01.png', 'kodim08.png', 'kodim13.png', 'kodim19.png', 'kodim20.png']


def edge_mask(rgb, q=0.85):
    lum = 0.299*rgb[:, :, 0] + 0.587*rgb[:, :, 1] + 0.114*rgb[:, :, 2]
    sx = ndimage.sobel(lum, axis=1); sy = ndimage.sobel(lum, axis=0)
    m = np.sqrt(sx*sx + sy*sy)
    return m > np.quantile(m, q)


def chroma_zip(gt, pred, mask, border=BORDER):
    g = gt.astype(np.float64); p = pred.astype(np.float64)
    dR = (p[:, :, 0]-p[:, :, 1]) - (g[:, :, 0]-g[:, :, 1])
    dB = (p[:, :, 2]-p[:, :, 1]) - (g[:, :, 2]-g[:, :, 1])
    zmag = np.abs(ndimage.laplace(dR)) + np.abs(ndimage.laplace(dB))
    h, w = gt.shape[:2]
    m = mask[border:h-border, border:w-border]
    v = zmag[border:h-border, border:w-border][m]
    cz_mean = float(np.mean(v)) if v.size else float('nan')
    cz_peak = float(ndimage.uniform_filter(zmag, 64).max())
    return cz_mean, cz_peak


def cpsnr(gt, pred, border=12):
    h, w = gt.shape[:2]
    g = gt[border:h-border, border:w-border].astype(np.float64)
    p = pred[border:h-border, border:w-border].astype(np.float64)
    mse = np.mean((g - p) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')


def run_exe(exe, img, extra_args=()):
    h, w = img.shape[:2]
    cfa = mosaicing_CFA_Bayer(img, 'RGGB').astype(np.float32)
    (tmp/'in.bin').write_bytes(np.ascontiguousarray(cfa).tobytes())
    subprocess.run([exe, str(w), str(h), '0x94949494',
                    str(tmp/'in.bin'), str(tmp/'out.bin'), *extra_args],
                   capture_output=True)
    return np.frombuffer((tmp/'out.bin').read_bytes(),
                         dtype=np.float32).reshape(h, w, 4)[:, :, :3]


METHODS = [
    ('ari-q2',   './test_ari.exe',   ['2']),
    ('amaze',    './test_amaze.exe', []),
    ('rcd',      './test_rcd.exe',   []),
    ('menon_r0', './test_menon.exe', ['0']),
    ('menon_r1', './test_menon.exe', ['1']),
]

# Accumulator
results = {name: {'cz_m': [], 'cz_pk': [], 'cp': []} for name, _, _ in METHODS}

for fn in EDGE_IMAGES:
    img = iio.imread(str(KD/fn)).astype(np.float32) / 255.0
    img = img[:, :, :3]
    h, w = img.shape[:2]; h &= ~1; w &= ~1; img = img[:h, :w]
    mask = edge_mask(img)

    print(f'\n{fn}:')
    for (name, exe, args) in METHODS:
        pred = run_exe(exe, img, args)
        cz_m, cz_pk = chroma_zip(img, pred, mask)
        cp = cpsnr(img, pred)
        results[name]['cz_m'].append(cz_m)
        results[name]['cz_pk'].append(cz_pk)
        results[name]['cp'].append(cp)
        print(f'  {name:<10} cz_m={cz_m:.4f}  cz_pk={cz_pk:.3f}  cp={cp:5.2f}')

print('\n' + '=' * 55)
print(f'{"method":<10} {"cz_mean":>10} {"cz_peak":>10} {"cpsnr":>8}')
print('-' * 55)
for (name, _, _) in METHODS:
    r = results[name]
    print(f'{name:<10} {np.mean(r["cz_m"]):>10.4f} '
          f'{np.mean(r["cz_pk"]):>10.4f} {np.mean(r["cp"]):>8.3f}')
print('\nLower cz = less chroma zipper (better edge quality).')
