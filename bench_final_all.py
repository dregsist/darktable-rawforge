"""Final comprehensive benchmark: ari, menon_r0, menon_r1, amaze, amaze+dual,
rcd, rcd+dual. Reports CPSNR (Kodak-24), chroma zipper (edge 5), and algo time.

Usage:
  python bench_final_all.py [--iso low|mid|high]
    low  = clean (sigma=0)
    mid  = ~ISO 1600 (sigma=0.02)
    high = ~ISO 6400 (sigma=0.05)
"""
import argparse
import subprocess
import re
import time
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from colour_demosaicing import mosaicing_CFA_Bayer
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument('--iso', default='low', choices=['low', 'mid', 'high'])
args = ap.parse_args()
NOISE_SIGMA = {'low': 0.0, 'mid': 0.02, 'high': 0.05}[args.iso]

KD = Path(r'C:/Users/dregsist/datasets/kodak')
tmp = Path(f'results/tmp_final_all_{args.iso}'); tmp.mkdir(parents=True, exist_ok=True)
BORDER = 32
DUAL_THRESHOLD = 0.20
RNG = np.random.default_rng(0)

EDGE_IMAGES = ['kodim01.png', 'kodim08.png', 'kodim13.png',
               'kodim19.png', 'kodim20.png']


# ---------- darktable dual blend (port of dual.c) ----------

def scharr_gradient_2d(p):
    gx = (47.0/255.0) * (np.roll(p, (1,1), (0,1)) - np.roll(p, (1,-1), (0,1))
                       + np.roll(p, (-1,1), (0,1)) - np.roll(p, (-1,-1), (0,1))) \
       + (162.0/255.0) * (np.roll(p, (0,1), (0,1)) - np.roll(p, (0,-1), (0,1)))
    gy = (47.0/255.0) * (np.roll(p, (1,1), (0,1)) - np.roll(p, (-1,1), (0,1))
                       + np.roll(p, (1,-1), (0,1)) - np.roll(p, (-1,-1), (0,1))) \
       + (162.0/255.0) * (np.roll(p, (1,0), (0,1)) - np.roll(p, (-1,0), (0,1)))
    return np.hypot(gx, gy)


def dual_blend(base, vng_lin, slider=DUAL_THRESHOLD):
    r = np.clip(base[:,:,0], 0, None); g = np.clip(base[:,:,1], 0, None); b = np.clip(base[:,:,2], 0, None)
    tmp = np.sqrt((r+g+b)/3.0)
    sch = np.clip(scharr_gradient_2d(tmp)/16.0, 0.0, 1.0)
    contrastf = 0.005 * (slider**1.1)
    ithr = 16.0 / max(contrastf, 1e-7)
    arg = np.clip(16.0 - ithr*sch, -80.0, 80.0)
    det = np.clip(1.0/(1.0+np.exp(arg)), 0.0, 1.0)
    mask = np.clip(ndimage.gaussian_filter(det, 2.0, mode='reflect'), 0, 1)[:,:,None]
    return mask*base + (1.0-mask)*vng_lin


# ---------- metrics ----------

def cpsnr(gt, pred, border=12):
    h,w = gt.shape[:2]
    g = gt[border:h-border, border:w-border].astype(np.float64)
    p = pred[border:h-border, border:w-border].astype(np.float64)
    mse = np.mean((g-p)**2)
    return 10*np.log10(1.0/mse) if mse > 0 else float('inf')


def edge_mask(rgb, q=0.85):
    lum = 0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]
    sx = ndimage.sobel(lum, axis=1); sy = ndimage.sobel(lum, axis=0)
    m = np.sqrt(sx*sx + sy*sy)
    return m > np.quantile(m, q)


def chroma_zipper(gt, pred, mask):
    g = gt.astype(np.float64); p = pred.astype(np.float64)
    dR = (p[:,:,0]-p[:,:,1]) - (g[:,:,0]-g[:,:,1])
    dB = (p[:,:,2]-p[:,:,1]) - (g[:,:,2]-g[:,:,1])
    zmag = np.abs(ndimage.laplace(dR)) + np.abs(ndimage.laplace(dB))
    h,w = gt.shape[:2]
    m = mask[BORDER:h-BORDER, BORDER:w-BORDER]
    v = zmag[BORDER:h-BORDER, BORDER:w-BORDER][m]
    cz_m = float(np.mean(v)) if v.size else float('nan')
    cz_pk = float(ndimage.uniform_filter(zmag, 64).max())
    return cz_m, cz_pk


# ---------- runner ----------

def make_noisy_cfa(img):
    cfa = mosaicing_CFA_Bayer(img, 'RGGB').astype(np.float32)
    if NOISE_SIGMA > 0:
        cfa = cfa + RNG.standard_normal(cfa.shape).astype(np.float32) * NOISE_SIGMA
        cfa = np.clip(cfa, 0.0, 1.0)
    return cfa


def run_exe(exe, img, extra=[], cfa_cache=None):
    h,w = img.shape[:2]
    cfa = cfa_cache if cfa_cache is not None else make_noisy_cfa(img)
    (tmp/'in.bin').write_bytes(np.ascontiguousarray(cfa).tobytes())
    t0 = time.time()
    subprocess.run([exe, str(w), str(h), '0x94949494',
                    str(tmp/'in.bin'), str(tmp/'out.bin'), *extra],
                   capture_output=True)
    dt = time.time()-t0
    out = np.frombuffer((tmp/'out.bin').read_bytes(),
                        dtype=np.float32).reshape(h,w,4)[:,:,:3]
    return out, dt


METHODS_SOLO = [
    ('ari',      './test_ari.exe',   ['11']),
    ('menon_r0', './test_menon.exe', ['0']),
    ('menon_r1', './test_menon.exe', ['1']),
    ('amaze',    './test_amaze.exe', []),
    ('rcd',      './test_rcd.exe',   []),
]
DUAL_BASES = {'amaze', 'rcd'}

# Preload Kodak-24
kodak = []
for i in range(1, 25):
    fn = f'kodim{i:02d}.png'
    img = iio.imread(str(KD/fn)).astype(np.float32)/255.0
    img = img[:,:,:3]
    h,w = img.shape[:2]; h &= ~1; w &= ~1; img = img[:h,:w]
    kodak.append((fn, img))

# CPSNR + time on all 24 (same noisy CFA per image across methods)
print(f'Running Kodak-24 iso={args.iso} sigma={NOISE_SIGMA}...')
stats = {}  # name -> dict
for fn, img in kodak:
    cfa = make_noisy_cfa(img)
    vng_out, vng_dt = None, 0
    for name, exe, extra in METHODS_SOLO:
        out, dt = run_exe(exe, img, extra, cfa_cache=cfa)
        d = stats.setdefault(name, {'cp': [], 't': []})
        d['cp'].append(cpsnr(img, out))
        d['t'].append(dt)
        if name in DUAL_BASES:
            if vng_out is None:
                vng_out, vng_dt = run_exe('./test_vng.exe', img, ['1'], cfa_cache=cfa)
            dual = dual_blend(out, vng_out)
            dn = name + '+dual'
            dd = stats.setdefault(dn, {'cp': [], 't': []})
            dd['cp'].append(cpsnr(img, dual))
            dd['t'].append(dt + vng_dt + 0.02)

# Zipper on edge 5 (also with same noisy CFA)
print('Running edge zipper...')
for fn in EDGE_IMAGES:
    img = next(im for (n, im) in kodak if n == fn)
    mask = edge_mask(img)
    cfa = make_noisy_cfa(img)
    vng_out = None
    for name, exe, extra in METHODS_SOLO:
        out, _ = run_exe(exe, img, extra, cfa_cache=cfa)
        d = stats.setdefault(name, {})
        cz_m, cz_pk = chroma_zipper(img, out, mask)
        d.setdefault('cz_m', []).append(cz_m)
        d.setdefault('cz_pk', []).append(cz_pk)
        if name in DUAL_BASES:
            if vng_out is None:
                vng_out, _ = run_exe('./test_vng.exe', img, ['1'], cfa_cache=cfa)
            dual = dual_blend(out, vng_out)
            dn = name + '+dual'
            dd = stats.setdefault(dn, {})
            cz_m_d, cz_pk_d = chroma_zipper(img, dual, mask)
            dd.setdefault('cz_m', []).append(cz_m_d)
            dd.setdefault('cz_pk', []).append(cz_pk_d)

ORDER = ['ari', 'amaze', 'amaze+dual', 'rcd', 'rcd+dual', 'menon_r1', 'menon_r0']

print()
print('='*74)
print(f'ISO={args.iso} (noise sigma={NOISE_SIGMA})')
print(f'{"method":<12} {"CPSNR":>7} {"cz_mean":>8} {"cz_peak":>8} {"time/img":>10} {"Mpix/s":>8}')
print('-'*74)
h, w = kodak[0][1].shape[:2]  # for Mpix calc
mpix_per_img = (h*w)/1e6  # approximate (all images similar size)
for name in ORDER:
    s = stats[name]
    cp = np.mean(s['cp'])
    czm = np.mean(s['cz_m'])
    czpk = np.mean(s['cz_pk'])
    t = np.mean(s['t'])
    mps = mpix_per_img / t
    print(f'{name:<12} {cp:>7.3f} {czm:>8.4f} {czpk:>8.4f} {t:>8.3f}s   {mps:>7.3f}')
