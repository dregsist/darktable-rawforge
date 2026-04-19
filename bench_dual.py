"""Compare solo vs dual-demosaic on clean (low ISO proxy) and noisy
(high ISO proxy) synthetic Bayer inputs.

Dual demosaic is a faithful Python port of darktable's
src/iop/demosaicing/dual.c, using default parameters:
  - dual_threshold = 0.20 (darktable default)
  - Scharr gradient on sqrt((R+G+B)/3) luminance
  - mask normalised by /16 then clipped to [0,1]
  - detail blend factor via sigmoid with ithreshold = 16 / contrastf
  - Gaussian blur (sigma=2) of the mask
  - final interpolatef(mask, base, vng_linear)

For high-ISO simulation we add Gaussian noise directly to the Bayer CFA
(float input is in [0,1]). Typical sensor models: sigma ~= 0.005 at low
ISO, ~= 0.02 at ISO 800-1600, ~= 0.05 at ISO 3200-6400.

VNG linear is generated through test_vng.exe with the only_linear=1 flag.
All other methods (RCD, AMaZE, Menon, ARI) use their own test harness.
"""
import subprocess
import argparse
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from colour_demosaicing import mosaicing_CFA_Bayer
from scipy import ndimage

HERE = Path(__file__).parent
KODAK_DIR = Path('C:/Users/dregsist/datasets/kodak')
BORDER = 12
FILTERS_HEX = '0x94949494'
DUAL_THRESHOLD = 0.20   # darktable default

SOLO = [
    ('rcd',      ['./test_rcd.exe'],   []),
    ('amaze',    ['./test_amaze.exe'], []),
    ('lmmse_1',  ['./test_lmmse.exe'], ['1']),
    ('menon_r1', ['./test_menon.exe'], ['1']),
    ('ari_q2',   ['./test_ari.exe'],   ['2']),
]
# darktable's actual dual-demosaic methods (from DT_IOP_DEMOSAIC_*_DUAL).
DUAL_BASES = [
    ('rcd',   ['./test_rcd.exe'],   []),
    ('amaze', ['./test_amaze.exe'], []),
]

EDGE_IMAGES = ['kodim01.png', 'kodim08.png', 'kodim13.png',
               'kodim19.png', 'kodim20.png']


# ---------- exact darktable detail mask / blend ------------------------

def scharr_gradient_2d(p):
    """Match common/math.h scharr_gradient, pixel-wise on 2D array."""
    gx = (47.0 / 255.0) * (np.roll(p, ( 1,  1), axis=(0, 1)) - np.roll(p, ( 1, -1), axis=(0, 1))
                         + np.roll(p, (-1,  1), axis=(0, 1)) - np.roll(p, (-1, -1), axis=(0, 1))) \
       + (162.0 / 255.0) * (np.roll(p, (0, 1), axis=(0, 1)) - np.roll(p, (0, -1), axis=(0, 1)))
    gy = (47.0 / 255.0) * (np.roll(p, ( 1,  1), axis=(0, 1)) - np.roll(p, (-1,  1), axis=(0, 1))
                         + np.roll(p, ( 1, -1), axis=(0, 1)) - np.roll(p, (-1, -1), axis=(0, 1))) \
       + (162.0 / 255.0) * (np.roll(p, (1, 0), axis=(0, 1)) - np.roll(p, (-1, 0), axis=(0, 1)))
    return np.hypot(gx, gy)


def dt_scharr_mask(rgb):
    """Match dt_masks_calc_scharr_mask (no white-balance adjustment)."""
    r = np.clip(rgb[:, :, 0], 0.0, None)
    g = np.clip(rgb[:, :, 1], 0.0, None)
    b = np.clip(rgb[:, :, 2], 0.0, None)
    tmp = np.sqrt((r + g + b) / 3.0)
    return np.clip(scharr_gradient_2d(tmp) / 16.0, 0.0, 1.0)


def dt_detail_blend(mask, threshold):
    """Match dt_masks_calc_detail_blend with detail=TRUE."""
    ithreshold = 16.0 / max(threshold, 1e-7)
    # sigmoid: 1 / (1 + exp(16 - ithreshold * val))
    arg = np.clip(16.0 - ithreshold * mask, -80.0, 80.0)
    return np.clip(1.0 / (1.0 + np.exp(arg)), 0.0, 1.0)


def dual_blend_dt(base, vng_linear, dual_slider=DUAL_THRESHOLD):
    contrastf = 0.005 * (dual_slider ** 1.1)
    sch = dt_scharr_mask(base)
    det = dt_detail_blend(sch, contrastf)
    mask = ndimage.gaussian_filter(det, sigma=2.0, mode='reflect')
    mask = np.clip(mask, 0.0, 1.0)[:, :, None]
    return mask * base + (1.0 - mask) * vng_linear, mask[:, :, 0]


# ---------- metrics ----------------------------------------------------

def cpsnr(gt, pred, border=BORDER):
    h, w = gt.shape[:2]
    g = gt[border:h-border, border:w-border].astype(np.float64)
    p = pred[border:h-border, border:w-border].astype(np.float64)
    mse = np.mean((g - p) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')


def edge_mask(rgb, quantile=0.85):
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    sx = ndimage.sobel(lum, axis=1); sy = ndimage.sobel(lum, axis=0)
    mag = np.sqrt(sx*sx + sy*sy)
    return mag > np.quantile(mag, quantile)


def edge_cpsnr_and_zipper(gt, pred, mask, border=BORDER):
    h, w = gt.shape[:2]
    gt2 = gt[border:h-border, border:w-border]
    pr2 = pred[border:h-border, border:w-border]
    m = mask[border:h-border, border:w-border]
    sq = (gt2.astype(np.float64) - pr2.astype(np.float64)) ** 2
    sq_m = sq[m]
    ec = 10*np.log10(1.0/np.mean(sq_m)) if sq_m.size and np.mean(sq_m) > 0 else float('nan')
    d = pred.astype(np.float64) - gt.astype(np.float64)
    lap = np.stack([ndimage.laplace(d[:, :, c]) for c in range(3)], axis=-1)
    zmag = np.mean(np.abs(lap), axis=-1)
    z = float(np.mean(zmag[border:h-border, border:w-border][m])) if m.any() else float('nan')
    return ec, z


# ---------- runners ----------------------------------------------------

def run_exe(cmd, extra, w, h, cfa_bytes, tmp, tag):
    in_p = tmp / f'{tag}_in.bin';  in_p.write_bytes(cfa_bytes)
    out_p = tmp / f'{tag}_out.bin'
    subprocess.run(cmd + [str(w), str(h), FILTERS_HEX, str(in_p), str(out_p)] + extra,
                   capture_output=True)
    return np.frombuffer(out_p.read_bytes(), dtype=np.float32).reshape(h, w, 4)[:, :, :3]


def make_cfa(img_rgb, noise_sigma=0.0, rng=None):
    cfa = mosaicing_CFA_Bayer(img_rgb, 'RGGB').astype(np.float32)
    if noise_sigma > 0.0:
        if rng is None:
            rng = np.random.default_rng(0)
        cfa = cfa + rng.standard_normal(cfa.shape).astype(np.float32) * noise_sigma
        cfa = np.clip(cfa, 0.0, 1.0).astype(np.float32)
    return cfa


# ---------- main --------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iso', default='low', choices=['low', 'mid', 'high'],
                    help='low=clean, mid=sigma 0.02 (~ISO 1600), high=sigma 0.05 (~ISO 6400)')
    ap.add_argument('--slider', type=float, default=DUAL_THRESHOLD)
    ap.add_argument('--out', default='results/dual')
    a = ap.parse_args()

    out_dir = HERE / a.out; tmp = out_dir / 'tmp'
    tmp.mkdir(parents=True, exist_ok=True)

    sigma = {'low': 0.0, 'mid': 0.02, 'high': 0.05}[a.iso]
    print(f'Dual-demosaic bench  iso={a.iso} (noise sigma={sigma})  '
          f'slider={a.slider} (contrastf={0.005*a.slider**1.1:.4g})\n')

    rng = np.random.default_rng(0)
    dual_names = {n for n, _, _ in DUAL_BASES}

    # Full Kodak-24 CPSNR
    cp = {}
    for i in range(1, 25):
        fname = f'kodim{i:02d}.png'
        fp = KODAK_DIR / fname
        if not fp.exists():
            continue
        img = iio.imread(str(fp)).astype(np.float32) / 255.0
        if img.ndim == 2: img = np.stack([img]*3, axis=-1)
        img = img[:, :, :3]
        h, w = img.shape[:2]
        cfa = make_cfa(img, sigma, rng)
        cfa_bytes = np.ascontiguousarray(cfa).tobytes()

        vng_lin = run_exe(['./test_vng.exe'], ['1'], w, h, cfa_bytes, tmp,
                          f'{fname[:-4]}_vnglin')

        for name, cmd, extra in SOLO:
            base = run_exe(cmd, extra, w, h, cfa_bytes, tmp, f'{fname[:-4]}_{name}')
            cp.setdefault(name, []).append(cpsnr(img, base))
            if name in dual_names:
                dual, _ = dual_blend_dt(base, vng_lin, a.slider)
                cp.setdefault(name+'+dual', []).append(cpsnr(img, dual))

    # Edge images: edge-CPSNR + zipper
    ec = {}; zz = {}
    for fname in EDGE_IMAGES:
        fp = KODAK_DIR / fname
        if not fp.exists(): continue
        img = iio.imread(str(fp)).astype(np.float32)/255.0
        img = (img[:, :, :3] if img.ndim==3 else np.stack([img]*3, axis=-1))
        h, w = img.shape[:2]
        mask = edge_mask(img)
        cfa_bytes = np.ascontiguousarray(make_cfa(img, sigma, rng)).tobytes()
        vng_lin = run_exe(['./test_vng.exe'], ['1'], w, h, cfa_bytes, tmp,
                          f'{fname[:-4]}_vnglin_e')
        for name, cmd, extra in SOLO:
            base = run_exe(cmd, extra, w, h, cfa_bytes, tmp, f'{fname[:-4]}_{name}_e')
            ec_b, z_b = edge_cpsnr_and_zipper(img, base, mask)
            ec.setdefault(name, []).append(ec_b); zz.setdefault(name, []).append(z_b)
            if name in dual_names:
                dual, _ = dual_blend_dt(base, vng_lin, a.slider)
                ec_d, z_d = edge_cpsnr_and_zipper(img, dual, mask)
                ec.setdefault(name+'+dual', []).append(ec_d)
                zz.setdefault(name+'+dual', []).append(z_d)

    # Report: show solo line, then +dual line only when present
    print(f'{"Algo":<16} {"CPSNR":>9}  {"Edge-CPSNR":>12} {"Zipper":>10}')
    print('-' * 52)
    for name, _, _ in SOLO:
        for suf in ('', '+dual'):
            nm = name + suf
            if nm not in cp: continue
            c_val = np.mean(cp[nm])
            e_val = np.mean(ec[nm]) if ec.get(nm) else float('nan')
            z_val = np.mean(zz[nm]) if zz.get(nm) else float('nan')
            print(f'  {nm:<14} {c_val:>9.3f}  {e_val:>11.3f} {z_val:>10.6f}')

    csv = out_dir / f'dual_{a.iso}.csv'
    with open(csv, 'w') as f:
        f.write('algo,cpsnr,edge_cpsnr,zipper\n')
        for nm in cp:
            f.write(f'{nm},{np.mean(cp[nm]):.4f},'
                    f'{np.mean(ec.get(nm, [float("nan")])):.4f},'
                    f'{np.mean(zz.get(nm, [float("nan")])):.6f}\n')
    print(f'\nCSV: {csv}')


if __name__ == '__main__':
    main()
