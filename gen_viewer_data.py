"""Generate viewer_data/ for viewer.html from Kodak (and optionally other)
ground-truth RGB images.

Output layout (preferred: dataset/method/image):
    viewer_data/
    <iso_tag>/
        __gt__/          kodim01.png ...
        rcd/             kodim01.png ...
        amaze/           kodim01.png ...
        rcd+dual/        kodim01.png ...
        amaze+dual/      kodim01.png ...
        menon_r1/        kodim01.png ...
        lmmse_1/         kodim01.png ...
        ari_q1/          kodim01.png ...
        ari_q2/          kodim01.png ...
    scores.json          (one file, contains all iso x method rows)

ISO simulation is Gaussian noise added directly to the float Bayer CFA:
    iso_low  -> sigma=0.00  (clean reference)
    iso_mid  -> sigma=0.02  (~ISO 800-1600 sensor RMS)
    iso_high -> sigma=0.05  (~ISO 3200-6400 sensor RMS)

Dual-demosaic is the Python port of darktable's src/iop/demosaicing/dual.c
(same code as bench_dual.py, default dual_threshold=0.20).
"""
import argparse
import json
import subprocess
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from colour_demosaicing import mosaicing_CFA_Bayer
from scipy import ndimage

HERE = Path(__file__).parent
DEFAULT_DATASET_DIR = Path('C:/Users/dregsist/datasets/kodak')
DEFAULT_OUT_DIR = HERE / 'viewer_data'
FILTERS_HEX = '0x94949494'
BORDER = 12

ISO_SIGMA = {
    'iso_low':  0.00,
    'iso_mid':  0.02,
    'iso_high': 0.05,
}

# (method_label, exe, extra_args). method_label becomes the folder name.
METHODS = [
    ('rcd',      './test_rcd.exe',   []),
    ('amaze',    './test_amaze.exe', []),
    ('menon_r1', './test_menon.exe', ['1']),
    ('lmmse_1',  './test_lmmse.exe', ['1']),
    ('ppg',      './test_ppg.exe',   []),
    ('vng',      './test_vng.exe',   []),
    ('ari_q1',   './test_ari.exe',   ['1']),
    ('ari_q2',   './test_ari.exe',   ['2']),
]

# darktable has dual variants only for RCD and AMaZE (+ Markesteijn 3-pass for X-Trans).
DUAL_BASES = ['rcd', 'amaze']
DUAL_SLIDER = 0.20   # darktable default


# ---------- darktable dual demosaic (Python port of dual.c) -----------

def scharr_gradient_2d(p):
    """Match common/math.h scharr_gradient."""
    gx = (47.0 / 255.0) * (np.roll(p, ( 1,  1), axis=(0, 1)) - np.roll(p, ( 1, -1), axis=(0, 1))
                         + np.roll(p, (-1,  1), axis=(0, 1)) - np.roll(p, (-1, -1), axis=(0, 1))) \
       + (162.0 / 255.0) * (np.roll(p, (0, 1), axis=(0, 1)) - np.roll(p, (0, -1), axis=(0, 1)))
    gy = (47.0 / 255.0) * (np.roll(p, ( 1,  1), axis=(0, 1)) - np.roll(p, (-1,  1), axis=(0, 1))
                         + np.roll(p, ( 1, -1), axis=(0, 1)) - np.roll(p, (-1, -1), axis=(0, 1))) \
       + (162.0 / 255.0) * (np.roll(p, (1, 0), axis=(0, 1)) - np.roll(p, (-1, 0), axis=(0, 1)))
    return np.hypot(gx, gy)


def dt_scharr_mask(rgb):
    r = np.clip(rgb[:, :, 0], 0.0, None)
    g = np.clip(rgb[:, :, 1], 0.0, None)
    b = np.clip(rgb[:, :, 2], 0.0, None)
    tmp = np.sqrt((r + g + b) / 3.0)
    return np.clip(scharr_gradient_2d(tmp) / 16.0, 0.0, 1.0)


def dt_detail_blend(mask, threshold):
    ithreshold = 16.0 / max(threshold, 1e-7)
    arg = np.clip(16.0 - ithreshold * mask, -80.0, 80.0)
    return np.clip(1.0 / (1.0 + np.exp(arg)), 0.0, 1.0)


def dual_blend_dt(base, vng_linear, slider=DUAL_SLIDER):
    contrastf = 0.005 * (slider ** 1.1)
    sch = dt_scharr_mask(base)
    det = dt_detail_blend(sch, contrastf)
    mask = ndimage.gaussian_filter(det, sigma=2.0, mode='reflect')
    mask = np.clip(mask, 0.0, 1.0)[:, :, None]
    return mask * base + (1.0 - mask) * vng_linear


# ---------- helpers ---------------------------------------------------

def run_exe(exe, extra, w, h, cfa_bytes, tmp_dir, tag):
    in_p = tmp_dir / f'{tag}_in.bin';  in_p.write_bytes(cfa_bytes)
    out_p = tmp_dir / f'{tag}_out.bin'
    subprocess.run([exe, str(w), str(h), FILTERS_HEX, str(in_p), str(out_p)] + extra,
                   capture_output=True, check=False)
    return np.frombuffer(out_p.read_bytes(), dtype=np.float32).reshape(h, w, 4)[:, :, :3]


def make_cfa(img_rgb, sigma, rng):
    cfa = mosaicing_CFA_Bayer(img_rgb, 'RGGB').astype(np.float32)
    if sigma > 0.0:
        cfa = cfa + rng.standard_normal(cfa.shape).astype(np.float32) * sigma
        cfa = np.clip(cfa, 0.0, 1.0).astype(np.float32)
    return cfa


def save_png(arr_rgb_float, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.clip(arr_rgb_float, 0.0, 1.0)
    u8 = (u8 * 255.0 + 0.5).astype(np.uint8)
    iio.imwrite(str(path), u8)


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


def edge_cpsnr(gt, pred, mask, border=BORDER):
    h, w = gt.shape[:2]
    gt2 = gt[border:h-border, border:w-border]
    pr2 = pred[border:h-border, border:w-border]
    m = mask[border:h-border, border:w-border]
    sq = (gt2.astype(np.float64) - pr2.astype(np.float64)) ** 2
    sq_m = sq[m]
    if sq_m.size == 0 or np.mean(sq_m) <= 0: return float('nan')
    return 10 * np.log10(1.0 / np.mean(sq_m))


def zipper_score(gt, pred, mask, border=BORDER):
    h, w = gt.shape[:2]
    d = pred.astype(np.float64) - gt.astype(np.float64)
    lap = np.stack([ndimage.laplace(d[:, :, c]) for c in range(3)], axis=-1)
    zmag = np.mean(np.abs(lap), axis=-1)
    m = mask[border:h-border, border:w-border]
    v = zmag[border:h-border, border:w-border][m]
    return float(np.mean(v)) if v.size else float('nan')


# ---------- main ------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', default=str(DEFAULT_DATASET_DIR),
                    help='folder of ground-truth RGB PNGs')
    ap.add_argument('--out', default=str(DEFAULT_OUT_DIR))
    ap.add_argument('--isos', nargs='+', default=['iso_low', 'iso_mid', 'iso_high'],
                    choices=list(ISO_SIGMA.keys()))
    ap.add_argument('--limit', type=int, default=0,
                    help='process only first N images (0 = all)')
    ap.add_argument('--names', nargs='+', default=None,
                    help='explicit list of image filenames (e.g. kodim01.png kodim08.png)')
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out)
    tmp_dir = out_dir / '_tmp'; tmp_dir.mkdir(parents=True, exist_ok=True)

    # Discover images
    if args.names:
        imgs = [dataset_dir / n for n in args.names]
    else:
        imgs = sorted(dataset_dir.glob('*.png'))
    if args.limit > 0:
        imgs = imgs[:args.limit]

    print(f'Generating viewer data in {out_dir}')
    print(f'  dataset: {dataset_dir}  ({len(imgs)} images)')
    print(f'  isos   : {args.isos}')
    print(f'  methods: {[m[0] for m in METHODS]} + dual variants of {DUAL_BASES}')

    # scores[iso_tag][scene_id][method] = {cpsnr, ecpsnr, zipper}
    scores_out = {}

    rng = np.random.default_rng(0)

    for iso in args.isos:
        sigma = ISO_SIGMA[iso]
        print(f'\n[{iso}] sigma={sigma}')
        iso_dir = out_dir / iso
        iso_dir.mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            fname = img_path.name
            tag   = img_path.stem
            img = iio.imread(str(img_path)).astype(np.float32) / 255.0
            if img.ndim == 2: img = np.stack([img]*3, axis=-1)
            img = img[:, :, :3]
            h, w = img.shape[:2]
            cfa = make_cfa(img, sigma, rng)
            cfa_bytes = np.ascontiguousarray(cfa).tobytes()
            mask = edge_mask(img)

            # Ground truth: save per-iso (identical across iso, but keeps lookup simple)
            save_png(img, iso_dir / '__gt__' / fname)
            scores_out.setdefault(tag, {})

            # VNG linear (for dual blending)
            vng_lin = run_exe('./test_vng.exe', ['1'], w, h, cfa_bytes, tmp_dir,
                              f'{iso}_{tag}_vnglin')

            # Each method
            outputs = {}
            for name, exe, extra in METHODS:
                out_rgb = run_exe(exe, extra, w, h, cfa_bytes, tmp_dir, f'{iso}_{tag}_{name}')
                outputs[name] = out_rgb
                save_png(out_rgb, iso_dir / name / fname)

            # Dual variants (RCD+dual, AMaZE+dual)
            for base in DUAL_BASES:
                blended = dual_blend_dt(outputs[base], vng_lin, DUAL_SLIDER)
                outputs[f'{base}+dual'] = blended
                save_png(blended, iso_dir / f'{base}+dual' / fname)

            # Scores
            entry = scores_out[tag]
            entry_iso = entry.setdefault(iso, {})
            for name, pred in outputs.items():
                entry_iso[name] = {
                    'cpsnr':  round(cpsnr(img, pred), 3),
                    'ecpsnr': round(edge_cpsnr(img, pred, mask), 3),
                    'zipper': round(zipper_score(img, pred, mask), 6),
                }
            print(f'  {fname:<14} '
                  f'ari_q2={entry_iso.get("ari_q2",{}).get("cpsnr","?")} '
                  f'amaze+dual={entry_iso.get("amaze+dual",{}).get("cpsnr","?")}')

    # Per-ISO viewer-format-A JSONs (one file per ISO layer).
    for iso in args.isos:
        iso_scores = {tag: iso_data.get(iso, {}) for tag, iso_data in scores_out.items()
                      if iso in iso_data}
        with open(out_dir / f'scores_{iso}.json', 'w', encoding='utf-8') as f:
            json.dump({'patches': iso_scores}, f, indent=1)

    # A convenience "all" file in viewer format B (methods array)
    methods_flat = {}
    for tag, iso_data in scores_out.items():
        for iso, meth_data in iso_data.items():
            for method, metrics in meth_data.items():
                methods_flat.setdefault(f'{iso}::{method}', []).append({
                    'tag': tag, **metrics
                })
    with open(out_dir / 'scores_all.json', 'w', encoding='utf-8') as f:
        json.dump({'methods': methods_flat}, f, indent=1)

    print(f'\nWrote scores per ISO: {out_dir}/scores_<iso>.json')
    print(f'Wrote combined scores: {out_dir}/scores_all.json')
    print('\nOpen viewer.html, "Open Folder" on viewer_data/,'
          ' then "Load Scores JSON" on scores_<iso>.json matching the ISO layer.')

    # Clean up tmp (optional)
    for p in tmp_dir.glob('*.bin'):
        try: p.unlink()
        except OSError: pass


if __name__ == '__main__':
    main()
