"""Generate viewer data from the darktable-tests RAW images.

Uses rawpy to decode mire1.cr2 and hlrecovery.arw into a float Bayer
(after black-level subtraction + white-level normalisation + camera
white balance) and a reference sRGB image produced by rawpy's
AAHD + full camera pipeline. Our test harnesses demosaic the Bayer,
and each per-method output is fit to the reference sRGB via a
3x3 CCM + bias so all methods land in the same display space.

This is strictly visual (no ground truth for CPSNR scoring — see PR
thread #20800). It lets us reproduce the exact mire1.cr2 image the
reviewer used to flag Menon/ARI artefacts and check whether the
current implementations still show the same issues.

Output layout matches viewer.html expectations:

    viewer_data_dttests/
    iso_low/
        __gt__/     mire1.png, hlrecovery.png     (rawpy AAHD reference)
        rcd/        mire1.png, ...
        amaze/      ...
        ari_q2/     ...
    scores_iso_low.json   (rough CPSNR vs rawpy reference — NOT ground truth)
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import imageio.v3 as iio
import rawpy
from scipy import ndimage

HERE = Path(__file__).parent
RAW_DIR = HERE / 'dt_tests_images'
OUT_DIR = HERE / 'viewer_data_dttests'
TIER = 'iso_low'
FIT_BORDER = 64
METRIC_BORDER = 32

SOURCES = [
    ('mire1',      'mire1.cr2'),
    ('hlrecovery', 'hlrecovery.arw'),
]

# Bayer pattern hex (both RAWs are RGGB per rawpy metadata).
FILTERS_HEX = 0x94949494

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
DUAL_BASES = ['rcd', 'amaze']
DUAL_SLIDER = 0.20


# ---------- sRGB gamma --------------------------------------------------

def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92,
                    ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(y):
    y = np.clip(y, 0.0, 1.0)
    return np.where(y <= 0.0031308, y * 12.92,
                    1.055 * (y ** (1.0 / 2.4)) - 0.055).astype(np.float32)


# ---------- 3x3 CCM + bias fit ------------------------------------------

def fit_ccm_3x3(sensor_rgb, ref_lin, border=FIT_BORDER):
    h, w = sensor_rgb.shape[:2]
    p = sensor_rgb[border:h-border, border:w-border].reshape(-1, 3).astype(np.float64)
    r = ref_lin   [border:h-border, border:w-border].reshape(-1, 3).astype(np.float64)
    X = np.hstack([p, np.ones((p.shape[0], 1))])
    sol, *_ = np.linalg.lstsq(X, r, rcond=None)
    return sol[:3].astype(np.float32), sol[3].astype(np.float32)


def apply_ccm(sensor_rgb, M, bias):
    flat = sensor_rgb.reshape(-1, 3) @ M
    return (flat + bias).reshape(sensor_rgb.shape)


# ---------- dt dual demosaic (sensor-space port) ------------------------

def scharr_gradient_2d(p):
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


def dual_blend_dt(base_sensor, vng_sensor, slider=DUAL_SLIDER):
    contrastf = 0.005 * (slider ** 1.1)
    sch = dt_scharr_mask(base_sensor)
    det = dt_detail_blend(sch, contrastf)
    mask = ndimage.gaussian_filter(det, sigma=2.0, mode='reflect')
    mask = np.clip(mask, 0.0, 1.0)[:, :, None]
    return mask * base_sensor + (1.0 - mask) * vng_sensor


# ---------- helpers -----------------------------------------------------

def run_exe(exe, extra, w, h, filters_hex, cfa_bytes, tmp_dir, tag):
    in_p = tmp_dir / f'{tag}_in.bin';  in_p.write_bytes(cfa_bytes)
    out_p = tmp_dir / f'{tag}_out.bin'
    subprocess.run([exe, str(w), str(h), f'0x{filters_hex:08X}', str(in_p), str(out_p)] + extra,
                   capture_output=True, check=False)
    return np.frombuffer(out_p.read_bytes(), dtype=np.float32).reshape(h, w, 4)[:, :, :3]


def save_png(rgb_srgb_float, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = (np.clip(rgb_srgb_float, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    iio.imwrite(str(path), u8)


def cpsnr(gt, pred, border=METRIC_BORDER):
    h, w = gt.shape[:2]
    g = gt[border:h-border, border:w-border].astype(np.float64)
    p = pred[border:h-border, border:w-border].astype(np.float64)
    mse = np.mean((g - p) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')


def edge_mask(rgb, quantile=0.85):
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    sx = ndimage.sobel(lum, axis=1); sy = ndimage.sobel(lum, axis=0)
    m = np.sqrt(sx*sx + sy*sy)
    return m > np.quantile(m, quantile)


def edge_cpsnr(gt, pred, mask, border=METRIC_BORDER):
    h, w = gt.shape[:2]
    gt2 = gt[border:h-border, border:w-border]
    pr2 = pred[border:h-border, border:w-border]
    m = mask[border:h-border, border:w-border]
    sq = (gt2.astype(np.float64) - pr2.astype(np.float64)) ** 2
    v = sq[m]
    if v.size == 0 or np.mean(v) <= 0: return float('nan')
    return 10 * np.log10(1.0 / np.mean(v))


def zipper_score(gt, pred, mask, border=METRIC_BORDER):
    h, w = gt.shape[:2]
    d = pred.astype(np.float64) - gt.astype(np.float64)
    lap = np.stack([ndimage.laplace(d[:, :, c]) for c in range(3)], axis=-1)
    zmag = np.mean(np.abs(lap), axis=-1)
    m = mask[border:h-border, border:w-border]
    v = zmag[border:h-border, border:w-border][m]
    return float(np.mean(v)) if v.size else float('nan')


def decode_raw_for_bench(raw_path):
    """Return (cfa_float, reference_srgb, height, width).

    cfa_float: float32 HxW Bayer after black-level subtraction, white-level
               normalisation, and camera white-balance multiplication at each
               Bayer position. Values roughly in [0, 1].
    reference_srgb: float32 HxWx3 sRGB from rawpy AAHD + full pipeline
                    (for 3x3 CCM fit alignment, not a ground truth).
    """
    with rawpy.imread(str(raw_path)) as r:
        # Crop-aligned to Bayer even offsets so the pattern stays RGGB.
        raw_vis = r.raw_image_visible.astype(np.float32)
        h = raw_vis.shape[0] & ~1
        w = raw_vis.shape[1] & ~1
        raw_vis = raw_vis[:h, :w]

        bl = np.array(r.black_level_per_channel, dtype=np.float32).mean()
        wl = float(r.white_level)
        cfa = (raw_vis - bl) / max(wl - bl, 1.0)
        cfa = np.clip(cfa, 0.0, 1.0)

        wb = np.array(r.camera_whitebalance, dtype=np.float32) / 1024.0
        # rawpy pattern indices: [[0,1],[3,2]] -> R,G,G,B for RGGB
        # camera_whitebalance order is [R, G1, B, G2]
        gain = np.ones_like(cfa)
        gain[0::2, 0::2] *= wb[0]  # R at (even, even)
        gain[0::2, 1::2] *= wb[1]  # G1 at (even, odd)
        gain[1::2, 0::2] *= wb[3] if wb[3] > 0 else wb[1]  # G2 at (odd, even)
        gain[1::2, 1::2] *= wb[2]  # B at (odd, odd)
        cfa *= gain
        cfa = np.clip(cfa, 0.0, 1.0).astype(np.float32)

        # Reference sRGB via rawpy AAHD + camera wb + standard sRGB gamma.
        # user_flip=0 disables auto-orientation so the output stays aligned
        # with raw_image_visible (otherwise rawpy may rotate the image).
        ref_srgb = r.postprocess(
            output_color=rawpy.ColorSpace.sRGB,
            gamma=(2.222, 4.5),
            no_auto_bright=True,
            use_camera_wb=True,
            user_flip=0,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AAHD,
            output_bps=16).astype(np.float32) / 65535.0

        # Align to the even-cropped raw so dimensions match
        ref_srgb = ref_srgb[:h, :w]

    return cfa, ref_srgb, h, w


def main():
    tmp_dir = OUT_DIR / '_tmp'; tmp_dir.mkdir(parents=True, exist_ok=True)
    tier_dir = OUT_DIR / TIER

    print(f'darktable-tests viewer gen  sources={len(SOURCES)}')
    print(f'Methods: {[m[0] for m in METHODS]} + dual({DUAL_BASES})')
    print(f'Fit: 3x3 CCM + bias against rawpy AAHD reference, border={FIT_BORDER}')

    results = {}
    for tag, fname in SOURCES:
        raw_path = RAW_DIR / fname
        if not raw_path.exists():
            print(f'  MISSING {raw_path}'); continue
        print(f'\n[{tag}]  decoding {fname} ...', flush=True)
        cfa, ref_srgb, h, w = decode_raw_for_bench(raw_path)
        print(f'  {h}x{w}  ref_srgb range [{ref_srgb.min():.3f}, {ref_srgb.max():.3f}]')
        cfa_bytes = np.ascontiguousarray(cfa).tobytes()
        ref_lin = srgb_to_linear(ref_srgb)
        mask = edge_mask(ref_srgb)

        save_png(ref_srgb, tier_dir / '__gt__' / f'{tag}.png')

        # VNG linear for dual blending
        vng_sensor = run_exe('./test_vng.exe', ['1'], w, h, FILTERS_HEX, cfa_bytes, tmp_dir,
                             f'{tag}_vnglin')

        sensor_outs = {}
        for name, exe, extra in METHODS:
            print(f'  {name} ...', end=' ', flush=True)
            sensor_outs[name] = run_exe(exe, extra, w, h, FILTERS_HEX, cfa_bytes, tmp_dir,
                                        f'{tag}_{name}')
            print('done')
        for base in DUAL_BASES:
            sensor_outs[f'{base}+dual'] = dual_blend_dt(sensor_outs[base], vng_sensor, DUAL_SLIDER)

        entry = results.setdefault(tag, {})
        for name, out_sensor in sensor_outs.items():
            M, bias = fit_ccm_3x3(out_sensor, ref_lin)
            fit_lin  = apply_ccm(out_sensor, M, bias)
            fit_srgb = linear_to_srgb(fit_lin)
            save_png(fit_srgb, tier_dir / name / f'{tag}.png')
            entry[name] = {
                'cpsnr':  round(cpsnr(ref_srgb, fit_srgb), 3),
                'ecpsnr': round(edge_cpsnr(ref_srgb, fit_srgb, mask), 3),
                'zipper': round(zipper_score(ref_srgb, fit_srgb, mask), 6),
            }

    with open(OUT_DIR / f'scores_{TIER}.json', 'w', encoding='utf-8') as f:
        json.dump({'patches': results}, f, indent=1)

    # Summary
    method_names = [m[0] for m in METHODS] + [f'{b}+dual' for b in DUAL_BASES]
    print('\n' + '=' * 70)
    print('Score vs rawpy AAHD reference (NOT ground truth -- visual guide only)')
    print('=' * 70)
    for metric, direction in [('cpsnr', '↑'), ('ecpsnr', '↑'), ('zipper', '↓')]:
        print(f'\n{metric.upper()} {direction}')
        print(f'{"method":<14}', end='')
        for tag, _ in SOURCES: print(f'{tag:>14}', end='')
        print()
        print('-' * (14 + 14 * len(SOURCES)))
        for m in method_names:
            print(f'{m:<14}', end='')
            for tag, _ in SOURCES:
                v = results.get(tag, {}).get(m, {}).get(metric)
                if v is None:
                    print(f'{"-":>14}', end='')
                else:
                    fmt = '.5f' if metric == 'zipper' else '.3f'
                    print(f'{v:>14{fmt}}', end='')
            print()

    for p in tmp_dir.glob('*.bin'):
        try: p.unlink()
        except OSError: pass


if __name__ == '__main__':
    main()
