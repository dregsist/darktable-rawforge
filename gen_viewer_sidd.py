"""Generate RAW-pipeline viewer data for SIDD medium scenes.

Each scene's RAW Bayer (gt_raw or noisy_raw) is demosaiced with every
method, then projected to sRGB via a per-scene, per-method 3x3 linear
colour correction matrix + 3-bias fit to the linearised gt_srgb. This
absorbs camera WB and CCM differences so we can see the real-sensor
demosaic behaviour without being masked by pipeline mismatch.

Output layout is identical to gen_viewer_data.py so the same viewer.html
works:

    viewer_data_sidd/
    <tier>/
        __gt__/         <scene>.png (gt_srgb)
        rcd/            <scene>.png (demosaic -> CCM -> sRGB)
        amaze/
        ...
    scores_<tier>.json  (viewer Load Scores)

Tiers:
    iso_clean  -> input = gt_raw     (averaged / low-noise reference)
    iso_noise  -> input = noisy_raw  (real sensor noise at the scene's ISO)
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from scipy import ndimage

HERE = Path(__file__).parent
SIDD_DIR = Path(r'C:/Users/dregsist/datasets/sidd/medium_bench')
DEFAULT_OUT = HERE / 'viewer_data_sidd'
FIT_BORDER  = 64
METRIC_BORDER = 32

PATTERN_HEX = {
    'RGGB': 0x94949494,
    'GRBG': 0x61616161,
    'GBRG': 0x49494949,
    'BGGR': 0x16161616,
}

SCENES = [
    ('0045', 'G4', 'BGGR', '010'),
    ('0059', 'G4', 'BGGR', '010'),
    ('0077', 'G4', 'BGGR', '010'),
    ('0099', 'G4', 'BGGR', '010'),
    ('0020', 'GP', 'BGGR', '010'),
    ('0036', 'GP', 'BGGR', '010'),
    ('0064', 'GP', 'BGGR', '010'),
    ('0083', 'GP', 'BGGR', '010'),
    ('0042', 'IP', 'RGGB', '010'),
    ('0069', 'IP', 'RGGB', '010'),
    ('0115', 'IP', 'RGGB', '010'),
    ('0165', 'IP', 'RGGB', '010'),
    ('0023', 'N6', 'BGGR', '010'),
    ('0048', 'N6', 'BGGR', '010'),
    ('0075', 'N6', 'BGGR', '010'),
    ('0120', 'N6', 'BGGR', '010'),
    ('0001', 'S6', 'GRBG', '010'),
    ('0010', 'S6', 'GRBG', '010'),
    ('0052', 'S6', 'GRBG', '010'),
    ('0154', 'S6', 'GRBG', '010'),
]

METHODS = [
    ('ari_q1',   './test_ari.exe',   ['1']),
    ('ari_q2',   './test_ari.exe',   ['2']),
    ('menon_r0', './test_menon.exe', ['0']),
]
DUAL_BASES = []
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
    """Return (M, bias) where M is (3,3) and bias is (3,).
    Maps sensor_rgb linear -> ref_lin linear, absorbing WB and CCM."""
    h, w = sensor_rgb.shape[:2]
    p = sensor_rgb[border:h-border, border:w-border].reshape(-1, 3).astype(np.float64)
    r = ref_lin   [border:h-border, border:w-border].reshape(-1, 3).astype(np.float64)
    X = np.hstack([p, np.ones((p.shape[0], 1))])
    sol, _, _, _ = np.linalg.lstsq(X, r, rcond=None)   # (4, 3)
    return sol[:3].astype(np.float32), sol[3].astype(np.float32)


def apply_ccm(sensor_rgb, M, bias):
    flat = sensor_rgb.reshape(-1, 3) @ M
    return (flat + bias).reshape(sensor_rgb.shape)


# ---------- dt dual demosaic (Python port) ------------------------------

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
    """Blend in sensor-RGB space before CCM so the mask and colours stay aligned."""
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
    u8 = np.clip(rgb_srgb_float, 0.0, 1.0)
    u8 = (u8 * 255.0 + 0.5).astype(np.uint8)
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
    return np.sqrt(sx*sx + sy*sy) > np.quantile(np.sqrt(sx*sx + sy*sy), quantile)


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


# ---------- main --------------------------------------------------------

def process_scene(scene_id, phone, pattern, frame, tier, tmp_dir, out_dir, results):
    filters_hex = PATTERN_HEX[pattern]
    tag = f'{scene_id}_{phone}_{pattern}_{frame}'
    gt_srgb_path = SIDD_DIR / f'{tag}_gt_srgb.npy'
    raw_path     = SIDD_DIR / f'{tag}_{"gt_raw" if tier == "iso_clean" else "noisy_raw"}.npy'
    if not gt_srgb_path.exists() or not raw_path.exists():
        print(f'  MISSING: {tag} ({tier})'); return

    gt_srgb = np.load(str(gt_srgb_path)).astype(np.float32)
    gt_srgb = np.clip(gt_srgb, 0.0, 1.0)
    raw     = np.load(str(raw_path)).astype(np.float32)
    h, w = raw.shape
    cfa_bytes = np.ascontiguousarray(raw).tobytes()

    ref_lin = srgb_to_linear(gt_srgb)
    mask = edge_mask(gt_srgb)

    tier_dir = out_dir / tier
    save_png(gt_srgb, tier_dir / '__gt__' / f'{tag}.png')

    vng_sensor = run_exe('./test_vng.exe', ['1'], w, h, filters_hex, cfa_bytes, tmp_dir,
                         f'{tier}_{tag}_vnglin')

    sensor_outs = {}
    for name, exe, extra in METHODS:
        sensor_outs[name] = run_exe(exe, extra, w, h, filters_hex, cfa_bytes, tmp_dir,
                                    f'{tier}_{tag}_{name}')
    for base in DUAL_BASES:
        sensor_outs[f'{base}+dual'] = dual_blend_dt(sensor_outs[base], vng_sensor, DUAL_SLIDER)

    entry = results.setdefault(tag, {}).setdefault(tier, {})
    for name, out_sensor in sensor_outs.items():
        M, bias = fit_ccm_3x3(out_sensor, ref_lin)
        fit_lin  = apply_ccm(out_sensor, M, bias)
        fit_srgb = linear_to_srgb(fit_lin)
        save_png(fit_srgb, tier_dir / name / f'{tag}.png')
        entry[name] = {
            'cpsnr':  round(cpsnr(gt_srgb, fit_srgb), 3),
            'ecpsnr': round(edge_cpsnr(gt_srgb, fit_srgb, mask), 3),
            'zipper': round(zipper_score(gt_srgb, fit_srgb, mask), 6),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    ap.add_argument('--tiers', nargs='+', default=['iso_clean', 'iso_noise'],
                    choices=['iso_clean', 'iso_noise'])
    ap.add_argument('--scenes', type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    tmp_dir = out_dir / '_tmp'; tmp_dir.mkdir(parents=True, exist_ok=True)
    scenes = SCENES[:args.scenes] if args.scenes > 0 else SCENES

    print(f'SIDD RAW-pipeline viewer gen  scenes={len(scenes)}  tiers={args.tiers}')
    print(f'Methods: {[m[0] for m in METHODS]} + dual({DUAL_BASES})')
    print(f'Fit: 3x3 CCM + bias per (scene, method), border={FIT_BORDER}')

    results = {}
    for i, (scene_id, phone, pattern, frame) in enumerate(scenes):
        tag = f'{scene_id}_{phone}_{pattern}_{frame}'
        print(f'\n[{i+1}/{len(scenes)}] {tag}  pattern={pattern}')
        for tier in args.tiers:
            print(f'  {tier} ...', end=' ', flush=True)
            process_scene(scene_id, phone, pattern, frame, tier, tmp_dir, out_dir, results)
            r = results.get(tag, {}).get(tier, {})
            if r:
                a = r.get('ari_q2', {}).get('cpsnr', '?')
                ad = r.get('amaze+dual', {}).get('cpsnr', '?')
                v = r.get('vng', {}).get('cpsnr', '?')
                print(f'ari_q2={a}  amaze+dual={ad}  vng={v}')

    for tier in args.tiers:
        tag_map = {tag: rec[tier] for tag, rec in results.items() if tier in rec}
        # Merge with existing JSON so previously-computed method scores are preserved.
        existing_path = out_dir / f'scores_{tier}.json'
        existing_patches = {}
        if existing_path.exists():
            try:
                existing_patches = json.loads(existing_path.read_text(encoding='utf-8')).get('patches', {})
            except Exception:
                existing_patches = {}
        for tag, new_methods in tag_map.items():
            merged = dict(existing_patches.get(tag, {}))
            merged.update(new_methods)
            existing_patches[tag] = merged
        with open(existing_path, 'w', encoding='utf-8') as f:
            json.dump({'patches': existing_patches}, f, indent=1)

    method_names = [m[0] for m in METHODS] + [f'{b}+dual' for b in DUAL_BASES]
    print('\n' + '=' * 80)
    for metric, direction in [('cpsnr', '↑'), ('ecpsnr', '↑'), ('zipper', '↓')]:
        print(f'\n{metric.upper()} {direction}   (pipeline mode with 3x3 CCM fit)')
        print(f'{"method":<14}', end='')
        for tier in args.tiers: print(f'{tier:>12}', end='')
        print()
        print('-' * (14 + 12 * len(args.tiers)))
        for m in method_names:
            print(f'{m:<14}', end='')
            for tier in args.tiers:
                vals = [results[tag][tier][m][metric]
                        for tag in results if tier in results[tag] and m in results[tag][tier]]
                if vals:
                    fmt = '.5f' if metric == 'zipper' else '.3f'
                    print(f'{np.mean(vals):>12{fmt}}', end='')
                else:
                    print(f'{"-":>12}', end='')
            print()
    print('=' * 80)

    for p in tmp_dir.glob('*.bin'):
        try: p.unlink()
        except OSError: pass


if __name__ == '__main__':
    main()
