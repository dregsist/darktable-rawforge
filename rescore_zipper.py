"""Re-score existing Kodak / darktable-tests / SIDD outputs with
chroma-Laplacian zipper metric instead of the luminance-Laplacian one.

The previous zipper score measured |Laplacian(pred - gt)| on RGB. Zipper is
fundamentally a chroma artefact (R-G / B-G alternating at edges) so we
switch to:

    zipper_chroma =  mean( |Laplacian(dR)| + |Laplacian(dB)| )
    where dR = (pred_R - pred_G) - (gt_R - gt_G)
          dB = (pred_B - pred_G) - (gt_B - gt_G)

Evaluated on the same edge mask as before. Lower = less zipper.

We also report a simple "fringe" score for completeness:

    fringe = mean( |dR| + |dB| )  -- absolute chroma deviation at edges.

Runs on the per-method PNGs already produced by gen_viewer_data.py
(Kodak) and gen_viewer_dt_tests.py (mire1/hlrecovery). SIDD is skipped
here because the CCM fit absorbs enough chroma that the comparison
would be unreliable.
"""
import json
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from scipy import ndimage

HERE = Path(__file__).parent
BORDER = 32


def edge_mask(rgb, quantile=0.85):
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    sx = ndimage.sobel(lum, axis=1)
    sy = ndimage.sobel(lum, axis=0)
    m = np.sqrt(sx*sx + sy*sy)
    return m > np.quantile(m, quantile)


def chroma_zipper(gt, pred, mask, border=BORDER):
    g = gt.astype(np.float64)
    p = pred.astype(np.float64)
    dR = (p[:, :, 0] - p[:, :, 1]) - (g[:, :, 0] - g[:, :, 1])
    dB = (p[:, :, 2] - p[:, :, 1]) - (g[:, :, 2] - g[:, :, 1])
    lapR = np.abs(ndimage.laplace(dR))
    lapB = np.abs(ndimage.laplace(dB))
    zmag = lapR + lapB
    h, w = gt.shape[:2]
    m = mask[border:h-border, border:w-border]
    v = zmag[border:h-border, border:w-border][m]
    return float(np.mean(v)) if v.size else float('nan')


def chroma_fringe(gt, pred, mask, border=BORDER):
    g = gt.astype(np.float64)
    p = pred.astype(np.float64)
    dR = np.abs((p[:, :, 0] - p[:, :, 1]) - (g[:, :, 0] - g[:, :, 1]))
    dB = np.abs((p[:, :, 2] - p[:, :, 1]) - (g[:, :, 2] - g[:, :, 1]))
    mag = dR + dB
    h, w = gt.shape[:2]
    m = mask[border:h-border, border:w-border]
    v = mag[border:h-border, border:w-border][m]
    return float(np.mean(v)) if v.size else float('nan')


def luma_zipper(gt, pred, mask, border=BORDER):
    g = gt.astype(np.float64)
    p = pred.astype(np.float64)
    d = p - g
    lap = np.stack([ndimage.laplace(d[:, :, c]) for c in range(3)], axis=-1)
    zmag = np.mean(np.abs(lap), axis=-1)
    h, w = gt.shape[:2]
    m = mask[border:h-border, border:w-border]
    v = zmag[border:h-border, border:w-border][m]
    return float(np.mean(v)) if v.size else float('nan')


def score_dataset(dataset_dir, ref_slot='__gt__'):
    """Score every method folder under dataset_dir against the ground-truth."""
    dataset_dir = Path(dataset_dir)
    # dataset layout: <dataset_dir>/<tier>/<method>/<scene>.png
    # Walk tiers
    tiers = [d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith('_')]
    tiers = [d for d in tiers if (d / ref_slot).exists()]
    if not tiers:
        # maybe single-tier layout without iso_*
        if (dataset_dir / ref_slot).exists():
            tiers = [dataset_dir]
    out = {}
    for tier in tiers:
        gt_dir = tier / ref_slot
        if not gt_dir.exists():
            continue
        gt_files = sorted(p.name for p in gt_dir.glob('*.png'))
        methods = sorted(d.name for d in tier.iterdir() if d.is_dir() and d.name != ref_slot
                         and not d.name.startswith('_'))
        t_out = {}
        for m in methods:
            rows = []
            for fn in gt_files:
                gt_p = gt_dir / fn
                pr_p = tier / m / fn
                if not pr_p.exists():
                    continue
                gt = iio.imread(str(gt_p)).astype(np.float32) / 255.0
                pr = iio.imread(str(pr_p)).astype(np.float32) / 255.0
                if gt.shape != pr.shape:
                    continue
                mask = edge_mask(gt)
                rows.append({
                    'scene':       fn,
                    'luma_zip':    luma_zipper(gt, pr, mask),
                    'chroma_zip':  chroma_zipper(gt, pr, mask),
                    'chroma_fri':  chroma_fringe(gt, pr, mask),
                })
            if not rows:
                continue
            t_out[m] = {
                'luma_zip':   float(np.mean([r['luma_zip']   for r in rows])),
                'chroma_zip': float(np.mean([r['chroma_zip'] for r in rows])),
                'chroma_fri': float(np.mean([r['chroma_fri'] for r in rows])),
                'n':          len(rows),
            }
        out[tier.name if tier != dataset_dir else 'all'] = t_out
    return out


def print_table(title, out, methods_order):
    print(f'\n{title}')
    print('=' * 78)
    for tier, m2v in out.items():
        print(f'\n  {tier}')
        print(f'    {"method":<14} {"luma_zip↓":>12} {"chroma_zip↓":>14} {"chroma_fri↓":>14}  n')
        print('    ' + '-' * 62)
        for m in methods_order:
            if m not in m2v:
                continue
            r = m2v[m]
            print(f'    {m:<14} {r["luma_zip"]:>12.5f} {r["chroma_zip"]:>14.5f} '
                  f'{r["chroma_fri"]:>14.5f}  {r["n"]}')


if __name__ == '__main__':
    method_order = ['rcd', 'rcd+dual', 'amaze', 'amaze+dual', 'menon_r1',
                    'lmmse_1', 'ari_q1', 'ari_q2', 'ppg', 'vng']

    # Kodak
    kodak_out = score_dataset(HERE / 'viewer_data')
    print_table('Kodak-24 chroma zipper re-score', kodak_out, method_order)

    # darktable-tests (mire1, hlrecovery)
    dttests_out = score_dataset(HERE / 'viewer_data_dttests')
    print_table('darktable-tests RAW (mire1 + hlrecovery, 3x3 CCM fit)',
                dttests_out, method_order)

    # SIDD — optional, may be noisy due to CCM fit absorbing chroma
    sidd_out = score_dataset(HERE / 'viewer_data_sidd')
    print_table('SIDD medium (3x3 CCM fit, noise-contaminated)', sidd_out, method_order)

    # Write JSON bundle
    all_out = {'kodak': kodak_out, 'dttests': dttests_out, 'sidd': sidd_out}
    json_p = HERE / 'results' / 'zipper_rescore.json'
    json_p.parent.mkdir(parents=True, exist_ok=True)
    with open(json_p, 'w', encoding='utf-8') as f:
        json.dump(all_out, f, indent=1)
    print(f'\nJSON: {json_p}')
