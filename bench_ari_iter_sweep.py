"""Sweep ARI iteration count: measure CPSNR (Kodak-24), chroma zipper (edge 5),
and internal algorithm time (kodim19) for itnum = 3, 5, 7, 9, 11.
"""
import subprocess
import re
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from colour_demosaicing import mosaicing_CFA_Bayer
from scipy import ndimage

KD = Path(r'C:/Users/dregsist/datasets/kodak')
tmp = Path('results/tmp_itersweep'); tmp.mkdir(parents=True, exist_ok=True)
BORDER = 32
EDGE_IMAGES = ['kodim01.png', 'kodim08.png', 'kodim13.png', 'kodim19.png', 'kodim20.png']
ITERS = [3, 5, 7, 9, 11]


def edge_mask(rgb, q=0.85):
    lum = 0.299*rgb[:, :, 0] + 0.587*rgb[:, :, 1] + 0.114*rgb[:, :, 2]
    sx = ndimage.sobel(lum, axis=1); sy = ndimage.sobel(lum, axis=0)
    m = np.sqrt(sx*sx + sy*sy)
    return m > np.quantile(m, q)


def chroma_zip(gt, pred, mask):
    g = gt.astype(np.float64); p = pred.astype(np.float64)
    dR = (p[:, :, 0]-p[:, :, 1]) - (g[:, :, 0]-g[:, :, 1])
    dB = (p[:, :, 2]-p[:, :, 1]) - (g[:, :, 2]-g[:, :, 1])
    zmag = np.abs(ndimage.laplace(dR)) + np.abs(ndimage.laplace(dB))
    h, w = gt.shape[:2]
    m = mask[BORDER:h-BORDER, BORDER:w-BORDER]
    v = zmag[BORDER:h-BORDER, BORDER:w-BORDER][m]
    cz_mean = float(np.mean(v)) if v.size else float('nan')
    cz_peak = float(ndimage.uniform_filter(zmag, 64).max())
    return cz_mean, cz_peak


def cpsnr(gt, pred, border=12):
    h, w = gt.shape[:2]
    g = gt[border:h-border, border:w-border].astype(np.float64)
    p = pred[border:h-border, border:w-border].astype(np.float64)
    mse = np.mean((g - p) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')


def run(img, iter_n):
    h, w = img.shape[:2]
    cfa = mosaicing_CFA_Bayer(img, 'RGGB').astype(np.float32)
    (tmp/'in.bin').write_bytes(np.ascontiguousarray(cfa).tobytes())
    r = subprocess.run(['./test_ari.exe', str(w), str(h), '0x94949494',
                        str(tmp/'in.bin'), str(tmp/'out.bin'), str(iter_n)],
                       capture_output=True, text=True)
    m = re.search(r'ari_demosaic: ([\d.]+) s', r.stderr)
    algo_t = float(m.group(1)) if m else float('nan')
    out = np.frombuffer((tmp/'out.bin').read_bytes(),
                        dtype=np.float32).reshape(h, w, 4)[:, :, :3]
    return out, algo_t


# Pre-load all Kodak-24 images
kodak_imgs = []
for i in range(1, 25):
    fn = f'kodim{i:02d}.png'
    img = iio.imread(str(KD/fn)).astype(np.float32) / 255.0
    img = img[:, :, :3]
    h, w = img.shape[:2]; h &= ~1; w &= ~1; img = img[:h, :w]
    kodak_imgs.append((fn, img))

# Edge subset (same images subset for zipper)
edge_imgs = [(fn, img) for fn, img in kodak_imgs if fn in EDGE_IMAGES]

print(f'{"iter":<6} {"cpsnr24":>9} {"cz_mean":>9} {"cz_peak":>9} {"time_k19":>10}')
print('-' * 55)

for itn in ITERS:
    # Kodak-24 CPSNR (no timing collected, first run is warmup)
    cps = []
    for fn, img in kodak_imgs:
        out, _ = run(img, itn)
        cps.append(cpsnr(img, out))

    # Edge zipper
    czm_list, czp_list = [], []
    for fn, img in edge_imgs:
        out, _ = run(img, itn)
        mask = edge_mask(img)
        czm, czp = chroma_zip(img, out, mask)
        czm_list.append(czm); czp_list.append(czp)

    # kodim19 timing (best of 5 to minimize noise)
    _, _ = run(kodak_imgs[18][1], itn)  # warmup
    ts = []
    for _ in range(5):
        _, t = run(kodak_imgs[18][1], itn)
        ts.append(t)
    best_t = min(ts)

    print(f'{itn:<6} {np.mean(cps):>9.3f} {np.mean(czm_list):>9.4f} '
          f'{np.mean(czp_list):>9.4f} {best_t:>8.3f}s')
