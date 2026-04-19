"""Python reference implementation of paper-exact RI/MLRI green interpolation
(Kiku 2013 / Kiku 2014). Used to debug the C port.

Flow (horizontal direction, RGGB):
  rawH      = 0.5 * (raw[:, :-2] + raw[:, 2:])    (pad replicate)
  GuideR_H  = R_channel + rawH * maskGr
  GuideB_H  = B_channel + rawH * maskGb
  tentGr_H  = GF(G*maskGr, GuideR_H, maskGr, h=5, v=0)
  tentGb_H  = GF(G*maskGb, GuideB_H, maskGb, h=5, v=0)
  resGr_H   = (G - tentGr_H) * maskGr
  resGb_H   = (G - tentGb_H) * maskGb
  res_upGr  = 0.5 * (resGr[:, :-2] + resGr[:, 2:])
  res_upGb  = 0.5 * (resGb[:, :-2] + resGb[:, 2:])
  green_H = G + (tentGr + res_upGr)*maskR + (tentGb + res_upGb)*maskB

For vertical we swap Gr <-> Gb masks (reference tomsangotw code pattern).
"""
import numpy as np
from scipy import ndimage
import imageio.v3 as iio
from colour_demosaicing import mosaicing_CFA_Bayer
from pathlib import Path


def build_masks_rggb(h, w):
    """RGGB-specific masks. Row 0: R G R G, Row 1: G B G B."""
    mR = np.zeros((h, w), dtype=np.float32)
    mG = np.zeros((h, w), dtype=np.float32)
    mB = np.zeros((h, w), dtype=np.float32)
    mGr = np.zeros((h, w), dtype=np.float32)
    mGb = np.zeros((h, w), dtype=np.float32)
    mR[0::2, 0::2] = 1     # even row, even col
    mGr[0::2, 1::2] = 1    # even row, odd col
    mGb[1::2, 0::2] = 1    # odd row, even col
    mB[1::2, 1::2] = 1     # odd row, odd col
    mG = mGr + mGb
    return mR, mG, mB, mGr, mGb


def guided_filter_masked(p, I, mask, rh, rv, eps):
    """Mask-aware guided filter (tomsangotw's guided_filter_modified).

    Differences vs my earlier version:
      - var is thresholded to >= 1e-5 (not just >= 0) so near-constant
        windows don't generate wild a values.
      - N=0 safety is handled by adding 1 to N (avoid div by zero), not by
        zeroing out a, b afterwards.
      - All statistics divided by N_safe (per-window valid count).
      - a, b smoothed over full window (N2 = window size constant).
    """
    kernel_shape = (2 * rv + 1, 2 * rh + 1)
    wsize = kernel_shape[0] * kernel_shape[1]
    def box(x):
        return ndimage.uniform_filter(x, size=kernel_shape, mode='reflect') * wsize

    N = box(mask)
    N_safe = N + (N == 0).astype(np.float32)
    N2 = wsize

    mean_I  = box(I * mask) / N_safe
    mean_p  = box(p) / N_safe
    mean_II = box(I * I * mask) / N_safe
    mean_Ip = box(I * p) / N_safe

    var = mean_II - mean_I * mean_I
    var = np.maximum(var, 1e-5)
    cov = mean_Ip - mean_I * mean_p

    a = cov / (var + eps)
    b = mean_p - a * mean_I

    mean_a = box(a) / N2
    mean_b = box(b) / N2
    return mean_a * I + mean_b


def bilinear_1d_h(src):
    # dst[x] = 0.5*(src[x-1] + src[x+1]) with mirror boundary
    return 0.5 * (np.roll(src, 1, axis=1) + np.roll(src, -1, axis=1))


def bilinear_1d_v(src):
    return 0.5 * (np.roll(src, 1, axis=0) + np.roll(src, -1, axis=0))


def green_direction(raw, mR, mG, mB, mGr, mGb, method, direction, eps=1e-6):
    """method: 'RI' or 'MLRI'. direction: 'H' or 'V'."""
    if direction == 'H':
        raw_bi = bilinear_1d_h(raw)
        rh, rv = 5, 0
        mGr_dir, mGb_dir = mGr, mGb
    else:
        raw_bi = bilinear_1d_v(raw)
        rh, rv = 0, 5
        mGr_dir, mGb_dir = mGb, mGr  # swap for V

    R_ch = raw * mR
    G_ch = raw * mG
    B_ch = raw * mB

    # Guides
    GuideR = R_ch + raw_bi * mGr_dir
    GuideB = B_ch + raw_bi * mGb_dir

    # GF predictions
    gf = guided_filter_masked  # could switch MLRI variant here

    tentGr = gf(G_ch * mGr_dir, GuideR, mGr_dir, rh, rv, eps)
    tentGb = gf(G_ch * mGb_dir, GuideB, mGb_dir, rh, rv, eps)

    # Residuals
    resGr = (G_ch - tentGr) * mGr_dir
    resGb = (G_ch - tentGb) * mGb_dir

    # Bilinear upsample
    if direction == 'H':
        resGr_up = bilinear_1d_h(resGr)
        resGb_up = bilinear_1d_h(resGb)
    else:
        resGr_up = bilinear_1d_v(resGr)
        resGb_up = bilinear_1d_v(resGb)

    # Compose
    green = G_ch.copy()
    green += (tentGr + resGr_up) * mR
    green += (tentGb + resGb_up) * mB

    return green


def demosaic_paper_ri(img_gt, pattern='RGGB', method='RI', direction='H'):
    """Just green interpolation for debugging; R/B stay mosaic."""
    cfa = mosaicing_CFA_Bayer(img_gt, pattern).astype(np.float32)
    h, w = cfa.shape
    mR, mG, mB, mGr, mGb = build_masks_rggb(h, w)
    green = green_direction(cfa, mR, mG, mB, mGr, mGb, method, direction)
    # Quick color output: green everywhere, R/B from raw bilinear for sanity
    out = np.zeros((h, w, 3), dtype=np.float32)
    out[:, :, 1] = green
    # Use simple bilinear for R, B (just for CPSNR check)
    out[:, :, 0] = ndimage.uniform_filter(cfa * mR, size=3, mode='reflect') * 9 / np.maximum(
        ndimage.uniform_filter(mR, size=3, mode='reflect') * 9, 1)
    out[:, :, 2] = ndimage.uniform_filter(cfa * mB, size=3, mode='reflect') * 9 / np.maximum(
        ndimage.uniform_filter(mB, size=3, mode='reflect') * 9, 1)
    return out, green


def cpsnr(gt, pred, border=12):
    h, w = gt.shape[:2]
    g = gt[border:h-border, border:w-border].astype(np.float64)
    p = pred[border:h-border, border:w-border].astype(np.float64)
    mse = np.mean((g - p) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')


def green_psnr(gt_green, pred_green, border=12):
    h, w = gt_green.shape
    g = gt_green[border:h-border, border:w-border].astype(np.float64)
    p = pred_green[border:h-border, border:w-border].astype(np.float64)
    mse = np.mean((g - p) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')


if __name__ == '__main__':
    KD = Path(r'C:/Users/dregsist/datasets/kodak')
    for fn in ['kodim01.png', 'kodim19.png', 'kodim23.png']:
        img = iio.imread(str(KD / fn)).astype(np.float32) / 255.0
        img = img[:, :, :3]
        h, w = img.shape[:2]
        # For RGGB, truncate to even dims
        h &= ~1; w &= ~1
        img = img[:h, :w]
        gt_green = img[:, :, 1]

        print(f'\n{fn}  ({h}x{w})')
        for direction in ('H', 'V'):
            _, g = demosaic_paper_ri(img, pattern='RGGB', method='RI', direction=direction)
            print(f'  RI-{direction}: green PSNR = {green_psnr(gt_green, g):6.3f} dB')
