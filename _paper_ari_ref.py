"""Port of Monno's ARI green-interpolation (ICIP 2015 MATLAB code).

Mirrors green_interpolation.m + guidedfilter.m + guidedfilter_MLRI.m
line-by-line so we can verify the algorithm in Python and then port to C.
"""
import numpy as np
from scipy import ndimage
from colour_demosaicing import mosaicing_CFA_Bayer
import imageio.v3 as iio
from pathlib import Path


def boxfilter(src, h, v):
    """Matlab boxfilter SUM (not mean) over (2*v+1, 2*h+1) window."""
    # h = horizontal radius, v = vertical radius
    ksize_h = 2 * h + 1
    ksize_v = 2 * v + 1
    # uniform_filter * window_size == box filter sum
    return ndimage.uniform_filter(src, size=(ksize_v, ksize_h),
                                  mode='reflect') * (ksize_h * ksize_v)


def imfilter(src, kernel):
    return ndimage.convolve(src, kernel, mode='reflect')


def guidedfilter(I, p, M, h, v, eps):
    """Monno's mask-aware guided filter with error-weighted a,b smoothing."""
    N = boxfilter(M, h, v)
    N = np.where(N == 0, 1.0, N)
    hei, wid = I.shape
    N2 = boxfilter(np.ones_like(I), h, v)

    mean_I  = boxfilter(I * M, h, v) / N
    mean_p  = boxfilter(p * M, h, v) / N
    mean_Ip = boxfilter(I * p * M, h, v) / N
    cov_Ip  = mean_Ip - mean_I * mean_p
    mean_II = boxfilter(I * I * M, h, v) / N
    var_I   = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # Error-weighted average of a, b (paper's key modification)
    dif = (boxfilter(I * I * M, h, v) * a * a + b * b * N
           + boxfilter(p * p * M, h, v)
           + 2 * a * b * boxfilter(I * M, h, v)
           - 2 * b * boxfilter(p * M, h, v)
           - 2 * a * boxfilter(p * I * M, h, v))
    dif = dif / N
    dif = np.sqrt(np.maximum(dif, 0))
    dif = np.where(dif < 1e-3, 1e-3, dif)
    dif = 1.0 / dif
    wdif = boxfilter(dif, h, v)
    mean_a = boxfilter(a * dif, h, v) / (wdif + 1e-4)
    mean_b = boxfilter(b * dif, h, v) / (wdif + 1e-4)

    return mean_a * I + mean_b


def guidedfilter_MLRI(G, R, mask, I, p, M, h, v, eps):
    """Paper's MLRI guided filter (Kiku 2014).

    Signature / semantics match guidedfilter_MLRI.m line-by-line:
      G, R       : intensity-domain guide and input  (for b and output)
      mask       : intensity-domain sample validity  (for mean_G, mean_R, dif)
      I, p       : Laplacian-domain guide and input  (for slope a)
      M          : Laplacian-domain sample validity  (for mean_Ip, mean_II)
    The slope uses origin-through regression on the Laplacian samples:
      a = mean(I*p) / (mean(I*I) + eps)
    b is then fit in the intensity domain: b = mean_R - a*mean_G.
    a, b are weighted-averaged by inverse RMS error.
    """
    N3 = boxfilter(mask, h, v)
    N3 = np.where(N3 == 0, 1.0, N3)
    N = boxfilter(M, h, v)
    N = np.where(N == 0, 1.0, N)

    # slope in Laplacian domain (through origin, no centring)
    mean_Ip = boxfilter(I * p * M, h, v) / N
    mean_II = boxfilter(I * I * M, h, v) / N
    a = mean_Ip / (mean_II + eps)

    # offset in intensity domain
    mean_G = boxfilter(G * mask, h, v) / N3
    mean_R = boxfilter(R * mask, h, v) / N3
    b = mean_R - a * mean_G

    # Error-weighted smoothing of a, b
    dif = (boxfilter(G * G * mask, h, v) * a * a + b * b * N3
           + boxfilter(R * R * mask, h, v)
           + 2 * a * b * boxfilter(G * mask, h, v)
           - 2 * b * boxfilter(R * mask, h, v)
           - 2 * a * boxfilter(R * G * mask, h, v))
    dif = dif / N3
    dif = np.sqrt(np.maximum(dif, 0))
    dif = np.where(dif < 1e-3, 1e-3, dif)
    dif = 1.0 / dif
    wdif = boxfilter(dif, h, v)
    mean_a = boxfilter(a * dif, h, v) / (wdif + 1e-4)
    mean_b = boxfilter(b * dif, h, v) / (wdif + 1e-4)

    return mean_a * G + mean_b


def green_interpolation_ari(mosaic, mask, pattern='rggb', eps=1e-6):
    """Exact port of Monno's green_interpolation.m."""
    rawq = mosaic.sum(axis=-1)
    h_im, w_im = rawq.shape

    # Gr / Gb sub-masks (depend on pattern)
    maskGr = np.zeros_like(rawq)
    maskGb = np.zeros_like(rawq)
    if pattern == 'grbg':
        maskGr[0::2, 0::2] = 1
        maskGb[1::2, 1::2] = 1
    elif pattern == 'rggb':
        maskGr[0::2, 1::2] = 1
        maskGb[1::2, 0::2] = 1
    elif pattern == 'gbrg':
        maskGb[0::2, 0::2] = 1
        maskGr[1::2, 1::2] = 1
    elif pattern == 'bggr':
        maskGb[0::2, 1::2] = 1
        maskGr[1::2, 0::2] = 1

    Mrh = mask[:, :, 0] + maskGr
    Mbh = mask[:, :, 2] + maskGb
    Mrv = mask[:, :, 0] + maskGb
    Mbv = mask[:, :, 2] + maskGr

    # Initial bilinear raw
    Kh = np.array([[0.5, 0, 0.5]], dtype=np.float32)
    Kv = Kh.T
    rawh = imfilter(rawq, Kh)
    rawv = imfilter(rawq, Kv)

    # Initial guides (per direction)
    Guidegrh = mosaic[:, :, 1] * maskGr + rawh * mask[:, :, 0]
    Guidegbh = mosaic[:, :, 1] * maskGb + rawh * mask[:, :, 2]
    Guiderh  = mosaic[:, :, 0] + rawh * maskGr
    Guidebh  = mosaic[:, :, 2] + rawh * maskGb
    Guidegrv = mosaic[:, :, 1] * maskGb + rawv * mask[:, :, 0]
    Guidegbv = mosaic[:, :, 1] * maskGr + rawv * mask[:, :, 2]
    Guiderv  = mosaic[:, :, 0] + rawv * maskGb
    Guidebv  = mosaic[:, :, 2] + rawv * maskGr

    # Initial window sizes
    h, v = 2, 1
    h2, v2 = 4, 0
    itnum = 11

    shape = maskGr.shape
    RI_w2h  = np.full(shape, 1e32, dtype=np.float32)
    RI_w2v  = np.full(shape, 1e32, dtype=np.float32)
    MLRI_w2h = np.full(shape, 1e32, dtype=np.float32)
    MLRI_w2v = np.full(shape, 1e32, dtype=np.float32)

    RI_Guidegrh = Guidegrh.copy(); RI_Guidegbh = Guidegbh.copy()
    RI_Guiderh = Guiderh.copy();  RI_Guidebh = Guidebh.copy()
    RI_Guidegrv = Guidegrv.copy(); RI_Guidegbv = Guidegbv.copy()
    RI_Guiderv = Guiderv.copy();  RI_Guidebv = Guidebv.copy()
    MLRI_Guidegrh = Guidegrh.copy(); MLRI_Guidegbh = Guidegbh.copy()
    MLRI_Guiderh  = Guiderh.copy();  MLRI_Guidebh  = Guidebh.copy()
    MLRI_Guidegrv = Guidegrv.copy(); MLRI_Guidegbv = Guidegbv.copy()
    MLRI_Guiderv  = Guiderv.copy();  MLRI_Guidebv  = Guidebv.copy()

    RI_Gh = Guidegrh + Guidegbh
    RI_Gv = Guidegrv + Guidegbv
    MLRI_Gh = Guidegrh + Guidegbh
    MLRI_Gv = Guidegrv + Guidegbv

    Fh5 = np.array([[-1, 0, 2, 0, -1]], dtype=np.float32)
    Fv5 = Fh5.T
    Kh3 = np.array([[0.5, 1, 0.5]], dtype=np.float32)
    Kv3 = Kh3.T
    grad_h = np.array([[-1, 0, 1]], dtype=np.float32)
    grad_v = grad_h.T

    for ittime in range(itnum):
        # RI tentative (Eq. 2)
        RI_tentativeGrh = guidedfilter(RI_Guiderh,  RI_Guidegrh, Mrh, h, v, eps)
        RI_tentativeGbh = guidedfilter(RI_Guidebh,  RI_Guidegbh, Mbh, h, v, eps)
        RI_tentativeRh  = guidedfilter(RI_Guidegrh, RI_Guiderh,  Mrh, h, v, eps)
        RI_tentativeBh  = guidedfilter(RI_Guidegbh, RI_Guidebh,  Mbh, h, v, eps)
        RI_tentativeGrv = guidedfilter(RI_Guiderv,  RI_Guidegrv, Mrv, v, h, eps)
        RI_tentativeGbv = guidedfilter(RI_Guidebv,  RI_Guidegbv, Mbv, v, h, eps)
        RI_tentativeRv  = guidedfilter(RI_Guidegrv, RI_Guiderv,  Mrv, v, h, eps)
        RI_tentativeBv  = guidedfilter(RI_Guidegbv, RI_Guidebv,  Mbv, v, h, eps)

        # MLRI tentative
        difR  = imfilter(MLRI_Guiderh,  Fh5)
        difGr = imfilter(MLRI_Guidegrh, Fh5)
        difB  = imfilter(MLRI_Guidebh,  Fh5)
        difGb = imfilter(MLRI_Guidegbh, Fh5)
        MLRI_tentativeRh  = guidedfilter_MLRI(MLRI_Guidegrh, MLRI_Guiderh,  Mrh, difGr, difR,  mask[:, :, 0], h2, v2, eps)
        MLRI_tentativeBh  = guidedfilter_MLRI(MLRI_Guidegbh, MLRI_Guidebh,  Mbh, difGb, difB,  mask[:, :, 2], h2, v2, eps)
        MLRI_tentativeGrh = guidedfilter_MLRI(MLRI_Guiderh,  MLRI_Guidegrh, Mrh, difR,  difGr, maskGr,         h2, v2, eps)
        MLRI_tentativeGbh = guidedfilter_MLRI(MLRI_Guidebh,  MLRI_Guidegbh, Mbh, difB,  difGb, maskGb,         h2, v2, eps)
        difR  = imfilter(MLRI_Guiderv,  Fv5)
        difGr = imfilter(MLRI_Guidegrv, Fv5)
        difB  = imfilter(MLRI_Guidebv,  Fv5)
        difGb = imfilter(MLRI_Guidegbv, Fv5)
        MLRI_tentativeRv  = guidedfilter_MLRI(MLRI_Guidegrv, MLRI_Guiderv,  Mrv, difGr, difR,  mask[:, :, 0], v2, h2, eps)
        MLRI_tentativeBv  = guidedfilter_MLRI(MLRI_Guidegbv, MLRI_Guidebv,  Mbv, difGb, difB,  mask[:, :, 2], v2, h2, eps)
        MLRI_tentativeGrv = guidedfilter_MLRI(MLRI_Guiderv,  MLRI_Guidegrv, Mrv, difR,  difGr, maskGb,         v2, h2, eps)
        MLRI_tentativeGbv = guidedfilter_MLRI(MLRI_Guidebv,  MLRI_Guidegbv, Mbv, difB,  difGb, maskGr,         v2, h2, eps)

        # residuals (Eq. 4)
        RI_residualGrh = (mosaic[:,:,1] - RI_tentativeGrh) * maskGr
        RI_residualGbh = (mosaic[:,:,1] - RI_tentativeGbh) * maskGb
        RI_residualRh  = (mosaic[:,:,0] - RI_tentativeRh)  * mask[:,:,0]
        RI_residualBh  = (mosaic[:,:,2] - RI_tentativeBh)  * mask[:,:,2]
        RI_residualGrv = (mosaic[:,:,1] - RI_tentativeGrv) * maskGb
        RI_residualGbv = (mosaic[:,:,1] - RI_tentativeGbv) * maskGr
        RI_residualRv  = (mosaic[:,:,0] - RI_tentativeRv)  * mask[:,:,0]
        RI_residualBv  = (mosaic[:,:,2] - RI_tentativeBv)  * mask[:,:,2]
        MLRI_residualGrh = (mosaic[:,:,1] - MLRI_tentativeGrh) * maskGr
        MLRI_residualGbh = (mosaic[:,:,1] - MLRI_tentativeGbh) * maskGb
        MLRI_residualRh  = (mosaic[:,:,0] - MLRI_tentativeRh)  * mask[:,:,0]
        MLRI_residualBh  = (mosaic[:,:,2] - MLRI_tentativeBh)  * mask[:,:,2]
        MLRI_residualGrv = (mosaic[:,:,1] - MLRI_tentativeGrv) * maskGb
        MLRI_residualGbv = (mosaic[:,:,1] - MLRI_tentativeGbv) * maskGr
        MLRI_residualRv  = (mosaic[:,:,0] - MLRI_tentativeRv)  * mask[:,:,0]
        MLRI_residualBv  = (mosaic[:,:,2] - MLRI_tentativeBv)  * mask[:,:,2]

        # Residual upsample (Eq. 5). Note Kh3 = [0.5, 1, 0.5] here (unit center!)
        RI_residualGrh   = imfilter(RI_residualGrh, Kh3)
        RI_residualGbh   = imfilter(RI_residualGbh, Kh3)
        RI_residualRh    = imfilter(RI_residualRh, Kh3)
        RI_residualBh    = imfilter(RI_residualBh, Kh3)
        MLRI_residualGrh = imfilter(MLRI_residualGrh, Kh3)
        MLRI_residualGbh = imfilter(MLRI_residualGbh, Kh3)
        MLRI_residualRh  = imfilter(MLRI_residualRh, Kh3)
        MLRI_residualBh  = imfilter(MLRI_residualBh, Kh3)
        RI_residualGrv   = imfilter(RI_residualGrv, Kv3)
        RI_residualGbv   = imfilter(RI_residualGbv, Kv3)
        RI_residualRv    = imfilter(RI_residualRv, Kv3)
        RI_residualBv    = imfilter(RI_residualBv, Kv3)
        MLRI_residualGrv = imfilter(MLRI_residualGrv, Kv3)
        MLRI_residualGbv = imfilter(MLRI_residualGbv, Kv3)
        MLRI_residualRv  = imfilter(MLRI_residualRv, Kv3)
        MLRI_residualBv  = imfilter(MLRI_residualBv, Kv3)

        # add tentative (Eq. 6)
        RI_Grh = (RI_tentativeGrh + RI_residualGrh) * mask[:,:,0]
        RI_Gbh = (RI_tentativeGbh + RI_residualGbh) * mask[:,:,2]
        RI_Rh  = (RI_tentativeRh  + RI_residualRh)  * maskGr
        RI_Bh  = (RI_tentativeBh  + RI_residualBh)  * maskGb
        RI_Grv = (RI_tentativeGrv + RI_residualGrv) * mask[:,:,0]
        RI_Gbv = (RI_tentativeGbv + RI_residualGbv) * mask[:,:,2]
        RI_Rv  = (RI_tentativeRv  + RI_residualRv)  * maskGb
        RI_Bv  = (RI_tentativeBv  + RI_residualBv)  * maskGr
        MLRI_Grh = (MLRI_tentativeGrh + MLRI_residualGrh) * mask[:,:,0]
        MLRI_Gbh = (MLRI_tentativeGbh + MLRI_residualGbh) * mask[:,:,2]
        MLRI_Rh  = (MLRI_tentativeRh  + MLRI_residualRh)  * maskGr
        MLRI_Bh  = (MLRI_tentativeBh  + MLRI_residualBh)  * maskGb
        MLRI_Grv = (MLRI_tentativeGrv + MLRI_residualGrv) * mask[:,:,0]
        MLRI_Gbv = (MLRI_tentativeGbv + MLRI_residualGbv) * mask[:,:,2]
        MLRI_Rv  = (MLRI_tentativeRv  + MLRI_residualRv)  * maskGb
        MLRI_Bv  = (MLRI_tentativeBv  + MLRI_residualBv)  * maskGr

        # criteria (Eq. 3) -- guide minus tentative, masked
        RI_criGrh = (RI_Guidegrh - RI_tentativeGrh) * Mrh
        RI_criGbh = (RI_Guidegbh - RI_tentativeGbh) * Mbh
        RI_criRh  = (RI_Guiderh  - RI_tentativeRh)  * Mrh
        RI_criBh  = (RI_Guidebh  - RI_tentativeBh)  * Mbh
        RI_criGrv = (RI_Guidegrv - RI_tentativeGrv) * Mrv
        RI_criGbv = (RI_Guidegbv - RI_tentativeGbv) * Mbv
        RI_criRv  = (RI_Guiderv  - RI_tentativeRv)  * Mrv
        RI_criBv  = (RI_Guidebv  - RI_tentativeBv)  * Mbv
        MLRI_criGrh = (MLRI_Guidegrh - MLRI_tentativeGrh) * Mrh
        MLRI_criGbh = (MLRI_Guidegbh - MLRI_tentativeGbh) * Mbh
        MLRI_criRh  = (MLRI_Guiderh  - MLRI_tentativeRh)  * Mrh
        MLRI_criBh  = (MLRI_Guidebh  - MLRI_tentativeBh)  * Mbh
        MLRI_criGrv = (MLRI_Guidegrv - MLRI_tentativeGrv) * Mrv
        MLRI_criGbv = (MLRI_Guidegbv - MLRI_tentativeGbv) * Mbv
        MLRI_criRv  = (MLRI_Guiderv  - MLRI_tentativeRv)  * Mrv
        MLRI_criBv  = (MLRI_Guidebv  - MLRI_tentativeBv)  * Mbv

        # Gradient of criteria
        RI_difcriGrh = np.abs(imfilter(RI_criGrh, grad_h))
        RI_difcriGbh = np.abs(imfilter(RI_criGbh, grad_h))
        RI_difcriRh  = np.abs(imfilter(RI_criRh,  grad_h))
        RI_difcriBh  = np.abs(imfilter(RI_criBh,  grad_h))
        MLRI_difcriGrh = np.abs(imfilter(MLRI_criGrh, grad_h))
        MLRI_difcriGbh = np.abs(imfilter(MLRI_criGbh, grad_h))
        MLRI_difcriRh  = np.abs(imfilter(MLRI_criRh,  grad_h))
        MLRI_difcriBh  = np.abs(imfilter(MLRI_criBh,  grad_h))
        RI_difcriGrv = np.abs(imfilter(RI_criGrv, grad_v))
        RI_difcriGbv = np.abs(imfilter(RI_criGbv, grad_v))
        RI_difcriRv  = np.abs(imfilter(RI_criRv,  grad_v))
        RI_difcriBv  = np.abs(imfilter(RI_criBv,  grad_v))
        MLRI_difcriGrv = np.abs(imfilter(MLRI_criGrv, grad_v))
        MLRI_difcriGbv = np.abs(imfilter(MLRI_criGbv, grad_v))
        MLRI_difcriRv  = np.abs(imfilter(MLRI_criRv,  grad_v))
        MLRI_difcriBv  = np.abs(imfilter(MLRI_criBv,  grad_v))

        # abs of criteria themselves
        RI_criGrh = np.abs(RI_criGrh); RI_criGbh = np.abs(RI_criGbh)
        RI_criRh  = np.abs(RI_criRh);  RI_criBh  = np.abs(RI_criBh)
        RI_criGrv = np.abs(RI_criGrv); RI_criGbv = np.abs(RI_criGbv)
        RI_criRv  = np.abs(RI_criRv);  RI_criBv  = np.abs(RI_criBv)
        MLRI_criGrh = np.abs(MLRI_criGrh); MLRI_criGbh = np.abs(MLRI_criGbh)
        MLRI_criRh  = np.abs(MLRI_criRh);  MLRI_criBh  = np.abs(MLRI_criBh)
        MLRI_criGrv = np.abs(MLRI_criGrv); MLRI_criGbv = np.abs(MLRI_criGbv)
        MLRI_criRv  = np.abs(MLRI_criRv);  MLRI_criBv  = np.abs(MLRI_criBv)

        # sum Gr+R / Gb+B criteria
        RI_criGRh = (RI_criGrh + RI_criRh) * Mrh
        RI_criGBh = (RI_criGbh + RI_criBh) * Mbh
        RI_criGRv = (RI_criGrv + RI_criRv) * Mrv
        RI_criGBv = (RI_criGbv + RI_criBv) * Mbv
        MLRI_criGRh = (MLRI_criGrh + MLRI_criRh) * Mrh
        MLRI_criGBh = (MLRI_criGbh + MLRI_criBh) * Mbh
        MLRI_criGRv = (MLRI_criGrv + MLRI_criRv) * Mrv
        MLRI_criGBv = (MLRI_criGbv + MLRI_criBv) * Mbv
        RI_difcriGRh = (RI_difcriGrh + RI_difcriRh) * Mrh
        RI_difcriGBh = (RI_difcriGbh + RI_difcriBh) * Mbh
        RI_difcriGRv = (RI_difcriGrv + RI_difcriRv) * Mrv
        RI_difcriGBv = (RI_difcriGbv + RI_difcriBv) * Mbv
        MLRI_difcriGRh = (MLRI_difcriGrh + MLRI_difcriRh) * Mrh
        MLRI_difcriGBh = (MLRI_difcriGbh + MLRI_difcriBh) * Mbh
        MLRI_difcriGRv = (MLRI_difcriGrv + MLRI_difcriRv) * Mrv
        MLRI_difcriGBv = (MLRI_difcriGbv + MLRI_difcriBv) * Mbv

        RI_crih = RI_criGRh + RI_criGBh
        RI_criv = RI_criGRv + RI_criGBv
        MLRI_crih = MLRI_criGRh + MLRI_criGBh
        MLRI_criv = MLRI_criGRv + MLRI_criGBv
        RI_difcrih = RI_difcriGRh + RI_difcriGBh
        RI_difcriv = RI_difcriGRv + RI_difcriGBv
        MLRI_difcrih = MLRI_difcriGRh + MLRI_difcriGBh
        MLRI_difcriv = MLRI_difcriGRv + MLRI_difcriGBv

        # smoothing sigma=2 Gaussian 5x5
        RI_crih = ndimage.gaussian_filter(RI_crih, sigma=2.0, truncate=1.5, mode='reflect')
        MLRI_crih = ndimage.gaussian_filter(MLRI_crih, sigma=2.0, truncate=1.5, mode='reflect')
        RI_difcrih = ndimage.gaussian_filter(RI_difcrih, sigma=2.0, truncate=1.5, mode='reflect')
        MLRI_difcrih = ndimage.gaussian_filter(MLRI_difcrih, sigma=2.0, truncate=1.5, mode='reflect')
        RI_criv = ndimage.gaussian_filter(RI_criv, sigma=2.0, truncate=1.5, mode='reflect')
        MLRI_criv = ndimage.gaussian_filter(MLRI_criv, sigma=2.0, truncate=1.5, mode='reflect')
        RI_difcriv = ndimage.gaussian_filter(RI_difcriv, sigma=2.0, truncate=1.5, mode='reflect')
        MLRI_difcriv = ndimage.gaussian_filter(MLRI_difcriv, sigma=2.0, truncate=1.5, mode='reflect')

        RI_wh = (RI_crih ** 2) * RI_difcrih
        RI_wv = (RI_criv ** 2) * RI_difcriv
        MLRI_wh = (MLRI_crih ** 2) * MLRI_difcrih
        MLRI_wv = (MLRI_criv ** 2) * MLRI_difcriv

        # update best pixels
        RI_pih   = RI_wh   < RI_w2h
        RI_piv   = RI_wv   < RI_w2v
        MLRI_pih = MLRI_wh < MLRI_w2h
        MLRI_piv = MLRI_wv < MLRI_w2v

        # guide update
        RI_Guidegrh = mosaic[:,:,1] * maskGr + RI_Grh
        RI_Guidegbh = mosaic[:,:,1] * maskGb + RI_Gbh
        RI_Guidegh  = RI_Guidegrh + RI_Guidegbh
        RI_Guiderh  = mosaic[:,:,0] + RI_Rh
        RI_Guidebh  = mosaic[:,:,2] + RI_Bh
        RI_Guidegrv = mosaic[:,:,1] * maskGb + RI_Grv
        RI_Guidegbv = mosaic[:,:,1] * maskGr + RI_Gbv
        RI_Guidegv  = RI_Guidegrv + RI_Guidegbv
        RI_Guiderv  = mosaic[:,:,0] + RI_Rv
        RI_Guidebv  = mosaic[:,:,2] + RI_Bv
        MLRI_Guidegrh = mosaic[:,:,1] * maskGr + MLRI_Grh
        MLRI_Guidegbh = mosaic[:,:,1] * maskGb + MLRI_Gbh
        MLRI_Guidegh  = MLRI_Guidegrh + MLRI_Guidegbh
        MLRI_Guiderh  = mosaic[:,:,0] + MLRI_Rh
        MLRI_Guidebh  = mosaic[:,:,2] + MLRI_Bh
        MLRI_Guidegrv = mosaic[:,:,1] * maskGb + MLRI_Grv
        MLRI_Guidegbv = mosaic[:,:,1] * maskGr + MLRI_Gbv
        MLRI_Guidegv  = MLRI_Guidegrv + MLRI_Guidegbv
        MLRI_Guiderv  = mosaic[:,:,0] + MLRI_Rv
        MLRI_Guidebv  = mosaic[:,:,2] + MLRI_Bv

        # Select smallest per-pixel criterion
        RI_Gh = np.where(RI_pih, RI_Guidegh, RI_Gh)
        MLRI_Gh = np.where(MLRI_pih, MLRI_Guidegh, MLRI_Gh)
        RI_Gv = np.where(RI_piv, RI_Guidegv, RI_Gv)
        MLRI_Gv = np.where(MLRI_piv, MLRI_Guidegv, MLRI_Gv)
        RI_w2h = np.where(RI_pih, RI_wh, RI_w2h)
        RI_w2v = np.where(RI_piv, RI_wv, RI_w2v)
        MLRI_w2h = np.where(MLRI_pih, MLRI_wh, MLRI_w2h)
        MLRI_w2v = np.where(MLRI_piv, MLRI_wv, MLRI_w2v)

        h  += 2; v  += 1
        h2 += 2; v2 += 1

    # Final combine (Eq. 8-ish): weighted average by inverse criterion
    RI_w2h   = 1.0 / (RI_w2h   + 1e-10)
    RI_w2v   = 1.0 / (RI_w2v   + 1e-10)
    MLRI_w2h = 1.0 / (MLRI_w2h + 1e-10)
    MLRI_w2v = 1.0 / (MLRI_w2v + 1e-10)
    w = RI_w2h + RI_w2v + MLRI_w2h + MLRI_w2v
    green = (RI_w2h * RI_Gh + RI_w2v * RI_Gv
             + MLRI_w2h * MLRI_Gh + MLRI_w2v * MLRI_Gv) / (w + 1e-32)

    # restore measured G
    imask_g = (mask[:, :, 1] == 0).astype(np.float32)
    green = green * imask_g + mosaic[:, :, 1]
    return np.clip(green, 0, 1)


def red_blue_interpolation(green, mosaic, mask, h=5, v=5, eps=1e-10):
    """Port of red_interpolation.m / blue_interpolation.m.
    Both follow the same MLRI-based tentative + 7x7 bicubic residual pattern."""
    F = np.array([[0, 0, -1, 0, 0],
                  [0, 0,  0, 0, 0],
                  [-1, 0, 4, 0, -1],
                  [0, 0,  0, 0, 0],
                  [0, 0, -1, 0, 0]], dtype=np.float32)

    # Bicubic 7x7 interpolation kernel (from s.m, a = -0.5).
    def s_fn(x, a=-0.5):
        ax = abs(x)
        if 1 < ax < 2:
            return a * ax**3 - 5*a*ax**2 + 8*a*ax - 4*a
        elif ax >= 2:
            return 0.0
        else:
            return (a + 2) * ax**3 - (a + 3) * ax**2 + 1
    s32, s12, s0, s1 = s_fn(3/2), s_fn(1/2), s_fn(0), s_fn(1)
    A = s32*s32; B = s32*s12; C = s12*s12
    D = s0*s12; E = s0*s32; Ff = s1*s12; Gg = s1*s32
    H = np.array([
        [A, Gg, B, E, B, Gg, A],
        [Gg, 0, Ff, 0, Ff, 0, Gg],
        [B, Ff, C, D, C, Ff, B],
        [E, 0, D, 1, D, 0, E],
        [B, Ff, C, D, C, Ff, B],
        [Gg, 0, Ff, 0, Ff, 0, Gg],
        [A, Gg, B, E, B, Gg, A],
    ], dtype=np.float32)

    results = {}
    for ch, name in [(0, 'red'), (2, 'blue')]:
        lap_c = imfilter(mosaic[:, :, ch], F)
        lap_g = imfilter(green * mask[:, :, ch], F)
        tentative = guidedfilter_MLRI(green, mosaic[:, :, ch],
                                      mask[:, :, ch],
                                      lap_g, lap_c, mask[:, :, ch],
                                      h, v, eps)
        tentative = np.clip(tentative, 0, 1)
        residual = mask[:, :, ch] * (mosaic[:, :, ch] - tentative)
        residual = imfilter(residual, H)
        results[name] = np.clip(residual + tentative, 0, 1)

    return results['red'], results['blue']


def demosaic_full(img, pattern='rggb'):
    h, w = img.shape[:2]
    h &= ~1; w &= ~1
    img = img[:h, :w].astype(np.float32)
    cfa = mosaicing_CFA_Bayer(img, pattern.upper()).astype(np.float32)
    mask = np.zeros((h, w, 3), dtype=np.float32)
    if pattern == 'rggb':
        mask[0::2, 0::2, 0] = 1
        mask[0::2, 1::2, 1] = 1
        mask[1::2, 0::2, 1] = 1
        mask[1::2, 1::2, 2] = 1
    mosaic3 = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        mosaic3[..., c] = cfa * mask[..., c]
    green = green_interpolation_ari(mosaic3, mask, pattern=pattern, eps=1e-6)
    red, blue = red_blue_interpolation(green, mosaic3, mask, h=5, v=5, eps=1e-10)
    return np.stack([red, green, blue], axis=-1)


def cpsnr_full(gt, pred, border=12):
    h, w = gt.shape[:2]
    g = gt[border:h-border, border:w-border].astype(np.float64)
    p = pred[border:h-border, border:w-border].astype(np.float64)
    mse = np.mean((g - p) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')


if __name__ == '__main__':
    KD = Path(r'C:/Users/dregsist/datasets/kodak')
    all_cp = []
    for i in range(1, 25):
        fn = f'kodim{i:02d}.png'
        img = iio.imread(str(KD / fn)).astype(np.float32) / 255.0
        img = img[:, :, :3]
        out = demosaic_full(img, 'rggb')
        gt = img[:out.shape[0], :out.shape[1]]
        cp = cpsnr_full(gt, out)
        all_cp.append(cp)
        print(f'{fn}  ARI full CPSNR = {cp:.3f} dB')
    print(f'--------\n  average     = {np.mean(all_cp):.3f} dB  (paper target 40-41)')
