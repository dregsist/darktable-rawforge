"""SIDD medium bench: all target methods (ari, menon_r0/r1, amaze(+dual),
rcd(+dual)), both iso_clean and iso_noise tiers, averaged across 20 scenes.
Uses per-scene 3x3 CCM + bias fit so sensor color space differences don't
mask demosaic artifacts."""
import subprocess
import time
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from scipy import ndimage

HERE = Path(__file__).parent
SIDD_DIR = Path(r'C:/Users/dregsist/datasets/sidd/medium_bench')
tmp = HERE / 'results' / 'tmp_sidd_bench'
tmp.mkdir(parents=True, exist_ok=True)

PATTERN_HEX = {'RGGB': 0x94949494, 'GRBG': 0x61616161,
               'GBRG': 0x49494949, 'BGGR': 0x16161616}

SCENES = [
    ('0045', 'G4', 'BGGR', '010'), ('0059', 'G4', 'BGGR', '010'),
    ('0077', 'G4', 'BGGR', '010'), ('0099', 'G4', 'BGGR', '010'),
    ('0020', 'GP', 'BGGR', '010'), ('0036', 'GP', 'BGGR', '010'),
    ('0064', 'GP', 'BGGR', '010'), ('0083', 'GP', 'BGGR', '010'),
    ('0042', 'IP', 'RGGB', '010'), ('0069', 'IP', 'RGGB', '010'),
    ('0115', 'IP', 'RGGB', '010'), ('0165', 'IP', 'RGGB', '010'),
    ('0023', 'N6', 'BGGR', '010'), ('0048', 'N6', 'BGGR', '010'),
    ('0075', 'N6', 'BGGR', '010'), ('0120', 'N6', 'BGGR', '010'),
    ('0001', 'S6', 'GRBG', '010'), ('0010', 'S6', 'GRBG', '010'),
    ('0052', 'S6', 'GRBG', '010'), ('0154', 'S6', 'GRBG', '010'),
]

METHODS = [
    ('ari',      './test_ari.exe',   ['11']),
    ('amaze',    './test_amaze.exe', []),
    ('rcd',      './test_rcd.exe',   []),
    ('menon_r1', './test_menon.exe', ['1']),
    ('menon_r0', './test_menon.exe', ['0']),
]
DUAL_BASES = {'amaze', 'rcd'}
DUAL_SLIDER = 0.20
FIT_BORDER = 64
METRIC_BORDER = 32


def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x/12.92, ((x+0.055)/1.055)**2.4).astype(np.float32)


def linear_to_srgb(y):
    y = np.clip(y, 0.0, 1.0)
    return np.where(y <= 0.0031308, y*12.92, 1.055*(y**(1.0/2.4))-0.055).astype(np.float32)


def fit_ccm_3x3(sensor_rgb, ref_lin, border=FIT_BORDER):
    h,w = sensor_rgb.shape[:2]
    p = sensor_rgb[border:h-border, border:w-border].reshape(-1,3).astype(np.float64)
    r = ref_lin[border:h-border, border:w-border].reshape(-1,3).astype(np.float64)
    X = np.hstack([p, np.ones((p.shape[0],1))])
    sol,_,_,_ = np.linalg.lstsq(X, r, rcond=None)
    return sol[:3].astype(np.float32), sol[3].astype(np.float32)


def apply_ccm(sensor, M, bias):
    return (sensor.reshape(-1,3) @ M + bias).reshape(sensor.shape)


def scharr_gradient_2d(p):
    gx = (47./255.)*(np.roll(p,(1,1),(0,1))-np.roll(p,(1,-1),(0,1))
                    +np.roll(p,(-1,1),(0,1))-np.roll(p,(-1,-1),(0,1))) \
       + (162./255.)*(np.roll(p,(0,1),(0,1))-np.roll(p,(0,-1),(0,1)))
    gy = (47./255.)*(np.roll(p,(1,1),(0,1))-np.roll(p,(-1,1),(0,1))
                    +np.roll(p,(1,-1),(0,1))-np.roll(p,(-1,-1),(0,1))) \
       + (162./255.)*(np.roll(p,(1,0),(0,1))-np.roll(p,(-1,0),(0,1)))
    return np.hypot(gx, gy)


def dual_blend(base, vng, slider=DUAL_SLIDER):
    r = np.clip(base[:,:,0],0,None); g = np.clip(base[:,:,1],0,None); b = np.clip(base[:,:,2],0,None)
    sch = np.clip(scharr_gradient_2d(np.sqrt((r+g+b)/3.0))/16.0, 0, 1)
    contrastf = 0.005*(slider**1.1)
    arg = np.clip(16.0 - (16.0/max(contrastf,1e-7))*sch, -80, 80)
    det = np.clip(1.0/(1.0+np.exp(arg)), 0, 1)
    mask = np.clip(ndimage.gaussian_filter(det, 2.0, mode='reflect'), 0, 1)[:,:,None]
    return mask*base + (1.0-mask)*vng


def run_exe(exe, extra, w, h, hex_filters, cfa_bytes, tag):
    in_p = tmp / f'{tag}_in.bin';  in_p.write_bytes(cfa_bytes)
    out_p = tmp / f'{tag}_out.bin'
    t0 = time.time()
    subprocess.run([exe, str(w), str(h), f'0x{hex_filters:08X}',
                    str(in_p), str(out_p), *extra], capture_output=True)
    dt = time.time()-t0
    out = np.frombuffer(out_p.read_bytes(), dtype=np.float32).reshape(h,w,4)[:,:,:3]
    return out, dt


def cpsnr(gt, pred, border=METRIC_BORDER):
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


def chroma_zipper(gt, pred, mask, border=METRIC_BORDER):
    g = gt.astype(np.float64); p = pred.astype(np.float64)
    dR = (p[:,:,0]-p[:,:,1]) - (g[:,:,0]-g[:,:,1])
    dB = (p[:,:,2]-p[:,:,1]) - (g[:,:,2]-g[:,:,1])
    zmag = np.abs(ndimage.laplace(dR)) + np.abs(ndimage.laplace(dB))
    h,w = gt.shape[:2]
    m = mask[border:h-border, border:w-border]
    v = zmag[border:h-border, border:w-border][m]
    cz_m = float(np.mean(v)) if v.size else float('nan')
    cz_pk = float(ndimage.uniform_filter(zmag, 64).max())
    return cz_m, cz_pk


def process_tier(tier):
    print(f'\n=== Tier: {tier} ===')
    stats = {}  # name -> lists
    times = {}
    for (scene_id, phone, pat, frame) in SCENES:
        tag = f'{scene_id}_{phone}_{pat}_{frame}'
        gt_path = SIDD_DIR / f'{tag}_gt_srgb.npy'
        raw_path = SIDD_DIR / f'{tag}_{"gt_raw" if tier=="iso_clean" else "noisy_raw"}.npy'
        if not gt_path.exists() or not raw_path.exists():
            print(f'  MISSING {tag}'); continue
        gt_srgb = np.clip(np.load(str(gt_path)).astype(np.float32), 0, 1)
        raw = np.load(str(raw_path)).astype(np.float32)
        h, w = raw.shape
        cfa_bytes = np.ascontiguousarray(raw).tobytes()
        ref_lin = srgb_to_linear(gt_srgb)
        mask = edge_mask(gt_srgb)

        sensor_outs = {}
        for name, exe, extra in METHODS:
            out, dt = run_exe(exe, extra, w, h, PATTERN_HEX[pat], cfa_bytes,
                              f'{tier}_{tag}_{name}')
            sensor_outs[name] = out
            times.setdefault(name, []).append(dt)
        # vng for dual
        vng, _ = run_exe('./test_vng.exe', ['1'], w, h, PATTERN_HEX[pat], cfa_bytes,
                         f'{tier}_{tag}_vng')
        for base in DUAL_BASES:
            sensor_outs[f'{base}+dual'] = dual_blend(sensor_outs[base], vng)
            times.setdefault(f'{base}+dual', []).append(np.mean(times.get(base, [0])))

        for name, out_sensor in sensor_outs.items():
            M, bias = fit_ccm_3x3(out_sensor, ref_lin)
            fit_srgb = linear_to_srgb(apply_ccm(out_sensor, M, bias))
            d = stats.setdefault(name, {'cp':[], 'czm':[], 'czpk':[]})
            d['cp'].append(cpsnr(gt_srgb, fit_srgb))
            czm, czpk = chroma_zipper(gt_srgb, fit_srgb, mask)
            d['czm'].append(czm); d['czpk'].append(czpk)
        print(f'  {tag}: ari={stats["ari"]["cp"][-1]:.2f}')

    print()
    print(f'{"method":<12} {"CPSNR":>7} {"cz_mean":>9} {"cz_peak":>9}')
    print('-'*50)
    order = ['ari', 'amaze', 'amaze+dual', 'rcd', 'rcd+dual', 'menon_r1', 'menon_r0']
    for name in order:
        if name not in stats: continue
        s = stats[name]
        print(f'{name:<12} {np.mean(s["cp"]):>7.3f} '
              f'{np.mean(s["czm"]):>9.4f} {np.mean(s["czpk"]):>9.4f}')


for tier in ['iso_clean', 'iso_noise']:
    process_tier(tier)
