"""
Zipper artifact check for all darktable demosaic methods.

For each strong-edge Kodak image:
  - edge-CPSNR (zipper-sensitive CPSNR on top-15% Sobel mask)
  - zipper score (mean |laplacian(pred - gt)| on edge mask, lower = better)
  - side-by-side crop PNG (GT + every algorithm)
"""
import sys
import subprocess
import argparse
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from colour_demosaicing import mosaicing_CFA_Bayer
from scipy import ndimage

PATTERNS = {'RGGB': 0x94949494, 'GRBG': 0x61616161,
            'GBRG': 0x49494949, 'BGGR': 0x16161616}

BORDER = 12
HERE = Path(__file__).parent
EXE = {
    'menon': HERE / 'test_menon.exe',
    'rcd':   HERE / 'test_rcd.exe',
    'ppg':   HERE / 'test_ppg.exe',
    'vng':   HERE / 'test_vng.exe',
    'lmmse': HERE / 'test_lmmse.exe',
    'ari':   HERE / 'test_ari.exe',
}
if (HERE / 'test_amaze.exe').exists():
    EXE['amaze'] = HERE / 'test_amaze.exe'

ALGOS = [
    ('menon_r1', 'menon', ['1']),
    ('menon_r0', 'menon', ['0']),
    ('rcd',      'rcd',   []),
    ('ppg',      'ppg',   []),
    ('vng',      'vng',   []),
    ('lmmse_1',  'lmmse', ['1']),
    ('lmmse_3',  'lmmse', ['3']),
    ('ari_q2',   'ari',   ['2']),
    ('ari_q3',   'ari',   ['3']),
]
if 'amaze' in EXE:
    ALGOS.append(('amaze', 'amaze', []))

EDGE_IMAGES = ['kodim01.png', 'kodim08.png', 'kodim13.png',
               'kodim19.png', 'kodim20.png']
KODAK_DIR = Path('C:/Users/dregsist/datasets/kodak')


def sobel_mag(g):
    sx = ndimage.sobel(g, axis=1); sy = ndimage.sobel(g, axis=0)
    return np.sqrt(sx*sx + sy*sy)


def run_algo(exe_name, w, h, fval, cfa, extra, tmp, tag):
    in_p = tmp / f'{tag}_in.bin'; out_p = tmp / f'{tag}_out.bin'
    in_p.write_bytes(np.ascontiguousarray(cfa, dtype=np.float32).tobytes())
    cmd = [str(EXE[exe_name]), str(w), str(h), f'0x{fval:08X}', str(in_p), str(out_p)] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0: return None
    return np.frombuffer(out_p.read_bytes(), dtype=np.float32).reshape(h, w, 4)[:, :, :3]


def edge_cpsnr(gt, pred, mask, border=BORDER):
    h, w = gt.shape[:2]
    g = gt  [border:h-border, border:w-border]
    p = pred[border:h-border, border:w-border]
    m = mask[border:h-border, border:w-border]
    d = (g.astype(np.float64) - p.astype(np.float64))**2
    dm = d[m]
    if dm.size == 0: return float('nan')
    mse = np.mean(dm)
    return 10*np.log10(1/mse) if mse > 0 else float('inf')


def zipper_score(gt, pred, mask, border=BORDER):
    h, w = gt.shape[:2]
    d = pred.astype(np.float64) - gt.astype(np.float64)
    lap = np.zeros_like(d)
    for c in range(3): lap[:,:,c] = ndimage.laplace(d[:,:,c])
    mag = np.mean(np.abs(lap), axis=-1)
    m = mask[border:h-border, border:w-border]
    v = mag[border:h-border, border:w-border][m]
    return float(np.mean(v)) if v.size else float('nan')


def find_edge_crops(emag, crop_size=128, n_crops=2, border=40):
    h, w = emag.shape
    ii = np.cumsum(np.cumsum(emag, axis=0), axis=1)
    def wsum(y,x,s):
        return (ii[y+s-1,x+s-1]-ii[y-1,x+s-1]-ii[y+s-1,x-1]+ii[y-1,x-1]
                if y>0 and x>0 else ii[y+s-1,x+s-1])
    cands = []
    for y in range(border, h - crop_size - border, 16):
        for x in range(border, w - crop_size - border, 16):
            cands.append((wsum(y,x,crop_size), y, x))
    cands.sort(reverse=True)
    sel = []
    for s,y,x in cands:
        if all(abs(y-sy)>=crop_size or abs(x-sx)>=crop_size for _,sy,sx in sel):
            sel.append((s,y,x))
        if len(sel) >= n_crops: break
    return [(y,x) for _,y,x in sel]


def up_nn(img, f=4):
    return np.repeat(np.repeat(img, f, axis=0), f, axis=1)


def save_strip(path, imgs, upscale=4, pad=8):
    ups = [up_nn(np.clip(im,0,1), upscale) for im in imgs]
    H, W = ups[0].shape[:2]
    sep = np.ones((H, pad, 3), dtype=np.float32) * 0.5
    parts = []
    for i, u in enumerate(ups):
        parts.append(u)
        if i < len(ups)-1: parts.append(sep)
    comb = np.concatenate(parts, axis=1)
    iio.imwrite(str(path), (comb*255).astype(np.uint8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='results/zipper_all')
    p.add_argument('--pattern', default='RGGB', choices=list(PATTERNS.keys()))
    a = p.parse_args()

    for k, ex in EXE.items():
        if not ex.exists():
            print(f'ERROR: {ex} not found'); sys.exit(1)

    out = Path(a.out); tmp = out / 'tmp'
    out.mkdir(parents=True, exist_ok=True); tmp.mkdir(exist_ok=True)
    fval = PATTERNS[a.pattern]

    names = [x[0] for x in ALGOS]
    print(f'Zipper check -- pattern={a.pattern}')
    hdr = f'{"Image":<12} {"Algo":<10} {"Edge-CPSNR":>12} {"Zipper":>10}'
    print(hdr); print('-'*len(hdr))

    rows = []
    for fname in EDGE_IMAGES:
        fp = KODAK_DIR / fname
        if not fp.exists(): fp = HERE / fname
        if not fp.exists(): print(f'{fname:<12}  MISSING'); continue
        img = iio.imread(str(fp)).astype(np.float32)/255.0
        if img.ndim == 2: img = np.stack([img]*3, axis=-1)
        if img.shape[2] > 3: img = img[:,:,:3]
        h, w = img.shape[:2]
        lum = 0.299*img[:,:,0]+0.587*img[:,:,1]+0.114*img[:,:,2]
        emag = sobel_mag(lum)
        thresh = np.quantile(emag, 0.85)
        mask = emag > thresh
        cfa = mosaicing_CFA_Bayer(img, a.pattern).astype(np.float32)

        outs = {}
        for name, exe_key, extra in ALGOS:
            o = run_algo(exe_key, w, h, fval, cfa, extra, tmp, f'{fname[:-4]}_{name}')
            outs[name] = o
            if o is None: continue
            ep = edge_cpsnr(img, o, mask); zs = zipper_score(img, o, mask)
            print(f'{fname:<12} {name:<10} {ep:>12.2f} {zs:>10.6f}')
            rows.append({'image': fname, 'algo': name, 'edge_cpsnr': ep, 'zipper': zs})
        print()

        # save crop strips
        crops = find_edge_crops(emag, 128, 2)
        for (y, x) in crops:
            imgs = [img[y:y+128, x:x+128]]
            for name, _, _ in ALGOS:
                if outs[name] is not None:
                    imgs.append(outs[name][y:y+128, x:x+128])
            save_strip(out / f'crop_{fname[:-4]}_y{y}_x{x}.png', imgs)

    csv = out / 'zipper_all.csv'
    with open(str(csv), 'w') as f:
        f.write('image,algo,edge_cpsnr_db,zipper_score\n')
        for r in rows:
            f.write(f'{r["image"]},{r["algo"]},{r["edge_cpsnr"]:.4f},{r["zipper"]:.6f}\n')
    print(f'CSV: {csv}')

    print('\nAverage across edge images:')
    print(f'{"Algo":<10} {"Edge-CPSNR":>12} {"Zipper":>10}')
    for name, _, _ in ALGOS:
        sub = [r for r in rows if r['algo'] == name]
        if not sub: continue
        ep = np.mean([r['edge_cpsnr'] for r in sub])
        zs = np.mean([r['zipper'] for r in sub])
        print(f'{name:<10} {ep:>12.2f} {zs:>10.6f}')


if __name__ == '__main__':
    main()
