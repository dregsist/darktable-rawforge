"""
Kodak-24 CPSNR benchmark for all darktable demosaic methods:
menon_r0, menon_r1, rcd, ppg, vng, lmmse_{0..4}, ari_q1/q2/q3.

Writes results/cpsnr_all.csv + console summary.
"""
import sys
import subprocess
import argparse
import urllib.request
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from colour_demosaicing import mosaicing_CFA_Bayer

KODAK_BASE = 'http://r0k.us/graphics/kodak/kodak/'
KODAK_IMGS = [f'kodim{i:02d}.png' for i in range(1, 25)]

PATTERNS = {
    'RGGB': 0x94949494,
    'GRBG': 0x61616161,
    'GBRG': 0x49494949,
    'BGGR': 0x16161616,
}

# common safe border for CPSNR eval (max of all methods' borders)
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
    ('lmmse_1',  'lmmse', ['1']),  # median (default darktable)
    ('lmmse_3',  'lmmse', ['3']),  # refine+medians
    ('ari_q1',   'ari',   ['1']),
    ('ari_q2',   'ari',   ['2']),
    ('ari_q3',   'ari',   ['3']),
]
if 'amaze' in EXE:
    ALGOS.append(('amaze', 'amaze', []))


def download_kodak(dest_dir):
    missing = [f for f in KODAK_IMGS if not (dest_dir / f).exists()]
    if not missing:
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    for fname in missing:
        url = KODAK_BASE + fname
        try:
            urllib.request.urlretrieve(url, str(dest_dir / fname))
        except Exception as e:
            print(f'  FAILED {fname}: {e}')


def cpsnr(gt, pred, border=BORDER):
    h, w = gt.shape[:2]
    g = gt  [border:h-border, border:w-border].astype(np.float64)
    p = pred[border:h-border, border:w-border].astype(np.float64)
    mse = np.mean((g - p) ** 2)
    return 10.0 * np.log10(1.0 / mse) if mse > 0 else float('inf')


def run_algo(exe_name, w, h, fval, cfa_bytes, extra_args, tmp_dir, tag):
    exe = EXE[exe_name]
    in_p  = tmp_dir / f'{tag}_in.bin'
    out_p = tmp_dir / f'{tag}_out.bin'
    in_p.write_bytes(cfa_bytes)
    cmd = [str(exe), str(w), str(h), f'0x{fval:08X}', str(in_p), str(out_p)] + extra_args
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f'  ERROR {exe.name}: {r.stderr.strip()}')
        return None
    raw = out_p.read_bytes()
    return np.frombuffer(raw, dtype=np.float32).reshape(h, w, 4)[:, :, :3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results')
    parser.add_argument('--pattern', default='RGGB', choices=list(PATTERNS.keys()))
    parser.add_argument('--kodak-dir', default='C:/Users/dregsist/datasets/kodak')
    args = parser.parse_args()

    for k, exe in EXE.items():
        if not exe.exists():
            print(f'ERROR: {exe} not found — run build.bat first'); sys.exit(1)

    out_dir = Path(args.out)
    tmp_dir = out_dir / 'tmp_all'
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(exist_ok=True)

    kodak_dir = Path(args.kodak_dir)
    download_kodak(kodak_dir)

    fval = PATTERNS[args.pattern]

    rows = []
    names = [a[0] for a in ALGOS]
    hdr = f'{"image":<12}' + ''.join(f'{n:>10}' for n in names)
    print(f'\nKodak-24 CPSNR (dB) -- pattern={args.pattern}, border={BORDER}')
    print(hdr)
    print('-' * len(hdr))

    for fname in KODAK_IMGS:
        fpath = kodak_dir / fname
        if not fpath.exists():
            print(f'{fname:<12}  MISSING'); continue
        img = iio.imread(str(fpath)).astype(np.float32) / 255.0
        if img.ndim == 2: img = np.stack([img]*3, axis=-1)
        if img.shape[2] > 3: img = img[:, :, :3]
        h, w = img.shape[:2]

        cfa = mosaicing_CFA_Bayer(img, args.pattern).astype(np.float32)
        cfa_bytes = np.ascontiguousarray(cfa).tobytes()

        row = {'image': fname}
        line = f'{fname:<12}'
        for name, exe_key, extra in ALGOS:
            out = run_algo(exe_key, w, h, fval, cfa_bytes, extra, tmp_dir, f'{fname[:-4]}_{name}')
            v = cpsnr(img, out) if out is not None else float('nan')
            row[name] = v
            line += f'{v:>10.2f}'
        print(line)
        rows.append(row)

    print('-' * len(hdr))
    avgs = {}
    for n in names:
        vals = [r[n] for r in rows if not np.isnan(r[n])]
        avgs[n] = float(np.mean(vals)) if vals else float('nan')
    line = f'{"average":<12}' + ''.join(f'{avgs[n]:>10.2f}' for n in names)
    print(line)

    csv = out_dir / 'cpsnr_all.csv'
    with open(str(csv), 'w') as f:
        f.write('image,' + ','.join(names) + '\n')
        for r in rows:
            f.write(r['image'] + ',' + ','.join(f'{r[n]:.4f}' for n in names) + '\n')
        f.write('average,' + ','.join(f'{avgs[n]:.4f}' for n in names) + '\n')
    print(f'\nCSV: {csv}')


if __name__ == '__main__':
    main()
