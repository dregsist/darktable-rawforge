"""
Speed benchmark for all darktable demosaic methods.

Measures median wall-clock on synthetic sizes (subprocess overhead subtracted).
Writes results/speed_all.csv.
"""
import sys
import time
import subprocess
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from colour_demosaicing import mosaicing_CFA_Bayer

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
    ('ari_q1',   'ari',   ['1']),
    ('ari_q2',   'ari',   ['2']),
    ('ari_q3',   'ari',   ['3']),
]
if 'amaze' in EXE:
    ALGOS.append(('amaze', 'amaze', []))

KODAK_DIR = Path('C:/Users/dregsist/datasets/kodak')

RUNS = 5
SIZES = [
    ('kodim19', 768,  512),
    ('2K',      2048, 1365),
    ('4K',      4096, 2732),
]


def time_exe(exe, args, runs=RUNS):
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        r = subprocess.run(args, capture_output=True)
        ts.append(time.perf_counter() - t0)
        if r.returncode != 0:
            return None
    return float(np.median(ts))


def main():
    for k, exe in EXE.items():
        if not exe.exists():
            print(f'ERROR: {exe} not found'); sys.exit(1)

    tmp = HERE / 'results' / 'speed_tmp_all'
    tmp.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)

    # Subprocess overhead baseline (16x16 menon)
    tiny = rng.random((16, 16), dtype=np.float32)
    (tmp / 'tiny.bin').write_bytes(np.ascontiguousarray(tiny).tobytes())
    overhead = []
    for _ in range(5):
        t0 = time.perf_counter()
        subprocess.run([str(EXE['menon']), '16','16','0x94949494',
                        str(tmp/'tiny.bin'), str(tmp/'tiny_out.bin'), '1'],
                       capture_output=True)
        overhead.append(time.perf_counter() - t0)
    ov = float(np.median(overhead))
    print(f'subprocess overhead ~ {ov*1000:.1f} ms\n')

    print(f'{"size":<8} {"w x h":<12} {"algo":<10} {"wall_ms":>10} {"algo_ms":>10} {"Mpix/s":>10}')
    print('-' * 64)

    rows = []
    for sname, w, h in SIZES:
        if sname == 'kodim19':
            kpath = KODAK_DIR / 'kodim19.png'
            if not kpath.exists():
                kpath = HERE / 'kodim19.png'
            img = iio.imread(str(kpath)).astype(np.float32)/255.
            h, w = img.shape[:2]
            cfa = mosaicing_CFA_Bayer(img, 'RGGB').astype(np.float32)
        else:
            rgb = rng.random((h, w, 3), dtype=np.float32)
            cfa = mosaicing_CFA_Bayer(rgb, 'RGGB').astype(np.float32)
        mpix = w*h/1e6
        cfa_bytes = np.ascontiguousarray(cfa).tobytes()
        in_p = tmp / f'{sname}_in.bin'
        in_p.write_bytes(cfa_bytes)

        for algo_name, exe_key, extra in ALGOS:
            out_p = tmp / f'{sname}_{algo_name}_out.bin'
            args = [str(EXE[exe_key]), str(w), str(h), '0x94949494',
                    str(in_p), str(out_p)] + extra
            twall = time_exe(exe=None, args=args)
            if twall is None:
                print(f'{sname:<8} {w}x{h:<8} {algo_name:<10}  ERROR')
                continue
            talgo = max(twall - ov, 0.001)
            mps = mpix / talgo
            print(f'{sname:<8} {w}x{h:<8} {algo_name:<10} {twall*1000:>10.1f} {talgo*1000:>10.1f} {mps:>10.2f}')
            rows.append({'size': sname, 'w': w, 'h': h, 'mpix': mpix,
                         'algo': algo_name, 'wall_ms': twall*1000,
                         'algo_ms': talgo*1000, 'mpix_s': mps})
        print()

    # per-algo average Mpix/s
    print('Average Mpix/s (mean across sizes):')
    print(f'{"algo":<10} {"Mpix/s":>10}')
    print('-' * 22)
    for a in ALGOS:
        name = a[0]
        vals = [r['mpix_s'] for r in rows if r['algo'] == name]
        if vals: print(f'{name:<10} {np.mean(vals):>10.2f}')

    csv = HERE / 'results' / 'speed_all.csv'
    with open(str(csv), 'w') as f:
        f.write('size,w,h,mpix,algo,wall_ms,algo_ms,mpix_s\n')
        for r in rows:
            f.write(f'{r["size"]},{r["w"]},{r["h"]},{r["mpix"]:.4f},'
                    f'{r["algo"]},{r["wall_ms"]:.2f},{r["algo_ms"]:.2f},{r["mpix_s"]:.4f}\n')
    print(f'\nCSV: {csv}')


if __name__ == '__main__':
    main()
