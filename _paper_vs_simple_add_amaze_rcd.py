"""Add AMaZE and RCD outputs to viewer_data_paper_vs_simple/iso_low/
so the user can 4-way compare: gt / simple / paper / amaze / rcd.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from colour_demosaicing import mosaicing_CFA_Bayer

HERE = Path(__file__).parent
KD = Path(r'C:/Users/dregsist/datasets/kodak')
OUT = HERE / 'viewer_data_paper_vs_simple' / 'iso_low'

EDGE_IMAGES = ['kodim01.png', 'kodim08.png', 'kodim13.png',
               'kodim19.png', 'kodim20.png']


def save_png(arr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
    iio.imwrite(str(path), u8)


def run_exe(exe, img, tmp_dir, extra_args=()):
    h, w = img.shape[:2]
    cfa = mosaicing_CFA_Bayer(img, 'RGGB').astype(np.float32)
    in_p = tmp_dir / 'in.bin';  in_p.write_bytes(np.ascontiguousarray(cfa).tobytes())
    out_p = tmp_dir / 'out.bin'
    cmd = [exe, str(w), str(h), '0x94949494', str(in_p), str(out_p), *extra_args]
    subprocess.run(cmd, capture_output=True)
    return np.frombuffer(out_p.read_bytes(), dtype=np.float32).reshape(h, w, 4)[:, :, :3]


tmp = HERE / 'results' / 'tmp_phase1'
tmp.mkdir(parents=True, exist_ok=True)

for fn in EDGE_IMAGES:
    img = iio.imread(str(KD / fn)).astype(np.float32) / 255.0
    img = img[:, :, :3]
    h, w = img.shape[:2]
    h &= ~1; w &= ~1
    img = img[:h, :w]

    pred_amaze = run_exe('./test_amaze.exe', img, tmp)
    pred_rcd   = run_exe('./test_rcd.exe',   img, tmp)

    save_png(pred_amaze, OUT / 'amaze' / fn)
    save_png(pred_rcd,   OUT / 'rcd'   / fn)
    print(f'  {fn}: amaze + rcd saved')

print(f'\nOutput: {OUT}')
print('Now viewer_data_paper_vs_simple has: __gt__ / simple / paper / amaze / rcd')
