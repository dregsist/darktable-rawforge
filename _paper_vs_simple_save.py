"""Render paper-exact (Python) and current-C-simplified outputs side by side
as PNGs in viewer_data_paper_vs_simple/ so the user can open them in
viewer.html and inspect chroma zipper differences directly.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from colour_demosaicing import mosaicing_CFA_Bayer

sys.path.insert(0, str(Path(__file__).parent))
from _paper_ari_ref import demosaic_full  # noqa

HERE = Path(__file__).parent
KD = Path(r'C:/Users/dregsist/datasets/kodak')
OUT = HERE / 'viewer_data_paper_vs_simple' / 'iso_low'

EDGE_IMAGES = ['kodim01.png', 'kodim08.png', 'kodim13.png',
               'kodim19.png', 'kodim20.png']


def save_png(arr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
    iio.imwrite(str(path), u8)


def run_simplified_c(img, tmp_dir):
    h, w = img.shape[:2]
    cfa = mosaicing_CFA_Bayer(img, 'RGGB').astype(np.float32)
    in_p = tmp_dir / 'in.bin';  in_p.write_bytes(np.ascontiguousarray(cfa).tobytes())
    out_p = tmp_dir / 'out.bin'
    subprocess.run(['./test_ari.exe', str(w), str(h), '0x94949494',
                    str(in_p), str(out_p), '2'], capture_output=True)
    return np.frombuffer(out_p.read_bytes(), dtype=np.float32).reshape(h, w, 4)[:, :, :3]


tmp = HERE / 'results' / 'tmp_phase1'
tmp.mkdir(parents=True, exist_ok=True)

for fn in EDGE_IMAGES:
    img = iio.imread(str(KD / fn)).astype(np.float32) / 255.0
    img = img[:, :, :3]
    h, w = img.shape[:2]
    h &= ~1; w &= ~1
    img = img[:h, :w]

    # Simple (C) — original full res
    pred_simpl = run_simplified_c(img, tmp)
    # Paper (Python, 11 iter) — same size (we already cropped img to even)
    pred_paper = demosaic_full(img, 'rggb')

    save_png(img,        OUT / '__gt__'    / fn)
    save_png(pred_simpl, OUT / 'c_port_q2' / fn)
    save_png(pred_paper, OUT / 'py_ref'    / fn)
    print(f'  {fn}: saved gt + c_port_q2 + py_ref')

print(f'\nOutput: {OUT}')
print('Open viewer.html on viewer_data_paper_vs_simple/ and 2-split '
      'paper vs simple to compare.')
