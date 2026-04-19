"""Add menon_r0 and menon_r1 outputs to viewer_data_paper_vs_simple/iso_low/."""
import subprocess
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

    pred_m0 = run_exe('./test_menon.exe', img, tmp, ['0'])
    pred_m1 = run_exe('./test_menon.exe', img, tmp, ['1'])

    save_png(pred_m0, OUT / 'menon_r0' / fn)
    save_png(pred_m1, OUT / 'menon_r1' / fn)
    print(f'  {fn}: menon_r0 + menon_r1 saved')

print(f'\nOutput: {OUT}')
print('Directories now: __gt__ / c_port_q2 / py_ref / amaze / rcd / menon_r0 / menon_r1')
