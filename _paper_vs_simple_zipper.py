"""Compare chroma-zipper (paper-exact Python ARI vs simplified current C).

Paper-exact:  _paper_ari_ref.py demosaic_full()  (Python 11-iter 16-GF ARI)
Simplified :  test_ari.exe  with q2 (our current implementation)

Metric: chroma-zipper = mean(|Laplacian(dR)|+|Laplacian(dB)|) on edge mask
        where dR = (R-G)_pred - (R-G)_gt, similarly dB

If paper-exact is NOT substantially better on zipper than simplified,
the 800-line C port delivers ~0 user-visible gain.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from colour_demosaicing import mosaicing_CFA_Bayer
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from _paper_ari_ref import demosaic_full  # noqa

HERE = Path(__file__).parent
KD = Path(r'C:/Users/dregsist/datasets/kodak')
BORDER = 32


def edge_mask(rgb, q=0.85):
    lum = 0.299*rgb[:, :, 0] + 0.587*rgb[:, :, 1] + 0.114*rgb[:, :, 2]
    sx = ndimage.sobel(lum, axis=1); sy = ndimage.sobel(lum, axis=0)
    m = np.sqrt(sx*sx + sy*sy)
    return m > np.quantile(m, q)


def chroma_zip(gt, pred, mask, border=BORDER):
    g = gt.astype(np.float64); p = pred.astype(np.float64)
    dR = (p[:, :, 0]-p[:, :, 1]) - (g[:, :, 0]-g[:, :, 1])
    dB = (p[:, :, 2]-p[:, :, 1]) - (g[:, :, 2]-g[:, :, 1])
    zmag = np.abs(ndimage.laplace(dR)) + np.abs(ndimage.laplace(dB))
    h, w = gt.shape[:2]
    m = mask[border:h-border, border:w-border]
    v = zmag[border:h-border, border:w-border][m]
    return float(np.mean(v)) if v.size else float('nan'), float(ndimage.uniform_filter(zmag, 64).max())


def cpsnr(gt, pred, border=12):
    h, w = gt.shape[:2]
    g = gt[border:h-border, border:w-border].astype(np.float64)
    p = pred[border:h-border, border:w-border].astype(np.float64)
    mse = np.mean((g - p) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')


def run_simplified_c(img, tmp_dir):
    """Run the current C test_ari.exe q2 implementation."""
    h, w = img.shape[:2]
    cfa = mosaicing_CFA_Bayer(img, 'RGGB').astype(np.float32)
    in_p = tmp_dir / 'in.bin';  in_p.write_bytes(np.ascontiguousarray(cfa).tobytes())
    out_p = tmp_dir / 'out.bin'
    subprocess.run(['./test_ari.exe', str(w), str(h), '0x94949494',
                    str(in_p), str(out_p), '2'], capture_output=True)
    return np.frombuffer(out_p.read_bytes(), dtype=np.float32).reshape(h, w, 4)[:, :, :3]


# Edge-heavy Kodak subset (same as bench_zipper_all.py)
EDGE_IMAGES = ['kodim01.png', 'kodim08.png', 'kodim13.png',
               'kodim19.png', 'kodim20.png']

tmp_dir = HERE / 'results' / 'tmp_phase1'
tmp_dir.mkdir(parents=True, exist_ok=True)

print(f'{"image":<13} {"paper":<18}  {"simple":<18}  {"Δcz mean":>10} {"Δcz peak":>10}  {"Δcpsnr":>9}')
print('-' * 95)
rows = []
for fn in EDGE_IMAGES:
    img = iio.imread(str(KD / fn)).astype(np.float32) / 255.0
    img = img[:, :, :3]
    h, w = img.shape[:2]
    h &= ~1; w &= ~1
    img = img[:h, :w]

    pred_paper = demosaic_full(img, 'rggb')
    pred_simpl = run_simplified_c(img, tmp_dir)

    # Clip to shapes
    ph, pw = pred_paper.shape[:2]
    gt_paper = img[:ph, :pw]
    gt_simpl = img

    m_paper = edge_mask(gt_paper)
    m_simpl = edge_mask(gt_simpl)

    cz_p_m, cz_p_pk = chroma_zip(gt_paper, pred_paper, m_paper)
    cz_s_m, cz_s_pk = chroma_zip(gt_simpl, pred_simpl, m_simpl)
    cp_p = cpsnr(gt_paper, pred_paper)
    cp_s = cpsnr(gt_simpl, pred_simpl)

    print(f'{fn:<13} cz={cz_p_m:.4f}/pk{cz_p_pk:.3f} cp{cp_p:5.2f}  '
          f'cz={cz_s_m:.4f}/pk{cz_s_pk:.3f} cp{cp_s:5.2f}  '
          f'{(cz_p_m-cz_s_m)*1000:>+10.4f} {(cz_p_pk-cz_s_pk)*1000:>+10.3f}  '
          f'{cp_p-cp_s:>+9.3f}')
    rows.append((cz_p_m, cz_p_pk, cp_p, cz_s_m, cz_s_pk, cp_s))

arr = np.array(rows)
print('-' * 95)
print(f'{"AVG":<13} cz={arr[:,0].mean():.4f}/pk{arr[:,1].mean():.3f} cp{arr[:,2].mean():5.2f}  '
      f'cz={arr[:,3].mean():.4f}/pk{arr[:,4].mean():.3f} cp{arr[:,5].mean():5.2f}  '
      f'{(arr[:,0].mean()-arr[:,3].mean())*1000:>+10.4f} {(arr[:,1].mean()-arr[:,4].mean())*1000:>+10.3f}  '
      f'{arr[:,2].mean()-arr[:,5].mean():>+9.3f}')

print()
print('Positive Δ = paper worse than simple; negative Δ = paper better.')
print('Units: Δcz mean/peak x1000, Δcpsnr in dB.')
