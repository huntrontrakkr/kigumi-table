"""Convert the exported test systems to flat binary files for the browser test page.

    cd ~/code/kigumi-table && uv run python tools/gpu-test/convert.py [--no-big]

matrices/<name>.mtx + <name>_rhs.npy + <name>_x.npy  ->  data/<name>.{indptr.i32, indices.i32,
data.f64, rhs.f64, xref.f64} (little-endian; rhs and xref column-major, column j at
[j*n, (j+1)*n)), plus data/manifest.json describing every case.

Unless --no-big: also a larger plate (plate_big, ~15k dof, ~90 load cases) built
with kigumi.plate.solve_plate exactly as the model builds its systems, with a
SuperLU reference solution -- the size the solver meets in practice.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MATRICES = HERE / "matrices"
DATA = HERE / "data"


def write_case(name: str, A: sp.csr_matrix, B: np.ndarray, X: np.ndarray, note: str, accept: dict) -> dict:
    A = sp.csr_matrix(A)
    A.sum_duplicates()
    A.sort_indices()
    n = A.shape[0]
    B = B.reshape(n, -1)
    X = X.reshape(n, -1)
    k = B.shape[1]
    files = {
        "indptr": f"{name}.indptr.i32",
        "indices": f"{name}.indices.i32",
        "data": f"{name}.data.f64",
        "rhs": f"{name}.rhs.f64",
        "xref": f"{name}.xref.f64",
    }
    A.indptr.astype("<i4").tofile(DATA / files["indptr"])
    A.indices.astype("<i4").tofile(DATA / files["indices"])
    A.data.astype("<f8").tofile(DATA / files["data"])
    np.asfortranarray(B).ravel(order="F").astype("<f8").tofile(DATA / files["rhs"])
    np.asfortranarray(X).ravel(order="F").astype("<f8").tofile(DATA / files["xref"])
    asym = abs(A - A.T).max() / abs(A).max()
    d = A.diagonal()
    ref_relres = np.linalg.norm(B - A @ X, axis=0) / np.maximum(np.linalg.norm(B, axis=0), 1e-300)
    meta = dict(name=name, n=int(n), nnz=int(A.nnz), k=int(k), files=files, note=note, accept=accept,
                diag_min=float(d.min()), diag_max=float(d.max()), rel_asym=float(asym),
                ref_relres_max=float(ref_relres.max()))
    print(f"  {name}: n={n} nnz={A.nnz} k={k} diag {d.min():.2e}..{d.max():.2e} "
          f"asym {asym:.1e} reference relres {ref_relres.max():.1e}")
    return meta


def exported(name: str, note: str, accept: dict) -> dict:
    A = scipy.io.mmread(MATRICES / f"{name}.mtx").tocsr()
    B = np.load(MATRICES / f"{name}_rhs.npy")
    X = np.load(MATRICES / f"{name}_x.npy")
    return write_case(name, A, B, X, note, accept)


def plate_big(h: float = 15.0) -> dict:
    """A 1200 x 700 mm rounded plate on four capitals, meshed at h, as kigumi.plate builds it."""
    sys.path.insert(0, str(ROOT))
    from shapely.geometry import box
    from kigumi import linalg, plate

    captured = {}

    def capture(K, B):
        captured["K"], captured["B"] = K.tocsr(), np.array(B, dtype=float)
        return np.zeros(B.shape)            # skip the solve here; the reference is made below

    region = box(0.0, 0.0, 1200.0, 700.0).buffer(-90.0).buffer(90.0)
    supports = [(np.array([x, y]), 35.0) for x in (160.0, 1040.0) for y in (150.0, 550.0)]
    linalg.set_accelerator(capture)
    try:
        plate.solve_plate(region, supports, load=1000.0, q=0.002, h=h)
    finally:
        linalg.set_accelerator(None)
    K, B = captured["K"], captured["B"]
    t = time.time()
    lu = linalg.factor(K)
    X = lu.solve(B)
    for _ in range(3):                      # float64 refinement: SuperLU alone stops near 1e-9
        X += lu.solve(B - K @ X)
    print(f"  plate_big reference (SuperLU + 3 refinements) {time.time() - t:.1f} s")
    return write_case("plate_big", K, B, X,
                      f"kigumi.plate.solve_plate on a 1200x700 mm rounded plate, h={h} mm, four capitals",
                      dict(relres=1e-8, err=1e-6))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-big", action="store_true", help="skip the ~15k-dof plate")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    cases = [
        exported("plate_four", "Kirchhoff plate (BFS elements, 4 dof/node) with 1e6x penalty supports",
                 dict(relres=1e-8, err=1e-6)),
        exported("grillage_four", "beams + glass plate + penalty contacts", dict(relres=1e-8, err=1e-6)),
    ]
    if not args.no_big:
        cases.append(plate_big())
    (DATA / "manifest.json").write_text(json.dumps(dict(cases=cases), indent=1))
    print(f"wrote {DATA / 'manifest.json'}")


if __name__ == "__main__":
    main()
