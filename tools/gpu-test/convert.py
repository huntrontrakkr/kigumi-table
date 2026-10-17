"""Convert the exported test systems to flat binary files for the browser test page.

    cd ~/code/kigumi-table && uv run python tools/gpu-test/convert.py [--no-big] [--export]

matrices/<name>.mtx + <name>_rhs.npy + <name>_x.npy  ->  data/<name>.{indptr.i32, indices.i32,
data.f64, rhs.f64, xref.f64} (little-endian; rhs and xref column-major, column j at
[j*n, (j+1)*n)), plus data/manifest.json describing every case.

matrices/ is not in git.  When its files are missing (or with --export, which
rewrites them) they are exported from the model: the 'field-four' preset's top
plate (the first 8 of its load cases) and its grillage with every contact active
(the dead load), with SuperLU references.  The matrices are those of the original
export; the hand-exported originals used other right-hand sides.

plate_model: the same 'field-four' plate with all 123 load cases, as the app solves it.

Unless --no-big: also a larger plate (plate_big, ~15k dof, 90 load cases) built
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


def reference(K, B) -> np.ndarray:
    """SuperLU solution polished by float64 refinement (SuperLU alone stops near 1e-9 here)."""
    from kigumi import linalg
    lu = linalg.factor(K)
    X = lu.solve(B)
    for _ in range(3):
        X += lu.solve(B - K @ X)
    return X


def plate_model() -> dict:
    """data/plate_model.*: the 'field-four' top plate with all its load cases, exactly the
    system kigumi.linalg.solve_many hands the GPU in the app (k = 123)."""
    sys.path.insert(0, str(ROOT))
    from kigumi import linalg
    from kigumi.build import build
    from kigumi.params import preset

    plate = []
    linalg.set_accelerator(lambda K, B: plate.append((K.tocsr(), np.array(B, dtype=float))) and None)
    try:
        build(preset("field-four"))
    finally:
        linalg.set_accelerator(None)
    K, B = plate[0]
    return write_case("plate_model", K, B, reference(K, B),
                      "the 'field-four' top plate with all its load cases, as solve_many passes it in the app",
                      dict(relres=1e-8, err=1e-6))


def export_model() -> None:
    """matrices/plate_four.* and matrices/grillage_four.* from the 'field-four' preset."""
    sys.path.insert(0, str(ROOT))
    from kigumi import linalg
    from kigumi.build import build
    from kigumi.checks import Grillage
    from kigumi.params import preset

    plate = []
    linalg.set_accelerator(lambda K, B: plate.append((K.tocsr(), np.array(B, dtype=float))) and None)
    try:
        d = build(preset("field-four"))
    finally:
        linalg.set_accelerator(None)
    K, B = plate[0]
    B = B[:, :8].copy()
    save_npz_case("plate_four", K, B, reference(K, B))

    # the grillage's first system: every contact active, under its dead load
    seen = {}
    factor0 = linalg.factor

    class Spy:
        def __init__(self, K):
            self.K, self.lu = K, factor0(K)

        def solve(self, b):
            seen.setdefault("system", (self.K, np.array(b, dtype=float)))
            return self.lu.solve(b)

    linalg.factor = lambda K: Spy(K.tocsr())
    try:
        g = Grillage(d)
        g.solve(g.f_self)
    finally:
        linalg.factor = factor0
    K, b = seen["system"]
    B = b.reshape(-1, 1)
    save_npz_case("grillage_four", K, B, reference(K, B))


def save_npz_case(name: str, K, B: np.ndarray, X: np.ndarray) -> None:
    MATRICES.mkdir(exist_ok=True)
    scipy.io.mmwrite(MATRICES / f"{name}.mtx", sp.coo_matrix(K), precision=17)
    np.save(MATRICES / f"{name}_rhs.npy", B)
    np.save(MATRICES / f"{name}_x.npy", X)
    print(f"  exported matrices/{name}: n={K.shape[0]} nnz={K.nnz} k={B.shape[1]}")


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
    X = reference(K, B)
    print(f"  plate_big reference (SuperLU + 3 refinements) {time.time() - t:.1f} s")
    return write_case("plate_big", K, B, X,
                      f"kigumi.plate.solve_plate on a 1200x700 mm rounded plate, h={h} mm, four capitals",
                      dict(relres=1e-8, err=1e-6))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-big", action="store_true", help="skip the ~15k-dof plate")
    ap.add_argument("--export", action="store_true", help="re-export matrices/ from the model")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    need = [MATRICES / f"{nm}{suffix}" for nm in ("plate_four", "grillage_four")
            for suffix in (".mtx", "_rhs.npy", "_x.npy")]
    if args.export or not all(f.exists() for f in need):
        export_model()
    cases = [
        exported("plate_four", "Kirchhoff plate (BFS elements, 4 dof/node) with 1e6x penalty supports",
                 dict(relres=1e-8, err=1e-6)),
        exported("grillage_four", "beams + glass plate + penalty contacts", dict(relres=1e-8, err=1e-6)),
    ]
    cases.append(plate_model())
    if not args.no_big:
        cases.append(plate_big())
    (DATA / "manifest.json").write_text(json.dumps(dict(cases=cases), indent=1))
    print(f"wrote {DATA / 'manifest.json'}")


if __name__ == "__main__":
    main()
