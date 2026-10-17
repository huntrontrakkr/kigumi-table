"""Call Blender to render a design (see render_blender.py)."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from .build import Design
from .export import export_npz

BLENDER = os.environ.get("KIGUMI_BLENDER", "blender")
SCRIPT = Path(__file__).with_name("render_blender.py")


def render(d: Design, out: Path, view: str = "hero", samples: int = 96, res: str = "1600x1100",
           explode: float = 0.0, extra: list[str] | None = None, timeout: int = 900) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        npz = export_npz(d, Path(td) / "d.npz", explode=explode)
        env = dict(os.environ)
        # WSL: Blender must load the WSL libcuda shim, not the native driver library
        if Path("/usr/lib/wsl/lib/libcuda.so").exists():
            env["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + env.get("LD_LIBRARY_PATH", "")
        cmd = [BLENDER, "-b", "--factory-startup", "--python", str(SCRIPT), "--",
               "--npz", str(npz), "--out", str(out.resolve()), "--view", view,
               "--samples", str(samples), "--res", res] + (extra or [])
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0 or not out.exists():
            raise RuntimeError("blender failed:\n" + r.stdout[-3000:] + r.stderr[-3000:])
    return out
