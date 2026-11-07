"""
install_shim.py — installs a permanent NumPy compatibility shim into a venv.

Why: SciPy's `sparse/_sputils.py` reads `np.ulong` at import time. NumPy 1.26
does not define it -> AttributeError -> `import scipy` fails -> `import rvc`
fails -> "Could not import the Applio/RVC voice converter class".

This script writes two files into the env's site-packages:

  _si_numpy_compat.py          the alias layer (np.ulong -> np.uint64, ...)
  zz_si_numpy_compat.pth       a startup hook: "import _si_numpy_compat"

Because .pth files are processed by site.py at interpreter startup, the aliases
exist BEFORE anything imports scipy. NumPy itself is NOT modified on disk, so
torch's ABI expectations are untouched.

Usage:
    python install_shim.py --env "C:\\...\\ApplioV3.7.1\\env"
    python install_shim.py --env "..." --remove
"""
import argparse
import subprocess
import sys
from pathlib import Path

SHIM_NAME = "_si_numpy_compat.py"
PTH_NAME = "zz_si_numpy_compat.pth"

SHIM_CODE = '''"""NumPy 1.x compatibility aliases required by SciPy (np.ulong etc.).

Auto-installed by install_shim.py. Remove with: install_shim.py --remove

Design note: at .pth execution time numpy is often NOT yet importable
(sys.path is still being built). So we do NOT import numpy here. Instead we
register a meta-path hook that patches numpy the moment scipy (or numba/rvc)
is about to be imported. By then numpy is guaranteed importable.
"""
import sys as _sys
import warnings as _warnings

_ALIASES = {
    "ulong":     ("uint64", "uint32"),
    "long":      ("int64", "int32"),
    "ulonglong": ("uint64",),
    "longlong":  ("int64",),
    "uintc":     ("uint32", "uint16"),
    "intc":      ("int32", "int16"),
    "uint":      ("uint32", "uint64"),
    "int_":      ("int64", "int32"),
    "float_":    ("float64",),
    "complex_":  ("complex128",),
}

_PATCHED = False


def _get(mod, name):
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        try:
            return getattr(mod, name)
        except AttributeError:
            return None


def apply():
    """Add the missing NumPy 1.x aliases. Returns list of names added."""
    global _PATCHED
    try:
        import numpy as np
    except Exception:
        return []
    added = []
    for name, targets in _ALIASES.items():
        if _get(np, name) is not None:
            continue
        for t in targets:
            val = _get(np, t)
            if val is not None:
                try:
                    setattr(np, name, val)
                    added.append(name)
                    break
                except Exception:
                    pass
    if added:
        _PATCHED = True
    return added


class _PatchFinder:
    """Patch numpy just before scipy/numba/rvc are imported."""

    _targets = ("scipy", "numba", "rvc", "librosa")

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in self._targets:
            apply()
        return None  # never intercept - let the normal import continue


def _install():
    if any(isinstance(f, _PatchFinder) for f in _sys.meta_path):
        return
    _sys.meta_path.insert(0, _PatchFinder())
    # best-effort immediate patch (works when numpy is already importable)
    apply()


_install()
'''


def site_packages(env: Path) -> Path:
    py = env / "Scripts" / "python.exe"
    if not py.exists():
        py = env / "bin" / "python"
    if not py.exists():
        raise SystemExit(f"[X] no python in {env}")
    out = subprocess.run(
        [str(py), "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True)
    p = out.stdout.strip()
    if not p or not Path(p).is_dir():
        raise SystemExit(f"[X] could not locate site-packages: {out.stderr[:300]}")
    return Path(p)


def verify(env: Path) -> bool:
    py = env / "Scripts" / "python.exe"
    if not py.exists():
        py = env / "bin" / "python"
    code = ("import warnings;warnings.simplefilter('ignore');"
            "import numpy;"
            "print('numpy', numpy.__version__, '| np.ulong', getattr(numpy,'ulong',None));"
            "import scipy.sparse;print('scipy.sparse OK')")
    r = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[-400:])
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--remove", action="store_true")
    a = ap.parse_args()

    env = Path(a.env)
    sp = site_packages(env)
    print(f"env           : {env}")
    print(f"site-packages : {sp}")

    shim = sp / SHIM_NAME
    pth = sp / PTH_NAME

    if a.remove:
        for f in (shim, pth):
            if f.exists():
                f.unlink()
                print(f"removed {f.name}")
        print("\nShim removed.")
        return

    shim.write_text(SHIM_CODE, encoding="utf-8")
    pth.write_text(f"import {SHIM_NAME[:-3]}\n", encoding="utf-8")
    print(f"wrote {shim.name}")
    print(f"wrote {pth.name}")
    print("\n=== verify ===")
    ok = verify(env)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
