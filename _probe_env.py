"""
_probe_env.py — does this interpreter actually work for RVC inference?

Exit code 0 = usable, 1 = not usable.
Prints one line per check so the batch file can show it.

Usage:
    python _probe_env.py --applio "C:\\...\\ApplioV3.7.1" --engine "C:\\...\\Engine"
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--applio", default="")
    ap.add_argument("--engine", default="")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if a.engine and os.path.isdir(a.engine):
        os.chdir(a.engine)
    paths = [p for p in (a.applio, a.engine) if p and os.path.isdir(p)]
    for p in reversed(paths):
        if p not in sys.path:
            sys.path.insert(0, p)

    ok = True

    def check(label, fn):
        nonlocal ok
        try:
            msg = fn()
            if not a.quiet:
                print(f"    {label:<16}: OK {msg or ''}")
            return True
        except Exception as e:
            ok = False
            if not a.quiet:
                print(f"    {label:<16}: FAIL {type(e).__name__}: {str(e)[:90]}")
            return False

    def numpy_c():
        import numpy
        return numpy.__version__

    def scipy_c():
        import scipy
        import scipy.sparse  # the module that touches np.ulong
        return scipy.__version__

    def torch_c():
        import torch
        return f"{torch.__version__} cuda={torch.cuda.is_available()}"

    def rvc_c():
        from rvc.infer.infer import VoiceConverter  # noqa: F401
        return "VoiceConverter"

    check("numpy", numpy_c)
    check("scipy.sparse", scipy_c)
    check("torch", torch_c)
    check("rvc", rvc_c)

    if not a.quiet:
        print(f"    {'RESULT':<16}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
