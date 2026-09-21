"""Regenerate every figure, animation, data file, and report.

Figure 7 writes the two-dimensional runs to outputs/cache, which the first
animation reads; Figure 6 writes the stress table that the second
animation reads; Figure 8 caches each resolution of its convergence
study, so an interrupted run resumes where it stopped.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["fig00_schematic.py", "fig01_canopy.py", "fig02_transmission.py",
         "fig03_spall.py", "fig04_bubble.py", "fig05_regimes.py",
         "fig06_diel.py", "fig07_fdtd.py", "fig08_verification.py",
         "anim01_diptych.py", "anim02_diel.py", "anim03_bubbles.py",
         "anim04_reverberation.py", "anim05_thickness.py",
         "make_reports.py"]

t0 = time.time()
for s in ORDER:
    print(f"--- {s}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, s)],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"{s} failed")
print(f"--- done in {time.time() - t0:.1f} s")
