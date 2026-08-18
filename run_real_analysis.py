#!/usr/bin/env python3
"""
comprises of all measurements in the report, computed on the real region bounds and
starting point from one detection.

Writes patterns_real.png and coverage_real.png, and prints the tables.

To run:
python3 run_real_analysis.py

Written with the help of Claude Opus 5, Anthropic
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from calculate_pattern import (calculate_spiral, calculate_circle, calculate_zigzag, calculate_figure_eight)
from compute_coverage import coverage
from compute_smoothness import smoothness_metrics

# IMPORT VALUES FROM REAL CAPTURES
START = np.array([-0.04317604702637159, 0.018008358745762322, 0.1])
BOUNDS = np.array([
    [0.10130116985996741,  0.1834433533709962,  0.05178152079283693],
    [0.09489065212402058, -0.15408180042334194, 0.03024450504275711],
    [-0.17810309606243288, 0.18571343094513404, 0.06539509440812385],
    [-0.1846769313093502, -0.147858713806498,   0.04536164774311002],
])
TOOL_RADIUS = 0.0297    # half of the 59.4 mm tool diameter
CELL = 0.005
MARGIN = {"Spiral": 0.0, "Concentric circles": 0.0, "Zigzag": 0.020, "Figure eight": 0.0}   # safety settings for each 
V_MAX = 0.345                 # m/s
A_MAX = 1.9                   # m/s^2
T_CTRL = 0.085                # s
PATTERNS = [
    ("Spiral",             lambda **k: calculate_spiral(START, BOUNDS, **{"margin": MARGIN["Spiral"], **k})),
    ("Concentric circles", lambda **k: calculate_circle(START, BOUNDS, **{"margin": MARGIN["Concentric circles"], **k})),
    ("Zigzag",             lambda **k: calculate_zigzag(START, BOUNDS, **{"margin": MARGIN["Zigzag"], **k})),
    ("Figure eight",       lambda **k: calculate_figure_eight(START, BOUNDS, **{"margin": MARGIN["Figure eight"], **k})),
]


def describe_region():
    xy = BOUNDS[:, :2]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    ext = hi - lo
    mid = (lo + hi) / 2
    off = START[:2] - mid
    print("REGION")
    print(f"  bounding box   {ext[0]*1000:.0f} x {ext[1]*1000:.0f} mm "
          f"= {ext[0]*ext[1]*1e4:.0f} cm²")
    print(f"  x {lo[0]:+.4f} to {hi[0]:+.4f}    y {lo[1]:+.4f} to {hi[1]:+.4f}")
    print(f"  corner z spans {BOUNDS[:,2].min():.4f} to {BOUNDS[:,2].max():.4f} m "
          f"({(BOUNDS[:,2].max()-BOUNDS[:,2].min())*1000:.0f} mm tilt across the region)")
    print(f"  pattern runs at z = {START[2]:.3f} m, i.e. "
          f"{(START[2]-BOUNDS[:,2].max())*1000:.0f} mm above the highest corner")
    print(f"  start offset from box centre: "
          f"{off[0]*1000:+.1f}, {off[1]*1000:+.1f} mm "
          f"({np.linalg.norm(off)/np.linalg.norm(ext/2)*100:.1f}% of the half-extent)")
    print("  -> for this detection the centroid sits almost exactly at the box centre\n")
    return lo, hi


def main():
    lo, hi = describe_region()

    print(f"PATTERNS  (tool half-width {TOOL_RADIUS*1000:.0f} mm; load figures assume "
              f"v_max {V_MAX} m/s, a_max {A_MAX} m/s^2)\n")
    print(f"{'pattern':20s} {'wpts':>5s} {'path m':>7s} {'cov %':>7s} "
          f"{'mean curv':>10s} {'corners':>8s} {'brake-acc':>10s} {'spd var':>8s} "
          f"{'time s':>7s}")
    results = {}
    for name, fn in PATTERNS:
        p = fn()
        cov, _ = coverage(p, bounds=BOUNDS, tool_radius=TOOL_RADIUS, cell=CELL)
        m = smoothness_metrics(p, v_max=V_MAX, a_max=A_MAX, t_ctrl=T_CTRL)
        results[name] = (p, cov, m)
        print(f"{name:20s} {len(p):5d} {m['path_length_m']:7.2f} {cov:7.1f} "
              f"{m['mean_curvature']:10.2f} {m['sharp_corners']:8d} "
              f"{m['brake_accel_cycles']:10d} {m['speed_variation']:8.2f} "
              f"{m['time']:7.2f}")

    print("\nMARGIN SWEEP — coverage %, and the gap this leaves to the boundary\n")
    print(f"{'margin mm':>10s} " + "".join(f"{n:>20s}" for n, _ in PATTERNS))
    for mm in [0, 5, 10, 15, 20, 25, 30]:
        row = []
        for name, fn in PATTERNS:
            try:
                p = fn(margin=mm / 1000.0)
                row.append(f"{coverage(p, bounds=BOUNDS, tool_radius=TOOL_RADIUS, cell=CELL)[0]:20.1f}")
            except Exception:
                row.append(f"{'infeasible':>20s}")
        print(f"{mm:10d} " + "".join(row))

    print("\nSTARTING-POINT SENSITIVITY — coverage % over 25 starts across the region\n")
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    xs = np.linspace(mid[0] - 0.33 * half[0], mid[0] + 0.33 * half[0], 5)
    ys = np.linspace(mid[1] - 0.33 * half[1], mid[1] + 0.33 * half[1], 5)
    print(f"{'pattern':20s} {'min':>7s} {'mean':>7s} {'max':>7s} {'spread':>8s}")
    for name, fn in PATTERNS:
        vals = []
        for x in xs:
            for y in ys:
                try:
                    fn2 = {"Spiral": calculate_spiral, "Concentric circles": calculate_circle,
                           "Zigzag": calculate_zigzag, "Figure eight": calculate_figure_eight}[name]
                    p = fn2(np.array([x, y, START[2]]), BOUNDS)
                    vals.append(coverage(p, bounds=BOUNDS, tool_radius=TOOL_RADIUS, cell=CELL)[0])
                except Exception:
                    pass
        v = np.array(vals)
        print(f"{name:20s} {v.min():7.1f} {v.mean():7.1f} {v.max():7.1f} "
              f"{np.ptp(v):8.1f}")

    for fname, mode in [("patterns_real.png", "path"), ("coverage_real.png", "cov")]:
        fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.0))
        for ax, (name, _) in zip(axes.ravel(), PATTERNS):
            p, cov, _ = results[name]
            if mode == "cov":
                _, mask = coverage(p, bounds=BOUNDS, tool_radius=TOOL_RADIUS, cell=CELL)
                ax.imshow(mask.T, 
                          origin="lower", 
                          extent=[lo[0], hi[0], lo[1], hi[1]],
                          cmap=ListedColormap(["#f0f0f0", "#4292c6"]), 
                          vmin=0, vmax=1,
                          interpolation="nearest")
                title = f"{name} — {cov:.1f}% covered"
            else:
                title = f"{name} — {len(p)} waypoints"
            ax.plot(p[:, 0], p[:, 1], "-", color="#08306b", lw=0.85)
            if mode == "path":
                ax.plot(p[:, 0], p[:, 1], ".", color="#08306b", ms=2)
            ax.plot(START[0], START[1], "o", mfc="none", mec="#d62728", ms=7, mew=1.3)
            ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, ec="#333", lw=1.2))
            ax.set_xlim(lo[0] - .02, hi[0] + .02); ax.set_ylim(lo[1] - .02, hi[1] + .02)
            ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
            ax.set_xlabel("x (m)", fontsize=8); ax.set_ylabel("y (m)", fontsize=8)
            ax.tick_params(labelsize=7); ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"\nwrote {fname}")


if __name__ == "__main__":
    main()
