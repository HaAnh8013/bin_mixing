"""
Measures how much of the bin floor each pattern actually sweeps.

To run:
python3 compute_coverage.py

Written with the help of Claude Opus 5, Anthropic
"""

import numpy as np
from scipy.spatial import cKDTree

from calculate_pattern import (calculate_spiral, calculate_circle, calculate_zigzag, calculate_figure_eight)

# DATA FROM THE REAL CAPTURES
BOUNDS = np.array([
    [0.10130116985996741,  0.1834433533709962,  0.05178152079283693],
    [0.09489065212402058, -0.15408180042334194, 0.03024450504275711],
    [-0.17810309606243288, 0.18571343094513404, 0.06539509440812385],
    [-0.1846769313093502, -0.147858713806498,   0.04536164774311002],
])
START = np.array([-0.04317604702637159, 0.018008358745762322, 0.1])
TOOL_RADIUS = 0.0297    # half of the 59.4 mm mixing tool
CELL = 0.005            # meters -- grid resolution
# ---------------------------------------------------------------------------


def densify(pts, step=0.002):
    """Sample points along the path at ~step spacing"""
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        d = np.linalg.norm(b[:2] - a[:2])
        n = max(2, int(np.ceil(d / step)) + 1)
        out.append(np.linspace(a[:2], b[:2], n))
    return np.vstack(out) if out else pts[:, :2]


def coverage(pts, bounds=BOUNDS, tool_radius=TOOL_RADIUS, cell=CELL):
    lo, hi = bounds[:, :2].min(axis=0), bounds[:, :2].max(axis=0)
    xs = np.arange(lo[0] + cell / 2, hi[0], cell)
    ys = np.arange(lo[1] + cell / 2, hi[1], cell)
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)

    dense = densify(pts)
    tree = cKDTree(dense)
    dist, _ = tree.query(grid, distance_upper_bound=tool_radius)
    covered = np.isfinite(dist)
    return 100.0 * covered.sum() / len(grid), covered.reshape(len(xs), len(ys))


def main():
    patterns = {
        # only for zigzag as a safety setting
        "Spiral": calculate_spiral(START, BOUNDS),
        "Concentric circles": calculate_circle(START, BOUNDS),
        "Zigzag": calculate_zigzag(START, BOUNDS, margin=0.020),
        "Figure eight": calculate_figure_eight(START, BOUNDS),
    }

    ext = BOUNDS[:, :2].max(axis=0) - BOUNDS[:, :2].min(axis=0)
    print(f"\nFlagged region {ext[0]*1000:.0f} x {ext[1]*1000:.0f} mm, "
          f"tool half-width {TOOL_RADIUS*1000:.1f} mm, grid {CELL*1000:.0f} mm\n")
    print(f"{'pattern':22s} {'waypoints':>10s} {'path (m)':>10s} "
          f"{'coverage %':>12s} {'cov/metre':>11s}")
    for name, pts in patterns.items():
        cov, _ = coverage(pts)
        length = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1).sum()
        print(f"{name:22s} {len(pts):10d} {length:10.2f} {cov:12.1f} "
              f"{cov/length:11.2f}")

    print("\ncov/metre is coverage divided by path length -- how efficiently the")
    print("pattern converts distance travelled into ground covered.\n")


if __name__ == "__main__":
    main()
