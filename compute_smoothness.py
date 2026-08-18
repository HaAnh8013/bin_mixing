"""
Measures how smooth each mixing pattern is, measured from the path geometry.

Answers two questions:
1. How sharply does this path turn?
- reports the mean curvature and corner count
- describes the shape of the path

2. How hard is this path on the arm?
- reports the number of brake-acceleration cycles and speed variation, derived from the speed profile
- describes the load the path puts on the arm

To run:
python3 compute_smoothness.py

Written with the help of Claude Opus 5, Anthropic
"""

import numpy as np

def resample(pts, step=0.005):
    """
    Resample a path at a uniform step along its length, to ensure for fair comparison between different patterns
    """
    pts = np.asarray(pts, dtype=float)[:, :2]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-12])     # drop duplicate points
    pts, s = pts[keep], s[keep]
    n = max(3, int(np.ceil(s[-1] / step)) + 1)
    s_new = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(s_new, s, pts[:, 0]), np.interp(s_new, s, pts[:, 1])]), s[-1]

def curvature(pts, step=0.005):
    """
    Curvature at each interior vertex of a uniformly resampled path.
    
    Returns (kappa, turn_angles, path_length). kappa is in 1/m, turn angles in radians.
    """
    p, length = resample(pts, step)
    v = np.diff(p, axis=0)
    n = np.linalg.norm(v, axis=1)
    good = n > 1e-12
    u = v[good] / n[good, None]

    cross = u[:-1, 0] * u[1:, 1] - u[:-1, 1] * u[1:, 0]
    dot = (u[:-1] * u[1:]).sum(axis=1)
    turn = np.arctan2(cross, dot)

    ds = 0.5 * (n[good][:-1] + n[good][1:])
    kappa = np.abs(turn) / np.maximum(ds, 1e-12)
    return kappa, turn, length


def waypoint_turns(pts):
    """
    Turn angle at each original waypoint, in radians.

    Computed on the waypoints as generated, not on a resampled path, as these are the direction changes received by the controller.
    """
    p = np.asarray(pts, dtype=float)[:, :2]
    v = np.diff(p, axis=0)
    n = np.linalg.norm(v, axis=1)
    good = n > 1e-12
    u = v[good] / n[good, None]
    cross = u[:-1, 0] * u[1:, 1] - u[:-1, 1] * u[1:, 0]
    dot = (u[:-1] * u[1:]).sum(axis=1)
    return np.abs(np.arctan2(cross, dot))


def speed_profile(pts, step=0.005, v_max=0.5, a_max=2.0, t_ctrl=0.085):
    """
    Fastest speed profile the path admits, and what that costs the arm.

    Reports brake-accelerate cycles and total speed variation independent of the resampling step.
    Peak acceleration is not reported.

    Returns a dict: time, v_mean, v_min, brake_accel_cycles, speed_variation.
    """
    p, length = resample(pts, step)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    keep = seg > 1e-12
    seg = seg[keep]
    if len(seg) < 2:
        return {"time": 0.0, "v_mean": 0.0, "v_min": 0.0, "brake_accel_cycles": 0, "speed_variation": 0.0}

    u = np.diff(p, axis=0)[keep] / seg[:, None]
    turn = np.r_[0.0, np.arccos(np.clip((u[:-1] * u[1:]).sum(axis=1), -1, 1)), 0.0]

    v = np.minimum(v_max, np.where(turn > 1e-9, a_max * t_ctrl / (2 * np.sin(np.maximum(turn, 1e-9) / 2)), v_max))
    for i in range(len(seg)):                                   # forward
        v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * a_max * seg[i]))
    for i in range(len(seg) - 1, -1, -1):                        # backward
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_max * seg[i]))

    # a cycle is a local minimum well below the straight-line limit: the arm braked into it and accelerated out again
    cycles = sum(1 for i in range(1, len(v) - 1)
                 if v[i] < 0.6 * v_max and v[i] <= v[i - 1] and v[i] < v[i + 1])
    return {
        "time": float(np.sum(2 * seg / (v[:-1] + v[1:]))),
        "v_mean": float(seg.sum() / np.sum(2 * seg / (v[:-1] + v[1:]))),
        "v_min": float(v.min()),
        "brake_accel_cycles": int(cycles),
        "speed_variation": float(np.abs(np.diff(v)).sum()),
    }


def smoothness_metrics(pts, step=0.005, accel_limit=None, speed=None, corner_deg=45.0, v_max=0.5, a_max=2.0, t_ctrl=0.085):
    """The numbers that go in the report's results table.

    IMPORTANT -- WHY THERE IS NO "PEAK CURVATURE" HERE.
    A path made of straight segments has corners where the direction changes
    discontinuously. At such a corner the true curvature is infinite, and any
    discrete estimate returns turn_angle / step -- which measures the
    resampling step, not the pattern. Reporting it would be meaningless and
    would rank the patterns by an arbitrary choice of step.

    So corners and curves are reported separately:

      * mean curvature       - total direction change per metre travelled.
                               Converges as the step shrinks, so it is a real
                               property of the path. This is the headline
                               number.
      * sharp corners        - how many places the arm must decelerate, turn
                               and accelerate again. Counted on the original
                               waypoints.
      * 95th-pct curvature   - characterises the genuinely curved sections
                               while ignoring the handful of corner vertices.
                               This is the one used for the acceleration
                               figures, since on a smooth arc a = v^2 k holds.

    At a corner the achievable speed is set by how the controller blends the
    corner, not by the pattern, so the pattern's contribution is the NUMBER of
    corners rather than a curvature value.
    """
    kappa, turn, length = curvature(pts, step)
    turns_wp = waypoint_turns(pts)
    k95 = float(np.percentile(kappa, 95))

    m = {
        "path_length_m": length,
        # total turning / length: sampling-independent, normalised for length
        "mean_curvature": float(np.abs(turn).sum() / length),
        "total_turning_rad": float(np.abs(turn).sum()),
        "curv_p95": k95,
        "radius_p95_mm": 1000.0 / k95 if k95 > 0 else np.inf,
        "sharp_corners": int((turns_wp > np.deg2rad(corner_deg)).sum()),
        "max_turn_deg": float(np.rad2deg(turns_wp.max())) if turns_wp.size else 0.0,
    }
    # A path that is mostly straight line has k95 = 0: there is no curvature
    # constraint on the straights at all, and its speed limit comes entirely
    # from how the controller handles the corners. Flag that rather than
    # printing an infinite speed.
    m.update(speed_profile(pts, step, v_max=v_max, a_max=a_max, t_ctrl=t_ctrl))
    m["curvature_limited"] = k95 > 1e-6
    if speed is not None:                          # equation (1)
        m["accel_on_curves"] = speed ** 2 * k95
    if accel_limit is not None and m["curvature_limited"]:   # equation (2)
        m["max_speed_m_s"] = float(np.sqrt(accel_limit / k95))
    else:
        m["max_speed_m_s"] = np.inf
    return m

def main():
    from calculate_pattern import (calculate_spiral, calculate_circle,
                                   calculate_zigzag, calculate_figure_eight)

    # CONFIGS DERIVED FROM REAL CAPTURES AND LOGGED RUNS
    BOUNDS = np.array([
            [0.10130116985996741,  0.1834433533709962,  0.05178152079283693],
            [0.09489065212402058, -0.15408180042334194, 0.03024450504275711],
            [-0.17810309606243288, 0.18571343094513404, 0.06539509440812385],
            [-0.1846769313093502, -0.147858713806498,   0.04536164774311002],
    ])
    START = np.array([-0.04317604702637159, 0.018008358745762322, 0.1])
    V_MAX = 0.345         # m/s
    A_MAX = 1.9           # m/s^2

    patterns = {
        "Spiral": calculate_spiral(START, BOUNDS),
        "Concentric circles": calculate_circle(START, BOUNDS),
        "Zigzag": calculate_zigzag(START, BOUNDS),
        "Figure eight": calculate_figure_eight(START, BOUNDS),
    }

    rows = {}
    for name, pts in patterns.items():
        m = smoothness_metrics(pts, v_max=V_MAX, a_max=A_MAX)
        rows[name] = m
        print(f"{name:22s} {m['mean_curvature']:10.2f} {m['sharp_corners']:8d} "
              f"{m['max_turn_deg']:9.1f}   {m['brake_accel_cycles']:12d} "
              f"{m['speed_variation']:10.2f} {m['time']:7.2f}")

    fewest = min(rows, key=lambda n: rows[n]["brake_accel_cycles"])
    lowest = min(rows, key=lambda n: rows[n]["mean_curvature"])

    print()
    print(f"  Least load on the arm           : {fewest} "
          f"({rows[fewest]['brake_accel_cycles']} brake-accelerate cycles)")
    print(f"  Least turning per metre         : {lowest} "
          f"({rows[lowest]['mean_curvature']:.2f} rad/m)")

if __name__ == "__main__":
    main()
