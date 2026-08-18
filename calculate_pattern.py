import numpy as np

def enforce_min_spacing(pts, min_chord=0.015):
    """
    Drop waypoints that are closer than min_chord distance to the previously kept one.
    Always keep the first and last point.

    Note: the final point is kept unconditionally, so the last segment may be
    shorter than min_chord if the path happens to end close to a kept point.
    """
    pts = np.asarray(pts, dtype=float)
    kept = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p[:2] - kept[-1][:2]) >= min_chord:
            kept.append(p)
    if np.linalg.norm(kept[-1][:2] - pts[-1][:2]) > 1e-9:
        kept.append(pts[-1])        # keep the end point
    return np.array(kept)

def calculate_spiral(starting_point, bounds, pitch=0.05, seg_len=0.03, margin=0, clip=False, aspect=None, min_chord=0.015):
    """
    Waypoints spiraling outwards from starting_point, kept inside the given bounds.
    
    Input:
        starting_point: the spiral center
        bounds:     four corner points (top_left, top_right, bottom_left, bottom_right) of the region/box the pattern must stay within
        pitch:      radial gap between consecutive turns - the smaller, the tighter the spiral
        seg_len:    target chord length between waypoints - sampling density
        margin:     extra clearance shrinking the bounds further
        clip:       if True, slide along the bounds; if False, stop at boundary
        aspect:     if None, derive the x/y stretch from the bounds' shape; else, create an explicit stretch
        min_chord:  the closest distance between 2 points

    Output: return an array of spiral-shaped waypoints with a constant z (at starting height)
    """
    # set starting point and bounds
    start = np.asarray(starting_point, dtype=float)
    corners = np.asarray(bounds, dtype=float)[:, :2]       # top_left, top_right, bottom_left, bottom_right
    lo = corners.min(axis=0) + margin
    hi = corners.max(axis=0) - margin
    if np.any(hi - lo <= 0):
        raise ValueError("bounds collapse after margin")

    # calculate stretch
    if aspect is None:
        extent = hi - lo
        aspect = extent / extent.max()      # fill the box's shape
    else:
        aspect = np.asarray(aspect, dtype=float)
        aspect = aspect / aspect.max()      # normalize


    b = pitch / (2.0 * np.pi)       # radius growth per radian
    r_max = np.max(np.linalg.norm(corners - start[:2], axis=1)) / aspect.min()

    pts, theta = [start.copy()], 0.0

    while True:
        local_r = b * theta * np.hypot(aspect[0] * np.cos(theta), aspect[1] * np.sin(theta))
        theta += seg_len / max(local_r, seg_len)
        r = b * theta

        if r > r_max:
            break
        p = start + np.array([r * aspect[0] * np.cos(theta), r * aspect[1] * np.sin(theta), 0.0])
        inside = np.all(p[:2] >= lo) and np.all(p[:2] <= hi)

        if not inside:
            if not clip:
                break       # first wall contact ends the spiral
            p[:2] = np.clip(p[:2], lo, hi)      # slide along the wall
        pts.append(p)
    return enforce_min_spacing(np.array(pts), min_chord)

def calculate_circle(start, bounds, pitch=0.05, seg_len=0.03, margin=0.0, clip=False, min_chord=0.015):
    """
    Waypoints in the shape of concentric rings around starting point, with inner ring first.

    Input:
        start:     coordinates of starting point
        bounds:    four corner points (top_left, top_right, bottom_left, bottom_right) of the bounds in which the pattern is to be created
        pitch:     radial gap between consecutive rings
        seg_len:    target chord length between waypoints - sampling density
        margin:     extra clearance shrinking the bounds further
        clip:       if True, keep growing to farthest corner, sliding along walls; if False, only rings that fully fit inside bounds
        min_chord: the closest distance between 2 points

    Output: return an array of concentric-circle-shaped waypoints with a constant z (at starting height)
    """
    start = np.asarray(start, dtype=float)      # set starting position

    # set bounds for pattern
    corners = np.asarray(bounds, dtype=float)[:, :2]      # top_left, top_right, bottom_left, bottom_right
    lo = corners.min(axis=0) + margin
    hi = corners.max(axis=0) - margin
    if np.any(hi - lo <= 0):
        raise ValueError("bounds collapse after margin")

    if clip:
        r_stop = np.max(np.linalg.norm(corners - start[:2], axis=1))
    else:   # largest ring that fully fits = distance to the nearest wall
        r_stop = np.min(np.concatenate([start[:2] - lo, hi - start[:2]]))

    pts, r = [start.copy()], pitch
    while r <= r_stop:
        n_seg = max(8, int(np.ceil(2 * np.pi * r / seg_len)))
        for th in np.linspace(0.0, 2 * np.pi, n_seg + 1):       # +1 closes the ring
            p = start + np.array([r * np.cos(th), r * np.sin(th), 0.0])
            if clip:
                p[:2] = np.clip(p[:2], lo, hi)
            pts.append(p)
        r += pitch
    return enforce_min_spacing(np.array(pts), min_chord)

def calculate_zigzag(start, bounds, pitch=0.05, margin=0.02, min_chord=0.015, center_out=False):
    """
    Waypoints boustrophedon sweep of the whole box. Passes run along the longer axis; entry corner is the box corner nearest to starting point.

    Note on the margin default: this is the only pattern whose passes reach the
    edge of the region, so it is the only one that can collide with the bin wall.
    It therefore defaults to a 20 mm clearance, where the other three default to
    zero because they never approach the boundary.

    Input:
      start:      coordinates of starting point
      bounds:     four corner points (top_left, top_right, bottom_left, bottom_right) of the bounds in which the pattern is to be created
      pitch:      gap between adjacent sweep lines
      margin:     extra clearance shrinking the bounds further
      min_chord:  the closest distance between 2 points
      center_out: if True, `start` must sit strictly inside the bounds along the long axis; the
                  path sweeps from `start` out to the near long-axis edge, returns to start,
                  then sweeps out to the far long-axis edge and returns to start again instead of a single corner-to-corner pass.

    Output: return an array of zigzag-shaped waypoints with a constant z (at starting height)
    """
    start = np.asarray(start, dtype=float)      # set starting point

    # set outside bounds
    corners = np.asarray(bounds, dtype=float)[:, :2]      # top_left, top_right, bottom_left, bottom_right
    lo = corners.min(axis=0) + margin
    hi = corners.max(axis=0) - margin
    extent = hi - lo
    if np.any(extent <= 0):
        raise ValueError("bounds collapse after margin")

    # set axis bounds
    long_ax = 0 if extent[0] >= extent[1] else 1        # passes along this axis
    short_ax = 1 - long_ax

    if center_out:
        if not (lo[long_ax] < start[long_ax] < hi[long_ax]):
            raise ValueError("start must lie strictly inside bounds along the long axis for center_out")

        pts = [start.copy()]
        for edge in (lo, hi):        # sweep toward the near edge first, then the far edge
            half_lo, half_hi = lo.copy(), hi.copy()
            half_lo[long_ax], half_hi[long_ax] = sorted((start[long_ax], edge[long_ax]))
            half_bounds = np.array([[half_lo[0], half_lo[1], start[2]],
                                     [half_hi[0], half_hi[1], start[2]]])
            leg = calculate_zigzag(start, half_bounds, pitch=pitch, margin=0.0, min_chord=min_chord)
            pts.extend(leg[1:])      # leg[0] duplicates `start`, already appended
            if np.linalg.norm(leg[-1, :2] - start[:2]) > 1e-9:
                pts.append(start.copy())     # explicit return to start
        return enforce_min_spacing(np.array(pts), min_chord)

    # entry corner: per axis, whichever wall is closer to `start`
    near = np.where(start[:2] - lo <= hi - start[:2], lo, hi)
    far = np.where(near == lo, hi, lo)

    # sweep lines across the short axis, spacing <= pitch, both edges included
    n_lines = max(2, int(np.ceil(extent[short_ax] / pitch)) + 1)
    across = np.linspace(near[short_ax], far[short_ax], n_lines)

    ends = (near[long_ax], far[long_ax])
    pts = []
    for i, c in enumerate(across):
        for v in ends if i % 2 == 0 else ends[::-1]:        # alternate direction
            p = np.empty(3)
            p[long_ax], p[short_ax], p[2] = v, c, start[2]
            pts.append(p)

    return enforce_min_spacing(np.array(pts), min_chord)

def calculate_figure_eight(start, bounds, seg_len=0.03, margin=0.0, scale=1.0, n_dense=4096, min_chord=0.015):
    """
    Waypoints following a figure-eight circuit centered on start, with lobes along the box's longer axis, sized to fit the bounds.

    Input:
        start:      the starting position
        bounds:     four corner points (top_left, top_right, bottom_left, bottom_right) of the region/box the pattern must stay within
        seg_len:    target chord length between waypoints - sampling density
        margin:     extra clearance shrinking the bounds further
        scale:      1.0 fills the available space; smaller shrinks the eight
        n_dense:    internal sampling resolution for arc-length resampling
        min_chord:  the closest distance between 2 points

    Output: return an array of figure-eight-shaped waypoints with a constant z (at starting height)
    """
    start = np.asarray(start, dtype=float)      # convert starting point to array

    # set bounds
    corners = np.asarray(bounds, dtype=float)[:, :2]      # top_left, top_right, bottom_left, bottom_right
    lo = corners.min(axis=0) + margin
    hi = corners.max(axis=0) - margin
    if np.any(hi - lo <= 0):
        raise ValueError("bounds collapse after margin")

    # per-axis half-size: how far the eight may reach from start each way
    half = np.minimum(start[:2] - lo, hi - start[:2]) * scale
    if np.any(half <= 0):
        raise ValueError("start lies outside the (margin-shrunk) bounds")

    long_ax = 0 if (hi - lo)[0] >= (hi - lo)[1] else 1      # lobes along this axis
    short_ax = 1 - long_ax

    # form the figure eight path
    # dense parametric trace: 1:2 Lissajous, starts and crosses at the center
    t = np.linspace(0.0, 2.0 * np.pi, n_dense)
    dense = np.empty((n_dense, 2))
    dense[:, long_ax] = half[long_ax] * np.sin(t)
    dense[:, short_ax] = half[short_ax] * np.sin(2.0 * t)

    # resample to ~seg_len spacing by cumulative arc length
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])          # distance along path
    n_pts = max(8, int(np.ceil(s[-1] / seg_len)))       # number of points 
    s_target = np.linspace(0.0, s[-1], n_pts + 1)
    xy = np.column_stack([np.interp(s_target, s, dense[:, 0]), np.interp(s_target, s, dense[:, 1])])

    pts = np.column_stack([start[0] + xy[:, 0],          # offsets around start
                            start[1] + xy[:, 1],
                            np.full(len(xy), start[2])])
    return enforce_min_spacing(pts, min_chord)