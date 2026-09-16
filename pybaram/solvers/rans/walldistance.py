# -*- coding: utf-8 -*-
import os
import numpy as np


def compute_wall_distance(be, ndims, neles, xc, xw):
    wall_dist = np.empty(neles)
    distf = _make_distance_refiner(ndims)

    try:
        _compute_with_pykdtree(be, xc, xw, wall_dist, distf)
    except ImportError:
        _compute_with_scipy(be, xc, xw, wall_dist, distf)

    return wall_dist


def _make_distance_refiner(ndims):
    if ndims == 2:
        from pybaram.utils.nb import dist2d_at

        def distf(i_begin, i_end, is_masked, idx, xw, xc, wdist):
            for _i in range(i_begin, i_end):
                k = is_masked[_i]
                for _j in range(idx.shape[1]):
                    j = idx[_i, _j]
                    status, distj = dist2d_at(xw[j][0], xw[j][1], xc[k])

                    if _j == 0:
                        dist = distj
                    else:
                        dist = min(dist, distj)

                    if status == 0:
                        break

                wdist[k] = dist

    elif ndims == 3:
        from pybaram.utils.nb import dist3d_at

        def distf(i_begin, i_end, is_masked, idx, xw, xc, wdist):
            for _i in range(i_begin, i_end):
                k = is_masked[_i]
                for _j in range(idx.shape[1]):
                    j = idx[_i, _j]
                    status, distj = dist3d_at(
                        xw[j][0], xw[j][1], xw[j][2], xc[k]
                    )

                    if _j == 0:
                        dist = distj
                    else:
                        dist = min(dist, distj)

                    if status == 0:
                        break

                wdist[k] = dist

    else:
        raise ValueError("Wall distance is only supported in 2D and 3D")

    return distf


def _compute_with_scipy(be, xc, xw, wdist, distf):
    from scipy.spatial import KDTree

    xwc = np.average(xw, axis=1)
    tree = KDTree(xwc)
    _compute_with_tree(be, tree, xc, xw, xwc, wdist, distf, _scipy_workers(be))


def _compute_with_pykdtree(be, xc, xw, wdist, distf):
    from pykdtree.kdtree import KDTree

    xwc = np.average(xw, axis=1)
    tree = KDTree(xwc)
    _compute_with_tree(be, tree, xc, xw, xwc, wdist, distf)


def _compute_with_tree(be, tree, xc, xw, xwc, wdist, distf, workers=None):
    fast_distance, fast_idx = _tree_query(tree, xc, workers)
    wdist[:] = fast_distance

    threshold = 2*np.max(np.linalg.norm(xw - xwc[:, None], axis=2), axis=1)
    mask = fast_distance < threshold[fast_idx]

    if not np.any(mask):
        return

    n_neighbor = _estimate_n_neighbor(xw, xwc)
    _, idx = _tree_query(tree, xc[mask], workers, k=n_neighbor)
    if idx.ndim == 1:
        idx = idx[:, None]

    is_masked = np.where(mask)[0]
    be.make_loop(len(is_masked), distf, host=True)[0](
        is_masked, idx, xw, xc, wdist
    )


def _estimate_n_neighbor(xw, xwc):
    nwall = len(xw)
    if nwall <= 50:
        return nwall

    #TODO: Need to refine
    # Use a search radius 2.5 times the representative face radius.  The
    # additional safety factor accounts for curvature and non-uniform surface
    # meshes which are not represented by the global mean face measure.
    beta = 2.5
    safety = 2.0

    face_radius = np.max(
        np.linalg.norm(xw - xwc[:, None], axis=2), axis=1
    )
    radius95 = np.percentile(face_radius, 95)

    if xw.shape[1] == 2:
        # A 2-D wall is a line: estimate how many segments fit in a search
        # interval whose radius is proportional to a representative segment.
        measure = np.linalg.norm(xw[:, 1] - xw[:, 0], axis=1)
        mean_measure = np.mean(measure)
        estimated = (
            safety*2*beta*radius95 / mean_measure
            if mean_measure else np.inf
        )
    else:
        # A 3-D wall is a surface: estimate how many triangles fit in a search
        # disk whose radius is proportional to a representative triangle.
        edge0 = xw[:, 1] - xw[:, 0]
        edge1 = xw[:, 2] - xw[:, 0]
        measure = 0.5*np.linalg.norm(np.cross(edge0, edge1), axis=1)
        mean_measure = np.mean(measure)
        estimated = (
            safety*np.pi*(beta*radius95)**2 / mean_measure
            if mean_measure else np.inf
        )

    # Keep the historical minimum and use the old 0.1% rule as an upper cap.
    # Degenerate wall faces fall back to that conservative upper bound.
    upper = max(nwall // 1000, 50)
    if not np.isfinite(estimated):
        return upper

    return min(max(int(np.ceil(estimated)), 50), upper)


def _tree_query(tree, x, workers=None, k=1):
    if workers is None:
        return tree.query(x, k=k)

    return tree.query(x, k=k, workers=workers)


def _scipy_workers(be):
    workers = getattr(be, 'cpu_workers', 1)

    if workers > 0:
        return workers
    elif workers <= 0:
        return -1
