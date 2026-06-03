# -*- coding: utf-8 -*-
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

    n_neighbor = min(max(len(xwc) // 1000, 50), len(xwc))
    _, idx = _tree_query(tree, xc[mask], workers, k=n_neighbor)
    if idx.ndim == 1:
        idx = idx[:, None]

    is_masked = np.where(mask)[0]
    be.make_loop(len(is_masked), distf, host=True)[0](
        is_masked, idx, xw, xc, wdist
    )


def _tree_query(tree, x, workers=None, k=1):
    if workers is None:
        return tree.query(x, k=k)

    return tree.query(x, k=k, workers=workers)


def _scipy_workers(be):
    if be.multithread == 'single':
        return 1

    return -1
