import closest_neighbour_bind
import numpy as np


def compute(ref_pts, query_pts):
    """Compute mutual closest neighbour between two point set using CUDA. Modified from https://github.com/vincentfpgarcia/kNN-CUDA

    Args:
        ref_pts: n x d
        query_pts: m x d
    Returns:
        ref_closest_dist: n
        ref_closest_index: n
        query_closest_dist: m
        query_closest_index: m
    """
    ref_closest_dist, ref_closest_index, query_closest_dist, query_closest_index = closest_neighbour_bind.compute(
        np.asfortranarray(ref_pts), np.asfortranarray(query_pts))

    return ref_closest_dist.astype(np.float64), ref_closest_index.astype(np.int64), query_closest_dist.astype(np.float64), query_closest_index.astype(np.int64)
