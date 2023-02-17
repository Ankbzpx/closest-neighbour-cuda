import numpy as np
import closest_neighbour
from icecream import ic
import time

from scipy.spatial.distance import cdist


def closest_neighbour_sp(ref_pts, query_pts):
    dist_mat = cdist(ref_pts, query_pts)
    ref_closest_index_sp = np.argmin(dist_mat, axis=1)
    ref_closest_dist_sp = np.min(dist_mat, axis=1)
    query_closest_index_sp = np.argmin(dist_mat, axis=0)
    query_closest_dist_sp = np.min(dist_mat, axis=0)

    return ref_closest_dist_sp.astype(np.float32), ref_closest_index_sp.astype(np.int32), query_closest_dist_sp.astype(np.float32), query_closest_index_sp.astype(np.int32)


if __name__ == '__main__':

    np.random.seed(0)

    ref_nb = 10000
    query_nb = 10000
    dim = 3

    test_iter = 10

    start_time = time.time()
    for i in range(test_iter):
        ref_pts = np.random.randn(ref_nb, 3)
        query_pts = np.random.randn(query_nb, 3)
        closest_neighbour_sp(ref_pts, query_pts)
    print(f"Scipy time: {(time.time() - start_time) / test_iter}")

    start_time = time.time()
    for i in range(test_iter):
        ref_pts = np.random.randn(ref_nb, 3)
        query_pts = np.random.randn(query_nb, 3)
        closest_neighbour.compute(np.asfortranarray(
            ref_pts), np.asfortranarray(query_pts))
    print(f"Cuda time: {(time.time() - start_time) / test_iter}")

    ref_pts = np.random.randn(ref_nb, 3)
    query_pts = np.random.randn(query_nb, 3)

    ref_closest_dist_sp, ref_closest_index_sp, query_closest_dist_sp, query_closest_index_sp = closest_neighbour_sp(
        ref_pts, query_pts)

    ref_closest_dist_cuda, ref_closest_index_cuda, query_closest_dist_cuda, query_closest_index_cuda = closest_neighbour.compute(
        np.asfortranarray(ref_pts), np.asfortranarray(query_pts))

    closest_index_valid = (ref_closest_index_sp != ref_closest_index_cuda).sum(
    ) == 0 and (query_closest_index_sp != query_closest_index_cuda).sum() == 0

    closest_dist_valid = np.average((ref_closest_dist_sp - ref_closest_dist_cuda)
                                    ) < 1e-7 and np.average((query_closest_dist_sp - query_closest_dist_cuda)) < 1e-7

    if closest_index_valid and closest_dist_valid:
        print("Test pass")
