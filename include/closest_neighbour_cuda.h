/**
 * For each input query point, locates the k-NN (indexes and distances) among
 * the reference points. This implementation uses global memory to store
 * reference and query points.
 *
 * @param ref        refence points
 * @param ref_nb     number of reference points
 * @param query      query points
 * @param query_nb   number of query points
 * @param dim        dimension of points
 * @param knn_dist_query   output array containing the query_nb distances
 * @param knn_index_query  output array containing the query_nb indexes
 * @param knn_dist_ref   output array containing the ref_nb distances
 * @param knn_index_ref  output array containing the ref_nb indexes
 */
bool closest_cuda(const float *ref, int ref_nb, const float *query,
                      int query_nb, int dim, float *knn_dist_ref,
                      int *knn_index_ref, float *knn_dist_query,
                      int *knn_index_query);