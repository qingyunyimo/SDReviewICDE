#include "ssd_index.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "neighbor.h"
#include "utils/timer.h"

namespace pipeann {
  // Uses the PCA-rotated query and the in-memory PCA navigation vectors for
  // graph traversal, then reranks the retained candidates against original
  // vectors in memory. No sector I/O or PQ distance is used.
  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::pca_memory_search(const T *query0, const float *pca_query,
                                               const uint64_t pca_dim, const uint64_t k_search,
                                               const uint64_t l_search,
                                               TagT *res_tags, float *res_dists,
                                               QueryStats *stats) {
    Timer query_timer;
    if (base_vecs_ == nullptr || base_vecs_num_ != num_points || base_vecs_dim_ != data_dim) {
      LOG(ERROR) << "Original vectors are not loaded. Call load_base_vectors() before PCA memory search.";
      return 0;
    }
    if (query0 == nullptr || pca_query == nullptr || res_tags == nullptr || pca_dim < navi_dim_ ||
        l_search == 0 || k_search == 0 || k_search > l_search) {
      LOG(ERROR) << "Invalid PCA memory search arguments: PCA dim=" << pca_dim
                 << ", navigation dim=" << navi_dim_ << ", K=" << k_search << ", L=" << l_search;
      return 0;
    }

    QueryBuffer<T> *query_buf = pop_query_buf(query0);
    query_buf->reset();
    const T *query = query_buf->aligned_query_T;
    auto &visited = *query_buf->visited;

    std::vector<Neighbor> candidates(l_search + 1);
    uint32_t candidate_count = 0;
    uint32_t cursor = 0;

    const uint32_t entry = static_cast<uint32_t>(medoid);
    const float entry_distance =
        l2_sqr(pca_query, reinterpret_cast<const float *>(get_navi_buf(entry)), navi_dim_);
    candidates[candidate_count++] = Neighbor(entry, entry_distance, true);
    visited.insert(entry);
    if (stats != nullptr) {
      stats->n_cmps++;
    }

    while (cursor < candidate_count) {
      uint32_t marker = cursor;
      while (marker < candidate_count && candidates[marker].visited) {
        ++marker;
      }
      if (marker == candidate_count) {
        break;
      }

      candidates[marker].visited = true;
      char *node_buf = get_navi_buf(candidates[marker].id);
      const char *nhood_buf = node_buf + navi_data_len_;
      const uint32_t neighbor_count = *reinterpret_cast<const uint32_t *>(nhood_buf);
      const uint32_t *neighbor_ids = reinterpret_cast<const uint32_t *>(nhood_buf + sizeof(uint32_t));
      const uint32_t bounded_neighbor_count =
          std::min<uint32_t>(neighbor_count, static_cast<uint32_t>(max_degree));

      uint32_t best_insert = candidate_count;
      for (uint32_t i = 0; i < bounded_neighbor_count; ++i) {
        const uint32_t id = neighbor_ids[i];
        if (id >= num_points || !visited.insert(id).second) {
          continue;
        }

        const float distance =
            l2_sqr(pca_query, reinterpret_cast<const float *>(get_navi_buf(id)), navi_dim_);
        if (stats != nullptr) {
          stats->n_cmps++;
        }
        if (candidate_count == l_search && distance >= candidates[candidate_count - 1].distance) {
          continue;
        }

        const uint32_t pos = InsertIntoPool(candidates.data(), candidate_count,
                                            Neighbor(id, distance, true));
        if (pos <= candidate_count) {
          if (candidate_count < l_search) {
            ++candidate_count;
          }
          best_insert = std::min(best_insert, pos);
        }
      }
      if (stats != nullptr) {
        stats->n_hops++;
      }
      cursor = best_insert <= marker ? best_insert : marker + 1;
    }

    std::vector<Neighbor> exact;
    exact.reserve(candidate_count);
    for (uint32_t i = 0; i < candidate_count; ++i) {
      const uint32_t id = candidates[i].id;
      const T *base = base_vecs_ + static_cast<size_t>(id) * base_vecs_dim_;
      exact.emplace_back(id, dist_cmp->compare(query, base, static_cast<unsigned>(data_dim)), false);
    }
    std::sort(exact.begin(), exact.end());

    const size_t result_count = std::min<size_t>(k_search, exact.size());
    for (size_t i = 0; i < result_count; ++i) {
      res_tags[i] = id2tag(exact[i].id);
      if (res_dists != nullptr) {
        res_dists[i] = exact[i].distance;
      }
    }

    push_query_buf(query_buf);
    if (stats != nullptr) {
      stats->total_us = static_cast<double>(query_timer.elapsed());
      stats->cpu_us = stats->total_us;
    }
    return result_count;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
