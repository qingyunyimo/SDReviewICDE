#include "ssd_index.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "neighbor.h"
#include "utils/timer.h"

namespace pipeann {
  // Greedy graph search in which every candidate distance is computed against
  // the original in-memory vector. This path uses neither PQ nor sector I/O.
  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::exact_memory_search(const T *query0, const uint64_t k_search,
                                                 const uint64_t l_search, TagT *res_tags,
                                                 float *res_dists, QueryStats *stats) {
    Timer query_timer;
    if (base_vecs_ == nullptr || base_vecs_num_ != num_points || base_vecs_dim_ != data_dim) {
      LOG(ERROR) << "Original vectors are not loaded. Call load_base_vectors() before exact memory search.";
      return 0;
    }
    if (query0 == nullptr || res_tags == nullptr || l_search == 0 || k_search == 0 || k_search > l_search) {
      LOG(ERROR) << "Invalid exact memory search arguments: K=" << k_search << ", L=" << l_search;
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
    const T *entry_vector = base_vecs_ + static_cast<size_t>(entry) * base_vecs_dim_;
    const float entry_distance = dist_cmp->compare(query, entry_vector, static_cast<unsigned>(data_dim));
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
      const uint32_t node_id = candidates[marker].id;
      char *node_buf = get_navi_buf(node_id);
      const char *nhood_buf = node_buf + navi_data_len_ + sizeof(float);
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

        const T *base = base_vecs_ + static_cast<size_t>(id) * base_vecs_dim_;
        const float distance = dist_cmp->compare(query, base, static_cast<unsigned>(data_dim));
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

    const size_t result_count = std::min<size_t>(k_search, candidate_count);
    for (size_t i = 0; i < result_count; ++i) {
      res_tags[i] = id2tag(candidates[i].id);
      if (res_dists != nullptr) {
        res_dists[i] = candidates[i].distance;
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
