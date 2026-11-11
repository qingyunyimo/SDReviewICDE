#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "neighbor.h"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#ifndef USE_AIO
#include "liburing.h"
#endif

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include "utils/timer.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"
#include "utils/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>

namespace pipeann {
    struct io_t {
        Neighbor nbr;
        unsigned page_id;
        unsigned loc;
        IORequest *read_req;
        bool operator>(const io_t &rhs) const {
            return nbr.distance > rhs.nbr.distance;
        }

        bool operator<(const io_t &rhs) const {
            return nbr.distance < rhs.nbr.distance;
        }

        bool finished() {
            return read_req->finished;
        }
    };

    template<typename T, typename TagT>
    size_t SSDIndex<T, TagT>::skipdisk_pcabit_search(const T *query1, const uint64_t k_search,
                                            const uint64_t l_search, TagT *res_tags, float *distances,
                                            const uint64_t beam_width, QueryStats *stats, bool in_memory, bool quantized_only, bool use_bound, const T *navigation_query) {
        if (navigation_query && (!mixed_pivot_ || !mixed_subpq_navigation_ || in_memory || quantized_only))
            throw std::invalid_argument("Independent navigation query requires mixed SSD subPQ search");
        if (mixed_pivot_ && mixed_cache_ && !mixed_subpq_navigation_)
            throw std::logic_error("Partial mixed filtering requires independent subPQ navigation");
        if (quantized_only && (!mixed_pivot_ || in_memory))
            throw std::logic_error("Quantized-only search requires a mixed index and no exact memory mode");
        if (mixed_estimation_ && (!mixed_pivot_ || quantized_only || !use_bound ||
            !std::isfinite(mixed_estimation_ratio_) || mixed_estimation_ratio_ <= 0))
            throw std::logic_error("Estimation requires mixed bound filtering and a positive finite ratio");
        if (in_memory && (!mixed_pivot_ || !base_vecs_ || base_vecs_num_ != num_points ||
                          base_vecs_dim_ != data_dim))
            throw std::logic_error("Load full PCA base vectors and a mixed index before memory search");
        QueryBuffer<T> *query_buf = pop_query_buf(query1);
        void *ctx = (in_memory || quantized_only) ? nullptr : reader->get_ctx(IORING_SETUP_SQPOLL);

        if (beam_width > MAX_N_SECTOR_READS) {
            LOG(ERROR) << "Beamwidth can not be higher than MAX_N_SECTOR_READS";
            crash();
        }

        const T *query = query_buf->aligned_query_T;

        // reset query
        query_buf->reset();

        T *data_buf = query_buf->coord_scratch;
        _mm_prefetch((char *) data_buf, _MM_HINT_T1);

        // sector scratch
        char *sector_scratch = query_buf->sector_scratch;
        float *dist_scratch = query_buf->aligned_dist_scratch;

        Timer query_timer, io_timer, cpu_timer, add_timer;
        std::vector<Neighbor> retset(l_search + 1);
        std::vector<uint32_t>  to_checked_nbrs(this->max_degree, 0);
        
        std::priority_queue<std::pair<float, uint32_t>> top_candidates;
        std::vector<std::pair<float, uint32_t>> quantized_candidates;
        // Unfiltered SSD candidates retain expansion order; never prune by an
        // approximate navigation distance or the current exact top-K threshold.
        std::queue<uint32_t> unfiltered_candidates;
        if (quantized_only) quantized_candidates.reserve(l_search);
        std::priority_queue<std::pair<float, uint32_t>,
                   std::vector<std::pair<float, uint32_t>>,
                   std::greater<std::pair<float, uint32_t>>> temp_candidates;
        float threshold = std::numeric_limits<float>::max();
        auto filtering_threshold = [&]() -> double {
            // Never estimate-prune before an exact top-K threshold exists.
            if (mixed_estimation_ && top_candidates.size() < k_search)
                return std::numeric_limits<double>::infinity();
            return mixed_estimation_ ? mixed_estimation_ratio_ * threshold : double(threshold);
        };
        auto &visited = *(query_buf->visited);
        unsigned cur_list_size = 0;
        unsigned k = 0;
        std::vector<std::pair<uint32_t, float>>  beam_ids; 
        beam_ids.reserve(beam_width);
        std::vector<uint32_t> frontier;
        frontier.reserve(beam_width);
        int64_t cur_beam_width = std::min(32ul, beam_width);
        std::vector<float>  restored_navi_vec(navi_dim_, 0.0f);
        std::vector<double> mixed_table;
        if (mixed_pivot_ && !((quantized_only || !use_bound) && mixed_subpq_navigation_))
            mixed_table = mixed_pivot_->distance_table(reinterpret_cast<const float *>(query));
        auto pivot_distance = [&](const char *code) -> float {
            if (mixed_pivot_)
                return static_cast<float>(mixed_pivot_->squared_distance(
                    reinterpret_cast<const float *>(query), mixed_table, code));
            bit_preserver_.loadFloatFromHigh16BitsAVX512(const_cast<char *>(code), restored_navi_vec.data());
            return l2_sqr((float *)query, restored_navi_vec.data(), navi_dim_);
        };

        uint32_t best_medoid = medoid;
        char* best_medoid_buf = get_navi_buf(best_medoid);
        const bool subpq_navigation = mixed_pivot_ && mixed_subpq_navigation_;
        float best_medoid_dist;
        if (subpq_navigation) {
            nbr_handler->initialize_query(navigation_query ? navigation_query : query, query_buf);
            nbr_handler->compute_dists(query_buf, &best_medoid, 1);
            best_medoid_dist = dist_scratch[0];
        } else {
            best_medoid_dist = pivot_distance(best_medoid_buf);
        }
        retset[cur_list_size].id = best_medoid;
        retset[cur_list_size].distance = best_medoid_dist;
        retset[cur_list_size++].visited = false;
        visited.insert(best_medoid);

        //lambda used in search
        auto compute_navidist_and_push_nbrs = [&](char* node_buf, uint32_t &nk){
            unsigned nnbrs = *(unsigned*) node_buf;
            unsigned* nbr_ids = (unsigned*) (node_buf + sizeof(unsigned));
            unsigned nbers_cand_size = 0;
            for(unsigned i = 0; i < nnbrs; i++){
                uint32_t nbr_id = nbr_ids[i];
                if(visited.find(nbr_id) == visited.end()){
                    visited.insert(nbr_id);
                    to_checked_nbrs[nbers_cand_size++] = nbr_id;
                }
            }
            if (subpq_navigation && nbers_cand_size)
                nbr_handler->compute_dists(query_buf, to_checked_nbrs.data(), nbers_cand_size);
            for(unsigned i = 0; i < nbers_cand_size; i++){
                uint32_t nbr_id = to_checked_nbrs[i];
                char* nbr_buf = get_navi_buf(nbr_id);
                float nber_dist = subpq_navigation ? dist_scratch[i] : pivot_distance(nbr_buf);
                if (mixed_pivot_ && stats) stats->n_cmps++;
                if (nber_dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search)) continue;

                Neighbor nn(nbr_id, nber_dist, true);
                auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
                if (cur_list_size < l_search) ++cur_list_size;
                // nk logs the best position in the retset that was updated due to neighbors of n.
                if (r < nk) nk = r;
            }
        };

        uint64_t l_k_search = k_search;
        auto compute_exact_dists_and_push = [&](const char *node_buf, const unsigned id) {
            if (stats) stats->n_exact++;
            float full_dist = l2_sqr((float*)query, (float*) node_buf, mixed_pivot_ ? data_dim : aligned_dim);
            top_candidates.emplace(std::make_pair(full_dist, id));
            if(top_candidates.size() > l_k_search){
                top_candidates.pop();
                threshold = top_candidates.top().first;
            }
            if (mixed_pivot_ && top_candidates.size() == l_k_search)
                threshold = top_candidates.top().first;
        };

        std::queue<io_t> on_flight_ios;
        auto send_read_req = [&](Neighbor &item) -> bool {
            if (quantized_only) throw std::logic_error("Exact rerank forbidden in quantized-only mode");
            item.flag = false;
            if (in_memory) {
                // IDs address the original-ID-order RAM array, not SSD sector locations.
                const T *base = base_vecs_ + size_t(item.id) * base_vecs_dim_;
                compute_exact_dists_and_push(reinterpret_cast<const char *>(base), item.id);
                return true;
            }

            // lock the corresponding page.
            //this->lock_idx(idx_lock_table, item.id, std::vector<uint32_t>(), true);
            const unsigned loc = id2loc(item.id), pid = loc_sector_no(loc);
            uint64_t &cur_buf_idx = query_buf->sector_idx;
            auto buf = sector_scratch + cur_buf_idx * size_per_io;
            auto &req = query_buf->reqs[cur_buf_idx];
            req = IORequest(static_cast<uint64_t>(pid) * SECTOR_LEN, size_per_io, buf, u_loc_offset(loc), max_node_len,
                            sector_scratch);
            reader->send_read_no_alloc(req, ctx);

            on_flight_ios.push(io_t{item, pid, loc, &req});
            cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;

            if (stats != nullptr) {
                stats->n_ios++;
            }
            return true;
        };

        auto send_best_read_req = [&](uint32_t n) -> unsigned {
            // auto io_st = std::chrono::high_resolution_clock::now();
            unsigned n_sent = 0;
            if (!use_bound) {
                while (!unfiltered_candidates.empty() && n_sent < n) {
                    Neighbor nn(unfiltered_candidates.front(), 0.0f, true);
                    unfiltered_candidates.pop();
                    n_sent += send_read_req(nn);
                }
                return n_sent;
            }
            while(!temp_candidates.empty() && n_sent < n){
                auto cand = temp_candidates.top();
                temp_candidates.pop();
                if(cand.first > filtering_threshold()){
                    break;
                }
                uint32_t id = cand.second;
                Neighbor nn(id, cand.first, true);
                n_sent += send_read_req(nn);
            }
            return n_sent;  
        };
        
        std::unordered_map<unsigned, char *> id_buf_map;
        auto poll_all = [&]() -> std::pair<int, int> {
            if (in_memory || quantized_only) return {0, 0};
            // poll once.
            reader->poll_all(ctx);
            unsigned n_in = 0, n_out = 0;
            while (!on_flight_ios.empty() && on_flight_ios.front().finished()) {
                io_t &io = on_flight_ios.front();
                //std::cout<<io.nbr.id<<","<<io.loc<<","<<io.nbr.distance<<";"<<std::endl;
                id_buf_map.insert(std::make_pair(io.nbr.id, offset_to_loc((char *) io.read_req->buf, io.loc)));
                io.nbr.distance <= retset[cur_list_size - 1].distance ? ++n_in : ++n_out;
                frontier.push_back(io.nbr.id);
                // unlock the corresponding page.
                //this->unlock_idx(idx_lock_table, io.nbr.id);
                on_flight_ios.pop();
            }
            return std::make_pair(n_in, n_out);
        };



        auto cpu2_st = std::chrono::high_resolution_clock::now();

        unsigned max_marker = 0;
        unsigned cur_total_io = 0;

        while(k < cur_list_size){
            unsigned nk = cur_list_size;
            uint32_t marker = k;
            uint32_t num_seen = 0;

            //cpu_timer.reset();
            while(marker < cur_list_size && num_seen < 1){
                if(!retset[marker].visited){
                    num_seen++;
                    retset[marker].visited = true;
                    beam_ids.push_back(std::make_pair(retset[marker].id, retset[marker].distance));
                }
                marker++;
            }
            //if(stats != nullptr){
            //    stats->cpu_us += cpu_timer.elapsed();
            //}

            k = marker - 1;

            auto [n_in, n_out] = poll_all();
            cur_total_io += n_in + n_out;

            cpu_timer.reset();
            unsigned sent_n = 0;
            if((int64_t) on_flight_ios.size() < cur_beam_width) {
                //sent_n = send_best_read_req(cur_beam_width - on_flight_ios.size());
                sent_n = send_best_read_req(1);
            }

            uint32_t bsize = beam_ids.size();
            for(uint32_t i = 0; i < bsize; i++){
                uint32_t node_id = beam_ids[i].first;
                if (mixed_pivot_ && stats) {
                    stats->n_hops++;
                    stats->candidate_hash = (stats->candidate_hash ^ node_id) * 1099511628211ULL;
                    if (stats->candidate_trace) stats->candidate_trace->push_back(node_id);
                }
                if (quantized_only) {
                    // Include every expanded node, even if later evicted from retset.
                    // Never compare a full-space lower bound to a quantized score.
                    quantized_candidates.emplace_back(beam_ids[i].second, node_id);
                    compute_navidist_and_push_nbrs(mixed_neighbors_buf(node_id), nk);
                    continue;
                }
                if (!use_bound) {
                    // Same graph expansion, but no pivot distance, bound queue or pruning.
                    if (in_memory) {
                        const T *base = base_vecs_ + size_t(node_id) * base_vecs_dim_;
                        compute_exact_dists_and_push(reinterpret_cast<const char *>(base), node_id);
                    } else {
                        unfiltered_candidates.push(node_id);
                    }
                    compute_navidist_and_push_nbrs(mixed_neighbors_buf(node_id), nk);
                    continue;
                }
                float q2p = std::sqrt(beam_ids[i].second);
                char* node_buf = mixed_filter_buf(node_id);
                if (!node_buf) {
                    // No representation => no bound/estimation. Zero priority
                    // guarantees exact verification even with a zero threshold.
                    temp_candidates.emplace(0.0f, node_id);
                    compute_navidist_and_push_nbrs(mixed_neighbors_buf(node_id), nk);
                    continue;
                }
                float* x2p_ptr = (float*)(node_buf + navi_data_len_);
                float x2p = *x2p_ptr;
                float lb_dist = std::abs(q2p - x2p);
                lb_dist = lb_dist * lb_dist;
                if (mixed_pivot_) {
                    // Reuse the LUT, but retain double precision for the bound;
                    // the float navigation score is only for graph ordering.
                    if (fp32_filter_dim_) {
                        const float *pivot=fp32_filter_data_.data()+size_t(node_id)*fp32_filter_dim_;
                        double squared=0;
                        for (size_t d=0;d<fp32_filter_dim_;++d) {
                            const double delta=double(query[d])-double(pivot[d]);
                            squared+=delta*delta;
                        }
                        // Same conservative roundoff allowance as mixed pivots; r=0.
                        lb_dist=mixed_pivot_->lower_bound(squared,0.0f);
                    } else {
                        lb_dist = mixed_pivot_->lower_bound(mixed_pivot_->squared_distance(
                            reinterpret_cast<const float *>(query), mixed_table, node_buf), x2p);
                    }
                    if (stats) stats->n_bounds++;
                }
                temp_candidates.push(std::make_pair(lb_dist, node_id));
                
                compute_navidist_and_push_nbrs(mixed_neighbors_buf(node_id), nk);
            }
            beam_ids.clear();

            uint32_t fsize = frontier.size();
            for(uint32_t i = 0; i < fsize; i++){
                auto it = id_buf_map.find(frontier[i]);
                if(it != id_buf_map.end()){
                    auto [id, buf] = *it;
                    compute_exact_dists_and_push(buf, id);
                }else{
                    std::cout<<"Error: frontier id not in id_buf_map!"<<std::endl;
                    crash();
                }
            }
            frontier.clear();

            if(stats != nullptr){
                stats->cpu_us += cpu_timer.elapsed();
                //std::cout<<stats->cpu_us<<";";
            }
            if (nk <= k)
                k = nk;  // k is the best position in retset updated in this round.
            else
                ++k;
        }

        if (quantized_only) {
            const size_t count = std::min<size_t>(k_search, quantized_candidates.size());
            std::partial_sort(quantized_candidates.begin(), quantized_candidates.begin() + count,
                              quantized_candidates.end());
            for (size_t i = 0; i < count; ++i) {
                res_tags[i] = id2tag(quantized_candidates[i].second);
                if (distances) distances[i] = quantized_candidates[i].first;
            }
            push_query_buf(query_buf);
            if (stats) stats->total_us = query_timer.elapsed();
            return count;
        }

        add_timer.reset();
        while(!on_flight_ios.empty() || (!use_bound && !unfiltered_candidates.empty()) ||
              (use_bound && !temp_candidates.empty() && temp_candidates.top().first <= filtering_threshold())){
            auto [n_in, n_out] = poll_all();
            //if(n_in + n_out > 0){
                //std::cout<<n_in<<","<<n_out<<","<<on_flight_ios.size()<<";"<<std::endl;
            //}
            
            unsigned more_sent = 0;
            if((int64_t) on_flight_ios.size() < cur_beam_width) {
                more_sent = send_best_read_req(cur_beam_width - on_flight_ios.size());
                if(stats != nullptr && !in_memory){
                    stats->n_addio += more_sent;
                }
            }

            uint32_t fsize = frontier.size();
            for(uint32_t i =0; i< fsize; i++){
                auto it = id_buf_map.find(frontier[i]);
                if(it != id_buf_map.end()){
                    auto [id, buf] = *it;
                    compute_exact_dists_and_push(buf, id);
                }else{
                    std::cout<<"Error: frontier id not in id_buf_map!"<<std::endl;
                    crash();
                }
            }
            frontier.clear();
        }

         if(stats != nullptr){
            stats->n_addtime += add_timer.elapsed();
        }

        while(!top_candidates.empty() && top_candidates.size() > k_search){
            top_candidates.pop();
        }
        const size_t result_count = top_candidates.size();
        for(uint64_t i =0; i<k_search; i++){
            if(top_candidates.empty()){
                break;
            }
            const size_t result_offset = mixed_pivot_ ? result_count - 1 - i : i;
            res_tags[result_offset] = mixed_pivot_ ? id2tag(top_candidates.top().second) : top_candidates.top().second;
            
            if(distances != nullptr){
                distances[result_offset] = top_candidates.top().first;
            }
            top_candidates.pop();
        }

        push_query_buf(query_buf);

        if (stats != nullptr) {
            stats->total_us = (double) query_timer.elapsed();
        }

        return mixed_pivot_ ? result_count : 0;
    }

    template class SSDIndex<float>;
    template class SSDIndex<int8_t>;
    template class SSDIndex<uint8_t>;
}
