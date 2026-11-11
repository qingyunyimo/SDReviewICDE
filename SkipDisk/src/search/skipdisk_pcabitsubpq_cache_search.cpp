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
    size_t SSDIndex<T, TagT>::skipdisk_pcabitsubpq_cache_search(const T *query1, const uint64_t k_search,
                                            const uint64_t l_search, TagT *res_tags, float *distances,
                                            const uint64_t beam_width, QueryStats *stats) {
        QueryBuffer<T> *query_buf = pop_query_buf(query1);
        void *ctx = reader->get_ctx(IORING_SETUP_SQPOLL);  // use SQ polling only for pipe search.

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
        std::priority_queue<std::pair<float, uint32_t>,
                   std::vector<std::pair<float, uint32_t>>,
                   std::greater<std::pair<float, uint32_t>>> temp_candidates;
        float threshold = std::numeric_limits<float>::max();
        auto &visited = *(query_buf->visited);
        unsigned cur_list_size = 0;
        unsigned k = 0;
        std::vector<uint32_t>  beam_ids; 
        beam_ids.reserve(beam_width);
        std::vector<uint32_t> frontier;
        frontier.reserve(beam_width);
        int64_t cur_beam_width = std::min(32ul, beam_width);
        std::vector<float>  restored_navi_vec(navi_dim_, 0.0f);

        nbr_handler->initialize_query(query, query_buf);
        uint32_t best_medoid = medoid;
        nbr_handler->compute_dists(query_buf, &best_medoid, 1);
        retset[cur_list_size].id = best_medoid;
        retset[cur_list_size].distance = dist_scratch[0];
        retset[cur_list_size++].visited = false;
        visited.insert(best_medoid);
        //lambda used in search
        
        auto compute_and_push_nbrs = [&](char* node_buf, uint32_t &nk){
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

            if(nbers_cand_size){
                nbr_handler->compute_dists(query_buf, to_checked_nbrs.data(), nbers_cand_size);
                for(unsigned m = 0; m < nbers_cand_size; m++){
                    const int nbor_id = to_checked_nbrs[m];
                    const float nbor_dist = dist_scratch[m];
                    if (stats != nullptr) {
                        stats->n_cmps++;
                    }
                    if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search))
                        continue;
                    Neighbor nn(nbor_id, nbor_dist, true);
                    // Return position in sorted list where nn inserted
                    auto r = InsertIntoPool(retset.data(), cur_list_size, nn);  // may be overflow in retset...
                    if (cur_list_size < l_search) {
                        ++cur_list_size;
                        if (unlikely(cur_list_size >= retset.size())) {
                            retset.resize(2 * cur_list_size);
                        }
                    }
                    // nk logs the best position in the retset that was updated due to
                    // neighbors of n.
                    if (r < nk)
                        nk = r;
                }
            }
        };

        uint64_t l_k_search = k_search;
        auto compute_exact_dists_and_push = [&](const char *node_buf, const unsigned id) {
            float full_dist = l2_sqr((float*)query1, (float*) node_buf, aligned_dim);
            top_candidates.emplace(std::make_pair(full_dist, id));
            if(top_candidates.size() > l_k_search){
                top_candidates.pop();
                threshold = top_candidates.top().first;
            }
        };

        std::queue<io_t> on_flight_ios;
        auto send_read_req = [&](Neighbor &item) -> bool {
            item.flag = false;

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
            while(!temp_candidates.empty() && n_sent < n){
                auto cand = temp_candidates.top();
                temp_candidates.pop();
                if(cand.first > threshold){
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
                    beam_ids.push_back(retset[marker].id);
                    if(stats != nullptr){
                        stats->n_hops++;
                    }
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
                uint32_t node_id = beam_ids[i];
                auto cache_it = graph_cache_map_udm_.find(node_id);
                char* node_buf = cache_it->second.second;

                if(cache_it->second.first){
                    
                    bit_preserver_.loadFloatFromHigh16BitsAVX512(node_buf, (float*)restored_navi_vec.data());
                    float q2p = l2_sqr((float*)query1, (float*) restored_navi_vec.data(), navi_dim_);
                    q2p = std::sqrt(q2p);
                    float* x2p_ptr = (float*)(node_buf + navi_data_len_);
                    float x2p = *x2p_ptr;
                    float lb_dist = std::abs(q2p - x2p);
                    lb_dist = lb_dist * lb_dist;
                    temp_candidates.push(std::make_pair(lb_dist, node_id));

                    compute_and_push_nbrs(node_buf + navi_data_len_ + 4, nk);    
                }else{
                    temp_candidates.push(std::make_pair(0.0f, node_id)); // push to temp_candidates to be read later.

                    compute_and_push_nbrs(node_buf, nk); 
                }
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

        add_timer.reset();
        while(!on_flight_ios.empty() || (!temp_candidates.empty() && temp_candidates.top().first <= threshold)){
            auto [n_in, n_out] = poll_all();
            //if(n_in + n_out > 0){
                //std::cout<<n_in<<","<<n_out<<","<<on_flight_ios.size()<<";"<<std::endl;
            //}
            
            unsigned more_sent = 0;
            if((int64_t) on_flight_ios.size() < cur_beam_width) {
                more_sent = send_best_read_req(cur_beam_width - on_flight_ios.size());
                if(stats != nullptr){
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
        for(uint64_t i =0; i<k_search; i++){
            if(top_candidates.empty()){
                break;
            }
            res_tags[i] = top_candidates.top().second;
            
            if(distances != nullptr){
                distances[i] = top_candidates.top().first;
            }
            top_candidates.pop();
        }

        push_query_buf(query_buf);

        if (stats != nullptr) {
            stats->total_us = (double) query_timer.elapsed();
        }

        return 0;
    }

    template class SSDIndex<float>;
    template class SSDIndex<int8_t>;
    template class SSDIndex<uint8_t>;
}