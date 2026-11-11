#include <cstring>
#include <omp.h>
#include <ssd_index.h>
#include <string.h>
#include <time.h>
#include <iomanip>

#include "utils/log.h"
#include "nbr/abstract_nbr.h"
#include "nbr/pq_nbr.h"
#include "nbr/rabitq_nbr.h"
#include "utils/timer.h"
#include "utils.h"
#include "aux_utils.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#include "mem_utils.h"
#include "utils/matrix.h"

#define WARMUP false
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_VECTORIZE

void print_stats(std::string category, std::vector<float> percentiles, std::vector<float> results) {
  std::cout << std::setw(20) << category << ": " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    std::cout << std::setw(8) << percentiles[s] << "%";
  }
  std::cout << std::endl;
  std::cout << std::setw(22) << " " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    std::cout << std::setw(9) << results[s];
  }
  std::cout << std::endl;
}

void load_data(const char* filename, float*& data, unsigned& num,
               unsigned& dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
//  std::cout << "data dimension: " << dim << std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim + 1) / 4);
  data = new float[num * dim];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}

template<typename T>
int search_disk_index(int argc, char **argv) {
  // load query bin
  T *query = nullptr;
  unsigned *gt_ids = nullptr;
  float *gt_dists = nullptr;
  uint32_t *tags = nullptr;
  size_t query_num, query_dim, gt_num, gt_dim;
  std::vector<uint64_t> Lvec;

  bool tags_flag = true;

  int index = 2;
  std::string index_prefix_path(argv[index++]);
  std::string data_name(argv[index++]);
  uint32_t num_threads = std::atoi(argv[index++]);
  uint32_t beamwidth = std::atoi(argv[index++]);
  uint32_t num_nodes_to_cache = std::atoi(argv[index++]);
  std::string query_bin(argv[index++]);
  std::string index_directory(argv[index++]);
  std::string truthset_bin(argv[index++]);
  uint64_t recall_at = std::atoi(argv[index++]);
  std::string dist_metric(argv[index++]);
  std::string nbr_type = argv[index++];
  int search_mode = std::atoi(argv[index++]);
  bool use_page_search = search_mode != 0;
  uint32_t mem_L = std::atoi(argv[index++]);

  pipeann::Metric m = dist_metric == "cosine" ? pipeann::Metric::COSINE : pipeann::Metric::L2;
  if (dist_metric != "l2" && m == pipeann::Metric::L2) {
    std::cout << "Unknown distance metric: " << dist_metric << ". Using default(L2) instead." << std::endl;
  }

  std::string disk_index_tag_file = index_prefix_path + "_disk.index.tags";

  bool calc_recall_flag = false;

  for (int ctr = index; ctr < argc; ctr++) {
    uint64_t curL = std::atoi(argv[ctr]);
    if (curL >= recall_at)
      Lvec.push_back(curL);
  }

  if (Lvec.size() == 0) {
    std::cout << "No valid Lsearch found. Lsearch must be at least recall_at" << std::endl;
    return -1;
  }

  std::cout << "Search parameters: #threads: " << num_threads << ", ";
  if (beamwidth <= 0)
    std::cout << "beamwidth to be optimized for each L value" << std::endl;
  else
    std::cout << " beamwidth: " << beamwidth << std::endl;

  pipeann::load_bin<T>(query_bin, query, query_num, query_dim);
  std::cout<< "Loaded query data. Num: " << query_num << " Dim: " << query_dim << std::endl;
  if (file_exists(truthset_bin)) {
    pipeann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim, &tags);
    if (gt_num != query_num) {
      std::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
    }
    calc_recall_flag = true;
  }

  std::shared_ptr<AlignedFileReader> reader = nullptr;
  reader.reset(new LinuxAlignedFileReader());
  pipeann::AbstractNeighbor<T> *nbr_handler = nullptr;
  if (nbr_type == "pq") {
    nbr_handler = new pipeann::PQNeighbor<T>();
  } else if (nbr_type == "rabitq") {
    nbr_handler = new pipeann::RaBitQNeighbor<T>();
  } else {
    std::cout << "Unknown nbr type: " << nbr_type << std::endl;
    return -1;
  }
  std::unique_ptr<pipeann::SSDIndex<T>> _pFlashIndex(new pipeann::SSDIndex<T>(m, reader, nbr_handler, tags_flag));

  //int res = _pFlashIndex->load_skip_navi_bit(num_threads, index_prefix_path.c_str());
  int res = _pFlashIndex->load_pca_index(index_prefix_path.c_str(), num_threads, true, use_page_search);
  
  if (res != 0) {
    return res;
  }

    //for PCA trans
    std::string mean_path = index_directory + "pcamean.bin";
    std::string trans_path = index_directory + "3.bin";
    Matrix<float> Q(query_bin.data());
    Matrix<float> P(trans_path.data());
    float *mean_data = NULL;
    size_t mean_num = 0; // one row
    size_t mean_dim = 0; // how many column
    pipeann::load_bin<float>(mean_path, mean_data, mean_num, mean_dim);

    size_t test_query_num = Q.n;
    size_t test_query_dim = Q.d;

    #ifdef EIGEN_VECTORIZE
        std::cout << "Using EIGEN_VECTORIZE!" << std::endl;
    #else
        std::cout << "EIGEN_VECTORIZE disabled!" << std::endl;
    #endif

    //timer
    auto start = std::chrono::high_resolution_clock::now();
        
    for(size_t i = 0; i<test_query_num; i++){
        for(size_t j = 0; j<test_query_dim; j++){
            Q.data[i*test_query_dim + j] = Q.data[i*test_query_dim + j] -  mean_data[j];
        }
    }

    Q = mul(Q, P);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    
    std::cout << "PCA transformation time: " << duration.count() / test_query_num << " microseconds" << std::endl;
    T *query_pca = (T*) Q.data;
  //cache the nodes for pipe search
  //_pFlashIndex->cache_for_pipe_search(index_prefix_path, num_nodes_to_cache);

  if (mem_L != 0) {
    auto mem_index_path = index_prefix_path + "_mem.index";
    LOG(INFO) << "Load memory index " << mem_index_path << " " << query_dim;
    _pFlashIndex->load_mem_index(m, query_dim, mem_index_path);
  }

  std::cout << "Actual memory usage: " << getCurrentRSS() / (1 << 20) << " Mb \n";

  omp_set_num_threads(num_threads);

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<uint32_t>> query_result_tags(Lvec.size());
  std::vector<std::vector<float>> query_result_dists(Lvec.size());

  auto run_tests = [&](uint32_t test_id, bool output) {
    pipeann::QueryStats *stats = new pipeann::QueryStats[query_num];
    uint64_t L = Lvec[test_id];

    query_result_ids[test_id].resize(recall_at * query_num);
    query_result_dists[test_id].resize(recall_at * query_num);
    query_result_tags[test_id].resize(recall_at * query_num);

    std::vector<uint64_t> query_result_tags_64(recall_at * query_num);
    std::vector<uint32_t> query_result_tags_32(recall_at * query_num);
    auto s = std::chrono::high_resolution_clock::now();
    
    if (search_mode == SearchMode::SKIPDISK_SEARCH) {

      for (int64_t i = 0; i < (int64_t) query_num; i++) {
        _pFlashIndex->skipdisk_pca_search(query_pca + (i * query_dim), (uint64_t) recall_at, (uint64_t) L,
                                  query_result_tags_32.data() + (i * recall_at),
                                  query_result_dists[test_id].data() + (i * recall_at), (uint64_t) beamwidth,
                                  stats + i);
      }
      //pipe_search  navikip_search pipe_pq_cache_search pipe_pq_cache_search1
    } else {
      std::cout << "Unknown search mode: " << search_mode << std::endl;
      exit(-1);
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    float qps = (float) ((1.0 * (double) query_num) / (1.0 * (double) diff.count()));

    pipeann::convert_types<uint32_t, uint32_t>(query_result_tags_32.data(), query_result_tags[test_id].data(),
                                               (size_t) query_num, (size_t) recall_at);

    float mean_latency = (float) pipeann::get_mean_stats(
        stats, query_num, [](const pipeann::QueryStats &stats) { return stats.total_us; });

    float latency_999 = (float) pipeann::get_percentile_stats(
        stats, query_num, 0.999f, [](const pipeann::QueryStats &stats) { return stats.total_us; });

    float mean_hops = (float) pipeann::get_mean_stats(stats, query_num,
                                                      [](const pipeann::QueryStats &stats) { return stats.n_hops; });
    auto mean_cpuus = (float) pipeann::get_mean_stats(stats, query_num,
                                                         [](const pipeann::QueryStats &stats) { return stats.cpu_us; });

    float mean_ios =
        (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.n_ios; });
    float mean_addios =
        (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.n_addio; });
    float mean_addtime =
        (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.n_addtime; });

    delete[] stats;

    if (output) {
      float recall = 0;
      if (calc_recall_flag) {
        /* Attention: in SPACEV, there may be multiple vectors with the same distance,
          which may cause lower than expected recall@1 (?) */
        recall = (float) pipeann::calculate_recall((uint32_t) query_num, gt_ids, gt_dists, (uint32_t) gt_dim,
                                                   query_result_tags[test_id].data(), (uint32_t) recall_at,
                                                   (uint32_t) recall_at);
      }

      std::cout << std::setw(6) << L << std::setw(12) << beamwidth << std::setw(12) << qps << std::setw(12)
                << mean_latency << std::setw(12) << latency_999 << std::setw(12) << mean_hops << std::setw(12)
                << mean_ios << std::setw(12) << mean_addios << std::setw(12) << mean_cpuus << std::setw(12) 
                << mean_addtime;
      if (calc_recall_flag) {
        std::cout << std::setw(12) << recall << std::endl;
      }
      /*
      bool file_output_flag = true;
      if (file_output_flag) {
        int saved_stdout = dup(STDOUT_FILENO);
        std::string result_output_file = "/home/ubuntu/results/" + std::string("pcanavi_") + data_name + std::to_string(recall_at) + ".log";
        FILE* fresult = freopen(result_output_file.c_str(), "a", stdout);
        if(fresult == nullptr) {
          std::cout << "Error opening result output file" << std::endl;
        }
        std::cout << L << " " << recall << " " << mean_latency << " " << latency_999 << std::endl; 
        fflush(stdout); 
        dup2(saved_stdout, STDOUT_FILENO);
        close(saved_stdout);
        if(recall > 99.5){
          exit(0);
        }
      }
      */
    }
  };

  // LOG(INFO) << "Use two ANNS for warming up...";
  // uint32_t prev_L = Lvec[0];
  // Lvec[0] = 200;
  // run_tests(0, false);
  // run_tests(0, false);
  // Lvec[0] = prev_L;
  // LOG(INFO) << "Warming up finished.";

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
   std::cout << std::setw(6) << "L" << std::setw(12) << "I/O Width" << std::setw(12) << "QPS" << std::setw(12)
            << "AvgLat(us)" << std::setw(12) << "P99 Lat" << std::setw(12) << "Mean Hops" << std::setw(12) << "Mean IOs"
            << std::setw(12) << "Add IOs" << std::setw(12) << "CPU(us)" << std::setw(12) << "Add Time(us)" << std::setw(12);
  if (calc_recall_flag) {
    std::cout << std::setw(12) << recall_string << std::endl;
  } else
    std::cout << std::endl;
  std::cout << "=============================================="
               "=============================================="
               "================================="
            << std::endl;
  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    run_tests(test_id, true);
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 12) {
    // tags == 1!
    std::cout << "Usage: " << argv[0]
              << " <index_type (float/int8/uint8)>  <index_prefix_path>"
                 " <num_threads>  <pipeline width> "
                 " <query_file.bin>  <truthset.bin (use \"null\" for none)> "
                 " <K> <similarity (cosine/l2)> <nbr_type (pq/rabitq)>"
                 " <search_mode(0 for beam search / 1 for page search / 2 for pipe search)> <mem_L (0 means not "
                 "using mem index)> <L1> [L2] etc."
              << std::endl;
    exit(-1);
  }

  if (std::string(argv[1]) == std::string("float"))
    search_disk_index<float>(argc, argv);
  else if (std::string(argv[1]) == std::string("int8"))
    search_disk_index<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    search_disk_index<uint8_t>(argc, argv);
  else
    std::cout << "Unsupported index type. Use float or int8 or uint8" << std::endl;
}
