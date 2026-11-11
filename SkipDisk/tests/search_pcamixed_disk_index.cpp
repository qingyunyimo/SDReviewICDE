#include "ssd_index.h"
#include "linux_aligned_file_reader.h"
#include "nbr/dummy_nbr.h"
#include "nbr/pq_nbr.h"
#include "utils/matrix.h"
#include "mem_utils.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <cstdlib>
#include <numeric>
#include <map>
#include <random>
#include <sstream>
#include <tuple>

static Matrix<float> read_matrix(const std::string &path) {
  std::ifstream src(path, std::ios::binary | std::ios::ate);
  if (!src) throw std::runtime_error("Cannot open matrix: " + path);
  const auto bytes = src.tellg();
  src.seekg(0);
  uint32_t shape[2];
  if (!src.read(reinterpret_cast<char *>(shape), 8) || !shape[0] || !shape[1] ||
      uint64_t(shape[0]) * shape[1] > (uint64_t(std::numeric_limits<int64_t>::max()) - 8) / 4 ||
      bytes != static_cast<std::streamoff>(8 + uint64_t(shape[0]) * shape[1] * 4))
    throw std::runtime_error("Invalid float32 bin matrix: " + path);
  Matrix<float> matrix(shape[0], shape[1]);
  if (!src.read(reinterpret_cast<char *>(matrix.data), uint64_t(shape[0]) * shape[1] * 4))
    throw std::runtime_error("Truncated matrix: " + path);
  return matrix;
}

static uint64_t positive(const char *s) {
  std::string value(s);
  size_t consumed = 0;
  if (value.empty() || value[0] == '-') throw std::invalid_argument("Expected a positive integer");
  auto n = std::stoull(value, &consumed);
  if (!n || consumed != value.size()) throw std::invalid_argument("Expected a positive integer");
  return n;
}

int main(int argc, char **argv) {
  std::string memory_base;
  std::string navigation_query_path;
  std::string fp32_filter_path, fp32_filter_dim;
  std::string estimation_lmd, estimation_epsilon;
  std::string cache_ids, cache_count_text;
  std::string candidate_frequency_prefix;
  bool quantized_only = false;
  bool trace_candidates = false;
  bool use_bound = true, compare_bounds = false, sweep = false;
  // Optional flags can appear in any order after the positional arguments.
  for (int i = 1; i < argc;) {
    const std::string flag = argv[i];
    int remove = 0;
    if (flag == "--navigation-query" && i + 1 < argc) {
      navigation_query_path=argv[i+1]; remove=2;
    }
    else if (flag == "--fp32-filter" && i + 2 < argc) {
      fp32_filter_path=argv[i+1]; fp32_filter_dim=argv[i+2]; remove=3;
    }
    else if (flag == "--filter-cache" && i + 2 < argc) {
      cache_ids = argv[i+1]; cache_count_text = argv[i+2]; remove = 3;
    }
    else if (flag == "--estimation" && i + 2 < argc) {
      estimation_lmd = argv[i+1]; estimation_epsilon = argv[i+2]; remove = 3;
    }
    else if (flag == "--no-bound") { use_bound = false; remove = 1; }
    else if (flag == "--compare-bounds") { compare_bounds = true; remove = 1; }
    else if (flag == "--sweep") { sweep = true; remove = 1; }
    else if (flag == "--memory-base" && i + 1 < argc) { memory_base = argv[i+1]; remove = 2; }
    else if (flag == "--quantized-only") { quantized_only = true; remove = 1; }
    else if (flag == "--trace-candidates") { trace_candidates = true; remove = 1; }
    else if (flag == "--candidate-frequency" && i + 1 < argc) {
      candidate_frequency_prefix = argv[i+1]; remove = 2;
    }
    if (remove) {
      for (int j = i; j + remove < argc; ++j) argv[j] = argv[j + remove];
      argc -= remove;
    } else ++i;
  }
  if (argc < 9 || argc > 11) {
    std::cerr << "Usage: " << argv[0]
              << " PREFIX QUERY.bin PCA_DIR_OR_- K L BEAM THREADS TRUTH.bin_OR_- [RESULT_PREFIX_OR_-] [SUBPQ_PREFIX] [--memory-base FULL_PCA_BASE.bin | --quantized-only]\n"
              << "Use '-' for PCA_DIR when queries are already PCA-transformed. L2 float32 only.\n"
              << "RAM/SSD options: --no-bound or --compare-bounds (1 repeat per mode; optional SKIPDISK_BENCH_REPEATS and SKIPDISK_BENCH_L_VALUES=50,100,200,400).\n";
    std::cerr << "Use --sweep with SKIPDISK_BENCH_L_VALUES for a single-mode RAM or SSD sweep.\n";
    std::cerr << "--trace-candidates saves expanded candidate IDs in generation order (diagnostic; affects timing).\n";
    std::cerr << "--navigation-query FILE supplies already-transformed subPQ queries with the same shape/row order; main queries/pivots/SSD remain in one common coordinate system. SSD subPQ only.\n";
    std::cerr << "--fp32-filter FULL_PCA_BASE.bin D loads an unquantized prefix for bound-only SSD filtering; requires SUBPQ_PREFIX.\n";
    std::cerr << "--estimation LMD.fvecs EPSILON.fvecs enables approximate mixed filtering (may reduce recall).\n";
    std::cerr << "--filter-cache IDS COUNT caches only the first COUNT selected pivots (0 allowed); requires SUBPQ_PREFIX. IDS='-' uses PREFIX_degree_sorted.id.\n";
    std::cerr << "--candidate-frequency PREFIX counts expanded candidates across queries and writes PREFIX_access_sorted.id and PREFIX_access_counts.bin; requires quantized-only subPQ, no truth, no warmup/sweep/trace.\n";
    return 1;
  }
  try {
    const std::string prefix = argv[1], query_path = argv[2], pca_dir = argv[3];
    if (!candidate_frequency_prefix.empty()) {
      if (!quantized_only || argc != 11 || trace_candidates || sweep || compare_bounds ||
          std::string(argv[8]) != "-" || std::getenv("SKIPDISK_BENCH_WARMUP"))
        throw std::invalid_argument("Candidate frequency requires quantized-only subPQ, no truth, no warmup/sweep/trace");
      for (const auto *suffix : {"_access_sorted.id", "_access_counts.bin"})
        if (std::filesystem::exists(candidate_frequency_prefix + suffix))
          throw std::invalid_argument("Refusing to overwrite candidate frequency output");
    }
    if (trace_candidates && (argc < 10 || std::string(argv[9]) == "-"))
      throw std::invalid_argument("Candidate tracing requires a result prefix");
    const uint64_t k = positive(argv[4]), beam = positive(argv[6]);
    uint64_t l = positive(argv[5]);
    const uint64_t threads = positive(argv[7]);
    if (k > l || l > UINT32_MAX - 1 || beam > MAX_N_SECTOR_READS || threads > 1024)
      throw std::invalid_argument("Require K <= L, BEAM <= 128, THREADS <= 1024");
    Matrix<float> queries = read_matrix(query_path);
    if (!queries.n || !queries.d) throw std::invalid_argument("Empty query file");
    if (pca_dir != "-") {
      std::string projection = (std::filesystem::path(pca_dir) / "3.bin").string();
      std::string mean_path = (std::filesystem::path(pca_dir) / "pcamean.bin").string();
      Matrix<float> p = read_matrix(projection);
      Matrix<float> mean = read_matrix(mean_path);
      if (p.n != queries.d || p.d != queries.d || mean.n != 1 || mean.d != queries.d)
        throw std::invalid_argument("PCA matrix/mean dimensions must match queries");
      for (size_t i = 0; i < queries.n; ++i)
        for (size_t d = 0; d < queries.d; ++d)
          queries.data[i * queries.d + d] -= mean.data[d];
      queries = mul(queries, p);
    }
    for (size_t i = 0; i < queries.n * queries.d; ++i)
      if (!std::isfinite(queries.data[i])) throw std::invalid_argument("Non-finite query");

    omp_set_num_threads(static_cast<int>(threads));
    std::shared_ptr<AlignedFileReader> reader = std::make_shared<LinuxAlignedFileReader>();
    const bool subpq = argc == 11;
    std::unique_ptr<Matrix<float>> navigation_queries;
    if (!navigation_query_path.empty()) {
      if (!subpq || !memory_base.empty() || quantized_only || !estimation_lmd.empty())
        throw std::invalid_argument("Independent navigation queries require exact SSD subPQ search");
      navigation_queries = std::make_unique<Matrix<float>>(read_matrix(navigation_query_path));
      if (navigation_queries->n != queries.n || navigation_queries->d != queries.d)
        throw std::invalid_argument("Independent navigation query shape mismatch");
      for (size_t i=0; i<queries.n*queries.d; ++i)
        if (!std::isfinite(navigation_queries->data[i])) throw std::invalid_argument("Non-finite navigation query");
      std::cout << "IndependentNavigationQuery=1 ProjectionOutsideTiming=1\n";
    }
    if (!fp32_filter_path.empty() && (!subpq || !memory_base.empty() || quantized_only ||
        !use_bound || compare_bounds || !cache_ids.empty() || !estimation_lmd.empty()))
      throw std::invalid_argument("FP32 filtering requires SSD subPQ bound-only search without cache/estimation");
    std::unique_ptr<pipeann::AbstractNeighbor<float>> neighbor;
    if (subpq) neighbor = std::make_unique<pipeann::PQNeighbor<float>>();
    else neighbor = std::make_unique<pipeann::DummyNeighbor<float>>();
    pipeann::SSDIndex<float> index(pipeann::Metric::L2, reader, neighbor.get(), false);
    const bool legacy = std::getenv("SKIPDISK_BENCH_LEGACY_BF16") &&
                        std::string(std::getenv("SKIPDISK_BENCH_LEGACY_BF16")) == "1";
    if (!estimation_lmd.empty() && (legacy || quantized_only || !use_bound || compare_bounds))
      throw std::invalid_argument("--estimation requires mixed filtering; incompatible with legacy, quantized-only, no-bound and compare-bounds");
    if (legacy && subpq) throw std::invalid_argument("Legacy benchmark and mixed subPQ are mutually exclusive");
    if (legacy && trace_candidates) throw std::invalid_argument("Candidate tracing requires a mixed index");
    if (quantized_only && (legacy || !memory_base.empty()))
      throw std::invalid_argument("Quantized-only mode cannot be combined with legacy or memory-base");
    if (legacy && !memory_base.empty()) throw std::invalid_argument("Memory mode requires mixed pivots");
    if ((!use_bound || compare_bounds) && (legacy || quantized_only))
      throw std::invalid_argument("Bound ablation requires a mixed index and exact reranking");
    uint64_t cache_count = UINT64_MAX;
    if (!cache_count_text.empty()) {
      if (!subpq || legacy || cache_count_text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("Filter cache requires independent subPQ and a nonnegative integer count");
      size_t consumed = 0;
      cache_count = std::stoull(cache_count_text, &consumed);
      if (consumed != cache_count_text.size() || cache_count >= UINT32_MAX)
        throw std::invalid_argument("Invalid filter cache count");
      if (cache_ids == "-") cache_ids = prefix + "_degree_sorted.id";
    }
    if (subpq) index.load_pca_mixed_subpq_index(prefix.c_str(), argv[10], static_cast<uint32_t>(threads), cache_ids, cache_count);
    else if (legacy) index.load_pca_bit_index(prefix.c_str(), static_cast<uint32_t>(threads));
    else index.load_pca_mixed_index(prefix.c_str(), static_cast<uint32_t>(threads));
    if (!estimation_lmd.empty()) index.load_mixed_estimation(estimation_lmd, estimation_epsilon);
    if (!fp32_filter_path.empty()) {
      const auto d=positive(fp32_filter_dim.c_str());
      if (d>queries.d) throw std::invalid_argument("FP32 filtering dimension exceeds query dimension");
      index.load_fp32_filter(fp32_filter_path,static_cast<uint32_t>(d));
    }
    std::cout << "FilteringMode=" << (index.mixed_estimation_ ? "MixedEstimation" : (use_bound ? "ExactBound" : "NoFilter"))
              << " EstimationRatio=" << index.mixed_estimation_ratio_
              << " FilteringDim=" << index.navi_dim_ << "\n";
    if ((!legacy && (queries.d != index.mixed_pivot_->full_dim || k > index.mixed_pivot_->points)) ||
        (legacy && queries.d < index.navi_dim_))
      throw std::invalid_argument("Query/index dimension mismatch or K exceeds point count");
    if (!memory_base.empty()) {
      // Check size/shape before using the shared base loader. This is a full
      // heap load, not a file-backed mmap; no vector file is read per query.
      std::ifstream base(memory_base, std::ios::binary | std::ios::ate);
      if (!base) throw std::invalid_argument("Cannot open memory base");
      const auto bytes = base.tellg();
      base.seekg(0);
      uint32_t shape[2] = {};
      if (!base.read(reinterpret_cast<char *>(shape), 8) || shape[0] != index.mixed_pivot_->points ||
          shape[1] != index.mixed_pivot_->full_dim ||
          bytes != static_cast<std::streamoff>(8 + uint64_t(shape[0]) * shape[1] * sizeof(float)))
        throw std::invalid_argument("Memory base shape/size mismatch");
      if (index.load_base_vectors(memory_base) != 0) throw std::runtime_error("Loading memory base failed");
      for (size_t i = 0; i < index.base_vecs_num_ * index.base_vecs_dim_; ++i)
        if (!std::isfinite(index.base_vecs_[i])) throw std::invalid_argument("Non-finite memory base");
      std::cout << "SearchStorage=RAM (BEAM ignored; no query sector I/O)\n";
    }
    if (memory_base.empty() && !quantized_only)
      std::cout << "SearchStorage=SSD DirectIO=1 Beam=" << beam << '\n';
    std::vector<std::tuple<uint64_t,bool,unsigned>> jobs;
    if (compare_bounds || sweep) {
      std::vector<uint64_t> lengths{l};
      if (const char *values = std::getenv("SKIPDISK_BENCH_L_VALUES")) {
        lengths.clear(); std::stringstream ss(values); std::string value;
        while (std::getline(ss,value,',')) {
          auto length = positive(value.c_str());
          if (length < k || length > UINT32_MAX - 1) throw std::invalid_argument("Invalid sweep L");
          lengths.push_back(length);
        }
        if (lengths.empty()) throw std::invalid_argument("Empty L sweep");
      }
      const uint64_t repeats = std::getenv("SKIPDISK_BENCH_REPEATS") ? positive(std::getenv("SKIPDISK_BENCH_REPEATS")) : 1;
      if (repeats > 1000) throw std::invalid_argument("Too many benchmark repeats");
      for (auto length : lengths) for (unsigned r=1; r<=repeats; ++r) {
        if (compare_bounds) for (bool filter : {true,false}) jobs.emplace_back(length,filter,r);
        else jobs.emplace_back(length,use_bound,r);
      }
      std::mt19937 rng(42); std::shuffle(jobs.begin(),jobs.end(),rng);
      if (const char *skip = std::getenv("SKIPDISK_BENCH_SKIP_PAIRS")) {
        std::set<std::string> completed; std::stringstream ss(skip); std::string value;
        while (std::getline(ss,value,',')) completed.insert(value);
        jobs.erase(std::remove_if(jobs.begin(),jobs.end(),[&](const auto &job) {
          return completed.count(std::to_string(std::get<0>(job))+":"+std::to_string(std::get<1>(job))) != 0;
        }),jobs.end());
      }
    } else jobs.emplace_back(l,use_bound,1);
    struct Reference {
      std::vector<uint32_t> ids;
      std::vector<float> distances;
      std::vector<pipeann::QueryStats> stats;
    };
    std::map<uint64_t,Reference> references;
    bool any_short = false;
    for (auto job : jobs) {
    unsigned repeat;
    std::tie(l,use_bound,repeat) = job;
    std::cout << "RUN_BEGIN Bound=" << use_bound << " L=" << l << " Repeat=" << repeat << std::endl;
    std::vector<uint32_t> ids(queries.n * k, UINT32_MAX);
    if (quantized_only)
      std::cout << "SearchStorage=QuantizedOnly ResultDistance="
                << (subpq ? "subPQ_ADC_squared_L2" : "mixed_pivot_squared_L2")
                << " Candidates=expanded_before_bound_pruning\n";
    std::vector<float> distances(queries.n * k, std::numeric_limits<float>::infinity());
    std::vector<pipeann::QueryStats> stats(queries.n);
    std::vector<std::vector<uint32_t>> candidate_traces(trace_candidates ? queries.n : 0);
    // One shared 64-bit counter per indexed point; traces live only for the
    // duration of one query, rather than growing with the number of queries.
    std::vector<uint64_t> access_counts(candidate_frequency_prefix.empty() ? 0 : index.mixed_pivot_->points, 0);
    uint64_t frequency_queries_done = 0;
    if (trace_candidates) {
      std::cout << "CandidateTrace=expanded_before_bound_or_IO TimingIsDiagnostic=1\n";
      for (size_t q=0; q<queries.n; ++q) stats[q].candidate_trace = &candidate_traces[q];
    }
    std::vector<size_t> counts(queries.n);
    std::vector<double> latency(queries.n);
    auto run_query = [&](size_t q, pipeann::QueryStats *stat, uint32_t *out_ids, float *out_distances) {
      if (quantized_only)
        return index.skipdisk_pcamixed_quantized_search(queries.data + q * queries.d, k, l,
                                                       out_ids, out_distances, stat);
      if (!memory_base.empty())
        return index.skipdisk_pcamixed_memory_search(queries.data + q * queries.d, k, l,
                                                     out_ids, out_distances, stat, use_bound);
      if (legacy) {
        index.skipdisk_pcabit_search(queries.data + q * queries.d, k, l, out_ids, out_distances, beam, stat);
        return static_cast<size_t>(std::count_if(out_ids, out_ids + k,
                                               [](uint32_t id) { return id != UINT32_MAX; }));
      }
      return index.skipdisk_pcamixed_search(queries.data + q * queries.d, k, l, out_ids, out_distances,
                                           beam, stat, false, false, use_bound,
                                           navigation_queries ? navigation_queries->data + q * queries.d : nullptr);
    };
    const size_t warmup = std::getenv("SKIPDISK_BENCH_WARMUP") ?
        std::min<size_t>(positive(std::getenv("SKIPDISK_BENCH_WARMUP")), queries.n) : 0;
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t q = 0; q < warmup; ++q) {
      std::vector<uint32_t> warm_ids(k, UINT32_MAX);
      std::vector<float> warm_distances(k);
      run_query(q, nullptr, warm_ids.data(), warm_distances.data());
    }
    const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t q = 0; q < queries.n; ++q) {
      std::vector<uint32_t> frequency_trace;
      if (!candidate_frequency_prefix.empty()) stats[q].candidate_trace = &frequency_trace;
      const auto query_start = std::chrono::steady_clock::now();
      counts[q] = run_query(q, &stats[q], ids.data() + q * k, distances.data() + q * k);
      latency[q] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - query_start).count();
      if (!candidate_frequency_prefix.empty()) {
        for (uint32_t id : frequency_trace) {
#pragma omp atomic update
          access_counts[id] += 1;
        }
        stats[q].candidate_trace = nullptr;
        uint64_t done;
#pragma omp atomic capture
        done = ++frequency_queries_done;
        if (done % 1000 == 0 || done == queries.n) {
#pragma omp critical(candidate_frequency_progress)
          std::cout << "CandidateFrequencyQueries=" << done << '/' << queries.n << std::endl;
        }
      }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double ios = 0, candidates = 0, exact = 0, bounds = 0, navigation = 0;
    size_t short_queries = 0;
    for (size_t q = 0; q < queries.n; ++q) {
      ios += stats[q].n_ios;
      candidates += stats[q].n_hops;
      exact += stats[q].n_exact;
      bounds += stats[q].n_bounds;
      navigation += stats[q].n_cmps;
      if (counts[q] != k) ++short_queries;
    }
    if (!candidate_frequency_prefix.empty()) {
      const uint64_t total = std::accumulate(access_counts.begin(), access_counts.end(), uint64_t(0));
      uint64_t expanded = 0;
      for (const auto &stat : stats) expanded += static_cast<uint64_t>(stat.n_hops);
      if (total != expanded || ios || exact || bounds || short_queries)
        throw std::runtime_error("Candidate frequency must match expansion counts with no reranking, filtering or query I/O");
      std::vector<uint32_t> ranking(access_counts.size());
      std::iota(ranking.begin(), ranking.end(), uint32_t(0));
      std::sort(ranking.begin(), ranking.end(), [&](uint32_t a, uint32_t b) {
        return access_counts[a] != access_counts[b] ? access_counts[a] > access_counts[b] : a < b;
      });
      // Same single-word uint32 header as degree_sorted.id and --filter-cache.
      std::ofstream out(candidate_frequency_prefix + "_access_sorted.id", std::ios::binary);
      out.exceptions(std::ofstream::failbit | std::ofstream::badbit);
      const uint32_t points = ranking.size();
      out.write(reinterpret_cast<const char *>(&points), sizeof(points));
      out.write(reinterpret_cast<const char *>(ranking.data()), ranking.size() * sizeof(uint32_t));
      out.close();
      pipeann::save_bin<uint64_t>(candidate_frequency_prefix + "_access_counts.bin",
                                 access_counts.data(), access_counts.size(), 1);
      std::cout << "CandidateFrequencyTotal=" << total << " AccessedPoints="
                << std::count_if(access_counts.begin(), access_counts.end(), [](uint64_t n) { return n != 0; })
                << " RankingTieBreak=IDAscending\n";
    }
    std::cout << "Queries=" << queries.n << " L=" << l << " QPS=" << queries.n / seconds
              << " MeanIO=" << ios / queries.n << " ShortResults=" << short_queries
              << " MeanCandidates=" << candidates / queries.n
              << " MeanExact=" << exact / queries.n << " MeanBounds=" << bounds / queries.n
              << " MeanNavigation=" << navigation / queries.n
              << " PrunedFraction=" << (candidates ? (candidates-exact)/candidates : 0) << '\n';
    std::sort(latency.begin(), latency.end());
    auto percentile = [&](double p) { return latency[static_cast<size_t>(std::ceil(p * latency.size())) - 1]; };
    std::cout << "MeanLatUs=" << std::accumulate(latency.begin(), latency.end(), 0.0) / queries.n
              << " P50LatUs=" << percentile(0.50) << " P95LatUs=" << percentile(0.95)
              << " P99LatUs=" << percentile(0.99) << " RSSMiB=" << getCurrentRSS() / 1048576.0
              << " PeakRSSMiB=" << getPeakRSS() / 1048576.0 << " Warmup=" << warmup << '\n';
    if (std::string(argv[8]) != "-") {
      unsigned *raw_ids = nullptr;
      float *raw_distances = nullptr;
      size_t rows = 0, cols = 0;
      pipeann::load_truthset(argv[8], raw_ids, raw_distances, rows, cols);
      std::unique_ptr<unsigned[]> truth(raw_ids);
      std::unique_ptr<float[]> truth_distances(raw_distances);
      if (rows != queries.n || cols < k) throw std::invalid_argument("Ground truth shape mismatch");
      size_t matches = 0;
      for (size_t q = 0; q < queries.n; ++q) {
        std::set<unsigned> expected(truth.get() + q * cols, truth.get() + q * cols + k);
        for (size_t j = 0; j < counts[q]; ++j) matches += expected.count(ids[q * k + j]);
      }
      std::cout << "Recall@" << k << '=' << double(matches) / (queries.n * k) << '\n';
    }
    if (compare_bounds) {
      const auto found = references.find(l);
      if (found == references.end()) references.emplace(l,Reference{ids,distances,stats});
      else {
        if (ids != found->second.ids || distances != found->second.distances)
          throw std::runtime_error("Bound modes produced different exact results");
        for (size_t q=0; q<queries.n; ++q)
          if (stats[q].n_hops != found->second.stats[q].n_hops ||
              stats[q].n_cmps != found->second.stats[q].n_cmps ||
              stats[q].candidate_hash != found->second.stats[q].candidate_hash)
            throw std::runtime_error("Bound modes changed graph traversal");
      }
      std::cout << "ResultsAndTraversalVerified=1\n";
    }
    if (argc >= 10 && std::string(argv[9]) != "-") {
      const std::string suffix = (compare_bounds || sweep) ? "_L"+std::to_string(l)+"_B"+std::to_string(use_bound)+"_R"+std::to_string(repeat) : "";
      pipeann::save_bin<uint32_t>(std::string(argv[9]) + suffix + "_ids.bin", ids.data(), queries.n, k);
      pipeann::save_bin<float>(std::string(argv[9]) + suffix + "_dists.bin", distances.data(), queries.n, k);
      std::vector<uint64_t> traversal(queries.n);
      for (size_t q=0; q<queries.n; ++q) traversal[q] = stats[q].candidate_hash;
      pipeann::save_bin<uint64_t>(std::string(argv[9]) + suffix + "_traversal.bin", traversal.data(), queries.n, 1);
      if (trace_candidates) {
        std::vector<uint64_t> offsets(queries.n + 1, 0);
        for (size_t q=0; q<queries.n; ++q) {
          if (candidate_traces[q].size() != static_cast<size_t>(stats[q].n_hops))
            throw std::runtime_error("Candidate trace/count mismatch");
          offsets[q+1] = offsets[q] + candidate_traces[q].size();
        }
        if (offsets.back() > UINT32_MAX) throw std::runtime_error("Candidate trace exceeds bin format limit");
        std::vector<uint32_t> flat;
        flat.reserve(offsets.back());
        for (const auto &trace : candidate_traces) flat.insert(flat.end(),trace.begin(),trace.end());
        pipeann::save_bin<uint64_t>(std::string(argv[9])+suffix+"_candidate_offsets.bin", offsets.data(), offsets.size(), 1);
        pipeann::save_bin<uint32_t>(std::string(argv[9])+suffix+"_candidate_ids.bin", flat.data(), flat.size(), 1);
      }
    }
    any_short |= short_queries != 0;
    std::cout << "RUN_END Bound=" << use_bound << " L=" << l << " Repeat=" << repeat << std::endl;
    }
    return any_short ? 2 : 0;
  } catch (const std::exception &e) {
    std::cerr << "Mixed-pivot search: " << e.what() << '\n';
    return 1;
  }
}
