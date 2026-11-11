#include "ssd_index.h"
#include "nbr/pq_nbr.h"

#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace pipeann {
namespace {
// The existing PQNeighbor loader expects DiskANN's four-offset format.
// Validate its offsets before its unchecked table construction is reached.
void validate_subpq_pivots(const std::string &path, uint32_t chunks) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::invalid_argument("Missing subPQ pivot file");
  const auto bytes = in.tellg();
  in.seekg(0);
  uint32_t shape[2];
  uint64_t offsets[4];
  if (!in.read(reinterpret_cast<char *>(shape), 8) || shape[0] != 4 || shape[1] != 1 ||
      !in.read(reinterpret_cast<char *>(offsets), sizeof(offsets)))
    throw std::invalid_argument("Expected DiskANN four-offset subPQ pivots");
  uint32_t dim = 0;
  for (size_t part = 0; part < 3; ++part) {
    if (offsets[part] < 40 || offsets[part] > uint64_t(bytes) ||
        offsets[part + 1] <= offsets[part] || offsets[part + 1] > uint64_t(bytes))
      throw std::invalid_argument("Invalid subPQ section offset");
    in.seekg(offsets[part]);
    if (!in.read(reinterpret_cast<char *>(shape), 8))
      throw std::invalid_argument("Truncated subPQ section");
    if (part == 0) {
      dim = shape[1];
      if (shape[0] != 256 || !dim || dim > 65536 || chunks > dim)
        throw std::invalid_argument("Invalid subPQ dimension or center count");
    } else if (shape[1] != 1 || shape[0] != (part == 1 ? dim : chunks + 1)) {
      throw std::invalid_argument("Invalid subPQ centroid/chunk shape");
    }
    const uint64_t count = uint64_t(shape[0]) * shape[1];
    if (8 + count * 4 > offsets[part + 1] - offsets[part])
      throw std::invalid_argument("Truncated subPQ section data");
    if (part < 2) {
      std::vector<float> values(count);
      if (!in.read(reinterpret_cast<char *>(values.data()), count * 4))
        throw std::invalid_argument("Truncated subPQ floats");
      for (float v : values) if (!std::isfinite(v))
        throw std::invalid_argument("Non-finite subPQ center");
    } else {
      std::vector<uint32_t> bounds(count);
      if (!in.read(reinterpret_cast<char *>(bounds.data()), count * 4) ||
          bounds.front() != 0 || bounds.back() != dim)
        throw std::invalid_argument("Invalid subPQ chunk boundaries");
      for (size_t c = 1; c < bounds.size(); ++c)
        if (bounds[c] <= bounds[c - 1]) throw std::invalid_argument("Empty or unordered subPQ chunk");
    }
  }
}
}  // namespace

template<typename T, typename TagT>
void SSDIndex<T, TagT>::load_fp32_filter(const std::string &path, uint32_t dim) {
  if (!mixed_pivot_ || !mixed_subpq_navigation_ || mixed_cache_ || mixed_estimation_ ||
      !dim || dim > data_dim || !std::is_same<T, float>::value)
    throw std::invalid_argument("FP32 filtering requires full-point subPQ L2 search and valid prefix");
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::invalid_argument("Cannot open FP32 filtering source");
  const auto bytes = in.tellg(); in.seekg(0);
  uint32_t shape[2] = {};
  if (!in.read(reinterpret_cast<char *>(shape),8) || shape[0] != num_points || shape[1] != data_dim ||
      bytes != std::streamoff(8 + uint64_t(shape[0])*shape[1]*4))
    throw std::invalid_argument("FP32 filtering source shape/size mismatch");
  std::vector<float> values(size_t(num_points)*dim), block(size_t(4096)*data_dim);
  for (size_t lo=0; lo<num_points; lo+=4096) {
    const size_t count=std::min(size_t(4096),size_t(num_points)-lo);
    MixedPivotCodec::read(in,block.data(),count*data_dim*sizeof(float));
    for (size_t i=0;i<count;++i) {
      for (size_t j=0;j<dim;++j)
        if (!std::isfinite(block[i*data_dim+j])) throw std::invalid_argument("Non-finite FP32 pivot");
      std::copy_n(block.data()+i*data_dim,dim,values.data()+(lo+i)*dim);
    }
  }
  fp32_filter_data_.swap(values); fp32_filter_dim_=dim;
  LOG(INFO) << "FP32FilterDim=" << dim << " FP32FilterBytes=" << fp32_filter_data_.size()*4;
}

template<typename T, typename TagT>
int SSDIndex<T, TagT>::load_pca_mixed_subpq_index(const char *index_prefix,
    const char *subpq_prefix, uint32_t num_threads, const std::string &cache_ids, uint64_t cache_count) {
  mixed_subpq_navigation_ = false;
  auto *pq = dynamic_cast<PQNeighbor<T> *>(nbr_handler);
  if (!pq || !subpq_prefix || !std::is_same<T, float>::value || data_is_normalized || !num_threads)
    throw std::invalid_argument("Mixed subPQ requires a PQNeighbor, float32 L2 and positive threads");
  // Check the code shape before loading or allocating fixed-size query scratch.
  std::ifstream codes(std::string(subpq_prefix) + "_navi_pq_compressed.bin", std::ios::binary);
  uint32_t shape[2] = {};
  if (!codes.read(reinterpret_cast<char *>(shape), sizeof(shape)) || !shape[0] ||
      !shape[1] || shape[1] > AbstractNeighbor<T>::MAX_BYTES_PER_NBR)
    throw std::invalid_argument("Invalid subPQ code shape or too many chunks");
  codes.seekg(0, std::ios::end);
  if (codes.tellg() != static_cast<std::streamoff>(8 + uint64_t(shape[0]) * shape[1]))
    throw std::invalid_argument("SubPQ code file size mismatch");
  validate_subpq_pivots(std::string(subpq_prefix) + "_navi_pq_pivots.bin", shape[1]);
  pq->load_navi(subpq_prefix);
  const int result = load_pca_mixed_index(index_prefix, num_threads, cache_ids, cache_count);
  if (pq->get_point_count() != num_points || !pq->get_dimension() ||
      pq->get_dimension() > data_dim || pq->get_code_size() > pq->get_dimension() ||
      max_degree > MAX_N_EDGES)
    throw std::invalid_argument("SubPQ must match point IDs/count and use a PCA prefix subspace");
  mixed_subpq_navigation_ = true;
  LOG(INFO) << "subPQ navigation: PCA prefix dimensions=" << pq->get_dimension()
            << " chunks=" << pq->get_code_size() << "; mixed pivots used only for bounds";
  return result;
}

template<typename T, typename TagT>
void SSDIndex<T, TagT>::load_pca_mixed_navi_graph(const std::string &path) {
  mixed_estimation_ = false;
  mixed_estimation_ratio_ = 1.0;
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open mixed-pivot index: " + path);
  auto codec = std::make_unique<MixedPivotCodec>();
  codec->load(in);
  if (codec->points != num_points || codec->full_dim != data_dim || medoid >= num_points)
    throw std::runtime_error("Mixed-pivot and SSD metadata do not match");
  const auto start = in.tellg();
  in.seekg(0, std::ios::end);
  const uint64_t bytes = uint64_t(codec->points) * codec->node_bytes();
  if (in.tellg() - start != static_cast<std::streamoff>(bytes))
    throw std::runtime_error("Mixed-pivot record file size mismatch");
  in.seekg(start);
  const size_t adjacency_bytes = (size_t(codec->degree) + 1) * sizeof(uint32_t);
  const size_t payload_bytes = codec->padded_bytes() + sizeof(float);
  const size_t graph_bytes = mixed_cache_ ? size_t(codec->points) * adjacency_bytes : bytes;
  std::vector<uint32_t> slots;
  std::vector<char> payload;
  if (mixed_cache_) {
    if (mixed_cache_count_ > codec->points)
      throw std::invalid_argument("Mixed cache count exceeds point count");
    std::ifstream ids(mixed_cache_ids_, std::ios::binary | std::ios::ate);
    if (!ids) throw std::invalid_argument("Cannot open mixed cache ID list");
    const auto id_bytes = ids.tellg(); ids.seekg(0);
    uint32_t count = 0;
    if (!ids.read(reinterpret_cast<char *>(&count), 4) || count > codec->points ||
        count < mixed_cache_count_ || id_bytes != std::streamoff(4 + uint64_t(count) * 4))
      throw std::invalid_argument("Invalid mixed cache ID list size");
    slots.assign(codec->points, UINT32_MAX);
    for (uint32_t slot = 0; slot < mixed_cache_count_; ++slot) {
      uint32_t id;
      MixedPivotCodec::read(ids, &id, sizeof(id));
      if (id >= codec->points || slots[id] != UINT32_MAX)
        throw std::invalid_argument("Duplicate or out-of-range selected cache ID");
      slots[id] = slot;
    }
    payload.resize(size_t(mixed_cache_count_) * payload_bytes);
  }
  std::unique_ptr<char, decltype(&std::free)> graph(static_cast<char *>(std::malloc(graph_bytes)), &std::free);
  if (!graph) throw std::bad_alloc();
  // Stream sparse-cache records: never allocate all points' filtering payloads.
  std::vector<char> record(mixed_cache_ ? codec->node_bytes() : 0);
  if (!mixed_cache_) MixedPivotCodec::read(in, graph.get(), bytes);
  for (size_t id = 0; id < codec->points; ++id) {
    if (mixed_cache_) MixedPivotCodec::read(in, record.data(), record.size());
    const char *row = mixed_cache_ ? record.data() : graph.get() + id * codec->node_bytes();
    for (size_t d = 0; d < codec->m; ++d)
      if (!std::isfinite(MixedPivotCodec::bf16(row + 2 * d)))
        throw std::runtime_error("Non-finite BF16 pivot");
    float radius;
    uint32_t count;
    std::memcpy(&radius, row + codec->padded_bytes(), 4);
    std::memcpy(&count, row + codec->padded_bytes() + 4, 4);
    if (!std::isfinite(radius) || radius < 0 || count > codec->degree)
      throw std::runtime_error("Invalid mixed-pivot radius or degree");
    const auto *ids = reinterpret_cast<const uint32_t *>(row + codec->padded_bytes() + 8);
    for (size_t j = 0; j < count; ++j)
      if (ids[j] >= num_points) throw std::runtime_error("Invalid mixed-pivot neighbor ID");
    if (mixed_cache_) {
      std::memcpy(graph.get() + id * adjacency_bytes, row + payload_bytes, adjacency_bytes);
      if (slots[id] != UINT32_MAX)
        std::memcpy(payload.data() + size_t(slots[id]) * payload_bytes, row, payload_bytes);
    }
  }
  std::free(navi_graph_);
  navi_graph_ = graph.release();
  navi_dim_ = codec->dim;
  navi_data_len_ = codec->padded_bytes();
  navi_node_len_ = mixed_cache_ ? adjacency_bytes : codec->node_bytes();
  mixed_cache_slots_.swap(slots);
  mixed_cache_data_.swap(payload);
  if (mixed_cache_)
    LOG(INFO) << "MixedCachePoints=" << mixed_cache_count_
              << " MixedCachePayloadBytes=" << mixed_cache_data_.size()
              << " MixedCacheMapBytes=" << mixed_cache_slots_.size() * sizeof(uint32_t)
              << " MixedCacheAdjacencyBytes=" << graph_bytes;
  max_degree = codec->degree;
  LOG(INFO) << "Mixed pivots: D=" << codec->dim << " BF16=" << codec->m
            << " PQ1D=" << codec->n
            << " PQ2D=" << (codec->extended_tiers ? codec->pq2_dims : codec->dim - codec->m - codec->n)
            << " PQ4D=" << codec->pq4_dims
            << " PQ8D=" << (codec->extended_tiers ? codec->dim - codec->m - codec->n - codec->pq2_dims - codec->pq4_dims : 0)
            << " code bytes=" << codec->code_bytes();
  mixed_pivot_ = std::move(codec);
}

template<typename T, typename TagT>
void SSDIndex<T, TagT>::load_mixed_estimation(const std::string &eigenvalues,
                                            const std::string &epsilon_file) {
  if (!mixed_pivot_) throw std::logic_error("Load mixed index before estimation parameters");
  // Legacy DADE files contain ONE uint32 length followed by float32 values,
  // not the two-word header used by ordinary .bin matrices.
  auto read_vector = [](const std::string &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::invalid_argument("Cannot open estimation parameters: " + path);
    const auto bytes = in.tellg(); in.seekg(0);
    uint32_t n = 0;
    if (!in.read(reinterpret_cast<char *>(&n), 4) || !n || n > 65537 ||
        bytes != std::streamoff(4 + uint64_t(n) * 4))
      throw std::invalid_argument("Invalid estimation vector size: " + path);
    std::vector<float> values(n);
    MixedPivotCodec::read(in, values.data(), n * sizeof(float));
    for (float v : values) if (!std::isfinite(v))
      throw std::invalid_argument("Estimation parameters must be finite");
    return values;
  };
  const auto values = read_vector(eigenvalues), errors = read_vector(epsilon_file);
  for (float v : values) if (v < 0)
    throw std::invalid_argument("Estimation eigenvalues must be nonnegative");
  // Relative-error quantiles can be negative. Do not clamp them: this
  // would change the calibrated threshold. A positive scale requires eps > -1.
  for (float e : errors) if (e <= -1)
    throw std::invalid_argument("Estimation epsilon must be greater than -1");
  const size_t d = mixed_pivot_->dim, D = mixed_pivot_->full_dim;
  if (!d || values.size() != D || errors.size() < d)
    throw std::invalid_argument("Estimation dimension mismatch (prefix d uses epsilon[d-1])");
  double total = 0, prefix = 0;
  for (size_t i = 0; i < D; ++i) { total += values[i]; if (i < d) prefix += values[i]; }
  if (!(total > 0) || !(prefix > 0)) throw std::invalid_argument("Zero estimation variance");
  // Previous convention, retained for reference:
  // const double ratio = d == D ? 1.0 : prefix / total * std::pow(1.0 + errors[d], 2);
  // getEpsilon.py column d-1 contains a relative SQUARED-distance error
  // for the first d coordinates, so (1 + epsilon) is not squared again.
  const double ratio = d == D ? 1.0 : prefix / total * (1.0 + errors[d - 1]);
  if (!std::isfinite(ratio) || ratio <= 0)
    throw std::invalid_argument("Invalid estimation ratio");
  mixed_estimation_ratio_ = ratio;
  mixed_estimation_ = true;
}

template<typename T, typename TagT>
size_t SSDIndex<T, TagT>::skipdisk_pcamixed_search(const T *query, uint64_t k_search,
    uint64_t l_search, TagT *res_tags, float *distances, uint64_t beam_width, QueryStats *stats,
    bool in_memory, bool quantized_only, bool use_bound, const T *navigation_query) {
  if (!mixed_pivot_ || !std::is_same<T, float>::value || data_is_normalized)
    throw std::logic_error("Load a float32 L2 mixed-pivot index before searching");
  if (mixed_cache_ && !mixed_subpq_navigation_)
    throw std::logic_error("Partial mixed filtering requires independent subPQ navigation");
  if (!query || !res_tags || !k_search || k_search > l_search || k_search > num_points ||
      l_search > std::numeric_limits<uint32_t>::max() - 1 ||
      !beam_width || beam_width > MAX_N_SECTOR_READS)
    throw std::invalid_argument("Invalid mixed-pivot search arguments");
  for (size_t d = 0; d < data_dim; ++d)
    if (!std::isfinite(query[d])) throw std::invalid_argument("Non-finite query coordinate");
  // Public API contract: an independent navigation query has data_dim entries,
  // even when the navigation codebook uses only a prefix.
  if (navigation_query)
    for (size_t d = 0; d < data_dim; ++d)
      if (!std::isfinite(navigation_query[d])) throw std::invalid_argument("Non-finite navigation query");
  return skipdisk_pcabit_search(query, k_search, l_search, res_tags, distances, beam_width, stats, in_memory, quantized_only, use_bound, navigation_query);
}

template class SSDIndex<float>;
template class SSDIndex<int8_t>;
template class SSDIndex<uint8_t>;
}  // namespace pipeann
