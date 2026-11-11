#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <istream>
#include <limits>
#include <stdexcept>
#include <vector>
#if defined(__AVX512F__) && defined(__AVX2__)
#include <immintrin.h>
#endif

namespace pipeann {

// PCA coordinates: m BF16 values, n one-dimensional PQ codes, then
// ceil((dim-m-n)/2) two-dimensional PQ codes (the last may be 1D).
// All codebooks have 256 float32 centroids and all PQ codes occupy one byte.
// Version 3 adds complete 2D/4D/8D tiers after the 1D tier (4/2/1 bits per
// coordinate on average). Version 1/2 retain their legacy odd-tail semantics.
class MixedPivotCodec {
 public:
  uint32_t points = 0, dim = 0, m = 0, n = 0, degree = 0;
  uint32_t full_dim = 0;  // Original/SSD dimension; dim is the stored PCA prefix.
  bool extended_tiers = false;
  uint32_t pq2_dims = 0, pq4_dims = 0;  // v3: remaining coordinates use 8D PQ.
  std::vector<float> centers;

  size_t chunks() const {
    if (extended_tiers)
      return n + pq2_dims / 2 + pq4_dims / 4 + (dim - m - n - pq2_dims - pq4_dims) / 8;
    return n + (size_t(dim) - m - n + 1) / 2;
  }
  size_t chunk_width(size_t c, size_t d) const {
    if (c < n) return 1;
    if (!extended_tiers) return std::min<size_t>(2, dim - d);
    if (c < size_t(n) + pq2_dims / 2) return 2;
    if (c < size_t(n) + pq2_dims / 2 + pq4_dims / 4) return 4;
    return 8;
  }
  size_t code_bytes() const { return 2 * size_t(m) + chunks(); }
  size_t padded_bytes() const { return (code_bytes() + 3) & ~size_t(3); }
  size_t node_bytes() const { return padded_bytes() + 8 + 4 * size_t(degree); }

  static void read(std::istream &in, void *dst, size_t bytes) {
    if (!in.read(static_cast<char *>(dst), bytes))
      throw std::runtime_error("Truncated mixed-pivot index");
  }

  void load(std::istream &in) {
    char magic[8];
    uint32_t header[7];
    read(in, magic, sizeof(magic));
    read(in, header, sizeof(header));
    if (std::memcmp(magic, "SKMIXPQ1", 8) || (header[0] < 1 || header[0] > 3))
      throw std::runtime_error("Unsupported mixed-pivot format");
    points = header[1]; dim = header[2]; m = header[3];
    n = header[4]; degree = header[5];
    full_dim = dim;
    extended_tiers = header[0] == 3;
    pq2_dims = pq4_dims = 0;
    if (header[0] >= 2) read(in, &full_dim, sizeof(full_dim));
    if (extended_tiers) {
      read(in, &pq2_dims, sizeof(pq2_dims));
      read(in, &pq4_dims, sizeof(pq4_dims));
    }
    if (!points || !dim || dim > 65536 || m > dim || n > dim - m || degree > 1024 ||
        full_dim < dim || full_dim > 65536)
      throw std::runtime_error("Invalid mixed-pivot dimensions or record layout");
    if (extended_tiers && (uint64_t(m) + n + pq2_dims + pq4_dims > dim ||
        pq2_dims % 2 || pq4_dims % 4 || (dim - m - n - pq2_dims - pq4_dims) % 8))
      throw std::runtime_error("Invalid mixed-pivot 2D/4D/8D tier alignment");
    if (header[6] != code_bytes()) throw std::runtime_error("Invalid mixed-pivot code length");
    centers.resize(size_t(dim - m) * 256);
    read(in, centers.data(), centers.size() * sizeof(float));
    for (float value : centers)
      if (!std::isfinite(value)) throw std::runtime_error("Non-finite PQ centroid");
  }

  static float bf16(const char *src) {
    uint16_t word;
    std::memcpy(&word, src, 2);
    uint32_t bits = uint32_t(word) << 16;
    float value;
    std::memcpy(&value, &bits, 4);
    return value;
  }

  // Local to each query: shared codebooks and index remain read-only.
  std::vector<double> distance_table(const float *query) const {
    std::vector<double> table(chunks() * 256);
    size_t d = m, offset = 0;
    for (size_t c = 0; c < chunks(); ++c) {
      size_t width = chunk_width(c, d);
      for (size_t k = 0; k < 256; ++k) {
        double sum = 0;
        for (size_t j = 0; j < width; ++j) {
          double diff = double(query[d + j]) - centers[offset + k * width + j];
          sum += diff * diff;
        }
        table[c * 256 + k] = sum;
      }
      d += width;
      offset += 256 * width;
    }
    return table;
  }

  // Reference/fallback implementation, also used by numerical regression tests.
  double squared_distance_scalar(const float *query, const std::vector<double> &table,
                                 const char *code) const {
    double sum = 0;
    for (size_t d = 0; d < m; ++d) {
      double diff = double(query[d]) - bf16(code + 2 * d);
      sum += diff * diff;
    }
    const auto *pq = reinterpret_cast<const uint8_t *>(code + 2 * m);
    for (size_t c = 0; c < chunks(); ++c) sum += table[c * 256 + pq[c]];
    return sum;
  }

  double squared_distance(const float *query, const std::vector<double> &table,
                          const char *code) const {
#if defined(__AVX512F__) && defined(__AVX2__)
    // Same BF16 bit expansion as BitPreserver, fused into distance evaluation:
    // no temporary decoded vector. Keep double arithmetic for the bound path.
    __m512d head_sum = _mm512_setzero_pd();
    size_t d = 0;
    for (; d + 8 <= m; d += 8) {
      const __m128i words = _mm_loadu_si128(reinterpret_cast<const __m128i *>(code + 2 * d));
      const __m256i bits = _mm256_slli_epi32(_mm256_cvtepu16_epi32(words), 16);
      const __m512d pivot = _mm512_cvtps_pd(_mm256_castsi256_ps(bits));
      const __m512d q = _mm512_cvtps_pd(_mm256_loadu_ps(query + d));
      const __m512d diff = _mm512_sub_pd(q, pivot);
      head_sum = _mm512_fmadd_pd(diff, diff, head_sum);
    }

    const auto *pq = reinterpret_cast<const uint8_t *>(code + 2 * m);
    __m512d pq_sum = _mm512_setzero_pd();
    const __m256i offsets = _mm256_setr_epi32(0, 256, 512, 768, 1024, 1280, 1536, 1792);
    size_t c = 0;
    const size_t count = chunks();
    for (; c + 8 <= count; c += 8) {
      // Read exactly 8 code bytes, including for an unaligned final block.
      const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(pq + c));
      const __m256i indices = _mm256_add_epi32(_mm256_cvtepu8_epi32(bytes), offsets);
      const __m512d values = _mm512_i32gather_pd(indices, table.data() + c * 256, 8);
      pq_sum = _mm512_add_pd(pq_sum, values);
    }
    double sum = _mm512_reduce_add_pd(head_sum) + _mm512_reduce_add_pd(pq_sum);
    // Scalar tails never read beyond the real BF16/PQ code length.
    for (; d < m; ++d) {
      const double diff = double(query[d]) - bf16(code + 2 * d);
      sum += diff * diff;
    }
    for (; c < count; ++c) sum += table[c * 256 + pq[c]];
    return sum;
#else
    return squared_distance_scalar(query, table, code);
#endif
  }

  // Radius is ||x[:dim] - decoded_pivot||, computed anew by the builder. Allow for
  // float32 radius storage and float32 exact-distance accumulation on SSD.
  float lower_bound(double squared_query_distance, float radius) const {
    double q = std::sqrt(squared_query_distance);
    const double eps = std::numeric_limits<float>::epsilon();
    double gap = std::max(0.0, std::abs(q - double(radius)) -
                                  4 * eps * (q + radius) - std::numeric_limits<float>::denorm_min());
    // The prefix-space bound is also a full-space bound. Roundoff allowance
    // uses the full SSD dimension, since exact reranking accumulates all terms.
    double lb = gap * gap * std::max(0.0, 1.0 - 4 * std::max(dim, full_dim) * eps);
    return std::max(0.0f, std::nextafter(static_cast<float>(lb), 0.0f));
  }
};
}  // namespace pipeann
