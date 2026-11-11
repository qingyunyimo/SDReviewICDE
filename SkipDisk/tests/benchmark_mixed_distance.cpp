// Standalone microbenchmark, excluding LUT construction and graph/SSD search.
#include "mixed_pivot.h"
#include <chrono>
#include <iostream>
#include <random>
#include <array>

using Codec = pipeann::MixedPivotCodec;
using Distance = double (Codec::*)(const float *, const std::vector<double> &, const char *) const;
static volatile double sink = 0;

__attribute__((noinline)) static double measure(const Codec &codec, Distance fn,
    const std::vector<float> &q, const std::vector<double> &table, const std::vector<char> &codes) {
  const size_t stride = codec.code_bytes(), count = 4096, passes = 8;
  double checksum = 0;
  const auto start = std::chrono::steady_clock::now();
  for (size_t pass = 0; pass < passes; ++pass)
    for (size_t i = 0; i < count; ++i)
      checksum += (codec.*fn)(q.data(), table, codes.data() + i * stride);
  const auto end = std::chrono::steady_clock::now();
  sink = checksum;
  return std::chrono::duration<double, std::nano>(end - start).count() / (count * passes);
}

int main() {
#if defined(__AVX512F__) && defined(__AVX2__)
  std::cout << "Backend=AVX512_double\n";
#else
  std::cout << "Backend=scalar_fallback\n";
#endif
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> random(-3, 3);
  for (auto cfg : {std::array<unsigned, 3>{1536, 128, 256}, {1536, 256, 512},
                  {1536, 512, 512}, {1536, 1536, 0}, {1536, 0, 0}, {512, 128, 128}, {513, 129, 127}}) {
    Codec codec;
    codec.dim = cfg[0]; codec.full_dim = 1536; codec.m = cfg[1]; codec.n = cfg[2];
    codec.centers.resize((codec.dim - codec.m) * 256);
    for (auto &v : codec.centers) v = random(rng);
    std::vector<float> query(codec.dim);
    for (auto &v : query) v = random(rng);
    auto table = codec.distance_table(query.data());
    std::vector<char> codes(codec.code_bytes() * 4096);
    for (size_t i = 0; i < 4096; ++i) {
      char *code = codes.data() + i * codec.code_bytes();
      for (size_t d = 0; d < codec.m; ++d) {
        float x = random(rng);
        uint32_t bits;
        std::memcpy(&bits, &x, 4);
        uint16_t word = bits >> 16;
        std::memcpy(code + 2 * d, &word, 2);
      }
      for (size_t c = 0; c < codec.chunks(); ++c) code[2 * codec.m + c] = rng() % 256;
      const double scalar = codec.squared_distance_scalar(query.data(), table, code);
      const double simd = codec.squared_distance(query.data(), table, code);
      if (std::abs(scalar - simd) > 1e-12 * std::max(1.0, scalar))
        throw std::runtime_error("SIMD mismatch in large/unaligned configuration");
    }
    std::vector<double> scalar_times, simd_times;
    for (size_t repeat = 0; repeat < 5; ++repeat) {
      if (repeat % 2) {
        simd_times.push_back(measure(codec, &Codec::squared_distance, query, table, codes));
        scalar_times.push_back(measure(codec, &Codec::squared_distance_scalar, query, table, codes));
      } else {
        scalar_times.push_back(measure(codec, &Codec::squared_distance_scalar, query, table, codes));
        simd_times.push_back(measure(codec, &Codec::squared_distance, query, table, codes));
      }
    }
    std::sort(scalar_times.begin(), scalar_times.end());
    std::sort(simd_times.begin(), simd_times.end());
    std::cout << "Dp=" << codec.dim << " m=" << codec.m << " n=" << codec.n
              << " scalar_ns=" << scalar_times[2] << " SIMD_ns=" << simd_times[2]
              << " speedup=" << scalar_times[2] / simd_times[2] << '\n';
  }
}
