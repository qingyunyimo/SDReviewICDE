#include "mixed_pivot.h"

#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

static void check(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

static std::vector<float> decode(const pipeann::MixedPivotCodec &codec, const char *code) {
  std::vector<float> p(codec.dim);
  for (size_t d = 0; d < codec.m; ++d) p[d] = codec.bf16(code + 2 * d);
  size_t d = codec.m, offset = 0;
  for (size_t c = 0; c < codec.chunks(); ++c) {
    size_t width = codec.chunk_width(c, d);
    uint8_t k = static_cast<uint8_t>(code[2 * codec.m + c]);
    for (size_t j = 0; j < width; ++j) p[d + j] = codec.centers[offset + k * width + j];
    d += width;
    offset += width * 256;
  }
  return p;
}

static double squared(const float *a, const float *b, size_t dim) {
  double sum = 0;
  for (size_t d = 0; d < dim; ++d) {
    double delta = double(a[d]) - b[d];
    sum += delta * delta;
  }
  return sum;
}

static void verify(const pipeann::MixedPivotCodec &codec, const char *code,
                   const float *x, const float *q, float radius) {
  auto p = decode(codec, code);
  auto table = codec.distance_table(q);
  double adc = codec.squared_distance(q, table, code);
  double reference = codec.squared_distance_scalar(q, table, code);
  check(std::abs(adc - reference) <= 1e-12 * std::max(1.0, reference),
        "SIMD differs from scalar distance");
  // Exercise an unaligned, exactly-sized record (no vector-width padding).
  std::vector<char> unaligned(codec.code_bytes() + 1);
  std::memcpy(unaligned.data() + 1, code, codec.code_bytes());
  check(codec.squared_distance(q, table, unaligned.data() + 1) == adc,
        "Unaligned SIMD distance mismatch");
  double direct = squared(q, p.data(), codec.dim);
  check(std::abs(adc - direct) <= 1e-12 * std::max(1.0, direct), "ADC does not match decoded pivot");
  double expected_radius = std::sqrt(squared(x, p.data(), codec.dim));
  check(std::abs(radius - expected_radius) <= 2e-7 * std::max(1.0, expected_radius), "Wrong pivot radius");
  check(codec.lower_bound(adc, radius) <= squared(q, x, codec.dim) + 1e-12,
        "Lower bound exceeds exact squared distance");
  if (codec.full_dim)
    check(codec.lower_bound(adc, radius) <= squared(q, x, codec.full_dim) + 1e-12,
          "Projected lower bound exceeds full-space distance");
}

static void unit_tests() {
  std::mt19937 rng(17);
  std::normal_distribution<float> random(0, 3);
  for (uint32_t dim : {1u, 2u, 7u, 16u, 31u}) {
    for (uint32_t m = 0; m <= dim; ++m) {
      for (uint32_t n = 0; n <= dim - m; ++n) {
        pipeann::MixedPivotCodec codec;
        codec.dim = dim; codec.m = m; codec.n = n;
        codec.centers.resize((dim - m) * 256);
        for (float &v : codec.centers) v = random(rng);
        std::vector<float> x(dim), q(dim);
        std::vector<char> code(codec.padded_bytes());
        for (auto &v : x) v = random(rng);
        for (size_t d = 0; d < m; ++d) {
          uint32_t bits;
          std::memcpy(&bits, &x[d], 4);
          uint16_t head = bits >> 16;
          std::memcpy(code.data() + 2 * d, &head, 2);
        }
        for (size_t c = 0; c < codec.chunks(); ++c) code[2 * m + c] = rng() % 256;
        auto p = decode(codec, code.data());
        float radius = std::sqrt(squared(x.data(), p.data(), dim));
        for (int trial = 0; trial < 8; ++trial) {
          for (size_t d = 0; d < dim; ++d) {
            // Include q=x, q=p, and collinear cases where the bound is tight.
            q[d] = trial == 0 ? x[d] : trial == 1 ? p[d] : trial == 2 ? 2 * x[d] - p[d] : random(rng);
          }
          verify(codec, code.data(), x.data(), q.data(), radius);
        }
      }
    }
  }
  pipeann::MixedPivotCodec codec;
  std::istringstream truncated("SKMIXPQ1");
  bool rejected = false;
  try { codec.load(truncated); } catch (const std::runtime_error &) { rejected = true; }
  check(rejected, "Truncated header accepted");
}

int main(int argc, char **argv) {
  try {
    unit_tests();
    if (argc == 4) {
      const size_t samples = std::stoull(argv[3]);
      check(samples > 0 && samples <= 10000, "Invalid sample count");
      std::ifstream in(argv[1], std::ios::binary), base(argv[2], std::ios::binary);
      pipeann::MixedPivotCodec codec;
      codec.load(in);
      const auto records_offset = in.tellg();
      uint32_t shape[2]; codec.read(base, shape, sizeof(shape));
      check(shape[0] == codec.points && shape[1] == codec.full_dim, "Base shape mismatch");
      std::vector<float> x(codec.full_dim), q(codec.full_dim);
      std::vector<char> row(codec.node_bytes());
      std::mt19937 rng(42);
      for (size_t s=0; s<samples; ++s) {
        const size_t id = rng() % codec.points, query = rng() % codec.points;
        in.seekg(records_offset + static_cast<std::streamoff>(id * codec.node_bytes()));
        codec.read(in, row.data(), row.size());
        base.seekg(8 + id * codec.full_dim * sizeof(float));
        codec.read(base, x.data(), x.size()*sizeof(float));
        base.seekg(8 + query * codec.full_dim * sizeof(float));
        codec.read(base, q.data(), q.size()*sizeof(float));
        float radius; std::memcpy(&radius, row.data()+codec.padded_bytes(), 4);
        verify(codec,row.data(),x.data(),q.data(),radius);
        verify(codec,row.data(),x.data(),x.data(),radius);
      }
    } else if (argc == 3) {
      std::ifstream in(argv[1], std::ios::binary), base(argv[2], std::ios::binary);
      pipeann::MixedPivotCodec codec;
      codec.load(in);
      uint32_t shape[2];
      codec.read(base, shape, sizeof(shape));
      check(shape[0] == codec.points && shape[1] == codec.full_dim, "Base shape mismatch");
      std::vector<float> vectors(size_t(shape[0]) * shape[1]);
      codec.read(base, vectors.data(), vectors.size() * sizeof(float));
      std::vector<char> row(codec.node_bytes());
      for (size_t id = 0; id < codec.points; ++id) {
        codec.read(in, row.data(), row.size());
        float radius;
        std::memcpy(&radius, row.data() + codec.padded_bytes(), 4);
        for (size_t q = 0; q < std::min<size_t>(32, codec.points); ++q)
          verify(codec, row.data(), vectors.data() + id * codec.full_dim,
                 vectors.data() + q * codec.full_dim, radius);
      }
    } else if (argc != 1) {
      throw std::invalid_argument("Usage: test_mixed_pivot [MIXED_INDEX PCA_BASE.bin [SAMPLE_COUNT]]");
    }
    std::cout << "Mixed pivot codec, ADC and lower-bound tests passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
