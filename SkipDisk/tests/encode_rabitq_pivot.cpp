// Export the existing RaBitQ sign codes and estimator factors for pivot studies.
#include "nbr/rabitqlib/quantization/rabitq_impl.hpp"
#include <fstream>
#include <stdexcept>

template<class T> std::vector<T> read_matrix(const char *path, uint32_t &n, uint32_t &d) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("Cannot open input");
  auto bytes=in.tellg(); in.seekg(0);
  in.read(reinterpret_cast<char*>(&n),4); in.read(reinterpret_cast<char*>(&d),4);
  if (!in || !n || !d || uint64_t(n)*d*sizeof(T)+8 != uint64_t(bytes))
    throw std::runtime_error("Invalid matrix shape/length");
  std::vector<T> values(size_t(n)*d);
  in.read(reinterpret_cast<char*>(values.data()),values.size()*sizeof(T));
  return values;
}
template<class T> void write_matrix(const std::string &path,const std::vector<T>&v,uint32_t n,uint32_t d) {
  std::ofstream out(path,std::ios::binary);
  out.write(reinterpret_cast<char*>(&n),4); out.write(reinterpret_cast<char*>(&d),4);
  out.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));
  if (!out) throw std::runtime_error("Failed writing output");
}
int main(int argc,char**argv) {
  try {
    if(argc!=5) throw std::runtime_error("Usage: encode_rabitq_pivot ROTATED.bin CENTERS.bin IDS.u32bin OUTPUT_PREFIX");
    uint32_t n,d,k,cd,ni,di;
    auto x=read_matrix<float>(argv[1],n,d), centers=read_matrix<float>(argv[2],k,cd);
    auto ids=read_matrix<uint32_t>(argv[3],ni,di);
    if(d%64 || cd!=d || ni!=n || di!=1) throw std::runtime_error("Shape mismatch or dimension not divisible by 64");
    for(float v:x) if(!std::isfinite(v)) throw std::runtime_error("Nonfinite vector");
    for(float v:centers) if(!std::isfinite(v)) throw std::runtime_error("Nonfinite center");
    for(auto id:ids) if(id>=k) throw std::runtime_error("Invalid center ID");
    std::vector<uint64_t> codes(size_t(n)*(d/64));
    std::vector<float> factors(size_t(n)*2);
    for(size_t i=0;i<n;++i) {
      float unused_error;
      rabitqlib::quant::rabitq_impl::one_bit::one_bit_compact_code<float,uint64_t>(
        x.data()+i*d,centers.data()+size_t(ids[i])*d,d,codes.data()+i*(d/64),
        factors[2*i],factors[2*i+1],unused_error,rabitqlib::METRIC_L2);
    }
    write_matrix(std::string(argv[4])+"_codes.bin",codes,n,d/64);
    write_matrix(std::string(argv[4])+"_factors.bin",factors,n,2);
    return 0;
  } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
