#include <iostream>
#include "utils.h"

#include <sys/mman.h>
#include <sys/stat.h> 
#include <fcntl.h>
#include <unistd.h>
#include "bitpreserver.h"
#include <bitset>
#include <immintrin.h>
#include <unordered_set>

#define READ_SECTOR_LEN (size_t) 4096
#define SECTOR_LEN 4096
#define READ_SECTOR_OFFSET(node_id) \
  ((_u64) node_id / nnodes_per_sector  + 1) * READ_SECTOR_LEN + ((_u64) node_id % nnodes_per_sector) * max_node_len;
#define INF 0xffffffff

#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))

typedef uint64_t _u64;
typedef int64_t  _s64;
typedef uint32_t _u32;
typedef int32_t  _s32;
typedef uint16_t _u16;
typedef int16_t  _s16;
typedef uint8_t  _u8;
typedef int8_t   _s8;

class MemoryMappedFile {
private:
    int fd;
    void* data;
    size_t size;
    
public:
    MemoryMappedFile(const char* filename) : fd(-1), data(nullptr), size(0) {
        fd = open(filename, O_RDONLY);
        if (fd == -1) {
            perror("open file error");
            exit(-1);
        }
        
        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            perror("fstat error");
            close(fd);
            exit(-1);
        }
        
        size = sb.st_size;
        data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) {
            perror("mmap error");
            close(fd);
            exit(-1);
        }
    }
    
    ~MemoryMappedFile() {
        if (data != nullptr && data != MAP_FAILED) {
            munmap(data, size);
        }
        if (fd != -1) {
            close(fd);
        }
    }
    
    const void* get_data() const { return data; }
    size_t get_size() const { return size; }
    
    template<typename T>
    const T* get_pointer(size_t offset = 0) const {
        return reinterpret_cast<const T*>(static_cast<const char*>(data) + offset);
    }

    void reset() {
        if (data != nullptr && data != MAP_FAILED) {
            munmap(data, size);
            data = nullptr;
        }
        if (fd != -1) {
            close(fd);
            fd = -1;
        }
    }
};

inline float l2_sqr( const float* __restrict__ vec0, const float* __restrict__ vec1, size_t dim) {
    float result = 0;
    size_t mul16 = dim - (dim & 0b1111);
    auto sum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i < mul16; i += 16) {
        auto xxx = _mm512_loadu_ps(&vec0[i]);
        auto yyy = _mm512_loadu_ps(&vec1[i]);
        auto ttt = _mm512_sub_ps(xxx, yyy);
        sum = _mm512_fmadd_ps(ttt, ttt, sum);
    }
    result = _mm512_reduce_add_ps(sum);
    for (; i < dim; ++i) {
        float tmp = vec0[i] - vec1[i];
        result += tmp * tmp;
    }

    return result;
}

// DiskANN changes how the meta is stored in the first sector of
// the _disk.index file after commit id 8bb74ff637cb2a77c99b71368ade68c62b7ca8e0 (exclusive)
// This function returns <is_new_version, vector of metas uint64_ts>
std::pair<bool, std::vector<_u64>> get_disk_index_meta(const std::string& path) {
    std::ifstream index_metadata(path, std::ios::binary);

    uint32_t nr, nc; // metadata itself is stored as bin format (nr is number of
                     // metadata, nc should be 1)
    READ_U32(index_metadata, nr);
    READ_U32(index_metadata, nc);

    assert(nr == 9);

    bool is_new_version = true;
    std::vector<_u64> metas;
    metas.resize(nr);
    index_metadata.read((char *)(metas.data()), sizeof(_u64) * nr);
    index_metadata.close();
    return {is_new_version, metas};
}

void transtobitlevelgraph(const char* indexname, const char* base_path, std::string index_prefix){
    _u64                               C;
    _u64                               _partition_nums;
    _u64                               _nd;
    _u64                               _dim;
    _u64                               _max_node_len;
    _u64                               _nnodes_per_sector;
 

    auto meta_pair = get_disk_index_meta(indexname);
    _u64 actual_index_size = get_file_size(indexname);
    _u64 expected_file_size, expected_npts;

    if (meta_pair.first) {
        // new version
        expected_file_size = meta_pair.second.back();
        expected_npts = meta_pair.second.front();
    } else {
        expected_file_size = meta_pair.second.front();
        expected_npts = meta_pair.second[1];
    }

    if (expected_file_size != actual_index_size) {
        std::cout << "File size mismatch for " << indexname
                    << " (size: " << actual_index_size << ")"
                    << " with meta-data size: " << expected_file_size << std::endl;
        exit(-1);
    }

    _nd = expected_npts;
    _dim = meta_pair.second[1];
    _max_node_len = meta_pair.second[3];
    _nnodes_per_sector = meta_pair.second[4];

    _u64            read_blk_size = 64 * 1024 * 1024;
    _u64            write_blk_size = read_blk_size;

    std::string navi_graph_path = index_prefix + "_navi_bit_graph.index";

    size_t index_partition_nums = DIV_ROUND_UP(_nd, _nnodes_per_sector);
    size_t size_per_io = SECTOR_LEN * (_nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, SECTOR_LEN));

    MemoryMappedFile mem_index_file(indexname);
    const char* mem_index_data = static_cast<const char*>(mem_index_file.get_data()) + SECTOR_LEN; // skip meta data

    std::vector<std::vector<unsigned>> full_graph;
    full_graph.resize(_nd);
    _u64 des = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : des)
    for(size_t i = 0; i < index_partition_nums; i++){
        std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(size_per_io);
        memset(sector_buf.get(), 0, size_per_io);
        memcpy(sector_buf.get(), mem_index_data + i * size_per_io, size_per_io);
        for(size_t j = 0; j < _nnodes_per_sector && i * _nnodes_per_sector + j < _nd; j++) {
            std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(_max_node_len);
            memset(node_buf.get(), 0, _max_node_len);
            memcpy(node_buf.get(), sector_buf.get() + j * _max_node_len, _max_node_len);
            unsigned &nnbr = *(unsigned *)(node_buf.get() + _dim * sizeof(float));
            unsigned *nhood_buf = (unsigned *)(node_buf.get() + (_dim * sizeof(float)) + sizeof(unsigned));
            std::vector<unsigned> tmp(nnbr);
            memcpy((char *)tmp.data(), nhood_buf, nnbr * sizeof(unsigned));
            full_graph[i * _nnodes_per_sector + j].assign(tmp.begin(), tmp.end());
            des += nnbr;
        }
    }
    mem_index_file.reset();
    std::cout << "avg neighbor size: " << des / (double) _nd << std::endl;

    MemoryMappedFile base_file(base_path);
    const char* file_data = static_cast<const char*>(base_file.get_data());
    unsigned data_num = *reinterpret_cast<const unsigned*>(file_data);
    unsigned data_dim = *(reinterpret_cast<const unsigned*>(file_data) + 1);
    const float* data_ptr = reinterpret_cast<const float*>(file_data + 2 * sizeof(unsigned));
    std::cout << "data num: " << data_num << " data dim: " << data_dim << std::endl;

    int navi_dim = 256;
    int chopbitsize = 16;

    BitPreserver preserver_navi(navi_dim, chopbitsize);

}

int main(){
    return 0;
}
