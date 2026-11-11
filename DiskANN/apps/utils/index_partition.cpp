//This file is created based on starling

#include <iostream>
#include "utils.h"
#include "defaults.h"
#include <sys/mman.h>
#include <sys/stat.h> 
#include <fcntl.h>
#include <unistd.h>
#include "bitpreserver.h"
#include <bitset>
#include <immintrin.h>
#include <unordered_set>
#define READ_SECTOR_LEN (size_t) 4096
#define READ_SECTOR_OFFSET(node_id) \
  ((_u64) node_id / nnodes_per_sector  + 1) * READ_SECTOR_LEN + ((_u64) node_id % nnodes_per_sector) * max_node_len;
#define INF 0xffffffff

const std::string partition_index_filename = "_tmp.index";

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


void extractAndSaveGraphNode(const char* indexname, std::string index_prefix){
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
        diskann::cout << "File size mismatch for " << indexname
                    << " (size: " << actual_index_size << ")"
                    << " with meta-data size: " << expected_file_size << std::endl;
        exit(-1);
    }

    _nd = expected_npts;
    _dim = meta_pair.second[1];
    _max_node_len = meta_pair.second[3];
    _nnodes_per_sector = meta_pair.second[4];
    std::cout<<"_nd: "<<_nd<<" _dim: "<<_dim<<" _max_node_len: "<<_max_node_len<<" _nnodes_per_sector: "<<_nnodes_per_sector<<std::endl;
    _u64            read_blk_size = 64 * 1024 * 1024;
    _u64            write_blk_size = read_blk_size;

    std::string graph_node_path = index_prefix + "_"  + "graph_nodes.index";

    size_t index_partition_nums = _nnodes_per_sector > 0 ? DIV_ROUND_UP(_nd, _nnodes_per_sector) : _nd;
    size_t _nodes_per_io = _nnodes_per_sector > 0 ? _nnodes_per_sector : 1;
    size_t size_per_io = _nnodes_per_sector > 0 ? diskann::defaults::SECTOR_LEN : DIV_ROUND_UP(_max_node_len, diskann::defaults::SECTOR_LEN) * diskann::defaults::SECTOR_LEN;

    MemoryMappedFile mem_index_file(indexname);
    const char* mem_index_data = static_cast<const char*>(mem_index_file.get_data()) + diskann::defaults::SECTOR_LEN; // skip meta data
    
    std::vector<std::vector<unsigned>> full_graph;
    full_graph.resize(_nd);
    _u64 des = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : des)
    for(size_t i = 0; i < index_partition_nums; i++){
        std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(size_per_io);
        memset(sector_buf.get(), 0, size_per_io);
        memcpy(sector_buf.get(), mem_index_data + i * size_per_io, size_per_io);
        for(size_t j = 0; j < _nodes_per_io && i * _nodes_per_io + j < _nd; j++) {
            std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(_max_node_len);
            memset(node_buf.get(), 0, _max_node_len);
            memcpy(node_buf.get(), sector_buf.get() + j * _max_node_len, _max_node_len);
            unsigned &nnbr = *(unsigned *)(node_buf.get() + _dim * sizeof(float));
            if(nnbr != 64){
                std::cout<<nnbr<<std::endl;
            }
            unsigned *nhood_buf = (unsigned *)(node_buf.get() + (_dim * sizeof(float)) + sizeof(unsigned));
            std::vector<unsigned> tmp(nnbr);
            memcpy((char *)tmp.data(), nhood_buf, nnbr * sizeof(unsigned));
            full_graph[i * _nodes_per_io + j].assign(tmp.begin(), tmp.end());
            des += nnbr;
        }
    }
    mem_index_file.reset();
    std::cout << "avg neighbor size: " << des / (double) _nd << std::endl;

    //save graph part
    cached_ofstream diskann_writer_navi(graph_node_path, write_blk_size);
    diskann_writer_navi.write((char*)&_nd, sizeof(unsigned));
    
    for(size_t i = 0; i < _nd; i++ ) {
        unsigned neighborsize = full_graph[i].size();
        char nboor_buf[neighborsize * sizeof(unsigned) + sizeof(unsigned)];
        memcpy(nboor_buf, (char*)&neighborsize, sizeof(unsigned));
        memcpy(nboor_buf + sizeof(unsigned), (char*)full_graph[i].data(), neighborsize * sizeof(unsigned));
        diskann_writer_navi.write(nboor_buf, neighborsize  * sizeof(unsigned) + sizeof(unsigned));
    }
    diskann_writer_navi.close();
}

void extractAndSaveGraphNodeFixDegreePad(const char* indexname, std::string index_prefix){
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
        diskann::cout << "File size mismatch for " << indexname
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

    std::string graph_node_path = index_prefix + "_"  + "graph_nodes.index";

    size_t index_partition_nums = DIV_ROUND_UP(_nd, _nnodes_per_sector);

    MemoryMappedFile mem_index_file(indexname);
    const char* mem_index_data = static_cast<const char*>(mem_index_file.get_data()) + diskann::defaults::SECTOR_LEN; // skip meta data
    
    std::vector<std::vector<unsigned>> full_graph;
    full_graph.resize(_nd);
    _u64 des = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : des)
    for(size_t i = 0; i < index_partition_nums; i++){
        std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(diskann::defaults::SECTOR_LEN);
        memset(sector_buf.get(), 0, diskann::defaults::SECTOR_LEN);
        memcpy(sector_buf.get(), mem_index_data + i * diskann::defaults::SECTOR_LEN, diskann::defaults::SECTOR_LEN);
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

    //save graph part
    size_t neighbor_width = 64;
    cached_ofstream diskann_writer_navi(graph_node_path, write_blk_size);
    diskann_writer_navi.write((char*)&_nd, sizeof(unsigned));
    
    for(size_t i = 0; i < _nd; i++ ) {
        unsigned neighborsize = full_graph[i].size();
        char nboor_buf[neighbor_width * sizeof(unsigned) + sizeof(unsigned)];
        memset(nboor_buf, 0, neighbor_width * sizeof(unsigned) + sizeof(unsigned));
        memcpy(nboor_buf, (char*)&neighborsize, sizeof(unsigned));
        memcpy(nboor_buf + sizeof(unsigned), (char*)full_graph[i].data(), neighborsize * sizeof(unsigned));
        diskann_writer_navi.write(nboor_buf, neighborsize  * sizeof(unsigned) + sizeof(unsigned));
    }
    diskann_writer_navi.close();
}

void transformIndex(const char* indexname, const char* base_path, std::string index_prefix){
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
        diskann::cout << "File size mismatch for " << indexname
                    << " (size: " << actual_index_size << ")"
                    << " with meta-data size: " << expected_file_size << std::endl;
        exit(-1);
    }
    _u64 _nd = expected_npts;
    _u64 _dim = meta_pair.second[1];

    _u64 _max_node_len = meta_pair.second[3];
    _u64 _nnodes_per_sector = meta_pair.second[4];
    unsigned index_partition_nums = DIV_ROUND_UP(_nd, _nnodes_per_sector);

    _u64            read_blk_size = 64 * 1024 * 1024;
    _u64            write_blk_size = read_blk_size;

   
    std::string new_index_name = index_prefix + "_disk_transformed.index";

    MemoryMappedFile base_file(base_path);
    const char* file_data = static_cast<const char*>(base_file.get_data());
    unsigned data_num = *reinterpret_cast<const unsigned*>(file_data);
    unsigned data_dim = *(reinterpret_cast<const unsigned*>(file_data) + 1);
    const float* data_ptr = reinterpret_cast<const float*>(file_data + 2 * sizeof(unsigned));
    std::cout << "data num: " << data_num << " data dim: " << data_dim << std::endl;

    cached_ofstream diskann_writer(new_index_name, write_blk_size);
    MemoryMappedFile mem_index_file(indexname);
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(diskann::defaults::SECTOR_LEN);
    memcpy(sector_buf.get(), mem_index_file.get_data(), diskann::defaults::SECTOR_LEN);
    diskann_writer.write(sector_buf.get(), diskann::defaults::SECTOR_LEN); //write meta data later
    const char* mem_index_data = static_cast<const char*>(mem_index_file.get_data()) + diskann::defaults::SECTOR_LEN; // skip meta data

    for(size_t i = 0; i < index_partition_nums; i++){
        memset(sector_buf.get(), 0, diskann::defaults::SECTOR_LEN);
        memcpy(sector_buf.get(), mem_index_data + i * diskann::defaults::SECTOR_LEN, diskann::defaults::SECTOR_LEN);
        for(size_t j = 0; j < _nnodes_per_sector && i * _nnodes_per_sector + j < _nd; j++) {
            float* node_data = (float*)(data_ptr + (i * _nnodes_per_sector + j) * data_dim);
            memcpy(sector_buf.get() + j * _max_node_len, (char*)node_data, _dim * sizeof(float));
        }
        diskann_writer.write(sector_buf.get(), diskann::defaults::SECTOR_LEN);
    }
    diskann_writer.close();
}

void splitData(const char* base_path, std::string index_prefix, int navi_dim){
    _u64            read_blk_size = 64 * 1024 * 1024;
    _u64            write_blk_size = read_blk_size;

    
    //path name for new index files
    std::string navi_part_path = index_prefix + "_"  + std::to_string(navi_dim) +"d_data.bin";
    cached_ofstream diskann_writer_navi(navi_part_path, write_blk_size);

    MemoryMappedFile base_file(base_path);
    const char* file_data = static_cast<const char*>(base_file.get_data());
    unsigned data_num = *reinterpret_cast<const unsigned*>(file_data);
    unsigned data_dim = *(reinterpret_cast<const unsigned*>(file_data) + 1);
    const float* data_ptr = reinterpret_cast<const float*>(file_data + 2 * sizeof(unsigned));
    std::cout << "data num: " << data_num << " data dim: " << data_dim << std::endl;
    
    
    //for navi part
    
    diskann_writer_navi.write((char*)&data_num, sizeof(unsigned));
    diskann_writer_navi.write((char*)&navi_dim, sizeof(unsigned));
    for(size_t i = 0; i < data_num; i++ ) {
        diskann_writer_navi.write((char*)(data_ptr + i * data_dim), navi_dim * sizeof(float));
    }
    
    diskann_writer_navi.close();
}


//对应的id也要保存，并且和数据一一对应
void randomGenFixNumData(const char* base_path, std::string output_path_prefix, unsigned fix_num){
    _u64            read_blk_size = 64 * 1024 * 1024;
    _u64            write_blk_size = read_blk_size;

    std::string output_path_prefix_id = output_path_prefix + "_id_" + std::to_string(fix_num) + ".bin";
    std::string output_path_prefix_data = output_path_prefix + "_data_" + std::to_string(fix_num) + ".bin";
    cached_ofstream diskann_writer_id(output_path_prefix_id, write_blk_size);
    cached_ofstream diskann_writer(output_path_prefix_data, write_blk_size);

    MemoryMappedFile base_file(base_path);
    const char* file_data = static_cast<const char*>(base_file.get_data());
    unsigned data_num = *reinterpret_cast<const unsigned*>(file_data);
    unsigned data_dim = *(reinterpret_cast<const unsigned*>(file_data) + 1);
    const float* data_ptr = reinterpret_cast<const float*>(file_data + 2 * sizeof(unsigned));
    std::cout << "data num: " << data_num << " data dim: " << data_dim << std::endl;

    diskann_writer.write((char*)&fix_num, sizeof(unsigned));
    diskann_writer.write((char*)&data_dim, sizeof(unsigned));

    diskann_writer_id.write((char*)&fix_num, sizeof(unsigned));
    diskann_writer_id.write((char*)&data_dim, sizeof(unsigned));

    std::unordered_set<unsigned> selected_ids;
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned> distribution(0, data_num -1);
    
    
    while(selected_ids.size() < fix_num){
        unsigned rand_id = distribution(generator);
        if(selected_ids.find(rand_id) == selected_ids.end()){
            selected_ids.insert(rand_id);
            diskann_writer.write((char*)(data_ptr + rand_id * data_dim), data_dim * sizeof(float));
            diskann_writer_id.write((char*)&rand_id, sizeof(unsigned));
        }
    }
    diskann_writer.close();
    diskann_writer_id.close();
}

//read data, cal the indegree and sort according to degree
void calDegreeAndSort(const char* indexname, std::string output_path_prefix){
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
        diskann::cout << "File size mismatch for " << indexname
                    << " (size: " << actual_index_size << ")"
                    << " with meta-data size: " << expected_file_size << std::endl;
        exit(-1);
    }

    _nd = expected_npts;
    _dim = meta_pair.second[1];
    _max_node_len = meta_pair.second[3];
    _nnodes_per_sector = meta_pair.second[4];

    size_t index_partition_nums = _nnodes_per_sector > 0 ? DIV_ROUND_UP(_nd, _nnodes_per_sector) : _nd;
    size_t _nodes_per_io = _nnodes_per_sector > 0 ? _nnodes_per_sector : 1;
    std::cout<<index_partition_nums<<" "<<_nodes_per_io<<" "<<_max_node_len<<" "<<diskann::defaults::SECTOR_LEN<<std::endl;
    size_t size_per_io = diskann::defaults::SECTOR_LEN * (_nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, diskann::defaults::SECTOR_LEN));

    MemoryMappedFile mem_index_file(indexname);
    const char* mem_index_data = static_cast<const char*>(mem_index_file.get_data()) + diskann::defaults::SECTOR_LEN; // skip meta data

    std::vector<unsigned> indegree(_nd, 0);
#pragma omp parallel for schedule(dynamic, 1)
    for(size_t i = 0; i < index_partition_nums; i++){
        std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(size_per_io);
        memset(sector_buf.get(), 0, size_per_io);
        memcpy(sector_buf.get(), mem_index_data + i * size_per_io, size_per_io);
        for(size_t j = 0; j < _nodes_per_io && i * _nodes_per_io + j < _nd; j++) {
            std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(_max_node_len);
            memset(node_buf.get(), 0, _max_node_len);
            memcpy(node_buf.get(), sector_buf.get() + j * _max_node_len, _max_node_len);
            unsigned &nnbr = *(unsigned *)(node_buf.get() + _dim * sizeof(float));
            unsigned *nhood_buf = (unsigned *)(node_buf.get() + (_dim * sizeof(float)) + sizeof(unsigned));
            for(size_t k =0; k< nnbr; k++){
                unsigned neighbor_id = nhood_buf[k];
#pragma omp atomic
                indegree[neighbor_id] +=1;
            }
        }
    }
    mem_index_file.reset();
    //sort according to indegree
    std::vector<std::pair<unsigned, unsigned>> degree_id_pair;
    for(unsigned i =0; i< indegree.size(); i++){
        degree_id_pair.push_back(std::make_pair(indegree[i], i));
    }
    std::sort(degree_id_pair.begin(), degree_id_pair.end(), std::greater<std::pair<unsigned, unsigned>>());
    //write to file
    std::string output_path = output_path_prefix + "_degree_sorted.id";
    cached_ofstream degree_writer(output_path, 64 * 1024 * 1024);
    degree_writer.write((char*)&_nd, sizeof(unsigned));
    for(unsigned i =0; i< degree_id_pair.size(); i++){
        degree_writer.write((char*)&degree_id_pair[i].second, sizeof(unsigned));
    }
    degree_writer.close();
    std::string output_path_degree = output_path_prefix + "_degree.value";
    cached_ofstream degree_value_writer(output_path_degree, 64 * 1024 * 1024);
    degree_value_writer.write((char*)&_nd, sizeof(unsigned));
    for(unsigned i =0; i< indegree.size(); i++){
        degree_value_writer.write((char*)&indegree[i], sizeof(unsigned));
    }
    degree_value_writer.close();
}

void calmedoidcacheindegree(std::string index_prefix){
    std::string degree_sorted_id_path = index_prefix + "_degree_sorted.id";
    std::string medoid_cache_id_path = index_prefix + "_medoid_list.bin";

    std::vector<unsigned> sorted_ids;
    std::vector<unsigned> medoid_cache_ids;
    std::ifstream degree_reader(degree_sorted_id_path);
    std::ifstream medoid_cache_reader(medoid_cache_id_path);
    uint32_t degree_nd;
    uint32_t medoid_cache_nd;
    degree_reader.read((char*)&degree_nd, sizeof(uint32_t));
    sorted_ids.resize(degree_nd);
    degree_reader.read((char*)sorted_ids.data(), degree_nd * sizeof(unsigned));

    medoid_cache_reader.read((char*)&medoid_cache_nd, sizeof(uint32_t));
    medoid_cache_ids.resize(medoid_cache_nd);
    medoid_cache_reader.read((char*)medoid_cache_ids.data(), medoid_cache_nd * sizeof(unsigned));
    std::cout<<"degree sorted nd: "<<degree_nd<<std::endl;
    std::cout<<"medoid cache nd: "<<medoid_cache_nd<<std::endl;

    //使用unordered_map存储degree排序后的id+位值，方便查询
    std::unordered_map<unsigned, unsigned> degree_id_map;
    for(size_t i =0; i< sorted_ids.size(); i++){
        degree_id_map[sorted_ids[i]] = i;
    }
    //find the position of medoid cache ids in degree sorted ids
    std::vector<unsigned> medoid_cache_degree_rank;
    for(size_t i =0; i< medoid_cache_ids.size(); i++){
        unsigned id = medoid_cache_ids[i];
        if(degree_id_map.find(id) != degree_id_map.end()){
            medoid_cache_degree_rank.push_back(degree_id_map[id]);
        }
    }

    //在medoid cache_degree_rank每间隔1000个degree rank，统计一下这前1000*i个在sorted_ids中的，最远的哪个位值，比如前1000个degree rank中，位值最远的一个是800，那么就记录800；前2000个degree rank中，位值最远的一个是900，那么就记录900；以此类推。这样就可以得到一个degree rank和位值的关系曲线。
    std::vector<std::pair<unsigned, unsigned>> degree_rank_bit_pair;
    unsigned total_in = 0;
    for(size_t i =0; i< medoid_cache_degree_rank.size() ; i++){
        unsigned cur_ranker = medoid_cache_degree_rank[i];
        if(cur_ranker > 5000000){
            total_in++;
        }
        if(i % 1000 == 0){
            std::cout<<i<<": "<<total_in<<", ";
        }
    }
    std::cout<<std::endl;
}

void transtobitlevelgraph(const char* indexname, const char* base_path, std::string index_prefix, int set_navi_dim, std::string dataname){
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

    

    size_t index_partition_nums = _nnodes_per_sector > 0 ? DIV_ROUND_UP(_nd, _nnodes_per_sector) : _nd;
    size_t _nodes_per_io = _nnodes_per_sector > 0 ? _nnodes_per_sector : 1;

    size_t size_per_io = diskann::defaults::SECTOR_LEN * (_nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, diskann::defaults::SECTOR_LEN));
    std::cout<<index_partition_nums<<" "<<_nodes_per_io<<" "<<size_per_io<<std::endl;
    std::cout<<_nd<<std::endl;

    MemoryMappedFile mem_index_file(indexname);
    const char* mem_index_data = static_cast<const char*>(mem_index_file.get_data()) + diskann::defaults::SECTOR_LEN; // skip meta data

    std::vector<std::vector<unsigned>> full_graph;
    full_graph.resize(_nd);
    _u64 des = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : des)
    for(size_t i = 0; i < index_partition_nums; i++){
        std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(size_per_io);
        memset(sector_buf.get(), 0, size_per_io);
        memcpy(sector_buf.get(), mem_index_data + i * size_per_io, size_per_io);
        for(size_t j = 0; j < _nodes_per_io && i * _nodes_per_io + j < _nd; j++) {
            std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(_max_node_len);
            memset(node_buf.get(), 0, _max_node_len);
            memcpy(node_buf.get(), sector_buf.get() + j * _max_node_len, _max_node_len);
            unsigned &nnbr = *(unsigned *)(node_buf.get() + _dim * sizeof(float));
            unsigned *nhood_buf = (unsigned *)(node_buf.get() + (_dim * sizeof(float)) + sizeof(unsigned));
            std::vector<unsigned> tmp(nnbr);
            memcpy((char *)tmp.data(), nhood_buf, nnbr * sizeof(unsigned));
            full_graph[i * _nodes_per_io + j].assign(tmp.begin(), tmp.end());
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

    int navi_dim = set_navi_dim;
    unsigned neighbor_width = 64;
    int chopbitsize = 16;

    std::string navi_graph_path = index_prefix + "_" + dataname + "_navi_bit_graph.index";

    BitPreserver preserver_navi(navi_dim, chopbitsize);

    cached_ofstream diskann_writer_navi(navi_graph_path, write_blk_size);
    //for navi part
    diskann_writer_navi.write((char*)&data_num, sizeof(unsigned));
    diskann_writer_navi.write((char*)&navi_dim, sizeof(unsigned));
    diskann_writer_navi.write((char*)&neighbor_width, sizeof(unsigned));
    diskann_writer_navi.write((char*)&chopbitsize, sizeof(unsigned));

    std::vector<float> restored_vec(navi_dim, 0.0f);

    for(size_t i = 0; i < data_num; i++ ) {
        auto buffer = preserver_navi.saveFloatHigherBitsLD16bit(data_ptr + i * data_dim, navi_dim);
        diskann_writer_navi.write(buffer.get(), preserver_navi.getChopSize());

        preserver_navi.loadFloatFromHigh16BitsAVX512(buffer.get(), restored_vec.data());
        float x2p = l2_sqr(restored_vec.data(), data_ptr + i * data_dim, navi_dim);
        float x2p_sqrt = std::sqrt(x2p);
        diskann_writer_navi.write((char*)&x2p_sqrt, sizeof(float));
        //write graph structure
        unsigned neighborsize = full_graph[i].size();
        char nboor_buf[neighbor_width * sizeof(unsigned) + sizeof(unsigned)];
        memcpy(nboor_buf, (char*)&neighborsize, sizeof(unsigned));
        memcpy(nboor_buf + sizeof(unsigned), (char*)full_graph[i].data(), neighborsize * sizeof(unsigned));
        //pad zeros
        //if(neighborsize < neighbor_width){
        //    memset(nboor_buf + sizeof(unsigned) + neighborsize * sizeof(unsigned), 0, (neighbor_width - neighborsize) * sizeof(unsigned));
        //}
        diskann_writer_navi.write(nboor_buf, neighbor_width * sizeof(unsigned) + sizeof(unsigned));
    }
    diskann_writer_navi.close();
    /*
    size_t size_data_per_io = diskann::defaults::SECTOR_LEN * (data_dim <= 1024 ? 1 : DIV_ROUND_UP(data_dim * sizeof(float), diskann::defaults::SECTOR_LEN));
    std::string data_sector_path = index_prefix + "_data_sector.bin";
    cached_ofstream diskann_writer_data(data_sector_path, write_blk_size);
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(diskann::defaults::SECTOR_LEN);
    diskann_writer_data.write(sector_buf.get(), diskann::defaults::SECTOR_LEN); //write meta data later
    
    for(size_t i = 0; i < data_num; i++){
        std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(size_data_per_io);
        memset(sector_buf.get(), 0, size_data_per_io);
        memcpy(sector_buf.get(), (char*)(data_ptr + i * data_dim), data_dim * sizeof(float));
        diskann_writer_data.write(sector_buf.get(), size_data_per_io);
    }
    diskann_writer_data.close();

    std::ofstream meta_writer(data_sector_path, std::ios::in | std::ios::out | std::ios::binary);
    meta_writer.seekp(0, std::ios::beg);
    std::vector<_u64> data_meta;
    data_meta.push_back(data_num);
    data_meta.push_back(data_dim);
    data_meta.push_back(meta_pair.second[2]); //medoid
    data_meta.push_back(data_dim * sizeof(float)); //max_node_len
    unsigned nodes_per_sector = diskann::defaults::SECTOR_LEN / (data_dim * sizeof(float));
    data_meta.push_back(nodes_per_sector); //_nnodes_per_sector for data
    _u64 data_file_size = diskann::defaults::SECTOR_LEN + size_data_per_io * data_num;
    data_meta.push_back(data_file_size);
    int meta_num = data_meta.size();
    int meta_dim = 1;
    meta_writer.write((char*)&meta_num, sizeof(int));
    meta_writer.write((char*)&meta_dim, sizeof(int));
    meta_writer.write((char*)data_meta.data(), data_meta.size() * sizeof(_u64));
    meta_writer.close();
    */
}

void transtodimlevelgraph(const char* indexname, const char* base_path, std::string index_prefix, int set_navi_dim, std::string dataname){
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

    

    size_t index_partition_nums = _nnodes_per_sector > 0 ? DIV_ROUND_UP(_nd, _nnodes_per_sector) : _nd;
    size_t _nodes_per_io = _nnodes_per_sector > 0 ? _nnodes_per_sector : 1;

    size_t size_per_io = diskann::defaults::SECTOR_LEN * (_nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, diskann::defaults::SECTOR_LEN));

    MemoryMappedFile mem_index_file(indexname);
    const char* mem_index_data = static_cast<const char*>(mem_index_file.get_data()) + diskann::defaults::SECTOR_LEN; // skip meta data

    std::vector<std::vector<unsigned>> full_graph;
    full_graph.resize(_nd);
    _u64 des = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : des)
    for(size_t i = 0; i < index_partition_nums; i++){
        std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(size_per_io);
        memset(sector_buf.get(), 0, size_per_io);
        memcpy(sector_buf.get(), mem_index_data + i * size_per_io, size_per_io);
        for(size_t j = 0; j < _nodes_per_io && i * _nodes_per_io + j < _nd; j++) {
            std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(_max_node_len);
            memset(node_buf.get(), 0, _max_node_len);
            memcpy(node_buf.get(), sector_buf.get() + j * _max_node_len, _max_node_len);
            unsigned &nnbr = *(unsigned *)(node_buf.get() + _dim * sizeof(float));
            unsigned *nhood_buf = (unsigned *)(node_buf.get() + (_dim * sizeof(float)) + sizeof(unsigned));
            std::vector<unsigned> tmp(nnbr);
            memcpy((char *)tmp.data(), nhood_buf, nnbr * sizeof(unsigned));
            full_graph[i * _nodes_per_io + j].assign(tmp.begin(), tmp.end());
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

    int navi_dim = set_navi_dim;
    unsigned neighbor_width = 64;
    int chopbitsize = 0;

    //std::string navi_graph_path = index_prefix + "_" + dataname + "_navi_graph.index" + std::to_string(navi_dim);
    std::string navi_graph_path = index_prefix + "_" + dataname + "_navi_graph.index";

    cached_ofstream diskann_writer_navi(navi_graph_path, write_blk_size);
    //for navi part
    diskann_writer_navi.write((char*)&data_num, sizeof(unsigned));
    diskann_writer_navi.write((char*)&navi_dim, sizeof(unsigned));
    diskann_writer_navi.write((char*)&neighbor_width, sizeof(unsigned));
    diskann_writer_navi.write((char*)&chopbitsize, sizeof(unsigned));

    for(size_t i = 0; i < data_num; i++ ) {
        char * navi_data = (char*)(data_ptr + i * data_dim);
        diskann_writer_navi.write(navi_data, navi_dim * sizeof(float));

        //write graph structure
        unsigned neighborsize = full_graph[i].size();
        char nboor_buf[neighbor_width * sizeof(unsigned) + sizeof(unsigned)];
        memcpy(nboor_buf, (char*)&neighborsize, sizeof(unsigned));
        memcpy(nboor_buf + sizeof(unsigned), (char*)full_graph[i].data(), neighborsize * sizeof(unsigned));
        //pad zeros
        if(neighborsize < neighbor_width){
            memset(nboor_buf + sizeof(unsigned) + neighborsize * sizeof(unsigned), 0, (neighbor_width - neighborsize) * sizeof(unsigned));
        }
        diskann_writer_navi.write(nboor_buf, neighbor_width * sizeof(unsigned) + sizeof(unsigned));
    }
    diskann_writer_navi.close();
    /*
    size_t size_data_per_io = diskann::defaults::SECTOR_LEN * (data_dim <= 1024 ? 1 : DIV_ROUND_UP(data_dim * sizeof(float), diskann::defaults::SECTOR_LEN));
    std::string data_sector_path = index_prefix + "_" + dataname + "_data_sector.bin";
    cached_ofstream diskann_writer_data(data_sector_path, write_blk_size);
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(diskann::defaults::SECTOR_LEN);
    diskann_writer_data.write(sector_buf.get(), diskann::defaults::SECTOR_LEN); //write meta data later
    
    for(size_t i = 0; i < data_num; i++){
        std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(size_data_per_io);
        memset(sector_buf.get(), 0, size_data_per_io);
        memcpy(sector_buf.get(), (char*)(data_ptr + i * data_dim), data_dim * sizeof(float));
        diskann_writer_data.write(sector_buf.get(), size_data_per_io);
    }
    diskann_writer_data.close();

    std::ofstream meta_writer(data_sector_path, std::ios::in | std::ios::out | std::ios::binary);
    meta_writer.seekp(0, std::ios::beg);
    std::vector<_u64> data_meta;
    data_meta.push_back(data_num);
    data_meta.push_back(data_dim);
    data_meta.push_back(meta_pair.second[2]); //medoid
    data_meta.push_back(data_dim * sizeof(float)); //max_node_len
    unsigned nodes_per_sector = diskann::defaults::SECTOR_LEN / (data_dim * sizeof(float));
    data_meta.push_back(nodes_per_sector); //_nnodes_per_sector for data
    _u64 data_file_size = diskann::defaults::SECTOR_LEN + size_data_per_io * data_num;
    data_meta.push_back(data_file_size);
    int meta_num = data_meta.size();
    int meta_dim = 1;
    meta_writer.write((char*)&meta_num, sizeof(int));
    meta_writer.write((char*)&meta_dim, sizeof(int));
    meta_writer.write((char*)data_meta.data(), data_meta.size() * sizeof(_u64));
    meta_writer.close();
    */
}

void transdatabitlevelScanMem(const char* base_path, std::string index_prefix, int navi_dim){
    MemoryMappedFile base_file(base_path);
    const char* file_data = static_cast<const char*>(base_file.get_data());
    unsigned data_num = *reinterpret_cast<const unsigned*>(file_data);
    unsigned data_dim = *(reinterpret_cast<const unsigned*>(file_data) + 1);
    const float* data_ptr = reinterpret_cast<const float*>(file_data + 2 * sizeof(unsigned));
    std::cout << "data num: " << data_num << " data dim: " << data_dim << std::endl;

    _u64            read_blk_size = 64 * 1024 * 1024;
    _u64            write_blk_size = read_blk_size;


    cached_ofstream diskann_writer_scan(index_prefix + "_data_scan_inmem" + std::to_string(navi_dim) + ".index", write_blk_size);

    int chopbitsize = 16;
    BitPreserver preserver(navi_dim, chopbitsize);
    diskann_writer_scan.write((char*)&data_num, sizeof(unsigned));
    diskann_writer_scan.write((char*)&navi_dim, sizeof(unsigned));
    diskann_writer_scan.write((char*)&chopbitsize, sizeof(unsigned));

    std::vector<float> restored_vec(navi_dim, 0.0f);
    
    for(size_t i = 0; i < data_num; i++ ) {
        auto buffer = preserver.saveFloatHigherBitsLD16bit(data_ptr + i * data_dim, navi_dim);
        preserver.loadFloatFromHigh16BitsAVX512(buffer.get(), restored_vec.data());
        float x2p = l2_sqr(restored_vec.data(), data_ptr + i * data_dim, navi_dim);
        float x2p_sqrt = std::sqrt(x2p);
        diskann_writer_scan.write((char*)&x2p_sqrt, sizeof(float));

        diskann_writer_scan.write(buffer.get(), preserver.getChopSize());
    }
    diskann_writer_scan.close();

}

void transdatatoblock(const char* indexname, const char* base_path, std::string index_prefix, std::string dataname){

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

    MemoryMappedFile base_file(base_path);
    const char* file_data = static_cast<const char*>(base_file.get_data());
    unsigned data_num = *reinterpret_cast<const unsigned*>(file_data);
    unsigned data_dim = *(reinterpret_cast<const unsigned*>(file_data) + 1);
    const float* data_ptr = reinterpret_cast<const float*>(file_data + 2 * sizeof(unsigned));
    std::cout << "data num: " << data_num << " data dim: " << data_dim << std::endl;

    size_t size_data_per_io = diskann::defaults::SECTOR_LEN * (data_dim <= 1024 ? 1 : DIV_ROUND_UP(data_dim * sizeof(float), diskann::defaults::SECTOR_LEN));
    std::string data_sector_path = index_prefix +  "_" + dataname + "_data_sector.bin";
    cached_ofstream diskann_writer_data(data_sector_path, write_blk_size);
    std::unique_ptr<char[]> sector_buf_meta = std::make_unique<char[]>(diskann::defaults::SECTOR_LEN);
    diskann_writer_data.write(sector_buf_meta.get(), diskann::defaults::SECTOR_LEN); //write meta data later

    for(size_t i = 0; i < data_num; i++){
        std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(size_data_per_io);
        memset(sector_buf.get(), 0, size_data_per_io);
        memcpy(sector_buf.get(), (char*)(data_ptr + i * data_dim), data_dim * sizeof(float));
        diskann_writer_data.write(sector_buf.get(), size_data_per_io);
    }
    diskann_writer_data.close();

    std::ofstream meta_writer(data_sector_path, std::ios::in | std::ios::out | std::ios::binary);
    meta_writer.seekp(0, std::ios::beg);
    std::vector<_u64> data_meta;
    data_meta.push_back(data_num);
    data_meta.push_back(data_dim);
    data_meta.push_back(meta_pair.second[2]); //medoid
    data_meta.push_back(data_dim * sizeof(float)); //max_node_len
    unsigned nodes_per_sector = diskann::defaults::SECTOR_LEN / (data_dim * sizeof(float));
    data_meta.push_back(nodes_per_sector); //_nnodes_per_sector for data
    _u64 data_file_size = diskann::defaults::SECTOR_LEN + size_data_per_io * data_num;
    data_meta.push_back(data_file_size);
    int meta_num = data_meta.size();
    int meta_dim = 1;
    meta_writer.write((char*)&meta_num, sizeof(int));
    meta_writer.write((char*)&meta_dim, sizeof(int));
    meta_writer.write((char*)data_meta.data(), data_meta.size() * sizeof(_u64));
    meta_writer.close();
}

int main(int argc, char** argv){
    int index = 1;
    int buildtype = std::stoi(argv[index++]);
    std::string dataname(argv[index++]);
    std::string index_prefix(argv[index++]);
    std::string dataset_path(argv[index++]);
    int subdim = std::stoi(argv[index++]);
    
    if(buildtype == 1){
        //tpye 1: split dataset into subdata set for PQ
        std::string pca_data_path = dataset_path + "3" + dataname + "_base.bin";
        splitData(pca_data_path.c_str(), index_prefix, subdim);
    }
    if(buildtype == 2){
        //type 2: transto datablock for disk data index.
        std::string indexname = index_prefix + "_" + dataname + "_disk.index";
        std::string pca_data_path = dataset_path + "3" + dataname + "_base.bin";
        transdatatoblock(indexname.c_str(), pca_data_path.c_str(), index_prefix, dataname);
    }
    if(buildtype == 3){
        //type 3: transto navi graph with bit encoding
        std::string indexname = index_prefix + "_" + dataname + "_disk.index";
        std::string pca_data_path = dataset_path + "3" + dataname + "_base.bin";
        transtobitlevelgraph(indexname.c_str(), pca_data_path.c_str(), index_prefix, subdim, dataname);    
    }
    if(buildtype == 4){
        //type 4: transto navi graph without bit encoding
        std::string indexname = index_prefix + "_" + dataname + "_disk.index";
        std::string pca_data_path = dataset_path + "3" + dataname + "_base.bin";
        transtodimlevelgraph(indexname.c_str(), pca_data_path.c_str(), index_prefix, subdim, dataname);    
    }
    if(buildtype == 5){
        //type 5: exact and save graph nodes
        std::string indexname = index_prefix + "_" + dataname + "_disk.index";
        extractAndSaveGraphNode(indexname.c_str(), index_prefix);
    }
    if(buildtype == 6){
        //type 6: cal degree and sort
        std::string indexname = index_prefix + "_" + dataname + "_disk.index";
        std::string nodes_prefix = index_prefix + "_" + dataname;
        calDegreeAndSort(indexname.c_str(), nodes_prefix);
    }
    return 0;
}
