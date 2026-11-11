#include "aligned_file_reader.h"
#include "ssd_index.h"
#include <malloc.h>
#include <filesystem>

#include <omp.h>
#include <cmath>
#include "nbr/abstract_nbr.h"
#include "ssd_index_defs.h"
#include "utils/timer.h"
#include "utils.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "utils/tsl/robin_set.h"

namespace pipeann {
  template<typename T>
  DiskNode<T>::DiskNode(uint32_t id, T *coords, uint32_t *nhood) : id(id) {
    this->coords = coords;
    this->nnbrs = *nhood;
    this->nbrs = nhood + 1;
  }

  // structs for DiskNode
  template struct DiskNode<float>;
  template struct DiskNode<uint8_t>;
  template struct DiskNode<int8_t>;

  template<typename T, typename TagT>
  SSDIndex<T, TagT>::SSDIndex(pipeann::Metric m, std::shared_ptr<AlignedFileReader> &fileReader,
                              AbstractNeighbor<T> *nbr_handler, bool tags, Parameters *params)
      : reader(fileReader), nbr_handler(nbr_handler), data_is_normalized(false), enable_tags(tags) {
    if (m == pipeann::Metric::COSINE) {
      if (std::is_floating_point<T>::value) {
        LOG(INFO) << "Cosine metric chosen for (normalized) float data."
                     "Changing distance to L2 to boost accuracy.";
        m = pipeann::Metric::L2;
        data_is_normalized = true;

      } else {
        LOG(ERROR) << "WARNING: Cannot normalize integral data types."
                   << " This may result in erroneous results or poor recall."
                   << " Consider using L2 distance with integral data types.";
      }
    }

    this->dist_cmp.reset(pipeann::get_distance_function<T>(m));

    if (params != nullptr) {
      this->beam_width = params->beam_width;
      this->l_index = params->L;
      this->range = params->R;
      this->maxc = params->C;
      this->alpha = params->alpha;
      LOG(INFO) << "Beamwidth: " << this->beam_width << ", L: " << this->l_index << ", R: " << this->range
                << ", C: " << this->maxc;
    }
    LOG(INFO) << "Use " << nbr_handler->get_name() << " as neighbor handler.";
  }

  template<typename T, typename TagT>
  SSDIndex<T, TagT>::~SSDIndex() {
    LOG(INFO) << "Lock table size: " << this->idx_lock_table.size();
    LOG(INFO) << "Page cache size: " << v2::cache.cache.size();

    if (load_flag) {
      this->destroy_buffers();
      reader->close();
    }

    if(navi_graph_ != nullptr){
      free(navi_graph_);
      navi_graph_ = nullptr;
    }

    if(pcabit_cache_ != nullptr){
      pipeann::aligned_free(pcabit_cache_);
      pcabit_cache_ = nullptr;
    }
    if(landmarks_dist_ != nullptr){
      delete [] landmarks_dist_;
      landmarks_dist_ = nullptr;
    }

    if(base_vecs_ != nullptr){
      delete [] base_vecs_;
      base_vecs_ = nullptr;
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::copy_index(const std::string &prefix_in, const std::string &prefix_out) {
    LOG(INFO) << "Copying disk index from " << prefix_in << " to " << prefix_out;
    std::filesystem::copy(prefix_in + "_disk.index", prefix_out + "_disk.index",
                          std::filesystem::copy_options::overwrite_existing);
    if (std::filesystem::exists(prefix_in + "_disk.index.tags")) {
      std::filesystem::copy(prefix_in + "_disk.index.tags", prefix_out + "_disk.index.tags",
                            std::filesystem::copy_options::overwrite_existing);
    } else {
      // remove the original tags.
      std::filesystem::remove(prefix_out + "_disk.index.tags");
    }

    // nbr.
    this->nbr_handler->load(prefix_in.c_str());
    this->nbr_handler->save(prefix_out.c_str());

    // partition data
    if (std::filesystem::exists(prefix_in + "_partition.bin.aligned")) {
      std::filesystem::copy(prefix_in + "_partition.bin.aligned", prefix_out + "_partition.bin.aligned",
                            std::filesystem::copy_options::overwrite_existing);
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::init_buffers(uint64_t n_threads) {
    uint64_t n_buffers = n_threads * 2;
    LOG(INFO) << "Init buffers for " << n_threads << " threads, setup " << n_buffers << " buffers.";
    this->thread_data_queue.null_T = nullptr;
    for (uint64_t i = 0; i < n_buffers; i++) {
      QueryBuffer<T> *data = new QueryBuffer<T>();
      this->init_query_buf(*data);
      this->thread_data_bufs.push_back(data);
      this->thread_data_queue.push(data);
      this->reader->register_buf(data->sector_scratch, MAX_N_SECTOR_READS * SECTOR_LEN, 0);
    }

#ifndef READ_ONLY_TESTS
    // background thread.
    LOG(INFO) << "Setup " << kBgIOThreads << " background I/O threads for insert...";
    for (int i = 0; i < kBgIOThreads; ++i) {
      bg_io_thread_[i] = new std::thread(&SSDIndex<T, TagT>::bg_io_thread, this);
      bg_io_thread_[i]->detach();
    }
#endif
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::destroy_buffers() {
#ifndef READ_ONLY_TESTS
    for (int i = 0; i < kBgIOThreads; ++i) {
      if (bg_io_thread_[i] != nullptr) {
        auto bg_task = new BgTask{
            .thread_data = nullptr, .writes = {}, .pages_to_unlock = {}, .pages_to_deref = {}, .terminate = true};
        bg_tasks.push(bg_task);
        bg_tasks.push_notify_all();
        bg_io_thread_[i] = nullptr;
      }
    }
#endif

    while (!this->thread_data_bufs.empty()) {
      auto buf = this->thread_data_bufs.back();
      pipeann::aligned_free((void *) buf->coord_scratch);
      pipeann::aligned_free((void *) buf->sector_scratch);
      pipeann::aligned_free((void *) buf->nbr_vec_scratch);
      pipeann::aligned_free((void *) buf->nbr_ctx_scratch);
      pipeann::aligned_free((void *) buf->aligned_dist_scratch);
      pipeann::aligned_free((void *) buf->aligned_query_T);
      this->thread_data_bufs.pop_back();
      this->thread_data_queue.pop();
      delete buf;
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_mem_index(Metric metric, const size_t query_dim, const std::string &mem_index_path) {
    if (mem_index_path.empty()) {
      LOG(ERROR) << "mem_index_path is needed";
      exit(-1);
    }
    mem_index_ = std::make_unique<pipeann::Index<T, uint32_t>>(metric, query_dim, 0, false, false, true);
    mem_index_->load(mem_index_path.c_str());
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load_base_vectors(const std::string &base_vectors_path) {
    if (!std::filesystem::exists(base_vectors_path)) {
      LOG(ERROR) << "Base vector file " << base_vectors_path << " does not exist.";
      return -1;
    }

    T *loaded = nullptr;
    size_t npts = 0, dim = 0;
    pipeann::load_bin<T>(base_vectors_path, loaded, npts, dim);
    if (npts != num_points || dim != data_dim) {
      LOG(ERROR) << "Base vector shape mismatch: got " << npts << " x " << dim
                 << ", expected " << num_points << " x " << data_dim;
      delete [] loaded;
      return -1;
    }

    // The normal search path compares against normalized vectors for cosine.
    // Keep the in-memory exact reranking path equivalent.
    if (data_is_normalized) {
      if constexpr (std::is_floating_point<T>::value) {
#pragma omp parallel for
        for (int64_t i = 0; i < static_cast<int64_t>(npts); ++i) {
          T *vec = loaded + static_cast<size_t>(i) * dim;
          const float norm = pipeann::compute_l2_norm(vec, dim);
          if (norm > 0.0f) {
            for (size_t d = 0; d < dim; ++d) {
              vec[d] = static_cast<T>(vec[d] / norm);
            }
          }
        }
      }
    }

    delete [] base_vecs_;
    base_vecs_ = loaded;
    base_vecs_num_ = npts;
    base_vecs_dim_ = dim;
    LOG(INFO) << "Loaded " << npts << " original vectors (dim=" << dim
              << ") for in-memory distance computation.";
    return 0;
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load(const char *index_prefix, uint32_t num_threads, bool new_index_format,
                              bool use_page_search) {
    std::string disk_index_file = std::string(index_prefix) + "_disk.index";
    this->_disk_index_file = disk_index_file;

    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);

    // load nbrs (e.g., PQ)
    nbr_handler->load(index_prefix);

    // read index metadata
    // open AlignedFileReader handle to index_file
    if (!std::filesystem::exists(disk_index_file)) {
      LOG(ERROR) << "Index file " << disk_index_file << " does not exist!";
      exit(-1);
    }

    this->destroy_buffers();  // in case of re-init.
    reader->open(disk_index_file, true, false);
    this->init_buffers(num_threads);
    this->max_nthreads = num_threads;

    // load page layout.
    this->use_page_search_ = use_page_search;
    this->load_page_layout(index_prefix, nnodes_per_sector, num_points);

    // load tags
    if (this->enable_tags) {
      std::string tag_file = disk_index_file + ".tags";
      LOG(INFO) << "Loading tags from " << tag_file;
      this->load_tags(tag_file);
    }

    load_flag = true;
    LOG(INFO) << "SSDIndex loaded successfully.";
    return 0;
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load_pca_index(const char *index_prefix, uint32_t num_threads, bool new_index_format, bool use_page_search) {
    std::string disk_index_file = std::string(index_prefix) + "_data_sector.bin";
    std::string navi_graph_path = std::string(index_prefix) + "_navi_graph.index";

    this->_disk_index_file = disk_index_file;
    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);

    // read index metadata
    // open AlignedFileReader handle to index_file
    if (!std::filesystem::exists(disk_index_file)) {
      LOG(ERROR) << "Index file " << disk_index_file << " does not exist!";
      exit(-1);
    }

    this->destroy_buffers();  // in case of re-init.
    reader->open(disk_index_file, true, false);
    this->init_buffers(num_threads);
    this->max_nthreads = num_threads;

    // load page layout.
    this->use_page_search_ = use_page_search;
    this->load_page_layout(index_prefix, nnodes_per_sector, num_points);

    // load tags
    if (this->enable_tags) {
      std::string tag_file = disk_index_file + ".tags";
      LOG(INFO) << "Loading tags from " << tag_file;
      this->load_tags(tag_file);
    }

    load_flag = true;
    LOG(INFO) << "SSDIndex loaded successfully.";

    
    load_pca_navi_graph(navi_graph_path);

    return 0;
  }


  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load_pca_subpq_index(const char *index_prefix, uint32_t num_threads, bool new_index_format, bool use_page_search) {
    std::string disk_index_file = std::string(index_prefix) + "_data_sector.bin";
    std::string navi_graph_path = std::string(index_prefix) + "_navi_graph.index";

    this->_disk_index_file = disk_index_file;
    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);

    nbr_handler->load_navi(index_prefix);
    
    // read index metadata
    // open AlignedFileReader handle to index_file
    if (!std::filesystem::exists(disk_index_file)) {
      LOG(ERROR) << "Index file " << disk_index_file << " does not exist!";
      exit(-1);
    }

    this->destroy_buffers();  // in case of re-init.
    reader->open(disk_index_file, true, false);
    this->init_buffers(num_threads);
    this->max_nthreads = num_threads;

    // load page layout.
    this->use_page_search_ = use_page_search;
    this->load_page_layout(index_prefix, nnodes_per_sector, num_points);

    // load tags
    if (this->enable_tags) {
      std::string tag_file = disk_index_file + ".tags";
      LOG(INFO) << "Loading tags from " << tag_file;
      this->load_tags(tag_file);
    }

    load_flag = true;
    LOG(INFO) << "SSDIndex loaded successfully.";

    
    load_pca_navi_graph(navi_graph_path);

    return 0;
  }
  
  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load_pca_bit_index(const char *index_prefix, uint32_t num_threads, bool new_index_format, bool use_page_search) {
    mixed_pivot_.reset();
    return load_pca_pivot_index(index_prefix, num_threads, use_page_search, false);
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load_pca_mixed_index(const char *index_prefix, uint32_t num_threads,
      const std::string &cache_ids, uint64_t cache_count) {
    mixed_cache_ = cache_count != UINT64_MAX;
    mixed_cache_ids_ = cache_ids;
    mixed_cache_count_ = mixed_cache_ ? cache_count : 0;
    if (!std::is_same<T, float>::value || data_is_normalized || num_threads == 0)
      throw std::invalid_argument("Mixed pivots require float32 L2 and positive thread count");
    return load_pca_pivot_index(index_prefix, num_threads, false, true);
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load_pca_pivot_index(const char *index_prefix, uint32_t num_threads,
                                           bool use_page_search, bool mixed) {
    mixed_subpq_navigation_ = false;
    std::string disk_index_file = std::string(index_prefix) + "_data_sector.bin";
    std::string navi_graph_path = std::string(index_prefix) + "_navi_bit_graph.index";

    this->_disk_index_file = disk_index_file;
    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    // Mixed pivots keep all graph edges in RAM; SSD records may contain only
    // coordinates. Avoid the legacy unsigned degree underflow in that layout.
    if (mixed && meta.max_node_len == meta.data_dim * sizeof(T)) meta.range = 0;
    this->init_metadata(meta);

    // read index metadata
    // open AlignedFileReader handle to index_file
    if (!std::filesystem::exists(disk_index_file)) {
      LOG(ERROR) << "Index file " << disk_index_file << " does not exist!";
      exit(-1);
    }

    this->destroy_buffers();  // in case of re-init.
    reader->open(disk_index_file, true, false);
    this->init_buffers(num_threads);
    this->max_nthreads = num_threads;

    // load page layout.
    this->use_page_search_ = use_page_search;
    this->load_page_layout(index_prefix, nnodes_per_sector, num_points);

    // load tags
    if (this->enable_tags) {
      std::string tag_file = disk_index_file + ".tags";
      LOG(INFO) << "Loading tags from " << tag_file;
      this->load_tags(tag_file);
    }

    load_flag = true;
    LOG(INFO) << "SSDIndex loaded successfully.";

    
    if (mixed)
      load_pca_mixed_navi_graph(std::string(index_prefix) + "_navi_mixed_graph.index");
    else
      load_pca_bit_navi_graph(navi_graph_path);

    return 0;
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load_pca_bit_pq_index(const char *index_prefix, uint32_t num_threads, bool new_index_format, bool use_page_search) {
    std::string disk_index_file = std::string(index_prefix) + "_data_sector.bin";
    std::string navi_graph_path = std::string(index_prefix) + "_navi_bit_graph.index";

    this->_disk_index_file = disk_index_file;
    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);

    // load nbrs (e.g., PQ)
    nbr_handler->load(index_prefix);

    // read index metadata
    // open AlignedFileReader handle to index_file
    if (!std::filesystem::exists(disk_index_file)) {
      LOG(ERROR) << "Index file " << disk_index_file << " does not exist!";
      exit(-1);
    }

    this->destroy_buffers();  // in case of re-init.
    reader->open(disk_index_file, true, false);
    this->init_buffers(num_threads);
    this->max_nthreads = num_threads;

    // load page layout.
    this->use_page_search_ = use_page_search;
    this->load_page_layout(index_prefix, nnodes_per_sector, num_points);

    // load tags
    if (this->enable_tags) {
      std::string tag_file = disk_index_file + ".tags";
      LOG(INFO) << "Loading tags from " << tag_file;
      this->load_tags(tag_file);
    }

    load_flag = true;
    LOG(INFO) << "SSDIndex loaded successfully.";

    
    load_pca_bit_navi_graph(navi_graph_path);

    return 0;
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load_pca_bit_subpq_index(const char *index_prefix, uint32_t num_threads, bool new_index_format, bool use_page_search) {
    std::string disk_index_file = std::string(index_prefix) + "_data_sector.bin";
    std::string navi_graph_path = std::string(index_prefix) + "_navi_bit_graph.index";
    std::cout<<"navi_graph_path: " << navi_graph_path << std::endl;

    this->_disk_index_file = disk_index_file;
    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);

    // load nbrs (e.g., PQ)
    nbr_handler->load_navi(index_prefix);
    //for rabitq
    //nbr_handler->load(index_prefix);

    // read index metadata
    // open AlignedFileReader handle to index_file
    if (!std::filesystem::exists(disk_index_file)) {
      LOG(ERROR) << "Index file " << disk_index_file << " does not exist!";
      exit(-1);
    }

    this->destroy_buffers();  // in case of re-init.
    reader->open(disk_index_file, true, false);
    this->init_buffers(num_threads);
    this->max_nthreads = num_threads;

    // load page layout.
    this->use_page_search_ = use_page_search;
    this->load_page_layout(index_prefix, nnodes_per_sector, num_points);

    // load tags
    if (this->enable_tags) {
      std::string tag_file = disk_index_file + ".tags";
      LOG(INFO) << "Loading tags from " << tag_file;
      this->load_tags(tag_file);
    }

    load_flag = true;
    LOG(INFO) << "SSDIndex loaded successfully.";

    
    load_pca_bit_navi_graph(navi_graph_path);

    return 0;
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load_pca_bit_subpq_index(const char *index_prefix, uint32_t num_threads, std::string subdim, bool new_index_format, bool use_page_search) {
    std::string disk_index_file = std::string(index_prefix) + "_data_sector.bin";
    std::string navi_graph_path = std::string(index_prefix) + "_navi_bit_graph.index" + std::string(subdim);
    std::cout<<"navi_graph_path: " << navi_graph_path << std::endl;

    this->_disk_index_file = disk_index_file;
    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);

    // load nbrs (e.g., PQ)
    nbr_handler->load_navi(index_prefix);
    //for rabitq
    //nbr_handler->load(index_prefix);

    // read index metadata
    // open AlignedFileReader handle to index_file
    if (!std::filesystem::exists(disk_index_file)) {
      LOG(ERROR) << "Index file " << disk_index_file << " does not exist!";
      exit(-1);
    }

    this->destroy_buffers();  // in case of re-init.
    reader->open(disk_index_file, true, false);
    this->init_buffers(num_threads);
    this->max_nthreads = num_threads;

    // load page layout.
    this->use_page_search_ = use_page_search;
    this->load_page_layout(index_prefix, nnodes_per_sector, num_points);

    // load tags
    if (this->enable_tags) {
      std::string tag_file = disk_index_file + ".tags";
      LOG(INFO) << "Loading tags from " << tag_file;
      this->load_tags(tag_file);
    }

    load_flag = true;
    LOG(INFO) << "SSDIndex loaded successfully.";

    
    load_pca_bit_navi_graph(navi_graph_path);

    return 0;
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load_pca_bit_subpq_cache_index(const char *index_prefix, uint32_t num_threads, bool new_index_format, bool use_page_search) {
    std::string disk_index_file = std::string(index_prefix) + "_data_sector.bin";
    std::string navi_graph_path = std::string(index_prefix) + "_navi_bit_graph.index";

    this->_disk_index_file = disk_index_file;
    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);

    // load nbrs (e.g., PQ)
    nbr_handler->load_navi(index_prefix);

    // read index metadata
    // open AlignedFileReader handle to index_file
    if (!std::filesystem::exists(disk_index_file)) {
      LOG(ERROR) << "Index file " << disk_index_file << " does not exist!";
      exit(-1);
    }

    this->destroy_buffers();  // in case of re-init.
    reader->open(disk_index_file, true, false);
    this->init_buffers(num_threads);
    this->max_nthreads = num_threads;

    // load page layout.
    this->use_page_search_ = use_page_search;
    this->load_page_layout(index_prefix, nnodes_per_sector, num_points);

    // load tags
    if (this->enable_tags) {
      std::string tag_file = disk_index_file + ".tags";
      LOG(INFO) << "Loading tags from " << tag_file;
      this->load_tags(tag_file);
    }

    load_flag = true;
    LOG(INFO) << "SSDIndex loaded successfully.";

    return 0;
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load_trim_index(const char *index_prefix, uint32_t num_threads, bool new_index_format, bool use_page_search) {
    std::string disk_index_file = std::string(index_prefix) + "_data_sector.bin";
    
    this->_disk_index_file = disk_index_file;
    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);

    // load nbrs (e.g., PQ)
    //nbr_handler->load(index_prefix);
    nbr_handler->load(index_prefix);
    //trim_nbr_handler = new pipeann::PQNeighbor<T>();
    //trim_nbr_handler->load(index_prefix);
    // read index metadata
    // open AlignedFileReader handle to index_file
    if (!std::filesystem::exists(disk_index_file)) {
      LOG(ERROR) << "Index file " << disk_index_file << " does not exist!";
      exit(-1);
    }

    this->destroy_buffers();  // in case of re-init.
    reader->open(disk_index_file, true, false);
    this->init_buffers(num_threads);
    this->max_nthreads = num_threads;

    // load page layout.
    this->use_page_search_ = use_page_search;
    this->load_page_layout(index_prefix, nnodes_per_sector, num_points);

    // load tags
    if (this->enable_tags) {
      std::string tag_file = disk_index_file + ".tags";
      LOG(INFO) << "Loading tags from " << tag_file;
      this->load_tags(tag_file);
    }

    load_flag = true;
    LOG(INFO) << "SSDIndex loaded successfully.";
    
    return 0;
  } 

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load_trim_index_navi(const char *index_prefix, uint32_t num_threads, bool new_index_format, bool use_page_search) {
    std::string disk_index_file = std::string(index_prefix) + "_data_sector.bin";
    
    this->_disk_index_file = disk_index_file;
    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);

    // load nbrs (e.g., PQ)
    //nbr_handler->load(index_prefix);
    nbr_handler->load_navi(index_prefix);
    trim_nbr_handler = new pipeann::PQNeighbor<T>();
    trim_nbr_handler->load(index_prefix);
    // read index metadata
    // open AlignedFileReader handle to index_file
    if (!std::filesystem::exists(disk_index_file)) {
      LOG(ERROR) << "Index file " << disk_index_file << " does not exist!";
      exit(-1);
    }

    this->destroy_buffers();  // in case of re-init.
    reader->open(disk_index_file, true, false);
    this->init_buffers(num_threads);
    this->max_nthreads = num_threads;

    // load page layout.
    this->use_page_search_ = use_page_search;
    this->load_page_layout(index_prefix, nnodes_per_sector, num_points);

    // load tags
    if (this->enable_tags) {
      std::string tag_file = disk_index_file + ".tags";
      LOG(INFO) << "Loading tags from " << tag_file;
      this->load_tags(tag_file);
    }

    load_flag = true;
    LOG(INFO) << "SSDIndex loaded successfully.";
    
    return 0;
  } 

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_pca_navi_graph(const std::string &navi_graph_path){
    mixed_pivot_.reset();
    std::ifstream in(navi_graph_path, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    size_t num_pts = 0;
    in.read((char*)&num_pts, 4);
    in.read((char*)&navi_dim_, 4);
    size_t max_navi_degree_ = 0;
    in.read((char*)&max_navi_degree_, 4);
    this->max_degree = max_navi_degree_;
    std::cout<<"navi graph num points: "<<num_pts<<", navi dim: "<<navi_dim_<<", max navi degree: "<<max_navi_degree_<<std::endl;
    in.read((char*)&this->high_significant_bits_, 4);

    //bit_preserver_.setNewParams(navi_dim_, this->high_significant_bits_);
        
    navi_node_len_ = navi_dim_ * sizeof(float) + sizeof(uint32_t) * (1 + max_navi_degree_);
    navi_data_len_ = navi_dim_ * sizeof(float);
    navi_graph_ = (char*) malloc(num_pts * navi_node_len_);
    if(!navi_graph_){
        std::cout << "malloc navigraph error" << std::endl;
        exit(-1);
    }
    in.read((char*)navi_graph_, num_pts * navi_node_len_);
    in.close();
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::outneighbor(){
    reverse_neighbors_.clear();
    reverse_neighbors_.resize(num_points);
    for(size_t i = 0; i < num_points; i++){
      char* node_ptr = navi_graph_ + i * navi_node_len_;
      float* coord_ptr = (float*) node_ptr;
      uint32_t* nhood_ptr = (uint32_t*)(node_ptr + navi_data_len_);
      uint32_t degree = *nhood_ptr;
      for(size_t j = 0; j < degree; j++){
        uint32_t nbr_id = *(nhood_ptr + 1 + j);
        reverse_neighbors_[nbr_id] = i;
      }
    }
  }


  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_pca_bit_navi_graph(const std::string &navi_bit_graph_path){
    mixed_pivot_.reset();
    std::ifstream in(navi_bit_graph_path, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    size_t num_pts = 0;
    in.read((char*)&num_pts, 4);
    in.read((char*)&navi_dim_, 4);
    size_t max_navi_degree_ = 0;
    in.read((char*)&max_navi_degree_, 4);
    this->max_degree = max_navi_degree_;
    std::cout<<"navi graph num points: "<<num_pts<<", navi dim: "<<navi_dim_<<", max navi degree: "<<max_navi_degree_<<std::endl;
    in.read((char*)&this->high_significant_bits_, 4);
    std::cout<<"high significant bits: "<<this->high_significant_bits_<<std::endl;

    bit_preserver_.setNewParams(navi_dim_, this->high_significant_bits_);
        
    navi_node_len_ = bit_preserver_.getChopSize() + sizeof(uint32_t) * (2 + max_navi_degree_);
    navi_data_len_ = bit_preserver_.getChopSize();
    navi_graph_ = (char*) malloc(num_pts * navi_node_len_);
    if(!navi_graph_){
        std::cout << "malloc navigraph error" << std::endl;
        exit(-1);
    }
    in.read((char*)navi_graph_, num_pts * navi_node_len_);
    in.close();
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_pcabit_cache(const char *index_prefix, size_t num_nodes_to_cache){

    std::string navi_graph_path = std::string(index_prefix) + "_navi_bit_graph.index";
    std::ifstream in(navi_graph_path, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error " << navi_graph_path << std::endl;
        exit(-1);
    }

    size_t num_pts = 0;
    in.read((char*)&num_pts, 4);
    in.read((char*)&navi_dim_, 4);
    size_t max_navi_degree_ = 0;
    in.read((char*)&max_navi_degree_, 4);
    this->max_degree = max_navi_degree_;
    std::cout<<"navi graph num points: "<<num_pts<<", navi dim: "<<navi_dim_<<", max navi degree: "<<max_navi_degree_<<std::endl;
    in.read((char*)&this->high_significant_bits_, 4);
    std::cout<<"high significant bits: "<<this->high_significant_bits_<<std::endl;

    bit_preserver_.setNewParams(navi_dim_, this->high_significant_bits_);
    navi_node_len_ = bit_preserver_.getChopSize() + sizeof(uint32_t) * (2 + max_navi_degree_);
    navi_data_len_ = bit_preserver_.getChopSize();

    size_t cache_size = num_nodes_to_cache * navi_node_len_ + (num_pts - num_nodes_to_cache) * sizeof(uint32_t) * (max_navi_degree_ + 1);

    pipeann::alloc_aligned((void**)&pcabit_cache_, cache_size, 1);
    memset(pcabit_cache_, 0, cache_size);

    std::vector<uint32_t> node_to_cache_list;
    std::string id_list_file = std::string(index_prefix) + "_degree_sorted.id";
    //std::string id_list_file = std::string(index_prefix) + "_merged_degree_medoid.id";
    std::ifstream id_list_in(id_list_file, std::ios::binary);

    if (!id_list_in.is_open()) {
        std::cout << "open file error " << id_list_file << std::endl;
        exit(-1);
    }
    size_t id_num = 0;
    id_list_in.read((char*)&id_num, 4);
    std::cout<<"id list num: "<<id_num<<std::endl;
    assert(num_nodes_to_cache <= id_num);
    node_to_cache_list.resize(num_nodes_to_cache);
    id_list_in.read((char*)node_to_cache_list.data(), num_nodes_to_cache * sizeof(uint32_t));
    id_list_in.close();
    
    size_t BLOCK_SIZE = 32;
    size_t num_blocks = DIV_ROUND_UP(num_nodes_to_cache, BLOCK_SIZE);
    for (size_t block = 0; block < num_blocks; block++)
    {
        size_t start_idx = block * BLOCK_SIZE;
        size_t end_idx = (std::min)(num_nodes_to_cache, (block + 1) * BLOCK_SIZE);

        // Copy offset into buffers to read into
        std::vector<uint32_t> nodes_to_read;
        std::vector<char *> coord_buffers;
        for (size_t node_idx = start_idx; node_idx < end_idx; node_idx++)
        {
            nodes_to_read.push_back(node_to_cache_list[node_idx]);
            coord_buffers.push_back(pcabit_cache_ + node_idx * navi_node_len_);  
        }

        // issue the reads
        size_t to_read_num = nodes_to_read.size();
        for(size_t i = 0; i < to_read_num; i++){
          in.seekg(sizeof(uint32_t) * 4 + nodes_to_read[i] * navi_node_len_, std::ios::beg);
          in.read(coord_buffers[i], navi_node_len_);
        }
        
        // check for success and insert into the cache.
        for (size_t i = 0; i < to_read_num; i++)
        {
          graph_cache_map_udm_.insert(std::make_pair(nodes_to_read[i], std::make_pair(true, coord_buffers[i])));    
        }
    }

    size_t left_nodes_idx = 0;

    for(size_t node_id = 0; node_id < num_pts; node_id++){
      if(graph_cache_map_udm_.find(node_id) == graph_cache_map_udm_.end()){
        size_t offset = sizeof(uint32_t) * 4 + node_id * navi_node_len_ + navi_data_len_ + sizeof(uint32_t); // offset to the neighbor list
        char* nbrs = pcabit_cache_ + num_nodes_to_cache * navi_node_len_ + left_nodes_idx * sizeof(uint32_t) * (max_navi_degree_ + 1);
        in.seekg(offset, std::ios::beg);
        in.read(nbrs, sizeof(uint32_t) * (max_navi_degree_ + 1));
        graph_cache_map_udm_.insert(std::make_pair(node_id, std::make_pair(false, nbrs)));
        left_nodes_idx++;
      }
    }

    in.close();
    LOG(INFO) << "PCA bit navi graph cache loaded with " << num_nodes_to_cache << " nodes cached.";
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_navi_graph(const std::string &index_prefix){
      std::cout<<"neighbor info for non-cached nodes..."<<std::flush;
      //对于非缓存节点，只读取邻居信息, 从navi_graph_nodes_中读取，并存入__coord_cache_buf中
      //其中node_list中存储的是已经缓存过的节点id，因此这些节点需要排除掉

      std::string navi_graph_file = std::string(index_prefix) + "_graph_nodes.index";
      std::ifstream in(navi_graph_file, std::ios::binary);
      if (!in.is_open()) {
          std::cout << "open file " << navi_graph_file << " error" << std::endl;
          exit(-1);
      }
      size_t num_pts = 0;
      in.read((char*)&num_pts, 4);

      navi_graph_nodes_.resize(num_pts);
      for(size_t i=0; i<num_pts; i++){
          uint32_t nnbrs = 0;
          in.read((char*)&nnbrs, 4);
          navi_graph_nodes_[i].resize(nnbrs);
          in.read((char*)navi_graph_nodes_[i].data(), nnbrs * sizeof(uint32_t));
      }
      in.close();
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_landmarks_dist(const char *index_prefix){
      std::string landmark_dist_path = std::string(index_prefix) + "_landmark_dist.bin";
      std::ifstream in_dist(landmark_dist_path, std::ios::binary);
      if (!in_dist.is_open()) {
        std::cerr << "Error opening file: " << landmark_dist_path << std::endl;
        return;
      }
      unsigned dist_num = 0;
      unsigned dist_dim = 0;
      in_dist.read(reinterpret_cast<char*>(&dist_num), sizeof(unsigned));
      in_dist.read(reinterpret_cast<char*>(&dist_dim), sizeof(unsigned));
      std::cout << "Loading landmark distances: num=" << dist_num << ", dim=" << dist_dim << std::endl;
      landmarks_dist_ = new float[dist_num];
      in_dist.read(reinterpret_cast<char*>(landmarks_dist_), dist_num * sizeof(float));
      in_dist.close();
  }

  template<typename T, typename TagT>
  uint64_t SSDIndex<T, TagT>::return_nd() {
    return this->num_points;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_page_layout(const std::string &index_prefix, const uint64_t nnodes_per_sector,
                                           const uint64_t num_points) {
    std::string partition_file = index_prefix + "_partition.bin.aligned";
    id2loc_.resize(num_points);  // pre-allocate space first.
    loc2id_.resize(cur_loc);     // pre-allocate space first.

    if (std::filesystem::exists(partition_file)) {
      LOG(INFO) << "Loading partition file " << partition_file;
      std::ifstream part(partition_file);
      uint64_t C, partition_nums, nd;
      part.read((char *) &C, sizeof(uint64_t));
      part.read((char *) &partition_nums, sizeof(uint64_t));
      part.read((char *) &nd, sizeof(uint64_t));
      if (nnodes_per_sector <= 1 || C != nnodes_per_sector) {
        // graph reordering is useful only when nnodes_per_sector > 1.
        LOG(ERROR) << "partition information not correct.";
        exit(-1);
      }
      LOG(INFO) << "Partition meta: C: " << C << " partition_nums: " << partition_nums;

      uint64_t page_offset = loc_sector_no(0);
      auto st = std::chrono::high_resolution_clock::now();

      constexpr uint64_t n_parts_per_read = 1024 * 1024;
      std::vector<unsigned> part_buf(n_parts_per_read * (1 + nnodes_per_sector));
      for (uint64_t p = 0; p < partition_nums; p += n_parts_per_read) {
        uint64_t nxt_p = std::min(p + n_parts_per_read, partition_nums);
        part.read((char *) part_buf.data(), sizeof(unsigned) * n_parts_per_read * (1 + nnodes_per_sector));
#pragma omp parallel for schedule(dynamic)
        for (uint64_t i = p; i < nxt_p; ++i) {
          uint32_t base = (i - p) * (1 + nnodes_per_sector);
          uint32_t s = part_buf[base];  // size of this partition
          for (uint32_t j = 0; j < s; ++j) {
            uint64_t id = part_buf[base + 1 + j];
            uint64_t loc = i * nnodes_per_sector + j;
            id2loc_[id] = loc;
            loc2id_[loc] = id;
          }
          for (uint32_t j = s; j < nnodes_per_sector; ++j) {
            loc2id_[i * nnodes_per_sector + j] = kInvalidID;
          }
        }
      }
      auto et = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "Page layout loaded in " << std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count()
                << " ms";
    } else {
      LOG(INFO) << partition_file << " does not exist, use equal partition mapping";
// use equal mapping for id2loc and page_layout.
#ifndef NO_MAPPING
#pragma omp parallel for
      for (size_t i = 0; i < this->num_points; ++i) {
        id2loc_[i] = i;
        loc2id_[i] = i;
      }
      for (size_t i = this->num_points; i < this->cur_loc; ++i) {
        loc2id_[i] = kInvalidID;
      }
#endif
    }
    LOG(INFO) << "Page layout loaded.";
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_tags(const std::string &tag_file_name, size_t offset) {
    size_t tag_num, tag_dim;
    std::vector<TagT> tag_v;
    this->tags.clear();

    if (!file_exists(tag_file_name)) {
      LOG(INFO) << "Tags file not found. Using equal mapping";
      // Equal mapping are by default eliminated in tags map.
    } else {
      LOG(INFO) << "Load tags from existing file: " << tag_file_name;
      pipeann::load_bin<TagT>(tag_file_name, tag_v, tag_num, tag_dim, offset);
      tags.reserve(tag_v.size());

#pragma omp parallel for num_threads(max_nthreads)
      for (size_t i = 0; i < tag_num; ++i) {
        tags.insert_or_assign(i, tag_v[i]);
      }
      LOG(INFO) << "Loaded " << tags.size() << " tags";
    }
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::get_vector_by_id(const uint32_t &id, T *vector_coords) {
    if (!enable_tags) {
      LOG(INFO) << "Tags are disabled, cannot retrieve vector";
      return -1;
    }
    uint32_t pos = id;
    size_t num_sectors = node_sector_no(pos);
    std::ifstream disk_reader(_disk_index_file.c_str(), std::ios::binary);
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(size_per_io);
    disk_reader.seekg(SECTOR_LEN * num_sectors, std::ios::beg);
    disk_reader.read(sector_buf.get(), size_per_io);
    char *node_coords = (offset_to_node(sector_buf.get(), pos));
    memcpy((void *) vector_coords, (void *) node_coords, data_dim * sizeof(T));
    return 0;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
