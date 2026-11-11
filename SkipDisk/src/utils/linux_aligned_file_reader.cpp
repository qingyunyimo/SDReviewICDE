#ifndef USE_AIO
#include "linux_aligned_file_reader.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <omp.h>
#include "aligned_file_reader.h"
#include "liburing.h"

#define MAX_EVENTS 256

namespace {
  constexpr uint64_t kNoUserData = 0;
  void execute_io(void *context, int fd, std::vector<IORequest> &reqs, uint64_t n_retries = 0, bool write = false) {
    io_uring *ring = (io_uring *) context;
    while (true) {
      for (uint64_t j = 0; j < reqs.size(); j++) {
        auto sqe = io_uring_get_sqe(ring);
        sqe->user_data = kNoUserData;
        if (write) {
          io_uring_prep_write(sqe, fd, reqs[j].buf, reqs[j].len, reqs[j].offset);
        } else {
          io_uring_prep_read(sqe, fd, reqs[j].buf, reqs[j].len, reqs[j].offset);
        }
      }
      io_uring_submit(ring);

      io_uring_cqe *cqe = nullptr;
      bool fail = false;
      for (uint64_t j = 0; j < reqs.size(); j++) {
        int ret = 0;
        do {
          ret = io_uring_wait_cqe(ring, &cqe);
        } while (ret == -EINTR);

        if (ret < 0 || cqe->res < 0) {
          fail = true;
          LOG(ERROR) << "Failed " << strerror(-ret) << " " << ring << " " << j << " " << reqs[j].buf << " "
                     << reqs[j].len << " " << reqs[j].offset;
          break;  // CQE broken.
        }
        io_uring_cqe_seen(ring, cqe);
      }
      if (!fail) {  // repeat until no fails.
        break;
      }
    }
  }
}  // namespace

LinuxAlignedFileReader::LinuxAlignedFileReader() {
  this->file_desc = -1;
}

LinuxAlignedFileReader::~LinuxAlignedFileReader() {
  int64_t ret;
  // check to make sure file_desc is closed
  ret = ::fcntl(this->file_desc, F_GETFD);
  if (ret == -1) {
    if (errno != EBADF) {
      std::cerr << "close() not called" << std::endl;
      // close file desc
      ret = ::close(this->file_desc);
      // error checks
      if (ret == -1) {
        std::cerr << "close() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno) << std::endl;
      }
    }
  }
}

namespace ioctx {
  static thread_local io_uring *ring = nullptr;
};

void *LinuxAlignedFileReader::get_ctx(int flag) {
  if (unlikely(ioctx::ring == nullptr)) {
    register_thread(flag);
  }
  return ioctx::ring;
}

void LinuxAlignedFileReader::register_thread(int flag) {
  if (ioctx::ring == nullptr) {
    ioctx::ring = new io_uring();
    io_uring_params params{};
    params.flags = flag;
    // Benchmark callers may pin the SQPOLL helper separately from the query
    // thread. Without this, a taskset-pinned caller can force both onto one CPU.
    const char *single_cpu = std::getenv("SKIPDISK_SQPOLL_CPU");
    const char *cpu_list = std::getenv("SKIPDISK_SQPOLL_CPUS");
    if ((flag & IORING_SETUP_SQPOLL) && (single_cpu || cpu_list)) {
      if (single_cpu && cpu_list)
        throw std::invalid_argument("Set only one of SKIPDISK_SQPOLL_CPU and SKIPDISK_SQPOLL_CPUS");
      std::string value;
      if (cpu_list) {
        // Each OpenMP query worker owns a ring and a separate polling helper.
        std::istringstream stream(cpu_list);
        std::vector<std::string> cpus;
        while (std::getline(stream, value, ',')) cpus.push_back(value);
        if (cpus.size() < static_cast<size_t>(omp_get_num_threads()) ||
            std::string(cpu_list).empty() || std::string(cpu_list).back() == ',')
          throw std::invalid_argument("SKIPDISK_SQPOLL_CPUS requires a CPU for each OpenMP worker");
        value = cpus.at(static_cast<size_t>(omp_get_thread_num()));
      } else value = single_cpu;
      size_t used = 0;
      const auto cpu = std::stoul(value, &used);
      if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos || used != value.size() ||
          cpu > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Invalid SKIPDISK_SQPOLL_CPU");
      params.flags |= IORING_SETUP_SQ_AFF;
      params.sq_thread_cpu = static_cast<uint32_t>(cpu);
    }
    const int result = io_uring_queue_init_params(MAX_EVENTS, ioctx::ring, &params);
    if (result < 0) {
      delete ioctx::ring; ioctx::ring = nullptr;
      throw std::runtime_error(std::string("io_uring setup failed: ") + strerror(-result));
    }
  }
}

void LinuxAlignedFileReader::deregister_thread() {
  io_uring_queue_exit(ioctx::ring);
  delete ioctx::ring;
  ioctx::ring = nullptr;
}

void LinuxAlignedFileReader::deregister_sync_thread() {
  if(ioctx::ring != nullptr){
    deregister_thread();
  }
}

void LinuxAlignedFileReader::deregister_all_threads() {
  return;
}

void LinuxAlignedFileReader::register_buf(void *buf, uint64_t buf_size, int mrid) {
  return;
}

void LinuxAlignedFileReader::open(const std::string &fname, bool enable_writes = false, bool enable_create = false) {
  int flags = O_DIRECT | O_LARGEFILE | O_RDWR;
  if (enable_create) {
    flags |= O_CREAT;
  }
  this->file_desc = ::open(fname.c_str(), flags, 0644);
  // error checks
  assert(this->file_desc != -1);
  //  std::cerr << "Opened file : " << fname << std::endl;
}

void LinuxAlignedFileReader::close() {
  //  int64_t ret;

  // check to make sure file_desc is closed
  ::fcntl(this->file_desc, F_GETFD);
  //  assert(ret != -1);

  ::close(this->file_desc);
  //  assert(ret != -1);
}

void LinuxAlignedFileReader::read(std::vector<IORequest> &read_reqs, void *ctx, bool async) {
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, read_reqs);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::write(std::vector<IORequest> &write_reqs, void *ctx, bool async) {
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, write_reqs, 0, true);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::read_fd(int fd, std::vector<IORequest> &read_reqs, void *ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, read_reqs);
}

void LinuxAlignedFileReader::write_fd(int fd, std::vector<IORequest> &write_reqs, void *ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, write_reqs, 0, true);
}

void LinuxAlignedFileReader::send_io(IORequest &req, void *ctx, bool write) {
  io_uring *ring = (io_uring *) ctx;
  auto sqe = io_uring_get_sqe(ring);
  if (sqe == nullptr) {
    LOG(ERROR) << "io_uring_get_sqe returned nullptr, ring is full";
    req.finished = true;  // avoid blocking caller-side queues
    return;
  }
  req.finished = false;
  sqe->user_data = (uint64_t) &req;
  if (write) {
    io_uring_prep_write(sqe, this->file_desc, req.buf, req.len, req.offset);
  } else {
    io_uring_prep_read(sqe, this->file_desc, req.buf, req.len, req.offset);
  }
  int submit_ret = io_uring_submit(ring);
  if (submit_ret < 0) {
    LOG(ERROR) << "io_uring_submit failed: " << strerror(-submit_ret);
    req.finished = true;  // unblock search loop even on submit failure
  }
}

void LinuxAlignedFileReader::send_io(std::vector<IORequest> &reqs, void *ctx, bool write) {
  io_uring *ring = (io_uring *) ctx;
  for (uint64_t j = 0; j < reqs.size(); j++) {
    auto sqe = io_uring_get_sqe(ring);
    if (sqe == nullptr) {
      LOG(ERROR) << "io_uring_get_sqe returned nullptr, ring is full";
      reqs[j].finished = true;
      continue;
    }
    reqs[j].finished = false;
    sqe->user_data = (uint64_t) &reqs[j];
    if (write) {
      io_uring_prep_write(sqe, this->file_desc, reqs[j].buf, reqs[j].len, reqs[j].offset);
    } else {
      io_uring_prep_read(sqe, this->file_desc, reqs[j].buf, reqs[j].len, reqs[j].offset);
    }
  }
  int submit_ret = io_uring_submit(ring);
  if (submit_ret < 0) {
    LOG(ERROR) << "io_uring_submit failed: " << strerror(-submit_ret);
    for (auto &r : reqs) {
      r.finished = true;
    }
  }
}

int LinuxAlignedFileReader::poll(void *ctx) {
  io_uring *ring = (io_uring *) ctx;
  io_uring_cqe *cqe = nullptr;
  int ret = io_uring_peek_cqe(ring, &cqe);
  if (ret < 0) {
    return ret;  // not finished yet.
  }
  if (cqe->res < 0) {
    LOG(ERROR) << "Failed " << strerror(-cqe->res);
  }
  IORequest *req = (IORequest *) cqe->user_data;
  if (req != nullptr) {
    req->finished = true;
  }
  io_uring_cqe_seen(ring, cqe);
  return 0;
}

void LinuxAlignedFileReader::poll_all(void *ctx) {
  io_uring *ring = (io_uring *) ctx;
  static __thread io_uring_cqe *cqes[MAX_EVENTS];
  int ret = io_uring_peek_batch_cqe(ring, cqes, MAX_EVENTS);
  if (ret < 0) {
    LOG(ERROR) << "io_uring_peek_batch_cqe failed: " << strerror(-ret);
    return;  // not finished yet.
  }
  for (int i = 0; i < ret; i++) {
    if (cqes[i]->res < 0) {
      LOG(ERROR) << "Failed " << strerror(-cqes[i]->res);
    }
    IORequest *req = (IORequest *) cqes[i]->user_data;
    if (req != nullptr) {
      req->finished = true;
    }
    io_uring_cqe_seen(ring, cqes[i]);
  }
}

void LinuxAlignedFileReader::poll_wait(void *ctx) {
  io_uring *ring = (io_uring *) ctx;
  io_uring_cqe *cqe = nullptr;
  int ret = 0;
  do {
    ret = io_uring_wait_cqe(ring, &cqe);
  } while (ret == -EINTR);
  if (ret < 0 || cqe->res < 0) {
    LOG(ERROR) << "Failed " << strerror(-cqe->res);
  }
  IORequest *req = (IORequest *) cqe->user_data;
  if (req != nullptr) {
    req->finished = true;
  }
  io_uring_cqe_seen(ring, cqe);
}

void LinuxAlignedFileReader::poll_wait_all(void *ctx) {
  io_uring *ring = (io_uring *) ctx;
  io_uring_cqe *cqe = nullptr;
  
  int ret = io_uring_wait_cqe(ring, &cqe);
  if(ret < 0) {
    LOG(ERROR) << "io_uring_wait_cqe failed: " << strerror(-ret);
    return;
  }
  
  //do {
  //  ret = io_uring_wait_cqe(ring, &cqe);
  //} while (ret == -EINTR);
  //if (ret < 0 || cqe->res < 0) {
  //  LOG(ERROR) << "Failed " << strerror(-cqe->res);
  //}
  unsigned completed = 0;
  unsigned head;
  io_uring_for_each_cqe(ring, head, cqe) {
    IORequest *req = (IORequest *) cqe->user_data;
    if (req != nullptr) {
      req->finished = true;
      completed++;
    }
  }
  
  if(completed > 0) {
    io_uring_cq_advance(ring, completed);
  }
}
#else
#include "linux_aligned_file_reader.h"

#include <libaio.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include "aligned_file_reader.h"
#include "utils/tsl/robin_map.h"
#include "utils.h"
#define MAX_EVENTS 256

namespace {
  typedef struct io_event io_event_t;
  typedef struct iocb iocb_t;

  void execute_io(void *ctx, int fd, std::vector<IORequest> &reqs, uint64_t n_retries = 0, bool write = false) {
    // break-up requests into chunks of size MAX_EVENTS each
    uint64_t n_iters = ROUND_UP(reqs.size(), MAX_EVENTS) / MAX_EVENTS;
    for (uint64_t iter = 0; iter < n_iters; iter++) {
      uint64_t n_ops = std::min((uint64_t) reqs.size() - (iter * MAX_EVENTS), (uint64_t) MAX_EVENTS);
      std::vector<iocb_t *> cbs(n_ops, nullptr);
      std::vector<io_event_t> evts(n_ops);
      std::vector<struct iocb> cb(n_ops);
      for (uint64_t j = 0; j < n_ops; j++) {
        if (write) {
          io_prep_pwrite(cb.data() + j, fd, reqs[j + iter * MAX_EVENTS].buf, reqs[j + iter * MAX_EVENTS].len,
                         reqs[j + iter * MAX_EVENTS].offset);
        } else {
          io_prep_pread(cb.data() + j, fd, reqs[j + iter * MAX_EVENTS].buf, reqs[j + iter * MAX_EVENTS].len,
                        reqs[j + iter * MAX_EVENTS].offset);
        }
      }

      // initialize `cbs` using `cb` array
      //

      for (uint64_t i = 0; i < n_ops; i++) {
        cbs[i] = cb.data() + i;
      }

      uint64_t n_tries = 0;
      while (n_tries <= n_retries) {
        // issue reads
        int64_t ret = io_submit((io_context_t) ctx, (int64_t) n_ops, cbs.data());
        // if requests didn't get accepted
        if (ret != (int64_t) n_ops) {
          LOG(ERROR) << "io_submit() failed; returned " << ret << ", expected=" << n_ops << ", ernno=" << errno << "="
                     << ::strerror((int) -ret) << ", try #" << n_tries + 1 << " ctx: " << ctx << "\n";
          exit(-1);
        } else {
          // wait on io_getevents
          ret = io_getevents((io_context_t) ctx, (int64_t) n_ops, (int64_t) n_ops, evts.data(), nullptr);
          // if requests didn't complete
          if (ret != (int64_t) n_ops) {
            LOG(ERROR) << "io_getevents() failed; returned " << ret << ", expected=" << n_ops << ", ernno=" << errno
                       << "=" << ::strerror((int) -ret) << ", try #" << n_tries + 1;
            exit(-1);
          } else {
            break;
          }
        }
      }
    }
  }
}  // namespace

LinuxAlignedFileReader::LinuxAlignedFileReader() {
  this->file_desc = -1;
}

LinuxAlignedFileReader::~LinuxAlignedFileReader() {
  int64_t ret;
  // check to make sure file_desc is closed
  ret = ::fcntl(this->file_desc, F_GETFD);
  if (ret == -1) {
    if (errno != EBADF) {
      std::cerr << "close() not called" << std::endl;
      // close file desc
      ret = ::close(this->file_desc);
      // error checks
      if (ret == -1) {
        std::cerr << "close() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno) << std::endl;
      }
    }
  }
}

namespace ioctx {
  static thread_local io_context_t ctx;
};

void *LinuxAlignedFileReader::get_ctx(int flag) {
  if (unlikely(ioctx::ctx == nullptr)) {
    register_thread(flag);
  }
  return (void *) ioctx::ctx;
}

void LinuxAlignedFileReader::register_thread(int flag) {
  if (ioctx::ctx == nullptr) {
    int ret = io_setup(MAX_EVENTS, &ioctx::ctx);
    if (ret != 0) {
      LOG(ERROR) << "io_setup() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno);
    }
  }
}

void LinuxAlignedFileReader::deregister_thread() {
  if (ioctx::ctx != nullptr) {
    io_destroy(ioctx::ctx);
    ioctx::ctx = nullptr;
  }
}

void LinuxAlignedFileReader::deregister_sync_thread() {
  // AIO uses one thread-local context for both synchronous and async calls.
  deregister_thread();
}

void LinuxAlignedFileReader::deregister_all_threads() {
}

void LinuxAlignedFileReader::register_buf(void *buf, uint64_t buf_size, int mrid) {
  return;
}

void LinuxAlignedFileReader::open(const std::string &fname, bool enable_writes = false, bool enable_create = false) {
  int flags = O_DIRECT | O_LARGEFILE | O_RDWR;
  if (enable_create) {
    flags |= O_CREAT;
  }
  this->file_desc = ::open(fname.c_str(), flags, 0644);
  // error checks
  assert(this->file_desc != -1);
  //  std::cerr << "Opened file : " << fname << std::endl;
}

void LinuxAlignedFileReader::close() {
  //  int64_t ret;

  // check to make sure file_desc is closed
  ::fcntl(this->file_desc, F_GETFD);
  //  assert(ret != -1);

  ::close(this->file_desc);
  //  assert(ret != -1);
}

void LinuxAlignedFileReader::read(std::vector<IORequest> &read_reqs, void *ctx, bool async) {
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, read_reqs);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::write(std::vector<IORequest> &write_reqs, void *ctx, bool async) {
  assert(this->file_desc != -1);
  execute_io(ctx, this->file_desc, write_reqs, 0, true);
  if (async == true) {
    std::cerr << "async only supported in Windows for now." << std::endl;
  }
}

void LinuxAlignedFileReader::read_fd(int fd, std::vector<IORequest> &read_reqs, void *ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, read_reqs);
}

void LinuxAlignedFileReader::write_fd(int fd, std::vector<IORequest> &write_reqs, void *ctx) {
  assert(this->file_desc != -1);
  execute_io(ctx, fd, write_reqs, 0, true);
}

void LinuxAlignedFileReader::send_io(std::vector<IORequest> &reqs, void *ctx, bool write) {
  uint64_t n_ops = std::min(reqs.size(), (uint64_t) MAX_EVENTS);
  std::vector<iocb_t *> cbs(n_ops, nullptr);
  std::vector<struct iocb> cb(n_ops);
  for (uint64_t j = 0; j < n_ops; j++) {
    reqs[j].finished = false;  // reset finished flag
    if (write) {
      io_prep_pwrite(cb.data() + j, this->file_desc, reqs[j].buf, reqs[j].len, reqs[j].offset);
    } else {
      io_prep_pread(cb.data() + j, this->file_desc, reqs[j].buf, reqs[j].len, reqs[j].offset);
    }
    cb[j].data = (void *) &reqs[j];  // set user data to point to the request
  }

  for (uint64_t i = 0; i < n_ops; i++) {
    cbs[i] = cb.data() + i;
  }

  // issue reads
  int64_t ret = io_submit((io_context_t) ctx, (int64_t) n_ops, cbs.data());
  // if requests didn't get accepted
  if (ret != (int64_t) n_ops) {
    LOG(ERROR) << "io_submit() failed; returned " << ret << ", expected=" << n_ops << ", " << strerror(errno);
  }
}

void LinuxAlignedFileReader::send_io(IORequest &req, void *ctx, bool write) {
  iocb_t cb;
  req.finished = false;  // reset finished flag
  if (write) {
    io_prep_pwrite(&cb, this->file_desc, req.buf, req.len, req.offset);
  } else {
    io_prep_pread(&cb, this->file_desc, req.buf, req.len, req.offset);
  }
  cb.data = (void *) &req;  // set user data to point to the request

  iocb_t *cbs[1] = {&cb};  // create an array of iocb_t pointers
  int ret = io_submit((io_context_t) ctx, 1, cbs);
  if (ret != 1) {
    LOG(ERROR) << "io_submit() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno);
  }
}

int LinuxAlignedFileReader::poll(void *ctx) {
  // Poll a single completed IO request in the io_uring context.
  io_event event;
  io_context_t io_ctx = (io_context_t) ctx;
  int ret = io_getevents(io_ctx, 0, 1, &event, nullptr);
  if (ret < 0) {
    return ret;  // not finished yet.
  }
  if (ret) {
    IORequest *req = (IORequest *) event.data;
    if (req != nullptr) {
      req->finished = true;
    }
  }
  return 0;
}

void LinuxAlignedFileReader::poll_all(void *ctx) {
  // Poll all completed IO requests in the io_uring context.
  static __thread io_event_t evts[MAX_EVENTS];
  io_context_t io_ctx = (io_context_t) ctx;
  int ret = io_getevents(io_ctx, 0, MAX_EVENTS, evts, nullptr);
  if (ret < 0) {
    LOG(ERROR) << "io_getevents() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno);
    return;  // not finished yet.
  }
  for (int i = 0; i < ret; i++) {
    IORequest *req = (IORequest *) evts[i].data;
    if (req != nullptr) {
      req->finished = true;
    }
  }
}

void LinuxAlignedFileReader::poll_wait(void *ctx) {
  io_event_t event;
  io_context_t io_ctx = (io_context_t) ctx;
  int ret = io_getevents(io_ctx, 1, 1, &event, nullptr);
  if (ret < 0) {
    LOG(ERROR) << "io_getevents() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno);
    return;  // not finished yet.
  }
  IORequest *req = (IORequest *) event.data;
  if (req != nullptr) {
    req->finished = true;
  }
}

void LinuxAlignedFileReader::poll_wait_all(void *ctx) {
  io_event_t events[MAX_EVENTS];
  const int count = io_getevents((io_context_t) ctx, 1, MAX_EVENTS, events, nullptr);
  if (count < 0) {
    throw std::runtime_error("AIO completion wait failed: " + std::to_string(count));
  }
  for (int i = 0; i < count; ++i) {
    auto *req = static_cast<IORequest *>(events[i].data);
    if (req != nullptr) req->finished = true;
  }
}

#endif

int LinuxAlignedFileReader::send_read_no_alloc(IORequest &req, void *ring) {
#ifndef READ_ONLY_TESTS
  if (!v2::cache.get(req.offset / SECTOR_LEN, (uint8_t *) req.buf)) {
    send_io(req, ring, false);
  } else {
    req.finished = true;  // mark as finished for cache miss
  }
#else
  send_io(req, ring, false);
#endif
  return 1;
}

int LinuxAlignedFileReader::send_read_no_alloc(std::vector<IORequest> &reqs, void *ring) {
#ifndef READ_ONLY_TESTS
  std::vector<IORequest> disk_read_reqs;  
  // fetch from cache.
  for (auto &req : reqs) {
    if (req.offset % SECTOR_LEN != 0 || req.len != SECTOR_LEN) {
      LOG(ERROR) << "Unaligned read offset: " << req.offset << ", len: " << req.len;
    }
    if (!v2::cache.get(req.offset / SECTOR_LEN, (uint8_t *) req.buf)) {
      disk_read_reqs.push_back(req);
    }
  }
  send_io(disk_read_reqs, ring, false);
  return disk_read_reqs.size();
#else
  send_io(reqs, ring, false);
  return reqs.size();
#endif
}

void LinuxAlignedFileReader::read_alloc(std::vector<IORequest> &read_reqs, void *ctx, std::vector<uint64_t> *page_ref) {
#ifndef READ_ONLY_TESTS
  std::vector<IORequest> disk_read_reqs;

  // TODO(gh): introduce size_per_io to cache.
  for (auto &req : read_reqs) {
    if (req.offset % SECTOR_LEN != 0) {
      LOG(ERROR) << "Unaligned read offset: " << req.offset << ", len: " << req.len;
      crash();
    }
    if (!v2::cache.get(req.offset / SECTOR_LEN, (uint8_t *) req.buf, true)) {
      disk_read_reqs.push_back(req);
    }
  }

  if (disk_read_reqs.size() > 0) {
    read(disk_read_reqs, ctx);
    //for (auto &req : disk_read_reqs) {
    //  v2::cache.put(req.offset / SECTOR_LEN, (uint8_t *) req.buf, true);
    //}
  }

  // ref.
  if (page_ref != nullptr) {
    for (auto &req : read_reqs) {
      page_ref->push_back(req.offset / SECTOR_LEN);
    }
  }
#else
  read(read_reqs, ctx);
#endif
}
