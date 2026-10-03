// src/glm/glm_main.cpp - strata-glm: GLM-5.3-Flash NVFP4 on one GPU, its routed experts in VRAM, RAM and on disk.
//
//   strata-glm --pack <dir from tools/glm_pack.py> --tokens <ids> [--max-new N] [--max-context C]
//              [--chunk T] [--profile P [--vram-experts N] [--ram-gib G] [--ram-reserve-gib R]]
//              [--dump-dir d] [--dump-logits f] [--routes f]
//
// Dense weights are read from the checkpoint's safetensors (the pack's dense.txt says where) into VRAM as BF16
// (the leading dense MLP layers decoded from NVFP4 to FP32). Each MoE layer's 8 routed experts are computed by
// Strata's NVFP4 expert kernel with FP32 activations (native_expert_grouped_f32). With --profile (tools/
// glm_profile.py) the hottest experts live in VRAM, the next ones in pinned RAM (TieredExpertSource) and the rest
// are read from the pack's experts.bin as they are routed; without it every expert is read from the file.
// --dump-dir writes, at the last prompt token, every layer's output streams (l%02d.f32) - what tools/glm_ref.py
// --dump-dir writes - and --dump-logits the next token's logits.
#include "glm_gemm.cuh"
#include "glm_kernels.cuh"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#if !defined(_WIN32)
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
// 408032gb: _fseeki64 is MSVC's; on Linux (64-bit off_t) fseeko is the same 64-bit seek
#define _fseeki64(f, off, whence) fseeko((f), (off_t) (off), (whence))
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <queue>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

// Unbuffered reads of experts.bin into 4 KiB-aligned destinations (the RAM tier's slots). A blob is split into
// kParts pieces so a single expert keeps several requests in flight; the blob lands at dst + offset % 4096. With
// identical copies on several drives (--mirror) each blob goes to the drive whose queue drains first at its speed.
class DiskReader {
public:
    static constexpr int kParts = 4, kJobs = 80, kDrives = 4;
    ~DiskReader() { close(); }
    /// paths: identical copies of experts.bin; gbps: each one's read rate (GB/s), for the balance
    bool open(const std::vector<std::string>& paths, const std::vector<double>& gbps, int threads) {
#if defined(_WIN32)
        nd_ = (int) std::min<size_t>(paths.size(), kDrives);
        for (int d = 0; d < nd_; ++d) {
            const int n = MultiByteToWideChar(CP_UTF8, 0, paths[(size_t) d].c_str(), -1, nullptr, 0);
            path_[d].assign((size_t) std::max(n, 1), L'\0');
            MultiByteToWideChar(CP_UTF8, 0, paths[(size_t) d].c_str(), -1, path_[d].data(), n);
            gbps_[d] = d < (int) gbps.size() ? gbps[(size_t) d] : 1.0;
            for (int t = 0; t < threads; ++t) th_.emplace_back([this, d] { loop(d); });
        }
        for (int i = 0; i < kJobs; ++i) { left_[i].store(0); bad_[i].store(0); }
        if (const char* e = std::getenv("GLM_LOW_INFLIGHT")) low_max_ = std::max(1, std::atoi(e));
        return nd_ > 0;
#else   // 408032gb: Linux - O_DIRECT pread per drive thread (the Windows path's unbuffered ReadFile)
        nd_ = (int) std::min<size_t>(paths.size(), kDrives);
        for (int d = 0; d < nd_; ++d) {
            upath_[d] = paths[(size_t) d];
            gbps_[d] = d < (int) gbps.size() ? gbps[(size_t) d] : 1.0;
            for (int t = 0; t < threads; ++t) th_.emplace_back([this, d] { loop(d); });
        }
        for (int i = 0; i < kJobs; ++i) { left_[i].store(0); bad_[i].store(0); }
        if (const char* e = std::getenv("GLM_LOW_INFLIGHT")) low_max_ = std::max(1, std::atoi(e));
        return nd_ > 0;
#endif
    }
    /// high: a read the current layer waits for (served before every low-priority prefetch piece)
    void start(int job, uint64_t off, uint64_t bytes, uint8_t* dst, bool high = true) {
        const uint64_t a0 = off / 4096 * 4096, len = (off - a0 + bytes + 4095) / 4096 * 4096;
        const int parts = kParts * std::max(1, nd_ - 0) + (nd_ > 1 ? 2 : 0);   // one drive: 4; two: 10 (6 + 4)
        const uint64_t piece = (len / (uint64_t) parts + 4095) / 4096 * 4096;
        int n = 0;
        {
            std::lock_guard<std::mutex> g(mu_);
            bad_[job].store(0);
            left_[job].store(1 << 30);   // held until every piece is queued
            for (uint64_t at = 0; at < len; at += piece, ++n) {
                const uint32_t pl = (uint32_t) std::min(piece, len - at);
                // each piece to the drive that gets to it first: a high one waits for the high queue and the pieces
                // in flight (low ones queued are passed), a low one for everything
                auto ahead = [&](int x) { return (double) (hq_[x] + fly_[x] + (high ? 0 : lqb_[x]) + pl) / gbps_[x]; };
                int d = 0;
                for (int x = 1; x < nd_; ++x)
                    if (ahead(x) < ahead(d)) d = x;
                (high ? q_[d] : lq_[d]).push_back({a0 + at, pl, dst + at, job});
                (high ? hq_[d] : lqb_[d]) += pl;
                ++reads_[d];
            }
            left_[job].store(n, std::memory_order_release);
        }
        for (int x = 0; x < nd_; ++x) cv_[x].notify_all();
    }
    bool wait(int job) {
        while (left_[job].load(std::memory_order_acquire) > 0) std::this_thread::yield();
        return bad_[job].load() == 0;
    }
    bool done(int job) const { return left_[job].load(std::memory_order_acquire) == 0; }
    /// a prefetch the current layer now needs: its queued pieces move to the high-priority queues
    void promote(int job) {
        bool moved = false;
        {
            std::lock_guard<std::mutex> g(mu_);
            for (int d = 0; d < nd_; ++d)
                for (auto it = lq_[d].begin(); it != lq_[d].end();) {
                    if (it->job != job) { ++it; continue; }
                    q_[d].push_back(*it);
                    lqb_[d] -= it->len;
                    hq_[d] += it->len;
                    it = lq_[d].erase(it);
                    moved = true;
                }
        }
        if (moved) for (int d = 0; d < nd_; ++d) cv_[d].notify_all();
    }
    long long reads(int d) const { return reads_[d]; }
    int drives() const { return nd_; }
    void close() {
        { std::lock_guard<std::mutex> g(mu_); quit_ = true; }
        for (auto& c : cv_) c.notify_all();
        for (auto& t : th_) t.join();
        th_.clear();
    }

private:
    struct Piece { uint64_t off; uint32_t len; uint8_t* dst; int job; };
    void loop(int d) {
#if defined(_WIN32)
        HANDLE h = CreateFileW(path_[d].c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
        LARGE_INTEGER fs{};
        if (h != INVALID_HANDLE_VALUE) GetFileSizeEx(h, &fs);
        for (;;) {
            Piece p{};
            bool low = false;
            {
                std::unique_lock<std::mutex> g(mu_);
                cv_[d].wait(g, [&] { return quit_ || !q_[d].empty() || (!lq_[d].empty() && low_fly_[d] < low_max_); });
                if (quit_) break;
                low = q_[d].empty();
                auto& q = low ? lq_[d] : q_[d];
                p = q.front();
                q.pop_front();
                (low ? lqb_[d] : hq_[d]) -= p.len;
                fly_[d] += p.len;
                if (low) ++low_fly_[d];
            }
            OVERLAPPED ov{};
            ov.Offset = (DWORD) p.off;
            ov.OffsetHigh = (DWORD) (p.off >> 32);
            DWORD got = 0;
            // the last blob's aligned window runs past the end of the file: only the bytes up to it must arrive
            const uint64_t need = std::min<uint64_t>(p.len, (uint64_t) fs.QuadPart > p.off ? (uint64_t) fs.QuadPart - p.off : 0);
            const bool ok = h != INVALID_HANDLE_VALUE && ReadFile(h, p.dst, p.len, &got, &ov) && got >= need && need > 0;
            if (!ok) {
                bad_[p.job].store(1);
                std::fprintf(stderr, "strata-glm: read of %u bytes at %llu (drive %d) failed: got %lu, error %lu\n", p.len,
                             (unsigned long long) p.off, d, (unsigned long) got, (unsigned long) GetLastError());
            }
            {
                std::lock_guard<std::mutex> g(mu_);
                fly_[d] -= p.len;
                if (low) --low_fly_[d];
            }
            if (low) cv_[d].notify_one();   // the next low piece may go
            left_[p.job].fetch_sub(1, std::memory_order_acq_rel);
        }
        if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
#else
        const int fdd = ::open(upath_[d].c_str(), O_RDONLY | O_DIRECT);   // unbuffered, like FILE_FLAG_NO_BUFFERING
        const int fdb = ::open(upath_[d].c_str(), O_RDONLY);              // fallback when a destination is not aligned
        struct stat st{};
        const uint64_t fsz = fdb >= 0 && ::fstat(fdb, &st) == 0 ? (uint64_t) st.st_size : 0;
        if (fdb < 0) std::fprintf(stderr, "strata-glm: cannot open %s: %s\n", upath_[d].c_str(), std::strerror(errno));
        for (;;) {
            Piece p{};
            bool low = false;
            {
                std::unique_lock<std::mutex> g(mu_);
                cv_[d].wait(g, [&] { return quit_ || !q_[d].empty() || (!lq_[d].empty() && low_fly_[d] < low_max_); });
                if (quit_) break;
                low = q_[d].empty();
                auto& q = low ? lq_[d] : q_[d];
                p = q.front();
                q.pop_front();
                (low ? lqb_[d] : hq_[d]) -= p.len;
                fly_[d] += p.len;
                if (low) ++low_fly_[d];
            }
            // the last blob's aligned window runs past the end of the file: only the bytes up to it must arrive
            const uint64_t need = std::min<uint64_t>(p.len, fsz > p.off ? fsz - p.off : 0);
            uint64_t got = 0;
            int err = 0;
            for (int fd = fdd >= 0 ? fdd : fdb; fd >= 0 && got < need;) {
                const ssize_t r = ::pread(fd, p.dst + got, (size_t) (p.len - got), (off_t) (p.off + got));
                if (r > 0) { got += (uint64_t) r; continue; }
                if (r == 0) break;                                           // end of file
                if (errno == EINTR) continue;
                if (errno == EINVAL && fd == fdd && fdb >= 0) { fd = fdb; continue; }   // O_DIRECT refused: buffered
                err = errno;
                break;
            }
            const bool ok = got >= need && need > 0;
            if (!ok) {
                bad_[p.job].store(1);
                std::fprintf(stderr, "strata-glm: read of %u bytes at %llu (drive %d) failed: got %llu, %s\n", p.len,
                             (unsigned long long) p.off, d, (unsigned long long) got, err ? std::strerror(err) : "short read");
            }
            {
                std::lock_guard<std::mutex> g(mu_);
                fly_[d] -= p.len;
                if (low) --low_fly_[d];
            }
            if (low) cv_[d].notify_one();
            left_[p.job].fetch_sub(1, std::memory_order_acq_rel);
        }
        if (fdd >= 0) ::close(fdd);
        if (fdb >= 0) ::close(fdb);
#endif
    }
    int nd_ = 0;
    std::string upath_[kDrives];   // 408032gb: the Linux paths
    std::wstring path_[kDrives];
    double gbps_[kDrives] = {};
    long long hq_[kDrives] = {}, lqb_[kDrives] = {}, fly_[kDrives] = {};   // bytes queued high / low, in flight (mu_)
    int low_fly_[kDrives] = {}, low_max_ = 2;                            // low pieces in flight a drive, and the cap
    long long reads_[kDrives] = {};
    std::vector<std::thread> th_;
    std::mutex mu_;
    std::condition_variable cv_[kDrives];
    std::deque<Piece> q_[kDrives], lq_[kDrives];
    std::atomic<int> left_[kJobs], bad_[kJobs];
    bool quit_ = false;
};

// The CPU's share of a decode layer's RAM experts: Strata's NVFP4 rows (AVX-512, activations as q8) straight from
// the pinned tier, while the GPU copies its own share over PCIe. Work units are blocks of rows, gate/up first,
// then (after the last gate/up block quantizes every expert's hidden) down; the workers spin between layers.
// start_multi() is the same for several tokens at once (--spec's verification pass): an expert's blob is read
// once for all the rows it was routed to, so a pass over T tokens costs the CPU what one token would.
class CpuExperts {
public:
    static constexpr int kMax = 8, kMaxMulti = 128, kMaxRows = 16, kMaxRowsTot = 128;
    static constexpr int kGuRows = 64, kDownRows = 128;
    ~CpuExperts() { stop(); }
    void init(int threads, const strata::kernels::cpu::NativeFmt& f) {
        f_ = f;
        act_.resize(f.act_bytes + 64);
        for (int e = 0; e < kMax; ++e) { ff_[e].assign((size_t) f.n_ff, 0.f); hq_[e].resize(f.h_bytes + 64); }
        act_m_.resize((size_t) kMaxRowsTot * (f.act_bytes + 64));
        hq_m_.resize((size_t) kMaxRowsTot * (f.h_bytes + 64));
        ff_m_.resize((size_t) kMaxRowsTot * f.n_ff);
        for (int t = 0; t < threads; ++t) th_.emplace_back([this] { loop(); });
    }
    bool ready() const { return !th_.empty(); }
    /// starts the n experts (blobs in host memory) on x (n_embd floats); out[e] holds each one's output after wait()
    void start(const float* x, const uint8_t* const* blobs, float* const* out, int n) {
        park();
        strata::kernels::cpu::native_quant_act(f_, x, act_.data());
        n_ = n;
        multi_ = false;
        for (int e = 0; e < n; ++e) { blob_[e] = blobs[e]; out_[e] = out[e]; }
        ua_ = (int) (f_.n_ff / kGuRows);
        ub_ = (int) (f_.n_embd / kDownRows);
        next_a_.store(0); done_a_.store(0); next_b_.store(0); done_b_.store(0);
        phase_b_.store(false); finished_.store(n == 0);
        gen_.fetch_add(1, std::memory_order_release);
    }
    /// the n experts over their own rows of x (n_embd floats a row): row r of expert e is x[rows[e][r]], and its
    /// result lands in out[e][r]. n <= kMaxMulti, all the rows together <= kMaxRowsTot.
    void start_multi(const float* x, const uint8_t* const* blobs, const int32_t* const* rows, const int32_t* nrows,
                     float* const* out, int n) {
        park();
        if (n > kMaxMulti) {
            std::fprintf(stderr, "cpu experts: %d experts, the limit is %d\n", n, kMaxMulti);
            std::exit(1);
        }
        n_ = n;
        multi_ = true;
        ntot_ = 0;
        for (int e = 0; e < n; ++e) {
            if (nrows[e] > kMaxRows || ntot_ + nrows[e] > kMaxRowsTot) {   // --spec is capped so this cannot happen
                std::fprintf(stderr, "cpu experts: %d rows for one expert, %d in all\n", nrows[e], ntot_);
                std::exit(1);
            }
            blob_[e] = blobs[e];
            nrow_[e] = nrows[e];
            for (int r = 0; r < nrows[e]; ++r) {
                const size_t k = (size_t) ntot_ + (size_t) r;
                strata::kernels::cpu::native_quant_act(f_, x + (size_t) rows[e][r] * f_.n_embd,
                                                       act_m_.data() + k * (f_.act_bytes + 64));
                av_[e][r] = act_m_.data() + k * (f_.act_bytes + 64);
                hqv_[e][r] = hq_m_.data() + k * (f_.h_bytes + 64);
                ffv_[e][r] = ff_m_.data() + k * (size_t) f_.n_ff;
                outv_[e][r] = out[(size_t) ntot_ + (size_t) r];
            }
            ntot_ += nrows[e];
        }
        ua_ = (int) (f_.n_ff / kGuRows);
        ub_ = (int) (f_.n_embd / kDownRows);
        next_a_.store(0); done_a_.store(0); next_b_.store(0); done_b_.store(0);
        phase_b_.store(false); finished_.store(n == 0);
        gen_.fetch_add(1, std::memory_order_release);
    }
    void wait() { while (!finished_.load(std::memory_order_acquire)) std::this_thread::yield(); }
    // A worker leaves the gate as soon as the generation changes, but it is still inside its claim loops when the
    // last unit lands: the producer has to wait for all of them back before it resets the counters and the shape,
    // or a straggler takes a unit of the next job with the previous job's n_ and pointers. --spec is what makes
    // that reachable - single-token jobs and multi-row ones now alternate.
    void park() {
        while (parked_.load(std::memory_order_acquire) < (int) th_.size() && !quit_.load(std::memory_order_relaxed))
            std::this_thread::yield();
    }
    void stop() {
        quit_.store(true);
        for (auto& t : th_) t.join();
        th_.clear();
    }

private:
    void loop() {
        namespace kc = strata::kernels::cpu;
        int seen = 0;
        for (;;) {
            parked_.fetch_add(1, std::memory_order_release);   // at the gate: a new shape may be published
            int g;
            while ((g = gen_.load(std::memory_order_acquire)) == seen) {
                if (quit_.load(std::memory_order_relaxed)) { parked_.fetch_sub(1, std::memory_order_release); return; }
                std::this_thread::yield();
            }
            parked_.fetch_sub(1, std::memory_order_release);   // left the gate: the shape must hold to the job's end
            seen = g;
            const bool m = multi_;
            const int na = n_ * ua_, nb = n_ * ub_;
            for (int u; (u = next_a_.fetch_add(1)) < na;) {
                const int e = u / ua_, r0 = (u % ua_) * kGuRows;
                if (m) {
                    kc::native_gu_rows(f_, blob_[e], av_[e], nrow_[e], ffv_[e], r0, r0 + kGuRows);
                } else {
                    const void* a = act_.data();
                    float* ff = ff_[e].data();
                    kc::native_gu_rows(f_, blob_[e], &a, 1, &ff, r0, r0 + kGuRows);
                }
                if (done_a_.fetch_add(1) + 1 == na) {   // the last gate/up block: every hidden is complete
                    if (m)
                        for (int k = 0; k < ntot_; ++k)
                            kc::native_quant_h(f_, ff_m_.data() + (size_t) k * f_.n_ff,
                                               hq_m_.data() + (size_t) k * (f_.h_bytes + 64));
                    else
                        for (int x = 0; x < n_; ++x) kc::native_quant_h(f_, ff_[x].data(), hq_[x].data());
                    phase_b_.store(true, std::memory_order_release);
                }
            }
            while (na > 0 && !phase_b_.load(std::memory_order_acquire)) std::this_thread::yield();
            for (int u; (u = next_b_.fetch_add(1)) < nb;) {
                const int e = u / ub_, r0 = (u % ub_) * kDownRows;
                if (m) {
                    kc::native_down_rows(f_, blob_[e], hqv_[e], nrow_[e], outv_[e], r0, r0 + kDownRows);
                } else {
                    const void* h = hq_[e].data();
                    float* o = out_[e];
                    kc::native_down_rows(f_, blob_[e], &h, 1, &o, r0, r0 + kDownRows);
                }
                if (done_b_.fetch_add(1) + 1 == nb) finished_.store(true, std::memory_order_release);
            }
        }
    }
    strata::kernels::cpu::NativeFmt f_;
    std::vector<std::thread> th_;
    std::vector<uint8_t> act_;
    std::vector<float> ff_[kMax];
    std::vector<uint8_t> hq_[kMax];
    const uint8_t* blob_[kMaxMulti] = {};
    float* out_[kMax] = {};
    // the multi-row job: one entry per (expert, row), the rows of an expert contiguous in the flat buffers
    std::vector<uint8_t> act_m_, hq_m_;
    std::vector<float> ff_m_;
    const void* av_[kMaxMulti][kMaxRows] = {};
    const void* hqv_[kMaxMulti][kMaxRows] = {};
    float* ffv_[kMaxMulti][kMaxRows] = {};
    float* outv_[kMaxMulti][kMaxRows] = {};
    int nrow_[kMaxMulti] = {};
    bool multi_ = false;
    int n_ = 0, ntot_ = 0, ua_ = 1, ub_ = 1;
    std::atomic<int> gen_{0}, next_a_{0}, done_a_{0}, next_b_{0}, done_b_{0}, parked_{0};
    std::atomic<bool> phase_b_{false}, finished_{true}, quit_{false};
};

// least recently used order over ids 0..n-1 (intrusive: most recent at head)
struct Lru {
    std::vector<int32_t> prev, next;
    std::vector<uint8_t> in;
    int32_t head = -1, tail = -1;
    void init(size_t n) { prev.assign(n, -1); next.assign(n, -1); in.assign(n, 0); head = tail = -1; }
    void remove(int32_t x) {
        if (!in[(size_t) x]) return;
        const int32_t p = prev[(size_t) x], q = next[(size_t) x];
        (p >= 0 ? next[(size_t) p] : head) = q;
        (q >= 0 ? prev[(size_t) q] : tail) = p;
        in[(size_t) x] = 0;
    }
    void touch(int32_t x) {
        remove(x);
        prev[(size_t) x] = -1;
        next[(size_t) x] = head;
        if (head >= 0) prev[(size_t) head] = x;
        head = x;
        if (tail < 0) tail = x;
        in[(size_t) x] = 1;
    }
};

// ---- GLM-5.3-Flash (config.json; checked against it at start)
constexpr int kLayers = 45, kDenseLead = 3, kEmbd = 4096, kHc = 4, kVocab = 154880;
constexpr int kKdaH = 64, kKdaD = 128, kKdaC = kKdaH * kKdaD;
constexpr int kMlaH = 64, kQLora = 1536, kR = 512, kDk = 256, kDv = 256;
constexpr int kIdxH = 32, kIdxD = 128, kKpool = 4, kIdxTopk = 2048;
constexpr int kNE = 288, kK = 8, kFF = 2048, kDenseFF = 12288;
constexpr float kEps = 1e-5f, kHcEps = 1e-6f, kLowerBound = -5.0f, kRouteScale = 2.5f, kSwigluLimit = 10.0f;
constexpr int kSinkhorn = 20;
bool is_dsa(int l) { return l % 4 == 3; }   // layers 3, 7, ..., 43 (config: full_attn_layers)

// --spec's drafter: a prompt lookup. The longest suffix of `seq` (2..8 tokens) that occurs earlier in it, and up
// to K tokens of what followed that occurrence - the continuation may overlap the suffix, so a repeat predicts
// itself. Empty when the sequence holds no repeat to predict from; then the pass is a plain decode step. The
// drafts never decide an output: the verification pass re-derives every position from the model's own logits.
std::vector<int> draft_ngram(const std::vector<int>& seq, int K) {
    const int n = (int) seq.size();
    for (int len = std::min(8, n - 1); len >= 2; --len) {
        const int* cur = seq.data() + (n - len);   // the suffix
        const int* hit = std::search(seq.data(), cur, cur, cur + len);
        if (hit == cur) continue;                  // no earlier occurrence of these `len` tokens
        std::vector<int> d;
        for (int i = 0; i < K && hit + len + i < seq.data() + n; ++i) d.push_back(hit[len + i]);
        return d;
    }
    return {};
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

struct TensorRef { std::string dtype, file; uint64_t offset = 0, bytes = 0; std::vector<int64_t> shape; };

struct Checkpoint {
    std::string dir;
    std::map<std::string, TensorRef> t;
    std::map<std::string, std::FILE*> files;
    bool open(const std::string& pack, std::string& err) {
        std::ifstream in(pack + "/dense.txt");
        if (!in) { err = "cannot open " + pack + "/dense.txt"; return false; }
        std::string line;
        while (std::getline(in, line)) {
            if (line.rfind("# model ", 0) == 0) { dir = line.substr(8); continue; }
            std::istringstream ss(line);
            std::string name;
            TensorRef r;
            int nd = 0;
            if (!(ss >> name >> r.dtype >> r.file >> r.offset >> r.bytes >> nd)) continue;
            r.shape.resize((size_t) nd);
            for (auto& d : r.shape) ss >> d;
            t[name] = r;
        }
        if (dir.empty() || t.empty()) { err = "dense.txt names no model or tensors"; return false; }
        return true;
    }
    const TensorRef& ref(const std::string& name) {
        auto it = t.find(name);
        if (it == t.end()) { std::fprintf(stderr, "strata-glm: no tensor %s\n", name.c_str()); std::exit(1); }
        return it->second;
    }
    std::vector<uint8_t> read(const std::string& name) {
        const TensorRef& r = ref(name);
        std::FILE*& f = files[r.file];
        if (!f) f = std::fopen((dir + "/" + r.file).c_str(), "rb");
        if (!f) { std::fprintf(stderr, "strata-glm: cannot open %s/%s\n", dir.c_str(), r.file.c_str()); std::exit(1); }
        std::vector<uint8_t> b(r.bytes);
        _fseeki64(f, (long long) r.offset, SEEK_SET);
        if (std::fread(b.data(), 1, b.size(), f) != b.size()) { std::fprintf(stderr, "strata-glm: short read of %s\n", name.c_str()); std::exit(1); }
        return b;
    }
    template <typename T> T* dev(const std::string& name, const char* want) {
        if (ref(name).dtype != want) { std::fprintf(stderr, "strata-glm: %s is %s, not %s\n", name.c_str(), ref(name).dtype.c_str(), want); std::exit(1); }
        const auto b = read(name);
        void* d = nullptr;
        CK(cudaMalloc(&d, b.size()));
        CK(cudaMemcpy(d, b.data(), b.size(), cudaMemcpyHostToDevice));
        return (T*) d;
    }
    glm::bf16* bf(const std::string& n) { return dev<glm::bf16>(n, "BF16"); }
    float* f32(const std::string& n) { return dev<float>(n, "F32"); }
    // an NVFP4 matrix (u8 codes, e4m3 scales, scale_2) decoded to FP32 on the device
    float* nvfp4_f32(const std::string& prefix, int rows, int cols) {
        uint8_t* w = dev<uint8_t>(prefix + ".weight", "U8");
        uint8_t* s = dev<uint8_t>(prefix + ".weight_scale", "F8_E4M3");
        const auto s2b = read(prefix + ".weight_scale_2");
        float s2 = 0.f;
        std::memcpy(&s2, s2b.data(), 4);
        float* out = nullptr;
        CK(cudaMalloc(&out, (size_t) rows * cols * sizeof(float)));
        glm::nvfp4_to_f32(w, s, s2, out, rows, cols, nullptr);
        CK(cudaDeviceSynchronize());
        cudaFree(w);
        cudaFree(s);
        return out;
    }
};

struct Layer {
    glm::bf16 *hc_attn_fn, *hc_attn_base, *hc_attn_scale, *hc_ffn_fn, *hc_ffn_base, *hc_ffn_scale;
    glm::bf16 *in_norm, *post_norm;
    // KDA
    glm::Mat wq, wk, wv, fa, fb, bproj, ga, gb, wo;
    glm::bf16* onorm;
    float *q_conv, *k_conv, *v_conv, *dt_bias, *A_log;
    // DSA
    glm::Mat qa, qb, kva, iwqb, iwk, iwp, igate;
    glm::bf16 *qa_norm, *kva_norm, *kvb, *ik_w, *ik_b, *iape;
    // MLP
    glm::Mat dg, du, dd;                                           // dense (layers 0-2): NVFP4 held exactly in BF16
    glm::bf16* router = nullptr;
    glm::Mat sg, su, sd;
    float* router_bias = nullptr;
    // state
    float *S = nullptr, *conv = nullptr;                           // KDA: [H][dh][dh], [3][C][3]
    glm::f16* lat = nullptr;                                       // DSA: the MLA latent of every position (FP16)
    int8_t* lat8 = nullptr;                                        // or int8 (--latent-i8), a scale per 64 values
    float* lat8s = nullptr;
    float *ikc = nullptr, *igc = nullptr, *pooled = nullptr;       // the indexer's key / gate window, its pools
};

struct Engine {
    Checkpoint ck;
    std::vector<Layer> L;
    // 408032gb (G4): this engine runs layers [lo, hi) - one stage of a layer pipeline; 0..kLayers is the whole model.
    // A stage loads only its layers, its tiers hold only its experts, it reads its input streams from `streams` /
    // `c_streams` when lo > 0 and leaves them there for the next stage when hi < kLayers (no head then).
    int lo = 0, hi = kLayers;
    bool own_moe(int m) const { const int l = m + kDenseLead; return l >= lo && l < hi; }
    glm::bf16* final_norm = nullptr;
    glm::Mat lm_head;
    // --dense-bf16: keep the checkpoint's BF16; default: FP8 E4M3 with a scale per row, quantized here once
    bool dense_fp8 = true;
    bool latent_i8 = false;   // --latent-i8: the MLA latent cache in int8 (half of FP16) - "k8" for an MLA, K and V one tensor
    // --dense-fp4 attn,shared,head: those matrices in NVFP4 (blocks of 16, a scale per row) instead of FP8 - half the
    // VRAM and the per-token read, lossy (the leading dense MLP is NVFP4 in the checkpoint: kept as it is)
    bool fp4_qk = false, fp4_vo = false, fp4_mla = false, fp4_shared = false, fp4_head = false;
    bool dense_i8 = false;   // --dense-i8: what would be FP8 is int8 with a scale per 32 values instead (Q8_0-like)
    size_t fp4_saved = 0;
    glm::Mat mat(const std::string& name, bool fp4 = false) {
        glm::Mat m;
        const TensorRef& r = ck.ref(name);
        glm::bf16* w = ck.bf(name);
        m.w = w;
        if (!dense_fp8 || r.shape.size() != 2 || r.shape[1] % 16) return m;
        const int rows = (int) r.shape[0], cols = (int) r.shape[1];
        if (fp4 && cols % 32 == 0 && (size_t) rows * cols >= ((size_t) 1 << 20)) {   // the small gates stay FP8
            uint8_t *q = nullptr, *bs = nullptr;
            float* sc = nullptr;
            CK(cudaMalloc(&q, (size_t) rows * cols / 2));
            CK(cudaMalloc(&bs, (size_t) rows * cols / 16));
            CK(cudaMalloc(&sc, (size_t) rows * sizeof(float)));
            glm::quant_nvfp4_rows(w, rows, cols, q, bs, sc, nullptr);
            CK(cudaDeviceSynchronize());
            CK(cudaFree(w));
            m.w = q;
            m.bsc = bs;
            m.scale = sc;
            fp4_saved += (size_t) rows * cols * 7 / 16;
            return m;
        }
        if (dense_i8 && cols % 32 == 0) {
            int8_t* q = nullptr;
            glm::f16* qs = nullptr;
            CK(cudaMalloc(&q, (size_t) rows * cols));
            CK(cudaMalloc(&qs, (size_t) rows * cols / 32 * sizeof(glm::f16)));
            glm::quant_i8_rows(w, rows, cols, q, qs, nullptr);
            CK(cudaDeviceSynchronize());
            CK(cudaFree(w));
            m.w = q;
            m.qs = qs;
            return m;
        }
        uint8_t* q = nullptr;
        float* sc = nullptr;
        CK(cudaMalloc(&q, (size_t) rows * cols));
        CK(cudaMalloc(&sc, (size_t) rows * sizeof(float)));
        glm::quant_fp8_rows(w, rows, cols, q, sc, nullptr);
        CK(cudaDeviceSynchronize());
        CK(cudaFree(w));
        m.w = q;
        m.scale = sc;
        m.fp8 = true;
        return m;
    }
    // an NVFP4 matrix of the checkpoint as it is (codes, E4M3 block scales), weight_scale_2 as the row scale
    glm::Mat nvfp4_mat(const std::string& prefix, int rows, int cols) {
        uint8_t* w = ck.dev<uint8_t>(prefix + ".weight", "U8");
        uint8_t* sc = ck.dev<uint8_t>(prefix + ".weight_scale", "F8_E4M3");
        const auto s2b = ck.read(prefix + ".weight_scale_2");
        float s2 = 0.f;
        std::memcpy(&s2, s2b.data(), 4);
        float* scale = nullptr;
        CK(cudaMalloc(&scale, (size_t) rows * sizeof(float)));
        glm::fill(scale, s2, rows, nullptr);
        CK(cudaDeviceSynchronize());
        glm::Mat m;
        m.w = w;
        m.bsc = sc;
        m.scale = scale;
        return m;
    }
    std::vector<glm::bf16> embed;                                   // host, BF16 [vocab][embd]
    int max_ctx = 8192;
    // the indexer's key / gate rows are needed only until their pool of kKpool completes: a window of ic_win rows
    // from position ic_base (a pool boundary), its incomplete tail moved to the front before each step
    int chunk_cap = 1, ic_win = 0;
    long long ic_base = 0;
    void ic_prepare(long long pos0, int T) {
        const long long P = pos0 - pos0 % kKpool;
        if (P != ic_base) {
            const long long src = P - ic_base;
            for (Layer& ly : L) {
                if (!ly.ikc) continue;
                for (long long r = 0; r < pos0 - P; ++r) {
                    CK(cudaMemcpyAsync(ly.ikc + (size_t) r * kIdxD, ly.ikc + (size_t) (src + r) * kIdxD, kIdxD * sizeof(float), cudaMemcpyDeviceToDevice, s));
                    CK(cudaMemcpyAsync(ly.igc + (size_t) r * kIdxD, ly.igc + (size_t) (src + r) * kIdxD, kIdxD * sizeof(float), cudaMemcpyDeviceToDevice, s));
                }
            }
            ic_base = P;
        }
        if (pos0 - ic_base + T > ic_win) { std::fprintf(stderr, "strata-glm: indexer window %d too small\n", ic_win); std::exit(1); }
    }
    // buffers
    static constexpr int kMixParts = 16;
    float *mixp = nullptr, *p_mixp = nullptr;                       // the hc mixes as kMixParts column slices
    float *streams, *mix, *x, *xn, *post, *comb, *y, *tmp1, *tmp2, *tmp3, *q, *k, *v, *gf, *b, *o, *gate;
    float *q_resid, *qm, *qa, *ctx, *vo, *iq, *ik, *ig, *iw, *score, *logits, *rows, *wts;
    int32_t *ids, *sel, *sel_cnt;
    static constexpr int kMlaSplit = 16;
    float* mla_part = nullptr;                                      // decode MLA: the key splits' partial results
    glm::bf16* emb_row;
    // experts
    std::FILE* experts = nullptr;
    strata::kernels::NativeExpertLayout XL;
    uint8_t* slots = nullptr;                                       // kK blobs on the device
    uint8_t* host_blobs = nullptr;                                  // pinned
    unsigned long long* grp_ptr;
    int32_t *grp_start, *n_groups, *ent_dst, *ent_tok;
    void* xscratch = nullptr;
    cudaStream_t s = nullptr;
    // ---- the prompt path (forward_chunk): T <= chunk tokens at once
    int chunk = 0;
    glm::Gemm gm;
    float *c_streams, *c_mix, *c_x, *c_xn, *c_y, *c_post, *c_comb, *c_t1, *c_t2, *c_t3, *c_q, *c_k, *c_v, *c_gf, *c_b, *c_o,
        *c_gate, *c_qr, *c_qm, *c_qa, *c_ctx, *c_vo, *c_iq, *c_iw, *c_score, *c_rows, *c_wts;
    int32_t *c_ids, *c_sel, *c_cnt, *c_grp_start, *c_ngroups, *c_ent_dst, *c_ent_tok;
    unsigned long long* c_grp_ptr;
    uint8_t* layer_slots = nullptr;                                 // a whole MoE layer's 288 blobs
    uint8_t* layer_host = nullptr;                                  // pinned staging, one layer
    void* c_xscratch = nullptr;
    glm::bf16* c_emb = nullptr;
    static constexpr int kSelLd = kIdxTopk + kKpool;
    // ---- the expert tiers (--profile); pairs are (MoE layer = l - kDenseLead, expert), as in experts.bin
    static constexpr int kMoe = kLayers - kDenseLead;
    strata::core::TieredExpertSource tier;
    bool tiered = false;
    std::vector<int32_t> res;                                       // per pair: its VRAM slot, or -1
    std::vector<uint8_t*> vslot;
    std::vector<int32_t> slot_pair;                                 // per VRAM slot: its pair
    static constexpr int kStage = 32;                               // staging slots: 2 x kBatch (prompt), kK (decode)
    // the prompt path's pipeline: disk experts read into a pinned ring, copied on pcs into one of two halves of
    // `stage` while the kernel of the other half runs on s
    static constexpr int kBatch = kStage / 2, kRing = 48, kRingJob0 = 16;
    uint8_t* pring = nullptr;
    size_t ring_stride = 0;
    cudaStream_t pcs = nullptr;
    cudaEvent_t ev_copy[2] = {}, ev_done[2] = {}, ev_ring[kRing] = {};
    unsigned long long* b_ptr[2] = {};                               // device, per half: kBatch blob pointers
    int32_t *b_start[2] = {}, *b_ng[2] = {}, *b_dst[2] = {}, *b_tok[2] = {};
    unsigned long long* h_ptr[2] = {};                               // pinned mirrors
    int32_t *h_start[2] = {}, *h_ng[2] = {}, *h_dst[2] = {}, *h_tok[2] = {};
    // MMQ (llama.cpp's int8 tensor-core products, Strata's moe_mmq): the layer's activations as q8_1 rows in the
    // order the experts are processed; each staging half's experts in one gate/up and one down launch
    bool prompt_mmq = true;                                          // --prompt-f32: the exact FP32 expert kernel
    std::unique_ptr<strata::prefill::mmq::Context> mmq_ctx;
    int32_t *m_src = nullptr, *m_dst = nullptr, *m_bounds = nullptr, *m_ident = nullptr, *m_rb[2] = {};
    int32_t *h_src = nullptr, *h_dstv = nullptr, *h_bounds = nullptr, *h_rb[2] = {};
    uint8_t *m_xq = nullptr, *m_hq = nullptr;
    float *m_gu = nullptr, *m_h = nullptr, *m_tails[2] = {};
    void init_prompt_mmq(int T) {
        namespace mq = strata::prefill::mmq;
        if (!mq::built() || !mq::supported(XL.gu_type)) { prompt_mmq = false; return; }
        const size_t E = (size_t) T * kK;
        CK(cudaMalloc(&m_src, E * sizeof(int32_t)));
        CK(cudaMalloc(&m_dst, E * sizeof(int32_t)));
        CK(cudaMalloc(&m_ident, E * sizeof(int32_t)));
        CK(cudaMalloc(&m_bounds, (kNE + 1) * sizeof(int32_t)));
        CK(cudaMallocHost(&h_src, E * sizeof(int32_t)));
        CK(cudaMallocHost(&h_dstv, E * sizeof(int32_t)));
        CK(cudaMallocHost(&h_bounds, (kNE + 1) * sizeof(int32_t)));
        for (int b = 0; b < 2; ++b) {
            CK(cudaMalloc(&m_rb[b], (kBatch + 1) * sizeof(int32_t)));
            CK(cudaMallocHost(&h_rb[b], (kBatch + 1) * sizeof(int32_t)));
            CK(cudaMalloc(&m_tails[b], kBatch * 4 * sizeof(float)));
        }
        CK(cudaMalloc(&m_xq, mq::q8_bytes((int64_t) E, kEmbd)));
        CK(cudaMalloc(&m_hq, mq::q8_bytes((int64_t) E, kFF)));
        CK(cudaMalloc(&m_gu, E * 2 * kFF * sizeof(float)));
        CK(cudaMalloc(&m_h, E * kFF * sizeof(float)));
        mq::iota(m_ident, (int64_t) E, s);
        mmq_ctx = std::make_unique<mq::Context>();
    }
    // The streamed prompt MoE (chunks of >= kStreamMin tokens: every expert of a layer is routed to): a pool with a
    // slot per expert id, filled from the start of each MoE layer - VRAM's experts device to device, RAM's over
    // PCIe, the disk's through the pinned ring - while the layer's attention computes; MMQ then runs over 16
    // consecutive ids at a time (an id nobody routes to has no rows). The pool waits for the previous layer's kernels.
    static constexpr int kStreamMin = 512;
    uint8_t* ppool = nullptr;
    size_t ps_stride = 0;
    cudaEvent_t ev_moe = nullptr;
    int ps_m = -1;
    std::vector<int> ps_disk;                                        // the staged layer's disk experts, by id
    size_t ps_next = 0;                                              // reads issued so far
    void init_prompt_pool() {   // before the expert tier sizes itself to the free VRAM
        ps_stride = (XL.bytes + 143) / 144 * 144;   // whole NVFP4 blocks between experts (MMQ steps in blocks)
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const size_t need = (size_t) kNE * ps_stride + (1 << 20);
        if (fr < need + (size_t) 4 * 1073741824ull) return;   // keep room for the tier and the reserve
        if (cudaMalloc(&ppool, need) != cudaSuccess) { (void) cudaGetLastError(); ppool = nullptr; return; }
        CK(cudaMemset(ppool, 0, need));
        CK(cudaEventCreateWithFlags(&ev_moe, cudaEventDisableTiming));
    }
    void ps_issue_reads(size_t upto) {
        const auto& lay = strata::kernels::cpu::expert_layout();
        for (; ps_next < ps_disk.size() && ps_next < upto; ++ps_next) {
            const int slot = (int) ((ring_pos + ps_next) % kRing);
            CK(cudaEventSynchronize(ev_ring[slot]));
            reader.start(kRingJob0 + slot, lay.blob_offset(ps_m, ps_disk[ps_next]), XL.bytes, pring + (size_t) slot * ring_stride);
        }
    }
    void prestage(int m) {   // before layer m's attention: every expert of the layer toward the pool
        if (!ppool || !tiered) return;
        CK(cudaStreamWaitEvent(pcs, ev_moe, 0));   // the previous layer's expert kernels have read the pool
        ps_m = m;
        ps_disk.clear();
        ps_next = 0;
        for (int e = 0; e < kNE; ++e) {
            const int32_t q = m * kNE + e;
            uint8_t* dst = ppool + (size_t) e * ps_stride;
            if (res[(size_t) q] >= 0) CK(cudaMemcpyAsync(dst, vslot[(size_t) res[(size_t) q]], XL.bytes, cudaMemcpyDeviceToDevice, pcs));
            else if (tier.has_copy(m, e)) CK(cudaMemcpyAsync(dst, tier.stable_blob(m, e), XL.bytes, cudaMemcpyHostToDevice, pcs));
            else ps_disk.push_back(e);
        }
        ps_issue_reads(kRing);
    }
    void moe_chunk_stream(int m, int T, const std::vector<std::vector<int32_t>>& by) {
        namespace mq = strata::prefill::mmq;
        const auto& lay = strata::kernels::cpu::expert_layout();
        const auto& f = lay.fmt[(size_t) m];
        const size_t S = ps_stride;
        CK(cudaEventSynchronize(ev_done[0]));   // the pinned arrays' previous uploads have run
        CK(cudaEventSynchronize(ev_done[1]));
        int32_t r = 0;
        for (int e = 0; e < kNE; ++e) {
            h_bounds[e] = r;
            for (const int32_t en : by[(size_t) e]) { h_src[r] = en / kK; h_dstv[r] = en; ++r; }
        }
        h_bounds[kNE] = r;
        CK(cudaMemcpyAsync(m_src, h_src, (size_t) r * sizeof(int32_t), cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(m_dst, h_dstv, (size_t) r * sizeof(int32_t), cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(m_bounds, h_bounds, (kNE + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, s));
        mq::quantize(c_xn, m_src, m_xq, XL.gu_type, kEmbd, kEmbd, r, s);
        size_t di = 0;   // the disk experts landed so far (by id order)
        for (int b0 = 0, bi = 0; b0 < kNE; b0 += kBatch, ++bi) {
            const int h = bi & 1, nb = std::min(kBatch, kNE - b0);
            // this batch's disk experts: wait for their reads, copy them into their slots
            while (di < ps_disk.size() && ps_disk[di] < b0 + nb) {
                const int e = ps_disk[di];
                ps_issue_reads(di + kRing);
                const int slot = (int) ((ring_pos + di) % kRing);
                const auto t = std::chrono::steady_clock::now();
                if (!reader.wait(kRingJob0 + slot)) { std::fprintf(stderr, "strata-glm: a prompt read failed\n"); std::exit(1); }
                ts.file_wait_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
                CK(cudaMemcpyAsync(ppool + (size_t) e * S, pring + (size_t) slot * ring_stride + lay.blob_offset(m, e) % 4096,
                                   XL.bytes, cudaMemcpyHostToDevice, pcs));
                CK(cudaEventRecord(ev_ring[slot], pcs));
                ++di;
            }
            const int32_t r0 = h_bounds[b0], nr = h_bounds[b0 + nb] - r0;
            if (nr == 0) continue;
            int32_t maxr = 0;
            for (int k = 0; k < nb; ++k) maxr = std::max<int32_t>(maxr, h_bounds[b0 + k + 1] - h_bounds[b0 + k]);
            CK(cudaEventSynchronize(ev_done[h]));
            for (int k = 0; k <= nb; ++k) h_rb[h][k] = h_bounds[b0 + k] - r0;
            CK(cudaEventRecord(ev_copy[h], pcs));
            CK(cudaMemcpyAsync(m_rb[h], h_rb[h], (size_t) (nb + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, s));
            CK(cudaStreamWaitEvent(s, ev_copy[h], 0));
            uint8_t* base = ppool + (size_t) b0 * S;
            glm::gather_tails(base, S, f.tail_off, nb, m_tails[h], s);
            mq::Product gu;
            gu.w = base; gu.type = XL.gu_type; gu.w_rows = 2 * kFF; gu.w_cols = kEmbd; gu.expert_bytes = S;
            gu.n = nb; gu.xq = m_xq; gu.bounds = m_bounds + b0; gu.ids = m_ident; gu.total_rows = r; gu.max_rows = maxr;
            gu.dst = m_gu; gu.ld_dst = 2 * kFF;
            mmq_ctx->run(gu, s);
            glm::swiglu_rows(m_gu, m_h, m_rb[h], nb, m_tails[h], r0, nr, kFF, kSwigluLimit, s);
            mq::quantize(m_h + (size_t) r0 * kFF, nullptr, m_hq, XL.d_type, kFF, kFF, nr, s);
            mq::Product dn;
            dn.w = base + f.down_off; dn.type = XL.d_type; dn.w_rows = kEmbd; dn.w_cols = kFF; dn.expert_bytes = S;
            dn.n = nb; dn.xq = m_hq; dn.bounds = m_rb[h]; dn.ids = m_dst + r0; dn.total_rows = nr; dn.max_rows = maxr;
            dn.dst = c_rows; dn.ld_dst = kEmbd;
            mmq_ctx->run(dn, s);
            glm::scale_entry_wts(c_wts, m_dst + r0, m_rb[h], nb, m_tails[h], nr, s);
            CK(cudaEventRecord(ev_done[h], s));
        }
        CK(cudaEventRecord(ev_moe, s));
        ring_pos += ps_disk.size();
        ts.ram += kNE - (long long) ps_disk.size();
        ts.file += (long long) ps_disk.size();
        ps_m = -1;
    }
    void init_prompt_pipe(int T) {   // idempotent: --spec sets the chunk path up again after release_prompt()
        const auto dev = [&](auto*& p, size_t n) { if (!p) CK(cudaMalloc((void**) &p, n)); };
        const auto pin = [&](auto*& p, size_t n) { if (!p) CK(cudaMallocHost((void**) &p, n)); };
        if (!pring) {
            ring_stride = ((size_t) XL.bytes + 8192 + 4095) / 4096 * 4096;
            CK(cudaHostAlloc((void**) &pring, (size_t) kRing * ring_stride, cudaHostAllocPortable));
        }
        if (!pcs) CK(cudaStreamCreateWithFlags(&pcs, cudaStreamNonBlocking));
        for (int b = 0; b < 2; ++b) {
            if (!ev_copy[b]) CK(cudaEventCreateWithFlags(&ev_copy[b], cudaEventDisableTiming));
            if (!ev_done[b]) CK(cudaEventCreateWithFlags(&ev_done[b], cudaEventDisableTiming));
            dev(b_ptr[b], kBatch * sizeof(unsigned long long));
            dev(b_start[b], (kBatch + 1) * sizeof(int32_t));
            dev(b_ng[b], sizeof(int32_t));
            dev(b_dst[b], (size_t) T * kK * sizeof(int32_t));
            dev(b_tok[b], (size_t) T * kK * sizeof(int32_t));
            pin(h_ptr[b], kBatch * sizeof(unsigned long long));
            pin(h_start[b], (kBatch + 1) * sizeof(int32_t));
            pin(h_ng[b], sizeof(int32_t));
            // these two are never freed, so they also have to hold a verification pass' entries
            const size_t ent = (size_t) std::max(T, logits_cap) * kK * sizeof(int32_t);
            pin(h_dst[b], ent);
            pin(h_tok[b], ent);
        }
        for (auto& ev : ev_ring) if (!ev) CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    }
    uint8_t* stage = nullptr;
    unsigned long long* gp_host = nullptr;                          // pinned, kK blob pointers per MoE layer
    struct TierStats { long long vram = 0, ram = 0, file = 0, cpu = 0; double file_wait_s = 0, pf_wait_s = 0; } ts;
    // what this conversation routes to: per pair, a count decayed by half every kHalfLife tokens (at each
    // rebalance); the tiers follow it, the startup profile only breaks ties
    static constexpr double kHalfLife = 1024;
    std::vector<float> heat;
    std::vector<int32_t> prior;
    long long heat_tokens = 0;
    // --policy lru (default): during decode VRAM caches RAM and RAM caches the disk, both least recently used.
    // A RAM hit is copied into VRAM anyway, so it lands in a VRAM slot (RAM keeps its copy: evicting from VRAM
    // is free); a disk hit is read anyway, so it lands in a RAM slot (of the least recent member outside VRAM).
    bool lru = true;
    // exclusive tiers (default with lru): a RAM hit moving to VRAM gives its RAM slot back (at the next layer, once
    // its copy has run), and an expert leaving VRAM without a RAM copy is written back (D2H: the idle direction)
    bool excl = false;   // --ram-exclusive
    const bool fake3 = std::getenv("GLM_FAKE3") != nullptr;   // tests: experts re-rounded to 3 bits (glm_moe.cu)
    // --vram-static F: the hottest F of VRAM's slots (by the prompt's routing, at each rebalance) are never evicted
    // and give their RAM copies up, so RAM caches that many more experts and the disk is read less; the rest of
    // VRAM stays an LRU over RAM. No write-backs while decoding (an expert leaving the static part at a rebalance is
    // written back to RAM then). An offline replay: 1100 of 1735 cut the disk reads a token 32-39% for 8-20% more
    // RAM -> VRAM copies (Strata-data/glm/policy/hybrid_sim.py).
    double vram_static = 0;
    std::vector<uint8_t> slot_static;
    long long static_n = 0;
    bool is_static(int32_t sl) const { return (size_t) sl < slot_static.size() && slot_static[(size_t) sl]; }
    // --skip-disk W (lossy, opt-in): a routed expert that is only on the disk and whose normalized routing weight is
    // below W is left out (the others' weights rescaled to the same sum) instead of waited for
    float skip_disk = 0.f;
    float skip_ram = 0.f;   // --skip-ram W (lossy, opt-in): the same for an expert in RAM (saves its PCIe copy)
    float* wts_pin = nullptr;
    long long skipped = 0;
    std::vector<int32_t> pending_free;                               // RAM copies of experts that moved to VRAM
    long long writebacks = 0;
    // a few free VRAM slots: a new expert takes one at once; the LRU victim that replaces it in the pool is written
    // back on `wb` (the D2H direction, beside the H2D copies), and the slot is reused once that copy has run
    static constexpr int kVPool = 16, kWbEv = 64;
    std::deque<std::pair<int32_t, int>> vfree;                       // (slot, writeback event or -1)
    cudaStream_t wb = nullptr;
    cudaEvent_t wb_ev[kWbEv] = {};
    int wb_next = 0;
    void evict_one(const int32_t* pj) {   // the least recent occupied slot outside pj into the pool
        int32_t sl = vlru.tail;
        while (sl >= 0) {
            const int32_t q = slot_pair[(size_t) sl];
            bool sel = false;
            for (int j = 0; pj && j < kK; ++j) sel |= pj[j] == q;
            if (!sel) break;
            sl = vlru.prev[(size_t) sl];
        }
        if (sl < 0) return;
        vlru.remove(sl);
        const int32_t v = slot_pair[(size_t) sl];
        int ev = -1;
        if (v >= 0) {
            res[(size_t) v] = -1;
            slot_pair[(size_t) sl] = -1;
            if (!tier.has_copy(v / kNE, v % kNE)) {
                if (tier.spares() == 0) {
                    const int32_t r = pj ? ram_victim(pj) : rlru.tail;
                    if (r >= 0) { tier.promote_done(r / kNE, r % kNE); rlru.remove(r); }
                }
                if (uint8_t* d = tier.demote_begin(v / kNE, v % kNE)) {
                    CK(cudaMemcpyAsync(d, vslot[(size_t) sl], XL.bytes, cudaMemcpyDeviceToHost, wb));
                    ev = wb_next;
                    wb_next = (wb_next + 1) % kWbEv;
                    CK(cudaEventRecord(wb_ev[ev], wb));
                    tier.demote_commit(v / kNE, v % kNE);
                    rlru.touch(v);
                    ++writebacks;
                }
            }
        }
        vfree.emplace_back(sl, ev);
    }
    void refill_pool() {
        if (!(lru && excl)) return;
        if (!wb) {
            CK(cudaStreamCreateWithFlags(&wb, cudaStreamNonBlocking));
            for (auto& ev : wb_ev) CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        }
        vfree.clear();
        while ((int) vfree.size() < kVPool) evict_one(nullptr);
        CK(cudaStreamSynchronize(wb));
    }
    // RAM tier eviction (default; --ram-lru: least recent): the RAM-only member with the lowest estimate of use - its
    // uses this decode (weighted 2^(t/512), so recent ones count more) plus its prompt frequency and the boot
    // profile's prior. Reuse barely depends on recency here (an offline replay: ARC, 2Q, LIRS ~ LRU), but ranking the
    // rarely used by their prior cut the disk misses 7-10% (tools: Strata-data/glm/policy).
    bool ram_freq = true;
    std::vector<double> pn, pbase;
    std::vector<uint32_t> plast, pver;
    struct PE { double k; uint32_t last, ver; int32_t p; };
    struct PGt { bool operator()(const PE& a, const PE& b) const { return a.k != b.k ? a.k > b.k : a.last > b.last; } };
    std::priority_queue<PE, std::vector<PE>, PGt> pheap;
    double pt0 = 0, dec_tok = 0;
    uint32_t pclk = 0;
    void pol_cand(int32_t p) { pheap.push({pn[(size_t) p] + pbase[(size_t) p], plast[(size_t) p], ++pver[(size_t) p], p}); }
    void pol_rebuild() {
        pheap = {};
        for (int32_t p = 0; p < (int32_t) pn.size(); ++p)
            if (res[(size_t) p] < 0 && tier.has_copy(p / kNE, p % kNE)) pol_cand(p);
    }
    void pol_reset(double prompt_tokens) {   // after the prompt's rebalance, from its routing counts (heat)
        const size_t np = (size_t) kMoe * kNE;
        pn.assign(np, 0.0);
        pbase.resize(np);
        plast.assign(np, 0);
        pver.assign(np, 0);
        pt0 = dec_tok;
        for (size_t q = 0; q < np; ++q) {
            const double r = std::min<double>(prior[q], (double) np);
            pbase[q] = 8.0 * heat[q] / std::max(1.0, prompt_tokens) + 64.0 * 2.0 * 8.0 / 288.0 * (1.0 - r / (double) np) + 0.01;
        }
        pol_rebuild();
    }
    void pol_use(int32_t p) {
        if (pn.empty()) return;
        pn[(size_t) p] += std::exp2((dec_tok - pt0) / 512.0);
        plast[(size_t) p] = ++pclk;
        if ((dec_tok - pt0) / 512.0 > 60) {   // rescale before the weights overflow
            const double f = std::exp2(-(dec_tok - pt0) / 512.0);
            for (size_t q = 0; q < pn.size(); ++q) { pn[q] *= f; pbase[q] *= f; }
            pt0 = dec_tok;
            pol_rebuild();
        }
    }
    int32_t pol_victim(const int32_t* pj) {
        PE held[kK];
        int nh = 0;
        int32_t v = -1;
        while (!pheap.empty()) {
            const PE e = pheap.top();
            pheap.pop();
            if (e.ver != pver[(size_t) e.p] || res[(size_t) e.p] >= 0 || !tier.has_copy(e.p / kNE, e.p % kNE)) continue;
            bool sel = false;
            for (int j = 0; j < kK; ++j) sel |= pj[j] == e.p;
            if (sel) { if (nh < kK) held[nh++] = e; continue; }
            ++pver[(size_t) e.p];
            v = e.p;
            break;
        }
        while (nh > 0) pheap.push(held[--nh]);
        if (pheap.size() > (size_t) 4 * kMoe * kNE) pol_rebuild();
        return v;
    }
    int32_t ram_pick(const int32_t* pj) {   // the RAM member a disk read replaces (outside VRAM and this selection)
        if (ram_freq && !pn.empty()) return pol_victim(pj);
        int32_t v = rlru.tail;
        while (v >= 0) {
            bool sel = false;
            for (int j = 0; j < kK; ++j) sel |= pj[j] == v;
            if (res[(size_t) v] < 0 && !sel) break;
            v = rlru.prev[(size_t) v];
        }
        return v;
    }
    int32_t ram_victim(const int32_t* pj) {   // the least recent RAM member outside VRAM and this layer's selection
        int32_t v = rlru.tail;
        while (v >= 0) {
            bool sel = false;
            for (int j = 0; j < kK; ++j) sel |= pj[j] == v;
            if (res[(size_t) v] < 0 && !sel) break;
            v = rlru.prev[(size_t) v];
        }
        return v;
    }
    Lru vlru, rlru;                                                  // over VRAM slots; over pairs (RAM members)
    DiskReader reader;
    std::vector<std::string> mirrors;                                // identical copies of experts.bin (--mirror)
    // --cpu-share F: the fraction of a decode layer's RAM hits the CPU computes (CpuExperts) instead of the GPU
    double cpu_share = 0, cpu_acc = 0;
    int cpu_threads = 14;
    CpuExperts cpux;
    float* x_host = nullptr;                                         // pinned: the layer's normed activation
    float* cpu_rows = nullptr;                                       // pinned: kK rows of kEmbd
    int32_t* hid_pin = nullptr;                                      // pinned: the router's ids
    unsigned long long* gq_host = nullptr;                           // pinned, per MoE layer: the GPU's blobs
    int32_t* ge_host = nullptr;                                      // pinned, per MoE layer: their rows, then the count
    int32_t* d_ent = nullptr;                                        // device: kK rows + the count
    cudaEvent_t ev_ids = nullptr;
    // decode's copies into VRAM slots run on xs, not behind the shared expert and the prediction on s
    cudaStream_t xs = nullptr;
    cudaEvent_t ev_xs = nullptr;
    // the experts already in VRAM run before the layer's copies land (their own launch); its blobs and rows are read
    // by the kernel straight from pinned memory (UVA), like the second launch's - no small H2D copies queued behind
    // the expert copies. ev_pfc: after the last speculative copies, which a resident expert may still be waiting for.
    unsigned long long* gqa_host = nullptr;
    int32_t* gea_host = nullptr;
    cudaEvent_t ev_pfc = nullptr;
    bool pfc_pending = false;
    // the speculative copies have their own stream: the layer's kernel waits for its own copies (ev_xs on xs) only;
    // a slot a speculative copy may still be writing is reused only after ev_pfc (spec_slot)
    cudaStream_t xp = nullptr;
    std::vector<uint8_t> spec_slot;
    // GLM_PREDICT=1 (measurement): layer l+1's experts predicted by its own hc_pre, norm and router applied to the
    // streams (A) as layer l's router sees them, (B) after layer l; scored against what layer l+1 then routes
    bool predict = false;
    bool prefetch = true;                                            // --no-prefetch: off
    int32_t* pf_ids = nullptr;                                       // pinned: layer l+1's predicted ids, kPred ranked
    static constexpr int kPred = 16;
    // what the prediction for each MoE layer was when its prefetch was decided, and where the disk reads a layer
    // still waited for (no prefetch) stood in it: rank 0..kPred-1, or kPred = not predicted
    int32_t pf_hist[kLayers][kPred];
    bool pf_hist_ok[kLayers] = {};
    long long miss_rank[kPred + 1] = {};
    cudaEvent_t ev_pred = nullptr;
    // disk reads started for the next layer: (pair, RAM slot) per reader job kK + i
    int pf_n = 0;
    int32_t pf_pair[kK] = {-1, -1, -1, -1, -1, -1, -1, -1};
    size_t ring_pos = 0;                                             // the prompt path's reads so far (ring order)
    uint8_t* pf_dst[kK];
    long long pf_reads = 0, pf_used = 0;
    // the less confident predictions (ranks pf_read_max..7) on the disk are read into a small pinned pool of their
    // own, not the RAM tier: a hit is copied from there, a miss costs the read and nothing else (no RAM member evicted)
    static constexpr int kSpool = 8, kSpoolJob0 = 64;
    bool pf_stage = false;                                           // --pf-stage: on (measured slower)
    uint8_t* spool = nullptr;
    size_t spool_stride = 0;
    int32_t sp_pair[kSpool] = {-1, -1, -1, -1, -1, -1, -1, -1};
    cudaEvent_t sp_ev[kSpool] = {};
    long long sp_reads = 0, sp_used = 0;
    void init_spool() {
        spool_stride = ((size_t) XL.bytes + 8192 + 4095) / 4096 * 4096;
        CK(cudaHostAlloc((void**) &spool, (size_t) kSpool * spool_stride, cudaHostAllocPortable));
        for (auto& ev : sp_ev) CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    }
    // RAM -> VRAM prefetch on its own stream: layer l+1's predicted RAM experts copied into VRAM slots behind layer
    // l's own copies, overlapping layer l's kernel and l+1's attention; layer l+1 waits for them before it copies
    cudaStream_t cs = nullptr;
    cudaEvent_t ev_cur = nullptr, ev_pf = nullptr;
    bool pf_pending = false;
    int32_t pf_vpair[kK];
    int pf_vn = 0;
    long long pf_copies = 0, pf_copies_used = 0;
    int pf_copy_max = 2;                                             // --pf-copies: the most confident N a layer
    int pf_read_max = kK;                                            // --pf-reads: disk reads for the top N predicted
    int32_t pf_vrank[kK];
    long long pf_rank_n[kK] = {}, pf_rank_used[kK] = {};              // prefetch copies by predicted rank
    void predict_enqueue(int l) {   // layer l's experts from the current streams, on the stream, into pf_ids
        using namespace glm;
        Layer& ly = L[(size_t) l];
        hc_mix(ly.hc_ffn_fn, streams, p_mixp, 24, kHc * kEmbd, kMixParts, s);
        hc_pre_finish(streams, p_mixp, ly.hc_ffn_base, ly.hc_ffn_scale, kEmbd, kEps, kHcEps, kSinkhorn, p_x, p_post, p_comb, s,
                      ly.post_norm, p_xn, kMixParts);
        gemv_bf16(ly.router, p_xn, p_log, kNE, kEmbd, 1, s);
        route_topk(p_log, ly.router_bias, kNE, kPred, kRouteScale, p_ids, p_wts, s);
        CK(cudaMemcpyAsync(pf_ids, p_ids, kPred * sizeof(int32_t), cudaMemcpyDeviceToHost, s));
        CK(cudaEventRecord(ev_pred, s));
    }
    // Prefetch reads into RAM (reader jobs kK..2kK-1) land in the background: a layer waits only for the ones it
    // routes to, and commits the others once they have arrived.
    void prefetch_land(int i, double* wait_s) {   // prefetch job i: wait for it, then it is a RAM member
        const int32_t q = pf_pair[i];
        const auto t = std::chrono::steady_clock::now();
        if (!reader.wait(kK + i)) { std::fprintf(stderr, "strata-glm: a prefetch read failed\n"); std::exit(1); }
        if (wait_s) *wait_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
        tier.demote_commit(q / kNE, q % kNE, (size_t) (strata::kernels::cpu::expert_layout().blob_offset(q / kNE, q % kNE) % 4096));
        rlru.touch(q);
        if (!pn.empty()) pol_cand(q);
        pf_pair[i] = -1;
        --pf_n;
    }
    void prefetch_commit_landed(const int32_t* sel) {   // the ones nobody waits for, once they have arrived
        for (int i = 0; i < kK; ++i) {
            const int32_t q = pf_pair[i];
            if (q < 0 || !reader.done(kK + i)) continue;
            bool need = false;
            for (int j = 0; j < kK; ++j) need |= sel[j] == q;
            if (!need) prefetch_land(i, nullptr);
        }
    }
    float *p_mix = nullptr, *p_x = nullptr, *p_xn = nullptr, *p_post = nullptr, *p_comb = nullptr, *p_log = nullptr, *p_wts = nullptr;
    int32_t* p_ids = nullptr;
    int32_t pred[2][kLayers][kK];
    bool pred_ok[2][kLayers] = {};
    long long pred_hit[2] = {}, pred_n[2] = {}, pred_miss_hit[2] = {}, pred_miss_n[2] = {};
    void predict_init() {
        predict = std::getenv("GLM_PREDICT") != nullptr;
        prefetch = prefetch && tiered && lru && !predict;
        if (!predict && !prefetch) return;
        CK(cudaMallocHost(&pf_ids, kPred * sizeof(int32_t)));
        CK(cudaEventCreateWithFlags(&ev_pred, cudaEventDisableTiming));
        CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&ev_cur, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&ev_pf, cudaEventDisableTiming));
        auto buf = [&](size_t n) { float* q = nullptr; CK(cudaMalloc(&q, n * sizeof(float))); return q; };
        p_mix = buf(32); p_x = buf(kEmbd); p_xn = buf(kEmbd); p_post = buf(4); p_comb = buf(16); p_log = buf(kNE); p_wts = buf(kPred);
        CK(cudaMalloc(&p_ids, kPred * sizeof(int32_t)));
    }
    void predict_layer(int v, int l) {   // layer l's experts from the current streams, into pred[v][l]
        using namespace glm;
        if (!predict || l >= hi || l < kDenseLead) return;   // l >= hi: the next stage's layer
        Layer& ly = L[(size_t) l];
        hc_mix(ly.hc_ffn_fn, streams, p_mixp, 24, kHc * kEmbd, kMixParts, s);
        hc_pre_finish(streams, p_mixp, ly.hc_ffn_base, ly.hc_ffn_scale, kEmbd, kEps, kHcEps, kSinkhorn, p_x, p_post, p_comb, s,
                      ly.post_norm, p_xn, kMixParts);
        gemv_bf16(ly.router, p_xn, p_log, kNE, kEmbd, 1, s);
        route_topk(p_log, ly.router_bias, kNE, kK, kRouteScale, p_ids, p_wts, s);
        CK(cudaMemcpyAsync(pred[v][l], p_ids, kK * sizeof(int32_t), cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        pred_ok[v][l] = true;
    }
    void predict_score(int l, const int32_t* hid) {
        if (!predict) return;
        const int m = l - kDenseLead;
        for (int v = 0; v < 2; ++v) {
            if (!pred_ok[v][l]) continue;
            for (int j = 0; j < kK; ++j) {
                bool hit = false;
                for (int i = 0; i < kK; ++i) hit |= pred[v][l][i] == hid[j];
                pred_hit[v] += hit;
                ++pred_n[v];
                if (tiered && res[(size_t) m * kNE + (size_t) hid[j]] < 0) { pred_miss_hit[v] += hit; ++pred_miss_n[v]; }
            }
            pred_ok[v][l] = false;
        }
    }
    void lru_rebuild(const std::vector<int32_t>& hot_first) {        // recency = the heat ranking
        vlru.init(vslot.size());
        rlru.init((size_t) kMoe * kNE);
        for (size_t s = 0; s < vslot.size(); ++s) if (slot_pair[s] < 0) vlru.touch((int32_t) s);   // empty: the tail
        for (size_t i = hot_first.size(); i-- > 0;) {                                              // the hottest last
            const int32_t p = hot_first[i];
            if (res[(size_t) p] >= 0 && !is_static(res[(size_t) p])) vlru.touch(res[(size_t) p]);
            if (tier.has_copy(p / kNE, p % kNE)) rlru.touch(p);
        }
    }
    // GLM_TIMING=1: per decode token, the MoE layers' expert copies (incl. waiting for the disk) and kernels
    bool timing = false;
    std::vector<cudaEvent_t> tev;                                   // per MoE layer: before copies, kernel, done
    struct Timing { double tok_s = 0, copy_ms = 0, kernel_ms = 0; long long tokens = 0; } tm;
    void timing_init() {
        timing = std::getenv("GLM_TIMING") != nullptr;
        if (!timing) return;
        tev.resize((size_t) kMoe * 3);
        for (auto& ev : tev) CK(cudaEventCreate(&ev));
    }
    void timing_token(double wall_s) {   // after forward() has synchronized
        if (!timing || !tiered) return;
        for (int m = 0; m < kMoe; ++m) {
            float a = 0, b = 0;
            CK(cudaEventElapsedTime(&a, tev[(size_t) m * 3], tev[(size_t) m * 3 + 1]));
            CK(cudaEventElapsedTime(&b, tev[(size_t) m * 3 + 1], tev[(size_t) m * 3 + 2]));
            tm.copy_ms += a;
            tm.kernel_ms += b;
        }
        tm.tok_s += wall_s;
        ++tm.tokens;
    }
    void note(int m, const int32_t* ids, size_t n) {
        float* h = heat.data() + (size_t) m * kNE;
        for (size_t i = 0; i < n; ++i) h[ids[i]] += 1.f;
    }

    void init_tier(const std::string& pack, const std::string& profile_path, long long vram_experts, double ram_gib,
                   double ram_reserve_gib, double vram_reserve_mib) {
        std::string err;
        std::vector<std::pair<int32_t, int32_t>> profile;
        int64_t pslots = 0;
        if (!strata::kernels::cpu::expert_layout_load(pack, kMoe, kNE, err) || !tier.open(pack, kMoe, kNE, err) ||
            !strata::core::read_expert_profile(profile_path, kMoe, kNE, profile, pslots, err)) {
            std::fprintf(stderr, "strata-glm: %s\n", err.c_str());
            std::exit(1);
        }
        // 408032gb: a stage ranks and holds only its own layers' experts
        profile.erase(std::remove_if(profile.begin(), profile.end(),
                                     [&](const std::pair<int32_t, int32_t>& q) { return !own_moe(q.first); }),
                      profile.end());
        CK(cudaStreamCreateWithFlags(&xs, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&xp, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&ev_xs, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&ev_pfc, cudaEventDisableTiming));
        CK(cudaMallocHost(&gqa_host, (size_t) kMoe * kK * sizeof(unsigned long long)));
        CK(cudaMallocHost(&gea_host, (size_t) kMoe * (kK + 1) * sizeof(int32_t)));
        CK(cudaMalloc(&stage, (size_t) kStage * XL.bytes + (1 << 20)));   // + MMQ's read past the last expert
        CK(cudaMemset(stage, 0, (size_t) kStage * XL.bytes + (1 << 20)));
        CK(cudaMallocHost(&x_host, kEmbd * sizeof(float)));
        CK(cudaMallocHost(&cpu_rows, (size_t) kK * kEmbd * sizeof(float)));
        CK(cudaMallocHost(&gq_host, (size_t) kMoe * kK * sizeof(unsigned long long)));
        CK(cudaMallocHost(&ge_host, (size_t) kMoe * (kK + 1) * sizeof(int32_t)));
        CK(cudaMalloc(&d_ent, (kK + 1) * sizeof(int32_t)));
        CK(cudaMallocHost(&gp_host, (size_t) kMoe * kK * sizeof(unsigned long long)));
        heat.assign((size_t) kMoe * kNE, 0.f);
        prior.assign((size_t) kMoe * kNE, INT32_MAX);
        for (size_t r = 0; r < profile.size(); ++r) prior[(size_t) profile[r].first * kNE + (size_t) profile[r].second] = (int32_t) r;
        // VRAM: the profile's head, as many slots as the free memory holds after the reserve
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const size_t keep = (size_t) ((vram_reserve_mib + (chunk > 0 ? 512 : 0)) * 1048576.0);
        long long nv = vram_experts >= 0 ? vram_experts : (long long) (fr > keep ? (fr - keep) / XL.bytes : 0);
        nv = std::min<long long>(nv, (long long) profile.size());
        constexpr int kBlock = 32;   // slots per allocation: WDDM places smaller blocks more readily
        for (long long i = 0; i < nv; i += kBlock) {
            const long long n = std::min<long long>(kBlock, nv - i);
            uint8_t* p = nullptr;
            if (cudaMalloc(&p, (size_t) n * XL.bytes) != cudaSuccess) { (void) cudaGetLastError(); break; }
            vblocks.push_back({vslot.size(), (size_t) n, p});
            for (long long j = 0; j < n; ++j) vslot.push_back(p + (size_t) j * XL.bytes);
        }
        nv = (long long) vslot.size();
        res.assign((size_t) kMoe * kNE, -1);
        // RAM: the next pairs by rank, as many as the budget holds; the rest stay in experts.bin
        const uint64_t avail = strata::core::available_ram_bytes();
        const uint64_t rkeep = (uint64_t) (ram_reserve_gib * 1073741824.0);
        uint64_t budget = avail > rkeep ? avail - rkeep : 0;
        if (ram_gib >= 0) budget = std::min<uint64_t>(budget, (uint64_t) (ram_gib * 1073741824.0));
        const std::vector<std::pair<int32_t, int32_t>> order(profile.begin() + nv, profile.end());
        if (!tier.load(order, budget, 0, 16, err)) { std::fprintf(stderr, "strata-glm: %s\n", err.c_str()); std::exit(1); }
        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<std::pair<int32_t, int32_t>> head(profile.begin(), profile.begin() + nv);
        slot_pair.assign(vslot.size(), -1);
        if (!tier.stream(head, [&](size_t k, const uint8_t* b) {
                if (cudaMemcpy(vslot[k], b, XL.bytes, cudaMemcpyHostToDevice) != cudaSuccess) return false;
                res[(size_t) head[k].first * kNE + (size_t) head[k].second] = (int32_t) k;
                slot_pair[k] = head[k].first * kNE + head[k].second;
                return true;
            }, 16, err)) {
            std::fprintf(stderr, "strata-glm: filling the VRAM tier: %s\n", err.c_str());
            std::exit(1);
        }
        tier.rerank(prior);
        tier.set_residency(res.data());
        tiered = true;
        {
            std::vector<std::string> paths{pack + "/experts.bin"};
            std::vector<double> gbps{10.0};
            for (const auto& mp : mirrors) { paths.push_back(mp); gbps.push_back(7.0); }
            // env (tests): GLM_GBPS=a,b - the drives' rates for the balance; GLM_DISK_THREADS - reads in flight a drive
            if (const char* g = std::getenv("GLM_GBPS"))
                for (size_t d = 0; d < gbps.size() && *g; ++d) {
                    gbps[d] = std::atof(g);
                    while (*g && *g != ',') ++g;
                    if (*g == ',') ++g;
                }
            const char* dt = std::getenv("GLM_DISK_THREADS");
            reader.open(paths, gbps, dt ? std::atoi(dt) : 8);
        }
        if (lru && pf_stage) init_spool();
        if (lru) {
            if (cpu_share > 0) cpux.init(cpu_threads, strata::kernels::cpu::expert_layout().fmt[0]);
            std::vector<int32_t> hot_first;
            for (const auto& pr : profile) hot_first.push_back(pr.first * kNE + pr.second);
            lru_rebuild(hot_first);
        }
        const double vs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const long long in_ram = tier.tier_members(), rest = (long long) profile.size() - nv - in_ram;
        std::fprintf(stderr, "strata-glm: experts: %lld in VRAM (%.1f GiB, filled in %.1f s), %lld in RAM (%s), %lld in "
                             "experts.bin\n", nv, nv * (double) XL.bytes / 1073741824.0, vs, in_ram, tier.note().c_str(),
                     rest);
    }

    // The tiers follow the conversation: the hottest pairs by `heat` into VRAM (up to max_vram moves; an expert
    // leaving VRAM that RAM should hold is copied back first), the next ones into RAM from the file (up to
    // max_ram reads, each evicting the coldest member). Then the heat decays with the tokens seen since the last one.
    void rebalance(long long max_vram, long long max_ram, bool verbose) {
        if (!tiered) return;
        const auto t0 = std::chrono::steady_clock::now();
        const size_t np = (size_t) kMoe * kNE;
        auto tier_of = [&](int32_t p) { return res[(size_t) p] >= 0 ? 0 : tier.has_copy(p / kNE, p % kNE) ? 1 : 2; };
        std::vector<int32_t> order(np);
        for (size_t i = 0; i < np; ++i) order[i] = (int32_t) i;
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {   // hotter; then where it is (no churn); prior
            const bool oa = own_moe(a / kNE), ob = own_moe(b / kNE);       // 408032gb: this stage's experts first
            if (oa != ob) return oa;
            if (heat[(size_t) a] != heat[(size_t) b]) return heat[(size_t) a] > heat[(size_t) b];
            const int ta = tier_of(a), tb = tier_of(b);
            if (ta != tb) return ta < tb;
            return prior[(size_t) a] < prior[(size_t) b];
        });
        std::vector<int32_t> rank(np);
        for (size_t i = 0; i < np; ++i) rank[(size_t) order[i]] = (int32_t) i;
        tier.rerank(rank);
        const size_t nv = vslot.size();
        const bool hybrid = lru && !excl && vram_static > 0;
        const size_t ns = hybrid ? (size_t) (vram_static * (double) nv) : 0;
        slot_static.resize(nv, 0);
        long long unstatic = 0;
        for (size_t sl = 0; sl < nv; ++sl) {   // static experts ranked below the new static part: back to inclusive
            if (!slot_static[sl]) continue;
            const int32_t o = slot_pair[sl];
            if (o >= 0 && (size_t) rank[(size_t) o] < ns) continue;
            slot_static[sl] = 0;
            --static_n;
            if (o < 0 || tier.has_copy(o / kNE, o % kNE)) continue;
            if (uint8_t* d = tier.demote_begin(o / kNE, o % kNE)) {
                CK(cudaMemcpy(d, vslot[sl], XL.bytes, cudaMemcpyDeviceToHost));
                tier.demote_commit(o / kNE, o % kNE);
                ++unstatic;
            } else {   // no RAM slot for it: out of VRAM too (file-backed)
                res[(size_t) o] = -1;
                slot_pair[sl] = -1;
            }
        }
        std::vector<int32_t> in, out;   // out: slots - the empty ones, then those of the coldest residents
        for (size_t i = 0; i < nv; ++i) if (res[(size_t) order[i]] < 0 && own_moe(order[i] / kNE)) in.push_back(order[i]);
        for (size_t sl = 0; sl < nv; ++sl) if (slot_pair[sl] < 0) out.push_back((int32_t) sl);
        for (size_t i = np; i-- > nv;) if (res[(size_t) order[i]] >= 0) out.push_back(res[(size_t) order[i]]);
        const size_t moves = std::min<size_t>({in.size(), out.size(), (size_t) std::max(0LL, max_vram)});
        std::vector<std::pair<int32_t, int32_t>> from_file;
        std::vector<int32_t> file_slot;
        long long demoted = 0, from_ram = 0;
        for (size_t j = 0; j < moves; ++j) {
            const int32_t s = out[j], o = slot_pair[(size_t) s], i = in[j];
            if (o >= 0) res[(size_t) o] = -1;
            if (o >= 0 && !tier.has_copy(o / kNE, o % kNE))
                if (uint8_t* d = tier.demote_begin(o / kNE, o % kNE)) {
                    CK(cudaMemcpy(d, vslot[(size_t) s], XL.bytes, cudaMemcpyDeviceToHost));
                    tier.demote_commit(o / kNE, o % kNE);
                    ++demoted;
                }
            if (tier.has_copy(i / kNE, i % kNE)) {
                CK(cudaMemcpy(vslot[(size_t) s], tier.stable_blob(i / kNE, i % kNE), XL.bytes, cudaMemcpyHostToDevice));
                if (!lru) tier.promote_done(i / kNE, i % kNE);   // exclusive: VRAM holds it now, its RAM slot is a spare
                res[(size_t) i] = s;
                slot_pair[(size_t) s] = i;
                ++from_ram;
            } else {
                from_file.emplace_back(i / kNE, i % kNE);
                file_slot.push_back(s);
            }
        }
        std::string err;
        if (!tier.stream(from_file, [&](size_t k, const uint8_t* b) {
                if (cudaMemcpy(vslot[(size_t) file_slot[k]], b, XL.bytes, cudaMemcpyHostToDevice) != cudaSuccess) return false;
                const int32_t p = from_file[k].first * kNE + from_file[k].second;
                res[(size_t) p] = file_slot[k];
                slot_pair[(size_t) file_slot[k]] = p;
                return true;
            }, 16, err)) {
            std::fprintf(stderr, "strata-glm: rebalance: %s\n", err.c_str());
            std::exit(1);
        }
        // RAM: the pairs ranked right after VRAM's (lru inclusive: from the top) that it does not hold, hottest first
        if (lru && excl)   // exclusive: VRAM's experts give their RAM copies back first
            for (size_t sl = 0; sl < nv; ++sl)
                if (slot_pair[sl] >= 0 && tier.has_copy(slot_pair[sl] / kNE, slot_pair[sl] % kNE))
                    tier.promote_done(slot_pair[sl] / kNE, slot_pair[sl] % kNE);
        pending_free.clear();
        for (size_t i = 0; i < ns; ++i) {   // the hottest residents become static: their RAM copies are spares now
            const int32_t p = order[i], sl = res[(size_t) p];
            if (sl < 0 || slot_static[(size_t) sl]) continue;
            slot_static[(size_t) sl] = 1;
            ++static_n;
            if (tier.has_copy(p / kNE, p % kNE)) tier.promote_done(p / kNE, p % kNE);
        }
        std::vector<std::pair<int32_t, int32_t>> want;
        const size_t cap = (size_t) tier.tier_slots(), r0 = lru && !excl ? 0 : nv;
        for (size_t i = r0, held = 0; i < np && held < cap && (long long) want.size() < max_ram; ++i) {
            const int32_t p = order[i];
            if (!own_moe(p / kNE)) break;   // own pairs come first in `order`: the rest belong to other stages
            if (res[(size_t) p] >= 0 && is_static(res[(size_t) p])) continue;
            ++held;
            if (((lru && !excl) || res[(size_t) p] < 0) && !tier.has_copy(p / kNE, p % kNE)) want.emplace_back(p / kNE, p % kNE);
        }
        const int64_t admitted = want.empty() ? 0 : tier.admit_from_file(want, 16, err);
        if (lru && ram_freq && !excl) pol_reset((double) heat_tokens);   // heat: the prompt's counts, before the decay
        const float f = (float) std::pow(0.5, (double) heat_tokens / kHalfLife);
        for (float& h : heat) h *= f;
        heat_tokens = 0;
        if (lru) { lru_rebuild(order); refill_pool(); }
        if (verbose) {
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "strata-glm: rebalance: VRAM %zu in (%lld from RAM, %zu from disk; %lld moved back to "
                                 "RAM), RAM %lld of %zu read in, %.2f s\n", moves, from_ram, from_file.size(), demoted,
                         (long long) admitted, want.size(), sec);
            if (hybrid) std::fprintf(stderr, "strata-glm: VRAM: %lld static (not in RAM), %zu LRU; %lld left the static part\n",
                                     static_n, nv - (size_t) static_n, unstatic);
        }
    }

    // --vram-static during decode: every few tokens the static part follows the RAM policy's estimate of use (decayed
    // decode uses + prompt frequency + prior). A pair whose estimate beats the coldest static one's by `hyst` swaps
    // in (from the VRAM LRU part: a change of role; from RAM: copied in), at most max_moves a call; the one swapped
    // out joins the LRU part with its RAM copy written back. Replay: chat_code's PCIe copies -7%, ~2 write-backs a
    // token (Strata-data/glm/policy/hybrid_resel.py).
    // The context caches (each DSA layer's latent and indexer pools) grow with the context instead of holding
    // max_ctx from the start (~1.9 GB at 262K): a short conversation keeps that VRAM for ~135 more expert slots. A
    // growth that does not fit takes the VRAM tier's last block of 32 slots (their experts stay in or go back to RAM).
    const int kCtxStep = std::getenv("GLM_CTX_STEP") ? std::atoi(std::getenv("GLM_CTX_STEP")) : 8192;   // env: tests
    int ctx_cap = 0;
    double vreserve_mib = 1500;
    struct VBlock { size_t start, n; uint8_t* base; };
    std::vector<VBlock> vblocks;
    void ctx_alloc(Layer& y, int cap, int keep) {   // cap positions, the first `keep` carried over
        auto grow = [&](auto*& ptr, size_t elem, size_t n_new, size_t n_keep) {
            void* q = nullptr;
            CK(cudaMalloc(&q, n_new * elem));
            if (ptr && n_keep) CK(cudaMemcpy(q, ptr, n_keep * elem, cudaMemcpyDeviceToDevice));
            if (ptr) CK(cudaFree(ptr));
            ptr = (std::remove_reference_t<decltype(ptr)>) q;
        };
        if (latent_i8) {
            grow(y.lat8, (size_t) kR, (size_t) cap, (size_t) keep);
            grow(y.lat8s, (size_t) (kR / 64) * sizeof(float), (size_t) cap, (size_t) keep);
        } else {
            grow(y.lat, (size_t) kR * sizeof(glm::f16), (size_t) cap, (size_t) keep);
        }
        grow(y.pooled, (size_t) kIdxD * sizeof(float), (size_t) (cap / kKpool + 1), keep ? (size_t) (keep / kKpool + 1) : 0);
    }
    size_t ctx_bytes(int cap) const {   // one DSA layer's context caches for cap positions
        const size_t per = latent_i8 ? (size_t) kR + (kR / 64) * sizeof(float) : (size_t) kR * sizeof(glm::f16);
        return (size_t) cap * per + (size_t) (cap / kKpool + 1) * kIdxD * sizeof(float);
    }
    bool retire_block() {   // the VRAM tier's last block back to the device
        if (vblocks.empty()) return false;
        const VBlock b = vblocks.back();
        for (size_t sl = b.start; sl < b.start + b.n; ++sl) {
            const int32_t p = slot_pair[sl];
            if (p >= 0) {
                if (!tier.has_copy(p / kNE, p % kNE))
                    if (uint8_t* d = tier.demote_begin(p / kNE, p % kNE)) {
                        CK(cudaMemcpy(d, vslot[sl], XL.bytes, cudaMemcpyDeviceToHost));
                        tier.demote_commit(p / kNE, p % kNE);
                        if (!rlru.in.empty()) rlru.touch(p);
                    }
                res[(size_t) p] = -1;
                if (!pn.empty() && tier.has_copy(p / kNE, p % kNE)) pol_cand(p);
            }
            if (sl < vlru.in.size()) vlru.remove((int32_t) sl);
            if (sl < slot_static.size() && slot_static[sl]) { slot_static[sl] = 0; --static_n; }
        }
        CK(cudaFree(b.base));
        vslot.resize(b.start);
        slot_pair.resize(b.start);
        if (slot_static.size() > b.start) slot_static.resize(b.start);
        vblocks.pop_back();
        ++retired_blocks;
        return true;
    }
    long long retired_blocks = 0;
    void ensure_ctx(int need) {   // at a token / chunk boundary
        if (need <= ctx_cap || ctx_cap >= max_ctx) return;
        int cap = std::max(need, ctx_cap + std::max(kCtxStep, ctx_cap / 4));
        cap = std::min(max_ctx, (cap + 4095) / 4096 * 4096);
        CK(cudaDeviceSynchronize());
        int nd = 0;
        for (const Layer& y : L) nd += y.pooled != nullptr;
        const size_t add = (size_t) nd * (ctx_bytes(cap) - ctx_bytes(ctx_cap)) + ctx_bytes(cap);   // + one layer in transit
        const size_t keep = (size_t) (vreserve_mib * 1048576.0);
        for (;;) {
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            if (fr >= add + keep || !retire_block()) break;
        }
        for (Layer& y : L)
            if (y.pooled) ctx_alloc(y, cap, ctx_cap);
        ctx_cap = cap;
    }
    long long restatic_moves = 0;
    void restatic(int max_moves, double hyst) {
        if (!(tiered && lru && !excl && vram_static > 0) || pn.empty() || static_n == 0) return;
        const size_t np = (size_t) kMoe * kNE, nv = vslot.size();
        const size_t ns = std::min((size_t) static_n, nv);
        auto sc = [&](int32_t p) { return pn[(size_t) p] + pbase[(size_t) p]; };
        std::vector<int32_t> ord(np);
        for (size_t i = 0; i < np; ++i) ord[i] = (int32_t) i;
        std::nth_element(ord.begin(), ord.begin() + (long long) ns, ord.end(), [&](int32_t a, int32_t b) { return sc(a) > sc(b); });
        std::vector<int32_t> in, out;
        for (size_t i = 0; i < ns; ++i) {
            const int32_t p = ord[i], r = res[(size_t) p];
            if (r >= 0 ? !is_static(r) : tier.has_copy(p / kNE, p % kNE)) in.push_back(p);
        }
        std::vector<uint8_t> top(np, 0);
        for (size_t i = 0; i < ns; ++i) top[(size_t) ord[i]] = 1;
        for (size_t sl = 0; sl < nv; ++sl)
            if (slot_static[sl] && slot_pair[sl] >= 0 && !top[(size_t) slot_pair[sl]]) out.push_back(slot_pair[sl]);
        std::sort(in.begin(), in.end(), [&](int32_t a, int32_t b) { return sc(a) > sc(b); });
        std::sort(out.begin(), out.end(), [&](int32_t a, int32_t b) { return sc(a) < sc(b); });
        const int32_t none[kK] = {-1, -1, -1, -1, -1, -1, -1, -1};
        for (size_t k = 0; k < std::min({in.size(), out.size(), (size_t) max_moves}); ++k) {
            const int32_t pi = in[k], po = out[k];
            if (sc(pi) <= hyst * sc(po)) break;
            const int32_t so = res[(size_t) po];
            // out: into the LRU part, with its RAM copy back (or out of VRAM if RAM has no room for it)
            slot_static[(size_t) so] = 0;
            --static_n;
            if (!tier.has_copy(po / kNE, po % kNE)) {
                if (tier.spares() == 0) {
                    const int32_t v = ram_pick(none);
                    if (v >= 0) { tier.promote_done(v / kNE, v % kNE); rlru.remove(v); }
                }
                if (uint8_t* d = tier.demote_begin(po / kNE, po % kNE)) {
                    CK(cudaMemcpy(d, vslot[(size_t) so], XL.bytes, cudaMemcpyDeviceToHost));
                    tier.demote_commit(po / kNE, po % kNE);
                    rlru.touch(po);
                } else {
                    res[(size_t) po] = -1;
                    slot_pair[(size_t) so] = -1;
                }
            }
            vlru.touch(so);
            // in: a resident of the LRU part changes role; one in RAM is copied into the LRU part's least recent slot
            int32_t r = res[(size_t) pi];
            if (r < 0) {
                r = vlru.tail;
                if (r < 0) break;
                const int32_t vv = slot_pair[(size_t) r];
                if (vv >= 0) {
                    res[(size_t) vv] = -1;
                    if (tier.has_copy(vv / kNE, vv % kNE)) pol_cand(vv);
                }
                CK(cudaMemcpy(vslot[(size_t) r], tier.stable_blob(pi / kNE, pi % kNE), XL.bytes, cudaMemcpyHostToDevice));
                slot_pair[(size_t) r] = pi;
                res[(size_t) pi] = r;
            }
            vlru.remove(r);
            slot_static[(size_t) r] = 1;
            ++static_n;
            if (tier.has_copy(pi / kNE, pi % kNE)) { tier.promote_done(pi / kNE, pi % kNE); rlru.remove(pi); }
            ++restatic_moves;
        }
    }

    // After the prompt: its buffers (and the staging slots LRU decode does not use) back to the expert tier.
    void release_prompt() {
        if (chunk == 0) return;
        CK(cudaStreamSynchronize(s));
        for (float* q : {c_streams, c_mix, c_x, c_xn, c_y, c_post, c_comb, c_t1, c_t2, c_t3, c_q, c_k, c_v, c_gf, c_b, c_o,
                         c_gate, c_qr, c_qm, c_qa, c_ctx, c_vo, c_iq, c_iw, c_score, c_rows, c_wts, m_gu, m_h})
            if (q) cudaFree(q);
        for (int32_t* q : {c_ids, c_sel, c_cnt, c_grp_start, c_ngroups, c_ent_dst, c_ent_tok, m_src, m_dst, m_bounds, m_ident,
                           b_start[0], b_start[1], b_ng[0], b_ng[1], b_dst[0], b_dst[1], b_tok[0], b_tok[1], m_rb[0], m_rb[1]})
            if (q) cudaFree(q);
        for (void* q : {(void*) c_grp_ptr, (void*) c_emb, c_xscratch, (void*) layer_slots, (void*) m_xq, (void*) m_hq,
                        (void*) b_ptr[0], (void*) b_ptr[1], (void*) m_tails[0], (void*) m_tails[1]})
            if (q) cudaFree(q);
        gm.release();
        glm::conv_silu_release();
        for (int b = 0; b < 2; ++b)
            b_ptr[b] = nullptr, b_start[b] = nullptr, b_ng[b] = nullptr, b_dst[b] = nullptr, b_tok[b] = nullptr;
        if (ppool) { cudaFree(ppool); ppool = nullptr; }
        if (lru && stage) { cudaFree(stage); stage = nullptr; }
        chunk = 0;
    }
    // new VRAM slots from what is free now (empty: rebalance and the LRU fill them first)
    void grow_vram_tier(double vram_reserve_mib) {
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const size_t keep = (size_t) (vram_reserve_mib * 1048576.0);
        long long added = 0;
        constexpr int kBlock = 32;
        while (fr > keep + (size_t) kBlock * XL.bytes) {
            uint8_t* q = nullptr;
            if (cudaMalloc(&q, (size_t) kBlock * XL.bytes) != cudaSuccess) { (void) cudaGetLastError(); break; }
            vblocks.push_back({vslot.size(), (size_t) kBlock, q});
            for (int j = 0; j < kBlock; ++j) { vslot.push_back(q + (size_t) j * XL.bytes); slot_pair.push_back(-1); }
            added += kBlock;
            fr -= (size_t) kBlock * XL.bytes;
        }
        if (added > 0) std::fprintf(stderr, "strata-glm: the prompt's buffers freed: %lld more VRAM slots (%zu)\n", added, vslot.size());
    }
    int qsub = 0;                                                   // the prompt path's DSA query sub-chunk
    // ---- --spec K: greedy speculative decoding ---------------------------------------------------------
    // A pass verifies [the last accepted token, the K drafts] with the prompt path at the current position and
    // runs the head for every row, so the pass's own argmax decides each position: the output is greedy by
    // construction, and its numbers are the prompt path's (FP32 experts, the CPU pool) rather than the decode
    // path's - self-consistent, which is what speculative decoding needs. What the pass leaves behind is the
    // KDA layers' recurrent state, advanced by K+1 tokens when only a prefix was accepted: spec_fix() rolls it
    // back from the snapshots kda_chunk() took. The MLA, indexer and pool caches are positional - their garbage
    // rows sit at or beyond the next pass's first position and are overwritten before anything reads them.
    bool spec_verify = false;                                       // true while a pass verifies
    int spec_T = 0;                                                 // its rows
    int logits_cap = 1;                                             // the head's rows the logits buffer holds
    std::vector<int> sp_slot;                                       // layer -> its KDA snapshot, -1 for MLA layers
    float *sp_S = nullptr, *sp_C = nullptr, *sp_raw = nullptr, *sp_in = nullptr, *sp_b = nullptr;
    size_t sp_n = 0;                                                // floats per layer in a raw / post-prep snap
    float *x_host_m = nullptr, *cpu_rows_m = nullptr;               // pinned: the CPU pool's input and output
    double cpu_acc_m = 0;                                           // the CPU's share, carried over the layers
    void init_spec(int T) {   // after the prompt: the chunk path again, small, for the verification passes
        release_prompt();   // a tiered run has done that already; a plain one still holds the prompt's buffers
        init_chunk(T, false);
        if (tiered) init_prompt_pipe(T);   // the ring is the disk tier's; without tiers the passes need nothing
        if (lru && !stage) {
            CK(cudaMalloc(&stage, (size_t) kStage * XL.bytes + (1 << 20)));
            CK(cudaMemset(stage, 0, (size_t) kStage * XL.bytes + (1 << 20)));
        }
        prompt_mmq = false;   // the FP32 grouped kernel and the CPU pool, as in decode
        sp_slot.assign(kLayers, -1);
        int nk = 0;
        for (int l = lo; l < hi; ++l) if (!is_dsa(l)) sp_slot[(size_t) l] = nk++;
        sp_n = (size_t) T * kKdaC;
        CK(cudaMalloc(&sp_S, (size_t) nk * kKdaH * kKdaD * sizeof(float)));
        CK(cudaMalloc(&sp_C, (size_t) nk * 3 * kKdaC * 3 * sizeof(float)));
        CK(cudaMalloc(&sp_raw, (size_t) nk * 3 * sp_n * sizeof(float)));
        CK(cudaMalloc(&sp_in, (size_t) nk * 3 * sp_n * sizeof(float)));
        CK(cudaMalloc(&sp_b, (size_t) nk * T * kKdaH * sizeof(float)));
        CK(cudaMallocHost(&x_host_m, (size_t) T * kEmbd * sizeof(float)));
        CK(cudaMallocHost(&cpu_rows_m, (size_t) T * kK * kEmbd * sizeof(float)));
        std::fprintf(stderr, "strata-glm: --spec: verifying %d tokens a pass, %d KDA states snapshotted\n", T, nk);
    }
    /// the KDA states back to the first `keep` tokens of the pass that just ran (keep >= 1)
    void spec_fix(int keep) {
        if (!sp_S) return;
        using namespace glm;
        for (int l = lo; l < hi; ++l) {
            const int sl = sp_slot[(size_t) l];
            if (sl < 0) continue;
            Layer& ly = L[(size_t) l];
            CK(cudaMemcpyAsync(ly.S, sp_S + (size_t) sl * kKdaH * kKdaD, (size_t) kKdaH * kKdaD * sizeof(float),
                               cudaMemcpyDeviceToDevice, s));
            CK(cudaMemcpyAsync(ly.conv, sp_C + (size_t) sl * 3 * kKdaC * 3, (size_t) 3 * kKdaC * 3 * sizeof(float),
                               cudaMemcpyDeviceToDevice, s));   // the state as it was before the pass
            const float* raw = sp_raw + (size_t) sl * 3 * sp_n;
            const float* in = sp_in + (size_t) sl * 3 * sp_n;
            conv_silu_seq(raw, ly.q_conv, ly.conv, c_q, kKdaC, keep, s);
            conv_silu_seq(raw + sp_n, ly.k_conv, ly.conv + (size_t) kKdaC * 3, c_k, kKdaC, keep, s);
            conv_silu_seq(raw + 2 * sp_n, ly.v_conv, ly.conv + (size_t) 2 * kKdaC * 3, c_v, kKdaC, keep, s);
            kda_scan(ly.S, in, in + sp_n, c_v, in + 2 * sp_n, sp_b + (size_t) sl * spec_T * kKdaH, c_o, kKdaH, kKdaD,
                     keep, s);   // v: the row above convolved it into c_v; q, k and the gate are the saved post-prep
            // ones (prep rewrites them in place), so the scan sees exactly what the pass fed it
        }
        CK(cudaStreamSynchronize(s));
    }
    void init_chunk(int T, bool whole_layer) {   // whole_layer: no tiers, each layer's 288 experts read at once
        chunk = T;
        qsub = std::min(T, std::getenv("GLM_QSUB") ? std::atoi(std::getenv("GLM_QSUB")) : 2048);   // env: tests
        // the widest split: the hc mixes' 4 streams, the dense MLP's rows, or a query sub-chunk's MLA context (64 x 512)
        gm.init(s, std::max({(size_t) qsub * kMlaH * kR, (size_t) T * kDenseFF, (size_t) T * kHc * kEmbd}));
        auto buf = [&](size_t n) { float* p = nullptr; CK(cudaMalloc(&p, n * sizeof(float))); return p; };
        c_streams = buf((size_t) T * kHc * kEmbd); c_mix = buf((size_t) T * 24); c_x = buf((size_t) T * kEmbd);
        c_xn = buf((size_t) T * kEmbd); c_y = buf((size_t) T * kEmbd); c_post = buf((size_t) T * 4); c_comb = buf((size_t) T * 16);
        c_t1 = buf((size_t) T * kDenseFF); c_t2 = buf((size_t) T * kDenseFF); c_t3 = buf((size_t) T * kDenseFF);
        c_q = buf((size_t) T * kKdaC); c_k = buf((size_t) T * kKdaC); c_v = buf((size_t) T * kKdaC); c_gf = buf((size_t) T * kKdaC);
        c_b = buf((size_t) T * kKdaH); c_o = buf((size_t) T * kKdaC); c_gate = buf((size_t) T * kKdaC);
        c_qr = buf((size_t) T * kQLora); c_qm = buf((size_t) qsub * kMlaH * kDk); c_qa = buf((size_t) qsub * kMlaH * kR);
        c_ctx = buf((size_t) qsub * kMlaH * kR); c_vo = buf((size_t) qsub * kMlaH * kDv); c_iq = buf((size_t) qsub * kIdxH * kIdxD);
        c_iw = buf((size_t) qsub * kIdxH); c_score = buf((size_t) qsub * (max_ctx / kKpool + 1));
        c_rows = buf((size_t) T * kK * kEmbd); c_wts = buf((size_t) T * kK);
        CK(cudaMalloc(&c_ids, (size_t) T * kK * sizeof(int32_t)));
        CK(cudaMalloc(&c_sel, (size_t) qsub * kSelLd * sizeof(int32_t)));
        CK(cudaMalloc(&c_cnt, (size_t) qsub * sizeof(int32_t)));
        CK(cudaMalloc(&c_grp_ptr, kNE * sizeof(unsigned long long)));
        CK(cudaMalloc(&c_grp_start, (kNE + 1) * sizeof(int32_t)));
        CK(cudaMalloc(&c_ngroups, sizeof(int32_t)));
        CK(cudaMalloc(&c_ent_dst, (size_t) T * kK * sizeof(int32_t)));
        CK(cudaMalloc(&c_ent_tok, (size_t) T * kK * sizeof(int32_t)));
        CK(cudaMalloc(&c_emb, (size_t) T * kEmbd * sizeof(glm::bf16)));
        if (whole_layer) {
            CK(cudaMalloc(&layer_slots, (size_t) kNE * XL.bytes));
            CK(cudaMallocHost(&layer_host, (size_t) kNE * XL.bytes));
        }
        CK(cudaMalloc(&c_xscratch, strata::kernels::native_expert_scratch_bytes((int64_t) T * kK, kFF)));
        glm::conv_silu_reserve((size_t) T * kKdaC);
        if (!whole_layer && T >= kStreamMin) init_prompt_pool();
        gm.reserve_w((size_t) kEmbd * kMlaH * kDv);   // the largest FP8 matrix the prompt multiplies (MLA's o_proj)
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        std::fprintf(stderr, "strata-glm: prompt path up to %d tokens a chunk; %.1f GiB of VRAM free\n", T, fr / 1073741824.0);
    }

    // --spec: a verification pass is a handful of tokens, and cuBLASLt dequantizes each quantized matrix to BF16 on
    // every call - bytes with nothing to do with T, and a rounding of its own that the decode token never sees.
    // gemv reads the quantized weights once for all its rows, exactly as the decode path does: the same weights and
    // the same numerics the drafts are judged against. Its kernels are instantiated for 1, 2 and 4 rows.
    void dmat(const glm::Mat& W, const float* X, float* Y, int N, int K, int T) {
        if (!spec_verify) { gm.wmat(W, X, Y, N, K, T); return; }
        for (int t = 0; t < T; t += 4)
            glm::gemv(W, X + (size_t) t * K, Y + (size_t) t * N, N, K, std::min(4, T - t), s);
    }
    void dmat16(const glm::bf16* W, const float* X, float* Y, int N, int K, int T) {
        if (!spec_verify) { gm.w16(W, X, Y, N, K, T); return; }
        for (int t = 0; t < T; t += 4)
            glm::gemv_bf16(W, X + (size_t) t * K, Y + (size_t) t * N, N, K, std::min(4, T - t), s);
    }

    void kda_chunk(Layer& ly, int T) {
        using namespace glm;
        dmat(ly.wq, c_xn, c_q, kKdaC, kEmbd, T);
        dmat(ly.wk, c_xn, c_k, kKdaC, kEmbd, T);
        dmat(ly.wv, c_xn, c_v, kKdaC, kEmbd, T);
        const int sl = spec_verify ? sp_slot[&ly - L.data()] : -1;   // --spec: what spec_fix() rolls back to
        if (sl >= 0) {
            constexpr size_t kc = (size_t) 3 * kKdaC * 3;   // the conv state: 3 taps a channel, for q, k and v
            CK(cudaMemcpyAsync(sp_S + (size_t) sl * kKdaH * kKdaD, ly.S, (size_t) kKdaH * kKdaD * sizeof(float),
                               cudaMemcpyDeviceToDevice, s));
            CK(cudaMemcpyAsync(sp_C + (size_t) sl * kc, ly.conv, kc * sizeof(float), cudaMemcpyDeviceToDevice, s));
            float* raw = sp_raw + (size_t) sl * 3 * sp_n;
            CK(cudaMemcpyAsync(raw, c_q, sp_n * sizeof(float), cudaMemcpyDeviceToDevice, s));
            CK(cudaMemcpyAsync(raw + sp_n, c_k, sp_n * sizeof(float), cudaMemcpyDeviceToDevice, s));
            CK(cudaMemcpyAsync(raw + 2 * sp_n, c_v, sp_n * sizeof(float), cudaMemcpyDeviceToDevice, s));
        }
        conv_silu_seq(c_q, ly.q_conv, ly.conv, c_q, kKdaC, T, s);
        conv_silu_seq(c_k, ly.k_conv, ly.conv + (size_t) kKdaC * 3, c_k, kKdaC, T, s);
        conv_silu_seq(c_v, ly.v_conv, ly.conv + (size_t) 2 * kKdaC * 3, c_v, kKdaC, T, s);
        dmat(ly.fa, c_xn, c_t1, kKdaD, kEmbd, T);
        dmat(ly.fb, c_t1, c_gf, kKdaC, kKdaD, T);
        dmat(ly.bproj, c_xn, c_b, kKdaH, kEmbd, T);
        kda_prep_rows(c_q, c_k, c_gf, ly.dt_bias, ly.A_log, kLowerBound, c_b, kKdaH, kKdaD, T, s);
        if (sl >= 0) {   // the scan's inputs, post-prep: the accepted prefix of them is what the state needs
            float* in = sp_in + (size_t) sl * 3 * sp_n;
            CK(cudaMemcpyAsync(in, c_q, sp_n * sizeof(float), cudaMemcpyDeviceToDevice, s));
            CK(cudaMemcpyAsync(in + sp_n, c_k, sp_n * sizeof(float), cudaMemcpyDeviceToDevice, s));
            CK(cudaMemcpyAsync(in + 2 * sp_n, c_gf, sp_n * sizeof(float), cudaMemcpyDeviceToDevice, s));
            CK(cudaMemcpyAsync(sp_b + (size_t) sl * spec_T * kKdaH, c_b, (size_t) T * kKdaH * sizeof(float),
                               cudaMemcpyDeviceToDevice, s));
        }
        kda_scan(ly.S, c_q, c_k, c_v, c_gf, c_b, c_o, kKdaH, kKdaD, T, s);
        dmat(ly.ga, c_xn, c_t1, kKdaD, kEmbd, T);
        dmat(ly.gb, c_t1, c_gate, kKdaC, kKdaD, T);
        kda_out_norm_rows(c_o, ly.onorm, c_gate, kEps, kKdaH, kKdaD, T, s);
        dmat(ly.wo, c_o, c_y, kEmbd, kKdaC, T);
    }

    void dsa_chunk(Layer& ly, int T, int pos0) {
        using namespace glm;
        dmat(ly.qa, c_xn, c_t1, kQLora, kEmbd, T);
        rmsnorm(c_t1, ly.qa_norm, kEps, c_qr, kQLora, T, s);
        dmat(ly.kva, c_xn, c_t1, kR, kEmbd, T);
        if (ly.lat8) rmsnorm_i8(c_t1, ly.kva_norm, kEps, ly.lat8 + (size_t) pos0 * kR, ly.lat8s + (size_t) pos0 * (kR / 64), kR, T, s);
        else rmsnorm_f16(c_t1, ly.kva_norm, kEps, ly.lat + (size_t) pos0 * kR, kR, T, s);
        dmat(ly.iwk, c_xn, c_t1, kIdxD, kEmbd, T);
        layernorm_rows(c_t1, ly.ik_w, ly.ik_b, 1e-6f, ly.ikc + (size_t) (pos0 - ic_base) * kIdxD, kIdxD, T, s);
        dmat(ly.igate, c_xn, ly.igc + (size_t) (pos0 - ic_base) * kIdxD, kIdxD, kEmbd, T);
        const int pool_lo = pos0 / kKpool, pool_hi = (pos0 + T) / kKpool;   // the pools this chunk completes
        idx_pool_rows(ly.ikc, ly.igc, ly.iape, ly.pooled, pool_lo - (int) (ic_base / kKpool), pool_hi - pool_lo, kKpool, kIdxD, s,
                      pool_lo);
        // the queries in sub-chunks of qsub (their buffers - scores over every pool, the absorbed q, the context - are
        // ~650 KB a token at 262K): every key and pool of the chunk is written above, the masks keep it causal
        const int budget = kIdxTopk / kKpool;
        static const bool dense = std::getenv("GLM_DENSE") != nullptr;   // tests: every visible token
        const long long ws = (long long) (kDk + kDv) * kR;
        for (int t0 = 0; t0 < T; t0 += qsub) {
            const int Ts = std::min(qsub, T - t0), p0 = pos0 + t0, phi = (p0 + Ts) / kKpool;
            const float* xn_s = c_xn + (size_t) t0 * kEmbd;
            const float* qr_s = c_qr + (size_t) t0 * kQLora;
            dmat(ly.qb, qr_s, c_qm, kMlaH * kDk, kQLora, Ts);
            const int32_t* sel = nullptr;
            if (phi > budget && !dense) {
                // some query sees more complete pools than the budget: the top ones by the indexer's score, then its tail
                dmat(ly.iwqb, qr_s, c_iq, kIdxH * kIdxD, kQLora, Ts);
                dmat(ly.iwp, xn_s, c_iw, kIdxH, kEmbd, Ts);
                static const bool idx_old = std::getenv("GLM_IDX_OLD") != nullptr;
                if (idx_old) idx_scores_rows(c_iq, c_iw, ly.pooled, c_score, phi, kIdxH, kIdxD, p0, kKpool, Ts, s);
                else idx_scores_tc(c_iq, c_iw, ly.pooled, c_score, phi, p0, kKpool, Ts, s);
                idx_select_rows(c_score, phi, p0, kKpool, budget, c_sel, kSelLd, c_cnt, Ts, s);
                sel = c_sel;
            }
            gm.heads16(ly.kvb, ws, false, c_qm, kMlaH * kDk, c_qa, kMlaH * kR, kR, kDk, Ts, kMlaH);
            mla_attend_tc(c_qa, ly.lat, ly.lat8, ly.lat8s, sel, c_cnt, kSelLd, p0, 1.0f / std::sqrt((float) kDk), c_ctx, Ts, 1,
                          nullptr, s);
            gm.heads16(ly.kvb + (size_t) kDk * kR, ws, true, c_ctx, kMlaH * kR, c_vo, kMlaH * kDv, kDv, kR, Ts, kMlaH);
            dmat(ly.wo, c_vo, c_y + (size_t) t0 * kEmbd, kEmbd, kMlaH * kDv, Ts);
        }
    }

    // The tiered prompt path through MMQ: every routed expert of the layer (VRAM's copied device to device, RAM's
    // and the disk's over PCIe) goes through the two staging halves; a half's experts are one gate/up and one down
    // launch. Rows (entries) are in processing order: VRAM, RAM, disk.
    void moe_chunk_mmq(int m, int T, const std::vector<std::vector<int32_t>>& by,
                       const std::vector<std::pair<int, const uint8_t*>>& resident, const std::vector<int>& ram,
                       const std::vector<std::pair<int32_t, int32_t>>& disk) {
        namespace mq = strata::prefill::mmq;
        const auto& lay = strata::kernels::cpu::expert_layout();
        const auto& f = lay.fmt[(size_t) m];
        // MMQ steps from one expert's matrix to the next in whole blocks (expert_bytes / 36): the blobs sit in the
        // staging slots at a stride that is a multiple of the block (and of 16), not at their own 14,155,792 bytes
        const size_t S = (XL.bytes + 143) / 144 * 144;
        std::vector<int> order;
        std::vector<const uint8_t*> vsrc;
        for (const auto& [e, b] : resident) { order.push_back(e); vsrc.push_back(b); }
        for (const int e : ram) order.push_back(e);
        for (const auto& d : disk) order.push_back(d.second);
        const size_t n = order.size(), nres = resident.size(), nram = ram.size();
        CK(cudaEventSynchronize(ev_done[0]));   // the previous layer's uploads from the pinned arrays have run
        CK(cudaEventSynchronize(ev_done[1]));
        int32_t r = 0;
        for (size_t j = 0; j < n; ++j) {
            h_bounds[j] = r;
            for (const int32_t en : by[(size_t) order[j]]) { h_src[r] = en / kK; h_dstv[r] = en; ++r; }
        }
        h_bounds[n] = r;
        CK(cudaMemcpyAsync(m_src, h_src, (size_t) r * sizeof(int32_t), cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(m_dst, h_dstv, (size_t) r * sizeof(int32_t), cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(m_bounds, h_bounds, (n + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, s));
        mq::quantize(c_xn, m_src, m_xq, XL.gu_type, kEmbd, kEmbd, r, s);
        // the disk reads start now, kRing in flight
        const size_t d0 = nres + nram;
        size_t next_read = 0;
        auto issue_reads = [&](size_t upto) {
            for (; next_read < disk.size() && next_read < upto; ++next_read) {
                const int slot = (int) (ring_pos + next_read) % kRing;
                CK(cudaEventSynchronize(ev_ring[slot]));
                reader.start(kRingJob0 + slot, lay.blob_offset(m, disk[next_read].second), XL.bytes, pring + (size_t) slot * ring_stride);
            }
        };
        issue_reads(kRing);
        for (size_t b0 = 0, bi = 0; b0 < n; b0 += kBatch, ++bi) {
            const int h = (int) (bi & 1);
            const int nb = (int) std::min<size_t>(kBatch, n - b0);
            CK(cudaEventSynchronize(ev_done[h]));
            uint8_t* half = stage + (size_t) h * kBatch * S;
            int32_t maxr = 0;
            for (int k = 0; k < nb; ++k) {
                const size_t i = b0 + (size_t) k;
                uint8_t* dst = half + (size_t) k * S;
                maxr = std::max<int32_t>(maxr, h_bounds[i + 1] - h_bounds[i]);
                if (i < nres) {
                    CK(cudaMemcpyAsync(dst, vsrc[i], XL.bytes, cudaMemcpyDeviceToDevice, pcs));
                } else if (i < d0) {
                    CK(cudaMemcpyAsync(dst, tier.stable_blob(m, order[i]), XL.bytes, cudaMemcpyHostToDevice, pcs));
                } else {
                    const size_t di = i - d0;
                    issue_reads(di + kRing);
                    const int slot = (int) (ring_pos + di) % kRing;
                    const auto t = std::chrono::steady_clock::now();
                    if (!reader.wait(kRingJob0 + slot)) { std::fprintf(stderr, "strata-glm: a prompt read failed\n"); std::exit(1); }
                    ts.file_wait_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
                    CK(cudaMemcpyAsync(dst, pring + (size_t) slot * ring_stride + lay.blob_offset(m, order[i]) % 4096,
                                       XL.bytes, cudaMemcpyHostToDevice, pcs));
                    CK(cudaEventRecord(ev_ring[slot], pcs));
                }
            }
            CK(cudaEventRecord(ev_copy[h], pcs));
            const int32_t r0 = h_bounds[b0], nr = h_bounds[b0 + (size_t) nb] - r0;
            for (int k = 0; k <= nb; ++k) h_rb[h][k] = h_bounds[b0 + (size_t) k] - r0;
            CK(cudaMemcpyAsync(m_rb[h], h_rb[h], (size_t) (nb + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, s));
            CK(cudaStreamWaitEvent(s, ev_copy[h], 0));
            glm::gather_tails(half, S, f.tail_off, nb, m_tails[h], s);
            mq::Product gu;
            gu.w = half; gu.type = XL.gu_type; gu.w_rows = 2 * kFF; gu.w_cols = kEmbd; gu.expert_bytes = S;
            gu.n = nb; gu.xq = m_xq; gu.bounds = m_bounds + b0; gu.ids = m_ident; gu.total_rows = r; gu.max_rows = maxr;
            gu.dst = m_gu; gu.ld_dst = 2 * kFF;
            mmq_ctx->run(gu, s);
            glm::swiglu_rows(m_gu, m_h, m_rb[h], nb, m_tails[h], r0, nr, kFF, kSwigluLimit, s);
            mq::quantize(m_h + (size_t) r0 * kFF, nullptr, m_hq, XL.d_type, kFF, kFF, nr, s);
            mq::Product dn;
            dn.w = half + f.down_off; dn.type = XL.d_type; dn.w_rows = kEmbd; dn.w_cols = kFF; dn.expert_bytes = S;
            dn.n = nb; dn.xq = m_hq; dn.bounds = m_rb[h]; dn.ids = m_dst + r0; dn.total_rows = nr; dn.max_rows = maxr;
            dn.dst = c_rows; dn.ld_dst = kEmbd;
            mmq_ctx->run(dn, s);
            glm::scale_entry_wts(c_wts, m_dst + r0, m_rb[h], nb, m_tails[h], nr, s);
            CK(cudaEventRecord(ev_done[h], s));
            static const bool serial = std::getenv("GLM_MMQ_SYNC") != nullptr;   // debugging: no overlap at all
            if (serial) { CK(cudaStreamSynchronize(s)); CK(cudaStreamSynchronize(pcs)); }
        }
        ring_pos += disk.size();
    }

    void moe_chunk(Layer& ly, int l, int T) {
        using namespace glm;
        dmat16(ly.router, c_xn, c_t1, kNE, kEmbd, T);
        route_rows(c_t1, ly.router_bias, kNE, kK, kRouteScale, c_ids, c_wts, T, s);
        dmat(ly.sg, c_xn, c_t1, kFF, kEmbd, T);
        dmat(ly.su, c_xn, c_t2, kFF, kEmbd, T);
        swiglu_clamp(c_t1, c_t2, c_t3, T * kFF, kSwigluLimit, s);
        dmat(ly.sd, c_t3, c_y, kEmbd, kFF, T);
        // the tokens grouped by expert
        std::vector<int32_t> hid((size_t) T * kK);
        CK(cudaMemcpyAsync(hid.data(), c_ids, hid.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, s));
        if (spec_verify && x_host_m)   // --spec: the CPU pool needs this layer's activations on the host
            CK(cudaMemcpyAsync(x_host_m, c_xn, (size_t) T * kEmbd * sizeof(float), cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        const int m = l - kDenseLead;
        // --spec: the experts the CPU pool computes, and the entries each takes (alive until cpux.wait())
        std::vector<const uint8_t*> cpu_blob;
        std::vector<const int32_t*> cpu_row;
        std::vector<int32_t> cpu_nrow, cpu_ent, cpu_tok;
        std::vector<float*> cpu_op;
        int ncpu = 0;
        std::vector<std::vector<int32_t>> by((size_t) kNE);
        for (int i = 0; i < T * kK; ++i) by[(size_t) hid[(size_t) i]].push_back(i);
        // one kernel over (expert, its blob on the device) groups; each writes only its own entries' rows
        auto run = [&](const std::vector<std::pair<int, const uint8_t*>>& grp) {
            if (grp.empty()) return;
            std::vector<unsigned long long> gp;
            std::vector<int32_t> gs, et, ed;
            for (const auto& [e, blob] : grp) {
                gp.push_back((unsigned long long) blob);
                gs.push_back((int32_t) et.size());
                for (const int32_t i : by[(size_t) e]) { et.push_back(i / kK); ed.push_back(i); }
            }
            gs.push_back((int32_t) et.size());
            const int32_t ng = (int32_t) gp.size();
            CK(cudaMemcpyAsync(c_grp_ptr, gp.data(), gp.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice, s));
            CK(cudaMemcpyAsync(c_grp_start, gs.data(), gs.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s));
            CK(cudaMemcpyAsync(c_ngroups, &ng, sizeof(int32_t), cudaMemcpyHostToDevice, s));
            CK(cudaMemcpyAsync(c_ent_tok, et.data(), et.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s));
            CK(cudaMemcpyAsync(c_ent_dst, ed.data(), ed.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s));
            if (fake3) glm::fake3_groups(c_grp_ptr, c_ngroups, ng, XL.tail_off / 36, XL.tail_off, s);
            strata::kernels::native_expert_grouped_f32(XL, c_grp_ptr, c_grp_start, c_ngroups, c_ent_dst, c_ent_tok, ng,
                                                       (int64_t) T * kK, c_xn, c_xscratch, c_rows, s);
            CK(cudaStreamSynchronize(s));   // the host vectors above, and the staging slots are free again
        };
        if (!tiered) {   // the whole layer's experts into their slots (one sequential read)
            _fseeki64(experts, (long long) m * kNE * (long long) XL.bytes, SEEK_SET);
            if (std::fread(layer_host, 1, (size_t) kNE * XL.bytes, experts) != (size_t) kNE * XL.bytes) {
                std::fprintf(stderr, "strata-glm: short read of layer %d's experts\n", l);
                std::exit(1);
            }
            CK(cudaMemcpyAsync(layer_slots, layer_host, (size_t) kNE * XL.bytes, cudaMemcpyHostToDevice, s));
            std::vector<std::pair<int, const uint8_t*>> all;
            for (int e = 0; e < kNE; ++e)
                if (!by[(size_t) e].empty()) all.emplace_back(e, layer_slots + (size_t) e * XL.bytes);
            run(all);
        } else {
            // the tiers: VRAM's experts in place; RAM's copied in kStage at a time; the disk's read by the tier's
            // threads and copied in as they arrive
            note(m, hid.data(), hid.size());
            std::vector<std::pair<int, const uint8_t*>> resident, batch;
            std::vector<int> ram;
            std::vector<std::pair<int32_t, int32_t>> disk;
            for (int e = 0; e < kNE; ++e) {
                if (by[(size_t) e].empty()) continue;
                const int32_t r = res[(size_t) m * kNE + (size_t) e];
                if (r >= 0) resident.emplace_back(e, vslot[(size_t) r]);
                else if (tier.has_copy(m, e)) ram.push_back(e);
                else disk.emplace_back(m, e);
            }
            // --spec: the RAM experts go to the CPU pool (its share, as in decode) - a PCIe copy of every one of
            // them would cost more than the whole pass. The disk's stay on the ring below.
            if (spec_verify && !prompt_mmq && cpu_share > 0 && cpux.ready() && x_host_m && !ram.empty()) {
                cpu_acc_m += cpu_share * (double) ram.size();
                const int want = std::min((int) cpu_acc_m, (int) ram.size());
                cpu_acc_m -= want;
                for (int i = 0; i < want; ++i)
                    for (const int32_t en : by[(size_t) ram[(size_t) i]]) cpu_ent.push_back(en);
                cpu_op.reserve(cpu_ent.size());
                cpu_tok.reserve(cpu_ent.size());
                for (const int32_t en : cpu_ent) {
                    cpu_op.push_back(cpu_rows_m + (size_t) en * kEmbd);
                    cpu_tok.push_back(en / kK);
                }
                cpu_blob.reserve(want);
                cpu_row.reserve(want);
                cpu_nrow.reserve(want);
                for (int off = 0, i = 0; i < want; ++i) {
                    const int e = ram[(size_t) i];
                    cpu_blob.push_back(tier.stable_blob(m, e));
                    cpu_row.push_back(cpu_tok.data() + off);
                    cpu_nrow.push_back((int32_t) by[(size_t) e].size());
                    off += (int) by[(size_t) e].size();
                }
                ncpu = want;
                ts.cpu += want;
                ram.erase(ram.begin(), ram.begin() + want);
                cpux.start_multi(x_host_m, cpu_blob.data(), cpu_row.data(), cpu_nrow.data(), cpu_op.data(), ncpu);
            }
            if (prompt_mmq && ps_m == m) {
                moe_chunk_stream(m, T, by);
                combine_rows_t(c_rows, c_wts, kK, c_y, kEmbd, T, s);
                if (routes) routes->push_back({l, hid});
                return;
            }
            ts.vram += (long long) resident.size();
            ts.ram += (long long) ram.size();
            ts.file += (long long) disk.size();
            if (prompt_mmq) {
                moe_chunk_mmq(m, T, by, resident, ram, disk);
                combine_rows_t(c_rows, c_wts, kK, c_y, kEmbd, T, s);
                if (routes) routes->push_back({l, hid});
                return;
            }
            // the disk reads start first (the longest wait), up to kRing in flight in the pinned ring
            const auto& lay = strata::kernels::cpu::expert_layout();
            size_t next_read = 0;
            auto issue_reads = [&](size_t upto) {
                for (; next_read < disk.size() && next_read < upto; ++next_read) {
                    const int slot = (int) (ring_pos + next_read) % kRing;
                    CK(cudaEventSynchronize(ev_ring[slot]));   // its previous blob has been copied out
                    reader.start(kRingJob0 + slot, lay.blob_offset(m, disk[next_read].second), XL.bytes, pring + (size_t) slot * ring_stride);
                }
            };
            issue_reads(kRing);
            run(resident);
            // the rest through the two halves of `stage`: copies of one half on pcs while the other half computes
            const size_t n_st = ram.size() + disk.size();
            for (size_t b0 = 0, bi = 0; b0 < n_st; b0 += kBatch, ++bi) {
                const int h = (int) (bi & 1);
                const size_t nb = std::min<size_t>(kBatch, n_st - b0);
                CK(cudaEventSynchronize(ev_done[h]));   // this half's previous kernel is done: slots and arrays are free
                int32_t ne = 0;
                for (size_t k = 0; k < nb; ++k) {
                    const size_t i = b0 + k;
                    uint8_t* dst = stage + ((size_t) h * kBatch + k) * XL.bytes;
                    int e;
                    if (i < ram.size()) {
                        e = ram[i];
                        CK(cudaMemcpyAsync(dst, tier.stable_blob(m, e), XL.bytes, cudaMemcpyHostToDevice, pcs));
                    } else {
                        const size_t di = i - ram.size();
                        e = disk[di].second;
                        issue_reads(di + kRing);
                        const int slot = (int) (ring_pos + di) % kRing;
                        const auto t = std::chrono::steady_clock::now();
                        if (!reader.wait(kRingJob0 + slot)) { std::fprintf(stderr, "strata-glm: a prompt read failed\n"); std::exit(1); }
                        ts.file_wait_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
                        CK(cudaMemcpyAsync(dst, pring + (size_t) slot * ring_stride + lay.blob_offset(m, e) % 4096, XL.bytes,
                                           cudaMemcpyHostToDevice, pcs));
                        CK(cudaEventRecord(ev_ring[slot], pcs));
                    }
                    h_ptr[h][k] = (unsigned long long) dst;
                    h_start[h][k] = ne;
                    for (const int32_t en : by[(size_t) e]) { h_tok[h][ne] = en / kK; h_dst[h][ne] = en; ++ne; }
                }
                h_start[h][nb] = ne;
                *h_ng[h] = (int32_t) nb;
                CK(cudaEventRecord(ev_copy[h], pcs));
                CK(cudaMemcpyAsync(b_ptr[h], h_ptr[h], nb * sizeof(unsigned long long), cudaMemcpyHostToDevice, s));
                CK(cudaMemcpyAsync(b_start[h], h_start[h], (nb + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, s));
                CK(cudaMemcpyAsync(b_ng[h], h_ng[h], sizeof(int32_t), cudaMemcpyHostToDevice, s));
                CK(cudaMemcpyAsync(b_dst[h], h_dst[h], (size_t) ne * sizeof(int32_t), cudaMemcpyHostToDevice, s));
                CK(cudaMemcpyAsync(b_tok[h], h_tok[h], (size_t) ne * sizeof(int32_t), cudaMemcpyHostToDevice, s));
                CK(cudaStreamWaitEvent(s, ev_copy[h], 0));
                strata::kernels::native_expert_grouped_f32(XL, b_ptr[h], b_start[h], b_ng[h], b_dst[h], b_tok[h], kBatch,
                                                           (int64_t) T * kK, c_xn, c_xscratch, c_rows, s);
                CK(cudaEventRecord(ev_done[h], s));
            }
            ring_pos += disk.size();
        }
        if (ncpu > 0) {   // --spec: the CPU's rows in, then the weighted combine. The stream sync at the top of
            cpux.wait();  // the next layer is what keeps cpu_rows_m intact until its rows have been copied.
            for (const int32_t en : cpu_ent)
                CK(cudaMemcpyAsync(c_rows + (size_t) en * kEmbd, cpu_rows_m + (size_t) en * kEmbd,
                                   (size_t) kEmbd * sizeof(float), cudaMemcpyHostToDevice, s));
        }
        combine_rows_t(c_rows, c_wts, kK, c_y, kEmbd, T, s);
        if (routes) routes->push_back({l, hid});
    }

    void dense_chunk(Layer& ly, int T) {
        using namespace glm;
        dmat(ly.dg, c_xn, c_t1, kDenseFF, kEmbd, T);
        dmat(ly.du, c_xn, c_t2, kDenseFF, kEmbd, T);
        swiglu_clamp(c_t1, c_t2, c_t3, T * kDenseFF, kSwigluLimit, s);
        dmat(ly.dd, c_t3, c_y, kEmbd, kDenseFF, T);
    }

    // routing log (for the expert profile): per MoE layer and chunk, the T*K chosen ids
    struct Route { int layer; std::vector<int32_t> ids; };
    std::vector<Route>* routes = nullptr;

    // T tokens at positions pos0..: the last one's next-token logits in `logits`
    void forward_chunk(const int* toks, int T, int pos0, const std::string& dump_dir, int nlog = 1) {
        ensure_ctx(pos0 + T);
        using namespace glm;
        heat_tokens += T;
        ic_prepare(pos0, T);
        if (lo == 0) {   // a later stage finds its input streams in c_streams (copied there by the driver)
            for (int t = 0; t < T; ++t)
                CK(cudaMemcpyAsync(c_emb + (size_t) t * kEmbd, embed.data() + (size_t) toks[t] * kEmbd, kEmbd * sizeof(bf16),
                                   cudaMemcpyHostToDevice, s));
            bf16_to_f32(c_emb, c_x, T * kEmbd, s);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < kHc; ++j)
                    CK(cudaMemcpyAsync(c_streams + ((size_t) t * kHc + j) * kEmbd, c_x + (size_t) t * kEmbd, kEmbd * sizeof(float),
                                       cudaMemcpyDeviceToDevice, s));
        }
        for (int l = lo; l < hi; ++l) {
            Layer& ly = L[(size_t) l];
            static const bool no_stream = std::getenv("GLM_NO_STREAM") != nullptr;
            if (l >= kDenseLead && T >= kStreamMin && prompt_mmq && !no_stream) prestage(l - kDenseLead);
            dmat16(ly.hc_attn_fn, c_streams, c_mix, 24, kHc * kEmbd, T);
            hc_pre_rows(c_streams, c_mix, ly.hc_attn_base, ly.hc_attn_scale, kEmbd, kEps, kHcEps, kSinkhorn, c_x, c_post, c_comb, T, s);
            rmsnorm(c_x, ly.in_norm, kEps, c_xn, kEmbd, T, s);
            if (is_dsa(l)) dsa_chunk(ly, T, pos0); else kda_chunk(ly, T);
            hc_post_rows(c_y, c_streams, c_post, c_comb, c_streams, kEmbd, T, s);
            dmat16(ly.hc_ffn_fn, c_streams, c_mix, 24, kHc * kEmbd, T);
            hc_pre_rows(c_streams, c_mix, ly.hc_ffn_base, ly.hc_ffn_scale, kEmbd, kEps, kHcEps, kSinkhorn, c_x, c_post, c_comb, T, s);
            rmsnorm(c_x, ly.post_norm, kEps, c_xn, kEmbd, T, s);
            if (l < kDenseLead) dense_chunk(ly, T); else moe_chunk(ly, l, T);
            hc_post_rows(c_y, c_streams, c_post, c_comb, c_streams, kEmbd, T, s);
            if (!dump_dir.empty()) {
                std::vector<float> h((size_t) kHc * kEmbd);
                CK(cudaMemcpyAsync(h.data(), c_streams + (size_t) (T - 1) * kHc * kEmbd, h.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, s));
                CK(cudaStreamSynchronize(s));
                char name[64];
                std::snprintf(name, sizeof name, "/l%02d.f32", l);
                if (std::FILE* f = std::fopen((dump_dir + name).c_str(), "wb")) { std::fwrite(h.data(), 4, h.size(), f); std::fclose(f); }
            }
        }
        if (hi == kLayers) {
            if (nlog <= 1) {
                hc_mean(c_streams + (size_t) (T - 1) * kHc * kEmbd, x, kEmbd, s);
                rmsnorm(x, final_norm, kEps, xn, kEmbd, 1, s);
                gemv(lm_head, xn, logits, kVocab, kEmbd, 1, s);
            } else {   // --spec: every row's next-token logits; logits[t] decides the token after position pos0+t
                for (int t = 0; t < nlog; ++t)
                    hc_mean(c_streams + (size_t) t * kHc * kEmbd, c_x + (size_t) t * kEmbd, kEmbd, s);
                rmsnorm(c_x, final_norm, kEps, c_xn, kEmbd, nlog, s);
                for (int t = 0; t < nlog; t += 4)   // gemv's kernels are built for 1, 2 and 4 rows
                    gemv(lm_head, c_xn + (size_t) t * kEmbd, logits + (size_t) t * kVocab, kVocab, kEmbd,
                         std::min(4, nlog - t), s);
            }
        }
        CK(cudaStreamSynchronize(s));   // the driver copies c_streams on to the next stage after this
    }

    void load(const std::string& pack) {
        ic_win = std::max(chunk_cap, 1) + 2 * kKpool;
        std::string err;
        if (!ck.open(pack, err)) { std::fprintf(stderr, "strata-glm: %s\n", err.c_str()); std::exit(1); }
        const auto t0 = std::chrono::steady_clock::now();
        L.resize(kLayers);
        for (int l = lo; l < hi; ++l) {
            Layer& y = L[(size_t) l];
            const std::string p = "model.language_model.layers." + std::to_string(l) + ".";
            y.hc_attn_fn = ck.bf(p + "hc_attn_fn"); y.hc_attn_base = ck.bf(p + "hc_attn_base"); y.hc_attn_scale = ck.bf(p + "hc_attn_scale");
            y.hc_ffn_fn = ck.bf(p + "hc_ffn_fn"); y.hc_ffn_base = ck.bf(p + "hc_ffn_base"); y.hc_ffn_scale = ck.bf(p + "hc_ffn_scale");
            y.in_norm = ck.bf(p + "input_layernorm.weight");
            y.post_norm = ck.bf(p + "post_attention_layernorm.weight");
            const std::string a = p + "self_attn.";
            if (!is_dsa(l)) {
                y.wq = mat(a + "q_proj.weight", fp4_qk); y.wk = mat(a + "k_proj.weight", fp4_qk);
                y.wv = mat(a + "v_proj.weight", fp4_vo);
                y.q_conv = ck.f32(a + "q_conv1d.weight"); y.k_conv = ck.f32(a + "k_conv1d.weight"); y.v_conv = ck.f32(a + "v_conv1d.weight");
                y.fa = mat(a + "f_a_proj.weight"); y.fb = mat(a + "f_b_proj.weight");
                y.dt_bias = ck.f32(a + "dt_bias"); y.A_log = ck.f32(a + "A_log");
                y.bproj = mat(a + "b_proj.weight"); y.ga = mat(a + "g_a_proj.weight"); y.gb = mat(a + "g_b_proj.weight");
                y.onorm = ck.bf(a + "o_norm.weight"); y.wo = mat(a + "o_proj.weight", fp4_vo);
                CK(cudaMalloc(&y.S, (size_t) kKdaH * kKdaD * kKdaD * sizeof(float)));
                CK(cudaMalloc(&y.conv, (size_t) 3 * kKdaC * 3 * sizeof(float)));
            } else {
                y.qa = mat(a + "q_a_proj.weight", fp4_mla); y.qa_norm = ck.bf(a + "q_a_layernorm.weight");
                y.qb = mat(a + "q_b_proj.weight", fp4_mla);
                y.kva = mat(a + "kv_a_proj_with_mqa.weight", fp4_mla); y.kva_norm = ck.bf(a + "kv_a_layernorm.weight"); y.kvb = ck.bf(a + "kv_b_proj.weight");
                y.wo = mat(a + "o_proj.weight", fp4_mla);
                const std::string ip = a + "indexer.";
                y.iwqb = mat(ip + "wq_b.weight", fp4_mla); y.iwk = mat(ip + "wk.weight"); y.ik_w = ck.bf(ip + "k_norm.weight");
                y.ik_b = ck.bf(ip + "k_norm.bias"); y.iwp = mat(ip + "weights_proj.weight");
                y.igate = mat(ip + "index_kpool_compress_gate"); y.iape = ck.bf(ip + "index_kpool_compress_ape");
                ctx_cap = std::min(max_ctx, kCtxStep);
                ctx_alloc(y, ctx_cap, 0);
                CK(cudaMalloc(&y.ikc, (size_t) ic_win * kIdxD * sizeof(float)));
                CK(cudaMalloc(&y.igc, (size_t) ic_win * kIdxD * sizeof(float)));
            }
            const std::string m = p + "mlp.";
            if (l < kDenseLead) {
                y.dg = nvfp4_mat(m + "gate_proj", kDenseFF, kEmbd);
                y.du = nvfp4_mat(m + "up_proj", kDenseFF, kEmbd);
                y.dd = nvfp4_mat(m + "down_proj", kEmbd, kDenseFF);
            } else {
                y.router = ck.bf(m + "gate.weight");
                y.router_bias = ck.f32(m + "gate.e_score_correction_bias");
                y.sg = mat(m + "shared_experts.gate_proj.weight", fp4_shared);
                y.su = mat(m + "shared_experts.up_proj.weight", fp4_shared);
                y.sd = mat(m + "shared_experts.down_proj.weight", fp4_shared);
            }
        }
        if (hi == kLayers) {   // the last stage owns the head
            final_norm = ck.bf("model.language_model.norm.weight");
            lm_head = mat("lm_head.weight", fp4_head);
        }
        if (fp4_saved) std::fprintf(stderr, "strata-glm: dense NVFP4 saves %.2f GB of VRAM\n", fp4_saved / 1e9);
        if (lo == 0) {   // the first stage owns the embedding
            const auto b = ck.read("model.language_model.embed_tokens.weight");
            embed.resize(b.size() / 2);
            std::memcpy(embed.data(), b.data(), b.size());
        }
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        std::fprintf(stderr, "strata-glm: dense weights loaded in %.1f s; %.1f GiB of VRAM free\n", sec, fr / 1073741824.0);

        CK(cudaStreamCreate(&s));
        auto buf = [&](size_t n) { float* p = nullptr; CK(cudaMalloc(&p, n * sizeof(float))); return p; };
        streams = buf(kHc * kEmbd); mix = buf(32); x = buf(kEmbd); xn = buf(kEmbd); post = buf(4); comb = buf(16);
        y = buf(kEmbd); tmp1 = buf(kDenseFF); tmp2 = buf(kDenseFF); tmp3 = buf(kDenseFF);
        q = buf(kKdaC); k = buf(kKdaC); v = buf(kKdaC); gf = buf(kKdaC); b = buf(kKdaH); o = buf(kKdaC); gate = buf(kKdaC);
        mixp = buf(24 * kMixParts); p_mixp = buf(24 * kMixParts);
        q_resid = buf(kQLora); qm = buf(kMlaH * kDk); qa = buf(kMlaH * kR); ctx = buf(kMlaH * kR); vo = buf(kMlaH * kDv);
        iq = buf(kIdxH * kIdxD); ik = buf(kIdxD); ig = buf(kIdxD); iw = buf(kIdxH); score = buf(max_ctx / kKpool + 1);
        logits = buf((size_t) kVocab * logits_cap); rows = buf(kK * kEmbd); wts = buf(kK);
        CK(cudaMalloc(&ids, kK * sizeof(int32_t)));
        CK(cudaMallocHost(&hid_pin, kK * sizeof(int32_t)));
        CK(cudaMallocHost(&wts_pin, 2 * kK * sizeof(float)));
        CK(cudaEventCreateWithFlags(&ev_ids, cudaEventDisableTiming));
        CK(cudaMalloc(&sel, (size_t) kSelLd * sizeof(int32_t)));
        CK(cudaMalloc(&sel_cnt, sizeof(int32_t)));
        CK(cudaMalloc(&mla_part, glm::mla_tc_part_floats(1, kMlaSplit) * sizeof(float)));
        CK(cudaMalloc(&emb_row, kEmbd * sizeof(glm::bf16)));

        // the routed experts: kK device slots, fed from experts.bin
        XL = strata::kernels::native_expert_layout(40, 40, kEmbd, kFF);
        XL.swiglu_limit = kSwigluLimit;
        experts = std::fopen((pack + "/experts.bin").c_str(), "rb");
        if (!experts) { std::fprintf(stderr, "strata-glm: cannot open %s/experts.bin\n", pack.c_str()); std::exit(1); }
        CK(cudaMalloc(&slots, (size_t) kK * XL.bytes));
        CK(cudaMallocHost(&host_blobs, (size_t) kK * XL.bytes));
        CK(cudaMalloc(&grp_ptr, kK * sizeof(unsigned long long)));
        CK(cudaMalloc(&grp_start, (kK + 1) * sizeof(int32_t)));
        CK(cudaMalloc(&n_groups, sizeof(int32_t)));
        CK(cudaMalloc(&ent_dst, kK * sizeof(int32_t)));
        CK(cudaMalloc(&ent_tok, kK * sizeof(int32_t)));
        std::vector<unsigned long long> gp(kK);
        std::vector<int32_t> gs(kK + 1), ed(kK), et(kK, 0);
        for (int j = 0; j < kK; ++j) { gp[j] = (unsigned long long) (slots + (size_t) j * XL.bytes); gs[j] = j; ed[j] = j; }
        gs[kK] = kK;
        const int32_t ng = kK;
        CK(cudaMemcpy(grp_ptr, gp.data(), kK * sizeof(unsigned long long), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(grp_start, gs.data(), (kK + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(n_groups, &ng, sizeof(int32_t), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ent_dst, ed.data(), kK * sizeof(int32_t), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ent_tok, et.data(), kK * sizeof(int32_t), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&xscratch, strata::kernels::native_expert_scratch_bytes(kK, kFF)));
        reset();
    }

    void reset() {
        for (auto& l : L) {
            if (l.S) CK(cudaMemset(l.S, 0, (size_t) kKdaH * kKdaD * kKdaD * sizeof(float)));
            if (l.conv) CK(cudaMemset(l.conv, 0, (size_t) 3 * kKdaC * 3 * sizeof(float)));
        }
    }

    void kda(Layer& ly) {
        using namespace glm;
        gemv(ly.wq, xn, q, kKdaC, kEmbd, 1, s);
        gemv(ly.wk, xn, k, kKdaC, kEmbd, 1, s);
        gemv(ly.wv, xn, v, kKdaC, kEmbd, 1, s);
        conv_silu_step(q, ly.q_conv, ly.conv, q, kKdaC, s);
        conv_silu_step(k, ly.k_conv, ly.conv + (size_t) kKdaC * 3, k, kKdaC, s);
        conv_silu_step(v, ly.v_conv, ly.conv + (size_t) 2 * kKdaC * 3, v, kKdaC, s);
        gemv(ly.fa, xn, tmp1, kKdaD, kEmbd, 1, s);
        gemv(ly.fb, tmp1, gf, kKdaC, kKdaD, 1, s);
        gemv(ly.bproj, xn, b, kKdaH, kEmbd, 1, s);
        kda_prep(q, k, gf, ly.dt_bias, ly.A_log, kLowerBound, b, kKdaH, kKdaD, s);
        kda_step(ly.S, q, k, v, gf, b, o, kKdaH, kKdaD, s);
        gemv(ly.ga, xn, tmp1, kKdaD, kEmbd, 1, s);
        gemv(ly.gb, tmp1, gate, kKdaC, kKdaD, 1, s);
        kda_out_norm(o, ly.onorm, gate, kEps, kKdaH, kKdaD, s);
        gemv(ly.wo, o, y, kEmbd, kKdaC, 1, s);
    }

    void dsa(Layer& ly, int pos) {
        using namespace glm;
        gemv(ly.qa, xn, tmp1, kQLora, kEmbd, 1, s);
        rmsnorm(tmp1, ly.qa_norm, kEps, q_resid, kQLora, 1, s);
        gemv(ly.qb, q_resid, qm, kMlaH * kDk, kQLora, 1, s);
        gemv(ly.kva, xn, tmp1, kR, kEmbd, 1, s);
        if (ly.lat8) rmsnorm_i8(tmp1, ly.kva_norm, kEps, ly.lat8 + (size_t) pos * kR, ly.lat8s + (size_t) pos * (kR / 64), kR, 1, s);
        else rmsnorm_f16(tmp1, ly.kva_norm, kEps, ly.lat + (size_t) pos * kR, kR, 1, s);
        // the indexer's caches and, when a pool completes, its pooled key
        gemv(ly.iwqb, q_resid, iq, kIdxH * kIdxD, kQLora, 1, s);
        gemv(ly.iwk, xn, tmp1, kIdxD, kEmbd, 1, s);
        layernorm(tmp1, ly.ik_w, ly.ik_b, 1e-6f, ly.ikc + (size_t) (pos - ic_base) * kIdxD, kIdxD, s);
        gemv(ly.igate, xn, ly.igc + (size_t) (pos - ic_base) * kIdxD, kIdxD, kEmbd, 1, s);
        if (pos % kKpool == kKpool - 1) {
            const int p0 = pos - (kKpool - 1);
            idx_pool(ly.ikc + (size_t) (p0 - ic_base) * kIdxD, ly.igc + (size_t) (p0 - ic_base) * kIdxD, ly.iape,
                     ly.pooled + (size_t) (pos / kKpool) * kIdxD,
                     kKpool, kIdxD, s);
        }
        // the selection (on the GPU): every visible token up to the budget of pools, then the top-scoring pools and
        // this query's incomplete tail
        const int n_pool = (pos + 1) / kKpool;
        const int32_t* use = nullptr;
        if (n_pool > kIdxTopk / kKpool) {
            gemv(ly.iwp, xn, iw, kIdxH, kEmbd, 1, s);
            idx_scores(iq, iw, ly.pooled, score, n_pool, kIdxH, kIdxD, s);
            idx_select_rows(score, n_pool, pos, kKpool, kIdxTopk / kKpool, sel, kSelLd, sel_cnt, 1, s);
            use = sel;
        }
        mla_absorb_q(qm, ly.kvb, qa, kMlaH, kDk, kDv, kR, s);
        {   // the keys split over blocks: 64 keys a split at least, 16 splits at most
            const int nk = std::min(pos + 1, kSelLd);
            const int nsplit = std::max(1, std::min(kMlaSplit, (nk + 127) / 128));
            mla_attend_tc(qa, ly.lat, ly.lat8, ly.lat8s, use, sel_cnt, kSelLd, pos, 1.0f / std::sqrt((float) kDk), ctx, 1, nsplit, mla_part, s);
        }
        mla_value(ctx, ly.kvb, vo, kMlaH, kDk, kDv, kR, s);
        gemv(ly.wo, vo, y, kEmbd, kMlaH * kDv, 1, s);
    }

    void moe(Layer& ly, int l) {
        using namespace glm;
        gemv_bf16(ly.router, xn, tmp1, kNE, kEmbd, 1, s);
        route_topk(tmp1, ly.router_bias, kNE, kK, kRouteScale, ids, wts, s);
        CK(cudaMemcpyAsync(hid_pin, ids, kK * sizeof(int32_t), cudaMemcpyDeviceToHost, s));
        if (skip_disk > 0 || skip_ram > 0) CK(cudaMemcpyAsync(wts_pin, wts, kK * sizeof(float), cudaMemcpyDeviceToHost, s));
        if (cpu_share > 0) CK(cudaMemcpyAsync(x_host, xn, kEmbd * sizeof(float), cudaMemcpyDeviceToHost, s));
        CK(cudaEventRecord(ev_ids, s));
        const bool pf = prefetch && l + 1 < hi;   // the next stage's layer: not prefetched across the boundary
        if (pf) predict_enqueue(l + 1);
        // the shared expert (BF16, clamped SwiGLU): the GPU runs it while the host sorts the routed ones
        gemv(ly.sg, xn, tmp1, kFF, kEmbd, 1, s);
        gemv(ly.su, xn, tmp2, kFF, kEmbd, 1, s);
        swiglu_clamp(tmp1, tmp2, tmp3, kFF, kSwigluLimit, s);
        gemv(ly.sd, tmp3, y, kEmbd, kFF, 1, s);
        CK(cudaEventSynchronize(ev_ids));
        int32_t hid[kK];
        std::memcpy(hid, hid_pin, sizeof(hid));
        predict_score(l, hid);
        const int m = l - kDenseLead;
        if (tiered && lru && tier.tier_slots() > 0 && vslot.size() >= (size_t) 2 * kK) {
            note(m, hid, kK);
            if (routes) routes->push_back({l, std::vector<int32_t>(hid, hid + kK)});
            if (timing) CK(cudaEventRecord(tev[(size_t) m * 3], s));
            unsigned long long* gp = gp_host + (size_t) m * kK;
            int32_t pj[kK];
            for (int j = 0; j < kK; ++j) pj[j] = m * kNE + hid[j];
            auto in_sel = [&](int32_t p) { for (int j = 0; j < kK; ++j) if (pj[j] == p) return true; return false; };
            // a VRAM slot for p: the least recent one outside this layer's selection
            auto vram_slot = [&](int32_t p) -> int32_t {
                if (excl && wb) {
                    if (vfree.empty()) evict_one(pj);
                    const auto [f, ev] = vfree.front();
                    vfree.pop_front();
                    if (ev >= 0) CK(cudaStreamWaitEvent(xs, wb_ev[ev], 0));   // its previous expert is written back
                    slot_pair[(size_t) f] = p;
                    res[(size_t) p] = f;
                    vlru.touch(f);
                    evict_one(pj);   // the pool stays full
                    return f;
                }
                int32_t sl = vlru.tail;
                while (sl >= 0 && slot_pair[(size_t) sl] >= 0 && in_sel(slot_pair[(size_t) sl])) sl = vlru.prev[(size_t) sl];
                if ((size_t) sl < spec_slot.size() && spec_slot[(size_t) sl]) {   // a speculative copy may still write it
                    CK(cudaStreamWaitEvent(xs, ev_pfc, 0));
                    std::fill(spec_slot.begin(), spec_slot.end(), 0);
                }
                const int32_t vv = slot_pair[(size_t) sl];
                if (excl && vv >= 0 && !tier.has_copy(vv / kNE, vv % kNE)) {   // write it back before the slot is reused
                    if (tier.spares() == 0) {
                        const int32_t v = ram_victim(pj);
                        if (v >= 0) { tier.promote_done(v / kNE, v % kNE); rlru.remove(v); }
                    }
                    if (uint8_t* d = tier.demote_begin(vv / kNE, vv % kNE)) {
                        CK(cudaMemcpyAsync(d, vslot[(size_t) sl], XL.bytes, cudaMemcpyDeviceToHost, s));
                        tier.demote_commit(vv / kNE, vv % kNE);
                        rlru.touch(vv);
                        ++writebacks;
                    }
                }
                if (vv >= 0) {
                    res[(size_t) vv] = -1;
                    if (!pn.empty() && tier.has_copy(vv / kNE, vv % kNE)) pol_cand(vv);   // only in RAM now
                }
                slot_pair[(size_t) sl] = p;
                res[(size_t) p] = sl;
                vlru.touch(sl);
                return sl;
            };
            // the most confident predictions for the next layer already in RAM, to VRAM on xs: issued right after this
            // layer's RAM copies when it waits for the disk anyway (the copy engine would idle), else after the event
            // the expert kernel waits for (beside the kernel and the next attention)
            bool pf_copied = false;
            auto pf_copy = [&]() {
                pf_copied = true;
                const int mn = m + 1;
                for (int i = 0; i < 3 && pf_vn < pf_copy_max; ++i) {   // ranks past 2: <= 64% routed there
                    const int32_t q = mn * kNE + pf_ids[i];
                    if (res[(size_t) q] >= 0 || !tier.has_copy(mn, pf_ids[i])) continue;
                    pf_vrank[pf_vn] = i;
                    const int32_t sl = vram_slot(q);
                    CK(cudaMemcpyAsync(vslot[(size_t) sl], tier.stable_blob(mn, pf_ids[i]), XL.bytes, cudaMemcpyHostToDevice, xp));
                    if (spec_slot.size() < vslot.size()) spec_slot.resize(vslot.size(), 0);
                    spec_slot[(size_t) sl] = 1;
                    rlru.touch(q);
                    pf_vpair[pf_vn++] = q;
                    ++pf_copies;
                }
                if (pf_vn > 0) { CK(cudaEventRecord(ev_pfc, xp)); pfc_pending = true; }
            };
            bool done[kK] = {};
            uint8_t* dslot[kK] = {};
            for (int i = 0; i < pf_vn; ++i) {
                bool used = false;
                for (int j = 0; j < kK; ++j) used |= pf_vpair[i] == pj[j];
                pf_copies_used += used;
                ++pf_rank_n[pf_vrank[i]];
                pf_rank_used[pf_vrank[i]] += used;
            }
            pf_vn = 0;
            if (pf_pending) { CK(cudaStreamWaitEvent(s, ev_pf, 0)); pf_pending = false; }
            for (const int32_t q : pending_free)   // their copies to VRAM ran before this layer's router
                if (res[(size_t) q] >= 0 && tier.has_copy(q / kNE, q % kNE)) { tier.promote_done(q / kNE, q % kNE); rlru.remove(q); }
            pending_free.clear();
            int pfj[kK], spj[kK];   // the prefetch job (RAM tier) / staging slot already holding j's expert, or -1
            for (int j = 0; j < kK; ++j) {
                pfj[j] = spj[j] = -1;
                for (int i = 0; i < kK; ++i) if (pf_pair[i] == pj[j]) pfj[j] = i;
                if (pfj[j] >= 0 && !reader.done(kK + pfj[j])) reader.promote(kK + pfj[j]);
                if (spool && res[(size_t) pj[j]] < 0 && !tier.has_copy(m, hid[j]))
                    for (int x = 0; x < kSpool; ++x) if (sp_pair[x] == pj[j]) spj[j] = x;
            }
            for (int j = 0; j < kK; ++j) {   // in VRAM already
                const int32_t r = res[(size_t) pj[j]];
                if (r < 0) continue;
                gp[j] = (unsigned long long) vslot[(size_t) r];
                if (!is_static(r)) vlru.touch(r);
                if (rlru.in[(size_t) pj[j]]) rlru.touch(pj[j]);
                pol_use(pj[j]);
                done[j] = true;
                ++ts.vram;
            }
            bool in_a[kK] = {};
            {   // the experts in VRAM now, beside this layer's copies (a speculative copy may still be landing: ev_pfc)
                unsigned long long* gqa = gqa_host + (size_t) m * kK;
                int32_t* gea = gea_host + (size_t) m * (kK + 1);
                int32_t na = 0;
                for (int j = 0; j < kK; ++j)
                    if (done[j]) { gqa[na] = gp[j]; gea[na] = j; in_a[j] = true; ++na; }
                gea[kK] = na;
                if (na > 0) {
                    if (pfc_pending) { CK(cudaStreamWaitEvent(s, ev_pfc, 0)); pfc_pending = false; }
                    if (fake3) glm::fake3_groups(gqa, gea + kK, kK, XL.tail_off / 36, XL.tail_off, s);
                    strata::kernels::native_expert_grouped_f32(XL, gqa, grp_start, gea + kK, gea, ent_tok, kK, kK, xn,
                                                               xscratch, rows, s);
                }
            }
            bool skipj[kK] = {};
            for (int j = 0; j < kK; ++j) {   // on the disk: into the RAM slot of the least recent member outside VRAM
                if (done[j] || tier.has_copy(m, hid[j]) || pfj[j] >= 0 || spj[j] >= 0) continue;
                if (std::max(skip_disk, skip_ram) > 0 && wts_pin[j] / kRouteScale < std::max(skip_disk, skip_ram)) {
                    skipj[j] = done[j] = true;
                    ++skipped;
                    continue;
                }
                if (tier.spares() == 0) {   // a spare first (one a static expert gave up), else the policy's victim
                    const int32_t v = ram_pick(pj);
                    if (v < 0) continue;   // nothing to evict: staged below through the tier's own read
                    tier.promote_done(v / kNE, v % kNE);   // its slot becomes the spare demote_begin takes
                    rlru.remove(v);
                }
                uint8_t* d = tier.demote_begin(m, hid[j]);
                if (d == nullptr) continue;
                reader.start(j, strata::kernels::cpu::expert_layout().blob_offset(m, hid[j]), XL.bytes, d);
                dslot[j] = d;
            }
            if (pf_hist_ok[m]) {   // the disk reads this layer waits for: where the prediction had them
                for (int j = 0; j < kK; ++j) {
                    if (dslot[j] == nullptr) continue;
                    int r = kPred;
                    for (int i = 0; i < kPred; ++i) if (pf_hist[m][i] == hid[j]) { r = i; break; }
                    ++miss_rank[r];
                }
                pf_hist_ok[m] = false;
            }
            // the CPU's share of the RAM hits, computed from the tier while the GPU copies its own
            int ncpu = 0, cj[kK];
            if (cpu_share > 0 && cpux.ready()) {
                int nram = 0;
                for (int j = 0; j < kK; ++j) if (!done[j] && dslot[j] == nullptr && pfj[j] < 0 && spj[j] < 0 && tier.has_copy(m, hid[j])) ++nram;
                cpu_acc += cpu_share * nram;
                const int want = (int) cpu_acc;
                cpu_acc -= want;
                const uint8_t* cb[kK];
                float* co[kK];
                for (int j = kK; j-- > 0 && ncpu < want;) {
                    if (done[j] || dslot[j] != nullptr || pfj[j] >= 0 || spj[j] >= 0 || !tier.has_copy(m, hid[j])) continue;
                    cj[ncpu] = j;
                    cb[ncpu] = tier.stable_blob(m, hid[j]);
                    co[ncpu] = cpu_rows + (size_t) j * kEmbd;
                    done[j] = true;
                    rlru.touch(pj[j]);
                    pol_use(pj[j]);
                    if (!pn.empty()) pol_cand(pj[j]);   // it stays only in RAM
                    ++ts.cpu;
                    ++ncpu;
                }
                if (ncpu > 0) cpux.start(x_host, cb, co, ncpu);
            }
            for (int j = 0; j < kK; ++j) {   // in RAM: copied into a VRAM slot (it was going to the GPU anyway)
                if (done[j] || dslot[j] != nullptr || pfj[j] >= 0 || spj[j] >= 0) continue;
                if (skip_ram > 0 && wts_pin[j] / kRouteScale < skip_ram) { skipj[j] = done[j] = true; ++skipped; continue; }
                const uint8_t* b = tier.has_copy(m, hid[j]) ? tier.stable_blob(m, hid[j]) : tier.blob(m, hid[j]);
                if (b == nullptr) { std::fprintf(stderr, "strata-glm: no bytes for expert %d of layer %d\n", hid[j], l); std::exit(1); }
                const bool in_ram = tier.has_copy(m, hid[j]);
                const int32_t sl = vram_slot(pj[j]);
                CK(cudaMemcpyAsync(vslot[(size_t) sl], b, XL.bytes, cudaMemcpyHostToDevice, xs));
                if (!in_ram) CK(cudaStreamSynchronize(xs));   // a mapped-file page, not ours to keep
                gp[j] = (unsigned long long) vslot[(size_t) sl];
                if (in_ram) { rlru.touch(pj[j]); ++ts.ram; if (excl) pending_free.push_back(pj[j]); } else ++ts.file;
                pol_use(pj[j]);
            }
            prefetch_commit_landed(pj);
            // the next layer's predicted experts on the disk (low priority), after this layer's work is queued
            if (pf) {
                CK(cudaEventSynchronize(ev_pred));
                const int mn = m + 1;
                std::memcpy(pf_hist[mn], pf_ids, sizeof(pf_hist[mn]));
                pf_hist_ok[mn] = true;
                bool waits = false;
                for (int j = 0; j < kK; ++j)   // a read still in flight: the copy engine would idle meanwhile
                    waits |= dslot[j] != nullptr || (pfj[j] >= 0 && !reader.done(kK + pfj[j])) ||
                             (spj[j] >= 0 && !reader.done(kSpoolJob0 + spj[j]));
                if (waits) pf_copy();

                for (int i = 0; i < pf_read_max && pf_n < kK; ++i) {
                    const int32_t q = mn * kNE + pf_ids[i];
                    if (res[(size_t) q] >= 0 || tier.has_copy(mn, pf_ids[i])) continue;
                    bool busy = false;
                    for (int x = 0; x < kK; ++x) busy |= pf_pair[x] == q;
                    if (busy) continue;
                    if (tier.spares() == 0) {
                        const int32_t v = ram_pick(pj);
                        if (v < 0) break;
                        tier.promote_done(v / kNE, v % kNE);
                        rlru.remove(v);
                    }
                    uint8_t* d = tier.demote_begin(mn, pf_ids[i]);
                    if (d == nullptr) continue;
                    int job = 0;
                    while (pf_pair[job] >= 0) ++job;
                    reader.start(kK + job, strata::kernels::cpu::expert_layout().blob_offset(mn, pf_ids[i]), XL.bytes, d, false);
                    pf_pair[job] = q;
                    pf_dst[job] = d;
                    ++pf_n;
                    ++pf_reads;
                }
                if (spool) {   // free the slots this layer did not use (a used one: once its copy has run)
                    for (int x = 0; x < kSpool; ++x) {
                        if (sp_pair[x] == -1) continue;
                        if (sp_pair[x] <= -2) {   // copied out at an earlier layer: free once that copy has run
                            if (cudaEventQuery(sp_ev[x]) == cudaSuccess) sp_pair[x] = -1;
                            else (void) cudaGetLastError();   // "not ready" is not an error to report later
                            continue;
                        }
                        bool mine = false;
                        for (int j = 0; j < kK; ++j) mine |= spj[j] == x;
                        if (mine || !reader.done(kSpoolJob0 + x)) continue;   // this layer lands it below / still reading
                        sp_pair[x] = -1;
                    }
                    for (int i = pf_read_max; i < kK; ++i) {
                        const int32_t q = mn * kNE + pf_ids[i];
                        if (res[(size_t) q] >= 0 || tier.has_copy(mn, pf_ids[i])) continue;
                        bool busy = false;
                        for (int x = 0; x < kK; ++x) busy |= pf_pair[x] == q;
                        for (int x = 0; x < kSpool; ++x) busy |= sp_pair[x] == q;
                        if (busy) continue;
                        int x = 0;
                        while (x < kSpool && sp_pair[x] >= 0) ++x;
                        if (x == kSpool) break;
                        reader.start(kSpoolJob0 + x, strata::kernels::cpu::expert_layout().blob_offset(mn, pf_ids[i]), XL.bytes,
                                     spool + (size_t) x * spool_stride, false);
                        sp_pair[x] = q;
                        ++sp_reads;
                    }
                }
            }
            for (int j = 0; j < kK; ++j) {   // this layer's experts staged in the pool
                if (spj[j] < 0) continue;
                const int x = spj[j];
                const auto t = std::chrono::steady_clock::now();
                if (!reader.wait(kSpoolJob0 + x)) { std::fprintf(stderr, "strata-glm: a staged read failed\n"); std::exit(1); }
                ts.pf_wait_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
                const int32_t sl = vram_slot(pj[j]);
                CK(cudaMemcpyAsync(vslot[(size_t) sl], spool + (size_t) x * spool_stride +
                                   strata::kernels::cpu::expert_layout().blob_offset(m, hid[j]) % 4096, XL.bytes,
                                   cudaMemcpyHostToDevice, xs));
                CK(cudaEventRecord(sp_ev[x], xs));
                sp_pair[x] = -2 - x;   // used: free once the copy has run (next layer)
                gp[j] = (unsigned long long) vslot[(size_t) sl];
                ++sp_used;
                pol_use(pj[j]);
                ++ts.file;
            }
            for (int j = 0; j < kK; ++j) {   // this layer's experts a prefetch was already reading
                if (pfj[j] < 0) continue;
                ++pf_used;
                prefetch_land(pfj[j], &ts.pf_wait_s);
                pol_use(pj[j]);
                const int32_t sl = vram_slot(pj[j]);
                CK(cudaMemcpyAsync(vslot[(size_t) sl], tier.stable_blob(m, hid[j]), XL.bytes, cudaMemcpyHostToDevice, xs));
                gp[j] = (unsigned long long) vslot[(size_t) sl];
                ++ts.file;
                if (excl) pending_free.push_back(pj[j]);
            }
            int nleft = 0;
            bool landed[kK] = {};
            for (int j = 0; j < kK; ++j) nleft += dslot[j] != nullptr;
            const auto t_disk = std::chrono::steady_clock::now();
            while (nleft > 0) {   // the disk reads, in the order they land
                int j = -1;
                for (int x = 0; x < kK && j < 0; ++x)
                    if (dslot[x] != nullptr && !landed[x] && reader.done(x)) j = x;
                if (j < 0) { std::this_thread::yield(); continue; }
                landed[j] = true;
                --nleft;
                const bool ok = reader.wait(j);
                if (!ok) { std::fprintf(stderr, "strata-glm: reading expert %d of layer %d failed\n", hid[j], l); std::exit(1); }
                const size_t pad = (size_t) (strata::kernels::cpu::expert_layout().blob_offset(m, hid[j]) % 4096);
                tier.demote_commit(m, hid[j], pad);
                rlru.touch(pj[j]);
                pol_use(pj[j]);
                const int32_t sl = vram_slot(pj[j]);
                CK(cudaMemcpyAsync(vslot[(size_t) sl], dslot[j] + pad, XL.bytes, cudaMemcpyHostToDevice, xs));
                gp[j] = (unsigned long long) vslot[(size_t) sl];
                ++ts.file;
                if (excl) pending_free.push_back(pj[j]);
                if (nleft == 0) ts.file_wait_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_disk).count();
            }
            // the GPU's groups: every expert but the CPU's, each writing its own row of `rows`
            unsigned long long* gq = gq_host + (size_t) m * kK;
            int32_t* ge = ge_host + (size_t) m * (kK + 1);
            int32_t ng = 0;
            for (int j = 0; j < kK; ++j) {
                bool cpu = skipj[j] || in_a[j];
                for (int c = 0; c < ncpu; ++c) cpu |= cj[c] == j;
                if (cpu) continue;
                gq[ng] = gp[j];
                ge[ng] = j;
                ++ng;
            }
            ge[kK] = ng;
            if (ng > 0) {   // the expert kernel waits for this layer's copies, nothing else does
                CK(cudaEventRecord(ev_xs, xs));
                CK(cudaStreamWaitEvent(s, ev_xs, 0));
            }
            if (pf && !pf_copied) pf_copy();   // no disk wait here: beside this layer's kernel and the next attention
            {
                float wall = 0.f, wkeep = 0.f;
                bool any = false;
                for (int j = 0; j < kK; ++j) { wall += wts_pin[j]; if (skipj[j]) any = true; else wkeep += wts_pin[j]; }
                if (any) {
                    float* nw = wts_pin + kK;
                    for (int j = 0; j < kK; ++j) nw[j] = skipj[j] ? 0.f : wts_pin[j] * (wkeep > 0 ? wall / wkeep : 0.f);
                    CK(cudaMemcpyAsync(wts, nw, kK * sizeof(float), cudaMemcpyHostToDevice, s));
                    for (int j = 0; j < kK; ++j)
                        if (skipj[j]) CK(cudaMemsetAsync(rows + (size_t) j * kEmbd, 0, kEmbd * sizeof(float), s));
                }
            }
            if (timing) CK(cudaEventRecord(tev[(size_t) m * 3 + 1], s));
            if (ng > 0)
            {
                if (fake3) glm::fake3_groups(gq, ge + kK, kK, XL.tail_off / 36, XL.tail_off, s);
                strata::kernels::native_expert_grouped_f32(XL, gq, grp_start, ge + kK, ge, ent_tok, kK, kK, xn, xscratch, rows, s);
            }
            if (ncpu > 0) {
                cpux.wait();
                for (int c = 0; c < ncpu; ++c)
                    CK(cudaMemcpyAsync(rows + (size_t) cj[c] * kEmbd, cpu_rows + (size_t) cj[c] * kEmbd,
                                       kEmbd * sizeof(float), cudaMemcpyHostToDevice, s));
            }
            combine_rows(rows, wts, kK, y, kEmbd, s);
            if (timing) CK(cudaEventRecord(tev[(size_t) m * 3 + 2], s));
            return;   // the RAM slots and cpu_rows read above are reused only after the next layer's router event
        }
        if (tiered) {
            // the tiers: a VRAM slot as it is; a RAM or disk expert copied into this layer's staging slot (the
            // disk reads were started by begin_layer and overlap the copies of the ones before them)
            note(m, hid, kK);
            if (routes) routes->push_back({l, std::vector<int32_t>(hid, hid + kK)});
            tier.begin_layer(m, hid, kK);
            if (timing) CK(cudaEventRecord(tev[(size_t) m * 3], s));
            unsigned long long* gp = gp_host + (size_t) m * kK;
            for (int j = 0; j < kK; ++j) {
                const size_t pi = (size_t) m * kNE + (size_t) hid[j];
                if (res[pi] >= 0) { gp[j] = (unsigned long long) vslot[(size_t) res[pi]]; ++ts.vram; continue; }
                const bool in_ram = tier.has_copy(m, hid[j]);
                const auto t = std::chrono::steady_clock::now();
                const uint8_t* b = tier.blob(m, hid[j]);
                if (in_ram) ++ts.ram;
                else { ++ts.file; ts.file_wait_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); }
                if (b == nullptr) { std::fprintf(stderr, "strata-glm: no bytes for expert %d of layer %d\n", hid[j], l); std::exit(1); }
                uint8_t* dst = stage + (size_t) j * XL.bytes;
                CK(cudaMemcpyAsync(dst, b, XL.bytes, cudaMemcpyHostToDevice, s));
                gp[j] = (unsigned long long) dst;
            }
            CK(cudaMemcpyAsync(grp_ptr, gp, kK * sizeof(unsigned long long), cudaMemcpyHostToDevice, s));
            if (timing) CK(cudaEventRecord(tev[(size_t) m * 3 + 1], s));
            strata::kernels::native_expert_grouped_f32(XL, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, kK, kK, xn,
                                                       xscratch, rows, s);
            combine_rows(rows, wts, kK, y, kEmbd, s);
            if (timing) CK(cudaEventRecord(tev[(size_t) m * 3 + 2], s));
            return;   // the staging slots and the disk buffers are free again once the next layer's router syncs
        }
        // no tiers: every routed expert read from experts.bin
        for (int j = 0; j < kK; ++j) {
            const long long off = ((long long) m * kNE + hid[j]) * (long long) XL.bytes;
            _fseeki64(experts, off, SEEK_SET);
            if (std::fread(host_blobs + (size_t) j * XL.bytes, 1, XL.bytes, experts) != XL.bytes) {
                std::fprintf(stderr, "strata-glm: short read of expert %d of layer %d\n", hid[j], l);
                std::exit(1);
            }
        }
        CK(cudaMemcpyAsync(slots, host_blobs, (size_t) kK * XL.bytes, cudaMemcpyHostToDevice, s));
        strata::kernels::native_expert_grouped_f32(XL, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, kK, kK, xn,
                                                   xscratch, rows, s);
        combine_rows(rows, wts, kK, y, kEmbd, s);
        CK(cudaStreamSynchronize(s));   // host_blobs is reused by the next layer
    }

    void dense_mlp(Layer& ly) {
        using namespace glm;
        gemv(ly.dg, xn, tmp1, kDenseFF, kEmbd, 1, s);
        gemv(ly.du, xn, tmp2, kDenseFF, kEmbd, 1, s);
        swiglu_clamp(tmp1, tmp2, tmp3, kDenseFF, kSwigluLimit, s);
        gemv(ly.dd, tmp3, y, kEmbd, kDenseFF, 1, s);
    }

    // one token at position pos: the next token's logits in `logits`
    void forward(int token, int pos, const std::string& dump_dir) {
        using namespace glm;
        ensure_ctx(pos + 1);
        ++heat_tokens;
        ++dec_tok;
        ic_prepare(pos, 1);
        if (lo == 0) {   // a later stage finds its input streams in `streams`
            CK(cudaMemcpyAsync(emb_row, embed.data() + (size_t) token * kEmbd, kEmbd * sizeof(bf16), cudaMemcpyHostToDevice, s));
            bf16_to_f32(emb_row, x, kEmbd, s);
            for (int j = 0; j < kHc; ++j) CK(cudaMemcpyAsync(streams + (size_t) j * kEmbd, x, kEmbd * sizeof(float), cudaMemcpyDeviceToDevice, s));
        }
        for (int l = lo; l < hi; ++l) {
            Layer& ly = L[(size_t) l];
            hc_mix(ly.hc_attn_fn, streams, mixp, 24, kHc * kEmbd, kMixParts, s);
            hc_pre_finish(streams, mixp, ly.hc_attn_base, ly.hc_attn_scale, kEmbd, kEps, kHcEps, kSinkhorn, x, post, comb, s,
                          ly.in_norm, xn, kMixParts);
            if (is_dsa(l)) dsa(ly, pos); else kda(ly);
            hc_post(y, streams, post, comb, streams, kEmbd, s);
            hc_mix(ly.hc_ffn_fn, streams, mixp, 24, kHc * kEmbd, kMixParts, s);
            hc_pre_finish(streams, mixp, ly.hc_ffn_base, ly.hc_ffn_scale, kEmbd, kEps, kHcEps, kSinkhorn, x, post, comb, s,
                          ly.post_norm, xn, kMixParts);
            predict_layer(0, l + 1);
            if (l < kDenseLead) dense_mlp(ly); else moe(ly, l);
            hc_post(y, streams, post, comb, streams, kEmbd, s);
            predict_layer(1, l + 1);
            if (!dump_dir.empty()) {
                std::vector<float> h((size_t) kHc * kEmbd);
                CK(cudaMemcpyAsync(h.data(), streams, h.size() * sizeof(float), cudaMemcpyDeviceToHost, s));
                CK(cudaStreamSynchronize(s));
                char name[64];
                std::snprintf(name, sizeof name, "/l%02d.f32", l);
                if (std::FILE* f = std::fopen((dump_dir + name).c_str(), "wb")) { std::fwrite(h.data(), 4, h.size(), f); std::fclose(f); }
            }
        }
        if (hi == kLayers) {
            hc_mean(streams, x, kEmbd, s);
            rmsnorm(x, final_norm, kEps, xn, kEmbd, 1, s);
            gemv(lm_head, xn, logits, kVocab, kEmbd, 1, s);
        }
        CK(cudaStreamSynchronize(s));
        if (xp && pfc_pending) {   // nothing of this token's speculation is in flight past it
            CK(cudaStreamSynchronize(xp));
            pfc_pending = false;
            std::fill(spec_slot.begin(), spec_slot.end(), 0);
        }
    }
};

}  // namespace

int main(int argc, char** argv) {
    std::string pack, tokens_path, dump_dir, dump_logits;
    int max_new = 16, max_ctx = 8192, chunk = 0;
    std::string routes_path, profile;
    long long vram_experts = -1;
    double ram_gib = -1, ram_reserve_gib = 24, vram_reserve_mib = 1500;   // leave RAM for Windows and the desktop
    int rebalance_every = -1;   // -1: 16 with --policy rebalance, none with lru
    std::string policy = "lru";
    std::vector<std::string> mirrors;
    double cpu_share = 0;
    int cpu_threads = 14;
    int spec = 0;             // --spec K: verify K drafted tokens a pass (0: one token at a time, as before)
    bool no_prefetch = false, dense_bf16 = false, prompt_f32 = false, ram_exclusive = false;
    double vram_static = 0.7;   // --vram-static 0: VRAM all LRU (inclusive)
    int restatic_every = 8;     // --restatic N: the static part follows the decode every N tokens (0: fixed at the prompt)
    // --skip-disk 0.1 by default: a routed expert only on the disk, not prefetched, under 10% of the routing weight
    // is left out (KL to BF16: 0.018 -> 0.017 short, 0.021 long; +13-15% tok/s); --skip-disk 0 restores exactness
    float skip_disk = 0.1f, skip_ram = 0.f;
    std::string teacher_path, step_logits_path;   // decode forced to these tokens; every step's logits written out
    int pf_copies = 2, pf_reads = 6;   // reads 6: chat_uk +1.3%, chat_code +0.7% over 4 (8: no better)
    // formats (KL to BF16 dense + FP16 latent, 48 teacher-forced steps after a 2600-token prompt, median): int8 dense
    // 0.0078 (old FP8: 0.0135); + NVFP4 KDA q/k 0.017 for 1 GB of VRAM; NVFP4 head / shared / mla add 0.006-0.012
    // each, all of them 0.041 (top-1 83%) - opt-in. --dense-fp4 none --dense-fp8 --latent-f16 restore the old ones.
    bool pf_stage_on = false, ram_lru = false, latent_i8 = true;
    std::string dense_fp4 = "kdaqk";
    bool dense_i8 = true;
    long long rebalance_vram = 32, rebalance_ram = 64;
    std::string layer_split, gpus;   // 408032gb (G4): --layer-split K1,K2,.. (first layer of each later stage), --gpus D0,D1,..
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--pack") pack = next();
        else if (a == "--tokens") tokens_path = next();
        else if (a == "--max-new") max_new = std::atoi(next().c_str());
        else if (a == "--max-context") max_ctx = std::atoi(next().c_str());
        else if (a == "--dump-dir") dump_dir = next();
        else if (a == "--dump-logits") dump_logits = next();
        else if (a == "--chunk") chunk = std::atoi(next().c_str());
        else if (a == "--routes") routes_path = next();
        else if (a == "--profile") profile = next();
        else if (a == "--vram-experts") vram_experts = std::atoll(next().c_str());
        else if (a == "--ram-gib") ram_gib = std::atof(next().c_str());
        else if (a == "--ram-reserve-gib") ram_reserve_gib = std::atof(next().c_str());
        else if (a == "--vram-reserve-mib") vram_reserve_mib = std::atof(next().c_str());
        else if (a == "--rebalance-every") rebalance_every = std::atoi(next().c_str());
        else if (a == "--policy") policy = next();
        else if (a == "--mirror") mirrors.push_back(next());
        else if (a == "--cpu-share") cpu_share = std::atof(next().c_str());
        else if (a == "--cpu-threads") cpu_threads = std::atoi(next().c_str());
        else if (a == "--spec") spec = std::atoi(next().c_str());
        else if (a == "--no-prefetch") no_prefetch = true;
        else if (a == "--pf-stage") pf_stage_on = true;
        else if (a == "--ram-lru") ram_lru = true;
        else if (a == "--latent-i8") latent_i8 = true;
        else if (a == "--latent-f16") latent_i8 = false;
        else if (a == "--dense-fp4" && i + 1 < argc) dense_fp4 = argv[++i];
        else if (a == "--dense-i8") dense_i8 = true;
        else if (a == "--dense-fp8") dense_i8 = false;
        else if (a == "--pf-copies") pf_copies = std::atoi(next().c_str());
        else if (a == "--pf-reads") pf_reads = std::atoi(next().c_str());
        else if (a == "--dense-bf16") dense_bf16 = true;
        else if (a == "--prompt-f32") prompt_f32 = true;
        else if (a == "--ram-exclusive") ram_exclusive = true;
        else if (a == "--vram-static") vram_static = std::atof(next().c_str());
        else if (a == "--restatic") restatic_every = std::atoi(next().c_str());
        else if (a == "--skip-disk") skip_disk = (float) std::atof(next().c_str());
        else if (a == "--skip-ram") skip_ram = (float) std::atof(next().c_str());
        else if (a == "--teacher") teacher_path = next();
        else if (a == "--dump-step-logits") step_logits_path = next();
        else if (a == "--rebalance-moves") { rebalance_vram = std::atoll(next().c_str()); rebalance_ram = 2 * rebalance_vram; }
        else if (a == "--layer-split") layer_split = next();
        else if (a == "--gpus") gpus = next();
        else { std::fprintf(stderr, "strata-glm: unknown argument %s\n", a.c_str()); return 2; }
    }
    if (pack.empty() || tokens_path.empty()) {
        std::fprintf(stderr, "usage: strata-glm --pack DIR --tokens FILE [--max-new N] [--max-context C] [--chunk T]\n"
                             "                  [--profile P [--vram-experts N] [--ram-gib G] [--ram-reserve-gib R] [--vram-reserve-mib M]\n"
                             "                   [--policy lru|rebalance] [--rebalance-every N] [--rebalance-moves M]]\n"
                             "                  [--dump-dir D] [--dump-logits F] [--routes F]\n"
                             "                  [--dense-fp4 GROUPS|none] [--dense-fp8] [--latent-f16] [--vram-static F]\n"
                             "                  [--spec K: greedy speculative decoding, K drafted tokens a pass]\n");
        return 2;
    }
    std::vector<int> prompt;
    {
        std::ifstream in(tokens_path);
        std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        for (char& c : all) if (c == ',') c = ' ';
        std::istringstream ss(all);
        int t;
        while (ss >> t) prompt.push_back(t);
    }
    if (prompt.empty()) { std::fprintf(stderr, "strata-glm: no tokens in %s\n", tokens_path.c_str()); return 2; }

    if (spec < 0 || spec > 15) { std::fprintf(stderr, "strata-glm: --spec is 0..15\n"); return 2; }
    if (spec > 0 && chunk <= 0) { std::fprintf(stderr, "strata-glm: --spec needs --chunk (the prompt path verifies)\n"); return 2; }
    if (spec > 0 && !teacher_path.empty()) { std::fprintf(stderr, "strata-glm: --spec does not take --teacher\n"); return 2; }
    // without the tiers a chunk re-reads all 288 experts of a layer, so a pass over T of them saves nothing
    if (spec > 0 && profile.empty()) { std::fprintf(stderr, "strata-glm: --spec needs --profile\n"); return 2; }
    if (policy != "lru" && policy != "rebalance") { std::fprintf(stderr, "strata-glm: --policy is lru or rebalance\n"); return 2; }
    if (rebalance_every < 0) rebalance_every = policy == "lru" ? 0 : 16;
    // 408032gb (G4): the layers [0, kLayers) split into stages, each its own Engine on its own GPU (pipeline).
    // One stage (the default) is the original engine. --gpus 0,0,.. (one GPU for every stage) is the bit-exact check.
    std::vector<int> cuts, devs;
    auto ints = [](const std::string& t, std::vector<int>& v) {
        std::stringstream ss(t);
        for (std::string c; std::getline(ss, c, ',');) if (!c.empty()) v.push_back(std::atoi(c.c_str()));
    };
    ints(layer_split, cuts);
    ints(gpus, devs);
    const int n_st = (int) cuts.size() + 1;
    for (int k = 0; k < (int) cuts.size(); ++k)
        if (cuts[(size_t) k] <= (k ? cuts[(size_t) k - 1] : 0) || cuts[(size_t) k] >= kLayers) {
            std::fprintf(stderr, "strata-glm: --layer-split wants increasing layers in 1..%d\n", kLayers - 1);
            return 2;
        }
    if (devs.empty()) { int d = 0; CK(cudaGetDevice(&d)); devs.assign((size_t) n_st, d); }
    if ((int) devs.size() != n_st) { std::fprintf(stderr, "strata-glm: --gpus needs %d devices for %d stages\n", n_st, n_st); return 2; }
    double ram_gib_stage = ram_gib;   // each stage pins its own RAM tier: split the budget instead of the first taking it all
    if (n_st > 1 && ram_gib < 0) {
        const double avail = strata::core::available_ram_bytes() / 1073741824.0;
        ram_gib_stage = std::max(0.0, (avail - ram_reserve_gib) / n_st);
    } else if (n_st > 1) {
        ram_gib_stage = ram_gib / n_st;
    }
    std::vector<Engine::Route> routes;
    std::vector<std::unique_ptr<Engine>> E;
    for (int si = 0; si < n_st; ++si) {
        CK(cudaSetDevice(devs[(size_t) si]));
        E.push_back(std::make_unique<Engine>());
        Engine& e = *E.back();
        e.lo = si == 0 ? 0 : cuts[(size_t) si - 1];
        e.hi = si + 1 < n_st ? cuts[(size_t) si] : kLayers;
        if (n_st > 1) std::fprintf(stderr, "strata-glm: stage %d: layers %d-%d on GPU %d\n", si, e.lo, e.hi - 1, devs[(size_t) si]);
        e.max_ctx = max_ctx;
        e.vreserve_mib = vram_reserve_mib;
        e.dense_fp8 = !dense_bf16;
        e.chunk_cap = chunk;
        e.logits_cap = spec > 0 ? spec + 1 : 1;   // --spec: the head's rows, one per position of a pass
        e.latent_i8 = latent_i8;
        e.dense_i8 = dense_i8;
        // a comma list: all, attn (= kda + mla), kda (= kdaqk + kdavo: KDA's q/k and v/o projections), mla (MLA and the
        // indexer's query), shared, head; none
        for (std::stringstream ss(dense_fp4); ss.good();) {
            std::string c;
            std::getline(ss, c, ',');
            const bool all = c == "all", attn = all || c == "attn", kda = attn || c == "kda";
            e.fp4_qk |= kda || c == "kdaqk";
            e.fp4_vo |= kda || c == "kdavo";
            e.fp4_mla |= attn || c == "mla";
            e.fp4_shared |= all || c == "shared";
            e.fp4_head |= all || c == "head";
        }
        e.lru = policy == "lru";
        e.excl = ram_exclusive;
        e.vram_static = vram_static;
        e.skip_disk = skip_disk;
        e.skip_ram = skip_ram;
        e.mirrors = mirrors;
        e.cpu_share = cpu_share;
        e.cpu_threads = cpu_threads;
        e.prefetch = !no_prefetch;
        e.pf_copy_max = std::min(pf_copies, kK);
        e.pf_stage = pf_stage_on;
        e.ram_freq = !ram_lru;
        e.pf_read_max = std::min(pf_reads, Engine::kPred);
        e.load(pack);
        if (!routes_path.empty()) e.routes = &routes;   // every stage appends its layers' records
        if (chunk > 0) e.init_chunk(chunk, profile.empty());
        if (!profile.empty()) e.init_tier(pack, profile, vram_experts, ram_gib_stage, ram_reserve_gib, vram_reserve_mib);
        if (chunk > 0 && e.tiered) {
            e.init_prompt_pipe(chunk);
            if (!prompt_f32) e.init_prompt_mmq(chunk);
            else e.prompt_mmq = false;
        }
        e.timing_init();
        e.predict_init();
    }
    Engine& e = *E.back();   // the stage with the head: logits and the summary below
    auto on = [&](int si) { CK(cudaSetDevice(devs[(size_t) si])); return std::ref(*E[(size_t) si]); };
    // one prompt chunk / one token through every stage; the residual streams move on between stages
    auto run_at = [&](const int* toks, int T, int p, const std::string& dd = std::string(), int nlog = 1) {
        for (int si = 0; si < n_st; ++si) {
            Engine& st = on(si);
            if (si > 0)
                CK(cudaMemcpyPeer(st.c_streams, devs[(size_t) si], E[(size_t) si - 1]->c_streams, devs[(size_t) si - 1],
                                  (size_t) T * kHc * kEmbd * sizeof(float)));
            st.forward_chunk(toks, T, p, dd, nlog);
        }
    };
    auto run_chunk = [&](size_t i, int T, const std::string& dd) { run_at(prompt.data() + i, T, (int) i, dd); };
    auto run_token = [&](int tok, int p, const std::string& dd) {
        for (int si = 0; si < n_st; ++si) {
            Engine& st = on(si);
            if (si > 0)
                CK(cudaMemcpyPeer(st.streams, devs[(size_t) si], E[(size_t) si - 1]->streams, devs[(size_t) si - 1],
                                  (size_t) kHc * kEmbd * sizeof(float)));
            st.forward(tok, p, dd);
        }
    };
    const char* nsys_env = std::getenv("GLM_NSYS");   // 1: decode, 2: the prompt (nsys --capture-range=cudaProfilerApi)
    if (nsys_env && std::atoi(nsys_env) == 2) CK(cudaProfilerStart());
    const auto t0 = std::chrono::steady_clock::now();
    if (chunk > 0) {
        for (size_t i = 0; i < prompt.size(); i += (size_t) chunk) {
            const int T = (int) std::min<size_t>((size_t) chunk, prompt.size() - i);
            run_chunk(i, T, i + T == prompt.size() ? dump_dir : std::string());
            std::fprintf(stderr, "strata-glm: %zu of %zu prompt tokens\n", i + T, prompt.size());
        }
    } else {
        for (size_t i = 0; i < prompt.size(); ++i)
            run_token(prompt[i], (int) i, i + 1 == prompt.size() ? dump_dir : std::string());
    }
    const double tp = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (nsys_env && std::atoi(nsys_env) == 2) CK(cudaProfilerStop());
    for (int si = 0; si < n_st; ++si) {
        Engine& st = on(si);
        if (st.tiered && chunk > 0) {
            st.release_prompt();
            st.grow_vram_tier(vram_reserve_mib);
        }
        if (st.tiered) {   // the prompt said which experts this conversation uses: the tiers follow it
            const double ps = (double) (st.ts.vram + st.ts.ram + st.ts.file);
            if (ps > 0)
                std::fprintf(stderr, "strata-glm: %sprompt experts: %.1f%% VRAM, %.1f%% RAM, %.1f%% disk\n",
                             n_st > 1 ? ("stage " + std::to_string(si) + ": ").c_str() : "", 100 * st.ts.vram / ps,
                             100 * st.ts.ram / ps, 100 * st.ts.file / ps);
            st.rebalance(1LL << 40, 1LL << 40, true);
        }
    }
    // the chunk path again for the verification passes: the prompt's buffers are released, the tiers have
    // settled, and a pass needs its own small set
    if (spec > 0)
        for (int si = 0; si < n_st; ++si) { Engine& st = on(si); st.init_spec(spec + 1); }
    on(n_st - 1);
    std::vector<float> lg(kVocab);
    CK(cudaMemcpy(lg.data(), e.logits, lg.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (!dump_logits.empty())
        if (std::FILE* f = std::fopen(dump_logits.c_str(), "wb")) { std::fwrite(lg.data(), 4, lg.size(), f); std::fclose(f); }
    std::fprintf(stderr, "strata-glm: prompt %zu tokens in %.1f s (%.2f tok/s)\n", prompt.size(), tp, prompt.size() / tp);
    std::vector<int> out;
    int pos = (int) prompt.size(), steps = 0;
    const Engine::TierStats ts0 = e.ts;
    std::vector<Engine::TierStats> ts0s;
    for (const auto& st : E) ts0s.push_back(st->ts);
    const bool nsys = nsys_env && std::atoi(nsys_env) == 1;
    if (nsys) CK(cudaProfilerStart());
    const auto t1 = std::chrono::steady_clock::now();
    std::vector<int> teacher;
    if (!teacher_path.empty()) {
        std::ifstream in(teacher_path);
        int t;
        while (in >> t) teacher.push_back(t);
    }
    std::FILE* step_f = step_logits_path.empty() ? nullptr : std::fopen(step_logits_path.c_str(), "wb");
    // --spec K: greedy speculative decoding. The drafter is a prompt lookup over `seq` (the prompt plus what has
    // been generated); a pass runs [the token just taken, the drafts] through the prompt path at the current
    // position and reads the head at every row, so each position's token is the model's own argmax there: the
    // drafts only choose which positions are computed together, never what comes out. `lg` holds the next
    // position's logits exactly as in a plain decode step, and the token after the last accepted one - the
    // bonus - is its argmax. A pass is the prompt path (FP32 experts, the CPU pool) rather than the decode
    // path, so its numbers differ from a step's by that much; being the only source of these tokens' output,
    // they are self-consistent, which is all greedy speculative decoding asks of a verifier.
    const auto eos_id = [](int t) { return t == 154820 || t == 154827 || t == 154829; };   // eos ids (config.json)
    const auto argmax = [](const float* p, size_t n) { return (int) (std::max_element(p, p + n) - p); };
    std::vector<int> seq = prompt, drafts;
    std::vector<float> lg_v;
    if (spec > 0) lg_v.resize((size_t) (spec + 1) * kVocab);
    long long sp_drafted = 0, sp_accepted = 0, sp_full = 0, sp_passes = 0;
    double sp_time = 0, sp_step_time = 0;
    long long sp_step_n = 0, sp_probe = 8, sp_mark = 0;
    bool sp_eos = false, sp_off = false;
    const auto spec_pass = [&]() -> int {   // the drafts accepted; -1: nothing to draft, take a plain step
        drafts = draft_ngram(seq, spec);
        if (drafts.empty() || pos + (int) drafts.size() + 1 > max_ctx) return -1;
        const int T = (int) drafts.size() + 1;
        std::vector<int> vt;
        vt.reserve((size_t) T);
        vt.push_back(out.back());
        vt.insert(vt.end(), drafts.begin(), drafts.end());
        for (int si = 0; si < n_st; ++si) { Engine& st = on(si); st.spec_verify = true; st.spec_T = T; }
        run_at(vt.data(), T, pos, std::string(), T);
        CK(cudaMemcpy(lg_v.data(), e.logits, (size_t) T * kVocab * sizeof(float), cudaMemcpyDeviceToHost));
        int a = 0;   // row t of the pass's logits judges draft t: it is the logits of the token before it
        while (a < (int) drafts.size() && argmax(lg_v.data() + (size_t) a * kVocab, kVocab) == drafts[(size_t) a]) ++a;
        // the KDA states back to the accepted prefix: what the pass advanced past it is discarded, and the next
        // pass rewrites those rows
        for (int si = 0; si < n_st; ++si) { Engine& st = on(si); st.spec_fix(a + 1); st.spec_verify = false; }
        for (int i = 0; i < a; ++i) {
            if ((int) out.size() >= max_new) break;
            out.push_back(drafts[(size_t) i]);
            seq.push_back(drafts[(size_t) i]);
            if (eos_id(drafts[(size_t) i])) { sp_eos = true; break; }
        }
        std::memcpy(lg.data(), lg_v.data() + (size_t) a * kVocab, lg.size() * sizeof(float));
        pos += a + 1;
        sp_drafted += (long long) drafts.size();
        sp_accepted += a;
        ++sp_passes;
        if (a == (int) drafts.size()) ++sp_full;
        return a;
    };
    for (int n = 0; (int) out.size() < max_new && pos < max_ctx; ++n) {
        const size_t n_before = out.size();
        if (step_f) std::fwrite(lg.data(), sizeof(float), lg.size(), step_f);
        if (!teacher.empty() && n >= (int) teacher.size()) break;
        const int tok = teacher.empty() ? (int) (std::max_element(lg.begin(), lg.end()) - lg.begin()) : teacher[(size_t) n];
        out.push_back(tok);
        if (spec > 0) seq.push_back(tok);
        if (eos_id(tok)) break;
        const auto tf = std::chrono::steady_clock::now();
        // A pass is only worth its cost while it lands more tokens than it spends in plain steps: both sides are
        // measured here, so the flag decides with this machine's own numbers rather than a constant. Every
        // sp_probe passes one position goes through a plain step - it is the same token a pass with no drafts
        // would take - until the plain side has enough samples to be believed.
        bool plain = spec == 0 || sp_off;
        if (!plain && sp_step_n < 6 && sp_passes - sp_mark >= sp_probe) { plain = true; sp_mark = sp_passes; }
        if (!plain && spec_pass() < 0) plain = true;   // nothing to draft from: a plain step
        if (plain) run_token(tok, pos++, std::string());
        // what the iteration put in the cache - a pass cut short by --max-new counts what it actually emitted
        const int adv = std::max(1, (int) out.size() - (int) n_before);
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - tf).count();
        if (plain) { e.timing_token(dt); sp_step_time += dt; ++sp_step_n; }
        else sp_time += dt;
        if (!sp_off && sp_passes >= 6 && sp_step_n >= 2) {
            const double per_pass = sp_time / (double) sp_passes, per_step = sp_step_time / (double) sp_step_n;
            if ((double) sp_accepted / (double) sp_passes + 1.0 <= per_pass / per_step) {
                sp_off = true;
                std::fprintf(stderr, "strata-glm: spec %d off after %lld passes: %.2f tokens a pass for %.1f ms, "
                                     "a plain step is %.1f ms\n", spec, sp_passes,
                             (double) sp_accepted / (double) sp_passes + 1.0, 1e3 * per_pass, 1e3 * per_step);
            }
        }
        steps += adv;
        for (int si = 0; si < n_st; ++si) {
            Engine& st = on(si);
            if (st.tiered && rebalance_every > 0 && steps % rebalance_every == 0) st.rebalance(rebalance_vram, rebalance_ram, false);
            if (st.tiered && restatic_every > 0 && steps % restatic_every == 0) st.restatic(16, 1.5);
        }
        on(n_st - 1);
        if (steps % 32 == 0) {   // the machine is also a desktop: say so when RAM runs low
            const double avail = strata::core::available_ram_bytes() / 1073741824.0;
            if (avail > 0 && avail < 8) std::fprintf(stderr, "strata-glm: only %.1f GiB of RAM left for Windows\n", avail);
        }
        if (sp_eos) break;
        if (plain) CK(cudaMemcpy(lg.data(), e.logits, lg.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }
    const double td = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    if (nsys) CK(cudaProfilerStop());
    if (step_f) std::fclose(step_f);
    if (e.skipped > 0 && steps > 0) std::fprintf(stderr, "strata-glm: skipped %.1f experts outside VRAM a token\n", (double) e.skipped / steps);
    if (!routes_path.empty()) {   // layer, then T*K ids per record (the prompt's chunks, then each decode step)
        if (std::FILE* f = std::fopen(routes_path.c_str(), "ab")) {
            for (const auto& r : routes) {
                const int32_t hdr[2] = {r.layer, (int32_t) r.ids.size()};
                std::fwrite(hdr, sizeof hdr, 1, f);
                std::fwrite(r.ids.data(), sizeof(int32_t), r.ids.size(), f);
            }
            std::fclose(f);
        }
    }
    std::printf("output :");
    for (int t : out) std::printf(" %d", t);
    std::printf("\n");
    std::fprintf(stderr, "strata-glm: decode %d steps in %.1f s (%.2f tok/s)\n", steps, td, steps / td);
    if (spec > 0)   // a pass ends with one token of the verifier's own argmax: the bonus
        std::fprintf(stderr, "strata-glm: spec %d: %lld passes, drafted %lld, accepted %lld (%.1f%%), bonus %lld, "
                             "full accepts %lld (%.1f%%)\n", spec, sp_passes, sp_drafted, sp_accepted,
                     sp_drafted ? 100.0 * (double) sp_accepted / (double) sp_drafted : 0.0, sp_passes, sp_full,
                     sp_passes ? 100.0 * (double) sp_full / (double) sp_passes : 0.0);
    if (n_st > 1 && steps > 0)
        for (int si = 0; si < n_st; ++si) {
            const Engine& st = *E[(size_t) si];
            const Engine::TierStats& b0 = ts0s[(size_t) si];
            const double v = (double) (st.ts.vram - b0.vram), r = (double) (st.ts.ram - b0.ram), f = (double) (st.ts.file - b0.file);
            const double c = (double) (st.ts.cpu - b0.cpu), n = std::max(1.0, v + r + f + c);
            std::fprintf(stderr, "strata-glm: stage %d (layers %d-%d, GPU %d) decode experts: %.1f%% VRAM, %.1f%% RAM to the GPU, "
                                 "%.1f%% RAM on the CPU, %.1f%% disk\n", si, st.lo, st.hi - 1, devs[(size_t) si], 100 * v / n,
                         100 * r / n, 100 * c / n, 100 * f / n);
        }
    if (e.tiered && steps > 0) {
        const double v = (double) (e.ts.vram - ts0.vram), r = (double) (e.ts.ram - ts0.ram), f = (double) (e.ts.file - ts0.file);
        const double c = (double) (e.ts.cpu - ts0.cpu), n = v + r + f + c;
        std::fprintf(stderr, "strata-glm: decode experts: %.1f%% VRAM, %.1f%% RAM to the GPU, %.1f%% RAM on the CPU, %.1f%% "
                             "disk (%.1f a token, %.1f ms a token waiting for the disk", 100 * v / n, 100 * r / n, 100 * c / n,
                     100 * f / n, f / steps, 1e3 * (e.ts.file_wait_s - ts0.file_wait_s + e.ts.pf_wait_s - ts0.pf_wait_s) / steps);
        for (int d = 0; d < e.reader.drives(); ++d) std::fprintf(stderr, "%s drive %d: %lld", d ? "," : ";", d, e.reader.reads(d));
        std::fprintf(stderr, ")\n");
    }
    if (e.tiered && steps > 0)
        std::fprintf(stderr, "strata-glm: disk waits a token: %.1f ms for unpredicted experts, %.1f ms for prefetches not landed yet "
                             "(%lld prefetch reads used)\n", 1e3 * (e.ts.file_wait_s - ts0.file_wait_s) / steps,
                     1e3 * (e.ts.pf_wait_s - ts0.pf_wait_s) / steps, e.pf_used);
    if (e.excl && steps > 0) std::fprintf(stderr, "strata-glm: exclusive tiers: %.1f writebacks a token\n", (double) e.writebacks / steps);
    if (e.prefetch && e.pf_reads > 0)
        std::fprintf(stderr, "strata-glm: prefetch: %lld disk reads for the next layer, %.1f%% of them routed there; "
                             "%lld copies RAM -> VRAM, %.1f%% routed there\n", e.pf_reads, 100.0 * e.pf_used / e.pf_reads,
                     e.pf_copies, 100.0 * e.pf_copies_used / std::max(1LL, e.pf_copies));
    if (e.sp_reads > 0)
        std::fprintf(stderr, "strata-glm: staged prefetch: %lld reads, %.1f%% of them routed there\n", e.sp_reads,
                     100.0 * e.sp_used / e.sp_reads);
    if (e.prefetch && e.pf_copies > 0) {
        std::fprintf(stderr, "strata-glm: prefetch copies routed there, by predicted rank:");
        for (int i = 0; i < kK; ++i)
            if (e.pf_rank_n[i]) std::fprintf(stderr, " %d: %.0f%% (%lld)", i, 100.0 * e.pf_rank_used[i] / e.pf_rank_n[i], e.pf_rank_n[i]);
        std::fprintf(stderr, "\n");
    }
    if (e.predict)
        for (int v = 0; v < 2; ++v)
            std::fprintf(stderr, "strata-glm: prediction %s: %.1f%% of the routed experts, %.1f%% of those outside VRAM\n",
                         v ? "(B) after layer l" : "(A) at layer l's router", 100.0 * e.pred_hit[v] / std::max(1LL, e.pred_n[v]),
                         100.0 * e.pred_miss_hit[v] / std::max(1LL, e.pred_miss_n[v]));
    {
        long long tot = 0;
        for (long long v : e.miss_rank) tot += v;
        if (tot > 0) {
            auto pct = [&](int a0, int a1) { long long n = 0; for (int i = a0; i < a1; ++i) n += e.miss_rank[i]; return 100.0 * n / tot; };
            std::fprintf(stderr, "strata-glm: disk reads waited for (%lld), by predicted rank: 0-5 %.0f%%, 6-7 %.0f%%, 8-11 %.0f%%, "
                                 "12-15 %.0f%%, not in the top 16 %.0f%%\n", tot, pct(0, 6), pct(6, 8), pct(8, 12), pct(12, 16),
                         pct(16, 17));
        }
    }
    if (spec > 0 && sp_passes > 0)   // a pass ends with one token of the verifier's own argmax: the bonus
        std::fprintf(stderr, "strata-glm: an accepted token %.1f ms (%lld passes, %lld tokens)\n",
                     1e3 * sp_time / (double) (sp_accepted + sp_passes), sp_passes, sp_accepted + sp_passes);
    if (e.timing && e.tm.tokens > 0) {
        const double n = (double) e.tm.tokens, tok = 1e3 * e.tm.tok_s / n;
        std::fprintf(stderr, "strata-glm: a decode token %.1f ms: expert copies (with the disk waits) %.1f, expert kernels "
                             "%.1f, the rest (dense layers, routers, syncs) %.1f\n", tok, e.tm.copy_ms / n,
                     e.tm.kernel_ms / n, tok - (e.tm.copy_ms + e.tm.kernel_ms) / n);
    }
    return 0;
}
