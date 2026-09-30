// src/glm/glm_main.cpp - strata-glm, M1: GLM-5.3-Flash NVFP4 decoded one token at a time, correct before fast.
//
//   strata-glm --pack <dir from tools/glm_pack.py> --tokens <ids> [--max-new N] [--max-context C]
//              [--dump-dir d] [--dump-logits f]
//
// Dense weights are read from the checkpoint's safetensors (the pack's dense.txt says where) into VRAM as BF16
// (the leading dense MLP layers decoded from NVFP4 to FP32). Each MoE layer's 8 routed experts are read from the
// pack's experts.bin and computed by Strata's NVFP4 expert kernel with FP32 activations (native_expert_grouped_f32).
// --dump-dir writes, at the last prompt token, every layer's output streams (l%02d.f32) - what tools/glm_ref.py
// --dump-dir writes - and --dump-logits the next token's logits.
#include "glm_gemm.cuh"
#include "glm_kernels.cuh"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ---- GLM-5.3-Flash (config.json; checked against it at start)
constexpr int kLayers = 45, kDenseLead = 3, kEmbd = 4096, kHc = 4, kVocab = 154880;
constexpr int kKdaH = 64, kKdaD = 128, kKdaC = kKdaH * kKdaD;
constexpr int kMlaH = 64, kQLora = 1536, kR = 512, kDk = 256, kDv = 256;
constexpr int kIdxH = 32, kIdxD = 128, kKpool = 4, kIdxTopk = 2048;
constexpr int kNE = 288, kK = 8, kFF = 2048, kDenseFF = 12288;
constexpr float kEps = 1e-5f, kHcEps = 1e-6f, kLowerBound = -5.0f, kRouteScale = 2.5f, kSwigluLimit = 10.0f;
constexpr int kSinkhorn = 20;
bool is_dsa(int l) { return l % 4 == 3; }   // layers 3, 7, ..., 43 (config: full_attn_layers)

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
    glm::bf16 *wq, *wk, *wv, *fa, *fb, *bproj, *ga, *gb, *onorm, *wo;
    float *q_conv, *k_conv, *v_conv, *dt_bias, *A_log;
    // DSA
    glm::bf16 *qa, *qa_norm, *qb, *kva, *kva_norm, *kvb, *iwqb, *iwk, *ik_w, *ik_b, *iwp, *igate, *iape;
    // MLP
    float *dg = nullptr, *du = nullptr, *dd = nullptr;           // dense (layers 0-2), FP32
    glm::bf16 *router = nullptr, *sg = nullptr, *su = nullptr, *sd = nullptr;
    float* router_bias = nullptr;
    // state
    float *S = nullptr, *conv = nullptr;                           // KDA: [H][dh][dh], [3][C][3]
    float *lat = nullptr, *ikc = nullptr, *igc = nullptr, *pooled = nullptr;   // DSA caches
};

struct Engine {
    Checkpoint ck;
    std::vector<Layer> L;
    glm::bf16 *final_norm = nullptr, *lm_head = nullptr;
    std::vector<glm::bf16> embed;                                   // host, BF16 [vocab][embd]
    int max_ctx = 8192;
    // buffers
    float *streams, *mix, *x, *xn, *post, *comb, *y, *tmp1, *tmp2, *tmp3, *q, *k, *v, *gf, *b, *o, *gate;
    float *q_resid, *qm, *qa, *ctx, *vo, *iq, *ik, *ig, *iw, *score, *logits, *rows, *wts;
    int32_t *ids, *sel;
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

    void init_chunk(int T) {
        chunk = T;
        gm.init(s, (size_t) T * kMlaH * kR);   // the widest split: the MLA context rows (64 x 512)
        auto buf = [&](size_t n) { float* p = nullptr; CK(cudaMalloc(&p, n * sizeof(float))); return p; };
        c_streams = buf((size_t) T * kHc * kEmbd); c_mix = buf((size_t) T * 24); c_x = buf((size_t) T * kEmbd);
        c_xn = buf((size_t) T * kEmbd); c_y = buf((size_t) T * kEmbd); c_post = buf((size_t) T * 4); c_comb = buf((size_t) T * 16);
        c_t1 = buf((size_t) T * kDenseFF); c_t2 = buf((size_t) T * kDenseFF); c_t3 = buf((size_t) T * kDenseFF);
        c_q = buf((size_t) T * kKdaC); c_k = buf((size_t) T * kKdaC); c_v = buf((size_t) T * kKdaC); c_gf = buf((size_t) T * kKdaC);
        c_b = buf((size_t) T * kKdaH); c_o = buf((size_t) T * kKdaC); c_gate = buf((size_t) T * kKdaC);
        c_qr = buf((size_t) T * kQLora); c_qm = buf((size_t) T * kMlaH * kDk); c_qa = buf((size_t) T * kMlaH * kR);
        c_ctx = buf((size_t) T * kMlaH * kR); c_vo = buf((size_t) T * kMlaH * kDv); c_iq = buf((size_t) T * kIdxH * kIdxD);
        c_iw = buf((size_t) T * kIdxH); c_score = buf((size_t) T * (max_ctx / kKpool + 1));
        c_rows = buf((size_t) T * kK * kEmbd); c_wts = buf((size_t) T * kK);
        CK(cudaMalloc(&c_ids, (size_t) T * kK * sizeof(int32_t)));
        CK(cudaMalloc(&c_sel, (size_t) T * kSelLd * sizeof(int32_t)));
        CK(cudaMalloc(&c_cnt, (size_t) T * sizeof(int32_t)));
        CK(cudaMalloc(&c_grp_ptr, kNE * sizeof(unsigned long long)));
        CK(cudaMalloc(&c_grp_start, (kNE + 1) * sizeof(int32_t)));
        CK(cudaMalloc(&c_ngroups, sizeof(int32_t)));
        CK(cudaMalloc(&c_ent_dst, (size_t) T * kK * sizeof(int32_t)));
        CK(cudaMalloc(&c_ent_tok, (size_t) T * kK * sizeof(int32_t)));
        CK(cudaMalloc(&c_emb, (size_t) T * kEmbd * sizeof(glm::bf16)));
        CK(cudaMalloc(&layer_slots, (size_t) kNE * XL.bytes));
        CK(cudaMallocHost(&layer_host, (size_t) kNE * XL.bytes));
        CK(cudaMalloc(&c_xscratch, strata::kernels::native_expert_scratch_bytes((int64_t) T * kK, kFF)));
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        std::fprintf(stderr, "strata-glm: prompt path up to %d tokens a chunk; %.1f GiB of VRAM free\n", T, fr / 1073741824.0);
    }

    void kda_chunk(Layer& ly, int T) {
        using namespace glm;
        gm.w16(ly.wq, c_xn, c_q, kKdaC, kEmbd, T);
        gm.w16(ly.wk, c_xn, c_k, kKdaC, kEmbd, T);
        gm.w16(ly.wv, c_xn, c_v, kKdaC, kEmbd, T);
        conv_silu_seq(c_q, ly.q_conv, ly.conv, c_q, kKdaC, T, s);
        conv_silu_seq(c_k, ly.k_conv, ly.conv + (size_t) kKdaC * 3, c_k, kKdaC, T, s);
        conv_silu_seq(c_v, ly.v_conv, ly.conv + (size_t) 2 * kKdaC * 3, c_v, kKdaC, T, s);
        gm.w16(ly.fa, c_xn, c_t1, kKdaD, kEmbd, T);
        gm.w16(ly.fb, c_t1, c_gf, kKdaC, kKdaD, T);
        gm.w16(ly.bproj, c_xn, c_b, kKdaH, kEmbd, T);
        kda_prep_rows(c_q, c_k, c_gf, ly.dt_bias, ly.A_log, kLowerBound, c_b, kKdaH, kKdaD, T, s);
        kda_scan(ly.S, c_q, c_k, c_v, c_gf, c_b, c_o, kKdaH, kKdaD, T, s);
        gm.w16(ly.ga, c_xn, c_t1, kKdaD, kEmbd, T);
        gm.w16(ly.gb, c_t1, c_gate, kKdaC, kKdaD, T);
        kda_out_norm_rows(c_o, ly.onorm, c_gate, kEps, kKdaH, kKdaD, T, s);
        gm.w16(ly.wo, c_o, c_y, kEmbd, kKdaC, T);
    }

    void dsa_chunk(Layer& ly, int T, int pos0) {
        using namespace glm;
        gm.w16(ly.qa, c_xn, c_t1, kQLora, kEmbd, T);
        rmsnorm(c_t1, ly.qa_norm, kEps, c_qr, kQLora, T, s);
        gm.w16(ly.qb, c_qr, c_qm, kMlaH * kDk, kQLora, T);
        gm.w16(ly.kva, c_xn, c_t1, kR, kEmbd, T);
        rmsnorm(c_t1, ly.kva_norm, kEps, ly.lat + (size_t) pos0 * kR, kR, T, s);
        gm.w16(ly.iwqb, c_qr, c_iq, kIdxH * kIdxD, kQLora, T);
        gm.w16(ly.iwk, c_xn, c_t1, kIdxD, kEmbd, T);
        layernorm_rows(c_t1, ly.ik_w, ly.ik_b, 1e-6f, ly.ikc + (size_t) pos0 * kIdxD, kIdxD, T, s);
        gm.w16(ly.igate, c_xn, ly.igc + (size_t) pos0 * kIdxD, kIdxD, kEmbd, T);
        const int pool_lo = pos0 / kKpool, pool_hi = (pos0 + T) / kKpool;   // the pools this chunk completes
        idx_pool_rows(ly.ikc, ly.igc, ly.iape, ly.pooled, pool_lo, pool_hi - pool_lo, kKpool, kIdxD, s);
        const int budget = kIdxTopk / kKpool;
        const int32_t* sel = nullptr;
        int max_sel = pos0 + T;
        static const bool dense = std::getenv("GLM_DENSE") != nullptr;   // tests: every visible token
        if (pool_hi > budget && !dense) {
            // some query sees more complete pools than the budget: the top ones by the indexer's score, then its tail
            gm.w16(ly.iwp, c_xn, c_iw, kIdxH, kEmbd, T);
            idx_scores_rows(c_iq, c_iw, ly.pooled, c_score, pool_hi, kIdxH, kIdxD, pos0, kKpool, T, s);
            std::vector<float> hs((size_t) T * pool_hi);
            CK(cudaMemcpyAsync(hs.data(), c_score, hs.size() * sizeof(float), cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            std::vector<int32_t> hsel((size_t) T * kSelLd), hcnt((size_t) T);
            std::vector<int> order;
            max_sel = 0;
            for (int t = 0; t < T; ++t) {
                const int pos = pos0 + t, vp = (pos + 1) / kKpool;
                int32_t* row = hsel.data() + (size_t) t * kSelLd;
                int n = 0;
                if (vp <= budget) {
                    for (int q = 0; q < vp * kKpool; ++q) row[n++] = q;
                } else {
                    order.resize((size_t) vp);
                    for (int i = 0; i < vp; ++i) order[(size_t) i] = i;
                    const float* sc = hs.data() + (size_t) t * pool_hi;
                    std::partial_sort(order.begin(), order.begin() + budget, order.end(),
                                      [&](int a, int c) { return sc[a] > sc[c]; });
                    for (int i = 0; i < budget; ++i)
                        for (int j = 0; j < kKpool; ++j) row[n++] = order[(size_t) i] * kKpool + j;
                }
                for (int q = vp * kKpool; q <= pos; ++q) row[n++] = q;   // the incomplete tail
                hcnt[(size_t) t] = n;
                max_sel = std::max(max_sel, n);
            }
            CK(cudaMemcpyAsync(c_sel, hsel.data(), hsel.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s));
            CK(cudaMemcpyAsync(c_cnt, hcnt.data(), hcnt.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s));
            CK(cudaStreamSynchronize(s));
            sel = c_sel;
        }
        const long long ws = (long long) (kDk + kDv) * kR;
        gm.heads16(ly.kvb, ws, false, c_qm, kMlaH * kDk, c_qa, kMlaH * kR, kR, kDk, T, kMlaH);
        mla_attend_rows(c_qa, ly.lat, sel, c_cnt, kSelLd, pos0, max_sel, 1.0f / std::sqrt((float) kDk), c_ctx, kMlaH, kR,
                        T, s);
        gm.heads16(ly.kvb + (size_t) kDk * kR, ws, true, c_ctx, kMlaH * kR, c_vo, kMlaH * kDv, kDv, kR, T, kMlaH);
        gm.w16(ly.wo, c_vo, c_y, kEmbd, kMlaH * kDv, T);
    }

    void moe_chunk(Layer& ly, int l, int T) {
        using namespace glm;
        gm.w16(ly.router, c_xn, c_t1, kNE, kEmbd, T);
        route_rows(c_t1, ly.router_bias, kNE, kK, kRouteScale, c_ids, c_wts, T, s);
        gm.w16(ly.sg, c_xn, c_t1, kFF, kEmbd, T);
        gm.w16(ly.su, c_xn, c_t2, kFF, kEmbd, T);
        swiglu_clamp(c_t1, c_t2, c_t3, T * kFF, kSwigluLimit, s);
        gm.w16(ly.sd, c_t3, c_y, kEmbd, kFF, T);
        // the whole layer's experts into their slots (one sequential read), the tokens grouped by expert
        std::vector<int32_t> hid((size_t) T * kK);
        CK(cudaMemcpyAsync(hid.data(), c_ids, hid.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, s));
        const int m = l - kDenseLead;
        _fseeki64(experts, (long long) m * kNE * (long long) XL.bytes, SEEK_SET);
        if (std::fread(layer_host, 1, (size_t) kNE * XL.bytes, experts) != (size_t) kNE * XL.bytes) {
            std::fprintf(stderr, "strata-glm: short read of layer %d's experts\n", l);
            std::exit(1);
        }
        CK(cudaMemcpyAsync(layer_slots, layer_host, (size_t) kNE * XL.bytes, cudaMemcpyHostToDevice, s));
        CK(cudaStreamSynchronize(s));
        std::vector<std::vector<int32_t>> by((size_t) kNE);
        for (int i = 0; i < T * kK; ++i) by[(size_t) hid[(size_t) i]].push_back(i);
        std::vector<unsigned long long> gp;
        std::vector<int32_t> gs, et, ed;
        for (int e = 0; e < kNE; ++e) {
            if (by[(size_t) e].empty()) continue;
            gp.push_back((unsigned long long) (layer_slots + (size_t) e * XL.bytes));
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
        strata::kernels::native_expert_grouped_f32(XL, c_grp_ptr, c_grp_start, c_ngroups, c_ent_dst, c_ent_tok, ng,
                                                   (int64_t) T * kK, c_xn, c_xscratch, c_rows, s);
        combine_rows_t(c_rows, c_wts, kK, c_y, kEmbd, T, s);
        CK(cudaStreamSynchronize(s));   // the host vectors above
        if (routes) routes->push_back({l, hid});
    }

    void dense_chunk(Layer& ly, int T) {
        using namespace glm;
        gm.w32(ly.dg, c_xn, c_t1, kDenseFF, kEmbd, T);
        gm.w32(ly.du, c_xn, c_t2, kDenseFF, kEmbd, T);
        swiglu_clamp(c_t1, c_t2, c_t3, T * kDenseFF, kSwigluLimit, s);
        gm.w32(ly.dd, c_t3, c_y, kEmbd, kDenseFF, T);
    }

    // routing log (for the expert profile): per MoE layer and chunk, the T*K chosen ids
    struct Route { int layer; std::vector<int32_t> ids; };
    std::vector<Route>* routes = nullptr;

    // T tokens at positions pos0..: the last one's next-token logits in `logits`
    void forward_chunk(const int* toks, int T, int pos0, const std::string& dump_dir) {
        using namespace glm;
        for (int t = 0; t < T; ++t)
            CK(cudaMemcpyAsync(c_emb + (size_t) t * kEmbd, embed.data() + (size_t) toks[t] * kEmbd, kEmbd * sizeof(bf16),
                               cudaMemcpyHostToDevice, s));
        bf16_to_f32(c_emb, c_x, T * kEmbd, s);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < kHc; ++j)
                CK(cudaMemcpyAsync(c_streams + ((size_t) t * kHc + j) * kEmbd, c_x + (size_t) t * kEmbd, kEmbd * sizeof(float),
                                   cudaMemcpyDeviceToDevice, s));
        for (int l = 0; l < kLayers; ++l) {
            Layer& ly = L[(size_t) l];
            gm.w16(ly.hc_attn_fn, c_streams, c_mix, 24, kHc * kEmbd, T);
            hc_pre_rows(c_streams, c_mix, ly.hc_attn_base, ly.hc_attn_scale, kEmbd, kEps, kHcEps, kSinkhorn, c_x, c_post, c_comb, T, s);
            rmsnorm(c_x, ly.in_norm, kEps, c_xn, kEmbd, T, s);
            if (is_dsa(l)) dsa_chunk(ly, T, pos0); else kda_chunk(ly, T);
            hc_post_rows(c_y, c_streams, c_post, c_comb, c_streams, kEmbd, T, s);
            gm.w16(ly.hc_ffn_fn, c_streams, c_mix, 24, kHc * kEmbd, T);
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
        hc_mean(c_streams + (size_t) (T - 1) * kHc * kEmbd, x, kEmbd, s);
        rmsnorm(x, final_norm, kEps, xn, kEmbd, 1, s);
        gemv_bf16(lm_head, xn, logits, kVocab, kEmbd, 1, s);
        CK(cudaStreamSynchronize(s));
    }

    void load(const std::string& pack) {
        std::string err;
        if (!ck.open(pack, err)) { std::fprintf(stderr, "strata-glm: %s\n", err.c_str()); std::exit(1); }
        const auto t0 = std::chrono::steady_clock::now();
        L.resize(kLayers);
        for (int l = 0; l < kLayers; ++l) {
            Layer& y = L[(size_t) l];
            const std::string p = "model.language_model.layers." + std::to_string(l) + ".";
            y.hc_attn_fn = ck.bf(p + "hc_attn_fn"); y.hc_attn_base = ck.bf(p + "hc_attn_base"); y.hc_attn_scale = ck.bf(p + "hc_attn_scale");
            y.hc_ffn_fn = ck.bf(p + "hc_ffn_fn"); y.hc_ffn_base = ck.bf(p + "hc_ffn_base"); y.hc_ffn_scale = ck.bf(p + "hc_ffn_scale");
            y.in_norm = ck.bf(p + "input_layernorm.weight");
            y.post_norm = ck.bf(p + "post_attention_layernorm.weight");
            const std::string a = p + "self_attn.";
            if (!is_dsa(l)) {
                y.wq = ck.bf(a + "q_proj.weight"); y.wk = ck.bf(a + "k_proj.weight"); y.wv = ck.bf(a + "v_proj.weight");
                y.q_conv = ck.f32(a + "q_conv1d.weight"); y.k_conv = ck.f32(a + "k_conv1d.weight"); y.v_conv = ck.f32(a + "v_conv1d.weight");
                y.fa = ck.bf(a + "f_a_proj.weight"); y.fb = ck.bf(a + "f_b_proj.weight");
                y.dt_bias = ck.f32(a + "dt_bias"); y.A_log = ck.f32(a + "A_log");
                y.bproj = ck.bf(a + "b_proj.weight"); y.ga = ck.bf(a + "g_a_proj.weight"); y.gb = ck.bf(a + "g_b_proj.weight");
                y.onorm = ck.bf(a + "o_norm.weight"); y.wo = ck.bf(a + "o_proj.weight");
                CK(cudaMalloc(&y.S, (size_t) kKdaH * kKdaD * kKdaD * sizeof(float)));
                CK(cudaMalloc(&y.conv, (size_t) 3 * kKdaC * 3 * sizeof(float)));
            } else {
                y.qa = ck.bf(a + "q_a_proj.weight"); y.qa_norm = ck.bf(a + "q_a_layernorm.weight"); y.qb = ck.bf(a + "q_b_proj.weight");
                y.kva = ck.bf(a + "kv_a_proj_with_mqa.weight"); y.kva_norm = ck.bf(a + "kv_a_layernorm.weight"); y.kvb = ck.bf(a + "kv_b_proj.weight");
                y.wo = ck.bf(a + "o_proj.weight");
                const std::string ip = a + "indexer.";
                y.iwqb = ck.bf(ip + "wq_b.weight"); y.iwk = ck.bf(ip + "wk.weight"); y.ik_w = ck.bf(ip + "k_norm.weight");
                y.ik_b = ck.bf(ip + "k_norm.bias"); y.iwp = ck.bf(ip + "weights_proj.weight");
                y.igate = ck.bf(ip + "index_kpool_compress_gate"); y.iape = ck.bf(ip + "index_kpool_compress_ape");
                CK(cudaMalloc(&y.lat, (size_t) max_ctx * kR * sizeof(float)));
                CK(cudaMalloc(&y.ikc, (size_t) max_ctx * kIdxD * sizeof(float)));
                CK(cudaMalloc(&y.igc, (size_t) max_ctx * kIdxD * sizeof(float)));
                CK(cudaMalloc(&y.pooled, (size_t) (max_ctx / kKpool + 1) * kIdxD * sizeof(float)));
            }
            const std::string m = p + "mlp.";
            if (l < kDenseLead) {
                y.dg = ck.nvfp4_f32(m + "gate_proj", kDenseFF, kEmbd);
                y.du = ck.nvfp4_f32(m + "up_proj", kDenseFF, kEmbd);
                y.dd = ck.nvfp4_f32(m + "down_proj", kEmbd, kDenseFF);
            } else {
                y.router = ck.bf(m + "gate.weight");
                y.router_bias = ck.f32(m + "gate.e_score_correction_bias");
                y.sg = ck.bf(m + "shared_experts.gate_proj.weight");
                y.su = ck.bf(m + "shared_experts.up_proj.weight");
                y.sd = ck.bf(m + "shared_experts.down_proj.weight");
            }
        }
        final_norm = ck.bf("model.language_model.norm.weight");
        lm_head = ck.bf("lm_head.weight");
        {
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
        q_resid = buf(kQLora); qm = buf(kMlaH * kDk); qa = buf(kMlaH * kR); ctx = buf(kMlaH * kR); vo = buf(kMlaH * kDv);
        iq = buf(kIdxH * kIdxD); ik = buf(kIdxD); ig = buf(kIdxD); iw = buf(kIdxH); score = buf(max_ctx / kKpool + 1);
        logits = buf(kVocab); rows = buf(kK * kEmbd); wts = buf(kK);
        CK(cudaMalloc(&ids, kK * sizeof(int32_t)));
        CK(cudaMalloc(&sel, (size_t) (kIdxTopk + kKpool) * sizeof(int32_t) + (size_t) max_ctx * sizeof(int32_t)));
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
        gemv_bf16(ly.wq, xn, q, kKdaC, kEmbd, 1, s);
        gemv_bf16(ly.wk, xn, k, kKdaC, kEmbd, 1, s);
        gemv_bf16(ly.wv, xn, v, kKdaC, kEmbd, 1, s);
        conv_silu_step(q, ly.q_conv, ly.conv, q, kKdaC, s);
        conv_silu_step(k, ly.k_conv, ly.conv + (size_t) kKdaC * 3, k, kKdaC, s);
        conv_silu_step(v, ly.v_conv, ly.conv + (size_t) 2 * kKdaC * 3, v, kKdaC, s);
        gemv_bf16(ly.fa, xn, tmp1, kKdaD, kEmbd, 1, s);
        gemv_bf16(ly.fb, tmp1, gf, kKdaC, kKdaD, 1, s);
        gemv_bf16(ly.bproj, xn, b, kKdaH, kEmbd, 1, s);
        kda_prep(q, k, gf, ly.dt_bias, ly.A_log, kLowerBound, b, kKdaH, kKdaD, s);
        kda_step(ly.S, q, k, v, gf, b, o, kKdaH, kKdaD, s);
        gemv_bf16(ly.ga, xn, tmp1, kKdaD, kEmbd, 1, s);
        gemv_bf16(ly.gb, tmp1, gate, kKdaC, kKdaD, 1, s);
        kda_out_norm(o, ly.onorm, gate, kEps, kKdaH, kKdaD, s);
        gemv_bf16(ly.wo, o, y, kEmbd, kKdaC, 1, s);
    }

    void dsa(Layer& ly, int pos) {
        using namespace glm;
        gemv_bf16(ly.qa, xn, tmp1, kQLora, kEmbd, 1, s);
        rmsnorm(tmp1, ly.qa_norm, kEps, q_resid, kQLora, 1, s);
        gemv_bf16(ly.qb, q_resid, qm, kMlaH * kDk, kQLora, 1, s);
        gemv_bf16(ly.kva, xn, tmp1, kR, kEmbd, 1, s);
        rmsnorm(tmp1, ly.kva_norm, kEps, ly.lat + (size_t) pos * kR, kR, 1, s);
        // the indexer's caches and, when a pool completes, its pooled key
        gemv_bf16(ly.iwqb, q_resid, iq, kIdxH * kIdxD, kQLora, 1, s);
        gemv_bf16(ly.iwk, xn, tmp1, kIdxD, kEmbd, 1, s);
        layernorm(tmp1, ly.ik_w, ly.ik_b, 1e-6f, ly.ikc + (size_t) pos * kIdxD, kIdxD, s);
        gemv_bf16(ly.igate, xn, ly.igc + (size_t) pos * kIdxD, kIdxD, kEmbd, 1, s);
        if (pos % kKpool == kKpool - 1) {
            const int p0 = pos - (kKpool - 1);
            idx_pool(ly.ikc + (size_t) p0 * kIdxD, ly.igc + (size_t) p0 * kIdxD, ly.iape, ly.pooled + (size_t) (pos / kKpool) * kIdxD,
                     kKpool, kIdxD, s);
        }
        // the selection: complete pools (all visible) up to the budget, then this query's incomplete tail
        const int n_pool = (pos + 1) / kKpool;
        std::vector<int32_t> hsel;
        if (n_pool <= kIdxTopk / kKpool) {
            for (int t = 0; t <= pos; ++t) hsel.push_back(t);
        } else {
            gemv_bf16(ly.iwp, xn, iw, kIdxH, kEmbd, 1, s);
            idx_scores(iq, iw, ly.pooled, score, n_pool, kIdxH, kIdxD, s);
            std::vector<float> hs((size_t) n_pool);
            CK(cudaMemcpyAsync(hs.data(), score, hs.size() * sizeof(float), cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            std::vector<int> order((size_t) n_pool);
            for (int i = 0; i < n_pool; ++i) order[(size_t) i] = i;
            std::partial_sort(order.begin(), order.begin() + kIdxTopk / kKpool, order.end(),
                              [&](int a, int c) { return hs[(size_t) a] > hs[(size_t) c]; });
            for (int i = 0; i < kIdxTopk / kKpool; ++i)
                for (int j = 0; j < kKpool; ++j) hsel.push_back(order[(size_t) i] * kKpool + j);
            for (int t = n_pool * kKpool; t <= pos; ++t) hsel.push_back(t);
        }
        CK(cudaMemcpyAsync(sel, hsel.data(), hsel.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s));
        mla_absorb_q(qm, ly.kvb, qa, kMlaH, kDk, kDv, kR, s);
        mla_attend(qa, ly.lat, sel, (int) hsel.size(), 1.0f / std::sqrt((float) kDk), ctx, kMlaH, kR, s);
        mla_value(ctx, ly.kvb, vo, kMlaH, kDk, kDv, kR, s);
        gemv_bf16(ly.wo, vo, y, kEmbd, kMlaH * kDv, 1, s);
        CK(cudaStreamSynchronize(s));   // hsel lives on this stack frame
    }

    void moe(Layer& ly, int l) {
        using namespace glm;
        gemv_bf16(ly.router, xn, tmp1, kNE, kEmbd, 1, s);
        route_topk(tmp1, ly.router_bias, kNE, kK, kRouteScale, ids, wts, s);
        // the shared expert (BF16, clamped SwiGLU)
        gemv_bf16(ly.sg, xn, tmp1, kFF, kEmbd, 1, s);
        gemv_bf16(ly.su, xn, tmp2, kFF, kEmbd, 1, s);
        swiglu_clamp(tmp1, tmp2, tmp3, kFF, kSwigluLimit, s);
        gemv_bf16(ly.sd, tmp3, y, kEmbd, kFF, 1, s);
        // the routed experts: read from experts.bin, computed on the GPU
        int32_t hid[kK];
        CK(cudaMemcpyAsync(hid, ids, sizeof(hid), cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        const int m = l - kDenseLead;
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
        gemv_f32(ly.dg, xn, tmp1, kDenseFF, kEmbd, 1, s);
        gemv_f32(ly.du, xn, tmp2, kDenseFF, kEmbd, 1, s);
        swiglu_clamp(tmp1, tmp2, tmp3, kDenseFF, kSwigluLimit, s);
        gemv_f32(ly.dd, tmp3, y, kEmbd, kDenseFF, 1, s);
    }

    // one token at position pos: the next token's logits in `logits`
    void forward(int token, int pos, const std::string& dump_dir) {
        using namespace glm;
        CK(cudaMemcpyAsync(emb_row, embed.data() + (size_t) token * kEmbd, kEmbd * sizeof(bf16), cudaMemcpyHostToDevice, s));
        bf16_to_f32(emb_row, x, kEmbd, s);
        for (int j = 0; j < kHc; ++j) CK(cudaMemcpyAsync(streams + (size_t) j * kEmbd, x, kEmbd * sizeof(float), cudaMemcpyDeviceToDevice, s));
        for (int l = 0; l < kLayers; ++l) {
            Layer& ly = L[(size_t) l];
            gemv_bf16(ly.hc_attn_fn, streams, mix, 24, kHc * kEmbd, 1, s);
            hc_pre_finish(streams, mix, ly.hc_attn_base, ly.hc_attn_scale, kEmbd, kEps, kHcEps, kSinkhorn, x, post, comb, s);
            rmsnorm(x, ly.in_norm, kEps, xn, kEmbd, 1, s);
            if (is_dsa(l)) dsa(ly, pos); else kda(ly);
            hc_post(y, streams, post, comb, streams, kEmbd, s);
            gemv_bf16(ly.hc_ffn_fn, streams, mix, 24, kHc * kEmbd, 1, s);
            hc_pre_finish(streams, mix, ly.hc_ffn_base, ly.hc_ffn_scale, kEmbd, kEps, kHcEps, kSinkhorn, x, post, comb, s);
            rmsnorm(x, ly.post_norm, kEps, xn, kEmbd, 1, s);
            if (l < kDenseLead) dense_mlp(ly); else moe(ly, l);
            hc_post(y, streams, post, comb, streams, kEmbd, s);
            if (!dump_dir.empty()) {
                std::vector<float> h((size_t) kHc * kEmbd);
                CK(cudaMemcpyAsync(h.data(), streams, h.size() * sizeof(float), cudaMemcpyDeviceToHost, s));
                CK(cudaStreamSynchronize(s));
                char name[64];
                std::snprintf(name, sizeof name, "/l%02d.f32", l);
                if (std::FILE* f = std::fopen((dump_dir + name).c_str(), "wb")) { std::fwrite(h.data(), 4, h.size(), f); std::fclose(f); }
            }
        }
        hc_mean(streams, x, kEmbd, s);
        rmsnorm(x, final_norm, kEps, xn, kEmbd, 1, s);
        gemv_bf16(lm_head, xn, logits, kVocab, kEmbd, 1, s);
        CK(cudaStreamSynchronize(s));
    }
};

}  // namespace

int main(int argc, char** argv) {
    std::string pack, tokens_path, dump_dir, dump_logits;
    int max_new = 16, max_ctx = 8192, chunk = 0;
    std::string routes_path;
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
        else { std::fprintf(stderr, "strata-glm: unknown argument %s\n", a.c_str()); return 2; }
    }
    if (pack.empty() || tokens_path.empty()) {
        std::fprintf(stderr, "usage: strata-glm --pack DIR --tokens FILE [--max-new N] [--max-context C] [--dump-dir D] [--dump-logits F]\n");
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

    Engine e;
    e.max_ctx = max_ctx;
    e.load(pack);
    std::vector<Engine::Route> routes;
    if (!routes_path.empty()) e.routes = &routes;
    if (chunk > 0) e.init_chunk(chunk);
    const auto t0 = std::chrono::steady_clock::now();
    if (chunk > 0) {
        for (size_t i = 0; i < prompt.size(); i += (size_t) chunk) {
            const int T = (int) std::min<size_t>((size_t) chunk, prompt.size() - i);
            e.forward_chunk(prompt.data() + i, T, (int) i, i + T == prompt.size() ? dump_dir : std::string());
            std::fprintf(stderr, "strata-glm: %zu of %zu prompt tokens\n", i + T, prompt.size());
        }
    } else {
        for (size_t i = 0; i < prompt.size(); ++i)
            e.forward(prompt[i], (int) i, i + 1 == prompt.size() ? dump_dir : std::string());
    }
    if (!routes_path.empty()) {   // layer, then T*K ids per record
        if (std::FILE* f = std::fopen(routes_path.c_str(), "ab")) {
            for (const auto& r : routes) {
                const int32_t hdr[2] = {r.layer, (int32_t) r.ids.size()};
                std::fwrite(hdr, sizeof hdr, 1, f);
                std::fwrite(r.ids.data(), sizeof(int32_t), r.ids.size(), f);
            }
            std::fclose(f);
        }
    }
    const double tp = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::vector<float> lg(kVocab);
    CK(cudaMemcpy(lg.data(), e.logits, lg.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (!dump_logits.empty())
        if (std::FILE* f = std::fopen(dump_logits.c_str(), "wb")) { std::fwrite(lg.data(), 4, lg.size(), f); std::fclose(f); }
    std::fprintf(stderr, "strata-glm: prompt %zu tokens in %.1f s (%.2f tok/s)\n", prompt.size(), tp, prompt.size() / tp);
    std::vector<int> out;
    int pos = (int) prompt.size();
    const auto t1 = std::chrono::steady_clock::now();
    for (int n = 0; n < max_new && pos < max_ctx; ++n) {
        const int tok = (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
        out.push_back(tok);
        if (tok == 154820 || tok == 154827 || tok == 154829) break;   // eos ids (config.json)
        e.forward(tok, pos++, std::string());
        CK(cudaMemcpy(lg.data(), e.logits, lg.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }
    const double td = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    std::printf("output :");
    for (int t : out) std::printf(" %d", t);
    std::printf("\n");
    std::fprintf(stderr, "strata-glm: decode %zu tokens in %.1f s (%.2f tok/s)\n", out.size(), td, out.size() / td);
    return 0;
}
