#ifndef CK_USE_OCP_FP8
#define CK_USE_OCP_FP8 1
#endif

#if CK_USE_OCP_FP8
#pragma message("Verification: CK_USE_OCP_FP8 is ENABLED. Using OCP FP8 types.")
#else
#error "Verification Failed: CK_USE_OCP_FP8 is NOT enabled!"
#endif

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace ck_tile {
namespace core {
namespace arch {
using TargetId = amdgcn_target_id;
} // namespace arch
} // namespace core
} // namespace ck_tile

struct MXFlatmmConfigBase16
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 256;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 128;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                = 1;
    static constexpr int TileParitionerGroupNum     = 8;
    static constexpr int TileParitionerM01          = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = true;

    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = false;
};

template <ck_tile::core::arch::TargetId Arch, typename FlatmmConfig>
struct MXFlatmmArchTraits
{
    static constexpr int BlockedXDLN_PerWarp = 2;
    using Config = FlatmmConfig;

    template <typename MXPipelineProblem>
    using MXFlatmmPipeline = ck_tile::MXFlatmmPipelineAGmemBGmemCRegV1<MXPipelineProblem>;
};

using MXTraits = MXFlatmmArchTraits<ck_tile::core::arch::TargetId::GFX950, MXFlatmmConfigBase16>;

static ck_tile::index_t round_up(ck_tile::index_t x, ck_tile::index_t tile)
{
    return ((x + tile - 1) / tile) * tile;
}

static void hip_check(hipError_t err, const char* msg)
{
    if(err != hipSuccess)
        throw std::runtime_error(std::string(msg) + ": " + hipGetErrorString(err));
}

template <typename LaunchFn>
float time_kernel_hip_events(hipStream_t stream, int warmup, int repeat, LaunchFn&& launch_once)
{
    warmup = std::max(warmup, 0);
    repeat = std::max(repeat, 1);

    for(int i = 0; i < warmup; ++i)
        launch_once();
    hip_check(hipStreamSynchronize(stream), "hipStreamSynchronize after warmup");

    hipEvent_t start_ev = nullptr;
    hipEvent_t stop_ev  = nullptr;
    hip_check(hipEventCreate(&start_ev), "hipEventCreate start");
    hip_check(hipEventCreate(&stop_ev), "hipEventCreate stop");

    hip_check(hipEventRecord(start_ev, stream), "hipEventRecord start");
    for(int i = 0; i < repeat; ++i)
        launch_once();
    hip_check(hipEventRecord(stop_ev, stream), "hipEventRecord stop");
    hip_check(hipEventSynchronize(stop_ev), "hipEventSynchronize stop");

    float elapsed_ms = 0.0f;
    hip_check(hipEventElapsedTime(&elapsed_ms, start_ev, stop_ev), "hipEventElapsedTime");

    hip_check(hipEventDestroy(start_ev), "hipEventDestroy start");
    hip_check(hipEventDestroy(stop_ev), "hipEventDestroy stop");

    return elapsed_ms / static_cast<float>(repeat);
}

template <typename T>
__device__ __forceinline__ T device_zero()
{
    if constexpr(std::is_same_v<T, ck_tile::e8m0_bexp_t>)
        return T{0.0f};
    else
        return ck_tile::type_convert<T>(0.0f);
}

template <typename T>
__device__ __forceinline__ T device_one()
{
    if constexpr(std::is_same_v<T, ck_tile::e8m0_bexp_t>)
        return T{1.0f};
    else
        return ck_tile::type_convert<T>(1.0f);
}

template <typename ADataType>
__global__ void pack_a_rows_kernel(ADataType* dst,
                                   const ADataType* src,
                                   int64_t m_real,
                                   int64_t m_pad,
                                   int64_t K,
                                   int64_t K_pad)
{
    const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = m_pad * K_pad;
    if(idx >= total)
        return;

    const int64_t row = idx / K_pad;
    const int64_t col = idx % K_pad;

    dst[idx] = (row < m_real && col < K) ? src[row * K + col] : device_zero<ADataType>();
}

template <typename ScaleType>
__global__ void preshuffle_a_scales_kernel(ScaleType* dst,
                                           const ScaleType* src,
                                           int64_t m_real,
                                           int64_t m_pad,
                                           int64_t QK,
                                           int64_t QK_pad)
{
    constexpr int XdlMNThread = MXFlatmmConfigBase16::N_Warp_Tile;
    constexpr int MNXdlPack   = 2;
    constexpr int KXdlPack    = 2;
    constexpr int XdlKThread  = ck_tile::get_warp_size() / XdlMNThread;

    const int64_t total = m_pad * QK_pad;
    const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= total)
        return;

    const int64_t n = idx / QK_pad;
    const int64_t k = idx % QK_pad;

    const int64_t n0    = n / (XdlMNThread * MNXdlPack);
    const int64_t tempn = n % (XdlMNThread * MNXdlPack);
    const int64_t n1    = tempn % XdlMNThread;
    const int64_t n2    = tempn / XdlMNThread;

    const int64_t k0    = k / (XdlKThread * KXdlPack);
    const int64_t tempk = k % (XdlKThread * KXdlPack);
    const int64_t k1    = tempk % XdlKThread;
    const int64_t k2    = tempk / XdlKThread;

    const int64_t K0 = QK_pad / (KXdlPack * XdlKThread);

    const int64_t outputIndex =
        n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
        k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
        k1 * MNXdlPack * KXdlPack * XdlMNThread + n1 * MNXdlPack * KXdlPack +
        k2 * MNXdlPack + n2;

    dst[outputIndex] = (n < m_real && k < QK) ? src[n * QK + k] : device_one<ScaleType>();
}

template <typename BDataType>
__global__ void preshuffle_b_weight_kernel(BDataType* dst,
                                           const BDataType* src,
                                           int64_t K,
                                           int64_t N,
                                           int64_t K_pad)
{
    constexpr int NLane       = MXFlatmmConfigBase16::N_Warp_Tile;
    constexpr int packed_size = ck_tile::numeric_traits<BDataType>::PackedSize;
    const int KPack           = 16 * packed_size;
    const int KLane           = ck_tile::get_warp_size() / NLane;

    const int64_t K_items_pad = K_pad / packed_size;
    const int64_t item_idx    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total       = N * K_items_pad;
    if(item_idx >= total)
        return;

    const int64_t n      = item_idx / K_items_pad;
    const int64_t k_item = item_idx % K_items_pad;
    const int64_t k      = k_item * packed_size;

    const int64_t n0    = n / NLane;
    const int64_t n1    = n % NLane;
    const int64_t k0    = k / (KLane * KPack);
    const int64_t tempk = k % (KLane * KPack);
    const int64_t k1    = tempk / KPack;
    const int64_t k2    = tempk % KPack;
    const int64_t K0    = K_pad / (KLane * KPack);

    const int64_t outputIndex =
        n0 * KPack * NLane * KLane * K0 +
        k0 * KPack * NLane * KLane +
        k1 * KPack * NLane + n1 * KPack + k2;

    dst[outputIndex] = (k < K) ? src[k + n * K] : device_zero<BDataType>();
}

template <typename ScaleType>
__global__ void preshuffle_b_scales_kernel(ScaleType* dst,
                                           const ScaleType* src,
                                           int64_t QK,
                                           int64_t N,
                                           int64_t N_pad,
                                           int64_t QK_pad)
{
    constexpr int XdlMNThread = MXFlatmmConfigBase16::N_Warp_Tile;
    constexpr int MNXdlPack   = 2;
    constexpr int KXdlPack    = 2;
    constexpr int XdlKThread  = ck_tile::get_warp_size() / XdlMNThread;

    const int64_t total = N_pad * QK_pad;
    const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= total)
        return;

    const int64_t n = idx / QK_pad;
    const int64_t k = idx % QK_pad;

    const int64_t n0    = n / (XdlMNThread * MNXdlPack);
    const int64_t tempn = n % (XdlMNThread * MNXdlPack);
    const int64_t n1    = tempn % XdlMNThread;
    const int64_t n2    = tempn / XdlMNThread;

    const int64_t k0    = k / (XdlKThread * KXdlPack);
    const int64_t tempk = k % (XdlKThread * KXdlPack);
    const int64_t k1    = tempk % XdlKThread;
    const int64_t k2    = tempk / XdlKThread;

    const int64_t K0 = QK_pad / (KXdlPack * XdlKThread);

    const int64_t outputIndex =
        n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
        k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
        k1 * MNXdlPack * KXdlPack * XdlMNThread + n1 * MNXdlPack * KXdlPack +
        k2 * MNXdlPack + n2;

    dst[outputIndex] = (n < N && k < QK) ? src[k + n * QK] : device_zero<ScaleType>();
}

template <typename CDataType>
__global__ void scatter_c_kernel(CDataType* dst,
                                 const CDataType* src,
                                 int64_t m_real,
                                 int64_t N)
{
    const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = m_real * N;
    if(idx >= total)
        return;
    dst[idx] = src[idx];
}

template <typename MXFlatmmArchTraitsT,
          typename ADataType,
          typename BDataType,
          typename DsDatatype,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename ScaleA,
          typename ScaleB,
          bool UsePersistentKernel = false,
          typename CDEElementWise  = ck_tile::element_wise::PassThrough>
float invoke_mx_flatmm_raw(const void* a_ptr,
                           const void* b_ptr,
                           void* c_ptr,
                           ck_tile::index_t M,
                           ck_tile::index_t N,
                           ck_tile::index_t K,
                           ck_tile::index_t stride_A,
                           ck_tile::index_t stride_B,
                           ck_tile::index_t stride_C,
                           ScaleA scale_a,
                           ScaleB scale_b,
                           int n_warmup,
                           int n_repeat)
{
    using FlatmmConfig = typename MXFlatmmArchTraitsT::Config;

    ck_tile::ScaleFlatmmHostArgs<ScaleA, ScaleB> args = {
        a_ptr,
        b_ptr,
        {},
        c_ptr,
        1,
        M,
        N,
        K,
        stride_A,
        stride_B,
        {},
        stride_C,
        scale_a,
        scale_b};

    using FlatmmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<FlatmmConfig::M_Tile, FlatmmConfig::N_Tile, FlatmmConfig::K_Tile>,
        ck_tile::sequence<FlatmmConfig::M_Warp, FlatmmConfig::N_Warp, FlatmmConfig::K_Warp>,
        ck_tile::sequence<FlatmmConfig::M_Warp_Tile,
                          FlatmmConfig::N_Warp_Tile,
                          FlatmmConfig::K_Warp_Tile>>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<FlatmmShape,
                                                   FlatmmConfig::TileParitionerGroupNum,
                                                   FlatmmConfig::TileParitionerM01>;

    using Traits = ck_tile::TileGemmTraits<FlatmmConfig::kPadM,
                                           FlatmmConfig::kPadN,
                                           FlatmmConfig::kPadK,
                                           ALayout,
                                           BLayout,
                                           CLayout,
                                           FlatmmConfig::NumWaveGroups>;

    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, FlatmmShape, Traits>;

    using BaseFlatmmPipeline = ck_tile::BaseFlatmmPipelineAGmemBGmemCRegV1<GemmPipelineProblem>;

    const ck_tile::index_t k_grain     = FlatmmConfig::K_Tile;
    const ck_tile::index_t k_split     = (K + k_grain - 1) / k_grain * k_grain;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(k_split);
    const bool has_hot_loop            = BaseFlatmmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseFlatmmPipeline::GetBlockLoopTailNum(num_loop);

    return BaseFlatmmPipeline::template TailHandler<true>(
        [&](auto has_hot_loop_, auto tail_num_) {
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_num_v     = tail_num_.value;

            using MXGemmTraits = ck_tile::TileGemmUniversalTraits<FlatmmConfig::kPadM,
                                                                  FlatmmConfig::kPadN,
                                                                  FlatmmConfig::kPadK,
                                                                  FlatmmConfig::DoubleSmemBuffer,
                                                                  ALayout,
                                                                  BLayout,
                                                                  CLayout,
                                                                  FlatmmConfig::TransposeC,
                                                                  FlatmmConfig::UseStructuredSparsity,
                                                                  UsePersistentKernel,
                                                                  FlatmmConfig::NumWaveGroups,
                                                                  true>;

            using MXPipelineProblem = ck_tile::MXFlatmmPipelineProblem<ADataType,
                                                                       BDataType,
                                                                       AccDataType,
                                                                       FlatmmShape,
                                                                       MXGemmTraits,
                                                                       FlatmmConfig::Scheduler,
                                                                       has_hot_loop_v,
                                                                       tail_num_v>;

            using MXFlatmmPipeline =
                typename MXFlatmmArchTraitsT::template MXFlatmmPipeline<MXPipelineProblem>;

            constexpr int BlockedXDLN_PerWarp = MXFlatmmArchTraitsT::BlockedXDLN_PerWarp;

            using GemmEpilogue =
                ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<ADataType,
                                                                           ADataType,
                                                                           DsDatatype,
                                                                           AccDataType,
                                                                           CDataType,
                                                                           DsLayout,
                                                                           CLayout,
                                                                           CDEElementWise,
                                                                           TilePartitioner::MPerBlock,
                                                                           TilePartitioner::NPerBlock,
                                                                           FlatmmConfig::M_Warp,
                                                                           FlatmmConfig::N_Warp,
                                                                           FlatmmConfig::M_Warp_Tile,
                                                                           FlatmmConfig::N_Warp_Tile,
                                                                           FlatmmConfig::K_Warp_Tile,
                                                                           MXPipelineProblem::TransposeC,
                                                                           FlatmmConfig::NumWaveGroups,
                                                                           false,
                                                                           1,
                                                                           FlatmmConfig::TiledMMAPermuteN,
                                                                           BlockedXDLN_PerWarp>>;

            using Kernel = ck_tile::MXFlatmmKernel<TilePartitioner, MXFlatmmPipeline, GemmEpilogue>;

            auto kargs = Kernel::MakeKernelArgs(args);
            const dim3 grids      = Kernel::GridSize(kargs);
            constexpr dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs))
                throw std::runtime_error("Arguments not supported for MXFlatmmKernel");

            hipStream_t stream = at::hip::getCurrentHIPStream();
            return time_kernel_hip_events(stream, n_warmup, n_repeat, [&] {
                hipLaunchKernelGGL(
                    (ck_tile::kentry<FlatmmConfig::kBlockPerCu, Kernel, decltype(kargs)>),
                    grids,
                    blocks,
                    0,
                    stream,
                    kargs);
                hip_check(hipGetLastError(), "launch MXFlatmmKernel");
            });
        },
        has_hot_loop,
        tail_num);
}

std::vector<torch::Tensor> ck_mx_prepack_a_mxfp8(torch::Tensor a, torch::Tensor a_scales)
{
    using ADataType = ck_tile::fp8_t;
    using ScaleType = ck_tile::e8m0_bexp_t;
    using index_t   = ck_tile::index_t;

    constexpr int ScaleGranularityK = 32;
    constexpr int kThreads = 256;

    TORCH_CHECK(a.is_cuda(), "a must be CUDA/HIP");
    TORCH_CHECK(a_scales.is_cuda(), "a_scales must be CUDA/HIP");
    TORCH_CHECK(a.dim() == 2, "a must be [M, K]");
    TORCH_CHECK(a_scales.dim() == 2, "a_scales must be [M, K/32]");
    TORCH_CHECK(a.scalar_type() == at::kFloat8_e4m3fn, "a must be torch.float8_e4m3fn");
    TORCH_CHECK(a_scales.scalar_type() == at::kFloat8_e8m0fnu, "a_scales must be torch.float8_e8m0fnu");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous row-major");
    TORCH_CHECK(a_scales.is_contiguous(), "a_scales must be contiguous row-major");

    const index_t M  = static_cast<index_t>(a.size(0));
    const index_t K  = static_cast<index_t>(a.size(1));
    const index_t QK = K / ScaleGranularityK;

    TORCH_CHECK(K % ScaleGranularityK == 0, "K must be divisible by 32");
    TORCH_CHECK(a_scales.size(0) == M && a_scales.size(1) == QK, "a_scales shape mismatch");

    const index_t M_pad  = round_up(M, MXFlatmmConfigBase16::M_Tile);
    const index_t K_pad  = round_up(K, MXFlatmmConfigBase16::K_Tile);
    const index_t QK_pad = K_pad / ScaleGranularityK;

    auto a_packed =
        torch::empty({static_cast<long long>(M_pad) * static_cast<long long>(K_pad)}, a.options());
    auto as_packed =
        torch::empty({static_cast<long long>(M_pad) * static_cast<long long>(QK_pad)}, a_scales.options());

    auto* a_dst  = reinterpret_cast<ADataType*>(a_packed.data_ptr());
    auto* as_dst = reinterpret_cast<ScaleType*>(as_packed.data_ptr());
    auto* a_src  = reinterpret_cast<const ADataType*>(a.data_ptr());
    auto* as_src = reinterpret_cast<const ScaleType*>(a_scales.data_ptr());

    hipStream_t stream = at::hip::getCurrentHIPStream();

    {
        const int64_t total = static_cast<int64_t>(M_pad) * static_cast<int64_t>(K_pad);
        hipLaunchKernelGGL(
            pack_a_rows_kernel<ADataType>,
            dim3((total + kThreads - 1) / kThreads),
            dim3(kThreads),
            0,
            stream,
            a_dst,
            a_src,
            M,
            M_pad,
            K,
            K_pad);
        hip_check(hipGetLastError(), "pack_a_rows_kernel");
    }

    {
        const int64_t total = static_cast<int64_t>(M_pad) * static_cast<int64_t>(QK_pad);
        hipLaunchKernelGGL(
            preshuffle_a_scales_kernel<ScaleType>,
            dim3((total + kThreads - 1) / kThreads),
            dim3(kThreads),
            0,
            stream,
            as_dst,
            as_src,
            M,
            M_pad,
            QK,
            QK_pad);
        hip_check(hipGetLastError(), "preshuffle_a_scales_kernel");
    }

    hip_check(hipStreamSynchronize(stream), "sync after A prepack");
    return {a_packed, as_packed};
}

std::vector<torch::Tensor> ck_mx_prepack_b_mxfp8(torch::Tensor b, torch::Tensor b_scales)
{
    using BDataType = ck_tile::fp8_t;
    using ScaleType = ck_tile::e8m0_bexp_t;
    using index_t   = ck_tile::index_t;

    constexpr int ScaleGranularityK = 32;
    constexpr int kThreads          = 256;
    constexpr int kScaleMNPad       = MXFlatmmConfigBase16::N_Warp_Tile * 2;

    TORCH_CHECK(b.is_cuda(), "b must be CUDA/HIP");
    TORCH_CHECK(b_scales.is_cuda(), "b_scales must be CUDA/HIP");
    TORCH_CHECK(b.dim() == 2, "b must be [K, N]");
    TORCH_CHECK(b_scales.dim() == 2, "b_scales must be [K/32, N]");
    TORCH_CHECK(b.scalar_type() == at::kFloat8_e4m3fn, "b must be torch.float8_e4m3fn");
    TORCH_CHECK(b_scales.scalar_type() == at::kFloat8_e8m0fnu, "b_scales must be torch.float8_e8m0fnu");

    const index_t K  = static_cast<index_t>(b.size(0));
    const index_t N  = static_cast<index_t>(b.size(1));
    const index_t QK = K / ScaleGranularityK;

    TORCH_CHECK(K % ScaleGranularityK == 0, "K must be divisible by 32");
    TORCH_CHECK(b_scales.size(0) == QK && b_scales.size(1) == N, "b_scales shape mismatch");
    TORCH_CHECK(b.stride(0) == 1 && b.stride(1) == K, "b must be CK column-major [K,N]");
    TORCH_CHECK(b_scales.stride(0) == 1 && b_scales.stride(1) == QK,
                "b_scales must be CK column-major [K/32,N]");

    const index_t K_pad  = round_up(K, MXFlatmmConfigBase16::K_Tile);
    const index_t QK_pad = K_pad / ScaleGranularityK;
    const index_t N_pad  = round_up(N, kScaleMNPad);

    auto b_packed =
        torch::empty({static_cast<long long>(K_pad) * static_cast<long long>(N)}, b.options());
    auto bs_packed =
        torch::empty({static_cast<long long>(QK_pad) * static_cast<long long>(N_pad)}, b_scales.options());

    auto* b_dst  = reinterpret_cast<BDataType*>(b_packed.data_ptr());
    auto* bs_dst = reinterpret_cast<ScaleType*>(bs_packed.data_ptr());
    auto* b_src  = reinterpret_cast<const BDataType*>(b.data_ptr());
    auto* bs_src = reinterpret_cast<const ScaleType*>(b_scales.data_ptr());

    hipStream_t stream = at::hip::getCurrentHIPStream();

    {
        const int64_t total_items =
            static_cast<int64_t>(N) * static_cast<int64_t>(K_pad / ck_tile::numeric_traits<BDataType>::PackedSize);
        hipLaunchKernelGGL(
            preshuffle_b_weight_kernel<BDataType>,
            dim3((total_items + kThreads - 1) / kThreads),
            dim3(kThreads),
            0,
            stream,
            b_dst,
            b_src,
            K,
            N,
            K_pad);
        hip_check(hipGetLastError(), "preshuffle_b_weight_kernel");
    }

    {
        const int64_t total_scales = static_cast<int64_t>(N_pad) * static_cast<int64_t>(QK_pad);
        hipLaunchKernelGGL(
            preshuffle_b_scales_kernel<ScaleType>,
            dim3((total_scales + kThreads - 1) / kThreads),
            dim3(kThreads),
            0,
            stream,
            bs_dst,
            bs_src,
            QK,
            N,
            N_pad,
            QK_pad);
        hip_check(hipGetLastError(), "preshuffle_b_scales_kernel");
    }

    hip_check(hipStreamSynchronize(stream), "sync after B prepack");
    return {b_packed, bs_packed};
}

float ck_mx_gemm_mxfp8_raw_prepacked(torch::Tensor a_packed,
                                     torch::Tensor a_scales_packed,
                                     torch::Tensor b_packed,
                                     torch::Tensor b_scales_packed,
                                     torch::Tensor c,
                                     std::int64_t M_real,
                                     std::int64_t N_real,
                                     std::int64_t K_real,
                                     int warmup = 20,
                                     int repeat = 50)
{
    using ADataType   = ck_tile::fp8_t;
    using BDataType   = ck_tile::fp8_t;
    using CDataType   = ck_tile::half_t;
    using AccDataType = float;
    using ScaleType   = ck_tile::e8m0_bexp_t;
    using RowMajor    = ck_tile::tensor_layout::gemm::RowMajor;
    using ColMajor    = ck_tile::tensor_layout::gemm::ColumnMajor;
    using index_t     = ck_tile::index_t;

    constexpr int ScaleGranularityK = 32;
    constexpr int kThreads = 256;

    TORCH_CHECK(a_packed.is_cuda(), "a_packed must be CUDA/HIP");
    TORCH_CHECK(a_scales_packed.is_cuda(), "a_scales_packed must be CUDA/HIP");
    TORCH_CHECK(b_packed.is_cuda(), "b_packed must be CUDA/HIP");
    TORCH_CHECK(b_scales_packed.is_cuda(), "b_scales_packed must be CUDA/HIP");
    TORCH_CHECK(c.is_cuda(), "c must be CUDA/HIP");

    TORCH_CHECK(a_packed.dim() == 1, "a_packed must be flat 1D buffer");
    TORCH_CHECK(a_scales_packed.dim() == 1, "a_scales_packed must be flat 1D buffer");
    TORCH_CHECK(b_packed.dim() == 1, "b_packed must be flat 1D buffer");
    TORCH_CHECK(b_scales_packed.dim() == 1, "b_scales_packed must be flat 1D buffer");
    TORCH_CHECK(c.dim() == 2, "c must be [M, N]");

    TORCH_CHECK(c.scalar_type() == at::kHalf, "c must be torch.float16");
    TORCH_CHECK(c.size(0) == M_real && c.size(1) == N_real, "c shape mismatch");
    TORCH_CHECK(K_real % ScaleGranularityK == 0, "K must be divisible by 32");

    const index_t M = static_cast<index_t>(M_real);
    const index_t N = static_cast<index_t>(N_real);
    const index_t K = static_cast<index_t>(K_real);

    const index_t M_pad  = round_up(M, MXFlatmmConfigBase16::M_Tile);
    const index_t K_pad  = round_up(K, MXFlatmmConfigBase16::K_Tile);
    const index_t QK_pad = K_pad / ScaleGranularityK;

    auto c_tmp = torch::zeros({M_pad, N}, c.options());

    auto* a_ptr  = reinterpret_cast<const ADataType*>(a_packed.data_ptr());
    auto* as_ptr = reinterpret_cast<ScaleType*>(a_scales_packed.data_ptr());
    auto* b_ptr  = reinterpret_cast<const BDataType*>(b_packed.data_ptr());
    auto* bs_ptr = reinterpret_cast<ScaleType*>(b_scales_packed.data_ptr());
    auto* c_tmp_ptr = reinterpret_cast<CDataType*>(c_tmp.data_ptr());
    auto* c_dst = reinterpret_cast<CDataType*>(c.data_ptr());

    const index_t stride_A = K_pad;
    const index_t stride_B = K_pad;
    const index_t stride_C = N;

    auto scale_a = ck_tile::FlatmmScalePointer<1, 32, ScaleType>{as_ptr, M_pad};
    auto scale_b = ck_tile::FlatmmScalePointer<1, 32, ScaleType>{bs_ptr, N};

    const float ms = invoke_mx_flatmm_raw<MXTraits,
                                          ADataType,
                                          BDataType,
                                          ck_tile::tuple<>,
                                          AccDataType,
                                          CDataType,
                                          RowMajor,
                                          ColMajor,
                                          ck_tile::tuple<>,
                                          RowMajor>(
        a_ptr,
        b_ptr,
        c_tmp_ptr,
        M_pad,
        N,
        K_pad,
        stride_A,
        stride_B,
        stride_C,
        scale_a,
        scale_b,
        warmup,
        repeat);

    hipStream_t stream = at::hip::getCurrentHIPStream();
    c.zero_();

    {
        const int64_t total = static_cast<int64_t>(M) * static_cast<int64_t>(N);
        hipLaunchKernelGGL(
            scatter_c_kernel<CDataType>,
            dim3((total + kThreads - 1) / kThreads),
            dim3(kThreads),
            0,
            stream,
            c_dst,
            c_tmp_ptr,
            M,
            N);
        hip_check(hipGetLastError(), "scatter_c_kernel");
    }

    hip_check(hipStreamSynchronize(stream), "sync after scatter");
    return ms;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ck_mx_prepack_a_mxfp8",
          &ck_mx_prepack_a_mxfp8,
          pybind11::arg("a"),
          pybind11::arg("a_scales"));

    m.def("ck_mx_prepack_b_mxfp8",
          &ck_mx_prepack_b_mxfp8,
          pybind11::arg("b"),
          pybind11::arg("b_scales"));

    m.def("ck_mx_gemm_mxfp8_raw_prepacked",
          &ck_mx_gemm_mxfp8_raw_prepacked,
          pybind11::arg("a_packed"),
          pybind11::arg("a_scales_packed"),
          pybind11::arg("b_packed"),
          pybind11::arg("b_scales_packed"),
          pybind11::arg("c"),
          pybind11::arg("M_real"),
          pybind11::arg("N_real"),
          pybind11::arg("K_real"),
          pybind11::arg("warmup") = 20,
          pybind11::arg("repeat") = 50);
}
