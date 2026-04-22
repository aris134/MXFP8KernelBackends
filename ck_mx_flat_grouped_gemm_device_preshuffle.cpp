// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef CK_USE_OCP_FP8
#define CK_USE_OCP_FP8 1
#endif

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/flatmm/kernel/mx_flatmm_kernel.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace ck_tile {
namespace core {
namespace arch {
using TargetId = amdgcn_target_id;
} // namespace arch
} // namespace core
} // namespace ck_tile

template <typename Layout>
static constexpr inline auto is_row_major(Layout)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<Layout>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

static void hip_check(hipError_t err, const char* msg)
{
    if(err != hipSuccess)
    {
        throw std::runtime_error(std::string(msg) + ": " + hipGetErrorString(err));
    }
}

template <typename T>
T make_value_from_float(float v)
{
    if constexpr(std::is_same_v<T, ck_tile::e8m0_bexp_t>)
        return T{v};
    else
        return ck_tile::type_convert<T>(v);
}

template <typename TensorT>
void fill_tensor_constant(TensorT& tensor, float v)
{
    using T = std::remove_cvref_t<decltype(tensor.mData[0])>;
    const T value = make_value_from_float<T>(v);
    for(auto& x : tensor.mData)
        x = value;
}

static ck_tile::index_t round_up(ck_tile::index_t x, ck_tile::index_t tile)
{
    return ((x + tile - 1) / tile) * tile;
}

template <typename LaunchFn>
float time_hip_events(hipStream_t stream, int warmup, int repeat, LaunchFn&& launch_once)
{
    warmup = std::max(warmup, 0);
    repeat = std::max(repeat, 1);

    for(int i = 0; i < warmup; ++i)
        launch_once();
    hip_check(hipStreamSynchronize(stream), "sync after warmup");

    hipEvent_t start_ev = nullptr;
    hipEvent_t stop_ev  = nullptr;
    hip_check(hipEventCreate(&start_ev), "event create start");
    hip_check(hipEventCreate(&stop_ev), "event create stop");

    hip_check(hipEventRecord(start_ev, stream), "event record start");
    for(int i = 0; i < repeat; ++i)
        launch_once();
    hip_check(hipEventRecord(stop_ev, stream), "event record stop");
    hip_check(hipEventSynchronize(stop_ev), "event sync stop");

    float elapsed_ms = 0.0f;
    hip_check(hipEventElapsedTime(&elapsed_ms, start_ev, stop_ev), "event elapsed");

    hip_check(hipEventDestroy(start_ev), "event destroy start");
    hip_check(hipEventDestroy(stop_ev), "event destroy stop");

    return elapsed_ms / static_cast<float>(repeat);
}

// -----------------------------------------------------------------------------
// Device-side preshuffle kernels
// -----------------------------------------------------------------------------

template <typename dtype, int NLane>
__global__ void preshuffle_weight_kernel(const dtype* __restrict__ src,
                                         dtype* __restrict__ dst,
                                         int K,
                                         int N)
{
    constexpr int packed_size = ck_tile::numeric_traits<dtype>::PackedSize;
    const int KPack = std::is_same_v<dtype, ck_tile::pk_fp6x16_t> ? 32 : 16 * packed_size;
    const int KLane = 64 / NLane;
    const int K0    = K / (KLane * KPack);

    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total  = N * (K / packed_size);
    if(linear >= total)
        return;

    const int n = linear / (K / packed_size);
    const int k = (linear % (K / packed_size)) * packed_size;

    const int n0 = n / NLane;
    const int n1 = n % NLane;

    const int k0    = k / (KLane * KPack);
    const int tempk = k % (KLane * KPack);
    const int k1    = tempk / KPack;
    const int k2    = tempk % KPack;

    const int outputIndex = n0 * KPack * NLane * KLane * K0 +
                            k0 * KPack * NLane * KLane +
                            k1 * KPack * NLane + n1 * KPack + k2;

    dst[outputIndex] = src[k * N + n];
}

template <typename dtype, bool KLast, int XdlMNThread>
__global__ void preshuffle_scale_kernel(const dtype* __restrict__ src,
                                        dtype* __restrict__ dst,
                                        int MN,
                                        int K)
{
    constexpr int MNXdlPack = 2;
    constexpr int KXdlPack  = 2;
    constexpr int XdlKThread = 64 / XdlMNThread;

    const int MN_padded = ((MN + XdlMNThread * MNXdlPack - 1) / (XdlMNThread * MNXdlPack)) *
                          (XdlMNThread * MNXdlPack);
    const int K0 = K / KXdlPack / XdlKThread;

    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total  = MN_padded * K;
    if(linear >= total)
        return;

    const int n = linear / K;
    const int k = linear % K;

    const int n0    = n / (XdlMNThread * MNXdlPack);
    const int tempn = n % (XdlMNThread * MNXdlPack);
    const int n1    = tempn % XdlMNThread;
    const int n2    = tempn / XdlMNThread;

    const int k0    = k / (XdlKThread * KXdlPack);
    const int tempk = k % (XdlKThread * KXdlPack);
    const int k1    = tempk % XdlKThread;
    const int k2    = tempk / XdlKThread;

    const int outputIndex =
        n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
        k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
        k1 * MNXdlPack * KXdlPack * XdlMNThread + n1 * MNXdlPack * KXdlPack +
        k2 * MNXdlPack + n2;

    dtype value{};
    if(n < MN)
    {
        if constexpr(KLast)
            value = src[n * K + k];
        else
            value = src[k * MN + n];
    }
    dst[outputIndex] = value;
}

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
    static constexpr bool DoubleSmemBuffer          = false;

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

namespace ck_tile {

template <class ScaleM = FlatmmScalePointer<-1>, class ScaleN = FlatmmScalePointer<-1>, index_t NumDTensor = 0>
struct GroupedMXFlatmmHostArgs
{
    CK_TILE_HOST explicit GroupedMXFlatmmHostArgs(const void* a_ptr_,
                                                  const void* b_ptr_,
                                                  const std::array<const void*, NumDTensor>& ds_ptr_,
                                                  void* e_ptr_,
                                                  index_t k_batch_,
                                                  index_t M_,
                                                  index_t N_,
                                                  index_t K_,
                                                  index_t stride_A_,
                                                  index_t stride_B_,
                                                  const std::array<index_t, NumDTensor>& stride_Ds_,
                                                  index_t stride_E_,
                                                  ScaleM scale_m_ = ScaleM{},
                                                  ScaleN scale_n_ = ScaleN{})
        : a_ptr(a_ptr_),
          b_ptr(b_ptr_),
          ds_ptr(ds_ptr_),
          e_ptr(e_ptr_),
          k_batch(k_batch_),
          M(M_),
          N(N_),
          K(K_),
          stride_A(stride_A_),
          stride_B(stride_B_),
          stride_Ds(stride_Ds_),
          stride_E(stride_E_),
          scale_m(scale_m_),
          scale_n(scale_n_)
    {
    }

    const void* a_ptr;
    const void* b_ptr;
    const std::array<const void*, NumDTensor> ds_ptr{};
    void* e_ptr;
    index_t k_batch;
    index_t M;
    index_t N;
    index_t K;
    index_t stride_A;
    index_t stride_B;
    const std::array<index_t, NumDTensor> stride_Ds{};
    index_t stride_E;
    ScaleM scale_m;
    ScaleN scale_n;
};

template <class ScaleM = FlatmmScalePointer<-1>, class ScaleN = FlatmmScalePointer<-1>, index_t NumDTensor = 0>
struct FlatmmTransKernelArg
{
    FlatmmKernelArgs<ScaleM, ScaleN, NumDTensor> group_karg;
    index_t block_start;
    index_t block_end;

    FlatmmTransKernelArg() = delete;

    FlatmmTransKernelArg(FlatmmKernelArgs<ScaleM, ScaleN, NumDTensor>&& karg,
                         index_t bl_start,
                         index_t bl_end)
        : group_karg{std::move(karg)}, block_start{bl_start}, block_end{bl_end}
    {
    }
};

template <typename TilePartitioner_, typename MXFlatmmPipeline_, typename EpiloguePipeline_>
struct GroupedMXFlatmmKernel
{
    using Base = MXFlatmmKernel<TilePartitioner_, MXFlatmmPipeline_, EpiloguePipeline_>;

    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using MXFlatmmPipeline = remove_cvref_t<MXFlatmmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using DsDataType       = remove_cvref_t<typename EpiloguePipeline::DsDataType>;
    static constexpr index_t NumDTensor_ = DsDataType::size();

    using OffsetTile1DPartitioner = OffsettedTile1DPartitioner<TilePartitioner>;
    static constexpr index_t kBlockSize = MXFlatmmPipeline::BlockSize;
    static constexpr bool UsePersistentKernel = false;

    CK_TILE_HOST static auto BlockSize() -> dim3
    {
        if(is_wave32()) return dim3(kBlockSize / 2);
        else return dim3(kBlockSize);
    }

    template <class ScaleM, class ScaleN>
    CK_TILE_HOST static auto GridSize(const std::vector<GroupedMXFlatmmHostArgs<ScaleM, ScaleN, NumDTensor_>>& gemm_descs) -> dim3
    {
        index_t grid_size = 0;
        for(const auto& it_desc : gemm_descs)
            grid_size += TilePartitioner::GridSize(it_desc.M, it_desc.N) * it_desc.k_batch;
        return dim3(grid_size, 1, 1);
    }

    template <class ScaleM, class ScaleN>
    CK_TILE_HOST static auto MakeKargs(const std::vector<GroupedMXFlatmmHostArgs<ScaleM, ScaleN, NumDTensor_>>& gemm_descs)
        -> std::vector<FlatmmTransKernelArg<ScaleM, ScaleN, NumDTensor_>>
    {
        std::vector<FlatmmTransKernelArg<ScaleM, ScaleN, NumDTensor_>> out;
        out.reserve(gemm_descs.size());
        index_t grid_size = 0;

        for(const auto& desc : gemm_descs)
        {
            const index_t M = desc.M, N = desc.N, K = desc.K;
            if(M == 0 || N == 0 || K == 0) continue;

            const index_t grid_size_grp = TilePartitioner::GridSize(M, N) * desc.k_batch;
            const index_t block_start = grid_size;
            const index_t block_end   = grid_size + grid_size_grp;
            grid_size += grid_size_grp;

            auto karg = FlatmmKernelArgs<ScaleM, ScaleN, NumDTensor_>{
                desc.a_ptr, desc.b_ptr, desc.ds_ptr, desc.e_ptr, desc.M, desc.N, desc.K,
                desc.stride_A, desc.stride_B, desc.stride_Ds, desc.stride_E, desc.k_batch,
                desc.scale_m, desc.scale_n};
            out.emplace_back(std::move(karg), block_start, block_end);
        }
        return out;
    }

    template <class ScaleM, class ScaleN>
    CK_TILE_HOST static bool IsSupportedArgument(const std::vector<FlatmmTransKernelArg<ScaleM, ScaleN, NumDTensor_>>& kargs)
    {
        for(const auto& karg : kargs)
            if(!Base::IsSupportedArgument(karg.group_karg)) return false;
        return true;
    }

    template <class ScaleM, class ScaleN>
    CK_TILE_DEVICE index_t FindGroupId(const FlatmmTransKernelArg<ScaleM, ScaleN, NumDTensor_>* gemm_desc_ptr,
                                       index_t block_id,
                                       index_t group_count) const
    {
        index_t left = 0, right = group_count, group_id = index_t((left + right) >> 1);
        while((!(block_id >= gemm_desc_ptr[group_id].block_start &&
                 block_id < gemm_desc_ptr[group_id].block_end)) &&
              left <= right)
        {
            if(block_id < gemm_desc_ptr[group_id].block_start) right = group_id;
            else left = group_id;
            group_id = index_t((left + right) >> 1);
        }
        return group_id;
    }

    template <bool U = UsePersistentKernel, typename = std::enable_if_t<!U>>
    CK_TILE_DEVICE void operator()(const void CK_TILE_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                   index_t group_count) const
    {
        using ScaleA_ = FlatmmScalePointer<1, 32, e8m0_bexp_t>;
        using ScaleB_ = FlatmmScalePointer<1, 32, e8m0_bexp_t>;
        using TransArg = FlatmmTransKernelArg<ScaleA_, ScaleB_, NumDTensor_>;

        const index_t block_id = ck_tile::get_block_1d_id();
        const auto gemm_desc_ptr = reinterpret_cast<const TransArg*>(
            cast_pointer_to_generic_address_space(gemm_descs_const));

        const index_t group_id = FindGroupId(gemm_desc_ptr, block_id, group_count);
        const auto& kargs = gemm_desc_ptr[group_id];
        const index_t local_block_id = block_id - kargs.block_start;
        const auto grid_size_2d = TilePartitioner::GridSize(kargs.group_karg.M, kargs.group_karg.N);
        const auto partition_idx = local_block_id % grid_size_2d;
        Base{}(kargs.group_karg, partition_idx);
    }
};

} // namespace ck_tile

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
          bool UsePersistentKernel = true,
          typename CDEElementWise  = ck_tile::element_wise::PassThrough>
float invoke_grouped_mx_flatmm_raw(const std::vector<ck_tile::index_t>& Ms_host,
                                   const std::vector<ck_tile::index_t>& Ns_host,
                                   const std::vector<ck_tile::index_t>& Ks_host,
                                   const std::vector<const void*>& a_ptrs_host,
                                   const std::vector<const void*>& b_ptrs_host,
                                   const std::vector<void*>& c_ptrs_host,
                                   const std::vector<ck_tile::index_t>& stride_A_host,
                                   const std::vector<ck_tile::index_t>& stride_B_host,
                                   const std::vector<ck_tile::index_t>& stride_C_host,
                                   const std::vector<ScaleA>& scale_a_host,
                                   const std::vector<ScaleB>& scale_b_host,
                                   int n_warmup,
                                   int n_repeat)
{
    using FlatmmConfig = typename MXFlatmmArchTraitsT::Config;
    using FlatmmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<FlatmmConfig::M_Tile, FlatmmConfig::N_Tile, FlatmmConfig::K_Tile>,
        ck_tile::sequence<FlatmmConfig::M_Warp, FlatmmConfig::N_Warp, FlatmmConfig::K_Warp>,
        ck_tile::sequence<FlatmmConfig::M_Warp_Tile, FlatmmConfig::N_Warp_Tile, FlatmmConfig::K_Warp_Tile>>;
    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<FlatmmShape,
                                                                       FlatmmConfig::TileParitionerGroupNum,
                                                                       FlatmmConfig::TileParitionerM01>;
    using Traits = ck_tile::TileGemmTraits<FlatmmConfig::kPadM,
                                           FlatmmConfig::kPadN,
                                           FlatmmConfig::kPadK,
                                           ALayout,
                                           BLayout,
                                           CLayout,
                                           FlatmmConfig::NumWaveGroups>;
    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, FlatmmShape, Traits>;
    using BaseFlatmmPipeline = ck_tile::BaseFlatmmPipelineAGmemBGmemCRegV1<GemmPipelineProblem>;

    if(Ks_host.empty()) throw std::runtime_error("No active groups");
    const ck_tile::index_t K0 = Ks_host.front();
    for(std::size_t i = 1; i < Ks_host.size(); ++i)
        if(Ks_host[i] != K0) throw std::runtime_error("Grouped MX flatmm requires uniform K");

    const ck_tile::index_t k_grain  = FlatmmConfig::K_Tile;
    const ck_tile::index_t k_split  = (K0 + k_grain - 1) / k_grain * k_grain;
    const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(k_split);
    const bool has_hot_loop         = BaseFlatmmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseFlatmmPipeline::GetBlockLoopTailNum(num_loop);

    return BaseFlatmmPipeline::template TailHandler<true>(
        [&](auto has_hot_loop_, auto tail_num_) {
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_num_v = tail_num_.value;

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
            using MXPipelineProblem = ck_tile::MXFlatmmPipelineProblem<ADataType, BDataType, AccDataType,
                                                                       FlatmmShape, MXGemmTraits,
                                                                       FlatmmConfig::Scheduler,
                                                                       has_hot_loop_v, tail_num_v>;
            using MXFlatmmPipeline = typename MXFlatmmArchTraitsT::template MXFlatmmPipeline<MXPipelineProblem>;
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
            using UnderlyingKernel = ck_tile::MXFlatmmKernel<TilePartitioner, MXFlatmmPipeline, GemmEpilogue>;
            using GroupedKernel = ck_tile::GroupedMXFlatmmKernel<TilePartitioner, MXFlatmmPipeline, GemmEpilogue>;
            using Desc = ck_tile::GroupedMXFlatmmHostArgs<ScaleA, ScaleB, 0>;

            std::vector<Desc> descs;
            descs.reserve(Ms_host.size());

            for(std::size_t i = 0; i < Ms_host.size(); ++i)
            {
                ck_tile::FlatmmKernelArgs<ScaleA, ScaleB, 0> impl_kargs{
                    a_ptrs_host[i], b_ptrs_host[i], {}, c_ptrs_host[i],
                    Ms_host[i], Ns_host[i], Ks_host[i],
                    stride_A_host[i], stride_B_host[i], {}, stride_C_host[i], 1,
                    scale_a_host[i], scale_b_host[i]};
                if(!UnderlyingKernel::IsSupportedArgument(impl_kargs))
                    throw std::runtime_error("Grouped MX flatmm: unsupported arguments");

                descs.emplace_back(a_ptrs_host[i], b_ptrs_host[i], std::array<const void*,0>{},
                                   c_ptrs_host[i], 1, Ms_host[i], Ns_host[i], Ks_host[i],
                                   stride_A_host[i], stride_B_host[i], std::array<ck_tile::index_t,0>{},
                                   stride_C_host[i], scale_a_host[i], scale_b_host[i]);
            }

            auto kargs = GroupedKernel::MakeKargs(descs);
            if(!GroupedKernel::IsSupportedArgument(kargs))
                throw std::runtime_error("Grouped MX flatmm: unsupported grouped arguments");

            ck_tile::DeviceMem ws_dev(kargs.size() * sizeof(typename decltype(kargs)::value_type));
            ws_dev.ToDevice(kargs.data());

            const dim3 grids  = GroupedKernel::GridSize(descs);
            const dim3 blocks = GroupedKernel::BlockSize();
            const auto d_workspace_const = ck_tile::cast_pointer_to_constant_address_space(ws_dev.GetDeviceBuffer());

            const auto kernel = ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(
                GroupedKernel{}, grids, blocks, 0, d_workspace_const, static_cast<ck_tile::index_t>(kargs.size()));

            auto clear_gemm_output = [&]() {};
            return ck_tile::launch_kernel_time_mask(
                ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat, true, true, 50},
                clear_gemm_output,
                kernel);
        },
        has_hot_loop,
        tail_num);
}

struct TimingSummary
{
    float weight_shuffle_ms = 0.0f;
    float a_scale_shuffle_ms = 0.0f;
    float b_scale_shuffle_ms = 0.0f;
    float preprocess_total_ms = 0.0f;
    float kernel_ms = 0.0f;
    float e2e_sum_ms = 0.0f;
};

template <typename T>
float launch_weight_preshuffle(const T* src, T* dst, int K, int N, hipStream_t stream, int warmup, int repeat)
{
    constexpr int threads = 256;
    const int total = N * (K / ck_tile::numeric_traits<T>::PackedSize);
    const int blocks = (total + threads - 1) / threads;
    return time_hip_events(stream, warmup, repeat, [&]{
        hipLaunchKernelGGL((preshuffle_weight_kernel<T, 16>), dim3(blocks), dim3(threads), 0, stream, src, dst, K, N);
        hip_check(hipGetLastError(), "launch preshuffle_weight_kernel");
    });
}

template <typename T, bool KLast>
float launch_scale_preshuffle(const T* src, T* dst, int MN, int K, hipStream_t stream, int warmup, int repeat)
{
    constexpr int threads = 256;
    constexpr int XdlMNThread = 16;
    constexpr int MNXdlPack = 2;
    const int MN_padded = ((MN + XdlMNThread * MNXdlPack - 1) / (XdlMNThread * MNXdlPack)) * (XdlMNThread * MNXdlPack);
    const int total = MN_padded * K;
    const int blocks = (total + threads - 1) / threads;
    return time_hip_events(stream, warmup, repeat, [&]{
        hipLaunchKernelGGL((preshuffle_scale_kernel<T, KLast, XdlMNThread>), dim3(blocks), dim3(threads), 0, stream, src, dst, MN, K);
        hip_check(hipGetLastError(), "launch preshuffle_scale_kernel");
    });
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "256", "rows per expert")
        .insert("n", "4096", "n dimension")
        .insert("k", "4096", "k dimension")
        .insert("num_experts", "8", "number of experts/groups")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("v", "1", "0: no validation, 1: host validation")
        .insert("warmup", "20", "warmup iterations")
        .insert("repeat", "50", "benchmark iterations")
        .insert("init", "0", "0: random, 1: constant(1), 2: random data + const scales");
    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K, const float max_accumulated_value)
{
    using ComputeType = std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(K);
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(max_accumulated_value, K);
    return ck_tile::make_tuple(rtol, atol);
}

int main(int argc, char* argv[])
{
    auto [ok, arg_parser] = create_args(argc, argv);
    if(!ok) return EXIT_FAILURE;

    using ADataType   = ck_tile::fp8_t;
    using BDataType   = ck_tile::fp8_t;
    using CDataType   = ck_tile::half_t;
    using AccDataType = float;
    using ScaleType   = ck_tile::e8m0_bexp_t;
    using Row         = ck_tile::tensor_layout::gemm::RowMajor;
    using Col         = ck_tile::tensor_layout::gemm::ColumnMajor;
    using FlatmmCfg   = typename MXTraits::Config;

    constexpr int ScaleGranularityM = 1;
    constexpr int ScaleGranularityN = 1;
    constexpr int ScaleGranularityK = 32;

    try
    {
        const ck_tile::index_t m_chunk = arg_parser.get_int("m");
        const ck_tile::index_t N       = arg_parser.get_int("n");
        const ck_tile::index_t K       = arg_parser.get_int("k");
        const ck_tile::index_t E       = arg_parser.get_int("num_experts");
        const int warmup               = arg_parser.get_int("warmup");
        const int repeat               = arg_parser.get_int("repeat");
        const int init_method          = arg_parser.get_int("init");
        const int validate             = arg_parser.get_int("v");

        if(K % ScaleGranularityK != 0) throw std::runtime_error("K must be divisible by 32 for MXFP8");
        if(E <= 0 || m_chunk <= 0) throw std::runtime_error("m and num_experts must be positive");

        const ck_tile::index_t M_total = m_chunk * E;
        const ck_tile::index_t QK      = K / ScaleGranularityK;
        const ck_tile::index_t m_pad   = round_up(m_chunk, FlatmmCfg::M_Tile);

        ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
        ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
        ck_tile::index_t stride_C = arg_parser.get_int("stride_c");

        stride_A = ck_tile::get_default_stride(m_pad, K, stride_A, is_row_major(Row{}));
        stride_B = ck_tile::get_default_stride(K, N, stride_B, is_row_major(Col{}));
        stride_C = ck_tile::get_default_stride(m_pad, N, stride_C, is_row_major(Row{}));

        const ck_tile::index_t scale_a_stride = QK;
        const ck_tile::index_t scale_b_stride = ck_tile::get_default_stride(QK, N, 0, is_row_major(Col{}));

        ck_tile::HostTensor<ADataType> a_total(
            ck_tile::host_tensor_descriptor(M_total, K, K, is_row_major(Row{})));
        ck_tile::HostTensor<ScaleType> a_scales_total(
            ck_tile::host_tensor_descriptor(M_total, QK, QK, is_row_major(Row{})));
        ck_tile::HostTensor<BDataType> b_all(
            ck_tile::host_tensor_descriptor(K, N * E, K, is_row_major(Col{})));
        ck_tile::HostTensor<ScaleType> b_scales_all(
            ck_tile::host_tensor_descriptor(QK, N * E, QK, is_row_major(Col{})));

        if(init_method == 0)
        {
            ck_tile::FillUniformDistribution<ADataType>{0.0f, 1.0f}(a_total);
            ck_tile::FillUniformDistribution<BDataType>{-0.5f, 0.5f}(b_all);
            ck_tile::FillUniformDistribution<ScaleType>{-2.f, 2.f}(a_scales_total);
            ck_tile::FillUniformDistribution<ScaleType>{-2.f, 2.f}(b_scales_all);
        }
        else if(init_method == 1)
        {
            fill_tensor_constant(a_total, 1.0f);
            fill_tensor_constant(b_all, 1.0f);
            fill_tensor_constant(a_scales_total, 1.0f);
            fill_tensor_constant(b_scales_all, 1.0f);
        }
        else
        {
            ck_tile::FillUniformDistribution<ADataType>{0.0f, 1.0f}(a_total);
            ck_tile::FillUniformDistribution<BDataType>{-0.5f, 0.5f}(b_all);
            fill_tensor_constant(a_scales_total, 1.0f);
            fill_tensor_constant(b_scales_all, 1.0f);
        }

        struct GroupMeta { ck_tile::index_t row_offset, m_real, m_pad; };
        std::vector<GroupMeta> metas;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_dev_bufs, b_raw_dev_bufs, b_shuf_dev_bufs, c_dev_bufs, a_scale_raw_dev_bufs, a_scale_shuf_dev_bufs, b_scale_raw_dev_bufs, b_scale_shuf_dev_bufs;
        std::vector<ck_tile::index_t> Ms_host, Ns_host, Ks_host, stride_A_host, stride_B_host, stride_C_host;
        std::vector<const void*> a_ptrs_host, b_ptrs_host;
        std::vector<void*> c_ptrs_host;
        std::vector<ck_tile::FlatmmScalePointer<ScaleGranularityM, ScaleGranularityK, ScaleType>> scale_a_host;
        std::vector<ck_tile::FlatmmScalePointer<ScaleGranularityN, ScaleGranularityK, ScaleType>> scale_b_host;

        TimingSummary timing;
        hipStream_t stream = nullptr;

        for(ck_tile::index_t expert = 0; expert < E; ++expert)
        {
            const ck_tile::index_t row_offset = expert * m_chunk;

            ck_tile::HostTensor<ADataType> a_group(
                ck_tile::host_tensor_descriptor(m_pad, K, stride_A, is_row_major(Row{})));
            ck_tile::HostTensor<ScaleType> a_scale_group(
                ck_tile::host_tensor_descriptor(m_pad, QK, scale_a_stride, is_row_major(Row{})));
            ck_tile::HostTensor<BDataType> b_group(
                ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(Col{})));
            ck_tile::HostTensor<ScaleType> b_scale_group(
                ck_tile::host_tensor_descriptor(QK, N, scale_b_stride, is_row_major(Col{})));

            fill_tensor_constant(a_group, 0.0f);
            fill_tensor_constant(a_scale_group, 1.0f);

            for(ck_tile::index_t i = 0; i < m_chunk; ++i)
                for(ck_tile::index_t k = 0; k < K; ++k)
                    a_group(i, k) = a_total(row_offset + i, k);

            for(ck_tile::index_t i = 0; i < m_chunk; ++i)
                for(ck_tile::index_t q = 0; q < QK; ++q)
                    a_scale_group(i, q) = a_scales_total(row_offset + i, q);

            for(ck_tile::index_t k = 0; k < K; ++k)
                for(ck_tile::index_t n = 0; n < N; ++n)
                    b_group(k, n) = b_all(k, expert * N + n);

            for(ck_tile::index_t q = 0; q < QK; ++q)
                for(ck_tile::index_t n = 0; n < N; ++n)
                    b_scale_group(q, n) = b_scales_all(q, expert * N + n);

            const std::size_t b_raw_bytes = static_cast<std::size_t>(K) * static_cast<std::size_t>(N) * sizeof(BDataType);
            const std::size_t b_shuf_bytes = static_cast<std::size_t>(K) * static_cast<std::size_t>(N) * sizeof(BDataType);
            const std::size_t a_scale_raw_bytes = static_cast<std::size_t>(m_pad) * static_cast<std::size_t>(QK) * sizeof(ScaleType);
            const std::size_t a_scale_shuf_bytes = static_cast<std::size_t>(round_up(m_pad, 32)) * static_cast<std::size_t>(QK) * sizeof(ScaleType);
            const std::size_t b_scale_raw_bytes = static_cast<std::size_t>(QK) * static_cast<std::size_t>(N) * sizeof(ScaleType);
            const std::size_t b_scale_shuf_bytes = static_cast<std::size_t>(round_up(N, 32)) * static_cast<std::size_t>(QK) * sizeof(ScaleType);

            a_dev_bufs.push_back(std::make_unique<ck_tile::DeviceMem>(a_group.get_element_space_size_in_bytes()));
            b_raw_dev_bufs.push_back(std::make_unique<ck_tile::DeviceMem>(b_raw_bytes));
            b_shuf_dev_bufs.push_back(std::make_unique<ck_tile::DeviceMem>(b_shuf_bytes));
            c_dev_bufs.push_back(std::make_unique<ck_tile::DeviceMem>(m_pad * N * sizeof(CDataType)));
            a_scale_raw_dev_bufs.push_back(std::make_unique<ck_tile::DeviceMem>(a_scale_raw_bytes));
            a_scale_shuf_dev_bufs.push_back(std::make_unique<ck_tile::DeviceMem>(a_scale_shuf_bytes));
            b_scale_raw_dev_bufs.push_back(std::make_unique<ck_tile::DeviceMem>(b_scale_raw_bytes));
            b_scale_shuf_dev_bufs.push_back(std::make_unique<ck_tile::DeviceMem>(b_scale_shuf_bytes));

            a_dev_bufs.back()->ToDevice(a_group.data());
            b_raw_dev_bufs.back()->ToDevice(b_group.data());
            c_dev_bufs.back()->SetZero();
            a_scale_raw_dev_bufs.back()->ToDevice(a_scale_group.data());
            b_scale_raw_dev_bufs.back()->ToDevice(b_scale_group.data());

            timing.weight_shuffle_ms += launch_weight_preshuffle(
                static_cast<const BDataType*>(b_raw_dev_bufs.back()->GetDeviceBuffer()),
                static_cast<BDataType*>(b_shuf_dev_bufs.back()->GetDeviceBuffer()),
                static_cast<int>(K), static_cast<int>(N), stream, warmup, repeat);

            timing.a_scale_shuffle_ms += launch_scale_preshuffle<ScaleType, true>(
                static_cast<const ScaleType*>(a_scale_raw_dev_bufs.back()->GetDeviceBuffer()),
                static_cast<ScaleType*>(a_scale_shuf_dev_bufs.back()->GetDeviceBuffer()),
                static_cast<int>(m_pad), static_cast<int>(QK), stream, warmup, repeat);

            timing.b_scale_shuffle_ms += launch_scale_preshuffle<ScaleType, false>(
                static_cast<const ScaleType*>(b_scale_raw_dev_bufs.back()->GetDeviceBuffer()),
                static_cast<ScaleType*>(b_scale_shuf_dev_bufs.back()->GetDeviceBuffer()),
                static_cast<int>(N), static_cast<int>(QK), stream, warmup, repeat);

            metas.push_back({row_offset, m_chunk, m_pad});
            Ms_host.push_back(m_pad);
            Ns_host.push_back(N);
            Ks_host.push_back(K);
            a_ptrs_host.push_back(a_dev_bufs.back()->GetDeviceBuffer());
            b_ptrs_host.push_back(b_shuf_dev_bufs.back()->GetDeviceBuffer());
            c_ptrs_host.push_back(c_dev_bufs.back()->GetDeviceBuffer());
            stride_A_host.push_back(stride_A);
            stride_B_host.push_back(stride_B);
            stride_C_host.push_back(stride_C);
            scale_a_host.push_back(
                ck_tile::FlatmmScalePointer<ScaleGranularityM, ScaleGranularityK, ScaleType>{
                    static_cast<ScaleType*>(a_scale_shuf_dev_bufs.back()->GetDeviceBuffer()), m_pad});
            scale_b_host.push_back(
                ck_tile::FlatmmScalePointer<ScaleGranularityN, ScaleGranularityK, ScaleType>{
                    static_cast<ScaleType*>(b_scale_shuf_dev_bufs.back()->GetDeviceBuffer()), N});
        }

        timing.preprocess_total_ms = timing.weight_shuffle_ms + timing.a_scale_shuffle_ms + timing.b_scale_shuffle_ms;

        timing.kernel_ms = invoke_grouped_mx_flatmm_raw<MXTraits,
                                                        ADataType,
                                                        BDataType,
                                                        ck_tile::tuple<>,
                                                        AccDataType,
                                                        CDataType,
                                                        Row,
                                                        Col,
                                                        ck_tile::tuple<>,
                                                        Row>(Ms_host, Ns_host, Ks_host, a_ptrs_host, b_ptrs_host,
                                                             c_ptrs_host, stride_A_host, stride_B_host, stride_C_host,
                                                             scale_a_host, scale_b_host, warmup, repeat);

        timing.e2e_sum_ms = timing.preprocess_total_ms + timing.kernel_ms;

        constexpr int APackedSize = ck_tile::numeric_traits<ADataType>::PackedSize;
        constexpr int BPackedSize = ck_tile::numeric_traits<BDataType>::PackedSize;
        const std::size_t flop = std::size_t(2) * M_total * N * K + std::size_t(2) * M_total * N * K / 32;
        const std::size_t num_byte = sizeof(ADataType) * M_total * K / APackedSize +
                                     sizeof(BDataType) * E * N * K / BPackedSize +
                                     sizeof(CDataType) * M_total * N +
                                     sizeof(ScaleType) * M_total * K / 32 +
                                     sizeof(ScaleType) * E * N * K / 32;

        const float kernel_tflops = static_cast<float>(flop) / 1.0e9f / timing.kernel_ms;
        const float e2e_tflops = static_cast<float>(flop) / 1.0e9f / timing.e2e_sum_ms;
        const float kernel_gbps = static_cast<float>(num_byte) / 1.0e6f / timing.kernel_ms;
        const float e2e_gbps = static_cast<float>(num_byte) / 1.0e6f / timing.e2e_sum_ms;

        std::cout << "Timing breakdown (device-side)\n"
                  << "  weight_preshuffle_ms: " << timing.weight_shuffle_ms << "\n"
                  << "  a_scale_preshuffle_ms: " << timing.a_scale_shuffle_ms << "\n"
                  << "  b_scale_preshuffle_ms: " << timing.b_scale_shuffle_ms << "\n"
                  << "  preprocess_total_ms: " << timing.preprocess_total_ms << "\n"
                  << "  kernel_ms: " << timing.kernel_ms << "\n"
                  << "  e2e_sum_ms: " << timing.e2e_sum_ms << "\n";

        std::cout << "Performance summary\n"
                  << "  kernel_tflops: " << kernel_tflops << "\n"
                  << "  e2e_tflops: " << e2e_tflops << "\n"
                  << "  kernel_gbps: " << kernel_gbps << "\n"
                  << "  e2e_gbps: " << e2e_gbps << "\n";

        bool pass = true;
        if(validate == 1)
        {
            ck_tile::HostTensor<CDataType> c_result(
                ck_tile::host_tensor_descriptor(M_total, N, N, is_row_major(Row{})));
            c_result.SetZero();

            for(std::size_t g = 0; g < metas.size(); ++g)
            {
                ck_tile::HostTensor<CDataType> c_group(
                    ck_tile::host_tensor_descriptor(metas[g].m_pad, N, stride_C, is_row_major(Row{})));
                c_dev_bufs[g]->FromDevice(c_group.data());

                for(ck_tile::index_t i = 0; i < metas[g].m_real; ++i)
                    for(ck_tile::index_t n = 0; n < N; ++n)
                        c_result(metas[g].row_offset + i, n) = c_group(i, n);
            }

            ck_tile::HostTensor<CDataType> c_ref(
                ck_tile::host_tensor_descriptor(M_total, N, N, is_row_major(Row{})));
            c_ref.SetZero();

            for(ck_tile::index_t expert = 0; expert < E; ++expert)
            {
                ck_tile::HostTensor<ADataType> a_ref(
                    ck_tile::host_tensor_descriptor(m_chunk, K, K, is_row_major(Row{})));
                ck_tile::HostTensor<BDataType> b_ref(
                    ck_tile::host_tensor_descriptor(K, N, K, is_row_major(Col{})));
                ck_tile::HostTensor<ScaleType> a_scale_ref(
                    ck_tile::host_tensor_descriptor(m_chunk, QK, QK, is_row_major(Row{})));
                ck_tile::HostTensor<ScaleType> b_scale_ref(
                    ck_tile::host_tensor_descriptor(QK, N, QK, is_row_major(Col{})));
                ck_tile::HostTensor<CDataType> c_group_ref(
                    ck_tile::host_tensor_descriptor(m_chunk, N, N, is_row_major(Row{})));
                c_group_ref.SetZero();

                const ck_tile::index_t row_offset = expert * m_chunk;
                for(ck_tile::index_t i = 0; i < m_chunk; ++i)
                    for(ck_tile::index_t k = 0; k < K; ++k)
                        a_ref(i, k) = a_total(row_offset + i, k);
                for(ck_tile::index_t i = 0; i < m_chunk; ++i)
                    for(ck_tile::index_t q = 0; q < QK; ++q)
                        a_scale_ref(i, q) = a_scales_total(row_offset + i, q);
                for(ck_tile::index_t k = 0; k < K; ++k)
                    for(ck_tile::index_t n = 0; n < N; ++n)
                        b_ref(k, n) = b_all(k, expert * N + n);
                for(ck_tile::index_t q = 0; q < QK; ++q)
                    for(ck_tile::index_t n = 0; n < N; ++n)
                        b_scale_ref(q, n) = b_scales_all(q, expert * N + n);

                ck_tile::reference_mx_gemm<ADataType, BDataType, ScaleType, AccDataType, CDataType>(
                    a_ref, b_ref, c_group_ref, a_scale_ref, b_scale_ref);

                for(ck_tile::index_t i = 0; i < m_chunk; ++i)
                    for(ck_tile::index_t n = 0; n < N; ++n)
                        c_ref(row_offset + i, n) = c_group_ref(i, n);
            }

            auto [rtol, atol] = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                K, 2.0f * static_cast<float>(K));

            pass = ck_tile::check_err(c_result, c_ref, "Error: Incorrect results!", rtol, atol);
            std::cout << "Relative error threshold: " << rtol
                      << " Absolute error threshold: " << atol << std::endl;
            std::cout << "The CPU verification result is: "
                      << (pass ? "correct" : "fail") << std::endl;
        }

        return pass ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
