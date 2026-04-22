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
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"
#include "ck_tile/ops/gemm_mx/kernel/gemm_mx_kernel.hpp"
#include "ck_tile/ops/gemm_mx/kernel/scale_pointer.hpp"
#include "ck_tile/ops/gemm_mx/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp"

template <typename Layout>
static constexpr inline auto is_row_major(Layout)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<Layout>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename ScaleM, typename ScaleN>
struct MXGemmHostArgs : ck_tile::UniversalGemmHostArgs<1, 1, 0>
{
    using Base = ck_tile::UniversalGemmHostArgs<1, 1, 0>;

    MXGemmHostArgs(const void* a_ptr,
                   const void* b_ptr,
                   void* c_ptr_,
                   ck_tile::index_t k_batch_,
                   ck_tile::index_t M_,
                   ck_tile::index_t N_,
                   ck_tile::index_t K_,
                   ck_tile::index_t stride_A_,
                   ck_tile::index_t stride_B_,
                   ck_tile::index_t stride_C_,
                   ScaleM scale_m_,
                   ScaleN scale_n_)
        : Base({a_ptr},
               {b_ptr},
               {},
               c_ptr_,
               k_batch_,
               M_,
               N_,
               K_,
               {stride_A_},
               {stride_B_},
               {},
               stride_C_),
          scale_m(scale_m_),
          scale_n(scale_n_)
    {
    }

    ScaleM scale_m;
    ScaleN scale_n;
};

struct MxGemmConfig
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 512;

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
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr bool Preshuffle                = false;

    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = false;
};

struct MXfp8_GemmConfig_256x64x128 : MxGemmConfig
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 128;
};

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ScaleM,
          typename ScaleN,
          bool persistent,
          bool Splitk>
float mx_gemm_calc(const MXGemmHostArgs<ScaleM, ScaleN>& args, const ck_tile::stream_config& s)
{
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;

    using MXGemmTraits = ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                                          GemmConfig::kPadN,
                                                          GemmConfig::kPadK,
                                                          GemmConfig::DoubleSmemBuffer,
                                                          ALayout,
                                                          BLayout,
                                                          CLayout,
                                                          GemmConfig::TransposeC,
                                                          GemmConfig::UseStructuredSparsity,
                                                          persistent,
                                                          GemmConfig::NumWaveGroups,
                                                          GemmConfig::Preshuffle>;

    using ComputeDataType = ADataType;
    using MXPipelineProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                    BDataType,
                                                                    AccDataType,
                                                                    GemmShape,
                                                                    MXGemmTraits,
                                                                    GemmConfig::Scheduler>;

    using MXGemmPipeline = ck_tile::MXGemmPipelineAgBgCrCompAsync<MXPipelineProblem>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                   GemmConfig::TileParitionerGroupNum,
                                                   GemmConfig::TileParitionerM01>;

    using GemmEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<ComputeDataType,
                                         ComputeDataType,
                                         ck_tile::tuple<>,
                                         AccDataType,
                                         CDataType,
                                         ck_tile::tuple<>,
                                         CLayout,
                                         ck_tile::element_wise::PassThrough,
                                         TilePartitioner::MPerBlock,
                                         TilePartitioner::NPerBlock,
                                         GemmConfig::M_Warp,
                                         GemmConfig::N_Warp,
                                         GemmConfig::M_Warp_Tile,
                                         GemmConfig::N_Warp_Tile,
                                         GemmConfig::K_Warp_Tile,
                                         MXPipelineProblem::TransposeC>>;

    using Kernel = ck_tile::MXGemmKernel<TilePartitioner, MXGemmPipeline, GemmEpilogue>;

    auto kargs = Kernel::MakeKernelArgs(std::array<const void*, 1>{args.as_ptr},
                                        std::array<const void*, 1>{args.bs_ptr},
                                        std::array<const void*, 0>{},
                                        args.e_ptr,
                                        args.k_batch,
                                        args.M,
                                        args.N,
                                        args.K,
                                        std::array<ck_tile::index_t, 1>{args.stride_As},
                                        std::array<ck_tile::index_t, 1>{args.stride_Bs},
                                        std::array<ck_tile::index_t, 0>{},
                                        args.stride_E,
                                        args.scale_m,
                                        args.scale_n);

    if(!Kernel::IsSupportedArgument(kargs))
    {
        throw std::runtime_error("MX GEMM: unsupported shape/configuration.");
    }

    const auto kernel = ck_tile::make_kernel<Kernel::kBlockPerCu>(
        Kernel{}, Kernel::GridSize(kargs), Kernel::BlockSize(), 0, kargs);

    auto clear_gemm_output = [&]() {
        if(args.k_batch > 1)
            (void)hipMemsetAsync(args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_);
    };

    return ck_tile::launch_kernel_time_mask(s, clear_gemm_output, kernel);
}

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ScaleM,
          typename ScaleN,
          bool UsePersistentKernel = false>
float invoke_mx_gemm(ck_tile::DeviceMem& a_dev_buf,
                     ck_tile::DeviceMem& b_dev_buf,
                     ck_tile::DeviceMem& c_dev_buf,
                     ck_tile::index_t M,
                     ck_tile::index_t N,
                     ck_tile::index_t K,
                     ck_tile::index_t stride_A,
                     ck_tile::index_t stride_B,
                     ck_tile::index_t stride_C,
                     ck_tile::index_t kbatch,
                     ScaleM scale_m,
                     ScaleN scale_n,
                     int n_warmup,
                     int n_repeat)
{
    MXGemmHostArgs<ScaleM, ScaleN> args(a_dev_buf.GetDeviceBuffer(),
                                        b_dev_buf.GetDeviceBuffer(),
                                        c_dev_buf.GetDeviceBuffer(),
                                        kbatch,
                                        M,
                                        N,
                                        K,
                                        stride_A,
                                        stride_B,
                                        stride_C,
                                        scale_m,
                                        scale_n);

    auto invoke_splitk_path = [&](auto split_k_) {
        return mx_gemm_calc<GemmConfig,
                            ADataType,
                            BDataType,
                            AccDataType,
                            CDataType,
                            ALayout,
                            BLayout,
                            CLayout,
                            ScaleM,
                            ScaleN,
                            UsePersistentKernel,
                            split_k_.value>(
            args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat, true, true, 50});
    };

    return (args.k_batch == 1) ? invoke_splitk_path(std::false_type{})
                               : invoke_splitk_path(std::true_type{});
}

// Pack [MN, K/32] e8m0_t scales into packed int32 layout, same as single standalone.
template <ck_tile::index_t MNPack      = 2,
          ck_tile::index_t KPack       = 2,
          ck_tile::index_t XdlMNThread = 16,
          ck_tile::index_t XdlKThread  = 4>
auto packScalesMNxK(const ck_tile::HostTensor<ck_tile::e8m0_t>& src, bool kLast)
{
    auto src_lengths                    = src.get_lengths();
    const ck_tile::index_t MN           = kLast ? src_lengths[0] : src_lengths[1];
    const ck_tile::index_t K_scale      = kLast ? src_lengths[1] : src_lengths[0];
    const ck_tile::index_t MN_packed    = MN / MNPack;
    const ck_tile::index_t K_packed     = K_scale / KPack;
    const ck_tile::index_t total_packed = MN_packed * K_packed;

    std::vector<int32_t> packed(total_packed);

    for(ck_tile::index_t packed_mn = 0; packed_mn < MN_packed; packed_mn++)
    {
        for(ck_tile::index_t packed_k = 0; packed_k < K_packed; packed_k++)
        {
            int32_t val               = 0;
            ck_tile::index_t mn_lane  = packed_mn % XdlMNThread;
            ck_tile::index_t mn_group = packed_mn / XdlMNThread;
            ck_tile::index_t k_lane   = packed_k % XdlKThread;
            ck_tile::index_t k_group  = packed_k / XdlKThread;
            for(ck_tile::index_t ik = 0; ik < KPack; ik++)
            {
                for(ck_tile::index_t imn = 0; imn < MNPack; imn++)
                {
                    ck_tile::index_t byteIdx = ik * MNPack + imn;
                    ck_tile::index_t orig_mn =
                        mn_group * XdlMNThread * MNPack + imn * XdlMNThread + mn_lane;
                    ck_tile::index_t orig_k =
                        k_group * XdlKThread * KPack + ik * XdlKThread + k_lane;

                    ck_tile::e8m0_t v = kLast ? src(orig_mn, orig_k) : src(orig_k, orig_mn);
                    val |= (static_cast<int32_t>(v.get()) << (byteIdx * 8));
                }
            }
            packed[packed_mn * K_packed + packed_k] = val;
        }
    }
    return packed;
}

namespace ck_tile {

template <typename ScaleM, typename ScaleN, index_t NumDTensor = 0>
struct MXGroupedGemmHostArgs : public GroupedGemmHostArgs<NumDTensor>
{
    using Base = GroupedGemmHostArgs<NumDTensor>;

    CK_TILE_HOST explicit MXGroupedGemmHostArgs(const void* a_ptr_,
                                                ScaleM scale_m_,
                                                const void* b_ptr_,
                                                ScaleN scale_n_,
                                                const std::array<const void*, NumDTensor>& ds_ptr_,
                                                void* e_ptr_,
                                                index_t k_batch_,
                                                index_t M_,
                                                index_t N_,
                                                index_t K_,
                                                index_t stride_A_,
                                                index_t stride_B_,
                                                const std::array<index_t, NumDTensor>& stride_Ds_,
                                                index_t stride_E_)
        : Base(a_ptr_,
               b_ptr_,
               ds_ptr_,
               e_ptr_,
               k_batch_,
               M_,
               N_,
               K_,
               stride_A_,
               stride_B_,
               stride_Ds_,
               stride_E_),
          scale_m(scale_m_),
          scale_n(scale_n_)
    {
    }

    ScaleM scale_m;
    ScaleN scale_n;
};

template <typename Karg>
struct MXGemmTransKernelArg
{
    Karg group_karg;
    index_t block_start;
    index_t block_end;

    CK_TILE_HOST MXGemmTransKernelArg(Karg&& karg_, index_t block_start_, index_t block_end_)
        : group_karg(std::move(karg_)), block_start(block_start_), block_end(block_end_)
    {
    }
};

template <typename TilePartitioner_,
          typename MXGemmPipeline_,
          typename EpiloguePipeline_,
          typename ScaleM_,
          typename ScaleN_,
          index_t NumDTensor_ = 0>
struct MXGroupedGemmKernel
{
    using Base            = ck_tile::MXGemmKernel<TilePartitioner_, MXGemmPipeline_, EpiloguePipeline_>;
    using TilePartitioner = remove_cvref_t<TilePartitioner_>;
    using HostArgs        = MXGroupedGemmHostArgs<ScaleM_, ScaleN_, NumDTensor_>;
    using KernelArgs      = typename Base::template KernelArgs<ScaleM_, ScaleN_>;
    using TransKernelArg  = MXGemmTransKernelArg<KernelArgs>;

    static constexpr index_t kBlockSize = Base::KernelBlockSize;
    static constexpr index_t kBlockPerCu = Base::kBlockPerCu;

    CK_TILE_HOST static auto GetWorkSpaceSize(index_t group_count) -> std::size_t
    {
        return group_count * sizeof(TransKernelArg);
    }

    CK_TILE_HOST static auto BlockSize() -> dim3 { return Base::BlockSize(); }

    CK_TILE_HOST static auto GridSize(const std::vector<HostArgs>& gemm_descs) -> dim3
    {
        index_t grid_size = 0;
        for(const auto& d : gemm_descs)
            grid_size += TilePartitioner::GridSize(d.M, d.N);
        return dim3(grid_size, 1, 1);
    }

    CK_TILE_HOST static auto MakeKargs(const std::vector<HostArgs>& gemm_descs)
    {
        std::vector<TransKernelArg> out;
        out.reserve(gemm_descs.size());

        index_t grid_size = 0;
        for(const auto& g : gemm_descs)
        {
            if(g.M == 0 || g.N == 0 || g.K == 0)
                continue;

            const index_t grid_size_grp = TilePartitioner::GridSize(g.M, g.N);
            const index_t block_start   = grid_size;
            const index_t block_end     = grid_size + grid_size_grp;
            grid_size += grid_size_grp;

            auto karg = Base::MakeKernelArgs(std::array<const void*, 1>{g.a_ptr},
                                             std::array<const void*, 1>{g.b_ptr},
                                             std::array<const void*, NumDTensor_>{g.ds_ptr},
                                             g.e_ptr,
                                             g.k_batch,
                                             g.M,
                                             g.N,
                                             g.K,
                                             std::array<index_t, 1>{g.stride_A},
                                             std::array<index_t, 1>{g.stride_B},
                                             g.stride_Ds,
                                             g.stride_E,
                                             g.scale_m,
                                             g.scale_n);
            out.emplace_back(std::move(karg), block_start, block_end);
        }
        return out;
    }

    CK_TILE_HOST static bool IsSupportedArgument(const std::vector<TransKernelArg>& kargs)
    {
        for(const auto& k : kargs)
            if(!Base::IsSupportedArgument(k.group_karg))
                return false;
        return true;
    }

    CK_TILE_DEVICE index_t FindGroupId(const TransKernelArg* ptr,
                                       index_t block_id,
                                       index_t group_count) const
    {
        index_t left = 0;
        index_t right = group_count;
        index_t group_id = (left + right) >> 1;

        while((!(block_id >= ptr[group_id].block_start && block_id < ptr[group_id].block_end)) &&
              left <= right)
        {
            if(block_id < ptr[group_id].block_start)
                right = group_id;
            else
                left = group_id;
            group_id = (left + right) >> 1;
        }
        return group_id;
    }

    CK_TILE_DEVICE void operator()(const void CK_TILE_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                   index_t group_count) const
    {
        const index_t block_id = get_block_1d_id();
        const auto ptr = reinterpret_cast<const TransKernelArg*>(
            cast_pointer_to_generic_address_space(gemm_descs_const));

        const index_t group_id = FindGroupId(ptr, block_id, group_count);
        const auto& kargs = ptr[group_id];
        const index_t local_partition_idx = block_id - kargs.block_start;
        Base{}(kargs.group_karg, local_partition_idx);
    }
};

} // namespace ck_tile

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "256", "rows per expert")
        .insert("n", "4096", "n dimension")
        .insert("k", "4096", "k dimension")
        .insert("num_experts", "8", "number of experts/groups")
        .insert("a_layout", "R", "A tensor data layout - Row by default")
        .insert("b_layout", "C", "B tensor data layout - Col by default")
        .insert("c_layout", "R", "C tensor data layout - Row by default")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("v", "1", "0. No validation, 1. Validation on CPU")
        .insert("mx_prec", "fp8xfp8", "data type for activation and weight, support: fp8xfp8")
        .insert("warmup", "20", "number of iterations before benchmark the kernel")
        .insert("repeat", "50", "number of iterations to benchmark the kernel")
        .insert("split_k", "1", "splitK value")
        .insert("init", "0", "0:random, 1:constant(1), 2:random data + const scales");
    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K, const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(K);
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value, K);
    return ck_tile::make_tuple(rtol, atol);
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;
    using ADataType = ck_tile::fp8_t;
    using BDataType = ck_tile::fp8_t;
    using AccDataType = float;
    using CDataType = ck_tile::fp16_t;
    using ScaleType = ck_tile::e8m0_t;
    using GemmConfig = MXfp8_GemmConfig_256x64x128;

    const std::string mx_prec  = arg_parser.get_str("mx_prec");
    const std::string a_layout = arg_parser.get_str("a_layout");
    const std::string b_layout = arg_parser.get_str("b_layout");

    if(a_layout != "R" || b_layout != "C")
        throw std::runtime_error("Only A=Row, B=Col layout is supported currently!");
    if(mx_prec != "fp8" && mx_prec != "fp8xfp8")
        throw std::runtime_error("This standalone currently supports only fp8/fp8xfp8.");

    const ck_tile::index_t m_chunk = arg_parser.get_int("m");
    const ck_tile::index_t N = arg_parser.get_int("n");
    const ck_tile::index_t K = arg_parser.get_int("k");
    const int group_count = arg_parser.get_int("num_experts");
    const int validation = arg_parser.get_int("v");
    const int n_warmup = arg_parser.get_int("warmup");
    const int n_repeat = arg_parser.get_int("repeat");
    const int kbatch = arg_parser.get_int("split_k");
    const int init_method = arg_parser.get_int("init");

    if(group_count <= 0 || m_chunk <= 0)
        throw std::runtime_error("m and num_experts must be positive");
    if(K % 32 != 0)
        throw std::runtime_error("K must be divisible by 32 for MX GEMM");
    if(K % GemmConfig::K_Tile != 0)
        throw std::runtime_error("K must be a multiple of K_Tile for this pipeline/config.");

    const ck_tile::index_t M_total = m_chunk * group_count;

    std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_dev_bufs, b_dev_bufs, c_dev_bufs;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> scale_a_dev_bufs, scale_b_dev_bufs;
    std::vector<ck_tile::HostTensor<ADataType>> a_hosts;
    std::vector<ck_tile::HostTensor<BDataType>> b_hosts;
    std::vector<ck_tile::HostTensor<CDataType>> c_hosts;
    std::vector<ck_tile::HostTensor<ScaleType>> scale_a_refs;
    std::vector<ck_tile::HostTensor<ScaleType>> scale_b_refs;

    std::vector<ck_tile::index_t> Ms(group_count, m_chunk), Ns(group_count, N), Ks(group_count, K);
    std::vector<ck_tile::index_t> stride_As(group_count), stride_Bs(group_count), stride_Cs(group_count);

    using ScaleM = ck_tile::MXScalePointer<ScaleType, 1, 32>;
    using ScaleN = ck_tile::MXScalePointer<ScaleType, 1, 32>;
    using HostDesc = ck_tile::MXGroupedGemmHostArgs<ScaleM, ScaleN>;

    std::vector<HostDesc> gemm_descs;
    gemm_descs.reserve(group_count);

    for(int i = 0; i < group_count; ++i)
    {
        stride_As[i] = ck_tile::get_default_stride(m_chunk, K, arg_parser.get_int("stride_a"), is_row_major(Row{}));
        stride_Bs[i] = ck_tile::get_default_stride(K, N, arg_parser.get_int("stride_b"), is_row_major(Col{}));
        stride_Cs[i] = ck_tile::get_default_stride(m_chunk, N, arg_parser.get_int("stride_c"), is_row_major(Row{}));

        a_hosts.emplace_back(ck_tile::host_tensor_descriptor(m_chunk, K, stride_As[i], is_row_major(Row{})));
        b_hosts.emplace_back(ck_tile::host_tensor_descriptor(K, N, stride_Bs[i], is_row_major(Col{})));
        c_hosts.emplace_back(ck_tile::host_tensor_descriptor(m_chunk, N, stride_Cs[i], is_row_major(Row{})));

        const ck_tile::index_t scale_k_size = K / 32;
        scale_a_refs.emplace_back(
            ck_tile::host_tensor_descriptor(m_chunk, scale_k_size, scale_k_size, is_row_major(Row{})));
        scale_b_refs.emplace_back(
            ck_tile::host_tensor_descriptor(scale_k_size, N, scale_k_size, is_row_major(Col{})));

        int seed = 1234 + i * 17;
        if(init_method == 0)
        {
            ck_tile::FillUniformDistribution<ADataType>{-2.f, 2.f, seed++}(a_hosts[i]);
            ck_tile::FillUniformDistribution<BDataType>{-2.f, 2.f, seed++}(b_hosts[i]);
            ck_tile::FillUniformDistribution<ScaleType>{0.001f, 10.f, seed++}(scale_a_refs[i]);
            ck_tile::FillUniformDistribution<ScaleType>{0.001f, 10.f, seed++}(scale_b_refs[i]);
        }
        else if(init_method == 1)
        {
            ck_tile::FillConstant<ADataType>{ADataType(1.f)}(a_hosts[i]);
            ck_tile::FillConstant<BDataType>{BDataType(1.f)}(b_hosts[i]);
            ck_tile::FillConstant<ScaleType>{ScaleType(1.f)}(scale_a_refs[i]);
            ck_tile::FillConstant<ScaleType>{ScaleType(1.f)}(scale_b_refs[i]);
        }
        else
        {
            ck_tile::FillUniformDistribution<ADataType>{-2.f, 2.f, seed++}(a_hosts[i]);
            ck_tile::FillUniformDistribution<BDataType>{-2.f, 2.f, seed++}(b_hosts[i]);
            ck_tile::FillConstant<ScaleType>{ScaleType(0.1f)}(scale_a_refs[i]);
            ck_tile::FillConstant<ScaleType>{ScaleType(0.1f)}(scale_b_refs[i]);
        }

        constexpr ck_tile::index_t MPerXdl_      = GemmConfig::M_Warp_Tile;
        constexpr ck_tile::index_t NPerXdl_      = GemmConfig::N_Warp_Tile;
        constexpr ck_tile::index_t KPerXdl_      = GemmConfig::K_Warp_Tile;
        constexpr ck_tile::index_t MIterPerWarp_ = GemmConfig::M_Tile / (GemmConfig::M_Warp * MPerXdl_);
        constexpr ck_tile::index_t NIterPerWarp_ = GemmConfig::N_Tile / (GemmConfig::N_Warp * NPerXdl_);
        constexpr ck_tile::index_t KIterPerWarp_ = GemmConfig::K_Tile / KPerXdl_;
        constexpr ck_tile::index_t MXdlPackEff =
            (MIterPerWarp_ >= 2 && MIterPerWarp_ % 2 == 0) ? 2 : 1;
        constexpr ck_tile::index_t NXdlPackEff =
            (NIterPerWarp_ >= 2 && NIterPerWarp_ % 2 == 0) ? 2 : 1;
        constexpr ck_tile::index_t KXdlPackEff =
            (KIterPerWarp_ >= 2 && KIterPerWarp_ % 2 == 0) ? 2 : 1;
        constexpr ck_tile::index_t XdlMNThread = GemmConfig::M_Warp_Tile;
        constexpr ck_tile::index_t XdlKThread  = 64 / XdlMNThread;

        auto scale_a_packed =
            packScalesMNxK<MXdlPackEff, KXdlPackEff, XdlMNThread, XdlKThread>(scale_a_refs[i], true);
        auto scale_b_packed =
            packScalesMNxK<NXdlPackEff, KXdlPackEff, XdlMNThread, XdlKThread>(scale_b_refs[i], false);

        a_dev_bufs.push_back(std::make_unique<ck_tile::DeviceMem>(a_hosts[i].get_element_space_size_in_bytes()));
        b_dev_bufs.push_back(std::make_unique<ck_tile::DeviceMem>(b_hosts[i].get_element_space_size_in_bytes()));
        c_dev_bufs.push_back(std::make_unique<ck_tile::DeviceMem>(c_hosts[i].get_element_space_size_in_bytes()));
        scale_a_dev_bufs.push_back(std::make_unique<ck_tile::DeviceMem>(scale_a_packed.size() * sizeof(int32_t)));
        scale_b_dev_bufs.push_back(std::make_unique<ck_tile::DeviceMem>(scale_b_packed.size() * sizeof(int32_t)));

        a_dev_bufs[i]->ToDevice(a_hosts[i].data());
        b_dev_bufs[i]->ToDevice(b_hosts[i].data());
        c_dev_bufs[i]->SetZero();
        scale_a_dev_bufs[i]->ToDevice(scale_a_packed.data());
        scale_b_dev_bufs[i]->ToDevice(scale_b_packed.data());

        auto* p_scale_a = reinterpret_cast<ScaleType*>(scale_a_dev_bufs[i]->GetDeviceBuffer());
        auto* p_scale_b = reinterpret_cast<ScaleType*>(scale_b_dev_bufs[i]->GetDeviceBuffer());

        gemm_descs.emplace_back(a_dev_bufs[i]->GetDeviceBuffer(),
                                ScaleM{p_scale_a},
                                b_dev_bufs[i]->GetDeviceBuffer(),
                                ScaleN{p_scale_b},
                                std::array<const void*, 0>{},
                                c_dev_bufs[i]->GetDeviceBuffer(),
                                kbatch,
                                m_chunk,
                                N,
                                K,
                                stride_As[i],
                                stride_Bs[i],
                                std::array<ck_tile::index_t, 0>{},
                                stride_Cs[i]);
    }

    using GemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
                               ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
                               ck_tile::sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;
    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape, GemmConfig::TileParitionerGroupNum, GemmConfig::TileParitionerM01>;
    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                                                 GemmConfig::kPadN,
                                                                 GemmConfig::kPadK,
                                                                 GemmConfig::DoubleSmemBuffer,
                                                                 Row,
                                                                 Col,
                                                                 Row,
                                                                 GemmConfig::TransposeC,
                                                                 GemmConfig::UseStructuredSparsity,
                                                                 true,
                                                                 GemmConfig::NumWaveGroups,
                                                                 GemmConfig::Preshuffle>;
    using UniversalGemmProblem =
        ck_tile::UniversalGemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, GemmUniversalTraits, GemmConfig::Scheduler>;
    using GemmPipeline = ck_tile::MXGemmPipelineAgBgCrCompAsync<UniversalGemmProblem>;
    using GemmEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<ADataType,
                                         BDataType,
                                         ck_tile::tuple<>,
                                         AccDataType,
                                         CDataType,
                                         ck_tile::tuple<>,
                                         Row,
                                         ck_tile::element_wise::PassThrough,
                                         TilePartitioner::MPerBlock,
                                         TilePartitioner::NPerBlock,
                                         GemmConfig::M_Warp,
                                         GemmConfig::N_Warp,
                                         GemmConfig::M_Warp_Tile,
                                         GemmConfig::N_Warp_Tile,
                                         GemmConfig::K_Warp_Tile,
                                         UniversalGemmProblem::TransposeC>>;
    using GroupedKernel = ck_tile::MXGroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue, ScaleM, ScaleN>;

    auto kargs = GroupedKernel::MakeKargs(gemm_descs);
    if(!GroupedKernel::IsSupportedArgument(kargs))
        throw std::runtime_error("Grouped MX GEMM arguments are not supported on this path.");

    ck_tile::DeviceMem workspace(kargs.size() * sizeof(typename decltype(kargs)::value_type));
    workspace.ToDevice(kargs.data());

    const dim3 grids = GroupedKernel::GridSize(gemm_descs);
    const dim3 blocks = GroupedKernel::BlockSize();

    const auto kernel = ck_tile::make_kernel<GemmConfig::kBlockPerCu>(
        GroupedKernel{},
        grids,
        blocks,
        0,
        ck_tile::cast_pointer_to_constant_address_space(workspace.GetDeviceBuffer()),
        static_cast<ck_tile::index_t>(kargs.size()));

    const float ave_time =
        ck_tile::launch_kernel_time_mask(ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat, true, true, 50},
                                         []() {},
                                         kernel);

    constexpr int APackedSize = ck_tile::numeric_traits<ADataType>::PackedSize;
    constexpr int BPackedSize = ck_tile::numeric_traits<BDataType>::PackedSize;
    const std::size_t flop = std::size_t(2) * M_total * N * K + std::size_t(2) * M_total * N * K / 32;
    const std::size_t num_byte = sizeof(ADataType) * M_total * K / APackedSize +
                                 sizeof(BDataType) * group_count * N * K / BPackedSize +
                                 sizeof(CDataType) * M_total * N +
                                 sizeof(ck_tile::e8m0_t) * M_total * K / 32 +
                                 sizeof(ck_tile::e8m0_t) * group_count * N * K / 32;
    float tflops = static_cast<float>(flop) / 1.E9f / ave_time;
    float gb_per_sec = num_byte / 1.E6f / ave_time;

    std::cout << "Run " << ck_tile::gemm_prec_str<ADataType, BDataType>() << " MX Grouped GEMM kernel "
              << " M_chunk = " << m_chunk
              << " Experts = " << group_count
              << " M_total = " << M_total
              << " N = " << N
              << " K = " << K
              << " : " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << std::endl;

    bool pass = true;
    if(validation > 0)
    {
        for(int i = 0; i < group_count; ++i)
        {
            c_dev_bufs[i]->FromDevice(c_hosts[i].data());

            ck_tile::HostTensor<CDataType> c_ref(
                ck_tile::host_tensor_descriptor(m_chunk, N, stride_Cs[i], is_row_major(Row{})));
            c_ref.SetZero();

            ck_tile::reference_mx_gemm<ADataType, BDataType, ScaleType, AccDataType, CDataType>(
                a_hosts[i], b_hosts[i], c_ref, scale_a_refs[i], scale_b_refs[i]);

            const float max_accumulated_value =
                *std::max_element(c_ref.mData.begin(), c_ref.mData.end());
            const auto rtol_atol =
                calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(K, max_accumulated_value);
            const double rtol = rtol_atol.at(ck_tile::number<0>{});
            const double atol = rtol_atol.at(ck_tile::number<1>{});
            pass &= ck_tile::check_err(c_hosts[i], c_ref, "Error: Incorrect results!", rtol, atol);
        }

        std::cout << "The CPU verification result is: " << (pass ? "correct" : "fail") << std::endl;
    }

    return pass ? 0 : -1;
}
