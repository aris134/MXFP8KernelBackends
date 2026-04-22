// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>

#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"
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

template <typename Layout>
static constexpr inline auto is_row_major(Layout)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<Layout>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename Layout>
using is_row_major_t = ck_tile::bool_constant<
    std::is_same_v<ck_tile::remove_cvref_t<Layout>, ck_tile::tensor_layout::gemm::RowMajor>>;

// Base FlatmmConfig with 16x16 warp tile on gfx950.
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

    static constexpr int GetNLane() { return Config::N_Warp_Tile; }

    template <typename DType>
    static auto preShuffleWeight(ck_tile::HostTensor<DType>& src)
    {
        constexpr ck_tile::index_t NLane = Config::N_Warp_Tile;
        auto src_lengths                 = src.get_lengths();
        const int K                      = src_lengths[0];
        const int N                      = src_lengths[1];
        constexpr int packed_size        = ck_tile::numeric_traits<DType>::PackedSize;
        int KPack                        = std::is_same_v<DType, ck_tile::pk_fp6x16_t>
                                               ? 32
                                               : 16 * packed_size;
        int KLane = ck_tile::get_warp_size() / NLane;
        int K0    = K / (KLane * KPack);

        ck_tile::HostTensor<DType> shuffled(ck_tile::HostTensorDescriptor({N * K}, {1}));

        for(int n = 0; n < N; ++n)
        {
            for(int k = 0; k < K; k += packed_size)
            {
                int n0 = n / NLane;
                int n1 = n % NLane;

                int k0    = k / (KLane * KPack);
                int tempk = k % (KLane * KPack);
                int k1    = tempk / KPack;
                int k2    = tempk % KPack;

                int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                                  k1 * KPack * NLane + n1 * KPack + k2;

                shuffled(outputIndex) = src(k, n);
            }
        }
        return shuffled;
    }

    template <bool KLast, typename DType>
    static auto preShuffleScale(ck_tile::HostTensor<DType>& src)
    {
        auto src_lengths = src.get_lengths();
        const auto MN    = KLast ? src_lengths[0] : src_lengths[1];
        const auto K     = KLast ? src_lengths[1] : src_lengths[0];

        constexpr size_t MNXdlPack   = 2;
        constexpr size_t KXdlPack    = 2;
        constexpr size_t XdlMNThread = Config::N_Warp_Tile;
        const size_t XdlKThread      = ck_tile::get_warp_size() / XdlMNThread;

        const auto MNPadded = ck_tile::integer_least_multiple(MN, XdlMNThread * MNXdlPack);
        ck_tile::HostTensor<DType> shuffled(ck_tile::HostTensorDescriptor({MNPadded * K}, {1}));

        const size_t K0 = K / KXdlPack / XdlKThread;

        for(size_t n = 0; n < static_cast<size_t>(MNPadded); ++n)
        {
            for(size_t k = 0; k < static_cast<size_t>(K); ++k)
            {
                const auto n0    = n / (XdlMNThread * MNXdlPack);
                const auto tempn = n % (XdlMNThread * MNXdlPack);
                const auto n1    = tempn % XdlMNThread;
                const auto n2    = tempn / XdlMNThread;

                const auto k0    = k / (XdlKThread * KXdlPack);
                const auto tempk = k % (XdlKThread * KXdlPack);
                const auto k1    = tempk % XdlKThread;
                const auto k2    = tempk / XdlKThread;

                const auto outputIndex =
                    n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
                    k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
                    k1 * MNXdlPack * KXdlPack * XdlMNThread + n1 * MNXdlPack * KXdlPack +
                    k2 * MNXdlPack + n2;

                if constexpr(KLast)
                    shuffled(outputIndex) = n < static_cast<size_t>(MN) ? src(n, k) : DType{};
                else
                    shuffled(outputIndex) = n < static_cast<size_t>(MN) ? src(k, n) : DType{};
            }
        }
        return shuffled;
    }
};

using MXFlatmm_GFX950_FP8FP8_Traits =
    MXFlatmmArchTraits<ck_tile::core::arch::TargetId::GFX950, MXFlatmmConfigBase16>;

template <typename MXFlatmmArchTraitsT,
          typename ADataType,
          typename BDataType,
          typename DsDatatype,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ScaleM,
          typename ScaleN,
          bool persistent,
          typename CDEElementWise,
          bool Splitk,
          bool HasHotLoop,
          ck_tile::TailNumber TailNum>
float mx_flatmm_calc(const ck_tile::ScaleFlatmmHostArgs<ScaleM, ScaleN>& args,
                     const ck_tile::stream_config& s)
{
    using FlatmmConfig = typename MXFlatmmArchTraitsT::Config;

    using FlatmmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<FlatmmConfig::M_Tile, FlatmmConfig::N_Tile, FlatmmConfig::K_Tile>,
        ck_tile::sequence<FlatmmConfig::M_Warp, FlatmmConfig::N_Warp, FlatmmConfig::K_Warp>,
        ck_tile::sequence<FlatmmConfig::M_Warp_Tile,
                          FlatmmConfig::N_Warp_Tile,
                          FlatmmConfig::K_Warp_Tile>>;

    using MXGemmTraits = ck_tile::TileGemmUniversalTraits<FlatmmConfig::kPadM,
                                                          FlatmmConfig::kPadN,
                                                          FlatmmConfig::kPadK,
                                                          FlatmmConfig::DoubleSmemBuffer,
                                                          ALayout,
                                                          BLayout,
                                                          ELayout,
                                                          FlatmmConfig::TransposeC,
                                                          FlatmmConfig::UseStructuredSparsity,
                                                          persistent,
                                                          FlatmmConfig::NumWaveGroups,
                                                          true>;

    using ComputeDataType = ADataType;
    static_assert(sizeof(ComputeDataType) >= sizeof(BDataType),
                  "mixed_prec_flatmm requires ADataType to be at least as wide as BDataType");

    constexpr auto scheduler = FlatmmConfig::Scheduler;
    ck_tile::ignore          = Splitk;

    constexpr int BlockedXDLN_PerWarp = MXFlatmmArchTraitsT::BlockedXDLN_PerWarp;

    using MXPipelineProblem = ck_tile::MXFlatmmPipelineProblem<ADataType,
                                                               BDataType,
                                                               AccDataType,
                                                               FlatmmShape,
                                                               MXGemmTraits,
                                                               scheduler,
                                                               HasHotLoop,
                                                               TailNum>;

    using MXFlatmmPipeline =
        typename MXFlatmmArchTraitsT::template MXFlatmmPipeline<MXPipelineProblem>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<FlatmmShape,
                                                   FlatmmConfig::TileParitionerGroupNum,
                                                   FlatmmConfig::TileParitionerM01>;

    using GemmEpilogue =
        ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<ComputeDataType,
                                                                   ComputeDataType,
                                                                   DsDatatype,
                                                                   AccDataType,
                                                                   CDataType,
                                                                   DsLayout,
                                                                   ELayout,
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
        throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");

    auto clear_gemm_output = [&]() {
        if(args.k_batch > 1)
            (void)hipMemsetAsync(args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_);
    };

    return ck_tile::launch_kernel_time_mask(
        s,
        clear_gemm_output,
        ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
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
float invoke_mx_flatmm(ck_tile::DeviceMem& a_dev_buf,
                       ck_tile::DeviceMem& b_shuffle_dev_buf,
                       ck_tile::DeviceMem& c_dev_buf,
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

    ck_tile::ScaleFlatmmHostArgs<ScaleA, ScaleB> args = {a_dev_buf.GetDeviceBuffer(),
                                                         b_shuffle_dev_buf.GetDeviceBuffer(),
                                                         {},
                                                         c_dev_buf.GetDeviceBuffer(),
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

    float ave_time = BaseFlatmmPipeline::template TailHandler<true>(
        [&](auto has_hot_loop_, auto tail_num_) {
            constexpr auto has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_num_v     = tail_num_.value;
            return mx_flatmm_calc<MXFlatmmArchTraitsT,
                                  ADataType,
                                  BDataType,
                                  DsDatatype,
                                  AccDataType,
                                  CDataType,
                                  ALayout,
                                  BLayout,
                                  DsLayout,
                                  CLayout,
                                  ScaleA,
                                  ScaleB,
                                  UsePersistentKernel,
                                  CDEElementWise,
                                  false,
                                  has_hot_loop_v,
                                  tail_num_v>(
                args,
                ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat, true, true, 50});
        },
        has_hot_loop,
        tail_num);

    constexpr int APackedSize = ck_tile::numeric_traits<ADataType>::PackedSize;
    constexpr int BPackedSize = ck_tile::numeric_traits<BDataType>::PackedSize;

    std::size_t flop     = std::size_t(2) * M * N * K + std::size_t(2) * M * N * K / 32;
    std::size_t num_byte = sizeof(ADataType) * M * K / APackedSize +
                           sizeof(BDataType) * N * K / BPackedSize + sizeof(CDataType) * M * N +
                           sizeof(ck_tile::e8m0_t) * M * K / 32 +
                           sizeof(ck_tile::e8m0_t) * N * K / 32;
    float tflops         = static_cast<float>(flop) / 1.E9f / ave_time;
    float gb_per_sec     = num_byte / 1.E6f / ave_time;

    std::cout << "Run " << ck_tile::gemm_prec_str<ADataType, BDataType>() << " Flatmm kernel "
              << " M = " << M << " N = " << N << " K = " << K << " StrideA = " << stride_A
              << " StrideB = " << stride_B << " StrideC = " << stride_C << " : " << ave_time
              << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, " << std::endl;

    return ave_time;
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "4096", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("k", "4096", "k dimension")
        .insert("a_layout", "R", "A tensor data layout - Row by default")
        .insert("b_layout", "C", "B tensor data layout - Col by default")
        .insert("c_layout", "R", "C tensor data layout - Row by default")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("v", "1", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
        .insert("mx_prec", "fp8xfp8", "data type for activation and weight, support: fp8xfp8")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("init", "0", "0:random, 1:constant(1), 2:random data + const scales")
        .insert("persistent", "0", "0: no persistent, 1: persistent kernel")
        .insert("warp_tile", "0", "0: 16x16x128 on gfx950.");
    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K, const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<(sizeof(ADataType) < sizeof(BDataType)), ADataType, BDataType>;
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(K);
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value, K);
    return ck_tile::make_tuple(rtol, atol);
}

template <typename PrecActType,
          typename PrecWeightType,
          typename CDataType,
          typename MXFlatmmArchTraitsT,
          bool UsePersistentKernel = false,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_mx_flatmm_with_layouts(const ck_tile::ArgParser& arg_parser,
                               const ALayout a_layout = ALayout{},
                               const BLayout b_layout = BLayout{},
                               const CLayout c_layout = CLayout{})
{
    using ADataType   = PrecActType;
    using BDataType   = PrecWeightType;
    using AccDataType = float;
    using ScaleType   = ck_tile::e8m0_t;

    constexpr int ScaleGranularityM = 1;
    constexpr int ScaleGranularityN = 1;
    constexpr int ScaleGranularityK = 32;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");

    const int validation  = arg_parser.get_int("v");
    const int init_method = arg_parser.get_int("init");
    const int n_warmup    = arg_parser.get_int("warmup");
    const int n_repeat    = arg_parser.get_int("repeat");

    stride_A = ck_tile::get_default_stride(M, K, stride_A, is_row_major(a_layout));
    stride_B = ck_tile::get_default_stride(K, N, stride_B, is_row_major(b_layout));
    stride_C = ck_tile::get_default_stride(M, N, stride_C, is_row_major(c_layout));

    const auto scale_stride_A = ck_tile::get_default_stride(
        M / ScaleGranularityM, K / ScaleGranularityK, 0, is_row_major(a_layout));
    const auto scale_stride_B = ck_tile::get_default_stride(
        K / ScaleGranularityK, N / ScaleGranularityN, 0, is_row_major(b_layout));

    if(K % ScaleGranularityK != 0)
        throw std::runtime_error("Flatmm: K must be a multiple of 32.");
    if(K % MXFlatmmArchTraitsT::Config::K_Tile != 0)
        throw std::runtime_error("Flatmm: K must be a multiple of K_Tile for this config.");
    if(K % ck_tile::numeric_traits<ADataType>::PackedSize != 0 ||
       K % ck_tile::numeric_traits<BDataType>::PackedSize != 0)
        throw std::runtime_error("Flatmm: K must be a multiple of packed size.");

    ck_tile::HostTensor<ADataType> a_host(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BDataType> b_host(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));
    ck_tile::HostTensor<CDataType> c_host(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(c_layout)));

    ck_tile::HostTensor<ScaleType> scale_a(
        ck_tile::host_tensor_descriptor(M / ScaleGranularityM,
                                        K / ScaleGranularityK,
                                        scale_stride_A,
                                        is_row_major(a_layout)));
    ck_tile::HostTensor<ScaleType> scale_b(
        ck_tile::host_tensor_descriptor(K / ScaleGranularityK,
                                        N / ScaleGranularityN,
                                        scale_stride_B,
                                        is_row_major(b_layout)));

    int seed = 1234;
    if(init_method == 0)
    {
        ck_tile::FillUniformDistribution<ADataType>{-2.f, 2.f, seed++}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{-2.f, 2.f, seed++}(b_host);
        ck_tile::FillUniformDistribution<ScaleType>{0.001f, 10.f, seed++}(scale_a);
        ck_tile::FillUniformDistribution<ScaleType>{0.001f, 10.f, seed++}(scale_b);
    }
    else if(init_method == 1)
    {
        ck_tile::FillConstant<ADataType>{ADataType(1.f)}(a_host);
        ck_tile::FillConstant<BDataType>{BDataType(1.f)}(b_host);
        ck_tile::FillConstant<ScaleType>{ScaleType(1.f)}(scale_a);
        ck_tile::FillConstant<ScaleType>{ScaleType(1.f)}(scale_b);
    }
    else if(init_method == 2)
    {
        ck_tile::FillUniformDistribution<ADataType>{-2.f, 2.f, seed++}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{-2.f, 2.f, seed++}(b_host);
        ck_tile::FillConstant<ScaleType>{ScaleType(0.1f)}(scale_a);
        ck_tile::FillConstant<ScaleType>{ScaleType(0.1f)}(scale_b);
    }
    else
    {
        throw std::runtime_error("Unsupported init mode.");
    }

    const auto b_shuffled_host  = MXFlatmmArchTraitsT::preShuffleWeight(b_host);
    const auto scale_a_shuffled = MXFlatmmArchTraitsT::template preShuffleScale<true>(scale_a);
    const auto scale_b_shuffled = MXFlatmmArchTraitsT::template preShuffleScale<false>(scale_b);

    ck_tile::DeviceMem a_dev_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_shuffled_dev_buf(b_shuffled_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_dev_buf(c_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem scale_a_dev_buf(scale_a_shuffled.get_element_space_size_in_bytes());
    ck_tile::DeviceMem scale_b_dev_buf(scale_b_shuffled.get_element_space_size_in_bytes());

    a_dev_buf.ToDevice(a_host.data());
    b_shuffled_dev_buf.ToDevice(b_shuffled_host.data());
    c_host.SetZero();
    c_dev_buf.ToDevice(c_host.data());
    scale_a_dev_buf.ToDevice(scale_a_shuffled.data());
    scale_b_dev_buf.ToDevice(scale_b_shuffled.data());

    auto scale_a_dev_ptr =
        ck_tile::FlatmmScalePointer<ScaleGranularityM, ScaleGranularityK, ScaleType>{
            static_cast<ScaleType*>(scale_a_dev_buf.GetDeviceBuffer()), M / ScaleGranularityM};
    auto scale_b_dev_ptr =
        ck_tile::FlatmmScalePointer<ScaleGranularityN, ScaleGranularityK, ScaleType>{
            static_cast<ScaleType*>(scale_b_dev_buf.GetDeviceBuffer()), N / ScaleGranularityN};

    (void)invoke_mx_flatmm<MXFlatmmArchTraitsT,
                           ADataType,
                           BDataType,
                           ck_tile::tuple<>,
                           AccDataType,
                           CDataType,
                           ALayout,
                           BLayout,
                           ck_tile::tuple<>,
                           CLayout,
                           decltype(scale_a_dev_ptr),
                           decltype(scale_b_dev_ptr),
                           UsePersistentKernel>(a_dev_buf,
                                                b_shuffled_dev_buf,
                                                c_dev_buf,
                                                M,
                                                N,
                                                K,
                                                stride_A,
                                                stride_B,
                                                stride_C,
                                                scale_a_dev_ptr,
                                                scale_b_dev_ptr,
                                                n_warmup,
                                                n_repeat);

    bool pass = true;
    if(validation > 0)
    {
        c_dev_buf.FromDevice(c_host.data());

        ck_tile::HostTensor<CDataType> c_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_ref.SetZero();

        ck_tile::reference_mx_gemm<ADataType, BDataType, ScaleType, AccDataType, CDataType>(
            a_host, b_host, c_ref, scale_a, scale_b);

        const float max_accumulated_value =
            *std::max_element(c_ref.mData.begin(), c_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, max_accumulated_value);
        const double rtol = rtol_atol.at(ck_tile::number<0>{});
        const double atol = rtol_atol.at(ck_tile::number<1>{});

        pass = ck_tile::check_err(c_host, c_ref, "Error: Incorrect results!", rtol, atol);

        std::cout << "Relative error threshold: " << rtol << " Absolute error threshold: " << atol
                  << std::endl;
        std::cout << "The CPU verification result is: " << (pass ? "correct" : "fail")
                  << std::endl;
    }

    return pass ? 0 : -1;
}

int run_mx_flatmm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    const std::string mx_prec      = arg_parser.get_str("mx_prec");
    const std::string a_layout     = arg_parser.get_str("a_layout");
    const std::string b_layout     = arg_parser.get_str("b_layout");
    const int persistent_opt       = arg_parser.get_int("persistent");
    const int warp_tile            = arg_parser.get_int("warp_tile");

    if(warp_tile != 0)
        throw std::runtime_error("Only warp_tile=0 (16x16x128 on gfx950) is supported.");
    if(a_layout != "R" || b_layout != "C")
        throw std::runtime_error("Only A=Row, B=Col layout is supported currently!");
    if(mx_prec != "fp8" && mx_prec != "fp8xfp8")
        throw std::runtime_error("This standalone currently supports only fp8/fp8xfp8.");
    if(persistent_opt != 0)
        throw std::runtime_error("Only non-persistent kernels are supported currently!");

    std::cout << "Using default warptile of 16x16x128." << std::endl;

    return run_mx_flatmm_with_layouts<ck_tile::fp8_t,
                                      ck_tile::fp8_t,
                                      ck_tile::fp16_t,
                                      MXFlatmm_GFX950_FP8FP8_Traits,
                                      false>(arg_parser, Row{}, Col{}, Row{});
}

int main(int argc, char* argv[])
{
    try
    {
        return run_mx_flatmm_example(argc, argv);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
