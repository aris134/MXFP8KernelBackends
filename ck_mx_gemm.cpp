// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "ck_tile/host.hpp"
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_mx/kernel/scale_pointer.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm_mx/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp"
#include "ck_tile/ops/gemm_mx/kernel/gemm_mx_kernel.hpp"

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

// GEMM config with 16x16 warp tile
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
    static constexpr bool DoubleSmemBuffer          = true; // comp_async uses double buffer
    static constexpr bool Preshuffle                = false;

    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = false;
};

struct MXfp8_GemmConfig16 : MxGemmConfig
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 256;
};

struct MXfp8_GemmConfig_256x64x128 : MxGemmConfig
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 128;
};

template <typename Layout>
using is_row_major_t = ck_tile::bool_constant<
    std::is_same_v<ck_tile::remove_cvref_t<Layout>, ck_tile::tensor_layout::gemm::RowMajor>>;

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
    static_assert(sizeof(ComputeDataType) >= sizeof(BDataType),
                  "mixed_prec_gemm requires ADataType is a wider type than BDataType");

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
        throw std::runtime_error(
            "MX GEMM: unsupported shape/configuration (for this config K must satisfy the pipeline/kernel requirements; set CK_TILE_LOGGING=1 for details).");
    }

    const auto kernel = ck_tile::make_kernel<Kernel::kBlockPerCu>(
        Kernel{}, Kernel::GridSize(kargs), Kernel::BlockSize(), 0, kargs);

    auto clear_gemm_output = [&]() {
        if(args.k_batch > 1)
        {
            (void)hipMemsetAsync(args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_);
        }
    };

    return ck_tile::launch_kernel_time_mask(s, clear_gemm_output, kernel);
}

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
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

    float ave_time = (args.k_batch == 1) ? invoke_splitk_path(std::false_type{})
                                         : invoke_splitk_path(std::true_type{});

    constexpr int APackedSize = ck_tile::numeric_traits<ADataType>::PackedSize;
    constexpr int BPackedSize = ck_tile::numeric_traits<BDataType>::PackedSize;

    std::size_t flop     = std::size_t(2) * M * N * K + std::size_t(2) * M * N * K / 32;
    std::size_t num_byte = sizeof(ADataType) * M * K / APackedSize +
                           sizeof(BDataType) * N * K / BPackedSize + sizeof(CDataType) * M * N +
                           sizeof(ck_tile::e8m0_t) * M * K / 32 +
                           sizeof(ck_tile::e8m0_t) * N * K / 32;
    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run " << ck_tile::gemm_prec_str<ADataType, BDataType>() << " MX GEMM kernel "
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
        .insert("b_layout", "C", "B tensor data layout - Row by default")
        .insert("c_layout", "R", "C tensor data layout - Row by default")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("v", "1", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
        .insert("mx_prec", "fp8xfp8", "data type for activation and weight, support: fp8xfp8")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("split_k", "1", "splitK value")
        .insert("init", "0", "0:random, 1:constant(1), 2:random data + const scales");
    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// Calculate relative and absolute error thresholds for MX GEMM
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

// Pack [MN, K/32] e8m0_t scales into [MN/MNPack, K/32/KPack] int32_t
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

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename GemmConfig,
          bool UsePersistentKernel,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_mx_gemm_with_layouts(int argc, char* argv[], ALayout, BLayout, CLayout)
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    ck_tile::index_t M        = arg_parser.get_int("m");
    ck_tile::index_t N        = arg_parser.get_int("n");
    ck_tile::index_t K        = arg_parser.get_int("k");
    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");
    int validation            = arg_parser.get_int("v");
    int n_warmup              = arg_parser.get_int("warmup");
    int n_repeat              = arg_parser.get_int("repeat");
    int kbatch                = arg_parser.get_int("split_k");
    int init_method           = arg_parser.get_int("init");

    using CDataType = ck_tile::fp16_t;

    if constexpr(!GemmConfig::kPadK)
    {
        if(K % GemmConfig::K_Tile != 0)
        {
            throw std::runtime_error(
                "MX GEMM: unsupported shape/configuration: K must be a multiple of K_Tile for this pipeline/config (no K padding support).");
        }
    }

    if(stride_A == 0)
        stride_A = ck_tile::get_default_stride(M, K, 0, is_row_major(ALayout{}));
    if(stride_B == 0)
        stride_B = ck_tile::get_default_stride(K, N, 0, is_row_major(BLayout{}));
    if(stride_C == 0)
        stride_C = ck_tile::get_default_stride(M, N, 0, is_row_major(CLayout{}));

    ck_tile::HostTensor<ADataType> a_host(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(ALayout{})));
    ck_tile::HostTensor<BDataType> b_host(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(BLayout{})));
    ck_tile::HostTensor<CDataType> c_host(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    using ScaleType               = ck_tile::e8m0_t;
    ck_tile::index_t scale_k_size = K / 32;

    ck_tile::index_t stride_scale_a =
        ck_tile::get_default_stride(M, scale_k_size, 0, is_row_major(ALayout{}));
    ck_tile::index_t stride_scale_b =
        ck_tile::get_default_stride(scale_k_size, N, 0, is_row_major(BLayout{}));

    ck_tile::HostTensor<ScaleType> scale_a_host(
        ck_tile::host_tensor_descriptor(M, scale_k_size, stride_scale_a, is_row_major(ALayout{})));
    ck_tile::HostTensor<ScaleType> scale_b_host(
        ck_tile::host_tensor_descriptor(scale_k_size, N, stride_scale_b, is_row_major(BLayout{})));
    int seed = 1234;
    switch(init_method)
    {
    case 0:
        ck_tile::FillUniformDistribution<ADataType>{-2.f, 2.f, seed++}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{-2.f, 2.f, seed++}(b_host);
        ck_tile::FillUniformDistribution<ScaleType>{0.001f, 10.f, seed++}(scale_a_host);
        ck_tile::FillUniformDistribution<ScaleType>{0.001f, 10.f, seed++}(scale_b_host);
        break;
    case 1:
        ck_tile::FillConstant<ADataType>{ADataType(1.f)}(a_host);
        ck_tile::FillConstant<BDataType>{BDataType(1.f)}(b_host);
        ck_tile::FillConstant<ScaleType>{ScaleType(1.f)}(scale_a_host);
        ck_tile::FillConstant<ScaleType>{ScaleType(1.f)}(scale_b_host);
        break;
    case 2:
        ck_tile::FillUniformDistribution<ADataType>{-2.f, 2.f, seed++}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{-2.f, 2.f, seed++}(b_host);
        ck_tile::FillConstant<ScaleType>{ScaleType(0.1f)}(scale_a_host);
        ck_tile::FillConstant<ScaleType>{ScaleType(0.1f)}(scale_b_host);
        break;
    default:
        throw std::runtime_error("Unsupported init mode.");
    }

    constexpr ck_tile::index_t MPerXdl_      = GemmConfig::M_Warp_Tile;
    constexpr ck_tile::index_t NPerXdl_      = GemmConfig::N_Warp_Tile;
    constexpr ck_tile::index_t KPerXdl_      = GemmConfig::K_Warp_Tile;
    constexpr ck_tile::index_t MIterPerWarp_ = GemmConfig::M_Tile / (GemmConfig::M_Warp * MPerXdl_);
    constexpr ck_tile::index_t NIterPerWarp_ = GemmConfig::N_Tile / (GemmConfig::N_Warp * NPerXdl_);
    constexpr ck_tile::index_t KIterPerWarp_ = GemmConfig::K_Tile / KPerXdl_;

    constexpr ck_tile::index_t MXdlPackEff = (MIterPerWarp_ >= 2 && MIterPerWarp_ % 2 == 0) ? 2 : 1;
    constexpr ck_tile::index_t NXdlPackEff = (NIterPerWarp_ >= 2 && NIterPerWarp_ % 2 == 0) ? 2 : 1;
    constexpr ck_tile::index_t KXdlPackEff = (KIterPerWarp_ >= 2 && KIterPerWarp_ % 2 == 0) ? 2 : 1;

    constexpr ck_tile::index_t XdlMNThread = GemmConfig::M_Warp_Tile;
    constexpr ck_tile::index_t XdlKThread  = 64 / XdlMNThread;

    auto scale_a_packed =
        packScalesMNxK<MXdlPackEff, KXdlPackEff, XdlMNThread, XdlKThread>(scale_a_host, true);
    auto scale_b_packed =
        packScalesMNxK<NXdlPackEff, KXdlPackEff, XdlMNThread, XdlKThread>(scale_b_host, false);

    ck_tile::DeviceMem a_dev_buf(a_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_dev_buf(b_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_dev_buf(c_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem scale_a_dev_buf(scale_a_packed.size() * sizeof(int32_t));
    ck_tile::DeviceMem scale_b_dev_buf(scale_b_packed.size() * sizeof(int32_t));

    a_dev_buf.ToDevice(a_host.data());
    b_dev_buf.ToDevice(b_host.data());
    c_dev_buf.SetZero();
    scale_a_dev_buf.ToDevice(scale_a_packed.data());
    scale_b_dev_buf.ToDevice(scale_b_packed.data());

    using ScaleM = ck_tile::MXScalePointer<ScaleType, 1, 32>;
    using ScaleN = ck_tile::MXScalePointer<ScaleType, 1, 32>;
    ScaleM scale_m(reinterpret_cast<ScaleType*>(scale_a_dev_buf.GetDeviceBuffer()));
    ScaleN scale_n(reinterpret_cast<ScaleType*>(scale_b_dev_buf.GetDeviceBuffer()));

    float ave_time = invoke_mx_gemm<GemmConfig,
                                    ADataType,
                                    BDataType,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    CLayout,
                                    ScaleM,
                                    ScaleN,
                                    UsePersistentKernel>(a_dev_buf,
                                                         b_dev_buf,
                                                         c_dev_buf,
                                                         M,
                                                         N,
                                                         K,
                                                         stride_A,
                                                         stride_B,
                                                         stride_C,
                                                         kbatch,
                                                         scale_m,
                                                         scale_n,
                                                         n_warmup,
                                                         n_repeat);

    (void)ave_time;

    bool pass = true;
    if(validation > 0)
    {
        c_dev_buf.FromDevice(c_host.data());

        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        ck_tile::reference_mx_gemm<ADataType, BDataType, ScaleType, AccDataType, CDataType>(
            a_host, b_host, c_m_n_host_ref, scale_a_host, scale_b_host);

        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, max_accumulated_value);
        const double rtol = rtol_atol.at(ck_tile::number<0>{});
        const double atol = rtol_atol.at(ck_tile::number<1>{});
        pass = ck_tile::check_err(c_host, c_m_n_host_ref, "Error: Incorrect results!", rtol, atol);

        std::cout << "Relative error threshold: " << rtol << " Absolute error threshold: " << atol
                  << std::endl;
        std::cout << "The CPU verification result is: " << (pass ? "correct" : "fail") << std::endl;
    }
    return pass ? 0 : -1;
}

int run_mx_gemm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    std::string mx_prec  = arg_parser.get_str("mx_prec");
    std::string a_layout = arg_parser.get_str("a_layout");
    std::string b_layout = arg_parser.get_str("b_layout");

    if(a_layout != "R" || b_layout != "C")
    {
        throw std::runtime_error("Only A=Row, B=Col layout is supported currently!");
    }

    if(mx_prec != "fp8" && mx_prec != "fp8xfp8")
    {
        throw std::runtime_error("This standalone currently supports only fp8/fp8xfp8.");
    }

    return run_mx_gemm_with_layouts<ck_tile::fp8_t,
                                    ck_tile::fp8_t,
                                    float,
                                    MXfp8_GemmConfig_256x64x128,//MXfp8_GemmConfig16,
                                    true>(argc, argv, Row{}, Col{}, Row{});
}

int main(int argc, char* argv[]) { return run_mx_gemm_example(argc, argv); }
