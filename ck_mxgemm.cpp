#ifndef CK_USE_OCP_FP8
#define CK_USE_OCP_FP8 1
#endif

#if CK_USE_OCP_FP8
#pragma message("Verification: CK_USE_OCP_FP8 is ENABLED. Using OCP FP8 types.")
#else
#error "Verification Failed: CK_USE_OCP_FP8 is NOT enabled!"
#endif

#include <array>
#include <cstring>
#include <iostream>
#include <vector>

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <pybind11/stl.h>

#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_mx/kernel/scale_pointer.hpp"
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
    static constexpr bool DoubleSmemBuffer          = false;
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

    const auto kernel = ck_tile::make_kernel<Kernel::kBlockPerCu>(
        Kernel{}, Kernel::GridSize(kargs), Kernel::BlockSize(), 0, kargs);

    return ck_tile::launch_kernel(s, kernel);
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
float invoke_mx_gemm(const ADataType* a_dev_ptr,
                     const BDataType* b_dev_ptr,
                     CDataType* c_dev_ptr,
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
    MXGemmHostArgs<ScaleM, ScaleN> args(static_cast<const void*>(a_dev_ptr),
                                        static_cast<const void*>(b_dev_ptr),
                                        static_cast<void*>(c_dev_ptr),
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
                           sizeof(BDataType) * N * K / BPackedSize +
                           sizeof(CDataType) * M * N +
                           sizeof(ck_tile::e8m0_t) * M * K / 32 +
                           sizeof(ck_tile::e8m0_t) * N * K / 32;
    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run " << ck_tile::gemm_prec_str<ADataType, BDataType>() << " MX GEMM kernel "
              << " M = " << M << " N = " << N << " K = " << K
              << " StrideA = " << stride_A
              << " StrideB = " << stride_B
              << " StrideC = " << stride_C
              << " : " << ave_time << " ms, "
              << tflops << " TFlops, "
              << gb_per_sec << " GB/s, "
              << std::endl;

    return ave_time;
}

float ck_mxgemm_mxfp8(torch::Tensor A,
                      torch::Tensor B,
                      torch::Tensor C,
                      torch::Tensor A_scales,
                      torch::Tensor B_scales,
                      ck_tile::index_t M,
                      ck_tile::index_t N,
                      ck_tile::index_t K,
                      int warmup = 20,
                      int repeat = 50)
{
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    using ADataType   = ck_tile::fp8_t;
    using BDataType   = ck_tile::fp8_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
    using ScaleType   = ck_tile::e8m0_t;
    using ALayout     = Row;
    using BLayout     = Col;
    using CLayout     = Row;
    using GemmConfig  = MXfp8_GemmConfig16;

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

    const ck_tile::index_t scale_k_size   = K / 32;
    const ck_tile::index_t stride_scale_a = scale_k_size;
    const ck_tile::index_t stride_scale_b = scale_k_size;

    auto A_scales_host = A_scales.contiguous();
    auto B_scales_host = B_scales.contiguous();

    const auto* a_scales_ptr =
        reinterpret_cast<const ScaleType*>(A_scales_host.data_ptr());
    const auto* b_scales_ptr =
        reinterpret_cast<const ScaleType*>(B_scales_host.data_ptr());

    ck_tile::HostTensor<ScaleType> A_scales_ht(
        ck_tile::host_tensor_descriptor(static_cast<std::size_t>(M),
                                        static_cast<std::size_t>(scale_k_size),
                                        static_cast<std::size_t>(stride_scale_a),
                                        ck_tile::bool_constant<true>{}));

    ck_tile::HostTensor<ScaleType> B_scales_ht(
        ck_tile::host_tensor_descriptor(static_cast<std::size_t>(scale_k_size),
                                        static_cast<std::size_t>(N),
                                        static_cast<std::size_t>(stride_scale_b),
                                        ck_tile::bool_constant<false>{}));

    std::memcpy(A_scales_ht.data(),
                a_scales_ptr,
                static_cast<size_t>(M) * static_cast<size_t>(scale_k_size) * sizeof(ScaleType));

    std::memcpy(B_scales_ht.data(),
                b_scales_ptr,
                static_cast<size_t>(scale_k_size) * static_cast<size_t>(N) * sizeof(ScaleType));

    auto A_scales_packed =
        packScalesMNxK<MXdlPackEff, KXdlPackEff, XdlMNThread, XdlKThread>(A_scales_ht, true);
    auto B_scales_packed =
        packScalesMNxK<NXdlPackEff, KXdlPackEff, XdlMNThread, XdlKThread>(B_scales_ht, false);

    ck_tile::DeviceMem A_scales_dev_buf(A_scales_packed.size() * sizeof(int32_t));
    ck_tile::DeviceMem B_scales_dev_buf(B_scales_packed.size() * sizeof(int32_t));

    A_scales_dev_buf.ToDevice(A_scales_packed.data());
    B_scales_dev_buf.ToDevice(B_scales_packed.data());

    using ScaleM = ck_tile::MXScalePointer<ScaleType, 1, 32>;
    using ScaleN = ck_tile::MXScalePointer<ScaleType, 1, 32>;

    ScaleM scale_m(reinterpret_cast<ScaleType*>(A_scales_dev_buf.GetDeviceBuffer()));
    ScaleN scale_n(reinterpret_cast<ScaleType*>(B_scales_dev_buf.GetDeviceBuffer()));

    const auto* a_dev_ptr = reinterpret_cast<const ADataType*>(A.data_ptr());
    const auto* b_dev_ptr = reinterpret_cast<const BDataType*>(B.data_ptr());
    auto* c_dev_ptr       = reinterpret_cast<CDataType*>(C.data_ptr());

    const ck_tile::index_t stride_A = K;
    const ck_tile::index_t stride_B = K;
    const ck_tile::index_t stride_C = N;
    const ck_tile::index_t kbatch   = 1;

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
                                    false>(a_dev_ptr,
                                           b_dev_ptr,
                                           c_dev_ptr,
                                           M,
                                           N,
                                           K,
                                           stride_A,
                                           stride_B,
                                           stride_C,
                                           kbatch,
                                           scale_m,
                                           scale_n,
                                           warmup,
                                           repeat);

    return ave_time;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ck_mxgemm_mxfp8",
          &ck_mxgemm_mxfp8,
          "CK MXFP8 GEMM");
}
