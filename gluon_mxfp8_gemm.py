import statistics
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


try:
    from triton.backends.amd.compiler import HIPOptions
    if not hasattr(HIPOptions, "maxnreg"):
        HIPOptions.maxnreg = None
except Exception:
    pass

try:
    import triton._utils as triton_utils
    if hasattr(triton_utils, "type_canonicalisation_dict"):
        triton_utils.type_canonicalisation_dict.setdefault("float8_e8m0fnu", "u8")
except Exception:
    pass


def is_hip_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip":
        return False
    props = torch.cuda.get_device_properties(0)
    name = props.name.lower()
    return ("gfx950" in name) or ("cdna4" in name)


def ceil_div(x, y):
    return (x + y - 1) // y


def ceil_to_multiple(x, multiple):
    return ceil_div(x, multiple) * multiple


FP8_MAX_E4M3 = 448.0


def _pow2_scale_from_maxabs(maxabs: torch.Tensor) -> torch.Tensor:
    safe = torch.where(maxabs > 0, maxabs, torch.ones_like(maxabs))
    raw = safe / FP8_MAX_E4M3
    raw = torch.where(raw > 0, raw, torch.ones_like(raw))
    log2_scale = torch.ceil(torch.log2(raw))
    scale = torch.pow(torch.tensor(2.0, device=maxabs.device, dtype=torch.float32), log2_scale)
    scale = torch.where(maxabs > 0, scale, torch.ones_like(scale))
    return scale


def quantize_a_mx32_fp8(a_bf16: torch.Tensor):
    M, K = a_bf16.shape
    assert K % 32 == 0
    qk = K // 32
    a_blocks = a_bf16.float().view(M, qk, 32)
    block_max = a_blocks.abs().amax(dim=2)
    scales_fp32 = _pow2_scale_from_maxabs(block_max)
    q_fp32 = (a_blocks / scales_fp32.unsqueeze(-1)).reshape(M, K)
    q = q_fp32.to(torch.float8_e4m3fn)
    scales = scales_fp32.to(torch.float8_e8m0fnu)
    return q.contiguous(), scales.contiguous()


def quantize_b_mx32_fp8(b_bf16: torch.Tensor):
    K, N = b_bf16.shape
    assert K % 32 == 0
    # Kernel expects logical B_scale as [N, K//32]
    b_nk = b_bf16.transpose(0, 1).contiguous()
    N2, K2 = b_nk.shape
    assert N2 == N and K2 == K
    qk = K // 32
    b_blocks = b_nk.float().view(N, qk, 32)
    block_max = b_blocks.abs().amax(dim=2)
    scales_fp32 = _pow2_scale_from_maxabs(block_max)
    q_nk = (b_blocks / scales_fp32.unsqueeze(-1)).reshape(N, K).to(torch.float8_e4m3fn)
    q_kn = q_nk.transpose(0, 1).contiguous()
    scales_nq = scales_fp32.to(torch.float8_e8m0fnu)
    return q_kn.contiguous(), scales_nq.contiguous()


def make_mxfp8_data(M, N, K, device="cuda"):
    a_bf16 = torch.randn((M, K), device=device, dtype=torch.bfloat16)
    b_bf16 = torch.randn((K, N), device=device, dtype=torch.bfloat16)

    A, A_scale = quantize_a_mx32_fp8(a_bf16)
    B, B_scale = quantize_b_mx32_fp8(b_bf16)
    return A, B, A_scale, B_scale


def pad_k_mxfp8(A, B, A_scale, B_scale, BLOCK_K=128):
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb

    K_pad = ceil_to_multiple(K, BLOCK_K)
    QK = ceil_div(K, 32)
    QK_pad = ceil_div(K_pad, 32)

    if K_pad == K:
        return A, B, A_scale, B_scale, K_pad

    A_pad = torch.zeros((M, K_pad), device=A.device, dtype=A.dtype)
    B_pad = torch.zeros((K_pad, N), device=B.device, dtype=B.dtype)
    A_pad[:, :K] = A
    B_pad[:K, :] = B

    A_scale_pad = torch.ones((M, QK_pad), device=A_scale.device, dtype=torch.float32).to(torch.float8_e8m0fnu)
    B_scale_pad = torch.ones((N, QK_pad), device=B_scale.device, dtype=torch.float32).to(torch.float8_e8m0fnu)
    A_scale_pad[:, :QK] = A_scale
    B_scale_pad[:, :QK] = B_scale

    return A_pad, B_pad, A_scale_pad, B_scale_pad, K_pad


def expand_scales_32(scales, K):
    return scales.float().repeat_interleave(32, dim=1)[:, :K]


def shuffle_scales_cdna4_mxfp8(scales: torch.Tensor, mfma_nonkdim: int):
    """
    Input logical shape: [rows, K//32]
    Output packed shape: [rows//32, (K//32) * 32]

    For mfma_nonkdim == 16:
      logical [32, 8] scale tile is split into four [16, 4] chunks:
        op0 = top-left
        op1 = top-right
        op2 = bottom-left
        op3 = bottom-right
      and packed in order:
        op0, op2, op1, op3
    """
    scales_shuffled = scales.clone()
    sm, sn = scales_shuffled.shape
    assert sm % 32 == 0
    assert sn % 4 == 0

    if mfma_nonkdim != 16:
        raise ValueError(f"optimized path only supports mfma_nonkdim=16, got {mfma_nonkdim}")

    scales_shuffled = scales_shuffled.view(sm // 32, 2, 16, sn // 4, 4)
    scales_shuffled = scales_shuffled.permute(0, 3, 1, 2, 4).contiguous()
    scales_shuffled = scales_shuffled.view(sm // 32, sn * 32)
    return scales_shuffled.contiguous()


@gluon.jit
def blocked_mxfp8_kernel_preshuffled_scales_opt16(
    out_ptr,
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    M,
    N,
    K,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    gl.static_assert(BLOCK_M == 128)
    gl.static_assert(BLOCK_N == 128)
    gl.static_assert(BLOCK_K == 128)

    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)

    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[2, 2],
    )

    a_load_layout: gl.constexpr = gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])
    b_load_layout: gl.constexpr = gl.BlockedLayout([1, 16], [32, 2], [4, 1], [1, 0])

    a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mfma_layout,
        k_width=16,
    )
    b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1,
        parent=mfma_layout,
        k_width=16,
    )

    a_scale_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        a_layout, [BLOCK_M, BLOCK_K // 32]
    )
    b_scale_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        b_layout, [BLOCK_N, BLOCK_K // 32]
    )

    c_store_layout: gl.constexpr = gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])
    scale_raw_layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [4, 1], [1, 0])

    acc = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mfma_layout)

    a_m_idx = off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, a_load_layout))[:, None]
    b_n_idx = off_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, b_load_layout))[None, :]

    a_scale_row = (off_m // 32) + gl.arange(
        0, BLOCK_M // 32, layout=gl.SliceLayout(1, scale_raw_layout)
    )[:, None]
    b_scale_row = (off_n // 32) + gl.arange(
        0, BLOCK_N // 32, layout=gl.SliceLayout(1, scale_raw_layout)
    )[:, None]

    a_scale_col_base = gl.arange(
        0, BLOCK_K, layout=gl.SliceLayout(0, scale_raw_layout)
    )[None, :]
    b_scale_col_base = gl.arange(
        0, BLOCK_K, layout=gl.SliceLayout(0, scale_raw_layout)
    )[None, :]

    a_k_idx_base = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, a_load_layout))[None, :]
    b_k_idx_base = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, b_load_layout))[:, None]

    qh: gl.constexpr = BLOCK_K // 128
    qk: gl.constexpr = BLOCK_K // 32

    for k0 in range(0, K, BLOCK_K):
        a_offs_k = k0 + a_k_idx_base
        a = gl.amd.cdna4.buffer_load(a_ptr, a_m_idx * K + a_offs_k)
        a = gl.convert_layout(a, a_layout)

        b_offs_k = k0 + b_k_idx_base
        b = gl.amd.cdna4.buffer_load(b_ptr, b_offs_k * N + b_n_idx)
        b = gl.convert_layout(b, b_layout)

        a_scale_col = (k0 // 32) * 32 + a_scale_col_base
        b_scale_col = (k0 // 32) * 32 + b_scale_col_base

        a_scale_raw = gl.amd.cdna4.buffer_load(
            a_scale_ptr,
            a_scale_row * ((K // 32) * 32) + a_scale_col,
        )
        b_scale_raw = gl.amd.cdna4.buffer_load(
            b_scale_ptr,
            b_scale_row * ((K // 32) * 32) + b_scale_col,
        )

        a_scale = a_scale_raw.reshape(
            BLOCK_M // 32, qh, 2, 16, 4
        ).permute(0, 2, 3, 1, 4).reshape(BLOCK_M, qk)

        b_scale = b_scale_raw.reshape(
            BLOCK_N // 32, qh, 2, 16, 4
        ).permute(0, 2, 3, 1, 4).reshape(BLOCK_N, qk)

        a_scale = gl.convert_layout(a_scale, a_scale_layout)
        b_scale = gl.convert_layout(b_scale, b_scale_layout)

        acc = gl.amd.cdna4.mfma_scaled(
            a, a_scale, "e4m3",
            b, b_scale, "e4m3",
            acc,
        )

    c = gl.convert_layout(acc, c_store_layout)

    out_offs_m = off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, c_store_layout))[:, None]
    out_offs_n = off_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, c_store_layout))[None, :]
    gl.amd.cdna4.buffer_store(c, out_ptr, out_offs_m * N + out_offs_n)


def prepare_mxfp8_gemm_preshuffled_scales(
    A,
    B,
    A_scale,
    B_scale,
    BLOCK_M=128,
    BLOCK_N=128,
    BLOCK_K=128,
):
    assert A.is_cuda and B.is_cuda and A_scale.is_cuda and B_scale.is_cuda
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb
    assert A.dtype == torch.float8_e4m3fn
    assert B.dtype == torch.float8_e4m3fn
    assert A_scale.dtype == torch.float8_e8m0fnu
    assert B_scale.dtype == torch.float8_e8m0fnu

    assert BLOCK_M == 128
    assert BLOCK_N == 128
    assert BLOCK_K == 128

    A_pad, B_pad, A_scale_pad, B_scale_pad, K_pad = pad_k_mxfp8(
        A, B, A_scale, B_scale, BLOCK_K=BLOCK_K
    )

    assert M % BLOCK_M == 0, f"M={M} must be divisible by BLOCK_M={BLOCK_M}"
    assert N % BLOCK_N == 0, f"N={N} must be divisible by BLOCK_N={BLOCK_N}"
    assert K_pad % BLOCK_K == 0
    assert M % 32 == 0
    assert N % 32 == 0

    A_scale_shuf = shuffle_scales_cdna4_mxfp8(A_scale_pad.contiguous(), 16)
    B_scale_shuf = shuffle_scales_cdna4_mxfp8(B_scale_pad.contiguous(), 16)

    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    grid = (M // BLOCK_M, N // BLOCK_N)

    return {
        "A_pad": A_pad,
        "B_pad": B_pad,
        "A_scale_shuf": A_scale_shuf,
        "B_scale_shuf": B_scale_shuf,
        "C": C,
        "M": M,
        "N": N,
        "K_pad": K_pad,
        "grid": grid,
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
    }


def launch_mxfp8_gemm_preshuffled_scales(prepared):
    blocked_mxfp8_kernel_preshuffled_scales_opt16[prepared["grid"]](
        prepared["C"],
        prepared["A_pad"],
        prepared["B_pad"],
        prepared["A_scale_shuf"],
        prepared["B_scale_shuf"],
        prepared["M"],
        prepared["N"],
        prepared["K_pad"],
        BLOCK_M=prepared["BLOCK_M"],
        BLOCK_N=prepared["BLOCK_N"],
        BLOCK_K=prepared["BLOCK_K"],
        num_warps=4,
        num_stages=1,
    )
    return prepared["C"]


def reference_mxfp8_gemm(A, B, A_scale, B_scale, BLOCK_K=128):
    A_pad, B_pad, A_scale_pad, B_scale_pad, K_pad = pad_k_mxfp8(
        A, B, A_scale, B_scale, BLOCK_K=BLOCK_K
    )

    A_scale_full = expand_scales_32(A_scale_pad, K_pad)
    B_scale_full = expand_scales_32(B_scale_pad, K_pad).T

    A_ref = A_pad.float() * A_scale_full
    B_ref = B_pad.float() * B_scale_full
    return A_ref @ B_ref


def benchmark_forward_ms(fn, warmup: int = 20, repeat: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        fn()
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / float(repeat)


SHAPES = [
    ("Llama-2-7B MBS=1", 4096, 12288, 4096),
    ("Llama-2-7B MBS=1", 4096, 4096, 4096),
    ("Llama-2-7B MBS=1", 4096, 22016, 4096),
    ("Llama-2-7B MBS=1", 4096, 4096, 11008),
    ("Llama-2-7B MBS=2", 8192, 12288, 4096),
    ("Llama-2-7B MBS=2", 8192, 4096, 4096),
    ("Llama-2-7B MBS=2", 8192, 22016, 4096),
    ("Llama-2-7B MBS=2", 8192, 4096, 11008),
    ("Llama-2-7B MBS=4", 16384, 12288, 4096),
    ("Llama-2-7B MBS=4", 16384, 4096, 4096),
    ("Llama-2-7B MBS=4", 16384, 22016, 4096),
    ("Llama-2-7B MBS=4", 16384, 4096, 11008),
    ("Llama-2-70B MBS=1", 4096, 10240, 8192),
    ("Llama-2-70B MBS=1", 4096, 8192, 8192),
    ("Llama-2-70B MBS=1", 4096, 57344, 8192),
    ("Llama-2-70B MBS=1", 4096, 8192, 28672),
    ("Llama-2-70B MBS=2", 8192, 10240, 8192),
    ("Llama-2-70B MBS=2", 8192, 8192, 8192),
    ("Llama-2-70B MBS=2", 8192, 57344, 8192),
    ("Llama-2-70B MBS=2", 8192, 8192, 28672),
    ("Llama-2-70B MBS=4", 16384, 10240, 8192),
    ("Llama-2-70B MBS=4", 16384, 8192, 8192),
    ("Llama-2-70B MBS=4", 16384, 57344, 8192),
    ("Llama-2-70B MBS=4", 16384, 8192, 28672),
    ("Llama-3.1-8B MBS=1", 8192, 6144, 4096),
    ("Llama-3.1-8B MBS=1", 8192, 4096, 4096),
    ("Llama-3.1-8B MBS=1", 8192, 28672, 4096),
    ("Llama-3.1-8B MBS=1", 8192, 4096, 14336),
    ("Llama-3.1-8B MBS=2", 16384, 6144, 4096),
    ("Llama-3.1-8B MBS=2", 16384, 4096, 4096),
    ("Llama-3.1-8B MBS=2", 16384, 28672, 4096),
    ("Llama-3.1-8B MBS=2", 16384, 4096, 14336),
    ("Llama-3.1-8B MBS=4", 32768, 6144, 4096),
    ("Llama-3.1-8B MBS=4", 32768, 4096, 4096),
    ("Llama-3.1-8B MBS=4", 32768, 28672, 4096),
    ("Llama-3.1-8B MBS=4", 32768, 4096, 14336),
    ("Llama-3.1-405B MBS=1", 8192, 18432, 16384),
    ("Llama-3.1-405B MBS=1", 8192, 16384, 16384),
    ("Llama-3.1-405B MBS=1", 8192, 106496, 16384),
    ("Llama-3.1-405B MBS=1", 8192, 16384, 53248),
    ("Llama-3.1-405B MBS=2", 16384, 18432, 16384),
    ("Llama-3.1-405B MBS=2", 16384, 16384, 16384),
    ("Llama-3.1-405B MBS=2", 16384, 106496, 16384),
    ("Llama-3.1-405B MBS=2", 16384, 16384, 53248),
    ("Llama-3.1-405B MBS=4", 32768, 18432, 16384),
    ("Llama-3.1-405B MBS=4", 32768, 16384, 16384),
    ("Llama-3.1-405B MBS=4", 32768, 106496, 16384),
    ("Llama-3.1-405B MBS=4", 32768, 16384, 53248),
    ("Qwen2.5-7B MBS=1", 8192, 4608, 3584),
    ("Qwen2.5-7B MBS=1", 8192, 3584, 3584),
    ("Qwen2.5-7B MBS=1", 8192, 37888, 3584),
    ("Qwen2.5-7B MBS=1", 8192, 3584, 18944),
    ("Qwen2.5-7B MBS=2", 16384, 4608, 3584),
    ("Qwen2.5-7B MBS=2", 16384, 3584, 3584),
    ("Qwen2.5-7B MBS=2", 16384, 37888, 3584),
    ("Qwen2.5-7B MBS=2", 16384, 3584, 18944),
    ("Qwen2.5-7B MBS=4", 32768, 4608, 3584),
    ("Qwen2.5-7B MBS=4", 32768, 3584, 3584),
    ("Qwen2.5-7B MBS=4", 32768, 37888, 3584),
    ("Qwen2.5-7B MBS=4", 32768, 3584, 18944),
    ("Qwen2.5-72B MBS=1", 8192, 10240, 8192),
    ("Qwen2.5-72B MBS=1", 8192, 8192, 8192),
    ("Qwen2.5-72B MBS=1", 8192, 59136, 8192),
    ("Qwen2.5-72B MBS=1", 8192, 8192, 29568),
    ("Qwen2.5-72B MBS=2", 16384, 10240, 8192),
    ("Qwen2.5-72B MBS=2", 16384, 8192, 8192),
    ("Qwen2.5-72B MBS=2", 16384, 59136, 8192),
    ("Qwen2.5-72B MBS=2", 16384, 8192, 29568),
    ("Qwen2.5-72B MBS=4", 32768, 10240, 8192),
    ("Qwen2.5-72B MBS=4", 32768, 8192, 8192),
    ("Qwen2.5-72B MBS=4", 32768, 59136, 8192),
    ("Qwen2.5-72B MBS=4", 32768, 8192, 29568),
    ("Mistral-7B MBS=1", 4096, 6144, 4096),
    ("Mistral-7B MBS=1", 4096, 4096, 4096),
    ("Mistral-7B MBS=1", 4096, 28672, 4096),
    ("Mistral-7B MBS=1", 4096, 4096, 14336),
    ("Mistral-7B MBS=2", 8192, 6144, 4096),
    ("Mistral-7B MBS=2", 8192, 4096, 4096),
    ("Mistral-7B MBS=2", 8192, 28672, 4096),
    ("Mistral-7B MBS=2", 8192, 4096, 14336),
    ("Mistral-7B MBS=4", 16384, 6144, 4096),
    ("Mistral-7B MBS=4", 16384, 4096, 4096),
    ("Mistral-7B MBS=4", 16384, 28672, 4096),
    ("Mistral-7B MBS=4", 16384, 4096, 14336),
]


def dedup_shapes(shapes):
    seen = set()
    out = []
    for item in shapes:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def bench_one_shape(label, M, N, K, device="cuda", block_m=128, block_n=128, block_k=128,
                    warmup=20, repeat=50):
    A, B, A_scale, B_scale = make_mxfp8_data(M, N, K, device=device)
    prepared = prepare_mxfp8_gemm_preshuffled_scales(
        A, B, A_scale, B_scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    gemm_fn = lambda: launch_mxfp8_gemm_preshuffled_scales(prepared)

    _ = gemm_fn()
    torch.cuda.synchronize()

    fwd_ms = benchmark_forward_ms(gemm_fn, warmup=warmup, repeat=repeat)
    flops = 2.0 * M * N * K
    tflops = flops / (fwd_ms * 1e-3) / 1e12

    return {
        "label": label,
        "M": M,
        "N": N,
        "K": K,
        "fwd_ms": fwd_ms,
        "tflops": tflops,
    }


def run_bench():
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda/HIP device required")
    if not is_hip_cdna4():
        raise RuntimeError("This example requires AMD CDNA4 / MI350X")

    device = "cuda"
    block_m = 128
    block_n = 128
    block_k = 128

    shapes = dedup_shapes(SHAPES)
    results = []

    print("device:", torch.cuda.get_device_properties(0).name)
    print("num_shapes:", len(shapes))
    print()

    for idx, (label, M, N, K) in enumerate(shapes, start=1):
        result = bench_one_shape(
            label, M, N, K,
            device=device,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            warmup=20,
            repeat=50,
        )
        results.append(result)
        print(
            f"[{idx:02d}/{len(shapes)}] {label:20s} "
            f"M={M:6d} N={N:6d} K={K:6d} "
            f"fwd_ms={result['fwd_ms']:.4f} tflops={result['tflops']:.2f}"
        )

    tflops_list = [r["tflops"] for r in results]
    avg_tflops = statistics.mean(tflops_list)
    median_tflops = statistics.median(tflops_list)

    print()
    print(f"Average TFLOPS: {avg_tflops:.2f}")
    print(f"Median TFLOPS:  {median_tflops:.2f}")


if __name__ == "__main__":
    run_bench()
