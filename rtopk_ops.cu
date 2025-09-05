#include "rtopk_ops.h"
#include "rtopk_kernel.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {

inline int choose_warps_per_block(int dim_origin) {
    if (dim_origin <= 1024) {
        return 8;
    } else if (dim_origin <= 2048) {
        return 4;
    } else if (dim_origin <= 4096) {
        return 2;
    } else {
        return 1;
    }
}

template <int WARPS_PER_BLOCK>
inline void launch_kernel(
    const float* data_ptr,
    float* value_ptr,
    int* index_ptr,
    int N,
    int dim_origin,
    int k,
    int max_iter,
    float precision,
    cudaStream_t stream)
{
    const dim3 block_dim(WARPS_PER_BLOCK * 32);
    const dim3 grid_dim((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    const size_t shared_mem_bytes = static_cast<size_t>(WARPS_PER_BLOCK) * static_cast<size_t>(dim_origin) * sizeof(float);

    rtopk_kernel<WARPS_PER_BLOCK><<<grid_dim, block_dim, shared_mem_bytes, stream>>>(
        const_cast<float*>(data_ptr), value_ptr, index_ptr,
        N, dim_origin, k, max_iter, precision);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // anonymous namespace

std::tuple<at::Tensor, at::Tensor> rtopk_cuda(
    const at::Tensor& input,
    int64_t k,
    int64_t max_iter,
    double precision)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [N, D]");
    TORCH_CHECK(k > 0, "k must be positive");
    TORCH_CHECK(max_iter >= 0, "max_iter must be non-negative");

    at::Tensor input_contig = input.contiguous();
    const int64_t N64 = input_contig.size(0);
    const int64_t D64 = input_contig.size(1);
    TORCH_CHECK(k <= D64, "k must be <= input.size(1)");

    const int N = static_cast<int>(N64);
    const int dim_origin = static_cast<int>(D64);
    const int kk = static_cast<int>(k);
    const int max_it = static_cast<int>(max_iter);
    const float prec = static_cast<float>(precision);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input_contig));
    auto stream = at::cuda::getCurrentCUDAStream();

    at::Tensor values = at::empty({N64, k}, input_contig.options());
    at::Tensor indices = at::empty({N64, k}, input_contig.options().dtype(at::kInt));

    const float* data_ptr = input_contig.data_ptr<float>();
    float* value_ptr = values.data_ptr<float>();
    int* index_ptr = indices.data_ptr<int>();

    const int w = choose_warps_per_block(dim_origin);
    switch (w) {
        case 8:
            launch_kernel<8>(data_ptr, value_ptr, index_ptr, N, dim_origin, kk, max_it, prec, stream.stream());
            break;
        case 4:
            launch_kernel<4>(data_ptr, value_ptr, index_ptr, N, dim_origin, kk, max_it, prec, stream.stream());
            break;
        case 2:
            launch_kernel<2>(data_ptr, value_ptr, index_ptr, N, dim_origin, kk, max_it, prec, stream.stream());
            break;
        default:
            launch_kernel<1>(data_ptr, value_ptr, index_ptr, N, dim_origin, kk, max_it, prec, stream.stream());
            break;
    }

    return {values, indices};
}


