#pragma once

#include <torch/extension.h>

// Launches the CUDA Top-K routine on a 2D float32 CUDA tensor of shape [N, D].
// Returns (values[N, K], indices[N, K]).
std::tuple<at::Tensor, at::Tensor> rtopk_cuda(
    const at::Tensor& input,
    int64_t k,
    int64_t max_iter,
    double precision);


