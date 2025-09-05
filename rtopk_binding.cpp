#include <torch/extension.h>
#include "rtopk_ops.h"

std::tuple<at::Tensor, at::Tensor> rtopk(
    const at::Tensor& input,
    int64_t k,
    int64_t max_iter = 10000,
    double precision = 0.0) {
    return rtopk_cuda(input, k, max_iter, precision);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rtopk", &rtopk, "RTopK CUDA (float32)",
          py::arg("input"),
          py::arg("k"),
          py::arg("max_iter") = 10000,
          py::arg("precision") = 0.0);
}


