#include <cuda/std/limits>
#include "kernel_config.cuh"

#include <backend_gpu.cuh>

namespace photon
{
    namespace gpu
    {
#include <compact_array_manual.inl>
#include <ndarray_core_gpu.inl>
#include <ndarray_view_gpu.inl>
#include <unary_ops_gpu.inl>
#include <scalar_ops_gpu.inl>
#include <ewise_ops_gpu.inl>
#include <reduction_ops_gpu.inl>
#include <matmul_gpu.inl>

        template class CompactArray<float>;
        template class NDArray<float>;

        template NDArray<float> ewise_add(const NDArray<float> &, const NDArray<float> &);
        template NDArray<float> ewise_sub(const NDArray<float> &, const NDArray<float> &);
        template NDArray<float> ewise_mul(const NDArray<float> &, const NDArray<float> &);
        template NDArray<float> ewise_div(const NDArray<float> &, const NDArray<float> &);
        template NDArray<float> ewise_pow(const NDArray<float> &, const NDArray<float> &);

        template NDArray<float> scalar_add(const NDArray<float> &, float);
        template NDArray<float> scalar_sub(const NDArray<float> &, float);
        template NDArray<float> scalar_mul(const NDArray<float> &, float);
        template NDArray<float> scalar_div(const NDArray<float> &, float);
        template NDArray<float> scalar_pow(const NDArray<float> &, float);

        template NDArray<float> scalar_rsub(const NDArray<float> &, float);
        template NDArray<float> scalar_rdiv(const NDArray<float> &, float);
        template NDArray<float> scalar_rpow(const NDArray<float> &, float);

        template NDArray<float> matmul(const NDArray<float> &, const NDArray<float> &);
    }
}
