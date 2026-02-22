#include <backend_cpu.hpp>
// Instantiate float versions of templates 
template class CompactArray<float>;
template class NDArray<float>;
template NDArray<float> ewise_add(const NDArray<float>&, const NDArray<float>&);
template NDArray<float> ewise_sub(const NDArray<float>&, const NDArray<float>&);
template NDArray<float> ewise_mul(const NDArray<float>&, const NDArray<float>&);
template NDArray<float> ewise_div(const NDArray<float>&, const NDArray<float>&);
template NDArray<float> ewise_pow(const NDArray<float>&, const NDArray<float>&);

template NDArray<float> scalar_add(const NDArray<float>&, float);
template NDArray<float> scalar_sub(const NDArray<float>&, float);
template NDArray<float> scalar_mul(const NDArray<float>&, float);
template NDArray<float> scalar_div(const NDArray<float>&, float);
template NDArray<float> scalar_pow(const NDArray<float>&, float);

template NDArray<float> scalar_rsub(const NDArray<float>&, float);
template NDArray<float> scalar_rdiv(const NDArray<float>&, float);
