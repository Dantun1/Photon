#include <vector>
#include <numeric>   
#include <functional> 
#include <cmath>    

template <typename T, typename Op>
NDArray<T> unary_op_kernel(const NDArray<T> &a, Op op)
{
    NDArray<T> target{a.get_shape()};
    const auto &shape = target.get_shape();
    const auto &strides = a.get_strides();
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());

    T *new_data = target.get_handle()->ptr();
    const T *old_data = a.get_handle()->ptr();

    size_t curr_idx = a.get_offset();
    DimVec indices(shape.size(), 0);
    for (size_t i = 0; i < total_size; i++)
    {
        new_data[i] = op(old_data[curr_idx]);

        for (int dim = static_cast<int>(shape.size()) - 1; dim >= 0; --dim)
        {
            indices[dim]++;
            curr_idx += strides[dim];

            if (indices[dim] < shape[dim])
            {
                break;
            }
            else
            {
                indices[dim] = 0;
                curr_idx -= shape[dim] * strides[dim];
            }
        }
    }
    return target;
}

template <typename T>
NDArray<T> NDArray<T>::neg() const
{
    return unary_op_kernel(*this, [](T scalar)
                           { return -scalar; });
}
template <typename T>
NDArray<T> NDArray<T>::exp() const
{
    return unary_op_kernel(*this, [](T scalar)
                           { return std::exp(scalar); });
}

template <typename T>
NDArray<T> NDArray<T>::log() const
{
    return unary_op_kernel(*this, [](T scalar)
                           { return std::log(scalar); });
}

template <typename T>
NDArray<T> NDArray<T>::sqrt() const
{
    return unary_op_kernel(*this, [](T scalar)
                           { return std::sqrt(scalar); });
}

template <typename T>
NDArray<T> NDArray<T>::sin() const
{
    return unary_op_kernel(*this, [](T scalar)
                           { return std::sin(scalar); });
}

template <typename T>
NDArray<T> NDArray<T>::cos() const
{
    return unary_op_kernel(*this, [](T scalar)
                           { return std::cos(scalar); });
}

template <typename T>
NDArray<T> NDArray<T>::tanh() const
{
    return unary_op_kernel(*this, [](T scalar)
                           { return std::tanh(scalar); });
}
