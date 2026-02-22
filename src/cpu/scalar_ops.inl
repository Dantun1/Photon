#include <vector>
#include <numeric>   
#include <functional> 
#include <cmath>    

template <typename T>
void NDArray<T>::setitem_scalar(const std::vector<Slice> &slice_ranges, T scalar)
{
    NDArray<T> target_view = this->slice(slice_ranges);
    DimVec target_shape = target_view.get_shape();
    size_t total_size = std::accumulate(target_shape.begin(), target_shape.end(), 1ULL, std::multiplies<size_t>());
    // Odometer logic based on shape/strides of the target view
    T *write_ptr = handle->ptr();
    DimVec indices(target_shape.size(), 0);
    size_t curr_idx = target_view.get_offset();
    for (size_t i = 0; i < total_size; i++)
    {
        write_ptr[curr_idx] = scalar;
        for (int dim = static_cast<int>(target_shape.size()) - 1; dim >= 0; --dim)
        {
            // increment by target view strides
            curr_idx += target_view.get_strides()[dim];
            indices[dim]++;
            if (indices[dim] == target_shape[dim])
            {
                indices[dim] = 0;
                curr_idx -= target_shape[dim] * target_view.get_strides()[dim];
            }
            else
            {
                break;
            }
        }
    }
}

/** Arithmetic scalar ops
 *
 */

template <typename T, typename Op>
NDArray<T> scalar_op_kernel(const NDArray<T> &a, T scalar, Op op)
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
        new_data[i] = op(old_data[curr_idx], scalar);

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
NDArray<T> scalar_add(const NDArray<T> &a, T b)
{
    return scalar_op_kernel(a, b, [](T a, T b)
                            { return a + b; });
}

template <typename T>
NDArray<T> scalar_sub(const NDArray<T> &a, T b)
{
    return scalar_op_kernel(a, b, [](T a, T b)
                            { return a - b; });
}
template <typename T>
NDArray<T> scalar_rsub(const NDArray<T> &a, T b)
{
    return scalar_op_kernel(a, b, [](T a, T b)
                            { return b - a; });
}

template <typename T>
NDArray<T> scalar_div(const NDArray<T> &a, T b)
{
    return scalar_op_kernel(a, b, [](T a, T b)
                            { return a / b; });
}

template <typename T>
NDArray<T> scalar_rdiv(const NDArray<T> &a, T b)
{
    return scalar_op_kernel(a, b, [](T a, T b)
                            { return b / a; });
}

template <typename T>
NDArray<T> scalar_mul(const NDArray<T> &a, T b)
{
    return scalar_op_kernel(a, b, [](T a, T b)
                            { return a * b; });
}

template <typename T>
NDArray<T> scalar_pow(const NDArray<T> &a, T b)
{
    return scalar_op_kernel(a, b, [](T a, T b)
                            { return std::pow(a, b); });
}

