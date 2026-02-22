#include <vector>
#include <numeric>   
#include <functional> 
#include <cmath>    


template <typename T>
void NDArray<T>::setitem_ewise(const std::vector<Slice> &slice_ranges, const NDArray<T> &source)
{
    NDArray<T> target_view = this->slice(slice_ranges);
    DimVec target_shape = target_view.get_shape();
    size_t total_size = std::accumulate(target_shape.begin(), target_shape.end(), 1ULL, std::multiplies<size_t>());

    // Try broadcasting if doesn't match, will throw error if incompatible.
    NDArray<T> broadcasted_source = (source.get_shape() == target_shape) ? source : source.broadcast(target_shape);

    // Odometer logic but two indices as 2 views being traversed.
    size_t write_idx = target_view.get_offset();
    size_t source_idx = broadcasted_source.get_offset();
    T *write_ptr = handle->ptr();
    const T *source_ptr = broadcasted_source.get_handle()->ptr();
    DimVec indices(target_shape.size(), 0);

    for (size_t i = 0; i < total_size; i++)
    {
        write_ptr[write_idx] = source_ptr[source_idx];
        for (int dim = static_cast<int>(target_shape.size()) - 1; dim >= 0; --dim)
        {
            write_idx += target_view.get_strides()[dim];
            source_idx += broadcasted_source.get_strides()[dim];
            indices[dim]++;
            if (indices[dim] == target_shape[dim])
            {
                indices[dim] = 0;
                write_idx -= target_shape[dim] * target_view.get_strides()[dim];
                source_idx -= target_shape[dim] * broadcasted_source.get_strides()[dim];
            }
            else
            {
                break;
            }
        }
    }
}

/** Ewise arithmetic ops
 *
 *
 */
template <typename T, typename Op>
NDArray<T> ewise_op_kernel(const NDArray<T> &a, const NDArray<T> &b, Op op)
{

    const auto &shape = broadcast_shape(a.get_shape(), b.get_shape());

    NDArray<T> broadcasted_b = (shape == b.get_shape()) ? b : b.broadcast(shape);
    NDArray<T> broadcasted_a = (shape == a.get_shape()) ? a : a.broadcast(shape);
    NDArray<T> target{shape};

    const auto &astrides = broadcasted_a.get_strides();
    const auto &bstrides = broadcasted_b.get_strides();
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());

    T *new_data = target.get_handle()->ptr();
    const T *aptr = broadcasted_a.get_handle()->ptr();
    const T *bptr = broadcasted_b.get_handle()->ptr();

    size_t a_idx = broadcasted_a.get_offset();
    size_t b_idx = broadcasted_b.get_offset();

    DimVec indices(shape.size(), 0);
    for (size_t i = 0; i < total_size; i++)
    {
        new_data[i] = op(aptr[a_idx], bptr[b_idx]);

        for (int dim = static_cast<int>(shape.size()) - 1; dim >= 0; --dim)
        {
            indices[dim]++;
            a_idx += astrides[dim];
            b_idx += bstrides[dim];
            if (indices[dim] < shape[dim])
            {
                break;
            }
            else
            {
                indices[dim] = 0;
                a_idx -= shape[dim] * astrides[dim];
                b_idx -= shape[dim] * bstrides[dim];
            }
        }
    }
    return target;
}

template <typename T>
NDArray<T> ewise_add(const NDArray<T> &a, const NDArray<T> &b)
{
    return ewise_op_kernel(a, b, [](T a, T b)
                           { return a + b; });
}

template <typename T>
NDArray<T> ewise_sub(const NDArray<T> &a, const NDArray<T> &b)
{
    return ewise_op_kernel(a, b, [](T a, T b)
                           { return a - b; });
}

template <typename T>
NDArray<T> ewise_mul(const NDArray<T> &a, const NDArray<T> &b)
{
    return ewise_op_kernel(a, b, [](T a, T b)
                           { return a * b; });
}

template <typename T>
NDArray<T> ewise_pow(const NDArray<T> &a, const NDArray<T> &b)
{
    return ewise_op_kernel(a, b, [](T a, T b)
                           { return std::pow(a, b); });
}

template <typename T>
NDArray<T> ewise_div(const NDArray<T> &a, const NDArray<T> &b)
{
    return ewise_op_kernel(a, b, [](T a, T b)
                           { return a / b; });
}
