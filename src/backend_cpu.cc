#include <vector>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <backend_cpu.hpp>

/*
 * Implementation of CompactArray methods.
 */

template <typename T>
CompactArray<T>::CompactArray(size_t size) : data(size) {}

template <typename T>
CompactArray<T>::CompactArray(const std::vector<T> &input) : data(input) {}

template <typename T>
CompactArray<T>::CompactArray(std::vector<T> &&input) : data{std::move(input)} {}

template <typename T>
size_t CompactArray<T>::size() const
{
    return data.size();
}

template <typename T>
void CompactArray<T>::print() const
{
    for (const auto &value : data)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}

template <typename T>
T *CompactArray<T>::ptr()
{
    return data.data();
}

template <typename T>
const T *CompactArray<T>::ptr() const
{
    return data.data();
}

/*
 * Implementation of NDArray methods.
 */

/*
 * Constructors
 */

template <typename T>
NDArray<T>::NDArray(const DimVec &shape) : shape{shape}, offset{0}
{
    // Make a new shared pointer to compact array with total size
    size_t total_size = std::accumulate(this->shape.begin(), this->shape.end(), 1, std::multiplies<size_t>());
    // Handle to zeroed compact array of total size
    handle = std::make_shared<CompactArray<T>>(total_size);
    initialise_strides();
}

template <typename T>
NDArray<T>::NDArray(std::vector<T> data) : offset{0}
{
    size_t size = data.size();
    handle = std::make_shared<CompactArray<T>>(std::move(data));
    shape = {size};
    initialise_strides();
}

template <typename T>
NDArray<T>::NDArray(std::vector<T> data, DimVec shape) : shape{std::move(shape)}, offset{0}
{
    size_t expected_size = std::accumulate(this->shape.begin(), this->shape.end(), 1, std::multiplies<size_t>());

    if (data.size() != expected_size)
    {
        throw std::invalid_argument("Data size does not match shape dimensions");
    }

    handle = std::make_shared<CompactArray<T>>(std::move(data));
    initialise_strides();
}

template <typename T>
NDArray<T>::NDArray(std::shared_ptr<CompactArray<T>> other, DimVec shape, DimVec strides, size_t offset) : handle{std::move(other)}, shape{std::move(shape)}, strides{std::move(strides)}, offset{offset}
{
}

template <typename T>
NDArray<T>::NDArray(std::shared_ptr<CompactArray<T>> other, DimVec shape, size_t offset) : handle{std::move(other)}, shape{std::move(shape)}, offset{offset}
{
    initialise_strides();
}

/**
 * View related member functions.
 *
 *
 */

template <typename T>
NDArray<T> NDArray<T>::make_compact() const
{
    // Need to allocate new compact array with matching shape and row major strides for given data.
    size_t new_size = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    auto new_handle = std::make_shared<CompactArray<T>>(new_size);

    T *new_data = new_handle->ptr();
    const T *old_data = handle->ptr();
    size_t curr_idx = offset;
    DimVec indices(shape.size(), 0);
    for (size_t i = 0; i < new_size; i++)
    {
        new_data[i] = old_data[curr_idx];

        for (int dim = static_cast<int>(shape.size()) - 1; dim >= 0; --dim)
        {
            indices[dim]++;
            curr_idx += strides[dim];

            if (indices[dim] < shape[dim])
            {
                // dim index less than shape dim, break as no carry handling needed
                break;
            }
            else
            {
                // reset dim index to 0, loop handles incrementing next dim
                indices[dim] = 0;
                // subtract total offset for current dim
                curr_idx -= shape[dim] * strides[dim];
            }
        }
    }
    return NDArray<T>(std::move(new_handle), shape, 0);
}

template <typename T>
NDArray<T> NDArray<T>::reshape(const DimVec &new_shape) const
{

    // New shape must have same elements.
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
    size_t current_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    if (new_size != current_size)
    {
        throw std::invalid_argument("New shape must have same number of elements as current shape");
    }

    // If contiguous, just change shape and strides, else make compact and then change shape and strides.

    NDArray<T> source = is_contiguous() ? *this : make_compact();

    DimVec new_strides(new_shape.size());
    size_t dim_stride = 1;
    for (int i = static_cast<int>(new_shape.size()) - 1; i >= 0; --i)
    {
        new_strides[i] = dim_stride;
        dim_stride *= new_shape[i];
    }
    return NDArray<T>(source.handle, new_shape, new_strides, source.offset);
}

template <typename T>
NDArray<T> NDArray<T>::slice(const std::vector<Slice> &slice_ranges) const
{
    size_t new_offset = offset;
    DimVec new_strides;
    DimVec new_shape;

    for (size_t i = 0; i < slice_ranges.size(); i++)
    {
        const auto &[start, stop, step, is_index] = slice_ranges[i];

        // shift offset (based on original strides)
        new_offset += start * strides[i];

        // if just scalar value for dim, we don't include in the
        // shape/strides, just shift offset
        if (is_index)
        {
            continue;
        }
        int64_t num_elements = (std::abs(stop - start) + std::abs(step) - 1) / std::abs(step);
        new_shape.push_back(static_cast<size_t>(num_elements)); // round up division for shape dim
        new_strides.push_back(strides[i] * step);
    }

    for (size_t i = slice_ranges.size(); i < shape.size(); i++)
    {
        // dims not in slice_ranges are unchanged, just add to new shape and strides.
        new_shape.push_back(shape[i]);
        new_strides.push_back(strides[i]);
    }

    return NDArray<T>(handle, new_shape, new_strides, new_offset);
}

template <typename T>
NDArray<T> NDArray<T>::transpose(const DimVec &axes) const
{
    if (axes.size() != shape.size())
    {
        throw std::invalid_argument("Invalid number of axes for transpose: must match number of dimensions");
    }
    DimVec new_shape(shape.size());
    DimVec new_strides(strides.size());
    for (size_t i = 0; i < axes.size(); i++)
    {
        if (axes[i] >= shape.size() or axes[i] < 0)
        {
            throw std::invalid_argument("Invalid axis index for transpose: must be between 0 and number of dimensions");
        }
        // Permute the shape, swap the strides according to the new order.
        new_shape[i] = shape[axes[i]];
        new_strides[i] = strides[axes[i]];
    }
    return NDArray<T>(handle, new_shape, new_strides, offset);
}

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

template <typename T>
DimVec broadcast_shape(const DimVec &s1, const DimVec &s2)
{
    int dims = static_cast<int>(std::max(s1.size(), s2.size()));
    DimVec out(dims, 1);

    // if dim
    for (size_t k = 0; k < dims; ++k)
    {
        const size_t d1 = (k < s1.size()) ? s1[dims - 1 - k] : 1;
        const size_t d2 = (k < s2.size()) ? s2[dims - 1 - k] : 1;

        if (d1 == d2 || d1 == 1 || d2 == 1)
        {
            out[dims - 1 - k] = std::max(d1, d2);
        }
        else
        {
            throw std::invalid_argument("Cannot broadcast arrays: incompatible shapes");
        }
    }

    return out;
}

template <typename T>
NDArray<T> NDArray<T>::broadcast(const DimVec &new_shape) const
{
    if (new_shape.size() < shape.size())
    {
        throw std::invalid_argument("Cannot broadcast to fewer dimensions");
    }
    DimVec new_strides(new_shape.size(), 0);

    for (int i = static_cast<int>(new_shape.size()) - 1, j = static_cast<int>(shape.size()) - 1; i >= 0; i--, j--)
    {
        if (j >= 0)
        {
            if (shape[j] == new_shape[i])
            {
                new_strides[i] = strides[j];
            }
            else if (shape[j] == 1)
            {
                new_strides[i] = 0;
            }
            else
            {
                throw std::invalid_argument("Cannot broadcast: incompatible shapes");
            }
        }
    }

    return NDArray<T>(handle, new_shape, new_strides, offset);
}

/**
 * Arithmetic operations.
 *
 * Return new NDArrays with new, compact handles.
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

/**
 *  Getters and utility methods
 *
 */

template <typename T>
bool NDArray<T>::has_row_major_strides() const
{
    size_t expected_stride = 1;
    // Ignoring 1 dims, check if strides match the expected row major strides for the shape
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
    {
        if (shape[i] > 1)
        {
            if (strides[i] != expected_stride)
                return false;
            expected_stride *= shape[i];
        }
    }
    return true;
}

template <typename T>
bool NDArray<T>::has_size_matching_shape() const
{
    size_t expected_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    return (handle->size() == expected_size);
}

template <typename T>
void NDArray<T>::initialise_strides()
{
    // Need strides for each shape.size()
    strides.resize(shape.size());
    // Each prior dim must skip over all elements of later dims.
    // Accumulate product
    size_t dim_stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
    {
        strides[i] = dim_stride;
        dim_stride *= shape[i];
    }
}

template <typename T>
DimVec NDArray<T>::get_shape() const
{
    return shape;
}

template <typename T>
DimVec NDArray<T>::get_strides() const
{
    return strides;
}

template <typename T>
size_t NDArray<T>::get_offset() const
{
    return offset;
}

template <typename T>
std::shared_ptr<CompactArray<T>> NDArray<T>::get_handle() const
{
    return handle;
}

template <typename T>
void NDArray<T>::print() const
{
    std::cout << "Shape: [";
    for (auto dim : shape)
    {
        std::cout << dim << " ";
    }
    std::cout << "], Strides: [";
    for (auto stride : strides)
    {
        std::cout << stride << " ";
    }
    std::cout << "], Offset: " << offset << std::endl;

    std::cout << "Data: ";
    const T *data_ptr = handle->ptr();
    // Print 10 elems to track handle
    for (size_t i = 0; i < 20; i++)
    {
        std::cout << data_ptr[i] << " ";
    }
}

template <typename T>
bool NDArray<T>::is_contiguous() const
{
    return has_row_major_strides();
}

template class CompactArray<float>;
template class NDArray<float>;