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
    // TODO: Negative slice values, add testing for this.
    if (slice_ranges.size() != shape.size())
    {
        throw std::invalid_argument("Number of slice ranges must match number of dimensions");
    }
    size_t new_offset = offset;
    DimVec new_strides(strides.size());
    DimVec sliced_shape(shape.size());
    for (size_t i = 0; i < shape.size(); i++)
    {
        const auto &[start, stop, step] = slice_ranges[i];
        if (step <= 0)
        {
            throw std::invalid_argument("Step size must be positive");
        }
        const int64_t dim = static_cast<int64_t>(shape[i]);
        if (start < 0 || stop < 0 || start > stop || stop > dim)
        {
            throw std::invalid_argument("Slice range out of bounds");
        }

        sliced_shape[i] = (stop - start + step - 1) / step;
        new_offset += start * strides[i];
        new_strides[i] = strides[i] * step;
    }
    return NDArray<T>(handle, sliced_shape, new_strides, new_offset);
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
            throw std::invalid_argument("Invalid axis index for transpose: must be less than number of dimensions");
        }
        // Permute the shape, swap the strides according to the new order.
        new_shape[i] = shape[axes[i]];
        new_strides[i] = strides[axes[i]];
    }
    return NDArray<T>(handle, new_shape, new_strides, offset);
}

// template <typename T>
// NDArray<T> NDArray<T>::broadcast_to(const DimVec &new_shape) const
// {
//     return ...;
// }

/*
 * Getters and utility methods
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

// template <typename T>
// DimVec NDArray<T>::get_sliced_shape(const std::vector<Slice> &slice) const
// {

// }

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
}

template <typename T>
bool NDArray<T>::is_contiguous() const
{
    return has_row_major_strides();
}

template class CompactArray<float>;
template class NDArray<float>;