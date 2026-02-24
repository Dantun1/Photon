#include <numeric>
#include <memory>
#include <vector>
#include <utility>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <view_helpers.inl>


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
