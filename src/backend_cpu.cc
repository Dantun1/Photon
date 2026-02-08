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
    size_t total_size = 1;
    for (size_t dim : shape)
    {
        total_size *= dim;
    }
    // Handle to zeroed compact array of total size
    handle = std::make_shared<CompactArray<T>>(total_size);
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

/*
 * Getters and utility methods
 */

template <typename T>
bool NDArray<T>::has_row_major_strides() const
{
    // Check if strides match row major order for the shape.
    size_t expected_stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
    {
        if (strides[i] != expected_stride)
        {
            return false;
        }
        expected_stride *= shape[i];
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
    // Row major strides, zero offset, total size matches elements in shape of view.
    return (has_row_major_strides() && offset == 0 && has_size_matching_shape());
}

template class CompactArray<float>;
template class NDArray<float>;