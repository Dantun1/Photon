#include <vector>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <numeric>
#include <memory>
#include <utility>


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
