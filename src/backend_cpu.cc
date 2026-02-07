#include <vector>
#include <iostream>
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
template <typename T>
void NDArray<T>::compute_strides()
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
NDArray<T>::NDArray(const std::vector<size_t> &shape) : shape{shape}, offset{0}
{
    // Make a new shared pointer to compact array with total size
    size_t total_size = 1;
    for (size_t dim : shape)
    {
        total_size *= dim;
    }
    // Handle to zeroed compact array of total size
    handle = std::make_shared<CompactArray<T>>(total_size);
    compute_strides();
}

template <typename T>
NDArray<T>::NDArray(std::vector<T> data, const std::vector<size_t> shape) : shape{std::move(shape)}, offset{0}
{
    // TODO
}

template <typename T>
NDArray<T>::NDArray(std::shared_ptr<CompactArray<T>> handle, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset) : handle{std::move(handle)}, shape{std::move(shape)}, strides{std::move(strides)}, offset{offset}
{
    // TODO
}

template class CompactArray<float>;
template class NDArray<float>;