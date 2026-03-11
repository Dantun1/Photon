#include <vector>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <numeric>
#include <memory>
#include <utility>

/*
 * Constructors
 */

// zeroed array from shape
template <typename T>
NDArray<T>::NDArray(const DimVec &shape) : _shape{shape}, _offset{0}
{
    // Make a new shared pointer to compact array with total size
    size_t total_size = std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<size_t>());
    // Handle to zeroed compact array of total size
    _handle = std::make_shared<CompactArray<T>>(total_size);
    initialise_strides();
}
// 1d array from data
template <typename T>
NDArray<T>::NDArray(std::vector<T> data) : _shape{data.size()}, _offset{0} 
{
    _handle = std::make_shared<CompactArray<T>>(std::move(data));
    initialise_strides();
} 

// ndarray given data + shape
template <typename T>
NDArray<T>::NDArray(std::vector<T> data, DimVec shape) : _shape{std::move(shape)}, _offset{0}
{
    size_t expected_size = std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<size_t>());

    if (data.size() != expected_size)
    {
        throw std::invalid_argument("Data size does not match shape dimensions");
    }
    _handle = std::make_shared<CompactArray<T>>(std::move(data));
    initialise_strides();
}

template <typename T>
NDArray<T>::NDArray(std::shared_ptr<CompactArray<T>> other, DimVec shape, DimVec strides, size_t offset) : _handle{std::move(other)}, _shape{std::move(shape)}, _strides{std::move(strides)}, _offset{offset}
{
}

template <typename T>
NDArray<T>::NDArray(std::shared_ptr<CompactArray<T>> other, DimVec shape, size_t offset) : _handle{std::move(other)}, _shape{std::move(shape)}, _offset{offset}
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
    for (int i = static_cast<int>(_shape.size()) - 1; i >= 0; --i)
    {
        if (_shape[i] > 1)
        {
            if (_strides[i] != expected_stride)
                return false;
            expected_stride *= _shape[i];
        }
    }
    return true;
}

template <typename T>
bool NDArray<T>::has_size_matching_shape() const
{ 
    size_t expected_size = std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<size_t>());
    return (_handle->size() == expected_size);
}

template <typename T>
void NDArray<T>::initialise_strides()
{
    // Need strides for each shape.size()
    _strides.resize(_shape.size());
    // Each prior dim must skip over all elements of later dims.
    // Accumulate product
    size_t dim_stride = 1;
    for (int i = static_cast<int>(_shape.size()) - 1; i >= 0; --i)
    {
        _strides[i] = dim_stride;
        dim_stride *= _shape[i];
    }
}

template <typename T>
bool NDArray<T>::is_contiguous() const {
  return has_row_major_strides();
}


template <typename T>
const DimVec& NDArray<T>::shape() const{
  return _shape;
}

template <typename T>
const DimVec& NDArray<T>::strides() const {
  return _strides;
}

template <typename T>
size_t NDArray<T>::offset() const {
  return _offset;
}

template <typename T>
std::shared_ptr<CompactArray<T>> NDArray<T>::handle(){
  return _handle;
}
template <typename T>
std::shared_ptr<const CompactArray<T>> NDArray<T>::handle() const {
  return _handle;
}



