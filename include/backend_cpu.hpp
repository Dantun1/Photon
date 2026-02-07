#pragma once
#include <vector>
#include <cstddef>

/**
 * @brief A compact array class that manages contiguous block of memory for a single data type. This is the underlying storage for NDArray.
 * Just a std::vector wrapper for now, but can be extended to support GPU memoryn in the future.
 *
 * @tparam T The numeric data type of the array elements.
 */
template <typename T>
class CompactArray
{
public:
    std::vector<T> data;

    CompactArray() = default;
    explicit CompactArray(size_t size);
    explicit CompactArray(const std::vector<T> &input);
    explicit CompactArray(std::vector<T> &&input);

    size_t size() const;
    void print() const;

    // pointers to data for gpu api.
    T *ptr();
    const T *ptr() const;
};

/**
 * @brief A multi-dimensional array class that provides a high-level interface for working with n-dimensional data.
 * * It manages the shape, strides, and offset for efficient indexing and slicing.
 * The actual data is stored in a CompactArray, which can be shared across multiple NDArray instances to enable views and slicing without copying data.
 *
 *
 * @tparam T The numeric data type of the array elements.
 */
template <typename T>
class NDArray
{
    // Manage the data via shared_ptr
    std::shared_ptr<CompactArray<T>> handle;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    size_t offset;

public:
    // Zeroed NDArray of shape
    explicit NDArray(const std::vector<size_t> &shape);
    // Create ndarray from existing vector + shape
    NDArray(std::vector<T> data, const std::vector<size_t> shape);
    // Construct new view of existing ndarray with new shape and strides, internal use
    NDArray(std::shared_ptr<CompactArray<T>> handle, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset = 0);

    void compute_strides();

    size_t size() const;
    std::vector<size_t> shape() const;
    std::vector<size_t> strides() const;
    void print() const;
};