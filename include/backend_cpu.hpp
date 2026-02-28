#pragma once
#include <vector>
#include <cstddef>
#include <memory>
#include <iostream>

using DimVec = std::vector<size_t>;

/**
 * @brief A compact array class that manages contiguous block of memory for a single data type. This is the underlying storage for NDArray.
 * Just a std::vector wrapper for now, but can be extended to support GPU memory in the future.
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
    // shared pointer to actual data
    std::shared_ptr<CompactArray<T>> handle;
    // the shape of the array view, attribute of the view
    DimVec shape;
    // the strides, attribute of the view, may not reflect memory layout
    DimVec strides;
    // the offset in the compact array, attribute of the view
    size_t offset;

    // Internal funcs for contiguity checks
    // Check if strides are row major order for the shape
    bool has_row_major_strides() const;
    // Check if underlying memory elements matches shape elements
    bool has_size_matching_shape() const;

public:
    struct Slice
    {
        int64_t start;
        int64_t stop;
        int64_t step;
        bool is_index = false;
    };

    // Zeroed NDArray of shape
    explicit NDArray(const DimVec &shape);
    // Create ndarray from existing vector + shape
    NDArray(std::vector<T> data, DimVec shape);
    // 1D array
    NDArray(std::vector<T> data);
    // Construct new view of existing ndarray with new shape and strides, internal use
    NDArray(std::shared_ptr<CompactArray<T>> handle, DimVec shape, DimVec strides, size_t offset = 0);
    NDArray(std::shared_ptr<CompactArray<T>> handle, DimVec shape, size_t offset = 0);

    // Helper function for initialising row major strides, called by constructors
    void initialise_strides();

    // View related funcs 
    NDArray<T> make_compact() const;
    NDArray<T> reshape(const DimVec &new_shape) const;
    NDArray<T> slice(const std::vector<Slice> &slice_ranges) const;
    NDArray<T> transpose(const DimVec &axes) const;
    NDArray<T> broadcast(const DimVec &new_shape) const;
    void setitem_scalar(const std::vector<Slice> &slice_ranges, T scalar);
    void setitem_ewise(const std::vector<Slice> &slice_ranges, const NDArray<T> &source);

    // Unary ops
    NDArray<T> neg() const;
    NDArray<T> exp() const;
    NDArray<T> log() const;
    NDArray<T> sqrt() const;
    NDArray<T> sin() const;
    NDArray<T> cos() const;
    NDArray<T> tanh() const;
    // Reductions
    NDArray<T> sum(const DimVec& axes, bool keepdims = false) const;
    NDArray<T> max(const DimVec& axes, bool keepdims = false) const;
    NDArray<T> min(const DimVec& axes, bool keepdims = false) const;

    DimVec get_shape() const;
    DimVec get_strides() const;
    size_t get_offset() const;
    std::shared_ptr<CompactArray<T>> get_handle() const;

    bool is_contiguous() const;

    void print() const;
};

// Scalar/Ewise op declarations
template <typename T>
NDArray<T> ewise_add(const NDArray<T> &a, const NDArray<T> &b);


template <typename T>
NDArray<T> scalar_add(const NDArray<T> &a, T b);

template <typename T>
NDArray<T> ewise_mul(const NDArray<T> &a, const NDArray<T> &b);

template <typename T>
NDArray<T> scalar_mul(const NDArray<T> &a, T b);

template <typename T>
NDArray<T> ewise_sub(const NDArray<T> &a, const NDArray<T> &b);

template <typename T>
NDArray<T> scalar_sub(const NDArray<T> &a, T b);

template <typename T>
NDArray<T> scalar_rsub(const NDArray<T> &a, T b);
  
template <typename T>
NDArray<T> ewise_div(const NDArray<T> &a, const NDArray<T> &b);

template <typename T>
NDArray<T> scalar_div(const NDArray<T> &a, T b);

template <typename T>
NDArray<T> scalar_rdiv(const NDArray<T> &a, T b);

template <typename T>
NDArray<T> ewise_pow(const NDArray<T> &a, const NDArray<T> &b);

template <typename T>
NDArray<T> scalar_pow(const NDArray<T> &a, T b);

template <typename T>
NDArray<T> matmul(const NDArray<T>& a, const NDArray<T>& b);

#include <compact_array.inl>
#include <ndarray_core.inl>
#include <ndarray_views.inl>
#include <unary_ops.inl>
#include <reduction_ops.inl>
#include <ewise_ops.inl>
#include <scalar_ops.inl>

// extern template class instantiation, if file imports backend_cpu.hpp, it does not
// implicitly create the template class 
extern template class CompactArray<float>;
extern template class NDArray<float>;

extern template NDArray<float> ewise_add(const NDArray<float>&, const NDArray<float>&);
extern template NDArray<float> ewise_sub(const NDArray<float>&, const NDArray<float>&);
extern template NDArray<float> ewise_mul(const NDArray<float>&, const NDArray<float>&);
extern template NDArray<float> ewise_div(const NDArray<float>&, const NDArray<float>&);
extern template NDArray<float> ewise_pow(const NDArray<float>&, const NDArray<float>&);

extern template NDArray<float> scalar_add(const NDArray<float>&, float);
extern template NDArray<float> scalar_sub(const NDArray<float>&, float);
extern template NDArray<float> scalar_mul(const NDArray<float>&, float);
extern template NDArray<float> scalar_div(const NDArray<float>&, float);
extern template NDArray<float> scalar_pow(const NDArray<float>&, float);

extern template NDArray<float> scalar_rsub(const NDArray<float>&, float);
extern template NDArray<float> scalar_rdiv(const NDArray<float>&, float);
extern template NDArray<float> matmul(const NDArray<float>& a, const NDArray<float>& b);

