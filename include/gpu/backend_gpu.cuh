#pragma once 

// to prevent shared library duplicating singleton, could be problem in future
#define PHOTON_API __attribute__((visibility("default")))

#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <functional>
#include <numeric>
#include <utility>
#include <cmath>
#include <cstdint>
#include <algorithm>


namespace photon::gpu {
    
    using DimVec = std::vector<size_t>;

    // -- Max dims supported in gpu backend -- 
    constexpr int MAX_DIMS = 10;

    // -- Metadata struct to pass to gpu kernels --
    struct TensorMeta {
        int rank;
        size_t shape[MAX_DIMS];
        size_t strides[MAX_DIMS];
        size_t offset;
    };


    // -- VRAM memory buffer -- 
    template <typename T>
    class CompactArray{
    public:
        CompactArray(size_t size);
        explicit CompactArray(std::vector<T> data);

        // Deleted copy ops to prevent double frees.
        CompactArray(const CompactArray &) = delete;
        CompactArray &operator=(const CompactArray &) = delete;
        ~CompactArray();

        // Explicit memcopy ops to/from host. 
        // This should be deferred/done only when host needs the actual data (lazily).
        void upload(const T *h_ptr, size_t count);
        void download(T *h_ptr, size_t count) const;

        size_t size() const;

        // pointer access so NDArray can execute kernels on data.
        const T *d_ptr() const;
        T *d_ptr();
    private:
        T *_d_ptr;
        size_t _size;
    };

    // -- Singleton for device properties + dynamic kernel dispatch -- 
    // (better than dependency injection alternative)
    struct PHOTON_API DeviceManager {
    public:
        int num_sms;
        int max_shared_memory_per_block;
        int max_threads_per_block;

        DeviceManager(const DeviceManager&) = delete;
        DeviceManager(DeviceManager&&) = delete;
        DeviceManager& operator=(const DeviceManager&) = delete;
        DeviceManager& operator=(DeviceManager&&) = delete;

        static const DeviceManager& get();
    private:
        DeviceManager();
    };

    // -- NDArray view over GPU memory -- 
    template <typename T>
    class NDArray {
    private:
        std::shared_ptr<CompactArray<T>> _handle;
        DimVec _shape;
        DimVec _strides;
        size_t _offset;


        // Internal funcs for contiguity checks
        // Check if strides are row major order for the shape
        bool has_row_major_strides() const;
        // Check if underlying memory elements matches shape elements
        bool has_size_matching_shape() const;

        // Helper function for initialising row major strides, called by constructors
        void initialise_strides();
            
        template <typename Op>
        NDArray<T> unary_dispatch(Op op) const;

        template <typename ReduceOp, typename AtomicOp>
        NDArray<T> reduction_dispatch(const DimVec& axes, ReduceOp op, AtomicOp at_op, bool keepdims) const;
    public:
        // slice struct for python slice manipulation
        struct Slice
        {
            int64_t start;
            int64_t stop;
            int64_t step;
            bool is_index = false;
        };


        explicit NDArray(const DimVec &shape);
        // Create ndarray from existing vector + shape
        NDArray(std::vector<T> data, DimVec shape);
        // 1D array
        NDArray(std::vector<T> data);
        // Construct new view of existing ndarray with new shape and strides, internal use
        NDArray(std::shared_ptr<CompactArray<T>> handle, DimVec shape, DimVec strides, size_t offset = 0);
        NDArray(std::shared_ptr<CompactArray<T>> handle, DimVec shape, size_t offset = 0);

        // View based ops
        NDArray<T> make_compact() const;
        NDArray<T> reshape(const DimVec &new_shape) const;
        NDArray<T> transpose(const DimVec &axes) const;
        NDArray<T> slice(const std::vector<Slice> &slice_ranges) const;
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
        NDArray<T> sum(const DimVec &axes, bool keepdims = false) const;
        NDArray<T> max(const DimVec &axes, bool keepdims = false) const;
        NDArray<T> min(const DimVec &axes, bool keepdims = false) const;

        // Utilities
        const DimVec &shape() const;
        const DimVec &strides() const;
        size_t offset() const;
        std::shared_ptr<CompactArray<T>> handle();
        std::shared_ptr<const CompactArray<T>> handle() const;
        TensorMeta meta() const; 

        bool is_contiguous() const;
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
    NDArray<T> scalar_rpow(const NDArray<T> &a, T b);

    template <typename T>
    NDArray<T> matmul(const NDArray<T> &a, const NDArray<T> &b);



    extern template class CompactArray<float>;
    extern template class NDArray<float>;
        
    extern template NDArray<float> ewise_add(const NDArray<float> &, const NDArray<float> &);
    extern template NDArray<float> ewise_sub(const NDArray<float> &, const NDArray<float> &);
    extern template NDArray<float> ewise_mul(const NDArray<float> &, const NDArray<float> &);
    extern template NDArray<float> ewise_div(const NDArray<float> &, const NDArray<float> &);
    extern template NDArray<float> ewise_pow(const NDArray<float> &, const NDArray<float> &);

    extern template NDArray<float> scalar_add(const NDArray<float> &, float);
    extern template NDArray<float> scalar_sub(const NDArray<float> &, float);
    extern template NDArray<float> scalar_mul(const NDArray<float> &, float);
    extern template NDArray<float> scalar_div(const NDArray<float> &, float);
    extern template NDArray<float> scalar_pow(const NDArray<float> &, float);

    extern template NDArray<float> scalar_rsub(const NDArray<float> &, float);
    extern template NDArray<float> scalar_rdiv(const NDArray<float> &, float);
    extern template NDArray<float> scalar_rpow(const NDArray<float> &, float);
}

