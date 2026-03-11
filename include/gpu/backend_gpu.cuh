#include <vector>
#include <memory>

using DimVec = std::vector<size_t>;

template <typename T>
class CompactArray
{
private:
    T* _d_ptr;
    size_t _size;

public:
    CompactArray(size_t size);
    explicit CompactArray(std::vector<T> data);
    // Deleted copy ops to prevent double frees.
    CompactArray(const CompactArray&) = delete;
    CompactArray& operator=(const CompactArray&) = delete;
    ~CompactArray();
    // explicit memcopy ops to/from host. PCIE bw is bottlenck in this implementation. 
    void upload(const T* h_ptr, size_t count);
    void download(T* h_ptr, size_t count) const;

    size_t size() const;
    
    // pointer access so NDArray can execute kernels on data.
    const T* d_ptr() const;
     T* d_ptr();
};


template <typename T>
class NDArray  
{
private:
    std::shared_ptr<CompactArray<T>> _handle;
    DimVec _shape;
    DimVec _strides;
    size_t _offset;

    // max dims supported 
    static constexpr int MAX_DIMS = 10;


    // Internal funcs for contiguity checks
    // Check if strides are row major order for the shape
    bool has_row_major_strides() const;
    // Check if underlying memory elements matches shape elements
    bool has_size_matching_shape() const;

    // Helper function for initialising row major strides, called by constructors
    void initialise_strides();
public:
    // slice struct for python slice manipulation
    struct Slice
    {
        int64_t start;
        int64_t stop;
        int64_t step;
        bool is_index = false;
    };
    //metadata struct to pass to gpu kernels
    struct TensorMeta {
      int rank;
      size_t shape[MAX_DIMS];
      size_t strides[MAX_DIMS];
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
    NDArray<T> reshape(const DimVec& new_shape) const;
    NDArray<T> transpose(const DimVec& axes) const;
    NDArray<T> slice(const std::vector<Slice> &slice_ranges) const;
    NDArray<T> broadcast(const DimVec &new_shape) const;
    // void setitem_scalar(const std::vector<Slice> &slice_ranges, T scalar);
    // void setitem_ewise(const std::vector<Slice> &slice_ranges, const NDArray<T> &source);

    
    // Utilities
    const DimVec& shape() const;
    const DimVec& strides() const;
    size_t offset() const;
    std::shared_ptr<CompactArray<T>> handle();
    std::shared_ptr<const CompactArray<T>> handle() const;     

    bool is_contiguous() const;
};


#include <compact_array_manual.inl>
#include <ndarray_core_gpu.inl> 
#include <ndarray_view_gpu.inl>


extern template class CompactArray<float>;
extern template class NDArray<float>;
