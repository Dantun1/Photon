template <typename T>
class CompactArray
{
private:
  T* d_ptr;
  size_t size;

public:
  CompactArray(size_t size);
  // Deleted copy ops to prevent double frees.
  CompactArray(const CompactArray&) = delete;
  CompactArray& operator=(const CompactArray&) = delete;
  ~CompactArray();
  // explicit memcopy ops to/from host. PCIE bw is bottlenck in this implementation. 
  void upload(const T* h_ptr, size_t count);
  void download(T* h_ptr, size_t count) const;

  void add_device(T scalar);


  size_t get_size() const;

  T* get_d_ptr();
  const T* get_d_ptr() const;
};


#include "../../src/gpu/ndarray_core.inl" 
