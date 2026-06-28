// need kernel for scalar ops etc

template <typename T>
__global__
void setitem_scalar_kernel(T scalar, T* dst, size_t size, TensorMeta meta) {
  // sets scalar value to a non contiguous destination tensor. maps 1d tid to coordinate
  // Implement same for contiguous tensor: this one is inefficient so only call when have to 
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= size) return;
  
  size_t temp_idx = tid;
  size_t write_idx = meta.offset;

  for (int dim = meta.rank - 1; dim >= 0; dim--) {
    // look into libdivide / int_fastdiv instead of mod and / 
    size_t dim_offset = temp_idx % meta.shape[dim];
    write_idx += dim_offset * meta.strides[dim];
    temp_idx /= meta.shape[dim];
  }
  
  dst[write_idx] = scalar;
}

template <typename T>
__global__ 
void setitem_scalar_contiguous_kernel(T scalar, T* dst, size_t size, TensorMeta meta) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= size) return;
  // coalesced write
  dst[tid+meta.offset] = scalar;
}

template <typename T>
void NDArray<T>::setitem_scalar(const std::vector<Slice>& slice_ranges, T scalar) {
  // get view over the selected subset
  NDArray<T> target_view = slice(slice_ranges);
  DimVec target_shape = target_view.shape();
  DimVec target_strides = target_view.strides();
  size_t size = std::accumulate(target_shape.begin(), target_shape.end(), 1ULL, std::multiplies<size_t>());
  
  if (size == 0) return;
  
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  TensorMeta tmeta = target_view.meta();
  // if NDArray buffer is contiguous in memory, call coalesced->coalesced for coalesced memory writes i.e. 4B floats = 32 writes in 1 memory access 
  if (target_view.is_contiguous()) {
    setitem_scalar_contiguous_kernel<<<blocks, threads>>>(
        scalar, target_view.handle()->d_ptr(),
        size, tmeta 
        );
  }
  else {
    // pass new view into kernel with scalar for setting
    setitem_scalar_kernel<<<blocks, threads>>>(
        scalar, target_view.handle()->d_ptr(),
        size, tmeta);
  }
  return;
}
