template <typename T>
__global__
void setitem_ewise_kernel(const T* src, T* dst, size_t size, TensorMeta src_meta, TensorMeta dst_meta) {
  // general case kernel, index maps to read + write
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= size) return;


  
  size_t read_idx = src_meta.offset;
  size_t write_idx = dst_meta.offset;
  
  for (int dim = src_meta.rank - 1; dim >= 0; dim--) {
    size_t dim_offset = tid % src_meta.shape[dim];
    read_idx += dim_offset * src_meta.strides[dim];
    write_idx += dim_offset * dst_meta.strides[dim];
    // skip the current dim to access offset into next dim
    tid /= src_meta.shape[dim];
  }

  dst[write_idx] = src[read_idx];
}

template <typename T>
__global__
void setitem_ewise_contiguous_kernel(const T* src, T* dst, size_t size, TensorMeta src_meta, TensorMeta dst_meta) {
  // called if the kernels are both contiguous, we can directly copy based on indices, coalesced reads + writes
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (tid >= size) return;
  
  dst[dst_meta.offset + tid] = src[src_meta.offset + tid];
}


template <typename T>
void NDArray<T>::setitem_ewise(const std::vector<Slice> &slice_ranges, const NDArray<T> &source) {
  auto target_view = slice(slice_ranges);
  DimVec target_shape = target_view.shape();
  DimVec target_strides = target_view.strides();
  size_t size = std::accumulate(target_shape.begin(), target_shape.end(), 1ULL, std::multiplies<size_t>());

  // attempt broadcast, throws if incompatible
  auto broadcasted_source = (target_shape == source.shape()) ? source : source.broadcast(target_shape);
  
  TensorMeta target_meta = target_view.meta();
  TensorMeta src_meta = broadcasted_source.meta();

  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  if (target_view.is_contiguous() && broadcasted_source.is_contiguous()) {
    setitem_ewise_contiguous_kernel<<<blocks, threads>>>(
        broadcasted_source.handle()->d_ptr(), 
        target_view.handle()->d_ptr(), 
        size, 
        src_meta, target_meta
        );
  }
  else {
    setitem_ewise_kernel<<<blocks, threads>>>(
        broadcasted_source.handle()->d_ptr(), 
        target_view.handle()->d_ptr(), 
        size, 
        src_meta, target_meta
        );
  }
}
