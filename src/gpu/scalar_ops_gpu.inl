// need kernel for scalar ops etc

template <typename T>
__global__
void setitem_scalar_kernel(T scalar, T* dst, size_t size, TensorMeta meta) {
  // sets scalar value to a non contiguous destination tensor
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= size) return;
  
  size_t temp_idx = tid;
  size_t write_idx = meta.offset;

  for (int dim = meta.rank - 1; dim >= 0; dim--) {
    size_t dim_offset = temp_idx % meta.shape[dim];
    write_idx += dim_offset * meta.strides[dim];
    temp_idx /= meta.shape[dim];
  }
  
  dst[write_idx] = scalar;
}


template <typename T>
void NDArray<T>::setitem_scalar(const std::vector<Slice>& slice_ranges, T scalar) {
  // get view over the selected subset
  NDArray<T> target_view = this->slice(slice_ranges);
  DimVec target_shape = target_view.shape();
  DimVec target_strides = target_view.strides();
  size_t size = std::accumulate(target_shape.begin(), target_shape.end(), 1ULL, std::multiplies<size_t>());
  
  if (size == 0) return;

  TensorMeta meta;
  meta.rank = target_shape.size();
  meta.offset = target_view.offset();
  for (size_t i = 0; i < meta.rank; ++i) {
    meta.shape[i] = target_shape[i];
    meta.strides[i] = target_strides[i];
  }

  // pass new view into kernel with scalar for setting
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  setitem_scalar_kernel<<<blocks, threads>>>(
      scalar, target_view.handle()->d_ptr(),
      size, meta);
  return;
}
