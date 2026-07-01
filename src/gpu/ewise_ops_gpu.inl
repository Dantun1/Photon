#include "core/view_helpers.inl"

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

// Standard ewise ops 

template <typename T, typename Op>
__global__
void ewise_op_contiguous_kernel(const T* src1, const T* src2, T* dst, size_t size, Op op) {

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= size) return;

  dst[tid] = op(src1[tid], src2[tid]);
}

template <typename T, typename Op>
__global__
void ewise_op_kernel(const T* src1, const T* src2, T* dst, size_t size, TensorMeta src1_meta, TensorMeta src2_meta, Op op) {

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= size) return;
  
  size_t read_src1_idx = src1_meta.offset;
  size_t read_src2_idx = src2_meta.offset;
  size_t temp_idx = tid;

  for (int dim = src1_meta.rank-1; dim >= 0; --dim) {
    size_t dim_offset = temp_idx % src1_meta.shape[dim];
    read_src1_idx += dim_offset * src1_meta.strides[dim];
    read_src2_idx += dim_offset * src2_meta.strides[dim];
    temp_idx /= src1_meta.shape[dim];
  }

  dst[tid] = op(src1[read_src1_idx], src2[read_src2_idx]);
}




template <typename T, typename Op>
NDArray<T> ewise_dispatch(const NDArray<T>& a, const NDArray<T>& b, Op op) {
  
  const auto& shape = broadcast_shape(a.shape(), b.shape());

  NDArray<T> new_array{shape};
  size_t size = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
  
  NDArray<T> broadcasted_a = (shape == a.shape()) ? a : a.broadcast(shape);
  NDArray<T> broadcasted_b = (shape == b.shape()) ? b : b.broadcast(shape);
  
  TensorMeta ameta = broadcasted_a.meta();
  TensorMeta bmeta = broadcasted_b.meta();
  // dispatch to appropriate kernel based on contiguity
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
    
  if (broadcasted_a.is_contiguous() && broadcasted_b.is_contiguous()) {
      ewise_op_contiguous_kernel<<<blocks, threads>>>(
          broadcasted_a.handle()->d_ptr() + ameta.offset,
          broadcasted_b.handle()->d_ptr() + bmeta.offset,
          new_array.handle()->d_ptr(),
          size,
          op
      );
  } else {
      ewise_op_kernel<<<blocks, threads>>>(
          broadcasted_a.handle()->d_ptr(),
          broadcasted_b.handle()->d_ptr(),
          new_array.handle()->d_ptr(), 
          size,
          ameta,
          bmeta,
          op
      );
  }
  return new_array;
}

template <typename T>
NDArray<T> ewise_add(const NDArray<T> &a, const NDArray<T> &b) {
    return ewise_dispatch(a, b, [] __device__ (T s1, T s2) { return s1 + s2; });
}

template <typename T>
NDArray<T> ewise_mul(const NDArray<T> &a, const NDArray<T> &b) {
    return ewise_dispatch(a, b, [] __device__ (T s1, T s2) { return s1 * s2; });
}

template <typename T>
NDArray<T> ewise_sub(const NDArray<T> &a, const NDArray<T> &b) {
    return ewise_dispatch(a, b, [] __device__ (T s1, T s2) { return s1 - s2; });
}

template <typename T>
NDArray<T> ewise_div(const NDArray<T> &a, const NDArray<T> &b) {
    return ewise_dispatch(a, b, [] __device__ (T s1, T s2) { return s1 / s2; });
}

template <typename T>
NDArray<T> ewise_pow(const NDArray<T> &a, const NDArray<T> &b) {
    return ewise_dispatch(a, b, [] __device__ (T s1, T s2) { return ::pow(s1, s2); });
}
