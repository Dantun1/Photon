#include <vector>
#include <numeric>   
#include <functional> 
#include <cmath>    
#include <view_helpers.inl>

/**Matmul
 */

template <typename T>
void matmul_2d_kernel(const T* src_a, const T* src_b, T* out, size_t offset_a,size_t offset_b, size_t offset_tgt, size_t M, size_t K, size_t P){
  // iterate over reduction dim second to 
  // optimise for contiguity of mem access when reading src b
  for (int i = 0; i< M; i++){
    for (int k = 0; k < K; k++){
      T a_val = src_a[offset_a + i * K + k];
      for (int j = 0; j < P; j++){
        // accumulate the dot products from columns as we iterate over reduction dims
        out[offset_tgt + i * P + j] += a_val * src_b[offset_b + k * P + j];
      }
    }
  }

}
// helper contiguous check
inline bool is_2d_contiguous(const DimVec& shape, const DimVec& strides){
  size_t rank = shape.size();
  if (rank < 2) return true;

  return (strides[rank-1] == 1 && strides[rank-2] == shape[rank-1]);
}

template <typename T>
NDArray<T> matmul(const NDArray<T>& a, const NDArray<T>& b){
  const auto ashape = a.get_shape();
  const auto bshape = b.get_shape();

  if (ashape[ashape.size()-1] != bshape[bshape.size()-2]){
    throw std::invalid_argument("Incompatible arrays for MatMul, M x K @ K x P required for non-batch dimensions");
  }
  
  DimVec a_batch_dims;
  DimVec b_batch_dims;
  for (int i = 0; i < ashape.size()-2; i++){
    a_batch_dims.push_back(ashape[i]);
  }

  for (int i = 0; i < bshape.size()-2; i++){
    b_batch_dims.push_back(bshape[i]);
  }

  // construct final dims for matmul
  auto batch_dims_broadcasted = broadcast_shape(a_batch_dims, b_batch_dims);

  // construct broadcasted a and b NDArrays 
  auto broadcasted_a_shape = batch_dims_broadcasted;
  broadcasted_a_shape.push_back(ashape[ashape.size()-2]);
  broadcasted_a_shape.push_back(ashape[ashape.size()-1]);
  NDArray<T> broadcasted_a = a.broadcast(broadcasted_a_shape);

  auto broadcasted_b_shape = batch_dims_broadcasted;
  broadcasted_b_shape.push_back(bshape[bshape.size()-2]);
  broadcasted_b_shape.push_back(bshape[bshape.size()-1]);
  NDArray<T> broadcasted_b = b.broadcast(broadcasted_b_shape);
  
  // Create NDArray we are writing to
  auto out_shape = batch_dims_broadcasted;
  out_shape.push_back(ashape[ashape.size()-2]); // M
  out_shape.push_back(bshape[bshape.size()-1]); // P
  NDArray<T> target(out_shape);
  
  // If last 2 dims of a and b are non contiguous, we need to compact them as kernel assumes contiguity
  // Note: this is currently inefficient as whole array is copied, need to figure way to optimise
  NDArray<T> final_a = is_2d_contiguous(broadcasted_a.get_shape(), broadcasted_a.get_strides()) 
                      ? broadcasted_a : broadcasted_a.make_compact();
  NDArray<T> final_b = is_2d_contiguous(broadcasted_b.get_shape(), broadcasted_b.get_strides()) 
                     ? broadcasted_b : broadcasted_b.make_compact();
  

  // The dims of the matmul i.e., MxK @ KxP = MxP
  const auto M = ashape[ashape.size()-2];
  const auto K = ashape[ashape.size()-1];
  const auto P = bshape[bshape.size()-1];

  // pointers to data to feed into kernel
  const T* src_a = final_a.get_handle()->ptr();
  const T* src_b = final_b.get_handle()->ptr();
  T* out = target.get_handle()->ptr();
  
  // data for batch index odometer logic so we can iteratively run matmul kernel with correct pointers to src/out arrays
  size_t batches = std::accumulate(batch_dims_broadcasted.begin(),batch_dims_broadcasted.end(), 1ULL, std::multiplies<size_t>());
  DimVec batch_indices(batch_dims_broadcasted.size());
  size_t offset_a = final_a.get_offset();
  size_t offset_b = final_b.get_offset();
  const DimVec strides_a = final_a.get_strides();
  const DimVec strides_b = final_b.get_strides();
  

  // We loop and apply the matmul over the batches, kernel assumes last 2 dims are contiguous
  for (int i = 0; i < batches; i++){
    
    // Each batch writes M * P elems to the target contiguously, offset_tgt skips these.
    auto offset_tgt = i * M * P;

    // Apply matmul kernel to the given 2d portions of a, b. Write to C contiguously
    matmul_2d_kernel(src_a, src_b, out, offset_a, offset_b, offset_tgt, M, K, P);
    // Increment the batch dim index tracker, adjust offsets of a and b appropriately to read next batch
    for (int dim = batch_indices.size()-1; dim >= 0; dim--){
      batch_indices[dim]++;
      offset_a += strides_a[dim];
      offset_b += strides_b[dim];
      if (batch_indices[dim] == batch_dims_broadcasted[dim]){
        offset_a -= strides_a[dim] * batch_dims_broadcasted[dim];
        offset_b -= strides_b[dim] * batch_dims_broadcasted[dim];
        batch_indices[dim] = 0;
      }
      else{
        break;
      }
    }
  }
  return target;
}


/** Key reductions
 *
 *
 */

template <typename T, typename Op>
NDArray<T> reduction_op_kernel(const NDArray<T>& a, const DimVec& axes, Op op, T init_val, bool keepdims){
  
  const DimVec& src_shape = a.get_shape();

  if (axes.size() > src_shape.size()) throw std::invalid_argument("Too many axes provided for reduction operation");
  
  DimVec tgt_shape;


  // bool mask for constant dim lookup for tgt shape construction
  std::vector<bool> is_removed(src_shape.size(),0);
  for (int axis: axes){
    if (axis < 0) axis += src_shape.size();
    
    if (axis < 0) throw std::invalid_argument("invalid axis provided, too negative");

    is_removed[axis] = true;
  }
  
  for (int i = 0; i < src_shape.size(); i++){
    if (is_removed[i]){
      if (keepdims) tgt_shape.push_back(1);
    }  
    else {
      tgt_shape.push_back(src_shape[i]);
    }
  }
  
  NDArray<T> target{tgt_shape.empty() ? DimVec{1} : tgt_shape};

  T* tgt_ptr = target.get_handle()->ptr();
  size_t tgt_total_size = target.get_handle()->size();
  
  // initialise value for reduction operation 
  for(size_t i=0; i<tgt_total_size; ++i) {
      tgt_ptr[i] = init_val;
  }

  const T* src_ptr = a.get_handle()->ptr();
  size_t src_total_size = std::accumulate(src_shape.begin(),src_shape.end(), 1ULL, std::multiplies<size_t>());

  size_t src_offset = a.get_offset();
  size_t tgt_offset = 0;
  // Iterate through the source indices, but don't increment offset of 
  // target ptr if dim is in mask. I.e., we set strides of those dims to 0
  DimVec indices(src_shape.size(),0);

  DimVec src_strides = a.get_strides();

  DimVec tgt_compact_strides = target.get_strides();
  DimVec tgt_strides_mapped(src_strides.size());
  // the compact stride index is handled differently if keepdims true/false, need explicit 
  // manipulation.
  size_t tgt_stride_idx = 0;
  
  for (int i = 0; i < tgt_strides_mapped.size(); i++){
    // set removed dim strides to 0
    if (is_removed[i]){
      tgt_strides_mapped[i] = 0;
      // keepdims means that the compact dims has an extra dim for reduced 1 dim, skip over.
      if (keepdims) tgt_stride_idx++;
    }
    else {
      // set to compact dim according to the idx
      tgt_strides_mapped[i] = tgt_compact_strides[tgt_stride_idx++];
    }
  }
  
  for (int i = 0; i < src_total_size; i++){
    tgt_ptr[tgt_offset] = op(tgt_ptr[tgt_offset], src_ptr[src_offset]);

    for (int dim = indices.size() - 1; dim >=0; dim--){
      
      indices[dim]++;
      src_offset += src_strides[dim];
      tgt_offset += tgt_strides_mapped[dim];

      if (indices[dim] == src_shape[dim]){
        indices[dim] = 0;
        src_offset -= src_strides[dim] * src_shape[dim];
        tgt_offset -= tgt_strides_mapped[dim] * src_shape[dim];
      }
      else {
        break;
      }
    }
  }
  return target;
}


template <typename T>
NDArray<T> NDArray<T>::sum(const DimVec& axes, bool keepdims) const{

  return reduction_op_kernel(*this, axes, [](T a, T b){ return a + b;}, (T) 0, keepdims);
}

template <typename T>
NDArray<T> NDArray<T>::max(const DimVec& axes, bool keepdims) const{

  return reduction_op_kernel(*this, axes, [](T a, T b){ return std::max(a,b);}, std::numeric_limits<T>::lowest(), keepdims);
}
template <typename T>
NDArray<T> NDArray<T>::min(const DimVec& axes, bool keepdims) const{

  return reduction_op_kernel(*this, axes, [](T a, T b){ return std::min(a,b);}, std::numeric_limits<T>::max(), keepdims);
}

