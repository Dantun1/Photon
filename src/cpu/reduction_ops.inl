#include <vector>
#include <numeric>   
#include <functional> 
#include <cmath>    


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

