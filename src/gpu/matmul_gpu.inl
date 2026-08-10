


// tile size as template var
template <typename T>
__global__
void batched_matmul_kernel(
    const T* a, const T* b,
    T* dst, 
    TensorMeta a_meta, TensorMeta b_meta,
    TensorMeta dst_meta,
    size_t M, size_t K, size_t N
    ) 
{
  // M and N split across grid x and y, z handles batch dims 

}





template <typename T>
NDArray<T> matmul(const NDArray<T> &a, const NDArray<T> &b) {
  const DimVec& a_shape = a.shape();
  const DimVec& b_shape = b.shape();

  validate_matmul_dims(a_shape, b_shape);

  DimVec a_batch_dims = get_matmul_batch_dims(a_shape);
  DimVec b_batch_dims = get_matmul_batch_dims(b_shape);

  // get final batch dims 
  DimVec batch_dims_broadcasted = broadcast_shape(a_batch_dims, b_batch_dims);

  // Src array setup
  DimVec a_shape_final = batch_dims_broadcasted;
  a_shape_final.insert(a_shape_final.end(), a_shape.end() - 2, a_shape.end());
  NDArray<T> a_final = a.broadcast(a_shape_final);

  DimVec b_shape_final = batch_dims_broadcasted;
  b_shape_final.insert(b_shape_final.end(), b_shape.end() - 2, b_shape.end());
  NDArray<T> b_final = b.broadcast(b_shape_final);


  // Out array setup
  size_t M = a_shape[a_shape.size()-2];
  size_t K = a_shape[a_shape.size()-1];
  size_t N = b_shape[b_shape.size()-1];

  DimVec output_shape = batch_dims_broadcasted;
  output_shape.push_back(M);
  output_shape.push_back(N);

  NDArray<T> output{output_shape};
}
