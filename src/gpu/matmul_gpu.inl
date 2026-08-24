#include "core/view_helpers.inl"

// tile size as template var
template <typename T, size_t TILE>
__global__ void batched_matmul_kernel(
    const T *a, const T *b,
    T *dst,
    TensorMeta a_meta, TensorMeta b_meta,
    TensorMeta dst_meta,
    size_t M, size_t K, size_t N)
{

  __shared__ T tile_a[TILE][TILE];
  __shared__ T tile_b[TILE][TILE];

  const size_t tx = threadIdx.x, ty = threadIdx.y;
  const size_t bx = blockIdx.x, by = blockIdx.y;

  // row and col into output array
  const size_t row = by * TILE + ty;
  const size_t col = bx * TILE + tx;

  // get offset over batch dims to get to this batch
  size_t tile_offset_a = a_meta.offset;
  size_t tile_offset_b = b_meta.offset;
  size_t tmp = blockIdx.z;
  for (int d = dst_meta.rank - 3; d >= 0; d--)
  {
    size_t idx = tmp % dst_meta.shape[d];
    tile_offset_a += idx * a_meta.strides[d];
    tile_offset_b += idx * b_meta.strides[d];
    tmp /= dst_meta.shape[d];
  }

  // get 2d matmul strides per input matrix
  size_t rank = a_meta.rank;
  const size_t a_rs = a_meta.strides[rank - 2], a_cs = a_meta.strides[rank - 1];
  const size_t b_rs = b_meta.strides[rank - 2], b_cs = b_meta.strides[rank - 1];

  // iterate over tiles and accumulate
  T acc = 0;
  for (size_t t = 0; t < K; t += TILE)
  {
    // index into reduction dim inside the tile for cooperative loading
    size_t a_col = t + tx;
    size_t b_row = t + ty;

    // threads load 1 elem directly iff row,col are in the bounds
    tile_a[ty][tx] = (row < M && a_col < K) ? a[tile_offset_a + row * a_rs + a_col * a_cs] : T(0);
    tile_b[ty][tx] = (col < N && b_row < K) ? b[tile_offset_b + col * b_cs + b_row * b_rs] : T(0);

    __syncthreads();

// unroll since compile time constant loop, less control instructions/address computation (cuobjdump -sass to see diff)
#pragma unroll
    for (int k = 0; k < TILE; ++k)
      acc += tile_a[ty][k] * tile_b[k][tx];

    __syncthreads();
  }

  if (row < M && col < N)
    dst[blockIdx.z * M * N + row * N + col] = acc;
}

template <typename T>
NDArray<T> matmul(const NDArray<T> &a, const NDArray<T> &b)
{
  const DimVec &a_shape = a.shape();
  const DimVec &b_shape = b.shape();

  validate_matmul_dims(a_shape, b_shape);

  DimVec a_batch_dims = get_matmul_batch_dims(a_shape);
  DimVec b_batch_dims = get_matmul_batch_dims(b_shape);

  // get final batch dims
  DimVec batch_dims_broadcasted = broadcast_shape(a_batch_dims, b_batch_dims);
  size_t B = std::accumulate(batch_dims_broadcasted.begin(), batch_dims_broadcasted.end(),
                             1ULL, std::multiplies<size_t>());

  if (B > 65535)
    throw std::invalid_argument("Batch count exceeds max grid z-dimension (65535)");

  // Src array setup
  DimVec a_shape_final = batch_dims_broadcasted;
  a_shape_final.insert(a_shape_final.end(), a_shape.end() - 2, a_shape.end());
  NDArray<T> a_final = a.broadcast(a_shape_final);

  DimVec b_shape_final = batch_dims_broadcasted;
  b_shape_final.insert(b_shape_final.end(), b_shape.end() - 2, b_shape.end());
  NDArray<T> b_final = b.broadcast(b_shape_final);

  // Out array setup
  size_t M = a_shape[a_shape.size() - 2];
  size_t K = a_shape[a_shape.size() - 1];
  size_t N = b_shape[b_shape.size() - 1];

  DimVec output_shape = batch_dims_broadcasted;
  output_shape.push_back(M);
  output_shape.push_back(N);

  NDArray<T> output{output_shape};

  constexpr size_t TILE_SIZE = kernelcfg::MATMUL_TILE_SIZE;

  // x,y into tiles, z for batch dims.
  dim3 grid(
      (N + TILE_SIZE - 1) / TILE_SIZE,
      (M + TILE_SIZE - 1) / TILE_SIZE,
      B);

  dim3 block(TILE_SIZE, TILE_SIZE, 1);

  batched_matmul_kernel<T, TILE_SIZE><<<grid, block>>>(
      a_final.handle()->d_ptr(), b_final.handle()->d_ptr(),
      output.handle()->d_ptr(),
      a_final.meta(), b_final.meta(),
      output.meta(),
      M, K, N);

  return output;
}
