#pragma once

inline DimVec broadcast_shape(const DimVec &s1, const DimVec &s2)
{
    int dims = static_cast<int>(std::max(s1.size(), s2.size()));
    DimVec out(dims, 1);

    for (size_t k = 0; k < dims; ++k)
    {
        // enforce same dim, or either 1. If diff size, throw except.
        const size_t d1 = (k < s1.size()) ? s1[s1.size() - 1 - k] : 1;
        const size_t d2 = (k < s2.size()) ? s2[s2.size() - 1 - k] : 1;

        if (d1 == d2 || d1 == 1 || d2 == 1)
        {
            out[dims - 1 - k] = std::max(d1, d2);
        }
        else
        {
            throw std::invalid_argument("Cannot broadcast arrays: incompatible shapes");
        }
    }

    return out;
}


inline void validate_matmul_dims(const DimVec& sh1, const DimVec& sh2) {

  if (sh1.size() <= 1 || sh2.size() <= 1) {
    throw std::invalid_argument("Too few dims for matmul");
  }

  if (sh1[sh1.size()-1] != sh2[sh2.size()-2]) {
    throw std::invalid_argument("Invalid reduction dim for matmul provided, need M x K @ K x P");
  }
}

inline DimVec get_matmul_batch_dims(const DimVec& shape) {
  DimVec batch_dims;
  for (int i = 0; i < shape.size() - 2; i++)
  {
    batch_dims.push_back(shape[i]);
  }

  return batch_dims;
}
