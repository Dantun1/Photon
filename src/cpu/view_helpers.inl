#pragma once
#include <vector>
#include <algorithm>
#include <stdexcept>



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
