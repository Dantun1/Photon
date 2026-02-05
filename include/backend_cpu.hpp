#pragma once
#include <vector>
#include <cstddef>

template <typename T>
class CompactArray
{
public:
    std::vector<T> data;

    CompactArray() = default;
    explicit CompactArray(size_t size);
    explicit CompactArray(const std::vector<T> &input);
    explicit CompactArray(std::vector<T> &&input);

    size_t size() const;
    void print() const;

    // pointers to data for gpu api.
    T *ptr();
    const T *ptr() const;
};