#pragma once
#include <vector>
#include <cstddef>

struct CompactArray
{
    std::vector<float> data;
    CompactArray(const std::vector<float> &input);
    size_t size();
    void print();
};