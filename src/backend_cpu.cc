#include <vector>
#include <iostream>
#include <backend_cpu.hpp>

CompactArray::CompactArray(const std::vector<float> &input)
    : data(input) {}

size_t CompactArray::size()
{
    return data.size();
}

void CompactArray::print()
{
    for (auto &val : data)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}