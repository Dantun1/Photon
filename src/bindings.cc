#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <backend_cpu.hpp>

namespace py = pybind11;

PYBIND11_MODULE(backend_cpu, m)
{
    py::class_<CompactArray<float>, std::shared_ptr<CompactArray<float>>>(m, "CompactArray")
        .def(py::init<const std::vector<float> &>())
        .def_readonly("data", &CompactArray<float>::data)
        .def("size", &CompactArray<float>::size)
        .def("print", &CompactArray<float>::print);

    py::class_<NDArray<float>>(m, "NDArray", py::buffer_protocol())
        .def(py::init<std::vector<float>, DimVec>())
        .def_buffer([](NDArray<float> &m) -> py::buffer_info
                    {
        // Strides in B for numpy
        DimVec strides_bytes = m.get_strides();
        for (auto &s : strides_bytes) {
            s *= sizeof(float);
        }

        return py::buffer_info(
            m.get_handle()->ptr() + m.get_offset(),   // Pointer to the start of data
            sizeof(float),                
            py::format_descriptor<float>::format(), // Dtype
            m.get_shape().size(),         // Ndims
            m.get_shape(),                
            strides_bytes                 
        ); })
        .def("transpose", &NDArray<float>::transpose)
        .def("reshape", &NDArray<float>::reshape)
        .def("make_compact", &NDArray<float>::make_compact)
        .def_property_readonly("shape", &NDArray<float>::get_shape)
        .def_property_readonly("strides", &NDArray<float>::get_strides);
}