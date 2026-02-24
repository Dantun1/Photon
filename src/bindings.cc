#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <backend_cpu.hpp>

namespace py = pybind11;

auto process_slices(const NDArray<float> &self, const py::object &index)
{
    // populate slice_ranges then call slice func with it to return the new view.
    std::vector<NDArray<float>::Slice> slice_ranges;
    const auto &shape = self.get_shape();

    // Tuple of slices, single slice, or single index.
    if (py::isinstance<py::tuple>(index))
    {
        auto tup = index.cast<py::tuple>();
        if (tup.size() > shape.size())
            throw py::index_error("Too many indices for array");
        for (size_t i = 0; i < tup.size(); ++i)
        {
            if (py::isinstance<py::slice>(tup[i]))
            {
                auto s = tup[i].cast<py::slice>();
                size_t start, stop, step, slicelength;
                s.compute(self.get_shape()[i], &start, &stop, &step, &slicelength);
                slice_ranges.push_back({(int64_t)start, (int64_t)stop, (int64_t)step, false});
            }
            else
            {
                int64_t idx = tup[i].cast<int64_t>();
                if (idx < 0)
                    idx += shape[i];
                if (idx < 0 || idx >= (int64_t)shape[i])
                    throw py::index_error("Index out of bounds");
                slice_ranges.push_back({idx, 0, 0, true});
            }
        }
    }
    else if (py::isinstance<py::slice>(index))
    {
        auto s = index.cast<py::slice>();
        size_t start, stop, step, slicelength;
        s.compute(self.get_shape()[0], &start, &stop, &step, &slicelength);
        slice_ranges.push_back({(int64_t)start, (int64_t)stop, (int64_t)step, false});
    }
    else
    {
        int64_t idx = index.cast<int64_t>();
        // if negative, convert to positive.
        if (idx < 0)
            idx += shape[0];
        // if still negative or too big, throw OOB error.
        if (idx < 0 || idx >= (int64_t)shape[0])
            throw py::index_error("Index out of bounds");
        slice_ranges.push_back({idx, 0, 0, true});
    }
    return slice_ranges;
}

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
        for (auto &s : strides_bytes)
        {
            s *= sizeof(float);
        }

        return py::buffer_info(
            m.get_handle()->ptr() + m.get_offset(), // Pointer to the start of data
            sizeof(float),
            py::format_descriptor<float>::format(), // Dtype
            m.get_shape().size(),                   // Ndims
            m.get_shape(),
            strides_bytes); })
        .def("transpose", &NDArray<float>::transpose)
        // operator funcs, scalar and ewise
        .def("__add__", &ewise_add<float>, py::is_operator())
        .def("__add__", &scalar_add<float>, py::is_operator())
        .def("__radd__", &scalar_add<float>, py::is_operator())
        .def("__sub__", &ewise_sub<float>, py::is_operator())
        .def("__sub__", &scalar_sub<float>, py::is_operator())
        .def("__rsub__", &scalar_rsub<float>, py::is_operator())
        .def("__mul__", &ewise_mul<float>, py::is_operator())
        .def("__mul__", &scalar_mul<float>, py::is_operator())
        .def("__rmul__", &scalar_mul<float>, py::is_operator())
        .def("__truediv__", &ewise_div<float>, py::is_operator())
        .def("__truediv__", &scalar_div<float>, py::is_operator())
        .def("__rtruediv__", &scalar_rdiv<float>, py::is_operator())
        .def("__pow__", &ewise_pow<float>, py::is_operator())
        .def("__pow__", &scalar_pow<float>, py::is_operator())
        .def("neg", &NDArray<float>::neg)
        .def("exp", &NDArray<float>::exp)
        .def("log", &NDArray<float>::log)
        .def("sqrt", &NDArray<float>::sqrt)
        .def("sin", &NDArray<float>::sin)
        .def("cos", &NDArray<float>::cos)
        .def("tanh", &NDArray<float>::tanh)
        //reduction ops
        .def("sum", &NDArray<float>::sum)
        .def("min", &NDArray<float>::min)
        .def("max", &NDArray<float>::max)
        .def("reshape", &NDArray<float>::reshape)
        .def("broadcast", &NDArray<float>::broadcast)
        .def("make_compact", &NDArray<float>::make_compact)
        .def("__matmul__", &matmul<float>, py::is_operator())
        .def("__getitem__", [](const NDArray<float> &self, py::object index)
             { 
                auto slice_ranges = process_slices(self, index);
                return self.slice(slice_ranges); })
        .def("__setitem__", [](NDArray<float> &self, py::object index, py::object value)
             {
                 auto slice_ranges = process_slices(self, index);
                 if (py::isinstance<py::float_>(value) || py::isinstance<py::int_>(value))
                 {
                     self.setitem_scalar(slice_ranges, value.cast<float>());
                 }
                 else if (py::isinstance<NDArray<float>>(value))
                 {
                     self.setitem_ewise(slice_ranges, value.cast<NDArray<float>>());
                 }
                 else
                 {
                     throw py::type_error("Value must be a scalar or NDArray");
                 } })
        .def_property_readonly("shape", &NDArray<float>::get_shape)
        .def_property_readonly("strides", &NDArray<float>::get_strides);
}
