#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "model/museformer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(model, m) {
    py::class_<Museformer>(m, "Museformer")
        .def(py::init<int, int, int, int, int, int>(),
             py::arg("vocab_size"),
             py::arg("max_seq_len"),
             py::arg("d_model"),
             py::arg("num_heads"),
             py::arg("num_layers"),
             py::arg("dim_ff"))
        .def("load",     &Museformer::load)
        .def("save",     &Museformer::save)
        .def("generate", &Museformer::generate);
}