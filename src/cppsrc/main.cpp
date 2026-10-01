#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "model/museformer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(museformer, m) {
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
        .def("forward",  &Museformer::forward)
        .def("generate", &Museformer::generate,
             py::arg("input_tokens"),
             py::arg("max_len"),
             py::arg("top_k") = 8,
             py::arg("temperature") = 1.0f,
             py::arg("allowed_tokens") = std::vector<int>{},
             py::arg("seed") = py::none(),
             py::arg("eos_token") = 1);
}
