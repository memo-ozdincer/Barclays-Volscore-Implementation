#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "volscore.hpp"

namespace py = pybind11;

PYBIND11_MODULE(volscore_wrapper, m) {
    py::class_<VolScore>(m, "VolScore")
        .def(py::init<>())
        .def("computeRealizedVol", &VolScore::computeRealizedVol)
        .def("computeVolScore", &VolScore::computeVolScore);
}
