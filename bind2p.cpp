/*
#include <string>
#include <map>
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
using namespace std;
namespace py = pybind11;
#include "analyze2p.h"
/*
PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::map<string, MolInfo>);
PYBIND11_MAKE_OPAQUE(std::map<string, ReacInfo>);
PYBIND11_MAKE_OPAQUE(std::map<string, EqnInfo>);
*/

PYBIND11_MODULE(a2p, m) {
	m.doc() = "Fast analysis functions for 2p mouse data";
	m.def("findFramePeak", &findFramePeak, py::return_value_policy::copy);
	m.def("estimateCS", &estimateCS, py::return_value_policy::copy);
	m.def("alignAllFrames", &alignAllFrames, py::return_value_policy::copy);
}

