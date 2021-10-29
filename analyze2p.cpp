/************************************************************************
 * This program is part of HILLTAU, a framework for fast compact
 * abstractions of signaling events.
 * Copyright	(C) 2021	Upinder S. Bhalla and NCBS
 * It is made available under the terms of the
 * GNU Public License version 3 or later.
 * See the file COPYING.LIB for the full notice.
************************************************************************/

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl_bind.h>
using namespace std;
namespace py = pybind11;
#include "analyze2p.h"
/**
#include <htHeader.h>
PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::map<string, MolInfo>);
PYBIND11_MAKE_OPAQUE(std::map<string, ReacInfo>);
PYBIND11_MAKE_OPAQUE(std::map<string, EqnInfo>);
*/



py::array_t< double > findFramePeak( 
				const py::array_t< double > &dfbf,
				const py::array_t< int >& startFrame,  
				const py::array_t< int >& endFrame,  
				int halfWidth
	)
{
	py::buffer_info dbuf = dfbf.request();
	py::buffer_info sfbuf = startFrame.request();
	py::buffer_info efbuf = endFrame.request();

	if (sfbuf.size != efbuf.size) {
		throw std::runtime_error("startFrame and endFrame size must match");
	}
	int numTrials = dbuf.shape[0];
	int numFrames = dbuf.shape[1];
	if (numTrials != efbuf.size) {
		throw std::runtime_error("dfbf size must match");
	}

	const double *pdbuf = (const double *) dbuf.ptr;
	const int *psfbuf = (const int *) sfbuf.ptr;
	const int *pefbuf = (const int *) efbuf.ptr;

	py::array_t<double> result = py::array_t<double>(2 * sfbuf.size);
	// Hack: Put both return arrays into the single one.
	double *presult = (double *) result.request().ptr; 

	for ( int i = 0; i < numTrials; i++, psfbuf++, pefbuf++ ) {
		int jmax = 0;
		double jval = -1e10;
		for ( int j = *psfbuf; j < *pefbuf; j++ ) {
			if ( jval < pdbuf[ i * numFrames + j ] ) {
				jval = pdbuf[ i * numFrames + j ];
				jmax = j;
			}
		}
		*presult = jval;
		*(presult + numTrials) = jmax;
		presult++;
	}
	// result.resize({numTrials, 2});
	return result;
}
