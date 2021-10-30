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

const int ALIGN_IDX = 94;


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

pair< double, double>  meanstd( const double* ptr, int size )
{
	double mean = 0.0;
	for ( int i = 0; i < size; i++ )
		mean += ptr[i];
	mean /= size;
	double stdev = 0.0;
	for ( int i = 0; i < size; i++ )
		stdev += pow( (ptr[i] - mean), 2 );
	return pair< double, double >( mean, sqrt( stdev/size ) );
}

// Take all the frames for all the cells for a given trial. Guess CS frame.
pair< int, double>  estimateCS( const py::array_t< double > &dfbf)
{
	py::buffer_info dbuf = dfbf.request();
	int numCells = dbuf.shape[0];
	int numFrames = dbuf.shape[1];

	const double *pdbuf = (const double *) dbuf.ptr;
	vector< int > frame( numFrames );
	vector< double > frameval;
	vector< double > sumframe( numFrames, 0.0 );
	for ( int i = 0; i < numCells; i++ ) {
		pair< double, double > p = meanstd( pdbuf + i * numFrames, numFrames );
		for ( int j = 0; j < numFrames; j++ ) {
			// double y = pdbuf[ i * numFrames + j ];
			double y = pdbuf[ i + j * numCells ];
			sumframe[j] += (y-p.first)/p.second;
		}
	}

	/**
	for ( int i = 0; i < numCells; i++ ) {
		for ( int j = 0; j < numFrames; j++ ) {
			double y = pdbuf[ i * numFrames + j ];
			sumframe[j] += y;
		}
	}
	**/
	double mean = 0.0;
	int jmax = 0;
	double jval = -1e10;
	for ( int j = 0; j < numFrames; j++ ) {
		double y = sumframe[j];
		mean += y;
		if ( jval < y ) {
			jval = y;
			jmax = j;
		}
	}
	mean /= numFrames;
	double sdev = 0.0;
	for ( int j = 0; j < numFrames; j++ ) {
		double y = sumframe[j] - mean;
		sdev += y * y;
	}
	sdev = sqrt( sdev/numFrames );
	double confidence = (jval - mean) / sdev;
	return pair< int, double> (jmax, confidence );
}



/**
// This does procrustean fitting. Still same # of frames.
// Does it all in place. If it works then that keeps it tight.
// py::array_t< double > alignAllFrames( 
int alignAllFrames( 
				py::array_t< double > &dfbf,
				const py::array_t< int >& startFrame)
{
	py::buffer_info dbuf = dfbf.request();
	py::buffer_info sfbuf = startFrame.request();

	int numTrials = dbuf.shape[0];
	int numFrames = dbuf.shape[1];
	if (numTrials != sfbuf.size) {
		throw std::runtime_error("dfbf size must match # of startFrame");
	}
	// cout << "numTrials = " << numTrials << "; numFrames = " << numFrames << endl;

	double *pdbuf = (double *) dbuf.ptr;
	const int *psfbuf = (const int *) sfbuf.ptr;

	//py::array_t<double> result = py::array_t<double>(dbuf.size );
	// double *presult = (double *) result.request().ptr; 

	for ( int i = 0; i < numTrials; i++ ) {
		int idxoffset = psfbuf[i] - ALIGN_IDX;
		if ( idxoffset == 0 )
			continue;
		vector< double > result( numFrames );
		int startidx = (idxoffset > 0 ? idxoffset: 0 );
		int endidx = idxoffset + numFrames;
		if ( endidx > numFrames )
			endidx = numFrames;
		for ( int j = 0; j < startidx; j++ )
			result[j] = 0.0;
			// presult[ i * numFrames + j] = 0.0;
		for ( int j = startidx; j < endidx; j++ )
			result[j] = pdbuf[ i * numFrames + j + idxoffset];
		for ( int j = endidx; j < numFrames; j++ )
			result[j] = 0.0;
		for ( int j = 0; j < numFrames; j++ )
			pdbuf[ i * numFrames + j] = result[j];
	}
	// result.resize( {numTrials, numFrames} );
	// return result;
	return 0;
}
**/

// This does procrustean fitting. Still same # of frames.
// This one returns the fitted frames
py::array_t< double > alignAllFrames( 
				const py::array_t< double > &dfbf,
				const py::array_t< int >& startFrame)
{
	py::buffer_info dbuf = dfbf.request();
	py::buffer_info sfbuf = startFrame.request();

	int numTrials = dbuf.shape[0];
	int numFrames = dbuf.shape[1];
	if (numTrials != sfbuf.size) {
		throw std::runtime_error("dfbf size must match # of startFrame");
	}

	const double *pdbuf = (const double *) dbuf.ptr;
	const int *psfbuf = (const int *) sfbuf.ptr;

	py::array_t<double> result = py::array_t<double>(dbuf.size );
	double *presult = (double *) result.request().ptr; 

	for ( int i = 0; i < numTrials; i++ ) {
		int idxoffset = psfbuf[i] - ALIGN_IDX;
		int startidx = (idxoffset > 0 ? idxoffset: 0 );
		int endidx = idxoffset + numFrames;
		if ( endidx > numFrames )
			endidx = numFrames;
		for ( int j = 0; j < startidx; j++ )
			presult[ i * numFrames + j] = 0.0;
		for ( int j = startidx; j < endidx; j++ )
			presult[i * numFrames + j] = pdbuf[ i * numFrames + j + idxoffset];
		for ( int j = endidx; j < numFrames; j++ )
			presult[i * numFrames + j] = 0.0;
	}
	result.resize( {numTrials, numFrames} );
	return result;
}
