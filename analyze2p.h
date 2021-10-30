/************************************************************************
 * This program is part of HILLTAU, a framework for fast compact
 * abstractions of signaling events.
 * Copyright	(C) 2021	Upinder S. Bhalla and NCBS
 * It is made available under the terms of the
 * GNU Public License version 3 or later.
 * See the file COPYING.LIB for the full notice.
************************************************************************/


py::array_t< double > findFramePeak( 
				const py::array_t< double > &dfbf,
				const py::array_t< int >& startFrame,  
				const py::array_t< int >& endFrame,  
				int halfWidth
);

pair< int, double>  estimateCS( const py::array_t< double > &dfbf);

// int alignAllFrames( 
//				py::array_t< double >& dfbf,
//				const py::array_t< int >& startFrame
// );
py::array_t< double > alignAllFrames( 
				const py::array_t< double >& dfbf,
				const py::array_t< int >& startFrame
);
