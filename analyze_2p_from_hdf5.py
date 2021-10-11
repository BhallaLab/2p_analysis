# CA1 hippocampal 2p Ca recording analysis pipeline. This loads all the
# data into pandas from the hdf5 dumpl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
import sys
import os
import argparse
import seaborn as sns

sns.set()

imagingSessionNames = ['1', '2', '3']
NUM_FRAMES = 240
PEAK_HALF_WIDTH = 3
hitKernel = np.array( [0.25, 0.5, 0.25] )


def main():
    parser = argparse.ArgumentParser( description = "This is a program for analyzing a 2P dataset stored in hdf5 format" )
    parser.add_argument( "-f", "--filename", type = str, help = "Required: Name of hdf5 file. ", default = "store_2p.h5" )
    parser.add_argument( "-st", "--sdev_thresh",  type = float, help = "Optional: Threshold of number of sdevs that the signal must have in order to count as a hit trial.", default = 2.0 )
    parser.add_argument( "-ht", "--hit_trial_thresh",  type = float, help = "Optional: Threshold of percentage of hit trials that each session must have in order to count as significant PSTH response.", default = 30.0 )
    parser.add_argument( "--trace_frames", type = float, nargs = 2, help = "Optional: start_frame end_frame.", default = [96, 99], metavar = ("start_frame", "end frame")  )
    parser.add_argument( "--baseline_frames", type = float, nargs = 2, help = "Optional: start_frame end_frame.", default = [80, 90], metavar = ("start_frame", "end frame")  )
    parser.add_argument( "-c", "--context", type = str, help = "Optional: Data context. Options are hrishi, soumya and synthetic", default = "soumya" )
    args = parser.parse_args()

    t0 = time.time()
    #pd.read_hdf("store_tl.h5", "table", where=["index>2"])
    p2data = pd.read_hdf(args.filename, "2pData")
    behavData = pd.read_hdf(args.filename, "behavData")
    print( "Time to load = ", time.time() - t0 )
    addColumns( p2data )
    #sns.pairplot( dataset.loc['G141','20190913'][['prePk1','prePos1','csPk','csPkPos','postPk1','postPos1']], hue='csPk' )

    #plt.show()
    return p2data, behavData


def findAndIsolateFramePeak( dfbf2, startFrame, endFrame, halfWidth ):
    '''
    # Returns value and position of peak in specified window, and
    # zeroes out dfbf2 in width around the peak, but within window.
    # Indexing: dfbf2[ cell*trial, frame]
    '''
    peakVal = [] 
    peakPos = [] 
    j = 0
    for d, s, e in zip( dfbf2, startFrame, endFrame ):
        window = d[ s:e ]
        pp = np.argmax( window )
        i0 = max( pp - halfWidth, s )
        i1 = min( pp + halfWidth, e )
        peakVal.append( np.max( window ) )
        peakPos.append( pp + s )
        dfbf2[j, i0:i1] = 0.0
    
    return np.array( peakVal ), peakPos


def addColumns( df ):

    y = df["frames"].tolist()
    csFrame = df["csFrame"]
    usFrame = df["usFrame"]
    print( "CSFRAME SHAPE === ", csFrame.shape, usFrame.shape )
    # Here we convert the ragged array of y to a padded array.
    length = max(map(len, y))
    dfbf2 = np.array([yi+[0.0]*(length-len(yi)) for yi in y])
    print( "LEEN = ", len( dfbf2 ), len( dfbf2[0] ), len( dfbf2[2000] ) )
    
    print( "SHAPE = ", dfbf2.shape, len(dfbf2) )
    # I want to build stdev of values below 80 percentile.
    perc = np.percentile( dfbf2, 80, axis = 1 )
    #np.std(np.ma.masked_where( b > np.repeat(pec,5).reshape( 20, 5 ), b ),1)
    ax1, ax2 = dfbf2.shape
    #print( "  DFBF2 = ", dfbf2.shape, sh[0], sh[1], ax1, ax2 )
    maskedDfbf2 = np.ma.masked_where( dfbf2 > np.repeat(perc, ax2 ).reshape( ax1, ax2 ), dfbf2 )
    sd = np.std( maskedDfbf2, 1 )
    mn = np.mean( maskedDfbf2, 1 )
    #sd = np.std(np.ma.masked_where( dfbf2 > np.repeat(perc, ax2 ).reshape( ax1, ax2 ), dfbf2 ),1)
    #print( "SHAPE = ", sd.shape, "  DFBF2 = ", dfbf2.shape )
    df['sdev80'] = sd
    df['mean80'] = mn
    zeros = [0] * len( csFrame )
    df['prePk1'], df['prePos1'] = findAndIsolateFramePeak( dfbf2, zeros, csFrame -1, PEAK_HALF_WIDTH )
    df['prePk2'], df['prePos2'] = findAndIsolateFramePeak( dfbf2, zeros, csFrame -1, PEAK_HALF_WIDTH )
    df['csPk'], df['csPkPos'] = findAndIsolateFramePeak( dfbf2, csFrame, usFrame, PEAK_HALF_WIDTH )
    df['postPk1'], df['postPos1'] = findAndIsolateFramePeak( dfbf2, usFrame, [len( dfbf2[0] )] * len( dfbf2 ),PEAK_HALF_WIDTH )
    df['postPk2'], df['postPos2'] = findAndIsolateFramePeak( dfbf2, usFrame, [len( dfbf2[0] )] * len( dfbf2 ), PEAK_HALF_WIDTH )
    print( df.head() )


if __name__ == '__main__':
    main()
