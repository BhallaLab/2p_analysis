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
    dataset = pd.read_hdf(args.filename, "table")
    print( "Time to load = ", time.time() - t0 )
    #sns.pairplot( dataset.loc['G141','20190913'][['prePk1','prePos1','csPk','csPkPos','postPk1','postPos1']], hue='csPk' )

    #plt.show()
    return dataset


def addColumns( df ):

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
    df['prePk1'], df['prePos1'] = findAndIsolateFramePeak( dfbf2, 0, csFrame -1, PEAKHALFWIDTH )
    df['prePk2'], df['prePos2'] = findAndIsolateFramePeak( dfbf2, 0, csFrame -1, PEAKHALFWIDTH )
    df['csPk'], df['csPkPos'] = findAndIsolateFramePeak( dfbf2, csFrame, usFrame, PEAKHALFWIDTH )
    df['postPk1'], df['postPos1'] = findAndIsolateFramePeak( dfbf2, usFrame, len( dfbf2[0] ),PEAKHALFWIDTH )
    df['postPk2'], df['postPos2'] = findAndIsolateFramePeak( dfbf2, usFrame, len( dfbf2[0] ), PEAKHALFWIDTH )
    print( df.head() )


if __name__ == '__main__':
    main()
