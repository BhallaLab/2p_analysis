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
BLANK_USFRAME_THRESH = 5.0  # # of stdevs beyond which we blank usframe
hitKernel = np.array( [0.25, 0.5, 0.25] )


def main():
    parser = argparse.ArgumentParser( description = "This is a program for analyzing a 2P dataset stored in hdf5 format" )
    parser.add_argument( "-f", "--filename", type = str, help = "Required: Name of hdf5 file. ", default = "store_2p.h5" )
    parser.add_argument( "-st", "--sdev_thresh",  type = float, help = "Optional: Threshold of number of sdevs that the signal must have in order to count as a hit trial.", default = 2.0 )
    parser.add_argument( "-ht", "--hit_trial_thresh",  type = float, help = "Optional: Threshold of percentage of hit trials that each session must have in order to count as significant PSTH response.", default = 30.0 )
    parser.add_argument( "--trace_frames", type = float, nargs = 2, help = "Optional: start_frame end_frame.", default = [87, 92], metavar = ("start_frame", "end frame")  )
    parser.add_argument( "--baseline_frames", type = float, nargs = 2, help = "Optional: start_frame end_frame.", default = [70, 80], metavar = ("start_frame", "end frame")  )
    parser.add_argument( "-c", "--context", type = str, help = "Optional: Data context. Options are hrishi, soumya and synthetic", default = "soumya" )
    args = parser.parse_args()

    t0 = time.time()
    #pd.read_hdf("store_tl.h5", "table", where=["index>2"])
    p2data = pd.read_hdf(args.filename, "2pData")
    # The pandas_2p.py will put the names in the future, for now set here. 
    p2data.index.names = ["mouse", "date", "cell", "trial"]
    p2data.sort_index( inplace = True )
    behavData = pd.read_hdf(args.filename, "behavData")
    print( "Time to load = ", time.time() - t0 )
    csFrame = np.full( len( p2data ), args.trace_frames[0] )
    usFrame = np.full( len( p2data ), args.trace_frames[1] )
    frames = addColumns( p2data, csFrame, usFrame )
    print( "Finished adding columns" )
    #sns.pairplot( dataset.loc['G141','20190913'][['prePk1','prePos1','csPk','csPkPos','postPk1','postPos1']], hue='csPk' )
    #displayPeakHisto( p2data, pkName = "csPk", posName = "csPkPos", numSdev = 3.0, hitRatio = 0.3, mouse = "G141", date = "20190913" )


    return p2data, behavData, frames

def displayAllFrames( p2data, frames, sortStartFrame = 40, sortEndFrame = -1, usFrame = -1 ):
    for mouse in p2data.index.levels[0]:
        for date in p2data.loc[mouse].index.get_level_values(0).unique():
            print( "Mouse = ", mouse, ", date = ", date )
            displayPSTH( p2data, frames, sortStartFrame = sortStartFrame, sortEndFrame = sortEndFrame, usFrame = usFrame, mouse = mouse, date = date )


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
    
    print( "Found Frame Peak between {} and {}".format( startFrame[0], endFrame[0] ) )
    return np.array( peakVal ), peakPos


def addColumns( df, csFrame, usFrame ):
    y = df["frames"].tolist()
    #csFrame = df["csFrame"]
    #usFrame = df["usFrame"]
    print( "CSFRAME SHAPE === ", csFrame.shape, usFrame.shape )
    # Here we convert the ragged array of y to a padded array.
    length = max(map(len, y))
    dfbf2 = np.array( [yi+[0.0]*(length-len(yi)) for yi in y] )
    print( "Finished Padding dfbf array, shape = ", dfbf2.shape )
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
    #df['paddedFrames' ] = dfbf2.tolist()
    print( "Computed mean and SD for dataframe")

    zeros = [0] * len( df )
    df['prePk1'], df['prePos1'] = findAndIsolateFramePeak( dfbf2, zeros, csFrame -1, PEAK_HALF_WIDTH )
    df['prePk2'], df['prePos2'] = findAndIsolateFramePeak( dfbf2, zeros, csFrame -1, PEAK_HALF_WIDTH )
    df['csPk'], df['csPkPos'] = findAndIsolateFramePeak( dfbf2, csFrame, usFrame, PEAK_HALF_WIDTH )
    df['postPk1'], df['postPos1'] = findAndIsolateFramePeak( dfbf2, usFrame, [len( dfbf2[0] )] * len( dfbf2 ),PEAK_HALF_WIDTH )
    df['postPk2'], df['postPos2'] = findAndIsolateFramePeak( dfbf2, usFrame, [len( dfbf2[0] )] * len( dfbf2 ), PEAK_HALF_WIDTH )
    print( df.head() )
    return pd.DataFrame( dfbf2.tolist(), index = df.index )


def displayPeakHisto( df, pkName = "csPk", posName = "csPkPos", numSdev = 3.0, hitRatio = 0.2, mouse = "G141", date = "20190913" ):
    # Generate histo of positions for a given cell on a given session.
    # Option 1: Add pos for cases where pk/sdev > numSdev
    # Option 2: Linear weight by pos*(pk/sdev)
    # Option 3: Like 2 but only show cells that clear numSdev threshold 
        # for hitRatio fraction of trials.
    # Sort all histos by peak time.
    pk = df.loc[(mouse, date)][pkName]
    pos = df.loc[(mouse, date)][posName]
    sdev = df.loc[(mouse, date)]["sdev80"]
    numTrials = len( pk[0] )
    numCells = len( pk ) // numTrials
    #################
    # Option 1
    sigpos = pos[pk/sdev > numSdev] # Generate cases for option 1.
    # foo.value_counts() gives a histogram table
    # sigpos.groupby(level=0).value_counts() gives 71 histos, one per cell
    # Now how to plot as multiple lines?
    #################
    # Option 2: Why not just do a normalized PSTH?

    print( pk )
    print( pos )
    print( sdev )
    return pk, pos, sdev

def displayPSTH( df, frames, sortStartFrame = 40, sortEndFrame = -1, usFrame = -1, mouse = "G141", date = "20190913" ):
    if sortEndFrame == -1:
        sortEndFrame = 1000
    # generate heatMap normalized PSTH (full-frame) for specified session.
    # Sorted by peak within specified range.
    # I want to average all the psths for a given cell
    # I want to average all the sdevs for a given cell
    # idxmax gives the index of the max value of a colum.
    # I want to sort by idxmax. Not look up array by idxmax.
    
    psth = frames.loc[ (mouse, date) ].mean( axis = 0, level = 0 )
    sdev = df.loc[(mouse, date)]["sdev80"].mean( axis = 0, level = 0 )
    ratio = psth.div( sdev, axis = 0 )

    # Check for usFrame as the one which has the highest peak 
    # averaged over all cells. Use this option if usFrame == -1.
    if usFrame == -1:
        mn = ratio.mean( axis = 0 )
        if mn.max() / mn.std() > BLANK_USFRAME_THRESH:
            usFrame = mn.idxmax()
            ratio.loc[:,usFrame] = 0    # blank out the biggest response.
        print( "{}/{} Max = {:.4f} at {}".format( mouse, date, mn.max(), mn.idxmax() ) )
    else:
        ratio.loc[:,usFrame] = 0    # blank out the defined US response.

    idxmax = ratio.loc[:, sortStartFrame:sortEndFrame].idxmax( axis = 1 )
    sortedIdx = idxmax.sort_values().index
    sortedRatio = ratio.reindex( sortedIdx )
    fig = plt.figure( figsize = (12,4 ))
    plt.imshow( np.log10( sortedRatio ) )
    plt.show()
    '''
    '''

    return psth, sdev


hasBar = False
def innerPlotHisto( mouse, histo ):
    global hasBar
    df = histo.loc[mouse]
    numFrames = df.index.levshape[1]
    data = np.array( df )
    data.shape = ( len( data ) // numFrames, numFrames )

    fig = plt.figure( num = mouse, figsize = ( 2, 10 ) )
    img = plt.imshow( data )
    plt.title( mouse )
    plt.ylabel( "Session day" )
    plt.xlabel( "Frame" )
    if not hasBar:
        fig.colorbar( img )
    plt.show()
    plt.pause( 0.1 )
    hasBar = True


def responderStats( df, frames, sigThresh ):
    '''
    Report 
        - fraction of cells responding (in a given window?),
            - Criteria for response: ampl, hit trial rate, window.
        - Hit trial rate
        - Mean Amplitude of response
            - Overall mean
            - Mean when it is a hit trial.
        - Tau of response
    '''
    # This gives indices of pks above thresh for each trial.
    bigPkIdx = df["csPkPos"][ (df["csPk"]/df["sdev80"]) > sigThresh ]
    # the above works fine till the threshold is so high some bins are zero.

    # I can do histograms with count, division = np.histogram( bigPkIdx, bins = [0, 1, 2] )
    # If the integer values are good for binning, I can use value_counts:
    # histo = bigPkIdx.groupby(level=["mouse", "date"] ).value_counts(sort=False, normalize = True)
    # 1. cumulate over each session for all cells. Draw a heatmap
    # of how this evolves over dates for each mouse.
    # Or, sum up the pks over time. Do they change?
    # Obtain a ratio wrt total # of cells. 
    histo = bigPkIdx.groupby(level=["mouse", "date"] ).value_counts(sort=False, normalize = True)

    plt.ion()
    for mouse in df.index.levels[0]:
        innerPlotHisto( mouse, histo )

    '''
    plt.figure( figsize = ( 4, 12 ) )
    #plt.imshow( histo.loc["G377"] ) # Doesn't work.
    n377 = np.array(histo.loc["G377"])
    numFrames = h377.index.levshape[1]
    n377.shape = ( len( n377 ) // numFrames, numFrames )
    plt.imshow( n377 )
    
    plt.show()
    plt.pause( 0.1 )
    '''

    # 2. Figure out how to count hit trials from this.

    # Trying to fix naming of the multiindex columns. It would help.
    return 0

def timeCellStats( df, frames ):
    '''
    # of cells with pk in window as a func of date for each animal.
    '''
    return 0

if __name__ == '__main__':
    main()
