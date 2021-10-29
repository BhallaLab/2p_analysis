# CA1 hippocampal 2p Ca recording analysis pipeline. This loads all the
# data into pandas from the hdf5 dumpl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
import sys
import os
import psutil   # to track memory usage.
import argparse
import seaborn as sns
import a2p

sns.set()

imagingSessionNames = ['1', '2', '3']
NUM_FRAMES = 240
PEAK_HALF_WIDTH = 3
BLANK_USFRAME_THRESH = 5.0  # # of stdevs beyond which we blank usframe
hitKernel = np.array( [0.25, 0.5, 0.25] )
FRAME_START = 0
FRAME_END = 232


def main():
    parser = argparse.ArgumentParser( description = "This is a program for analyzing a 2P dataset stored in hdf5 format" )
    parser.add_argument( "-f", "--filename", type = str, help = "Required: Name of hdf5 file. ", default = "store_2p.h5" )
    parser.add_argument( "-st", "--sdev_thresh",  type = float, help = "Optional: Threshold of number of sdevs that the signal must have in order to count as a hit trial.", default = 2.0 )
    parser.add_argument( "-ht", "--hit_trial_thresh",  type = float, help = "Optional: Threshold of percentage of hit trials that each session must have in order to count as significant PSTH response.", default = 30.0 )
    parser.add_argument( "--trace_frames", type = float, nargs = 2, help = "Optional: start_frame end_frame.", default = [87, 92], metavar = ("start_frame", "end frame")  )
    parser.add_argument( "--baseline_frames", type = float, nargs = 2, help = "Optional: start_frame end_frame.", default = [70, 80], metavar = ("start_frame", "end frame")  )
    args = parser.parse_args()

    t0 = time.time()
    #pd.read_hdf("store_tl.h5", "table", where=["index>2"])
    mouseNames = pd.read_hdf(args.filename, "MouseNames")
    #print( mouseNames )
    mdf = []
    for mn in mouseNames[0]:
        mdf.append( pd.read_hdf( args.filename, mn ) )
        print( "Loading mouse data : ", mn )
    p2data = pd.concat( mdf, keys = mouseNames[0] )
    t1 = time.time()
    print ("Loading took {:.2f} sec and uses {:.2f} GB ".format( t1 - t0, psutil.Process(os.getpid()).memory_info().rss/(1024**3) ) )
    # The pandas_2p.py has put the names for the multi-index in p2data:
    print( "INDEX NAMES = ", p2data.index.names )
    p2data.sort_index( inplace = True )
    print ("Sorting took {:.2f} sec and uses {:.2f} GB ".format( time.time() - t1, psutil.Process(os.getpid()).memory_info().rss/(1024**3) ) )
    t1 = time.time()
    behavData = pd.read_hdf(args.filename, "behavData")
    print ("Reading Behav data took {:.2f} sec and uses {:.2f} GB ".format( time.time() - t1, psutil.Process(os.getpid()).memory_info().rss/(1024**3) ) )
    t1 = time.time()
    #csFrame = np.full( len( p2data ), args.trace_frames[0] )
    #usFrame = np.full( len( p2data ), args.trace_frames[1] )
    #print ("adding CS and US took {:.2f} sec and uses {:.2f} %B ".format( time.time() - t1, psutil.Process(os.getpid()).memory_info().rss/(1024**3) ) )
    addColumns( p2data )
    print ("adding columns took {:.2f} sec and uses {:.2f} GB ".format( time.time() - t1, psutil.Process(os.getpid()).memory_info().rss/(1024**3) ) )
    print( "Finished adding columns" )
    #sns.pairplot( dataset.loc['G141','20190913'][['prePk1','prePos1','csPk','csPkPos','postPk1','postPos1']], hue='csPk' )
    #displayPeakHisto( p2data, pkName = "csPk", posName = "csPkPos", numSdev = 3.0, hitRatio = 0.3, mouse = "G141", date = "20190913" )


    return p2data, behavData

def displayAllFrames( p2data, sortStartFrame = 40, sortEndFrame = -1, usFrame = -1 ):
    for mouse in p2data.index.levels[0]:
        for date in p2data.loc[mouse].index.get_level_values(0).unique():
            print( "Mouse = ", mouse, ", date = ", date )
            displayPSTH( p2data, sortStartFrame = sortStartFrame, sortEndFrame = sortEndFrame, usFrame = usFrame, mouse = mouse, date = date )


def findAndIsolateFramePeak( dfbf2, startFrame, endFrame, halfWidth ):
    '''
    # Returns value and position of peak in specified window, and
    # zeroes out dfbf2 in width around the peak, but within window.
    # Indexing: dfbf2[ cell*trial, frame]
    '''
    peakVal = [] 
    peakPos = [] 
    j = 0
    
    t1 = time.time()
    for i, [s, e] in enumerate( zip( startFrame, endFrame ) ):
        window = dfbf2.iloc[i, s:e]
        peakVal.append( window.max() )
        peakPos.append( window.argmax() )
        pp = window.argmax()
    print ("Finding Frame Pk took {:.2f} sec and uses {:.2f} GB ".format( time.time() - t1, psutil.Process(os.getpid()).memory_info().rss/(1024**3) ) )
    
    return np.array( peakVal ), peakPos


def addColumns( df ):
    t1 = time.time()
    csFrame = df["csFrame"]
    usFrame = df["usFrame"]
    print( "CSFRAME SHAPE === ", csFrame.shape, usFrame.shape )
    dfbf2 = df.iloc[:, FRAME_START:FRAME_END]
    print( "dfbf2 SHAPE === ", dfbf2.shape ) 
    #perc = dfbf2.quantile( 0.80, axis = 1 )
    sd = dfbf2.std( axis = 1 )
    mn = dfbf2.mean( axis = 1 )

    #print( "SHAPE = ", sd.shape, "  DFBF2 = ", dfbf2.shape )
    df['sdev'] = sd
    df['mean'] = mn
    #df['paddedFrames' ] = dfbf2.tolist()
    print( "Computed mean and SD for dataframe")
    print ("mean and SD took {:.2f} sec and uses {:.2f} GB ".format( time.time() - t1, psutil.Process(os.getpid()).memory_info().rss/(1024**3) ) )
    t1 = time.time()

    zeros = [0] * len( df )

    ret = a2p.findFramePeak( dfbf2, zeros, csFrame-1, PEAK_HALF_WIDTH )
    # findFramePeak returns a single vector, first half is pk values and 
    # second half is indices of those pks.
    df['prePk1'] = ret[:len( usFrame )]
    df['prePos'] = ret[len( usFrame):].astype(int)

    ret = a2p.findFramePeak( dfbf2, csFrame, usFrame, PEAK_HALF_WIDTH )
    df['csPk'] = ret[:len( usFrame )]
    df['csPos'] = ret[len( usFrame ):].astype(int)

    ret = a2p.findFramePeak( dfbf2, usFrame, [dfbf2.shape[1]] * len( usFrame ) ,PEAK_HALF_WIDTH )
    df['postPk1'] = ret[:len( usFrame )]
    df['postPos'] = ret[len( usFrame ):].astype(int)

    print ( "csPk = ", df['csPk'] )
    print ( "csPos = ", df['csPos'] )
    print( df.head() )

    print ("pk and pos took {:.2f} sec and uses {:.2f} GB ".format( time.time() - t1, psutil.Process(os.getpid()).memory_info().rss/(1024**3) ) )


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

def displayPSTH( df, sortStartFrame = 40, sortEndFrame = -1, usFrame = -1, mouse = "G141", date = "20190913" ):
    if sortEndFrame == -1:
        sortEndFrame = 1000
    # generate heatMap normalized PSTH (full-frame) for specified session.
    # Sorted by peak within specified range.
    # I want to average all the psths for a given cell
    # I want to average all the sdevs for a given cell
    # idxmax gives the index of the max value of a colum.
    # I want to sort by idxmax. Not look up array by idxmax.
    
    psth = df.loc[ (mouse, date)].iloc[:, START_FRAME: END_FRAME ].mean( axis = 0, level = 0 )
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
def innerPlotHisto( figname, mouse, data, fignum ):
    global hasBar
    #df = histo.loc[mouse]
    #numFrames = df.index.levshape[1]
    #data = np.array( df )
    #data.shape = ( len( data ) // numFrames, numFrames )

    fig = plt.figure( figname )
    ax = plt.subplot( 5, 1, fignum )
    img = ax.imshow( data )
    ax.set_title( mouse )
    ax.set_ylabel( "Session/day" )
    ax.set_xlabel( "Frame" )
    hasBar = True
    return img


def responderStats( df, sigThresh = 5.0, hitSigThresh = 2.5, hitThresh = 0.15, pk = "csPk", pos = "csPkPos" ):
    hitKernel = np.array( [0.67, 1.0, 0.67] )
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
    pkRange = df[pos].max() + 1
    # Shape it by dat[ dates, bins ]
    bins = range( pkRange +1 )
    fig1 = plt.figure( "psthStats", figsize = (6, 9 ) )
    fig2 = plt.figure( "hitStats", figsize = (6, 9 ) )
    fig3 = plt.figure( "hitPlots", figsize = (6, 9 ) )
    ax1 = plt.subplot( 2, 1, 1 )
    ax2 = plt.subplot( 2, 1, 2 )

    ax1.set_title( "Time cell evolution over sessions" )
    ax1.set_xlabel( "Session #" )
    ax1.set_ylabel( "# of time cells" )
    ax2.set_title( "Time cell evolution over frames" )
    ax2.set_xlabel( "Frame #" )
    ax2.set_ylabel( "# of time cells" )

    bigPkIdx = df[pos][ (df[pk]/df["sdev80"]) > sigThresh ]
    hitPkIdx = df[pos][ (df[pk]/df["sdev80"]) > hitSigThresh ]
    #I'm sure there is a clean way to get the num of dates for each mouse.
    # Need to normalize by number of cells.
    for mousenum, mouse in enumerate( df.index.levels[0] ):
        mdf = df.loc[mouse]
        numTrials = mdf.index.levshape[-1]
        dates = mdf.index.unique( level = "date" )
        histo = np.zeros( (len( dates ), pkRange ) )
        hitHisto = np.zeros( (len( dates ), pkRange ) )
        for i, date in enumerate( dates ):
            bp = bigPkIdx.loc[mouse, date]
            hp = hitPkIdx.loc[mouse, date]
            numCells = mdf.loc[date, :, 0].shape[0]
            assert( numCells > 0 )
            increment = 1.0 / numCells
            for b in bp:    # iterating over all cells * all trails. 'b' is the frame index of pk
                histo[i, b] += increment
            for cell in hp.index.unique( level = "cell" ):
                hits = np.histogram( hp.loc[cell], bins )
                # Now apply a convolution across the hits to smooth it out.
                smoothHits = np.convolve( hits[0], hitKernel, mode = "same" )
                hitHisto[i] += (smoothHits > hitThresh*numTrials )
            # innerHitHisto[ cell#, frame# ]
            # Here we can threshold innerHitHisto to find a time cell, and its frame.
            # Would like a raster of verified time cells for each session.

        innerPlotHisto( "psthStats", mouse, histo, mousenum + 1 )
        innerPlotHisto( "hitStats", mouse, hitHisto, mousenum + 1 )
        ax1.plot( range( hitHisto.shape[0] ), np.sum( hitHisto, axis = 1 ), label = mouse )
        ax2.plot( range( hitHisto.shape[1] ), np.sum( hitHisto, axis = 0 ), label = mouse )
    #fig1.colorbar( img )
    ax1.legend()
    ax2.legend()
    plt.show()



    # I can do histograms with count, division = np.histogram( bigPkIdx, bins = [0, 1, 2] )
    # If the integer values are good for binning, I can use value_counts:
    # histo = bigPkIdx.groupby(level=["mouse", "date"] ).value_counts(sort=False, normalize = True)
    # 1. cumulate over each session for all cells. Draw a heatmap
    # of how this evolves over dates for each mouse.
    # Or, sum up the pks over time. Do they change?
    # Obtain a ratio wrt total # of cells. 
    #histo = bigPkIdx.groupby(level=["mouse", "date"] ).value_counts(sort=False, normalize = True)
    # Here we try again:
    # histo = bigPkIdx.groupby(level=["mouse", "date"] ).value_counts(sort=False, normalize = True)
    # dat = np.array( histo.loc["G141"] 

    # dat = np.zeros( bigPkIdx.loc[ mouse ]





    '''
    # Value counts leaves out bins, so I should do np.histogram
    numFrames = max( bigPkIdx ) - min( bigPkIdx )
    histo = np.histogram( bigPkIdx.groupby(level=["mouse", "date"] ), bins = numFrames )
    # Value counts leaves out bins, so I should do np.histogram
    bar = df[pos][ (df[pk]/df["sdev80"]) > sigThresh ].groupby( level = ["mouse", "date"] )

    if pk == "csPk":
        width = 2
    else:
        width = 5

    plt.ion()
    for mouse in df.index.levels[0]:
        innerPlotHisto( mouse, histo, width )
    '''

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

    return 0

def timeCellStats( df, frames ):
    '''
    # of cells with pk in window as a func of date for each animal.
    '''
    return 0

if __name__ == '__main__':
    main()
