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
START_FRAME = 0
END_FRAME = 232

def reportMemoryUse( name, t1 ):
    t2 = time.time()
    mem = psutil.Process(os.getpid()).memory_info().rss + psutil.Process(os.getpid()).memory_info().vms
    print ("{} took {:.2f} sec and uses {:.2f} GB ".format( name, t2 - t1, mem/(1024**3) ) )
    return t2

def main():
    parser = argparse.ArgumentParser( description = "This is a program for analyzing a 2P dataset stored in hdf5 format" )
    parser.add_argument( "-f", "--filename", type = str, help = "Required: Name of hdf5 file. ", default = "store_2p.h5" )
    parser.add_argument( "-st", "--sdev_thresh",  type = float, help = "Optional: Threshold of number of sdevs that the signal must have in order to count as a hit trial.", default = 2.0 )
    parser.add_argument( "-ht", "--hit_trial_thresh",  type = float, help = "Optional: Threshold of percentage of hit trials that each session must have in order to count as significant PSTH response.", default = 30.0 )
    parser.add_argument( "--trace_frames", type = float, nargs = 2, help = "Optional: start_frame end_frame.", default = [87, 92], metavar = ("start_frame", "end frame")  )
    parser.add_argument( "--baseline_frames", type = float, nargs = 2, help = "Optional: start_frame end_frame.", default = [70, 80], metavar = ("start_frame", "end frame")  )
    parser.add_argument( '-daf', '--display_all_frames', action='store_true', help='Flag: when set, it displays PSTH of all frames.' )
    args = parser.parse_args()

    t1 = time.time()
    p2data, behavData = loadData( args )
    t1 = reportMemoryUse( "Loaded and preprocessed all data", t1 )
    if args.display_all_frames:
        displayAllFrames( p2data )
    

def loadData( args ):
    t1 = time.time()
    #pd.read_hdf("store_tl.h5", "table", where=["index>2"])
    mouseNames = pd.read_hdf(args.filename, "MouseNames")
    #mouseNames = [["G405"]]
    #print( mouseNames )
    mdf = []
    for mn in mouseNames[0]:
        df1 = pd.read_hdf( args.filename, mn )
        idx = df1.index
        cs = fillCS( df1 )
        #print( "CS LEN = {}, CS = {}".format( len( cs ), cs[:20] ) )
        #print( df )
        '''
        print( "BEFORE: ")
        tempdf1 = np.array(df1.iloc[505010:505400, START_FRAME:END_FRAME])
        mt1 = tempdf1.mean( axis = 1 )
        plt.figure( num = "BEFORE")
        plt.imshow( (tempdf1.transpose() / mt1).transpose() )
        '''
        temp = a2p.alignAllFrames( df1.iloc[:, START_FRAME:END_FRAME], cs )
        '''
        print( "AFTER: ")
        temp1 = temp[505010:505400, :] 
        mt1 = temp1.mean( axis = 1 )
        plt.figure( num = "AFTER")
        plt.imshow( (temp1.transpose() / mt1).transpose() )
        plt.show()
        '''
        df = pd.DataFrame( temp, index = idx )
        df["csFrame"] = cs
        df["usFrame"] = cs + 5
        #print( df )
        mdf.append( df )
        t1 = reportMemoryUse( mn, t1 )
    p2data = pd.concat( mdf, keys = mouseNames[0], names = ["mouse", "date", "cell", "trial"] )
    #print( "COOOOOOOOOOLUMN names = ", p2data.columns.values.tolist( ) )
    t1 = reportMemoryUse( "Loading", t1 )

    p2data.sort_index( inplace = True )
    t1 = reportMemoryUse( "Sorting", t1 )

    behavData = pd.read_hdf(args.filename, "behavData")
    t1 = reportMemoryUse( "Reading Behav data", t1 )

    analyzeBehaviour( behavData, p2data, args )
    t1 = reportMemoryUse( "analyzeBehaviour", t1 )

    addColumns( p2data )
    t1 = reportMemoryUse( "adding columns", t1 )

    #sns.pairplot( dataset.loc['G141','20190913'][['prePk1','prePos1','csPk','csPkPos','postPk1','postPos1']], hue='csPk' )
    #displayPeakHisto( p2data, pkName = "csPk", posName = "csPkPos", numSdev = 3.0, hitRatio = 0.3, mouse = "G141", date = "20190913" )

    return p2data, behavData

def estimateCS( gb ):
    #plt.imshow( np.array( gb ) )
    #plt.show()
    return a2p.estimateCS( np.array( gb ) )

def expandCS( csScore, numCells ):
    CS_MIN = 80
    CS_MAX = 120
    CONFIDENCE_THRESH = 5.0
    unit = np.array( [i[0] for i in csScore] )
    confidence = np.array( [i[1] for i in csScore] )
    unit[ confidence < CONFIDENCE_THRESH ] = 92
    unit[ (unit < CS_MIN) | (unit > CS_MAX)] = 97
    return np.tile( unit, numCells )

def fillCS( df1 ):
    cs = []
    for date in df1.index.get_level_values(0).unique():
        df = df1.loc[date].iloc[:,START_FRAME:END_FRAME]
        numTrials = len( df[0][0] )
        #print( "IN ESTIMATE CS for ", date )
        csScore = df.groupby( level = "trial" ).apply( estimateCS )
        cs.extend( expandCS( csScore, len(df)//numTrials ) )
    return np.array( cs )
    #df1["csFrame"] = cs
    #df1["usFrame"] = cs + 5

def analyzeBehaviour( behavData, p2data, args ):
    # Currently a placeholder.
    # Just put in better estimates for the trace period.
    #df = p2data.loc['G394','20210130'].iloc[:,START_FRAME:END_FRAME]
    '''
    df = p2data.loc['G404','20210304'].iloc[:,START_FRAME:END_FRAME]
    print( "DF SHAPE = ", df.shape )
    print( df )
    trialNum = 2
    print( df.loc[ df.index.get_level_values(1) == trialNum] )
    for trialNum in range( 60 ):
        fr, conf = a2p.estimateCS( np.array(df.loc[ df.index.get_level_values(1) == trialNum] ))
        print( "Trial {}: frame = {}, confidence = {:.2f}".format( trialNum, fr, conf ) )
    arr = df.groupby( level = "trial" )
    print( "ARRRRRRRRRRR: " )
    print( arr )
    ret = df.groupby( level = "trial" ).apply( estimateCS )
    print("RET = " )
    print( ret )
    print( ret.shape )
    print("NP SHAPE  = ", np.array( ret).shape )
    '''
    '''
    cs = []
    for mouse in p2data.index.levels[0]:
        for date in p2data.loc[mouse].index.get_level_values(0).unique():
            df = p2data.loc[mouse,date].iloc[:,START_FRAME:END_FRAME]
            numTrials = len( df[0][0] )
            csScore = df.groupby( level = "trial" ).apply( estimateCS )
            cs.extend( expandCS( csScore, len(df)//numTrials ) )
    print( "IN analyzeBehavior, ", len( cs ), len( p2data ) )

    cs = np.array( cs )
    p2data["csFrame"] = cs
    p2data["usFrame"] = cs + 5
    '''
    #p2data["csFrame"] = np.full( len( p2data ), args.trace_frames[0] )
    #p2data["usFrame"] = np.full( len( p2data ), args.trace_frames[1] )
    return

def displayAllFrames( p2data, sortStartFrame = 40, sortEndFrame = -1 ):
    for mouse in p2data.index.levels[0]:
        for date in p2data.loc[mouse].index.get_level_values(0).unique():
            print( "Mouse = ", mouse, ", date = ", date )
            displayPSTH( p2data, sortStartFrame = sortStartFrame, sortEndFrame = sortEndFrame, csFrame = 94, mouse = mouse, date = date )


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
    print ("Finding Frame Pk took {:.2f} sec and uses {:.2f} GB ".format( time.time() - t1, psutil.Process(os.getpid()).memory_info().used/(1024**3) ) )
    
    return np.array( peakVal ), peakPos


def addColumns( df ):
    t1 = time.time()
    csFrame = df["csFrame"]
    usFrame = df["usFrame"]
    #print( "CSFRAME SHAPE === ", csFrame.shape, usFrame.shape )
    dfbf2 = df.iloc[:, START_FRAME:END_FRAME]
    print( "dfbf2 SHAPE === ", dfbf2.shape ) 
    #perc = dfbf2.quantile( 0.80, axis = 1 )
    sd = dfbf2.std( axis = 1 )
    mn = dfbf2.mean( axis = 1 )

    #print( "SHAPE = ", sd.shape, "  DFBF2 = ", dfbf2.shape )
    df['sdev'] = sd
    df['mean'] = mn
    #df['paddedFrames' ] = dfbf2.tolist()
    print( "Computed mean and SD for dataframe")
    t1 = reportMemoryUse( "Mean and SD", t1 )

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

    t1 = reportMemoryUse( "pk and pos", t1 )

def displayPeakHisto( df, pkName = "csPk", posName = "csPkPos", numSdev = 3.0, hitRatio = 0.2, mouse = "G141", date = "20190913" ):
    # Generate histo of positions for a given cell on a given session.
    # Option 1: Add pos for cases where pk/sdev > numSdev
    # Option 2: Linear weight by pos*(pk/sdev)
    # Option 3: Like 2 but only show cells that clear numSdev threshold 
        # for hitRatio fraction of trials.
    # Sort all histos by peak time.
    pk = df.loc[(mouse, date)][pkName]
    pos = df.loc[(mouse, date)][posName]
    sdev = df.loc[(mouse, date)]["sdev"]
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

def displayPSTH( df, sortStartFrame = 40, sortEndFrame = -1, csFrame = -1, mouse = "G141", date = "20190913" ):
    if sortEndFrame == -1:
        sortEndFrame = END_FRAME
    # generate heatMap normalized PSTH (full-frame) for specified session.
    # Sorted by peak within specified range.
    # I want to average all the psths for a given cell
    # I want to average all the sdevs for a given cell
    # idxmax gives the index of the max value of a colum.
    # I want to sort by idxmax. Not look up array by idxmax.
    #plt.imshow( df.loc[ (mouse,date,1)].iloc[:, START_FRAME: END_FRAME] )
    #plt.show()
    
    psth = df.loc[ (mouse, date)].iloc[:, START_FRAME: END_FRAME ].mean( axis = 0, level = 0 )
    #print( psth )
    sdev = df.loc[(mouse, date)]["sdev"].mean( axis = 0, level = 0 )
    ratio = psth.div( sdev, axis = 0 )
    #print( "RATIO SHAPE  = ", ratio.shape )

    # Check for csFrame as the one which has the highest peak 
    # averaged over all cells. Use this option if csFrame == -1.
    if csFrame == -1:
        mn = ratio.mean( axis = 0 )
        if mn.max() / mn.std() > BLANK_USFRAME_THRESH:
            csFrame = mn.idxmax()
            ratio.loc[:,csFrame] = 0    # blank out the biggest response.
        print( "{}/{} Max = {:.4f} at {};   Min = {}".format( mouse, date, mn.max(), mn.idxmax(), mn.min() ) )
    else:
        ratio.loc[:,csFrame] = 0    # blank out the defined CS response.
        #for i, j in enumerate( csFrame ): # Most unpanda
        #    ratio[i, j] = 0

    idxmax = ratio.iloc[:, sortStartFrame:sortEndFrame].idxmax( axis = 1 )
    sortedIdx = idxmax.sort_values().index
    sortedRatio = ratio.reindex( sortedIdx )
    fig = plt.figure( figsize = (12,12 ))
    #plt.imshow( np.log10( sortedRatio ) )
    plt.imshow( sortedRatio )
    plt.show()

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

    bigPkIdx = df[pos][ (df[pk]/df["sdev"]) > sigThresh ]
    hitPkIdx = df[pos][ (df[pk]/df["sdev"]) > hitSigThresh ]
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
    bar = df[pos][ (df[pk]/df["sdev"]) > sigThresh ].groupby( level = ["mouse", "date"] )

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
