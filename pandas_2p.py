# CA1 hippocampal 2p Ca recording analysis pipeline. This loads all the
# data into pandas.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys
import os
import argparse

# hrishkeshn/imaging_Sorted_for_Analysis/G405/2021-321/1/tiff_files.

# processed df/f data for a full imaging session:
# hrishkeshn/imaging_Sorted_for_Analysis/ImageAnalysis/G405/20210321/1/F_G405_202010304_plane1.mat

# Cell registraton data.
# hrishkeshn/imaging_Sorted_for_Analysis/ImageAnalysis/G405/20210321/1/regops_G405_20210304.mat


# Behaviour Raw data:
# hrishikeshn/BehaviourRawData/G405/20210122_G405_All3_28/002.tiff
behav_sessions = ["All1", "All3", "SoAn1", "An1", "An2", "An3", "Hr7"]
# 20210122_G405_All3_28 -> 28 session number, typically linked to date.
# There are num_trials tiff files with the raw data
# There is an analysis dir with pickle files for online process data.

# Behaviour proceseed data:
# hrishkeshn/imaging_Sorted_for_Analysis/ImageAnalysis/G405/20210321/1/G405_20210321_behav.mat


def checkSoumyaDataFileName( mouse, date, fname ):
    return (mouse + "_" + date in fname) and ("_wholeTrial_B" in fname) and fname[-4:] == ".mat" # for Soumya

def checkHrishiDataFileName( mouse, date, fname ):
    return fname == mouse + "_" + date + ".mat" # for Hrishi

class Context:
    def __init__( self, name, imagingMice = [], behaviourMice = [], dataDirectory = "", fileNamePrefix = "", checkFname = checkSoumyaDataFileName ):
        self.name = name
        self.imagingMice = imagingMice
        self.behavourMice = behaviourMice
        self.dataDirectory = dataDirectory
        self.fileNamePrefix = fileNamePrefix
        self.checkFname = checkFname

soumyaContext = Context( "soumya", 
    imagingMice = ['G141', 'G142', 'G313', 'G377', 'G71'],
    behaviourMice = ['G141', 'G142', 'G313', 'G377', 'G71'],
    dataDirectory = "/home1/bhalla/soumyab/CalciumDataAnalysisResults/Preprocessed_files/",
    fileNamePrefix = "wholeTrial_B",
    checkFname = checkSoumyaDataFileName )


hrishiContext = Context( "hrishi", 
    imagingMice = ['G394', 'G396', 'G404', 'G405', 'G407', 'G408', 'G409'],
    behaviourMice=['G394', 'G396', 'G404', 'G405', 'G407', 'G408', 'G409'],
    dataDirectory = "/home1/bhalla/hrishikeshn/Imaging_Sorted_for_Analysis/Suite2p_analysis/",
    fileNamePrefix = "2D",
    checkFname = checkHrishiDataFileName )

dataContext = soumyaContext

imagingSessionNames = ['1', '2', '3']
NUM_FRAMES = 240
hitKernel = np.array( [0.25, 0.5, 0.25] )

class Cell:
    def __init__( self, index, dfbf, sdevThresh = 3.0, hitThresh = 30.0 ):
        self.index = index
        self.dfbf = dfbf
        self.sdevThresh = sdevThresh
        self.hitThresh = hitThresh / 100.0 # convert from percent.
        #print( "DFBF = ", dfbf.shape, "  mean= ", np.mean(dfbf), "     hitThresh = ", hitThresh )

    def psth( self, b0 = 80, b1 = 90 ):
        peakSeparation = 2

        hitVec = np.zeros( self.dfbf.shape[1] )
        psth = np.zeros( self.dfbf.shape[1] )

        #return [], psth

        for trial in self.dfbf:
            baseline = np.mean( trial[b0:b1] )
            sdev = np.std( trial[b0:b1] )
            #print( "p", end = "" )
            #sys.stdout.flush()

            if sdev <= 0.0:
                continue
            psth += trial
            if np.any( np.isinf( psth ) ):
                print( "INFFFFF   ", trial )
                quit()
            hitVec += ((trial - baseline)/ sdev > self.sdevThresh )

        # Do a convolution for hitVec
        smooth = np.convolve( hitVec, hitKernel, mode = "same" )
        # Go through and find peak times. They have to be separated by a window.
        pk = []
        while max( smooth ) > self.hitThresh:
            pk1 = np.argmax( smooth )
            lo = max( 0, pk1 - peakSeparation )
            hi = min( len( smooth ), pk1 + peakSeparation )
            smooth[ lo:hi] = 0.0
            pk.append( pk1 )

        #print("PSTH =======",  psth )
        return pk, psth

    def hits( self, b0 = 80, b1 = 90 ):
        window = 3
        ret = np.zeros( self.dfbf.shape[1] - window )
        for trial in self.dfbf:
            sdev = np.std( trial[b0:b1] ) * self.sdevThresh
            t = np.zeros( (window, len(trial) - window) )
            for i in range( window ):
                t[i] = trial[i:-window+i]
            mx = np.max( t, axis = 0 )
            mn = np.min( t, axis = 0 )
            assert( len( mx ) == len( trial ) - window )
            ret += ( (mx -mn) > sdev )
        #print("{}    ".format(ret[96]), end = "")
        return (ret / len( self.dfbf )) > self.hitThresh


class Session:
    def __init__( self, mouseName, date, cells ):
        self.date = date
        self.cells = cells
        self.mouseName = mouseName
        self.idx = 0

    def setIndex( self, idx ):
        self.index = idx

    def analyze( self ):
        self.pkPos = np.zeros( NUM_FRAMES )
        self.totPSTH = np.zeros( NUM_FRAMES )
        self.totHits = np.zeros( NUM_FRAMES )
        self.numSig = 0
        for c in self.cells:
            pk, psth = c.psth()
            self.numSig += (len( pk ) > 0 )
            for p in pk:
                self.pkPos[p] += 1

            self.totPSTH[:len(psth)] += psth
            hits = c.hits()
            self.totHits[:len(hits)] += hits

        return self.numSig, self.pkPos, self.totPSTH, self.totHits

class Mouse:
    def __init__( self, sessions ):
        # Make a list of sessions sorted by date.
        self.sessions = sorted(sessions.items(), key = lambda kv: kv[0])
        self.numBehav = 0

    def analyze( self ):
        self.totPkPos = np.zeros( NUM_FRAMES )
        self.totPSTH = np.zeros( NUM_FRAMES )
        self.totHits = np.zeros( NUM_FRAMES )
        self.numSig = 0
        for s in self.sessions:
            self.numSig += s[1].numSig
            self.totPkPos[:len(s[1].pkPos)] += s[1].pkPos
            self.totPSTH[:len(s[1].totPSTH)] += s[1].totPSTH
            self.totHits[:len(s[1].totHits)] += s[1].totHits

    def analyzeTrends():
        return np.zeros(len( trends ) )


def main():
    global dataContext
    parser = argparse.ArgumentParser( description = "This is a dispatcher program for sweeping through the a 2P dataset and executing an analysis pipeline" )
    parser.add_argument( "-b", "--basepath", type = str, help = "Optional: Base path for data. It is organized as follows:\n basePath/Imaging/mouse_name/date/trial and\n basePath/Behaviour/mouse_name/date/trial ", default = soumyaContext.dataDirectory )
    parser.add_argument( "-st", "--sdev_thresh",  type = float, help = "Optional: Threshold of number of sdevs that the signal must have in order to count as a hit trial.", default = 2.0 )
    parser.add_argument( "-ht", "--hit_trial_thresh",  type = float, help = "Optional: Threshold of percentage of hit trials that each session must have in order to count as significant PSTH response.", default = 30.0 )
    parser.add_argument( "--trace_frames", type = float, nargs = 2, help = "Optional: start_frame end_frame.", default = [96, 99], metavar = ("start_frame", "end frame")  )
    parser.add_argument( "--baseline_frames", type = float, nargs = 2, help = "Optional: start_frame end_frame.", default = [80, 90], metavar = ("start_frame", "end frame")  )
    parser.add_argument( "-c", "--context", type = str, help = "Optional: Data context. Options are hrishi, soumya and synthetic", default = "soumya" )
    args = parser.parse_args()

    if args.context == "soumya":
        dataContext = soumyaContext
    elif args.context == "hrishi":
        dataContext = hrishiContext

    trends = []
    psth_params = [args.sdev_thresh, args.hit_trial_thresh] + args.trace_frames + args.baseline_frames
    numSig = 0
    numTot = 0
    numSessions = 0
    numCells = 0
    numBehaviour = 0
    totalPkPos = np.zeros( NUM_FRAMES )
    totalPSTH = np.zeros( NUM_FRAMES )
    totalHits = np.zeros( NUM_FRAMES )
    mouse = {}

    mouseFrames = []
    sessionFrames = []
    mouseNameList = []
    for mouseName in dataContext.imagingMice:
        print( "\nMouse: ", mouseName )
        frames = []
        dates = []
        for date in os.listdir( dataContext.dataDirectory + mouseName ):
            if len(date) != 8:
                continue
            countSession = 0
            for matfile in os.listdir( dataContext.dataDirectory + mouseName + "/" + date + "/" ):
                if dataContext.checkFname( mouseName, date, matfile ):
                    dat = loadmat( dataContext.dataDirectory + mouseName + "/" + date + "/" + matfile )
                    if not 'dfbf' in dat:
                        print( "BAAAAAD: ",  mouseName + "/" + date + "/" + matfile )
                        continue
                    # current version of numpy doesn't handle posinf
                    #dfbf = np.nan_to_num( dat['dfbf'], posinf = 0.0, neginf = 0.0 )
                    dfbf = dat['dfbf']
                    # Reshape into columns of time and rows of (cell,trial)
                    sh = dfbf.shape
                    # Do integer division
                    idx1 = [ i // sh[1] for i in range( sh[0] * sh[1] ) ]
                    idx2 = [ i % sh[1] for i in range( sh[0] * sh[1] ) ]
                    dfbf2 = dfbf.reshape(sh[0] * sh[1], -1 )
                    #print("SHAPE2 = ", dfbf2.shape )
                    df = pd.DataFrame(dfbf2, index=[idx1, idx2])
                    frames.append( pd.DataFrame(dfbf2, index=[idx1, idx2]) )
                    dates.append( date )
                    #print( df.head() )
                    #print( df.tail() )
                    cells, trials, numframes = dfbf.shape
                    #print( "Found: {}/{}/{} with {} cells, {} trials and {} frames".format( mouseName, date, matfile, cells, trials, numframes ) )
                    print( ".", end = "" )
                    sys.stdout.flush()
                    numCells += sh[0]
                    countSession = 1
                    break
            numSessions += countSession
            behavBaseDir = dataContext.dataDirectory + mouseName + "/" + date + "/behaviour/"
            #print( "KEYS ==========",  dates, "    NUM-frames = ", len( frames ) )
            if not "behaviour" in os.listdir( dataContext.dataDirectory + mouseName + "/" + date ):
                #print( "WARNING: No behaviour in: ", mouseName + "/" + date )
                print( "x", end = "")
            else:
                for behavDir in os.listdir( behavBaseDir ):
                    if behavDir.find( mouseName ) != -1:
                        for matfile in os.listdir( behavBaseDir + behavDir ):
                            spl = matfile.split( "." )
                            if spl[-1] == "mat":
                                if spl[0].find("_fec") != -1:
                                    #print( "Behav: {}/{}.mat".format( behavDir, matfile ) )
                                    print( "b", end = "" )
                                    numBehaviour += 1

        sessionFrames.append( pd.concat( frames, keys = dates ) )
        mouseNameList.append( mouseName )
        print( "\nAnalyze Mouse: ", mouseName )
    print( "KEYS ==========",  mouseNameList, "    NUM-frames = ", len( sessionFrames ), "  ", len( mouseNameList ) )
    fullSet = pd.concat( sessionFrames, keys = mouseNameList )
    #fullSet = pd.concat( sessionFrames )

    print( "\nNUM MICE = ", len(dataContext.imagingMice), "NUM_SESSIONS = ", numSessions, "NUM_BEHAVIOUR", numBehaviour )
    print( "NUM SIG = ", numSig, " num Cells = ", numCells )
    #print( "Pk Pos = ", totalPkPos )
    #print( "PSTH = ", totalPSTH )
    fullSet.to_hdf("store_2p.h5", "table", format = "fixed", append=False)
    #fullSet.to_csv("store_2p.csv", float_format = "%4f" )

    '''
    plt.figure( figsize = ( 7, 16 ))
    ax1 = plt.subplot( 4, 1, 1 )
    ax1.plot( np.arange( len( totalPkPos ) ), totalPkPos, label = "Pk pos" )
    ax1.legend()
    ax2 = plt.subplot( 4, 1, 2 )
    #print( totalPSTH )
    ax2.plot( np.arange( len( totalPSTH ) ), totalPSTH, label = "PSTH" )
    ax2.legend()
    ax3 = plt.subplot( 4, 1, 3 )
    ax3.plot( np.arange( len( totalHits ) ), totalHits, label = "Hits" )
    ax3.legend()
    ax4 = plt.subplot( 4, 1, 4 )
    for mouseName, mouse in mouse.items():
        print( "re-Analyzing mouse:", mouseName )
        mouse.analyze()
        ax4.plot( np.arange( len( mouse.totHits ) ), mouse.totHits, label = mouseName )
    ax4.legend()


    plt.show()
    '''


if __name__ == '__main__':
    main()
