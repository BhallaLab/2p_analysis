# CA1 hippocampal 2p Ca recording analysis pipeline. This loads all the
# data into pandas.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
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
    t0 = time.time()
    fullSet.to_hdf("store_2p.h5", "table", format = "fixed", append=False)
    print( "Time to save = ", time.time() - t0 )
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
