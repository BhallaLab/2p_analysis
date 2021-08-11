# CA1 hippocampal 2p Ca recording analysis pipeline. This is the dispatcher program

import numpy as np
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




# storage.ncbs.res.in/soumyab/BehaviourData_camera/g5_791

# Uncomment for Soumya analysis.
imagingMice = ['G141', 'G142', 'G313', 'G377', 'G71']
behaviourMice = ['G141', 'G142', 'G313', 'G377', 'G71']
dataDirectory = "/home1/bhalla/soumyab/CalciumDataAnalysisResults/Preprocessed_files/"
fileNamePrefix = "wholeTrial_B"
'''
# uncomment for Hrishi analysis.
dataDirectory = "/home1/bhalla/hrishikeshn/Imaging_Sorted_for_Analysis/Suite2p_analysis/"
imagingMice = ['G394', 'G396', 'G404', 'G405', 'G407', 'G408', 'G409']
behaviourMice = ['G394', 'G396', 'G404', 'G405', 'G407', 'G408', 'G409']
fileNamePrefix = "2D"
'''

imagingSessionNames = ['1', '2', '3']
NUM_FRAMES = 240
hitKernel = np.array( [0.25, 0.5, 0.25] )

def checkDataFileName( mouse, date, fname ):
    #return fname == mouse + "_" + date + ".mat" # for Hrishi
    return (mouse + "_" + date in fname) and ("_wholeTrial_B" in fname) and fname[-4:] == ".mat" # for Soumya

class Cell:
    def __init__( self, index, dfbf, sdevThresh = 3.0, hitThresh = 30.0 ):
        self.index = index
        self.dfbf = dfbf
        self.sdevThresh = sdevThresh
        self.hitThresh = hitThresh / 100.0 # convert from percent.
        #print( "DFBF = ", dfbf.shape, "  mean= ", np.mean(dfbf), "     hitThresh = ", hitThresh )

    def psth( self, b0 = 80, b1 = 90 ):
        peakSeparation = 5

        hitVec = np.zeros( self.dfbf.shape[1] )
        psth = np.zeros( self.dfbf.shape[1] )
        for trial in self.dfbf:
            baseline = np.mean( trial[b0:b1] )
            sdev = np.std( trial[b0:b1] )
            psth += trial - baseline # Note this includes bad trials.
            #print( "p", end = "" )
            #sys.stdout.flush()

            if sdev <= 0.0:
                continue
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
        ret = np.zeros( self.dfbf.shape[1] )
        for trial in self.dfbf:
            sdev = np.std( trial[b0:b1] ) * self.sdevThresh
            #print( "SDEV = ", sdev )
            if len(trial) > window:
                for i in range( len( trial ) - window ):
                    if ( np.max( trial[i:i+window] ) - np.min( trial[i:i+window] ) ) > sdev:
                        ret[i] += 1.0

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
        pkPos = np.zeros( NUM_FRAMES )
        totPSTH = np.zeros( NUM_FRAMES )
        totHits = np.zeros( NUM_FRAMES )
        numSig = 0
        for c in self.cells:
            pk, psth = c.psth()
            numSig += (len( pk ) > 0 )
            for p in pk:
                pkPos[p] += 1

            totPSTH[:len(psth)] += psth
            hits = c.hits()
            totHits[:len(hits)] += hits

        return numSig, pkPos, totPSTH, totHits

class Mouse:
    def __init__( self, trends ):
        self.trends = trends

    def analyzeTrends():
        return np.zeros(len( trends ) )


def main():
    parser = argparse.ArgumentParser( description = "This is a dispatcher program for sweeping through the a 2P dataset and executing an analysis pipeline" )
    parser.add_argument( "-b", "--basepath", type = str, help = "Optional: Base path for data. It is organized:\n basePath/Imaging/mouse_name/date/trial and\n basePath/Behaviour/mouse_name/date/trial ", default = dataDirectory )
    parser.add_argument( "-st", "--sdev_thresh",  type = float, help = "Optional: Threshold of number of sdevs that the signal must have in order to count as a hit trial.", default = 2.0 )
    parser.add_argument( "-ht", "--hit_trial_thresh",  type = float, help = "Optional: Threshold of percentage of hit trials that each session must have in order to count as significant PSTH response.", default = 30.0 )
    parser.add_argument( "--trace_frames", type = float, nargs = 2, help = "Optional: start_frame end_frame.", default = [96, 99], metavar = ("start_frame", "end frame")  )
    parser.add_argument( "--baseline_frames", type = float, nargs = 2, help = "Optional: start_frame end_frame.", default = [80, 90], metavar = ("start_frame", "end frame")  )
    args = parser.parse_args()


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
    for mouseName in imagingMice:
        print( "\nMouse: ", mouseName )
        for date in os.listdir( args.basepath + mouseName ):
            if len(date) != 8:
                continue
            countSession = 0
            sessions = {}
            for matfile in os.listdir( args.basepath + mouseName + "/" + date + "/" ):
                if checkDataFileName( mouseName, date, matfile ):
                    cells = []
                    dat = loadmat( args.basepath + mouseName + "/" + date + "/" + matfile )
                    if not 'dfbf' in dat:
                        print( "BAAAAAD: ",  mouseName + "/" + date + "/" + matfile )
                        continue
                    dfbf = np.nan_to_num( dat['dfbf'] )
                    #cells, trials, frames = dfbf.shape
                    #print( "Found: {}/{}/{}.mat with {} cells, {} trials and {} frames".format( mouseName, date, spl[0], cells, trials, frames ) )
                    print( ".", end = "" )
                    sys.stdout.flush()

                    for idx, data in enumerate( dfbf, 0 ):
                        cells.append( Cell( idx, data, sdevThresh = args.sdev_thresh, hitThresh = args.hit_trial_thresh ) )
                    #psth.extend( unit_analysis.psth( dfbf, psth_params))
                    numCells += len( cells )
                    countSession = 1

                    sessions[date] = Session( mouseName, date, cells )
            numSessions += countSession
            #mouse[mouseName] = Mouse( sessions )
            behavBaseDir = args.basepath + mouseName + "/" + date + "/behaviour/"
            if not "behaviour" in os.listdir( args.basepath + mouseName + "/" + date ):
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

        for i, s in enumerate( sorted( sessions ) ):
            sess = sessions[s]
            sess.setIndex( i )
            ns, pkPos, psth, hits = sess.analyze()
            numSig += ns
            totalPkPos += pkPos
            totalPSTH[:len(psth)] += psth
            totalHits[:len(hits)] += hits
            print( "hits = ", hits )
            # What I really want to do is to attach each analysis output to the
            # behavioural stage. For now I just have sequential day of recording.

    print( "\nNUM MICE = ", len(imagingMice), "NUM_SESSIONS = ", numSessions, "NUM_BEHAVIOUR", numBehaviour )
    print( "NUM SIG = ", numSig, " num Cells = ", numCells )
    #print( "Pk Pos = ", totalPkPos )
    #print( "PSTH = ", totalPSTH )

    plt.figure( figsize = ( 7, 12 ))
    ax1 = plt.subplot( 3, 1, 1 )
    ax1.plot( np.arange( len( totalPkPos ) ), totalPkPos, label = "Pk pos" )
    ax1.legend()
    ax2 = plt.subplot( 3, 1, 2 )
    ax2.plot( np.arange( len( totalPSTH ) ), totalPSTH, label = "PSTH" )
    ax2.legend()
    ax3 = plt.subplot( 3, 1, 3 )
    ax3.plot( np.arange( len( totalHits ) ), totalHits, label = "Hits" )
    ax3.legend()


    plt.show()


if __name__ == '__main__':
    main()
