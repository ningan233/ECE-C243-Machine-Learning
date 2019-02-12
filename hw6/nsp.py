# -*- coding: utf-8 -*-
"""
ECE C143/C243 Homework-3
Helper functions for Neural Signal Processing

"""
import numpy as np
import matplotlib.pyplot as plt
import pdb

def GeneratePoissonSpikeTrain( T, rate ):
#GENERATEPOISSONSPIKETRAIN Summary of this function goes here
#   T in ms
#   r in spikes/s
#   returns spike_train, a collection of spike times

    spike_train = np.array(0)
    time = 0

    while time <= T:
        time_next_spike = np.random.exponential(1/rate * 1000)
        time = time + time_next_spike
        spike_train = np.append(spike_train,time)

    #discard last spike if happens after T
    if (spike_train[np.size(spike_train)-1] > T) :
        spike_train = spike_train[:-1]
    return spike_train

"""
Input S is spike_times described in Jupyter Notebook.
"""

def PlotSpikeRaster(S):
    gap = 3
    mark = 5
    pad = 30

    numSpikeTrains = np.size(S);
    for s in range(numSpikeTrains):
        offset = pad + gap + s*(gap+mark)
        train = S[s]
        if np.size(train)!=0 :
            train = train[:]
            for t in train.T :
                plt.plot([t,t], [offset, offset+mark], color=[0,0,0])

    plt.xlabel('Time (ms)')
    plt.ylim([0,offset+mark+gap+pad])
    
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[int(np.ceil(window_len/2-1)):int(np.ceil(-(window_len/2)))] 



def bin(X, binWidth,binType):
    [dims,numSamples] = X.shape
    if binType == 'first':
        numBins = np.ceil(float(numSamples)/binWidth).astype(int)
    else:
        numBins = np.floor(float(numSamples)/binWidth).astype(int)
    
    binX = np.zeros((dims, numBins),dtype = list)
    
    for i in range(numBins):
        binStart = i*binWidth
        binStop  = (i+1)*binWidth
        if binType == 'sum' :
            binX[:,i] = np.sum(X[:, binStart : binStop].todense(), 1).T
        elif binType == 'mean' :
            binX[:,i] = np.mean(X[:, binStart : binStop].todense(), 1).T
        elif binType ==  'first':
            binX[:,i] = np.asarray(X[:,binStart].todense().T)
        elif binType ==  'last':
            binX[:,i] = X[:, binStop]        
    return binX
