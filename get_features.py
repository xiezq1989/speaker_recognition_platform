# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:10:05 2019

@author: Administrator
"""

from scipy.io import wavfile
import numpy as np
import sys,os
from python_speech_features import mfcc

def get_feature(path):
    mfcc_feature_all = []
    for category in os.listdir(path):
        fs, signal = wavfile.read(path +category)
        mfcc_feature = mfcc(signal, fs)
        #print(mfcc_feature)
        #print(len(mfcc_feature))
        mfcc_feature_all.append(mfcc_feature)
    #print(mfcc_feature_all)
    return mfcc_feature_all

get_feature(sys.argv[1])
