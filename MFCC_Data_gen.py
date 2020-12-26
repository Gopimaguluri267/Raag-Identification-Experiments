import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import librosa
import librosa.display
from matplotlib import image


class mfcc_data:
    def mfcc(self, x):
        samples, sampling_rate = librosa.load(x)
        mel = librosa.feature.melspectrogram(samples, sr=sampling_rate, n_mels=128)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=13)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        return delta2_mfcc

    def mfcc_matrix(self, delta2_mfcc, y):
        plt.figure(figsize=(14,8))
        librosa.display.specshow(delta2_mfcc)
        t='.jpg'
        n = (y[:-4])+t
        plt.savefig(n)
        img = plt.imread(n)
        return img
    
#The output is the numpy array of the MFCC image which is given as input to the CNN to predict the raag.