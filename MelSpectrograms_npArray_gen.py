import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
from matplotlib import image

class mel_spectrograms:
    
    def mel_spec(self, x):
        samples, sampling_rate = librosa.load(x)
        stft = np.abs(librosa.stft(samples))**2
        M = librosa.feature.melspectrogram(S=stft)
        M = librosa.feature.melspectrogram(y=samples, sr=sampling_rate, n_mels=128, fmax=8000)
        log_M = librosa.power_to_db(M, ref=np.max)
        plt.figure(figsize=(10, 8))
        plt.subplot('211')
        librosa.display.specshow(log_M, y_axis='mel', fmax=8000, x_axis='time')
        plt.tight_layout()
        t='.jpg'
        name=x[:-4]+t
        plt.savefig(name)
        img = plt.imread(name)
        return img

#The output is the numpy array of the Mel Spectrogram image which is given as input to CNN.