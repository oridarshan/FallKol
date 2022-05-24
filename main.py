# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import librosa
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras import layers
import keras
from keras.models import Sequential
from tensorflow import metrics


def checkFallen(file_path: str):
    audio, sample_rate = librosa.load(file_path, mono=True)
    rms = librosa.feature.rms(y=audio)
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate)
    to_append = [np.mean(chroma_stft), np.mean(rms), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff),
                 np.mean(zcr)]
    for e in mfcc:
        to_append.append(np.mean(e))
    # to_append += ' true' if 'not' not in file_path else ' false'

    model = keras.models.load_model('trained_model')
    print('model loaded')
    ans = model.predict(to_append)

    if ans[0] > ans[1]:
        ans2 = 0
        print("Not Fall")
    else:
        print("Fall")
        ans2 = 1


if __name__ == '__main__':
    checkFallen('Data/Audio/fall109.wav')

