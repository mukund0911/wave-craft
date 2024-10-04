import pickle
import librosa
import numpy as np

MODEL_PATH = './backend/models/speech_music_classifier/sm_model.pkl'

def sm_inference(PATH):
    y, sr = librosa.load(PATH, mono=False, sr=None)
    y_mono = np.mean(y, axis=0)
    mfcc = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=40)
    mfcc_mean = mfcc.mean(axis=1)

    # Load Speech-Music classifier 
    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
    prediction = clf.predict(mfcc_mean.reshape(1, -1))

    return 'Music' if prediction[0] == 0 else 'Speech'

