import os
# import _pickle as cPickle
import pickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM
import python_speech_features as mfcc
from sklearn import preprocessing
import warnings
import argparse
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Predict gender from voice")
parser.add_argument("--train", help="Path to training data")
parser.add_argument("--model", help="Path to save trained model")
args = parser.parse_args()

def get_MFCC(sampling_rate,audio):
    features = mfcc.mfcc(audio,sampling_rate, 0.025, 0.01, 13,appendEnergy = False)
    features = preprocessing.scale(features)
    return features

for gen in ['/male', '/female']:
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(args.train+gen):
        for file in f:
            if '.wav' in file:
                files.append(os.path.join(r, file))

    # files    = [os.path.join(args.train,f) for f in os.listdir(args.train) if
    #              f.endswith('.wav')]
    features = np.asarray(());

    for f in files:
        sampling_rate,audio = read(f)
        vector   = get_MFCC(sampling_rate,audio)
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

    gmm = GMM(n_components = 8, max_iter = 200, covariance_type='diag',
            n_init = 3)
    gmm.fit(features)

    picklefile = f.split("/")[-2].split(".wav")[0]+".gmm"
    pickle.dump(gmm,open(os.path.join(args.model ,picklefile),'wb'))
    print('modeling completed for gender:',picklefile)
