import os
import pickle
import numpy as np
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn import preprocessing
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Predict gender from voice")
parser.add_argument("--test", help="Path to test data")
parser.add_argument("--model", help="Path to saved model")
args = parser.parse_args()

def get_MFCC(sampling_rate,audio):
    features = mfcc.mfcc(audio,sampling_rate, 0.025, 0.01, 13,appendEnergy = False)
    feat     = np.asarray(())
    for i in range(features.shape[0]):
        temp = features[i,:]
        if np.isnan(np.min(temp)):
            continue
        else:
            if feat.size == 0:
                feat = temp
            else:
                feat = np.vstack((feat, temp))
    features = feat
    features = preprocessing.scale(features)
    return features

gmm_files = [os.path.join(args.model,fname) for fname in os.listdir(args.model) if fname.endswith('.gmm')]
models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
genders   = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]

# files[0] has male clips, files[1] has female clips
files = [[], []]

true = [0,0]
false = [0,0]
total = [0,0]
for j, gen in enumerate(['female', 'male']):
    # r=root, d=directories, f = files
    for r, d, f in os.walk(os.path.join(args.test, gen)):
        for file in f:
            if '.wav' in file:
                files[j].append(os.path.join(r, file))
    for f in files[j]:
        # print(f.split("/")[-1])
        sampling_rate, audio  = read(f)
        features   = get_MFCC(sampling_rate,audio)
        scores     = None
        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            gmm    = models[i]         #checking with each model one by one
            scores = np.array(gmm.score(features))
            log_likelihood[i] = scores.sum()
        winner = np.argmax(log_likelihood)
        if(winner == j):
            true[j]+=1
        else:
            false[j]+=1
        total[j]+=1
        # print("\tdetected as - ", genders[winner],"\n\tscores:female ",log_likelihood[0],",male ", log_likelihood[1],"\n")
accuracy = (sum(true)/sum(total))
print("Accuracy = ", accuracy*100, "%")
