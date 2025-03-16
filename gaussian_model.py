import numpy as np
from sklearn.mixture import GaussianMixture
import scipy.io.wavfile as wav
import pickle
import os
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy
import warnings
import audiofeatures as audiofeatures
import preprocessing as preprocessing
warnings.filterwarnings("ignore")

def train_model(x):
    file_paths=open("development_set_enroll.txt" ,'w')
    i=1
    while i<6:
        file_paths.write(x+'_'+str(i)+".wav\n")
        i+=1
    file_paths.close()
    #path to training data
    source   = r"training_audios/"
    #path where training speakers will be saved
    dest = r"models/"
    train_file = "development_set_enroll.txt"        
    file_paths = open(train_file,'r')
    count = 1
    features = np.asarray(())
    for path in file_paths:    
        path = path.strip()
        # read the audio
        preprocessing.pre_process(source+path)
        rate,sig = wav.read(source+path)
        mfcc_feat=audiofeatures.extract_features(sig,rate)
        # extract MFCC 
        
        if features.size == 0:
            features = mfcc_feat
        else:
            features = np.vstack((features, mfcc_feat))
        if count == 5:    
            gmm = GaussianMixture(n_components = 16, max_iter = 200, covariance_type='diag',n_init = 10)
            gmm.fit(features)
            picklefile = path.split("_")[0]+".gmm"
            # name=picklefile.plit("_")[0]
            pickle.dump(gmm,open(dest + picklefile,'wb'))
            features = np.asarray(())
            count = 0
        count = count + 1
    return True

    
# hey= train_model("nikhil")
# print(hey)