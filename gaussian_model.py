import numpy as np
from sklearn.mixture import GaussianMixture
import scipy.io.wavfile as wav
import pickle
import os
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy
import warnings
import audiofeatures
import preprocessing
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
    print(os.getcwd())
    count = 1
    # Extracting features for each speaker (5 files per speakers)
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
        # when features of 5 files of speaker are concatenated, then do model training
        if count == 5:    
            gmm = GaussianMixture(n_components = 16, max_iter = 200, covariance_type='diag',n_init = 10)
            gmm.fit(features)
            
            # dumping the trained gaussian model
            picklefile = path.split("-")[0]+".gmm"
            pickle.dump(gmm,open(dest + picklefile,'wb'))
            features = np.asarray(())
            count = 0
        count = count + 1
#train_model('akshaya')