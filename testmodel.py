import numpy as np
import scipy
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from sklearn.mixture import GaussianMixture
import pickle
import os
import audiofeatures
import warnings
warnings.filterwarnings("ignore")

def test(): 
    modelpath = "models"       
    
    gmm_files = [os.path.join(modelpath,fname) for fname in 
                  os.listdir(modelpath) if fname.endswith('.gmm')]
    
    #Load the Gaussian gender Models
    models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers   = [fname.split("\\")[-1].split(".gmm")[0] for fname 
                  in gmm_files]
    # Read the test directory and get the list of test audio files 
      
        
         
    path="training_audios/akshaya_5.wav"
    rate,sig = wav.read(path)
    mfcc_feat=audiofeatures.extract_features(sig,rate)
        
    log_likelihood = np.zeros(len(models)) 
        
    for i in range(len(models)):
        gmm    = models[i]  #checking with each model one by one
        scores = np.array(gmm.score(mfcc_feat))
        log_likelihood[i] = scores.sum()
        print(speakers[i]+str(log_likelihood[i]))
    winner = np.argmax(log_likelihood)
    #split the string to get the name of the speaker using regular expressions for delimiters
    file_name = os.path.basename(speakers[winner]).split('.')[0]
    # Split by underscore and take the first part
    name = file_name.split('_')[0]
    print ("\tdetected as - ", name)
    return speakers[winner]
test()