#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vaibhav
"""

import numpy as np
import sidekit,os
from sklearn import mixture
import pickle
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Extracting features from Audio
# =============================================================================
def featureExtractor(subFolder_path):
    e = sidekit.FeaturesExtractor(audio_filename_structure= subFolder_path + '{}',
                                   feature_filename_structure= None,#subFolder_path + '{}.h5',
                                   sampling_frequency=8000,
                                   lower_frequency=0,
                                   higher_frequency=4000,
                                   filter_bank="log",
                                   filter_bank_size=24,
                                   window_size=0.025,
                                   shift=0.01,
                                   ceps_number=20,
                                   vad="snr",
                                   snr=40,
                                   pre_emphasis=0.97,
                                   save_param=["vad", "energy", "cep", "fb"],
                                   keep_all_features=False)   
    return e   
 
# =============================================================================
#  Feature Server Extraction
# =============================================================================

def featureServer(ext):
    fs = sidekit.FeaturesServer(features_extractor=ext,
                                feature_filename_structure= None,
                                sources= None, #['cep'],
                                dataset_list=['cep', 'fb', 'vad', 'energy'],
                                #dataset_list=["cep"],
                                mask="[0-12]",
                                #mask = None,
                                feat_norm="cmvn",
                                global_cmvn=None,
                                dct_pca=False,
                                dct_pca_config=None,
                                sdc=False,
                                sdc_config=None,
                                delta=True,
                                double_delta=True,
                                delta_filter=None,
                                context=None,
                                traps_dct_nb=None,
                                rasta=True,
                                keep_all_features=False)
   
    return fs
 


# =============================================================================
# # to Calculate the Accuracy : 
# =============================================================================

def calculateAccuracy(testDataPath):
    #testDataPath =  '/'.join (path.split('/')[:-2]) + '/TestData/'
    y_predict = []
    folders = os.listdir(testDataPath)
    folders = [i for i in folders if '.DS_Store' not in i ]
    y_true =[]
    print('Starting Feature Extraction Job from testAudio Files. It might take some time...!' )
    for folder in folders :
        
        folder_path = testDataPath + folder + '/'
        subFolders = os.listdir(folder_path)
        subFolders = [f for f in subFolders if f == 'wav'][0]
        subFolder_path = folder_path + subFolders + '/'
        
        files = os.listdir(subFolder_path)
        print('Starting Audio Feature Extraction for {} speaker whic has {} files'.format(folder,len(files)))    
        for file in files:
            testPath = subFolder_path + file
            y_predict.append(findSpeaker(testPath))
            y_true.append(folder)
            
        print('done with {}'.format(folder))
            
    
    lbl = LabelEncoder()
    #x1 = [ item.split('-')[0] for item in y_true]
    lbl.fit(y_true)
    y_true_enc = lbl.transform(y_true)
    
    #x2 = [ item.split('-')[0] for item in y_predict]
    y_predict_enc = lbl.transform(y_predict)
    print ('The Accuracy of the Model is : ', accuracy_score(y_true_enc,y_predict_enc))




#==============================================================================
#           Speaker Identification
#==============================================================================

def findSpeaker(fpath):
    #fpath = '/Users/vk250027/Documents/Audio_Processing/Samples/mwalma/wav/cc-32.wav'
    modelPath =  './Models/'
    #getting all the models
    #path = os.path.dirname(os.path.abspath('__file__')).split('/')
    #path = '/'.join(path[:-1]) + '/New_Speech_Verification/Models/'
    models = os.listdir(modelPath)
    models = [item for item in models if 'gmm_' in item]
   
    #get the Speaker info from the model itself:
    speaker = [str(str(item).split('.')[0]).split('_')[1] for item in models]
   
    #gettign the Actual models
    models = [pickle.load(open(modelPath + item,'rb')) for item in models if 'gmm_' in item]
   
    #print(speaker[7])
    #Getting the file and extracting the features
   
    #'/Users/vk250027/Documents/Audio_Processing/Samples/mwalma/wav/cc-32.wav'
    filePath = '/'.join(fpath.split('/')[:-1]) + '/'
    file = fpath.split('/')[-1]
    #print(file)
   
    e = featureExtractor(filePath)
    fs = featureServer(e)
   
    vector = fs.load(file)[0]  #take only the 1st element of tupple
   
    
    # Checking the Individual model
    chances = np.zeros(len(models))
    for i in range(0,len(models)):
        score = np.array(models[i].score(vector))
        chances[i] = score.sum()
   
    #identifying the person from the Audio
    index = int(np.argmax(chances))
    #print(speaker[index])   
    return speaker[index]



#==============================================================================
#           Speaker Verification
#==============================================================================

# =============================================================================
# def verify(fpath, claimedSpeaker='calamity', THRESHOLD=0.5):
#     #print(fpath)
#     #fpath = '/Users/as186194/Desktop/TestData/calamity/wav/as0012.wav'
#     #path = os.path.dirname(os.path.abspath('__file__')).split('/')
#     #path = '/'.join(path[:-1]) + '/New_Speech_Verification/Models/'
#     modelPath = path + 'Models/'
#     models = os.listdir(modelPath)
#     models = [item for item in models if 'gmm_' in item]
#     model = [item for item in models if claimedSpeaker in item]
#    
#     #get the Speaker info from the model itself:
#     speaker = [str(str(item).split('.')[0]).split('_')[1] for item in model]
#    
#     #gettign the Actual models
#     model = [pickle.load(open(modelPath +item,'rb')) for item in model if 'gmm_' in item]
#    
#     filePath = '/'.join(fpath.split('/')[:-1]) + '/'
#     file = fpath.split('/')[-1]
#     #print(file)
#    
#     e = featureExtractor(filePath)
#     fs = featureServer(e)
#     
#     vector = fs.load(file)[0]
#     
#     score = np.array(model[0].score(vector))
#     chances = score.sum()
#     print(chances)
#     if chances > -51.0:
#         return "accepted"
#     else:
#         return 'Rejected'
# =============================================================================


 
# =============================================================================
"""  Training a GMM model for every Speaker """
# =============================================================================
 
def startTraining():
    features = np.asarray(())
    y =[]
    path = os.path.dirname(os.path.abspath('__file__')) + '/'
    dPath = path + 'Data/'
    #path = '/'.join(path[:-1]) + '/New_Speech_Verification/Data'
     
     
    folders = os.listdir(dPath)
    #folders = [folder for folder in folders if folder == 'Data']
    folders = [folder for folder in folders if '.DS_Store' not in folder]
    for i,folder in enumerate (folders):
        print('\n\nIteration : {} - Started {}'.format(i,folder))
        ts = time.time()
       
        folder_path = dPath  + folder + '/'
        subFolders = os.listdir(folder_path)
        subFolders = [f for f in subFolders if f == 'wav'][0]
        subFolder_path = folder_path + subFolders + '/'
        print(subFolder_path)
        '''
        Getting the Extractor
        '''
        e = featureExtractor(subFolder_path)
       
        #getting Feature Server
       
        fs = featureServer(e)
     
       
        files = os.listdir(subFolder_path)
        #print(subFolder_path)
        files = [item for item in files if '.h5' not in item]
        files = [item for item in files if item not in '.DS_Store']
        files.sort()
        
        for k, file in enumerate (files) :
           
            vector = fs.load(file)[0]
           
            if len(features) == 0:
                features = vector
                y.append(folder)
            else:
                features = np.vstack((features, vector))
                y.append(folder)
           
        print('Training GMM on {} files'.format(len(files)) )
        gmm = mixture.GaussianMixture(n_components = len(folders), max_iter = 200, covariance_type='diag',n_init = 3)
        gmm.fit(features)
        
         
        with open (path + 'Models/gmm_' + str(folder) + '.pkl','wb' ) as f:
            pickle.dump(gmm,f)
            print('Model Saved..!')
               
    print('Done {} which took {:.2f} sec. Size of Feature vector is {} and length of y is {}'.format(folder,(time.time() - ts), features.shape,len(y)))
    features = np.asarray(())

 
    




def main(mode, fileLoc=None, testDataPath = None):
    ''' mode == 0 Training Mode'''
    if mode == 0 :
        startTraining()
        
    elif mode == 1:
        ''' Find Speaker Mode'''
        if fileLoc is not None:
            print(findSpeaker(fileLoc))
        else:
            print('file location is not mentioned') 
    
    elif mode == 2:
        ''' Find accuracy on Test Data '''
        if testDataPath is not None:
           print(calculateAccuracy(testDataPath))
        else:
            print('Test Data path is not mentioned')
        
        


















"""  Evaluation  """
path = '/'.join((os.path.dirname(os.path.abspath('__file__')) + '/').replace('\\','/').split('/')[:-2]) + '/'


''' Mode 0 : Trainign Mode 
    Mode 1 : Identify the Speaker
    Mode 2: Calculate the accuracy on Test Data
'''
main(mode=2,fileLoc = path+'TestData/Speaker-B/wav/eti0407.wav',
     testDataPath= path + 'TestData/')












#testSpeaker(path+'TestData/Speaker-B/wav/eti0199.wav')


# =============================================================================
# #Calculate the accuracy
# =============================================================================

#calculateAccuracy(path+'TestData/')


