import pandas as  pd
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sktree import ObliqueRandomForestClassifier, PatchObliqueRandomForestClassifier
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import pickle
import scipy.stats as ss
from scipy import ndimage



df = pd.read_csv('/cis/home/jdey4/spmmouse_segment/uwmadison.csv')

path = '/cis/home/jdey4/spmmouse_segment/'
subjects = os.listdir(path)

current_depth = 275
current_width = 347
current_height = 245

desired_depth = 113
desired_width = 137
desired_height = 113

depth = current_depth / desired_depth
width = current_width / desired_width
height = current_height / desired_height

depth_factor = 1 / depth
width_factor = 1 / width
height_factor = 1 / height

X = []
y = []
file_no = 0
count = 0
count_ = 0
IDs = set(df['participant_id'])


for subject in tqdm(subjects):
    if subject in IDs:
        
        count += 1
        #print(count, count_)
        gender = list(df[df['participant_id']==subject]['sex'])
        sex = int(gender[0]=='F')
        
        
        current_file = os.path.join(path, subject)
        file_count = 0
        files = glob.glob(current_file+'/*')

        for file in files:
            if 'mwc1' in file:
                try:
                    img = nb.load(file).get_fdata()
                    img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=1)
                    file_count +=1 
                except:
                    break
                
        for file in files:
            if 'mwc2' in file:
                try:
                    img_ = nb.load(file).get_fdata()
                    img_ = ndimage.zoom(img_, (depth_factor, width_factor, height_factor), order=1)
                    img = np.concatenate((img, img_),axis=0)
                    file_count +=1 
                except:
                    break
                

        '''if len(tmp)<2:
            print(subject, ' has less files')'''
           
        if file_count==2:
            X.append(img.reshape(1,-1))
            y.append(sex)

    
X = np.concatenate(X,axis=0)
X = np.nan_to_num(X)
y = np.array(y)


print('data shape', X.shape)
print('mean label', np.mean(y))

####################################################
total_models = 200

predicted_proba_ = []
for ii in tqdm(range(total_models)):
    with open('morf_models/model'+str(ii)+'.pickle','rb') as f:
        morf = pickle.load(f)
        predicted_proba_.append(
            morf.predict_proba(X)
        )
        del morf

predicted_proba = np.mean(predicted_proba_,axis=0)
predicted_label = np.argmax(predicted_proba,axis=1)
print('MORF accuracy ', np.mean(predicted_label==y))

