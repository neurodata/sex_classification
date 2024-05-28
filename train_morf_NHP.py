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
                    img = ndimage.zoom(img, (depth_factor, width_factor, height_factor), order=1).reshape(-1)
                    file_count +=1 
                except:
                    break
                
        for file in files:
            if 'mwc2' in file:
                try:
                    img_ = nb.load(file).get_fdata()
                    img_ = ndimage.zoom(img_, (depth_factor, width_factor, height_factor), order=1).reshape(-1)
                    img = np.concatenate((img, img_))
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

total_models = 1000
idx = list(range(len(y)))

np.random.seed(0)
np.random.shuffle(idx)
train_samples = int(len(y)*0.8)
test_samples = len(y) - train_samples
train_ids = idx[:train_samples]

feature_imp = []
for ii in tqdm(range(total_models)):
    np.random.seed(ii)
    np.random.shuffle(train_ids)
    idx_chosen = train_ids[:int(len(y)*0.8*0.7)]
    
    morf = PatchObliqueRandomForestClassifier(n_estimators=1, max_patch_dims=np.array((4, 4, 4)), data_dims=np.array((113, 137, 113)))
    morf.fit(X[idx_chosen,:113*137*113], y[idx_chosen])

    with open('morf_models/NHP_model'+str(ii)+'_gray.pickle','wb') as f:
        pickle.dump(morf, f)

    feature_imp.append(
            morf.feature_importances_
        )
    del morf

with open('feature_imp_gray_NHP.pickle','wb') as f:
    pickle.dump(feature_imp, f)



feature_imp = []
for ii in tqdm(range(total_models)):
    np.random.seed(ii)
    np.random.shuffle(train_ids)
    idx_chosen = train_ids[:int(len(y)*0.8*0.7)]
    
    morf = PatchObliqueRandomForestClassifier(n_estimators=1, max_patch_dims=np.array((4, 4, 4)), data_dims=np.array((113, 137, 113)))
    morf.fit(X[idx_chosen,113*137*113:], y[idx_chosen])

    with open('morf_models/NHP_model'+str(ii)+'_white.pickle','wb') as f:
        pickle.dump(morf, f)

    feature_imp.append(
            morf.feature_importances_
        )
    del morf

with open('feature_imp_white_NHP.pickle','wb') as f:
    pickle.dump(feature_imp, f)
    


feature_imp = []
    
train_ids = idx[:train_samples]
for ii in tqdm(range(total_models)):
    random.shuffle(y)
    
    np.random.seed(ii)
    np.random.shuffle(train_ids)
    idx_chosen = train_ids[:int(len(y)*0.8*0.7)]
    
    morf = PatchObliqueRandomForestClassifier(n_estimators=1, max_patch_dims=np.array((4, 4, 4)), data_dims=np.array((113, 137, 113)), n_jobs=-1)
    morf.fit(X[idx_chosen,:113*137*113], y[idx_chosen])

    feature_imp.append(
            morf.feature_importances_
        )
    del morf

with open('feature_imp_gray_NHP_random.pickle','wb') as f:
    pickle.dump(feature_imp, f)


feature_imp = []
    
train_ids = idx[:train_samples]
for ii in tqdm(range(total_models)):
    random.shuffle(y)
    
    np.random.seed(ii)
    np.random.shuffle(train_ids)
    idx_chosen = train_ids[:int(len(y)*0.8*0.7)]
    
    morf = PatchObliqueRandomForestClassifier(n_estimators=1, max_patch_dims=np.array((4, 4, 4)), data_dims=np.array((113, 137, 113)), n_jobs=-1)
    morf.fit(X[idx_chosen,113*137*113:], y[idx_chosen])

    feature_imp.append(
            morf.feature_importances_
        )
    del morf

with open('feature_imp_white_NHP_random.pickle','wb') as f:
    pickle.dump(feature_imp, f)





