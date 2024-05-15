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



df = pd.read_excel('/cis/home/jdey4/data_MRI/subjects_age_sex_data_MRI.xlsx')
df_quality = pd.read_csv('/cis/home/jdey4/data_MRI/QC_catreport.csv')

path = '/cis/home/jdey4/data_MRI/'
subjects = os.listdir(path)
X = []
y = []
file_no = 0
count = 0
count_ = 0
IDs = set(df['ID'])
quality_ID = set(df_quality['sub'])
for subject in tqdm(subjects):
    if subject in IDs and subject in quality_ID:
        #print(df[df['ID']==subject]['Sex'])
        IQR = list(df_quality[df_quality['sub']==subject]['Weighted average (IQR)'])[0]
        count += 1
        #print(IQR)
        if IQR is np.nan:
            continue
            
        if IQR[-1] == '%':
            continue
        
        if float(IQR) < 60:
            continue

        count_ += 1
        #print(count, count_)
        gender = list(df[df['ID']==subject]['Sex'])
        sex = int(gender[0]=='FEMALE')
        
        
        current_file = os.path.join(path, subject)
        file_count = 0
        files = glob.glob(current_file+'/mri/*')

        for file in files:
            if 'mwp1' in file:
                try:
                    img = nb.load(file).get_fdata().reshape(-1)
                    file_count +=1 
                except:
                    break
                
        for file in files:
            if 'mwp2' in file:
                try:
                    img = np.concatenate((img, nb.load(file).get_fdata().reshape(-1)))
                    file_count +=1 
                except:
                    break
                

        '''if len(tmp)<2:
            print(subject, ' has less files')'''
           
        if file_count==2:
            X.append(img.reshape(1,-1))
            y.append(sex)

    
X = np.concatenate(X,axis=0)
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

for ii in tqdm(range(650,total_models)):
    np.random.seed(ii)
    np.random.shuffle(train_ids)
    idx_chosen = train_ids[:int(len(y)*0.8*0.7)]
    
    morf = PatchObliqueRandomForestClassifier(n_estimators=1, max_patch_dims=np.array((4, 4, 4)), data_dims=np.array((113, 137, 113)), n_jobs=-1)
    morf.fit(X[idx_chosen,:113*137*113], y[idx_chosen])

    with open('morf_models/model'+str(ii)+'_gray.pickle','wb') as f:
        pickle.dump(morf, f)

    del morf

    morf = PatchObliqueRandomForestClassifier(n_estimators=1, max_patch_dims=np.array((4, 4, 4)), data_dims=np.array((113, 137, 113)), n_jobs=-1)
    morf.fit(X[idx_chosen,113*137*113:], y[idx_chosen])

    with open('morf_models/model'+str(ii)+'_white.pickle','wb') as f:
        pickle.dump(morf, f)

    del morf



#feature_imp_gray = []
#feature_imp_white = []
with open('feature_imp_gray_random.pickle','rb') as f:
    feature_imp_gray = pickle.load(f)

with open('feature_imp_white_random.pickle','rb') as f:
    feature_imp_white = pickle.load(f)


train_ids = idx[:train_samples]
for ii in tqdm(range(650,total_models)):
    random.shuffle(y)
    
    np.random.seed(ii)
    np.random.shuffle(train_ids)
    idx_chosen = train_ids[:int(len(y)*0.8*0.7)]
    
    morf = PatchObliqueRandomForestClassifier(n_estimators=1, max_patch_dims=np.array((4, 4, 4)), data_dims=np.array((113, 137, 113)), n_jobs=-1)
    morf.fit(X[idx_chosen,:113*137*113], y[idx_chosen])

    feature_imp_gray.append(
            morf.feature_importances_
        )

    del morf

    morf = PatchObliqueRandomForestClassifier(n_estimators=1, max_patch_dims=np.array((4, 4, 4)), data_dims=np.array((113, 137, 113)), n_jobs=-1)
    morf.fit(X[idx_chosen,113*137*113:], y[idx_chosen])

    feature_imp_white.append(
            morf.feature_importances_
        )

    del morf


with open('feature_imp_gray_random.pickle','wb') as f:
    pickle.dump(feature_imp_gray, f)

with open('feature_imp_white_random.pickle','wb') as f:
    pickle.dump(feature_imp_white, f)

