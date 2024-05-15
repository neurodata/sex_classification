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


total_models = 1000
feature_imp = []
with open('feature_imp_gray_NHP.pickle','rb') as f:
    feature_imp = pickle.load(f)

#print(len(feature_imp), 'its the length')

for ii in tqdm(range(500,total_models)):
    with open('morf_models/NHP_model'+str(ii)+'_gray.pickle','rb') as f:
        morf = pickle.load(f)
        feature_imp.append(
            morf.feature_importances_
        )
        del morf


with open('feature_imp_gray_NHP.pickle','wb') as f:
    pickle.dump(feature_imp, f)


#############################################################
feature_imp = []
with open('feature_imp_white_NHP.pickle','rb') as f:
    feature_imp = pickle.load(f)
    
for ii in tqdm(range(500,total_models)):
    with open('morf_models/NHP_model'+str(ii)+'_white.pickle','rb') as f:
        morf = pickle.load(f)
        feature_imp.append(
            morf.feature_importances_
        )
        del morf


with open('feature_imp_white_NHP.pickle','wb') as f:
    pickle.dump(feature_imp, f)

