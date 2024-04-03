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


total_models = 200
feature_imp = []
for ii in tqdm(range(total_models)):
    with open('morf_models/model'+str(ii)+'.pickle','rb') as f:
        morf = pickle.load(f)
        feature_imp.append(
            morf.feature_importances_
        )
        del morf

for ii in tqdm(range(total_models)):
    with open('morf_shuffled_models/model'+str(ii)+'.pickle','rb') as f:
        morf = pickle.load(f)
        feature_imp.append(
            morf.feature_importances_
        )
        del morf


with open('feature_imp.pickle','wb') as f:
    pickle.dump(feature_imp, f)