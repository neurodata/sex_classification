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
from joblib import Parallel, delayed
import numpy as np
import multiprocessing

class PermutationTest():
    r"""
    Feature importance test statistic and p-value.
    """

    def __init__(self, n_estimators, feature_importance):
        self.n_estimators = n_estimators
        self.feature_importance = ss.rankdata(1-feature_importance, method="max", axis=1)
        

    def _statistics(self, feature_rank, shuffle=True):
        r"""
        Helper function that calulates the feature importance
        test statistic.
        """
        idx = list(range(2 * self.n_estimators))
        
        if shuffle:
            np.random.shuffle(idx)
        
        stat = np.mean(feature_rank[idx[:self.n_estimators]]\
                       <feature_rank[idx[self.n_estimators:]],axis=0)

        return stat


    def test(self, n_repeats = 1000, n_jobs = -1):
        r"""
        Calculates p values for fearture imprtance test.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The data matrix.
        y : ArrayLike of shape (n_samples, n_outputs)
            The target matrix.
        n_repeats : int, optional
            Number of times to sample the null distribution, by default 1000.
        n_jobs : int, optional
            Number of workers to use, by default 1000.

        Returns
        -------
        stat : float
            The computed discriminability statistic.
        pvalue : float
            The computed one sample test p-value.
        """

        stat = self._statistics(
                    self.feature_importance,
                    shuffle=False
                )
        count = np.zeros(self.feature_importance[0].shape, dtype=float)

        if n_jobs == -1:
            cpu_count = multiprocessing.cpu_count()
        else: 
            cpu_count = n_jobs
            
        segment_len = int(np.ceil(len(self.feature_importance[0])/cpu_count))
        fragments = [ii*segment_len for ii in range(segment_len)]
        fragments.append(-1)
        

        for _ in tqdm(range(n_repeats)):
            null_stat = Parallel(n_jobs=n_jobs, verbose=False)(delayed(self._statistics)(self.feature_importance[:,fragments[ii]:fragments[ii+1]]) for ii in range(cpu_count))
            null_stat = np.concatenate(null_stat, axis=0)
            count += (null_stat >= stat) * 1
            
        p_val = (1 + count) / (1 + n_repeats)

        return stat, p_val


if __name__ == "__main__":
    total_models = 1000
    reps = 10000
    
    with open('feature_imp_white_NHP.pickle','rb') as f:
        feature_imp1 = pickle.load(f)

    with open('feature_imp_white_NHP_random.pickle','rb') as f:
        feature_imp2 = pickle.load(f)

    feature_imp = np.concatenate(
                        (
                            feature_imp1,
                            feature_imp2
                        ),
                        axis=0
                )


    test = PermutationTest(n_estimators=total_models, feature_importance=feature_imp)
    stat, p_val = test.test(n_repeats = reps, n_jobs=-1)

    with open('feature_imp_white_pval_NHP.pickle','wb') as f:
        pickle.dump(p_val, f)

    