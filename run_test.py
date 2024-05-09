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
        

    def _statistics(self, idx):
        r"""
        Helper function that calulates the feature importance
        test statistic.
        """
        diff_rank = self.feature_importance[idx[:self.n_estimators]] - \
            self.feature_importance[idx[self.n_estimators:]]
        stat = np.mean(diff_rank<0,axis=0)

        return stat

    def _perm_stat(self):
        r"""
        Helper function that calulates the null distribution.
        """

        idx = list(range(2 * self.n_estimators))
        np.random.shuffle(idx)

        return self._statistics(idx)

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

        stat = self._statistics(list(range(2 * self.n_estimators)))
        count = np.zeros(self.feature_importance[0].shape, dtype=float)

        if n_jobs == -1:
            cpu_count = multiprocessing.cpu_count()
        else: 
            cpu_count = n_jobs
            
        loops = int(np.ceil(n_repeats/cpu_count))
        n_repeats = loops*cpu_count

        for _ in tqdm(range(loops)):
            null_stat = Parallel(n_jobs=n_jobs, verbose=False)(delayed(self._perm_stat)() for _ in range(cpu_count))
            count += np.sum((null_stat >= stat) * 1, axis=0)

            del null_stat
            
        p_val = (1 + count) / (1 + n_repeats)

        return stat, p_val


if __name__ == "__main__":
    total_models = 500
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
    stat, p_val = test.test(n_repeats = reps, n_jobs=22)

    with open('feature_imp_white_pval_NHP.pickle','wb') as f:
        pickle.dump(p_val, f)

    