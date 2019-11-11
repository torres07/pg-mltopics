# -*- coding: utf-8 -*-
# @Author: pedrotorres
# @Date:   2019-11-11 17:08:54
# @Last Modified by:   pedrotorres
# @Last Modified time: 2019-11-11 17:08:59

import tensorly as tl
import numpy as np
import pandas as pd
from tensorly.decomposition import parafac

data_beatles = pd.read_csv('sbeatles.csv', delimiter=' , ', engine='python', names=['subject', 'predicate', 'object'], skiprows=1)
data_beatles = data_beatles.dropna()

subjects = list(data_beatles['subject'].unique())
objects = list(data_beatles['object'].unique())
predicates = list(data_beatles['predicate'].unique())
k, l, m = len(subjects), len(objects), len(predicates)

T = np.zeros(k * l * m, dtype=np.float32).reshape((k, l, m))

predicates_freq = data_beatles['predicate'].value_counts()
alpha = predicates_freq[predicates_freq.idxmax()]

for _, r in data_beatles.iterrows():
	i, j, k = subjects.index(r[0]), objects.index(r[2]), predicates.index(r[1])
	T[i, j, k] = 1 + np.log(alpha / predicates_freq[r[1]])

ub = min(k * l, min(l * m, k * m))
print('upper bound: {}'.format(ub))

PARAFAC = parafac(T, rank=32, n_iter_max=512, verbose=True, normalize_factors=True)

import pickle

with open('pp_parafac_ub_32_norm.pkl', 'wb') as f:
	pickle.dump(PARAFAC, f)