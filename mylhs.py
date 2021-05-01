# # pip install lhsmdu
# import lhsmdu
# lhsmdu.setRandomSeed(110)
# k = lhsmdu.sample(2, 20)  # Latin Hypercube Sampling with multi-dimensional uniformity


from pyDOE import *
import pandas as pd, numpy as np

# mnist
from init import hyp_rngs

lhs_x = lhs(n=len(hyp_rngs), samples=70, criterion='maximin')
lhs_df = pd.DataFrame()
for i, (k, v) in enumerate(hyp_rngs.items()):
    if k not in ['lr', 'blur_prec']:
        lhs_df[k] = np.floor(lhs_x[:, i] * (v[1] - v[0]) + v[0]).astype(int)
    else:
        lhs_df[k] = np.round(lhs_x[:, i] * (v[1]-v[0]) + v[0], 4)

lhs_df.to_csv("lhs-mnist.csv", index=False)

