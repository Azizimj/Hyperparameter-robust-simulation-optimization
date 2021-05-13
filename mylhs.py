# # pip install lhsmdu
# import lhsmdu
# lhsmdu.setRandomSeed(110)
# k = lhsmdu.sample(2, 20)  # Latin Hypercube Sampling with multi-dimensional uniformity


from pyDOE import *
import pandas as pd, numpy as np
from init import hyp_rngs

def mylhs_f(fname, num_points):
    # np.random.seed(110)
    hyp_rngs.update({'blur_prec': (0, .7)})
    lhs_x = lhs(n=len(hyp_rngs), samples=num_points, criterion='maximin')
    lhs_df = pd.DataFrame()
    for i, (k, v) in enumerate(hyp_rngs.items()):
        if k not in ['lr', 'blur_prec']:
            lhs_df[k] = np.floor(lhs_x[:, i] * (v[1] - v[0]) + v[0]).astype(int)
        else:
            lhs_df[k] = np.round(lhs_x[:, i] * (v[1]-v[0]) + v[0], 4)

    lhs_df.to_csv(fname, index=False)
    return fname

