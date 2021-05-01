# # pip install lhsmdu
# import lhsmdu
# lhsmdu.setRandomSeed(110)
# k = lhsmdu.sample(2, 20)  # Latin Hypercube Sampling with multi-dimensional uniformity


from pyDOE import *
import pandas as pd, numpy as np

# mnist
hyp_rngs = {'lr': (1e-4, 1e-1), 'batch_size': (10, 64), 'fc_size': (30, 200), 'mxp_krnl': (2, 10),
            'blur_prec': (0.2, .8)}

lhs_x = lhs(n=len(hyp_rngs), samples=70, criterion='maximin')
lhs_df = pd.DataFrame()
for i, (k, v) in enumerate(hyp_rngs.items()):
    if k not in ['lr', 'blur_prec']:
        lhs_df[k] = np.floor(lhs_x[:, i] * (v[1] - v[0]) + v[0]).astype(int)
    else:
        lhs_df[k] = np.round(lhs_x[:, i]* (v[1]-v[0])+ v[0], 4)

lhs_df.to_csv("lhs-mnist.csv", index=False)

