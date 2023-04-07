import joblib
import torch
from tqdm import tqdm
import numpy as np

data = joblib.load(open('data/dense_train_feat.pkl', 'rb'))

for j,d in enumerate(data):
    if j==0:
        index=2
    else:
        index=1
    with tqdm(total=len(d)) as p:
        for i, t in enumerate(d):
            if isinstance(t[index],np.ndarray):
                t[index]=torch.from_numpy(t[index])
            t[index] = t[index].cuda()
            p.update(1)

for j,d in enumerate(data):
    if j==0:
        index=2
    else:
        index=1
    with tqdm(total=len(d)) as p:
        for i, t in enumerate(d):
            if torch.isnan(t[index]).sum() > 0:
                print('NAN')
            if torch.isinf(t[index]).sum() > 0:
                print('inf')
            p.update(1)

print(0)
