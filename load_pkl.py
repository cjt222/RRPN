import pickle
import pprint
import numpy as np

path = "/paddle/rrpn_baseline/RRPN_pytorch/models/IC-13-15-17-Trial-renew/model_0000020.pkl"

#pkl_file = open(path)
inf = np.load(path, allow_pickle=True)
#print(inf['backbone.body.stem.conv1.weight'].cpu().numpy())
k = 0
for i in inf:
    if "weight" in i or "bias" in i:
        print(i)
        k += 1
print(k)
