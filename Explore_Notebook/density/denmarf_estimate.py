
from denmarf import DensityEstimate
import numpy as np


save_dir = "/home/wergillius/Project/TIGON/Density/denmarf_pickle"


DM_ay = np.load('/home/wergillius/Project/TIGON/Input/clu7_DiffMap.npy', allow_pickle=True)
timepoints = [ 3,   7,  12,  27,  49,  76, 112, 161, 269]


for i, dm in enumerate(DM_ay):    
    de = DensityEstimate(device='cuda:0', use_cuda=True).fit(dm, num_epochs=500)
    de.save(f"{save_dir}/DiffMap30_day{timepoints[i]}th.pkl")

for i, dm in enumerate(DM_ay):    
    de = DensityEstimate(device='cuda:0', use_cuda=True).fit(dm[:,:10], num_epochs=500)
    de.save(f"{save_dir}/DiffMap10_day{timepoints[i]}th.pkl")