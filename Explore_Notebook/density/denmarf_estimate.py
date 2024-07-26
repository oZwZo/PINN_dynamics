
from denmarf import DensityEstimate
import numpy as np


save_dir = "/ssd/users/Wergillius/Project/PINN_dynamics/Explore_Notebook/density"


DM_ay = np.load('/ssd/users/Wergillius/Project/PINN_dynamics/data/mkery_DM10.npy', allow_pickle=True)
timepoints = [ 3,   7,  12,  27,  49,  76, 112, 161, 269]


for i, dm in enumerate(DM_ay):    
    de = DensityEstimate(device='cuda:2', use_cuda=True).fit(dm, p_train=1, num_epochs=500)
    de.save(f"{save_dir}/DiffMap30_day{timepoints[i]}th.pkl")

for i, dm in enumerate(DM_ay):    
    de = DensityEstimate(device='cuda:2', use_cuda=True).fit(dm[:,:5], p_train=1, num_epochs=500)
    de.save(f"{save_dir}/DiffMap5_day{timepoints[i]}th.pkl")