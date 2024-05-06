import os
import torch 
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset

from . import reader
from . import PINNs
from . import funtions

###               ###
#     read data     #
###               ###

pt_path = 'dataExample.pt'
train_DS = reader.Pdyn_ExtractDataset(Data_pt=pt_path, n_grid=300, collocation_points=300, n_repeat=10)
val_DS= reader.Pdyn_ExtractDataset(Data_pt=pt_path, n_grid=300, collocation_points=600, n_repeat=1)

train_Loader = DataLoader(train_DS, batch_size=1, num_workers=4)
val_Loader = DataLoader(val_DS, batch_size=1, num_workers=4)



###                ###
#     load model     #
###                ###
