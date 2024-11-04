import os, sys, re
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import PINN
from PINN import reader, models, pl, tl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from TorchDiffEqPack import odesolve
from torchdiffeq import odeint_adjoint as odeint

import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Patch

os.chdir("/ssd/users/Wergillius/Project/PINN_dynamics")

# CHANG THIS !!!!!
ckpt_path = "logs/klein_subset-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_4/checkpoints/epoch=101-total_loss=0.32358119.ckpt"

result_dir = "results/" + "/".join(ckpt_path.split("/")[1:3]) 
result_base = re.match(r".*/(version_\d{1,2})/.*", ckpt_path).group(1)

if not os.path.exists(result_dir):
    try:
        os.mkdir(result_dir)
    except:
        os.mkdir(os.path.dirname(result_dir))
        os.mkdir(result_dir)


model_name = ckpt_path.split("/")[2].replace("_tsense","")
model_class = eval(f"models.{model_name}")

# MODEL CLASS and define model
pde_model = model_class.load_from_checkpoint(ckpt_path)
device = pde_model.device

data_name = ckpt_path.split("/")[1].split("-")[0]
adata = sc.read_h5ad(f"data/{data_name}.h5ad")
timepoints = adata.uns['pop']['t']

cellstate_key = ckpt_path.split("/")[1].split("-")[1].split("_n")[0]

n_timepoint = int(ckpt_path.split("/")[1].split("-")[1].split("_n")[1])
n_dimension = 5

# the DS
DS_t7 = reader.TwoTimpepoint_AnnDS(AnnData=adata, n_timepoint=n_timepoint,  
                            n_dimension = n_dimension,
                              cellstate_key=cellstate_key,  #'DM_EigenVector'
                              log_transform=False,
                              norm_time=False,
                              batchsize = 300
                              )

# adata
t7_ad = DS_t7.adata.copy()
u_b = DS_t7.u_b.cpu().numpy().reshape(n_timepoint, -1)
cellstate = torch.from_numpy(DS_t7.cellstate).float().requires_grad_()

if "u" in dir(pde_model):
    u_pred_ls = []
    chunk_size= 500

    s_ts = DS_t7.s.float().to(device)
    t_ts = DS_t7.t_b.float().to(device)

    with torch.no_grad():
        for i in tqdm(range(0, len(t_ts), chunk_size)):
            u_pred = torch.exp(pde_model.u(s_ts[i:i+chunk_size],  t_ts[i:i+chunk_size]))
            u_pred_ls.append(u_pred.detach().cpu().numpy())
    u_pred_b = np.concatenate(u_pred_ls, axis=0).reshape(n_timepoint,-1)

    PINN.pl.params_in_umap(t7_ad, u_b, param='u', cell_of_t=False);
    PINN.pl.params_in_umap(t7_ad, u_pred_b, param='u pred by forward', cell_of_t=False);

# other params
dynamics_params = ['g', 'D', 'v']

if pde_model.time_sensitive:
    for param in dynamics_params:
        param_pred = pde_model.predict_param(DS_t7, param=param);
        if param_pred.shape[1] != t7_ad.shape[0]:
            param_pred = param_pred.reshape(n_timepoint, -1, n_dimension)
            param_pred = pde_model.trace_div(param_pred,cellstate)
        
        # saved
        np.save(f"{result_dir}/{result_base}_{param}.npy", param_pred)
        PINN.pl.params_in_umap(t7_ad, param_pred, param=param, cell_of_t=False);
else:
    for param in dynamics_params:
        t7_ad.obs[param] = pde_model.predict_param(DS_t7, param=param);
    sc.pl.umap(t7_ad, color=dynamics_params, ncols=3, frameon=False, size=50)    

# with torch.no_grad():
#     for t in timepoints:
#             for i in range(0, len(cellstate), chunk_size):
#             s0 = cellstate[i:i+chunk_size].to(device)

#             cellstate = c
#             t_input
#             pde_model.g()

##
## predict u with ode integrat
u_int_all = [u_b[0]]
chunk_size= 500
t_list = timepoints / 10
it = 0
for t0, t1 in tqdm(zip(t_list[:-1], t_list[1:])):

    u_t_ls = []

    for i in range(0, len(cellstate), chunk_size):

        s0 = cellstate[i:i+chunk_size].to(device)
        tompos_u0 = torch.from_numpy(u_b[it, i:i+chunk_size]).float().to(device).requires_grad_()
        init_condition = (tompos_u0, s0)

        step_size = np.around((t1 - t0)/15, decimals=1).item() 
        step_size = step_size if step_size > 0 else 0.05
        step_size = min(step_size, 0.4)


        u_t, s_t = odeint(
                        pde_model,
                        y0 = init_condition,
                        t = torch.tensor([t0, t1]).type(torch.float32).to(device),
                        atol=1e-4,
                        rtol=1e-4,
                        method='dopri5',
                        adjoint_options={'norm':'seminorm'},
                    )

        torch.cuda.empty_cache()

        u_int = torch.nn.functional.relu(u_t[-1])
        u_t_ls.append(u_int.detach().cpu().numpy())
        del u_t, s_t
        torch.cuda.empty_cache()

    u_int = np.concatenate(u_t_ls)
    
    u_int_all.append(u_int)
    it+=1

u_int_all = np.stack(u_int_all)

print(u_b.sum(axis=1))
print(u_int_all.sum(axis=1))


PINN.pl.params_in_umap(t7_ad, u_b, param='u b', cell_of_t=True);
PINN.pl.params_in_umap(t7_ad, u_int_all, param='u by integrat', cell_of_t=True);
PINN.pl.params_in_umap(t7_ad, u_int_all, timepoints=timepoints, param='u by integrat', cell_of_t=False);


ct_key = 'label_man'

# add the density into obs
for i, d in enumerate(timepoints):
    t7_ad.obs[f'u_int_{d}'] = u_int_all[i]
    t7_ad.obs[f'u_obs_{d}'] = u_b[i]

    t7_ad.obs[f'p_int_{d}'] = u_int_all[i] / u_int_all[i].sum()
    t7_ad.obs[f'p_obs_{d}'] = u_b[i] / u_b[i].sum()


obs = t7_ad.obs.copy()

proint_columns = ['p_int_%s'%t for t in timepoints] 
probs_columns =['p_obs_%s'%t for t in timepoints] 
# density by cell type


cm_celltype = dict(zip(t7_ad.obs[ct_key].cat.categories ,t7_ad.uns[f'{ct_key}_colors']))


p_by_celltype = obs[probs_columns+proint_columns+[ct_key]].groupby(ct_key).agg("sum")
p_celltype_melt = pd.melt(p_by_celltype.reset_index(), id_vars=[ct_key])
p_celltype_melt['data'] = p_celltype_melt['variable'].str.extract(r"p_(\w{3})_\d")
p_celltype_melt['time'] = p_celltype_melt['variable'].str.extract(r"p_\w{3}_(\d*)")


plt.figure(figsize=(9, 3), dpi=300)
ax=PINN.pl.stack_catplot(x='time', y='value', cat='data', stack=ct_key, data=p_celltype_melt , palette=cm_celltype)
ax.set_xlabel("time")
ax.set_ylabel("cell type proportion")


fig, axs = plt.subplots(1, n_timepoint, figsize=(3*n_timepoint, 6), dpi=300, sharey=True)
for i, t in enumerate(timepoints):
    t = str(t)
    ax=axs[i]

    sns.barplot(data=p_celltype_melt.query("`time` == @t"), 
                # color=ct_key, #palette=cm_celltype,
                edgecolor='gray', width=0.7,
                x = 'value', y=ct_key, hue='data', ax=ax)
    
    for bars, hatch, legend_handle in zip(ax.containers, ['', '//'], ax.legend_.legendHandles):
        for bar, color in zip(bars, cm_celltype.values()):
            alpha = 1 if hatch == '' else 0.5
            bar.set_alpha(alpha)
            bar.set_facecolor(color)
            bar.set_hatch(hatch)
        # update the existing legend, use twice the hatching pattern to make it denser
        legend_handle.set_hatch(hatch + hatch)

    sns.despine()
    axs[i].set_xlabel("")
    if i!=0:
        axs[i].legend([], frameon=False)

axs[0].set_ylabel("")
axs[1].set_xlabel("cell type proportion")

