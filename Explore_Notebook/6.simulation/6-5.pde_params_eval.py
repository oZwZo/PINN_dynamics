import os, sys
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import PINN
from PINN import reader, models, pl, tl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from TorchDiffEqPack import odesolve
from torchdiffeq import odeint

import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Patch

os.chdir("/ssd/users/Wergillius/Project/PINN_dynamics")

# CHANG THIS !!!!!
ckpt_path = "logs/tom_pos-DM_scaled_n9/pde_params_tsense/lightning_logs/version_1/checkpoints/epoch=174-total_loss=0.88114280.ckpt"

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
n_dimension =10

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
            u_pred = pde_model(s_ts[i:i+chunk_size],  t_ts[i:i+chunk_size])
            u_pred_ls.append(u_pred.detach().cpu().numpy())
    u_pred_b = np.concatenate(u_pred_ls, axis=0).reshape(n_timepoint,-1)

    PINN.pl.params_in_umap(t7_ad, u_b, param='u', cell_of_t=False);
    PINN.pl.params_in_umap(t7_ad, u_pred_b, param='u pred by forward', cell_of_t=False);

# other params
dynamics_params = ['g', 'v', 'D']

if pde_model.time_sensitive:
    for param in dynamics_params:
        param_pred = pde_model.predict_param(DS_t7, param=param);
        if param_pred.shape[1] != t7_ad.shape[0]:
            param_pred = param_pred.reshape(5, -1, n_dimension)
            param_pred = pde_model.trace_div(param_pred,)
        PINN.pl.params_in_umap(t7_ad, param_pred, param=param, cell_of_t=False);
else:
    for param in dynamics_params:
        t7_ad.obs[param] = pde_model.predict_param(DS_t7, param=param);
    sc.pl.umap(t7_ad, color=dynamics_params, ncols=3, frameon=False, size=50)    

with torch.no_grad():
for t in timepoints:
        for i in range(0, len(cellstate), chunk_size):
        s0 = cellstate[i:i+chunk_size].to(device)

        cellstate = c
        t_input
        pde_model.g()

##
## predict u with ode integrat
u_int_all = [u_b[0]]
chunk_size= 500
# t_list = timepoints / 15
it = 0
for t0, t1 in tqdm(zip(timepoints[:-1], timepoints[1:])):

    u_t_ls = []

    for i in range(0, len(cellstate), chunk_size):

        s0 = cellstate[i:i+chunk_size].to(device)
        tompos_u0 = torch.from_numpy(u_b[it, i:i+chunk_size]).float().to(device).requires_grad_()
        init_condition = (tompos_u0, s0)

        step_size = np.around((t1 - t0)/15, decimals=1).item() 
        step_size = step_size if step_size > 0 else 0.05
        step_size = min(step_size, 0.4)


        u_t, s_t = odeint(
                        pde_model.ode_func,
                        init_condition,
                        torch.tensor([t0, t1]).type(torch.float32).to(device)/15,
                        atol=1e-8,
                        rtol=1e-8,
                        method='midpoint',
                        options = {'step_size': step_size}
                    )

        torch.cuda.empty_cache()

        u_t_ls.append(u_t[-1].detach().cpu().numpy())
        del u_t, s_t
        torch.cuda.empty_cache()

    u_int = np.concatenate(u_t_ls)
    
    u_int_all.append(u_int)
    it+=0

u_int_all = np.stack(u_int_all)
print(u_b.sum(axis=1))
print(u_int_all.sum(axis=1))


PINN.pl.params_in_umap(t7_ad, u_int_all[:-2], param='u by integrat', cell_of_t=False);
PINN.pl.params_in_umap(t7_ad, u_int_all, timepoints=timepoints, param='u by integrat', cell_of_t=False);


# add the density into obs
for i, d in enumerate(timepoints):
    t7_ad.obs[f'u_int_{d}'] = u_int_all[i]
    t7_ad.obs[f'u_obs_{d}'] = u_b[i]

    t7_ad.obs[f'p_int_{d}'] = u_int_all[i] / u_int_all[i].sum()
    t7_ad.obs[f'p_obs_{d}'] = u_b[i] / u_b[i].sum()


obs = t7_ad.obs.copy()

proint_columns = ['p_int_3', 'p_int_7', 'p_int_12',
       'p_int_27', 'p_int_49', 'p_int_76', 'p_int_112', 'p_int_161',
       'p_int_269'] 
probs_columns =['p_obs_3', 'p_obs_7', 'p_obs_12', 'p_obs_27', 'p_obs_49',
       'p_obs_76', 'p_obs_112', 'p_obs_161', 'p_obs_269']
# density by cell type


cm_celltype = dict(zip(t7_ad.obs.anno_man.cat.categories ,t7_ad.uns['anno_man_colors']))
cm_leiden = dict(zip(t7_ad.obs.leiden.cat.categories ,t7_ad.uns['leiden_colors']))


p_by_celltype = obs[probs_columns+proint_columns+['anno_man']].groupby("anno_man").agg("sum")
p_celltype_melt = pd.melt(p_by_celltype.reset_index(), id_vars=['anno_man'])
p_celltype_melt['data'] = p_celltype_melt['variable'].str.extract(r"p_(\w{3})_\d")
p_celltype_melt['time'] = p_celltype_melt['variable'].str.extract(r"p_\w{3}_(\d*)")

p_by_leiden = obs[probs_columns+proint_columns+['leiden']].groupby("leiden").agg("sum")
p_leiden_melt = pd.melt(p_by_leiden.reset_index(), id_vars=['leiden'])
p_leiden_melt['data'] = p_leiden_melt['variable'].str.extract(r"p_(\w{3})_\d")
p_leiden_melt['time'] = p_leiden_melt['variable'].str.extract(r"p_\w{3}_(\d*)")

def stack_catplot(x, y, cat, stack, data, palette=sns.color_palette('Reds')):
    ax = plt.gca()
    # pivot the data based on categories and stacks
    # df = data.pivot_table(values=y, index=[cat, x], columns=stack, 
    #                       dropna=False, aggfunc='sum').fillna(0)
    ncat = data[cat].nunique()
    nx = data[x].nunique()
    nstack = data[stack].nunique()
    range_x = np.arange(nx)
    width = 0.8 / ncat # width of each bar
    
    for i, c in enumerate(data[cat].unique()):
        # iterate over categories, i.e., Conditions
        # calculate the location of each bar
        loc_x = (0.5 + i - ncat / 2) * width + range_x
        bottom = 0

        for j, s in enumerate(data[stack].unique()):
            # iterate over stacks, i.e., Hosts
            # obtain the height of each stack of a bar
            height_df = data.query(f"`{cat}` == @c & `{stack}`==@s")
            height_df = height_df.set_index(x)
            height = height_df.loc[data[x].unique(), y]
            # plot the bar, you can customize the color yourself
            
            hatch = '/' if i == 1 else None
            barcontainer = ax.bar(x=loc_x, height=height, 
                                  bottom=bottom, width=width*0.7, 
                                    color=palette[s], 
                                    # zorder=10, 
                                    lw=0.1,
                                    hatch=hatch, label=f"{c}: {s}")
            
  
            for bc in barcontainer:
                bc._hatch_color = mpl.colors.to_rgba("w")
                bc.stale = True
            
            # change the bottom attribute to achieve a stacked barplot
            bottom += height

    # make xlabel
    ax.set_xticks(range_x)
    ax.set_xticklabels(data[x].unique(), rotation=45)
    ax.set_ylabel(y)
    # make legend
    plt.legend(
            #     [Patch(facecolor=palette[i]) for i in range(ncat * nstack)], 
            #    [f"{c}: {s}" for c in data[cat].unique() for s in data[stack].unique()],
               bbox_to_anchor=(1.05, 0.8), loc='upper left', borderaxespad=0., ncol=2)
    plt.grid()
    return ax


plt.figure(figsize=(9, 3), dpi=300)
ax=stack_catplot(x='time', y='value', cat='data', stack='anno_man', data=p_celltype_melt , palette=cm_celltype)
ax.set_xlabel("time")
ax.set_ylabel("cell type proportion")


plt.figure(figsize=(10, 5), dpi=300)
ax=stack_catplot(x='time', y='value', cat='data', stack='leiden', data=p_leiden_melt , palette=cm_leiden)
ax.set_xlabel("time")
ax.set_ylabel("cell cluster proportion")
