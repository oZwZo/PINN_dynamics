%load_ext autoreload
%autoreload 2
import os, sys, re
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import time
import PINN
from PINN import reader, models, pl, tl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from TorchDiffEqPack import odesolve
from torchdiffeq import odeint_adjoint as odeint

import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Patch

from matplotlib.backends.backend_pdf import PdfPages

os.chdir("/ssd/users/Wergillius/Project/PINN_dynamics")

def savefig(fig, name):
    fig.savefig(f'{plot_save}/{name}.png', transparent=True, bbox_inches='tight', dpi=200)



ckpt_path = "logs/tom_pos-DM_scaled_n[0, 1, 2, 3, 4, 6, 8]/pde_params_tsense/lightning_logs/version_4/checkpoints/epoch=95-total_loss=1.33532071.ckpt"
if __name__ == '__main__':
    ckpt_path = sys.argv[1] if not sys.argv[1].endswith("json") else ckpt_path
    

# CHANGE THIS !!!!!
n_dimension = 10
ct_key = 'anno_man'
use_device="cuda:3"


date = time.strftime("%b%d")
result_dir = "results/" + "/".join(ckpt_path.split("/")[1:3]) + f"_{date}_onbase"
# result_base = re.match(r".*/(version_\d{1,2})/.*", ckpt_path).group(1)
result_base = ckpt_path.split("/")[4]

if not os.path.exists(result_dir):
    try:
        os.mkdir(result_dir)
    except:
        os.mkdir(os.path.dirname(result_dir))
        os.mkdir(result_dir)

plot_save = f'{result_dir}/{result_base}_plot'
if not os.path.exists(plot_save):
    os.mkdir(plot_save)


# load model
model_name = ckpt_path.split("/")[2].replace("_tsense","")
model_class = eval(f"models.{model_name}")

# MODEL CLASS and define model
pde_model = model_class.load_from_checkpoint(ckpt_path, map_location=use_device)
device = pde_model.device

data_name = ckpt_path.split("/")[1].split("-")[0]
adata = sc.read_h5ad(f"data/{data_name}.h5ad")
full_full_timepoints = adata.uns['pop']['t']


cellstate_key = ckpt_path.split("/")[1].split("-")[1].split("_n")[0]
timepoint_idx = eval(ckpt_path.split("/")[1].split("-")[1].split("_n")[1])
timepoint_idx = timepoint_idx if type(timepoint_idx)==list else  np.arange(timepoint_idx)



# the DS
DS_full = reader.TwoTimpepoint_AnnDS(AnnData=adata, timepoint_idx=None,  
                            n_dimension = n_dimension,
                              cellstate_key=cellstate_key,  #'DM_EigenVector'
                              log_transform=False,
                              norm_time=False,
                              batchsize = 300
                              )

DS_sub = reader.TwoTimpepoint_AnnDS(AnnData=adata, timepoint_idx=None,  
                    n_dimension = n_dimension,
                        cellstate_key=cellstate_key,  #'DM_EigenVector'
                        log_transform=False,
                        norm_time=False,
                        batchsize = 300
                        )                       

# adata
t7_ad = DS_full.adata.copy()
u_b = DS_full.u_b.cpu().numpy().reshape(DS_full.T_b.shape[0], -1)
cellstate = torch.from_numpy(DS_full.cellstate).float()
timepoints = DS_full.popD['t']

try:
    duds = DS_full.duds.copy()
except AttributeError:
    pass


imputed_t_idx = [i for i,t in enumerate(timepoints) if i not in timepoint_idx]
# if cellstate.shape[0] > t7_ad.shape[0]:
#     t7_ad = full_adata

DM_range = (cellstate.max(axis=0).values - cellstate.min(axis=0).values).cpu().numpy()

# FORWARD 
u_pred_ls = []
v_ls = []
D_ls = []
g_ls = []
chunk_size= 500

s_ts = DS_full.s.float().to(device)
t_ts = DS_full.t_b.float().to(device)

with torch.no_grad():
    for i in tqdm(range(0, len(t_ts), chunk_size)):                              
        s_in = s_ts[i:i+chunk_size]
        t_in = t_ts[i:i+chunk_size]

        v_pred = pde_model.v(s_in, t_in)
        g_pred = pde_model.g(s_in, t_in)
        D_pred = pde_model.D(s_in, t_in)
        
        v_ls.append(v_pred.detach().cpu().numpy())
        g_ls.append(g_pred.detach().cpu().numpy())
        D_ls.append(D_pred.detach().cpu().numpy())

        if "u" in dir(pde_model):
            u_pred = torch.exp(pde_model.u(s_in, t_in))
            u_pred_ls.append(u_pred.detach().cpu().numpy())

n_timepoint =  DS_full.T_b.shape[0]
if "u" in dir(pde_model):
    u_pred_b = np.concatenate(u_pred_ls, axis=0).reshape(n_timepoint,-1)
    fig1, axs1 = PINN.pl.params_in_umap(t7_ad, u_b, param='u', cell_of_t=False);
    fig2, axs2 = PINN.pl.params_in_umap(t7_ad, u_pred_b, param='u forward', cell_of_t=False);

    
# # other params

v_pred_ay = np.concatenate(v_ls, axis=0).reshape(n_timepoint, -1, n_dimension)
g_pred_ay = np.concatenate(g_ls, axis=0).reshape(n_timepoint,-1)
D_pred_ay = np.concatenate(D_ls, axis=0).reshape(n_timepoint,-1)
    

for param in ['g', 'D']:
    param_pred = locals()[f'{param}_pred_ay']
    npy_savepath = f"{result_dir}/{result_base}_{param}.npy"
    np.save(npy_savepath, param_pred)
    print(param, 'saved to ', npy_savepath)
    fig_param, axs_param = PINN.pl.params_in_umap(t7_ad, param_pred, param=param, cell_of_t=False);
    #save

    fig_param.savefig(f'{plot_save}/{param}.png', transparent=True, bbox_inches='tight', dpi=200)

nabla_v = pde_model.predict_nabla_v(DS_full).sum(axis=-1);
fig_v1, axs_v = PINN.pl.params_in_umap(t7_ad, nabla_v, param=r'$\nabla v$', cell_of_t=False);


v_norm = v_pred_ay / DM_range[None,None,:]
v_norm2 = np.sqrt(np.power(v_norm,2).mean(axis=2))
fig_v2, axs_v = PINN.pl.params_in_umap(t7_ad, v_norm2, param=r'$||v||^2$', cell_of_t=False);

np.save(f"{result_dir}/{result_base}_v.npy", v_pred_ay)
np.save(f"{result_dir}/{result_base}_v_norm.npy", v_norm2)
# PINN.pl.params_in_umap(t7_ad, v_norm2, param=r'$||v||^2$', cell_of_t=True);

if "u" in dir(pde_model):
    savefig(fig1, "observed_density")
    savefig(fig2, "forward_pred_density")

savefig(fig_v1, "nabla_v")
savefig(fig_v2, "v_norm")

# v_stream plot

## generate adata base on cellsate coords
import scvelo as scv
cellstate_ad = PINN.tl.make_coord_adata(t7_ad, n_dimension=n_dimension, cellstate_key=cellstate_key, v = v_pred_ay)
vkeys = [k for k in list(cellstate_ad.layers.keys()) if k.endswith("v")]
sc.pp.neighbors(cellstate_ad, n_neighbors=15)


for vkey in vkeys:

    # compute velocity graph
    scv.tl.velocity_graph(cellstate_ad, vkey=vkey, xkey='cellstate', n_jobs=20)

    # vis
    fig_velocity = plt.figure(dpi=100, figsize=(4,4))
    ax = fig_velocity.gca()
    scv.pl.velocity_embedding_stream(cellstate_ad, color=ct_key, vkey=vkey, 
                                     basis='umap', ax=ax, 
                                     legend_loc='right', alpha=0.01,
                                     title=vkey, 
                                     save=f"{plot_save}/{vkey}.png")

#
# predict u with ode integrat


u_int_all_ls = []
chunk_size= 100
t_list = timepoints/ timepoints[0] * pde_model.time_scale_factor

for it, itp1 in tqdm(zip(timepoint_idx[:-1], timepoint_idx[1:])):

    u_t_ls = []

    for i in range(0, len(cellstate), chunk_size):

        s0 = cellstate[i:i+chunk_size].to(device).requires_grad_()
        tompos_u0 = torch.from_numpy(u_b[it, i:i+chunk_size]).float().to(device).requires_grad_()
        # duds_0 = torch.from_numpy(duds[it, i:i+chunk_size, :]).float().to(device).requires_grad_()
        
        init_condition = (tompos_u0, s0, duds_0) if model_name == 'pde_u_free' else (tompos_u0, s0)

        int_out = odeint(
                        pde_model,
                        y0 = init_condition,
                        t = torch.tensor(t_list[it:itp1+1]).type(torch.float32).to(device),
                        atol=pde_model.ode_tol,
                        rtol=pde_model.ode_tol,
                        method='dopri5',
                        adjoint_options={'norm':'seminorm'},
                    )

        u_t = int_out[0]
        s_t = int_out[1]

        # torch.cuda.empty_cache()

        u_int = torch.nn.functional.relu(u_t[1:])
        u_t_ls.append(u_int.detach().cpu().numpy())
        del u_t, s_t
        # torch.cuda.empty_cache()

    u_int = np.concatenate(u_t_ls, axis=len(u_int.shape)-1)
    
    u_int_all_ls.append(u_int)
    # it+=1

u_int_ay = np.stack(u_int_all_ls) if len(u_int.shape)==1 else np.concatenate(u_int_all_ls, axis=0)
u_int_all = np.concatenate([u_b[[0]], u_int_ay], axis=0)

print(u_b.sum(axis=1))
print(u_int_all.sum(axis=1))

obs_u = np.log10(u_b + 1e-20)
siml_u = np.log10(u_int_all + 1e-20)

thres = np.quantile(siml_u, [0.05, 0.90])
siml_u_clipped = np.clip(siml_u, a_min=thres[0], a_max=thres[1])

fig_obs, axs = PINN.pl.params_in_umap(t7_ad, obs_u[timepoint_idx], timepoints=timepoints[timepoint_idx], param='\nlog10 observed density', cell_of_t=True);
fig_int1, axs = PINN.pl.params_in_umap(t7_ad, siml_u_clipped[timepoint_idx],timepoints=timepoints[timepoint_idx], param='\nlog10 simulated density', cell_of_t=True);

imputed_t_idx = np.array(timepoint_idx)[:-1]+1
fig_obs, axs = PINN.pl.params_in_umap(t7_ad, obs_u[imputed_t_idx], timepoints=timepoints[imputed_t_idx], param='\nlog10 observed density', cell_of_t=True);
fig_int1, axs = PINN.pl.params_in_umap(t7_ad, siml_u_clipped[imputed_t_idx],timepoints=timepoints[imputed_t_idx], param='\nlog10 simulated density', cell_of_t=True);



fig_obs, axs = PINN.pl.params_in_umap(t7_ad, u_b[timepoint_idx], timepoints=timepoints[timepoint_idx], param='\nlog10 observed density', cell_of_t=True);
fig_int1, axs = PINN.pl.params_in_umap(t7_ad, u_int_all[timepoint_idx],timepoints=timepoints[timepoint_idx], param='\nlog10 simulated density', cell_of_t=True);

imputed_t_idx = np.array(timepoint_idx)[:-1]+1
fig_obs, axs = PINN.pl.params_in_umap(t7_ad, u_b[imputed_t_idx], timepoints=timepoints[imputed_t_idx], param='\nlog10 observed density', cell_of_t=True);
fig_int1, axs = PINN.pl.params_in_umap(t7_ad, u_int_all[imputed_t_idx],timepoints=timepoints[imputed_t_idx], param='\nlog10 simulated density', cell_of_t=True);



fig_obs, axs = PINN.pl.params_in_umap(t7_ad, obs_u, timepoints=timepoint_label, param='\nlog10 observed density', cell_of_t=True);
fig_int1, axs = PINN.pl.params_in_umap(t7_ad, siml_u_clipped,timepoints=timepoint_label, param='\nlog10 simulated density', cell_of_t=True);

fig_obs, axs = PINN.pl.params_in_umap(t7_ad, obs_u, timepoints=timepoint_label, param='\nlog10 observed density', cell_of_t=False);
fig_int2, axs = PINN.pl.params_in_umap(t7_ad, siml_u_clipped, timepoints=timepoint_label, param='u by integrat', cell_of_t=False);


# long term
u_int_all = [u_b[0]]
chunk_size= 500
t_list = full_full_timepoints / 10
it = 0

u_t_ls = []

for i in range(0, len(cellstate), chunk_size):

    s0 = cellstate[i:i+chunk_size].to(device).requires_grad_()
    tompos_u0 = torch.from_numpy(u_b[0, i:i+chunk_size]).float().to(device).requires_grad_()
    init_condition = (tompos_u0, s0)

    step_size = np.around((t1 - t0)/15, decimals=1).item() 
    step_size = step_size if step_size > 0 else 0.05
    step_size = min(step_size, 0.4)

    u_t, s_t = odeint(
                    pde_model,
                    y0 = init_condition,
                    t = torch.tensor(t_list).type(torch.float32).to(device),
                    atol=1e-4,
                    rtol=1e-4,
                    method='dopri5',
                    adjoint_options={'norm':'seminorm'},
                )

    torch.cuda.empty_cache()

    u_int = torch.nn.functional.relu(u_t)
    u_t_ls.append(u_int.detach().cpu().numpy())
    del u_t, s_t
    torch.cuda.empty_cache()

u_int_all = np.concatenate(u_t_ls, axis=1)

print(u_b.sum(axis=1))
print(u_int_all.sum(axis=1))

obs_u = np.log10(u_b + 1e-20)
siml_u = np.log10(u_int_all + 1e-20)

thres = np.quantile(siml_u, [0.05, 0.90])
siml_u_clipped = np.clip(siml_u, a_min=thres[0], a_max=thres[1])

fig_obs, axs = PINN.pl.params_in_umap(t7_ad, obs_u, param='\nlog10 observed density', cell_of_t=True);
fig_int1, axs = PINN.pl.params_in_umap(t7_ad, siml_u_clipped, param='\nlog10 simulated density', cell_of_t=True);
fig_int2, axs = PINN.pl.params_in_umap(t7_ad, siml_u_clipped, timepoints=full_full_timepoints, param='u by integrat', cell_of_t=False);

from scipy.stats import pearsonr
from scipy.special import kl_div


def vis_perform(W, y_label, kind='bar'):
    fig = plt.figure()
    ax = fig.gca()

    plt_fn = eval(f"ax.{kind}")
    ax.plot(range(len(timepoints)),  W, color='lightgray', alpha=0.5)
    plt_fn(timepoint_idx, W[timepoint_idx],  color='navy', label='observed')
    plt_fn(imputed_t_idx, W[imputed_t_idx], color='orange', label='imputed')
    ax.set_xlabel("timepoints")
    ax.set_ylabel(y_label)
    ax.set_xticks(range(len(timepoints)))
    ax.set_xticklabels(timepoints)
    ax.legend()
    return fig, ax

def W_distance(u_b, u_int_all, p=2):
    distance_ls = []
    for t in range(u_b.shape[0]):
        p_b = u_b[t] / u_b[t].sum()
        p_int = u_int_all[t] / u_int_all[t].sum()
        if p==1:
            w = np.abs(p_b - p_int)
        elif p > 1:
            w = np.power(p_b - p_int, p)
            w = w**(1/p)
        distance_ls.append(np.sum(p_b*w))
    return np.array(distance_ls)

KLD_ls = []
for t in range(u_b.shape[0]):
    p_b = u_b[t] / u_b[t].sum()
    p_int = u_int_all[t] / u_int_all[t].sum()
    p_b += 1e-34
    p_int += 1e-34

    KLD_ls.append(kl_div(p_b, p_int).sum())
KLD_ls= np.array(KLD_ls)

W1 = W_distance(u_b, u_int_all, p=1)
W2 = W_distance(u_b, u_int_all)

perform_df=pd.DataFrame({'KLD': KLD_ls, 'W1':W1, "W2":W2})
perform_df['timepoints'] = timepoints
perform_df['datatype'] = ['observed' if i in timepoint_idx else 'imputed' for i,t in enumerate(timepoints) ]
perform_df.to_csv(f"{result_dir}/{result_base}_perform.csv",index=False)

vis_perform(KLD_ls, "KLD : true v.s. predicted density", kind='scatter')
vis_perform(W1, "W1 distance", kind='bar')
vis_perform(W2, "W2 distance", kind='bar')



# add the density into obs
full_timepoints = timepoints
for i, d in enumerate(timepoints):
    t7_ad.obs[f'u_int_{d}'] = u_int_all[i]
    t7_ad.obs[f'u_obs_{d}'] = u_b[i]

    t7_ad.obs[f'p_int_{d}'] = u_int_all[i] / u_int_all[i].sum()
    t7_ad.obs[f'p_obs_{d}'] = u_b[i] / u_b[i].sum()

gflow, vflow, dflow = pde_model.statify_flow(DS_full)
stratified_flow = {
    "dilution flow" : gflow,
    'drift flow' : vflow,
    'diffuse flow' : dflow
}

flow_keys = []
for key, flow in stratified_flow.items():
    for i, d in enumerate(timepoints[1:]):
        t7_ad.obs[f'Day{d} {key}'] = flow[i]
        flow_keys.append(f'Day{d} {key}')
    # visualize
    fig_flow, axs = PINN.pl.params_in_umap(t7_ad, flow, timepoints=full_timepoints[1:], param=key, cell_of_t=False);
    fig_flow.savefig(f'{plot_save}/{key}.png', transparent=True, bbox_inches='tight', dpi=200)

# flow by cell type
cm_celltype = dict(zip(t7_ad.obs[ct_key].cat.categories ,t7_ad.uns[f'{ct_key}_colors']))

for i, d in enumerate(full_timepoints[1:]):

    # stratified
    flow_key_obs = t7_ad.obs.groupby(ct_key).agg({f'Day{d} {key}':'sum' for key in stratified_flow})
    flow_melt_obs = flow_key_obs.reset_index().melt(id_vars=ct_key, var_name='flow')

    g1 = sns.catplot(data = flow_melt_obs, y = ct_key, hue=ct_key, x = 'value', col='flow', kind='bar', sharex=False, palette=cm_celltype)
    g1.savefig(f'{plot_save}/Day{d}_stratify_flow.png', transparent=True, bbox_inches='tight', dpi=200)

    # stratified flow fold chanage
    density_change = t7_ad.obs[f'u_int_{d}'].values - t7_ad.obs[f'u_int_{full_timepoints[i]}'].values
    t7_ad.obs[f'Day{d} u change'] = density_change

    agg_change = t7_ad.obs.groupby(ct_key).agg({f'Day{d} u change':'sum'}).values
    flow_fc_obs = flow_key_obs / agg_change
    flow_melt_fc = flow_fc_obs.reset_index().melt(id_vars=ct_key, var_name='flow')

    g2 = sns.catplot(data = flow_melt_fc, y = ct_key, hue=ct_key, x = 'value', col='flow', kind='bar', sharex=False, palette=cm_celltype)
    g2.savefig(f'{plot_save}/Day{d}_contrib.png', transparent=True, bbox_inches='tight', dpi=200)


    norm_u = t7_ad.obs.groupby(ct_key).agg({f'u_int_{d}':'sum'}).values
    flow_key_norm = flow_key_obs / norm_u
    flow_melt_norm = flow_key_norm.reset_index().melt(id_vars=ct_key, var_name='flow')

    g3 = sns.catplot(data = flow_melt_norm, y = ct_key, hue=ct_key, x = 'value', col='flow', kind='bar', sharex=False, palette=cm_celltype)
    g3.savefig(f'{plot_save}/Day{d}_stratify_flow_norm.png', transparent=True, bbox_inches='tight', dpi=200)



obs = t7_ad.obs.copy()
cm_celltype = dict(zip(t7_ad.obs[ct_key].cat.categories ,t7_ad.uns[f'{ct_key}_colors']))

proint_columns = ['p_int_%s'%t for t in full_timepoints] 
probs_columns =['p_obs_%s'%t for t in full_timepoints] 
# density by cell type

p_by_celltype = obs[probs_columns+proint_columns+[ct_key]].groupby(ct_key).agg("sum")
p_celltype_melt = pd.melt(p_by_celltype.reset_index(), id_vars=[ct_key])
p_celltype_melt['data'] = p_celltype_melt['variable'].str.extract(r"p_(\w{3})_\d")
p_celltype_melt['time'] = p_celltype_melt['variable'].str.extract(r"p_\w{3}_(\d*)")

plt.figure(figsize=(9, 3), dpi=300)
ax=PINN.pl.stack_catplot(x='time', y='value', cat='data', stack=ct_key, data=p_celltype_melt , palette=cm_celltype)
ax.set_xlabel("time")
ax.set_ylabel("cell type proportion")
p_celltype_melt['log_value'] = p_celltype_melt['value'].apply(np.log)
p_celltype_melt['log_value'] = p_celltype_melt['log_value'] - p_celltype_melt['log_value'].min()

def celltype_proportion(density_key='value', timepoints=timepoints, x_lim=None):
    n_timepoint = len(timepoints)
    fig_subplot, axs = plt.subplots(1, n_timepoint, figsize=(3*n_timepoint, 6), dpi=300, sharey=True)

    for i, t in enumerate(timepoints):
        t = str(t)
        ax=axs[i]

        sns.barplot(data=p_celltype_melt.query("`time` == @t"), 
                    # color=ct_key, #palette=cm_celltype,
                    edgecolor='gray', width=0.7,
                    x = density_key, y=ct_key, hue='data', ax=ax)
        
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
        axs[i].set_title(t)
        if i!=0:
            axs[i].legend([], frameon=False)
        if x_lim is not None:
            axs[i].set_xlim(0, x_lim)

    axs[0].set_ylabel("")
    axs[n_timepoint//2].set_xlabel("cell type proportion")

celltype_proportion(density_key='value', timepoints=timepoints[timepoint_idx], x_lim=0.1)
celltype_proportion(density_key='value', timepoints=timepoints[imputed_t_idx], x_lim=0.1)


celltype_proportion(density_key='log_value', timepoints=timepoints[timepoint_idx], x_lim=None)
celltype_proportion(density_key='log_value', timepoints=timepoints[imputed_t_idx], x_lim=None)





savefig(fig_subplot, "celltype_proportion")
savefig(fig_int1, 'simulation_cell_T')
savefig(fig_int2, 'simulation')


print("===============================================================")
print("finished")
print("visualization saved to ", f'{result_dir}/{result_base}_plot')
