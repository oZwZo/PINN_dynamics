# %load_ext autoreload
# %autoreload 2
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

main_dir="/home/wergillius/Project/PINN_dynamics"
os.chdir(main_dir)
TM1=[0,1,2,3,4,6,8]
TM2=[0,1,2,3,5]

# CHANGE THIS !!!!!

ckpt_path = "logs/tom_pos-DM_scaled_n[0, 1, 2, 3, 4, 6, 8]/pde_params_tsense/lightning_logs/version_1/checkpoints/epoch=121-total_loss=8.91327095.ckpt"
# ckpt_path = "logs/Weinreb_clone2-DM_EigenVectors_multiscaled_n3/pde_params_tsense/lightning_logs/version_1/checkpoints/epoch=191-total_loss=3.19271231.ckpt"

if __name__ == '__main__':
    ckpt_path = sys.argv[1] if not sys.argv[1].endswith("json") else ckpt_path
    cuda = sys.argv[2] if len(sys.argv) > 2  else 'cuda:1'


n_dimension = 10 if 'tom_pos' in ckpt_path else 5
ct_key = 'anno_man'
cuda = "cuda:%d"%cuda if cuda.isdigit() else cuda


# check result dir and create
date = time.strftime("%b%d")
result_dir = "results/" + "/".join(ckpt_path.split("/")[1:3]) + f"_{date}_onbase"
result_base = re.match(r".*/(.*version_\d{1,2})/.*", ckpt_path).group(1)
plot_save = f'{result_dir}/{result_base}_plot'
tl.make_dir(os.path.join(main_dir, result_dir))
tl.make_dir(os.path.join(main_dir, plot_save))

def savefig(fig, name):
    fig.savefig(f'{plot_save}/{name}.png', transparent=True, bbox_inches='tight', dpi=200)


# load model
model_name = ckpt_path.split("/")[2].replace("_tsense","")
model_class = eval(f"models.{model_name}")

# MODEL CLASS and define model
pde_model = model_class.load_from_checkpoint(ckpt_path, map_location=cuda)
device = pde_model.device

data_name = ckpt_path.split("/")[1].split("-")[0]
adata = sc.read_h5ad(f"data/{data_name}.h5ad")
timepoints = adata.uns['pop']['t']


cellstate_key = ckpt_path.split("/")[1].split("-")[1].split("_n")[0]
timepoint_idx = eval(ckpt_path.split("/")[1].split("-")[1].split("_n")[1])
timepoint_idx = range(timepoint_idx) if isinstance(timepoint_idx, int) else timepoint_idx

# detecting pre-computed duds
duds_path = f"data/{data_name}_duds.npy"
if os.path.exists(duds_path):
    precomputed_duds = np.load(duds_path)
else:
    precomputed_duds = None

DS_full = reader.Duds_AnnDS(
                            AnnData=adata, 
                            timepoint_idx=None, 
                            precomputed_duds=precomputed_duds,
                            n_dimension = n_dimension,
                            cellstate_key=cellstate_key,  #'DM_EigenVector'
                            log_transform=False,
                            norm_time=False,
                            deltax_key="Delta_DM",
                            # base_cellstate = base_cellstate,
                            batchsize=100)
# adata
t7_ad = DS_full.adata.copy()
duds = DS_full.duds.copy()
u_b = DS_full.u_b.cpu().numpy().reshape(DS_full.T_b.shape[0], -1)
cellstate = torch.from_numpy(DS_full.cellstate).float()
timepoint_label = DS_full.popD['t']



duds_path = f"data/tom_neg_duds.npy"
if os.path.exists(duds_path):
    precomputed_duds_neg = np.load(duds_path)
else:
    precomputed_duds_neg = None

# DS_tomneg = reader.Duds_AnnDS(
#                             AnnData=sc.read_h5ad(f"data/tom_neg.h5ad"), 
#                             timepoint_idx=None, 
#                             precomputed_duds=precomputed_duds_neg,
#                             n_dimension = n_dimension,
#                             cellstate_key=cellstate_key,  #'DM_EigenVector'
#                             log_transform=False,
#                             norm_time=False,
#                             deltax_key=None,
#                             # base_cellstate = base_cellstate,
#                             batchsize=100)
# tom_neg_u_b = DS_tomneg.u_b.cpu().numpy().reshape(DS_full.T_b.shape[0], -1)
# tom_neg_cellstate = torch.from_numpy(DS_full.cellstate).float()

DM_range = (cellstate.max(axis=0).values - cellstate.min(axis=0).values).cpu().numpy()

# FORWARD 
u_pred_ls, g_pred_ay, v_pred_ay, D_pred_ay = tl.forward_get_params(pde_model, DS_full)


if "u" in dir(pde_model):
    n_timepoint =  DS_full.T_b.shape[0]
    u_pred_b = np.concatenate(u_pred_ls, axis=0).reshape(n_timepoint,-1)
    fig1, axs1 = PINN.pl.params_in_umap(t7_ad, u_b, timepoints=timepoint_label, param='u', cell_of_t=False);
    fig2, axs2 = PINN.pl.params_in_umap(t7_ad, u_pred_b, timepoints=timepoint_label, param='u forward', cell_of_t=False);


for param in ['g', 'D']:
    param_pred = locals()[f'{param}_pred_ay']
    npy_savepath = f"{result_dir}/{result_base}_{param}.npy"
    np.save(npy_savepath, param_pred)
    print(param, 'saved to ', npy_savepath)
    fig_param, axs_param = PINN.pl.params_in_umap(t7_ad, param_pred, param=param,timepoints=timepoint_label, cell_of_t=False);
    #save

    fig_param.savefig(f'{plot_save}/{param}.png', transparent=True, bbox_inches='tight', dpi=200)

nabla_v = pde_model.predict_param(DS_full, param='v');
fig_v1, axs_v = PINN.pl.params_in_umap(t7_ad, nabla_v, timepoints=timepoint_label, param=r'$\nabla v$', cell_of_t=False);


v_norm = v_pred_ay / DM_range[None,None,:]
v_norm2 = np.sqrt(np.power(v_norm,2).mean(axis=2))
fig_v2, axs_v = PINN.pl.params_in_umap(t7_ad, v_norm2,timepoints=timepoint_label, param=r'$||v||^2$', cell_of_t=False);

np.save(f"{result_dir}/{result_base}_v.npy", v_pred_ay)
np.save(f"{result_dir}/{result_base}_v_norm.npy", v_norm2)
savefig(fig_v1, "nabla_v")
savefig(fig_v2, "v_norm")
# PINN.pl.params_in_umap(t7_ad, v_norm2, param=r'$||v||^2$', cell_of_t=True);

# short-term 

u_int_all = tl.density_shortterm_simulation(pde_model, DS_full, timepoint_idx)
print(u_b.sum(axis=1))
print(u_int_all.sum(axis=1))

obs_u = np.log10(u_b + 1e-20)
siml_u = np.log10(u_int_all + 1e-20)

thres = np.quantile(siml_u, [0.05, 1])
siml_u_clipped = np.clip(siml_u, a_min=thres[0], a_max=thres[1])

fig_logobs1, axs = PINN.pl.params_in_umap(t7_ad, obs_u[timepoint_idx], timepoints=timepoints[timepoint_idx], param='\nlog10 observed density', cell_of_t=True);
fig_logint1, axs = PINN.pl.params_in_umap(t7_ad, siml_u_clipped[timepoint_idx],timepoints=timepoints[timepoint_idx], param='\nlog10 simulated density', cell_of_t=True);
savefig(fig_logobs1, "Taining_Timpoint_observed_log_density")
savefig(fig_logint1, "Taining_Timpoint_simulated_log_density")

imputed_t_idx = [i for i, t in enumerate(timepoints) if i not in timepoint_idx]
if len(imputed_t_idx) > 0:
    fig_logobs2, axs = PINN.pl.params_in_umap(t7_ad, obs_u[imputed_t_idx], timepoints=timepoints[imputed_t_idx], param='\nlog10 observed density', cell_of_t=True);
    fig_logint2, axs = PINN.pl.params_in_umap(t7_ad, siml_u_clipped[imputed_t_idx],timepoints=timepoints[imputed_t_idx], param='\nlog10 simulated density', cell_of_t=True);
    savefig(fig_logobs2, "Imputed_Timpoint_observed_log_density")
    savefig(fig_logint2, "Imputed_Timpoint_simulated_log_density")

fig_obs, axs = PINN.pl.params_in_umap(t7_ad, u_b[timepoint_idx], timepoints=timepoints[timepoint_idx], param='\nlog10 observed density', cell_of_t=True);
fig_int1, axs = PINN.pl.params_in_umap(t7_ad, u_int_all[timepoint_idx],timepoints=timepoints[timepoint_idx], param='\nlog10 simulated density', cell_of_t=True);
savefig(fig_obs, "Taining_Timpoint_observed_density")
savefig(fig_int1, "Taining_Timpoint_simulated_density")

if len(imputed_t_idx) > 0:
    fig_obs2, axs = PINN.pl.params_in_umap(t7_ad, u_b[imputed_t_idx], timepoints=timepoints[imputed_t_idx], param='\nlog10 observed density', cell_of_t=True);
    fig_int2, axs = PINN.pl.params_in_umap(t7_ad, u_int_all[imputed_t_idx],timepoints=timepoints[imputed_t_idx], param='\nlog10 simulated density', cell_of_t=True);
    savefig(fig_obs2, "Imputed_Timpoint_observed_density")
    savefig(fig_int2, "Imputed_Timpoint_simulated_density")

# long term
# start_idx = 1
# u_int_all = [u_b[start_idx]]
# chunk_size= 500
# t_list = timepoints[start_idx:] / 10
# it = start_idx

# u_t_ls = []

# for i in range(0, len(cellstate), chunk_size):

#     s0 = cellstate[i:i+chunk_size].to(device).requires_grad_()
#     tompos_u0 = torch.from_numpy(u_b[it, i:i+chunk_size]).float().to(device).requires_grad_()
#     duds_0 = torch.from_numpy(duds[it, i:i+chunk_size, :]).float().to(device).requires_grad_()
#     init_condition = (tompos_u0, s0, duds_0)

#     step_size = np.around((t1 - t0)/15, decimals=1).item() 
#     step_size = step_size if step_size > 0 else 0.05
#     step_size = min(step_size, 0.4)

#     u_t, s_t, dudt_ds = odeint(
#                     pde_model,
#                     y0 = init_condition,
#                     t = torch.tensor(t_list).type(torch.float32).to(device),
#                     atol=pde_model.ode_tol,
#                     rtol=pde_model.ode_tol,
#                     method='dopri5',
#                     adjoint_options={'norm':'seminorm'},
#                 )

#     torch.cuda.empty_cache()

#     u_int = torch.nn.functional.relu(u_t)
#     u_t_ls.append(u_int.detach().cpu().numpy())
#     del u_t, s_t
#     torch.cuda.empty_cache()

# u_int_all = np.concatenate(u_t_ls, axis=1)

# print(u_b[start_idx:].sum(axis=1))
# print(u_int_all.sum(axis=1))

# obs_u = np.log10(u_b + 1e-20)
# siml_u = np.log10(u_int_all + 1e-20)

# thres = np.quantile(siml_u, [0.05, 0.90])
# siml_u_clipped = np.clip(siml_u, a_min=thres[0], a_max=thres[1])

# fig_obs, axs = PINN.pl.params_in_umap(t7_ad, obs_u[start_idx:], timepoints=timepoints[start_idx:], param='\nlog10 observed density', cell_of_t=True);
# fig_int1, axs = PINN.pl.params_in_umap(t7_ad, siml_u_clipped, timepoints=timepoints[start_idx:], param='\nlog10 simulated density', cell_of_t=True);

# fig_obs, axs = PINN.pl.params_in_umap(t7_ad, obs_u[start_idx:], timepoints=timepoints[start_idx:], param='\nlog10 observed density', cell_of_t=False);
# fig_int2, axs = PINN.pl.params_in_umap(t7_ad, siml_u_clipped, timepoints=timepoints[start_idx:], param='u by integrat', cell_of_t=False);


from scipy.stats import pearsonr
from scipy.special import kl_div


def vis_perform(W, y_label, kind='bar'):
    sns.set_theme(style='ticks', font_scale=1.3)
    fig = plt.figure(dpi=300)
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

KLD_fig, ax = vis_perform(KLD_ls, "KLD : true v.s. predicted density", kind='scatter')
W1_fig, ax = vis_perform(W1, "W1 distance", kind='bar')
W2_fig, ax = vis_perform(W2, "W2 distance", kind='bar')

savefig(KLD_fig, "KLD")
savefig(W1_fig, "W1")
savefig(W2_fig, "W2")

# add the density into obs
cm_celltype = dict(zip(t7_ad.obs[ct_key].cat.categories ,t7_ad.uns[f'{ct_key}_colors']))

for i, d in enumerate(timepoints):
    t7_ad.obs[f'u_int_{d}'] = u_int_all[i]
    t7_ad.obs[f'u_obs_{d}'] = u_b[i]

    t7_ad.obs[f'p_int_{d}'] = u_int_all[i] / u_int_all[i].sum()
    t7_ad.obs[f'p_obs_{d}'] = u_b[i] / u_b[i].sum()

if model_name == 'pde_params':
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
        fig_flow, axs = PINN.pl.params_in_umap(t7_ad, flow, timepoints=timepoints[1:], param=key, cell_of_t=False);
        fig_flow.savefig(f'{plot_save}/{key}.png', transparent=True, bbox_inches='tight', dpi=200)

    # flow by cell typ

    for i, d in enumerate(timepoints[1:]):

        # stratified
        flow_key_obs = t7_ad.obs.groupby(ct_key).agg({f'Day{d} {key}':'sum' for key in stratified_flow})
        flow_melt_obs = flow_key_obs.reset_index().melt(id_vars=ct_key, var_name='flow')

        g1 = sns.catplot(data = flow_melt_obs, y = ct_key, hue=ct_key, x = 'value', col='flow', kind='bar', sharex=False, palette=cm_celltype)
        g1.savefig(f'{plot_save}/Day{d}_stratify_flow.png', transparent=True, bbox_inches='tight', dpi=200)

        # stratified flow fold chanage
        density_change = t7_ad.obs[f'u_int_{d}'].values - t7_ad.obs[f'u_int_{timepoints[i]}'].values
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


# Cell type proportion

obs = t7_ad.obs.copy()
cm_celltype = dict(zip(t7_ad.obs[ct_key].cat.categories ,t7_ad.uns[f'{ct_key}_colors']))

proint_columns = ['p_int_%s'%t for t in timepoints] 
probs_columns =['p_obs_%s'%t for t in timepoints] 
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

from PINN.plotting_fns import density_plot
fig_subplot_log = density_plot.celltype_proportion(p_celltype_melt, ct_key=ct_key, cm_celltype=cm_celltype, density_key='log_value', timepoints=timepoints[timepoint_idx], x_lim=None)
savefig(fig_subplot_log, "celltype_proportion_logscaled")

if len(imputed_t_idx) >0:
    fig_subplot = density_plot.celltype_proportion(p_celltype_melt, ct_key=ct_key, cm_celltype=cm_celltype, density_key='value', timepoints=timepoints[imputed_t_idx], x_lim=0.1)
    savefig(fig_subplot, "celltype_proportion")

if "u" in dir(pde_model):
    savefig(fig1, "observed_density")
    savefig(fig2, "forward_pred_density")



print("===============================================================")
print("finished")
print("visualization saved to ", f'{result_dir}/{result_base}_plot')
