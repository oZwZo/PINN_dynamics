%load_ext autoreload
%autoreload 2
import os, sys, re
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import time
import PINN
from functools import partial
from PINN import reader, models, pl, tl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from TorchDiffEqPack import odesolve
from torchdiffeq import odeint_adjoint as odeint
import mellon
import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Patch

from matplotlib.backends.backend_pdf import PdfPages

main_dir="/ssd/users/Wergillius/Project/PINN_dynamics"
os.chdir(main_dir)
TM1=[0,1,2,4]
# CHANGE THIS !!!!!

ckpt_path = "logs/log_gast_nTMAll_full/logrithmic_pde_tsense/lightning_logs/version_5/checkpoints/epoch=33-total_loss=4.70090246.ckpt"


if __name__ == '__main__':
    ckpt_path = sys.argv[1] if not sys.argv[1].endswith("json") else ckpt_path
    cuda = sys.argv[2] if len(sys.argv) > 2  else 'cuda:1'

data_name = 'gastrulation'
cellstate_key = 'DM_EigenVectors'
timepoint_key = 'timepoint_tx_days'
n_dimension = 25
ct_key = 'anno_man'
cuda = "2"
cuda = "cuda:%s"%cuda if cuda.isdigit() else cuda
fast_mode = False
subsample = False


# check result dir and create
date = time.strftime("%b%d")
result_dir = "results/" + "/".join(ckpt_path.split("/")[1:3]) + f"_{date}_onbase"
result_base = re.match(r".*/(.*version_\d{1,2})/.*", ckpt_path).group(1)
plot_save = f'{result_dir}/{result_base}_plot'

print("result will save to ", result_dir)
tl.make_dir(os.path.join(main_dir, result_dir))
tl.make_dir(os.path.join(main_dir, plot_save))

def savefig(fig, name):
    fig.savefig(f'{plot_save}/{name}.png', transparent=True, bbox_inches='tight', dpi=200)


# load model
model_name = ckpt_path.split("/")[2].replace("_tsense","")
model_class = eval(f"models.{model_name}")

# # MODEL CLASS and define model
pde_model = model_class.load_from_checkpoint(ckpt_path, map_location=cuda)
device = pde_model.device


adata = sc.read_h5ad(f"data/{data_name}.h5ad")
timepoints = adata.uns['pop']['t']


timepoint_idx = TM1

# detecting pre-computed duds
duds_path = f"data/{data_name}_mellon_dloguds.npy"
if os.path.exists(duds_path):
    precomputed_duds = np.load(duds_path)
else:
    precomputed_duds = None


# time-continous mellon
if not os.path.exists(f"data/{data_name}_mellon_timecontinuous_predictor.json"):
    X = adata.obsm[cellstate_key]
    X_times = adata.obs[timepoint_key].astype(int).apply(np.log)
    ls_time_estimate = 1.5 * np.mean(np.diff(np.log(timepoints)))
    t_est = mellon.TimeSensitiveDensityEstimator(d=2, ls_time=ls_time_estimate)
    # Fit the estimator to the data
    t_est.fit(X, X_times)
    t_est.predict.to_json(f"data/{data_name}_mellon_timecontinuous_predictor.json") 

t_pred = mellon.Predictor.from_json(f"data/{data_name}_mellon_timecontinuous_predictor.json")

# test mellon
predictors = []
for i, t in enumerate(adata.uns['pop']['t']):
    # den_fun = lambda x : np.clip(t_est.predict(x, np.full((x.shape[0],), np.log(t))), *threshold[i])
    if data_name == 'tom_pos':
        t=np.log(t)
    den_fun = partial(t_pred, time = t) 
    predictors.append(den_fun) 
predictors= np.array(predictors)



DataSet_class = reader.Duds_AnnDS_fastmode if fast_mode else reader.Duds_AnnDS
fast_mode_args = {
    "n_pseudobulk":None, "pseudobulk_key":'pseudo_bulk', "resolution":1000
}
fast_mode_args = fast_mode_args if fast_mode else {}

if subsample:
    adata_sub = sc.pp.subsample(adata, fraction=0.05, random_state=0, copy=True)
    subset_index = [cb in adata_sub.obs_names for cb in adata.obs_names]
    if precomputed_duds is not None:
        precomputed_duds = precomputed_duds[:, subset_index,:]
        adata_raw = adata.copy()
        adata = adata_sub

DS_full = DataSet_class(
                        AnnData=adata, 
                        timepoint_idx=None, 
                        precomputed_duds=precomputed_duds,
                        timepoint_key = timepoint_key,
                        n_dimension = n_dimension,
                        cellstate_key=cellstate_key,  #'DM_EigenVector'
                        log_transform=True,
                        norm_time='min_minus',
                        deltax_key="Delta_DM",
                        density_funs = predictors,
                        # base_cellstate = base_cellstate,
                        batchsize=100)
pde_model.log_transform = True

if fast_mode:
    bulk_ad = sc.AnnData(
        X = DS_full.cellstate,
        var = ["DM_%d"%i for i in range(n_dimension)],
        obs = DS_full.adata.obs['pseudo_bulk'].unique()
    )
    bulk_ad.obsm[cellstate_key] = bulk_ad.X
    bulk_ad.obsm['X_umap'] = tl.get_pseudobulk(adata, 'X_umap_paper')

    adata.obs.groupby('pseudo_bulk')

# adata
t7_ad = DS_full.adata.copy()
duds = DS_full.duds.copy()
u_b = DS_full.u_b.cpu().numpy().reshape(DS_full.T_b.shape[0], -1)
u_b_raw = np.stack([predict(DS_full.cellstate) for predict in predictors])
cellstate = torch.from_numpy(DS_full.cellstate).float()
timepoint_label = DS_full.popD['t']

PINN.pl.params_in_umap(adata, u_b, param='DS log u', cell_of_t=False)

retest_u_clip = np.stack([p(DS_full.cellstate) for p in predictors])
PINN.pl.params_in_umap(adata, retest_u_clip, cell_of_t=False)



DM_range = (cellstate.max(axis=0).values - cellstate.min(axis=0).values).cpu().numpy()
DM_range = np.where(DM_range==0, 1, DM_range)
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

out_int_all = PINN.tl.density_shortterm_simulation(pde_model, DS_full, timepoint_idx, timepoints=(timepoints-timepoints[0])/pde_model.time_scale_factor, return_all=True)
u_int_all = out_int_all[0]

u_int_all = np.concatenate([u_b[[0]], u_int_all])
print(np.exp(u_b).sum(axis=1))
print(np.exp(u_int_all).sum(axis=1))


thres = np.quantile(u_int_all, [0.05, 1])
siml_u_clipped = np.clip(u_int_all, a_min=thres[0], a_max=thres[1])
imputed_t_idx = [i for i, t in enumerate(timepoints) if i not in timepoint_idx]


fig_obs, axs = PINN.pl.params_in_umap(t7_ad, u_b[timepoint_idx], timepoints=timepoints[timepoint_idx], param='\nlog observed density', cell_of_t=True);
fig_int1, axs = PINN.pl.params_in_umap(t7_ad, u_int_all[timepoint_idx],timepoints=timepoints[timepoint_idx], param='\nlog simulated density', cell_of_t=True);
savefig(fig_obs, "Taining_Timpoint_observed_density")
savefig(fig_int1, "Taining_Timpoint_simulated_density")

fig_obs, axs = PINN.pl.params_in_umap(t7_ad, u_b, timepoints=timepoints, param='\nlog observed density', cell_of_t=False);
fig_int1, axs = PINN.pl.params_in_umap(t7_ad, u_int_all,timepoints=timepoints, param='\nlog simulated density', cell_of_t=False);

if len(imputed_t_idx) > 0:
    fig_obs2, axs = PINN.pl.params_in_umap(t7_ad, u_b[imputed_t_idx], timepoints=timepoints[imputed_t_idx], param='\nlog observed density', cell_of_t=True);
    fig_int2, axs = PINN.pl.params_in_umap(t7_ad, u_int_all[imputed_t_idx],timepoints=timepoints[imputed_t_idx], param='\nlog simulated density', cell_of_t=True);
    savefig(fig_obs2, "Imputed_Timpoint_observed_density")
    savefig(fig_int2, "Imputed_Timpoint_simulated_density")


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
