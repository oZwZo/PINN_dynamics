# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: PINN_torch
#     language: python
#     name: python3
# ---

# %%
import os
os.chdir("/ssd/users/Wergillius/Project/PINN_dynamics")
import sys
import numpy as np
import pandas as pd

# %%
import scanpy as sc
import palantir

# %%
import matplotlib.pyplot as plt

# %% [markdown]
# # read the cellbarcode and compute density

# %%
ad = sc.read_h5ad("data/combined_filt.h5ad")

# %%
ery_mk = pd.read_csv("data/input_pseudo_dyn_ery_mk_dpt_feb_22_with_index.csv", index_col=0)
ery_mk_ad = ad[ery_mk.index].copy()
ery_mk_ad.obs['branch'] = ery_mk.loc[ery_mk_ad.obs_names, 'pseudodynamics_branch_class'].values
ery_mk_ad

# %%
sc.pl.umap(ery_mk_ad, color='leiden')

# %%
import seaborn as sns

# %%
ery_mk_ad

# %%
fig, axs = plt.subplots(3, 9, figsize=(19,5),  gridspec_kw={'hspace':0.6 ,'wspace':0.3})

for i_t, t in enumerate(ery_mk_ad.obs['timepoint_tx_days'].unique()):
    obs_t = ery_mk_ad.obs.query("`timepoint_tx_days` == @t")

    #  fate probability density
    # sns.kdeplot(ad_t.obsm['palantir_fate_probabilities']['mk'], ax=axs[0,i_t], label='mk')
    
    sns.kdeplot(obs_t.dpt_pseudotime.values, ax=axs[0, i_t], label='ery')
    
    sns.kdeplot(obs_t.query("`branch` != 'mk'").dpt_pseudotime.values, ax=axs[1, i_t], label='ery')

    #  plt
    # sns.kdeplot(ad_t[ad_t.obsm['branch_masks']['mk']].obs['palantir_pseudotime'].values, ax=axs[1,i_t], label='mk')
    sns.kdeplot(obs_t.query("`branch` != 'ery'").dpt_pseudotime.values, ax=axs[2, i_t], label='mk')



# %%
fig, axs = plt.subplots(3, 9, figsize=(19,5), sharey=True,  gridspec_kw={'hspace':0.6 ,'wspace':0.3})

for i_t, t in enumerate(ery_mk_ad.obs['timepoint_tx_days'].unique()):
    obs_t = ery_mk_ad.obs.query("`timepoint_tx_days` == @t")

    #  fate probability density
    # sns.kdeplot(ad_t.obsm['palantir_fate_probabilities']['mk'], ax=axs[0,i_t], label='mk')
    
    sns.kdeplot(obs_t.dpt_pseudotime.values, ax=axs[0, i_t], label='ery')
    
    sns.kdeplot(obs_t.query("`branch` != 'mk'").palantir_pseudotime.values, ax=axs[1, i_t], label='ery')

    #  plt
    # sns.kdeplot(ad_t[ad_t.obsm['branch_masks']['mk']].obs['palantir_pseudotime'].values, ax=axs[1,i_t], label='mk')
    sns.kdeplot(obs_t.query("`branch` != 'ery'").palantir_pseudotime.values, ax=axs[2, i_t], label='mk')



# %% [markdown]
# # run palantir

# %%
ery_mk_ad = sc.read_h5ad("data/ery_mk.h5ad")

# %%
ery_mk_ad.uns['iroot'] = ery_mk_ad.obs['HSCscore'].argmax()

# %%
ery_mk_ad.uns['iroot']

# %%
sc.pp.neighbors(ery_mk_ad, use_rep='X_pca_harmony', n_neighbors=30)

# %%
# Run diffusion maps
dm_res = palantir.utils.run_diffusion_maps(ery_mk_ad, pca_key='X_pca_harmony',n_components=5)
ms_data = palantir.utils.determine_multiscale_space(ery_mk_ad)

# %%
dm_res['EigenVectors'].shape

# %%
ms_data.shape

# %% [markdown]
# ### magic impute expression

# %%
imputed_X = palantir.utils.run_magic_imputation(ery_mk_ad)
palantir.plot.plot_diffusion_components(ery_mk_ad)

# %%
print(ery_mk_ad.obsm['DM_EigenVectors'].shape)
print(ery_mk_ad.obsm['DM_EigenVectors_multiscaled'].shape)

# %%
ery_mk = pd.read_csv("data/input_pseudo_dyn_ery_mk_dpt_feb_22_with_index.csv", index_col=0)

# terminal mk
mk_terminal_idx = ery_mk.query('`pseudodynamics_branch_class` == "mk"').dpt_pseudotime.argmax()
mk_terminal_cb = ery_mk.query('`pseudodynamics_branch_class` == "mk"').iloc[mk_terminal_idx].name
print(mk_terminal_cb)

# terminal ery
ery_terminal_idx = ery_mk.query('`pseudodynamics_branch_class` == "ery"').dpt_pseudotime.argmax()
ery_terminal_cb = ery_mk.query('`pseudodynamics_branch_class` == "ery"').iloc[ery_terminal_idx].name
print(ery_terminal_cb)

# root
root_idx = ery_mk_ad.uns['iroot']
root_cb = ery_mk_ad.obs_names[root_idx]
print(root_cb)

# %%
all_states = pd.Series(
    ["mk", "ery", "root"],
    index=[mk_terminal_cb, ery_terminal_cb, root_cb],
)

terminal_states = pd.Series(
    ["mk", "ery"],
    index=[mk_terminal_cb, ery_terminal_cb],
)

# %%
palantir.plot.highlight_cells_on_umap(ery_mk_ad, all_states)
plt.show()

# %%
start_cell = root_cb
pr_res = palantir.core.run_palantir(
    ery_mk_ad, start_cell, num_waypoints=500, terminal_states=terminal_states
)

# %%
palantir.plot.plot_palantir_results(ery_mk_ad, s=3)
plt.show()

# %%
sc.pl.umap(ery_mk_ad, color=['palantir_pseudotime'], )

# %%
ery_mk_ad.uns['pop']

# %%
ery_mk_ad.write_h5ad('data/ery_mk_Aug1.h5ad')

# %% [markdown]
# # branch

# %%
masks = palantir.presults.select_branch_cells(ery_mk_ad, q=.02, eps=.02)

# %%
palantir.plot.plot_branch_selection(ery_mk_ad)
plt.show()

# %%
palantir.plot.plot_trajectory(ery_mk_ad, "ery")

# %%
palantir.plot.plot_trajectory(ery_mk_ad, "mk")

# %% [markdown]
# # cell rank

# %%
ery_mk_ad = sc.read_h5ad('data/ery_mk_Aug1.h5ad')

# %%
ery_mk_ad

# %%
palantir.plot.plot_diffusion_components(ery_mk_ad, dm_res='DM_EigenVectors_multiscaled')
plt.show()

# %%
ery_mk_ad.obsm['DM_EigenVectors_multiscaled'].shape

# %%
print(ery_mk_ad.obsm['DM_EigenVectors_multiscaled'].std(axis=0))

# %%
print(ery_mk_ad.obsm['DM_EigenVectors'].std(axis=0))

# %%
# transition matrix
import cellrank as cr
import scvelo as scv
import PINN

# %%
ery_mk_ad

# %%
ck = cr.kernels.ConnectivityKernel(ery_mk_ad)
ck.compute_transition_matrix()

pk = cr.kernels.PseudotimeKernel(ery_mk_ad, time_key="palantir_pseudotime")
pk.compute_transition_matrix(threshold_scheme='soft')

# %%
pk.plot_projection()

combined_kernel = ck *0.8 + pk*0.2
combined_kernel.compute_transition_matrix()
combined_kernel.plot_projection()

# %%
from PINN.functions import reader_funs

# %%
T_M = pk.transition_matrix
cellstate_key = 'DM_EigenVectors'
delta_DM, neighbors = reader_funs.sample_deltax_from_transition(ery_mk_ad, T_M, xkey=cellstate_key)

cellstate_ad = PINN.tl.make_coord_adata(ery_mk_ad, cellstate_key=cellstate_key, n_dimension=7)
cellstate_ad.layers['velocity'] = delta_DM


# compute velocity graph
vkey = 'velocity'
scv.tl.velocity_graph(cellstate_ad,  vkey='velocity', xkey='cellstate', n_jobs=20)

# %%
# vis
fig_velocity = plt.figure(dpi=50, figsize=(4,4))
ax = fig_velocity.gca()
scv.pl.velocity_embedding_stream(cellstate_ad, color='anno_man', vkey=vkey, 
                                basis='umap', ax=ax, 
                                legend_loc='right', alpha=0.01,
                                title='sampling with Softthres pseudotime kernel TM ', )
                                # save=f"{result_dir}/{vkey}_velo.png")

# %%
knn_dx = []
for i in range(20):
    knn_delta_dm, neighbors = reader_funs.sample_deltax_from_knn(ery_mk_ad, T_M, xkey=cellstate_key, temperature=1, progressbar=False)
    knn_dx.append(knn_delta_dm)

knn_dx = np.stack(knn_dx)

# %%
knn_dx.shape

# %%
# compute velocity graph
vkey = 'velocity from KNN sampled idx'

cellstate_key = 'DM_EigenVectors'
cellstate_ad.layers[vkey] = knn_dx.mean(axis=0)
scv.tl.velocity_graph(cellstate_ad,  vkey=vkey, xkey='cellstate', n_jobs=20)
# vis
fig_velocity = plt.figure(dpi=50, figsize=(4,4))
ax = fig_velocity.gca()
scv.pl.velocity_embedding_stream(cellstate_ad, color='anno_man', vkey=vkey, 
                                basis='umap', ax=ax, 
                                legend_loc='right', alpha=0.01,
                                title=vkey, )
                                # save=f"{result_dir}/{vkey}_velo.png")

# %%
ery_mk_ad.obsm['delta_DM'] = knn_dx.mean(axis=0)

# %% [markdown]
# ## DM eigen vector scaled by std

# %%
DM = ery_mk_ad.obsm['DM_EigenVectors'].copy()
ery_mk_ad.obsm['DM_EigenVectors_scaled'] = DM / DM.std(axis=0)

# %%
# compute velocity graph
vkey = 'velocity from KNN sampled idx'
cellstate_key = 'DM_EigenVectors_scaled'

# sample
knn_dx_scaled = []
for i in range(20):
    knn_delta_dm, neighbors = reader_funs.sample_deltax_from_knn(ery_mk_ad, T_M, xkey=cellstate_key, temperature=1, progressbar=False)
    knn_dx_scaled.append(knn_delta_dm)

knn_dx_scaled = np.stack(knn_dx_scaled)


# 
cellstate_ad = PINN.tl.make_coord_adata(ery_mk_ad, cellstate_key=cellstate_key, n_dimension=5)
cellstate_ad.layers[vkey] = knn_dx_scaled.mean(axis=0)


# vis
scv.tl.velocity_graph(cellstate_ad,  vkey=vkey, xkey='cellstate', n_jobs=20)
fig_velocity = plt.figure(dpi=50, figsize=(4,4))
ax = fig_velocity.gca()
scv.pl.velocity_embedding_stream(cellstate_ad, color='anno_man', vkey=vkey, 
                                basis='umap', ax=ax, 
                                legend_loc='right', alpha=0.01,
                                title=vkey, )
                                # save=f"{result_dir}/{vkey}_velo.png")

# %%
ery_mk_ad.obs['timepoint_tx_days'] = ery_mk_ad.obs['timepoint_tx_days'].astype(int)

# %%
ery_mk_ad.write_h5ad('data/ery_mk_Aug1.h5ad')

# %% [markdown]
# # check density

# %%
from PINN import reader

# %%
for space in ['DM_EigenVectors', 'DM_EigenVectors_scaled', 'DM_EigenVectors_multiscaled']:

    full_DS = reader.TwoTimpepoint_AnnDS(
                                AnnData=ery_mk_ad, 
                                split = None,
                                n_dimension = None,
                                cellstate_key= space,
                                log_transform=False,
                                norm_time=True,
                                deltax_key='Delta_DM',
                                # kde_kws = {'bw_method':1},
                                batchsize=300)
                                
    PINN.pl.params_in_umap(ery_mk_ad, full_DS.u_b, param='u');

# %%
full_DS_volume = reader.TwoTimpepoint_AnnDS(
                            AnnData=ery_mk_ad, 
                            knn_volume = True, #!!!!! THIS ARG
                            split = None,
                            n_dimension = None,
                            cellstate_key= 'DM_EigenVectors_scaled',
                            log_transform=False,
                            norm_time=True,
                            deltax_key='Delta_DM',
                            batchsize=300)
                            
PINN.pl.params_in_umap(ery_mk_ad, full_DS_volume.u_b, param='volumed scaled u');
PINN.pl.params_in_umap(ery_mk_ad, np.log(full_DS_volume.u_b+1e-9), param='volumed scaled u');

# %% [markdown]
# # test KNN distance for correcting KDE Density

# %%
X =  ery_mk_ad.obsm['DM_EigenVectors_scaled'].copy()
dist = ery_mk_ad.obsp['distances'].copy()
conn = ery_mk_ad.obsp['connectivities'].copy()

# %%
dist = ery_mk_ad.obsp['distances'].copy()
conn = ery_mk_ad.obsp['connectivities'].copy()

Vol = []
DM_vol = []
DM_vol_mean = []
for i in range(dist.shape[0]):
    nz_dist_u = dist[i].data
    r1 = nz_dist_u[np.nonzero(nz_dist_u)].min()
    r2 = nz_dist_u.max()
    Vol.append(np.sqrt(r1*r2))
    
    DM_dist = np.sum((X[conn[i].indices] - X[[i]])**2,axis=1)*0.5
    min_DMdist = DM_dist[np.nonzero(DM_dist)].min()
    max_DMdist = DM_dist.max()
    DM_vol.append(np.sqrt(min_DMdist*max_DMdist))
    DM_vol_mean.append( DM_dist.mean() )

vol = np.array(Vol)
DM_vol = np.array(DM_vol)
DM_vol_mean = np.array(DM_vol_mean)

# %% [markdown]
# # KNN distance in PCA space

# %%
ery_mk_ad.obs['PCA_volume'] = vol

thres = np.quantile(Vol, 0.99)
vol_clip = np.clip(vol, 0, thres)
ery_mk_ad.obs['PCA_volume_clip'] = vol_clip
sc.pl.umap(ery_mk_ad, color=['PCA_volume','PCA_volume_clip'])

# %% [markdown]
# # KNN distance in DM space

# %%
# ery_mk_ad.obs['volume'] = vol_clip
ery_mk_ad.obs['DM_volume'] = DM_vol

thres = np.quantile(DM_vol, 0.99)
ery_mk_ad.obs['DM_volume_clip'] = np.clip(DM_vol, a_min=0, a_max=thres)
sc.pl.umap(ery_mk_ad, color=['DM_volume','DM_volume_clip'])

# %%
ery_mk_ad.obs['DM_vol_mean'] = DM_vol_mean

thres = np.quantile(DM_vol_mean, 0.99)
ery_mk_ad.obs['DM_vol_mean_clip'] = np.clip(DM_vol_mean, a_min=0, a_max=thres)
sc.pl.umap(ery_mk_ad, color=['DM_vol_mean','DM_vol_mean_clip'])

# %%
ery_mk_ad.write_h5ad('data/ery_mk_Aug1.h5ad')

# %% [markdown]
# ### Density Estimation corrected by DM mean distance 

# %%
from importlib import reload
reload(reader)

# %%
for space in ['DM_EigenVectors_scaled', 'DM_EigenVectors_multiscaled']:
    full_DS_volume = reader.TwoTimpepoint_AnnDS(
                                AnnData=ery_mk_ad, 
                                knn_volume = True, #!!!!! THIS ARG
                                split = None,
                                n_dimension = None,
                                cellstate_key= space,
                                log_transform=False,
                                norm_time=True,
                                deltax_key='Delta_DM',
                                batchsize=300)
                                
    fig, axs = PINN.pl.params_in_umap(ery_mk_ad, full_DS_volume.u_b, param=r'\n $E{Vol_{DM}}$ smoothed $u$');
    fig.suptitle(space, y=1.3)
    fig, axs = PINN.pl.params_in_umap(ery_mk_ad, np.log(full_DS_volume.u_b+1e-9), param=r'\n $E{Vol_{DM}}$ smoothed $\log u$');
    fig.suptitle(space, y=1.3)

# %% [markdown]
# ## find gene trend

# %%
ery_mk_ad

# %%
gene_trends = palantir.presults.compute_gene_trends(
    ery_mk_ad,
    expression_key="MAGIC_imputed_data",
)

# %% [markdown]
# find the differentially expressed genes

# %%
ery_mk_ad.obs.leiden.cat.categories
deg_df = sc.get.rank_genes_groups_df(ad, group=['1', '7', '8', '9', '11', '20', '0'])
deg_df.head(3)

# %%
diff_degs = deg_df.query("`pvals_adj` < 0.01 & `logfoldchanges` > 2").names.values
diff_degs = np.unique([g for g in diff_degs if g in ery_mk_ad.var_names])
print(len(diff_degs))

# %% [markdown]
# ## ery related genes

# %%
# of 
communities = palantir.presults.cluster_gene_trends(ery_mk_ad, 
    "ery", diff_degs, n_neighbors = 20)


palantir.plot.plot_gene_trend_clusters(ery_mk_ad, "ery")
plt.show()

# %% [markdown]
# ## mk related genes

# %%
# of 
communities = palantir.presults.cluster_gene_trends(ery_mk_ad, 
    "mk", diff_degs, n_neighbors = 20)


palantir.plot.plot_gene_trend_clusters(ery_mk_ad, "mk")
plt.show()

# %% [markdown]
# ### find gene with the highest correlation

# %%
from scipy.stats import pearsonr, spearmanr

# %%
from tqdm import tqdm

# %%
for gene in tqdm(diff_degs):
    pass

# %% [markdown]
# ## ery trajectory

# %%
ery_ad = ery_mk_ad[ery_mk_ad.obsm['branch_masks']['ery']]
pseudotime = ery_ad.obs.palantir_pseudotime.values.reshape(-1,1)

# %%
expression = ery_ad.layers['MAGIC_imputed_data']

ery_R = []
ery_Rou = []
i = 0

for i in tqdm(range(ery_mk_ad.shape[1])):
    ery_R.append(pearsonr(expression[:,i], pseudotime.flatten())[0])
    ery_Rou.append(spearmanr(expression[:,i], pseudotime.flatten())[0])
    i+=1
    
ery_R = np.array(ery_R)
ery_R = np.where(np.isnan((ery_R)), 0, ery_R)
ery_mk_ad.var['ery_Magic_pdt_r'] = ery_R

ery_Rou = np.array(ery_Rou)
ery_Rou = np.where(np.isnan((ery_Rou)), 0, ery_Rou)
ery_mk_ad.var['ery_Magic_pdt_spr'] = ery_Rou

# %%
expression = ery_ad.X.A

ery_R = []
i = 0
for i in tqdm(range(ery_mk_ad.shape[1])):
    ery_R.append(pearsonr(expression[:,i], pseudotime.flatten())[0])
    i+=1
    
ery_R = np.array(ery_R)
ery_R = np.where(np.isnan((ery_R)), 0, ery_R)
ery_mk_ad.var['ery_X_pdt_r'] = ery_R

# %%
ery_top_genes = ery_mk_ad.var[['symbol','ery_Magic_pdt_spr', 'ery_Magic_pdt_r', 'ery_X_pdt_r']].sort_values('ery_Magic_pdt_r', ascending=False)
ery_genes = ery_top_genes.iloc[:10].index
ery_top_genes.head()

# %%
palantir.plot.plot_trend(ery_ad, "ery", "Kcnn4", color="branch", position_layer="MAGIC_imputed_data")
palantir.plot.plot_trend(ery_ad, "ery", "Kel", color="branch", position_layer="MAGIC_imputed_data")
palantir.plot.plot_trend(ery_ad, "ery", "Gimap1", color="branch", position_layer="MAGIC_imputed_data")
plt.show()

# %%
palantir.plot.plot_trend(ery_ad, "ery", "Gm15915", color="branch", position_layer="X")
palantir.plot.plot_trend(ery_ad, "ery", "Adgrg1", color="branch", position_layer="X")
plt.show()

# %% [markdown]
# ## mk trajectory

# %%
mk_ad = ery_mk_ad[ery_mk_ad.obsm['branch_masks']['mk']]

# %%
sc.pl.umap(mk_ad, color=['palantir_pseudotime', 'dpt_pseudotime'])

# %%
expression = mk_ad.layers['MAGIC_imputed_data']
pseudotime = mk_ad.obs.palantir_pseudotime.values.reshape(-1,1)

mk_R = []
mk_Rou = []
i = 0
for i in tqdm(range(ery_mk_ad.shape[1])):
    mk_R.append(pearsonr(expression[:,i], pseudotime.flatten())[0])
    mk_Rou.append(spearmanr(expression[:,i], pseudotime.flatten())[0])
    i+=1
    
mk_R = np.array(mk_R)
mk_R = np.where(np.isnan((mk_R)), 0, mk_R)
ery_mk_ad.var['mk_Magic_pdt_r'] = mk_R

mk_Rou = np.array(mk_Rou)
mk_Rou = np.where(np.isnan((mk_Rou)), 0, mk_Rou)
ery_mk_ad.var['mk_Magic_pdt_spr'] = mk_Rou

# %%
expression = mk_ad.X.A

mk_R = []
i = 0
for i in tqdm(range(ery_mk_ad.shape[1])):
    mk_R.append(pearsonr(expression[:,i], pseudotime.flatten())[0])
    i+=1
    
mk_R = np.array(mk_R)
mk_R = np.where(np.isnan((mk_R)), 0, mk_R)
ery_mk_ad.var['mk_X_pdt_r'] = mk_R

# %%
mk_top_genes = ery_mk_ad.var[['symbol', "mk_Magic_pdt_spr",'mk_Magic_pdt_r', 'mk_X_pdt_r']].sort_values('mk_Magic_pdt_r', ascending=False)
mk_genes = mk_top_genes.iloc[:10].index
mk_top_genes.head()

# %%
palantir.plot.plot_trend(mk_ad, "mk", "Actb", color="branch", position_layer="MAGIC_imputed_data")
palantir.plot.plot_trend(mk_ad, "mk", "Pf4", color="branch", position_layer="MAGIC_imputed_data")
palantir.plot.plot_trend(mk_ad, "mk", "Gata3", color="branch", position_layer="MAGIC_imputed_data")
plt.show()

# %%
ery_mk_ad.write_h5ad("data/ery_mk.h5ad")

# %%
palantir.plot.plot_gene_trend_heatmaps(ery_mk_ad, ery_genes);

# %%
palantir.plot.plot_gene_trend_heatmaps(ery_mk_ad, mk_genes);

# %% [markdown]
# ## Density

# %% [markdown]
# Let's take `Actb` for mk and `Kcnn4` for ery

# %%
# cell state coordinates
S = ery_mk_ad[:,['Actb', 'Kcnn4']].layers['MAGIC_imputed_data']

# rescale S
S = (S - S.min(0).reshape(-1,2)) / (S.max(0) - S.min(0)).reshape(-1,2)
ery_mk_ad.obsm['Actb_Kcnn4_scaled_S'] = S

# %%
S.min(0), S.max(0)

# %%
fig, axs = plt.subplots(1,2, figsize=(7,3), dpi=80)
X_umap = ery_mk_ad.obsm['X_umap']

axs[0].scatter(X_umap[:,0], X_umap[:,1], c=S[:,0], s=10)
axs[0].set_title("Actb - MK")

axs[1].scatter(X_umap[:,0], X_umap[:,1], c=S[:,1], s=10)
axs[1].set_title("Kcnn4 - Ery")

# %%
ery_mk_ad.write_h5ad("data/ery_mk.h5ad")

# %% [markdown]
# Let visualize the density change over time

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
fig, axs = plt.subplots(1, 9, figsize=(19,1.7), dpi=300, sharey=True,  gridspec_kw={'hspace':0.6 ,'wspace':0.3})

for i_t, t in enumerate(sorted(ery_mk_ad.obs['timepoint_tx_days'].unique())):
    ad_t = ery_mk_ad[ery_mk_ad.obs.query("`timepoint_tx_days` == @t").index]
    
    sns.kdeplot(x=ad_t.obsm['Actb_Kcnn4_scaled_S'][:,0], 
                y=ad_t.obsm['Actb_Kcnn4_scaled_S'][:,1], 
                bw_method = 0.6,
                fill=True, ax=axs[i_t])

    axs[i_t].set_xlabel('')
    axs[i_t].set_title('Day %d' %(t))
    
    axs[i_t].set_ylim(-0.2,1.2)
    axs[i_t].set_xlim(-0.2,1.2)
    axs[i_t].set_xticks([0,0.5,1.0])
axs[4].set_xlabel('Actb \n(MK trajectory)')
axs[0].set_ylabel('Kcnn4 \n(Ery trajectory)')

# %%
density_fn = gaussian_kde(S.T, bw_method=0.8)

# %%
density_fn

# %% [markdown]
# # fate probability

# %%
ery_mk_ad = sc.read_h5ad("data/ery_mk.h5ad")

# %%
ery_mk_ad.obsm_keys()

# %%
ery_mk_ad.obsm['palantir_fate_probabilities']

# %% [markdown]
# ## density of fate probability, pseudotime, or magic smoothed genes

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
fig, axs = plt.subplots(2, 9, figsize=(18,4), gridspec_kw={'hspace':0.6 ,'wspace':0.3})

for i_t, t in enumerate(sorted(ery_mk_ad.obs['timepoint_tx_days'].unique())):
    ad_t = ery_mk_ad[ery_mk_ad.obs.query("`timepoint_tx_days` == @t").index]

    mk_adt = ad_t[ad_t.obsm['branch_masks']['mk']]
    ery_adt = ad_t[ad_t.obsm['branch_masks']['ery']]

    #  fate probability density
    sns.kdeplot(mk_adt.obsm['palantir_fate_probabilities']['mk'], ax=axs[0,i_t], label='mk')
    sns.kdeplot(ery_adt.obsm['palantir_fate_probabilities']['ery'], ax=axs[0,i_t], label='ery')

    #  plt
    sns.kdeplot(mk_adt.obs['palantir_pseudotime'].values, ax=axs[1,i_t], label='mk')
    sns.kdeplot(ery_adt.obs['palantir_pseudotime'].values, ax=axs[1,i_t], label='ery')


    
    axs[0,i_t].set_title('day %d' %t)
    axs[0,i_t].set_xlabel('fate_probability')
    axs[0,i_t].set_xlim(0,1)
    axs[1,i_t].set_xlabel('palantir_pseudotime')
    axs[1,i_t].set_xlim(-0.05,1.05)

axs[0,0].legend(ncol=1)

axs[0,0].set_ylabel('mk trajectory \n Density')
axs[1,0].set_ylabel('mk trajectory \n Density')


# %%
# (n_cell,) -> (n_cell,2)
pseudotime = ery_mk_ad.obs.palantir_pseudotime.values.reshape(-1,1)
pseudotime = np.broadcast_to(pseudotime, (pseudotime.shape[0],2))

# make sure the 
Fate_P = ery_mk_ad.obsm['palantir_fate_probabilities'].values

# %%
# make sure the pseudotime and fate probability has the same shape
Fate_coord = np.multiply(Fate_P**1.5, pseudotime**0.3) 
Fate_coord.min(0), Fate_coord.max(0)

# %%
# rescale to 0,1
Fate_coord = (Fate_coord - Fate_coord.min(0).reshape(-1,2)) / (Fate_coord.max(0) - Fate_coord.min(0)).reshape(-1,2)
Fate_coord.min(0), Fate_coord.max(0)

# %%
fig, axs = plt.subplots(1,2, figsize=(7,3), dpi=80)
X_umap = ery_mk_ad.obsm['X_umap']

coords = Fate_coord #ery_mk_ad.obsm['Fate_scaled_dpt']

axs[0].scatter(X_umap[:,0], X_umap[:,1], c=coords[:,0], s=10)
axs[0].set_title("Fate dpt - MK")

axs[1].scatter(X_umap[:,0], X_umap[:,1], c=coords[:,1], s=10)
axs[1].set_title("Kcnn4 dpt - Ery")

# %%
ery_mk_ad.obsm['Fate_scaled_dpt'] = Fate_coord

# %%
fig, axs = plt.subplots(1, 9, figsize=(19,1.7), dpi=300, sharey=True,  gridspec_kw={'hspace':0.6 ,'wspace':0.3})

for i_t, t in enumerate(sorted(ery_mk_ad.obs['timepoint_tx_days'].unique())):
    ad_t = ery_mk_ad[ery_mk_ad.obs.query("`timepoint_tx_days` == @t").index]
    
    sns.kdeplot(x=ad_t.obsm['Fate_scaled_dpt'][:,0], 
                y=ad_t.obsm['Fate_scaled_dpt'][:,1], 
                bw_method = 0.7,
                fill=True, ax=axs[i_t])

    axs[i_t].set_xlabel('')
    axs[i_t].set_title('Day %d' %(t))
    
    axs[i_t].set_ylim(-0.2,1.2)
    axs[i_t].set_xlim(-0.2,1.2)
    axs[i_t].set_xticks([0,0.5,1.0])
    
axs[4].set_xlabel('MK trajectory')
axs[0].set_ylabel('Ery trajectory)')

# %%
fig, axs = plt.subplots(1,2, figsize=(7,3), dpi=80)
X_umap = ery_mk_ad.obsm['X_umap']

coords = ery_mk_ad.obsm['palantir_fate_probabilities'].values

axs[0].scatter(X_umap[:,0], X_umap[:,1], c=coords[:,0], s=10)
axs[0].set_title("Fate - MK")

axs[1].scatter(X_umap[:,0], X_umap[:,1], c=coords[:,1], s=10)
axs[1].set_title("Fate - Ery")

# %%
ery_mk_ad.write_h5ad("data/ery_mk.h5ad")

# %%
