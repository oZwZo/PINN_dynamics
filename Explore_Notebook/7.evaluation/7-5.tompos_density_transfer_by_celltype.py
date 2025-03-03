%load_ext autoreload
%autoreload 2
import os,sys
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import PINN
from PINN import reader, models
from torchdiffeq import odeint
from TorchDiffEqPack.odesolver import odesolve
from torchdyn.core import NeuralODE
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm.auto import tqdm

os.chdir("/ssd/users/Wergillius/Project/PINN_dynamics")


# CHAGNE HERE !!
# model_ckpt = "logs/tom_pos-DM_scaled_n9/pde_params_tsense/lightning_logs/version_0/checkpoints/epoch=257-total_loss=0.76506275.ckpt"
model_ckpt = "logs/tom_pos-DM_scaled_n[0, 1, 2, 3, 4, 6, 8]/pde_params_tsense/lightning_logs/version_4/checkpoints/epoch=87-total_loss=1.34044445.ckpt"
device = 'cuda:4'


### 
# dataset input params
adata = sc.read_h5ad(f"data/tom_pos.h5ad")
TFs = pd.read_table("data/TF_list_mm.txt")['TF'].values

# sc.pp.subsample(adata, fraction=0.1)
cellstate_key = "DM_scaled"
n_dimension = 10
base_cellstate = adata.obsm[cellstate_key][:,:n_dimension].copy()

Dataset = reader.TwoTimpepoint_AnnDS(
                    AnnData=adata, 
                    timepoint_idx=9, 
                    n_dimension=10,
                    cellstate_key=cellstate_key,  #'DM_EigenVector'
                    log_transform=False,
                    norm_time=False,
                    deltax_key=None,
                    batchsize= 100)


##
# load model 
pde_model = models.pde_params.load_from_checkpoint(model_ckpt)
pde_model = pde_model.to(device)
DT = models.Density_Transfer(pde_model)


# initial cell

timepoint_tx_days = sorted(adata.obs.timepoint_tx_days.unique())
t0 = timepoint_tx_days[0]


def get_cb_of_celltype(adata, celltype, celltype_key='anno_man', n_interval=10):
    
    celltype_cbs = []
    for it,t in tqdm(enumerate(timepoint_tx_days[:-1])):
        # if it==0:
        #     torch.cuda.memory._record_memory_history()

        integrate_time = np.linspace(t/t0, timepoint_tx_days[it+1]/t0 ,n_interval+1) / pde_model.time_scale_factor

        # find cell of time
        start_cell = adata.obs.query(f"`{celltype_key}` == @celltype & `timepoint_tx_days` == @t").index
        start_cell = list(start_cell)
        celltype_cbs.append(start_cell)

    return celltype_cbs

def density_transfer_by_celltype(celltype, celltype_key='anno_man', n_interval = 10, ncell = 100):
    
    Tmaps = {}
    Tmaps_norm = {}
    S_traj_lookup = {}
    celltype_cbs = []

    for it,t in tqdm(enumerate(timepoint_tx_days[:-1])):
        # if it==0:
        #     torch.cuda.memory._record_memory_history()

        integrate_time = np.linspace(t/t0, timepoint_tx_days[it+1]/t0 ,n_interval+1) / pde_model.time_scale_factor
        print(integrate_time)

        # find cell of time
        start_cell = adata.obs.query(f"`{celltype_key}` == @celltype & `timepoint_tx_days` == @t").index
        start_cell = list(start_cell)
        celltype_cbs.append(start_cell)
        if len(start_cell) == 0:
            continue

        # define initital density and cellstates
        cell_index = [np.where(adata.obs_names == x)[0].item() for x in start_cell]
        u0 = Dataset.u_b[it, cell_index].float().to(device)
        s0 = torch.from_numpy(Dataset.cellstate[cell_index]).float().to(device)

        S_trajectory = DT.cellstate_drift(s0, integrate_time)
        Tmaps_t, Tmaps_t_norm = DT.transition_by_batch(s0, u0, integrate_time, n_interval=n_interval, ncell=ncell)

        del u0, s0
        
        S_traj_lookup[str(t)] = S_trajectory
        Tmaps_norm[str(t)] = Tmaps_t_norm
        Tmaps[str(t)] = Tmaps_t

    ## 
    # v4 , interval by pop t
    np.save(f"results/tompos_Tmap/Cellstate_traj_TM1_v4_{celltype}.npy", S_traj_lookup)
    np.save(f"results/tompos_Tmap/Tmaps_norm_TM1_v4_{celltype}.npy", Tmaps_norm)
    np.save(f"results/tompos_Tmap/Tmaps_TM1_v4_{celltype}.npy", Tmaps)

    return Tmaps, Tmaps_norm, celltype_cbs

for ct in tqdm(adata.obs.anno_man.cat.categories):  #:['Meg', 'Eos']

    if ct == 'Eos':
        continue

    # main function
    if '/' in ct:
        ct = ct.replace("/","-")

    save_dir = f"results/tompos_Tmap/{ct}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if os.path.exists(f"results/tompos_Tmap/Tmaps_norm_TM1_v4_{ct}.npy"):
        # S_traj_lookup = np.load(f"results/tompos_Tmap/Cellstate_traj_TM1_v4_{ct}.npy", allow_pickle=True).item()
        Tmaps_norm = np.load(f"results/tompos_Tmap/Tmaps_norm_TM1_v4_{ct}.npy", allow_pickle=True).item()
        Tmaps = np.load(f"results/tompos_Tmap/Tmaps_TM1_v4_{ct}.npy", allow_pickle=True).item()
        ct_cbs = get_cb_of_celltype(adata, ct)
    else:
        Tmaps, Tmap_norm , ct_cbs = density_transfer_by_celltype(ct)
    
    if np.sum([len(cb) for cb in ct_cbs]) == 0:
        continue

    # reshape and flatten transport map
    Tmap_all = np.concatenate(list(Tmaps_norm.values()), axis=0)
    Tmap_all_flat = np.transpose(Tmap_all, axes=[0,2,1]).reshape(-1,100)
    nz_id = np.where(Tmap_all_flat.sum(axis=1)!=0)[0]
    Tmap_all_flat[nz_id] = Tmap_all_flat[nz_id] / Tmap_all_flat[nz_id].sum(axis=1,keepdims=True)

    # hierarchical clustering and viz
    num_clusters = 6
    cluster_flat, reordered_indices, g = PINN.pl.truncated_clustermap(Tmap_all_flat, num_clusters, truncate_mode="level", p=3)
    g.fig.savefig(f"{save_dir}/clustermap_flat.png")

        
    # assign cluster label
    adata.obs['celltype_cluster_flat'] = 'na'
    named_cluster_flats = ['%s_c%s'%(ct,c) for c in cluster_flat]
    adata.obs.loc[np.concatenate(ct_cbs), 'celltype_cluster_flat'] = named_cluster_flats
    adata.obs['celltype_cluster_flat'] = pd.Categorical(adata.obs['celltype_cluster_flat'].values, sorted(adata.obs['celltype_cluster_flat'].unique()))
    # color
    color_palette = dict(zip(cluster_flat.astype(str), g.row_colors))
    color_palette = {'%s_c%s'%(ct,c):v for c,v in color_palette.items()}
    color_palette['na'] = (0.498,0.498,0.498)
    adata.uns['celltype_cluster_flat_colors'] = [color_palette[i] for i in  adata.obs['celltype_cluster_flat'].cat.categories]


    # viz proportion
    fig_propo = PINN.pl.obs_composition(adata[np.concatenate(ct_cbs)], 'timepoint_tx_days', 'celltype_cluster_flat')
    fig_propo.gca().legend(fontsize=23, bbox_to_anchor=(1 , 0.8))
    fig_propo
    fig_propo.savefig(f"{save_dir}/dyncluster_flat_proportion.png")

    
    # viz in umap
    fig_umap, axs = plt.subplots(1,2,figsize=(8,3), gridspec_kw={'wspace':0.35})
    sc.pl.umap(adata, size=30,color='celltype_cluster_flat', groups=['%s_c%s'%(ct,c) for c in range(1,num_clusters+1)], show=False, ax=axs[0], frameon=False)
    sc.pl.umap(adata[adata.obs.anno_man==ct], size=100,color='celltype_cluster_flat', groups=['%s_c%s'%(ct,c) for c in range(1,num_clusters+1)], show=False, ax=axs[1], frameon=False)
    fig_umap.savefig(f"{save_dir}/dyncluster_flat_umap.png")

    fig_umap2, axs = plt.subplots(1,8,figsize=(17,2), gridspec_kw={'wspace':0.35})
    for i,t in enumerate(Tmaps_norm.keys()):
        t = int(t)
        sc.pl.umap(adata[adata.obs.anno_man==ct], size=30,  show=False, ax=axs[i], frameon=False)
        sc.pl.umap(adata[adata.obs.query("`anno_man` ==@ct & `timepoint_tx_days` == @t").index], size=40,
                   color='celltype_cluster_flat', show=False,
                    ax=axs[i], frameon=False, title = f'{ct} cluster Day {t}', legend_loc='on data')


    # DEG
    adata_ct = adata[adata.obs.celltype_cluster_flat!='na'].copy()
    sc.tl.rank_genes_groups(adata_ct, groupby='celltype_cluster_flat')
    # sc.pl.rank_genes_groups_dotplot(adata_ct,  n_genes=5, groups=['HSC_c1', 'HSC_c2', 'HSC_c3','HSC_c4', 'HSC_c6'])
    sc.pl.rank_genes_groups_dotplot(
            adata_ct,
            n_genes=5,
            values_to_plot="logfoldchanges", cmap='bwr',
            min_logfoldchange=3,
            colorbar_title='log fold change'
        )
    

    groups=['HSC_c1','HSC_c4']
    sc.tl.rank_genes_groups(adata_ct, groupby='celltype_cluster_flat',  groups=['HSC_c1','HSC_c4'])

    c1_deg_df = sc.get.rank_genes_groups_df(adata_ct, group='HSC_c1')
    c1_detfs = c1_deg_df[c1_deg_df.names.isin(TFs)]
    c1_marker_TFs = c1_deg_df.query("`pvals_adj` <1e-3").sort_values("logfoldchanges", ascending=False).names[:10]
    
    c4_deg_df = sc.get.rank_genes_groups_df(adata_ct, group='HSC_c4')
    c4_detfs = c4_deg_df[c4_deg_df.names.isin(TFs)]
    c4_marker_TFs = c4_deg_df.query("`pvals_adj` <1e-3").sort_values("logfoldchanges", ascending=False).names[:10]

    var_names = {"HSC_c1": c1_marker_TFs, "HSC_c4": c4_marker_TFs}
    sc.pl.rank_genes_groups_dotplot(
        adata_ct[adata_ct.obs.query("`celltype_cluster_flat`in @groups").index],
        groups=['HSC_c1','HSC_c4'],
        var_names=var_names,
        # values_to_plot="logfoldchanges",
        # cmap='bwr',
        # colorbar_title='log fold change',
    )
