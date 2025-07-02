# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# %load_ext autoreload
# %autoreload 2

import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # or a lower number like 2, 4, or 8

import sys, re, torch, time, PINN, palantir
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm.auto import tqdm
# PINN
from PINN import reader, models, pl, tl
import de_test
# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages

from de_select_fun import *

os.chdir("/ssd/users/Wergillius/Project/PINN_dynamics")

# %%
from importlib import reload

# %%
mmTF = pd.read_csv("data/mouse_geneset/allTFs_mm.txt", sep='\t',  names=['gene_symbol'])

adata = sc.read_h5ad("data/tom_pos.h5ad")
adata_raw = adata.raw.to_adata()

ad_palantir = sc.read_h5ad("data/tom_pos_palantir_Jun4.h5ad")
ad_palantir

impute_x_reload = np.load("data/tom_pos_raw_impute_X.npy", allow_pickle=True).item()
adata_raw.layers['MAGIC_imputed_data'] = impute_x_reload

#  Compute TF trends along palantir pseudotime
for key in ['g', 'v_sum', 'v_norm', 'nabla_v', 'D', 'DM_EigenVectors_multiscaled', 'palantir_fate_probabilities', 'branch_masks']:
    adata_raw.obsm[key] = ad_palantir.obsm[key].copy()

for key in [ 'DM_Kernel', 'DM_Similarity', 'DM_connectivities', 'DM_distances']:
    adata_raw.obsp[key] = ad_palantir.obsp[key].copy()

adata_raw.obs['palantir_pseudotime'] = ad_palantir.obs['palantir_pseudotime'].copy()

impute_x_reload = np.load("data/tom_pos_raw_impute_X.npy", allow_pickle=True).item()
adata_raw.layers['MAGIC_imputed_data'] = impute_x_reload
tf_trends = palantir.presults.compute_gene_trends(
    adata_raw,
    expression_key='MAGIC_imputed_data',  # Use the layer with TF activity
    pseudo_time_key='palantir_pseudotime',
)

# %%
adata_lin = {}
v_project = {}
g_project = {}
x_project = {}
pdt_bin_lineage = {}

for lineage in ['Neu', 'Meg', 'Ery']:
    ad = adata_raw[adata_raw.obsm['branch_masks'][lineage]].copy()
    pdt_max = ad.obs['palantir_pseudotime'].max()

    # Normalize the pseudotime
    x = np.linspace(0, pdt_max, 100)
    x_project[lineage] = (x[1:] + x[:-1])/2
    v_project[lineage] = tl.aggregate_params_by_pseudotime(ad, ad.obsm['v_norm'].T, 
                                            param_names='v', timepoints=None, 
                                            pseudotime_key='palantir_pseudotime',
                                            nbins=100, return_y=True)

    g_project[lineage] = tl.aggregate_params_by_pseudotime(ad, ad.obsm['g'].T, 
                                            param_names='g', timepoints=None, 
                                            pseudotime_key='palantir_pseudotime',
                                            nbins=100, return_y=True)

    pdt_bins = np.linspace(0, pdt_max, 21)
    pdt_bin_lineage[lineage] = pdt_bins
    pdt_label = np.round((pdt_bins[1:] + pdt_bins[:-1])/2, 2)
    ad.obs['pseudotime_bin'] = pd.cut(ad.obs['palantir_pseudotime']
                                            ,bins=20,labels = pdt_label).astype(float)
    
    adata_lin[lineage] = ad
    print(adata_lin[lineage].obs['pseudotime_bin'].unique())

# %%
# Ery

# %%
expression_matrix, cell_time, genes = get_input(adata_lin, 'Ery')
gam_fit_Ery = de_test.run_fitGAM_parallel(expression_matrix, cell_time, genes, n_cores=10)


Ery_expr_df = prepare_expression_data(gam_fit_Ery, pseudotime=x_project['Ery'])
Ery_expr_df.to_csv("results/tompos_Ery_GAM/GAM_scaled_expression.csv", index=True)

# %%
import pickle as pkl

# %%
lineage = 'Ery'

for day_id in [0,1,7]:

    day = adata_raw.uns['pop']['t'][day_id]

    steepest_region = find_steepest_region(x_project, v_project, day_id, min_bins=30)
    pseudotime_range = (0.2, 
                    steepest_region[lineage]['end_bin'])
    # # Run association tests

    print(f"Running association test for {lineage} lineage...")
    test = de_test.AssociationTest(gam_fit_Ery, ['lineage'])
    result_lin_day = test.association_test(restrcited_pseudotime=pseudotime_range)
    result_lin_day['gene_symbol'] = gam_fit_Ery['gene_symbol']
    result_lin_day.to_csv(f"results/tompos_{lineage}_GAM/Day{day}_associationtest.csv", index=False)

    # select significantly DE genes and annotate TF
    print(f"Selecting significantly DE genes for {lineage} lineage...")
    significant_genes = select_top_de_genes(result_lin_day, qval_threshold=1e-2)
    significant_genes['is_TF'] = significant_genes['gene_symbol'].isin(mmTF['gene_symbol'])
    significant_genes.set_index("gene_symbol", inplace=True)

    # compute parameter derivative
    print(f"Selecting drift derivative related genes for {lineage} lineage...")
    dx = x_project[lineage][0]
    drift_derivatives = np.gradient(v_project[lineage][day_id], dx)

    # compute correlation and mutual information
    corr_list = [pearsonr(Ery_expr_df[gene].values[:-3], drift_derivatives[8:-3])[0] for gene in significant_genes.index]
    mi_list = [calculate_mutual_information(Ery_expr_df.loc[:,gene].values[:-3], drift_derivatives[8:-3]) for gene in significant_genes.index]

    significant_genes['pearson_r'] = corr_list
    significant_genes['mutual_information'] = mi_list

    significant_genes.to_csv(f"results/tompos_{lineage}_GAM/Day{day}_sigDE.csv", index=True)

# %%
Ery_sig_gene_df = pd.read_csv(f"results/tompos_{lineage}_GAM/Day{day}_sigDE.csv", index_col=0)
print(Ery_sig_gene_df.shape)
Ery_sig_gene_df.head()

# %%
Ery_sig_gene_df[['pearson_r','mutual_information']].hist()

# %%
Ery_sig_TFs = Ery_sig_gene_df.query("`pearson_r` > 0.05 & `mutual_information`>0.4 & `is_TF` == True" ).sort_values("mutual_information", ascending=False)
print(Ery_sig_TFs.shape)

# %%
dpi = 60
fig, axs = plt.subplots(1,2, figsize=(12, 4), dpi=dpi, frameon=False)


sig_TFs_list = Ery_sig_TFs.index.to_list()
sig_gene_expr = Ery_expr_df.loc[:,sig_TFs_list].iloc[6:]
ax1, ax2 = plot_de_genes_with_rate(
    sig_gene_expr, 
    steepest_region, 
    lineage='Ery', 
    day_idx=8,
    v_project=v_project,  # Assuming this is available from notebook
    x_project=x_project,
    ax = axs[0]
    )

axs[1].axis("off")
ax1.legend(bbox_to_anchor=(1.2, 0.95), loc='upper left', ncol=len(sig_TFs_list)//10, frameon=False)
ax1.set_title("Ery Day161 drift related TFs")

if dpi==600:
    fig.savefig("figures/tompos_tradeseq/Ery_Day161_DETFs.pdf")

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
