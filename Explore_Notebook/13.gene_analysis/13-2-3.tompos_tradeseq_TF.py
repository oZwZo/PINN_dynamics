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
# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages

os.chdir("/ssd/users/Wergillius/Project/PINN_dynamics")

# %%
import de_test

# %%
import decoupler as dc

# %%
adata = sc.read_h5ad("data/tom_pos.h5ad")
adata_raw = adata.raw.to_adata()

ad_palantir = sc.read_h5ad("data/tom_pos_palantir_Jun4.h5ad")
ad_palantir


# %%
impute_x_reload = np.load("data/tom_pos_raw_impute_X.npy", allow_pickle=True).item()
adata_raw.layers['MAGIC_imputed_data'] = impute_x_reload

# %%
for key in ['g', 'v_sum', 'v_norm', 'nabla_v', 'D', 'DM_EigenVectors_multiscaled', 'palantir_fate_probabilities', 'branch_masks']:
    adata_raw.obsm[key] = ad_palantir.obsm[key].copy()

for key in [ 'DM_Kernel', 'DM_Similarity', 'DM_connectivities', 'DM_distances']:
    adata_raw.obsp[key] = ad_palantir.obsp[key].copy()

adata_raw.obs['palantir_pseudotime'] = ad_palantir.obs['palantir_pseudotime'].copy()

impute_x_reload = np.load("data/tom_pos_raw_impute_X.npy", allow_pickle=True).item()
adata_raw.layers['MAGIC_imputed_data'] = impute_x_reload


# %%
#  Compute TF trends along palantir pseudotime
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
def find_steepest_region(x_values, rate_values,  min_bins=10):
    """
    Identify the pseudotime region with steepest slope
    
    Parameters:
    x_values: dict with pseudotime bins per lineage
    rate_values: dict with rate values (v or g) per lineage
    day_idx : int, the index to select which day of the paramter to use
    min_bins: minimum number of consecutive bins to consider
    
    Returns: dict with steepest region info per lineage
    """
    steepest_regions = {}
    

    # Calculate slopes using central differences
    dx = np.diff(x_values)[0]  # Assuming uniform spacing
    slopes = np.gradient(rate_values, dx)
    
    # Calculate absolute slopes for windowed analysis
    window_size = min_bins
    mean_abs_slopes = np.array([
        np.mean(np.abs(slopes[i:i+window_size]))
        for i in range(len(slopes) - window_size + 1)
    ])
    
    # Find window with maximum mean absolute slope
    max_idx = np.argmax(mean_abs_slopes)
    start_bin = x_values[max_idx]
    end_bin = x_values[max_idx + window_size - 1]
    
    steepest_regions = {
        'start_bin': start_bin,
        'end_bin': end_bin,
        'mean_slope': mean_abs_slopes[max_idx],
        'window_idx': (max_idx, max_idx + window_size)
    }
    
    return steepest_regions

def find_steepest_region(x_values, rate_values, day_idx, min_bins=10):
    """
    Identify the pseudotime region with steepest slope
    
    Parameters:
    x_values: dict with pseudotime bins per lineage
    rate_values: dict with rate values (v or g) per lineage
    day_idx : int, the index to select which day of the paramter to use
    min_bins: minimum number of consecutive bins to consider
    
    Returns: dict with steepest region info per lineage
    """
    steepest_regions = {}
    
    for lineage in x_values.keys():
        # Calculate slopes using central differences
        dx = np.diff(x_values[lineage])[0]  # Assuming uniform spacing
        slopes = np.gradient(rate_values[lineage][day_idx], dx)
        
        # Calculate absolute slopes for windowed analysis
        window_size = min_bins
        mean_abs_slopes = np.array([
            np.mean(np.abs(slopes[i:i+window_size]))
            for i in range(len(slopes) - window_size + 1)
        ])
        
        # Find window with maximum mean absolute slope
        max_idx = np.argmax(mean_abs_slopes)
        start_bin = x_values[lineage][max_idx]
        end_bin = x_values[lineage][max_idx + window_size - 1]
        
        steepest_regions[lineage] = {
            'start_bin': start_bin,
            'end_bin': end_bin,
            'mean_slope': mean_abs_slopes[max_idx],
            'window_idx': (max_idx, max_idx + window_size)
        }
    
    return steepest_regions


# %%
np.gradient(v_project['Neu'][7], np.diff(x_project["Neu"])[0]).shape

# %%
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from de_test import fitGAM, run_fitGAM_parallel


# %%
def get_input(adata_lin, lineage):
    """
    prepare input for fitGAM
    """
    ad = adata_lin[lineage]
    cell_time = ad.obs["palantir_pseudotime"].values
    if 'MAGIC_imputed_data' in ad.layers:
        expression_matrix = ad.layers['MAGIC_imputed_data'].A.copy()
    else:
        expression_matrix = ad.X.A.copy()
    
    genes = ad.var_names.values
    
    return expression_matrix, cell_time, genes


# %%
for lineage in ['Ery', 'Meg', 'Neu']:
    expression_matrix, cell_time, genes = get_input(adata_lin, lineage)

    # Run parallel GAM fitting
    # it takes around 7 minutes
    gam_fit2 = run_fitGAM_parallel(expression_matrix, cell_time, genes, n_cores=10)
    de_test.save_gamfit(gam_fit2, save_dir=f'results/tompos_{lineage}_GAM/')

# %% [markdown]
# # new Associtation Test

# %%
from importlib import reload
reload(de_test)

# %% [markdown]
# # select genes 

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
from statsmodels.stats.multitest import multipletests

def select_top_de_genes(results_df, qval_threshold=0.05):
    """
    Select top differentially expressed genes based on p-value threshold
    
    Parameters:
    results_df: DataFrame with association test results
    qval_threshold: p-value threshold for selecting DE genes
    
    Returns: list of gene indices passing threshold
    """
    # Filter significant genes
    significant = results_df[~results_df['pvalue'].isna()]
    significant['fdr'] = multipletests(significant['pvalue'], method='fdr_bh')[1]
    
    # Sort by Wald statistic
    significant = significant.query("`fdr` <= @qval_threshold").sort_values('waldStat', ascending=False)
    
    return significant

def prepare_expression_data(gam_fit, pseudotime=None):
    """
    Prepare and scale gene expression data for plotting
    
    Parameters:
    gam_fit: fitted GAM models
    pseudotime: optional custom pseudotime points
    
    Returns: DataFrame with scaled expression values
    """
    if pseudotime is None:
        # Create prediction grid
        pseudotime = gam_fit['pseudotime']
        pred_grid = np.linspace(pseudotime.min(), pseudotime.max(), 100)
        pseudotime = pred_grid
    
    
    # Process each gene
    scaler = MinMaxScaler()
    
    Expr_list = []
    gene_symbol = []
    for i, gene_idx in enumerate(gam_fit['gene_names']):
        model_info = gam_fit['models'][gene_idx]
        
        # Get prediction function
        if model_info['model']:
            pred = model_info['pred_func'](pseudotime)
            scaled_pred = scaler.fit_transform(pred.reshape(-1, 1)).flatten()
            Expr_list.append(scaled_pred)
            gene_symbol.append(gam_fit['gene_symbol'][i])
        else:
            continue
            
        # Get predictions
        
        # Initialize DataFrame
        expr_df = pd.DataFrame(np.stack(Expr_list).T, columns=gene_symbol)
        expr_df.index = pseudotime
        # Scale and add to DataFrame
        # expr_df[f'gene_{gene_idx}'] = scaler.fit_transform(predictions.reshape(-1, 1)).flatten()
    
    return expr_df

# %%
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

def calculate_mutual_information(
    parameter: np.ndarray,
    predicted_expression: np.ndarray,
    n_neighbors: int = 3,
    random_state: int = 42
) -> float:
    """
    Calculate mutual information between a parameter and predicted expression.
    
    Parameters:
        parameter (np.ndarray): Parameter values (e.g., pseudotime)
        predicted_expression (np.ndarray): Predicted expression values from GAM model
        n_neighbors (int): Number of neighbors to use for MI estimation
        random_state (int): Random state for reproducibility
        
    Returns:
        float: Mutual information value
    """
    # Ensure proper shape
    if parameter.ndim == 1:
        parameter = parameter.reshape(-1, 1)
        
    # Calculate mutual information
    mi = mutual_info_regression(
        parameter,
        predicted_expression,
        n_neighbors=n_neighbors,
        random_state=random_state
    )
    
    return float(mi[0])


# %%
mmTF = pd.read_csv("data/mouse_geneset/allTFs_mm.txt", sep='\t',  names=['gene_symbol'])

steepest_region_Day3 = find_steepest_region(x_project, v_project, 0)
steepest_region_Day7 = find_steepest_region(x_project, v_project, 1)
steepest_region_Day161 = find_steepest_region(x_project, v_project, 7)

# %%


for lineage in ['Neu', 'Meg', 'Ery']:

    print(f"\nprocessing {lineage}")
    print("loading GAM model")
    
    gam_fit = de_test.load_gamfit(f'results/tompos_{lineage}_GAM/')
    
    print("preparing and scaling gene trends")
    lineage_expr_df = prepare_expression_data(gam_fit, pseudotime=x_project[lineage])
    lineage_expr_df.to_csv(f"results/tompos_{lineage}_GAM/GAM_scaled_expression.csv", index=True)

    # lineage_expr_df = lineage_expr_df.iloc[3:-3]

    for day_id in [0, 1, 7]:

        day = adata_raw.uns['pop']['t'][day_id]
        print(f"== processing Day {day} ==")

        steepest_region = find_steepest_region(x_project, v_project, day_id)
        pseudotime_range = (steepest_region[lineage]['start_bin'], 
                        steepest_region[lineage]['end_bin'])

        # # Run association tests
        test = de_test.AssociationTest(gam_fit, ['lineage'])
        result_lin_day = test.association_test(restrcited_pseudotime=pseudotime_range)
        result_lin_day['gene_symbol'] = gam_fit['gene_symbol']

        result_lin_day.to_csv(f"results/tompos_{lineage}_GAM/Day{day}_associationtest.csv", index=False)

        # select significantly DE genes and annotate TF
        significant_genes = select_top_de_genes(result_lin_day, qval_threshold=1e-2)
        significant_genes['is_TF'] = significant_genes['gene_symbol'].isin(mmTF['gene_symbol'])
        significant_genes.set_index("gene_symbol", inplace=True)

        # compute parameter derivative
        dx = x_project[lineage][0]
        drift_derivatives = np.gradient(v_project[lineage][day_id], dx)

        # compute correlation and mutual information
        corr_list = [pearsonr(lineage_expr_df.loc[:,gene].values[3:-3], drift_derivatives[3:-3])[0] for gene in significant_genes.index]
        mi_list = [calculate_mutual_information(lineage_expr_df.loc[:,gene].values[3:-3], drift_derivatives[3:-3]) for gene in significant_genes.index]

        significant_genes['pearson_r'] = corr_list
        significant_genes['mutual_information'] = mi_list

        significant_genes.to_csv(f"results/tompos_{lineage}_GAM/Day{day}_sigDE.csv", index=True)

# %% [markdown]
# # Neu

# %%
significant_genes

# %%
sig_TFs = significant_genes.query("`meanLogFC`>1 & `mutual_information`>0.7 & `is_TF` == True" ).sort_values("mutual_information", ascending=False)
s = significant_genes.query("`meanLogFC`>2 & `pearson_r` > 0" ).sort_values("mutual_information", ascending=False)

# %%
sig_TFs

# %%
sig_genes

# %% [markdown]
# # visualization

# %%
from itertools import cycle
import seaborn as sns
import matplotlib.pyplot as plt

def plot_de_genes_with_rate(
    expr_df, 
    steepest_region, 
    lineage='Neu', 
    day_idx=0,
    v_project=v_project,  # Assuming this is available from notebook
    x_project=x_project,    # Assuming this is available from notebook
    ax=None
    ):
    """
    Plot scaled expression of DE genes with rate information
    
    Parameters:
    ------
    expr_df: DataFrame with scaled expression data
    steepest_region: dict with steepest region info
    lineage: cell lineage to plot
    day_idx: index of timepoint to show rate
    ax: optional matplotlib axes to plot on
    
    Returns: 
    ------
    matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    
    # Create color and line style combinations
    colors = sns.color_palette()  # Get current Seaborn color palette
    linestyles = ['-', '--', '-.', ':']  # Available line styles
    
    # Create all possible combinations of color and line style
    style_cycle = cycle([(color, linestyle) for linestyle in linestyles for color in colors ])
    
    # Plot gene expression profiles
    genes = expr_df.columns.to_list()
    expr_dpt = expr_df.index.to_numpy()
    
    # Store legend elements
    legend_elements = []
    
    for col in genes:
        color, linestyle = next(style_cycle)
        line = ax.plot(
            expr_dpt,
            expr_df[col],
            color=color,
            linestyle=linestyle,
            alpha=0.7,
            label=f'{col.split("_")[-1]}'
        )
        # Add to legend elements
        legend_elements.append(plt.Line2D([0], [0], color=color, linestyle=linestyle, label=f'{col.split("_")[-1]}'))
    
    # Add rate information on secondary axis
    ax2 = ax.twinx()
    
    # Get rate data
    rate_values = v_project[lineage][day_idx]
    rate_pseudotime = x_project[lineage]

    mask = np.logical_and(rate_pseudotime<=expr_dpt.max(), rate_pseudotime>=expr_dpt.min())
    
    # Plot rate
    rate_line = ax2.plot(
        rate_pseudotime[mask],
        rate_values[mask],
        color='navy',
        linewidth=2,
        linestyle='--',
        label=f'Rate (Day {day_idx})'
    )
    
    # Highlight steepest slope region
    start_bin = steepest_region[lineage]['start_bin']
    end_bin = steepest_region[lineage]['end_bin']
    
    ax.axvspan(
        start_bin, 
        end_bin, 
        alpha=0.2, 
        color='gray',
        label='Steepest Slope Region'
    )
    
    # Formatting
    ax.set_title('Scaled Expression of Top DE Genes with Rate Profile')
    ax.set_xlabel('Pseudotime')
    ax.set_ylabel('Scaled Expression')
    ax2.set_ylabel('Rate Value', color='navy')
    
    # Combine legends
    lines1 = legend_elements
    lines2 = [plt.Line2D([0], [0], color='navy', linewidth=2, linestyle='--', label=f'Rate (Day {day_idx})')]
    
    ax.legend(
        lines1 + lines2, 
        [l.get_label() for l in lines1] + [l.get_label() for l in lines2],
        loc='upper right',
        bbox_to_anchor=(1.2, 1)
    )
    
    # Style adjustments
    ax.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', colors='navy')
    
    return ax, ax2


# %%
gam_fit

# %% [markdown]
# # Neu Day 7

# %%
GAMfit_dict = {}

# %%
GAMfit_dict['Neu'] = gam_fit_Ery.copy()

# %%
sig_genes.index.to_numpy()

# %%
dpi = 60
fig, axs = plt.subplots(1,2, figsize=(12, 4), dpi=dpi, frameon=False)


sig_genes_list = sig_genes.index.to_list()
sig_gene_expr = Neu_expr_df.loc[:,sig_genes_list].iloc[6:]
ax1, ax2 = plot_de_genes_with_rate(
    sig_gene_expr, 
    steepest_region_Day7, 
    lineage='Neu', 
    day_idx=1,
    v_project=v_project,  # Assuming this is available from notebook
    x_project=x_project,
    ax = axs[0]
    )

axs[1].axis("off")
ax1.legend(bbox_to_anchor=(1.2, 0.95), loc='upper left', ncol=len(sig_genes_list)//10, frameon=False)
ax1.set_title("Neu Day7 drift related genes")

if dpi==600:
    fig.savefig("figures/tompos_tradeseq/Neu_Day7_DEGenes.pdf")

# %%
dpi = 60
fig, axs = plt.subplots(1,2, figsize=(12, 4), dpi=dpi, frameon=False)


sig_TFs_list = sig_TFs.index.to_list()
sig_gene_expr = Neu_expr_df.loc[:,sig_TFs_list].iloc[6:]
ax1, ax2 = plot_de_genes_with_rate(
    sig_gene_expr, 
    steepest_region_Day7, 
    lineage='Neu', 
    day_idx=1,
    v_project=v_project,  # Assuming this is available from notebook
    x_project=x_project,
    ax = axs[0]
    )

axs[1].axis("off")
ax1.legend(bbox_to_anchor=(1.2, 0.95), loc='upper left', ncol=len(sig_TFs_list)//10, frameon=False)
ax1.set_title("Neu Day7 drift related TFs")

if dpi==600:
    fig.savefig("figures/tompos_tradeseq/Neu_Day7_DETFs.pdf")

# %% [markdown]
# # Meg Day 161

# %%
cell_time

# %%
expression_matrix, cell_time, genes = get_input(adata_lin, 'Meg')
gam_fit = run_fitGAM_parallel(expression_matrix, cell_time, genes, n_cores=10)

Meg_expr_df = prepare_expression_data(gam_fit, pseudotime=x_project['Meg'])
Meg_expr_df.to_csv("results/tompos_Meg_GAM/GAM_scaled_expression.csv", index=True)

# %%
Meg_expr_df = prepare_expression_data(gam_fit, pseudotime=x_project['Meg'][8:])
# Meg_expr_df.to_csv("results/tompos_Meg_GAM/GAM_scaled_expression.csv", index=True)

# %%
Meg_expr_df.shape

# %%
lineage = 'Meg'
day = 161

for day_id in [0,1]:

    day = adata_raw.uns['pop']['t'][day_id]

    steepest_region = find_steepest_region(x_project, v_project, day_id, min_bins=30)
    pseudotime_range = (steepest_region[lineage]['start_bin'], 
                    steepest_region[lineage]['end_bin'])
    pseudotime_range = (0.27,0.4)
    # # Run association tests

    print(f"Running association test for {lineage} lineage...")
    test = de_test.AssociationTest(gam_fit, ['lineage'])
    result_lin_day = test.association_test(restrcited_pseudotime=pseudotime_range)
    result_lin_day['gene_symbol'] = gam_fit['gene_symbol']
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
    corr_list = [pearsonr(Meg_expr_df[gene].values[10:-3], drift_derivatives[10:-3])[0] for gene in significant_genes.index]
    mi_list = [calculate_mutual_information(Meg_expr_df.loc[:,gene].values[10:-3], drift_derivatives[10:-3]) for gene in significant_genes.index]

    significant_genes['pearson_r'] = corr_list
    significant_genes['mutual_information'] = mi_list

    significant_genes.to_csv(f"results/tompos_{lineage}_GAM/Day{day}_sigDE.csv", index=True)

# %%
significant_genes['pearson_r'].hist()

# %%
Meg_sig_genes = significant_genes.query("`meanLogFC`>2 & `fdr` <1e-3 & `pearson_r` > 0.1 & `mutual_information`>0.4" ).sort_values(["mutual_information"], ascending=False)
print(Meg_sig_genes.shape)

Meg_sig_genes

# %%
Meg_sig_TFs = significant_genes.query("`meanLogFC`>1 & `pearson_r` > 0 & `mutual_information`>0.3 & `is_TF` == True" ).sort_values("mutual_information", ascending=False)
print(Meg_sig_TFs.shape)

# %%
dpi = 60
fig, axs = plt.subplots(1,2, figsize=(12, 4), dpi=dpi, frameon=False)


sig_genes_list = Meg_sig_genes.index.to_list()
sig_gene_expr = Meg_expr_df.loc[:,sig_genes_list]#.iloc[10:]
ax1, ax2 = plot_de_genes_with_rate(
    sig_gene_expr, 
    steepest_region, 
    lineage='Meg', 
    day_idx=1,
    v_project=v_project,  # Assuming this is available from notebook
    x_project=x_project,
    ax = axs[0]
    )

axs[1].axis("off")
ax1.legend(bbox_to_anchor=(1.2, 0.95), loc='upper left', ncol=(len(sig_genes_list)//10), frameon=False)
ax1.set_title("Meg Day161 drift related genes")

if dpi==600:
    fig.savefig("figures/tompos_tradeseq/Meg_Day161_DEgenes.pdf")


# %%
dpi = 60
fig, axs = plt.subplots(1,2, figsize=(12, 4), dpi=dpi, frameon=False)


sig_TFs_list = Meg_sig_TFs.index.to_list()
sig_gene_expr = Meg_expr_df.loc[:,sig_TFs_list].iloc[10:]
ax1, ax2 = plot_de_genes_with_rate(
    sig_gene_expr, 
    steepest_region, 
    lineage='Meg', 
    day_idx=1,
    v_project=v_project,  # Assuming this is available from notebook
    x_project=x_project,
    ax = axs[0]
    )

axs[1].axis("off")
ax1.legend(bbox_to_anchor=(1.2, 0.95), loc='upper left', ncol=len(sig_TFs_list)//10, frameon=False)
ax1.set_title("Meg Day161 drift related TFs")

if dpi==600:
    fig.savefig("figures/tompos_tradeseq/Meg_Day161_DETFs.pdf")

# %% [markdown]
# # Ery

# %%
expression_matrix, cell_time, genes = get_input(adata_lin, 'Ery')
gam_fit_Ery = run_fitGAM_parallel(expression_matrix, cell_time, genes, n_cores=10)
GAMfit_dict['Ery'] = gam_fit_Ery.copy()

# Ery_expr_df = prepare_expression_data(gam_fit_Ery, pseudotime=x_project['Ery'][8:])
# Ery_expr_df.to_csv("results/tompos_Ery_GAM/GAM_scaled_expression.csv", index=True)

# %%
expression_matrix, cell_time, genes = get_input(adata_lin, 'Ery')
gam_fit_Ery = de_test.run_fitGAM_parallel(expression_matrix, cell_time, genes, n_cores=10)
GAMfit_dict['Ery'] = gam_fit_Ery.copy()

Ery_expr_df = prepare_expression_data(gam_fit_Ery, pseudotime=x_project['Ery'])
# Ery_expr_df.to_csv("results/tompos_Ery_GAM/GAM_scaled_expression.csv", index=True)

# %%
Ery_expr_df = prepare_expression_data(gam_fit_Ery, pseudotime=x_project['Ery'])

# %%
cell_time.min(), cell_time.max()

# %%
pseudotime_range

# %%
pseudotime_range = (steepest_region[lineage]['start_bin'], 
                    steepest_region[lineage]['end_bin'])

# %%
pseudotime_range

# %%
plt.plot(x_project['Ery'], v_project['Ery'][1])

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
print(f"Running association test for {lineage} lineage...")
test = de_test.AssociationTest(gam_fit_Ery, ['lineage'])
result_lin_day = test.association_test(restrcited_pseudotime=pseudotime_range)
result_lin_day['gene_symbol'] = gam_fit_Ery['gene_symbol']
# result_lin_day.to_csv(f"results/tompos_{lineage}_GAM/Day{day}_associationtest.csv", index=False)


# %%
Ery_sig_gene_df = significant_genes
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
