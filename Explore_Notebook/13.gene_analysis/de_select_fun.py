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

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from de_test import fitGAM, run_fitGAM_parallel
from itertools import cycle

from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

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



def plot_de_genes_with_rate(
    expr_df, 
    steepest_region, 
    lineage, 
    day_idx,
    v_project,  # Assuming this is available from notebook
    x_project,    # Assuming this is available from notebook
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