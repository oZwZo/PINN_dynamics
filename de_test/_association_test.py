from scipy.stats import ttest_1samp
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from warnings import catch_warnings, simplefilter
from statsmodels.genmod.families.family import NegativeBinomial
from statsmodels.gam.api import GLMGam, BSplines
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations
from typing import Literal, Optional, Union, List
import numpy as np
import pandas as pd
from scipy.linalg import qr, solve_triangular
from scipy.stats import chi2
from scipy import sparse
import logging
from tqdm.contrib.concurrent import process_map 
from functools import partial

from ._base import logger

class AssociationTest:
    """Python implementation of tradeSeq associationTest"""
    
    def __init__(self, gam_fit, lineage_names):
        self.gam_fit = gam_fit
        self.lineage_names = lineage_names
    
    def association_test(
        self,
        global_test: bool = True,
        lineages: bool = False,
        l2fc: float = 0,
        contrast_type: str = "start",
        n_points: Optional[int] = None,
        restrcited_pseudotime: Optional[List] = None,
        inverse: str = "qr"
    ) -> pd.DataFrame:
        # Extract model components
        models = self.gam_fit['models']
        pseudotime = self.gam_fit['pseudotime']
        n_genes = len(models)
        
        # Default n_points = 2 * n_knots
        if n_points is None:
            n_knots = 6  # Default knot count in fitGAM
            n_points = 2 * n_knots
        
        # Determine number of lineages
        n_lineages = 1  # Simplified for single lineage
        
        # Create contrast points
        if restrcited_pseudotime is None:
            maxT = np.max(pseudotime)
            contrast_points = np.linspace(0.01, maxT, n_points)
        else:
            assert len(restrcited_pseudotime) == 2, "please pass a list of [min_pdt, max_pdt]"
            contrast_points = np.linspace(*restrcited_pseudotime, n_points)
        
        # Initialize results storage
        results = []
        
        for gene_idx, model_info in models.items():
            model = model_info['model']
            # predict_func = model_info['pred_func']
            
            if model is None:
                # Skip genes without a valid model
                results.append({
                    'gene': gene_idx,
                    'waldStat': np.nan,
                    'df': np.nan,
                    'pvalue': np.nan,
                    'meanLogFC': np.nan
                })
                continue
                
            try:
                # Get model coefficients and covariance
                beta = model.params
                bs_fn = model.model.smoother
                if beta.ndim == 1:
                    beta = beta.reshape(-1, 1)  # Ensure column vector
                sigma = model.cov_params()
                
                # Build predictor matrix at contrast points
                # X_points = []
                # for t in contrast_points:
                #     # Create design row for this point
                #     design_row = _get_predict_custom_point_df(
                #         t, 1, n_lineages
                #     )
                #     # Predict using model
                #     X_points.append(bs_fn.transform(design_row))
                # X_points = np.vstack(X_points)
                X_points = bs_fn.transform(contrast_points)
                
                # Build contrast matrix L (n_points-1 x n_params)
                L = np.zeros((n_points - 1, X_points.shape[1]))  # Use X_points.shape[1] for correct dimensions
                
                if contrast_type == "start":
                    for j in range(n_points - 1):
                        L[j] = X_points[j + 1] - X_points[0]
                    L_reduced = L[[-1]]
                elif contrast_type == "end":
                    for j in range(n_points - 1):
                        L[j] = X_points[j] - X_points[-1]
                    L_reduced = L[[-1]]
                elif contrast_type == "consecutive":
                    for j in range(n_points - 1):
                        L[j] = X_points[j + 1] - X_points[j]
                    L_reduced = L[:]
                else:
                    raise ValueError(
                        f"Invalid contrast_type {contrast_type}. "
                        "Contrast type has to be 'start', 'end' or 'consecutive'"
                    )
                
                # Validate shapes
                logger.debug(f"beta shape: {beta.shape}")
                logger.debug(f"sigma shape: {sigma.shape}")
                logger.debug(f"L shape: {L.shape}")
                
                
                # Calculate rank from reduced L matrix
                # pivot = _linearly_independent_rows(L.T)
                

                # reshape L_reduced to match beta
                L_reduced = np.broadcast_to(L_reduced, (L_reduced.shape[0], beta.shape[0]))

                # Check dimensions before matrix operations
                if L_reduced.shape[1] != beta.shape[0]:
                    raise ValueError(
                        f"Dimension mismatch: L matrix has {L.shape[1]} columns but beta has {beta.shape[0]} rows"
                    )
                
                # Perform Wald test with proper shapes
                est_reduced = L_reduced @ beta  # Now shapes should match
                cov_contrast = L_reduced @ sigma @ L_reduced.T
                
                # Invert covariance matrix
                if inverse == "chol":
                    try:
                        # Cholesky decomposition
                        L_chol = np.linalg.cholesky(cov_contrast)
                        inv_cov = solve_triangular(
                            L_chol, 
                            solve_triangular(L_chol, np.eye(L_chol.shape[0]), lower=True, trans='T'),
                            lower=True
                        )
                    except np.linalg.LinAlgError:
                        logger.warning("Cholesky decomposition failed, using pseudo-inverse")
                        inv_cov = np.linalg.pinv(cov_contrast)
                elif inverse == "qr":
                    try:
                        # QR decomposition
                        Q_cov, R_cov = qr(cov_contrast, mode='economic')
                        inv_cov = solve_triangular(R_cov, Q_cov.T)
                    except np.linalg.LinAlgError:
                        logger.warning("QR decomposition failed, using pseudo-inverse")
                        inv_cov = np.linalg.pinv(cov_contrast)
                elif inverse == "eigen":
                    # Eigen decomposition
                    try:
                        w, v = np.linalg.eigh(cov_contrast)
                        inv_cov = v @ np.diag(1/w) @ v.T
                    except np.linalg.LinAlgError:
                        logger.warning("Eigen decomposition failed, using pseudo-inverse")
                        inv_cov = np.linalg.pinv(cov_contrast)
                else:  # "generalized" or fallback
                    inv_cov = np.linalg.pinv(cov_contrast)
                
                # Compute Wald statistic
                try:
                    wald = est_reduced.T @ inv_cov @ est_reduced
                    wald = max(0, wald[0, 0] if wald.shape[0] > 1 else wald[0])
                except Exception as e:
                    logger.error(f"Wald computation failed: {str(e)}")
                    logger.error(f"est_reduced shape: {est_reduced.shape}, inv_cov shape: {inv_cov.shape}")
                    raise
                
                # Degrees of freedom and p-value
                df = L_reduced.shape[0]
                pval = chi2.sf(wald, df)
                
                # Compute mean log fold change
                mean_log_fc = np.mean(np.abs(est_reduced))
                # if len(est_reduced) > 1:
                #     mean_log_fc = np.mean(np.abs(est_reduced))
                # else:
                #     mean_log_fc = est_reduced.item()
                
                results.append({
                    'gene': gene_idx,
                    'waldStat': wald[0],
                    'df': df,
                    'pvalue': pval[0],
                    'meanLogFC': mean_log_fc
                })
                
            except Exception as e:
                print(f"Error processing gene {gene_idx}: {str(e)}")
                results.append({
                    'gene': gene_idx,
                    'waldStat': np.nan,
                    'df': np.nan,
                    'pvalue': np.nan,
                    'meanLogFC': np.nan
                })
                
        return pd.DataFrame(results)

class PseudotimeRestrictedAssociationTest(AssociationTest):
    """Association test restricted to specific pseudotime range"""
    
    def __init__(self, gam_fit, lineage_names, pseudotime_range=None):
        super().__init__(gam_fit, lineage_names)
        self.pseudotime_range = pseudotime_range
        
    def association_test(self, *args, pseudotime_range=None, **kwargs):
        """
        Override to allow specifying pseudotime range
        
        Parameters:
        pseudotime_range: tuple (min, max) for pseudotime filtering
        """
        # Use instance-level pseudotime_range unless overridden
        range_to_use = pseudotime_range if pseudotime_range else self.pseudotime_range
        
        # Filter cells by pseudotime range
        if range_to_use:
            mask = (self.gam_fit['pseudotime'] >= range_to_use[0]) & \
                   (self.gam_fit['pseudotime'] <= range_to_use[1])
            
            # Create filtered GAM fit
            filtered_models = {}
            for gene_idx, model_info in self.gam_fit['models'].items():
                # Filter expression data
                expr = model_info['expr'][mask]
                cell_time = self.gam_fit['pseudotime'][mask]
                
                # Re-fit model on filtered data if model exists
                if model_info['model']:
                    try:
                        bs = BSplines(cell_time.reshape(-1, 1), 
                                    df=model_info['model'].model.smoother.df, 
                                    degree=3)
                        gam = GLMGam(endog=expr, smoother=bs, 
                                    alpha=1e-6, 
                                    family=NegativeBinomial(alpha=1.0))
                        
                        with catch_warnings():
                            simplefilter("ignore")
                            res = gam.fit()
                            
                        filtered_models[gene_idx] = {
                            'model': res,
                            'pred_func': interp1d(cell_time, res.predict(), 
                                                 fill_value='extrapolate'),
                            'expr': expr
                        }
                    except:
                        # Fallback to original model
                        filtered_models[gene_idx] = model_info
                else:
                    filtered_models[gene_idx] = model_info
                    
            # Create filtered gam_fit
            filtered_gam_fit = {
                'models': filtered_models,
                'gene_symbol': self.gam_fit['gene_symbol'],
                'gene_names': self.gam_fit['gene_names'],
                'pseudotime': self.gam_fit['pseudotime'][mask]
            }
            
            # Run test with filtered data
            return super().association_test(*args, **kwargs, gam_fit=filtered_gam_fit)
        
        return super().association_test(*args, **kwargs)


def _run_test_chunk(gene_indices, gam_fit, pseudotime_range):
    """
    Helper function to run association test on gene subset

    Parameters:
    gene_indices: indices of genes to process
    gam_fit: fitted GAM models
    pseudotime_range: tuple (min, max) for pseudotime filtering

    Returns: DataFrame with results for processed genes
    """
    

    # Filter genes
    filtered_models = {
        idx: gam_fit['models'][idx] for idx in gene_indices
    }

    # Create filtered gam_fit
    filtered_gam_fit = {
        'models': filtered_models,
        'gene_symbol': [gam_fit['gene_symbol'][idx] for idx in gene_indices],
        'gene_names': gene_indices,
        'pseudotime': gam_fit['pseudotime']
    }

    # Create restricted test instance
    test = AssociationTest(
        filtered_gam_fit, ['lineage']
    )

    # Run test
    results = test.association_test(restrcited_pseudotime=pseudotime_range)
    results['gene'] = gene_indices  # Keep original gene indices

    return results


def run_association_test_parallel(gam_fit, pseudotime_range, chunk_size=10, n_cores=20):
    """
    Run association tests in parallel within specified pseudotime range

    Parameters:
    gam_fit: fitted GAM models
    pseudotime_range: tuple (min, max) for pseudotime filtering
    n_cores: number of CPU cores to use

    Returns: DataFrame with results
    """
    # Split genes into chunks of one gene per task for fine-grained progress updates
    n_genes = len(gam_fit['models'])
    gene_indices = np.array_split(np.arange(n_genes), chunk_size*n_cores)  # One gene per chunk

    # Create partial function with fixed parameters
    worker_func = partial(
        _run_test_chunk,
        gam_fit=gam_fit,
        pseudotime_range=pseudotime_range
    )

    # Use tqdm's process_map for parallel execution with progress bar
    chunk_results = process_map(
        worker_func,
        gene_indices,
        max_workers=n_cores,
        chunksize=1,  # Ensures per-gene progress update
        desc="Processing genes",
        smoothing=0.05
    )

    # Combine results
    return pd.concat(chunk_results)

def Tradeseqpy_associationTest(gam_fit, l2fc=0, contrastType='end'):
    """
    code from tradeSeq-py by Weiler P

    Performs an association test to identify genes changing along pseudotime.

    Parameters:
        gam_fit (dict): Output from fitGAM
        l2fc (float): Log2 fold change threshold
        contrastType (str): Type of contrast ('end' or 'linear')

    Returns:
        pd.DataFrame: DE results
    """
    results = []
    models = gam_fit['models']
    pseudotime = gam_fit['pseudotime']

    for gene_idx in models:
        model_data = models[gene_idx]
        res = model_data['model']
        expr = model_data['expr']
        
        # Skip if no model was fitted
        if res is None:
            logFC = 0
            pval = 1.0
        else:
            pred = model_data['pred_func'](pseudotime)
            # Simple t-test between early and late cells
            mid_time = np.median(pseudotime)
            early = pseudotime < mid_time
            late = pseudotime >= mid_time
            
            # Handle empty groups
            if not np.any(early) or not np.any(late):
                logFC = 0
                pval = 1.0
            else:
                logFC = np.log2(pred[late].mean() / (pred[early].mean() + 1e-5))
                pval = ttest_1samp(logFC, 0)[1] if logFC != 0 else 1.0
                
        results.append({
            'gene': gene_idx,
            'log2FoldChange': logFC,
            'pvalue': pval
        })

    df = pd.DataFrame(results).set_index('gene')
    return df




