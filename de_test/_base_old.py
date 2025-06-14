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



def fitGAM(count_matrix, cell_time, n_knots=7, cell_weights=None, gene_symbol=None):
    """
    Fits a GAM model per gene using pseudotime.

    Parameters:
        count_matrix (np.ndarray): Gene expression matrix (cells x genes)
        cell_time (np.ndarray): Pseudotime values for cells
        n_knots (int): Number of knots for spline fitting
        cell_weights (np.ndarray): Optional weights for each cell

    Returns:
        dict: Dictionary containing fitted models and metadata
    """
    if cell_weights is None:
        cell_weights = np.ones(len(cell_time))

    models = {}
    design_matrix = np.linspace(0, 1, len(np.unique(cell_time)))
    design_matrix = np.repeat(design_matrix[np.newaxis, :], count_matrix.shape[1], axis=0)  # shape (genes, timepoints)

    for i_gene in range(count_matrix.shape[1]):
        expr = count_matrix[:, i_gene]
        
        # Check for constant expression or insufficient data
        if np.all(expr == expr[0]) or len(np.unique(cell_time)) < 2:
            
            # interp1d is not defined, fix it
            # define interp1d
            # interp1d = lambda x, y, **kwargs: np.interp(x, x[0], y[0], **kwargs)
            pred_func = interp1d(cell_time, np.zeros_like(cell_time), fill_value='extrapolate')
            models[i_gene] = {
                'model': None,
                'design': design_matrix[i_gene],
                'pred_func': pred_func,
                'expr': expr
            }
            continue
            
        # Create B-spline basis
        try:
            bs = BSplines(cell_time.reshape(-1, 1), df=n_knots, degree=3)
        except ValueError as e:
            pred_func = interp1d(cell_time, np.zeros_like(cell_time), fill_value='extrapolate')
            models[i_gene] = {
                'model': None,
                'design': design_matrix[i_gene],
                'pred_func': pred_func,
                'expr': expr
            }
            continue
        
        # Fit GAM using GLMGam with explicit endog and smoother
        try:
            gam = GLMGam(endog=expr, smoother=bs, alpha=1e-6, family=NegativeBinomial(alpha=1.0))  # Add small regularization
            
            # Suppress perfect separation warnings
            with catch_warnings():
                simplefilter("ignore")
                res = gam.fit()
                
            if res is None:
                pred_func = interp1d(cell_time, np.zeros_like(cell_time), fill_value='extrapolate')
                models[i_gene] = {
                    'model': None,
                    'design': design_matrix[i_gene],
                    'pred_func': pred_func,
                    'expr': expr
                }
            else:
                models[i_gene] = {
                    'model': res,
                    'design': design_matrix[i_gene],
                    'pred_func': interp1d(cell_time, res.predict(), fill_value='extrapolate'),
                    'expr': expr
                }
        except:
            # Fallback to linear regression if GAM fails
            lr = LinearRegression()
            X = cell_time.reshape(-1, 1)
            lr.fit(X, expr)
            pred_func = interp1d(cell_time, lr.predict(X), fill_value='extrapolate')
            models[i_gene] = {
                'model': lr,
                'design': design_matrix[i_gene],
                'pred_func': pred_func,
                'expr': expr
            }
    return {
        'models': models,
        'gene_symbol': gene_symbol,
        'gene_names' :np.arange(count_matrix.shape[1]),
        'pseudotime': cell_time
    }

# Add a simple wrapper function for easier use
def fit_gam_simple(expr, time, n_knots=7):
    """
    Simple wrapper for GAM fitting

    Parameters:
        expr (np.ndarray): Gene expression values
        time (np.ndarray): Pseudotime values
        n_knots (int): Number of knots for spline fitting

    Returns:
        tuple: (predicted_values, model)
    """
    # Check for constant expression
    if np.all(expr == expr[0]) or len(np.unique(time)) < 2:
        return np.zeros_like(time), None
    
    # Create B-spline basis
    try:
        bs = BSplines(time.reshape(-1, 1), df=n_knots, degree=3)
    except ValueError as e:
        return np.zeros_like(time), None
    
    # Fit GAM using GLMGam with explicit endog and smoother
    gam = GLMGam(endog=expr, smoother=bs)
    
    # Suppress perfect separation warnings
    with catch_warnings():
        simplefilter("ignore")
        res = gam.fit()
        
    if res is None:
        return np.zeros_like(time), None
    else:
        return res.predict(), res


class DifferentialExpressionTest(ABC):
    """Abstract base class for a DifferentialExpressionTest."""

    def __init__(self, model):
        """Initialize DifferentialExpressionTest class.

        Parameters
        ----------
        model
            Fitted GAM class.
        """
        self._model = model

    @abstractmethod
    def __call__(self, **kwargs):
        """Perform the DifferentialExpressionTest."""


def _wald_test(
    prediction: np.ndarray,
    contrast: np.ndarray,
    sigma: np.ndarray,
    method: Literal["qr", "pinv", "inv"] = "qr"
):
    """Perform Wald test with Python-native linear algebra."""
    # Find linearly independent rows
    q, r, piv = qr(contrast.T, pivoting=True, mode='economic')
    rank = np.sum(np.abs(np.diag(r)) > 1e-10)
    if rank == 0:
        return np.nan, np.nan, np.nan
    
    # Reduce to independent rows
    piv = piv[:rank]
    contrast_reduced = contrast[piv]
    prediction_reduced = prediction[piv]
    
    # Compute covariance of contrasts
    cov_contrast = contrast_reduced @ sigma @ contrast_reduced.T
    
    # Invert based on selected method
    if method == "qr":
        q_cov, r_cov = qr(cov_contrast, mode='economic')
        try:
            inv_cov = solve_triangular(r_cov, q_cov.T, lower=False)
        except:
            inv_cov = np.linalg.pinv(cov_contrast)
    elif method == "pinv":
        inv_cov = np.linalg.pinv(cov_contrast)
    elif method == "inv":
        inv_cov = np.linalg.inv(cov_contrast)
    else:
        raise ValueError("Invalid inversion method")
    
    # Compute Wald statistic
    wald = prediction_reduced @ inv_cov @ prediction_reduced
    wald = max(0, wald)  # Ensure non-negative
    
    # Degrees of freedom and p-value
    df = rank
    pval = chi2.sf(wald, df)
    
    return wald, df, pval


def _wald_test_fc(
    beta: np.ndarray,
    sigma: np.ndarray,
    L: np.ndarray,
    l2fc: float = 0,
    inverse: Literal["qr", "chol", "eigen", "generalized"] = "qr"
):
    """Python implementation of waldTestFC from tradeSeq"""
    # Apply log fold change threshold
    log_fc_cutoff = np.log(2**l2fc) if l2fc != 0 else 0
    
    # Ensure beta is column vector
    if beta.ndim == 1:
        beta = beta.reshape(-1, 1)
        
    # Transpose L to match beta dimensions (n_params x n_contrasts)
    L = L.T
    
    # Find linearly independent rows using QR decomposition
    Q, R, P = qr(L.T, pivoting=True, mode='economic')
    rank = np.sum(np.abs(np.diag(R)) > 1e-8)
    if rank == 0:
        return np.nan, np.nan, np.nan
    
    # Reduce to independent rows
    piv = P[:rank]
    L_reduced = L[:, piv]
    
    # Compute contrast estimates
    est = L_reduced.T @ beta
    
    # Apply fold change threshold
    if l2fc != 0:
        est = np.sign(est) * np.maximum(0, np.abs(est) - log_fc_cutoff)
    
    # Compute covariance matrix
    cov_contrast = L_reduced.T @ sigma @ L_reduced
    
    # Invert covariance matrix
    if inverse == "chol":
        try:
            # Cholesky decomposition
            L_chol = np.linalg.cholesky(cov_contrast)
            inv_cov = solve_triangular(
                L_chol, 
                solve_triangular(L_chol, np.eye(rank), lower=True, trans='T'),
                lower=True
            )
        except:
            inv_cov = np.linalg.pinv(cov_contrast)
    elif inverse == "qr":
        try:
            # QR decomposition
            Q_cov, R_cov = qr(cov_contrast, mode='economic')
            inv_cov = solve_triangular(R_cov, Q_cov.T)
        except:
            inv_cov = np.linalg.pinv(cov_contrast)
    elif inverse == "eigen":
        # Eigen decomposition
        w, v = np.linalg.eigh(cov_contrast)
        inv_cov = v @ np.diag(1/w) @ v.T
    else:  # "generalized" or fallback
        inv_cov = np.linalg.pinv(cov_contrast)
    
    # Compute Wald statistic
    wald = est.T @ inv_cov @ est
    wald = max(0, wald)  # Ensure non-negative
    
    # Degrees of freedom and p-value
    df = rank
    pval = chi2.sf(wald, df)
    
    return wald, df, pval


def _get_predict_custom_point_df(
    pseudotime: float, 
    lineage_id: int,
    n_lineages: int,
    conditions: Optional[np.ndarray] = None,
    condition_id: Optional[int] = None
) -> np.ndarray:
    """Python implementation of .getPredictCustomPointDf"""
    # Create base design row (zeros)
    design_row = np.zeros(n_lineages * 2)  # t + l for each lineage
    
    # Set lineage indicator
    design_row[lineage_id - 1] = 1  # lineage indicator
    
    # Set pseudotime
    design_row[n_lineages + lineage_id - 1] = pseudotime
    
    # Handle conditions if present
    if conditions is not None and condition_id is not None:
        # Expand design for conditions
        expanded = np.zeros(n_lineages * (1 + len(np.unique(conditions))))
        # Set condition-specific lineage
        condition_offset = lineage_id * len(np.unique(conditions)) + condition_id - 1
        expanded[condition_offset] = 1
        # Set pseudotime
        expanded[len(expanded)//2 + lineage_id - 1] = pseudotime
        return expanded
    
    return design_row


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
        inverse: str = "qr"
    ) -> pd.DataFrame:
        # Extract model components
        models = self.gam_fit['models']
        pseudotime = self.gam_fit['pseudotime']
        n_genes = len(models)
        
        # Default n_points = 2 * n_knots
        if n_points is None:
            n_points = 12  # Default knot count in fitGAM is 6
        
        # Determine number of lineages
        n_lineages = 1  # Simplified for single lineage
        
        # Create contrast points
        maxT = np.max(pseudotime)
        contrast_points = np.linspace(0.01, maxT, n_points)
        
        # Initialize results storage
        results = []
        
        for gene_idx, model_info in models.items():
            model = model_info['model']
            predict_func = model_info['pred_func']
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
                if beta.ndim == 1:
                    beta = beta.reshape(-1, 1)  # Ensure column vector
                sigma = model.cov_params()
                
                # Build predictor matrix at contrast points
                X_points = []
                for t in contrast_points:
                    # Create design row for this point
                    design_row = _get_predict_custom_point_df(
                        t, 1, n_lineages
                    )
                    # Predict using model
                    X_points.append(predict_func(design_row))
                X_points = np.vstack(X_points)
                
                # Build contrast matrix L (n_points-1 x n_params)
                L = np.zeros((n_points - 1, X_points.shape[1]))
                # L_base = predict_func(contrast_points)
                
                if contrast_type == "start":
                    for j in range(1, n_points):
                        L[j-1] = X_points[j] - X_points[0]
                elif contrast_type == "end":
                    for j in range(n_points - 1):
                        L[j] = X_points[j] - X_points[-1]
                elif contrast_type == "consecutive":
                    for j in range(n_points - 1):
                        L[j] = X_points[j+1] - X_points[j]
                # if contrast_type == "start":
                #     L = L_base[1:] - L_base[0]
                # elif contrast_type == "end":
                #     L = L_base[:-1] - L_base[-1]
                # elif contrast_type == "consecutive":
                #     L = L_base[1:] - L_base[:-1]
                else:
                    raise ValueError("Invalid contrast_type")
                
                # Perform global test
                if global_test:
                    wald, df, pval = _wald_test_fc(
                        beta, sigma, L, l2fc, inverse
                    )
                else:
                    wald, df, pval = np.nan, np.nan, np.nan
                
                # Compute fold changes
                fc = L
                mean_log_fc = np.mean(np.abs(fc))
                
                results.append({
                    'gene': gene_idx,
                    'waldStat': wald,
                    'df': df,
                    'pvalue': pval,
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
        
        # Convert to DataFrame
        return pd.DataFrame(results)


class BetweenLineageTest(DifferentialExpressionTest):
    """Class for performing association tests between lineages using GAMs."""

    def __init__(self, model, lineage_names):
        super().__init__(model)
        self._model = model
        self.lineage_names = lineage_names

    def associationTest(
        self,
        pseudotimes: List[np.ndarray],
        lineages: Union[List[int], np.ndarray],
        pairwise_test: bool = False,
        global_test: bool = True,
        l2fc: float = 0,
        n_points: int = None
    ):
        """Perform association tests between lineages."""
        # If n_points not provided, default to 2 * number of knots
        if n_points is None:
            n_points = 2 * self._model.smoother.df
        
        result = defaultdict(dict)
        lineage_ids = np.concatenate([
            np.full(len(pseudotimes[i]), i) for i in lineages
        ])
        all_pseudotimes = np.concatenate(pseudotimes)
        
        for var_id in range(self._model.exog.shape[1]):
            var_name = f"Gene_{var_id}"
            try:
                # Get covariance matrix for this gene
                cov = self._model.cov_params()
                if cov.size == 0:
                    continue
                    
                # Get predictions and linear predictor matrix
                predictions = self._model.predict(exog=self._model.exog)
                lp_matrix = self._model.exog
                
                # Calculate differences between lineages
                pred_diffs = []
                lp_diffs = []
                for lineage_a, lineage_b in combinations(lineages, 2):
                    mask_a = (lineage_ids == lineage_a)
                    mask_b = (lineage_ids == lineage_b)
                    
                    pred_diff = predictions[mask_a] - predictions[mask_b]
                    lp_diff = lp_matrix[mask_a] - lp_matrix[mask_b]
                    
                    pred_diffs.append(pred_diff)
                    lp_diffs.append(lp_diff)
                
                # Apply fold change threshold
                for pred in pred_diffs:
                    log_fc_cutoff = l2fc / np.log2(np.e)
                    pred[np.abs(pred) < log_fc_cutoff] = 0
                
                # Pairwise tests
                if pairwise_test:
                    for (pred_diff, lp_diff, (lineage_a, lineage_b)) in zip(
                        pred_diffs, lp_diffs, combinations(lineages, 2)
                    ):
                        wald_stat, df, p_value = _wald_test(
                            pred_diff, lp_diff, cov
                        )
                        key = f"between {self.lineage_names[lineage_a]} and {self.lineage_names[lineage_b]}"
                        result[key][var_name] = (wald_stat, df, p_value, np.mean(pred_diff))
                
                # Global test
                if global_test and len(pred_diffs) > 0:
                    all_pred_diff = np.concatenate(pred_diffs)
                    all_lp_diff = np.vstack(lp_diffs)
                    wald_stat, df, p_value = _wald_test(all_pred_diff, all_lp_diff, cov)
                    result["globally"][var_name] = (
                        wald_stat, df, p_value, np.mean(np.concatenate(pred_diffs)))
                        
            except Exception as e:
                print(f"Error processing {var_name}: {str(e)}")
                continue
                
        return self._create_result_dataframe(result)

    def _create_result_dataframe(self, result_dict):
        dfs = []
        for test_type, gene_data in result_dict.items():
            df = pd.DataFrame.from_dict(gene_data, orient='index',
                                        columns=['wald_stat', 'df', 'p_value', 'log_fc'])
            df['test_type'] = test_type
            df['gene'] = df.index
            dfs.append(df)
        
        return pd.concat(dfs).reset_index(drop=True)

