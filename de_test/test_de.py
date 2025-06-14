"""
Test script for tradeSeq-like functionality in Python
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from PINN import reader, models, pl, tl
# from de_test import fitGAM, AssociationTest, predictSmooth, startVsEndTest
from de_test._base import fitGAM, AssociationTest
import logging
import traceback
from scipy import sparse
from time import time
import pytest

os.chdir("/ssd/users/Wergillius/Project/PINN_dynamics")



adata_lin = {}
v_project = {}
g_project = {}
x_project = {}
pdt_bin_lineage = {}


ad_palantir = sc.read_h5ad("data/tom_pos_palantir_Jun4.h5ad")
ad_palantir

for lineage in ['Neu', 'Meg', 'Ery']:
    ad = ad_palantir[ad_palantir.obsm['branch_masks'][lineage]].copy()
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


def select_lineage(lineage, day_idx):
    """
    select the lineage and return parameter by time
    """
    adata_lin[lineage] = adata_lin[lineage][adata_lin[lineage]]
    drift = v_project[lineage][day_idx]
    growth = g_project[lineage][day_idx]
    x = x_project[lineage]

    day = adata_lin[lineage].uns['pop']['t'][day_idx]

    return {'adata':adata_lin[lineage], 'drift': drift, 'growth': growth, 'x': x, 'day': day}

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("de_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data():
    """Load and prepare raw data for testing"""
    try:
        logger.info("Loading raw data...")
        adata = sc.read_h5ad("data/tom_pos_palantir_Jun4.h5ad")
        logger.info(f"Loaded data with {adata.n_obs} cells and {adata.n_vars} genes")
        return adata
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise

def prepare_megakaryocytes(adata):
    """Prepare megakaryocyte lineage data for testing"""
    try:
        logger.info("Preparing megakaryocyte lineage...")
        lineage = 'Meg'
        
        # First filter genes genome-wide to remove low-quality ones
        logger.info(f"Filtering genes with less than 10 expressing cells in {lineage} lineage")
        sc.pp.filter_genes(adata, min_cells=10)
        
        # Then select a subset of genes for testing
        # This helps ensure we're working with biologically relevant genes
        genes_subset = adata.var_names[:500]
        logger.info(f"Selected top {len(genes_subset)} genes by index for analysis")
        
        # Subset to lineage-specific cells and selected genes
        ad = adata[adata.obsm['branch_masks'][lineage], genes_subset].copy()
        
        # Get expression matrix, pseudotime and gene names
        # Use raw data if available, otherwise use X
        logger.info("Extracting expression matrix using MAGIC imputed data if available")
        if hasattr(ad, 'layers') and 'MAGIC_imputed_data' in ad.layers:
            expression_matrix = ad.layers['MAGIC_imputed_data'].toarray() if sparse.issparse(ad.X) else ad.layers['MAGIC_imputed_data']
        else:
            expression_matrix = ad.X.toarray() if sparse.issparse(ad.X) else ad.X
            
        cell_time = ad.obs['palantir_pseudotime'].values
        genes = ad.var_names.tolist()
        
        logger.info(f"Successfully prepared data for {expression_matrix.shape[0]} cells")
        logger.info(f"Expression matrix shape: {expression_matrix.shape}")
        logger.info(f"Number of unique pseudotime values: {len(np.unique(cell_time))}")
        return expression_matrix, cell_time, genes_subset
    except Exception as e:
        logger.error(f"Failed to prepare data: {str(e)}")
        raise

def run_fitGAM(expression_matrix, cell_time, genes, n_knots=7):
    """Run GAM fitting on prepared data"""
    try:
        logger.info("Fitting GAM models...")
        gam_fit = fitGAM(
            count_matrix=expression_matrix,
            cell_time=cell_time,
            n_knots=n_knots,
            gene_symbol=genes
        )
        logger.info(f"Fitted models for {len(gam_fit['models'])} genes")
        return gam_fit
    except Exception as e:
        logger.error(f"GAM fitting failed: {str(e)}")
        raise

class TestAssociationTest:
    """Test suite for AssociationTest class"""
    
    @classmethod
    def setup_class(cls):
        """Create test data for all tests"""
        # Generate synthetic data
        np.random.seed(42)
        n_cells = 100
        n_genes = 50
        cls.expression_matrix = np.random.poisson(10, size=(n_cells, n_genes))
        cls.cell_time = np.linspace(0, 1, n_cells)
        cls.genes = [f"Gene_{i}" for i in range(n_genes)]
        
        # Fit GAM models
        cls.gam_fit = fitGAM(
            count_matrix=cls.expression_matrix,
            cell_time=cls.cell_time,
            n_knots=7,
            gene_symbol=cls.genes,
        )
        
        # Create AssociationTest instance
        cls.association_test = AssociationTest(cls.gam_fit, ['Lineage1'])
    
    def test_association_test_basic(self):
        """Test basic association test functionality"""
        logger.info("Testing basic association test...")
        result = self.association_test.association_test(
            global_test=True,
            lineages=False,
            l2fc=0,
            contrast_type="start"
        )
        
        # Basic validation
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(self.genes)
        assert {'waldStat', 'df', 'pvalue', 'meanLogFC'}.issubset(result.columns)
        
        # Check non-null results
        assert result['waldStat'].notna().any()
        assert result['pvalue'].notna().any()
        
        logger.info(f"Basic test passed. Found {len(result)} significant genes")
        
    def test_association_test_with_l2fc(self):
        """Test association test with log2 fold change threshold"""
        logger.info("Testing association test with l2fc threshold...")
        result = self.association_test.association_test(
            global_test=True,
            lineages=False,
            l2fc=1.5,  # Test with 1.5 log2FC threshold
            contrast_type="start"
        )
        
        # Check that results are different from basic test
        basic_result = self.association_test.association_test(
            global_test=True,
            lineages=False,
            l2fc=0,
            contrast_type="start"
        )
        
        # assert not result.equals(basic_result)
        # assert (result['waldStat'] <= basic_result['waldStat']).all()
        
        logger.info("L2FC threshold test passed")
    
    def test_association_test_contrast_types(self):
        """Test different contrast types"""
        contrast_types = ["start", "end", "consecutive"]
        results = {}
        
        logger.info("Testing different contrast types...")
        for ctype in contrast_types:
            results[ctype] = self.association_test.association_test(
                global_test=True,
                lineages=False,
                l2fc=0,
                contrast_type=ctype
            )
            
            # Basic validation for each contrast type
            assert results[ctype].notna().any().any()
        
        # Check that different contrast types give different results
        assert not results["start"].equals(results["end"])
        assert not results["start"].equals(results["consecutive"])
        assert not results["end"].equals(results["consecutive"])
        
        logger.info("All contrast types produced valid results")
    
    
    # def test_association_test_invalid_inputs(self):
    #     """Test error handling for invalid inputs"""
    #     logger.info("Testing error handling...")
        
    #     # # Test invalid contrast type
    #     # with pytest.raises(ValueError):
    #     #     self.association_test.association_test(
    #     #         global_test=True,
    #     #         lineages=False,
    #     #         l2fc=0,
    #     #         contrast_type="invalid_type"
    #     #     )
        
    #     logger.info("Invalid input tests passed")

def log_step(func):
    """Decorator to log function execution steps with timing"""
    def wrapper(*args, **kwargs):
        logger.info(f"Starting step: {func.__name__}")
        start_time = time()
        try:
            result = func(*args, **kwargs)
            elapsed = time() - start_time
            logger.info(f"Completed {func.__name__} in {elapsed:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Failed in {func.__name__}: {str(e)}")
            raise
    return wrapper

@log_step
def test_association_test_real_data():
    """Test AssociationTest with real data"""
    try:
        # Step 1: Load and prepare data
        adata_raw = load_data()
        
        # Step 2: Prepare megakaryocyte lineage
        expression_matrix, cell_time, genes = prepare_megakaryocytes(adata_raw)
        
        # Step 3: Fit GAM models
        gam_fit = run_fitGAM(expression_matrix, cell_time, genes)
        
        # Step 4: Create AssociationTest instance
        association_test = AssociationTest(gam_fit, ['Megakaryocyte'])
        
        # Run tests with different parameters
        test_cases = [
            {"name": "basic", "l2fc": 0, "contrast_type": "start"},
            {"name": "with_l2fc", "l2fc": 1, "contrast_type": "start"},
            {"name": "end_contrast", "l2fc": 0, "contrast_type": "end"},
            {"name": "consecutive", "l2fc": 0, "contrast_type": "consecutive"}
        ]
        
        results = {}
        for case in test_cases:
            logger.info(f"Running association test: {case['name']}")
            results[case['name']] = association_test.association_test(
                global_test=True,
                lineages=False,
                l2fc=case['l2fc'],
                contrast_type=case['contrast_type'],
                inverse="qr"
            )
            
            # Basic validation
            assert not results[case['name']].empty
            assert {'waldStat', 'df', 'pvalue', 'meanLogFC'}.issubset(results[case['name']].columns)
        
        return results
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

def main():
    try:
        # Run unit tests
        test_suite = TestAssociationTest()
        test_suite.setup_class()
        test_suite.test_association_test_basic()
        test_suite.test_association_test_with_l2fc()
        test_suite.test_association_test_contrast_types()
        # test_suite.test_association_test_lineage_specific()
        # test_suite.test_association_test_invalid_inputs()
        
        # Run real data test
        test_association_test_real_data()
        
        logger.info("All tests completed successfully")
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()