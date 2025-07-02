from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

def startVsEndTest(gam_fit, time_range=None):
    """
    Compares expression at early vs late pseudotime points.

    Parameters:
        gam_fit (dict): Output from fitGAM
        time_range (tuple or None): Range of pseudotime to consider; full range if None

    Returns:
        pd.DataFrame: Results of comparison
    """
    results = []
    pseudotime = gam_fit['pseudotime']
    models = gam_fit['models']

    if time_range is not None:
        mask = (pseudotime >= time_range[0]) & (pseudotime <= time_range[1])
        pseudotime = pseudotime[mask]

    for gene_idx in models:
        pred = models[gene_idx]['model'].predict()
        # Take first and last 10% of pseudotime
        cutoff = np.quantile(pseudotime, [0.1, 0.9])
        early = pseudotime < cutoff[0]
        late = pseudotime > cutoff[1]
        t_stat, pval = ttest_ind(pred[early], pred[late])
        results.append({
            'gene': gene_idx,
            't_stat': t_stat,
            'pvalue': pval
        })

    return pd.DataFrame(results).set_index('gene')


