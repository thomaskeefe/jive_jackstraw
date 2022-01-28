import numpy as np
import numba
from tqdm import tqdm  # progress bar

class JIVEJackstraw:
    """
    Determines which variables have common loadings that are statistically
    significantly nonzero."
    """
    def __init__(self):
        self.results = []

    def fit(self, datablock, cns, alpha=.05, bonferroni=True):
        """Fit the JIVEJackstraw object.

        Parameters
        ----------
        datablock : (n x d) numpy array
                    Data block of interest
        cns : (n x joint_rank) numpy array
              Common normalized scores from a JIVE analysis
        alpha : numeric
                The desired level of the test
        bonferroni : boolean
                     Whether to use Bonferroni correction to ensure the total
                     level does not exceed alpha
        """

        if not (isinstance(datablock, np.ndarray) and isinstance(cns, np.ndarray)):
            raise ValueError("Only numpy arrays are supported at this time.")

        d = datablock.shape[1]
        joint_rank = cns.shape[1]

        if bonferroni:
            alpha = alpha/(d*joint_rank)

        for joint_comp_number in range(joint_rank):
            joint_comp_scores = cns[:, joint_comp_number]

            F_obs, F_null, significant, p_values = jive_jackstraw(datablock.T, joint_comp_scores, alpha)
            self.results.append({'F_obs': F_obs,
                                'F_null': F_null,
                                'significant': significant,
                                'p_values': p_values
            })

@numba.njit
def OLS_F_stat(y, x):
    """Returns the F statistic for y ~ x + 1 (i.e. an intercept is added).
    Much faster than statsmodels.OLS, but speedup from numba is only apparent
    for n < 1000.

    Parameters
    ----------
    y : 1-D numpy array
        response vector
    x : 1-D numpy array
        predictor vector
    """
    n = len(x)
    ybar = np.mean(y)
    xbar = np.mean(x)
    y_cen = y - ybar
    x_cen = x - xbar
    beta1 = np.dot(x_cen, y_cen) / np.dot(x_cen, x_cen)
    beta0 = ybar - beta1 * xbar
    yhat = beta0 + beta1 * x
    e = y - yhat
    sse1 = np.dot(e, e)
    sse0 = np.dot(y_cen, y_cen)
    F = (sse0 - sse1) / (sse1 / (n-2))
    return F

def dropnan(arr):
    "Drop nans from an array"
    return arr[~np.isnan(arr)]

def compute_p_values(observed_f_stats, null_f_stats):
    """Compute empirical p-values from the observed F statistics and
    their corresponding null statistics.

    Parameters
    ----------
    observed_f_stats : length d numpy array
                       observed F statistics, one for each feature
                       in the datablock
    null_f_stats : (d, 10) array
                   null F statistics, 10 for each original feature.
    """
    p_values = []
    for i, f_obs in enumerate(observed_f_stats):
        # if a feat is constant, the F is nan so we set empircal p-val = 1
        if np.isnan(f_obs):
            p = 1
        else:
            f_nulls = dropnan(null_f_stats[i, :])
            p = np.mean(f_obs < f_nulls)
        p_values.append(p)
    return np.array(p_values)

@numba.njit
def generate_null_f_stats(random_features, joint_comp_scores):
    """Generate null f stats for 10 random features without replacement

    Parameters
    ----------
    random_features : (10, n) numpy array
                      a 10-feature subset of the datablock of interest, features as rows
    joint_comp_scores : 1-D numpy array
                        Array of scores for a single joint component
    """
    null_f_stats = []
    for feature in random_features:
        # if the feature is constant, set F = nan so that it is left out
        # of the empircal p-value computation.
        if np.var(feature) == 0:
            null_f_stats.append(np.nan)
        else:
            permuted = np.random.permutation(feature)
            f = OLS_F_stat(permuted, joint_comp_scores)
            null_f_stats.append(f)

    return null_f_stats

def jive_jackstraw(datablock_t, joint_comp_scores, alpha):
    """Internal jackstraw function. Note: users should use the .fit()
    method of the JIVEJackstraw class instead of this function.

    Parameters
    ----------
    datablock_t : numpy array
                  datablock for which to compute loadings, transposed so that
                  columns are observations
    joint_comp_scores : 1-D numpy array
                        Array of scores for a single joint component
    alpha : numeric
            the desired level of each test
    """
    d = datablock_t.shape[0]

    observed_f_stats = []
    for feature in datablock_t:  # rows are features
        # if the feature is constant, set F = nan
        if np.var(feature) == 0:
            observed_f_stats.append(np.nan)
        else:
            f = OLS_F_stat(feature, joint_comp_scores)
            observed_f_stats.append(f)

    observed_f_stats = np.array(observed_f_stats)

    # Generate 10 null F-statistics per feature
    null_f_stats = []
    with tqdm(total=d) as progressbar:
        for i in range(d):
            random_features = datablock_t[np.random.randint(0, d, size=10), :]
            null_f_stats.extend(generate_null_f_stats(random_features, joint_comp_scores))
            progressbar.update(1)

    null_f_stats = np.array(null_f_stats).reshape(d, 10)

    p_values = compute_p_values(observed_f_stats, null_f_stats)
    significant_vars = np.where(p_values <= alpha)[0]  # np.where returns length-1 tuple

    return (observed_f_stats, null_f_stats, significant_vars, p_values)
