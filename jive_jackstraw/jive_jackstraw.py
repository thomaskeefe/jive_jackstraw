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

        datablock: (n x d) numpy array of one data block
        cns: (n x joint_rank) numpy array of common normalized scores
        alpha: the desired level of the test
        bonferroni: whether to use Bonferroni correction to ensure the level is < alpha
        """

        if not (isinstance(datablock, np.ndarray) and isinstance(cns, np.ndarray)):
            raise ValueError("Only numpy arrays are supported at this time.")

        d = datablock.shape[1]
        if bonferroni:
            alpha = alpha/d

        joint_rank = cns.shape[1]

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
    """High performance simple linear regression (with intercept).
    260x faster than statsmodels.OLS, and 20x faster than using
    the non-jitted version.
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
    p_values = []
    for i, f_obs in enumerate(observed_f_stats):
        f_nulls = dropnan(null_f_stats[i, :])
        p = np.mean(f_obs < f_nulls)
        p_values.append(p)
    return np.array(p_values)

@numba.njit
def generate_null_f_stats(random_features, joint_comp_scores):
    "Generate null f stats for 10 random features without replacement"
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
    "Note, works on TRANSPOSED data, and only numpy arrays"
    d = datablock_t.shape[0]

    observed_f_stats = []
    for feature in datablock_t:  # rows are features
        # if the feature is constant, set F = -1 to make the empirical p-value 1.
        if np.var(feature) == 0:
            observed_f_stats.append(-1)
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
