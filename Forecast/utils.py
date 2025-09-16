import numpy as np

def quantile_bins(x, n_bins=20):
    x = np.asarray(x)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x, qs)
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.quantile(x, [0, 0.5, 1.0])
    bin_ids = np.clip(np.digitize(x, edges, right=False) - 1, 0, len(edges) - 2)
    return edges, bin_ids

def pinball_loss(y, q, alpha):
    y = np.asarray(y)
    q = np.asarray(q)
    e = y - q
    return np.where(e >= 0, alpha * e, (alpha - 1) * e)

def mae(y, m):
    y = np.asarray(y); m = np.asarray(m)
    return np.abs(y - m)

def mse(y, mu):
    y = np.asarray(y); mu = np.asarray(mu)
    return (y - mu) ** 2

def empirical_quantile(a, alpha):
    a = np.asarray(a)
    if a.size == 0:
        return np.nan
    return np.quantile(a, alpha, method='linear')
