import numpy as np
import pandas as pd
from .utils import quantile_bins, empirical_quantile

def reliability_mean(y, yhat, n_bins=20):
    y = np.asarray(y).astype(float)
    yhat = np.asarray(yhat).astype(float)
    edges, bin_ids = quantile_bins(yhat, n_bins=n_bins)
    rows = []
    for b in range(len(edges) - 1):
        idx = (bin_ids == b)
        if np.any(idx):
            rows.append({
                'x_pred_bin': int(b),
                'x_pred_mean': float(np.mean(yhat[idx])),
                'y_empirical_mean': float(np.mean(y[idx])),
                'count': int(np.sum(idx)),
            })
    return pd.DataFrame(rows)

def reliability_proba(y_binary, p_pred, n_bins=20):
    y = np.asarray(y_binary).astype(int)
    p = np.asarray(p_pred).astype(float)
    edges, bin_ids = quantile_bins(p, n_bins=n_bins)
    rows = []
    for b in range(len(edges) - 1):
        idx = (bin_ids == b)
        if np.any(idx):
            rows.append({
                'x_pred_bin': int(b),
                'x_pred_mean': float(np.mean(p[idx])),
                'y_empirical_rate': float(np.mean(y[idx])),
                'count': int(np.sum(idx)),
            })
    return pd.DataFrame(rows)

def reliability_quantile(y, qhat, alpha, n_bins=20):
    y = np.asarray(y).astype(float)
    qhat = np.asarray(qhat).astype(float)
    edges, bin_ids = quantile_bins(qhat, n_bins=n_bins)
    rows = []
    for b in range(len(edges) - 1):
        idx = (bin_ids == b)
        if np.any(idx):
            rows.append({
                'x_pred_bin': int(b),
                'x_pred_mean': float(np.mean(qhat[idx])),
                'y_empirical_q': float(empirical_quantile(y[idx], alpha)),
                'alpha': float(alpha),
                'count': int(np.sum(idx)),
            })
    return pd.DataFrame(rows)

def reliability_surface_2d(y, pred, z, functional='mean', alpha=None, n_bins_pred=10, n_bins_z=6):
    y = np.asarray(y).astype(float)
    pred = np.asarray(pred).astype(float)
    z = np.asarray(z).astype(float)
    pred_edges, pred_bins = quantile_bins(pred, n_bins=n_bins_pred)
    z_edges, z_bins = quantile_bins(z, n_bins=n_bins_z)
    rows = []
    for bp in range(len(pred_edges) - 1):
        for bz in range(len(z_edges) - 1):
            idx = (pred_bins == bp) & (z_bins == bz)
            if not np.any(idx):
                continue
            xbar = float(np.mean(pred[idx]))
            zmid = float(np.mean(z[idx]))
            if functional == 'mean':
                t_emp = float(np.mean(y[idx]))
            elif functional == 'proba':
                t_emp = float(np.mean(y[idx]))  # y must be 0/1
            elif functional == 'quantile' and alpha is not None:
                t_emp = float(empirical_quantile(y[idx], alpha))
            else:
                raise ValueError('Unsupported functional or missing alpha.')
            rows.append({
                'bin_pred': int(bp),
                'bin_z': int(bz),
                'x_pred_mean': xbar,
                'z_mid': zmid,
                't_empirical': t_emp,
                'n': int(np.sum(idx)),
            })
    return pd.DataFrame(rows)
