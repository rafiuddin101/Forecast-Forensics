import numpy as np
from .calibrators import IsotonicCalibrator1D, QuantileCalibrator1D
from .utils import mse, mae, pinball_loss, empirical_quantile

def _ref_forecast(y, functional, alpha=None):
    if functional == 'mean':
        ref = np.mean(y)
    elif functional == 'median':
        ref = np.median(y)
    elif functional == 'quantile' and alpha is not None:
        ref = empirical_quantile(y, alpha)
    else:
        raise ValueError('Unsupported functional or missing alpha.')
    return ref

def _score(y, pred, functional, alpha=None):
    y = np.asarray(y).astype(float)
    pred = np.asarray(pred).astype(float)
    if functional == 'mean':
        return np.mean(mse(y, pred))
    elif functional == 'median':
        return np.mean(mae(y, pred))
    elif functional == 'quantile' and alpha is not None:
        return np.mean(pinball_loss(y, pred, alpha))
    else:
        raise ValueError('Unsupported functional or missing alpha.')

def decompose(y, yhat, functional='mean', alpha=None, n_bins=20):
    y = np.asarray(y).astype(float)
    yhat = np.asarray(yhat).astype(float)
    ref = _ref_forecast(y, functional, alpha)
    score_ref = _score(y, np.full_like(y, ref), functional, alpha)
    if functional in ('mean', 'median'):
        cal = IsotonicCalibrator1D(increasing=True).fit(yhat, y)
        yhat_cal = cal.predict(yhat)
    elif functional == 'quantile' and alpha is not None:
        cal = QuantileCalibrator1D(alpha=alpha, n_bins=n_bins, increasing=True).fit(yhat, y)
        yhat_cal = cal.predict(yhat)
    else:
        raise ValueError('Unsupported functional or missing alpha.')
    score_raw = _score(y, yhat, functional, alpha)
    score_cal = _score(y, yhat_cal, functional, alpha)
    UNC = score_ref
    MCB = max(0.0, score_raw - score_cal)
    DSC = max(0.0, score_ref - score_cal)
    score_recon = UNC - DSC + MCB
    return {
        'functional': functional,
        'alpha': alpha,
        'score': score_raw,
        'UNC': UNC,
        'DSC': DSC,
        'MCB': MCB,
        'score_calibrated': score_cal,
        'score_identity_close': float(abs(score_raw - score_recon)),
    }
