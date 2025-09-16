import numpy as np
from .utils import empirical_quantile
try:
    from sklearn.isotonic import IsotonicRegression
    _HAS_SK = True
except Exception:
    _HAS_SK = False

def _pav_isotonic_fit(x, y, increasing=True):
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    x_unique, idx, counts = np.unique(x, return_index=True, return_counts=True)
    y_sums = np.add.reduceat(y, idx)
    y_avg = y_sums / counts
    if increasing:
        g = y_avg.tolist()
        w = counts.astype(float).tolist()
        i = 0
        while i < len(g) - 1:
            if g[i] <= g[i + 1]:
                i += 1
            else:
                new_g = (g[i] * w[i] + g[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
                new_w = w[i] + w[i + 1]
                g[i:i + 2] = [new_g]
                w[i:i + 2] = [new_w]
                if i > 0:
                    i -= 1
        # Expand to piecewise-constant over x_unique via interpolation
        y_fit = np.interp(x_unique, x_unique, np.array(g) if len(g)==len(x_unique) else y_avg)
    else:
        xu, yf = _pav_isotonic_fit(x, -y, increasing=True)
        return xu, -yf
    return x_unique, y_fit

class IsotonicCalibrator1D:
    def __init__(self, increasing=True):
        self.increasing = increasing
        self._iso = None
        self._x_unique = None
        self._y_at_unique = None

    def fit(self, x, target):
        x = np.asarray(x).astype(float)
        target = np.asarray(target).astype(float)
        if _HAS_SK:
            iso = IsotonicRegression(increasing=self.increasing, out_of_bounds='clip')
            y_fit = iso.fit_transform(x, target)
            order = np.argsort(x)
            self._x_unique, idx = np.unique(x[order], return_index=True)
            self._y_at_unique = y_fit[order][idx]
            self._iso = iso
        else:
            xu, yf = _pav_isotonic_fit(x, target, increasing=self.increasing)
            self._x_unique, self._y_at_unique = xu, yf
        return self

    def predict(self, x_new):
        x_new = np.asarray(x_new).astype(float)
        if self._iso is not None:
            return self._iso.predict(x_new)
        return np.interp(x_new, self._x_unique, self._y_at_unique)

class QuantileCalibrator1D:
    def __init__(self, alpha, n_bins=20, increasing=True):
        self.alpha = float(alpha)
        self.n_bins = int(n_bins)
        self.increasing = increasing
        self.base_iso = IsotonicCalibrator1D(increasing=increasing)
        self.knots_x = None
        self.knots_y = None

    def fit(self, q_pred, y):
        q_pred = np.asarray(q_pred).astype(float)
        y = np.asarray(y).astype(float)
        qs = np.linspace(0, 1, self.n_bins + 1)
        edges = np.quantile(q_pred, qs)
        edges = np.unique(edges)
        if len(edges) < 3:
            edges = np.quantile(q_pred, [0, 0.5, 1.0])
        bin_ids = np.clip(np.digitize(q_pred, edges, right=False) - 1, 0, len(edges)-2)
        xk = []
        yk = []
        for b in range(len(edges) - 1):
            idx = (bin_ids == b)
            if np.any(idx):
                xk.append(float(np.mean(q_pred[idx])))
                yk.append(float(empirical_quantile(y[idx], self.alpha)))
        xk = np.array(xk); yk = np.array(yk)
        self.base_iso.fit(xk, yk)
        self.knots_x = self.base_iso._x_unique
        self.knots_y = self.base_iso._y_at_unique
        return self

    def predict(self, q_pred_new):
        return self.base_iso.predict(q_pred_new)
