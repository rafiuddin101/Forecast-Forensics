import numpy as np
import pandas as pd

def make_synthetic_hetero(n=5000, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, size=n)
    z_cov = rng.normal(0, 1, size=n)
    mu_true = 1.5 * np.sin(x) + 0.5 * x
    sigma_true = 0.5 + 0.75 * (x > 0) + 0.3 * (x**2)
    mix = rng.binomial(1, 0.1, size=n)
    eps = rng.normal(0, 1, size=n) * (1 - mix) + rng.standard_t(df=3, size=n) * mix
    y = mu_true + sigma_true * eps
    mean_pred = mu_true + 0.3 * (x < 0) - 0.2 * (x >= 0)
    scale_pred = 0.8 * sigma_true
    q_01_pred = mean_pred + scale_pred * (-1.1)
    q_05_pred = mean_pred
    q_09_pred = mean_pred + scale_pred * (1.1)
    event = (y > 0).astype(int)
    p_true = 1 / (1 + np.exp(-(mu_true / (sigma_true + 1e-6))))
    p_pred = 0.9 * p_true + 0.05
    df = pd.DataFrame({
        'y': y,
        'mean_pred': mean_pred,
        'q_0.1_pred': q_01_pred,
        'q_0.5_pred': q_05_pred,
        'q_0.9_pred': q_09_pred,
        'event': event,
        'p_pred': p_pred,
        'z_cov': z_cov,
    })
    return df
