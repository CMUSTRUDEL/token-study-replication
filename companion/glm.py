"""Poisson-family GLM with log link, fitted by IRLS.

The plan (token-study.tex, Appendix) specifies Poisson quasi-likelihood: the
mean model is consistent under variance misspecification, and only intervals
depend on the dispersion treatment. The pipeline reads fitted values, deviance
shares, and coefficients off the mean model, so plain IRLS is all it needs.
No statsmodels dependency.
"""

import numpy as np


def irls_poisson(X, y, max_iter=100, tol=1e-9, ridge=1e-8):
    """Fit E[y] = exp(X beta). Returns beta.

    `ridge` is a tiny Tikhonov term that keeps the weighted normal equations
    solvable when dummies are near-collinear (e.g. a week in which almost
    nobody was active); it does not visibly move any coefficient of interest.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta = np.zeros(p)
    # McCullagh-Nelder start: eta from a smoothed copy of y itself, so the
    # first working response is well-scaled even when y spans decades
    eta = np.log(np.maximum(y, 0.0) + 0.1 * max(y.mean(), 1.0))
    dev_old = np.inf
    for _ in range(max_iter):
        mu = np.exp(np.clip(eta, -30, 30))
        z = eta + (y - mu) / mu
        w = mu
        Xw = X * w[:, None]
        A = X.T @ Xw + ridge * np.eye(p)
        b = X.T @ (w * z)
        beta_new = np.linalg.solve(A, b)
        eta = X @ beta_new
        dev = poisson_deviance(y, np.exp(np.clip(eta, -30, 30)))
        if abs(dev_old - dev) < tol * (abs(dev) + 1e-12):
            beta = beta_new
            break
        beta, dev_old = beta_new, dev
    return beta


def poisson_deviance(y, mu):
    """Unit Poisson deviance summed over observations (scale-free in the
    quasi-likelihood sense: dispersion cancels from deviance *shares*)."""
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2.0 * np.sum(term - (y - mu))


def design(blocks):
    """Column-stack a list of (name, array-or-matrix) into X plus name list.
    First block should be the intercept."""
    cols, names = [], []
    for name, arr in blocks:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            cols.append(arr[:, None])
            names.append(name)
        else:
            cols.append(arr)
            names.extend(f"{name}{j}" for j in range(arr.shape[1]))
    return np.hstack(cols), names


def dummies(codes, drop_first=True):
    """0/1 dummy matrix from integer codes 0..K-1."""
    codes = np.asarray(codes)
    K = codes.max() + 1
    M = np.zeros((codes.size, K))
    M[np.arange(codes.size), codes] = 1.0
    return M[:, 1:] if drop_first else M
