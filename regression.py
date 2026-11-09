"""Linear, Ridge, and Lasso regression implemented with NumPy only.

Each model expects X to be an (n_samples, n_features) array WITHOUT
an intercept column. Intercepts are fitted internally and never penalized.

Objectives:
 Linear: (1/(2n)) ||y - (Xw+b)||_2^2
 Ridge:  (1/(2n)) ||y - (Xw+b)||_2^2 + (alpha/2)||w||_2^2
 Lasso:  (1/(2n)) ||y - (Xw+b)||_2^2 + alpha||w||_1
"""
import numpy as np


def _validate_X_y(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be nonempty 2D array")
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must have one value per row of X")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain only finite values")
    return X, y


def _validate_predict(X, n_features):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != n_features:
        raise ValueError(f"X must have shape (n_samples, {n_features})")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    return X


class LinearRegression:
    def fit(self, X, y):
        X, y = _validate_X_y(X, y)
        n = X.shape[0]
        A = np.column_stack((np.ones(n), X))
        # Minimum-norm least-squares solution also works with singular X^T X.
        params = np.linalg.lstsq(A, y, rcond=None)[0]
        self.intercept_ = float(params[0])
        self.coef_ = params[1:]
        return self

    def predict(self, X):
        X = _validate_predict(X, len(self.coef_))
        return X @ self.coef_ + self.intercept_


class RidgeRegression:
    def __init__(self, alpha=1.0):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.alpha = float(alpha)

    def fit(self, X, y):
        X, y = _validate_X_y(X, y)
        n, d = X.shape
        # Centering means intercept is not regularized.
        x_mean = X.mean(axis=0)
        y_mean = y.mean()
        Xc = X - x_mean
        yc = y - y_mean
        self.coef_ = np.linalg.solve(
            (Xc.T @ Xc) / n + self.alpha * np.eye(d),
            (Xc.T @ yc) / n,
        )
        self.intercept_ = float(y_mean - x_mean @ self.coef_)
        return self

    def predict(self, X):
        X = _validate_predict(X, len(self.coef_))
        return X @ self.coef_ + self.intercept_


def _soft_threshold(value, threshold):
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)


class LassoRegression:
    """Cyclic coordinate descent; standardizes predictors internally."""
    def __init__(self, alpha=1.0, max_iter=10000, tol=1e-7):
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be >= 0")
        if max_iter < 1 or tol <= 0:
            raise ValueError("max_iter must be >= 1 and tol > 0")
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def fit(self, X, y):
        X, y = _validate_X_y(X, y)
        n, d = X.shape
        x_mean = X.mean(axis=0)
        x_scale = X.std(axis=0)
        x_scale[x_scale == 0] = 1.0
        Z = (X - x_mean) / x_scale
        y_mean = y.mean()
        yc = y - y_mean

        weights = np.zeros(d)
        residual = yc.copy() # residual = yc - Z @ weights
        squared_norms = np.sum(Z * Z, axis=0) / n
        self.converged_ = False
        for it in range(self.max_iter):
            old_weights = weights.copy()
            for j in range(d):
                if squared_norms[j] == 0:
                    continue
                # Put feature j's old contribution back in the residual.
                residual += Z[:, j] * weights[j]
                rho = Z[:, j] @ residual / n
                weights[j] = _soft_threshold(rho, self.alpha) / squared_norms[j]
                residual -= Z[:, j] * weights[j]
            if np.max(np.abs(weights - old_weights)) < self.tol:
                self.converged_ = True
                break

        self.n_iter_ = it + 1
        self.coef_ = weights / x_scale
        self.intercept_ = float(y_mean - x_mean @ self.coef_)
        return self

    def predict(self, X):
        X = _validate_predict(X, len(self.coef_))
        return X @ self.coef_ + self.intercept_


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must have matching nonempty shapes")
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = np.sum((y_true - y_true.mean())**2)
    r2 = float(1 - np.sum(err**2) / denom) if denom > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}
