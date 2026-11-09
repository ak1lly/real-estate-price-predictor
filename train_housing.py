"""Train all three models on an piptrain.csv.

Run: python train_housing.py data/train.csv
Requires: numpy, pandas. No machine-learning libraries.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from regression import LinearRegression, RidgeRegression, LassoRegression, regression_metrics

FEATURES = ["GrLivArea", "OverallQual", "BedroomAbvGr", "FullBath", "GarageCars", "YearBuilt"]


def main(csv_path):
    df = pd.read_csv(csv_path)
    cols = FEATURES + ["SalePrice"]
    df = df[cols].dropna()
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["SalePrice"].to_numpy(dtype=float)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    cutoff = int(0.8 * len(X))
    tr, te = perm[:cutoff], perm[cutoff:]
    X_train, X_test = X[tr], X[te]
    y_train, y_test = y[tr], y[te]

    # Fit preprocessing parameters on training subset ONLY.
    means = X_train.mean(axis=0)
    scales = X_train.std(axis=0)
    scales[scales == 0] = 1.0
    X_train = (X_train - means) / scales
    X_test = (X_test - means) / scales

    models = {
        "Linear": LinearRegression(),
        "Ridge": RidgeRegression(alpha=0.1),
        "Lasso": LassoRegression(alpha=1000.0),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = regression_metrics(y_test, model.predict(X_test))
        print(f"{name}: MAE=${metrics['MAE']:,.0f}, RMSE=${metrics['RMSE']:,.0f}, R2={metrics['R2']:.3f}")
        print("  coefficients (per 1 SD input change):", dict(zip(FEATURES, np.round(model.coef_, 1))))
    print("\nNote: tune alpha using cross-validation on training data only.")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/train.csv")
    main(path)
