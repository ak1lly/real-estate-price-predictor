import numpy as np
from regression import LinearRegression, RidgeRegression, LassoRegression, regression_metrics

rng = np.random.default_rng(42)
X = rng.normal(size=(250, 3))
y = 80000 + X @ np.array([12000., -7000., 0.]) + rng.normal(0, 1000, 250)

for name, model in [
    ("Linear", LinearRegression()),
    ("Ridge", RidgeRegression(alpha=0.1)),
    ("Lasso", LassoRegression(alpha=200.0)),
]:
    model.fit(X[:200], y[:200])
    print(f"{name:7} | intercept={model.intercept_:,.2f} | coefficients={np.round(model.coef_, 2)}")
    print("    test metrics:", regression_metrics(y[200:], model.predict(X[200:])))

# On exactly linear data, all models should fit the known intercept.
exact_y = 5 + 2 * X[:, 0] - 3 * X[:, 1]
ols = LinearRegression().fit(X, exact_y)
assert np.allclose(ols.predict(X), exact_y)
assert np.isclose(ols.intercept_, 5)

# A zero penalty recovers OLS on full-rank data.
lasso0 = LassoRegression(alpha=0).fit(X, exact_y)
assert np.allclose(lasso0.predict(X), exact_y, atol=1e-5)

# A very large penalty zeros coefficients without shrinking the intercept.
lasso_big = LassoRegression(alpha=1e8).fit(X, y)
assert np.allclose(lasso_big.coef_, 0)
assert np.isclose(lasso_big.intercept_, y.mean())
print("All checks passed.")
