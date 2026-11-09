# Real Estate Regression (from scratch)

Models in `regression.py`:
- `LinearRegression` — NumPy least squares
- `RidgeRegression(alpha)` — closed-form solution with L2 penalty
- `LassoRegression(alpha)` — cyclic coordinate descent with soft thresholding

All models use `fit(X, y)` and `predict(X)`. Pass **raw feature columns only**: no manually augmented intercept column. Both `(n,)` and `(n,1)` target arrays are accepted, while predictions are returned as `(n,)`.

## Run tests

```bash
pip install numpy pandas
python test_regression.py
```

## Run on Ames house price data

Download the Kaggle House Prices `train.csv`, save it to `data/train.csv` relative to this folder, then run:

```bash
python train_housing.py data/train.csv
```

`train_housing.py` uses a fixed 80/20 split and normalizes with training-only means/stds. The regularization strengths are examples; select them through cross-validation within the training set before reporting final test performance.

## Mathematical objectives

With `n` observations, input matrix `X` and target `y`, all models include an **unpenalized intercept** `b`:

- Linear: `min (1/(2n)) ||y-Xw-b||_2^2`
- Ridge: `min (1/(2n)) ||y-Xw-b||_2^2 + (alpha/2)||w||_2^2`
- Lasso: `min (1/(2n)) ||y-Xw-b||_2^2 + alpha||w||_1`

Lasso internally standardizes the features so `alpha` applies to the standardized coefficients, and maps fitted coefficients back to the original input units. For a fair comparison across models, the included housing script standardizes inputs externally for all three.
