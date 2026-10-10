# House Price Prediction: Multiple Linear Regression

Predicting home sale prices in Ames, Iowa with a multiple linear regression model, built on the Kaggle **House Prices** training set (1,460 homes, 80 features).

## Results
| | |
|---|---|
| **Adjusted R²** | **0.927** |
| RMSE (log10 scale) | 0.043, a typical error of about ±10% of sale price |
| Final predictors | 83 (from 20+ cleaned features), **all VIF < 10** |

**Key drivers of sale price:** overall quality, neighborhood (Crawford, Stone Brook, Somerset, Northridge Heights, Northridge carry a premium), above-ground living area, kitchen quality, and garage presence.

## Approach
1. **Target:** `SalePrice` is right-skewed, so a log10 transformation was applied (`SalePrice_log10`).
2. **Variable-by-variable EDA and cleaning** for 20+ features: descriptive statistics, univariate and bivariate plots, and correlation with price.
   - Missing `LotFrontage` imputed with a linear regression on `LotArea`
   - Skewed variables log-transformed (`LotArea`, `GrLivArea`); outliers winsorized at ±3 SD
   - Rare categories (< 1%) combined; years converted to ages (`HouseAge`, `HomeAge`)
   - Categorical variables one-hot encoded; continuous variables z-score standardized
3. **Modeling:** OLS regression, with checks for linearity, normal errors with mean 0, and homoscedasticity. Extreme residuals (|z| > 2.5) were removed.
4. **Multicollinearity:** one reference category was dropped from each of the 17 one-hot groups to avoid the dummy variable trap, redundant transformations were dropped, and the highest-VIF predictor was removed iteratively until all VIF < 10.

## Files
| File | Description |
|---|---|
| `Kaggle_Project_Thu_Nguyen.ipynb` | Full analysis: EDA, cleaning, regression model |
| `train.csv` | Kaggle House Prices training data (1,460 rows) |
| `requirements.txt` | Python packages |

## How to run
```bash
pip install -r requirements.txt
jupyter notebook Kaggle_Project_Thu_Nguyen_Part_2.ipynb
```

## Tools
Python · pandas · NumPy · statsmodels · pingouin · SciPy · seaborn · Matplotlib

## Data source
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques), Kaggle (Ames Housing dataset, Dean De Cock).
