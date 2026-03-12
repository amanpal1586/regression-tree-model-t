# Statistical Modeling for Regression Tree and Predictive Analysis
 
> **Tools & Technologies:** Python · NumPy · Pandas · Matplotlib · Scikit-learn · Statistical Modeling

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Models Implemented](#models-implemented)
   - [Linear Regression](#1-linear-regression)
   - [Polynomial Regression](#2-polynomial-regression)
   - [Ridge Regression](#3-ridge-regression)
   - [Regression Tree](#4-regression-tree)
4. [Statistical Theory & Definitions](#statistical-theory--definitions)
   - [Curve Fitting](#curve-fitting)
   - [Error Minimisation (Loss Functions)](#error-minimisation-loss-functions)
   - [Overfitting & Underfitting](#overfitting--underfitting)
   - [Regularisation](#regularisation)
   - [Bias-Variance Tradeoff](#bias-variance-tradeoff)
5. [Performance Metrics](#performance-metrics)
6. [Visualisations](#visualisations)
7. [Project Structure](#project-structure)
8. [How to Run](#how-to-run)
9. [Results Summary](#results-summary)
10. [Key Takeaways](#key-takeaways)

---

## Project Overview

This project demonstrates the full pipeline of **statistical and machine learning regression modelling** applied to a real-world-style dataset. The goal is to predict vehicle fuel consumption (mpg) from engine size (litres) using four different regression approaches — ranging from simple linear models to decision-tree-based nonlinear predictors.

The project bridges **classical statistical theory** (least squares, error minimisation, curve fitting) with **modern ML techniques** (regularisation, decision trees, pipeline-based modelling), showcasing both the mathematical foundations and practical Python implementations.

---

## Dataset Description

| Property | Value |
|---|---|
| Domain | Automotive / Engineering |
| Features | Engine Size (litres, continuous) |
| Target | Fuel Consumption (mpg, continuous) |
| Samples | 300 observations |
| Train / Test Split | 80% / 20% |
| Relationship | Nonlinear (quadratic + sinusoidal + Gaussian noise) |

The underlying data-generating function is:

```
y = 55 - 8x + 0.9x² - 2.5·sin(1.5x) + ε,   ε ~ N(0, 2.5)
```

This intentionally introduces **nonlinearity** to stress-test linear models and demonstrate where regression trees and polynomial models excel.

---

## Models Implemented

### 1. Linear Regression

**Definition:**  
Linear Regression models the relationship between a dependent variable `y` and one or more independent variables `X` by fitting a straight line (hyperplane in higher dimensions) that minimises the **sum of squared residuals**.

**Mathematical Form:**
```
ŷ = β₀ + β₁x
```

**Parameter Estimation (Ordinary Least Squares):**
```
β = (XᵀX)⁻¹ Xᵀy
```

**Key Assumptions:**
- Linearity between X and y
- Homoscedasticity (constant variance of residuals)
- Independence of observations
- Normally distributed residuals

**Python Implementation:**
```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
```

**When to use:** Baseline model; when relationship is expected to be approximately linear.

---

### 2. Polynomial Regression

**Definition:**  
Polynomial Regression extends linear regression by adding **polynomial terms** of the predictor variable, enabling the model to capture nonlinear relationships while still using linear regression machinery on the transformed features.

**Mathematical Form (degree d):**
```
ŷ = β₀ + β₁x + β₂x² + β₃x³ + ... + βdxᵈ
```

**Key Idea:**  
By mapping `x → [x, x², x³, ..., xᵈ]`, the problem is still linear in parameters, so OLS still applies. The **PolynomialFeatures** transformer creates these new feature columns automatically.

**Degree Selection:**
- Degree 1 → Linear Regression
- Degree 2 → Quadratic (parabola)
- Degree 3 → Cubic (inflection points)
- Higher degrees → Risk of overfitting

**Python Implementation:**
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

poly_pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=3, include_bias=False)),
    ("lr",   LinearRegression())
])
poly_pipeline.fit(X_train, y_train)
```

**When to use:** When scatter plot suggests a curved, nonlinear relationship.

---

### 3. Ridge Regression

**Definition:**  
Ridge Regression (also called **L2 Regularisation** or **Tikhonov Regularisation**) is a regularised version of linear (or polynomial) regression that penalises large coefficients to prevent overfitting.

**Mathematical Form:**
```
β_ridge = argmin { ||y - Xβ||² + α·||β||² }
```

Where:
- `||y - Xβ||²` is the ordinary least-squares loss
- `α·||β||²` is the L2 penalty term
- `α` (alpha) is the regularisation strength hyperparameter

**Closed-Form Solution:**
```
β_ridge = (XᵀX + αI)⁻¹ Xᵀy
```

**Effect of Alpha:**
| Alpha Value | Effect |
|---|---|
| α = 0 | Same as OLS (no regularisation) |
| Small α | Slight shrinkage of coefficients |
| Large α | Heavy shrinkage, coefficients → 0 |
| α → ∞ | All coefficients → 0 (underfitting) |

**Python Implementation:**
```python
from sklearn.linear_model import Ridge

ridge_pipeline = Pipeline([
    ("poly",  PolynomialFeatures(degree=3, include_bias=False)),
    ("ridge", Ridge(alpha=1.0))
])
ridge_pipeline.fit(X_train, y_train)
```

**When to use:** When polynomial/linear models overfit; when features are correlated (multicollinearity).

---

### 4. Regression Tree

**Definition:**  
A Regression Tree (a type of **Decision Tree Regressor**) partitions the feature space into rectangular regions through recursive binary splitting, then assigns the **mean of training samples** in each region as the predicted value.

**Algorithm — Recursive Binary Splitting:**
```
For each feature j and split point s:
  Find (j*, s*) = argmin [ MSE(left region) + MSE(right region) ]
  Split data into two nodes
  Repeat recursively until stopping criterion met
```

**Split Criterion — Mean Squared Error:**
```
MSE_split = (1/nL)·Σ(yi - ȳL)² + (1/nR)·Σ(yi - ȳR)²
```

Where:
- `nL`, `nR` = number of samples in left/right child nodes
- `ȳL`, `ȳR` = mean target values in left/right nodes

**Stopping Criteria (Error Minimisation Controls):**
| Parameter | Meaning |
|---|---|
| `max_depth` | Maximum tree depth (limits complexity) |
| `min_samples_split` | Minimum samples required to split a node |
| `min_samples_leaf` | Minimum samples required at a leaf node |

**Prediction:**  
For a new input `x`, traverse the tree from root to leaf → predict the mean `ȳ` of training samples in that leaf.

**Key Properties:**
- Captures **nonlinear** relationships naturally
- Produces **piecewise-constant** predictions (step function)
- Non-parametric: no assumption about data distribution
- Prone to overfitting if `max_depth` not controlled

**Python Implementation:**
```python
from sklearn.tree import DecisionTreeRegressor

rt = DecisionTreeRegressor(
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
rt.fit(X_train, y_train)
```

**When to use:** When the relationship is highly nonlinear, non-monotonic, or has abrupt changes.

---

## Statistical Theory & Definitions

### Curve Fitting

**Definition:**  
Curve fitting is the process of constructing a mathematical function (curve) that best fits a series of data points. It is a form of regression analysis that minimises the deviation between the observed data and the model's predictions.

**Types:**
- **Interpolation** — curve passes through every data point exactly
- **Regression (Approximation)** — curve minimises error across all points (does not need to pass through every point)

**Least Squares Curve Fitting:**
The most common criterion. Find parameters `β` such that:
```
S = Σᵢ [yᵢ - f(xᵢ; β)]²  →  minimised
```

**Polynomial Curve Fitting:**
```
f(x) = a₀ + a₁x + a₂x² + ... + aₙxⁿ
```
Higher-degree polynomials fit training data better but may overfit.

---

### Error Minimisation (Loss Functions)

**Definition:**  
A loss function quantifies the difference between predicted values `ŷ` and actual values `y`. Training a model means finding parameters that minimise this function.

**Common Loss Functions:**

| Loss Function | Formula | Notes |
|---|---|---|
| Mean Squared Error (MSE) | `(1/n)·Σ(yᵢ - ŷᵢ)²` | Penalises large errors heavily |
| Root MSE (RMSE) | `√MSE` | Same units as target variable |
| Mean Absolute Error (MAE) | `(1/n)·Σ|yᵢ - ŷᵢ|` | Robust to outliers |
| Sum of Squared Residuals (SSR) | `Σ(yᵢ - ŷᵢ)²` | OLS minimises this |
| R² Score | `1 - SSR/SST` | Proportion of variance explained |

**Residual:**  
The difference between an observed value and the predicted value:
```
eᵢ = yᵢ - ŷᵢ
```

---

### Overfitting & Underfitting

**Overfitting:**  
The model learns the training data too well, including its noise, and fails to generalise to new data.
- Symptoms: High Train R², Low Test R²
- Causes: Too many parameters, too deep tree, too high polynomial degree

**Underfitting:**  
The model is too simple to capture the underlying pattern.
- Symptoms: Low Train R² and Low Test R²
- Causes: Linear model on nonlinear data, insufficient features

**How to detect:**  
Compare Train vs Test performance. Large gap = overfitting; both low = underfitting.

---

### Regularisation

**Definition:**  
Regularisation adds a penalty term to the loss function to discourage complex models, reducing overfitting by shrinking model parameters.

**L1 Regularisation (Lasso):**
```
Loss = MSE + α·Σ|βⱼ|
```
Effect: Sets some coefficients exactly to zero (feature selection)

**L2 Regularisation (Ridge):**
```
Loss = MSE + α·Σβⱼ²
```
Effect: Shrinks all coefficients toward zero (but not exactly zero)

**ElasticNet:** Combination of L1 + L2

---

### Bias-Variance Tradeoff

**Definition:**  
Every model's prediction error can be decomposed into three components:
```
Expected MSE = Bias² + Variance + Irreducible Noise
```

| Term | Definition |
|---|---|
| **Bias** | Error from incorrect assumptions (model too simple) |
| **Variance** | Error from sensitivity to small fluctuations in training data |
| **Irreducible Noise** | Inherent noise in the data — cannot be reduced |

**Tradeoff:**
- Increasing model complexity → **decreases Bias, increases Variance**
- Decreasing model complexity → **increases Bias, decreases Variance**
- Goal: Find the sweet spot that minimises **total error**

**Relation to our models:**
- Linear Regression: High bias, low variance
- High-degree Polynomial: Low bias, high variance
- Ridge: Reduces variance by shrinking coefficients
- Regression Tree (depth=4): Balanced; max_depth controls variance

---

## Performance Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **R² Score** | `1 - SSR/SST` | 1.0 = perfect; 0 = predicts mean; <0 = worse than mean |
| **RMSE** | `√(Σ(y-ŷ)²/n)` | Average prediction error in original units |
| **MAE** | `Σ|y-ŷ|/n` | Median-like average error, robust to outliers |
| **Train R²** | Computed on training data | Higher = better fit |
| **Test R²** | Computed on unseen test data | Key metric for generalisation |

**Results from this project:**

| Model | Train R² | Test R² | Test RMSE | Test MAE |
|---|---|---|---|---|
| Linear Regression | 0.4963 | 0.3890 | 2.3163 | 1.8623 |
| Polynomial (deg=3) | 0.5180 | 0.4734 | 2.1503 | 1.7115 |
| Ridge (deg=3, α=1) | 0.5178 | 0.4711 | 2.1551 | 1.7110 |
| Regression Tree | 0.6059 | 0.4278 | 2.2417 | 1.7718 |

---

## Visualisations

| Figure | Description |
|---|---|
| `fig1_model_predictions.png` | 2×2 subplot showing each model's fitted curve vs train/test data |
| `fig2_model_comparison.png` | All 4 models overlaid on a single plot for visual comparison |
| `fig3_metrics_comparison.png` | Bar charts comparing R², RMSE, and MAE across all models |
| `fig4_tree_structure.png` | Full regression tree diagram showing all nodes, splits, and leaf values |
| `fig5_residuals.png` | Residual plots for all 4 models — checks model assumptions |
| `fig6_tree_nonlinear.png` | Regression tree step-function vs the true nonlinear underlying curve |

---

## Project Structure

```
regression_project/
│
├── regression_analysis.py      # Main Python script (all models + visualisations)
├── README.md                   # This file
├── theory_report.pdf           # LaTeX-compiled PDF of all statistical theory
│
├── fig1_model_predictions.png
├── fig2_model_comparison.png
├── fig3_metrics_comparison.png
├── fig4_tree_structure.png
├── fig5_residuals.png
└── fig6_tree_nonlinear.png
```

---

## How to Run

**Prerequisites:**
```bash
pip install numpy pandas matplotlib scikit-learn
```

**Run the main script:**
```bash
python regression_analysis.py
```

**Expected Output:**
- Console: Dataset statistics + model performance metrics table
- Files: 6 PNG visualisation plots saved to current directory

---

## Results Summary

- **Best Test R²:** Polynomial Regression (0.4734) — captures the curved relationship
- **Lowest RMSE:** Polynomial Regression (2.1503 mpg) — best generalisation
- **Best Train R²:** Regression Tree (0.6059) — most flexible, fits training data best
- **Linear Regression:** Underperforms due to the nonlinear nature of the data
- **Ridge vs Polynomial:** Nearly identical performance — L2 penalty had minimal effect at α=1.0

---

## Key Takeaways

1. **Model selection matters** — Linear regression fails on nonlinear data; polynomial and tree models capture the curvature better.
2. **Regression trees are powerful** — They naturally model nonlinear, piecewise relationships without assuming any functional form.
3. **Statistical error minimisation** — All models optimise some form of MSE; the tree algorithm greedily minimises MSE at each split.
4. **Regularisation adds stability** — Ridge reduces coefficient magnitude and prevents overfitting in high-degree polynomial models.
5. **Always evaluate on test data** — Train performance alone is misleading; generalisation (Test R²) is the true measure of model quality.

---

*Built with Python · NumPy · Pandas · Matplotlib · Scikit-learn*
