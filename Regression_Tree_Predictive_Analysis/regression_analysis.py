"""
Statistical Modeling for Regression Tree and Predictive Analysis
================================================================
Tools: Python, NumPy, Pandas, Matplotlib, Scikit-learn
Models: Linear, Polynomial, Ridge, Regression Tree
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 1.  DATASET GENERATION  (real-world style)
# ──────────────────────────────────────────────
np.random.seed(42)
n = 300

# Feature: engine size (litres)  1.0 – 5.0
engine_size = np.random.uniform(1.0, 5.0, n)

# Target: fuel consumption (mpg) – nonlinear relationship + noise
fuel_consumption = (
    55
    - 8  * engine_size
    + 0.9 * engine_size**2
    - 2.5 * np.sin(engine_size * 1.5)
    + np.random.normal(0, 2.5, n)
)

df = pd.DataFrame({"engine_size": engine_size,
                   "fuel_consumption_mpg": fuel_consumption})

print("=" * 60)
print("  Dataset: Vehicle Engine Size vs Fuel Consumption")
print("=" * 60)
print(df.describe().round(2))
print()

# ──────────────────────────────────────────────
# 2.  TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
X = df[["engine_size"]].values
y = df["fuel_consumption_mpg"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ──────────────────────────────────────────────
# 3.  MODELS
# ──────────────────────────────────────────────

# 3a. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# 3b. Polynomial Regression (degree = 3)
poly_pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=3, include_bias=False)),
    ("lr",   LinearRegression())
])
poly_pipeline.fit(X_train, y_train)

# 3c. Ridge Regression (degree = 3 features, alpha = 1.0)
ridge_pipeline = Pipeline([
    ("poly",  PolynomialFeatures(degree=3, include_bias=False)),
    ("ridge", Ridge(alpha=1.0))
])
ridge_pipeline.fit(X_train, y_train)

# 3d. Regression Tree (max_depth = 4 — controls overfitting via error minimisation)
rt = DecisionTreeRegressor(max_depth=4,
                           min_samples_split=10,
                           min_samples_leaf=5,
                           random_state=42)
rt.fit(X_train, y_train)

# ──────────────────────────────────────────────
# 4.  EVALUATION METRICS
# ──────────────────────────────────────────────
models = {
    "Linear Regression":     lr,
    "Polynomial (deg=3)":    poly_pipeline,
    "Ridge (deg=3, α=1)":    ridge_pipeline,
    "Regression Tree":       rt,
}

results = []
for name, model in models.items():
    y_pred_tr = model.predict(X_train)
    y_pred_te = model.predict(X_test)
    results.append({
        "Model":       name,
        "Train R²":    round(r2_score(y_train, y_pred_tr), 4),
        "Test R²":     round(r2_score(y_test,  y_pred_te), 4),
        "Test RMSE":   round(np.sqrt(mean_squared_error(y_test, y_pred_te)), 4),
        "Test MAE":    round(mean_absolute_error(y_test, y_pred_te), 4),
    })

metrics_df = pd.DataFrame(results)
print("=" * 60)
print("  Model Performance Metrics")
print("=" * 60)
print(metrics_df.to_string(index=False))
print()

# ──────────────────────────────────────────────
# 5.  VISUALISATIONS
# ──────────────────────────────────────────────
X_range = np.linspace(X.min(), X.max(), 400).reshape(-1, 1)

colors = {
    "Linear Regression":  "#E74C3C",
    "Polynomial (deg=3)": "#2ECC71",
    "Ridge (deg=3, α=1)": "#9B59B6",
    "Regression Tree":    "#F39C12",
}

# ── Figure 1 : Model Predictions vs Data ──────
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle("Regression Models — Predictions vs Actual Data",
              fontsize=15, fontweight="bold", y=1.01)
axes = axes.flatten()

for ax, (name, model) in zip(axes, models.items()):
    ax.scatter(X_train, y_train, alpha=0.35, s=25,
               color="#3498DB", label="Train data")
    ax.scatter(X_test, y_test, alpha=0.6, s=35,
               color="#1ABC9C", marker="^", label="Test data")
    y_line = model.predict(X_range)
    ax.plot(X_range, y_line, color=colors[name],
            linewidth=2.5, label=name)
    r2 = r2_score(y_test, model.predict(X_test))
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    ax.set_title(f"{name}\nTest R²={r2:.3f}  RMSE={rmse:.3f}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Engine Size (L)", fontsize=10)
    ax.set_ylabel("Fuel Consumption (mpg)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fig1_model_predictions.png",
            dpi=150, bbox_inches="tight")
print("[Saved] fig1_model_predictions.png")
plt.close()

# ── Figure 2 : All Models on One Plot ─────────
fig2, ax = plt.subplots(figsize=(12, 7))
ax.scatter(X, y, alpha=0.2, s=20, color="#95A5A6", label="All data")
ax.scatter(X_test, y_test, alpha=0.6, s=40,
           color="#3498DB", marker="^", label="Test data", zorder=5)

for name, model in models.items():
    y_line = model.predict(X_range)
    ax.plot(X_range, y_line, color=colors[name],
            linewidth=2.2, label=name)

ax.set_title("All Regression Models — Comparison",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Engine Size (L)", fontsize=12)
ax.set_ylabel("Fuel Consumption (mpg)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig2_model_comparison.png",
            dpi=150, bbox_inches="tight")
print("[Saved] fig2_model_comparison.png")
plt.close()

# ── Figure 3 : Metrics Bar Chart ──────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle("Model Performance Comparison",
              fontsize=14, fontweight="bold")

bar_colors = list(colors.values())
short_names = ["Linear", "Poly(3)", "Ridge(3)", "Reg. Tree"]
metrics_to_plot = [("Test R²", "Test R²", "higher is better"),
                   ("Test RMSE", "Test RMSE", "lower is better"),
                   ("Test MAE",  "Test MAE",  "lower is better")]

for ax3, (col, title, note) in zip(axes3, metrics_to_plot):
    vals = metrics_df[col].values
    bars = ax3.bar(short_names, vals, color=bar_colors, edgecolor="white",
                   linewidth=1.2, alpha=0.88)
    for bar, v in zip(bars, vals):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(vals)*0.01,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax3.set_title(f"{title}\n({note})", fontsize=11, fontweight="bold")
    ax3.set_ylabel(title, fontsize=10)
    ax3.grid(axis="y", alpha=0.3)
    ax3.set_ylim(0, max(vals) * 1.18)

plt.tight_layout()
plt.savefig("fig3_metrics_comparison.png",
            dpi=150, bbox_inches="tight")
print("[Saved] fig3_metrics_comparison.png")
plt.close()

# ── Figure 4 : Regression Tree Visualisation ──
fig4, ax4 = plt.subplots(figsize=(20, 8))
plot_tree(rt, feature_names=["Engine Size (L)"],
          filled=True, rounded=True, fontsize=9, ax=ax4,
          impurity=False, precision=2)
ax4.set_title("Regression Tree Structure (max_depth=4)",
              fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("fig4_tree_structure.png",
            dpi=150, bbox_inches="tight")
print("[Saved] fig4_tree_structure.png")
plt.close()

# ── Figure 5 : Residual Plots ─────────────────
fig5, axes5 = plt.subplots(2, 2, figsize=(14, 10))
fig5.suptitle("Residual Analysis — All Models",
              fontsize=14, fontweight="bold")
axes5 = axes5.flatten()

for ax5, (name, model) in zip(axes5, models.items()):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    ax5.scatter(y_pred, residuals, alpha=0.6, s=35,
                color=colors[name], edgecolors="white", linewidths=0.5)
    ax5.axhline(0, color="black", linewidth=1.5, linestyle="--")
    ax5.set_title(f"{name} — Residuals", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Predicted Values", fontsize=10)
    ax5.set_ylabel("Residuals", fontsize=10)
    ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fig5_residuals.png",
            dpi=150, bbox_inches="tight")
print("[Saved] fig5_residuals.png")
plt.close()

# ── Figure 6 : Regression Tree — Nonlinear Prediction Detail ──
fig6, ax6 = plt.subplots(figsize=(12, 6))
ax6.scatter(X_train, y_train, alpha=0.3, s=25,
            color="#3498DB", label="Train data")
ax6.scatter(X_test, y_test, alpha=0.7, s=45,
            color="#1ABC9C", marker="^", label="Test data", zorder=5)

# Step-function nature of tree
y_tree_line = rt.predict(X_range)
ax6.plot(X_range, y_tree_line, color="#F39C12",
         linewidth=2.5, label="Regression Tree (step)")

# True underlying curve
y_true_curve = (55 - 8*X_range.ravel()
                + 0.9*X_range.ravel()**2
                - 2.5*np.sin(X_range.ravel()*1.5))
ax6.plot(X_range, y_true_curve, color="#2C3E50",
         linewidth=1.5, linestyle="--", alpha=0.7, label="True curve")

ax6.set_title("Regression Tree — Nonlinear Prediction\n"
              "via Statistical Error Minimisation (MSE splits)",
              fontsize=13, fontweight="bold")
ax6.set_xlabel("Engine Size (L)", fontsize=12)
ax6.set_ylabel("Fuel Consumption (mpg)", fontsize=12)
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig6_tree_nonlinear.png",
            dpi=150, bbox_inches="tight")
print("[Saved] fig6_tree_nonlinear.png")
plt.close()

print("\n All visualisations saved successfully.")
print("\nFinal Metrics Summary:")
print(metrics_df.to_string(index=False))
