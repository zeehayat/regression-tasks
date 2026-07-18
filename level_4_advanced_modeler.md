# Chapter 4: Level 4 Advanced Modeler — Ensembles, Nonlinearities, Tuning, and Interpretability

Welcome to Chapter 4! You've graduated from basic linear regression. You know how to preprocess data and audit a model. Now, we are going to unlock the heavy machinery of tabular regression: Tree Ensembles (Random Forests, Gradient Boosting), target transformations, hyperparameter optimization, and model interpretability.

By the end of this chapter, you'll be able to build models that dominate Kaggle competitions and tabular research datasets, and explain exactly how they make their decisions.

---

## 16. Nonlinear Regression Models

Suppose someone collected crop yield data ($X$, containing `soil_ph` and `average_temperature_celsius`). They want to find crop yield. They notice that the relationship is curved and interactive. A linear line is too rigid (underfits). What do we do?

We move beyond straight lines:

### 16.1 Polynomials and Splines
* **Polynomials:** We already saw how adding $X^2$ fits a curve. However, high-degree polynomials (e.g., $X^5$) behave wildly near the boundaries (extrapolation).
* **Splines:** Fit piecewise curves joined at specific points called *knots*. They are smooth, flexible, and behave better at boundaries.

### 16.2 Generalized Additive Models (GAMs)
GAMs assume that features have smooth, additive nonlinear effects:
$$y = \beta_0 + f_1(x_1) + f_2(x_2) + \dots + \epsilon$$
where each $f(x)$ is a smooth function (usually a spline) learned from the data. They provide a beautiful balance: they are flexible like machine learning, but interpretable like linear regression (you can plot the shape of each $f_i$ directly!).

---

## 17. Tree-Based Regression (The tabular workhorses)

Suppose you have categoricals, missing values, and nonlinear relationships. You don't want to spend hours manually engineering polynomials and scaling features. What do we do?

We use Decision Trees and Random Forests.

### 17.1 Decision Trees
A decision tree splits the feature space recursively based on rules. E.g., "Is estimated duration > 24 months? Yes/No." It predicts the mean of the training observations in each final box (leaf).
* **Pros:** Non-parametric, handles interactions automatically, needs no feature scaling.
* **Cons:** Highly unstable, deep trees overfit easily.
* **Hyperparameters:** `max_depth` (complexity), `min_samples_leaf` (prevents small leaf groups).

### 17.2 Random Forest (Bagging)
Train $M$ different decision trees on different bootstrap samples of the training data. For each tree, select a random subset of features at each split. Average the predictions of all trees.
* **Why it works:** Individual trees have high variance (unstable). Averaging them cancels out the variance while preserving the low bias.
* **The Extrapolation Limit (CRITICAL):** Tree-based models **cannot extrapolate** beyond the range of the training target. If the largest crop yield in your training data is 30 tons/acre, a Random Forest can never predict 31, no matter how much fertilizer or rain you give it. Linear models can extrapolate; trees cannot.

---

## 18. Gradient Boosting (The tabular kings)

Suppose a Random Forest still isn't accurate enough. What do we do?

We use Gradient Boosting. While Random Forest trains trees in parallel, Gradient Boosting trains them **sequentially**.
Each tree is trained to predict the **residuals** (errors) of the previous trees.

$$\hat{y}_0 = \bar{y} \text{ (the mean)}$$
$$\hat{y}_1 = \hat{y}_0 + \eta \cdot \text{tree}_1(y - \hat{y}_0)$$
$$\hat{y}_2 = \hat{y}_1 + \eta \cdot \text{tree}_2(y - \hat{y}_1)$$
where $\eta$ is the **learning rate** (usually 0.01 to 0.1). By taking small steps, the model slowly and precisely fits complex surfaces.
* **Libraries to know:** `XGBoost`, `LightGBM`, `CatBoost`, and scikit-learn's `HistGradientBoostingRegressor`.

---

## 19. SVR, KNN, and Kernel Regression

* **KNN Regression:** Finds the $k$ nearest neighbors in feature space and averages their targets. 
  * *Warning:* Fails in high dimensions (curse of dimensionality) and requires scaling.
* **Support Vector Regression (SVR):** Finds a function that has at most $\epsilon$ deviation from the target, ignoring small errors. Extremely slow on datasets with > 50,000 rows.

---

## 20. Target Transformations and Skewed Targets

Suppose someone collected infrastructure project budgets. They want to predict `actual_cost_million_pkr`. 
They plot the target and see a massive right skew (most projects cost 5M, a few cost 2000M).
If they train directly, the OLS loss function ($MSE$) will be dominated by the 2000M projects, resulting in poor predictions for normal-sized projects. What do we do?

We transform the target:

### 20.1 The Log Transform
$$y_{\text{transformed}} = \log(1 + y) = \text{log1p}(y)$$
Train your model on $y_{\text{transformed}}$. When making predictions, invert the transform:
$$\hat{y} = \exp(\hat{y}_{\text{transformed}}) - 1 = \text{expm1}(\hat{y}_{\text{transformed}})$$
In `scikit-learn`, use `TransformedTargetRegressor` to handle this automatically:
```python
from sklearn.compose import TransformedTargetRegressor
model = TransformedTargetRegressor(regressor=Ridge(), func=np.log1p, inverse_func=np.expm1)
```

---

## 21. Multi-Output Regression

Suppose you want to predict both `actual_cost_million_pkr` and `actual_duration_months` for a project. What do we do?
* **Option A (Separate Models):** Train two models. Simple, but ignores correlations between cost and duration.
* **Option B (Regressor Chain):** Predict duration first. Then, feed the *predicted* duration as a feature to the cost model. This captures the correlation without leaking the true actual duration!

---

## 22. Hyperparameter Tuning

We must choose parameters like tree depth or Ridge alpha.
* **Bayesian Optimization (Optuna):** Instead of checking all combinations (Grid Search) or guessing randomly (Random Search), Bayesian optimization builds a probabilistic model of the loss and chooses the next set of hyperparameters strategically.
* **Early Stopping:** Stop training boosting models when validation error stops improving to prevent overfitting.

---

## 23. Cross-Validation (The proper way)

* **Nested CV:** Use an inner loop of CV to find hyperparameters, and an outer loop of CV to evaluate final performance. This prevents hyperparameter tuning from leaking test information.

---

## 24. Model Interpretation

Do not treat machine learning as a black box. 

### 24.1 Permutation Importance
Shuffle the values of a feature and measure how much the validation score drops. If shuffling `fertilizer` destroys your score, it's an important feature.

### 24.2 Partial Dependence Plots (PDP) & ICE Plots
Show how the predicted target changes as one feature changes, holding all other features constant. 

### 24.3 SHAP (Shapley Additive exPlanations)
Based on game theory. It calculates how much each feature contributes to the deviation of a single prediction from the global average prediction.
* *Vibe Check:* If community A has a predicted development score of 72, SHAP can tell you: "Baseline is 52. Income added +15, health distance added +8, and terrain subtracted -3."

---

## Chapter 4 Exercises

Time to deploy the heavy artillery.

### Exercise 4.1: Tree Extrapolation Experiment
Write a script to test the extrapolation limit of tree models:
1. Load `data/kp_agricultural_yields.csv`.
2. Split the data chronologically (or by elevation):
   * Train set: farms at elevation < 1500 meters.
   * Test set: farms at elevation $\ge$ 1500 meters.
3. Train a `LinearRegression` and a `RandomForestRegressor(max_depth=10, random_state=42)` using only `elevation` to predict `crop_yield_tons_per_acre`.
4. Predict on the test set and plot predictions. 
5. Explain why the Random Forest predictions flatline above 1500 meters while the Linear model does not.

**Helpful Comments:**
* Plot using a scatter plot of `elevation` vs `crop_yield_tons_per_acre`, and overlay the prediction lines for both models.
* *Comic Relief:* If your Random Forest predictions look like a horizontal shelf at high elevation, congratulations! You have witnessed the tree model's inability to think outside its training box. Remember: trees are great inside their boundaries, but they are terrible tourists.

### Exercise 4.2: Target Transformation with Skewed Costs
On the infrastructure projects dataset:
1. Train a `HistGradientBoostingRegressor` to predict `actual_cost_million_pkr` using a random split.
2. Train another `HistGradientBoostingRegressor` wrapped in a `TransformedTargetRegressor` using `np.log1p` and `np.expm1`.
3. Compute MAE, RMSE, and RMSLE (Root Mean Squared Log Error) on the test set.
4. Plot the residual scatter plots for both models. Compare.

**Helpful Comments:**
* Use `mean_squared_log_error` from `sklearn.metrics` for RMSLE.

### Exercise 4.3: Tuning and Explaining with Optuna and SHAP
Using the agricultural dataset:
1. Write an Optuna study to tune `max_depth`, `learning_rate`, and `n_estimators` for a `LightGBM` (or `HistGradientBoostingRegressor`) model. Use 3-fold cross-validation.
2. Train the best model on the training set.
3. Compute SHAP values for 100 random test samples (`shap.TreeExplainer` or `shap.Explainer`).
4. Generate a summary plot of SHAP values and explain the top three most important features.

**Helpful Comments:**
* Install SHAP using `pip install shap` if you haven't already.
* Verify if the SHAP dependence plot for `soil_ph` matches the optimal range curve.

---

## References

1. **The Elements of Statistical Learning (ESL):** Hastie, Tibshirani, and Friedman. Chapters 9 (Additive Models, Trees) and 10 (Boosting). [Free PDF](https://hastie.su.domains/ElemStatLearn/) - The ultimate reference for tree algorithms.
2. **SHAP Documentation:** [SHAP GitHub & Docs](https://shap.readthedocs.io/) - Read the tutorials to understand Shapley values.
3. **Optuna Documentation:** [Optuna Guide](https://optuna.readthedocs.io/) - Learn how to define search spaces.
4. **Grinsztajn et al. (2022), "Why do tree-based models still outperform deep learning on tabular data?":** [Link to Paper](https://arxiv.org/abs/2207.08815) - A modern research paper demonstrating why trees remain the kings of tabular data regression.
