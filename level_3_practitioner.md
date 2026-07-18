# Chapter 3: Level 3 Practitioner — Preprocessing, Engineering, Diagnostics, and Regularization

Welcome to Chapter 3! This is the most crucial chapter of the guide. If you master this chapter, you will transition from a "beginner who feeds data to a black box and prays" to a "practitioner who designs robust preprocessing pipelines, diagnoses why a model is failing, and optimizes it using math."

We have a lot of ground to cover. We will discuss preprocessing pipelines, engineering new features, deep-diving into linear regression interpretation, running regression diagnostics, applying regularization (Ridge and Lasso), and performing systematic error analysis.

---

## 11. Data Preprocessing (Building Leakproof Pipelines)

Suppose someone collected data containing missing values, categorical labels (like district names), and numeric values of completely different scales (like population in the thousands and literacy rates as percentages). Now, they want to prepare this data for a regularized linear model. What do we do?

We build a preprocessing pipeline. Preprocessing is not just a cleaning step; it is part of the model itself.

### 11.1 Numeric Scaling
Linear models, Support Vector Machines, and KNN are highly sensitive to feature scales. If features are on different scales, the model will think the feature with the largest numbers (like `population`) is the most important.
* **StandardScaler:** Normalizes features to have mean 0 and standard deviation 1:
  $$z = \frac{x - \mu}{\sigma}$$
  Use this by default for most models.
* **MinMaxScaler:** Scales features to be between 0 and 1. Useful when you have hard boundaries.
* **RobustScaler:** Uses median and Interquartile Range (IQR) instead of mean and standard deviation. Use this if your data is full of outliers that you don't want to distort the scaling.

### 11.2 Categorical Encoding
Computers only understand numbers. If you feed the string `"Peshawar"` to a model, it will crash.
* **One-Hot Encoding:** Creates binary columns for each category (e.g., `is_Peshawar`, `is_Mardan`).
  * *Warning:* If you have 36 districts, one-hot encoding will add 36 new columns. This is fine for linear models but can slow down trees.
  * *The Dummy Variable Trap:* If you one-hot encode all categories, the columns will sum to 1. This creates perfect multicollinearity. To avoid this, set `drop='first'` in your encoder.
* **Target Encoding:** Replaces each category with the average target value for that category. Extremely powerful but highly prone to data leakage if not done inside cross-validation.

### 11.3 Missing Data Imputation
Real datasets have holes.
* **SimpleImputer:** Fills missing values with the mean, median, or most frequent value.
* **MissingIndicator:** Adds a binary column indicating if a value was missing. Sometimes, the fact that a value is missing is the most important feature!

> [!CRITICAL]
> **Pipeline Rule:** Always fit your transformers (scalers, encoders, imputers) on the **training set only**, and then use them to transform the validation and test sets. If you scale your entire dataset before splitting, the test set mean and variance will leak into your training set, leading to over-optimistic performance claims.

---

## 12. Feature Engineering

Suppose someone collected raw farm data ($X$, containing `soil_ph` and `fertilizer_used_bags_per_acre`). They want to find crop yield. They know that fertilizer helps, but applying too much burns the crops. They also know that fertilizer doesn't work if the soil is too acidic or dry. What do we do?

We engineer features using domain knowledge to capture these nonlinearities and interactions.

### 12.1 Interaction Terms
An interaction occurs when the effect of one feature depends on the value of another feature.
* **Formula:** $X_1 \times X_2$
* **Agriculture Example:** `fertilizer_used_bags_per_acre` $\times$ `irrigation_type`. Fertilizer is useless without water. An interaction term captures this multiplicative effect.

### 12.2 Polynomial Features
When the relationship curves, we add powers of the feature.
* **Formula:** $X^2$, $X^3$
* **Agriculture Example:** Soil pH has an optimal point. By adding `soil_ph` and `soil_ph` squared, we let the model fit a downward-curving parabola:
  $$\hat{y} = \beta_0 + \beta_1(\text{soil\_ph}) + \beta_2(\text{soil\_ph})^2$$
  If $\beta_2$ is negative, we have successfully captured the optimal pH curve!

### 12.3 Ratios and Log Transforms
* **Ratios:** `public_funding` / `population`. This represents funding *per person*, which is far more informative than raw funding.
* **Log Transforms:** $\log(1 + x)$. For right-skewed positive features (like `approved_cost` or `population`), a log transform reduces the range, bringing extreme values closer and making the distribution more Gaussian.

---

## 13. Linear Regression in Depth

Suppose we fit a linear model on the development dataset:
$$\text{development\_score} = 45.2 + 0.12(\text{road\_index}) - 0.35(\text{health\_dist}) + 1.5(\text{doctors}) + 2.1(\text{funding\_M\_PKR})$$

Let's interpret this.

### 13.1 Interpretation of Coefficients
* **Numeric features:** "Holding all other variables constant, a 1-unit increase in `public_funding` (1 million PKR) is associated with an average increase of 2.1 points in the `development_score`."
* **Standardized units:** If features were scaled using `StandardScaler`, a 1-unit increase means "one standard deviation increase." This allows you to directly compare coefficient magnitudes to see which feature is the strongest predictor.
* **Categorical reference categories:** If we drop the first category (e.g., `Plain` terrain) during one-hot encoding, it becomes our reference. A coefficient of $-5.0$ for `Mountainous` terrain means: "Mountainous communities score 5.0 points lower on average compared to Plain communities, holding all other features constant."

### 13.2 Multicollinearity and Variance Inflation Factor (VIF)
Multicollinearity occurs when features are highly correlated. In our development index dataset, `literacy_rate` and `school_enrollment_rate` carry identical information.
* **The Symptom:** Your coefficients will have massive standard errors. One run, `literacy_rate` has a coefficient of $+15.0$ and `school_enrollment` has $-12.0$. The next run, they swap. The model is confused because it can't tell which feature is doing the work.
* **Variance Inflation Factor (VIF):** Measures how much the variance of an estimated coefficient is increased due to collinearity.
  $$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
  where $R_j^2$ is the $R^2$ obtained by regressing feature $j$ on all other features.
  * **Rule of thumb:** VIF > 5 indicates moderate multicollinearity. VIF > 10 is a serious issue.

---

## 14. Assumptions and Diagnostics (How to Audit Your Model)

For a linear model's statistical inferences (like p-values and confidence intervals) to be valid, several assumptions must hold. You cannot just fit OLS and trust the summary. You must audit it.

### 14.1 Key OLS Assumptions
1. **Linearity:** The relationship between features and target is linear in the parameters.
2. **Independence:** Observations are independent of each other (no spatial or temporal correlation).
3. **Homoscedasticity:** The variance of the residuals is constant across all prediction levels.
4. **Exogeneity:** The expected value of the errors given the features is 0. (No omitted variables that affect both features and target).
5. **Normality of Errors:** Residuals are normally distributed (mainly required for small-sample t-tests).

### 14.2 Visual Diagnostics
To check these assumptions, we run diagnostics on our residuals $e_i = y_i - \hat{y}_i$:

* **Residuals vs. Fitted Plot:** Plot $\hat{y}$ on the x-axis and $e$ on the y-axis.
  * *Good:* A random cloud of points centered around 0.
  * *Bad (Curve):* Indicates missing nonlinearities (you need polynomials or splines).
  * *Bad (Funnel):* Indicates **Heteroscedasticity** (variance changes as predictions get larger). This is extremely common in project cost data, where large budgets have much larger absolute errors than small budgets.
* **Normal Q-Q Plot:** Compares residual quantiles against normal distribution quantiles. If points lie on a straight diagonal line, errors are normally distributed.
* **Influence and Leverage (Cook's Distance):** Measures how much your coefficients change if you delete a single observation. Mega-projects in infrastructure are high-leverage points; if you include them, they might pull the entire line towards them, ruining predictions for the other 99% of normal-sized projects.

---

## 15. Regularized Linear Models

When multicollinearity is high, or when you have too many features, OLS overfits or becomes unstable. To fix this, we add a penalty to the loss function.

```text
+-----------------------+------------------------------------------+------------------------------------------+
|      Model Type       |             L1 Penalty (Lasso)           |             L2 Penalty (Ridge)           |
+-----------------------+------------------------------------------+------------------------------------------+
| Penalty Expression    |           alpha * sum(|beta_j|)          |          alpha * sum(beta_j^2)           |
| Coefficient Behavior  | Can shrink coefficients exactly to zero. | Shrinks coefficients towards zero,       |
|                       | (Performs automatic feature selection).  | but keeps all of them.                   |
| Best Use Case         | When you want a sparse model with fewer  | When you have many correlated features   |
|                       | features.                                | that you want to keep.                   |
+-----------------------+------------------------------------------+------------------------------------------+
```

### 15.1 ElasticNet
ElasticNet combines both penalties:
$$\text{Loss} = \text{MSE} + \alpha \cdot \left( \lambda \sum | \beta_j | + \frac{1 - \lambda}{2} \sum \beta_j^2 \right)$$
where $\lambda$ (the `l1_ratio`) controls the balance between Lasso and Ridge.

---

## 25. Error Analysis and Model Debugging

Suppose you trained a model and got a validation MAE of 12.0. 
Don't stop there. Open the hood and look at the errors.

### 25.1 Subgroup Error Analysis
Calculate metrics for subgroups. In the infrastructure dataset, group by `project_sector` or `project_scale`. You might find:
* Road projects: MAE of 5M PKR.
* Energy projects: MAE of 80M PKR.
The model is failing on Energy! A global MAE of 12M hid this failure.

### 25.2 The Debugging Sequence
1. Check the residuals vs fitted plot.
2. Sort observations by largest absolute residual. Look at the top 10 worst predictions. What do they have in common? Are they all from mountainous areas? Are they all organic farms?
3. Add interaction terms or non-linear terms to target those specific errors.

---

## Chapter 3 Exercises

Let's act like real practitioners.

### Exercise 3.1: Preprocessing & Leakage Audit
Load the agricultural yields dataset. Write a pipeline using `scikit-learn`'s `Pipeline` and `ColumnTransformer` that:
1. Imputes missing values (if any) using median for numeric features, and most frequent for categoricals.
2. One-hot encodes `crop_type` and `irrigation_type`, dropping the first category to avoid the dummy variable trap.
3. Scales numeric features using `StandardScaler`.
4. Fits a `Ridge` regression.
5. Use `cross_validate` with 5-fold CV to evaluate the model. Ensure that scaling and encoding are performed *inside* the CV loop to prevent data leakage.

**Helpful Comments:**
* Your pipeline should look something like:
  ```python
  preprocessor = ColumnTransformer(transformers=[...])
  pipeline = Pipeline(steps=[('preprocess', preprocessor), ('model', Ridge())])
  ```

### Exercise 3.2: Feature Engineering Challenge
Using the agricultural dataset:
1. Create a polynomial feature for `soil_ph` (pH squared).
2. Create an interaction term between `fertilizer_used_bags_per_acre` and `irrigation_type`.
3. Train two Ridge models: one with these engineered features, and one without.
4. Compare validation $R^2$. Print the coefficients of the engineered model and interpret the coefficient of `soil_ph` vs `soil_ph^2`.

**Helpful Comments:**
* Check if the coefficient of `soil_ph^2` is negative. A negative coefficient confirms that yield decreases when pH is too high or too low, verifying the optimal pH curve built by `generate_data.py`.

### Exercise 3.3: Diagnosing OLS via statsmodels
Fit an OLS model using `statsmodels.api.OLS` on the development dataset to predict `development_score`.
1. Print `model.summary()`.
2. Find the VIF of `literacy_rate` and `school_enrollment_rate`.
3. Plot a residual vs fitted scatter plot. Look for any curvature or funnels.
4. Run a Breusch-Pagan test for heteroscedasticity (`statsmodels.stats.diagnostic.het_breuschpagan`).
5. Re-fit the model using HC3 robust standard errors (`cov_type='HC3'`). Compare the confidence intervals of your coefficients.

**Helpful Comments:**
* *Comic Relief:* If your literacy VIF is over 50, don't panic. The model is just letting you know that you passed it two copies of the same textbook. Drop one of them, and watch your standard errors drop.

---

## References

1. **Applied Predictive Modeling:** Max Kuhn and Kjell Johnson. Excellent chapters on data preprocessing and feature engineering.
2. **statsmodels Diagnostics:** [statsmodels Diagnostic Examples](https://www.statsmodels.org/stable/examples/notebooks/generated/regression_diagnostics.html) - Step-by-step code for regression checks.
3. **Breiman (2001), "Statistical Modeling: The Two Cultures":** [Link to Paper](https://doi.org/10.1214/ss/1009213726) - A legendary paper discussing OLS vs Machine Learning prediction paradigms. Essential reading for research-level work.
