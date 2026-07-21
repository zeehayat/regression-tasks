# Chapter 3 Companion Guide: From a Trustworthy Baseline to a Research-Grade Regression Study
## A First-Principles Mathematical & Computational Reference Manual

This companion guide is an exhaustive, first-principles reference manual accompanying **Chapter 3: From a Trustworthy Baseline to a Research-Grade Regression Study** [cite: 18]. It unpacks mathematical derivations, feature engineering hypotheses, advanced regularized regression ($L_1$/$L_2$), robust diagnostic methodologies, and the mechanics of a pre-specified research benchmark for the Microhydro Power (MHP) Cost Estimator [cite: 18].

All mathematical notation is strictly formatted using standard LaTeX delimiters (`$inline$` and `$$display$$`) for seamless rendering.

---

## Table of Contents
1. **Day 13: Preprocessing Is Part of the Model**
2. **Day 14: Feature Engineering as Model Specification**
3. **Day 15: Ridge Regression: Shrinkage and Stability**
4. **Day 16: Lasso, Elastic Net, and Sparse Models**
5. **Day 17: Multicollinearity, Leverage, and Influence**
6. **Day 18: Heteroskedasticity and Dependent Data**
7. **Day 19: Robust and Quantile Regression**
8. **Day 20: A Pre-Specified Regression Benchmark**
9. **Master Rosetta Stone & Formula Sheet**

---

## 1. Day 13: Preprocessing Is Part of the Model

Preprocessing (handling missing data and categorical variables) is not neutral "cleaning"; it mathematically alters the distribution of the data [cite: 18]. Therefore, preprocessing must be learned exclusively inside the training folds of cross-validation to prevent leakage [cite: 18].

### Missing Data Mechanisms
* **MCAR (Missing Completely at Random):** Missingness is unrelated to the observed or missing data [cite: 18].
* **MAR (Missing at Random):** Missingness depends only on *observed* information (e.g., surveys from early years are missing more often) [cite: 18].
* **MNAR (Missing Not at Random):** Missingness depends on the unobserved value itself (e.g., remote projects omit survey distances because the distance was too difficult to establish) [cite: 18].

### Handling Missing Data
* **Complete-Case Analysis (Deletion):** Deleting rows with missing values can severely bias the model by excluding specific sub-populations (e.g., deleting all remote projects) [cite: 18].
* **Median Imputation + Missing Indicator:** Imputing missing values with the training median ($x_{ij}^{imp} = 	ext{median}(X_{j,train})$) changes the distribution (reduces variance) [cite: 18]. We add a binary missingness indicator ($M_{ij} = 1 - R_{ij}$) so the model can learn an offset for imputed rows [cite: 18].

### Categorical Encoding
* **One-Hot Encoding:** Creates binary indicators. One category must be dropped as the reference to prevent exact collinearity [cite: 18].
* **Target Encoding:** Replaces a category with the mean of the target variable for that category ($TE(c) = rac{1}{n_c}\sum_{i:C_i=c}y_i$) [cite: 18]. 
  * *Critical Rule:* Target encoding must use **cross-fitting** inside the training data to prevent same-row target leakage [cite: 18].

---

## 2. Day 14: Feature Engineering as Model Specification

Feature engineering alters the columns of the design matrix $X$. The regression remains *linear in parameters* ($eta$) even if the engineered feature represents a non-linear relationship (like $x^2$) [cite: 18].

### Feature Transformations
* **Polynomials (Quadratic):** $\hat{y} = eta_0 + eta_1x + eta_2x^2$ [cite: 18].
  * *Derivative Interpretation:* The slope is $rac{df(x)}{dx} = eta_1 + 2eta_2x$. The curve has a stationary point at $x^* = -rac{eta_1}{2eta_2}$ [cite: 18]. You must check if this turning point actually exists within the credible, observed data range [cite: 18].
* **Interactions:** $\hat{y} = eta_0 + eta_1x_1 + eta_2x_2 + eta_3x_1x_2$ [cite: 18].
  * The effect of $x_1$ now depends on the value of $x_2$: $rac{\partial\hat{y}}{\partial x_1} = eta_1 + eta_3x_2$ [cite: 18].
* **The Hierarchy Principle:** If a model includes a higher-order term (like $x^2$ or $x_1x_2$), it must generally retain the lower-order "main effects" ($x$, $x_1$, $x_2$) so the interaction's meaning is properly anchored [cite: 18].

---

## 3. Day 15: Ridge Regression: Shrinkage and Stability

When predictors are highly correlated, unpenalized OLS coefficients can become massive and unstable [cite: 18]. Ridge regression stabilizes this by adding a penalty on the squared Euclidean length of the parameter vector [cite: 18].

### The Ridge Objective and Solution
The objective function adds an $L_2$ penalty controlled by hyperparameter $\lambda$ [cite: 18]:
$$J_{	ext{ridge}}(eta) = \lVert y - Xeta Vert_2^2 + \lambda\lVertetaVert_2^2$$
Taking the derivative and setting it to zero yields the closed-form Ridge solution [cite: 18]:
$$\hat{eta}_{	ext{ridge}} = (X^	op X + \lambda I)^{-1}X^	op y$$

### Why It Works: SVD Shrinkage
Using Singular Value Decomposition ($X=U\Sigma V^	op$), the Ridge fitted values become [cite: 18]:
$$\hat{y}_{	ext{ridge}} = U\,\operatorname{diag}\left(rac{\sigma_j^2}{\sigma_j^2+\lambda}ight)U^	op y$$
* In standard OLS, a tiny singular value ($\sigma_j$) massively amplifies noise [cite: 18].
* Ridge applies a shrinkage factor $s_j(\lambda) = rac{\sigma_j^2}{\sigma_j^2+\lambda}$. Weak directions (small $\sigma$) are suppressed toward zero, stabilizing the predictions [cite: 18].

**Crucial Scaling Requirement:** Because the penalty applies to coefficient magnitude, all features must be standardized ($z = rac{x-\mu}{s}$) before fitting Ridge. Otherwise, features with small arbitrary units (like millimeters) will be unfairly penalized [cite: 18].

---

## 4. Day 16: Lasso, Elastic Net, and Sparse Models

### The Lasso Objective
Lasso replaces the $L_2$ penalty with an $L_1$ penalty (sum of absolute values) [cite: 18]:
$$J_{	ext{lasso}}(eta) = rac{1}{2n}\lVert y - Xeta Vert_2^2 + lpha\lVertetaVert_1$$

### Soft Thresholding and Exact Zeros
The $L_1$ penalty has sharp "corners" in its geometric constraint [cite: 18]. When optimized, coefficients can be pushed to **exactly zero** [cite: 18]. For orthonormal features, the coordinate solution is the soft-thresholding operator [cite: 18]:
$$\hat{eta}_j = \operatorname{sign}(z_j)\max(|z_j| - lpha, 0)$$
* **WARNING:** A zero lasso coefficient does *not* prove a feature has no scientific relationship with the target. It merely means the feature was dropped under this specific penalty and correlation structure [cite: 18].

### Elastic Net
Combines both penalties using a mixing ratio $ho$ [cite: 18]:
$$J_{	ext{EN}}(eta) = 	ext{Loss} + lphaho\lVertetaVert_1 + rac{lpha(1-ho)}{2}\lVertetaVert_2^2$$
This provides the exact sparsity of Lasso while maintaining the correlated-feature stability of Ridge [cite: 18].

---

## 5. Day 17: Multicollinearity, Leverage, and Influence

### Variance Inflation Factor (VIF)
VIF measures how much a feature's coefficient variance is inflated due to correlation with other features [cite: 18]:
$$\operatorname{VIF}_j = rac{1}{1 - R_j^2}$$
* A VIF of 9 means the standard error is inflated by a factor of $\sqrt{9} = 3$ [cite: 18]. 
* There is no universal "cutoff" for VIF. High VIF does not inherently bias predictions, it simply increases parameter uncertainty [cite: 18].

### Leverage vs. Influence
* **Leverage ($h_{ii}$):** Measures how unusual a row's predictor configuration is. Extracted from the diagonal of the Hat Matrix $H = X(X^	op X)^{-1}X^	op$ [cite: 18].
* **Influence (Cook's Distance $D_i$):** Measures how much the fitted model would change if observation $i$ were deleted [cite: 18]. It combines both the residual size and the leverage [cite: 18]:
  $$D_i = rac{e_i^2}{k s^2} rac{h_{ii}}{(1 - h_{ii})^2}$$

---

## 6. Day 18: Heteroskedasticity and Dependent Data

### Robust Covariance (HC0 - HC3)
If error variance changes across observations (Heteroskedasticity), standard OLS standard errors are incorrect [cite: 18]. The **Sandwich Estimator** corrects this [cite: 18]:
$$\operatorname{Var}(\hat{eta}) = (X^	op X)^{-1} X^	op \Omega X (X^	op X)^{-1}$$
* **HC3** adjusts the "meat" of the sandwich by heavily weighting squared residuals by their leverage [cite: 18]: $rac{e_i^2}{(1 - h_{ii})^2}$.
* *Limitation:* HC3 fixes variance estimation; it *does not* fix a misspecified mean model (e.g., missing a curved relationship) [cite: 18].

### Cluster-Robust Standard Errors
If projects in the same district share unobserved shocks, we must group errors by cluster $g$ [cite: 18]:
$$	ext{Meat} = \sum_{g=1}^G (X_g^	op e_g)(X_g^	op e_g)^	op$$
* *Limitation:* Asymptotic cluster confidence requires many independent clusters [cite: 18]. Five districts are too few for reliable large-$G$ asymptotic confidence [cite: 18].

---

## 7. Day 19: Robust and Quantile Regression

### Huber Regression (Robust to Outliers)
Squared loss highly penalizes outliers. **Huber Loss** transitions from quadratic (near zero) to linear (in the tails) beyond a threshold $\delta$ [cite: 18]:
$$L_\delta(r) = egin{cases} rac{1}{2}r^2 & |r| \le \delta \ \delta(|r| - rac{1}{2}\delta) & |r| > \delta \end{cases}$$
Huber downweights vertical outliers, preventing extreme cost values from pulling the entire regression line [cite: 18].

### Quantile Regression
OLS estimates the conditional *mean* [cite: 18]. Quantile regression uses asymmetric **Pinball Loss** to estimate a specific conditional quantile $	au$ [cite: 18]:
$$ho_	au(u) = egin{cases} 	au u & u \ge 0 \ (	au - 1)u & u < 0 \end{cases}$$
* If $	au = 0.80$, underpredicting the cost is penalized 4 times more than overpredicting [cite: 18]. This yields the 80th-percentile cost prediction, which is ideal for conservative contingency budgeting [cite: 18].

---

## 8. Day 20: A Pre-Specified Regression Benchmark

Exploratory modeling can easily turn noise into a publishable result [cite: 18]. To build a research-grade study, use a **Pre-Specified Prediction Contract** [cite: 18]:

1. **Temporal Boundary:** Validate only on data chronologically *before* the validation year [cite: 18].
2. **Locked Test Set:** The final test set (e.g., 2023-2025 projects) must not be touched until all feature engineering, imputation, and hyperparameter tuning is complete [cite: 18].
3. **Pipeline Encapsulation:** All missing-data handling, target encoding, standardizing, and modeling must occur inside a `Pipeline` to prevent train-test leakage [cite: 18].
4. **Subgroup Audits:** Overall MAE can hide failures. Always audit performance by operational subgroups (e.g., remote vs. accessible sites) [cite: 18].

---

## 9. Master Rosetta Stone & Formula Sheet

| Concept | Formula | Use |
|---|---|---|
| Standardisation | $z_{ij}=(x_{ij}-ar x_j)/s_j$ | Comparable penalty scale using training statistics [cite: 18] |
| Ridge objective | $\lVert y-XetaVert_2^2+\lambda\lVertetaVert_2^2$ | Stable $L_2$ shrinkage [cite: 18] |
| Ridge estimate | $(X^	op X+\lambda I)^{-1}X^	op y$ | Closed form for centred penalised slopes [cite: 18] |
| Ridge shrinkage | $\sigma_j^2/(\sigma_j^2+\lambda)$ | Fitted-value retention in singular direction $j$ [cite: 18] |
| Lasso objective | $rac{\lVert y-XetaVert_2^2}{2n}+lpha\lVertetaVert_1$ | Sparse shrinkage [cite: 18] |
| Soft threshold | $S(z,lpha)=\operatorname{sign}(z)(\lvert zvert-lpha)_+$ | One-coordinate lasso solution [cite: 18] |
| Elastic net | Loss $+lphaho\lVertetaVert_1+rac{lpha(1-ho)}{2}\lVertetaVert_2^2$ | Sparsity plus correlated-feature stabilisation [cite: 18] |
| VIF | $1/(1-R_j^2)$ | Conditional variance inflation for slope $j$ [cite: 18] |
| Hat matrix | $H=X(X^	op X)^{-1}X^	op$ | Maps outcomes to OLS fitted values [cite: 18] |
| Average leverage | $k/n$ | Reference leverage for $k$ design columns [cite: 18] |
| Studentised residual | $e_i/[s\sqrt{1-h_{ii}}]$ | Residual adjusted for its model variance [cite: 18] |
| Cook's distance | $e_i^2h_{ii}/[ks^2(1-h_{ii})^2]$ | Aggregate case influence [cite: 18] |
| General sandwich | $(X^	op X)^{-1}X^	op\Omega X(X^	op X)^{-1}$ | Coefficient covariance under general error covariance [cite: 18] |
| HC3 residual square | $e_i^2/(1-h_{ii})^2$ | Stronger leverage correction for heteroskedasticity [cite: 18] |
| Huber loss | quadratic inside $\delta$, linear outside | Reduce large-residual gradient [cite: 18] |
| Pinball loss | $u(	au-\mathbb1\{u<0\})$ | Proper loss for a conditional quantile $	au$ [cite: 18] |
