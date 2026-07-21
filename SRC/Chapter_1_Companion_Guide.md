# Chapter 1 Companion Guide: From a Question to Ordinary Least Squares
## A First-Principles Mathematical & Computational Reference Manual

This companion guide is designed as an exhaustive, first-principles reference manual to accompany **Chapter 1: From a Question to Ordinary Least Squares** [cite: 12]. It unpacks the mathematical derivations, provides intuitive mental models, and breaks down the code for the Microhydro Power (MHP) Cost Estimator [cite: 12].

All mathematical notation is strictly formatted using standard LaTeX delimiters (`$inline$` and `$$display$$`) for seamless rendering.

---

## Table of Contents
1. **Day 1: Defining the Question (Prediction vs. Causation)**
2. **Day 2: Data Vocabulary and Matrix Shapes**
3. **Day 3: Linear Predictions and the Intercept**
4. **Day 4: Residuals, Loss, and OLS Geometry**
5. **Day 5: Calculus of OLS and Rank Failure**
6. **Master Rosetta Stone & Formula Sheet**

---

## 1. Day 1: Defining the Question (Prediction vs. Causation)

Regression simply constructs a numerical relationship between a target variable $y$ and observed features $X$ [cite: 12]. However, the *purpose* of the regression dictates how you interpret the results [cite: 12].

### The Three Jobs of Regression
* **Prediction:** Estimating an unknown outcome based on observed features [cite: 12].
  * *Success:* Low error on genuinely new projects [cite: 12].
  * *Danger:* The model may fail when external conditions change [cite: 12].
* **Explanation:** Describing the conditional association between variables [cite: 12].
  * *Success:* Stable, interpretable coefficients with measured uncertainty [cite: 12].
  * *Danger:* Omitted variables and correlated features can distort interpretation [cite: 12].
* **Causal Inference:** Estimating the effect of a specific intervention [cite: 12].
  * *Success:* A credible estimate comparing observed outcomes to an unobserved counterfactual [cite: 12].
  * *Danger:* Mistaking a mere association for the effect of a direct action [cite: 12].

### Confounding and Simpson's Paradox
* A **confounder** is a hidden common cause related to both the decision/exposure and the outcome [cite: 12].
* For example, rugged terrain might cause both the use of helicopter transport and higher cost overruns [cite: 12].
* Ignoring this confounder makes helicopters look like the cause of the expense, even if they actually saved money compared to manual labor [cite: 12].
* **Simpson's Paradox** occurs when an aggregated dataset shows one trend (e.g., a downward slope), but the trend reverses within specific subgroups (e.g., an upward slope for both accessible and remote terrain groups) [cite: 12].

---

## 2. Day 2: Data Vocabulary and Matrix Shapes

Before applying statistics, data must be correctly represented as scalars, vectors, matrices, or tensors [cite: 12]. 

### The Vocabulary of Regression
* **Observation:** One unit represented by a single row in the data (e.g., one completed MHP project) [cite: 12].
* **Feature:** An input measured *before* the target is known (e.g., planned capacity) [cite: 12].
* **Target:** The outcome variable to be estimated (e.g., actual project cost) [cite: 12].
* **Parameter:** A value learned from the data by the fitting algorithm (e.g., cost coefficient for road distance) [cite: 12].
* **Hyperparameter:** A setting chosen by the analyst outside the fitting calculation (e.g., whether to include an intercept) [cite: 12].

### Mathematical Array Shapes
* $X \in \mathbb{R}^{n 	imes p}$: A real-valued design matrix with $n$ rows (projects) and $p$ feature columns [cite: 12].
* $y \in \mathbb{R}^{n}$: A real-valued target vector with $n$ entries [cite: 12].
* $eta \in \mathbb{R}^{p}$: A parameter vector with one entry per feature [cite: 12].

**Shape Validation in Code:**
NumPy strictly distinguishes between shapes [cite: 12].
* `(3,)` is a 1D vector [cite: 12].
* `(3, 1)` is a 2D column matrix [cite: 12].
* `(1, 3)` is a 2D row matrix [cite: 12].

### Target Leakage
* Including a feature recorded *after* the outcome is known (e.g., "number of contract amendments" to predict final cost) is called **target leakage** [cite: 12].
* This creates a deceptively accurate model that cannot be used for early-stage prediction [cite: 12].

---

## 3. Day 3: Linear Predictions and the Intercept

### The Linear Equation
For a single project $i$ with a single feature, the linear model is [cite: 12]:
$$\hat{y}_i = eta_0 + eta_1x_i$$
* $eta_0$ is the **intercept**: the model's predicted target when $x_i = 0$ [cite: 12].
* $eta_1$ is the **slope**: the predicted change in $y$ associated with a one-unit increase in $x$ [cite: 12].

### Why We Add a Column of Ones
* To compute predictions using matrix multiplication ($\hat{y} = Xeta$), we prepend a column of $1$s to the design matrix $X$ [cite: 12].
* When multiplying $Xeta$, the parameter $eta_0$ is multiplied by $1$ and added to every project's prediction, effectively creating an affine transformation (translation + scaling) [cite: 12].
* Without this column of ones, the model is forced through the origin ($\hat{y} = 0$ when $x = 0$), which may not fit the observed baseline [cite: 12].

### Unit Scaling
* Changing a feature's units changes its coefficient, not the final prediction [cite: 12].
* If cable length is measured in meters instead of kilometers ($x_{metres} = 1000 x_{km}$), the parameter scales inversely ($eta_{metres} = rac{eta_{km}}{1000}$) [cite: 12].
* Therefore, comparing coefficient magnitudes across differently scaled features is misleading [cite: 12].

---

## 4. Day 4: Residuals, Loss, and OLS Geometry

### Residuals
A residual is the difference between the actual observed value and the model's prediction [cite: 12].
$$e_i = y_i - \hat{y}_i$$
* Positive residual ($e_i > 0$): The model underpredicted the actual cost [cite: 12].
* Negative residual ($e_i < 0$): The model overpredicted the actual cost [cite: 12].

### Error Metrics
* **Sum of Squared Residuals (SSR):** $SSR = \sum_{i=1}^{n} e_i^2 = e^T e$ [cite: 12].
  * *Why square?* It prevents positive and negative residuals from canceling out, heavily penalizes large outliers, and creates a smooth, differentiable surface [cite: 12].
* **Mean Squared Error (MSE):** $rac{1}{n} \sum e_i^2$ [cite: 12].
* **Root Mean Squared Error (RMSE):** $\sqrt{MSE}$ (Interpretable in the original target units) [cite: 12].
* **Mean Absolute Error (MAE):** $rac{1}{n} \sum |e_i|$ (More robust to massive outliers than RMSE) [cite: 12].

### The Geometric Proof of OLS
* **Goal:** OLS seeks the prediction vector $\hat{y}$ (in the column space of $X$) that is closest to the observed target $y$ [cite: 12].
* **Orthogonality:** The shortest distance from $y$ to the column space is a perpendicular drop. Therefore, the residual vector $e$ must be perpendicular to every column of $X$ [cite: 12].
  $$X^T e = 0$$
* **Pythagorean Theorem:** For any other prediction $q$ in the column space, let the displacement be $d = \hat{y} - q$ [cite: 12].
  * $y - q = e + d$ [cite: 12].
  * Because $e$ is perpendicular to $d$ ($e^T d = 0$), expanding the squared length yields: $\lVert y - q Vert_2^2 = \lVert e Vert_2^2 + \lVert d Vert_2^2$ [cite: 12].
  * Since $\lVert d Vert_2^2 \ge 0$, no alternative prediction $q$ can be closer to $y$ than $\hat{y}$ [cite: 12].

---

## 5. Day 5: Calculus of OLS and Rank Failure

While geometry tells us *what* OLS does, calculus tells us *how* to find the exact parameters [cite: 12].

### The Matrix Derivative
* The objective function is $S(eta) = (y - Xeta)^T(y - Xeta)$ [cite: 12].
* Expanded: $S(eta) = y^Ty - 2eta^TX^Ty + eta^TX^TXeta$ [cite: 12].
* Differentiating with respect to $eta$ yields the gradient: 
  $$
abla_eta S(eta) = -2X^Ty + 2X^TXeta$$
* Setting the gradient to zero at the minimum yields the **Normal Equations** [cite: 12]:
  $$X^TX\hat{eta} = X^Ty$$

### Fitting in Code: `lstsq` vs. Inverse
* Mathematically, if $X^TX$ is invertible, the solution is $\hat{eta} = (X^TX)^{-1}X^Ty$ [cite: 12].
* Computationally, explicitly calculating the inverse squares the condition number and causes numerical instability [cite: 12].
* **Best Practice:** Use `np.linalg.lstsq(X, y)` directly, which utilizes stable matrix decomposition rather than raw inversion [cite: 12].

### Perfect Multicollinearity (Rank Failure)
* If one column in the design matrix is an exact multiple of another (e.g., having both cable length in km and cable length in meters), the matrix loses **full column rank** [cite: 12].
* The model cannot uniquely identify the parameters because infinitely many coefficient pairs yield the exact same combined prediction [cite: 12].
* The solution is to remove the redundant feature from the matrix [cite: 12].

---

## 6. Master Rosetta Stone & Formula Sheet

### Conceptual Formulas

| Concept | Equation | Meaning |
|---|---|---|
| Linear prediction | $\hat{y}=Xeta$ | Weighted combination of design columns [cite: 12] |
| Residual | $e=y-\hat{y}$ | Actual minus predicted [cite: 12] |
| SSR | $e^Te=\sum_i e_i^2$ | Total squared training error [cite: 12] |
| MSE | $SSR/n$ | Mean squared error [cite: 12] |
| RMSE | $\sqrt{SSR/n}$ | Squared-error summary in target units [cite: 12] |
| MAE | $rac{1}{n}\sum_i|e_i|$ | Mean absolute miss [cite: 12] |
| Orthogonality | $X^Te=0$ | Residual perpendicular to every design column [cite: 12] |
| Normal equations | $X^TX\hat{eta}=X^Ty$ | First-order condition for OLS [cite: 12] |
| Closed form | $(X^TX)^{-1}X^Ty$ | Unique solution when $X$ has full column rank [cite: 12] |
| Scalar slope | $rac{\sum_i(x_i-ar{x})(y_i-ar{y})}{\sum_i(x_i-ar{x})^2}$ | One-feature slope with intercept [cite: 12] |
| Scalar intercept | $ar{y}-\hat{eta}_1ar{x}$ | Makes the line pass through the means [cite: 12] |
| Hessian | $2X^TX$ | Curvature of the SSR surface [cite: 12] |

### Python/NumPy Implementations

| Mathematics | Meaning | NumPy |
|---|---|---|
| $Xeta$ | Matrix–vector product | `X @ beta` [cite: 12] |
| $X^T$ | Transpose | `X.T` [cite: 12] |
| $x^Ty$ | Dot product | `x @ y` or `np.dot(x, y)` [cite: 12] |
| $I$ | Identity matrix | `np.eye(p)` [cite: 12] |
| $\mathbf{1}$ | Vector of ones | `np.ones(n)` [cite: 12] |
| $\lVert vVert_2$ | Euclidean length | `np.linalg.norm(v)` [cite: 12] |
