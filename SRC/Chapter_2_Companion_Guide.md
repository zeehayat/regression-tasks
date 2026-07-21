# Chapter 2 Companion Guide: From a Fitted Line to a Trustworthy Model

## A beginner-first guide to numerical stability, probability, optimisation, evaluation, and responsible model revision

This guide accompanies `chapter2.md`. It assumes only the material developed in Chapter 1, and it explains the new Python, mathematics, probability, statistics, and machine-learning tools from first principles.

Chapter 1 answered: “How do we construct and fit an ordinary least-squares line?” Chapter 2 asks the harder question:

> When should anyone trust that fitted line for a real decision?

Trust is not created by a low training error or a library call. It requires a chain of evidence:

1. the numerical calculation is stable;
2. probability claims are tied to explicit assumptions;
3. optimisation has genuinely converged;
4. evaluation data represent the intended deployment;
5. preprocessing and model selection do not leak information;
6. the model beats useful baselines on unseen cases;
7. average metrics do not conceal systematic failure; and
8. revisions are evaluated on new evidence.

The running case remains the fictional microhydro power (MHP) cost estimator for Khyber Pakhtunkhwa (KP). The target is final project cost in **constant 2025 million PKR**, predicted at technical appraisal before procurement and construction.

---

## How to use this guide

For each of Days 6–12:

1. Read the corresponding section here before the main chapter.
2. Type the smallest code example yourself.
3. Predict the output before running it.
4. Run the main chapter's longer proof and build code.
5. deliberately create the listed failure.
6. Explain the result in plain language and in the MHP context.
7. Complete the exit check without looking back.

Use four questions whenever you meet a new method:

- **What is learned?** Coefficients, means, scales, categories, or hyperparameters?
- **From which rows?** Training, validation, a cross-validation fold, or test?
- **What claim follows?** Computation, prediction, uncertainty, or causation?
- **What can break it?** Poor conditioning, wrong assumptions, leakage, shift, or small samples?

---

# 0. Foundations for an absolute beginner

## 0.1 What Chapter 1 already gave us

Ordinary least squares (OLS) chooses coefficients that minimise squared residuals:

$$
\hat\beta
=\arg\min_\beta\lVert y-X\beta\rVert_2^2.
$$

You should recall:

- $X$ is the design matrix;
- $y$ is the target vector;
- $\beta$ is a coefficient vector;
- $\hat y=X\hat\beta$ contains fitted predictions;
- $e=y-\hat y$ contains residuals; and
- full column rank gives a unique coefficient vector.

Chapter 2 does not discard this mathematics. It surrounds it with the computational and evidential practices required for responsible use.

## 0.2 The prediction-time rule

Imagine an appraisal officer on the day a cost estimate must be issued. A feature is eligible only if it is legitimately available then.

| Field | Available at appraisal? | Eligible? | Reason |
|---|---:|---:|---|
| Planned capacity | Yes | Yes | Comes from the proposed design |
| Estimated cable length | Yes | Yes | Comes from the survey |
| Road distance | Yes | Yes | Comes from the access assessment |
| Terrain index | Yes | Cautiously | Assessor consistency must be checked |
| District | Yes | Depends | Use depends on whether deployment includes unseen districts |
| Start year | Yes | Depends | May represent time change if used correctly |
| Final material bill | No | No | Created during construction; leaks the outcome |
| Actual completion time | No | No | It is a later outcome |

Constant-price cost removes general inflation from the target. It does not remove changes in procurement, technology, project mix, or data collection.

## 0.3 Installing the software

Chapter 2 adds pandas, SciPy, and scikit-learn:

```bash
python -m pip install numpy pandas matplotlib scipy scikit-learn
```

Record versions when an analysis must be reproducible:

```python
import numpy as np
import pandas as pd
import scipy
import sklearn

print(np.__version__)
print(pd.__version__)
print(scipy.__version__)
print(sklearn.__version__)
```

A reproducible result identifies its data recipe, code, random seeds, package versions, and evaluation procedure.

## 0.4 NumPy ideas used repeatedly

### Rows, columns, and `axis`

For an array with shape `(n, p)`:

```python
X.mean(axis=0)  # one mean for each column
X.mean(axis=1)  # one mean for each row
```

`axis=0` collapses the row direction and leaves one result per feature. Feature scaling therefore uses `axis=0`.

### Boolean masks

```python
later = start_year >= 2023
X_test = X[later]
```

The comparison creates `True` or `False` for every row. Indexing with it retains the `True` rows.

### Random-number generators and seeds

```python
rng = np.random.default_rng(2026)
noise = rng.normal(0.0, 2.0, size=100)
```

A seed makes a pseudo-random sequence reproducible. It does not make one random split scientifically correct or remove sampling uncertainty.

### Useful linear-algebra functions

```python
np.linalg.cond(X)                    # condition number
np.linalg.matrix_rank(X)             # numerical rank
np.linalg.qr(X, mode="reduced")      # QR decomposition
np.linalg.svd(X, full_matrices=False) # singular value decomposition
np.linalg.lstsq(X, y, rcond=None)    # least-squares solution
```

## 0.5 A pandas survival kit

A pandas `DataFrame` is a labeled table. A `Series` is one labeled column.

```python
df = make_mhp_projects()

print(df.head())
print(df.shape)

features = ["planned_capacity_kw", "road_distance_km"]
X = df[features]                         # DataFrame, two columns
y = df["actual_cost_2025_million_pkr"]  # Series, one column

train = df[df["start_year"] <= 2021].copy()
```

Important operations:

- `df[columns]` selects named columns;
- `df.loc[row_labels, columns]` selects by labels;
- `df.iloc[row_positions]` selects by integer positions;
- `sort_values` orders rows;
- `groupby` calculates summaries within groups;
- `pd.concat` combines tables; and
- `.copy()` makes an explicit independent table for modification.

Pandas labels help preserve meaning. Converting too early to raw arrays can hide a swapped-column or row-alignment error.

## 0.6 Understanding the chapter's data generator

`make_mhp_projects(n=360, seed=2026)` creates a reproducible teaching dataset rather than loading confidential project records. Its structure deliberately gives every later diagnostic something meaningful to discover.

The generator:

1. samples districts and start years;
2. creates appraisal-stage capacity, road distance, terrain, and cable features;
3. adds district-specific cost differences;
4. adds a threshold surcharge when road distance exceeds 18 km;
5. adds a curved capacity contribution using capacity squared;
6. adds a time trend;
7. makes noise larger for more remote projects;
8. constructs actual constant-price cost; and
9. creates a tempting final material bill from actual cost after construction.

These choices introduce:

- **nonlinearity:** one straight line cannot exactly represent the squared-capacity term or road threshold;
- **group structure:** districts have shared differences;
- **temporal change:** later years have systematically changed cost;
- **heteroskedasticity:** outcome noise grows with road distance; and
- **target leakage:** final material bill contains later information strongly related to cost.

Important generator expressions:

```python
rng = np.random.default_rng(seed)
road_distance_km = np.clip(values, 0.2, 42.0)
remote_surcharge = 7.0 * (road_distance_km > 18.0)
capacity_curve = 0.000025 * planned_capacity_kw ** 2
noise = rng.normal(0.0, noise_sd)
```

`np.clip` limits values to a stated range. Multiplying a Boolean array by `7.0` turns `False` into 0 and `True` into 7. A vector `noise_sd` gives each project its own Gaussian standard deviation.

The final `sort_values(["start_year", "project_id"])` makes chronological analysis convenient. `reset_index(drop=True)` replaces the old row labels after sorting. Saving CSV with `index=False` avoids writing a meaningless extra index column.

Knowing the data-generating recipe helps interpret this exercise. In real analysis, that recipe is unknown; diagnostics and domain evidence are used to learn where a proposed model is inadequate.

## 0.7 The scikit-learn pattern

Most scikit-learn objects follow one interface:

```python
model.fit(X_train, y_train)
predictions = model.predict(X_new)
```

Transformers use:

```python
transformer.fit(X_train)
X_train_changed = transformer.transform(X_train)
X_new_changed = transformer.transform(X_new)
```

`fit` learns from data. `transform` applies what was learned. `fit_transform` combines them for training data. Calling `fit` on test data would let the test set teach the procedure.

Other tools introduced in the chapter:

- `Pipeline`: chains transformations and a model within one data boundary;
- `ColumnTransformer`: applies different transformations to different columns;
- `SimpleImputer`: learns replacement values for missing data;
- `OneHotEncoder`: turns categories into indicator columns;
- `StandardScaler`: learns means and scales;
- `clone`: makes a fresh unfitted copy of a procedure;
- `GridSearchCV`: searches hyperparameters using cross-validation; and
- `DummyRegressor`: supplies a simple baseline.

## 0.8 Mathematical and probability language

### Logarithms and exponentials

$\exp(a)=e^a$ is an exponential. $\log$ is its inverse. Useful rules are:

$$
\log(ab)=\log a+\log b,
\qquad
\log(a^b)=b\log a.
$$

These rules turn a product of many probability densities into a sum of log densities.

### Random variables, expectation, and variance

A random variable represents a numerical outcome before it is observed. Uppercase $Y$ often denotes the random variable and lowercase $y$ its realised value.

Expectation $\mathbb E[Y]$ is a probability-weighted long-run mean. Variance measures spread around that mean:

$$
\operatorname{Var}(Y)=\mathbb E[(Y-\mathbb E[Y])^2].
$$

Standard deviation is the square root of variance and returns to the original units.

### Distributions and density

A continuous probability density describes relative concentration. Probability over an interval is area under the density:

$$
P(a\le Y\le b)=\int_a^b f(y)\,dy.
$$

The integral symbol $\int$ is continuous addition. You do not need to calculate these integrals manually in this chapter.

### Samples and sampling variability

A dataset is one sample from a wider process. If we could repeatedly collect new datasets and refit the model, estimates would change. This repeated-sample variation is the basis of standard errors and frequentist confidence intervals.

### Quantiles

The $q$ quantile is a value at or below which proportion $q$ of observations lies. The median is the 0.5 quantile. A 90th percentile absolute error is a threshold at or below which 90% of observed absolute errors fall.

---

# Day 6 — Scaling and numerical conditioning

## 6.1 Floating-point numbers are approximations

Computers store a finite number of binary digits. Many decimal fractions have no finite binary representation:

```python
print(0.1 + 0.2)
print((0.1 + 0.2) == 0.3)
```

The first result is extremely close to 0.3; the exact comparison is false. Use `np.isclose` or `np.allclose` for computed floats.

Finite precision is manageable when algorithms avoid magnifying tiny errors unnecessarily.

## 6.2 Conditioning and numerical stability are different

An **ill-conditioned problem** turns a tiny input change into a large solution change. **Numerical stability** describes whether an algorithm adds avoidable error while solving the problem.

- Conditioning belongs to the problem and its representation.
- Stability belongs to the computational method.
- A stable solver cannot invent information missing from nearly duplicate features.

For a full-column-rank matrix:

$$
\kappa_2(X)=\frac{\sigma_{\max}(X)}{\sigma_{\min}(X)}.
$$

$\sigma_{\max}$ and $\sigma_{\min}$ are the largest and smallest singular values. A ratio near 1 is favourable. A large ratio means some coefficient directions affect predictions far less than others, so noise or rounding can create large coefficient changes.

There is no universal pass/fail threshold. Precision, scale, noise, intended claim, and consequences matter.

### Full rank is not good conditioning

A matrix can have no exactly redundant columns yet contain two almost identical columns. It is full rank in exact or tolerance-based arithmetic but can be badly conditioned. Rank answers “how many directions?” Conditioning asks “how unevenly are those directions represented?”

## 6.3 Why differing units cause imbalance

Capacity may be hundreds of kW, road distance tens of km, and terrain only 1–5. Their columns live on different numerical scales.

Standardisation transforms entry $i,j$:

$$
z_{ij}=\frac{x_{ij}-\mu_j}{s_j},
$$

where $\mu_j$ and $s_j$ come from the training column.

- $z=0$: at the training mean;
- $z=1$: one training standard deviation above it;
- $z=-2$: two training standard deviations below it.

Scaling changes representation, not information. It does not guarantee generalisation and cannot repair redundancy.

## 6.4 Build a scaler from scratch

The chapter's `StandardScalerFromScratch` teaches four distinct operations:

1. `fit` validates a 2D matrix and learns one mean and scale per column;
2. `transform` subtracts stored means and divides by stored scales;
3. `inverse_transform` reconstructs original units; and
4. `fit_transform` fits and then transforms the training rows.

```python
self.mean_ = X.mean(axis=0)
self.scale_ = X.std(axis=0, ddof=0)
Z = (X - self.mean_) / self.scale_
```

NumPy broadcasts the `(p,)` means and scales across all `n` rows. A constant feature has zero standard deviation, making division impossible. The scaler rejects it rather than producing infinities or `NaN`.

### Population versus sample standard deviation

$$
s_{population}=\sqrt{\frac1n\sum_i(x_i-\bar x)^2},
$$

$$
s_{sample}=\sqrt{\frac1{n-1}\sum_i(x_i-\bar x)^2}.
$$

`ddof=0` uses $n$ and matches scikit-learn's `StandardScaler`. `ddof=1` uses $n-1$, common when estimating a population variance. The correct denominator depends on purpose; it is not one rule for all statistics.

## 6.5 Scaling must respect the information boundary

Correct procedure:

```python
scaler.fit(X_train)
Z_train = scaler.transform(X_train)
Z_validation = scaler.transform(X_validation)
Z_test = scaler.transform(X_test)
```

Incorrect procedure:

```python
scaler.fit(np.vstack([X_train, X_test]))
```

The incorrect version learns the future test distribution. Test values may look less extreme only because the test set has already influenced preprocessing. Means, scales, imputations, categories, and feature-selection decisions are learned parameters and must be fitted only on permitted training rows.

## 6.6 Scaling changes coefficient units, not fitted OLS predictions

For scaled coefficients $\gamma$:

$$
\hat y=\gamma_0+\sum_j\gamma_j\frac{x_j-\mu_j}{s_j}.
$$

Rearrangement gives raw-unit coefficients:

$$
\beta_j=\frac{\gamma_j}{s_j},
$$

$$
\beta_0=\gamma_0-\sum_j\frac{\gamma_j\mu_j}{s_j}.
$$

With an intercept and the same unpenalised OLS feature space, raw and standardised fits can produce the same predictions. Coefficient numbers and units change because one scaled unit means one training standard deviation.

This equivalence may not hold unchanged for penalised models, constrained optimisation, finite gradient-descent iterations, or preprocessing that changes information.

## 6.7 Condition number before and after scaling

Compare the whole design including the intercept:

```python
raw_design = np.column_stack([np.ones(X.shape[0]), X])
scaled_design = np.column_stack([np.ones(Z.shape[0]), Z])

print(np.linalg.cond(raw_design))
print(np.linalg.cond(scaled_design))
```

Scaling often improves unit-driven imbalance. If a feature is an almost exact copy of another, the smallest singular value may remain tiny.

## 6.8 Failure laboratory

- Add capacity in watts and kW: exact redundancy and rank loss.
- Add an almost identical road-time proxy: poor conditioning despite possible full rank.
- Add a constant feature: zero scaling denominator and redundancy with the intercept.
- Fit the scaler on all years: preprocessing leakage.

For each failure, distinguish computational symptoms, information content, and invalid evaluation claims.

## 6.9 Which methods benefit from scaling?

| Method | Usually scale? | Reason |
|---|---:|---|
| OLS with stable decomposition | Often | Conditioning and coefficient comparison |
| Gradient descent | Yes | Balances curvature and learning-rate behaviour |
| Ridge/lasso | Yes | Penalties act on coefficient magnitude |
| k-nearest neighbours | Yes | Distance depends on units |
| Support vector machines | Usually | Margins and kernels depend on scale |
| Decision trees | Usually unnecessary | Splits depend on order and thresholds |

Any learned transformation must still be fitted inside the training partition or cross-validation fold.

## Day 6 exit check

You are ready when you can explain:

1. why full rank does not guarantee stable coefficients;
2. why scaling improves representation without adding information;
3. which quantities a scaler learns; and
4. why test data cannot influence them.

---

# Day 7 — QR, SVD, rank, and the pseudoinverse

## 7.1 Why not form the normal equations?

The normal equations are correct:

$$
X^TX\hat\beta=X^Ty.
$$

But, for full-column-rank $X$:

$$
\kappa_2(X^TX)=\kappa_2(X)^2.
$$

A condition number of $10^6$ becomes roughly $10^{12}$. QR and SVD work with $X$ directly and avoid this unnecessary magnification.

## 7.2 QR decomposition

For $n\geq p$ and full column rank, reduced QR writes:

$$
X=QR,
$$

where:

- $Q$ has shape $n\times p$;
- $Q^TQ=I_p$;
- $R$ has shape $p\times p$; and
- $R$ is upper triangular, meaning entries below its diagonal are zero.

Substitute into least squares. The fitted projection is built from $Q$, and coefficients satisfy:

$$
R\hat\beta=Q^Ty.
$$

Solve this triangular system by back substitution:

```python
Q, R = np.linalg.qr(X, mode="reduced")
beta_qr = np.linalg.solve(R, Q.T @ y)
```

Do not invert $R$ explicitly.

## 7.3 Orthonormal means perpendicular and unit length

For distinct columns $q_j,q_k$:

$$
q_j^Tq_k=0.
$$

For each column:

$$
q_j^Tq_j=1.
$$

Thus the columns of $Q$ are clean perpendicular unit directions spanning the same column space as $X$. $Q^Ty$ measures how much of $y$ lies along each direction.

Check numerically:

```python
assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]))
```

Basic QR without pivoting is most straightforward for full-rank problems. Rank-revealing QR uses additional column pivoting.

## 7.4 Singular value decomposition

The reduced SVD is:

$$
X=U\Sigma V^T.
$$

- $V^T$ rotates parameter coordinates;
- diagonal $\Sigma$ stretches each direction by a singular value; and
- $U$ rotates the result into observation space.

$U$ and $V$ contain orthonormal directions. Singular values are nonnegative and usually ordered largest to smallest.

A tiny singular value means movement along one parameter direction barely changes predictions. The data provide little information for distinguishing coefficients in that direction.

## 7.5 The Moore–Penrose pseudoinverse

For positive singular values:

$$
X^+=V\Sigma^+U^T,
$$

where $\Sigma^+$ replaces retained singular values $\sigma_j$ with $1/\sigma_j$ and leaves discarded directions at zero.

Then:

$$
\hat\beta=X^+y=V\Sigma^+U^Ty.
$$

If multiple coefficient vectors minimise residual length, the pseudoinverse returns the one with minimum Euclidean coefficient norm. This precise convention does not make redundant coefficients separately identifiable.

## 7.6 Read the from-scratch SVD solver

The chapter's solver:

1. computes `U, singular_values, Vt`;
2. chooses a tolerance;
3. marks singular values above tolerance as retained;
4. safely inverts only retained values;
5. calculates $V\Sigma^+U^Ty$; and
6. reports numerical rank and the full singular-value spectrum.

Default tolerance:

```python
rcond = np.finfo(float).eps * max(X.shape)
tolerance = rcond * singular_values[0]
keep = singular_values > tolerance
```

`np.finfo(float).eps` is machine epsilon, a scale for floating-point resolution near 1. Multiplying by matrix size and the largest singular value creates a scale-aware numerical rule.

## 7.7 Numerical rank depends on tolerance

Exact mathematics distinguishes zero from nonzero. Floating-point computation must decide whether a tiny value represents information or numerical residue.

Changing tolerance can change reported rank and the pseudoinverse solution. When rank matters, record:

- feature scaling;
- singular values;
- tolerance or `rcond` rule;
- software version; and
- sensitivity of predictions and coefficients.

## 7.8 Compare computational routes

| Route | Calculation | Strength | Caution |
|---|---|---|---|
| Normal equations | Solve $X^TX\beta=X^Ty$ | Direct link to derivation | Squares condition number |
| QR | Solve $R\beta=Q^Ty$ | Stable and efficient for full rank | Rank deficiency needs care |
| SVD | Use $V\Sigma^+U^Ty$ | Reveals rank and weak directions | More computational work |
| `lstsq` | Library least-squares routine | Appropriate default | Diagnostics still require interpretation |

Agreement of predictions across routes is a useful code check. Coefficients can disagree in a rank-deficient problem because multiple solutions produce the same projection.

## 7.9 Singular-value spectrum

Plot singular values on a logarithmic vertical axis. A large drop to a tiny final value reveals a weak direction that might be invisible on a linear scale.

The spectrum is diagnostic, not an automatic deletion rule. Investigate whether weakness arises from duplicated units, feature construction, narrow sampling, or a meaningful but poorly represented comparison.

## 7.10 `StableOLS`

The chapter's deterministic estimator stores:

- fitted parameters;
- the residual-sum array returned by `lstsq`;
- numerical rank;
- singular values;
- number of parameters;
- condition number; and
- whether full column rank holds.

This separates two outputs:

> A solver returned a prediction-minimising vector.

from:

> Every individual coefficient is uniquely identified and stable enough to interpret.

The first can be true while the second is false.

## Day 7 exit check

Complete:

$$
X=QR\Rightarrow R\hat\beta=Q^Ty,
$$

$$
X=U\Sigma V^T\Rightarrow\hat\beta=V\Sigma^+U^Ty.
$$

Manager-friendly explanation of a tiny singular value: “The data barely distinguish one combination of effects, so small data changes can produce large changes in the reported coefficients.”

---

# Day 8 — Probability, likelihood, and uncertainty

## 8.1 Deterministic OLS versus a stochastic model

OLS projection needs no probability distribution:

$$
\hat\beta=\arg\min_\beta\lVert y-X\beta\rVert_2^2.
$$

A statistical model adds claims about outcome variation:

$$
y=X\beta+\varepsilon.
$$

- $\beta$ is an unknown population parameter;
- $X\beta$ is the model's systematic conditional mean; and
- $\varepsilon$ represents unmodeled random influences.

“Error” here does not mean data-entry error.

## 8.2 Errors are not residuals

True error:

$$
\varepsilon_i=y_i-x_i^T\beta.
$$

Observed residual:

$$
e_i=y_i-x_i^T\hat\beta.
$$

We do not know true $\beta$, so we do not observe true errors. Residuals depend on the fitted sample and obey constraints such as summing to zero with an intercept. They are diagnostic proxies, not independent observations of the error distribution.

## 8.3 The Gaussian distribution

For mean $\mu$ and variance $\sigma^2$:

$$
f(y\mid\mu,\sigma^2)
=\frac1{\sqrt{2\pi\sigma^2}}
\exp\left[-\frac{(y-\mu)^2}{2\sigma^2}\right].
$$

- $\mu$ locates the centre;
- $\sigma$ controls spread;
- equal positive and negative deviations have equal density; and
- large deviations receive rapidly declining density.

A density value is not itself the probability of one exact continuous outcome. Probability is area over an interval.

## 8.4 The classical Gaussian linear model

$$
Y_i\mid X_i=x_i\sim\mathcal N(x_i^T\beta,\sigma^2),
$$

equivalently:

$$
\varepsilon\sim\mathcal N(0,\sigma^2I).
$$

This compact statement assumes:

1. the conditional mean is linear in chosen design columns;
2. conditional errors have mean zero;
3. conditional variance is constant;
4. errors are independent under the sampling design; and
5. Gaussian errors support the exact finite-sample results used here.

Running `LinearRegression` does not verify these assumptions.

## 8.5 Probability and likelihood reverse what is held fixed

Probability asks how data would vary for fixed parameter values. Likelihood holds the observed data fixed and compares parameter values:

$$
L(\beta,\sigma^2\mid X,y)
=\prod_{i=1}^n
\frac1{\sqrt{2\pi\sigma^2}}
\exp\left[-\frac{(y_i-x_i^T\beta)^2}{2\sigma^2}\right].
$$

Likelihood is not the probability that a fixed parameter value is true.

## 8.6 Log-likelihood connects Gaussian MLE to OLS

Products of many small densities can underflow. Logs turn products into sums without changing which positive likelihood is largest:

$$
\ell(\beta,\sigma^2)
=-\frac n2\log(2\pi)
-\frac n2\log(\sigma^2)
-\frac1{2\sigma^2}\sum_i(y_i-x_i^T\beta)^2.
$$

For fixed $\sigma^2$, only the final term depends on $\beta$. Maximising log-likelihood therefore minimises SSR.

$$
\boxed{\text{Gaussian maximum likelihood for }\beta=\text{OLS}.}
$$

This equivalence does not establish that the real errors are Gaussian.

## 8.7 Residual variance and degrees of freedom

After fitting $k$ parameters including the intercept:

$$
s^2=\frac{SSR}{n-k}.
$$

Fitting $k$ independent parameters imposes constraints and leaves $n-k$ residual degrees of freedom. Under the classical model, this is an unbiased estimator of $\sigma^2$. Gaussian maximum likelihood uses $SSR/n$; the criteria differ.

Residual standard deviation is $s=\sqrt{s^2}$ and has target units. It is not the same as a coefficient's standard error.

## 8.8 Coefficient sampling variability

Under classical assumptions and fixed $X$:

$$
\hat\beta\sim\mathcal N\left(\beta,\sigma^2(X^TX)^{-1}\right).
$$

Estimate covariance with:

$$
\widehat{\operatorname{Var}}(\hat\beta)=s^2(X^TX)^{-1}.
$$

Coefficient $j$'s standard error is the square root of covariance diagonal entry $j$.

Large standard errors can result from high outcome noise, small samples, narrow feature ranges, or multicollinearity. They measure sampling uncertainty under assumptions, not whether code syntax succeeded.

## 8.9 Coefficient confidence intervals

$$
\hat\beta_j\pm t_{1-\alpha/2,n-k}SE(\hat\beta_j).
$$

Frequentist interpretation: if the full sampling and interval procedure were repeated under its assumptions, proportion $1-\alpha$ of intervals would cover the fixed true parameter. It is not strictly a 95% posterior probability that this one fixed parameter lies in this one interval.

An interval's units match its coefficient. It does not make an association causal.

## 8.10 Mean-response confidence interval versus prediction interval

For design row $x_0$:

$$
\hat y_0=x_0^T\hat\beta.
$$

Mean-response standard error:

$$
SE_{mean}(x_0)=s\sqrt{x_0^T(X^TX)^{-1}x_0}.
$$

Individual prediction standard error:

$$
SE_{prediction}(x_0)
=s\sqrt{1+x_0^T(X^TX)^{-1}x_0}.
$$

The extra 1 represents individual project noise, so the prediction interval is wider. Both usually widen far from the centre of observed feature values because extrapolated means are less precisely estimated.

## 8.11 Read `GaussianOLS`

The chapter extends `StableOLS` to:

1. require full rank;
2. calculate residuals and SSR;
3. require positive residual degrees of freedom;
4. estimate residual variance;
5. calculate the classical covariance matrix and standard errors;
6. use Student's $t$ critical values; and
7. return coefficient, mean-response, and individual-prediction intervals.

`np.einsum("ij,jk,ik->i", ...)` efficiently calculates one quadratic form $x_i^TAx_i$ per new row. It supplies each row's leverage-like uncertainty factor.

The class uses an inverse after full-rank checking to show the textbook covariance formula. Production inference should use mature decomposition-based methods and covariance estimators appropriate to heteroskedastic or clustered data.

## 8.12 Assumptions map

| Claim | Required support |
|---|---|
| A least-squares solution was computed | Valid finite arrays and correct numerical method |
| Coefficients are unique | Full column rank |
| Predictions generalise | Evaluation representing deployment |
| Classical intervals have stated coverage | Mean, variance, dependence, distribution, and sampling assumptions |
| A coefficient is causal | A causal identification design |

Normal-looking residuals do not fix confounding. Heteroskedastic residuals do not mean the least-squares projection was computed incorrectly.

## 8.13 Heteroskedasticity

**Heteroskedasticity** means conditional variance changes across observations. In the synthetic data, remote projects have larger noise. A residual-versus-fitted plot may show a fan shape.

OLS may still estimate a useful conditional mean under suitable conditions, but equal-variance standard errors and constant-width uncertainty become questionable. Later work introduces heteroskedasticity-robust and cluster-aware inference.

## Day 8 exit check

Distinguish:

- parameter versus estimate;
- unobserved error versus fitted residual;
- probability versus likelihood;
- residual standard deviation versus coefficient standard error; and
- confidence interval for a mean versus prediction interval for one project.

---

# Day 9 — Gradient descent from the OLS gradient

## 9.1 Why learn an iterative method?

QR or SVD is generally preferable for moderate OLS. Gradient descent matters because the same optimisation pattern works for models without convenient direct solutions and for very large datasets.

“Iterative” does not mean “more accurate.” For convex OLS, a correct converged implementation should approach the same minimum as `lstsq`.

## 9.2 MSE objective and gradient

$$
J(\beta)=\frac1n\lVert y-X\beta\rVert_2^2,
$$

$$
\nabla_\beta J(\beta)=\frac2nX^T(X\beta-y).
$$

Shape check:

| Quantity | Shape |
|---|---:|
| $X$ | $n\times k$ |
| $\beta$ | $k$ |
| $X\beta-y$ | $n$ |
| gradient | $k$ |

The gradient has one component per parameter and points in the direction of steepest local increase.

## 9.3 Update rule

$$
\beta^{(t+1)}
=\beta^{(t)}-\eta\nabla J(\beta^{(t)}).
$$

- Minus means move downhill.
- $\eta$ is the learning rate.
- Too small means slow progress.
- Too large causes overshooting, oscillation, or divergence.
- A stopping rule decides when further changes are negligible.

In a one-parameter walk, print both coefficient and loss each step. Try learning rates `0.0001`, `0.05`, `0.2`, and `1.0`; recognising patterns matters more than seeing one successful run.

## 9.4 Why scaling helps gradient descent

Very different feature scales create a long narrow MSE valley. One learning rate zigzags across steep directions while moving slowly along shallow ones. Standardisation makes curvature more balanced and a useful learning rate easier to find.

Scaling does not fix a wrong gradient sign, leakage, nonlinearity, or an unrepresentative validation design.

## 9.5 Read `GradientDescentOLS`

The chapter's batch optimiser:

1. creates a design matrix;
2. starts all coefficients at zero;
3. calculates predictions, MSE, and the full-data gradient;
4. rejects non-finite loss or gradient;
5. stores loss history;
6. updates coefficients;
7. stops when coefficient movement is below tolerance; and
8. otherwise stops at `max_iter`.

The `for ... else` construct runs `else` only if the loop did not reach `break`. It records that maximum iterations were exhausted.

Stopping because parameter movement is small is not automatically proof of the correct optimum. Also inspect gradient size, loss history, and agreement with a trusted direct solver.

## 9.6 Verify against SVD and finite differences

Two independent checks are essential:

- fitted coefficients and predictions approach `StableOLS`/`lstsq` on scaled data;
- analytic gradient agrees with a centred finite-difference approximation.

For component $j$:

$$
\frac{\partial J}{\partial\beta_j}
\approx\frac{J(\beta+hu_j)-J(\beta-hu_j)}{2h}.
$$

A smooth falling loss is not sufficient—a wrong gradient can still look orderly.

## 9.7 Convergence patterns

| Loss history | Likely issue | Response |
|---|---|---|
| Smooth decline to plateau | Convergence | Compare with direct solver and gradient norm |
| Very slow decline | Small rate or poor scaling | Scale and tune rate |
| Oscillation | Rate too large | Reduce rate |
| Explosion or non-finite values | Divergence or sign/scale bug | Stop and audit |
| Training improves, validation worsens | Overfit or shift | Evaluation problem, not optimiser success |

A logarithmic y-axis helps show changes across many orders of magnitude.

## 9.8 Batch, stochastic, and mini-batch descent

- **Batch:** uses every training row for every update.
- **Stochastic:** uses one randomly chosen row.
- **Mini-batch:** uses a small subset.

Stochastic methods are cheaper per update but noisy. They require decisions about shuffling, batch size, learning-rate schedules, seeds, and stopping. Chapter 2 uses batch descent so code stays directly aligned with the equation.

## 9.9 Parameters and hyperparameters

| Quantity | Type | How obtained |
|---|---|---|
| Intercept and slopes | Parameters | Learned by minimising training loss |
| Scaler means and scales | Preprocessing parameters | Learned from training features |
| Learning rate | Hyperparameter | Chosen outside coefficient updates |
| Maximum iterations | Hyperparameter | Chosen before fitting |
| Tree depth | Hyperparameter | Selected using development evidence |

When validation results guide a learning rate, validation has participated in model selection. The final test must remain untouched.

## Day 9 exit check

Write the gradient and update rule from memory. Then explain why equal minimum training MSE does not demonstrate equal future-project performance.

---

# Day 10 — Generalisation, baselines, and model complexity

## 10.1 The full learning procedure

$$
\hat f=\mathcal A(D_{train}).
$$

$\mathcal A$ includes feature definitions, cleaning, imputation, scaling, encoding, model family, hyperparameter search, random seeds, and selection rules. Honest evaluation must reproduce the entire procedure using only permitted information.

The named algorithm alone is not the unit of comparison.

## 10.2 Training risk versus deployment risk

Empirical training risk:

$$
\widehat R_{train}(f)
=\frac1{n_{train}}\sum_{i\in train}L(y_i,f(x_i)).
$$

Generalisation risk:

$$
R(f)=\mathbb E_{(X,Y)\sim P_{deployment}}[L(Y,f(X))].
$$

The deployment distribution is only partially observed. A held-out set estimates relevant performance only when its construction represents the cases, groups, and time periods of intended use.

## 10.3 Constant baselines are genuine models

For squared error:

$$
S(c)=\sum_i(y_i-c)^2.
$$

Setting the derivative to zero proves:

$$
\hat c=\bar y_{train}.
$$

The training mean is the best constant under squared loss. Under absolute loss, any training median minimises total absolute error because it balances observations on either side.

Fit the baseline on training targets only, then score it and candidate models on exactly the same held-out rows with exactly the same metric.

The mean is pulled more strongly by extreme values; the median is more resistant. Baseline and metric should match the decision loss.

## 10.4 What failure to beat baseline teaches

Possible causes include:

- weak features;
- incorrect row or column alignment;
- distribution shift;
- excessive flexibility;
- insufficient data; or
- an earlier leaky experiment that set unrealistic expectations.

The response is investigation, not hiding the baseline.

## 10.5 Underfitting and overfitting

- **Underfitting:** procedure is too restricted; training and validation errors are both high.
- **Overfitting:** procedure adapts to training-specific noise; training error is low but validation error is high.

Training error normally decreases as flexibility increases. Validation error may decrease and then rise. Choose complexity using validation evidence, not training victory.

## 10.6 Bias–variance decomposition

At fixed input $x_0$, assume:

$$
Y=f(x_0)+\varepsilon,
\quad\mathbb E[\varepsilon]=0,
\quad\operatorname{Var}(\varepsilon)=\sigma^2.
$$

Across repeated training datasets:

$$
\mathbb E[(Y-\hat f(x_0))^2]
=\left(\mathbb E[\hat f(x_0)]-f(x_0)\right)^2
+\operatorname{Var}(\hat f(x_0))
+\sigma^2.
$$

The terms are squared bias, procedure variance, and irreducible noise. This is a repeated-sampling identity under squared loss, not three quantities fully visible in one split. Do not label one model “high variance” solely from one graph.

## 10.7 A shallow decision tree

A regression tree repeatedly splits feature space. For feature $j$ and threshold $s$:

$$
R_L=\{x:x_j\leq s\},\qquad R_R=\{x:x_j>s\}.
$$

Under squared loss, each leaf predicts its training-target mean. A split minimises combined within-child SSR.

Trees can represent thresholds and interactions without manually specifying them. They are piecewise constant and do not linearly extrapolate beyond observed ranges.

- Depth 0: one constant leaf.
- Small depth: broad nonlinear structure.
- Large depth: small leaves and possible memorisation.

A shallow tree is a useful nonlinear comparator, not automatically the correct deployment model.

## 10.8 Read the first library comparison

The main chapter compares:

- `DummyRegressor(strategy="mean")`;
- `LinearRegression`;
- depth-3 `DecisionTreeRegressor`; and
- an unrestricted tree.

`random_state` makes the split or tree behaviour reproducible. The unrestricted tree may nearly eliminate training error. Only held-out error addresses generalisation.

The first random split demonstrates the API; Day 11 determines whether that split matches deployment.

## 10.9 Learning curves

A learning curve plots training and held-out performance as training size grows.

| Pattern | Possible lesson |
|---|---|
| Both errors stay high | Model or features may underfit |
| Large gap narrows | More data may reduce variance |
| Gap remains wide | Excess flexibility or structural leakage |
| Curves unstable | Too little data or unsuitable split |

One sampled subset per size makes a noisy curve. Repeat sampling or use deployment-appropriate cross-validation for stronger evidence.

## 10.10 Research discussion: Breiman's “Two Cultures”

Breiman contrasted:

- **data modeling:** specify a stochastic model and study its parameters;
- **algorithmic modeling:** treat the mechanism as partly unknown and prioritise predictive performance.

Beginner lessons:

1. An interpretable equation is not automatically an adequate description of reality.
2. Predictive accuracy does not automatically explain a mechanism or identify causation.
3. Prediction claims need out-of-sample evaluation.
4. Complete procedures, not mathematical elegance alone, should be compared.

For MHP work, a probabilistic linear model can communicate conditional relationships and uncertainty, while a shallow tree checks whether broad nonlinearity matters. Modern practice can combine these traditions; the paper's framing is a provocative argument, not a command to choose one camp forever.

Replication exercise: over many synthetic datasets, compare OLS and increasing tree depths under a fixed validation design. Plot training and validation MAE and identify where flexibility stops helping new cases.

## Day 10 exit check

Derive the mean baseline, explain why the median minimises absolute error, and give an example where the lowest-training-error model is the least responsible choice.

---

# Day 11 — Honest splitting, leakage, and cross-validation

## 11.1 Training, validation, and test roles

| Partition | Allowed | Not allowed |
|---|---|---|
| Training | Fit preprocessing and model parameters | Final performance claim |
| Validation | Select features, model family, and hyperparameters | Later call it untouched test data |
| Test | One final evaluation after procedure is fixed | Tuning or revising the procedure |

Typical workflow:

```text
train parameters
    -> validate/select procedure
    -> refit selected procedure on train + validation
    -> test once
```

Percentages such as 60/20/20 are examples, not laws. Time, groups, sample size, and deployment decide the split.

## 11.2 A split simulates deployment

### Random split

Appropriate for new, approximately independent and exchangeable observations under the same mixture of conditions. It is unsafe when related entities, time order, or deployment shifts matter.

### Group split

Keep related observations together. For unseen-district deployment:

$$
G_{train}\cap G_{test}=\varnothing.
$$

The group need not be a model feature; it identifies dependence and intended transfer.

### Temporal split

For future prediction:

$$
\max(t_{train})<\min(t_{test}).
$$

Forward-chaining trains on earlier history and validates on the next period. It exposes changing procurement, technology, measurement, or project mix.

### Combined new-time/new-group deployment

Future projects in unseen districts require both boundaries. If the dataset contains no analogue, a splitter cannot create evidence; external validation is needed.

## 11.3 Leakage catalogue

**Target leakage:** use target or post-outcome proxy, such as final material bill.

**Preprocessing leakage:** learn imputation, scaling, categories, transformations, or feature selection from validation/test rows.

**Temporal leakage:** allow future information to predict the past or use values recorded after prediction time.

**Group leakage:** related projects appear on both sides of an evaluation intended for new groups.

**Duplicate leakage:** exact or near-duplicate records cross partitions.

**Test-set leakage:** repeatedly inspect test scores and modify the procedure.

**Label leakage through cleaning:** use difficult target outcomes to decide which rows or feature values to remove.

Leakage is about illegitimate information flow, not merely whether a target column is visibly present.

## 11.4 Why the leaky feature looks spectacular

The synthetic final material bill is strongly based on actual cost and is known only later. Adding it dramatically lowers random-holdout MAE but answers a useless appraisal question: reconstruct final cost after much of it is already known.

High accuracy is not evidence of valid timing. Always perform the desk-at-prediction-time test.

## 11.5 Pipelines protect statistical validity

The numeric branch:

```text
numeric columns -> median imputation -> standardisation
```

The categorical branch:

```text
district -> most-frequent imputation -> one-hot encoding
```

`ColumnTransformer` combines them, then `Pipeline` fits the model. In every cross-validation split, the pipeline:

1. fits preprocessing only on the current training fold;
2. transforms that training fold;
3. fits the model;
4. transforms the validation fold using learned training values; and
5. predicts the validation fold.

`OneHotEncoder(handle_unknown="ignore")` prevents an unseen category from crashing transformation. It does not guarantee good predictions for that group. `drop="first"` removes one redundant category indicator when an intercept is present.

A pipeline is not just code tidiness. It defines the correctly bounded learning procedure.

## 11.6 K-fold cross-validation

Divide development data into $K$ folds. Each fold is held out once while the others train the procedure. Average held-out losses:

$$
\widehat L_{CV}=\frac1K\sum_{k=1}^K L_k.
$$

Every cross-validated prediction comes from a fitted procedure that did not train on that row. However, cross-validation estimates only the deployment question represented by its splitter.

Scikit-learn scoring names such as `neg_mean_absolute_error` return negative losses because its selection API treats larger scores as better. Negate them to report positive MAE.

## 11.7 Cross-validation is a family

| Splitter | Preserves | Question |
|---|---|---|
| `KFold` | Row separation | New independent-like observations |
| `GroupKFold` | Group separation | New groups |
| `TimeSeriesSplit` or forward loop | Time order | Future observations |
| Nested CV | Tuning/evaluation separation | Performance of a selection procedure |

Regression target binning for stratification does not replace time or group structure and must be documented carefully.

## 11.8 Grouped and temporal validation

With five district folds, report every district's MAE, RMSE, and $R^2$, not only their mean. One fold may reveal operational failure.

Temporal fold losses have different training sizes and represent different periods. They are not identical repeated measurements. The trend over years may be more informative than one average.

Use `clone(pipeline)` inside manual loops to ensure a fresh unfitted procedure for every fold.

## 11.9 Hyperparameter selection creates optimism

For depths in set $\mathcal D$:

$$
\hat d=\arg\min_{d\in\mathcal D}\widehat L_{validation}(d).
$$

The selected depth looks best partly because of genuine performance and partly because of validation noise. Reporting the same validation score as final is optimistically biased.

The more irrelevant choices tried, the more opportunities noise has to create an apparent winner.

## 11.10 Nested cross-validation

- **Inner loop:** selects hyperparameters using only the outer-training data.
- **Outer loop:** evaluates the entire inner-selection-and-refit procedure on untouched outer data.

For every outer split:

1. create outer train and outer test;
2. run an inner search entirely inside outer train;
3. refit the inner winner on all outer train;
4. predict outer test once; and
5. store outer performance.

The evaluated object is not a fixed depth chosen in advance. It is the complete procedure that searches a specified grid with a specified splitter, score, preprocessing pipeline, seed policy, and refitting rule.

Nested CV is expensive because selection is repeated. That repetition is the protection.

## 11.11 Research discussion: Cawley and Talbot

The paper asks whether a model-selection criterion itself can be overfit. Its key lesson is that selection noise can be comparable with reported differences between algorithms.

The unit of comparison should be:

$$
\text{learning algorithm}+\text{model-selection procedure}.
$$

“Decision tree” is incomplete. Specify candidate depths and leaf sizes, splitter, metric, preprocessing, search procedure, refit rule, and randomness.

Nested CV or a final untouched test addresses selection optimism under the chosen sampling design. It cannot fix a wrong deployment simulation, feature leakage, unmeasured distribution shift, or missing hard projects.

Replication exercise: as the number of irrelevant hyperparameter choices grows, compare the best inner score with outer performance across many synthetic datasets.

## 11.12 Locked-test protocol

Before opening test results, write and timestamp:

1. target and units;
2. prediction time;
3. eligible and prohibited features;
4. exclusion rules;
5. split design;
6. candidate procedures;
7. selection metric;
8. subgroup reports;
9. uncertainty method; and
10. deployment-blocking conditions.

If test results motivate revision, revision is legitimate but the test set has become development data. A fresh final claim needs new independent evidence.

## Day 11 exit check

Draw inner and outer nested-CV loops, state what each may influence, and explain why a pipeline is part of statistical validity.

---

# Day 12 — Metrics, diagnostics, uncertainty, and revision

## 12.1 Begin with signed error

$$
e_i=y_i-\hat y_i.
$$

- positive: actual cost is larger, so model underpredicted;
- negative: prediction is larger, so model overpredicted.

Mean error:

$$
ME=\frac1n\sum_i e_i.
$$

Opposite signs cancel. ME detects directional bias but cannot measure total accuracy.

## 12.2 MAE, MSE, RMSE, and median absolute error

$$
MAE=\frac1n\sum_i|e_i|,
$$

$$
MSE=\frac1n\sum_i e_i^2,
\qquad RMSE=\sqrt{MSE},
$$

$$
MedAE=\operatorname{median}(|e_i|).
$$

| Metric | Units | Sensitivity | Plain reading |
|---|---|---|---|
| ME | target | signs cancel | average direction of miss |
| MAE | target | linear penalty | average absolute miss |
| RMSE | target | emphasises large errors | square-root average squared miss |
| MedAE | target | highly resistant to a minority of extremes | central typical miss |

MAE is not immune to outliers; it is less dominated than RMSE. MedAE can look reassuring while a minority of projects fail catastrophically. Also report a high absolute-error quantile such as $Q_{0.90}(|e|)$.

## 12.3 Held-out $R^2$

$$
R^2
=1-\frac{\sum_i(y_i-\hat y_i)^2}
{\sum_i(y_i-\bar y_{evaluation})^2}.
$$

- 1: perfect predictions;
- 0: same SSR as predicting the evaluation-set mean;
- negative: worse than that evaluation-set-mean reference.

Standard held-out $R^2$ uses the evaluation targets' mean in its denominator. A deployable dummy predicts a value learned from training targets. Under shift these are not equal. Always score an explicit training-fitted baseline.

$R^2$ is dimensionless and depends on target variation. Identical absolute errors can yield different $R^2$ across districts with different cost ranges.

## 12.4 Why MAPE can be dangerous

$$
MAPE=\frac{100}{n}\sum_i\left|\frac{y_i-\hat y_i}{y_i}\right|.
$$

It is undefined at zero, explodes near zero, gives the same absolute miss more weight for small projects, and introduces asymmetries. Use it only when proportional error genuinely matches the decision.

Possible scale-aware alternatives:

- report metrics by project scale;
- use weighted absolute percentage error for a justified portfolio meaning:

$$
WAPE=\frac{\sum_i|y_i-\hat y_i|}{\sum_i|y_i|};
$$

- model log cost when multiplicative structure is plausible; or
- compare scaled error with a defined baseline.

No substitute is universal.

## 12.5 Asymmetric decision loss

If underbudgeting is more costly:

$$
L(e)=
\begin{cases}
c_{under}|e|,&e>0,\\
c_{over}|e|,&e\leq0,
\end{cases}
\qquad c_{under}>c_{over}.
$$

Quantile or pinball loss for level $\tau$ is:

$$
L_\tau(y,\hat q)=
\begin{cases}
\tau(y-\hat q),&y\geq\hat q,\\
(1-\tau)(\hat q-y),&y<\hat q.
\end{cases}
$$

An estimated 80th conditional quantile can support contingency planning. Label it as a quantile, not as a mean plus an arbitrary buffer.

## 12.6 Training loss, selection metric, and reporting metrics

- Training loss determines fitted parameters.
- Selection metric chooses among procedures.
- Reporting metrics communicate operational consequences.

A model can minimise training MSE, be selected by validation MAE, and be reported with test MAE, RMSE, ME, MedAE, $R^2$, and subgroup errors. Pre-specify primary and secondary metrics rather than choosing after seeing which favours a preferred model.

## 12.7 Keep a row-level prediction table

Retain:

- project ID;
- actual target and unit;
- prediction;
- signed, absolute, and squared errors;
- district and year;
- project scale and remoteness; and
- any evaluation-fold identifier.

Metrics can be regenerated from rows. Aggregate scores cannot reconstruct lost individual failures.

## 12.8 Held-out diagnostic views

**Error versus prediction:** curvature, fan shape, and offset.

**Error over time:** policy, procurement, technology, measurement, or drift.

**Error by district:** representation, group relationships, or measurement differences.

**Error by project scale/remoteness:** whether overall averages hide high-risk projects.

**Signed-error distribution:** asymmetry and extreme tails.

The four-panel function in the chapter validates required columns, plots these views, and returns the Matplotlib figure. Plots generate hypotheses; they do not identify causes by themselves.

## 12.9 Subgroup evaluation without false certainty

For each meaningful group report:

- count;
- mean signed error;
- MAE;
- MedAE; and
- 90th percentile absolute error.

Small subgroup estimates are noisy. Do not rank five-project districts as a stable league table. Consider intervals, domain knowledge, replication, and whether groups are weighted by project count, community importance, or budget exposure.

## 12.10 Bootstrap interval for held-out MAE

A row bootstrap:

1. retains fixed held-out actual/prediction pairs;
2. samples $n$ row indices with replacement;
3. calculates MAE on each resample;
4. repeats thousands of times; and
5. takes lower and upper empirical quantiles.

“With replacement” means a project can appear multiple times or not at all in one resample. The percentile interval approximates sampling variability under the assumption that held-out rows are suitable resampling units.

If projects share districts or contractors, row resampling may understate dependence. A cluster bootstrap resamples independent groups. The resampling unit must match the data-generating structure.

The seed makes the simulation reproducible; the number of repeats controls Monte Carlo smoothness, not the underlying amount of evidence.

## 12.11 Fold standard deviation is not a confidence interval

Cross-validation training sets overlap, so fold scores are dependent. Folds may also represent substantively different districts or years. Fold standard deviation describes variation across those folds; it is not automatically a standard error or 95% interval. Dividing by $\sqrt K$ does not make dependence disappear.

## 12.12 Research discussion: Bengio and Grandvalet

The paper proves that, under its broad setting, no universal unbiased estimator of $K$-fold cross-validation variance can be recovered from ordinary fold results. Overlapping training sets create covariance terms that are not generally identifiable.

It does not say cross-validation is useless or uncertainty impossible. It warns against a convenient universal variance formula.

Practical lessons:

- never relabel fold SD as a confidence interval;
- preserve individual fold results and their identities;
- use repeated/nested designs, independent tests, bootstrap, or model-based methods only with assumptions stated; and
- treat small performance differences cautiously.

For district folds, variation may represent real geographic heterogeneity rather than interchangeable noise.

Replication exercise: across many independently generated datasets, compare variation among folds within one dataset with variation of mean CV performance across datasets.

## 12.13 Diagnostics do not authorise test tuning

Correct workflow after finding a test failure:

1. report the original pre-specified result;
2. use diagnostics to propose a substantive mechanism;
3. label changes as post-test development;
4. revise using development data; and
5. obtain new independent or prospective evidence.

Do not overwrite history and reuse the same test set as untouched.

## 12.14 Responsible revision protocol

```text
structured error
    -> proposed mechanism
    -> measurement/timing audit
    -> development-data revision
    -> rerun full selection procedure
    -> new independent evaluation
```

Curvature may motivate a capacity-squared feature; changing spread may motivate quantile or variance modeling; drift may motivate a known-at-deployment time feature; district failure may reveal measurement inconsistency rather than a need to add district identity.

## Day 12 exit check

You should be able to state:

1. positive signed error means underprediction;
2. held-out $R^2$ uses the evaluation-set mean reference;
3. low overall MAE can hide remote-project failure;
4. fold SD is not automatically an interval; and
5. test-driven revision requires new evaluation evidence.

---

# Chapter 2 capstone — An auditable MHP evaluation pipeline

## Capstone objective

Estimate final constant-2025 cost at appraisal for projects beginning in 2023 onward. The deliverable is the documented learning and evaluation procedure, not only the winning score.

## Pass 1: evaluation contract

State before modeling:

- unit of observation;
- target, unit, and price basis;
- prediction time;
- eligible and prohibited features;
- deployment population;
- primary metric and decision rationale;
- validation design;
- locked test period;
- subgroup reports; and
- conditions that block deployment.

## Pass 2: boundary audit

Confirm:

- project ID is an identifier, not a feature;
- final material bill is prohibited;
- transformations fit only inside training partitions;
- time ordering is correct;
- duplicates do not cross partitions; and
- test outcomes do not guide category handling or cleaning.

## Pass 3: compare complete procedures

Compare:

1. training-mean baseline;
2. linear regression with imputation, scaling, and one-hot encoding; and
3. trees whose depth is selected on validation data.

Use validation MAE as the primary selection criterion. Retain RMSE, MedAE, $R^2$, and ME as secondary reports.

## Pass 4: select, refit, and test once

- Train: 2016–2021.
- Validate: 2022.
- Test: 2023–2025.
- Select procedure using validation MAE only.
- Refit the chosen procedure on train plus validation.
- Open final test once.

The capstone uses `clone` so every candidate begins unfitted. All candidates share one preprocessing recipe, which makes comparison procedural and prevents leakage.

## Pass 5: diagnose without rewriting history

Report:

- overall metrics and a bootstrap MAE interval;
- district and remoteness results with counts;
- the ten largest absolute errors;
- temporal drift evidence; and
- whether errors meet actual planning tolerance.

If diagnostics motivate a revision, mark the start of a new development cycle.

## Read the runnable capstone by function

| Function/component | Role |
|---|---|
| `make_preprocessor` | Defines numeric and categorical training transformations |
| `make_pipeline` | Joins preprocessing to one model |
| `regression_metrics` | Computes pre-specified test summaries |
| `bootstrap_mae_interval` | Quantifies row-resampling uncertainty |
| `prediction_table` | Preserves project-level evidence |
| `subgroup_report` | Summarises operational groups |
| feature audit assertions | Enforce timing and identifier rules |
| chronological masks | Create train, validation, and test periods |
| candidate loop | Fits each procedure on train and scores validation |
| selected refit | Learns from all development data after selection |
| final test block | Produces the one locked evaluation |

Run it, then inspect the exported prediction CSV rather than reading only console averages.

## Capstone interpretation checklist

1. Did the selected procedure beat the training-fitted baseline on validation and test?
2. Did procedure ranking change between validation and test?
3. Is error small relative to planning tolerance, not merely relative to a competitor?
4. Which groups fail, and how many observations support each estimate?
5. Does ME show systematic underbudgeting?
6. Do later years show drift?
7. Why is final material bill prohibited despite its predictive score?
8. Are tree thresholds predictive patterns rather than causal effects?
9. Does an OLS win prove the true mechanism is linear? No.
10. What fresh evidence is needed after revision?

## Self-assessment rubric

| Area | Research-ready beginner habit |
|---|---|
| Contract | Timing, units, permitted information, deployment, and blockers are explicit |
| Numerics | Scaling, rank, conditioning, solver, and tolerance are examined |
| Probability | Interval type and assumptions are stated |
| Optimisation | Gradient is checked and convergence compared with a direct solver |
| Validation | Split matches deployment and selection remains inside its boundary |
| Metrics | Baseline, decision loss, uncertainty, and subgroup risk are integrated |
| Diagnostics | Patterns generate hypotheses without reusing test as confirmation |
| Reproducibility | Data recipe, seed, code, versions, and outputs are traceable |
| Communication | Recommendation includes magnitude, limitations, and blocking conditions |

---

# Formula sheet with plain-language readings

| Concept | Formula | Read it as... |
|---|---|---|
| Standardisation | $z_{ij}=(x_{ij}-\mu_j)/s_j$ | Centre and scale using training statistics |
| Condition number | $\kappa_2(X)=\sigma_{max}/\sigma_{min}$ | Imbalance between strongest and weakest directions |
| QR | $X=QR$ | Orthonormal directions times triangular coordinates |
| QR least squares | $R\hat\beta=Q^Ty$ | Project, then solve a triangular system |
| SVD | $X=U\Sigma V^T$ | Rotate, stretch, rotate |
| Pseudoinverse | $X^+=V\Sigma^+U^T$ | Invert retained singular directions |
| SVD solution | $\hat\beta=V\Sigma^+U^Ty$ | Minimum-norm least-squares solution |
| Statistical model | $y=X\beta+\varepsilon$ | Conditional mean plus unobserved error |
| Gaussian errors | $\varepsilon\sim\mathcal N(0,\sigma^2I)$ | Independent equal-variance normal errors |
| Log-likelihood | $C-\frac n2\log\sigma^2-\frac{SSR}{2\sigma^2}$ | Gaussian fit rewards smaller SSR |
| Residual variance | $s^2=SSR/(n-k)$ | Noise estimate using residual degrees of freedom |
| Coefficient covariance | $s^2(X^TX)^{-1}$ | Classical sampling uncertainty matrix |
| Coefficient CI | $\hat\beta_j\pm t^*SE(\hat\beta_j)$ | Repeated-sampling interval procedure |
| Mean SE | $s\sqrt{x_0^T(X^TX)^{-1}x_0}$ | Uncertainty in estimated conditional mean |
| Prediction SE | $s\sqrt{1+x_0^T(X^TX)^{-1}x_0}$ | Mean uncertainty plus individual noise |
| MSE objective | $J=\lVert y-X\beta\rVert^2/n$ | Average squared training loss |
| MSE gradient | $\nabla J=2X^T(X\beta-y)/n$ | Local uphill direction |
| GD update | $\beta^{t+1}=\beta^t-\eta\nabla J$ | Take a learning-rate-sized downhill step |
| Mean baseline | $\hat c=\bar y_{train}$ | Best constant under squared loss |
| Median baseline | $\hat c=\operatorname{median}(y_{train})$ | Best constant under absolute loss |
| Training risk | $n^{-1}\sum_iL(y_i,f(x_i))$ | Average observed training loss |
| Generalisation risk | $\mathbb E_{deployment}[L(Y,f(X))]$ | Expected loss where model will be used |
| CV loss | $K^{-1}\sum_kL_k$ | Average held-out fold loss |
| ME | $n^{-1}\sum_i e_i$ | Directional error |
| MAE | $n^{-1}\sum_i|e_i|$ | Average magnitude of error |
| RMSE | $\sqrt{n^{-1}\sum_i e_i^2}$ | Large-error-sensitive summary in target units |
| MedAE | $\operatorname{median}(|e_i|)$ | Central absolute miss |
| $R^2$ | $1-SSR/TSS$ | Squared-error comparison with evaluation mean |
| WAPE | $\sum|e_i|/\sum|y_i|$ | Portfolio absolute error relative to total magnitude |
| Pinball loss | Piecewise $\tau$-weighted loss | Asymmetric loss targeting a conditional quantile |

## Shape sheet

For $n$ observations and $k$ design columns:

| Object | Shape |
|---|---|
| $X$ | $(n,k)$ |
| $y$, predictions, residuals | $(n,)$ |
| $\beta$, gradient | $(k,)$ |
| Reduced $Q$ | $(n,k)$ |
| $R$ | $(k,k)$ |
| Reduced $U$ | $(n,r)$ or `(n, min(n,k))` computationally |
| singular values | `(min(n,k),)` |
| $V^T$ | `(min(n,k), k)` |
| coefficient covariance | $(k,k)$ |

---

# Expanded glossary

**Algorithm:** Computational procedure. A learning procedure also includes preprocessing, tuning, and selection.

**Baseline:** Simple reference model a candidate should beat under the same evaluation.

**Batch gradient descent:** Uses all training observations for each update.

**Bias–variance decomposition:** Repeated-sampling squared-error decomposition into squared bias, estimator variance, and irreducible noise.

**Bootstrap:** Repeatedly resamples observed units with replacement to approximate sampling variability.

**Condition number:** Measure of how strongly perturbations can be amplified.

**Confidence interval:** Repeated-sampling procedure designed for stated long-run coverage under assumptions.

**Constant prices:** Monetary values converted to a common price-year basis.

**Cross-validation:** Development design where rows or groups take turns as held-out data.

**Data leakage:** Illegitimate information flow across timing, training, selection, or evaluation boundaries.

**Degrees of freedom:** Independent information remaining after fitted constraints in a specified model.

**Density:** Relative concentration of a continuous probability distribution.

**Distribution shift:** Difference between training and deployment distributions.

**Empirical risk:** Average loss on an observed dataset.

**Error term:** Unobserved deviation from a population conditional mean.

**Exchangeable:** Observations whose ordering does not change their joint probabilistic role under a model; informally, similarly generated units for a random split.

**Extrapolation:** Prediction outside feature regions well represented in training.

**Generalisation:** Performance on genuinely new intended-deployment cases.

**Gradient descent:** Iterative optimisation that moves parameters opposite the gradient.

**Heteroskedasticity:** Conditional error variance changes with observations or features.

**Hyperparameter:** Setting chosen outside parameter fitting, often using validation evidence.

**Ill-conditioning:** Small input changes can cause large solution changes.

**Imputation:** Replacement of missing values using a stated learned rule.

**Irreducible noise:** Outcome variability not eliminated by knowing the modeled relationship.

**Learning curve:** Training and evaluation performance plotted against training-set size.

**Likelihood:** Function comparing parameter values with observed data fixed.

**Log-likelihood:** Logarithm of likelihood, converting products to sums.

**Maximum likelihood:** Chooses parameters maximising likelihood under a specified model.

**Mini-batch gradient descent:** Uses a subset of training rows for each update.

**Nested cross-validation:** Outer evaluation loop surrounding inner model selection.

**Numerical rank:** Singular directions retained as nonzero at a stated tolerance and scale.

**Numerical stability:** Extent to which an algorithm avoids magnifying rounding error unnecessarily.

**One-hot encoding:** Represents a category with indicator columns.

**Orthonormal:** Mutually perpendicular vectors, each of length one.

**Overfitting:** Learning sample-specific patterns that do not generalise.

**Parameter:** Value learned inside a fitted model or transformer.

**Pipeline:** One bounded procedure joining learned transformations and a model.

**Prediction interval:** Interval for an individual future outcome, including outcome noise.

**Pseudoinverse:** Generalised inverse expressing least-squares solutions in full- or deficient-rank cases.

**QR decomposition:** Factorisation into orthonormal-column $Q$ and upper-triangular $R$.

**Quantile:** Value below which a specified proportion lies.

**Residual:** Observed target minus fitted prediction.

**Selection bias in evaluation:** Optimism when data used for selection also influence reported performance.

**Singular value:** Nonnegative SVD scale describing strength of a matrix direction.

**Stochastic gradient descent:** Uses one observation per parameter update.

**Test set:** Data reserved for final evaluation after procedure is fixed.

**Training set:** Data permitted to fit transformations and model parameters.

**Underfitting:** Failure to capture useful structure because procedure is too restricted.

**Validation set:** Development data used to compare procedures and choose hyperparameters.

---

# Research reading guide

For each paper, answer:

1. What exact question is asked?
2. Is the object a parameter, predictor, algorithm, selection procedure, or evaluation estimate?
3. Is the evidence a theorem, simulation, benchmark, observational analysis, or case study?
4. What did the authors actually establish?
5. Which assumptions or settings limit transfer?
6. What should change in the MHP workflow?

| Paper | Main object | Warning | Workflow consequence |
|---|---|---|---|
| Breiman (2001) | Modeling cultures | Elegant stochastic models may be weakly checked | Compare transparent and algorithmic models on purpose-relevant evidence |
| Cawley & Talbot (2010) | Selection procedure | Model selection itself can overfit | Evaluate complete selection procedures with nested CV or locked test |
| Bengio & Grandvalet (2004) | CV variance | No universal unbiased variance estimator from usual folds | Do not turn fold SD into an automatic interval |

Small replication portfolio:

1. Generate many datasets and compare tree-depth training/validation optimism.
2. Increase the number of hyperparameters tried and measure inner-to-outer gaps.
3. Compare within-dataset fold variation with across-dataset variation of mean CV performance.

For every notebook, pre-write the question and expected result, save seeds and versions, label axes and units, separate observation from interpretation, and state one limitation.

---

# Suggested seven-day study schedule

| Activity | Minutes |
|---|---:|
| Retrieve Chapter 1 knowledge | 15 |
| Work through central derivation | 40 |
| Run and alter proof code | 35 |
| Inspect or create figure | 25 |
| Extend estimator/evaluation | 40 |
| Break and diagnose | 30 |
| Exit check and reflection | 15 |
| **Total** | **200** |

Day 11 may require two sessions. Nested validation is conceptually dense; drawing every inner and outer boundary is more valuable than memorising an API call.

Morning retrieval prompts:

- After Day 6: What does condition number measure, and where does a scaler fit?
- After Day 7: How do QR and SVD solve least squares?
- After Day 8: Why does Gaussian MLE produce OLS, and which interval is wider?
- After Day 9: Write the gradient and explain learning rate.
- After Day 10: Derive the mean baseline and define deployment risk.
- After Day 11: Which nested-CV loop selects and which evaluates?
- After Day 12: Why is one average score insufficient?

---

# Common beginner confusions

| Confusion | Correction |
|---|---|
| Scaling adds information | It changes representation only |
| Full rank means stable coefficients | Full-rank matrices can be ill-conditioned |
| SVD fixes multicollinearity | It diagnoses directions and selects a solution; it cannot identify redundant effects |
| OLS requires normal raw data | OLS fitting does not; particular inference formulas add assumptions |
| Residual equals true error | Residual uses estimated parameters and fitting constraints |
| Gradient descent is automatically more accurate | It is another optimiser and should reach the same OLS minimum |
| Lowest training error wins | Training error rewards flexibility and cannot establish generalisation |
| Random split is always objective | It can violate group and time structure |
| Pipeline is code convenience | It protects the learning boundary |
| CV removes need for final evaluation | Repeated selection can overfit development evidence |
| Test $R^2=0$ equals training dummy | Standard denominator uses evaluation-set mean |
| Fold SD is a 95% interval | Fold scores are dependent and possibly heterogeneous |

---

# Responsible interpretation in the KP context

Keep asking:

- Was each field genuinely known at appraisal?
- Are costs in the same base year and do they treat community contributions consistently?
- Does road distance always mean distance to an all-weather road?
- Is terrain scored consistently across survey teams?
- Do projects share contracts, contractors, valleys, or administrative processes?
- Did procurement policy change during the data period?
- Are delayed or abandoned remote projects absent from completed-project data?
- Does a random split place near-identical local conditions on both sides?
- Would systematic underprediction deny remote communities adequate budgets?
- Are subgroup differences supported by enough projects?

Numerical correctness, statistical validity, and institutional responsibility meet at these questions.

---

# Where Chapter 3 begins

Chapter 2 builds an honest framework around OLS and a nonlinear comparator. Chapter 3 can now add:

1. polynomial and interaction features;
2. ridge and lasso regularisation;
3. geometric and Bayesian views of penalties;
4. robust and quantile regression;
5. heteroskedasticity-robust and cluster-aware inference;
6. influence, leverage, and Cook's distance;
7. repeated and nested model comparison; and
8. a pre-registered benchmark study.

Every future model must travel through the same chain:

```text
deployment question
    -> permitted information
    -> complete learning procedure
    -> matching validation design
    -> uncertainty and subgroup diagnostics
    -> bounded decision
```

A trustworthy model is not certified by elegant mathematics or an impressive score alone. Trust grows from an auditable chain of stable computation, explicit assumptions, legitimate information, deployment-matched evaluation, quantified uncertainty, investigated failures, and restraint in the final claim.
