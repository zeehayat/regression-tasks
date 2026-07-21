# Chapter 3 Companion Guide: From a Trustworthy Baseline to a Research-Grade Regression Study

## A beginner-first guide to preprocessing, feature design, regularisation, diagnostics, robust inference, quantiles, and auditable research

This guide accompanies `chapter3.md`. It assumes you have completed Chapters 1 and 2, but it continues to explain new Python, mathematics, and statistics from first principles.

Chapter 1 built ordinary least squares (OLS). Chapter 2 surrounded it with stable computation and honest evaluation. Chapter 3 develops the habits of a regression practitioner:

1. treat preprocessing as part of the fitted model;
2. express substantive hypotheses through features;
3. stabilise complex designs through regularisation;
4. find projects and predictor directions that dominate results;
5. match uncertainty calculations to variance and dependence;
6. change the loss or conditional target when the decision requires it; and
7. freeze the full comparison procedure before opening final evidence.

The goal is not to collect model names. It is to make every link auditable:

```text
question
    -> data
    -> representation
    -> estimator
    -> validation
    -> uncertainty
    -> bounded claim
```

At every arrow ask: What was learned? From which rows? Which assumption entered? What could leak? What could shift? What evidence would change the decision?

---

## How to use this guide

For Days 13–20:

1. Read the matching guide section.
2. Reproduce the central equation on paper.
3. Type and alter the smallest code example.
4. Run the main chapter's complete laboratory.
5. Create the stated failure deliberately.
6. Write one supported conclusion and one unsupported conclusion.
7. Complete the exit check without notes.

When evaluating a technique, separate these layers:

- **representation:** which columns enter the model;
- **target:** mean, median, or another quantile;
- **fitting criterion:** squared, Huber, pinball, or penalised loss;
- **uncertainty:** classical, heteroskedasticity-consistent, clustered, bootstrap, or none;
- **evaluation:** which untouched cases simulate deployment; and
- **interpretation:** predictive, explanatory, or causal.

Methods acting on different layers are not interchangeable.

---

# 0. Prerequisite and software foundations

## 0.1 Retrieve these Chapter 1 and 2 ideas

Before proceeding, explain without notes:

- $\hat y=X\hat\beta$ and $e=y-\hat y$;
- the OLS objective and gradient;
- full rank, singular values, and condition number;
- why scaling and imputation fit only on training data;
- train, validation, test, grouped CV, temporal CV, and nested CV;
- error term versus residual;
- coefficient confidence interval versus project prediction interval; and
- ME, MAE, RMSE, held-out $R^2$, and subgroup evaluation.

Chapter 3 relies on these rather than replacing them.

## 0.2 The MHP prediction contract

One row is one fictional microhydro power project. The model predicts final project cost in constant 2025 million PKR at technical appraisal, before procurement and construction.

Chapter 3 adds:

- missing road-distance surveys;
- contractor experience counts;
- categorical access mode;
- a correlated route-length measurement;
- curved capacity costs;
- cable-by-road interactions;
- larger variability for remote projects; and
- a few extreme but legitimate logistics events.

The complete hidden `road_distance_km` remains in the synthetic generator only so learners can audit how missingness was created. In a real incomplete dataset, the true missing values are unavailable. Using the hidden column as a feature would be leakage.

## 0.3 Software setup

```bash
python -m pip install numpy pandas scipy matplotlib scikit-learn statsmodels
```

Scikit-learn supports predictive pipelines and regularised estimators. Statsmodels supplies mature inferential and influence diagnostics. Educational from-scratch functions reveal mechanisms; verify them against tested libraries before relying on them.

Record Python and package versions, seeds, input-data versions, and output files for every benchmark.

## 0.4 Understanding the practitioner data generator

`make_mhp_practitioner_data` extends Chapter 2's synthetic data.

```python
df["contractor_experience_projects"] = rng.poisson(lam=5.0, size=n)
df["access_mode"] = np.select(conditions, choices, default="porter_or_air")
df["road_distance_observed_km"] = df["road_distance_km"].mask(missing_road)
```

- A Poisson draw creates nonnegative event counts.
- `np.select` applies the first true condition and otherwise uses a default.
- `.mask(condition)` replaces true positions with missing values.

Missingness is more likely for difficult access and early years. Extreme logistics events are added only to some remote projects. These are genuine outcomes within the simulated population, not automatic deletion candidates.

Always distinguish:

- **latent complete value:** available to the simulation author;
- **observed appraisal value:** available to the fitted procedure; and
- **post-appraisal field:** prohibited at prediction time.

## 0.5 New mathematical language

### Norms

For coefficient vector $\beta$:

$$
\lVert\beta\rVert_1=\sum_j|\beta_j|,
\qquad
\lVert\beta\rVert_2^2=\sum_j\beta_j^2.
$$

The $L_1$ norm sums magnitudes. Squared $L_2$ sums squares. Lasso and ridge use them as penalties.

### Trace

The trace of a square matrix is the sum of diagonal elements:

$$
\operatorname{tr}(A)=\sum_j A_{jj}.
$$

Trace appears in average leverage and ridge effective degrees of freedom.

### Indicator functions

$\mathbb 1\{condition\}$ equals 1 when the condition is true and 0 otherwise. It is the mathematical version of a Boolean converted to a number.

### Partial derivatives and conditional slopes

If a prediction depends on several inputs, $\partial\hat y/\partial x_j$ measures the local change in prediction as $x_j$ changes while other displayed inputs remain fixed.

### Subgradients

$|b|$ has no single derivative at zero because its left slope is -1 and right slope is +1. A subgradient uses every slope in the interval $[-1,1]$ at that kink. This permits the lasso optimum to remain exactly zero.

### Positive definite matrices

A symmetric matrix $A$ is positive definite when:

$$
v^TAv>0
$$

for every nonzero $v$. Such curvature gives a unique minimum for a quadratic problem.

### MAP estimation

Maximum a posteriori (MAP) estimation maximises a Bayesian posterior density. Under particular priors, its objective matches ridge or lasso. A MAP point is not a full posterior distribution and does not by itself quantify uncertainty.

## 0.6 New Python and scikit-learn patterns

### Custom transformers

A scikit-learn transformer implements `fit` and `transform`:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class ExampleTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.learned_value_ = X["column"].median()
        return self

    def transform(self, X):
        X = X.copy()
        X["centered"] = X["column"] - self.learned_value_
        return X
```

The trailing underscore marks learned state. `fit` must use only the current training fold. `transform` applies stored state to validation, test, or deployment rows.

### `statsmodels`

```python
import statsmodels.api as sm

X_design = sm.add_constant(X, has_constant="add")
result = sm.OLS(y, X_design).fit()
print(result.summary())
```

Statsmodels generally expects you to add the intercept explicitly. Its summary is not automatically valid merely because it prints p-values; design and covariance assumptions still matter.

### Logarithmic grids

```python
alphas = np.logspace(-4, 4, 100)
```

This creates penalty values from $10^{-4}$ to $10^4$. Regularisation strength often needs orders-of-magnitude exploration rather than equally spaced values.

---

# Day 13 — Preprocessing is part of the model

## 13.1 Missingness is data

For feature $j$ and row $i$:

$$
R_{ij}=
\begin{cases}
1,&X_{ij}\text{ observed},\\
0,&X_{ij}\text{ missing}.
\end{cases}
$$

The missingness mechanism describes how $R$ relates to observed and unobserved values. It is not a moral label for data collectors.

## 13.2 MCAR, MAR, and MNAR

Let $X_{obs}$ be observed data and $X_{mis}$ missing values.

### Missing completely at random (MCAR)

$$
P(R\mid X_{obs},X_{mis})=P(R).
$$

Missingness is unrelated to observed and missing values, such as an independent random scanner failure.

### Missing at random (MAR)

$$
P(R\mid X_{obs},X_{mis})=P(R\mid X_{obs}).
$$

After conditioning on observed information, missingness no longer depends on the unseen value. “At random” does not mean “for no reason.” Early year and observed access mode might explain recording probability.

### Missing not at random (MNAR)

Even after conditioning on observed information, missingness depends on an unseen value or unobserved factor. Extremely long access distances may be omitted precisely because they are hard to measure.

Observed data generally cannot prove which mechanism holds or rule out MNAR. Sensitivity analysis must state alternative assumptions.

## 13.3 Complete-case analysis changes the population

Deleting every incomplete row can:

- reduce precision;
- change district and access-mode composition;
- remove remote projects systematically; and
- bias estimates unless deletion is justified for the analysis.

Compare retained and excluded rows on observed fields, including year, terrain, access mode, and retrospective outcomes where appropriate for an audit. This can reveal selection but cannot prove deletion safe.

Do not use the target in a deployment-time imputer merely because it was legitimate in a retrospective missingness investigation.

## 13.4 Median imputation is a learned model

$$
x_{ij}^{imp}=
\begin{cases}
x_{ij},&R_{ij}=1,\\
\operatorname{median}(X_{j,train}),&R_{ij}=0.
\end{cases}
$$

Median imputation preserves rows and offers a useful predictive baseline, but it does not recover truth:

- values pile up at the median;
- variance shrinks;
- relationships with other variables weaken; and
- ordinary formulas treating imputed values as observed understate uncertainty.

The median must be learned independently inside every training fold.

## 13.5 Missingness indicators

$$
M_{ij}=1-R_{ij}.
$$

Adding $M$ lets a model learn an offset for rows whose value was imputed. In scikit-learn:

```python
SimpleImputer(strategy="median", add_indicator=True)
```

The future missing value gets the training median plus indicator 1. An observed future value remains unchanged and gets indicator 0.

An indicator can exploit stable predictive information in recording patterns. It does not make MNAR ignorable, identify a cause, or restore the missing distribution.

## 13.6 Single and multiple imputation serve different purposes

Single imputation makes one completed feature matrix. Multiple imputation creates $m$ plausible completed datasets, runs the analysis on each, and pools estimates and uncertainty.

For estimate $\hat Q_l$ and within-imputation variance $U_l$:

$$
\bar Q=\frac1m\sum_{l=1}^m\hat Q_l,
$$

$$
\bar U=\frac1m\sum_{l=1}^mU_l,
$$

$$
B=\frac1{m-1}\sum_{l=1}^m(\hat Q_l-\bar Q)^2,
$$

$$
T=\bar U+\left(1+\frac1m\right)B.
$$

$\bar U$ is ordinary within-completed-data uncertainty. $B$ measures sensitivity to plausible missing values. The final factor acknowledges finite imputations.

Generating several datasets is not enough. Imputation and analysis models must support the estimand, include relevant missingness predictors, respect deployment timing, and fit separately inside evaluation folds. Prediction performance is scored on observed held-out outcomes.

## 13.7 Categorical encodings impose hypotheses

### One-hot encoding

With A as reference:

$$
\hat y=\beta_0+\beta_BI(B)+\beta_CI(C).
$$

$\beta_B$ is B minus A conditional on other features. Changing reference changes coefficient labels but not fitted values. With an intercept, dropping one category prevents exact redundant coding.

### Ordinal encoding

Mapping low, medium, high to 0, 1, 2 imposes order. A linear coefficient additionally imposes equal gaps. Never assign numbers to districts merely because software needs numeric input.

### Frequency encoding

Replace a category with its training frequency. This says category commonness is predictive and makes equally frequent categories indistinguishable.

### Target encoding

$$
TE(c)=\frac1{n_c}\sum_{i:C_i=c}y_i.
$$

This uses outcome information. Naively encoding training rows with their own category mean leaks each row's target, especially for rare or unique categories.

## 13.8 Smoothed and cross-fitted target encoding

Smoothed category mean:

$$
\widetilde{TE}(c)
=\frac{n_c\bar y_c+m\bar y}{n_c+m}.
$$

Rare categories shrink toward the global training mean. Smoothing reduces variance but does not eliminate same-row leakage.

Cross-fitting procedure:

1. split development training data into internal folds;
2. calculate category statistics on the other folds;
3. encode the held-out fold;
4. combine out-of-fold training encodings;
5. fit a final mapping on all development rows only for future transformation; and
6. map unseen categories to a pre-specified fallback such as training global mean.

The internal splitter must also respect groups and time. Random target-encoding folds can leak district or future information while avoiding same-row leakage.

Target encoding therefore needs two protections:

- cross-fitting within development; and
- complete separation from the final test.

## 13.9 A leakproof mixed-type pipeline

Numeric branch:

```text
median imputation + missing indicators -> standardisation
```

Categorical branch:

```text
most-frequent imputation -> one-hot encoding
```

`ColumnTransformer` combines branches, then a model fits transformed columns. Every cross-validation fold learns its own medians, indicator availability, scales, and categories.

`handle_unknown="ignore"` prevents unseen categories from crashing. It does not demonstrate good transfer to them.

## 13.10 Research discussion: Rubin on missing data

Rubin formalised when a missingness process may be ignored for particular likelihood or sampling analyses.

Beginner lessons:

1. MAR is conditional on observed information.
2. Ignorability depends on the mechanism and inferential framework.
3. Observed values generally cannot rule out MNAR.
4. Filling blanks is not propagating missing-value uncertainty.

MHP implication: missing remote-road surveys require recording-process knowledge and sensitivity analysis, not only a median-filled column.

Replication: simulate identical complete data followed by MCAR, MAR, and MNAR deletion. Compare complete cases, median imputation, median-plus-indicator prediction, and multiple-imputation interval coverage. Keep predictive error separate from coefficient bias.

## Day 13 exit check

Explain why MAR is not “missing for no reason,” why a missing indicator does not solve MNAR, and why target encoding needs cross-fitting plus final-test separation.

---

# Day 14 — Feature engineering is model specification

## 14.1 Linear in parameters can be nonlinear in a feature

$$
\hat y=\beta_0+\beta_1x+\beta_2x^2
$$

is curved in $x$ but linear in coefficients. The design matrix simply contains $1,x,x^2$. OLS or regularised linear regression can fit it using familiar machinery.

Creating a column changes the family of possible relationships. It is a mathematical hypothesis, not free accuracy.

## 14.2 Centre before powers and interactions

Let:

$$
x_c=x-c,
$$

where $c$ is a meaningful reference such as the training median.

$$
\hat y=\gamma_0+\gamma_1x_c+\gamma_2x_c^2.
$$

Then:

- $\gamma_0$ is prediction at $x=c$;
- $\gamma_1$ is slope at $x=c$;
- $x_c$ and $x_c^2$ are often less correlated; and
- numerical interpretation improves.

Data-derived centres are learned preprocessing and fit only inside training folds.

## 14.3 Interpret quadratics through derivatives

For:

$$
f(x)=\beta_0+\beta_1x+\beta_2x^2,
$$

the local slope is:

$$
f'(x)=\beta_1+2\beta_2x.
$$

A stationary point occurs at:

$$
x^*=-\frac{\beta_1}{2\beta_2},
$$

when $\beta_2\neq0$.

- $\beta_2>0$: convex, stationary point is a minimum.
- $\beta_2<0$: concave, stationary point is a maximum.

Before interpreting, check whether $x^*$ lies inside observed and credible deployment support, whether uncertainty is acceptable, whether the specification fits, and whether a causal claim is identified. The sign of $\beta_2$ alone proves none of these.

## 14.4 Continuous-by-continuous interactions

$$
\hat y=\beta_0+\beta_1x_1+\beta_2x_2+\beta_3x_1x_2.
$$

Conditional slopes:

$$
\frac{\partial\hat y}{\partial x_1}=\beta_1+\beta_3x_2,
$$

$$
\frac{\partial\hat y}{\partial x_2}=\beta_2+\beta_3x_1.
$$

$\beta_1$ is the slope of $x_1$ only at $x_2=0$. Centring makes zero a meaningful reference. An MHP cable-by-road interaction allows cable cost slope to vary with remoteness.

## 14.5 Continuous-by-binary interactions

Let $D=1$ for porter/air access and 0 for reference access:

$$
\hat y=\beta_0+\beta_1x+\beta_2D+\beta_3xD.
$$

For $D=0$:

$$
\hat y=\beta_0+\beta_1x.
$$

For $D=1$:

$$
\hat y=(\beta_0+\beta_2)+(\beta_1+\beta_3)x.
$$

$\beta_2$ changes intercept and $\beta_3$ changes slope.

## 14.6 Hierarchy principle

When including $x^2$, normally retain $x$. When including $x_1x_2$, retain both main effects.

This:

- anchors higher-order interpretation;
- avoids unintended constraints; and
- makes predictions invariant to harmless shifts of feature origin.

Hierarchy is a disciplined default, not an absolute theorem. Depart only for a defensible substantive constraint.

## 14.7 Ratios make strong assumptions

$$
r=\frac{x_1}{x_2}.
$$

Risks:

- denominator near zero creates instability;
- both measurements may contain error;
- separate size information is lost;
- denominator may be post-outcome and leak; and
- low ratio error may not imply low total-budget error.

“Cost per kW” changes the estimand from total cost. Converting it back requires capacity weighting and careful loss interpretation.

## 14.8 Log transformations

For positive $x$, use $\log x`. For nonnegative counts, $\log(1+x)` is common.

Interpretations for small changes:

- level–log: $y=\beta_0+\beta_1\log x$; 1% more $x$ corresponds to about $0.01\beta_1$ target units;
- log–level: $\log y=\beta_0+\beta_1x$; one more unit of $x$ corresponds to about $100\beta_1$ percent change when $\beta_1$ is small;
- log–log: coefficient is elasticity.

Logs compress large values but do not guarantee normality, and normal features are not an OLS requirement. Exponentiating a predicted log target generally returns a conditional median under log-error assumptions, not automatically the mean. Retransformation bias requires treatment.

## 14.9 Piecewise linear hinges and splines

Hinge at knot $k$:

$$
(x-k)_+=\max(0,x-k).
$$

Model:

$$
\hat y=\beta_0+\beta_1x+\beta_2(x-k)_+.
$$

- below $k$: slope $\beta_1$;
- above $k$: slope $\beta_1+\beta_2$.

Splines use basis functions:

$$
f(x)=\sum_{m=1}^M\theta_mB_m(x).
$$

Once knots and degree define the bases, coefficients remain linearly fitted. Knot count, placement, and degree control flexibility and belong inside model selection unless pre-specified.

Splines usually behave more locally than high-degree global polynomials, but no fitted curve certifies unsupported extrapolation.

## 14.10 Feature explosion

Degree-2 expansion of $p$ inputs can create:

- $p$ main effects;
- $p$ squares; and
- $p(p-1)/2$ pairwise interactions.

This increases computation, collinearity, search opportunities, and noise fitting. Domain knowledge limits candidates; regularisation stabilises them; validation evaluates the complete procedure.

## 14.11 Read `MHPFeatureEngineer`

Its `fit` validates required columns and learns training medians. `transform` creates:

- centred capacity and its square;
- centred cable-by-road interaction;
- an 18 km road hinge; and
- `log1p` contractor experience.

Pipeline order matters: a feature transformer operating on missing road values must either follow imputation or handle missing values deliberately. Every generated column needs an information receipt:

- raw fields used;
- whether target was used;
- rows used to learn constants;
- units;
- mechanism represented;
- valid range; and
- availability at appraisal.

## 14.12 Extrapolation audit

Plot fitted functions inside and outside observed ranges. A quadratic and hinge can agree over 80–850 kW yet diverge severely at 1,300 kW. Validation on existing support cannot certify a region with no comparable projects.

## Day 14 exit check

Interpret a quadratic through its derivative, explain an interaction using conditional slopes, apply hierarchy, and state why feature generation repeats inside every validation training fold.

---

# Day 15 — Ridge regression: shrinkage and stability

## 15.1 Why accept bias?

Nearly redundant predictors can yield large opposite OLS coefficients that cancel in predictions. Small sample changes reassign their contribution. Ridge accepts deliberate bias to reduce coefficient variance and possibly improve future prediction.

## 15.2 Ridge objective

For centred target and standardised features, with intercept separated:

$$
J_{ridge}(\beta)
=\lVert y-X\beta\rVert_2^2
+\lambda\lVert\beta\rVert_2^2,
\qquad\lambda\geq0.
$$

- $\lambda=0$: OLS where unique.
- small positive $\lambda$: mild shrinkage.
- large $\lambda$: penalised slopes approach zero.
- intercept is ordinarily not penalised.

Packages normalise loss differently using $n$, 2, or $2n$. Therefore `alpha` and $\lambda$ values are not portable without checking the documented objective.

## 15.3 Derive the ridge solution

Gradient:

$$
\nabla J
=-2X^Ty+2X^TX\beta+2\lambda\beta.
$$

Set to zero:

$$
(X^TX+\lambda I)\hat\beta_{ridge}=X^Ty.
$$

Thus:

$$
\hat\beta_{ridge}
=(X^TX+\lambda I)^{-1}X^Ty.
$$

In code, solve the system; do not explicitly invert.

### Why a unique solution exists

For nonzero $v$ and $\lambda>0$:

$$
v^T(X^TX+\lambda I)v
=\lVert Xv\rVert_2^2+\lambda\lVert v\rVert_2^2>0.
$$

The system is positive definite even when $X^TX$ is singular. Ridge chooses a unique coefficient vector but does not create scientific information to distinguish identical features.

## 15.4 Scaling defines the penalty

The same distance measured in metres needs a coefficient one-thousandth the per-kilometre coefficient. Penalising raw magnitudes treats arbitrary units differently.

Standardising inside each fold makes one coefficient unit correspond to one training standard deviation. It does not make features causal, equally useful, Gaussian, or outlier-free.

## 15.5 SVD explains ridge shrinkage

For $X=U\Sigma V^T$:

$$
\hat\beta_{ridge}
=V\operatorname{diag}\left(\frac{\sigma_j}{\sigma_j^2+\lambda}\right)U^Ty.
$$

Fitted values:

$$
\hat y_{ridge}
=U\operatorname{diag}\left(
\frac{\sigma_j^2}{\sigma_j^2+\lambda}
\right)U^Ty.
$$

Direction-specific retention:

$$
s_j(\lambda)=\frac{\sigma_j^2}{\sigma_j^2+\lambda}.
$$

Large singular directions retain more. Weak directions shrink more. OLS's $1/\sigma_j` amplification is replaced by $\sigma_j/(\sigma_j^2+\lambda)`.

## 15.6 Effective degrees of freedom

Ridge smoother matrix:

$$
H_\lambda=X(X^TX+\lambda I)^{-1}X^T.
$$

Effective flexibility:

$$
df_{eff}(\lambda)
=\operatorname{tr}(H_\lambda)
=\sum_{j=1}^r\frac{\sigma_j^2}{\sigma_j^2+\lambda}.
$$

This need not be an integer. It equals rank at zero penalty for the centred component and decreases as shrinkage grows.

## 15.7 Constrained geometry

Ridge is equivalent for a corresponding $t$ to:

$$
\min_\beta\lVert y-X\beta\rVert_2^2
\quad\text{subject to}\quad
\lVert\beta\rVert_2^2\leq t.
$$

In two dimensions the constraint is a circle. SSR contours are ellipses. Their first contact gives the ridge solution. A smooth circle generally does not touch at an axis, so ridge shrinks but rarely makes exact zeros.

## 15.8 Read `RidgeFromScratch`

It:

1. validates nonnegative alpha;
2. learns training means and scales;
3. centres target;
4. solves $Z^TZ+\alpha I$;
5. converts standardised slopes back to raw units; and
6. reconstructs the intercept.

Constant columns receive scale 1 to avoid division by zero, but should still be investigated. Predictions should match `StandardScaler` plus scikit-learn `Ridge` under the same objective.

## 15.9 Coefficient paths and tuning

Plot standardised coefficients against log penalty. Paths reveal whether correlated slopes stabilise and how quickly weak directions shrink.

The path is development information. Select alpha with pre-specified validation, never by choosing an attractive final-test coefficient.

## 15.10 Bias–variance trade-off

Ridge increases bias. If variance falls by more, expected prediction error improves. It is not universally superior: adequate sample size, few well-conditioned features, and correct linear structure may favour OLS.

## 15.11 Bayesian MAP interpretation

Gaussian likelihood:

$$
y\mid X,\beta\sim\mathcal N(X\beta,\sigma^2I).
$$

Gaussian prior on standardised slopes:

$$
\beta\sim\mathcal N(0,\tau^2I).
$$

Negative log posterior is proportional to:

$$
\frac1{2\sigma^2}\lVert y-X\beta\rVert_2^2
+\frac1{2\tau^2}\lVert\beta\rVert_2^2.
$$

Thus MAP has ridge form with $\lambda=\sigma^2/\tau^2$. Smaller prior variance means stronger shrinkage. This interpretation depends on that prior and does not provide the whole posterior.

## 15.12 Ridge limitations

Ridge does not automatically provide causality, valid post-tuning p-values, leakage protection, outlier resistance, missing-data validity, or variable selection. Tuning makes selection part of the estimator; fixed-model uncertainty formulas do not automatically include it.

## 15.13 Research discussion: Hoerl and Kennard

Their proposal showed how biased estimates can lower coefficient mean squared error in nonorthogonal problems. Read separately:

- instability being addressed;
- diagonal modification of normal equations;
- coefficient MSE versus modern test prediction error;
- original ridge-constant choice versus nested/temporal validation; and
- dependence on feature units and linear assumptions.

Replication: across many samples with $x_2=x_1+\epsilon$, compare OLS and ridge coefficient bias, variance, coefficient MSE, and test prediction error as correlation increases.

## Day 15 exit check

Derive ridge normal equations, explain singular-direction shrinkage and effective degrees of freedom, and state why scaling defines penalty meaning.

---

# Day 16 — Lasso, elastic net, and sparse models

## 16.1 Lasso objective

For centred target and standardised design:

$$
J_{lasso}(\beta)
=\frac1{2n}\lVert y-X\beta\rVert_2^2
+\alpha\lVert\beta\rVert_1.
$$

This matches scikit-learn's convention. The intercept is separate and unpenalised. Because absolute value has a kink at zero, there is no ordinary closed-form matrix inverse—and exact zero coefficients become possible.

## 16.2 Subgradient condition

For nonzero $b$:

$$
\frac{d|b|}{db}=\operatorname{sign}(b).
$$

At zero:

$$
\partial|b|=[-1,1].
$$

Optimality:

$$
-\frac1nX^T(y-X\hat\beta)+\alpha z=0,
$$

where $z_j=1` for positive coefficients, -1 for negative coefficients, and any value in $[-1,1]` at zero. A coefficient remains zero when its residual correlation lies within the penalty threshold.

## 16.3 Derive soft thresholding

For orthonormal standardised columns:

$$
\frac1nX^TX=I,
\qquad
z_j=\frac1nx_j^Ty.
$$

One-coordinate objective becomes:

$$
\frac12(\beta_j-z_j)^2+\alpha|\beta_j|.
$$

Cases:

- $z_j>\alpha$: $\hat\beta_j=z_j-\alpha$;
- $z_j<-\alpha$: $\hat\beta_j=z_j+\alpha$;
- $|z_j|\leq\alpha$: $\hat\beta_j=0$.

Combined:

$$
S(z,\alpha)
=\operatorname{sign}(z)\max(|z|-\alpha,0).
$$

Soft thresholding both shrinks and selects.

## 16.4 Why $L_1$ geometry produces zeros

The constrained form uses a diamond:

$$
\min_\beta\lVert y-X\beta\rVert_2^2
\quad\text{subject to}\quad
\lVert\beta\rVert_1\leq t.
$$

Diamond corners lie on coordinate axes. Expanding elliptical SSR contours often first touch a corner. Ridge's circular boundary usually touches away from axes.

## 16.5 Coordinate descent

For coordinate $j$, calculate the partial residual excluding it:

$$
r^{(j)}=y-\sum_{k\neq j}x_k\beta_k.
$$

Update:

$$
\beta_j\leftarrow
\frac{S\left(x_j^Tr^{(j)}/n,\alpha\right)}{x_j^Tx_j/n}.
$$

Cycle through coefficients until convergence. The from-scratch class centres and scales, caches column energy, updates one coordinate at a time, and checks maximum coefficient change.

Production solvers also examine optimality through a duality gap. Verify educational predictions against scikit-learn with identical scaling, alpha, tolerance, and objective normalisation.

## 16.6 Correlated predictors make selection unstable

When road distance and surveyed route carry overlapping information, lasso can select either one across resamples while predictions remain similar.

A zero means:

> This feature was not used by this penalised fit at this penalty, sample, and representation.

It does not mean no relationship, no importance, or no causal role.

Selection-frequency bootstraps are descriptive unless embedded in a formal stability-selection method. Use transformed feature names from the fitted pipeline rather than guessing coefficient order.

## 16.7 Elastic net

$$
J_{EN}(\beta)
=\frac1{2n}\lVert y-X\beta\rVert_2^2
+\alpha\rho\lVert\beta\rVert_1
+\frac{\alpha(1-\rho)}2\lVert\beta\rVert_2^2.
$$

Scikit-learn calls $\rho$ `l1_ratio`.

- $\rho=1$: lasso.
- $\rho=0`: ridge-like, though use `Ridge` for pure $L_2`.
- between 0 and 1: sparsity plus correlated-feature stabilisation.

Both alpha and ratio belong inside tuning.

## 16.8 Bayesian lasso interpretation

With Gaussian outcome errors and independent Laplace priors:

$$
p(\beta_j\mid b)=\frac1{2b}\exp\left(-\frac{|\beta_j|}{b}\right),
$$

the posterior mode has lasso form after matching normalisations. A continuous Laplace prior assigns zero probability to the event $\beta_j=0$ exactly; a zero MAP is not posterior proof of absence.

## 16.9 Paths and the one-standard-error rule

A validation minimum may be broad and noisy. The one-standard-error rule selects the most regularised candidate whose estimated validation loss is within one estimated standard error of the minimum.

It encodes a simplicity/stability preference under near-ties; it is not a theorem of optimality. Temporal folds are dependent and heterogeneous, so report year scores and describe any standard-error approximation.

## 16.10 Post-selection inference warning

Selecting three variables from forty with lasso, then fitting ordinary OLS on those three using the same data, makes classical intervals too optimistic because they pretend selection never occurred.

Options:

- focus on honestly held-out prediction;
- confirm on a new sample;
- use valid selective/debiased methods with assumptions; or
- pre-specify a low-dimensional explanatory model.

Regularisation solves fitting and prediction problems, not inference after searching.

## 16.11 Research discussion: Tibshirani and LARS

Tibshirani's lasso combines ridge-like shrinkage with subset-like sparsity through an $L_1$ constraint. Read its motivation, geometry, evidence, original computation, and causal/inferential boundaries.

Efron and colleagues later introduced least-angle regression (LARS), efficiently tracing a lasso path under suitable conditions. Coordinate descent and LARS illuminate different computational geometry.

Replication: generate correlated feature pairs, repeat samples, and compare lasso versus elastic net on MAE, active-set size, selection frequency, and correlation of predictions across fits.

## Day 16 exit check

Derive soft thresholding, explain the diamond geometry, and explain why prediction stability and variable-selection stability are different.

---

# Day 17 — Multicollinearity, leverage, and influence

## 17.1 Four diagnostic questions

| Idea | Question |
|---|---|
| Multicollinearity | Are predictor directions weakly distinguished? |
| Target outlier | Is the outcome surprising under this model? |
| Leverage | Is the predictor combination unusual? |
| Influence | Would removing this row substantially change the fit? |

A row can have high leverage with a small residual, or a large residual with ordinary leverage. Influence usually combines discrepancy and leverage. Every label is relative to a particular fitted design.

## 17.2 Consequences of collinearity depend on purpose

- Exact collinearity: rank deficiency and non-unique OLS coefficients.
- Near collinearity: unique but unstable coefficients.
- In-range prediction: combined fitted values may remain stable.
- Interpretation: conditional slopes are uncertain.
- Extrapolation: cancellation between large slopes may fail.
- Computation: rounding errors can be amplified.

Under the classical model, collinearity increases variance; it does not itself bias OLS.

## 17.3 Variance inflation factor

Regress feature $x_j$ on other predictors and obtain auxiliary $R_j^2$. Then:

$$
VIF_j=\frac1{1-R_j^2}.
$$

Slope variance can be written:

$$
\operatorname{Var}(\hat\beta_j\mid X)
=\frac{\sigma^2}
{\sum_i(x_{ij}-\bar x_j)^2(1-R_j^2)}.
$$

Relative to a corresponding orthogonal reference, variance is inflated by VIF and standard error by $\sqrt{VIF_j}$. VIF 9 means threefold standard-error inflation, not “90% wrong.”

There is no universal cutoff. High VIF is structurally expected with polynomials and interactions and may affect explanation more than interpolation. Do not remove a necessary confounder just to lower it.

Calculate VIF on the actual transformed design, not merely raw inputs. Singular-value diagnostics reveal multi-feature weak directions that individual VIFs may summarise incompletely.

## 17.4 Hat matrix and leverage

$$
\hat y=X(X^TX)^{-1}X^Ty=Hy,
$$

$$
H=X(X^TX)^{-1}X^T.
$$

Row $i$ leverage:

$$
h_{ii}=x_i^T(X^TX)^{-1}x_i.
$$

For a full-rank design with $k$ columns:

$$
0\leq h_{ii}\leq1,
\qquad
\sum_i h_{ii}=\operatorname{tr}(H)=k.
$$

Average leverage is $k/n$. Rules like $2k/n$ and $3k/n$ are screening conventions, not automatic exclusions. Rare deployment-relevant projects may legitimately have high leverage.

## 17.5 Studentised residuals

Since $e=(I-H)\varepsilon$ under the model:

$$
\operatorname{Var}(e_i\mid X)=\sigma^2(1-h_{ii}).
$$

Internally studentised residual:

$$
r_i=\frac{e_i}{s\sqrt{1-h_{ii}}},
\qquad
s^2=\frac{\sum_i e_i^2}{n-k}.
$$

It adjusts raw residuals for their model-implied variance. External studentisation estimates variance after deleting row $i$ and is useful in formal outlier tests, which also require multiple-testing care.

## 17.6 Cook's distance

$$
D_i
=\frac{e_i^2}{ks^2}
\frac{h_{ii}}{(1-h_{ii})^2}.
$$

It is proportional to aggregate fitted-value change after deleting row $i$:

$$
D_i
=\frac{\sum_m(\hat y_m-\hat y_{m(i)})^2}{ks^2}.
$$

$D_i>4/n$ is an investigation flag, not a deletion rule.

## 17.7 Read the diagnostic code

The educational function computes:

1. OLS coefficients and fitted values;
2. residuals;
3. leverage with row-wise quadratic forms;
4. residual MSE;
5. studentised residuals; and
6. Cook's distance.

Use `statsmodels.get_influence()` as a reference. The explicit inverse mirrors formulas; production implementations should use stable factorizations.

## 17.8 Diagnostic quartet

1. Residual versus fitted: curvature, spread, groups.
2. Normal Q–Q: tail departure when normal inference matters.
3. Scale–location: $\sqrt{|r_i|}$ versus fitted for changing variance.
4. Residual versus leverage with Cook contours: influential combinations.

Plots suggest mechanisms but do not select remedies. A funnel may reflect multiplicative noise, omitted groups, wrong mean structure, or data errors.

## 17.9 Influence investigation protocol

1. Verify source, units, duplicate status, and transcription.
2. Confirm target-population and prediction-time eligibility.
3. Understand why predictors are rare or outcome surprising.
4. compare pre-justified specifications.
5. report sensitivity of coefficients, validation, and conclusions.
6. preserve valid in-scope rows in the primary analysis.

Document any data-error deletion rule so it can be applied without viewing a desired result.

## 17.10 Regularised influence differs

OLS formulas apply to unpenalised OLS. Ridge uses $H_\lambda$ and effective degrees of freedom. Lasso changes active set nonlinearly, so copying OLS Cook formulas is unsafe. Assess complex selected pipelines through case-deletion refits or carefully designed resampling.

## Day 17 exit check

Distinguish residual, leverage, and influence; derive average leverage; and explain VIF as conditional variance inflation rather than a variable-deletion command.

---

# Day 18 — Heteroskedasticity and dependent data

## 18.1 Coefficients and uncertainty are separate layers

If the conditional mean is correctly linear and:

$$
\mathbb E[\varepsilon\mid X]=0,
$$

OLS can remain unbiased even when error variance changes.

Under homoskedastic independent errors:

$$
\operatorname{Var}(\varepsilon\mid X)=\sigma^2I,
$$

$$
\operatorname{Var}(\hat\beta\mid X)=\sigma^2(X^TX)^{-1}.
$$

When variance or dependence differs, OLS coefficients stay the same but covariance, standard errors, tests, and intervals change.

## 18.2 General sandwich covariance

For general error covariance $\Omega$:

$$
\operatorname{Var}(\hat\beta\mid X)
=(X^TX)^{-1}X^T\Omega X(X^TX)^{-1}.
$$

Outer matrices are bread; middle is meat. For independent heteroskedastic observations, $\Omega$ is diagonal with different variances.

## 18.3 HC0 through HC3

HC0:

$$
\widehat{Var}_{HC0}(\hat\beta)
=(X^TX)^{-1}
\left(\sum_i e_i^2x_ix_i^T\right)
(X^TX)^{-1}.
$$

| Estimator | Residual contribution | Purpose |
|---|---:|---|
| HC0 | $e_i^2$ | Basic asymptotic sandwich |
| HC1 | $\frac n{n-k}e_i^2$ | Degrees-of-freedom scaling |
| HC2 | $e_i^2/(1-h_{ii})$ | Leverage-related residual shrinkage |
| HC3 | $e_i^2/(1-h_{ii})^2$ | Stronger leave-one-out-like correction |

HC3 is often reasonable in modest samples, not universally guaranteed.

## 18.4 Build and verify HC covariance

The chapter function:

1. computes bread $(X^TX)^{-1}$;
2. fits OLS and residuals;
3. calculates leverage;
4. chooses the HC residual weights;
5. computes `X.T @ (omega[:, None] * X)` as meat; and
6. sandwiches the result.

Compare coefficients and covariance with `statsmodels.OLS(...).fit(cov_type="HC3")`.

## 18.5 What HC standard errors do not fix

They do not repair:

- wrong conditional mean;
- omitted variables or endogeneity;
- leakage;
- measurement-error bias;
- unsupported causal contrasts;
- row dependence when using independent-row HC; or
- out-of-sample prediction.

“Robust” always means robust to a stated violation while retaining other assumptions.

## 18.6 Heteroskedasticity tests

Breusch–Pagan relates squared residuals to predictors. White's test allows a richer variance auxiliary model.

- small samples may lack power;
- huge samples flag trivial departures;
- rejection may reflect mean misspecification; and
- non-rejection does not prove equal variance.

Use design knowledge and plots. If changing variance is expected by construction, pre-specifying HC3 can be more coherent than choosing covariance after a significance test.

## 18.7 Cluster dependence

For cluster $g$:

$$
\text{meat}
=\sum_{g=1}^G(X_g^Te_g)(X_g^Te_g)^T.
$$

Cluster-robust covariance permits arbitrary within-cluster covariance while relying on approximate independence across clusters. The number of independent clusters, not row count, drives asymptotic justification.

## 18.8 Five districts are too few for casual asymptotics

Five clusters can produce a software number without reliable reference distribution. Possible responses:

- gather more independent districts;
- state a hierarchical model explicitly;
- use justified small-cluster methods such as wild cluster bootstrap;
- use randomisation inference when design permits; or
- limit claims and show district sensitivity descriptively.

No technique converts five independent shocks into fifty.

## 18.9 Choose cluster level from dependence

- shared district shocks: district;
- repeated rows per project: project;
- village-level treatment: assignment unit constrains inference;
- dependence across district and year: one-way clustering may be insufficient.

Do not choose the level that makes significance attractive. Time and spatial dependence need methods with their own bandwidth and asymptotic assumptions.

## 18.10 Prediction intervals need a variance model

HC covariance improves uncertainty of estimated mean coefficients. It does not automatically make project-level interval widths adapt to remoteness.

Options include conditional scale models, quantile regression, conformal methods under stated exchangeability/shift assumptions, or hierarchical probability models.

## 18.11 Research discussion: White

White developed consistent OLS covariance estimation under unknown heteroskedasticity subject to regularity conditions.

Read:

- target asymptotic distribution;
- failure of homoskedastic covariance;
- squared-residual construction;
- sensitivity to sample size and leverage; and
- assumptions that remain beyond variance form.

Replication: under variance $\left(0.5+|x|\right)^2$, compare classical and HC3 95% slope-interval coverage across repeated samples. Then omit a true quadratic term and observe that HC3 does not repair mean bias.

## Day 18 exit check

Construct the sandwich, explain HC3's leverage correction, and explain why few clusters are a design limitation.

---

# Day 19 — Robust and quantile regression

## 19.1 Three meanings of “robust”

| Method | Fitted target/loss | Changed layer |
|---|---|---|
| OLS + HC3 | OLS mean/squared loss | covariance only |
| Huber regression | robust conditional location | coefficient fit/loss |
| Quantile regression | chosen conditional quantile | target and loss |

They answer different questions.

## 19.2 Why squared loss reacts strongly

OLS uses $\frac12r^2`, whose derivative with respect to residual is $r`. Residual magnitude 20 contributes twenty times the local gradient magnitude of residual 1.

Large residuals may be errors, omitted mechanisms, heavy tails, mixed regimes, rare genuine events, or out-of-scope rows. Changing loss reduces sensitivity but does not diagnose the cause.

## 19.3 Huber loss and score

$$
L_\delta(r)=
\begin{cases}
\frac12r^2,&|r|\leq\delta,\\
\delta\left(|r|-\frac12\delta\right),&|r|>\delta.
\end{cases}
$$

Derivative:

$$
\psi_\delta(r)=
\begin{cases}
r,&|r|\leq\delta,\\
\delta\operatorname{sign}(r),&|r|>\delta.
\end{cases}
$$

Large gradients are clipped. Huber remains smooth at $\pm\delta$ and is less sensitive to vertical outliers.

## 19.4 Iteratively reweighted least squares intuition

For $r\neq0$:

$$
w(r)=\frac{\psi_\delta(r)}r
=\begin{cases}
1,&|r|\leq\delta,\\
\delta/|r|,&|r|>\delta.
\end{cases}
$$

Iterate:

1. fit current coefficients;
2. calculate residuals and weights;
3. solve weighted least squares; and
4. repeat.

Production Huber regression accounts for scale because a numerical threshold changes meaning when target units change.

## 19.5 Huber is not robust to every leverage point

A predictor-space outlier can pull the fitted surface toward itself and end with a modest residual. Residual clipping alone may not protect against it. Combine Huber with leverage, case deletion, provenance, and deployment-population checks.

In scikit-learn, `epsilon` sets the scaled-residual transition and `alpha` adds $L_2$ regularisation. Both tune inside validation. Smaller epsilon is not automatically better.

## 19.6 Conditional quantiles

OLS targets:

$$
\mathbb E[Y\mid X=x].
$$

Conditional quantile:

$$
Q_Y(\tau\mid x)
=\inf\{q:P(Y\leq q\mid X=x)\geq\tau\}.
$$

- 0.5: conditional median.
- 0.8: upper planning quantile.
- 0.95: more conservative upper quantile.

A quantile is not a confidence interval for a mean.

## 19.7 Derive pinball loss

For residual $u=y-q$:

$$
\rho_\tau(u)
=u(\tau-\mathbb 1\{u<0\})
=\begin{cases}
\tau u,&u\geq0,\\
(\tau-1)u,&u<0.
\end{cases}
$$

At $\tau=0.8$, underprediction costs 0.8 per unit and overprediction 0.2: four times as much. Expected risk derivative is:

$$
R'(q)=F_Y(q)-\tau.
$$

The optimum satisfies the quantile condition. At jumps, a subgradient produces the set-valued definition.

## 19.8 Linear quantile regression

Assume:

$$
Q_Y(\tau\mid X=x)=x^T\beta_\tau.
$$

Fit:

$$
\hat\beta_\tau
=\arg\min_\beta\sum_i\rho_\tau(y_i-x_i^T\beta),
$$

possibly with regularisation. Coefficients can differ across quantiles because spread and distribution shape change with features.

Scikit-learn `QuantileRegressor.alpha` controls $L_1$ regularisation, so scaling and validation remain necessary.

## 19.9 Evaluate quantile predictions

Use mean pinball loss at the target quantile. Check held-out empirical coverage:

$$
\frac1m\sum_i\mathbb 1\{y_i\leq\hat q_\tau(x_i)\}.
$$

For a calibrated stable 0.8 conditional-quantile model, overall coverage should be near 0.8. Also report coverage by pre-specified group and predicted-quantile bins with counts and uncertainty.

Coverage alone is insufficient: an enormous upper prediction covers everything but is useless. Pinball loss balances calibration and sharpness according to the target.

## 19.10 Quantile crossing

Separately fitted models can yield:

$$
\hat Q(0.5\mid x)>\hat Q(0.8\mid x).
$$

This violates ordering. Check within deployment-like ranges. Remedies include joint non-crossing fitting, full-distribution models, validated rearrangement, or simpler representation. Sorting predictions post hoc changes the procedure and must be validated.

## 19.11 Decision basis for an upper quantile

If underprediction costs $c_u` per unit and overprediction costs $c_o`:

$$
\tau=\frac{c_u}{c_u+c_o}.
$$

When underbudgeting costs four times as much, $\tau=0.8$. This is a decision-based choice made before outcomes, not a quantile chosen for attractive test coverage.

## 19.12 Research discussions: Huber and Koenker–Bassett

Huber's robust-location work formalised an efficiency-versus-contamination trade-off, explaining linear central score and clipped tails. Its location setting does not automatically solve regression leverage.

Replication: contaminate 5% of normal observations with a much wider distribution and compare mean, median, and Huber location across repeats.

Koenker and Bassett generalised sample quantiles to regression through asymmetric absolute loss, allowing conditional effects at different distribution points.

Replication: simulate heteroskedastic $Y=2+3X+(0.5+X)\varepsilon$, fit 0.1, 0.5, and 0.9 regressions, compare theoretical slopes, and check held-out coverage.

## Day 19 exit check

Distinguish HC3, Huber, and quantile regression; derive Huber and pinball losses; and evaluate an upper quantile overall and by subgroup.

---

# Day 20 — A pre-specified regression benchmark

## 20.1 Exploration, confirmation, and benchmarks

- **Exploratory:** choices adapt to observed results; findings generate hypotheses.
- **Confirmatory:** primary question and analysis are fixed in advance; deviations recorded.
- **Predictive benchmark:** candidate procedures and selection rule are fixed; locked test evaluates the selected result.

Exploration is essential. Mislabeling an adaptive result as pre-specified is the problem.

## 20.2 Registered prediction contract

Freeze:

| Component | Choice |
|---|---|
| Unit | One eligible MHP project |
| Prediction time | Technical appraisal |
| Target | Final constant-2025 cost |
| Development | Start through 2022 |
| Locked test | 2023–2025 |
| Validation | Expanding years 2019–2022 |
| Point metric | Equal-year-average MAE |
| Secondary metrics | RMSE, signed error, year spread |
| Planning target | Conditional 80th percentile |
| Planning metric | 0.8 pinball loss; coverage diagnostic |
| Subgroups | Access mode and district with counts |
| Exclusions | Documented eligibility/source errors only |
| Test use | Once after selection |

Equal year weighting gives each year equal influence. Project weighting answers a different question. Either can be justified if fixed in advance.

## 20.3 Compare complete procedures

A candidate includes:

1. imputation and indicators;
2. categorical encoding;
3. centring and engineered features;
4. standardisation;
5. estimator and loss;
6. penalty and other hyperparameters; and
7. convergence settings.

The chapter pre-specifies historical mean, raw and engineered OLS, ridge grid, elastic-net grid, Huber grid, and median/0.8 quantile grids.

Point candidates need not estimate the same scientific functional: squared loss targets mean, MAE's population optimum is median, and Huber targets a robust location. Comparing them under operational MAE is legitimate for a point decision, but do not relabel an MAE-selected robust location as expected cost.

## 20.4 Temporal boundary

For validation year $t`, fit every learned transformation and estimator only on years before $t`. Aggregate the pre-specified year losses, select one procedure, refit through 2022, and test on 2023–2025 once.

Feature design, influence decisions, and metric choice cannot use locked-test information.

## 20.5 Read the end-to-end code

### Constants and feature lists

They make target, validation years, test boundary, seed, eligible raw features, engineered features, and categories explicit. Assertions prohibit latent road distance and post-construction material bill.

### Feature engineer

Learns capacity/year centres per fold and creates capacity square, time centre, road hinge, cable-road interaction, and log experience.

### Candidate factories

Return fresh complete pipelines for point and quantile procedures. A modest theory-guided grid controls selection opportunity.

### Expanding-window scoring

Clones and refits every candidate for every year. It stores candidate, year, validation count, and loss rather than keeping only an average.

### Selection

Ranks candidates by equal-year mean and worst-year tie breaker. A broad practical tie should be reported rather than over-precisely ranked.

### Locked test

Refits winner on all development rows, predicts test once, and compares with pre-specified raw OLS reference.

### Quantile track

Selects median and 80th-quantile procedures separately by matching pinball loss, then records coverage and crossing.

### Outputs

Writes validation fold scores, summaries, row-level locked predictions, subgroup audit, and JSON metadata containing environment and selections.

Assertions protect only their stated contract. They do not replace a full audit.

## 20.6 Read validation before winner

Ask:

- Is the lead consistent or one-year-driven?
- Is it operationally meaningful?
- Does raw OLS nearly tie?
- Did candidates converge?
- Does the tie rule prefer a stable simpler model?

Minimum validation loss after a search is optimistic as a performance estimate. The locked test evaluates the chosen procedure.

## 20.7 Interpret the locked test narrowly

Report:

- date ranges and sample sizes;
- full candidate families and selected procedure;
- MAE, RMSE, signed error;
- comparison with raw OLS;
- median and 0.8 pinball loss;
- overall/subgroup 0.8 coverage;
- crossing rate;
- shift evidence; and
- plan deviations.

It estimates performance in the represented later period, not permanent universal superiority.

## 20.8 Paired bootstrap difference

$$
\Delta=MAE_{selected}-MAE_{raw\ OLS}.
$$

Negative favours selected. Resample projects as paired rows because both models predict the same cases. The supplied row-bootstrap interval is descriptive and assumes row exchangeability; it ignores district/time dependence. Show year/district results and avoid unjustified precision.

## 20.9 Subgroup audit without leaderboard fishing

Pre-specify groups for operational reasons. Report counts, errors, quantile coverage, and crossing. Do not search many slices until one looks dramatic, then present it as confirmed. Post-hoc findings become hypotheses for new evidence.

## 20.10 Model card and reproducibility

Model card should state:

- purpose and prohibited uses;
- prediction time and target units;
- training period/population;
- features and missingness policy;
- candidate selection and metrics;
- overall and subgroup performance;
- quantile calibration;
- limitations and monitoring;
- human override and accountability.

Reproducibility checklist includes source code, seed, versions, data generator/input hash, frozen contract, candidate grid, all fold scores, row predictions, metadata, and deviation log.

Reproducibility means another researcher can repeat the procedure. It does not prove the procedure is scientifically correct.

## 20.11 Research discussion: Varma and Simon

Their work demonstrates selection bias when cross-validation both tunes and reports performance. Nested CV or a locked independent evaluation separates tuning from assessment.

Replication: increase candidate settings from 5 to 50 to 500 and compare best inner score with outer/independent performance. Selection optimism should grow with search opportunity.

## 20.12 Research report structure

1. Question and deployment decision.
2. Data and eligibility.
3. Registered contract and deviations.
4. Complete candidate procedures.
5. Validation design and selection rule.
6. Locked results, diagnostics, influence, convergence.
7. Pre-specified sensitivity and labeled post-hoc analysis.
8. Limitations: sampling, dependence, shift, causality, synthetic status.
9. Reproducibility artifacts.
10. Bounded conclusion.

Conclusion template:

> Under expanding-window comparison through 2022, procedure A had the lowest pre-specified average validation MAE. After refitting on all development projects, its 2023–2025 MAE was X million PKR versus Y for raw OLS. Performance was weakest for subgroup Z based on n projects. The 80th-quantile procedure covered C% overall and C-Z% in subgroup Z. Results support appraisal prediction within represented districts and ranges; they do not identify causal effects or establish transfer to new regions.

Fill every placeholder and retain unfavourable results.

## Day 20 exit check

You are finished when another researcher can trace every number from frozen contract through pipeline, fold scores, selected procedure, locked predictions, subgroup audit, metadata, and limitations.

---

# Chapter 3 synthesis and laboratories

## How the techniques form one system

- Preprocessing changes information received.
- Feature engineering changes expressible relationships.
- Regularisation changes fitting criterion.
- Diagnostics reveal weak directions and dominating cases.
- Robust covariance changes uncertainty, not coefficients.
- Huber changes loss and fitted location.
- Quantile regression changes conditional target.
- Validation evaluates the whole procedure.
- Locked evidence bounds the final claim.

## Derivation workshop

Complete on paper and verify numerically:

1. ridge normal equations;
2. ridge SVD coefficient and fitted-value factors;
3. positive definiteness of $X^TX+\lambda I$;
4. lasso soft thresholding under $X^TX/n=I$;
5. $\sum_i h_{ii}=k$;
6. $\operatorname{Var}(e_i\mid X)=\sigma^2(1-h_{ii})$;
7. sandwich covariance from the OLS coefficient expression;
8. pinball expected-risk derivative;
9. cost ratio leading to $\tau=c_u/(c_u+c_o)$; and
10. slope when a predictor appears in a square and interaction.

## Practical laboratories

### A — Missingness as information

Compare MCAR and remote-dependent missingness using complete cases, median, median-plus-indicator, and multiple imputation. Separate prediction error from inferential coverage.

### B — Regularisation under correlation

Vary predictor correlation from 0 to 0.999. Across repeats report OLS/ridge coefficient variance and bias, lasso selection frequency, and elastic-net prediction.

### C — Influence map

Create ordinary, vertical-outlier, high-leverage-on-line, and high-leverage-large-residual rows. Compare residual, leverage, and Cook distance.

### D — Covariance failure

Compare classical, HC0, and HC3 coverage under equal/changing variance and several leverage patterns.

### E — Dependence

Increase projects per district while fixing district count. Observe why row count does not replace independent clusters.

### F — Robust target choice

Contaminate outcomes and compare OLS, Huber, median, and ridge under mean- and median-target losses.

### G — Quantile calibration

Fit median and 0.8 quantiles under equal/changing spread. Check loss, overall/group coverage, and crossing.

### H — Selection optimism

Compare 5, 50, and 500 candidate searches using inner versus outer performance.

## Self-assessment rubric

| Dimension | Ready for this level when you can... |
|---|---|
| Prediction contract | make population, decision, timing, target, units, and exclusions executable |
| Preprocessing | justify missingness assumptions, category policy, unknown levels, and fold boundaries |
| Feature design | state mechanism, hierarchy, derivative, support, units, and extrapolation limits |
| Regularisation | derive objectives, explain paths and scaling, and tune without test access |
| Diagnostics | separate collinearity, residual size, leverage, and influence and audit provenance |
| Uncertainty | justify classical, HC, or clustered covariance and state remaining assumptions |
| Robust targets | distinguish mean, robust location, median, and upper quantile and use matching losses |
| Validation | preserve temporal selection, locked test, search budget, tie rule, and adaptation log |
| Reporting | retain predictions, failed candidates, subgroup counts, uncertainty limits, and deviations |
| Reproducibility | enable another researcher to trace every claim to code and saved output |

---

# Formula sheet with plain-language readings

| Concept | Formula | Meaning |
|---|---|---|
| Missing observed indicator | $R_{ij}\in\{0,1\}$ | Whether a value is observed |
| Missing indicator | $M_{ij}=1-R_{ij}$ | Whether imputation occurred |
| Rubin pooled estimate | $\bar Q=m^{-1}\sum_l\hat Q_l$ | Average across imputations |
| Rubin pooled variance | $T=\bar U+(1+1/m)B$ | Within plus missing-value uncertainty |
| Smoothed target encoding | $(n_c\bar y_c+m\bar y)/(n_c+m)$ | Shrink rare category means |
| Quadratic slope | $\beta_1+2\beta_2x$ | Local change at $x$ |
| Turning point | $-\beta_1/(2\beta_2)$ | Zero local quadratic slope |
| Interaction slope | $\beta_1+\beta_3x_2$ | Effect of $x_1` conditional on $x_2` |
| Hinge | $(x-k)_+=\max(0,x-k)$ | Add a slope change at knot $k$ |
| Standardisation | $(x_{ij}-\bar x_j)/s_j$ | Define comparable penalty scale |
| Ridge objective | $\lVert y-X\beta\rVert^2+\lambda\lVert\beta\rVert_2^2$ | Fit plus squared shrinkage |
| Ridge estimate | $(X^TX+\lambda I)^{-1}X^Ty$ | Closed-form centred slopes |
| Ridge retention | $\sigma_j^2/(\sigma_j^2+\lambda)$ | Fitted-value signal retained in direction $j$ |
| Ridge effective df | $tr(H_\lambda)$ | Smooth flexibility |
| Lasso objective | $\lVert y-X\beta\rVert^2/(2n)+\alpha\lVert\beta\rVert_1$ | Fit plus sparse shrinkage |
| Soft threshold | $sign(z)\max(|z|-\alpha,0)$ | Shrink and set small coordinates to zero |
| Elastic net | Loss $+\alpha\rho L_1+\alpha(1-\rho)L_2^2/2$ | Sparsity plus correlated-feature stability |
| VIF | $1/(1-R_j^2)$ | Slope-variance inflation |
| Hat matrix | $X(X^TX)^{-1}X^T$ | Maps outcomes to OLS fits |
| Average leverage | $k/n$ | Mean design leverage |
| Studentised residual | $e_i/[s\sqrt{1-h_{ii}}]$ | Variance-adjusted residual |
| Cook distance | $e_i^2h_{ii}/[ks^2(1-h_{ii})^2]$ | Case influence on fitted values |
| General sandwich | $(X^TX)^{-1}X^T\Omega X(X^TX)^{-1}$ | Coefficient covariance for general error covariance |
| HC0 meat | $\sum_i e_i^2x_ix_i^T$ | Independent heteroskedastic residual contribution |
| HC3 residual square | $e_i^2/(1-h_{ii})^2$ | Leverage-corrected contribution |
| Cluster meat | $\sum_g(X_g^Te_g)(X_g^Te_g)^T$ | Arbitrary within-cluster covariance |
| Huber loss | Quadratic centre, linear tails | Clip large-residual gradients |
| Conditional quantile | $\inf\{q:P(Y\leq q\mid X=x)\geq\tau\}$ | Outcome distribution point at level $\tau$ |
| Pinball loss | $u(\tau-\mathbb1\{u<0\})$ | Proper asymmetric quantile loss |
| Quantile coverage | $m^{-1}\sum_i\mathbb1\{y_i\leq\hat q_\tau(x_i)\}$ | Held-out calibration diagnostic |

Penalty values depend on loss normalisation. Never copy `alpha` or $\lambda$ across implementations without checking the exact objective.

---

# Expanded glossary

**Active set:** Features with nonzero lasso or elastic-net coefficients at one tuning value and fit.

**Bias–variance trade-off:** Deliberate estimator bias may reduce sample sensitivity enough to improve prediction.

**Complete-case analysis:** Restrict analysis to rows with every required value observed.

**Conditional quantile:** Quantile of outcome distribution for a specified feature vector.

**Cook's distance:** Scaled aggregate fitted-value change caused by deleting one observation.

**Cross-fitting:** Build each training-row representation using a fit that excluded its fold.

**Elastic net:** Regression combining $L_1` and squared $L_2` penalties.

**Effective degrees of freedom:** Smooth flexibility measure such as $\operatorname{tr}(H_\lambda)$.

**Frequency encoding:** Replace category with its training frequency.

**Hat matrix:** Linear map from outcomes to OLS fitted outcomes.

**HC covariance:** Heteroskedasticity-consistent coefficient covariance estimator.

**Heteroskedasticity:** Conditional error variance changes across observations.

**Hierarchy principle:** Retain lower-order terms defining polynomial or interaction effects.

**Huber loss:** Quadratic near zero and linear for large absolute residuals.

**Influence:** Sensitivity of a fitted result to an observation's presence.

**Leverage:** Unusualness of a predictor combination relative to the design.

**Lasso:** Least squares with an $L_1$ penalty capable of exact zeros.

**MAP:** Posterior-mode estimate under a Bayesian model.

**MAR:** Missingness no longer depends on missing values after conditioning on observed information.

**MCAR:** Missingness independent of observed and missing data under the model.

**MNAR:** Missingness still depends on unobserved information after conditioning.

**Multiple imputation:** Several plausible completions used to propagate missing-value uncertainty.

**One-hot encoding:** Binary columns representing nominal categories.

**Pinball loss:** Asymmetric absolute loss whose population minimiser is a chosen quantile.

**Post-selection inference:** Inference accounting for data-driven model or hypothesis selection.

**Quantile crossing:** Predicted lower quantile exceeds a higher quantile at the same features.

**Regularisation:** Penalty or constraint controlling flexibility or encoding prior preference.

**Ridge:** Least squares with squared $L_2$ penalty.

**Sandwich covariance:** Bread matrices surrounding estimated score/error covariance meat.

**Shrinkage:** Pull coefficients toward a reference, usually zero.

**Soft thresholding:** Reduce magnitude by a threshold and set smaller values to zero.

**Spline:** Curve represented by weighted local basis functions.

**Studentised residual:** Residual divided by its estimated model-based standard deviation.

**Target encoding:** Category represented by smoothed, cross-fitted outcome summaries.

**VIF:** Conditional slope-variance inflation relative to an orthogonal reference.

---

# Research-paper reading ladder

| Paper | Problem | Replication focus |
|---|---|---|
| Rubin (1976) | Missing-data mechanisms and ignorability | Prediction and coverage under MCAR/MAR/MNAR |
| Hoerl & Kennard (1970) | Biased shrinkage for nonorthogonal regression | Stability under collinearity |
| Tibshirani (1996) | $L_1$ shrinkage and selection | Sparse prediction and unstable selection |
| Efron et al. (2004) | Efficient geometric lasso paths | LARS versus coordinate paths |
| White (1980) | Heteroskedasticity-consistent covariance | Interval coverage under changing variance |
| Huber (1964) | Robust location under contamination | Efficiency versus contamination |
| Koenker & Bassett (1978) | Regression quantiles | Conditional spread and coverage |
| Varma & Simon (2006) | Selection bias in CV estimates | Nested versus non-nested evaluation |

Six reading passes:

1. record citation, question, venue, and vocabulary;
2. scan abstract, introduction, figures, and conclusion;
3. build a symbol table;
4. reproduce one central derivation;
5. reconstruct one simulation or result; and
6. list assumptions, failure modes, and transfer limits.

Reading means reconstructing an argument well enough to identify where it could fail.

---

# Suggested eight-day study rhythm

| Segment | Minutes |
|---|---:|
| Retrieve previous exit check | 15 |
| Construct concept and intuition | 35 |
| Mathematical derivation | 40 |
| Code laboratory | 55 |
| Failure laboratory | 25 |
| Research reading | 20 |
| Bounded reflection | 10 |
| **Total** | **200** |

Regularisation and covariance days may require two sessions. Slow reconstruction is preferable to treating formulas as incantations.

---

# Common beginner misconceptions

| Misconception | Correction |
|---|---|
| Imputation neutrally fills blanks | It is a learned representation with assumptions |
| MAR means no reason for missingness | MAR allows dependence on observed information |
| Smoothing target encoding prevents leakage | Cross-fitting is still required |
| Polynomial regression is nonlinear estimation | It is linear in coefficients after feature expansion |
| Negative quadratic coefficient proves an optimum | Derivative, support, uncertainty, and causality still matter |
| Scaling makes raw features important | It defines conditioning and penalty units |
| Ridge discovers the true correlated feature | It stabilises a combined direction, not causal identity |
| Lasso zero proves no relationship | Zero is sample-, penalty-, and representation-dependent |
| VIF above 5 requires deletion | Cutoffs are conventions; respond to consequences |
| Large residual means high leverage | Residual concerns outcome; leverage concerns predictors |
| Cook flag means delete | It triggers provenance and sensitivity investigation |
| HC3 is robust regression | HC3 changes covariance, not fitted OLS coefficients |
| Many rows fix five-cluster inference | Independent cluster count remains five |
| Quantile is a confidence interval | It is a conditional outcome-distribution target |
| Global 80% coverage proves calibration everywhere | Subgroups can severely under-cover |
| Reproducible means scientifically correct | Reproducibility repeats both good and bad designs |

---

# Ethical and contextual questions

The KP setting is fictional and cannot evaluate actual districts. A real study requires local technical, financial, administrative, and community knowledge.

Ask:

- Who bears harm when remote cost is underpredicted?
- Does historical underinvestment become encoded as a district effect?
- Are appraisal fields measured consistently across locations?
- Is an access mode “rare” because data collection excluded it?
- Are missing surveys evidence of structural access barriers?
- Could contractor or district coefficients create unfair rankings?
- Are extreme projects genuine obligations rather than disposable outliers?
- Who may override the model, for which reasons, and with what record?
- Which groups need minimum calibration or error safeguards?
- Would deployment redirect resources away from poorly represented communities?

Technical competence without contextual accountability is not research readiness.

---

# Where Chapter 4 begins

Chapter 4 can introduce classification and generalised linear models while reusing everything learned here:

- Bernoulli likelihood, probability, odds, and log odds;
- logistic regression gradients and convex optimisation;
- regularised classification;
- calibration and proper scoring rules;
- decision thresholds and class imbalance;
- precision–recall analysis;
- subgroup error and fairness definitions;
- nonlinear tree benchmarks;
- external/prospective validation; and
- a locked classification study.

The practitioner does not leave linear-model thinking behind. Design matrices, interactions, penalties, influence, temporal validation, and post-selection caution all remain.

Research readiness begins when another person can inspect the complete argument, reproduce every number, identify every assumption, and understand exactly how strongly the evidence supports the final decision.
