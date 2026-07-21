# Chapter 4 Companion Guide: From Probabilities to Accountable Decisions

## A beginner-first guide to classification, probability, decisions, ensembles, explanations, fairness, and validation

This guide accompanies `chapter4.md`. It assumes no fluency in Python, mathematics, or statistics. When Chapter 4 uses a compact formula or a library object, this guide slows down, names every part, and connects it to a small example.

Chapter 4 is about much more than predicting `0` or `1`. It builds an accountable chain:

```text
define the event
    -> estimate a probability
    -> test probability quality
    -> choose an action rule
    -> examine who receives errors and actions
    -> test where the evidence transports
```

Each arrow can fail independently. A sophisticated model cannot repair a badly defined label. A well-calibrated probability does not choose an ethical policy. A fairness table cannot prove that an intervention helps.

---

## How to use this guide

For each day:

1. Read the companion section before the matching chapter section.
2. Work through the tiny numerical example with paper or a calculator.
3. Type the code rather than only reading it.
4. change one value and predict what will happen before rerunning it.
5. Complete the chapter's larger build exercise.
6. Deliberately create one of the listed failures.
7. Explain the exit check in ordinary language.

Keep these five objects separate throughout the chapter:

- **label:** what later counts as an event;
- **features:** what is known when the prediction is made;
- **score or probability:** what the fitted model returns;
- **action rule:** how a person or policy uses the output; and
- **claim:** what the study's evidence actually supports.

---

## 0. Foundations for a complete beginner

### 0.1 The running example

One row represents one fictional microhydro power project. At technical appraisal, the model estimates the chance that final inflation-adjusted cost will exceed 115% of the approved appraisal budget.

The positive label is:

```python
major_cost_overrun = 1
```

The negative label is:

```python
major_cost_overrun = 0
```

A high probability may send a project to senior engineering review. It must not automatically reject the project. Prediction and action are different stages.

The data are simulated teaching data. District or access-mode patterns in them are not facts about real places.

### 0.2 Essential Python vocabulary

Python evaluates expressions and stores results in names:

```python
probability = 0.20
threshold = 0.15
send_to_review = probability >= threshold
print(send_to_review)  # True
```

- `=` assigns a value.
- `>=` compares two values.
- `True` and `False` are Boolean values.
- `#` starts a comment that Python ignores.
- a function packages reusable instructions.

```python
def odds(p):
    return p / (1 - p)

print(odds(0.20))
```

The indented line belongs to the function. `return` sends a result back to the caller.

### 0.3 Arrays, tables, and labels

NumPy arrays store many values of the same general kind:

```python
import numpy as np

y = np.array([1, 0, 1, 0])
p = np.array([0.80, 0.30, 0.60, 0.10])
```

Here `y[i]` is an observed label and `p[i]` is a predicted probability for the same row.

Pandas DataFrames are labelled tables:

```python
import pandas as pd

projects = pd.DataFrame({
    "project_id": ["P1", "P2", "P3"],
    "probability": [0.72, 0.18, 0.41],
})
print(projects["probability"].mean())
```

Important patterns used in Chapter 4:

- `frame["column"]` selects one column;
- `.mean()` calculates an average;
- `.groupby("district")` forms groups;
- `.fit(X, y)` learns from features `X` and labels `y`;
- `.predict_proba(X)[:, 1]` selects the probability of class 1;
- `random_state=...` makes randomized code reproducible.

### 0.4 Mathematical symbols used in the chapter

- $Y$ is a random binary outcome; $y_i$ is the observed label for row $i$.
- $X$ is the complete feature table; $x_i$ is one row.
- $n$ is the number of rows.
- $sum$ means “add these terms.”
- $prod$ means “multiply these terms.”
- $log$ is the natural logarithm.
- $e^z$ is the exponential function.
- $mathbb E$ means an average over a probability distribution.
- $P(A)$ means the probability of event $A$.
- $P(A\mid B)$ means the probability of $A$ among cases for which $B$ is true.
- $\mathbb 1\{condition\}$ is 1 when the condition is true and 0 otherwise.
- $\partial$ denotes a derivative when a function has several inputs.
- $X^\top$ is the transpose of a matrix: rows and columns exchange roles.

### 0.5 Four different meanings of “positive”

Classification language can be misleading. “Positive” need not mean good.

- **actual positive:** the event occurred, $Y=1$;
- **predicted positive:** the action rule marked the case, $\hat Y=1$;
- **true positive:** the event occurred and the rule marked it;
- **false positive:** the rule marked it but the event did not occur.

In this chapter, a positive event is a major cost overrun, while a positive action is senior review.

### 0.6 Training, validation, calibration, and test data

- **Training data** fit model parameters.
- **Validation data** compare models and tune settings.
- **Calibration data** can learn a mapping from model scores to probabilities.
- **Test data** estimate final performance after choices are frozen.

One row may sometimes receive different roles through carefully designed cross-validation, but information from its evaluation role must not leak into its training role.

### 0.7 The prediction-time rule

Imagine stopping the project clock at technical appraisal. A feature is allowed only if it is available then. Final material bills, final costs, and events learned during construction are prohibited, even if they make prediction easy.

This simple question catches much leakage:

> Could a real reviewer know this value at the exact moment the prediction must be issued?

### 0.8 Installing the software

From a terminal in the project directory, install the core packages:

```bash
python -m pip install numpy pandas scipy matplotlib scikit-learn
```

- NumPy supplies arrays and numerical functions.
- pandas supplies labelled tables.
- SciPy supplies probability distributions and optimization utilities.
- Matplotlib draws diagnostic plots.
- scikit-learn supplies preprocessing, models, metrics, and validation tools.

SHAP is optional and needed only for its interpretation laboratories:

```bash
python -m pip install shap
```

An installation command changes the current Python environment. Record Python, operating-system, and package versions so another learner can reproduce the run. A seed reproduces pseudo-random choices under the same software; it does not make a scientific result universally stable.

### 0.9 Understanding `chapter4_data.py`

The Chapter 4 generator begins with Chapter 3's fictional project table, then creates an appraisal budget from appraisal-time variables:

```python
df = make_mhp_practitioner_data(n=n, seed=seed)
rng = np.random.default_rng(seed + 1)
```

`n` is the number of rows. A separate random generator produces reproducible budget noise without silently reusing Chapter 3's random stream.

A dictionary maps district labels to simulated budget adjustments:

```python
district_term = df["district"].map(district_budget_adjustment).to_numpy()
```

`.map(...)` looks up one adjustment per row. `.to_numpy()` converts the pandas Series into a NumPy array for arithmetic.

The appraisal-budget expression combines linear, squared, categorical, time, and random terms. It deliberately uses only information available at appraisal. The binary event is constructed by comparison:

```python
df["major_cost_overrun"] = (
    df["actual_cost_2025_million_pkr"]
    > 1.15 * df["appraisal_budget_2025_million_pkr"]
).astype(int)
```

The comparison first creates `True` or `False`; `.astype(int)` converts them to 1 and 0. Final cost is legitimate for constructing the later outcome but prohibited as a prediction feature.

Finally:

```python
df.sort_values(["start_year", "project_id"]).reset_index(drop=True)
```

puts earlier years first and gives rows a fresh numeric index. This ordering is essential for time-based splitting, but sorting alone does not make every generic cross-validator time-aware.

The generator retains final-cost fields for teaching audits. Their presence in the same DataFrame does not authorize using them. Real data would need accounting review, inflation rules, close-out dates, contract-change treatment, and dispute policies.

### 0.10 Pipelines and column transformations

A scikit-learn `Pipeline` runs steps in order:

```python
from sklearn.pipeline import Pipeline

procedure = Pipeline([
    ("preprocess", preprocessor),
    ("model", classifier),
])
```

When cross-validation calls `fit`, preprocessing learns only from that fold's training rows. Validation rows receive the stored transformation. This is why a pipeline is part of the scientific procedure, not merely convenient syntax.

A `ColumnTransformer` can send numeric and categorical columns through different paths. Numeric columns may receive median imputation and standardization. Categorical columns may receive most-frequent imputation and one-hot encoding. `handle_unknown="ignore"` prevents a previously unseen validation category from crashing prediction, but it does not prove that the new category is represented adequately.

For classifiers, the fitted procedure usually exposes:

```python
procedure.predict_proba(X_new)[:, 1]
```

Use the class-1 probability for probability metrics and registered action rules. `.predict(...)` normally applies an estimator default threshold and can discard information needed for calibration and decision analysis.

---

# Day 21 — The outcome, probability, and prediction contract

## 21.1 Begin with time, not an algorithm

A binary column hides a story. Define:

- **unit:** what one row represents;
- **eligibility:** which rows belong in the study;
- **index time:** when prediction occurs;
- **event:** what makes the outcome 1;
- **horizon:** how long outcomes are observed;
- **negative rule:** when 0 is justified;
- **unresolved rule:** what happens before the outcome matures; and
- **competing outcomes:** how cancellation, merger, or scope change are handled.

An unfinished project is unknown, not automatically 0. Coding it negative gives older projects more time to become positive and contaminates the meaning of the label.

If event timing and incomplete follow-up are central, a time-to-event model may be more honest than a fixed binary outcome.

## 21.2 Label construction is measurement

“Overrun above 15%” is not a natural fact until accounting rules are specified. Ask:

- Is the original or revised budget the denominator?
- Are inflation and exchange rates handled consistently?
- Does approved scope expansion count?
- When is the final account considered closed?
- Can accounting practice differ across locations?

The primary definition must be fixed before model comparison. Alternate 10%, 15%, and 20% definitions are useful sensitivity analyses, not invitations to choose the easiest result afterward.

## 21.3 Probability, odds, and log odds

A probability $p$ lies from 0 to 1. Odds compare the event probability with the non-event probability:

$$
o=\frac{p}{1-p}.
$$

For $p=0.20$:

$$
o=\frac{0.20}{0.80}=0.25.
$$

This is “one event for four non-events.” It is not a 25% probability.

Convert odds back to probability:

$$
p=\frac{o}{1+o}.
$$

Log odds are:

$$
\operatorname{logit}(p)=\log\left(\frac{p}{1-p}\right).
$$

Probabilities are bounded, but log odds range from negative infinity to positive infinity. At $p=0.5$, odds are 1 and log odds are 0.

```python
import numpy as np

def probability_to_log_odds(p):
    if not 0 < p < 1:
        raise ValueError("p must be strictly between 0 and 1")
    return np.log(p / (1 - p))

def log_odds_to_probability(z):
    return 1 / (1 + np.exp(-z))
```

## 21.4 Bernoulli outcomes

A Bernoulli variable has two possible values:

$$
Y\sim\operatorname{Bernoulli}(p),\qquad Y\in\{0,1\}.
$$

Its compact probability formula is:

$$
P(Y=y)=p^y(1-p)^{1-y}.
$$

Substitute both values:

- for $y=1$, it becomes $p$;
- for $y=0$, it becomes $1-p$.

Its mean and variance are:

$$
\mathbb E[Y]=p,
\qquad
\operatorname{Var}(Y)=p(1-p).
$$

Variance is greatest at $p=0.5$ and small near 0 or 1. Binary data therefore do not have constant conditional variance.

## 21.5 Prevalence and the constant baseline

Prevalence is the event fraction in a named sample and time window:

$$
\hat\pi=\frac{\text{number of events}}{\text{number of eligible resolved rows}}.
$$

It can change by period, group, label definition, follow-up, and sampling plan.

A no-feature probability model predicts training prevalence for every future row. If 18 of 100 training projects overrun, it predicts 0.18 for each validation project. This is a serious baseline for Brier score and log loss.

Never calculate a fold's baseline using the full dataset; that reveals future prevalence.

## 21.6 Selection into the labelled sample

Let $S=1$ mean a resolved label is available. Training estimates:

$$
P(Y=1\mid X,S=1),
$$

while deployment may need $P(Y=1\mid X)$. These differ if fast-closing or well-documented projects are overrepresented.

Compare labelled and unresolved eligible projects using appraisal-time features, year, group, and follow-up length. This does not prove selection is harmless, but it exposes obvious differences.

## 21.7 Beginner checklist

Before Day 22, be able to:

- describe the event without referring to a dataset column name;
- distinguish 0 from unknown;
- convert 0.20 to odds and log odds;
- explain why prevalence is not timeless;
- construct a training-only constant probability baseline; and
- name one way labelled projects could be unrepresentative.

---

# Day 22 — Logistic regression from Bernoulli likelihood

## 22.1 Why a straight probability line is awkward

A line $p=\beta_0+\beta_1x$ can predict $p<0$ or $p>1$. Logistic regression instead makes log odds linear:

$$
\log\left(\frac{p}{1-p}\right)=x^\top\beta.
$$

The sigmoid converts any real score $z=x^\top\beta$ into a valid probability:

$$
p=\sigma(z)=\frac{1}{1+e^{-z}}.
$$

“Linear logistic regression” means linear on the log-odds scale. The probability curve is S-shaped.

## 22.2 Vectors and the linear score

With two features and an intercept:

$$
z=\beta_0+\beta_1x_1+\beta_2x_2.
$$

The shorthand $x^\top\beta$ means multiply matching feature and coefficient values and add them. The intercept is represented by a constant feature equal to 1.

The sigmoid derivative is:

$$
\sigma'(z)=\sigma(z)[1-\sigma(z)]=p(1-p).
$$

It is largest at $p=0.5$. The same log-odds change has a larger probability effect near 0.5 than near 0 or 1.

## 22.3 Likelihood and log loss

Likelihood asks: under coefficients $\beta$, how plausible are all observed labels?

$$
L(\beta)=\prod_i p_i^{y_i}(1-p_i)^{1-y_i}.
$$

Products of many small probabilities are numerically awkward, so take logarithms. Logs turn products into sums:

$$
\ell(\beta)=\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)].
$$

Maximizing log likelihood is equivalent to minimizing negative average log likelihood:

$$
J(\beta)=-\frac1n\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)].
$$

This is binary cross-entropy or log loss. A confident wrong answer has very large loss. Code clips probabilities away from exact 0 and 1 before taking logs.

## 22.4 Gradient and Hessian without mystery

A derivative says how fast a quantity changes. A gradient collects one derivative for each coefficient:

$$
\nabla J=\frac1nX^\top(p-y).
$$

The vector $p-y$ contains probability errors. Multiplication by $X^\top$ accumulates how those errors align with each feature.

The Hessian records curvature:

$$
H=\frac1nX^\top WX,
\qquad W_{ii}=p_i(1-p_i).
$$

Because every diagonal weight is nonnegative, $H$ is positive semidefinite. The logistic negative log likelihood is convex: it has no deceptive local valleys, although separation or redundant columns can prevent a finite unique solution.

## 22.5 Gradient descent, Newton's method, and IRLS

Gradient descent repeatedly moves opposite the gradient:

$$
\beta_{new}=\beta_{old}-\eta\nabla J.
$$

$\eta$ is the learning rate. Too large can overshoot; too small can be slow.

Newton's method also uses curvature:

$$
\beta_{new}=\beta_{old}-H^{-1}\nabla J.
$$

For generalized linear models, the Newton calculation can be written as iteratively reweighted least squares (IRLS). It repeatedly updates probabilities, weights, and a working least-squares problem. This is a computational device, not a claim that binary errors are Gaussian.

## 22.6 Regularisation

An $L_2$ penalty adds:

$$
\frac{\lambda}{2}\sum_{j=1}^p\beta_j^2.
$$

It discourages very large slopes and stabilizes correlated or sparse designs. The intercept is usually not penalized. Scaling matters because the same numeric coefficient represents different changes in metres, kilometres, or rupees.

In scikit-learn, `C` is inverse penalty strength:

- smaller `C` means stronger regularisation;
- larger `C` means weaker regularisation.

Packages normalize loss differently, so identical numeric settings need not mean identical objectives.

## 22.7 Coefficients and odds ratios

A slope $\beta_j$ is the conditional log-odds change for a one-unit feature increase, holding represented covariates fixed. Exponentiating gives an odds ratio:

$$
OR=e^{\beta_j}.
$$

If $\beta_j=0.4$, then $OR\approx1.49$: odds multiply by 1.49. This is not a 49 percentage-point probability increase.

The local probability slope is:

$$
\frac{\partial p}{\partial x_j}=\beta_jp(1-p).
$$

Therefore the probability change depends on starting risk. Coefficients are associational unless a separate causal design justifies a causal interpretation.

## 22.8 Complete and quasi-complete separation

Complete separation occurs when a feature rule perfectly divides every 0 from every 1. Unpenalised coefficients can grow without bound while likelihood keeps improving. Quasi-separation leaves a few boundary ties.

Warning signs:

- convergence warnings;
- huge coefficients or standard errors;
- probabilities nearly exactly 0 or 1;
- perfect training accuracy; and
- extreme sensitivity to a few rows.

Check leakage and data errors first. Then consider justified category pooling, regularisation, bias-reduced inference, or more observations in sparse regions. Never celebrate perfect accuracy before this audit.

## 22.9 Minimal library workflow

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1.0, max_iter=10_000)
model.fit(X_train, y_train)
probability = model.predict_proba(X_valid)[:, 1]
```

- `fit` learns from training rows.
- `predict_proba` returns one column per class.
- `[:, 1]` selects class-1 probabilities.
- a convergence warning is evidence to investigate, not text to suppress.

---

# Day 23 — Proper scoring and calibration

## 23.1 Three different evaluation questions

- **Discrimination:** are event cases generally ranked higher?
- **Calibration:** do stated probabilities match event frequencies?
- **Decision value:** do resulting actions perform well under stated consequences?

A model can rank perfectly but call a 20% event a 90% risk. It can be calibrated overall while barely separating cases. One number cannot represent all three goals.

## 23.2 Calibration in ordinary language

Perfect calibration means:

$$
P(Y=1\mid P=p)=p.
$$

Among many comparable cases receiving 0.20, roughly 20% should experience the event. This validates repeated forecasts, not an individual claim that a particular case is “20% true.”

Calibration belongs to a population and time. It can break after prevalence, workflow, or conditional relationships change.

## 23.3 Brier score and propriety

The Brier score is mean squared probability error:

$$
\operatorname{Brier}=\frac1n\sum_i(p_i-y_i)^2.
$$

Lower is better. Suppose true probability is $q$ but someone reports $p$. Expected squared loss is:

$$
q(p-1)^2+(1-q)p^2=p^2-2pq+q.
$$

Its derivative is $2(p-q)$, which is zero only at $p=q$. Truthful reporting minimizes expected loss. That makes Brier score strictly proper.

## 23.4 Log loss and propriety

For one row:

$$
-y\log p-(1-y)\log(1-p).
$$

If true probability is $q$, expected loss is:

$$
-q\log p-(1-q)\log(1-p).
$$

Its derivative is $(p-q)/[p(1-p)]$, giving the minimum at $p=q$. Log loss punishes extreme wrong confidence more strongly than Brier score.

## 23.5 Reliability tables and diagrams

A reliability table groups similar probabilities, then reports:

- number of cases;
- mean predicted probability; and
- observed event fraction.

If a bin has mean prediction 0.31 and event rate 0.29, it is close in that bin. But binning is descriptive:

- tiny bins are uncertain;
- empty regions supply no evidence;
- equal-width and equal-count bins differ;
- results change with boundaries; and
- a diagonal-looking graph says nothing about ranking.

Always show counts or data density and preferably uncertainty.

## 23.6 Brier decomposition

For deliberately binned forecasts, Brier score can be written as:

$$
\text{reliability}-\text{resolution}+\text{uncertainty}.
$$

- **Reliability** penalizes mean forecast/event-rate disagreement within bins.
- **Resolution** rewards groups whose event rates differ from overall prevalence.
- **Uncertainty** is $\bar y(1-\bar y)$, the variability associated with prevalence.

With continuously varying forecasts, a binned decomposition depends on the grouping and can omit within-bin structure. Treat it as a diagnostic, not an invariant fingerprint.

## 23.7 Calibration intercept and slope

On untouched validation data, form predicted log odds $z$ and fit:

$$
\operatorname{logit}P(Y=1)=a+bz.
$$

Ideally $a=0$ and $b=1$.

- $a<0$ often means average risk is too high.
- $a>0$ often means average risk is too low.
- $b<1$ often means predictions are too extreme.
- $b>1$ often means the risk range is too narrow.

These are estimates with uncertainty. Training predictions make the assessment optimistic.

## 23.8 Sigmoid and isotonic calibration

Sigmoid calibration fits a logistic mapping from score to probability. It is relatively stable but assumes an S-shaped correction.

Isotonic calibration fits a flexible nondecreasing staircase. It assumes higher scores should never map to lower risks, but it can overfit small calibration sets.

Valid calibration patterns use:

- a dedicated later calibration period;
- cross-fitted development predictions; or
- calibration entirely inside every outer validation fold.

The final calibrated procedure must be judged on untouched data. Fitting and assessing a calibrator on the same probabilities is leakage.

## 23.9 Why ECE is insufficient

Expected calibration error averages absolute bin gaps. It changes with bins, can hide local failures, and is not a strictly proper scoring rule for choosing forecasts. Report a proper score, a reliability display, sample support, and uncertainty.

---

# Day 24 — Thresholds, ranking, and class imbalance

## 24.1 A probability is not an action

A threshold rule is:

$$
a_i=\mathbb 1\{p_i\ge t\}.
$$

Changing $t$ changes actions without changing the probability model. A default 0.5 has no special ethical or economic status.

## 24.2 Confusion matrix from four counts

| | Event | No event |
|---|---:|---:|
| Review | TP | FP |
| No review | FN | TN |

From these counts:

$$
\text{sensitivity}=\frac{TP}{TP+FN},
\qquad
\text{specificity}=\frac{TN}{TN+FP},
$$

$$
\text{precision}=\frac{TP}{TP+FP},
\qquad
\text{accuracy}=\frac{TP+TN}{n}.
$$

Sensitivity asks what fraction of events were reviewed. Precision asks what fraction of reviewed projects actually had events. Their denominators differ.

Balanced accuracy averages sensitivity and specificity, giving the two classes equal weight. Always report raw counts because rates can be unstable in small groups.

## 24.3 Deriving a cost threshold

Suppose unnecessary review costs $C_{FP}$ and missed overrun costs $C_{FN}$. With calibrated risk $p$:

$$
C_{review}=C_{FP}(1-p),
$$

$$
C_{no\ review}=C_{FN}p.
$$

Review when the first is no larger than the second. Rearranging gives:

$$
t=\frac{C_{FP}}{C_{FP}+C_{FN}}.
$$

If a miss costs five times an unnecessary review, $t=1/6\approx0.167$.

Real systems can have limited capacity, varying event severity, delayed reviews, human behavior, and project-specific costs. Then use explicit expected utility or constrained allocation rather than pretending one threshold captures everything.

## 24.4 Threshold tables and $F_1$

A validation threshold table should list review rate, sensitivity, specificity, precision, confusion counts, and cost for many plausible thresholds. Select according to the registered goal, never on the locked test.

$F_1$ is the harmonic mean of precision and recall:

$$
F_1=2\frac{PR}{P+R}.
$$

It omits true negatives and is not a monetary cost, probability score, capacity rule, or severity model. Use it only when its trade-off matches the real task.

## 24.5 ROC curves and AUC

An ROC curve plots true-positive rate against false-positive rate over thresholds. ROC AUC is the probability that a randomly chosen positive gets a higher score than a randomly chosen negative, with half credit for ties.

ROC AUC measures ranking, not calibration. It averages threshold regions that operations may never use. A partial ROC region can be useful only if selected in advance.

## 24.6 Precision–recall curves

A PR curve plots precision against recall. Its no-skill precision baseline is prevalence, so it makes rare-event difficulty visible.

Precision changes when prevalence changes, even if class-conditional score distributions remain fixed. Average precision is a particular weighted summary across recall increments and is not identical to every trapezoidal “PR AUC.” Name the implementation.

## 24.7 What imbalance does and does not mean

Class imbalance merely describes unequal class frequency. It can make accuracy misleading, rare-event estimates noisy, and random folds deficient in events.

Possible responses answer different questions:

- **class weighting** changes how fitting errors are valued;
- **over/undersampling** changes the training distribution;
- **SMOTE** synthesizes points between minority neighbors;
- **threshold movement** changes actions while retaining the probability model.

Resampling must occur inside each training fold. Synthetic cases are not new independent event evidence and may violate mixed, constrained, geographic, or temporal structure. Weighted or resampled outputs may require probability correction and representative calibration.

## 24.8 Capacity-constrained review

If only 30 cases can be reviewed, a top-30 policy ranks each cohort. Its effective threshold changes with the cohort.

Evaluate event yield, recall among the top $k$, selection stability, subgroup allocation, and the usefulness of actual probabilities. Describe it as a rank allocation policy, not a fixed-threshold classifier.

---

# Day 25 — Nonlinear, kernel, and multiple-class models

## 25.1 Several meanings of nonlinear

Logistic regression has an S-shaped probability even when log odds are linear. Nonlinearity may instead enter through:

- polynomial terms such as $x^2$;
- interactions such as $x_1x_2$;
- spline basis functions;
- neighborhoods or kernels; or
- a complex learned decision surface.

Always name the scale on which a relationship is linear or nonlinear.

## 25.2 Polynomial and interaction logistic models

For a centered quadratic:

$$
\operatorname{logit}(p)=\beta_0+\beta_1x_c+\beta_2x_c^2.
$$

Centering means subtracting a meaningful reference, which makes coefficients and computation easier to interpret. The probability derivative is:

$$
\frac{\partial p}{\partial x}=p(1-p)(\beta_1+2\beta_2x_c).
$$

For an interaction:

$$
\operatorname{logit}(p)=\beta_0+\beta_1x_1+\beta_2x_2+\beta_3x_1x_2.
$$

The slope for $x_1$ depends on $x_2$. Log-odds interaction and probability interaction are not the same object, and neither proves causal effect modification.

## 25.3 Splines and generalized additive models

A spline represents a smooth curve using several basis columns $B_m(x)$:

$$
\operatorname{logit}(p)=\beta_0+\sum_m\theta_mB_m(x).
$$

A generalized additive model combines smooth feature functions:

$$
g(\mathbb E[Y\mid X])=\beta_0+f_1(x_1)+\cdots+f_p(x_p).
$$

For binary logistic GAMs, $g$ is the logit. Additivity improves readability but does not automatically learn interactions. Knot count, smoothness penalty, and chosen features belong inside validation. Plot curves with data density and uncertainty; smooth extrapolation beyond support is still unsupported.

## 25.4 K-nearest-neighbor classification

KNN finds the $k$ closest training points and averages their labels:

$$
\hat p(x)=\frac1k\sum_{i\in N_k(x)}y_i.
$$

Small $k$ is flexible and noisy. Large $k$ is smoother and drifts toward prevalence.

Distance silently encodes assumptions about scaling, categories, missingness, and feature relevance. Without scaling, a variable measured in thousands can dominate one measured from 1 to 5.

In high dimensions, points become sparse and distances often become similar: the curse of dimensionality. KNN needs disciplined features, adequate local data, and no claims beyond observed neighborhoods.

## 25.5 Support-vector machines

For labels encoded $-1,+1$, an SVM finds a separating score $f(x)$ with a wide margin. Soft-margin SVM uses hinge loss:

$$
\max(0,1-yf(x)).
$$

The objective balances a small weight norm with margin violations. In the common `C` parameterization:

- small `C` allows more violations and stronger regularisation;
- large `C` emphasizes fitting violations.

Support vectors are points on or inside the margin that directly shape the solution.

## 25.6 Kernels and the RBF kernel

A kernel replaces an explicit inner product with a similarity function. The radial-basis-function kernel is:

$$
K(x,x')=\exp[-\gamma\lVert x-x'\rVert_2^2].
$$

Large $\gamma$ makes similarity very local and permits wiggly boundaries. Small $\gamma$ creates broader, smoother similarity. Scaling is essential, and `C` plus $\gamma$ need honest nested tuning.

An SVM margin is not a probability. `probability=True` adds a calibration-like step and extra computation; its probability quality still requires untouched evaluation.

## 25.7 Multiclass classification and softmax

Multiclass outcomes have more than two mutually exclusive categories. Softmax turns class scores $z_k$ into probabilities summing to one:

$$
P(Y=k\mid x)=\frac{e^{z_k}}{\sum_j e^{z_j}}.
$$

Subtract the largest score before exponentiating for numerical stability. Cross-entropy sums the negative log probability assigned to each observed class.

One-versus-rest and one-versus-one decompose the problem differently and can have different calibration behavior. Do not invent classes to hide censoring or competing-event problems.

## 25.8 Macro, micro, and weighted metrics

- **Macro:** calculate each class metric, then average classes equally.
- **Weighted macro:** weight class metrics by class count.
- **Micro:** pool all class decisions before calculating the metric.

Macro can give ten examples the same influence as ten thousand; micro can conceal a rare-class failure. Report class support and per-class results.

## 25.9 Multilabel classification and chains

Multilabel means several labels can be true for one row. Independent models ignore dependence. A classifier chain predicts labels sequentially and uses earlier predictions as later inputs.

During training, true upstream future labels would leak information unavailable at deployment. Train downstream stages using out-of-fold upstream predictions, and feed only predictions at deployment. Chain order is itself a modeling choice.

---

# Day 26 — Classification trees and random forests

## 26.1 How a tree makes probabilities

A tree repeatedly splits feature space. Each terminal leaf predicts its event fraction:

$$
\hat p_L=\frac{\text{events in leaf }L}{\text{rows in leaf }L}.
$$

Every case in the leaf receives the same probability. A leaf with one event and no non-events reports 1.0 but has almost no support.

## 26.2 Gini impurity and entropy

For class proportions $p_k$:

$$
I_G=1-\sum_kp_k^2.
$$

For two classes, $I_G=2p(1-p)$. It is 0 for a pure node and largest at an even split.

Entropy is:

$$
I_H=-\sum_kp_k\log p_k,
$$

where $0\log0$ is defined as 0. Both measure mixing, but they need not choose the same split.

## 26.3 Weighted split improvement

A split is judged by parent impurity minus size-weighted child impurity:

$$
\Delta I=I_P-\frac{n_L}{n_P}I_L-\frac{n_R}{n_P}I_R.
$$

Weighting prevents a tiny pure child from receiving all the credit while leaving most cases mixed. The tree greedily chooses the best available split now; it does not search every possible future tree.

## 26.4 Controlling tree complexity

- `max_depth` limits path length.
- `min_samples_leaf` protects leaf support.
- `min_samples_split` controls when splitting is considered.
- `max_leaf_nodes` caps leaves.
- `ccp_alpha` controls cost-complexity pruning.

Large leaves often improve probability stability. Select complexity inside validation and inspect leaf event counts, not only accuracy.

## 26.5 Preprocessing and extrapolation

Numeric trees do not need standardization because feature order is unchanged by a strictly monotone transformation. But categorical handling, missing values, rare categories, and high-cardinality IDs remain implementation-specific.

Beyond the training range, queries often fall into the same outer leaf, producing flat predictions. This is not evidence of safe extrapolation; it is one unsupported behavior.

## 26.6 Bagging and why correlation matters

Bagging fits trees on bootstrap samples and averages probabilities:

$$
\hat p_{bag}(x)=\frac1M\sum_m\hat p_m(x).
$$

If tree predictions have variance $\sigma^2$ and pairwise correlation $\rho$, the average has idealized variance:

$$
\rho\sigma^2+\frac{1-\rho}{M}\sigma^2.
$$

More trees shrink the independent portion but cannot remove shared error. Diversity matters.

## 26.7 Random forests

A random forest also offers each split only a random feature subset. Strong variables cannot dominate every tree, which can reduce correlation and let alternative patterns contribute.

Important settings include tree count, candidate features per split, leaf size, depth, bootstrap size, and class weights. More trees reduce Monte Carlo noise; they do not repair leakage, bias, or miscalibration.

## 26.8 Out-of-bag prediction

A bootstrap sample of size $n$ contains about 63.2% distinct rows. For each training row, an OOB prediction averages only trees that did not include that row.

OOB estimates are useful internal evidence under bootstrap-like exchangeability, but ordinary bootstraps mix years and groups. They do not replace forward, grouped, geographic, or external validation.

## 26.9 Forest probability quality and importance

Forest probabilities average leaf proportions. Leaf size, class weighting, and sampling alter their spread, so good ROC AUC does not guarantee calibration.

Impurity importance credits accumulated split gains. It can favor continuous or high-cardinality features and split credit strangely among correlated variables. Held-out permutation methods help, but no importance score is a causal effect.

---

# Day 27 — Gradient boosting and honest tuning

## 27.1 Bagging versus boosting

Bagging fits many learners mostly independently and averages them. Boosting builds sequentially:

$$
F_M(x)=F_0(x)+\eta\sum_{m=1}^Mh_m(x).
$$

Each new weak learner tries to reduce the current loss. $F$ is a raw score, $h_m$ is often a shallow tree, $\eta$ is learning rate, and $M$ is the number of stages.

## 27.2 Gradient descent in function space

Boosting changes a function rather than one fixed coefficient vector. At stage $m$, calculate negative loss gradients:

$$
r_{im}=-\left.\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}\right|_{F=F_{m-1}}.
$$

Fit a weak learner to these pseudo-residuals, then add a small version of it. For squared error they equal ordinary residuals. For other losses they do not.

## 27.3 Logistic pseudo-residuals and initial score

For logistic loss, $p=\sigma(F)$ and:

$$
\frac{\partial L}{\partial F}=p-y.
$$

Therefore the negative gradient is:

$$
r=y-p.
$$

Boosting begins at training log prevalence:

$$
F_0=\log\left(\frac{\hat\pi}{1-\hat\pi}\right).
$$

Its sigmoid is the constant prevalence baseline. Each tree then adds structure.

## 27.4 Interacting complexity controls

- smaller learning rate usually needs more stages;
- deeper or larger trees learn higher-order interactions;
- minimum leaf size protects local support;
- row and feature subsampling add randomness;
- penalties and leaf rules depend on the implementation.

There is no universal best range. A parameter setting only has meaning with a package version, dataset, preprocessing procedure, and search budget.

## 27.5 Histogram boosting

Histogram methods bin numeric feature values and search over bins, saving time and memory. Implementations differ in missing-value routing, categorical support, objectives, and regularisation. XGBoost, LightGBM, CatBoost, and scikit-learn estimators are related, not interchangeable.

Read the documentation for the exact version. Raw strings are not automatically accepted merely because numeric missing values are.

## 27.6 Early stopping is model selection

Early stopping chooses the training length using validation loss. A hidden random validation fraction can violate a temporal design.

Use an order-respecting validation period, tune iteration count inside temporal folds, or implement expanding-window selection. The locked test must never decide when to stop.

## 27.7 Grid, random, and sequential search

- **Grid search** evaluates a fixed Cartesian grid; transparent but wasteful in many dimensions.
- **Random search** samples from stated distributions; often explores influential dimensions better for a fixed budget.
- **Bayesian/sequential search** uses prior trials to choose promising next trials; efficient but still adaptive and capable of overfitting validation.

Scale parameters often need log-uniform sampling. Uniformly sampling `C` from 0.0001 to 1000 spends almost all draws at the large end; uniformly sampling $\log_{10}C$ spreads trials across orders of magnitude.

Record spaces, distributions, seed, trial count, failures, stopping, hardware, and every result.

## 27.8 Nested evaluation

The inner loop tunes preprocessing, features, hyperparameters, calibration, resampling, early stopping, and possibly threshold. The outer loop evaluates the resulting selection procedure.

Outer splits must still match deployment. Random nested CV does not cure temporal or group mismatch.

## 27.9 Fair search budgets and uncertainty

A default logistic model versus 1,000 boosted-model trials is a comparison of complete procedures and budgets, not pure algorithms. Report performance as well as computation.

The best inner score is the minimum of many noisy estimates and is optimistically selected. Use a locked test or outer folds, report all fold scores and paired differences, align uncertainty with the sampling unit, and focus on practical differences.

---

# Day 28 — Interpreting fitted models without inventing causes

## 28.1 Specify the explanation request

Before choosing a method, state:

- object: one prediction, global behavior, or performance dependence;
- audience: developer, reviewer, policymaker, or affected party;
- scale: probability, log odds, margin, loss, or action;
- reference: compared with what background;
- scope: fitted model or real-world mechanism; and
- use: debugging, contesting, understanding, or justifying.

No explanation method turns prediction into causation.

## 28.2 Intrinsic and post-hoc interpretation

Logistic coefficients, sparse scorecards, shallow trees, additive shape models, and monotonic models expose constrained structures. Post-hoc methods probe a fitted model afterward.

A simple model can be readable and wrong. A post-hoc method can be useful but adds perturbation, approximation, and reference assumptions. Match fidelity and complexity to the use.

## 28.3 Coefficients are not generic importance

Logistic coefficients depend on units, coding, correlated variables, interactions, regularisation, omissions, measurement error, and sampling. Comparing a one-kilometre coefficient with a one-million-rupee coefficient is meaningless without a defined perturbation.

## 28.4 Permutation importance

On held-out data, shuffle feature $j$ and measure score deterioration:

$$
I_j=S(f,D)-S(f,D_j^{perm}).
$$

It asks how much this fitted model's held-out performance relies on information disrupted by this permutation. It does not ask how much the feature causes the outcome.

Correlated substitutes can make each feature look weak. Unconditional shuffling can create impossible combinations. Consider grouped or conditional permutations and clearly state the added assumptions. Retraining without a feature asks a different question.

## 28.5 Partial dependence and ICE

Partial dependence replaces a focal feature across rows and averages predictions:

$$
PD_S(x_S)=\mathbb E_{X_C}[f(x_S,X_C)].
$$

Other features retain their row-specific values; they are not all fixed at one shared value. Correlated features can create unsupported synthetic combinations.

ICE shows the corresponding curve for one row:

$$
ICE_i(x_S)=f(x_S,x_{iC}).
$$

Diverging ICE curves reveal model heterogeneity hidden by an average. Both remain model perturbations, not causal experiments. Show support and output scale.

## 28.6 Accumulated local effects

ALE estimates local prediction changes inside observed feature intervals, averages them, accumulates them across the range, and centers the result. It often stays closer to observed joint support than PDP under correlation.

It still depends on bins, density, and meaningful local perturbations. Sparse regions remain uncertain.

## 28.7 Local surrogate explanations

A local surrogate perturbs points near one case, queries the black box, weights by proximity, and fits a simple approximation. Results depend on neighborhood, scaling, perturbation distribution, kernel, surrogate family, and seed.

Always report local fidelity. A readable but inaccurate surrogate does not explain the original model.

## 28.8 Shapley values and SHAP

Shapley values average a feature's marginal contribution over every order in which features can join a coalition:

$$
\phi_j=\sum_{S\subseteq N\setminus\{j\}}
\frac{|S|!(M-|S|-1)!}{M!}[v(S\cup\{j\})-v(S)].
$$

For two features, each can enter first or second, so average its contribution across the two orders. Additive SHAP explanations satisfy:

$$
f(x)=\phi_0+\sum_j\phi_j
$$

on the explainer's stated scale.

The hard question is what $v(S)$ means when features are “missing.” Interventional-style explanations can break dependencies; conditional-style explanations preserve modeled dependencies and can credit proxies. Neither is automatically causal.

State the algorithm, background population, dependence treatment, output scale, link, approximation, and version. Log-odds contributions cannot simply be read as probability-point contributions because the sigmoid is nonlinear.

## 28.9 Stability and legitimacy

Repeat explanations across seeds, temporal folds, bootstrap fits, correlated substitutions, and plausible backgrounds. Instability may be the most decision-relevant finding.

Explanations cannot repair an invalid label, excluded population, leakage, poor calibration, unfair allocation, lack of recourse, or an ineffective intervention. Explanation follows system validity; it does not substitute for it.

---

# Day 29 — Fairness, external validation, and the locked study

## 29.1 Begin before metrics

Ask whether the task is legitimate, the label is equally measured, the action helps or burdens, important populations are absent, records and decisions can be contested, and an accountable institution can respond to harm.

A parity number diagnoses one distributional property. It is not a complete ethical theory.

## 29.2 Group attributes have different roles

Separate:

- attributes used as model features;
- attributes retained only for auditing;
- attributes used in decision policy; and
- governance or legal restrictions.

Dropping a group attribute does not prevent proxies or historical labels from reproducing disparities. Retaining an attribute for audit does not require using it for prediction.

## 29.3 Demographic parity

Demographic parity asks for equal positive-action rates:

$$
P(\hat Y=1\mid A=a)=P(\hat Y=1\mid A=b).
$$

It may matter for a scarce benefit or burden, but it ignores observed outcomes. If review is supportive and structural barriers create different risks, equal review rates could under-support a group. Meaning depends on the action.

## 29.4 Equal opportunity and equalized odds

Equal opportunity asks for equal TPR among actual events:

$$
P(\hat Y=1\mid Y=1,A=a)=P(\hat Y=1\mid Y=1,A=b).
$$

Equalized odds requires both TPR and FPR equality, equivalently $\hat Y$ independent of $A$ given $Y$. These describe thresholded actions, not probability calibration.

## 29.5 Predictive parity and group calibration

Predictive parity asks for equal precision among positive actions. Group calibration asks that within each group, cases assigned probability $p$ experience events at rate $p$.

These condition on different quantities. With different base rates and imperfect prediction, several desirable parity properties generally cannot all hold. Choosing among them is a normative and operational decision, not metric shopping.

## 29.6 Measurement disparity and small groups

If remote projects close accounts later and unresolved cases become 0, the dataset can manufacture group differences. Audit label completeness, follow-up, missing features, measurement errors, and appeals by group before adjusting algorithms.

Intersectional groups can reveal hidden failures but create small samples and repeated comparisons. Report counts, confusion cells, uncertainty, pre-specified versus exploratory status, privacy rules, and collection plans. Lack of evidence is not evidence of equality.

## 29.7 Types of validation

- **Internal random:** similar sampled rows from one source.
- **Internal grouped:** whole known groups held out.
- **Temporal:** later periods in the broad same system.
- **Geographic/external:** independent represented locations or institutions.
- **Prospective:** model frozen before future cases accrue.
- **Impact evaluation:** causal comparison of using versus not using the intervention.

A future-year test in the same districts is temporal, not broad geographic validation. Prospective prediction performance still does not prove that review improves outcomes.

## 29.8 Dataset shift

Training and deployment distributions may differ:

- **covariate shift:** $P(X)$ changes while $P(Y\mid X)$ is assumed stable;
- **label shift:** $P(Y)$ changes while $P(X\mid Y)$ is assumed stable;
- **concept/conditional shift:** $P(Y\mid X)$ changes.

These are assumptions, not labels that an unlabeled dashboard proves. Monitor feature distributions, missingness, risk distribution, mature prevalence, calibration, actions, subgroup metrics, pipeline changes, overrides, and appeals.

## 29.9 The locked Chapter 4 protocol

The chapter freezes:

- projects through 2022 for development;
- 2019–2022 as forward validation years;
- 2023–2025 as a locked temporal test;
- log loss as the primary metric, equally averaged across validation years;
- Brier, ROC AUC, and average precision as secondary views;
- a $1/6$ action threshold from FP cost 1 and FN cost 5;
- access mode and district audits with raw counts; and
- prior, logistic, engineered logistic, forest, and histogram-boosting candidates.

The test is opened once after selection. Threshold, calibration, preprocessing, and candidates are part of the frozen procedure.

## 29.10 Reading the benchmark code

The code:

1. creates and time-sorts simulated projects;
2. separates development from locked test;
3. asserts prohibited future fields are not features;
4. builds all candidate pipelines;
5. fits each candidate on years before each validation year;
6. selects by mean validation log loss;
7. refits the selected procedure on all development rows;
8. predicts the locked test once;
9. compares with a fixed logistic reference using paired row losses;
10. produces subgroup and reliability tables; and
11. saves scores, predictions, audits, and environment metadata.

The custom transformer learns centers only in `fit`, then reuses them in `transform`. `clone` creates a fresh unfitted copy. Pipelines ensure imputation, scaling, encoding, feature engineering, and modeling obey the same fold boundary.

## 29.11 Why the threshold is not tuned

The threshold comes from a registered simplified cost ratio; the model is selected for probability quality. Tuning the threshold on the locked test would spend the test and blur probability estimation with action optimization.

If costs are uncertain, pre-register sensitivity or decision-curve analyses over plausible costs rather than selecting whichever makes test results attractive.

## 29.12 Subgroups and uncertainty

For every group report sample size, events, prevalence, mean probability, Brier score, review rate, TP, FP, FN, TN, sensitivity, specificity, precision, and cost.

Differences can arise from chance, prevalence, measurement, model error, threshold effects, or structural inequity. A rate alone cannot identify the mechanism.

The chapter's paired bootstrap resamples rows. Shared year or district shocks violate simple row exchangeability, so its interval is descriptive. Better research aligns resampling with independent sampling units and collects more periods and sites.

## 29.13 Prospective shadow deployment

Before influencing real decisions:

1. freeze code, environment, feature definitions, and policy;
2. issue timestamped predictions without changing decisions;
3. verify feature availability and latency;
4. record human actions, overrides, and reasons;
5. wait until outcomes mature;
6. assess probability, action, subgroup, and workflow results;
7. establish contest, incident, pause, and retraining rules; and
8. consider a separate ethical impact evaluation.

Once a deployed system changes decisions, later labels reflect both the world and the intervention. Drift monitoring must account for that feedback.

## 29.14 The bounded conclusion

A defensible conclusion names the represented projects and period, selected procedure, locked probability and action results, subgroup limitations, and next validation step. It explicitly does not claim causal risk factors, fairness, intervention benefit, or transport to untested regions.

## 29.15 Structure of the final research report

The report should let a reader reconstruct decisions, not only admire the winning model:

1. **Abstract:** event, population, prediction time, design, locked result, and main limitation.
2. **Label study:** data source, window, unresolved cases, and group measurement audit.
3. **Prediction contract:** intended and prohibited uses, feature timing, and action.
4. **Methods:** complete candidates, folds, search budget, scores, threshold, and uncertainty.
5. **Validation:** every candidate and year, including failures and near ties.
6. **Locked results:** probability, ranking, calibration, and action evidence.
7. **Subgroup audit:** raw counts, rates, uncertainty, and exploratory labels.
8. **Interpretation:** method, output scale, background, support, fidelity, and stability.
9. **Transport:** temporal, geographic, workflow, and outcome-maturity limits.
10. **Governance:** human review, contestability, monitoring, incidents, and retraining.
11. **Reproducibility:** data and code versions, environment, seeds, predictions, and deviations.
12. **Conclusion:** a bounded statement separating prediction from cause and benefit.

Save every locked-test probability, not only aggregate metrics. Predictions permit paired comparisons, reliability reconstruction, subgroup audit, and later error review. Preserve deviations from the protocol rather than quietly rewriting the plan.

---

# Chapter 4 synthesis and practical laboratories

## S.1 The six evidence layers

1. **Event measurement:** Is the label valid and mature?
2. **Probability estimation:** Does the fitted procedure estimate supported risks?
3. **Probability evaluation:** Are ranking and calibration adequate on untouched data?
4. **Decision policy:** Are consequences, capacity, and human roles explicit?
5. **Group and governance audit:** Who receives errors, actions, recourse, and accountability?
6. **Transport and impact:** Does performance persist elsewhere, and does use help?

Later layers cannot repair earlier ones.

## S.2 Derivations to complete by hand

1. Convert probability to odds, log odds, and back.
2. Derive Bernoulli mean and variance from the two outcomes.
3. Derive binary log loss from the likelihood.
4. Derive $X^\top(p-y)/n$ and $X^\top WX/n$.
5. Show why the Hessian is positive semidefinite.
6. Derive $\partial p/\partial x_j=\beta_jp(1-p)$.
7. Prove Brier and log-loss propriety.
8. Derive the binned Brier decomposition and state its bin dependence.
9. Derive the cost threshold.
10. Use Bayes' rule to express precision through prevalence, sensitivity, and specificity.
11. Derive binary Gini $2p(1-p)$.
12. Derive variance of an average of equally correlated trees.
13. Derive the logistic pseudo-residual $y-p$.
14. Calculate all coalitions for a three-feature Shapley game.
15. Express equalized odds as conditional independence.

## S.3 Ten laboratories

### Laboratory A: label maturity

Simulate different close-out times. Compare coding incomplete cases as 0 with waiting for a fixed outcome window. Plot prevalence and calibration by project age.

### Laboratory B: logistic optimization

Compare gradient descent, Newton/IRLS, and a library solver under scaling, collinearity, and near separation. Record loss, gradient norm, iterations, and runtime.

### Laboratory C: probability quality

Build two forecasts with equal 0.5-threshold accuracy but different confidence. Compare Brier, log loss, reliability, calibration intercept, and slope.

### Laboratory D: prevalence shift

Keep positive and negative score distributions fixed while changing class frequency. Observe ROC stability and changes in precision, PR curves, probability calibration, and decision cost.

### Laboratory E: nonlinear support

Fit spline logistic, KNN, RBF SVM, forest, and boosting on a known two-dimensional probability surface. Evaluate within support and beyond it.

### Laboratory F: forest correlation

Add redundant strong predictors and vary feature subsampling. Measure individual-tree performance, tree correlation, ensemble Brier score, and stability.

### Laboratory G: adaptive tuning

Search 5, 50, and 500 configurations on noise. Compare best inner validation loss with outer and independent-test loss.

### Laboratory H: explanation disagreement

Fit near-tied models using correlated features differently. Compare coefficients, impurity and permutation importance, PDP, ICE, ALE, and SHAP under multiple backgrounds.

### Laboratory I: fairness trade-offs

Simulate two groups with different prevalence. Compare one threshold, equal-TPR thresholds, equalized-odds post-processing, and equal review rates. Preserve every individual decision change.

### Laboratory J: prospective registry

Store feature timestamps, data version, model version, probability, threshold, action, override, outcome-maturity date, and final event. Try reconstructing omitted fields later to see why prospective logging matters.

## S.4 Model-family comparison questions

For every candidate ask:

- What representation does it assume?
- Does it need scaling or encoding?
- What probability target does fitting optimize?
- Which settings were learned from validation data?
- How does it behave outside feature support?
- Does it return probabilities or only scores?
- How will it be calibrated?
- What computation and monitoring does it require?
- How stable are its predictions and explanations?
- What claim does the evaluation design permit?

---

# Formula sheet with plain-language readings

| Formula | Plain-language reading |
|---|---|
| $p/(1-p)$ | event odds |
| $\log[p/(1-p)]$ | log odds on the real line |
| $1/(1+e^{-z})$ | turn a real score into probability |
| $p^y(1-p)^{1-y}$ | probability of one binary observation |
| $p(1-p)$ | Bernoulli variance |
| $-[y\log p+(1-y)\log(1-p)]$ | penalty for one probability forecast |
| $X^\top(p-y)/n$ | logistic-loss gradient |
| $X^\top WX/n$ | logistic curvature |
| $e^{\beta_j}$ | conditional odds multiplier for one unit |
| $n^{-1}\sum(p_i-y_i)^2$ | Brier score |
| $P(Y=1\mid P=p)=p$ | calibration definition |
| $TP/(TP+FN)$ | sensitivity or recall |
| $TN/(TN+FP)$ | specificity |
| $TP/(TP+FP)$ | precision |
| $(TPR+TNR)/2$ | balanced accuracy |
| $2PR/(P+R)$ | $F_1$ summary |
| $C_{FP}/(C_{FP}+C_{FN})$ | simplified cost threshold |
| $1-\sum p_k^2$ | Gini impurity |
| $I_P-(n_LI_L+n_RI_R)/n_P$ | split improvement |
| $M^{-1}\sum_m\hat p_m(x)$ | bagged probability |
| $\rho\sigma^2+(1-\rho)\sigma^2/M$ | variance left after averaging correlated learners |
| $\exp[-\gamma\lVert x-x'\rVert^2]$ | RBF similarity |
| $\max(0,1-yf)$ | hinge loss |
| $e^{z_k}/\sum_je^{z_j}$ | softmax class probability |
| $F_m=F_{m-1}+\eta h_m$ | one boosting update |
| $y-p$ | logistic negative gradient |
| $S(D)-S(D_j^{perm})$ | permutation score decrease |
| $\mathbb E[f(x_S,X_C)]$ | partial dependence |
| $f(x)=\phi_0+\sum_j\phi_j$ | additive SHAP explanation |

---

# Expanded glossary

**Action threshold** — A rule converting probability or score into an action.

**Average precision** — A prevalence-sensitive weighted summary of precision over recall increments.

**Bagging** — Fitting learners on bootstrap samples and averaging their outputs.

**Baseline** — A simple reference procedure a candidate should improve upon.

**Bernoulli distribution** — A probability model for one outcome coded 0 or 1.

**Brier score** — Mean squared error of binary probability forecasts.

**Calibration** — Agreement between forecast probabilities and event frequencies in a stated population.

**Classifier chain** — A multilabel method that feeds earlier label predictions into later models.

**Complete separation** — Perfect feature-based class division that makes unpenalised logistic estimates diverge.

**Concept shift** — A change in $P(Y\mid X)$.

**Confusion matrix** — TP, FP, FN, and TN action counts.

**Covariate shift** — A change in $P(X)$ under an assumed stable $P(Y\mid X)$.

**Cross-entropy** — Negative log likelihood used to assess or fit class probabilities.

**Demographic parity** — Equality of positive-action rates across groups.

**Discrimination** — Ability to rank events above non-events.

**Early stopping** — Choosing training length using validation performance.

**Equal opportunity** — Equality of true-positive rates across groups.

**Equalized odds** — Equality of both true-positive and false-positive rates across groups.

**Event horizon** — The period during which a future event is assessed.

**Generalized additive model** — A linked outcome mean represented by a sum of smooth feature functions.

**Gini impurity** — A measure of class mixing within a tree node.

**Gradient** — The vector of first derivatives of an objective.

**Gradient boosting** — Sequential additive fitting along negative loss gradients.

**Hessian** — A matrix of second derivatives describing local curvature.

**ICE** — One row's fitted prediction curve as a focal feature is varied.

**Impact evaluation** — A causal design estimating the effect of using an intervention.

**Index time** — The moment at which eligibility, features, and prediction are anchored.

**Isotonic calibration** — A flexible nondecreasing mapping from score to probability.

**Kernel** — A function representing similarity or an implicit feature-space inner product.

**Label shift** — A prevalence change under assumed stable $P(X\mid Y)$.

**Likelihood** — Plausibility of observed data as a function of model parameters.

**Log loss** — Negative log probability assigned to observed classes.

**Log odds** — Logarithm of event odds.

**Macro average** — Equal-weight mean of class-specific metrics.

**Multiclass** — More than two mutually exclusive classes.

**Multilabel** — Several binary labels may be true for one row.

**Nested validation** — Inner model selection surrounded by outer procedure evaluation.

**Odds ratio** — Multiplicative change in conditional odds.

**Out of bag** — Rows omitted from a tree's bootstrap sample and usable for that tree's internal assessment.

**Partial dependence** — Average fitted output after replacing focal feature values across rows.

**Permutation importance** — Held-out score deterioration after shuffling feature information.

**Precision** — Event fraction among positive actions.

**Prevalence** — Event fraction in a specified population, sample, and window.

**Proper scoring rule** — A score whose expected optimum is truthful probability reporting.

**Prospective validation** — Evaluation on future cases after the procedure is frozen.

**Pseudo-residual** — Negative loss gradient used as a boosting-stage target.

**Random forest** — Bootstrap trees with random feature availability at splits.

**Recall** — Sensitivity or true-positive rate.

**Reliability diagram** — Binned mean probabilities compared with event rates.

**ROC AUC** — Ranking summary over true- and false-positive rates.

**Sensitivity** — Event fraction assigned positive action.

**SHAP** — Additive feature-attribution methods based on Shapley ideas.

**Sigmoid** — The function mapping a real score into a value between 0 and 1.

**Specificity** — Non-event fraction assigned negative action.

**Support vector** — A point on or within the SVM margin that shapes the solution.

---

# Research-paper reading ladder

| Paper | Main question to reconstruct | Beginner replication |
|---|---|---|
| Nelder & Wedderburn (1972) | How do distribution, linear predictor, link, and iterative fitting form a GLM? | Implement IRLS and create separation |
| Brier (1950) | Why evaluate probability forecasts directly? | Verify squared-loss propriety by simulation |
| Niculescu-Mizil & Caruana (2005) | Why can ranking and calibration differ by algorithm? | Calibrate four model families without leakage |
| Davis & Goadrich (2006) | How are ROC and PR spaces related? | Alter prevalence while keeping score distributions fixed |
| Breiman (2001) | How do forest strength and tree correlation affect error? | Measure correlation while changing `max_features` |
| Friedman (2001) | How is boosting function-space optimization? | Fit stumps to $y-p$ stage by stage |
| Grinsztajn et al. (2022) | Under which benchmark design do tree methods excel on tabular data? | Give four families equal search budgets |
| Lundberg & Lee (2017) | Which axioms define additive feature attribution? | Change background and dependence assumptions |
| Hardt, Price & Srebro (2016) | What do equal opportunity and equalized odds require? | Compare parity rules, utility, and changed decisions |

Read every paper in seven passes:

1. State its question in one falsifiable sentence.
2. Identify the mathematical object estimated or bounded.
3. Rewrite the algorithm as pseudocode.
4. List sampling, measurement, model, and optimization assumptions.
5. Reproduce one derivation, table, or simulation.
6. Construct a setting where the method should fail.
7. State what new evidence application to MHP projects would require.

---

# Suggested nine-day study rhythm

Use roughly three hours per day:

1. 15 minutes: retrieve yesterday's exit check.
2. 20 minutes: define the event or decision scenario.
3. 35 minutes: work a tiny numerical example.
4. 45 minutes: derive the central equation.
5. 55 minutes: type, run, and alter code.
6. 25 minutes: create one deliberate failure.
7. 20 minutes: reconstruct one paper claim.
8. 10 minutes: write one bounded conclusion.

Do not compress all days into a model-comparison weekend. The label, probability, action, explanation, and governance distinctions need repeated retrieval.

---

# Common beginner misconceptions

- Binary does not mean fully observed.
- Unknown is not automatically 0.
- Odds are not probabilities.
- Logistic regression is linear on log odds, not probability.
- An odds ratio is not a percentage-point risk change.
- A convergence warning is not harmless formatting.
- Accuracy is not probability quality.
- ROC AUC is not calibration.
- A 0.5 threshold is not neutral.
- $F_1$ is not a general cost function.
- Class weighting and threshold movement are not interchangeable.
- SMOTE does not create independent event evidence.
- KNN and RBF kernels require meaningful scaling.
- SVM margins are not probabilities.
- Trees avoid numeric scaling but not preprocessing decisions.
- OOB evidence is not automatically temporal evidence.
- Classification boosting does not always fit ordinary residuals.
- Early stopping uses validation information.
- More tuning creates more selection opportunity.
- Coefficients and importance values are not causal effects.
- PDP changes a feature across rows; it does not hold all other features at one common value.
- SHAP depends on background, dependence treatment, and output scale.
- Dropping group attributes does not prove fairness.
- One parity metric does not define justice.
- Temporal validation is not broad external validation.
- Prospective prediction does not prove intervention benefit.

---

# Ethical and governance questions

Discuss these with model developers, reviewers, funders, affected communities, and domain experts:

- Who defines a major overrun and can challenge that definition?
- Is review supportive, burdensome, or both?
- Should high predicted risk trigger more resources instead of restriction?
- Who supplies false-positive and false-negative costs?
- How is limited review capacity allocated?
- Who may correct a feature record or appeal an action?
- How are overrides recorded without punishing sound judgment?
- Which groups have incomplete labels or missing features?
- What disparities require investigation, mitigation, or stopping?
- Who can pause the system after an incident?
- When does a changed workflow require recalibration or full revalidation?
- How will an impact study determine whether review actually helps?

---

# Final readiness checklist

You are ready to leave Chapter 4 when you can:

- defend unit, eligibility, event, time boundaries, and unresolved-outcome policy;
- move among probability, odds, and log odds;
- derive Bernoulli likelihood, log loss, logistic gradient, and Hessian;
- interpret coefficients without confusing odds and risk;
- recognize separation and respond safely;
- distinguish discrimination, calibration, and utility;
- prove Brier and log loss are proper;
- design calibration without leakage;
- calculate every confusion-matrix rate;
- derive and challenge a cost threshold;
- explain ROC and PR behavior under prevalence change;
- distinguish weighting, resampling, calibration, and thresholding;
- describe spline, KNN, SVM, softmax, and multilabel assumptions;
- derive Gini gain and forest averaging behavior;
- derive logistic boosting pseudo-residuals;
- place tuning and early stopping inside honest validation;
- define the perturbation and scale behind every explanation;
- calculate a small Shapley example;
- compare fairness definitions and explain their conflicts;
- distinguish internal, temporal, geographic, prospective, and impact evidence;
- run the locked benchmark without revising it after test exposure; and
- write a conclusion limited to the data, period, workflow, and action actually studied.

The central lesson is simple even though the machinery is rich:

> A classifier is not only a model. It is an event definition, a probability procedure, an evaluation design, an action policy, and an accountable institution.
