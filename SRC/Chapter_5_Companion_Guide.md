# Chapter 5 Companion Guide: From Event Labels to Event Histories

## A beginner-first guide to survival analysis, censoring, competing risks, dynamic prediction, and registered evidence

This guide accompanies `chapter5.md`. It assumes you are still learning Python, mathematics, and statistics. It explains the vocabulary and symbols before using them, develops the main estimators from small risk sets, and repeatedly separates prediction from causal claims.

Chapter 4 asked whether an event occurs inside a fixed window. Chapter 5 preserves more information:

```text
when observation begins
    -> how long a subject remains event-free
    -> which event happens first
    -> why observation ends
    -> what can be predicted at a stated horizon
```

The central fact is:

> If a project has no observed warning when follow-up ends, we often know “not yet,” not “never.”

Survival analysis is also called time-to-event or event-history analysis. It applies to project warnings, machine failures, customer churn, loan defaults, hospital outcomes, and many other processes—not only death.

---

## How to use this guide

For each of Days 30–38:

1. Draw the timeline before reading the formula.
2. List who is genuinely at risk immediately before each event.
3. Work the smallest numerical example by hand.
4. Type the short code and alter one record.
5. Run the main chapter's full laboratory.
6. Deliberately create one listed failure.
7. State the estimand, assumptions, and allowed conclusion in ordinary language.

Keep these objects separate:

- **event process:** what can happen and in what order;
- **observation process:** when and why records become unavailable;
- **estimand:** the exact survival, hazard, incidence, or causal quantity wanted;
- **estimator:** the calculation used under assumptions;
- **prediction:** a future probability or score at a stated time and horizon;
- **decision:** an action based on consequences and capacity; and
- **claim:** what the design actually supports.

---

## 0. Foundations for a complete beginner

### 0.1 The running project question

One row begins as one fictional microhydro project approved at technical appraisal. The event of interest is the first formal warning that projected final cost exceeds 115% of the approved budget.

- Time zero is appraisal approval.
- A warning is event type 1.
- Cancellation before warning is event type 2.
- Database closure or loss to follow-up is censoring.
- Predictions concern 12, 24, or 36 months.
- Month-12 progress can only enter a prediction made at month 12 or later.

A high predicted incidence may trigger review, not automatic cancellation. The data are simulated and say nothing factual about real projects or districts.

### 0.2 Python language used in this chapter

```python
import numpy as np
import pandas as pd

time = np.array([2.0, 3.0, 5.0])
event = np.array([1, 0, 1])
```

- an array stores several values;
- `time[i]` and `event[i]` refer to the same subject;
- `event == 1` produces a Boolean mask;
- `time[event == 1]` keeps event times;
- `np.sum(mask)` counts `True` values;
- `np.unique(...)` finds distinct values;
- `np.sort(...)` orders them;
- `&` means elementwise “and”; and
- `|` means elementwise “or.”

Pandas provides labelled tables:

```python
history = pd.DataFrame({
    "project": ["A", "B", "C"],
    "time": [2.0, 3.0, 5.0],
    "event": [1, 0, 1],
})
```

Parentheses matter in masks:

```python
at_risk = (history["entry"] <= t) & (history["exit"] >= t)
```

### 0.3 Mathematical vocabulary

- $T$ is a latent event time: the time that would be observed with sufficient follow-up.
- $C$ is censoring time: when observation stops without the event.
- $Y=\min(T,C)$ is observed follow-up time.
- $\Delta$ is 1 if the event was observed and 0 if censored.
- $t$ is a particular time; $\tau$ is often a chosen prediction horizon.
- $P(A)$ is the probability of $A$.
- $P(A\mid B)$ is probability of $A$ among cases satisfying $B$.
- $f(t)$ is a probability density.
- $\mathbb E$ is an average over a distribution.
- $\sum$ means add; $\prod$ means multiply.
- $\int$ accumulates infinitely small contributions over a continuous range.
- $\exp(z)=e^z$ and $\log$ is its inverse.
- $x^\top\beta$ means multiply matching features and coefficients, then add.
- $u-$ means the instant just before time $u$.

### 0.4 A timeline contains more information than a binary label

Suppose A has a warning at month 18 and B is event-free when the database closes at month 18:

```text
A: approval ---------------- warning
B: approval ---------------- database closes
                               month 18
```

For A, $T_A=18$. For B, $T_B>18$. Calling B a permanent non-event changes an inequality into a claim about infinity. Calling month 18 its event invents an event. Dropping B discards 18 known event-free months.

### 0.5 Installing software

Core code uses:

```bash
python -m pip install numpy pandas scipy matplotlib scikit-learn
```

Optional applied laboratories can use:

```bash
python -m pip install lifelines scikit-survival
```

Record Python and package versions. Survival packages can differ in tie handling, variance estimates, censoring conventions, and prediction outputs. The same function name does not guarantee the same estimand.

### 0.6 Understanding `chapter5_data.py`

The generator creates three hidden times per project:

- time to overrun warning;
- time to cancellation; and
- time to censoring.

The first one determines the observable record:

```python
all_times = np.column_stack([overrun_time, cancel_time, censor_time])
first_type = np.argmin(all_times, axis=1)
observed_time = all_times[np.arange(n), first_type]
```

`column_stack` creates a three-column matrix. `argmin(..., axis=1)` returns the column holding the smallest time for each row.

Event type is coded:

```text
0 = censored
1 = overrun warning
2 = cancellation
```

Later cohorts receive less possible administrative follow-up because the database closes at one calendar date. A separate exponential time simulates loss to follow-up.

`legacy_registry_entry_months` supports delayed-entry exercises. The primary inception cohort starts everyone at approval, so entry is zero. `month12_progress_gap_sd` is future information for the appraisal model but legitimate among observable, event-free month-12 survivors. `senior_review_by_month6` is deliberately confounded and exists to demonstrate why an observational treatment coefficient is not a causal effect.

The hidden potential times are not returned as usable columns. Real construction would need dated warnings, adjudication of back-dated records, stable event definitions, cancellation reasons, and a verified close date.

### 0.7 Baseline, dynamic, and causal questions

- **Baseline prediction:** using appraisal information, what is cause-1 incidence by month 36?
- **Dynamic prediction:** among projects still observable and event-free at month 12, what is incidence during the next 24 months using history through month 12?
- **Causal question:** what would incidence have been under review versus no review?

They have different populations, time zeros, features, assumptions, and designs. A better month-12 model cannot be used at appraisal. A prognostic model cannot identify the effect of review merely by adding a review column.

---

# Day 30 — The event-history contract

## 30.1 Observed time and status

We observe:

$$
Y_i=\min(T_i,C_i),
\qquad
\Delta_i=\mathbb 1(T_i\le C_i).
$$

If $\Delta=1$, then $T=Y$. If $\Delta=0$, then $T>Y$. A censored row contributes genuine partial information.

With competing events, store event type beside time rather than one binary indicator. Cancellation is known outcome information, while database closure leaves the next event unknown.

## 30.2 The seven-part protocol

Freeze before association analysis:

1. **Unit:** one eligible project.
2. **Origin:** approval at appraisal.
3. **Entry:** when it begins contributing to risk sets.
4. **Event:** first formal warning under a stable rule.
5. **Competing events:** cancellation or other events that prevent warning.
6. **Censoring:** why observation ends without a recorded event.
7. **Horizons:** 12, 24, and 36 months.

Analysis time and calendar time differ. Two projects can both be at follow-up month 12 in different years; calendar change can still affect transport.

## 30.3 Four incomplete-observation mechanisms

**Right censoring:** event, if it happens, is after last observation. We know $T>C$.

**Left censoring:** event happened before a bound, but exact time is unknown. We know $T\le L$.

**Interval censoring:** event happened between inspections: $L<T\le R$.

**Left truncation/delayed entry:** a subject appears only after surviving event-free to entry time. Earlier failures are absent from the sample.

Left censoring concerns an unknown time for an observed subject. Left truncation changes which subjects are observed. Replacing interval-censored times by endpoints or midpoints is an approximation that needs sensitivity analysis.

## 30.4 Risk sets

Immediately before time $t$:

$$
R(t)=\{i:L_i\le t\le Y_i\}.
$$

A subject must have entered, remain observed, and remain event-free. Under a start–stop convention this may be written start $<t\le$ stop; state the convention.

For A `(0,3,event)`, B `(0,5,censor)`, C `(2,6,event)`, and D `(4,7,censor)`:

- at month 3: A, B, C are at risk;
- at month 6: C and D are at risk.

D has not entered at month 3. A failed and B was censored before month 6.

## 30.5 Independent censoring

A common assumption is:

$$
T\perp C\mid X.
$$

Conditional on recorded $X$, censoring provides no extra information about event time. This can fail when deteriorating projects disappear for unrecorded reasons.

Observed data cannot prove the absence of hidden informative censoring. Document reasons, compare censoring across periods and groups, include justified predictors in censoring models, and run sensitivity analyses.

Administrative censoring can differ by approval year without automatically being informative, provided calendar time and the mechanism are handled appropriately.

## 30.6 Censoring versus competing risk

Censoring means the future event is unknown after observation ends. A competing event means a known event has changed or prevented the future event of interest. Treating cancellation as censoring can be useful for estimating a cause-specific hazard, but it does not make cancellation missing information for absolute incidence.

## 30.7 Audit before estimation

Check:

- unique project IDs;
- origin, entry, exit, and event ordering;
- positive follow-up and entry no later than exit;
- impossible or duplicate event types;
- date precision and ties;
- censoring reasons by year and group;
- event counts and numbers at risk at every intended horizon;
- missing features and measurements recorded after prediction time; and
- whether event definitions changed over calendar time.

Do not begin with a survival curve. Begin with whether the timeline is coherent.

---

# Day 31 — Survival, hazard, and Kaplan–Meier

## 31.1 Distribution, density, survival, and hazard

For event time $T$:

$$
F(t)=P(T\le t)
$$

is cumulative event probability, and:

$$
S(t)=P(T>t)=1-F(t)
$$

is survival or event-free probability.

For continuous time, density $f(t)$ describes probability concentration near $t$. Hazard is:

$$
f(t)=\frac{dF(t)}{dt}=-\frac{dS(t)}{dt},
\qquad
h(t)=\frac{f(t)}{S(t)}.
$$

The first identity says density is the rate at which cumulative event probability rises, equivalently the rate at which survival falls. Dividing by $S(t)$ conditions the density on having survived to $t$.

$$
h(t)=\lim_{\Delta t\downarrow0}
\frac{P(t\le T<t+\Delta t\mid T\ge t)}{\Delta t}.
$$

Hazard is an instantaneous rate among current survivors. It can exceed 1 because it is per time unit, not a probability.

For a short interval, event probability is approximately $h(t)\Delta t$. Under constant hazard $h$, exact interval probability is $1-e^{-h\Delta t}$.

## 31.2 Cumulative hazard

$$
H(t)=\int_0^t h(u)\,du.
$$

For a single event process:

$$
S(t)=e^{-H(t)},
\qquad
H(t)=-\log S(t).
$$

Derivation intuition: multiply survival across tiny intervals, take logs to turn the product into a sum, then let interval width shrink.

If hazard is constant $\lambda$, then $H(t)=\lambda t$ and $S(t)=e^{-\lambda t}$. The median solves $S(t)=0.5$, giving $\log 2/\lambda$.

## 31.3 Kaplan–Meier product limit

At event time $t_j$, let $n_j$ be at risk just before and $d_j$ the number of events. Conditional survival through that time is:

$$
1-\frac{d_j}{n_j}.
$$

Multiply these conditional survivals:

$$
\widehat S(t)=\prod_{t_j\le t}\left(1-\frac{d_j}{n_j}\right).
$$

Events create downward steps. Censoring changes later risk-set denominators but does not create a survival step.

For records `(2,event)`, `(3,censor)`, `(5,event)`, `(5,event)`, `(7,censor)`:

- at 2: $n=5,d=1$, so survival is $4/5$;
- censoring at 3 leaves three at risk before 5;
- at 5: $n=3,d=2$, so survival becomes $(4/5)(1/3)=4/15$.

Tied event/censor timestamps need an explicit precision convention. File row order must not decide risk sets.

## 31.4 Greenwood uncertainty

$$
\widehat{Var}\{\widehat S(t)\}
\approx
\widehat S(t)^2
\sum_{t_j\le t}\frac{d_j}{n_j(n_j-d_j)}.
$$

Applied software often transforms the curve before forming intervals so limits remain between 0 and 1. Greenwood uncertainty does not include event-definition error, informative censoring, clustering, or selection.

Late intervals widen when few subjects remain. Always show numbers at risk; a smooth-looking long tail with three projects is weak evidence.

## 31.5 Nelson–Aalen cumulative hazard

$$
\widehat H(t)=\sum_{t_j\le t}\frac{d_j}{n_j}.
$$

$e^{-\widehat H(t)}$ approximates survival but is not exactly Kaplan–Meier because $-\log(1-d/n)$ is only approximately $d/n$ for small event fractions.

Nelson–Aalen is useful for hazard-scale diagnostics and resembles the baseline cumulative-hazard estimator used after Cox fitting.

## 31.6 Log-rank comparison

At pooled event time $j$, if group 1 has $n_{1j}$ of $n_j$ at-risk subjects and $d_j$ events, its expected count under equal hazards is:

$$
e_{1j}=d_j\frac{n_{1j}}{n_j}.
$$

The test accumulates observed minus expected events and scales by a hypergeometric variance. It is most sensitive to roughly proportional separation.

Crossing curves can differ at important horizons yet have a small log-rank statistic. A $p$-value is not an effect size, probability forecast, or causal result.

## 31.7 Cause-specific warning curve

If cancellation is treated as censoring, the resulting Kaplan–Meier curve describes a cause-specific/net-risk construction. Label it clearly. One minus that curve is not real-world warning incidence in the presence of cancellation; Day 35 supplies Aalen–Johansen.

---

# Day 32 — Cox regression from partial likelihood

## 32.1 The proportional-hazards model

$$
h(t\mid x)=h_0(t)e^{x^\top\beta}.
$$

$h_0(t)$ is an unspecified baseline hazard. The exponential covariate score multiplies it. For profiles $a,b$:

$$
\frac{h(t\mid x_a)}{h(t\mid x_b)}
=e^{(x_a-x_b)^\top\beta}.
$$

The baseline cancels, so the ratio is constant over time under proportional hazards. The hazards themselves may rise or fall.

## 32.2 Hazard-ratio interpretation

If $\beta=0.30$, then $e^{0.30}\approx1.35$. Among projects still event-free and observed, the fitted instantaneous rate is 35% higher for a one-unit difference, conditional on modeled covariates.

It is not a 35-point probability increase, risk ratio, time ratio, eventual-event ratio, individual guarantee, or causal effect. Accompany hazard ratios with absolute survival or cumulative incidence at relevant horizons.

## 32.3 Partial likelihood

If subject $i(j)$ is the sole event at $t_j$, conditional probability that this risk-set member is the one who fails is:

$$
\frac{e^{x_{i(j)}^\top\beta}}
{\sum_{k\in R(t_j)}e^{x_k^\top\beta}}.
$$

$h_0(t_j)$ cancels from numerator and denominator. Multiply across event times to obtain partial likelihood. It is “partial” because it estimates relative-hazard coefficients without simultaneously choosing a baseline-hazard formula.

## 32.4 Log partial likelihood, score, and information

Taking logs converts products to sums:

$$
\ell_p(\beta)=\sum_j\left[x_{i(j)}^\top\beta-
\log\sum_{k\in R(t_j)}e^{x_k^\top\beta}\right].
$$

The score is:

$$
U(\beta)=\sum_j[x_{i(j)}-\bar x(\beta,t_j)],
$$

where $\bar x$ is the risk-score-weighted feature mean in the risk set. The optimum balances observed event-subject covariates against expected risk-set covariates.

The negative Hessian is a sum of weighted risk-set covariance matrices. It drives optimization and model-based covariance estimates.

## 32.5 Tied event times

- **Exact:** averages possible event orders; expensive.
- **Efron:** progressively removes fractions of tied-event score; often accurate.
- **Breslow:** uses the same full denominator for a tied group; simple but less accurate with many ties.

Record the package default. The chapter's educational Cox implementation uses Breslow ties.

## 32.6 Reading the scratch Cox class

For each event time, `_objective`:

1. identifies deaths and the risk set;
2. adds event linear scores;
3. subtracts tied-event count times `logsumexp` of risk scores;
4. calculates weighted risk-set feature means; and
5. adds an optional $L_2$ penalty.

`logsumexp` avoids overflow when exponentiated scores are large. `L-BFGS-B` numerically minimizes negative partial log likelihood. The class then estimates Breslow baseline increments.

It is educational, not production-ready: no Efron ties, delayed entry, strata, robust covariance, sampling weights, or sparse matrices.

## 32.7 Absolute survival after fitting

The Breslow baseline cumulative hazard is:

$$
\widehat H_0(t)=\sum_{t_j\le t}
\frac{d_j}{\sum_{k\in R(t_j)}e^{x_k^\top\hat\beta}}.
$$

Then:

$$
\widehat S(t\mid x)=
\exp[-\widehat H_0(t)e^{x^\top\hat\beta}].
$$

A Cox linear predictor is a relative-risk score, not a probability. Absolute prediction needs the baseline event process and target population.

## 32.8 Penalization and event information

Ridge subtracts $\lambda\sum\beta_j^2/2$ from partial log likelihood. Lasso and elastic net use $L_1$ or mixed penalties. Scaling and penalty selection must occur inside resampling.

Information depends strongly on event counts and risk-set comparisons, not simply total rows. Thirty-five events cannot reliably support hundreds of flexible coefficients merely because thousands of censored rows exist.

Penalization cannot repair wrong time zero, dependent censoring, missing competing events, or leakage.

## 32.9 Clustering, frailty, fixed effects, and strata

- Cluster-robust covariance changes standard errors, not coefficients.
- Shared frailty adds latent group heterogeneity and changes the model.
- District fixed effects estimate represented-district conditional contrasts.
- Stratified Cox gives strata separate baseline hazards while constraining other hazard ratios to be common.

Choose based on sampling unit, dependence, estimand, and prediction destination—not the smallest $p$-value.

---

# Day 33 — Assumptions, diagnostics, and alternative time scales

## 33.1 What proportional hazards requires

For two profiles, the log-hazard difference must remain constant over analysis time. It does not require a constant baseline hazard.

Terrain might matter during early access works but less during commissioning. A single coefficient can average a fading or sign-changing effect and obscure the operational story.

## 33.2 Complementary log-log plots

Under PH:

$$
\log[-\log S(t\mid x)]
=\log[-\log S_0(t)]+x^\top\beta.
$$

Grouped curves of $\log[-\log\widehat S(t)]$ against $\log t$ should be roughly parallel. This is a rough visual diagnostic; sparse tails, arbitrary grouping, and competing events can mislead.

## 33.3 Schoenfeld residuals

At an event time:

$$
r_j=x_{i(j)}-\bar x(\hat\beta,t_j).
$$

This is observed event-subject feature minus fitted risk-set expectation. Under a stable effect, residuals should not systematically trend with event time. Scaled residuals and smooth curves help reveal changing coefficients.

A non-significant global test does not prove PH. Large samples flag trivial deviations; small samples miss important ones. Examine size, shape, uncertainty, and decision consequences.

## 33.4 Time-varying coefficients and stratification

One extension is:

$$
h(t\mid x)=h_0(t)\exp\{x\beta+xg(t)\gamma\}.
$$

With $g(t)=\log t$, $\log HR(t)=\beta+\gamma\log t$. A prespecified before/after milestone may communicate better. Searching many cut points and keeping the strongest exaggerates evidence.

If a variable is a nuisance with nonproportional effect, stratification can give it separate baseline hazards; then it receives no single coefficient.

## 33.5 Functional form is separate

PH concerns stability over time. Linearity concerns how a continuous predictor enters the log hazard. Diagnose curvature with justified transformations, restricted cubic splines, supported effect plots, and out-of-sample comparison.

Cutting a continuous feature at a data-chosen value loses information, creates a false jump, and adds selection multiplicity.

## 33.6 Other residuals and influence

Martingale residual:

$$
M_i=\Delta_i-\widehat H_i(Y_i).
$$

It compares observed event count with fitted cumulative hazard and helps diagnose functional form. Deviance residuals symmetrize it. Score residuals and DFBETA-like measures examine coefficient influence.

Residuals are investigation tools, not automatic deletion rules. A rare influential project can be precisely the population the model must serve.

## 33.7 Accelerated failure time models

$$
\log T=x^\top\beta+\sigma\varepsilon.
$$

$e^{\beta_j}$ is a modeled time ratio. A value 1.20 stretches the event-time distribution by 20%, conditional on covariates.

The error distribution determines baseline shape: extreme-value gives Weibull, normal gives log-normal, logistic gives log-logistic. AFT interpretation is often intuitive but imposes more distributional structure than Cox. Check tails and never extrapolate far beyond supported follow-up because AIC is lower.

## 33.8 Restricted mean survival time

$$
RMST(\tau)=\mathbb E[\min(T,\tau)]=\int_0^\tau S(t)\,dt.
$$

It is area under the survival curve through a chosen horizon and is expressed in event-free time, such as months. RMST differences do not require PH.

Choose $\tau$ before outcome inspection and within follow-up support for every compared group. The unrestricted mean is not identifiable when the tail remains censored; RMST is deliberately bounded.

## 33.9 Selection uncertainty and Bayesian extension

Standard intervals condition on the selected variables, splines, interactions, and time scale. Extensive diagnostic-driven selection makes them too narrow for the full procedure. Prespecify a primary model, label exploration, use shrinkage, and bootstrap the entire selection process for prediction performance when appropriate.

A Bayesian survival model combines event/censoring likelihood with priors on coefficients and baseline components. Posterior draws can propagate uncertainty into curves, RMST, and decisions; hierarchical priors can partially pool districts.

Results remain conditional on event definition, likelihood, censoring, priors, and computation. Perform prior and posterior predictive checks, convergence diagnostics, and prior sensitivity. “Bayesian” does not automatically mean causal, robust, or calibrated.

---

# Day 34 — Time-varying information and dynamic prediction

## 34.1 Baseline versus changing covariates

Baseline values are known at time zero. A time-varying value $X_i(t)$ changes during follow-up:

$$
h_i(t)=h_0(t)e^{\beta^\top X_i(t)}.
$$

At event time $t$, use only the value known immediately before $t$. Backfilling a future maximum delay into early risk sets leaks the future.

## 34.2 Start–stop data

If schedule status changes at month 10 and warning occurs at 17:

| Project | Start | Stop | Status | Event at stop |
|---|---:|---:|---:|---:|
| P1 | 0 | 10 | 0 | 0 |
| P1 | 10 | 17 | 1 | 1 |

Each row is an interval of constant information. Multiple rows are not independent subjects. Audit overlaps, gaps, duplicated events, update timestamps, and observation frequency. Split and resample by subject, never interval row.

## 34.3 Internal and external covariates

An external covariate evolves independently of subject state, such as a regional price index. An internal covariate, such as accumulated cost, is generated by the evolving project.

Internal values may predict strongly but can be shaped by earlier condition and intervention, creating time-dependent confounding and selection. Their Cox coefficients are associational without a causal design.

## 34.4 Landmark prediction

At landmark $s$:

- retain subjects observable and event-free at $s$;
- use history available by $s$;
- condition explicitly on survival to $s$; and
- predict in $(s,s+\tau]$.

The target is:

$$
P(T\le s+\tau,J=1\mid T>s,\mathcal H(s)).
$$

This population consists of survivors to $s$, not the baseline cohort. A project censored exactly at the landmark is not known observable afterward unless a timestamp convention justifies inclusion.

## 34.5 Immortal-time bias

Calling projects “ever reviewed by month 6” from appraisal guarantees that reviewed projects survived event-free and observable long enough to receive review. Their pre-review time is immortal with respect to that classification.

Defensible choices depend on the question:

- model review as a correctly aligned time-varying exposure;
- align both strategies at month 6 in a target-trial emulation;
- use a month-6 landmark among survivors; or
- use review only in predictions made at or after it is known.

Adding eventual review to a baseline Cox model does not fix the bias.

## 34.6 Recurrent events

First-event analysis discards later warnings or outages. Andersen–Gill represents repeated start–stop intervals and models event intensity, often with subject-clustered covariance.

Other models target event order, gap time, or latent frailty. Decide whether the scientific quantity is first event, total burden, time between events, or individual propensity.

## 34.7 Joint models

Landmarking summarizes history at selected times. Joint models connect a longitudinal measurement process with event time through shared latent structure, potentially handling noisy repeated values and informative observation/dropout.

They require assumptions for trajectory, association, event process, missingness, and random effects or priors. Use them when longitudinal and event processes are scientifically inseparable, not simply because they are sophisticated.

---

# Day 35 — Competing risks and multi-state processes

## 35.1 Event type and cumulative incidence

Let $T$ be time to the first event of any type and $J$ identify its cause. Cause-$k$ cumulative incidence is:

$$
F_k(t)=P(T\le t,J=k).
$$

$F_1(36)$ is real-world probability of a warning by month 36 before cancellation. All-cause survival $S(t)$ means no event type has occurred.

## 35.2 Cause-specific hazards

$$
\lambda_k(t)=\lim_{\Delta t\downarrow0}
\frac{P(t\le T<t+\Delta t,J=k\mid T\ge t)}{\Delta t}.
$$

Overall hazard is $\sum_k\lambda_k(t)$, and:

$$
S(t)=\exp\left[-\sum_k\int_0^t\lambda_k(u)du\right].
$$

Cause-$k$ incidence is:

$$
F_k(t)=\int_0^tS(u-)\lambda_k(u)du.
$$

First remain free of every event until just before $u$, then experience cause $k$.

## 35.3 Why one minus Kaplan–Meier overstates incidence

If cancellation is censored, $1-\widehat S_{KM}$ imagines cancelled projects remain capable of later warnings. This is a hypothetical net-risk quantity, not observed-world warning probability. The discrepancy grows as competing incidence grows.

## 35.4 Aalen–Johansen by hand

At event time $t_j$:

$$
\Delta\widehat F_k(t_j)=\widehat S(t_j-)\frac{d_{kj}}{n_j},
$$

$$
\widehat S(t_j)=\widehat S(t_j-)
\left(1-\frac{\sum_kd_{kj}}{n_j}\right).
$$

In the chapter's five-project example, warning increments are 0.20 at month 2 and 0.30 at month 5; cancellation incidence is 0.20 and remaining survival is 0.30. The three quantities sum to 1.

The scratch function loops over all event times, uses one shared all-event-free risk set, adds each cause's incidence increment using survival-before, and then updates survival. Censoring only changes later risk sets.

## 35.5 Cause-specific Cox and coherent predictions

Fit one Cox model per cause:

$$
\lambda_k(t\mid x)=\lambda_{0k}(t)e^{x^\top\beta_k}.
$$

Other causes leave the cause-$k$ risk set at their event time. Combine all fitted cause-specific hazards to produce coherent CIFs.

A covariate can raise cause-1 hazard yet lower cause-1 incidence by increasing a competing cause even more. Rate and probability are different targets.

## 35.6 Fine–Gray subdistribution hazard

Fine–Gray uses a special risk set that retains competing-event subjects under weighting conventions and models a subdistribution hazard linked to one CIF.

A subdistribution hazard ratio is not an instantaneous rate ratio among currently event-free subjects, not a probability ratio, and not the same as a cause-specific hazard ratio. Separate Fine–Gray models for every cause do not automatically create probabilities that sum coherently.

Choose the model from the estimand, not a familiar output table.

## 35.7 Multi-state models

A first-event model stops after warning or cancellation. A multi-state system can represent Active, Warning, Cancelled, and Completed, with transition-specific hazards and state-occupation probabilities.

A Markov model assumes future transitions depend on current state and covariates, not full history. A semi-Markov model can use time since state entry. State definitions, allowed transitions, and timestamps are measurement choices.

Use multi-state analysis when questions include completion after warning, time spent delayed, or probability of occupying a state—not just the first event.

---

# Day 36 — Nonlinear and hierarchical survival models

## 36.1 When flexibility is justified

Cox can already contain splines and interactions. Consider other learners for complex thresholds, local effects, many regulated predictors, or difficult time-varying effects when event information is sufficient.

Keep a strong regularized Cox reference. A tiny predictive gain may not justify worse calibration, auditability, computation, or maintenance.

## 36.2 Survival trees

A survival tree splits subjects using event-process criteria such as log-rank separation. Each terminal node estimates a survival curve or cumulative hazard.

It cannot regress observed follow-up time with ordinary squared error because censored time is not the event time. Control terminal **event counts**, not only rows. Specify tie handling, competing-risk target, depth, pruning, and leaf output.

Log-rank splitting favors proportional separation and may miss crossing hazards or differences at one operational horizon. Split criteria must match the prediction target.

## 36.3 Random survival forests

Typically:

1. bootstrap subjects;
2. grow deep survival trees;
3. offer random feature subsets at splits;
4. estimate terminal cumulative hazards or survival curves; and
5. average across trees.

Feature and bootstrap randomness reduce tree correlation. OOB prediction is internal evidence, not temporal, geographic, or external transport. Clustered data require cluster-aware sampling.

Tune tree count, feature subset, minimum terminal events, sampling, and split rule using censoring-aware metrics at prespecified horizons.

## 36.4 Survival boosting

For Cox boosting:

$$
\mathcal L(f)=-\sum_{i:\Delta_i=1}
\left[f(x_i)-\log\sum_{k\in R(T_i)}e^{f(x_k)}\right].
$$

Weak learners reduce a loss coupled through risk sets. Other methods optimize AFT likelihood, discrete hazards, or ranking. Name the objective.

Learning rate, tree complexity, stages, and early stopping interact. Early stopping consumes validation evidence and belongs inside development folds.

## 36.5 Discrete-time survival

Create one row per subject-interval while still at risk and estimate conditional interval probability:

$$
q_{it}=P(T\text{ in interval }t\mid T\text{ not earlier},X_i).
$$

Then:

$$
S_i(t)=\prod_{u\le t}(1-q_{iu}).
$$

Logistic, complementary log-log, boosted, or neural models can estimate these probabilities. Interval rows from one subject are dependent. Random row splitting leaks identities and later history. Multiple causes need coherent multinomial transitions.

## 36.6 Neural survival models

A neural model may output a Cox score, discrete hazard, distribution, survival curve, or cause-specific incidence. The brand name does not reveal the estimand.

Check censoring loss, monotonic curves, probability coherence, time representation, event count, tuning, and temporal/external validation. Deep learning is most plausible with large event counts and rich inputs; ordinary tabular cohorts often favor simpler regularized models.

## 36.7 Shared frailty and partial pooling

$$
h_{ij}(t)=h_0(t)e^{x_{ij}^\top\beta+u_j}.
$$

$u_j$ is a latent group effect. Sparse districts partially pool toward the population distribution. Frailty induces within-group dependence and risk-set selection over time.

Frailty variance does not identify why districts differ. For a new district, an existing-district random effect is unavailable; prediction may integrate over the frailty distribution or require local data.

## 36.8 Resampling destinations and explanation

Hold out according to the generalization target: future project, new district, future period, or both. “Generalizes” is incomplete without a destination.

Survival permutation importance needs a survival metric. PDP or SHAP must name whether output is $S(t)$, hazard score, or $F_k(t)$ and at which horizon. Importance may change with time. These are fitted-model explanations, not causal decompositions.

---

# Day 37 — Censoring-aware evaluation and decisions

## 37.1 Complete prediction contract

Every metric must name prediction time, eligible population, event type, horizon, competing-event handling, censoring adjustment, validation destination, and whether it assesses ranking, probability, or action.

“C-index 0.74” omits most of the scientific question.

## 37.2 Harrell concordance

If subject $i$ has the earlier observed time and experiences cause 1 then, its pair with a later subject is comparable. A score is concordant if it ranks $i$ higher.

$$
\widehat C=\frac{\text{concordant}+0.5\times\text{score ties}}
{\text{comparable pairs}}.
$$

Harrell C excludes ambiguous pairs and can depend on censoring. It is a global ranking measure, not horizon-specific probability accuracy. Uno C uses inverse censoring weights under additional assumptions.

## 37.3 Time-dependent AUC

At $\tau$, define cases and controls using a stated incident/dynamic or cumulative/dynamic convention, then adjust for censoring. Different conventions answer different questions.

Plot supported horizons. Selecting the time where AUC peaks is outcome-driven. A model can rank early events well and later events poorly.

## 37.4 IPCW Brier score

Without censoring, cause-1 Brier at $\tau$ is:

$$
\frac1n\sum_i[\mathbb 1(T_i\le\tau,J_i=1)-\widehat F_{1i}(\tau)]^2.
$$

Status is unknown for subjects censored before $\tau$. IPCW upweights subjects whose status is observed by the inverse probability of remaining uncensored.

Estimate $G(t)=P(C>t)$ with a reverse Kaplan–Meier or justified conditional censoring model, using training/reference data. A competing event before $\tau$ is known not to be cause 1 and remains observed; it is not censored.

Small $\widehat G$ produces unstable weights. Restrict horizons or prespecify weight truncation and sensitivity analyses. Integrated Brier averages over a stated time range whose weights and maximum time must be reported.

## 37.5 Reading the scratch IPCW function

- Subjects with any observed event by the horizon receive inverse $G$ just before their event.
- Cause-1 events have outcome 1; competing events have outcome 0.
- Subjects still at risk after the horizon receive inverse $G(\tau)$ and outcome 0.
- Subjects censored on or before the horizon receive zero weight because status is unknown.

This recovers information only under the censoring assumption. If censoring depends on features, model $G(t\mid X)$ appropriately and validate it.

## 37.6 Calibration at a horizon

For each prediction group:

1. calculate predictions without validation outcomes;
2. report mean predicted CIF;
3. estimate observed CIF with Aalen–Johansen;
4. show events, competing events, censoring, numbers at risk, and uncertainty.

Raw observed cause-1 proportion mishandles censored status. Tiny bins cannot establish calibration. Any flexible recalibration needs separate development evidence.

## 37.7 Inference, prediction, and conformal methods

A narrow coefficient interval does not imply accurate individual probabilities. Coefficient inference, ranking, calibration, forecast uncertainty, and decisions require different evidence.

Ordinary split conformal cannot use $|T-\hat T|$ for censored rows because $T$ is unknown. Dropping censored rows breaks exchangeability. Conformal survival methods add censoring assumptions, weighting, bounds, or alternative targets. State whether the output is a lower bound, interval, or probability band and the population receiving coverage.

## 37.8 Validation destinations

- Apparent performance uses training data and is optimistic.
- Internal validation estimates optimism within one source.
- Temporal validation uses later approvals.
- Geographic validation holds out places.
- External validation uses an independently collected source.
- Prospective validation freezes the system before live future cases.

Describe shifts in predictors, censoring, competing events, follow-up, measurement, and interventions. A random 20% holdout is not external validation.

## 37.9 Horizon-specific decisions

For calibrated incidence $p$ and simplified costs:

$$
p>\frac{C_{FP}}{C_{FP}+C_{FN}}.
$$

This assumes a currently available action, common costs, encoded intervention effectiveness, and an appropriate horizon. Capacity limits can instead select the top 10%, but calibration still informs expected yield.

A dynamic policy must specify at each landmark its population, remaining horizon, model, inputs, action, threshold/capacity, override, appeal, and next evaluation. Monitor repeated alerts, lead time, false alarms, burden, and feedback from actions.

## 37.10 Subgroup evaluation

Report group counts, event types, censoring, follow-up, CIF, calibration, discrimination when estimable, actions, and uncertainty. A software AUC based on four events is not stable evidence.

Differences may reflect sampling, measurement, baseline incidence, competing risks, access to intervention, or model failure. Tie fairness analysis to the concrete action and institution.

---

# Day 38 — The registered locked study and causal boundary

## 38.1 Registration

Before touching the lockbox, freeze the question, cohort, time zero, entry, causes, censoring, horizons, prediction timestamps, estimands, splits, preprocessing, candidates, tuning, metrics, censoring weights, competing-risk method, groups, missing data, uncertainty, decision policy, sensitivity analyses, stopping, updates, and allowed claim language.

Registration permits exploration; it labels it honestly and makes deviations visible.

## 38.2 The Chapter 5 protocol

The chapter uses projects through 2022 for development, forward validation years 2020–2022, and 2023–2025 as one locked temporal test. It selects by cause-1 IPCW Brier averaged equally over 12, 24, and 36 months and validation years.

Candidates include historical Aalen–Johansen CIF and raw/engineered cause-specific Cox systems with fixed ridge choices. Separate cause models are combined into coherent incidence. The study also reports concordance, calibration, access-mode and district audits, and paired locked-test uncertainty.

The feature-free historical CIF is essential: complexity must beat a credible event-history baseline.

## 38.3 Reading the benchmark

`DesignBuilder` learns medians, means, scales, and category levels only from its training fold. It optionally constructs prespecified nonlinear features.

`CauseSpecificCoxSystem` fits one Cox model for warning and one for cancellation. `cif_from_cause_specific_models` combines their hazard increments:

1. union all cause-specific event times;
2. calculate each subject's cause increments;
3. convert total integrated hazard to event probability $1-e^{-dH}$;
4. allocate it among causes in proportion to increments;
5. add incidence using prior event-free survival; and
6. update survival.

The result is nonnegative and obeys $F_1+F_2+S=1$ within numerical precision.

`HistoricalCIF` predicts the training cohort's Aalen–Johansen values for everyone. Forward folds train only on earlier years. The lockbox is opened after candidate selection. The development reverse-KM curve supplies censoring weights.

The script saves every fold score, candidate summary, individual locked prediction, calibration table, subgroup audits, and metadata. Assertions prohibit final cost, future progress, future review, outcome time, and event type as baseline features.

## 38.4 What it does not solve

Real applications may need conditional censoring weights, delayed entry, Efron ties, clustered resampling, interval-censoring models, better missingness handling, external validation, and different state definitions. Executable does not mean universally valid.

## 38.5 Paired bootstrap scope

The benchmark resamples locked projects and recomputes the Brier difference while fits and censoring curve remain fixed. It measures validation-sample variation conditional on those frozen objects.

It omits development variability, selection uncertainty, censoring-model uncertainty, clustering, adjudication error, and transport. End-to-end procedure uncertainty would repeat preprocessing, validation, selection, fitting, and evaluation at the correct sampling unit.

## 38.6 Sensitivity analysis

Prespecified analyses change one assumption at a time: horizons, Efron ties, conditional censoring, composite events, ambiguous-date exclusion, district bootstrap, spline terms, or a month-12 landmark.

Sensitivity shows dependence on choices. It must not become a menu from which the best-looking answer is selected.

## 38.7 Prediction is not treatment effect

Projects with latent difficulty are more likely to receive review and warnings. The back-door path:

```text
review <- latent difficulty -> warning
```

creates confounding by indication. Event-free survival needed to receive review can add immortal-time bias. Time-varying condition may influence review and be influenced by earlier review.

Adding every available variable to Cox does not guarantee control: mediators, colliders, unmeasured confounders, positivity, and timing matter.

## 38.8 Target-trial emulation

A hypothetical experiment specifies:

- eligibility: active warning-free projects at month 6;
- strategies: immediate review versus no review in a grace period;
- common time zero: month 6;
- follow-up: through month 36, cancellation, or censoring;
- outcome and competing-risk estimand;
- causal contrast: incidence difference under strategies; and
- analysis aligned with baseline/time-varying confounding.

Observational methods include standardization, inverse-probability weighting, g-formula, marginal structural models, and doubly robust estimation. They require consistency, positivity, conditional exchangeability, and correct nuisance estimation. Design and identification—not a regression name—support causal claims.

## 38.9 Research integrity and paper reading

Maintain a deviation log with date, reason, lockbox visibility, affected components, analysis status, and approval. Freeze tested code and environment; keep raw data read-only; script derived data; preserve a dictionary and timestamp audit.

A selected model losing to historical CIF is a valid result. Reopening search turns the old test into development evidence and requires a new future test.

For every paper map its problem, estimand, observed data, assumptions, estimator, uncertainty, evidence, failures, software additions, and smallest replication. Read core derivation and limitations, not only summaries.

---

# Capstone companion: exercises, reporting, and assessment

## C.1 How to approach the exercises

For every exercise, write four lines before code:

1. **Observed record:** what time, status, entry, cause, and features are available?
2. **Target:** survival, hazard, CIF, dynamic risk, performance, or causal contrast?
3. **Assumptions:** censoring, truncation, PH, distribution, exchangeability, or transport?
4. **Check:** which hand calculation, invariant, simulation, or independent implementation could reveal an error?

Foundation exercises build inequalities, delayed-entry risk sets, hazard-to-probability conversion, KM, Nelson–Aalen, Greenwood, and RMST. Modeling exercises cover partial likelihood, ties, non-PH, and AFT alternatives. Dynamic/competing exercises cover KM overstatement, coherent cause predictions, landmark populations, and immortal time. Research exercises stress censoring sensitivity, external validation, paper replication, and registration.

## C.2 Essential invariants

Automated checks should verify:

- entry never follows exit;
- interval start is less than stop;
- event type belongs to the allowed set;
- at most one first event occurs per project;
- risk-set counts match a hand example;
- survival is nonincreasing and between 0 and 1;
- each CIF is nondecreasing and between 0 and 1;
- $S+\sum_kF_k=1$ in the simple competing-risk system;
- predictions use no post-prediction features;
- train years precede validation/test years;
- preprocessing learns only from training rows; and
- locked predictions are saved before aggregate reporting.

## C.3 Research-ready report

1. Abstract: cohort, time zero, causes, horizons, design, locked result, limitation.
2. Event-history construction: sources, timestamps, adjudication, entry, censoring.
3. Estimands and prediction contract.
4. Candidate procedures, folds, tuning, censoring, ties, competing-risk method.
5. Every validation fold and event count.
6. Locked discrimination, Brier, calibration, and incidence.
7. Groups with counts, causes, censoring, follow-up, and uncertainty.
8. Diagnostics and prespecified sensitivity analyses.
9. Transport and supported follow-up limits.
10. Decision rule, workflow, overrides, monitoring, and contestability.
11. Reproducibility artifacts and deviations.
12. Bounded predictive conclusion and separate causal question.

A research-ready study can conclude that no candidate transports or that follow-up cannot support the horizon. Refusing an unsupported claim is success.

---

# Chapter 5 synthesis and practical laboratories

## S.1 One history, several targets

| Question | Target | Typical method |
|---|---|---|
| Time until any first event | $S(t)$ or $H(t)$ | Kaplan–Meier, Nelson–Aalen |
| Feature association with current cause-1 rate | $\lambda_1(t\mid x)$ | cause-specific Cox |
| Real-world cause-1 probability | $F_1(t\mid x)$ | Aalen–Johansen or combined cause hazards |
| Updated risk at month 12 | conditional landmark CIF | landmark model |
| Flexible prediction | horizon survival/CIF | spline Cox, forest, boosting |
| Effect of review | causal contrast under strategies | experiment or target-trial emulation |

## S.2 Derivations to complete by hand

1. Derive what is known under $Y=\min(T,C)$ and $\Delta$.
2. Construct risk sets with delayed entry.
3. Derive $S(t)=e^{-H(t)}$.
4. Derive KM as a product of conditional survivals.
5. Calculate Greenwood variance and Nelson–Aalen hazard.
6. Derive log-rank observed-minus-expected counts.
7. Derive one Cox partial-likelihood factor and score.
8. Recover Breslow baseline survival.
9. Derive complementary-log-log parallelism under PH.
10. Calculate RMST as a staircase area.
11. Construct start–stop intervals and a landmark estimand.
12. Derive Aalen–Johansen CIF increments.
13. Combine cause-specific hazard increments coherently.
14. Enumerate concordance-comparable pairs.
15. Derive IPCW Brier contributions for event, competitor, survivor, and censoring.
16. Derive a horizon-specific cost threshold.
17. Draw the confounding and immortal-time paths for review.

## S.3 Practical laboratories

### Laboratory A: incomplete records

Generate latent event and censoring times, reveal only observed time/status, move the database-close date, and state every censored inequality.

### Laboratory B: product-limit estimation

Calculate delayed-entry KM, Greenwood, Nelson–Aalen, median survival if reached, and RMST with ties.

### Laboratory C: Cox recovery

Simulate exponential and Weibull PH data, vary censoring and ties, compare Breslow/Efron, then generate a sign-changing coefficient.

### Laboratory D: diagnostics and alternative scale

Compare constant Cox, prespecified time interaction, spline form, AFT, and RMST under crossing hazards.

### Laboratory E: dynamic information

Create month-6, 12, and 18 landmark cohorts. Compare baseline and current-history prediction without pooling unlike survivor populations.

### Laboratory F: competing risks

Hold warning hazard fixed while increasing cancellation. Compare $1-$KM, Aalen–Johansen, cause-specific hazards, and Fine–Gray interpretation.

### Laboratory G: flexible survival

Give regularized Cox, spline Cox, survival forest, and boosting equal temporal folds and tuning budgets. Select using supported-horizon IPCW Brier.

### Laboratory H: censoring shift

Hold event process fixed while changing censoring. Compare Harrell C, Uno C, IPCW Brier, calibration, and weight stability.

### Laboratory I: external destination

Hold out a district. Report feature, missingness, cause, censoring, and follow-up shifts before model metrics.

### Laboratory J: causal boundary

Demonstrate confounding by indication and immortal time, then write a target-trial protocol whose time zero and strategies align.

---

# Formula sheet with plain-language readings

| Formula | Meaning |
|---|---|
| $Y=\min(T,C)$ | observed follow-up ends at event or censoring |
| $\Delta=\mathbb 1(T\le C)$ | whether the event was observed |
| $S(t)=P(T>t)$ | event-free probability beyond time $t$ |
| $F(t)=1-S(t)$ | cumulative event probability without competing causes |
| $h(t)$ | instantaneous rate among those still at risk |
| $H(t)=\int_0^th(u)du$ | cumulative hazard |
| $S(t)=e^{-H(t)}$ | link from cumulative rate to survival |
| $\prod(1-d_j/n_j)$ | Kaplan–Meier survival |
| $S^2\sum d/[n(n-d)]$ | Greenwood variance |
| $\sum d_j/n_j$ | Nelson–Aalen cumulative hazard |
| $d_jn_{1j}/n_j$ | expected group-1 events in log-rank calculation |
| $h_0(t)e^{x^\top\beta}$ | Cox proportional hazard |
| $e^{\beta_j}$ | conditional hazard ratio for one unit |
| $\sum[x_{event}-\bar x_R]$ | Cox score |
| $e^{-\widehat H_0(t)e^{x^\top\beta}}$ | Cox absolute survival |
| $\log T=x^\top\beta+\sigma\varepsilon$ | AFT model |
| $\int_0^\tau S(t)dt$ | RMST through $\tau$ |
| $F_k(t)=P(T\le t,J=k)$ | cause-$k$ cumulative incidence |
| $S(t-)d_{kj}/n_j$ | Aalen–Johansen CIF increment |
| $S_i(t)=\prod_{u\le t}(1-q_{iu})$ | discrete-time survival |
| $(concordant+0.5ties)/comparable$ | Harrell concordance |
| $1/G(t)$ | inverse censoring weight |
| $C_{FP}/(C_{FP}+C_{FN})$ | simplified decision threshold |

---

# Expanded glossary

**Aalen–Johansen estimator** — Product-integral estimator of transition probabilities and competing-risk CIFs.

**Accelerated failure time model** — Model in which covariates multiply the event-time scale through log time.

**Administrative censoring** — Planned follow-up end at database or study closure.

**At risk** — Entered, observable, and free of the defined absorbing event immediately before a time.

**Baseline hazard** — Time-varying reference hazard in a PH model.

**Calibration at a horizon** — Agreement between predicted and observed event probability by that time.

**Cause-specific hazard** — Instantaneous rate of one cause among subjects free of all causes.

**Censoring** — Incomplete event-time observation; right censoring tells us event time exceeds censor time.

**Concordance** — Ranking agreement across usable event-time pairs.

**Competing event** — An event that prevents or fundamentally changes the event of interest.

**Cumulative hazard** — Hazard accumulated over time.

**Cumulative incidence function** — Probability of one particular cause by time in the presence of competitors.

**Delayed entry** — Risk-set entry after time zero because of left truncation.

**Dynamic prediction** — Updated prediction using history available at a later prediction time.

**Fine–Gray model** — Regression model for a cause's subdistribution hazard linked to its CIF.

**Frailty** — Latent random effect representing unobserved heterogeneity or shared dependence.

**Greenwood formula** — Approximate KM variance estimator.

**Hazard** — Instantaneous event rate conditional on remaining at risk; not a probability.

**Hazard ratio** — Ratio of hazards; time-constant under PH.

**Independent censoring** — Conditional assumption that censoring adds no further event-time information.

**Interval censoring** — Event known only to lie within an interval.

**IPCW** — Inverse probability of censoring weighting.

**Kaplan–Meier estimator** — Product-limit survival estimator based on event-time risk sets.

**Landmark analysis** — Prediction among survivors at a chosen update time using information then available.

**Left censoring** — Event occurred before a known bound.

**Left truncation** — Subjects are sampled only after surviving to an entry time.

**Log-rank test** — Risk-set observed-versus-expected group comparison.

**Multi-state model** — Model for transitions among several states rather than only first event.

**Nelson–Aalen estimator** — Sum of events divided by risk-set size, estimating cumulative hazard.

**Partial likelihood** — Cox likelihood component estimating relative hazard without specifying baseline shape.

**Proportional hazards** — Covariate hazard ratios remain constant over analysis time.

**Recurrent event** — Event type that may occur multiple times for one subject.

**Restricted mean survival time** — Expected event-free time capped at a finite horizon.

**Risk set** — Subjects eligible, entered, observed, and event-free immediately before a time.

**Schoenfeld residual** — Event-subject covariate minus its risk-set-weighted expectation.

**Subdistribution hazard** — Hazard-like quantity constructed to model a competing-risk CIF.

**Time-varying covariate** — Feature changing during follow-up and aligned to when known.

**Target trial** — Explicit hypothetical randomized experiment an observational causal analysis seeks to emulate.

---

# Research-paper reading ladder

| Paper | Central idea | Beginner replication |
|---|---|---|
| Kaplan & Meier (1958) | nonparametric estimation from incomplete lifetimes | hand-calculate every product term |
| Cox (1972) | relative-hazard regression through risk-set conditioning | compare two baseline hazard shapes |
| Schoenfeld (1982) | partial-likelihood residuals for model checking | simulate stable and fading effects |
| Andersen & Gill (1982) | counting-process framework for histories and recurrent events | build valid start–stop risk sets |
| van Houwelingen (2007) | dynamic prediction by landmarking | compare several landmark populations |
| Aalen & Johansen (1978) | product-integral transition probabilities | verify CIFs plus survival sum to one |
| Fine & Gray (1999) | regression tied to a competing-risk CIF | separate cause-specific and subdistribution interpretations |
| Ishwaran et al. (2008) | survival-specific forest construction | compare equal-budget temporal validation |
| Graf et al. (1999) | prediction error with censoring | implement horizon IPCW Brier |
| Uno et al. (2011) | censoring limitation of common concordance | vary only censoring across simulations |
| Candès et al. (2023) | conformal prediction under survival censoring | state target and coverage assumptions |

For every methods paper, map problem, estimand, observed record, assumptions, estimator, uncertainty, evidence, failures, software gap, and a minimal falsifying replication.

---

# Suggested nine-day study rhythm

Use roughly three hours daily:

1. 15 minutes: retrieve the prior exit check.
2. 20 minutes: draw event and observation timelines.
3. 35 minutes: construct a tiny risk set.
4. 45 minutes: derive the central estimator.
5. 55 minutes: type and alter code.
6. 25 minutes: create a failure case.
7. 20 minutes: reconstruct a paper claim.
8. 10 minutes: write an estimand, assumption, and bounded conclusion.

Survival skill grows by repeatedly rebuilding risk sets. Do not reduce the chapter to memorizing package calls.

---

# Common beginner misconceptions

- Censored does not mean event-free forever.
- Left censoring is not left truncation.
- A competing event is not missing follow-up.
- Hazard is a rate, not a probability.
- Hazard ratio is not a risk ratio, probability difference, time ratio, or causal effect.
- Censoring changes later risk sets, not KM directly.
- One minus KM is not a cause-specific real-world incidence with competitors.
- A log-rank $p$-value is not effect size or causality.
- PH does not require constant hazard.
- PH and linear functional form are separate assumptions.
- A non-significant diagnostic does not prove an assumption.
- A Cox risk score is not an absolute probability.
- Rows do not equal information; event counts matter.
- Future progress cannot enter a baseline model.
- Start–stop rows from one subject are dependent.
- Ever-treated grouping can create immortal-time bias.
- Different landmarks contain different survivor populations.
- Fine–Gray and cause-specific hazard ratios differ.
- Ordinary regression on observed follow-up time mishandles censoring.
- OOB performance is not temporal transport.
- A C-index is not calibration or decision value.
- Subjects censored before a horizon do not have known negative status.
- A random holdout is not external validation.
- Split conformal is not automatically valid with unknown event times.
- A predictive treatment coefficient is not an intervention effect.
- Reopening a lockbox makes it development data.

---

# Ethical, operational, and governance questions

- Who defines and adjudicates a formal warning?
- Could warning practices differ by district or period?
- Why does follow-up end, and who becomes invisible?
- Does cancellation represent failure, protection, choice, or several processes?
- Is review supportive, burdensome, or both?
- What horizon allows meaningful preventive action?
- Who sets false-alert and missed-warning consequences?
- Can teams correct dates, features, states, and decisions?
- How are repeated alerts and override burden monitored?
- Which groups lack enough follow-up for a promised horizon?
- Who can pause the system when calibration or censoring changes?
- How do actions alter later warnings, cancellations, and recorded data?
- What new causal design is needed before claiming review benefit?

---

# Final readiness checklist

You are ready to leave Chapter 5 when you can:

- define unit, origin, entry, event types, censoring, and horizon;
- distinguish four incomplete-observation mechanisms;
- construct delayed-entry risk sets;
- state independent censoring as an unprovable conditional assumption;
- move among density, survival, hazard, and cumulative hazard;
- derive KM, Greenwood, Nelson–Aalen, and log-rank calculations;
- derive Cox partial likelihood, score, and baseline survival;
- interpret hazard ratios without probability or causal overreach;
- document tie handling and event-information limits;
- diagnose PH separately from functional form;
- interpret AFT time ratios and RMST;
- align time-varying information using start–stop or landmarks;
- explain recurrent events, joint models, and immortal time;
- derive Aalen–Johansen incidence and distinguish Fine–Gray;
- explain survival trees, forests, boosting, discrete time, and frailty;
- design resampling for the destination;
- calculate concordance and IPCW Brier contributions;
- build censoring-aware calibration at supported horizons;
- separate ranking, probability, decision, coefficient, and forecast uncertainty;
- execute the registered temporal benchmark once;
- preserve predictions, metadata, deviations, and unfavorable results; and
- distinguish prognostic association from a target-trial causal effect.

The entire chapter can be remembered as:

> clock → risk set → event process → probability → decision → bounded claim.
