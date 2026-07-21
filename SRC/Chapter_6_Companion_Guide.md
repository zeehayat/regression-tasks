# Chapter 6 Companion Guide: From Prediction to Intervention

## A beginner-first guide to causal inference, experiments, observational designs, reproducibility, and responsible deployment

This guide accompanies `chapter6.md`. It assumes you are still learning Python, mathematics, statistics, and causal language. The chapter is advanced because the questions are demanding, not because explanations should be obscure.

Earlier chapters asked what happened, what is associated, what is likely, and when an event may occur. Chapter 6 asks:

> What would change if we deliberately chose one action rather than another?

That question cannot be answered by adding a treatment column to a predictive model. A causal study needs a defined intervention, counterfactual contrast, assignment story, identification assumptions, estimator, uncertainty procedure, and population.

```text
policy question
    -> target trial and estimand
    -> causal structure and identification
    -> design and estimation
    -> diagnostics and sensitivity
    -> reproducible result
    -> monitored decision system
```

---

## How to use this guide

For Days 39–48:

1. Write the causal question without naming a model.
2. Specify population, time zero, treatment, comparator, outcome, horizon, and scale.
3. Draw the assignment mechanism or DAG.
4. State every identification assumption in plain language.
5. Derive the smallest estimator by hand.
6. Type and deliberately break the code.
7. Identify what no observed-data diagnostic can test.
8. Write one supported and one unsupported conclusion.

Keep these layers separate:

- **estimand:** the counterfactual quantity wanted;
- **identification:** assumptions connecting it to observable data;
- **estimator:** a finite-sample calculation;
- **inference:** uncertainty under the design and procedure;
- **transport:** the population or setting receiving the claim;
- **policy:** the action rule and consequences; and
- **production:** how the system is delivered, monitored, challenged, and stopped.

---

## 0. Foundations for a complete beginner

### 0.1 The running causal question

At month 6, eligible fictional microhydro projects may receive a structured senior-review package. The outcome is a formal major-overrun warning between just after month 6 and month 36.

The primary question is:

> Among projects active, observable, and warning-free at month 6, what would be the 36-month warning-risk difference if all received review at month 6 versus if none received it?

This fixes:

- unit: eligible project;
- time zero: month 6;
- treatment: defined review package at time zero;
- comparator: no package at time zero;
- outcome: warning through month 36;
- follow-up: 30 months;
- population: month-6 eligible projects; and
- scale: risk difference.

The data are simulated. Nothing about their districts, projects, or review effects is real-world evidence.

### 0.2 Essential notation

- $A=1$ means treatment; $A=0$ means comparator.
- $Y$ is the observed outcome.
- $Y(1)$ and $Y(0)$ are potential outcomes under the two interventions.
- $X$ contains pre-treatment covariates.
- $Z$ often denotes randomized assignment or an instrument.
- $M$ often denotes a mediator.
- $S$ can denote selection into the observed sample.
- $E(Y)$ is an average over a population distribution.
- $P(A=1\mid X)$ is treatment probability among profile $X$.
- $\perp$ means statistical independence.
- $\hat q$ is an estimate of unknown quantity $q$.
- $\bar A_k$ means treatment history through visit $k$.

Subscripts index subjects or visits; superscripts such as $Y^d$ name a potential outcome under strategy $d$, not exponentiation.

### 0.3 Python patterns

```python
treated = df[df["senior_review_month6"] == 1]
control = df[df["senior_review_month6"] == 0]
naive_difference = (
    treated["warning_by_month36"].mean()
    - control["warning_by_month36"].mean()
)
```

This computes an observed group difference. Code correctness does not make it causal.

Other recurring patterns:

- `.copy()` prevents accidental mutation of a shared table;
- `np.where(condition, a, b)` selects rowwise values;
- `np.clip(p, .02, .98)` limits estimated probabilities but changes the procedure;
- `.predict_proba(X)[:, 1]` selects class-1 probability;
- `KFold` creates cross-fitting partitions;
- `random_state` reproduces pseudo-random choices under the same environment.

### 0.4 Understanding the simulation generator

`make_mhp_causal_data` simulates baseline covariates, then assigns review preferentially to difficult projects:

```python
propensity = expit(linear_assignment_score)
senior_review = rng.binomial(1, propensity, n)
```

`expit(z)=1/(1+e^{-z})` converts a score into probability. Terrain, cable length, budget, experience, access, and district affect both assignment and warning risk, creating measured confounding.

The generator creates probabilities under no review and review, then potential outcomes `y0` and `y1`. Observed outcome is:

```python
observed_y = np.where(senior_review == 1, y1, y0)
```

Counterfactual columns are hidden unless explicitly revealed for simulation validation. Real datasets never reveal both potential outcomes or the true propensity. Simulation truth tests whether code can work under a known world; it does not validate assumptions in empirical data.

### 0.5 Installing and recording software

```bash
python -m pip install numpy pandas scipy matplotlib scikit-learn statsmodels
```

Optional causal libraries include DoWhy, EconML, DoubleML, CausalML, and relevant R packages. Interfaces change, so use current official documentation and record exact versions.

A package implements an estimator. It cannot define treatment versions, justify exchangeability, choose a valid instrument, detect every collider, or accept responsibility for policy.

### 0.6 Prediction, association, and intervention

$$
P(Y=1\mid A,X)
$$

describes observed outcomes conditional on recorded action and features.

$$
P\{Y(a)=1\}
$$

describes a hypothetical intervention setting action to $a$. The conditioning bar and intervention are not interchangeable. Riskier projects can receive more review, so reviewed projects may have worse observed outcomes even if review helps.

---

# Day 39 — Counterfactuals, estimands, and identification

## 39.1 Potential outcomes

For project $i$:

- $Y_i(1)$ is outcome under review;
- $Y_i(0)$ is outcome under no review.

Individual causal contrast is $Y_i(1)-Y_i(0)$, but only one is observed:

$$
Y_i=A_iY_i(1)+(1-A_i)Y_i(0).
$$

This fundamental problem cannot be solved by sample size: the same project in the same circumstance cannot simultaneously receive both actions. Designs identify average contrasts using comparisons and assumptions.

## 39.2 Choose the estimand first

Common targets:

$$
ATE=E\{Y(1)-Y(0)\},
$$

$$
ATT=E\{Y(1)-Y(0)\mid A=1\},
$$

$$
CATE(x)=E\{Y(1)-Y(0)\mid X=x\},
$$

and policy value $E[Y\{d(X)\}]$ under rule $d$.

ATE, ATT, untreated-population effects, overlap effects, and policy values average over different populations or assignments. “The treatment effect” is incomplete.

For binary risks $p_1,p_0$:

- risk difference: $p_1-p_0$;
- risk ratio: $p_1/p_0$;
- odds ratio: $[p_1/(1-p_1)]/[p_0/(1-p_0)]$.

Scales differ. Conditional and marginal odds ratios can differ without confounding because odds ratios are non-collapsible. Register the scale rather than switching to the most dramatic result.

For event time, name horizon, competing-event strategy, survival, incidence, or restricted mean.

## 39.3 Four identification assumptions

### Consistency

If $A=a$, observed outcome equals $Y(a)$. Treatment must be well defined. A five-minute signature and six-week engineering review are not automatically one intervention.

### Conditional exchangeability

$$
\{Y(1),Y(0)\}\perp A\mid X.
$$

Given adequate pre-treatment covariates, action behaves as if independent of potential outcomes. In observational data this no-unmeasured-confounding claim is not testable from outcomes alone.

### Positivity

$$
0<P(A=1\mid X=x)<1
$$

for target profiles. If all remote projects are reviewed, untreated remote outcomes cannot be learned without extrapolation. Restriction, new data, or a different estimand changes the question honestly.

### No interference

One project's treatment must not change another's outcome under simple notation. Shared engineers can create spillovers. Cluster or network estimands may be needed.

## 39.4 The g-formula identification step

Under consistency, exchangeability, and positivity:

$$
E\{Y(a)\}=E_X[E(Y\mid A=a,X)].
$$

Exchangeability inserts observed treatment group $A=a$ into the conditional potential-outcome mean; consistency replaces potential with observed outcome; positivity ensures relevant comparisons exist.

This derivation justifies standardization—not loyalty to one regression package.

## 39.5 Identification, estimation, and inference

- Identification asks whether an exactly known observed distribution determines the estimand under assumptions.
- Estimation approximates required quantities from finite data.
- Inference quantifies sampling and procedural uncertainty.

More data reduce some estimation error but cannot repair undefined interventions, colliders, positivity gaps, or hidden confounding. Confidence intervals are conditional on identification and model assumptions.

---

# Day 40 — Causal diagrams and adjustment

## 40.1 DAG basics

A directed acyclic graph has variables as nodes and assumed direct causal relations as arrows. “Acyclic” means following arrow directions cannot return to the starting node.

A DAG is not learned truth. It is a compact, criticizable causal argument. Include unmeasured variables when scientifically relevant; omission from the dataset does not erase them from the world.

## 40.2 Variable roles depend on the question

- Confounder: common cause $A\leftarrow C\rightarrow Y$.
- Mediator: causal pathway $A\rightarrow M\rightarrow Y$.
- Collider: common effect $A\rightarrow K\leftarrow Y$.
- Instrument: causes treatment but affects outcome only through treatment under strong assumptions.
- Proxy: measured stand-in related to an unmeasured construct.
- Selection variable: controls entry or observation and can induce bias.

A variable's role depends on time, intervention, outcome, and graph. “Always adjust for age” is not causal reasoning.

## 40.3 Confounders and backdoor paths

A backdoor path starts with an arrow into treatment. A valid adjustment set blocks all such paths without conditioning on forbidden descendants that open bias.

Confounders are chosen from substantive causal knowledge, not treatment/outcome $p$-values, predictive importance, or automated stepwise selection. Weakly associated confounders can matter; strong predictors need not confound.

## 40.4 Mediators and estimand changes

Adjusting for a mediator removes part of the pathway and no longer targets the total effect. Direct and indirect effects need precisely defined interventions and additional assumptions, especially with mediator–outcome confounding affected by treatment.

“Control for everything” silently changes the causal question.

## 40.5 Collider and selection bias

In $A\rightarrow K\leftarrow Y$, $A$ and $Y$ can be independent before conditioning. Restricting to or adjusting for $K$ opens association between its causes.

Examples include analyzing only approved, tested, hospitalized, completed, or fully documented cases when selection is affected by exposure and prognosis. Missingness indicators can also be colliders.

## 40.6 Descendants, instruments, and bias amplification

Treatment descendants are usually post-treatment and inappropriate for total-effect adjustment. Conditioning on an instrument-like strong treatment predictor can amplify residual bias from an unmeasured confounder without helping outcome prediction.

Covariate choice must follow the graph and estimand, not a generic baseline-variable checklist.

## 40.7 Selection and transport

A selection diagram marks mechanisms differing between study and target settings. Transport may require standardizing study effects over target covariates under assumptions that conditional causal relations are shared.

External predictive accuracy does not by itself prove causal transport. Treatment versions, adherence, effect modifiers, outcome measurement, and interference can change.

## 40.8 DAG workflow

1. Fix treatment, comparator, outcome, horizon, and population.
2. Order variables by time.
3. Add causes even when unmeasured.
4. Trace causal and backdoor paths.
5. Identify candidate adjustment sets.
6. Reject sets containing harmful post-treatment or collider variables.
7. Compare assumptions with actual timestamps and measurements.
8. Publish and challenge the graph before effect estimation.

Different plausible DAGs imply sensitivity analyses or that identification is not credible.

---

# Day 41 — Randomised experiments and noncompliance

## 41.1 What randomization does

Random assignment makes $Z$ independent of baseline potential outcomes in expectation through a known mechanism. It balances measured and unmeasured baseline causes on average, not necessarily exactly in one sample.

Analyze according to the randomization design. Allocation concealment prevents foreknowledge; blinding reduces behavior and measurement changes. Randomization does not solve attrition, nonadherence, interference, outcome mismeasurement, or transport.

## 41.2 Intention-to-treat

$$
\widehat\tau_{ITT}=\bar Y_{Z=1}-\bar Y_{Z=0}.
$$

ITT estimates effect of assignment/offer in the trial population. Preserve assignment groups regardless of receipt. Comparing recipients breaks randomization because adherence can reflect prognosis.

With binary outcomes report absolute risks and differences, not only odds ratios. Use randomization-based, robust, or prespecified model-assisted uncertainty appropriate to design.

## 41.3 Blocking and covariate adjustment

Blocking randomizes within prespecified strata and should be reflected in analysis. Baseline adjustment can improve precision when prespecified and correctly handled; post-randomization adjustment can bias the total assignment effect.

Chance baseline imbalance is not evidence randomization failed. Do not choose adjustment variables based on observed imbalance significance tests.

## 41.4 Cluster randomization and interference

If offices or districts are randomized, the independent assignment unit is the cluster. Individual-level standard errors falsely treat correlated subjects as independent.

Few clusters require careful randomization inference or small-sample corrections. Analyze recruitment timing because knowledge of cluster assignment can affect who enters.

Spillovers violate no-interference assumptions. Define direct, indirect, total, or saturation effects and randomize at a level consistent with delivery.

## 41.5 Noncompliance and LATE

Assignment $Z$ differs from treatment received $A$. The first stage is:

$$
E(A\mid Z=1)-E(A\mid Z=0).
$$

Under relevance, assignment independence, exclusion, and monotonicity, the Wald ratio:

$$
\frac{E(Y\mid Z=1)-E(Y\mid Z=0)}
{E(A\mid Z=1)-E(A\mid Z=0)}
$$

identifies a local average effect for compliers whose receipt changes with assignment. It is not automatically the ATE. Exclusion can fail if assignment itself motivates action; monotonicity excludes defiers.

Weak first stages create unstable ratios and unusual inference.

## 41.6 Per-protocol, attrition, and missing outcomes

Per-protocol effects concern adherence to a strategy and require adjustment for post-assignment predictors of adherence and outcome. Simply restricting to adherers is biased.

Attrition can destroy exchangeability if related to assignment and prognosis. Report reasons and rates by arm, preserve assignment, model or weight observation under explicit assumptions, and perform sensitivity analyses for missing outcomes.

Complete-case precision does not prove missingness is harmless.

## 41.7 Minimal randomized laboratory

Simulate assignment independently, generate noncompliance and outcomes, then calculate assignment risk difference, receipt difference, first stage, and Wald ratio. Repeat many trials to distinguish one noisy realization from estimator behavior.

---

# Day 42 — Standardisation, matching, and weighting

## 42.1 Three views of the same identified target

Under the same consistency, exchangeability, and positivity assumptions:

### Standardization

$$
\psi=E_X\{m_1(X)-m_0(X)\},
\quad m_a(X)=E(Y\mid A=a,X).
$$

Fit outcome means, predict every row under both actions, average, subtract.

### Inverse probability weighting

$$
\psi=E\left[\frac{AY}{e(X)}-\frac{(1-A)Y}{1-e(X)}\right].
$$

Weight each observed outcome by inverse probability of receiving its action.

### Matching or balancing

Construct treated and untreated comparisons with similar confounder distributions, then average the contrast over the intended population.

Agreement cannot test unmeasured exchangeability because all methods can share the same hidden bias.

## 42.2 Regression standardization

1. Fit $m(A,X)$.
2. copy every row and set $A=1$; predict $\hat m_1(X_i)$.
3. Set $A=0$; predict $\hat m_0(X_i)$.
4. Average each intervention world.
5. Subtract.

Even with logistic regression this produces a marginal risk difference. The treatment coefficient alone is a conditional log-odds ratio, not the ATE.

## 42.3 Propensity scores

$$
e(X)=P(A=1\mid X).
$$

The true propensity is a balancing score under the causal assumptions. Its job is design and covariate balance, not maximum treatment AUC. Perfect separation reveals absent comparisons, not a wonderful model.

Use propensity scores for matching, strata, IPW, overlap weighting, or balancing estimation. Select specifications using causal knowledge, overlap, and balance—not the preferred effect estimate.

## 42.4 Matching changes the target population

Document distance, caliper, ratio, replacement, exact matches, order, discards, balance, overlap, and outcome analysis. Discarding unmatched units often changes ATE to an effect in matched/overlap-support units.

Design matching without repeatedly inspecting outcomes. Account for reused controls, pairs, and matching weights in inference.

## 42.5 Weighting and pseudo-populations

ATE weight:

$$
w_i=\frac{A_i}{e(X_i)}+\frac{1-A_i}{1-e(X_i)}.
$$

ATT keeps treated weight 1 and controls weight $e/(1-e)$. Stabilized weights can reduce variance. Overlap weights emphasize treatment equipoise and target the overlap population.

Extreme weights reveal weak support or misspecification. Truncation trades variance for bias and defines a registered procedure. Do not tune truncation after seeing effects.

## 42.6 Balance and effective sample size

Standardized mean difference:

$$
SMD_j=\frac{\bar X_{1j}-\bar X_{0j}}
{\sqrt{(s_{1j}^2+s_{0j}^2)/2}}.
$$

Inspect weighted continuous distributions, categories, nonlinear terms, and interactions required by the graph. A threshold like 0.1 is a convention, not proof.

$$
ESS=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}.
$$

Concentrated weights can reduce thousands of rows to far less effective information. Plot treatment-specific propensities and inspect influential profiles.

## 42.7 Reading the adjustment code

The outcome pipeline includes treatment and baseline covariates, then predicts copied rows under both treatments. The propensity pipeline predicts treatment from baseline covariates. Preprocessing and model fitting must remain inside the relevant design/resampling boundary.

Clipping propensities is not innocuous. Report raw range, clipping rule, clipped count, balance, weights, ESS, and sensitivity. Flexible nuisance models still require causal covariate timing.

---

# Day 43 — Doubly robust estimation and sensitivity

## 43.1 AIPW score

$$
\hat\phi_i=\hat m_1(X_i)-\hat m_0(X_i)
+\frac{A_i\{Y_i-\hat m_1(X_i)\}}{\hat e(X_i)}
-\frac{(1-A_i)\{Y_i-\hat m_0(X_i)\}}{1-\hat e(X_i)}.
$$

Average scores to estimate ATE. The first term standardizes predicted outcomes. Weighted residual terms correct prediction error using observed outcomes.

## 43.2 Double robustness—precise meaning

Under identification and regularity conditions, AIPW can be consistent if either the outcome models or propensity model are correctly specified.

It does not mean two wrong models cancel, hidden confounding vanishes, positivity is optional, measurement no longer matters, or standard errors are automatic. Robustness concerns nuisance functions inside an already identified problem.

## 43.3 Influence-function uncertainty

$$
\hat\psi=\frac1n\sum_i\hat\phi_i,
\qquad
\widehat{SE}(\hat\psi)=\frac{SD(\hat\phi_i)}{\sqrt n}.
$$

Large scores reveal influential residuals and small propensities. Clustered data require cluster-level influence aggregation or appropriate bootstrap.

## 43.4 Cross-fitting

Split observations into folds. Train nuisance models outside each fold and predict $e,m_1,m_0$ inside the held-out fold. After all rows have out-of-fold predictions, compute the effect score once.

Cross-fitting reduces own-observation overfit and enables flexible nuisance estimation under rate conditions. It does not tune itself, prevent preprocessing leakage, or repair causal assumptions. Any tuning stays within nuisance training data.

## 43.5 TMLE and double machine learning

TMLE updates initial outcome predictions along a targeted fluctuation submodel and can preserve outcome bounds. Double/debiased machine learning uses orthogonal scores and sample splitting so small nuisance errors have reduced first-order influence.

These are frameworks, not magic brands. Report estimand, score, learners, folds, tuning, truncation, variance, and failures.

## 43.6 Whole-procedure bootstrap

Resample the independent unit, redo folds, transformations, tuning, nuisance fitting, score calculation, and effect estimation. Bootstrapping only saved predictions omits fitting uncertainty.

Use cluster or block bootstrap where independence is higher-level. Compare full-procedure and influence intervals; disagreement is diagnostic, not permission to choose the favorable interval.

## 43.7 Sensitivity to hidden confounding

Observed balance cannot prove exchangeability. Methods include bias-parameter grids, tipping points, E-values for suitable ratio measures, Rosenbaum bounds for matched designs, negative controls, quantitative bias analysis, and partial-identification bounds.

Each has assumptions. Calibrate sensitivity parameters against measured covariates and domain evidence. An E-value is not a certificate; a negative control only detects biases it shares.

## 43.8 Specification and overlap sensitivity

Change one registered choice at a time: nonlinear terms, justified covariate sets, weight truncation, overlap target, outcome definition, missing-data scenario, or negative control.

Agreement among models sharing the same flawed identification strategy is not causal robustness.

---

# Day 44 — Target trials and time-varying treatment

## 44.1 Write the imaginary experiment

A target trial specifies eligibility, strategies and versions, assignment, time zero, outcome/horizon, causal contrast, and analysis. Observational emulation follows that protocol as closely as data allow.

Eligibility assessment, treatment assignment, and follow-up must align at time zero. Selecting month-12 survivors while calling review at month 6 baseline creates a mismatched population.

## 44.2 Immortal-time and time-zero bias

If “treated” means receiving review any time in the first year but follow-up starts at approval, treated projects must survive warning-free until review. Assigning this guaranteed time to treatment can make it appear protective.

Solutions include aligned time zero, time-varying treatment, repeated trials, clone–censor–weight, or properly analyzed grace periods. Adding time-to-review as a baseline covariate does not repair design.

## 44.3 Longitudinal histories

At visit $k$, $L_k$ is covariate history observed before decision $A_k$. Bars denote history:

$$
\bar L_k=(L_0,\ldots,L_k),
\quad
\bar A_k=(A_0,\ldots,A_k).
$$

A dynamic strategy $d$ maps history to action. $Y^d$ is outcome under sustained strategy. Define realistic grace periods and adherence.

## 44.4 Treatment–confounder feedback

Earlier treatment affects later health; later health affects both future treatment and outcome. Conventional adjustment for later health blocks part of early treatment effect, while omitting it confounds later treatment.

This motivates longitudinal g-methods, not ordinary baseline regression.

## 44.5 Sequential assumptions

$$
Y^{\bar a}\perp A_k\mid \bar L_k,\bar A_{k-1}
$$

is sequential exchangeability. Strategies also require sequential positivity, longitudinal consistency, censoring assumptions, and interference treatment. Richer history makes support harder.

## 44.6 Parametric g-formula

Fit models for next covariates and outcomes given past history; sample baseline units; impose strategy; simulate covariates forward; impose later actions; simulate outcomes; average; repeat for each strategy.

It standardizes over complete histories and depends on correct longitudinal process models. Make Monte Carlo simulation error negligible relative to sampling uncertainty.

## 44.7 Marginal structural models

Stabilized longitudinal treatment weight:

$$
SW_i=\prod_k
\frac{P(A_{ik}\mid\bar A_{i,k-1},L_{i0})}
{P(A_{ik}\mid\bar A_{i,k-1},\bar L_{ik})}.
$$

Additional censoring weights handle measured informative loss. Fit a marginal outcome model in the pseudo-population.

Inspect probabilities and weights by visit, late-history support, balance, ESS, truncation sensitivity, and treatment/censoring calibration. Multiplication can make long-horizon weights unstable.

## 44.8 Clone–censor–weight and repeated trials

Clone a subject into every strategy compatible at baseline, artificially censor a clone on deviation, and weight continued compatibility. Clones share origin, so inference must account for dependence. Artificial censoring requires its own ignorability model.

Repeated target trials enroll eligible projects at successive months, align assignment and follow-up each time, then pool with trial/calendar controls and project-clustered inference. Treatment versions and policy periods may change.

## 44.9 Censoring and competing outcomes

Cancellation may remain a competing event for total real-world incidence, join a composite policy-failure outcome, be eliminated in a hypothetical estimand under stronger assumptions, or enter separable-effects analysis.

Calling it censoring changes the question and can create informative loss. Carry Chapter 5's event-process discipline into causal work.

---

# Day 45 — Quasi-experimental designs

## 45.1 Difference-in-differences

$$
\hat\tau_{DiD}=(\bar Y_{T,post}-\bar Y_{T,pre})
-(\bar Y_{C,post}-\bar Y_{C,pre}).
$$

Identification requires treated group's untreated counterfactual trend to match comparison trend on the chosen scale. Pre-trends can reveal problems but cannot prove post-policy parallel trends.

Threats include anticipation, changing composition, concurrent shocks, measurement change, spillovers, functional form, and serial correlation. Cluster inference at policy-assignment level.

## 45.2 Event studies and staggered adoption

Event studies show effects relative to adoption and pre-period patterns. With staggered timing and heterogeneous effects, ordinary two-way fixed effects can combine inappropriate comparisons and negative weights.

Use group-time estimands with never-treated or not-yet-treated comparisons, transparent aggregation, cohort/event-time support, and modern methods such as Callaway–Sant'Anna when assumptions fit.

## 45.3 Regression discontinuity

For cutoff $c$:

$$
\tau_{RD}=\lim_{r\downarrow c}E(Y\mid R=r)
-\lim_{r\uparrow c}E(Y\mid R=r).
$$

This is local to the cutoff. It requires continuity of potential outcomes absent treatment and no precise manipulation around the threshold.

Plot raw data and treatment probability, inspect running-variable density and baseline continuity, use low-order local polynomials, justified bandwidths, robust bias correction, and bandwidth/kernel sensitivity. Avoid high-order global polynomials.

Fuzzy RD divides outcome discontinuity by treatment discontinuity and identifies a local complier effect under IV-like assumptions.

## 45.4 Instrumental variables

Wald ratio:

$$
\frac{E(Y\mid Z=1)-E(Y\mid Z=0)}
{E(A\mid Z=1)-E(A\mid Z=0)}.
$$

Requires relevance, independence, exclusion, and usually monotonicity. Under heterogeneity it targets compliers, not population ATE. Weak instruments yield instability; a strong first stage does not defend exclusion.

Distance to a review office may directly affect outcomes, violating exclusion. Attack the DAG before fitting two-stage regression.

## 45.5 Synthetic control and interrupted time series

Synthetic control weights unaffected donors to reproduce a treated unit's pre-policy trajectory. It needs stable long pre-period, unaffected donors, good fit, and placebos.

Interrupted time series estimates level/slope changes and needs enough observations, seasonality/autocorrelation handling, stable measurement, and absence of concurrent explanations. A comparison series strengthens it.

## 45.6 Placebos and falsification

Use fake dates/cutoffs, outcomes treatment cannot affect, pre-treatment effects, covariate continuity, instrument associations with baseline outcomes, and leave-one-donor-out checks.

Passing tests cannot prove identification; failing a well-chosen falsification requires investigation.

---

# Day 46 — Heterogeneous effects and policy learning

## 46.1 CATE is still an average

$$
\tau(x)=E\{Y(1)-Y(0)\mid X=x\}.
$$

It is an average for profile/subgroup, not an observed individual effect. Even randomized trials do not reveal both outcomes for one person.

## 46.2 Confirmatory and exploratory heterogeneity

Prespecify a small set of plausible modifiers, coding, functional form, scale, interaction contrast, multiplicity control, sample/overlap requirements, and missingness handling.

“Significant here, nonsignificant there” does not prove interaction; test the difference. Exploratory discoveries need independent confirmation and must be labeled hypothesis-generating.

## 46.3 Effect scale matters

Constant risk ratio can yield larger absolute benefit in high-risk groups. Constant risk difference can yield varying ratios. Report baseline risks and register the policy-relevant scale.

## 46.4 Subgroup AIPW scores

Cross-fitted AIPW scores can be averaged within prespecified groups to estimate subgroup effects, with group-appropriate uncertainty. Nuisance models can be global while effect aggregation is subgroup-specific.

Small groups, poor treatment overlap, multiple testing, and learned subgroup boundaries undermine apparent precision.

## 46.5 Meta-learners

- S-learner fits one outcome model including treatment.
- T-learner fits separate treated/control models.
- X-learner imputes pseudo-effects and can help unequal group sizes.
- R-/DR-learners residualize or use orthogonal pseudo-outcomes.

Their assumptions and tuning matter. Cross-fitting reduces overfit but does not create exchangeability.

## 46.6 Honest causal trees and forests

Honesty separates data choosing tree splits from data estimating leaf effects. Causal forests average honest trees and can estimate conditional effects with uncertainty under conditions.

Tune for treatment-effect estimation rather than outcome prediction. Require overlap and sufficient treated/control observations in leaves. Importance does not prove true effect modification.

## 46.7 Evaluating heterogeneity without labels

Individual treatment effects are unobserved, so ordinary prediction RMSE is unavailable. Use simulation, randomized validation, held-out policy value, calibration of grouped scores, best-linear-predictor tests, and stability across samples.

Ranking apparent effects can overfit noise even if average ATE is correct.

## 46.8 From CATE to policy

A policy treats when expected benefit exceeds cost subject to capacity and constraints. Estimate policy value on data not used to learn the rule, using randomized comparison or doubly robust off-policy evaluation.

Compare with simple treat-all, treat-none, and risk-based rules. Account for treatment cost, harm, capacity, uncertainty, and implementation. A high-risk subject need not benefit most.

## 46.9 Fairness and transport

Audit who is eligible, has overlap, receives action, benefits, bears burden, and is excluded. Maximizing average value can concentrate resources or harms. Protected group rules may face legal and ethical constraints.

Transporting CATE requires stable conditional effects or measured effect modifiers plus target-population data. New settings can differ in treatment version, adherence, resources, baseline risk, and interference.

---

# Day 47 — Comparison, reproducibility, and responsible production

## 47.1 Define what is compared

Distinguish algorithms, tuned procedures, estimands, decision policies, and complete systems. Equalize data, folds, preprocessing, search budgets, censoring/uncertainty rules, and action definitions where the scientific comparison requires it.

## 47.2 Paired evaluation

Compare methods on the same observations or resamples and analyze paired differences. Pairing removes shared case difficulty from variance.

Repeated CV folds overlap heavily. Fold scores are not independent replicates, so ordinary paired $t$-tests on folds understate uncertainty. Use nested/locked evaluation, corrected resampled tests with limitations, bootstrap at the independent unit, or dataset-level comparison across genuinely independent tasks.

## 47.3 Practical equivalence and multiplicity

Predefine the smallest practically important difference and consider equivalence/noninferiority, not only “different from zero.” Multiple models, metrics, horizons, and groups create selection opportunities; control or label multiplicity and preserve all results.

## 47.4 Ablation studies

Removing a feature or component asks how this fitted procedure changes without it. It does not establish the component's causal importance. Rerun preprocessing and tuning fairly for each ablation, use paired evidence, and disclose the full ablation set.

## 47.5 Reproducibility levels

- Computational reproducibility: same data/code/environment regenerate outputs.
- Replicability: independent study obtains compatible evidence.
- Robustness: conclusions survive justified analysis choices.
- Generalizability/transport: results hold in new populations/settings.

One does not imply the others.

## 47.6 Research bundle

Include registration, README, scripted cohort construction, data dictionary, environment lock, seeds/determinism notes, tests, analysis code, all results, individual predictions/scores where allowed, figures, model objects, manifests/hashes, data sheet, model card, causal-study card, deviation log, and plain-language summary.

No hidden notebook state or manual spreadsheet step should be required.

## 47.7 Seeds, hashes, and manifests

A seed controls specified pseudo-randomness but cannot guarantee identical results across libraries, hardware, parallelism, or versions. Record all seed locations and deterministic limitations.

A cryptographic hash fingerprints an exact file; it does not prove correctness. A manifest lists paths, sizes, hashes, sources, timestamps, licenses, and roles. Hash raw and derived artifacts while respecting privacy.

## 47.8 Data sheets, model cards, and causal-study cards

Data sheet: motivation, creators, units, collection, sampling, timestamps, missingness, labels, groups, consent/governance, limitations, update history.

Model card: intended use, prohibited use, population, inputs/outputs, training, metrics, groups, thresholds, calibration, uncertainty, failures, monitoring, version.

Causal-study card: target trial, estimand, DAG, identification assumptions, assignment mechanism, overlap, adjustment, diagnostics, sensitivity, transport, deviations, allowed wording.

## 47.9 Training–serving consistency

Verify production feature definitions, units, category handling, missingness, time availability, model version, threshold, and action exactly match validation. Use schema tests, fixture predictions, shadow runs, and parity checks.

## 47.10 Monitor the decision system

Monitor service health, input/measurement drift, missingness, predicted values, treatment/action rates, capacity, overrides, subgroup burden, delayed outcomes, calibration/impact when mature, incidents, appeals, and downstream consequences.

Actions create feedback: intervention changes outcomes and future labels; selection changes who is observed; staff adapt to scores. Monitoring a score distribution alone misses system drift.

## 47.11 Retraining, rollback, and retirement

Predefine triggers, owners, validation gates, prospective/shadow periods, versioning, rollback artifacts, communication, and retirement criteria. Retraining is a new intervention and can invalidate causal interpretation. Preserve prior versions and decision logs.

---

# Day 48 — The registered final study

## 48.1 Registration and lockbox

Freeze causal question, target trial, estimand, DAG, adjustment set, data rules, exclusions, missingness, primary estimator, nuisance learners/tuning, diagnostics, sensitivity, uncertainty, groups, claim rule, and code version before primary outcomes are opened.

Lockbox access should be logged and technically separated. After opening, changes are deviations or exploratory. Reusing the test to revise the procedure consumes it.

## 48.2 Executable primary analysis

A complete script regenerates cohort, validates timestamps, constructs treatment/outcome, fits the registered method, computes diagnostics and effect, produces uncertainty, saves artifacts, and exits on failed integrity assertions.

Run synthetic fixtures with known truth and known failure cases before real outcomes. Assertions should check eligibility, treatment timing, outcome window, feature timing, overlap, finite scores, reproducible counts, and output schema.

## 48.3 Integrity before interpretation

Verify cohort counts and flow, duplicates, missingness, treatment provenance, balance, overlap, weights/ESS, nuisance predictions, influential scores, seed/environment, hash manifest, and lockbox log before interpreting the estimate.

If a primary integrity test fails, stop. Do not rationalize because the estimate looks plausible.

## 48.4 Writing the result

Report population, intervention/comparator, time zero, horizon, estimand/scale, design, identification assumptions, estimator, diagnostics, primary estimate and interval, sensitivity, deviations, transport limits, and allowed decision implication.

A narrow interval does not incorporate unmeasured confounding or undefined treatment versions. Null, harmful, unstable, or unidentified results must be reported.

## 48.5 Three graduation capstones

### Capstone A: target-trial emulation

Emulate senior review using aligned eligibility and treatment, causal adjustment, overlap/balance, sensitivity, and bounded claim.

### Capstone B: threshold or rollout evaluation

Use RD, DiD, IV, or another defensible assignment mechanism; define its local/group-time/complier estimand and falsification evidence.

### Capstone C: prospective model-impact trial

Randomize or prospectively compare use of a decision system, recording actions, adherence, spillovers, harms, capacity, subgroup outcomes, and workflow impact.

Prediction performance and intervention impact are separate endpoints.

## 48.6 Oral defense

Be ready to defend why the estimand matters, how treatment is defined, where time zero lies, which paths are adjusted, what is unmeasured, where positivity fails, why estimator matches identification, how uncertainty was calculated, what sensitivity means, where results transport, and what would falsify the claim.

“The software did it” is not a research defense.

---

# Capstone companion: exercises and certification

## C.1 Exercise method

Before coding each exercise, write:

1. target trial or assignment mechanism;
2. counterfactual estimand and population;
3. identification assumptions;
4. estimator and diagnostic;
5. deliberate failure case; and
6. bounded conclusion.

The exercises progress from potential-outcome arithmetic and DAG adjustment through trials, noncompliance, propensity design, standardization, doubly robust failure matrices, whole-procedure bootstrap, hidden-confounding tipping points, longitudinal treatment, DiD, RD, IV, honest heterogeneity, reproducibility, incidents, and a registered study.

## C.2 Certification dimensions

A research-ready submission:

- defines treatment versions and time zero;
- separates observed and counterfactual quantities;
- makes identification assumptions criticizable;
- matches adjustment to causal structure;
- demonstrates overlap and balance;
- propagates procedural uncertainty;
- distinguishes confirmatory and exploratory work;
- preserves unfavorable results and deviations;
- produces a clean research bundle;
- communicates transport and sensitivity limits; and
- connects deployment to monitoring, appeal, rollback, and prospective impact evaluation.

An honest conclusion that the effect is unidentified or too uncertain can pass. Unsupported certainty cannot.

---

# Chapter 6 synthesis and practical laboratories

## S.1 One question, six layers

1. Measurement: are treatment, outcome, time, and population valid?
2. Counterfactual target: which intervention worlds and scale?
3. Identification: which design and assumptions connect data to target?
4. Estimation/inference: which procedure and uncertainty?
5. Translation: where does result transport and what decision follows?
6. Production: how is action delivered, monitored, contested, and stopped?

Reproducibility preserves the chain; it cannot repair a wrong early link.

## S.2 Derivations to complete

1. Observed outcome from potential outcomes.
2. ATE, ATT, CATE, risk difference/ratio/odds ratio.
3. G-formula identification steps.
4. Backdoor adjustment for a small DAG.
5. ITT, first stage, and Wald LATE.
6. Standardization and IPW identification.
7. SMD and ESS.
8. AIPW score and influence standard error.
9. Longitudinal stabilized weights.
10. Two-period DiD.
11. Sharp and fuzzy RD contrasts.
12. Policy value and incremental net benefit.
13. Paired method differences and dependence explanation.

## S.3 Practical laboratories

### Laboratory A: simulation truth

Reveal both potential outcomes only after freezing analysis; compare naive, standardization, IPW, and AIPW over repeated samples.

### Laboratory B: DAG traps

Simulate a confounder, mediator, collider, selection variable, proxy, and instrument. Adjust valid and invalid sets and trace bias paths.

### Laboratory C: randomized noncompliance

Simulate compliers, always-takers, never-takers, and exclusion violation. Compare ITT, receipt contrast, first stage, and LATE.

### Laboratory D: overlap failure

Make treatment deterministic for one profile. Compare ATE, restricted, ATT, and overlap targets, weights, ESS, and extrapolation.

### Laboratory E: double robustness

Cross correct/incorrect outcome and propensity models over repetitions. Add hidden confounding to show all adjustment estimators fail together.

### Laboratory F: longitudinal feedback

Simulate three treatment decisions with confounder feedback. Compare naive adjustment, g-formula, and marginal structural model.

### Laboratory G: quasi-experiment contamination

Break DiD with concurrent shock, RD with manipulation, and IV with direct instrument effect. Test design-specific placebos.

### Laboratory H: honest heterogeneity

Use separate discovery and estimation samples; evaluate policy value on untouched data under true constant and varying effects.

### Laboratory I: reproducibility handoff

Give a peer only the bundle. Record and repair every hidden dependency needed to reproduce cohort, estimate, interval, and figure.

### Laboratory J: production incident

Simulate unit change, delayed labels, group action spike, or feedback. Exercise detection, containment, rollback, communication, and prevention.

---

# Formula sheet with plain-language readings

| Formula | Meaning |
|---|---|
| $Y=AY(1)+(1-A)Y(0)$ | observed potential outcome |
| $E[Y(1)-Y(0)]$ | population ATE |
| $E[Y(1)-Y(0)\mid A=1]$ | ATT |
| $E[Y(1)-Y(0)\mid X=x]$ | CATE |
| $\{Y(1),Y(0)\}\perp A\mid X$ | conditional exchangeability |
| $0<P(A=1\mid X)<1$ | positivity |
| $E_X[E(Y\mid A=a,X)]$ | g-formula intervention mean |
| $\bar Y_{Z=1}-\bar Y_{Z=0}$ | ITT estimate |
| outcome ITT / treatment first stage | Wald LATE |
| $e(X)=P(A=1\mid X)$ | propensity score |
| $A/e+(1-A)/(1-e)$ | ATE weight |
| $SMD$ | standardized covariate imbalance |
| $(\sum w)^2/\sum w^2$ | effective sample size |
| $m_1-m_0+A(Y-m_1)/e-(1-A)(Y-m_0)/(1-e)$ | AIPW score |
| $SD(\phi)/\sqrt n$ | influence-score standard error |
| $\prod_k P(A_k\mid\text{limited past})/P(A_k\mid\text{full past})$ | longitudinal stabilized weight |
| $(T_{post}-T_{pre})-(C_{post}-C_{pre})$ | DiD contrast |
| outcome jump / treatment jump | fuzzy-RD or IV local ratio |
| $E[Y\{d(X)\}]$ | policy value |

---

# Expanded glossary

**ATE** — Average potential-outcome contrast in a stated population.

**ATT** — Average contrast among actually treated units.

**Backdoor path** — Treatment–outcome path beginning with an arrow into treatment.

**Balance** — Similarity of treatment-group covariate distributions after design.

**CATE** — Average effect conditional on baseline profile or subgroup.

**Causal estimand** — Counterfactual contrast including intervention, comparator, population, outcome, horizon, and scale.

**Collider** — Common effect; conditioning can associate its causes.

**Consistency** — Observed outcome equals potential outcome under received well-defined treatment.

**Confounder** — Pre-treatment common cause of treatment and outcome for the question.

**Cross-fitting** — Nuisance models trained outside observations receiving their predictions.

**DAG** — Directed acyclic graph encoding causal assumptions.

**Difference-in-differences** — Treated change minus comparison change under parallel counterfactual trends.

**Double robustness** — Consistency if one of two nuisance models is correct, given identification.

**Effective sample size** — Concentration summary for nonnegative weights.

**Exchangeability** — Treatment independent of potential outcomes, by design or conditionally.

**Exclusion restriction** — Instrument affects outcome only through treatment.

**G-formula** — Standardization of conditional outcome distributions over target covariates/histories.

**Honesty** — Separate data for structure discovery and effect estimation.

**Instrument** — Treatment shifter satisfying relevance, independence, exclusion, and often monotonicity.

**Interference** — One unit's treatment affects another unit's outcome.

**ITT** — Effect of randomized assignment or offer.

**LATE** — Average effect among instrument compliers under assumptions.

**Marginal structural model** — Model for counterfactual outcomes under histories, often fit with inverse weights.

**Mediator** — Post-treatment variable on a causal pathway.

**Negative control** — Variable expected to share bias but not the causal relation.

**Nuisance function** — Auxiliary propensity or outcome function needed for target estimation.

**Overlap** — Credible presence of both treatment comparisons across target profiles.

**Policy value** — Expected outcome or utility under an assignment rule.

**Positivity** — Required actions have nonzero probability in target histories.

**Potential outcome** — Outcome under a specified intervention.

**Propensity score** — Conditional treatment probability given baseline covariates.

**Regression discontinuity** — Local design exploiting an assignment cutoff.

**Sensitivity analysis** — Structured variation of assumptions or choices.

**Standardization** — Averaging conditional predictions under interventions over a target distribution.

**Target trial** — Protocol of the ideal randomized experiment being emulated.

**Time-zero bias** — Bias from misaligned eligibility, treatment classification, and follow-up.

**Transportability** — Conditions under which a causal result applies to another target setting.

---

# Research-paper reading ladder

| Paper | Central contribution | Replication focus |
|---|---|---|
| Rubin (1974), Holland (1986) | potential outcomes and causal estimands | observed versus missing counterfactuals |
| Greenland, Pearl & Robins (1999) | causal diagrams and adjustment | collider and mediator failures |
| Rosenbaum & Rubin (1983) | propensity-score balancing | balance before outcomes |
| Bang & Robins (2005) | doubly robust estimation | four-model failure matrix |
| Chernozhukov et al. (2018) | orthogonal scores and cross-fitting | in-sample versus cross-fit nuisance predictions |
| Hernán & Robins (2016) | target-trial emulation | align time zero |
| Robins, Hernán & Brumback (2000) | marginal structural models | treatment-confounder feedback |
| Angrist, Imbens & Rubin (1996) | IV and local effects | exclusion and complier population |
| Hahn, Todd & van der Klaauw (2001) | RD identification | bandwidth and manipulation |
| Callaway & Sant'Anna (2021) | staggered DiD group-time effects | heterogeneous adoption |
| Athey & Imbens (2016), Wager & Athey (2018) | honest trees and causal forests | discovery/estimation separation |
| Nadeau & Bengio (2003) | dependence in resampled comparison | overlapping-fold uncertainty |
| Gebru et al. (2021), Mitchell et al. (2019) | dataset and model documentation | complete research cards |

For each paper identify estimand, assignment mechanism, population, assumptions, estimator, uncertainty, evidence, failure mode, and smallest falsifying simulation.

---

# Suggested ten-day study rhythm

Each day:

1. 15 minutes: retrieve yesterday's assumptions.
2. 20 minutes: write target trial/assignment mechanism.
3. 30 minutes: draw DAG or timeline.
4. 40 minutes: derive central identification or estimator.
5. 55 minutes: code and validate on simulation.
6. 25 minutes: deliberately violate one assumption.
7. 20 minutes: reconstruct one paper claim.
8. 15 minutes: write diagnostics, sensitivity, and bounded result.

Do not learn causal inference as a menu of estimators. Repeatedly rebuild the path from intervention to observed comparison.

---

# Common beginner misconceptions

- Conditioning is not intervention.
- Both potential outcomes are never observed for one unit.
- “Treatment effect” needs population, interventions, outcome, horizon, and scale.
- More data do not fix identification.
- No measured imbalance does not prove exchangeability.
- Adjusting for every variable can create bias.
- Mediator adjustment changes the estimand.
- Collider conditioning opens paths.
- Randomization balances in expectation, not perfectly in every sample.
- Receipt comparison is not ITT.
- IV usually identifies a complier effect, not ATE.
- Propensity AUC is not the design objective.
- Extreme weights are scientific lack-of-support warnings.
- Matching discards can change the target population.
- Double robustness does not repair two wrong models or hidden confounding.
- Cross-fitting does not repair causal assumptions.
- A non-significant sensitivity or placebo test does not prove validity.
- Time-varying treatment cannot be backfilled to baseline.
- Pre-trends do not prove parallel trends.
- RD is local to a cutoff.
- High predicted risk is not high treatment benefit.
- CATE is not an individual causal label.
- Cross-validation folds are not independent datasets.
- Reproducibility is not replication or validity.
- A model card does not absolve unsafe deployment.
- Retraining changes the intervention system.
- A lockbox revised against is no longer a test.

---

# Ethical and governance questions

- Is the intervention sufficiently defined and deliverable?
- Who receives benefits, burdens, or opportunity costs?
- Does treating one project reduce resources for another?
- Who is excluded by positivity or missingness?
- Can affected teams contest data, assignment, and outcome definitions?
- Are local/complier effects being generalized beyond their population?
- Are subgroup policies lawful and legitimate?
- Who approves deviations from registration?
- Who owns incident response, rollback, and retirement?
- Are delayed harms and spillovers monitored?
- Will feedback change future labels and eligibility?
- What prospective design can test actual system impact?

---

# Final readiness checklist

You are ready to complete the book when you can:

- translate policy language into a target trial and estimand;
- distinguish predictive conditioning from intervention;
- explain potential outcomes and identification assumptions;
- draw confounders, mediators, colliders, instruments, and selection;
- defend an adjustment set from paths and timing;
- analyze randomized assignment, clusters, attrition, and noncompliance;
- derive standardization, IPW, SMD, ESS, and AIPW;
- diagnose overlap and balance without outcome fishing;
- implement cross-fitting and whole-procedure uncertainty;
- quantify sensitivity to hidden confounding;
- align longitudinal treatment and time zero;
- explain g-formula, MSM, clone–censor–weight, and repeated trials;
- distinguish DiD, RD, IV, synthetic control, and interrupted time series;
- investigate heterogeneity honestly and evaluate policies out of sample;
- compare methods using paired evidence and practical equivalence;
- build data, model, and causal-study documentation;
- verify training-serving parity and monitor feedback;
- define retraining, rollback, and retirement governance;
- execute a registered lockbox analysis without outcome-driven revision; and
- state exactly what the result does not establish.

The book's final principle is:

> Reproducibility preserves a causal argument; it cannot create one.
