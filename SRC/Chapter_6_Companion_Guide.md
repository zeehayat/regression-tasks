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

## Day 39 mastery assignment

### Level 1 — Vocabulary and recognition

1. For “Does a reminder reduce missed appointments?”, identify $A$, $Y$, $Y(1)$, $Y(0)$, $X$, population, time zero, and horizon.
2. Classify each quantity as observed, counterfactual, or sometimes observed: $A$, $Y$, $Y(1)$, $Y(0)$, $P(Y=1\mid A=1)$, and $E[Y(1)]$.
3. Explain the fundamental problem of causal inference without using the word “counterfactual.”
4. Translate consistency, exchangeability, positivity, and no interference into one sentence each for senior review.

### Level 2 — Hand calculations

Given six simulated pairs $(Y(1),Y(0))$: `(0,1), (0,0), (1,1), (0,1), (1,1), (0,0)` and assignments `1,0,1,0,1,0`:

1. calculate every individual effect, ATE, ATT, and average effect among untreated;
2. construct the observed $Y$ vector;
3. calculate the naive treated-minus-control risk difference;
4. calculate causal bias as naive contrast minus ATE; and
5. explain why this calculation is possible only in a simulation.

For $p_1=0.18$ and $p_0=0.30$, calculate risk difference, risk ratio, odds under each action, odds ratio, and number of warnings prevented per 1,000 projects.

### Level 3 — Applied design and code

1. Run `make_mhp_causal_data` without revealing counterfactuals. Produce a cohort table by treatment with counts, outcome risk, and baseline summaries.
2. Write three distinct target trials for the same review program: effect of offer, effect of receipt, and effect of sustained adherence.
3. Reveal simulation truth only after freezing the three estimands. Compare their populations and explain which observed columns cannot identify each without assumptions.
4. Modify the generator so every remote project is treated. Diagnose the resulting positivity failure by profile.

### Level 4 — Research challenge

Choose an intervention in education, health, infrastructure, finance, or public administration. Write a one-page estimand specification containing treatment versions, comparator, eligibility, time zero, outcome, competing events, horizon, population, scale, interference unit, and the exact counterfactual contrast. Propose one realistic violation of each identification assumption and state whether more observations would fix it.

### Probing questions

- Can an ATE be well identified when individual treatment effects cannot be observed?
- When would ATT be more useful than ATE for a policy decision?
- If the risk difference is constant across populations, must the risk ratio be constant?
- Does excellent treatment-overlap prove exchangeability?
- Which part of your intervention definition is most vulnerable to hidden treatment versions?

### Completion evidence

You have mastered Day 39 when another reader can reconstruct your target trial, reproduce every hand calculation, distinguish estimand from estimator, and identify at least one untestable assumption without prompting.

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

## Day 40 mastery assignment

### Level 1 — Path recognition

For each structure, name the variable role and state whether conditioning is normally appropriate for a total effect:

1. $A\leftarrow C\rightarrow Y$;
2. $A\rightarrow M\rightarrow Y$;
3. $A\rightarrow K\leftarrow Y$;
4. $Z\rightarrow A\rightarrow Y$; and
5. $A\rightarrow S\leftarrow U\rightarrow Y$ with analysis restricted to $S=1$.

Trace every open path from $A$ to $Y$ before and after conditioning.

### Level 2 — Draw and criticize

1. Draw a DAG for review and warnings containing terrain, contractor experience, review, updated month-12 progress, cancellation, and warning.
2. Add one unmeasured management-quality variable, one selection variable, one proxy, and one possible instrument.
3. Find a minimally sufficient total-effect adjustment set.
4. Construct three invalid adjustment sets: one containing a mediator, one opening a collider, and one failing to block confounding.
5. Explain in words which path each invalid set opens or leaves open.

### Level 3 — Simulation laboratory

Simulate four datasets:

1. confounding only;
2. confounding plus a mediator;
3. treatment and outcome both causing selection; and
4. treatment and an unmeasured outcome cause both causing a collider.

For each, compare unadjusted regression with graph-appropriate and graph-inappropriate adjustment. Repeat at least 500 times and plot estimator bias. Confirm that a highly predictive adjustment variable can increase causal bias.

### Level 4 — Research challenge

Interview or consult a domain process description before seeing outcomes. Produce a DAG change log documenting who proposed each arrow, what evidence supports it, which nodes are unmeasured, which alternative graphs remain plausible, and how estimates would be reported across those graphs. Include a transport/selection diagram for a second setting.

### Probing questions

- Can a variable be a confounder for one estimand and a mediator for another?
- Why can conditioning on complete records create bias?
- When might adjustment for an instrument amplify residual confounding?
- What would make two experts disagree about a valid adjustment set?
- Which arrows in your DAG are most consequential yet least measurable?

### Completion evidence

Mastery requires correctly tracing paths without software, defending covariate timing, explaining why every adjusted variable is included, and stating how a plausible alternative DAG changes identification.

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

## Day 41 mastery assignment

### Level 1 — Trial mechanics

1. Distinguish allocation sequence generation, allocation concealment, blinding, treatment receipt, adherence, and outcome ascertainment.
2. Explain why baseline balance is expected across repeated randomizations but not guaranteed in one trial.
3. Identify the independent analysis unit in individual, classroom, clinic, and district randomization.
4. State the causal question answered by ITT.

### Level 2 — Hand calculations

For a trial with 200 assigned to review and 200 to control, 34 versus 50 warnings, and 150 versus 20 actually receiving review:

1. calculate arm risks, ITT risk difference, risk ratio, and first stage;
2. calculate the Wald estimate and name its target population;
3. compare treatment-received risks and explain why their difference is not protected by randomization; and
4. calculate cluster-level means for a supplied four-district version before attempting individual regression.

### Level 3 — Randomization simulation

1. Simulate 1,000 blocked trials and verify average ITT bias and interval coverage.
2. Add noncompliance driven by prognosis. Compare ITT, as-treated, per-protocol restriction, and Wald LATE.
3. Violate exclusion by making assignment itself motivate better reporting. Explain the changed meaning of the Wald ratio.
4. Simulate 12 cluster-randomized districts with within-district correlation. Compare naive individual standard errors with cluster-aware uncertainty.
5. Add differential attrition and implement best/worst-case outcome bounds.

### Level 4 — Research challenge

Write a randomized evaluation protocol for a realistic intervention. Include randomization unit, blocking, concealment, treatment versions, adherence measurement, spillovers, primary ITT, secondary per-protocol strategy, attrition prevention, missing-outcome sensitivity, analysis unit, stopping, harms, and transport population.

### Probing questions

- Can ITT be policy-relevant when adherence is low?
- When is cluster randomization necessary rather than convenient?
- Why does exclusion concern assignment as well as treatment?
- What population comprises compliers, and can they be identified individually?
- Could blinding be impossible yet randomized evidence remain useful?

### Completion evidence

Mastery means you can recover the effect of assignment from raw arm counts, diagnose why receipt comparisons fail, state every LATE assumption, and align uncertainty with the randomization unit.

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

## Day 42 mastery assignment

### Level 1 — Method mapping

1. Match standardization, IPW, matching, stratification, ATT weighting, and overlap weighting to the population each targets.
2. Explain why a treatment-prediction AUC of 1 can be disastrous for ATE estimation.
3. Identify which components model outcome, treatment assignment, or covariate distribution.
4. Explain why propensity analysis should initially hide outcomes.

### Level 2 — Hand calculations

For five rows with supplied $A,Y,e(X),m_1(X),m_0(X)$:

1. calculate standardized intervention means and ATE;
2. calculate Horvitz–Thompson ATE contributions;
3. calculate ATE, ATT, stabilized, and overlap weights;
4. calculate weighted means, one SMD, and ESS; and
5. identify which row dominates and why.

Then fit a two-stratum example by hand and show that standardization and correctly weighted estimation target the same marginal contrast.

### Level 3 — Design laboratory

1. Hide outcomes and design four comparisons: nearest-neighbor matching, propensity strata, ATE IPW, and overlap weighting.
2. Produce propensity-overlap plots, love plots for SMDs, weight histograms, ESS, discarded-profile table, and target-population description.
3. Freeze designs, reveal outcomes, and estimate effects with uncertainty respecting pairs/weights.
4. Remove terrain from the propensity model and observe whether balance diagnostics detect the omission.
5. Force deterministic treatment in remote projects. Compare unrestricted ATE with restricted and overlap-population effects.

### Level 4 — Research challenge

Design an observational adjustment analysis without using outcome data. Justify the confounder set from a DAG, choose a primary target population, define acceptable overlap, register match/weight rules, specify what happens if balance fails, and write the exact sentence that will replace an ATE claim if support is inadequate.

### Probing questions

- When matching discards treated units, whose effect remains?
- Can all SMDs be small while nonlinear or joint imbalance remains?
- Is weight truncation a numerical fix or an estimand/procedure change?
- Why can standardization extrapolate silently where IPW becomes visibly unstable?
- Which method makes your target population easiest to explain?

### Completion evidence

Mastery requires reproducing weights and standardized predictions by hand, reporting balance beyond one threshold, identifying the target population after design, and refusing unsupported extrapolation.

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

## Day 43 mastery assignment

### Level 1 — AIPW anatomy

1. Label the plug-in contrast and two residual-correction terms in the AIPW score.
2. Explain what happens to a treated row's correction when $e(X)$ approaches zero.
3. State double robustness precisely and list five things it does not guarantee.
4. Explain cross-fitting to a reader who knows train/test splitting but not influence functions.

### Level 2 — Score calculation

Using a six-row table of $A,Y,\hat e,\hat m_1,\hat m_0$:

1. compute every AIPW score;
2. average them and calculate the influence-score standard error;
3. identify the most influential row;
4. recalculate after clipping propensities; and
5. explain the bias–variance trade-off introduced by clipping.

### Level 3 — Four-cell failure matrix

Simulate a nonlinear confounded outcome and compare g-computation, IPW, in-sample AIPW, and cross-fitted AIPW when:

1. both nuisance models are adequate;
2. only the outcome model is adequate;
3. only the propensity model is adequate; and
4. neither is adequate.

Repeat across sample sizes and overlap levels. Report bias, RMSE, coverage, nuisance calibration, maximum score, and failure rate. Then add an unmeasured confounder and demonstrate that all four statistically sophisticated estimators can converge to the wrong causal answer.

### Level 4 — Uncertainty and sensitivity study

Implement a whole-procedure bootstrap that resamples the independent unit, recreates folds, retunes nuisance learners, and recomputes AIPW. Compare percentile, normal, and influence intervals. Add a quantitative bias analysis calibrated to the strongest measured confounders and identify a decision-relevant tipping point.

### Probing questions

- Why can good nuisance prediction coexist with biased effect estimation?
- Which product of nuisance errors matters for orthogonal estimation?
- What uncertainty is omitted when saved scores alone are bootstrapped?
- How should folds change for clustered or temporal data?
- What hidden-confounding strength would be plausible in your domain?

### Completion evidence

Mastery means you can calculate the score manually, implement out-of-fold nuisance prediction without leakage, demonstrate double robustness by simulation, and present a substantively calibrated sensitivity statement.

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

## Day 44 mastery assignment

### Level 1 — Timeline diagnosis

1. Draw eligibility, assignment, treatment receipt, outcome follow-up, censoring, and competing-event times for five example projects.
2. Mark every immortal period created by an “ever treated” definition.
3. Rewrite a misaligned observational question so eligibility, strategy assignment, and follow-up share time zero.
4. Distinguish baseline, time-varying, and dynamic strategies.

### Level 2 — Longitudinal notation

For three visits, write $L_0,A_0,L_1,A_1,L_2,A_2,Y$ in temporal order. Specify two static strategies and one realistic dynamic rule. State sequential exchangeability and positivity at each decision.

Given numerator and denominator treatment probabilities at each visit, calculate individual stabilized weights, final ESS, and the change after adding censoring weights.

### Level 3 — Simulation laboratory

1. Simulate treatment–confounder feedback where $A_0$ affects $L_1$, which affects $A_1$ and $Y$.
2. Compare naive final-treatment regression, conventional adjustment for $L_1$, parametric g-formula, and an MSM.
3. Plot weight growth by visit and truncate on a registered grid.
4. Implement clone–censor–weight for two sustained strategies and cluster inference by original subject.
5. Emulate repeated monthly trials and verify one project never crosses train/test or independent-unit boundaries incorrectly.

### Level 4 — Research challenge

Write a full target-trial table for a longitudinal intervention with grace period. Define eligibility, strategies, assignment, time zero, follow-up, outcome, competing events, adherence, causal contrast, covariate history, censoring, analysis, and transport. Create a data-availability matrix showing whether every required variable is measured before each decision.

### Probing questions

- Why does adjusting for $L_1$ both help and harm in treatment–confounder feedback?
- When does a dynamic strategy violate positivity even if each single treatment is common?
- What new assumption does artificial censoring introduce?
- How should cancellation enter a total-effect versus hypothetical estimand?
- When do repeated trials improve information but worsen transport ambiguity?

### Completion evidence

Mastery requires detecting immortal time from a timeline, computing longitudinal weights, explaining g-formula versus MSM assumptions, and producing an implementable target-trial protocol whose variables exist at the needed times.

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

## Day 45 mastery assignment

### Level 1 — Design identification

For ten short scenarios, choose adjustment, randomized trial, DiD, RD, IV, synthetic control, or interrupted time series. For each, name the assignment leverage, target population, estimand, and central untestable assumption.

### Level 2 — Hand calculations

1. Calculate a two-period DiD from four cell means and reconstruct the missing treated counterfactual mean.
2. Calculate sharp-RD jumps from left/right limits and explain why the effect is local.
3. Calculate reduced form, first stage, and Wald IV ratio from binary assignment counts.
4. Construct a simple three-donor synthetic-control weighted pre-period trajectory.
5. Calculate an interrupted-series level and slope change from a supplied segmented regression.

### Level 3 — Failure laboratories

1. DiD: simulate parallel trends, then add anticipation, heterogeneous staggered effects, serial correlation, composition change, and a concurrent shock. Compare two-way fixed effects with group-time effects.
2. RD: simulate a sharp cutoff, estimate local linear effects over bandwidths, then add score manipulation and a discontinuous baseline covariate.
3. IV: simulate valid compliance types, then add weak relevance, direct instrument effect, and defiers.
4. Synthetic control: remove one donor at a time and run placebo interventions.
5. Interrupted series: add seasonality and autocorrelation, then show naive standard errors.

### Level 4 — Research challenge

Find a real or hypothetical policy assignment mechanism and prepare a design memo before outcomes. Include institutional history, exact estimand, comparison population, manipulation/anticipation/spillover risks, falsification tests, inference unit, data support, and reasons alternative designs are weaker or stronger.

### Probing questions

- Why can many parallel pre-periods still fail to identify DiD?
- Who are controls under staggered adoption?
- How far from an RD cutoff may the result reasonably travel?
- Can instrument relevance be strong while exclusion is implausible?
- Which placebo would most directly expose your design's likely failure?

### Completion evidence

Mastery means you can derive each basic contrast, state its local/group-time/complier population, select design-specific diagnostics, and decline “natural experiment” language without a defended assignment mechanism.

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

## Day 46 mastery assignment

### Level 1 — Interpretation

1. Explain why CATE is not an individual treatment effect.
2. Give an example of constant risk ratio with varying risk difference.
3. Distinguish prognostic risk from treatment benefit.
4. Explain why subgroup significance versus nonsignificance is not an interaction test.

### Level 2 — Prespecified subgroup work

Using cross-fitted AIPW scores and a prespecified binary modifier:

1. calculate group-specific effects and standard errors;
2. calculate the interaction contrast;
3. report baseline risks, overlap, treated/control counts, and multiplicity plan;
4. repeat on risk-difference and risk-ratio scales; and
5. explain why conclusions differ across scales.

### Level 3 — Learner comparison

Simulate constant and heterogeneous effects. Compare S-, T-, X-, R-, and DR-learners plus honest causal forest using nested/cross-fitted nuisance estimation. Evaluate:

- ATE recovery;
- calibration of grouped predicted effects;
- policy value on untouched data;
- stability of modifier rankings;
- overlap within learned subgroups; and
- performance under a truly constant effect.

Do not evaluate individual CATE RMSE on real data where individual effects are unobserved.

### Level 4 — Policy learning challenge

Learn a capacity-constrained rule that may treat 20% of projects. Compare treat-all, treat-none, highest-risk, and highest-estimated-benefit policies using doubly robust held-out policy value. Add treatment cost, uncertainty penalty, minimum group access, and a rule-complexity constraint. Produce a policy card describing eligibility, expected value, burden, exclusions, support, appeals, and monitoring.

### Probing questions

- When can a high-risk rule outperform an estimated-benefit rule?
- How does weak overlap masquerade as large heterogeneity?
- What is gained and lost through honest sample splitting?
- Which fairness constraint belongs in policy learning rather than model fitting?
- What evidence is required to transport a modifier to another setting?

### Completion evidence

Mastery requires separating risk and benefit, testing interactions directly, evaluating learned policies out of sample, and reporting uncertainty/support for every subgroup used in a decision.

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

## Day 47 mastery assignment

### Level 1 — Reproducibility audit

Classify each artifact as needed for computational reproducibility, replication, robustness, transport, or governance: source data hash, environment lock, seed, independent dataset, sensitivity grid, model card, causal-study card, incident runbook, and external validation.

Explain why a seed, notebook, container, or hash alone is insufficient.

### Level 2 — Paired comparison

Given two models' losses on the same 20 projects:

1. calculate rowwise differences, mean difference, standard error, and interval;
2. compare with an incorrect unpaired calculation;
3. define a smallest practically important difference and assess equivalence;
4. adjust interpretation for five metrics and four horizons; and
5. explain why ten cross-validation fold scores are not ten independent studies.

### Level 3 — Build the research bundle

From a clean checkout or directory, create a one-command workflow that regenerates cohort count, primary estimate, interval, table, and figure. Include:

- README and entry command;
- environment lock;
- data and artifact manifests with hashes;
- deterministic limitations;
- schema and synthetic fixture tests;
- full result tables, not only winners;
- data sheet, model card, and causal-study card;
- registration and deviation log; and
- plain-language summary.

Give the bundle to a peer without verbal help and record every hidden dependency.

### Level 4 — Production stress test

Design and rehearse incidents for a unit conversion, unseen category, delayed outcomes, subgroup action spike, capacity failure, corrupted feature timestamp, feedback-driven label change, and harmful override pattern. For each specify detection, severity, owner, containment, rollback, communication, root-cause analysis, and prevention.

### Probing questions

- Can an exactly reproducible result be causally invalid?
- What does a hash prove and not prove?
- Which changes require recalibration, full revalidation, or a new causal study?
- How does deployment change the future data-generating process?
- What retirement signal cannot be solved by retraining?

### Completion evidence

Mastery requires an independent peer successfully reproducing the registered outputs, a defensible paired comparison, complete cards/manifests, and a tested rollback path.

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

## Day 48 mastery assignment

### Level 1 — Protocol reconstruction

Read a published or fictional result paragraph and reconstruct its population, treatment, comparator, time zero, outcome, horizon, estimand, design, assumptions, estimator, uncertainty, and transport claim. Mark every missing element.

### Level 2 — Lockbox rehearsal

1. Create synthetic development and lockbox files with hashes.
2. Write integrity tests before opening lockbox outcomes.
3. Freeze configuration, code revision, environment, and allowed claim rules.
4. Simulate an integrity-test failure and practice stopping without inspecting the effect.
5. Open once, save individual scores/predictions where allowed, and create an immutable access/deviation record.

### Level 3 — Complete registered mini-study

Choose one:

1. baseline target-trial emulation with standardization/IPW/AIPW;
2. RD, DiD, or IV analysis tied to a genuine assignment mechanism; or
3. prospective randomized model-impact evaluation.

Submit registration before primary outcome access. Deliver cohort flow, DAG, estimand, primary and secondary methods, diagnostics, sensitivity, uncertainty, groups, full results, research bundle, cards, and a 300-word bounded conclusion.

### Level 4 — Adversarial oral defense

Ask a peer to challenge:

- treatment versions and time zero;
- missing confounders and collider adjustment;
- positivity and target population;
- estimator/estimand mismatch;
- multiplicity and researcher choices;
- uncertainty unit;
- sensitivity calibration;
- transport, interference, and implementation;
- unfavorable or null findings; and
- conditions that would reverse the decision.

Revise documentation, not the locked primary estimate, in response.

### Research and probing questions

- What observation would falsify the deployment claim?
- Which assumption contributes the largest uncertainty not shown in the interval?
- Would a different stakeholder choose a different estimand or effect scale?
- If the primary model loses to a simple baseline, what happens next?
- Which outcome or harm should be measured prospectively before deployment?
- What evidence would justify expanding from the study population?
- When is “we cannot identify this effect” the most useful result?

### Completion evidence

Chapter mastery is demonstrated when the study can be rerun from a clean environment, the lockbox history is auditable, every assumption is connected to evidence or sensitivity, an adversarial reader can reproduce the result, and the conclusion remains bounded after criticism.

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
