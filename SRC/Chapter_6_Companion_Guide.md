# Chapter 6 Companion Guide: From Prediction to Intervention
## A First-Principles Mathematical & Computational Reference Manual

This companion guide is an exhaustive, first-principles reference manual accompanying **Chapter 6: From Prediction to Intervention** [cite: 24]. It unpacks the mathematics of causal inference, directed acyclic graphs (DAGs), potential outcomes, quasi-experimental designs, and the absolute necessity of research reproducibility for the Microhydro Power (MHP) Cost Estimator [cite: 24].

All mathematical notation is strictly formatted using standard LaTeX delimiters (`$inline$` and `$$display$$`) for seamless rendering.

---

## Table of Contents
1. **Day 39: Counterfactuals, Estimands, and Identification**
2. **Day 40: Causal Diagrams and Adjustment**
3. **Day 41: Randomised Experiments and Noncompliance**
4. **Day 42: Standardisation, Matching, and Weighting**
5. **Day 43: Doubly Robust Estimation and Sensitivity**
6. **Day 44: Target Trials and Time-Varying Treatment**
7. **Day 45: Quasi-Experimental Designs**
8. **Day 46: Heterogeneous Effects and Policy Learning**
9. **Day 47 & 48: Reproducibility and The Registered Study**
10. **Master Rosetta Stone & Formula Sheet**

---

## 1. Day 39: Counterfactuals, Estimands, and Identification

A predictive model answers "What is the expected outcome if I observe $X$ and $A$?" ($P(Y=1 \mid A, X)$) [cite: 24]. A causal model answers "What is the outcome if I *force* action $A$?" ($P\{Y(a)=1\}$) [cite: 24]. They are fundamentally different questions [cite: 24].

### Potential Outcomes
For any project $i$, there are two potential outcomes [cite: 24]:
* $Y_i(1)$: The warning outcome if project $i$ *received* the senior review [cite: 24].
* $Y_i(0)$: The warning outcome if project $i$ *did not receive* the senior review [cite: 24].
The **fundamental problem of causal inference** is that we can only observe one of these states for a given project at a given time [cite: 24]:
$$Y_i = A_iY_i(1) + (1-A_i)Y_i(0)$$

### Causal Estimands
Before picking an algorithm, you must define your target quantity [cite: 24]:
* **Average Treatment Effect (ATE):** $E\{Y(1) - Y(0)\}$ [cite: 24]. What if *everyone* was reviewed vs. *no one*? [cite: 24]
* **Average Treatment Effect on the Treated (ATT):** $E\{Y(1) - Y(0) \mid A=1\}$ [cite: 24]. What was the effect for the projects that *actually were* reviewed? [cite: 24]

### The Four Identification Assumptions
To identify a causal effect from observational data, four strict assumptions must hold [cite: 24]:
1. **Consistency:** If a unit receives treatment $A=a$, its observed outcome exactly equals its potential outcome $Y(a)$ [cite: 24]. This requires the treatment to be well-defined (no hidden variations of "review") [cite: 24].
2. **Conditional Exchangeability:** $\{Y(1), Y(0)\} \perp A \mid X$ [cite: 24]. After adjusting for $X$, treatment is assigned as-if randomly [cite: 24]. *There are no unmeasured confounders.* [cite: 24]
3. **Positivity:** $0 < P(A=1 \mid X=x) < 1$ [cite: 24]. Every sub-population must have some probability of receiving (or not receiving) the treatment [cite: 24].
4. **No Interference:** One project's treatment does not affect another project's outcome (e.g., reviewing one project doesn't consume all the engineers and doom the next project) [cite: 24].

---

## 2. Day 40: Causal Diagrams and Adjustment

A Directed Acyclic Graph (DAG) is a rigorous visual map of your causal assumptions [cite: 24]. It tells you exactly what to control for, and more importantly, what *not* to control for [cite: 24].

### Variable Roles in a DAG
* **Confounder ($A \leftarrow C \rightarrow Y$):** Causes both treatment and outcome [cite: 24]. *Must be adjusted for.* [cite: 24]
* **Mediator ($A \rightarrow M \rightarrow Y$):** A step on the causal pathway [cite: 24]. *Do not adjust for this if you want the total effect.* [cite: 24]
* **Collider ($A \rightarrow K \leftarrow Y$):** Caused by both treatment and outcome [cite: 24]. *Do NOT adjust for this. Doing so opens a biased path.* [cite: 24]

### The Backdoor Criterion
A set of variables $X$ is sufficient for adjustment if [cite: 24]:
1. It contains no descendants of the treatment [cite: 24].
2. It blocks every "backdoor path" from $A$ to $Y$ (paths that have an arrow pointing into $A$) [cite: 24].

---

## 3. Day 41: Randomised Experiments and Noncompliance

Randomization physically forces the exchangeability assumption to be true by severing all backdoor paths into $A$ [cite: 24].

### Intention-to-Treat (ITT)
If you randomly assign ($Z$) a review, but some managers refuse to execute the review ($A$), you must analyze based on the *assignment* ($Z$), not the *receipt* ($A$) [cite: 24].
$$\widehat{\tau}_{\text{ITT}} = \bar Y_{Z=1} - \bar Y_{Z=0}$$
Comparing only the people who *received* treatment breaks randomization, because compliance is driven by confounding factors (e.g., highly competent managers are more likely to comply) [cite: 24].

### Instrumental Variables & LATE
Under strict assumptions (Relevance, Independence, Exclusion, Monotonicity), we can use the assignment $Z$ as an instrument to find the effect on the **Compliers** [cite: 24]:
$$\widehat\tau_{\text{LATE}} = \frac{E(Y \mid Z=1) - E(Y \mid Z=0)}{E(A \mid Z=1) - E(A \mid Z=0)}$$

---

## 4. Day 42: Standardisation, Matching, and Weighting

### Propensity Scores
The propensity score is the probability of receiving treatment given the covariates [cite: 24]: 
$$e(X) = P(A=1 \mid X)$$
It is a balancing score, not a predictive model. If the propensity model separates the classes perfectly, you have a severe Positivity violation [cite: 24].

### Inverse-Probability Weighting (IPW)
IPW creates a pseudo-population where treatment assignment is independent of the covariates [cite: 24]. The ATE weights are [cite: 24]:
$$w_i = \frac{A_i}{e(X_i)} + \frac{1-A_i}{1-e(X_i)}$$
*Danger:* Extreme weights signal a lack of overlap [cite: 24]. A few heavily weighted observations can destroy the variance of your estimate [cite: 24].

---

## 5. Day 43: Doubly Robust Estimation and Sensitivity

### Augmented Inverse-Probability Weighting (AIPW)
AIPW combines outcome modeling ($\hat{m}$) and propensity weighting ($\hat{e}$) [cite: 24].
$$\widehat\psi_{\text{AIPW}} = \frac{1}{n}\sum_{i=1}^n\left[ \hat m_1(X_i) - \hat m_0(X_i) + \frac{A_i\{Y_i-\hat m_1(X_i)\}}{\hat e(X_i)} - \frac{(1-A_i)\{Y_i-\hat m_0(X_i)\}}{1-\hat e(X_i)} \right]$$
* **"Doubly Robust"** means the estimator is consistent if *either* the outcome model *or* the propensity model is correctly specified [cite: 24]. It does *not* fix unmeasured confounding [cite: 24].

### Cross-Fitting
To prevent the machine learning models ($\hat{m}, \hat{e}$) from overfitting to their own training data, we use $K$-fold cross-fitting [cite: 24]. Nuisance models are trained on $K-1$ folds, and the causal score is evaluated on the held-out fold [cite: 24].

---

## 6. Day 44: Target Trials and Time-Varying Treatment

### Target Trial Emulation
To prevent biases in observational data, define the hypothetical randomized trial you *wish* you could have run [cite: 24]. Align Eligibility, Treatment Assignment, and Follow-up exactly at "Time Zero" [cite: 24].

### Time-Zero Bias (Immortal Time Bias)
If a project is classified as "Reviewed" because it was reviewed at month 10, but observation started at month 0, the project was "immortal" for 10 months [cite: 24] (if it failed at month 5, it could never have reached the review) [cite: 24]. This artificially makes the review look highly protective [cite: 24].

---

## 7. Day 45: Quasi-Experimental Designs

When you cannot measure all confounders, Quasi-Experiments exploit natural assignment mechanisms [cite: 24].

* **Difference-in-Differences (DiD):** Compares the pre/post change in a treated group to the pre/post change in a control group [cite: 24]. Requires the **Parallel Trends Assumption**: absent treatment, the groups would have evolved in parallel [cite: 24].
* **Regression Discontinuity (RD):** Explores treatments assigned by a strict threshold (e.g., all projects with risk $> 80$ get reviewed) [cite: 24]. It estimates a local effect right at the boundary $\lim_{r \downarrow c} E(Y \mid R=r) - \lim_{r \uparrow c} E(Y \mid R=r)$ [cite: 24].

---

## 8. Day 46: Heterogeneous Effects and Policy Learning

### Conditional Average Treatment Effect (CATE)
The effect of a treatment for a specific covariate profile $x$ [cite: 24]:
$$\tau(x) = E\{Y(1) - Y(0) \mid X=x\}$$

### Honest Causal Forests
Searching for subgroups that react strongly to treatment is highly prone to overfitting [cite: 24]. **Honest Causal Forests** split the data into two parts: one part to define the tree splits, and a completely separate part to estimate the causal effect within those leaves [cite: 24].

---

## 9. Day 47 & 48: Reproducibility and The Registered Study

A causal claim is worthless if the exact data, pipeline, hyperparameters, and code cannot be audited [cite: 24].

### The Research Bundle
Your final deliverable must include [cite: 24]:
1. A time-stamped **Protocol** outlining the exact DAG, estimand, and models [cite: 24].
2. An **Environment Lock** (Python/package versions) [cite: 24].
3. Fixed **Random Seeds** [cite: 24].
4. A **Deviation Log** recording any changes made after opening the data [cite: 24].
5. **Model and Causal-Study Cards** detailing assumptions, unmeasured confounders, and populations where the model should *not* be applied [cite: 24].

---

## 10. Master Rosetta Stone & Formula Sheet

| Concept | Formula | Purpose |
|---|---|---|
| ATE Estimand | $E\{Y(1)-Y(0)\}$ | Population average causal effect [cite: 24] |
| Risk Difference | $p_1 - p_0$ | Absolute effect scale [cite: 24] |
| ITT Effect | $\bar Y_{Z=1} - \bar Y_{Z=0}$ | Effect of assignment, ignoring compliance [cite: 24] |
| LATE (Wald Ratio) | $\frac{E(Y\mid Z=1)-E(Y\mid Z=0)}{E(A\mid Z=1)-E(A\mid Z=0)}$ | Complier effect under IV assumptions [cite: 24] |
| Propensity Score | $e(X) = P(A=1\mid X)$ | Conditional probability of treatment [cite: 24] |
| IPW Weights | $\frac{A_i}{e(X_i)}+\frac{1-A_i}{1-e(X_i)}$ | Pseudo-population weighting [cite: 24] |
| Standardised Mean Difference | $\frac{\bar X_{1j}-\bar X_{0j}}{\sqrt{(s_{1j}^2+s_{0j}^2)/2}}$ | Covariate balance diagnostic [cite: 24] |
| Effective Sample Size | $(\sum w_i)^2 / \sum w_i^2$ | Information retention under weighting [cite: 24] |
| AIPW Score | $\hat m_1 - \hat m_0 + \frac{A(Y-\hat m_1)}{\hat e} - \frac{(1-A)(Y-\hat m_0)}{1-\hat e}$ | Doubly robust orthogonal estimation [cite: 24] |
| DiD Estimator | $(\bar Y_{T,post}-\bar Y_{T,pre}) - (\bar Y_{C,post}-\bar Y_{C,pre})$ | Effect estimation assuming parallel trends [cite: 24] |
| RD Estimator | $\lim_{r\downarrow c}E(Y\mid R=r) - \lim_{r\uparrow c}E(Y\mid R=r)$ | Local effect at assignment threshold [cite: 24] |
| CATE | $E\{Y(1)-Y(0)\mid X=x\}$ | Conditional effect for a specific profile [cite: 24] |
| Corrected CV Variance | $\sqrt{(\frac{1}{r}+q)s_d^2}$ | Penalizes variance for overlapping folds [cite: 24] |
