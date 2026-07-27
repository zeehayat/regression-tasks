# Chapter 4 Concept Help

## Open a card only when a classification idea feels unclear

Chapter 4 separates probability, ranking, action, explanation, and fairness. Do not read every card first. Open the one that answers the question currently blocking you, then return to the main chapter. The [full Chapter 4 Companion Guide](chapter4_companion.html) contains the longer explanations.

<details>
<summary><strong>1. A label, a probability, and an action are different</strong></summary>

**Short version:** A label records an outcome, a model estimates a probability, and a policy turns that probability into an action. These are three different layers.

The label must be defined at a legitimate prediction time. A probability such as 0.23 does not mean “the project is positive”; it means the model assigns an estimated event chance. A threshold or top-$k$ rule decides who receives review.

**Try:** name the event window, prediction time, and action for a major cost-overrun review.

**Go deeper:** [Day 21 — The outcome, probability, and prediction contract](chapter4_companion.html#day-21--the-outcome-probability-and-prediction-contract).

</details>

<details>
<summary><strong>2. Probability, odds, and log odds are not interchangeable</strong></summary>

**Short version:** Probability is between 0 and 1. Odds are $p/(1-p)$. Log odds are the logarithm of odds and can take any real value.

Logistic regression makes log odds linear in the features, then maps the linear score back through the sigmoid to obtain a valid probability. An odds ratio is not a percentage-point change in probability; the probability change depends on the starting risk.

**Try:** convert $p=0.20$ to odds and explain what an odds ratio of 1.5 does not say.

**Go deeper:** [Day 22 — Logistic regression from Bernoulli likelihood](chapter4_companion.html#day-22--logistic-regression-from-bernoulli-likelihood).

</details>

<details>
<summary><strong>3. Calibration is different from discrimination</strong></summary>

**Short version:** Discrimination asks whether higher-risk cases tend to rank above lower-risk cases. Calibration asks whether predicted probabilities match observed frequencies.

A model can have excellent ROC AUC and poor probability quality. Brier score and log loss evaluate probability forecasts; reliability tables and diagrams show whether a group receiving probability 0.20 experiences the event about 20% of the time.

Calibration must be learned on data separate from the base fit. A calibrator trained on the same predictions it corrects can overfit.

**Try:** explain why two models with the same labels at threshold 0.5 can still have different log loss.

**Go deeper:** [Day 23 — Proper scoring and calibration](chapter4_companion.html#day-23--proper-scoring-and-calibration).

</details>

<details>
<summary><strong>4. A threshold is a policy choice</strong></summary>

**Short version:** The default threshold 0.5 is not a law. A threshold should reflect false-positive cost, false-negative cost, available review capacity, and the quality of the probabilities.

F1 combines precision and recall, but it is not a monetary cost function. ROC AUC measures ranking, not calibration. Precision depends on prevalence, so it can change when the deployment population changes even if the class-conditional score distributions do not.

**Try:** derive the simplified threshold when missing an event costs five times as much as an unnecessary review.

**Go deeper:** [Day 24 — Thresholds, ranking, and class imbalance](chapter4_companion.html#day-24--thresholds-ranking-and-class-imbalance).

</details>

<details>
<summary><strong>5. Class imbalance changes the evidence, not the meaning of “positive”</strong></summary>

**Short version:** A rare event is not automatically a defective dataset. It changes the variance of probability estimates, the usefulness of accuracy, and the operational meaning of false positives.

Class weighting, oversampling, undersampling, and threshold movement do different things. Weighted training can improve ranking or recall while making raw outputs no longer represent population probabilities without calibration.

**Try:** state why an accuracy of 90% can be useless when prevalence is 10%.

**Go deeper:** [What imbalance does and does not mean](chapter4_companion.html#day-24--thresholds-ranking-and-class-imbalance).

</details>

<details>
<summary><strong>6. Nonlinear classifiers still need honest evaluation</strong></summary>

**Short version:** Splines, KNN, kernels, trees, forests, and boosting change the shape of the score function. They do not remove the need for scaling where relevant, leakage control, calibration, valid resampling, or a decision contract.

Trees do not require feature scaling in the same way distance-based methods do. KNN is sensitive to scale. Boosting has interacting learning-rate, depth, and stage choices, so early stopping and tuning are validation operations.

**Try:** choose which of KNN, a tree, and a forest is most directly affected by feature scale, and explain why.

**Go deeper:** [Day 25 — Nonlinear, kernel, and multiple-class models](chapter4_companion.html#day-25--nonlinear-kernel-and-multiple-class-models) and [Day 27 — Gradient boosting and honest tuning](chapter4_companion.html#day-27--gradient-boosting-and-honest-tuning).

</details>

<details>
<summary><strong>7. Explanations are not causes</strong></summary>

**Short version:** Coefficients, permutation importance, PDP, ICE, ALE, local surrogates, and SHAP answer different descriptive questions. None automatically identifies what an intervention would change.

Correlated features can share or obscure importance. PDP can evaluate unsupported combinations. SHAP depends on its background distribution, feature-dependence treatment, and output scale. Always state the population, contrast, and explanation target.

**Try:** explain why a feature can be important for prediction without being a good intervention target.

**Go deeper:** [Day 28 — Interpreting fitted models without inventing causes](chapter4_companion.html#day-28--interpreting-fitted-models-without-inventing-causes).

</details>

<details>
<summary><strong>8. Fairness metrics encode different commitments</strong></summary>

**Short version:** Demographic parity, equal opportunity, equalised odds, predictive parity, and group calibration are different constraints. With differing base rates, they may not all be achievable at once.

Fairness begins with label quality, measurement, access to review, and the consequences of errors—not only a parity formula. Small intersectional groups need uncertainty-aware reporting. External and prospective validation test whether the system transports beyond its development sample.

**Try:** name one fairness metric and one reason it cannot be the complete ethical analysis.

**Go deeper:** [Day 29 — Fairness, external validation, and the locked study](chapter4_companion.html#day-29--fairness-external-validation-and-the-locked-study).

</details>

## Still stuck?

Open the full companion guide and complete the relevant day’s exit check. If the difficulty is with validation, leakage, metrics, or uncertainty, revisit the Chapter 2 and Chapter 3 concept-help pages first.
