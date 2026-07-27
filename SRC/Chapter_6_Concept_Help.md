# Chapter 6 Concept Help

## Open a card only when a causal idea feels unclear

Chapter 6 asks what would change under an intervention. Do not read every card first. Open the card that answers the question currently blocking you, then return to the main chapter. The [full Chapter 6 Companion Guide](chapter6_companion.html) contains the longer explanations and exercises.

<details>
<summary><strong>1. Potential outcomes are the two worlds we cannot observe together</strong></summary>

**Short version:** $Y(1)$ is what the same unit would experience under treatment; $Y(0)$ is what it would experience under the comparator. The individual effect is $Y(1)-Y(0)$, but only one world is observed for each unit.

An average causal effect is identified by design or assumptions, not by simply comparing two columns.

**Try:** state the two potential outcomes for a project eligible for senior review.

**Go deeper:** [Day 39 — Counterfactuals, estimands, and identification](chapter6_companion.html#day-39--counterfactuals-estimands-and-identification).

</details>

<details>
<summary><strong>2. An estimand must name the population, treatment, outcome, time, and scale</strong></summary>

**Short version:** “The effect” is incomplete. Specify who, compared interventions, outcome, follow-up, and whether the contrast is a risk difference, ratio, odds ratio, or another quantity.

ATE, ATT, and a conditional effect answer different questions and need not agree.

**Try:** complete: “Among ___, compare ___ versus ___ for ___ by ___ on the ___ scale.”

**Go deeper:** [Day 39 — Counterfactuals, estimands, and identification](chapter6_companion.html#day-39--counterfactuals-estimands-and-identification).

</details>

<details>
<summary><strong>3. Identification assumptions connect observed data to causal effects</strong></summary>

**Short version:** Consistency, exchangeability, positivity, and non-interference are assumptions—not output from a regression package.

Exchangeability says that, conditional on the chosen covariates, treatment assignment is not confounded for the target contrast. Positivity says every relevant profile has a nonzero chance of each treatment. No observed-data test proves either one completely.

**Try:** explain why a propensity score near 0 or 1 is a practical warning about positivity.

**Go deeper:** [Day 39 — Counterfactuals, estimands, and identification](chapter6_companion.html#day-39--counterfactuals-estimands-and-identification).

</details>

<details>
<summary><strong>4. A DAG is a map of assumptions, not a decoration</strong></summary>

**Short version:** Directed acyclic graphs help distinguish confounders, mediators, colliders, instruments, and selection variables.

Adjusting for a common cause can block a backdoor path. Adjusting for a collider can create a spurious association. Adjusting for a mediator changes the effect being estimated.

**Try:** identify one pre-treatment common cause of review assignment and overrun risk in the running case.

**Go deeper:** [Day 40 — Causal diagrams and adjustment](chapter6_companion.html#day-40--causal-diagrams-and-adjustment).

</details>

<details>
<summary><strong>5. Randomisation makes assignment independent of potential outcomes in expectation</strong></summary>

**Short version:** In a well-run randomised experiment, the assignment mechanism—not statistical adjustment—creates comparability on average.

Intention-to-treat estimates the effect of assignment. It remains meaningful when people do not adhere, but it is not automatically the effect of treatment received.

**Try:** distinguish assignment, treatment receipt, and adherence in one sentence each.

**Go deeper:** [Day 41 — Randomised experiments and noncompliance](chapter6_companion.html#day-41--randomised-experiments-and-noncompliance).

</details>

<details>
<summary><strong>6. Observational adjustment requires overlap and measured confounding control</strong></summary>

**Short version:** Standardisation, matching, and weighting re-create a comparison using observed covariates; they cannot repair an unmeasured confounder or a population with no overlap.

Inspect balance after adjustment, extreme weights, effective sample size, and whether the target population is still represented.

**Try:** explain why a very large inverse-probability weight is a diagnostic signal, not merely a nuisance.

**Go deeper:** [Day 42 — Standardisation, matching, and weighting](chapter6_companion.html#day-42--standardisation-matching-and-weighting).

</details>

<details>
<summary><strong>7. Doubly robust does not mean assumption-free</strong></summary>

**Short version:** An augmented estimator can remain consistent if one of two nuisance models is correctly specified under its regularity conditions, but it still needs identification, adequate overlap, and honest uncertainty.

Flexible nuisance models should be separated from effect estimation through cross-fitting when appropriate. “Doubly robust” is not a guarantee against poor data or a wrong causal question.

**Try:** name the two nuisance components in a treatment-effect estimator.

**Go deeper:** [Day 43 — Doubly robust estimation and sensitivity](chapter6_companion.html#day-43--doubly-robust-estimation-and-sensitivity).

</details>

<details>
<summary><strong>8. A target trial prevents time-zero and immortal-time errors</strong></summary>

**Short version:** Write the hypothetical trial first: eligibility, assignment, treatment strategies, time zero, follow-up, outcome, contrast, and analysis.

If treatment status or eligibility is defined using information after time zero, the comparison can accidentally grant some subjects guaranteed event-free time.

**Try:** state why month 6 must be the same eligibility and assignment time for both review strategies.

**Go deeper:** [Day 44 — Target trials and time-varying treatment](chapter6_companion.html#day-44--target-trials-and-time-varying-treatment).

</details>

<details>
<summary><strong>9. Quasi-experiments rely on design-specific assumptions</strong></summary>

**Short version:** Difference-in-differences needs a credible parallel-trends story; regression discontinuity needs local continuity around a threshold; instrumental variables need relevance and exclusion assumptions.

These designs are not interchangeable substitutes for randomisation. The estimand and target population can be local or policy-specific.

**Try:** name the key untestable assumption for one quasi-experimental design.

**Go deeper:** [Day 45 — Quasi-experimental designs](chapter6_companion.html#day-45--quasi-experimental-designs).

</details>

<details>
<summary><strong>10. Reproducibility preserves a study; it does not identify an effect</strong></summary>

**Short version:** Versioned code, data provenance, manifests, model cards, and deviation logs make an analysis inspectable. They cannot rescue confounding, poor overlap, or an undefined estimand.

Heterogeneous effects and policy learning also require pre-specified modifiers, honest validation, and attention to who receives the resulting action.

**Try:** write one supported claim and one claim your design cannot support.

**Go deeper:** [Day 47 — Comparison, reproducibility, and responsible production](chapter6_companion.html#day-47--comparison-reproducibility-and-responsible-production) and [Day 48 — The registered final study](chapter6_companion.html#day-48--the-registered-final-study).

</details>

## Still stuck?

Open the full companion guide and complete that day’s smallest hand calculation before running code. If the difficulty is with probability, regression, survival, or prediction, revisit the earlier concept-help pages first.
