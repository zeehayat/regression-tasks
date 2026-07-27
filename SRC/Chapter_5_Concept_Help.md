# Chapter 5 Concept Help

## Open a card only when a survival-analysis idea feels unclear

Chapter 5 is about preserving *when* an event happens and why observation ends. Do not read every card first. Open the card that answers the question blocking you, then return to the main chapter. The [full Chapter 5 Companion Guide](chapter5_companion.html) contains the longer explanations and exercises.

<details>
<summary><strong>1. Censoring does not mean “no event”</strong></summary>

**Short version:** If follow-up ends before the event is observed, we know only that the event had not happened yet.

For observed time $Y=\min(T,C)$, an event indicator of 0 tells us $T>C$, not $T=\infty$. Treating every censored project as a permanent non-event biases ordinary classification.

**Try:** write the inequality you know for a project censored at month 18.

**Go deeper:** [Day 30 — The event-history contract](chapter5_companion.html#day-30--the-event-history-contract).

</details>

<details>
<summary><strong>2. A risk set is “still eligible immediately before the event”</strong></summary>

**Short version:** At time $t$, the risk set contains subjects who have entered, remain under observation, and have not already had an event.

It is not everyone in the original dataset. Delayed entry, censoring, and earlier events change membership over time.

**Try:** draw a timeline for three projects and list the risk set just before each event time.

**Go deeper:** [Day 31 — Survival, hazard, and Kaplan–Meier](chapter5_companion.html#day-31--survival-hazard-and-kaplanmeier).

</details>

<details>
<summary><strong>3. Survival, hazard, and cumulative incidence answer different questions</strong></summary>

**Short version:** Survival is the chance of no event by time $t$; hazard is an instantaneous event rate among those currently at risk; cumulative incidence is the probability that a specified event has happened by $t$.

A hazard ratio is not a probability ratio. A survival curve can fall even when the hazard is changing, and competing events affect incidence.

**Try:** explain why “hazard 0.02” is not the same statement as “2% probability.”

**Go deeper:** [Day 31 — Survival, hazard, and Kaplan–Meier](chapter5_companion.html#day-31--survival-hazard-and-kaplanmeier) and [Day 35 — Competing risks and multi-state processes](chapter5_companion.html#day-35--competing-risks-and-multi-state-processes).

</details>

<details>
<summary><strong>4. Kaplan–Meier uses the changing denominator</strong></summary>

**Short version:** At each event time, Kaplan–Meier multiplies the previous survival estimate by the fraction who remain event-free after accounting for the current risk set.

Censored observations reduce later risk sets but are not counted as failures. The estimator is a product of conditional survival fractions, not a simple proportion of zeros.

**Try:** compute one product-limit update when 1 of 5 at-risk subjects fails.

**Go deeper:** [Day 31 — Survival, hazard, and Kaplan–Meier](chapter5_companion.html#day-31--survival-hazard-and-kaplanmeier).

</details>

<details>
<summary><strong>5. Cox regression models relative hazard, not a causal effect</strong></summary>

**Short version:** A Cox coefficient describes how covariates multiply the hazard, conditional on the model and observed history.

The partial likelihood estimates relative effects without specifying the baseline hazard. A hazard ratio above 1 does not mean a proportional increase in event probability, and it does not prove that changing the predictor would cause the event rate to change.

**Try:** state what a hazard ratio of 1.5 does—and does not—say for two otherwise comparable projects.

**Go deeper:** [Day 32 — Cox regression from partial likelihood](chapter5_companion.html#day-32--cox-regression-from-partial-likelihood) and [Day 38 — The registered locked study and causal boundary](chapter5_companion.html#day-38--the-registered-locked-study-and-causal-boundary).

</details>

<details>
<summary><strong>6. Proportional hazards is an assumption to check</strong></summary>

**Short version:** Proportional hazards says the hazard ratio stays constant over analysis time.

Crossing curves, time interactions, and Schoenfeld-residual patterns can signal that the assumption is implausible. Alternatives include time-varying effects, stratification, or an accelerated-failure-time model.

**Try:** name one plot and one model-based check for time-varying effects.

**Go deeper:** [Day 33 — Assumptions, diagnostics, and alternative time scales](chapter5_companion.html#day-33--assumptions-diagnostics-and-alternative-time-scales).

</details>

<details>
<summary><strong>7. Time-varying information must respect the clock</strong></summary>

**Short version:** A predictor may enter a model only after it was actually observed.

Start–stop rows represent changing covariates. Landmark prediction restricts the population to subjects still observable and event-free at the landmark. Using a later value at baseline creates leakage; requiring survival until a future time can create immortal-time bias.

**Try:** decide whether a month-12 progress measure is allowed in an appraisal-time prediction.

**Go deeper:** [Day 34 — Time-varying information and dynamic prediction](chapter5_companion.html#day-34--time-varying-information-and-dynamic-prediction).

</details>

<details>
<summary><strong>8. Competing events change the estimand</strong></summary>

**Short version:** If cancellation prevents a warning, it is not ordinary censoring for the real-world probability of a warning.

Cause-specific hazard describes the warning rate among those currently free of all events. The cumulative incidence function describes the actual probability of warning by a horizon while cancellation remains possible. Kaplan–Meier censoring of cancellations usually overstates that probability.

**Try:** distinguish “warning rate among those still at risk” from “probability of a warning by month 36.”

**Go deeper:** [Day 35 — Competing risks and multi-state processes](chapter5_companion.html#day-35--competing-risks-and-multi-state-processes).

</details>

<details>
<summary><strong>9. Dynamic prediction changes the population and time zero</strong></summary>

**Short version:** A month-12 prediction is for projects alive, observable, and event-free at month 12, using information available by month 12.

It is not the same estimand as a baseline prediction made at appraisal. Report the landmark population, prediction window, and information cutoff explicitly.

**Try:** complete the sentence: “Among projects that are ___ at month 12, we predict ___ during ___.”

**Go deeper:** [Day 34 — Time-varying information and dynamic prediction](chapter5_companion.html#day-34--time-varying-information-and-dynamic-prediction).

</details>

<details>
<summary><strong>10. Survival metrics must account for unknown outcomes</strong></summary>

**Short version:** Ordinary accuracy, ROC AUC, and Brier calculations cannot simply treat censored cases as known labels.

Harrell’s concordance handles some censoring but has limitations. IPCW Brier scores and censoring-aware calibration estimate performance at a stated horizon under a censoring model.

**Try:** name the horizon whenever you report a survival prediction metric.

**Go deeper:** [Day 37 — Censoring-aware evaluation and decisions](chapter5_companion.html#day-37--censoring-aware-evaluation-and-decisions).

</details>

## Still stuck?

Open the full companion guide and complete that day’s smallest hand calculation before running the code. If the difficulty is with likelihood, bootstrap uncertainty, leakage, or causal language, revisit the Chapter 2–4 concept-help pages first.
