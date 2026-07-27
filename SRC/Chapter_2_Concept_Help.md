# Chapter 2 Concept Help

## Read this only when a Chapter 2 idea feels unclear

Chapter 2 is intentionally ambitious. Do not read every explanation before starting. Open one card when you meet a term you cannot explain, try the tiny example, and then return to the main chapter.

The full companion guide is available as [Chapter 2 Companion Guide](chapter2_companion.html).

<details>
<summary><strong>1. Scaling versus conditioning</strong></summary>

**Short version:** Scaling changes the numerical units of columns. Conditioning describes how sensitive the calculation is to small changes or rounding. A matrix can have full rank and still be poorly conditioned.

If one feature is measured in kilometres and another in watts, their magnitudes may be very different. QR or SVD can still solve the problem, but scaling often makes the numerical geometry easier to handle. Scaling must be learned from training data only.

**Tiny example:** changing 2 km to 2,000 m changes a coefficient’s number but not a correctly converted prediction.

**Try:** explain why multiplying one design column by 1,000 does not create new information.

**Go deeper:** [Day 6 — Scaling and numerical conditioning](chapter2_companion.html#day-6--scaling-and-numerical-conditioning).

</details>

<details>
<summary><strong>2. QR, SVD, rank, and the pseudoinverse</strong></summary>

**Short version:** QR rewrites the design using perpendicular directions. SVD goes further: it reveals the strength of every informed direction. Tiny singular values signal directions that are weakly identified.

The inverse formula is useful for deriving OLS, but directly forming an inverse is not the safest application method. `lstsq`, QR, and SVD are solver routes; none of them repairs a coefficient that the data do not identify.

**Tiny example:** if one column is exactly twice another, the predictions may be identifiable while the two individual coefficients are not.

**Try:** state what rank deficiency would look like in a singular-value list.

**Go deeper:** [Day 7 — QR, SVD, rank, and the pseudoinverse](chapter2_companion.html#day-7--qr-svd-rank-and-the-pseudoinverse).

</details>

<details>
<summary><strong>3. Probability, likelihood, and error terms</strong></summary>

**Short version:** OLS can be calculated as a geometric projection without probability. Probability enters when we describe how outcomes could vary across repeated or possible projects.

An **error term** belongs to the data-generating model. A **residual** is the observed leftover after fitting one dataset. Likelihood holds the observed data fixed and asks which parameter values make those data most plausible.

Under a Gaussian error model with constant variance, maximising likelihood for the coefficients gives the same answer as minimising squared error. That equivalence depends on the stated model; it is not a universal proof that the data are Gaussian.

**Try:** say which object is fixed in likelihood: the observed data or the parameter.

**Go deeper:** [Day 8 — Probability, likelihood, and uncertainty](chapter2_companion.html#day-8--probability-likelihood-and-uncertainty).

</details>

<details>
<summary><strong>4. Confidence interval versus prediction interval</strong></summary>

**Short version:** A confidence interval can describe uncertainty about an average response. A prediction interval describes where one new project outcome may fall, so it is wider because it includes project-to-project noise.

Neither interval is automatically causal. Classical intervals also rely on assumptions about the mean structure, variance, dependence, and sampling process.

**Tiny example:** uncertainty about the mean cost for many comparable sites is narrower than uncertainty about the actual cost of one new site.

**Try:** explain what the extra “1” inside a new-observation variance term represents.

**Go deeper:** [Confidence interval versus prediction interval](chapter2_companion.html#day-8--probability-likelihood-and-uncertainty).

</details>

<details>
<summary><strong>5. Gradient descent and learning rate</strong></summary>

**Short version:** Gradient descent repeatedly moves parameters opposite the gradient. The learning rate controls the step size; it is a chosen hyperparameter, not something learned by the OLS formula.

For a scaled design and MSE objective:

$$
\beta_{t+1}=\beta_t-\eta\nabla\operatorname{MSE}(\beta_t).
$$

Too large a step can overshoot or diverge. Too small a step can make progress painfully slow. Scaling helps because the loss surface is less stretched across parameter directions.

**Try:** predict what a steadily increasing loss means before changing the code.

**Go deeper:** [Day 9 — Gradient descent from the OLS gradient](chapter2_companion.html#day-9--gradient-descent-from-the-ols-gradient).

</details>

<details>
<summary><strong>6. Training, validation, and test data</strong></summary>

**Short version:** Training data fit the procedure. Validation data guide choices. A locked test set supports the final performance claim. If the test result changes your model, it has become development information.

The split should match the deployment question: random rows for exchangeable new projects, groups for new districts or contractors, and time for future projects.

**Try:** name the split you would use to predict next year’s projects in districts absent from training.

**Go deeper:** [Day 11 — Honest splitting, leakage, and cross-validation](chapter2_companion.html#day-11--honest-splitting-leakage-and-cross-validation).

</details>

<details>
<summary><strong>7. Leakage and cross-validation</strong></summary>

**Short version:** Leakage occurs when information from evaluation data crosses into fitting, preprocessing, feature selection, or model choice. It makes performance look better than the deployment procedure deserves.

Cross-validation creates several training/validation views. A pipeline keeps learned transformations inside each training fold. Grouped and temporal validation are alternatives to ordinary random folds when the deployment boundary is grouped or chronological.

**Tiny example:** computing a mean or standard deviation using the test rows before fitting is leakage, even if the target is never used.

**Try:** identify the boundary violation in “scale the complete dataset, then split.”

**Go deeper:** [Leakage catalogue](chapter2_companion.html#day-11--honest-splitting-leakage-and-cross-validation).

</details>

<details>
<summary><strong>8. Metrics, diagnostics, and bootstrap uncertainty</strong></summary>

**Short version:** RMSE, MAE, median absolute error, $R^2$, and asymmetric loss answer different questions. One average score cannot reveal every operational failure.

Keep a row-level prediction table. Inspect signed errors against predictions, time, district, and project scale. A bootstrap interval for held-out MAE resamples the appropriate evaluation units; if projects are clustered, resampling individual rows may be wrong.

**Try:** choose MAE or RMSE for a policy that is especially sensitive to very large cost misses, and justify the choice.

**Go deeper:** [Day 12 — Metrics, diagnostics, uncertainty, and revision](chapter2_companion.html#day-12--metrics-diagnostics-uncertainty-and-revision).

</details>

## If you are still stuck

Open the full companion guide, search for the concept name, and complete that day’s exit check. If the idea is still unclear, return to Chapter 1’s projection, residual, or held-out-evaluation foundations before pushing ahead.
