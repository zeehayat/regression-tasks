# Chapter 3 Concept Help

## Open a card only when a Chapter 3 idea feels unclear

Chapter 3 is a practitioner chapter, not a menu of buttons. Use one card at the moment you need it, then return to the main chapter. The [full Chapter 3 Companion Guide](chapter3_companion.html) contains the longer explanations and code.

<details>
<summary><strong>1. Missingness is information, not just a blank cell</strong></summary>

**Short version:** MCAR, MAR, and MNAR describe different relationships between the missingness process and the data. You usually cannot prove the mechanism from the observed table alone.

Imputation is a learned rule. It must be fitted inside the training boundary. A missingness indicator can preserve the fact that a value was absent, but it does not magically solve MNAR or recover an unobserved value without uncertainty.

**Try:** explain why fitting a median on the complete dataset before splitting is leakage.

**Go deeper:** [Day 13 — Preprocessing is part of the model](chapter3_companion.html#day-13--preprocessing-is-part-of-the-model).

</details>

<details>
<summary><strong>2. Categorical encoding is a modelling assumption</strong></summary>

**Short version:** One-hot encoding treats categories as separate indicators. Ordinal encoding imposes an order. Frequency encoding says commonness carries information. Target encoding uses outcomes and therefore needs cross-fitting.

Naive target encoding can let a row’s own target help construct its feature. That is target leakage. For grouped or temporal deployment, the cross-fitting split must respect groups or time as well as rows.

**Try:** state what information one-hot encoding does not claim about the distance between categories.

**Go deeper:** [Categorical encodings impose hypotheses](chapter3_companion.html#day-13--preprocessing-is-part-of-the-model).

</details>

<details>
<summary><strong>3. Feature engineering is model specification</strong></summary>

**Short version:** A quadratic, interaction, log, ratio, hinge, or spline changes the question the model can express. “Linear regression” means linear in the coefficients, not necessarily linear in the raw features.

For an interaction, a main-effect coefficient is conditional on the value of the other variable. Keep lower-order terms when the hierarchy principle applies. Centre variables before powers and interactions when zero would otherwise be an arbitrary reference.

**Try:** interpret the slope of $x_1$ in $y=\beta_0+\beta_1x_1+\beta_2x_2+\beta_3x_1x_2$.

**Go deeper:** [Day 14 — Feature engineering is model specification](chapter3_companion.html#day-14--feature-engineering-is-model-specification).

</details>

<details>
<summary><strong>4. Ridge trades a little bias for stability</strong></summary>

**Short version:** Ridge adds a squared-coefficient penalty to the loss. It shrinks coefficients toward zero, usually without setting them exactly to zero, and can stabilise predictions when features are correlated.

Standardisation matters because the penalty acts on coefficient magnitude. The intercept is normally not penalised. Ridge improves a predictive procedure; it does not make a coefficient causal or automatically suitable for classical inference.

**Try:** explain why changing kilometres to metres can change a ridge fit even when OLS predictions are converted correctly.

**Go deeper:** [Day 15 — Ridge regression](chapter3_companion.html#day-15--ridge-regression-shrinkage-and-stability).

</details>

<details>
<summary><strong>5. Lasso creates sparsity, but selection is not truth</strong></summary>

**Short version:** Lasso uses an absolute-value penalty. Its kink at zero makes exact zero coefficients possible. Elastic net combines $L_1$ sparsity with $L_2$ stabilisation.

When predictors are correlated, lasso may select one variable and drop another nearly interchangeable one. A selected variable is not automatically the uniquely important variable, and a zero coefficient is not proof of no causal role. Post-selection inference needs its own methods.

**Try:** distinguish “sparse predictive representation” from “scientifically proven absence.”

**Go deeper:** [Day 16 — Lasso, elastic net, and sparse models](chapter3_companion.html#day-16--lasso-elastic-net-and-sparse-models).

</details>

<details>
<summary><strong>6. Multicollinearity, leverage, and influence are different</strong></summary>

**Short version:** Multicollinearity concerns the design and coefficient stability. Leverage concerns how unusual an input row is. Influence concerns how much a row changes a fitted result when removed or perturbed.

A high-leverage row can have a small residual and still matter greatly. A large residual does not automatically mean high leverage. Use the diagnostic quartet and investigate the project record before deleting a point.

**Try:** name one reason an influential project could be a valuable warning rather than a bad observation.

**Go deeper:** [Day 17 — Multicollinearity, leverage, and influence](chapter3_companion.html#day-17--multicollinearity-leverage-and-influence).

</details>

<details>
<summary><strong>7. “Robust” names three different protections</strong></summary>

**Short version:** Robust standard errors protect a stated uncertainty calculation against some variance misspecification. Robust regression changes the loss to reduce sensitivity to vertical outliers. Quantile regression models a conditional quantile rather than the conditional mean.

None automatically fixes confounding, dependence, bad features, leverage, or a wrong deployment question. Cluster-robust inference also needs enough independent clusters; five districts is not a comfortable large-sample regime.

**Try:** choose whether HC3, Huber regression, or quantile regression answers a given question, and say what it does not fix.

**Go deeper:** [Day 18 — Heteroskedasticity and dependent data](chapter3_companion.html#day-18--heteroskedasticity-and-dependent-data) and [Day 19 — Robust and quantile regression](chapter3_companion.html#day-19--robust-and-quantile-regression).

</details>

<details>
<summary><strong>8. A benchmark is a procedure, not a winner’s name</strong></summary>

**Short version:** A research-grade comparison pre-specifies the prediction time, feature boundary, splits, preprocessing, candidate procedures, metric, and revision rule before looking at the locked test result.

Compare complete pipelines, not isolated estimators. If the test result changes the procedure, record that as post-test development and obtain a fresh final evaluation. A good benchmark can conclude that procedures are practically tied or that evidence is too limited.

**Try:** list the three pieces of information that must be fixed before a locked test is opened.

**Go deeper:** [Day 20 — A pre-specified regression benchmark](chapter3_companion.html#day-20--a-pre-specified-regression-benchmark).

</details>

## Still stuck?

Open the full companion guide and complete that day’s exit check. If the difficulty is about splitting, scaling, confidence intervals, or gradient descent, revisit the Chapter 2 concept-help page first.
