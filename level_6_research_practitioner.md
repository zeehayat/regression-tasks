# Chapter 6: Level 6 Research-Level Practitioner — Comparison, Reproducibility, Production, and Capstones

Welcome to Chapter 6! This is the summit. You have traveled from a complete zero to someone who understands the math, preprocessing, non-linear ensembles, hierarchical models, and conformal intervals. 

But there is one final step. To be a research-level practitioner, you must know how to design bulletproof experiments that others can replicate, statistically compare model architectures, document model limitations, prepare models for deployment, and write academic-grade reports.

In this chapter, we will discuss statistical model comparison, reproducibility protocols, production monitoring, dataset tracks, and finally, your three graduation capstones.

---

## 31. Research-Grade Model Comparison (Beyond Average Metrics)

Suppose someone collected fold-level evaluation errors from a 10-fold cross-validation run on two models:
* Model A (Ridge Regression): Average MAE = $12.35$
* Model B (LightGBM): Average MAE = $12.18$

Now, they want to find out: *Is Model B actually better than Model A, or is the $0.17$ difference just random noise?* What do we do?

We perform a paired, resampled comparison.

### 31.1 Paired Folds
Do not just compare the two averages. Because both models were evaluated on the **exact same folds**, the errors are paired.
Let $d_f = \text{MAE}_{A,f} - \text{MAE}_{B,f}$ be the difference in fold $f$. We perform a **paired statistical test** on these differences:
* **Wilcoxon Signed-Rank Test:** A non-parametric test that evaluates whether the median difference is zero. Perfect if differences aren't normally distributed.
* **Nadeau-Bengio Corrected Resampled t-test:** Corrects for the fact that cross-validation folds are not independent (since training sets overlap).

### 31.2 Ablation Studies
In research papers, you must prove *why* your model works. An ablation study removes one component at a time to measure its contribution. E.g.:
1. Full Model (LightGBM + engineered features + log target) $\to$ MAE: 10.2
2. Remove log target $\to$ MAE: 12.8 (Ablation effect: $+2.6$)
3. Remove engineered features $\to$ MAE: 10.9 (Ablation effect: $+0.7$)
This proves that the target log transformation was the main driver of performance.

---

## 32. Reproducibility and Experiment Tracking

If a researcher claims an $R^2$ of 0.85 in a paper, but nobody else can run their code and get the same result, that paper belongs in the trash.

### 32.1 The Reproducibility Checklist
1. **Seed everything:** Set `random_state=42` in scikit-learn, `np.random.seed(42)`, and `random.seed(42)`.
2. **Data Hash:** Save the SHA256 checksum of your CSV files. If the data changes slightly, your results will change.
   ```bash
   sha256sum data/*.csv
   ```
3. **Environment Capture:** Always export your package versions:
   ```bash
   pip freeze > requirements.txt
   ```

### 32.2 Model Cards
Every model you train should have a **Model Card** (originally proposed by Margaret Mitchell et al. at Google). It is a short document stating:
* **Intended Use:** What is this model for? E.g., pre-construction budgeting.
* **Out-of-Scope Use:** What is it NOT for? E.g., predicting real-time daily cost changes.
* **Metrics:** Overall and subgroup performance (e.g., MAE by district terrain).
* **Ethical Considerations:** Biases in training data.

---

## 33. Regression in Production

A model in production is a living organism. It needs feeding and care.

### 33.1 Training-Serving Consistency
If you train your model using `scikit-learn`'s `StandardScaler`, you must save that fitted scaler object (e.g., using `pickle` or `joblib`) and use it to scale incoming production queries. If you scale production queries using their own mean, your predictions will be garbage.

### 33.2 Monitoring and Drift
* **Concept:** *Suppose someone deployed a crop yield model. In the third year, a massive heatwave strikes KP. The average temperature in the input queries jumps from 22°C to 34°C. The model predictions start failing. What do we do?*
* We monitor **Feature Drift**. We compare the distribution of incoming features against our training distributions using tests like the Kolmogorov-Smirnov test. If drift is high, we must retrain the model with fresh data.

---

## 34. Dataset-Specific Study Tracks

We have laid out structured practice paths for each dataset. Refer to [34. Dataset-Specific Study Tracks](file:///var/www/documentation/regression-tasks/regression_study_guide.md#L3142-L3268) for details. Use these tracks to build up your code portfolio.

---

## 35. Skill Tests and Level Certification

Review the [Final Scoring Rubric](file:///var/www/documentation/regression-tasks/regression_study_guide.md#L3351-L3378). You are ready to claim Level 6 certification once you have successfully completed one of the following research capstones.

---

## 36. Research Capstones (Your Graduation Requirements)

Choose **one** of the following capstones to complete. Write a 5-to-8 page research-grade report detailing your findings.

### Capstone A: The Development Index Allocator
* **Question:** Is public funding associated with a higher development score after accounting for terrain, literacy rates, and income, and what assumptions are required to interpret this association causally?
* **Requirements:** Fit OLS with robust standard errors, check VIF, fit a mixed-effects model by district, draw a DAG, and write a policy memo stating the causal limitations.

### Capstone B: Crop Yield Decision Support
* **Question:** Can we predict crop yield with calibrated uncertainty and produce crop-specific insights about soil pH and temperature?
* **Requirements:** Fit a Generalized Additive Model (GAM) for soil pH, compare boosting and Random Forest, calculate conformal prediction intervals, and write an extension officer handbook.

### Capstone C: Infrastructure Overrun Risk Analyzer
* **Question:** Can we predict infrastructure cost and duration with honest validation and calibrated risk intervals suitable for government budgeting?
* **Requirements:** Multi-output regression chain (predict duration, then predict cost), Yeo-Johnson target transformations, quantile regression, and a budget risk report.

---

## Chapter 6 Exercises

These are the final challenges. No training wheels.

### Exercise 6.1: Paired Statistical Comparison
Write a Python script `compare_models.py` to:
1. Load `data/kp_subdistrict_development_index.csv`.
2. Set up a 10-fold cross-validation loop.
3. In each fold, train a `Ridge` regression and a `HistGradientBoostingRegressor`.
4. Save the fold-level test MAEs for both models.
5. Perform a Wilcoxon signed-rank test (`scipy.stats.wilcoxon`) on the differences.
6. Print the test statistic, p-value, and the mean difference with its 95% confidence interval. State whether the difference is statistically significant.

**Helpful Comments:**
* Ensure your random state is identical so both models see the exact same splits.

### Exercise 6.2: Creating a Model Card
Choose the best model you trained for the agricultural yield dataset. Write a markdown file `MODEL_CARD.md` that contains:
1. **Model Details:** Developer, Model Date, Model Version, Model Type.
2. **Intended Use:** Primary uses and target users.
3. **Factors:** Subgroup variables (crop types, organic status).
4. **Metrics:** Model performance metrics (overall and split by crop type).
5. **Training Data & Evaluation Data:** Source and details.
6. **Quantitative Analyses:** Graphs showing prediction residuals.
7. **Caveats & Limitations:** Discussion of synthetic data constraints.

**Helpful Comments:**
* Refer to the Google model card paper in the reading list for styling ideas.

### Exercise 6.3: Running an Ablation Study
On the infrastructure projects dataset:
1. Write a script that trains your best tuned model.
2. Run three ablation runs where you sequentially:
   * Remove the target log transformation.
   * Remove the contractor experience feature.
   * Remove the sector categorical column.
3. Save the test MAE and RMSE for each run.
4. Print a markdown table showing the ablation results. Write a paragraph explaining which component is the most critical.

---

## References

1. **Model Cards for Model Reporting:** Margaret Mitchell et al. [Link to Paper](https://arxiv.org/abs/1810.03993) - Read this to write your Exercise 6.2.
2. **Inference for the Generalization Error:** Nadeau and Bengio (2003). [Link to Paper](https://link.springer.com/article/10.1023/A:1024068626366) - Math on why standard t-tests on CV folds are biased.
3. **Kolmogorov-Smirnov Test for Data Drift:** [Scipy KS-Test Docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html) - Read this to understand how to monitor data distributions in production.
