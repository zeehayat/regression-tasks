# Regression Study Guide: From First Principles to Research Level

## A self-paced, project-based curriculum using the KP synthetic regression datasets

Welcome, future regression wizard! If you want to move from zero knowledge of regression to an applied researcher who can confidently talk about statistical inference, causal identification, and non-parametric ensembles, you are in the right place.

This curriculum is structured around three custom-built synthetic datasets representing the subdistricts, farms, and public infrastructure projects of Khyber Pakhtunkhwa (KP), Pakistan. By working through the chapters and coding the models from scratch, you will build true, research-grade expertise.

We have split the comprehensive study guide into modular chapters, each addressing a specific skill level. 

---

## 📚 Curriculum Table of Contents

### 🚀 [Chapter 0: Level 0 Orientation](file:///var/www/documentation/regression-tasks/level_0_orientation.md)
* **Goal:** Landing on Planet Regression.
* **Topics:** How to use the guide, project environment setup, running `generate_data.py`, and understanding the three KP regression datasets.
* **Exercises:** The Data Handshake, Spotting the Imposters (Identifiers), and Tracing the Data-Generating Code.

### 🔍 [Chapter 1: Level 1 Data Explorer](file:///var/www/documentation/regression-tasks/level_1_data_explorer.md)
* **Goal:** Looking at the numbers (and the math behind them).
* **Topics:** Prediction vs. Explanation vs. Causation, regression vocabulary, OLS math derivation, loss functions (MSE, MAE, Huber, Quantile), and Exploratory Data Analysis (EDA).
* **Exercises:** OLS from Scratch in numpy, the heavy tail of infrastructure cost, and agricultural pH curve visualization.

### 🧪 [Chapter 2: Level 2 Beginner Modeler](file:///var/www/documentation/regression-tasks/level_2_beginner_modeler.md)
* **Goal:** Your first models and honest evaluation.
* **Topics:** Dummy baseline models, linear regression training, train-test splitting rules, data leakage prevention, Group splits (generalizing to new districts), Time splits (forecasting the future), and metrics (MAE, RMSE, $R^2$).
* **Exercises:** Baseline pipeline construction, breaking the random split, and evaluating skewed cost metrics.

### 🛠️ [Chapter 3: Level 3 Practitioner](file:///var/www/documentation/regression-tasks/level_3_practitioner.md)
* **Goal:** Preprocessing, engineering, diagnostics, and regularization.
* **Topics:** Leakproof ColumnTransformers, numeric scaling (Standard, MinMaxScaler, Robust), categorical encoding (One-Hot, Target), feature engineering (interactions, polynomials, ratios), coefficient interpretation, multicollinearity, VIF, residual assumptions & diagnostics, and regularization (Ridge, Lasso, ElasticNet).
* **Exercises:** Leakproof pipelines, feature engineering challenge, and OLS diagnostics via statsmodels.

### 🌲 [Chapter 4: Level 4 Advanced Modeler](file:///var/www/documentation/regression-tasks/level_4_advanced_modeler.md)
* **Goal:** Ensembles, non-linearities, tuning, and interpretability.
* **Topics:** Polynomial splines, Generalized Additive Models (GAMs), Decision Trees, Random Forest bagging (and the tree extrapolation limit!), Gradient Boosting (XGBoost, LightGBM), target transformations (log1p/expm1), Regressor Chains, Optuna tuning, and model interpretation (permutation importance, PDP, ICE, SHAP).
* **Exercises:** Tree extrapolation experiment, target transforms with skewed costs, and Optuna/SHAP tuning.

### 🎓 [Chapter 5: Level 5 Research Apprentice](file:///var/www/documentation/regression-tasks/level_5_research_apprentice.md)
* **Goal:** Inference, causality, advanced families, and uncertainty.
* **Topics:** Parameter standard errors, t-tests, robust covariance estimators (HC3), causal confounding vs. mediators vs. colliders, Directed Acyclic Graphs (DAGs), Generalized Linear Models (GLMs), mixed-effects hierarchical models, Bayesian regression, and conformal prediction prediction intervals.
* **Exercises:** Causal regression adjustment, district hierarchical mixed-effects model, and conformal prediction from scratch.

### 🏆 [Chapter 6: Level 6 Research-Level Practitioner](file:///var/www/documentation/regression-tasks/level_6_research_practitioner.md)
* **Goal:** Model comparisons, reproducibility, production readiness, and capstones.
* **Topics:** Statistical comparisons (Wilcoxon signed-rank, Nadeau-Bengio resampled t-test), ablation studies, reproducibility checklists, Model Cards, feature drift monitoring, and three graduation capstone projects.
* **Exercises:** Paired statistical comparison, creating a Model Card, and running an ablation study.

### 🗄️ [Chapter 7: Appendices and Resources](file:///var/www/documentation/regression-tasks/appendices_and_resources.md)
* **Goal:** Ready-to-copy code templates, textbook roadmaps, primary papers to study, and a comprehensive glossary of terms.

---

## 🛠️ Quick Start

To begin your regression journey:
1. Initialize your Python environment and generate the datasets following the steps in **[Chapter 0](file:///var/www/documentation/regression-tasks/level_0_orientation.md)**.
2. Follow the learning rhythm for each chapter: read, implement, break, and solve.
3. Don't skip the exercises—they are designed to make you a master of the material.

Good luck!
