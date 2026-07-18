# Chapter 7: Appendices and Resources — Templates, Glossary, and Reading List

Welcome to Chapter 7! This is your reference vault. Keep this page bookmarked. When you are writing a script and forget how to set up `TransformedTargetRegressor` or how to write a split conformal prediction interval from scratch, copy these clean templates.

We have compiled the code templates, a curated list of research papers to read, the reading roadmap, and a comprehensive glossary of terms.

---

## 37. Code Templates

These are ready-to-run boilerplate scripts. Modify them for your exercises.

### 37.1 General Preprocessing & Modeling Pipeline
Use this template to build leakproof preprocessing pipelines for any model.

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def evaluate_regression(y_true, y_pred):
    """Computes standard regression metrics."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }

def build_preprocessing_pipeline(X, scale_numeric=True):
    """Creates a ColumnTransformer for categoricals and numerics."""
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler() if scale_numeric else 'passthrough')
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

def train_and_score(df, target, drop_cols, model, scale_numeric=True):
    """Splits data, builds pipeline, fits model, and returns scores."""
    X = df.drop(columns=drop_cols + [target])
    y = df[target]

    preprocessor = build_preprocessing_pipeline(X, scale_numeric=scale_numeric)
    
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    
    return pipe, evaluate_regression(y_test, pred)
```

### 37.2 GroupKFold Cross-Validation
Use this template to perform district-aware cross-validation.

```python
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.base import clone

def evaluate_group_cv(X, y, groups, pipeline, n_splits=5):
    """Evaluates a pipeline using GroupKFold cross-validation."""
    cv = GroupKFold(n_splits=n_splits)
    results = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        # Clone model to prevent fitted states carrying over
        fold_pipeline = clone(pipeline)
        
        # Split data
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # Fit and predict
        fold_pipeline.fit(X_train, y_train)
        pred = fold_pipeline.predict(X_test)
        
        # Score
        scores = evaluate_regression(y_test, pred)
        scores["fold"] = fold
        results.append(scores)

    return pd.DataFrame(results)
```

### 37.3 statsmodels OLS with Robust Standard Errors
Use this template for classical hypothesis testing and VIF evaluation.

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def fit_ols_with_diagnostics(df, target, features):
    """Fits statsmodels OLS and checks multicollinearity."""
    X = df[features].copy()
    X = sm.add_constant(X)  # Add intercept!
    y = df[target]

    # Calculate VIF
    vifs = pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        index=X.columns
    )
    print("--- Variance Inflation Factors (VIF) ---")
    print(vifs)
    print()

    # Fit regular OLS
    model = sm.OLS(y, X).fit()
    
    # Fit OLS with robust standard errors (HC3)
    robust_model = sm.OLS(y, X).fit(cov_type="HC3")

    return model, robust_model
```

### 37.4 Split Conformal Prediction Intervals
Use this template to generate prediction intervals with coverage guarantees.

```python
import numpy as np
from sklearn.model_selection import train_test_split

def compute_split_conformal_interval(X, y, pipeline, alpha=0.10):
    """Computes split conformal prediction intervals."""
    # Split into train-calibration and test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Split train-calibration into proper train and calibration
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42
    )

    # Train model
    pipeline.fit(X_train, y_train)

    # Predict calibration set and calculate absolute residuals
    cal_pred = pipeline.predict(X_cal)
    cal_residuals = np.abs(y_cal - cal_pred)

    # Find conformal quantile threshold
    n_cal = len(X_cal)
    q = np.quantile(cal_residuals, (1 - alpha) * (1 + 1/n_cal))

    # Predict test set
    test_pred = pipeline.predict(X_test)
    lower_bound = test_pred - q
    upper_bound = test_pred + q

    # Calculate empirical coverage
    coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
    avg_width = np.mean(upper_bound - lower_bound)

    return {
        "coverage": coverage,
        "average_width": avg_width,
        "threshold": q,
        "predictions": test_pred,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "y_true": y_test
    }
```

---

## 38. Reading Roadmap and Books

If you can only buy three physical books for your research career, buy these:
1. **Introduction to Statistical Learning with Applications in R/Python (ISLR/ISLP):** Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani. [Free Download](https://www.statlearning.com/) - The friendliest gateway to machine learning.
2. **Causal Inference: The Mixtape:** Scott Cunningham. [Free Online Book](https://mixtape.scunning.com/) - An incredibly engaging, informal introduction to causal econometrics.
3. **Applied Predictive Modeling:** Max Kuhn and Kjell Johnson. The absolute best book on feature engineering, preprocessing pipelines, and practical model tuning.

---

## 39. Research Papers to Study

When you are ready to read primary literature, study these in order. For each paper, don't just read the abstract; focus on the **methodology section** to see how they justified their validation choices.

| Paper | Key Contribution | Why it matters to you |
|---|---|---|
| **Hoerl & Kennard (1970)** | Ridge Regression | First formal mathematical proof of why shrinking coefficients stabilizes multicollinear systems. |
| **Tibshirani (1996)** | Lasso Regression | Introduced the L1 penalty, showing how to do parameter estimation and feature selection simultaneously. |
| **Breiman (2001)** | Random Forests | Proved that bagging unstable trees creates a powerful non-parametric ensemble. |
| **Chen & Guestrin (2016)** | XGBoost | Details the system design and split-finding mathematics that make boosted trees scale to billions of rows. |
| **Nadeau & Bengio (2003)** | Inference for Generalization Error | Explains why standard t-tests on cross-validation folds underestimate variance, and provides the mathematical correction. |
| **Margaret Mitchell et al. (2019)** | Model Cards for Model Reporting | The seminal paper outlining how to document model limits, datasets, and ethical considerations. |

---

## 40. Glossary

* **Ablation Study:** The process of removing one component of an algorithm or pipeline at a time to measure its specific contribution to overall performance.
* **Bias-Variance Tradeoff:** The fundamental tension in statistical learning. Simple models have high bias and low variance (underfit). Complex models have low bias and high variance (overfit).
* **Collider Variable:** A variable that is a common effect of two other variables. Controlling for a collider in a regression model creates spurious correlation between its causes.
* **Conformer/Conformal Prediction:** A framework that uses calibration residuals to produce prediction intervals with guaranteed finite-sample coverage under the assumption of exchangeability.
* **Data Leakage:** An error where information from the test set or future time periods influences the model training process, leading to invalid, over-optimistic evaluation metrics.
* **Design Matrix ($X$):** A matrix where each row is an observation and each column is a feature (including a column of 1s representing the intercept).
* **Heteroscedasticity:** A condition where the variance of the residuals is not constant across all levels of the independent variables (creates a funnel shape on a residual plot).
* **Multicollinearity:** A statistical phenomenon where two or more features in a regression model are highly correlated, making coefficient estimates unstable.
* **Ordinary Least Squares (OLS):** A optimization method for estimating linear coefficients by minimizing the sum of squared residuals.
* **Residual ($e_i$):** The difference between the true target value and the model's prediction: $y_i - \hat{y}_i$.
* **Variance Inflation Factor (VIF):** A metric measuring the severity of multicollinearity for a specific feature. VIF $> 10$ indicates serious collinearity.
