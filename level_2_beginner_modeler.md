# Chapter 2: Level 2 Beginner Modeler — Your First Models and Honest Evaluation

Welcome back! You've generated the data and explored it in Chapter 1. Now, it's time to build your first models. 

Wait, don't get too excited. We aren't building a deep neural network that burns through GPUs yet. We are going to start with the humblest, most honest models possible. Why? Because if you don't know how well a dumb model performs, you won't know if your fancy model is actually smart or just expensive.

This chapter is organized as **one topic per day** — three days of focused learning. Here's the map:

| Day | Title |
|---|---|
| [Day 11](#day-11-first-regression-models--building-the-baselines) | First Regression Models — Building the Baselines |
| [Day 12](#day-12-model-training-validation-and-testing) | Model Training, Validation, and Testing |
| [Day 13](#day-13-evaluation-metrics--how-to-judge-a-model) | Evaluation Metrics — How to Judge a Model |

> [!NOTE]
> **How to Pace Yourself:** Just like in Chapter 1, we organize the material as one topic per day. Apply Chapter 0's five-step rhythm (read the theory → explain it back → run the experiments → break it on purpose → solve the exercise) to each day before moving to the next. Day 11 builds your baselines, Day 12 covers honest splitting, and Day 13 gives you the metrics to judge them.

---

## Day 11: First Regression Models — Building the Baselines

*Today's goal: by the end of today, you should be able to build and fit your first baseline models in scikit-learn and explain why a "brainless" model is the essential starting point for all regression tasks.*

Suppose someone collected data from communities in KP ($X$, containing attributes like population, utilities access, etc.) and they want to predict the `development_score` ($y$). What do we do?

We start with three baseline choices:

1. **The Dummy Regressor:** This model is completely brainless. It looks at the training set target and always predicts the mean (or median), regardless of the inputs. If the average development score in KP is 52.3, the dummy regressor predicts 52.3 for every single village.
   * *Why use it?* It is your baseline floor. If your machine learning model can't beat the training mean, your model is useless.
2. **Linear Regression:** The classic straight line. It assumes that each feature has an additive, linear effect on the target:
   $$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots$$
   Under the hood, `LinearRegression` solves this using the ordinary least squares (OLS) closed-form matrix equation: $\hat\beta = (X^TX)^{-1}X^Ty$, which you derived and calculated by hand in Chapter 1 Day 5.
3. **Decision Tree Regressor (Shallow):** Our first step into nonlinear territory. It splits the data into nested decision regions (e.g., "Is population > 5,000? Yes/No") and predicts the average target value inside each terminal region (leaf node). By limiting its depth (e.g., `max_depth=3` or `max_depth=5`), we keep it shallow and prevent it from overfitting/memorizing the data.

### 11.1 Hands-On: Fitting the Baselines in Python

Before writing a full training script, let's see how simple it is to instantiate and fit these models using `scikit-learn`:

```python
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Fictional toy data representing 4 subdistricts
# Features: [literacy_rate, health_facility_distance_km]
X_toy = np.array([[60.0, 5.0], [70.0, 10.0], [80.0, 2.0], [50.0, 12.0]])
# Target: development_score
y_toy = np.array([45.0, 55.0, 65.0, 35.0])

# 1. Dummy Regressor (predicts mean of y_toy)
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_toy, y_toy)
print(f"Dummy Predictions: {dummy.predict(X_toy)}")

# 2. Linear Regression (OLS)
lr = LinearRegression()
lr.fit(X_toy, y_toy)
print(f"Linear Reg Coefficients: {lr.coef_}, Intercept: {lr.intercept_:.3f}")

# 3. Decision Tree Regressor (shallow baseline)
tree = DecisionTreeRegressor(max_depth=2, random_state=42)
tree.fit(X_toy, y_toy)
print(f"Decision Tree Predictions: {tree.predict(X_toy)}")
```

### Day 11 Recap
* **Always build a baseline first:** If a complex model does not outperform a dummy mean predictor, it is worthless.
* **DummyRegressor** serves as the performance floor by outputting a constant prediction (the mean or median of the training target).
* **LinearRegression** fits a linear equation to the features using the closed-form OLS solution.
* **Shallow DecisionTreeRegressor** splits features into axis-aligned partitions and predicts regional averages, serving as a non-linear baseline.

> [!TIP]
> **Quick Check:** Suppose you build a machine learning model to predict crop yield. The average yield in the training set is 15 tons per acre. A Dummy Regressor predicts 15 tons for every farm. Your model gets a Mean Squared Error of 25, while the Dummy Regressor gets a Mean Squared Error of 20. Should you deploy your model?
> *Answer:* No! Your model is performing worse than the dummy baseline (it has a higher Mean Squared Error). It is actually worse than predicting a constant mean. You should debug your features or model architecture.

### Day 11 Exercise: Training the Baselines
Write a Python script `train_baselines.py` to:
1. Load `data/kp_subdistrict_development_index.csv`.
2. Drop the identifier `community_id` (see Exercise 0.2 in Chapter 0).
3. Separate features ($X$) and target ($y = \text{development\_score}$).
4. Split the data randomly into 80% train and 20% test.
5. Set up a preprocessing pipeline using `scikit-learn`:
   * Scale numeric features using `StandardScaler`.
   * One-hot encode categorical features using `OneHotEncoder(handle_unknown='ignore')`.
6. Train three models on the training set:
   * `DummyRegressor(strategy='mean')`
   * `LinearRegression()`
   * `DecisionTreeRegressor(max_depth=5, random_state=42)`
7. Evaluate all three models on the test set. Print a table showing MAE, RMSE, and $R^2$ for each.

Here is the starter-code skeleton for `train_baselines.py`. Save this script in the repository and fill in the missing code:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_preprocess(filepath: str):
    """Loads the KP development index dataset, separates features and target, and splits the data.

    Args:
        filepath (str): Path to the CSV dataset file.

    Returns:
        tuple: A four-element tuple containing (X_train, X_test, y_train, y_test)
            representing the training features, testing features, training targets,
            and testing targets respectively.
    """
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Drop the community_id column
    df = df.drop(columns=["community_id"])
    
    # Separate features and target
    X = df.drop(columns=["development_score"])
    y = df["development_score"]
    
    # Split into 80% train, 20% test with random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def build_pipeline(X_train: pd.DataFrame, regressor) -> Pipeline:
    """Builds a scikit-learn pipeline wrapping preprocessing and the given regressor.

    Preprocessing scales numeric columns and one-hot encodes categorical columns.

    Args:
        X_train (pd.DataFrame): The training feature DataFrame (used to infer column types).
        regressor (object): The scikit-learn regressor object to fit.

    Returns:
        Pipeline: Preconfigured scikit-learn Pipeline.
    """
    # Dynamically select numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Define transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )
    
    # Construct and return Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])
    return pipeline

def evaluate_model(pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, model_name: str) -> dict:
    """Fits the pipeline on the training set, predicts on the test set, and computes evaluation metrics.

    Args:
        pipeline (Pipeline): Preconfigured scikit-learn Pipeline.
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target values.
        y_test (pd.Series): Testing target values.
        model_name (str): The name of the model.

    Returns:
        dict: A dictionary containing 'MAE', 'RMSE', and 'R2' metrics.
    """
    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)
    
    # Predict on the test data
    preds = pipeline.predict(X_test)
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"--- {model_name} Results ---")
    print(f"  MAE : {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2  : {r2:.4f}\n")
    
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess("data/kp_subdistrict_development_index.csv")
    
    models = {
        "Dummy Mean Regressor": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=5, random_state=42)
    }
    
    for name, model in models.items():
        pipeline = build_pipeline(X_train, model)
        evaluate_model(pipeline, X_train, X_test, y_train, y_test, name)
```

> [!TIP]
> **Comic Relief:** If your linear model gets an $R^2$ of 0.99, check if you accidentally forgot to drop the target column from $X$. You'd be surprised how often people "predict" the target by passing the target itself as a feature. It's a great way to look like a genius in front of your screen and a fool in front of your colleagues.

---

## Day 12: Model Training, Validation, and Testing

*Today's goal: design robust data splitting strategies and understand the physical mechanism of data leakage so that your evaluations represent honest real-world performance.*

Suppose you want to know if your model is ready for the real world. You trained it on your data, and it got 99% accuracy. Are you done? 

**Absolutely not.** If you test your model on the same data you used to train it, you are cheating. It's like a teacher giving students the exact same questions from the study guide on the final exam. They didn't learn calculus; they learned how to memorize.

To prevent this, we split our data into roles:

```text
+-----------------------------------+-----------------------+-----------------------+
|            Train (60%)            |    Validation (20%)   |       Test (20%)      |
+-----------------------------------+-----------------------+-----------------------+
|  Used to fit model parameters     | Used to compare       | Used ONCE at the end  |
|  (e.g., learn coefficients).      | models and tune       | for a final honest    |
|                                   | hyperparameters.      | evaluation.           |
+-----------------------------------+-----------------------+-----------------------+
```

### 12.1 The Danger of Test Set Leakage
If you look at the test set, change your model structure, test it again, tune hyperparameters on it, and repeat... your test set is no longer an independent test set! It has become part of your training loop. You have leaked information, and your final performance reports will be overly optimistic lies. Keep the test set locked in a vault until the very end.

### 12.2 Group Splits (Generalization to Unseen Areas)
Suppose you split your communities randomly. Since villages in the same district share similar terrain, culture, and local government policies, a random split means village A in Peshawar might be in the training set, while village B in Peshawar is in the test set. The model will perform beautifully because it already "knows" Peshawar.

But what if the government deploys your model to a brand-new district (like Chitral) that was never in the training set? 
A random split will lie to you about how well the model generalizes. Instead, we use a **Group Split** (like `GroupKFold` on the `district` column) where entire districts are held out. This tests if our model understands the underlying features of *development*, or if it just memorized *districts*.

### 12.3 Time Splits (Forecasting the Future)
For the infrastructure dataset, projects have a `year_of_initiation`. If you use projects from 2025 to predict projects in 2018, you are leaking the future. In the real world, you cannot use future inflation rates or contractor portfolios to predict past costs. For real deployment, you train on early years (e.g., 2015–2022) and test on later years (e.g., 2023–2025) to evaluate chronological forecasting ability.

### 12.4 Hands-On: Partitioning Strategies in Python

Here is how you execute these three different splitting strategies using `scikit-learn`:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold

# Create a small mock DataFrame to illustrate splits
data = {
    'district': ['Peshawar', 'Peshawar', 'Mardan', 'Mardan', 'Swat', 'Swat', 'Chitral', 'Chitral'],
    'year': [2018, 2019, 2018, 2019, 2018, 2019, 2018, 2019],
    'feature': [1.2, 1.5, 0.8, 1.1, 2.1, 2.3, 0.5, 0.7],
    'target': [10.2, 12.5, 8.4, 9.1, 18.2, 19.5, 5.2, 6.1]
}
df = pd.DataFrame(data)

# 1. Random Split
X_train, X_test, y_train, y_test = train_test_split(df[['feature']], df['target'], test_size=0.25, random_state=42)
print(f"Random Train Indices: {X_train.index.tolist()}")

# 2. Group Split (Hold out entire districts)
gkf = GroupKFold(n_splits=3)
for fold, (train_idx, test_idx) in enumerate(gkf.split(df, df['target'], groups=df['district'])):
    train_districts = df.iloc[train_idx]['district'].unique()
    test_districts = df.iloc[test_idx]['district'].unique()
    print(f"Fold {fold}: Train Districts {train_districts} | Test Districts {test_districts}")

# 3. Temporal Split (Train on past, test on future)
train_mask = df['year'] <= 2018
test_mask = df['year'] > 2018
print(f"Temporal Train Years: {df[train_mask]['year'].tolist()}")
print(f"Temporal Test Years: {df[test_mask]['year'].tolist()}")
```

### Day 12 Recap
* **Always separate data roles:** Train (fit weights), Validation (tune settings/compare models), Test (final honest scoring).
* **Test leakage** occurs when test set patterns influence model parameters, hyperparameter tuning, or preprocessing.
* **Group splits** evaluate spatial generalization by holding out entire groups (like districts) to prevent geographic leakage.
* **Temporal splits** sort data chronologically and split on a timestamp to evaluate forecasting and prevent leakage of future parameters.

> [!TIP]
> **Quick Check:** You are building a model to predict tomato yields based on soil and weather data collected over 10 years. If you randomly split your observations, what kind of leakage could occur, and how should you split the data instead?
> *Answer:* Random splitting will leak weather patterns across years (e.g., weather from the same year will be in both train and test, making it easy for the model to memorize that specific year's yield). Instead, you should use a temporal split, training on the first 8 years and testing on the last 2 years, or use group-by-year cross-validation.

### Day 12 Exercise: Breaking the Random Split (Group Split)
Let's see if our model actually understands development or just memorized district names. Modify your script or write a new script `group_validation.py` using the skeleton below:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def evaluate_group_cv(filepath: str):
    """Loads the dataset, builds a preprocessing pipeline, and runs 5-fold GroupKFold cross-validation on district groups.

    Args:
        filepath (str): Path to the CSV dataset file.
    """
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Drop community_id
    df = df.drop(columns=["community_id"])
    
    # Separate target and features
    X = df.drop(columns=["development_score"])
    y = df["development_score"]
    groups = df["district"]
    
    # NOTE: We drop 'district' from features to prevent the model from memorizing 
    # district-specific biases, forcing it to generalize purely on development features.
    X_features = X.drop(columns=["district"])
    
    # Preprocessing pipeline
    numeric_cols = X_features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_features.select_dtypes(include=["object", "category"]).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    
    # Define 5-fold GroupKFold
    gkf = GroupKFold(n_splits=5)
    
    # Run cross-validation
    cv_results = cross_validate(
        pipeline, X_features, y, groups=groups, cv=gkf,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
        return_train_score=False
    )
    
    # Extract results
    mae_scores = -cv_results["test_neg_mean_absolute_error"]
    rmse_scores = -cv_results["test_neg_root_mean_squared_error"]
    r2_scores = cv_results["test_r2"]
    
    print(f"--- GroupKFold CV Results ---")
    print(f"  Mean MAE : {mae_scores.mean():.4f} (+/- {mae_scores.std():.4f})")
    print(f"  Mean RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")
    print(f"  Mean R2  : {r2_scores.mean():.4f} (+/- {r2_scores.std():.4f})")

if __name__ == "__main__":
    evaluate_group_cv("data/kp_subdistrict_development_index.csv")
```

After running this, compare these cross-validation scores against the test scores you got from the random split in Day 11. Write down:
1. Why did the average test $R^2$ drop?
2. What does this tell you about deploying this model in a brand-new district?

---

## Day 13: Evaluation Metrics — How to Judge a Model

*Today's goal: mathematically define MAE, RMSE, and $R^2$, and select the correct metric based on the business costs of small versus large errors.*

Suppose your model predicted the cost of five projects. How do we measure the errors?

Let's trace a concrete example table:

| Project | True Cost ($y$) | Predicted Cost ($\hat{y}$) | Error ($y - \hat{y}$) | Absolute Error ($|y - \hat{y}|$) | Squared Error ($(y - \hat{y})^2$) |
|---|---|---|---|---|---|
| 1 | 10M | 12M | -2M | 2M | 4M |
| 2 | 50M | 45M | +5M | 5M | 25M |
| 3 | 100M | 105M | -5M | 5M | 25M |
| 4 | 20M | 15M | +5M | 5M | 25M |
| 5 | 500M | 600M | -100M | 100M | 10,000M |

Let's compute the standard metrics using this table:

### 13.1 Mean Absolute Error (MAE)
$$\text{MAE} = \frac{\sum_{i=1}^n \vert y_i - \hat{y}_i\vert}{n} = \frac{2 + 5 + 5 + 5 + 100}{5} = \frac{117}{5} = 23.4\text{ Million PKR}$$
* **What it means:** On average, our predictions are off by 23.4 million PKR. It is highly intuitive, robust to outliers, and easy to explain to stakeholders.

### 13.2 Root Mean Squared Error (RMSE)
$$\text{MSE} = \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{n} = \frac{4 + 25 + 25 + 25 + 10000}{5} = \frac{10079}{5} = 2015.8$$
$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{2015.8} = 44.9\text{ Million PKR}$$
* **What it means:** On average, our predictions are off by 44.9 million PKR. Why is it so much higher than MAE? Because of Project 5. The 100M error was squared to 10,000M, dominating the calculation.
* **Rule of thumb:** Use RMSE when large errors are catastrophic. If underestimating a budget by 100M bankrupts the department, use RMSE to punish the model severely for large mistakes.

### 13.3 R-squared ($R^2$)
$$R^2 = 1 - \frac{\text{Residual Sum of Squares (RSS)}}{\text{Total Sum of Squares (TSS)}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
* **What it means:** The proportion of variance in the target that is explained by the features. 
  * $R^2 = 1$: Perfect predictions.
  * $R^2 = 0$: No better than predicting the training mean.
  * $R^2 < 0$: Worse than predicting the training mean (occurs when a model makes wild predictions that deviate worse than a horizontal line).

### 13.4 Hands-On: Evaluation in Python

Here is how you compute these metrics in python using `scikit-learn`:

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_true = np.array([10.0, 50.0, 100.0, 20.0, 500.0])
y_pred = np.array([12.0, 45.0, 105.0, 15.0, 600.0])

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r2:.4f}")
```

### Day 13 Recap
* **MAE** measures average absolute deviance, scaling linearly; it is robust against single massive outliers.
* **RMSE** measures the square root of squared deviances, placing a heavy quadratic penalty on large errors.
* **R-squared ($R^2$)** scales performance relative to a baseline dummy predictor, showing the fraction of variance explained.

> [!TIP]
> **Quick Check:** A housing agency is predicting rental costs. If underestimating a rent by $500 leads to severe budgeting shortfalls, while overestimating it by $500 is just slightly inconvenient, is MAE or RMSE a better metric for training and optimizing this model? Why?
> *Answer:* RMSE is better because it heavily penalizes larger errors. However, note that MSE/RMSE penalizes *any* large error symmetrically (both positive and negative). If the agency specifically cares about *asymmetric* penalties (underestimating is worse than overestimating), they should actually look into asymmetric loss functions, like the Quantile Loss introduced in Chapter 1 Day 8!

### Day 13 Exercise: Evaluating Skewed Cost Metrics
Load the infrastructure dataset. Fit a simple linear model to predict `actual_cost_million_pkr`.
1. Calculate MAE, RMSE, and $R^2$ on the test set.
2. Sort your test set by `approved_cost_million_pkr` and compute the MAE separately for:
   * "Small scale" projects (approved cost < 22M)
   * "Mega scale" projects (approved cost > 520M)
3. Comment on whether a single global MAE tells the full story for this dataset.

Here is the starter-code skeleton for `evaluate_skewed.py`. Complete the script:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_skewed_data(filepath: str):
    """Loads the infrastructure dataset, splits it, trains a pipeline to predict actual cost, and evaluates metrics overall and on subsets.

    Args:
        filepath (str): Path to the kp_infrastructure_projects.csv dataset.
    """
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Drop identifiers and co-targets
    df = df.drop(columns=["project_id", "actual_duration_months"])
    
    # Target is actual_cost_million_pkr
    X = df.drop(columns=["actual_cost_million_pkr"])
    y = df["actual_cost_million_pkr"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Dynamic columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    
    # Fit model
    pipeline.fit(X_train, y_train)
    
    # Predict on test
    preds = pipeline.predict(X_test)
    
    # Calculate global metrics
    global_mae = mean_absolute_error(y_test, preds)
    global_rmse = np.sqrt(mean_squared_error(y_test, preds))
    global_r2 = r2_score(y_test, preds)
    
    print(f"--- Global Infrastructure Cost Results ---")
    print(f"  MAE : {global_mae:.4f}")
    print(f"  RMSE: {global_rmse:.4f}")
    print(f"  R2  : {global_r2:.4f}\n")
    
    # Create a validation DataFrame for test set
    test_results = pd.DataFrame({
        "approved_cost": X_test["approved_cost_million_pkr"],
        "true_cost": y_test,
        "pred_cost": preds
    })
    
    # Subset 1: Small scale (approved cost < 22M)
    small_scale = test_results[test_results["approved_cost"] < 22]
    small_mae = mean_absolute_error(small_scale["true_cost"], small_scale["pred_cost"])
    print(f"--- Small Scale Projects (Approved Cost < 22M) ---")
    print(f"  Sample count: {len(small_scale)}")
    print(f"  MAE         : {small_mae:.4f}\n")
    
    # Subset 2: Mega scale (approved cost > 520M)
    mega_scale = test_results[test_results["approved_cost"] > 520]
    mega_mae = mean_absolute_error(mega_scale["true_cost"], mega_scale["pred_cost"])
    print(f"--- Mega Scale Projects (Approved Cost > 520M) ---")
    print(f"  Sample count: {len(mega_scale)}")
    print(f"  MAE         : {mega_mae:.4f}\n")

if __name__ == "__main__":
    evaluate_skewed_data("data/kp_infrastructure_projects.csv")
```

---

## Chapter 2 Capstone: The Temporal Forecasting Pipeline

You have learned how to build baselines (Day 11), split data honestly using random, group, and time-based strategies (Day 12), and judge model performance using MAE, RMSE, and $R^2$ (Day 13). The Capstone asks you to combine all these concepts by building a model for the **KP Infrastructure Projects Dataset** to predict `actual_duration_months`.

Complete the following tasks in a single script `duration_capstone.py`:
1. **Load and Prepare:** Load `data/kp_infrastructure_projects.csv`. Drop `project_id` and the co-target `actual_cost_million_pkr` to prevent target leakage.
2. **Temporal Split:** Split the data chronologically using `year_of_initiation`. Use projects initiated in years up to 2021 for training, year 2022 for validation, and years 2023–2025 for testing. Print the sizes of each split.
3. **Build the Pipeline:** Set up a preprocessing pipeline:
   * Standard scale numeric features (e.g., `approved_cost_million_pkr`, `estimated_duration_months`).
   * One-hot encode categorical features (e.g., `project_sector`, `district`, `terrain_complexity`, `project_scale`).
4. **Train and Compare:** Fit three models on the training set and evaluate them on both the validation set and test set:
   * `DummyRegressor(strategy='mean')`
   * `LinearRegression()`
   * `DecisionTreeRegressor(max_depth=6, random_state=42)`
5. **Report & Analyze:**
   * Print a comparison table of MAE, RMSE, and $R^2$ on the validation and test sets.
   * Answer: Why is a temporal split more honest here than a random split?
   * Answer: Did the decision tree outperform the linear regression? Why or why not?

Here is the starter-code skeleton for `duration_capstone.py`:

```python
import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_temporal_split(filepath: str):
    """Loads the infrastructure dataset, drops leakages/identifiers, and splits chronologically.

    Args:
        filepath (str): Path to the infrastructure projects CSV file.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test) splits.
    """
    df = pd.read_csv(filepath)
    
    # Drop identifiers and cost leakage
    df = df.drop(columns=["project_id", "actual_cost_million_pkr"])
    
    # Define splits
    train_df = df[df["year_of_initiation"] <= 2021]
    val_df = df[df["year_of_initiation"] == 2022]
    test_df = df[df["year_of_initiation"] >= 2023]
    
    # Separate features and target (actual_duration_months)
    # Drop year_of_initiation as a feature so it doesn't leak time trend
    y_train = train_df["actual_duration_months"]
    X_train = train_df.drop(columns=["actual_duration_months", "year_of_initiation"])
    
    y_val = val_df["actual_duration_months"]
    X_val = val_df.drop(columns=["actual_duration_months", "year_of_initiation"])
    
    y_test = test_df["actual_duration_months"]
    X_test = test_df.drop(columns=["actual_duration_months", "year_of_initiation"])
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_pipeline(X_train: pd.DataFrame, regressor) -> Pipeline:
    """Builds a scikit-learn pipeline wrapping preprocessing and the given regressor.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        regressor (object): scikit-learn regressor.

    Returns:
        Pipeline: Preconfigured scikit-learn Pipeline.
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

def evaluate(pipeline: Pipeline, X_train, X_val, X_test, y_train, y_val, y_test, name: str):
    """Fits the model and evaluates on validation and test splits.

    Args:
        pipeline (Pipeline): Preconfigured Pipeline.
        X_train, X_val, X_test: Feature splits.
        y_train, y_val, y_test: Target splits.
        name (str): Model name.
    """
    pipeline.fit(X_train, y_train)
    
    val_preds = pipeline.predict(X_val)
    test_preds = pipeline.predict(X_test)
    
    print(f"=== {name} ===")
    print(f"  [Validation] MAE: {mean_absolute_error(y_val, val_preds):.4f} | R2: {r2_score(y_val, val_preds):.4f}")
    print(f"  [Test]       MAE: {mean_absolute_error(y_test, test_preds):.4f} | R2: {r2_score(y_test, test_preds):.4f}\n")

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_temporal_split("data/kp_infrastructure_projects.csv")
    print(f"Split sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}\n")
    
    models = {
        "Dummy Mean Regressor": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor (depth=6)": DecisionTreeRegressor(max_depth=6, random_state=42)
    }
    
    for name, model in models.items():
        pipeline = build_pipeline(X_train, model)
        evaluate(pipeline, X_train, X_val, X_test, y_train, y_val, y_test, name)
```

---

## Chapter 2 Recap: Setting the Standards

* **Baseline is the Floor:** Never optimize or report machine learning results without fitting a Dummy Regressor first. Beat the mean, or go home.
* **Memorization is not Learning:** Standard train-test splits evaluate generic variance. Group splits (GroupKFold) evaluate spatial variance (geographic generalization). Temporal splits evaluate chronological forecasting and prevent time-based target leakage.
* **Leakage is cheating:** Testing on data that was used in any step of preprocessing (like global fitting of a StandardScaler) or feature engineering yields overly optimistic, false metrics.
* **MAE vs RMSE:** MAE is robust and scale-linear. RMSE is sensitive to large deviations due to quadratic squaring, making it ideal when outlier errors carry severe business costs.

---

## Formula Cheat Sheet

| Concept | Formula / Rule | Day |
|---|---|---|
| Dummy Prediction (Mean) | $\hat{y}_i = \bar{y} = \frac{1}{n}\sum_{j=1}^n y_j$ | 11 |
| Linear Regression Prediction | $\hat{y}_i = x_i^T\beta = \beta_0 + \beta_1 x_{i1} + \dots$ | 11 |
| Decision Tree Region Prediction | $\hat{y}_i = \frac{1}{\vert R_m\vert} \sum_{j \in R_m} y_j$ for $x_i \in R_m$ | 11 |
| Mean Absolute Error (MAE) | $\text{MAE} = \frac{1}{n}\sum_{i=1}^n \vert y_i - \hat{y}_i\vert$ | 13 |
| Mean Squared Error (MSE) | $\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$ | 13 |
| Root Mean Squared Error (RMSE) | $\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$ | 13 |
| R-squared ($R^2$) | $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | 13 |

---

## References

1. **scikit-learn Pipelines:** [Pipeline Tutorial](https://scikit-learn.org/stable/modules/compose.html) - Standard compose API guide.
2. **Cross-Validation Guide:** [sklearn CV User Guide](https://scikit-learn.org/stable/modules/cross_validation.html) - Details on GroupKFold, KFold, and TimeSeriesSplit.
3. **Evaluating Regression Models:** [Evaluation Metrics Overview](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) - Standard API documentation on regression losses.
4. **Generalization & Splitting Theory:** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Chapter 7 (Model Assessment and Selection).
