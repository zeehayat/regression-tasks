# Chapter 1 Companion Guide: From a Question to Ordinary Least Squares

## A beginner-first guide to the Python, mathematics, statistics, and judgement behind Chapter 1

This guide accompanies `chapter1.md`. It assumes that you have never programmed in Python, have not studied statistics, and may be uncomfortable with algebra. Nothing important is treated as “obvious.”

The chapter's running example is a cost estimator for fictional microhydro power (MHP) projects in Khyber Pakhtunkhwa (KP). The goal is not merely to produce a number. It is to understand the complete chain:

1. ask a precise question;
2. represent the data correctly;
3. form linear predictions;
4. measure prediction errors;
5. find the ordinary least-squares solution;
6. check when that solution is meaningful; and
7. communicate what the result can and cannot support.

You do not need to memorise every formula on the first reading. Aim to translate each formula into a sentence and each line of code into a small operation you understand.

---

## How to use this guide

For each day in Chapter 1:

1. Read the corresponding section here first.
2. Type the small examples yourself. Copying and pasting hides useful mistakes.
3. Read the main chapter and run its longer code blocks.
4. Change one number and predict what will happen before rerunning the code.
5. Complete the “Check your understanding” questions without looking back.
6. Keep a notebook containing new words, errors you encountered, and how you fixed them.

When an equation feels difficult, use this three-part translation:

- **objects:** What are the numbers, vectors, or matrices?
- **operation:** What arithmetic is performed?
- **meaning:** What does the result say about an MHP project?

---

## 0. Foundations you need before Day 1

### 0.1 The running case and its units

One **observation** is one completed MHP project. Possible input fields include:

| Field | Symbol | Unit | Plain-language meaning |
|---|---:|---:|---|
| Number of projects | $n$ | projects | Number of rows in the dataset |
| Number of features | $p$ | features | Number of input columns before an intercept is added |
| Cable length | $x_1$ | km | Length of distribution cable |
| Hydraulic head | $x_2$ | m | Vertical fall used to drive the turbine |
| Road distance | $x_3$ | km | Distance from an all-weather road |
| Terrain difficulty | $x_4$ | index 1–5 | An ordered assessment of access difficulty |
| Planned capacity | $x_5$ | kW | Intended electricity-generating capacity |
| Actual project cost | $y$ | million PKR | Outcome that the model will estimate |

Units are part of the meaning. If cost is measured in million PKR and road distance in km, a road-distance coefficient is measured in **million PKR per km**. A coefficient without units is incomplete.

The chapter's projects are synthetic. They are suitable for learning but not for evaluating a real district, contractor, organisation, or community.

### 0.2 Installing and running Python

Python executes instructions from top to bottom. Chapter 1 uses Python 3.11 or later, NumPy for numerical arrays, and Matplotlib for plots.

In a terminal:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib
```

On Windows PowerShell, activation is usually:

```powershell
.venv\Scripts\Activate.ps1
```

The virtual environment keeps this project's packages separate from other Python projects. You can run code in a `.py` file, a Jupyter notebook, or an editor's interactive Python window.

### 0.3 A small Python survival kit

#### Values, names, and arithmetic

```python
project_name = "MHP-CH-003"  # text, called a string
road_distance = 18.5         # decimal number, called a float
terrain_index = 4            # whole number, called an integer
is_complete = True           # Boolean: True or False

base_cost = 8.0
cost_per_km = 4.0
prediction = base_cost + cost_per_km * road_distance
print(prediction)
```

`=` assigns a value to a name. It does not mean “is mathematically equal forever.” `#` begins a comment that Python ignores. Common arithmetic operators are `+`, `-`, `*`, `/`, and `**` for powers.

#### Lists and NumPy arrays

```python
import numpy as np

cost_list = [19.0, 23.0, 30.0]
cost_array = np.array(cost_list, dtype=float)

print(cost_array[0])      # first value; Python counts from zero
print(cost_array.shape)   # (3,)
print(cost_array.mean())  # arithmetic mean
```

A Python list is general-purpose. A NumPy array is designed for numerical work and has a fixed shape. `dtype=float` asks NumPy to store floating-point numbers.

#### Functions

```python
def predict_cost(distance_km, base=8.0, rate=4.0):
    """Return a simple cost prediction in million PKR."""
    return base + rate * distance_km

estimate = predict_cost(3.0)
```

`def` creates a reusable function. Indentation defines the function body. Inputs are called arguments or parameters in Python terminology; do not confuse these with fitted statistical parameters. `return` sends a result back to the caller.

#### Conditions and errors

```python
def require_positive(value):
    if value <= 0:
        raise ValueError("value must be positive")
    return value
```

An exception stops an invalid calculation near its source. A clear early error is better than a plausible but wrong result later.

#### Loops

```python
total = 0.0
for value in [2.0, 3.0, 4.0]:
    total += value
print(total)  # 9.0
```

The loop visits each item. `total += value` is shorthand for `total = total + value`.

#### Classes, objects, and dataclasses

The chapter builds an `MHPCostEstimator` class. A **class** is a blueprint; an **object** is one created instance. `self` means “this particular object.” A method is a function attached to a class.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class AnalysisContract:
    purpose: str
    target_name: str

contract = AnalysisContract(
    purpose="prediction",
    target_name="actual_cost_m_pkr",
)
```

The type annotations describe expected values but do not by themselves validate everything. `frozen=True` prevents accidental changes after creation. Names ending in `_`, such as `parameters_`, conventionally mean values learned during fitting.

#### Assertions and floating-point comparisons

```python
result = 0.1 + 0.2
assert np.isclose(result, 0.3)
```

An `assert` is an executable claim. Decimal arithmetic on a computer has tiny rounding errors, so use `np.isclose` or `np.allclose` for calculated floating-point results instead of exact `==`.

### 0.4 The mathematical language used in the chapter

#### Symbols and subscripts

- $x$ is usually an input feature.
- $y$ is the observed target.
- $\hat{y}$, read “y-hat,” is a prediction.
- $\beta$, read “beta,” is a parameter or vector of parameters.
- $i$ identifies a row or project.
- $j$ identifies a feature column.
- $x_{ij}$ means feature $j$ for project $i$.

A hat means “estimated from data.” Thus $\hat{\beta}$ is the fitted parameter value, not the unknown population value $\beta$.

#### Sums, means, powers, roots, and absolute values

The symbol $\sum$ means “add a sequence of values.” For example:

$$
\sum_{i=1}^{3}x_i=x_1+x_2+x_3.
$$

The arithmetic mean is:

$$
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i.
$$

$x^2$ means $x$ multiplied by itself. $\sqrt{x}$ is the nonnegative square root. $|x|$ is absolute value, the distance from zero: $|-3|=3$.

#### Scalars, vectors, matrices, and tensors

- A **scalar** is one number, such as one project's cost.
- A **vector** is an ordered one-dimensional collection, such as all project costs.
- A **matrix** is a rectangular two-dimensional collection, such as projects by features.
- A **tensor** is the general term for an array with any number of axes.

Do not confuse the number of tensor axes with **matrix rank**, which counts independent directions and is introduced on Day 5.

#### Dot products and matrix multiplication

For vectors $a=[a_1,a_2]$ and $b=[b_1,b_2]$:

$$
a^Tb=a_1b_1+a_2b_2.
$$

NumPy writes this as `a @ b`. A matrix-vector product performs one dot product for every matrix row.

#### Transpose and length

$X^T$ is the transpose of $X$: rows become columns. In NumPy it is `X.T`.

The Euclidean length of a vector is:

$$
\lVert v\rVert_2=\sqrt{v_1^2+v_2^2+\cdots+v_n^2}.
$$

NumPy computes it with `np.linalg.norm(v)`.

#### Functions, derivatives, gradients, and Hessians

A mathematical function maps an input to an output. A derivative measures the local rate at which an output changes when one input changes. At the bottom of a smooth bowl, the slope is zero.

When a function has several inputs, a **partial derivative** changes one input while holding the others fixed. A **gradient** collects all partial derivatives into a vector. A **Hessian** collects all second partial derivatives into a matrix and describes curvature.

This intuition is enough to begin Day 5; the full derivation is developed there.

---

# Day 1 — Decide what question regression should answer

## 1.1 Regression is a mechanism, not a purpose

Regression constructs a numerical relationship between a target $y$ and observed features $X$. The same fitted equation can be used for three different jobs:

| Job | Core question | Evidence of success | Main danger |
|---|---|---|---|
| Prediction | What outcome should we expect for a new project? | Low error on genuinely new projects | Memorising old data or failing when conditions change |
| Explanation | How is one feature associated with cost, conditional on included features? | Stable, interpretable estimates with uncertainty | Omitted or correlated variables distort interpretation |
| Causal inference | What would change if we intervened? | A credible comparison with the relevant counterfactual | Treating association as the effect of an action |

The calculation does not decide which question you meant. The purpose determines what data, design, validation, and wording are defensible.

## 1.2 Prediction

A prediction rule can be written as:

$$
\hat{y}_i=f(x_{i1},x_{i2},\ldots,x_{ip}).
$$

For project $i$, the function $f$ converts its feature values into an estimated cost $\hat{y}_i$.

A useful predictor does not need every feature to be causal. River crossings may help predict transport difficulty even if “removing one crossing” is not a meaningful intervention. Prediction should be judged on projects that were not used to fit the model. Training accuracy alone may reward memorisation.

## 1.3 Explanation and conditional association

Consider:

$$
\hat{y}_i=\beta_0+\beta_1\,\text{road distance}_i
+\beta_2\,\text{capacity}_i.
$$

$\beta_1$ describes the fitted cost difference associated with one additional kilometre of road distance **while the included capacity value is held fixed**. This is a conditional association.

“Holding included features fixed” does not control for omitted features, measurement errors, or structural differences between districts. Explanatory analysis also needs uncertainty. The same numerical coefficient is less persuasive when estimated from 6 projects than from 600 comparable projects.

## 1.4 Causation, interventions, and counterfactuals

A causal question compares outcomes under different actions. Suppose the action is completing an access road before construction. For one project, the causal effect would compare:

$$
Y_i(\text{road completed})-Y_i(\text{road not completed}).
$$

Only one of these outcomes can usually be observed for the same project. The unobserved alternative is the **counterfactual**. Causal inference needs a study design and assumptions that make another comparison stand in for this missing outcome.

A regression coefficient is therefore not automatically a causal effect. A causal statement must define:

- the intervention;
- the affected population;
- the outcome and time horizon;
- the counterfactual alternative; and
- why the comparison identifies that effect.

## 1.5 Confounding, mediators, and colliders

A **confounder** is a common cause of both an exposure or decision and an outcome. Rugged terrain may make helicopter transport more likely and independently make a project more expensive. Ignoring terrain can make helicopters appear responsible for high cost even if they reduce cost relative to weeks of manual carriage.

In a causal diagram:

```text
Rugged terrain ──> Helicopter use
       │
       └─────────> Project cost
```

Not every available variable should be controlled:

- A **mediator** lies on the pathway from action to outcome. Controlling it may remove part of the effect being studied.
- A **collider** is caused by two other variables. Conditioning on it can create a misleading association.

The beginner-safe rule is not “control for everything.” It is “state the causal story and justify the adjustment set.”

## 1.6 Simpson's paradox

Simpson's paradox occurs when an aggregated trend differs from, or reverses, trends within meaningful groups.

| Terrain | Cable km | Cost, million PKR |
|---|---:|---:|
| Accessible | 4 | 18 |
| Accessible | 5 | 20 |
| Accessible | 6 | 22 |
| Accessible | 7 | 24 |
| Remote | 1 | 30 |
| Remote | 2 | 33 |
| Remote | 3 | 36 |
| Remote | 4 | 39 |

Within accessible projects, cost rises by about 2 million PKR per km. Within remote projects, it rises by about 3 million PKR per km. Yet the combined slope is about -2 because costly remote projects mostly have shorter cables.

For a line with one feature and an intercept:

$$
\hat{\beta}_1=
\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}
{\sum_i(x_i-\bar{x})^2}.
$$

The numerator measures whether $x$ and $y$ move together. The denominator measures variation in $x$. The numerical reversal does not by itself identify a causal relationship; subject-matter knowledge explains why aggregation is misleading.

## 1.7 Write an estimand before code

An **estimand** is the precise quantity an analysis intends to estimate.

- Prediction: “The expected final cost, in million PKR at the stated price basis, of an approved project using design-stage information.”
- Explanation: “The adjusted difference in final cost associated with one additional kilometre from an all-weather road, conditional on planned capacity and measured terrain.”
- Causation: “The average change in final cost caused by completing an access road before civil works among approved remote projects.”

Also state the unit of observation, intended use, prediction time, target units, and decision-maker. If these are vague, a more complicated model will not fix the question.

## 1.8 The `AnalysisContract`

The chapter begins the application with an immutable dataclass:

```python
from dataclasses import dataclass
from typing import Literal

AnalysisPurpose = Literal["prediction", "explanation", "causation"]

@dataclass(frozen=True)
class AnalysisContract:
    purpose: AnalysisPurpose
    target_name: str
    target_unit: str
    unit_of_observation: str
    intended_use: str

    def __post_init__(self):
        if self.purpose not in {"prediction", "explanation", "causation"}:
            raise ValueError("invalid purpose")
        for name, value in vars(self).items():
            if not str(value).strip():
                raise ValueError(f"{name} cannot be empty")
```

`Literal` documents the allowed choices. `__post_init__` runs after creation and validates them. `vars(self)` provides the object's fields as name-value pairs. `.strip()` removes surrounding whitespace, so a string containing only spaces is rejected.

Deliberately try `purpose="forecasting"` and an empty intended use. Predict the errors first. This demonstrates a general engineering principle: make invalid states fail clearly and early.

## Day 1 check

You should now be able to answer:

1. Why can a feature help prediction without being causal?
2. What does “holding the other included features fixed” mean?
3. What is the missing counterfactual in an access-road study?
4. How can rugged terrain confound a transport-cost comparison?
5. Why can an aggregated slope have the opposite sign from group-specific slopes?

Key idea: a coefficient is a causal effect only when the design and assumptions make the comparison equivalent to the stated intervention and counterfactual.

---

# Day 2 — Give data the correct roles, shapes, units, and provenance

## 2.1 Regression vocabulary

| Term | Meaning | MHP example |
|---|---|---|
| Observation | One unit represented by one row | One completed project |
| Feature | An input used to form a prediction | Planned capacity |
| Target | The outcome to estimate | Actual project cost |
| Parameter | A number learned during fitting | Road-distance coefficient |
| Hyperparameter/design choice | A setting chosen outside fitting | Whether to include an intercept |
| Prediction | Model output for one observation | 42 million PKR |
| Residual | Actual target minus prediction | $45-42=3$ million PKR |
| Loss | Rule used to penalise error during fitting | Squared residual |
| Metric | Summary used to report performance | RMSE |
| Estimator | Procedure that learns parameters | OLS |
| Fitted model | Estimator plus learned values | OLS plus fitted intercept and slopes |

The distinction between a parameter and hyperparameter is procedural: fitting learns a parameter from data; the analyst chooses a hyperparameter or modeling design.

## 2.2 Arrays and shape notation

The central objects are:

$$
X\in\mathbb{R}^{n\times p},\qquad
y\in\mathbb{R}^{n},\qquad
\beta\in\mathbb{R}^{p}.
$$

$\mathbb{R}$ means real-valued numbers. $X$ has $n$ project rows and $p$ feature columns. $y$ has one target per row. $\beta$ has one weight per feature. After adding an intercept, $X$ and $\beta$ each gain one column or entry.

For three projects and two features:

$$
X=
\begin{bmatrix}
12&100\\
25&150\\
8&80
\end{bmatrix},\qquad
y=\begin{bmatrix}28\\44\\22\end{bmatrix}.
$$

Row 2 means that the second project has road distance 25 km, planned capacity 150 kW, and actual cost 44 million PKR.

## 2.3 NumPy shapes are not decoration

```python
a = np.array([28.0, 44.0, 22.0])          # (3,)
b = np.array([[28.0], [44.0], [22.0]])    # (3, 1)
c = np.array([[28.0, 44.0, 22.0]])        # (1, 3)
```

- `(3,)` is a one-dimensional array.
- `(3, 1)` is a two-dimensional column matrix.
- `(1, 3)` is a two-dimensional row matrix.

They contain the same numbers but behave differently during multiplication and broadcasting. In this application, `y` uses shape `(n,)` and `X` always becomes 2D.

Useful shape tools:

```python
x = np.array([1.0, 2.0, 3.0])
X_one_feature = x.reshape(-1, 1)  # infer rows, make one column
X_two_features = np.column_stack([x, x ** 2])
flat_again = X_one_feature.reshape(-1)
```

## 2.4 The design matrix and row alignment

The feature matrix supplied to regression is the **design matrix**. The word “design” means the arrangement of predictor columns; it does not imply an experiment.

The $i$th row of $X$ and the $i$th value of $y$ must describe the same project. Equal row counts are necessary but insufficient. If `X` is sorted by district and `y` by project ID, the program can run successfully while learning nonsense.

Preserve `project_id` while assembling and checking data, even if it is not used as a numerical feature. Join or sort `X` and `y` together, then assert identifiers align.

## 2.5 A data dictionary belongs to the model

A data dictionary should record:

- field name and plain-language definition;
- data type;
- unit or allowed categories;
- when the value becomes available;
- who measured it and how;
- missing-value rule;
- valid range; and
- price year or price basis for monetary targets.

This information determines whether two rows mean the same thing and whether a feature is legitimate at prediction time.

### Target leakage

**Target leakage** occurs when training uses information that would not legitimately be available when a real prediction is made. “Number of contract amendments” may predict final cost well but is unavailable during initial budgeting and may be caused by the overrun itself.

Leakage creates impressive-looking training or test results if the split does not reproduce the real timeline. Prevent it by stating the prediction time first, then checking every feature's availability and provenance.

## 2.6 Parsing nested JSON

JSON represents nested records with objects and lists. Python's `json.loads` converts JSON text into dictionaries.

```python
import json

record = json.loads(raw_json)
distance = record["design"]["road_distance_km"]

X_one = np.array([[
    record["design"]["road_distance_km"],
    record["design"]["planned_capacity_kw"],
    record["design"]["terrain_index"],
]], dtype=float)

y_one = np.array([
    record["completion"]["actual_cost_m_pkr"]
], dtype=float)

assert X_one.shape == (1, 3)
assert y_one.shape == (1,)
```

The double opening bracket means one row containing three columns. Assertions make the intended mathematical shapes executable.

## 2.7 Matrix-vector multiplication is repeated weighted addition

Without an intercept:

$$
\hat{y}_i=\sum_{j=1}^{p}x_{ij}\beta_j.
$$

For each project, multiply every feature by its matching coefficient, then add the products.

```python
y_hat_loop = np.zeros(X.shape[0])
for i in range(X.shape[0]):
    total = 0.0
    for j in range(X.shape[1]):
        total += X[i, j] * beta[j]
    y_hat_loop[i] = total

y_hat_matrix = X @ beta
assert np.allclose(y_hat_loop, y_hat_matrix)
```

`X.shape[0]` is the number of rows and `X.shape[1]` the number of columns. The compact `@` expression performs the same nested-loop calculation.

## 2.8 Input validation, line by line

The chapter's `_validate_inputs` method:

1. converts input to floating-point NumPy arrays;
2. turns 1D training `X` into one feature column;
3. rejects arrays that are not 2D;
4. rejects empty rows or columns;
5. rejects `NaN` and infinity;
6. flattens a one-column target to `(n,)`;
7. rejects other target shapes; and
8. checks equal row counts.

Important expressions:

```python
X = np.asarray(X, dtype=float)
X = X.reshape(-1, 1)
np.isfinite(X).all()
```

`np.asarray` accepts lists or arrays. `-1` asks NumPy to infer the needed dimension. `np.isfinite` is false for `NaN`, positive infinity, and negative infinity; `.all()` requires every entry to pass.

Validation cannot detect correctly shaped but misaligned projects. That requires identifiers and data-provenance checks before numerical arrays discard labels.

## Day 2 failure laboratory

Predict the result of each case:

- 10 feature rows and 8 targets: row mismatch;
- a feature containing `np.nan`: non-finite input;
- shape `(0, 2)`: no observations;
- shape `(2, 2, 2)`: too many dimensions;
- equal-length `X` and reversed `y`: no shape error, but corrupted meaning.

For 70 projects and 5 raw features, `X.shape == (70, 5)` and `y.shape == (70,)`. After adding an intercept, the design matrix is `(70, 6)` and the parameter vector is `(6,)`.

---

# Day 3 — Build linear predictions

## 3.1 From words to an equation

“Start with 8 million PKR and add 4 million PKR per kilometre of cable” becomes:

$$
\hat{y}_i=8+4x_i.
$$

The general one-feature model is:

$$
\hat{y}_i=\beta_0+\beta_1x_i.
$$

- $\beta_0$ is the **intercept**, the prediction when $x=0$.
- $\beta_1$ is the **slope**, the predicted change for one unit more of $x$.

If cost is million PKR and cable is km, the slope is million PKR per km. The intercept can improve predictions without describing a physically possible zero-feature project. It is an algebraic baseline, and interpreting it literally may require extrapolating beyond observed data.

## 3.2 Several features are still one weighted sum

$$
\hat{y}_i=\beta_0
+\beta_1\,\text{road distance}_i
+\beta_2\,\text{capacity}_i
+\beta_3\,\text{terrain}_i.
$$

For one project, this is a dot product. For all projects simultaneously:

$$
\hat{y}=X\beta.
$$

“Linear” means linear in the parameters: predictions are sums of parameter-weighted columns. A model may still include transformed features such as $x^2$, provided their coefficients enter linearly.

## 3.3 Why a column of ones creates an intercept

For three projects and one feature:

$$
X=
\begin{bmatrix}
1&x_1\\
1&x_2\\
1&x_3
\end{bmatrix},\qquad
\beta=
\begin{bmatrix}\beta_0\\\beta_1\end{bmatrix}.
$$

Multiplication gives:

$$
X\beta=
\begin{bmatrix}
\beta_0+x_1\beta_1\\
\beta_0+x_2\beta_1\\
\beta_0+x_3\beta_1
\end{bmatrix}.
$$

Every prediction receives the same $\beta_0$ because every first-column value is 1. Without that column, the fitted line is forced through $(0,0)$. With it, the model is technically affine—a linear combination plus translation—but is conventionally called linear regression.

In NumPy:

```python
X = np.column_stack([np.ones(x.size), x])
y_hat = X @ beta
```

A column of twos would estimate $2\beta_0$ as the common offset, so its coefficient would be half the conventional intercept. Ones give the coefficient itself the natural intercept meaning.

## 3.4 Shape algebra

If an intercept has been added:

$$
X\in\mathbb{R}^{n\times(p+1)},\qquad
\beta\in\mathbb{R}^{p+1},\qquad
X\beta\in\mathbb{R}^{n}.
$$

The inner dimensions must match:

$$
(n\times(p+1))(p+1\times1)=n\times1.
$$

Always check shapes before values. If `X.shape == (40, 6)` and `beta.shape == (5,)`, multiplication is impossible because six design columns need six parameters.

### Mathematics-to-NumPy map

| Mathematics | Meaning | NumPy |
|---|---|---|
| $X\beta$ | Matrix-vector product | `X @ beta` |
| $X^T$ | Transpose | `X.T` |
| $x^Ty$ | Dot product | `x @ y` |
| $I$ | Identity matrix | `np.eye(p)` |
| $\mathbf{1}$ | Ones vector | `np.ones(n)` |
| $\lVert v\rVert_2$ | Euclidean length | `np.linalg.norm(v)` |

## 3.5 Two views of matrix multiplication

**Row view:** each row of $X$ is one project, and its dot product with $\beta$ makes one prediction.

**Column view:** if columns are $x_0,x_1,\ldots,x_p$, then:

$$
X\beta=\beta_0x_0+\beta_1x_1+\cdots+\beta_px_p.
$$

The columns are building blocks; the coefficients decide how much of each to combine. All possible combinations form the **column space** of $X$. This column view prepares the geometry of Day 4.

Matrices can also represent rotations, stretches, reflections, and shears. That transformation viewpoint is useful, but in regression the immediate meaning of $X\beta$ is a weighted construction of a prediction vector.

## 3.6 With and without an intercept

For a one-feature fit through the origin:

$$
\hat{\beta}=\frac{x^Ty}{x^Tx}.
$$

This model must predict zero when $x=0$. A model with an intercept can translate vertically and will often fit a nonzero baseline better. Plotting both lines reveals whether forcing the origin creates systematic errors.

The chapter uses:

```python
beta = np.linalg.solve(X.T @ X, X.T @ y)
```

This solves the linear system directly. Day 5 explains why application code should avoid `np.linalg.inv(X.T @ X) @ X.T @ y`.

## 3.7 Scaling and unit conversions

If kilometres become metres:

$$
x_{m}=1000x_{km}.
$$

To preserve predictions:

$$
\beta_m=\frac{\beta_{km}}{1000}.
$$

Then $\beta_mx_m=\beta_{km}x_{km}$. Changing feature units changes the coefficient inversely, not the prediction. Changing the target from million PKR to PKR multiplies the intercept, slopes, residuals, RMSE, and MAE by one million; MSE and SSR multiply by one million squared.

Coefficient magnitude alone does not measure importance when features have different units or scales.

## 3.8 Prediction code and shape ambiguity

The chapter's `predict` method must:

1. reject prediction before fitting;
2. convert a single new project from `(p,)` to `(1, p)`;
3. validate finite values and a 2D shape;
4. require the same number and order of features used in fitting;
5. add the intercept; and
6. return `X_design @ parameters_`.

There is a context-dependent ambiguity:

- during fitting, a 1D `X` means many observations of one feature;
- during prediction, a 1D `X` means one observation with many features.

Explicit 2D input is safest: `[[20.0, 170.0, 4.0]]` clearly means one project with three features.

## 3.9 Broadcasting is not matrix multiplication

```python
correct = X @ beta
elementwise = X * beta
```

`@` returns one dot product per row, so its result is a prediction vector. `*` performs elementwise multiplication. NumPy may **broadcast** `beta` across every row, returning a matrix of unadded products. It can run without an error while answering the wrong question.

This is a crucial debugging lesson: code that runs is not necessarily mathematically correct.

## Day 3 check

Explain both statements in your own words:

- `X @ beta` is one dot product per project.
- `X @ beta` is a weighted combination of the columns of `X`.

Then state the shapes of the raw features, design matrix, coefficient vector, and predictions for 40 projects and 5 features.

---

# Day 4 — Residuals, loss, metrics, and OLS geometry

## 4.1 Residuals

For project $i$:

$$
e_i=y_i-\hat{y}_i.
$$

- $e_i>0$: the actual cost is higher; the model underpredicted.
- $e_i<0$: the actual cost is lower; the model overpredicted.
- $e_i=0$: prediction equals the observed target.

For all projects:

$$
e=y-\hat{y}=y-X\beta.
$$

This guide follows **actual minus predicted**. Some software uses the opposite sign, so always check the convention.

A residual is calculated after fitting from observed data. It is not exactly the same as an unobservable population error term, even though introductory writing sometimes uses the words interchangeably.

## 4.2 Why OLS squares residuals

The **sum of squared residuals** (SSR), also called RSS or SSE, is:

$$
SSR(\beta)=\sum_{i=1}^{n}(y_i-x_i^T\beta)^2
=(y-X\beta)^T(y-X\beta)=e^Te.
$$

Squaring:

1. stops positive and negative residuals cancelling;
2. penalises large misses much more heavily;
3. produces a smooth differentiable objective; and
4. turns fitting into a squared Euclidean-distance problem.

These are properties, not proof that squared loss is always institutionally correct. One data-entry error can dominate the result.

## 4.3 Work the metrics by hand

Suppose:

$$
y=[3,5],\qquad \hat{y}=[4,3].
$$

Residuals are $e=[-1,2]$.

$$
SSR=(-1)^2+2^2=5,
$$

$$
MSE=\frac{5}{2}=2.5,
$$

$$
RMSE=\sqrt{2.5}\approx1.581,
$$

$$
MAE=\frac{|-1|+|2|}{2}=1.5.
$$

| Metric | Formula | Units | What it communicates |
|---|---|---|---|
| SSR | $\sum e_i^2$ | squared target units | Total squared fitting objective |
| MSE | $SSR/n$ | squared target units | Average squared loss |
| RMSE | $\sqrt{MSE}$ | target units | Squared-loss summary in communicable units |
| MAE | $\sum|e_i|/n$ | target units | Average absolute miss |

RMSE reacts more strongly than MAE to large residuals. Neither is automatically “better.” Select and interpret metrics according to the decision and inspect unusual observations.

Training MSE here divides by $n$. An estimate of residual variance may divide by $n-p-1$ when an intercept and $p$ raw features are fitted. These are different quantities; always state the denominator.

## 4.4 OLS searches inside the column space

The target vector $y$ has one coordinate per project, so it lies in $\mathbb{R}^n$. Every possible prediction $X\beta$ is a weighted combination of design columns and therefore lies in the column space of $X$.

OLS cannot choose any vector in $\mathbb{R}^n$. It chooses the attainable vector $\hat{y}=X\hat{\beta}$ closest to $y$ under squared Euclidean distance. This is an **orthogonal projection**.

The geometric picture is:

- column space: a line, plane, or higher-dimensional flat surface of attainable predictions;
- $y$: the observed target vector, usually outside that surface;
- $\hat{y}$: the closest point on the surface;
- $e=y-\hat{y}$: the perpendicular arrow from prediction to observation.

## 4.5 The Pythagorean minimum proof

Let $q$ be any other attainable prediction. Define:

$$
e=y-\hat{y},\qquad d=\hat{y}-q.
$$

Then:

$$
y-q=e+d.
$$

$e$ is perpendicular to the column space, while $d$ lies inside it, so $e^Td=0$. Expanding:

$$
\begin{aligned}
\lVert y-q\rVert_2^2
&=(e+d)^T(e+d)\\
&=e^Te+e^Td+d^Te+d^Td\\
&=\lVert e\rVert_2^2+\lVert d\rVert_2^2\\
&\geq\lVert e\rVert_2^2.
\end{aligned}
$$

The other prediction's squared distance equals the OLS distance plus another nonnegative squared length. Therefore it cannot be smaller.

## 4.6 Orthogonality and the normal equations

Perpendicular to every design column means:

$$
X^Te=0.
$$

Substitute $e=y-X\hat{\beta}$:

$$
X^T(y-X\hat{\beta})=0,
$$

so:

$$
X^TX\hat{\beta}=X^Ty.
$$

These are the **normal equations**. “Normal” means perpendicular here; it has nothing to do with the normal probability distribution.

If the design includes a ones column, one orthogonality equation is:

$$
\mathbf{1}^Te=\sum_i e_i=0.
$$

Thus training residuals sum to approximately zero when an intercept is included. This does not imply a good fit: residuals of `[-100, 100]` sum to zero but are enormous.

## 4.7 Verify geometry in code

```python
beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
y_hat = X @ beta_hat
e = y - y_hat

assert np.allclose(X.T @ e, 0.0, atol=1e-10)
assert np.isclose(e.sum(), 0.0, atol=1e-10)

beta_other = beta_hat + np.array([2.0, -1.0])
q = X @ beta_other
d = y_hat - q

assert np.isclose(e @ d, 0.0, atol=1e-10)
assert np.isclose(
    np.linalg.norm(y - q) ** 2,
    np.linalg.norm(e) ** 2 + np.linalg.norm(d) ** 2,
)
```

`atol` is an absolute tolerance that allows tiny floating-point error. The chapter's three-dimensional plot makes the same relationships visible when there are three observations and two independent design columns.

## 4.8 Metrics in the estimator

`_metrics_from_predictions` should flatten both arrays, require equal shapes, calculate `residuals = y - y_hat`, and return Python floats for SSR, MSE, RMSE, and MAE.

`evaluate(X, y)` validates supplied data, calls `predict`, then computes metrics. The method does not call them “test metrics” because an array's provenance—training, validation, or test—is the caller's responsibility.

## 4.9 Outliers and data investigation

Changing 26 to 260 creates a residual hundreds of times larger than the others. Its square can dominate SSR and RMSE and pull fitted coefficients toward it. MAE changes less because it does not square the miss.

Do not immediately delete an outlier or declare MAE superior. Ask:

- Is it a data-entry error?
- Is it a genuine extreme project?
- Does it belong to a different project type?
- Is the chosen loss aligned with the real consequences of large budget misses?
- How do results change with and without it, and why?

---

# Day 5 — Derive, fit, and stress-test ordinary least squares

## 5.1 A one-parameter error bowl

Start with no intercept:

$$
\hat{y}_i=\beta x_i.
$$

The objective is:

$$
S(\beta)=\sum_i(y_i-\beta x_i)^2.
$$

Expand the square:

$$
(y_i-\beta x_i)^2=y_i^2-2\beta x_iy_i+\beta^2x_i^2.
$$

After summing and differentiating:

$$
\frac{dS}{d\beta}=-2\sum_i x_iy_i+2\beta\sum_i x_i^2.
$$

At the bottom of a smooth bowl the derivative is zero. Solving gives:

$$
\hat{\beta}=\frac{\sum_i x_iy_i}{\sum_i x_i^2}.
$$

The second derivative is:

$$
\frac{d^2S}{d\beta^2}=2\sum_i x_i^2.
$$

If at least one $x_i$ is nonzero, this is positive, proving a unique minimum rather than a maximum. A grid search in the chapter independently checks that the formula lands at the bottom of the error bowl.

## 5.2 One feature with an intercept

Now:

$$
S(\beta_0,\beta_1)=\sum_i(y_i-\beta_0-\beta_1x_i)^2.
$$

There are two unknowns, so set two partial derivatives to zero. The intercept equation gives:

$$
\hat{\beta}_0=\bar{y}-\hat{\beta}_1\bar{x}.
$$

Therefore the fitted line passes through $(\bar{x},\bar{y})$. Substitution into the slope equation gives:

$$
\hat{\beta}_1=
\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}
{\sum_i(x_i-\bar{x})^2}.
$$

If all $x_i$ values are identical, the denominator is zero. The data contain no horizontal variation, so they cannot identify a slope.

The centred numerator resembles covariance and the denominator resembles variance. Different textbook definitions divide these sums by $n$ or $n-1$, but that common factor cancels in the slope ratio.

## 5.3 Expand the matrix objective

For multiple features:

$$
S(\beta)=(y-X\beta)^T(y-X\beta).
$$

Because $(y-X\beta)^T=y^T-\beta^TX^T$:

$$
S(\beta)=y^Ty-y^TX\beta-\beta^TX^Ty+\beta^TX^TX\beta.
$$

The middle expressions are both one-number scalars and are transposes of one another, so they are equal:

$$
S(\beta)=y^Ty-2\beta^TX^Ty+\beta^TX^TX\beta.
$$

Check shapes as you read: $y^Ty$ is one number, $\beta^TX^Ty$ is one number, and $\beta^TX^TX\beta$ is one number. An objective function must return one number for each candidate parameter vector.

## 5.4 The matrix gradient

Use three derivative facts:

1. $y^Ty$ contains no $\beta$, so its gradient is zero.
2. For constant vector $c$, $\nabla_\beta(\beta^Tc)=c$.
3. $\nabla_\beta(\beta^TA\beta)=(A+A^T)\beta$.

Here $A=X^TX$, which is symmetric because $(X^TX)^T=X^TX$. Therefore:

$$
\nabla_\beta S(\beta)=-2X^Ty+2X^TX\beta.
$$

At a minimum the gradient is zero:

$$
-2X^Ty+2X^TX\hat{\beta}=0,
$$

which produces the same normal equations found geometrically:

$$
X^TX\hat{\beta}=X^Ty.
$$

### Why the quadratic derivative has two terms

Writing $\beta^TA\beta=\sum_j\sum_k\beta_jA_{jk}\beta_k$, a derivative with respect to $\beta_r$ finds occurrences on the left ($j=r$) and the right ($k=r$). Those contributions form $A\beta$ and $A^T\beta$. Symmetry makes them identical, producing $2A\beta$.

### Finite-difference audit

A centred finite difference estimates one gradient component:

$$
\frac{\partial S}{\partial\beta_j}\approx
\frac{S(\beta+h u_j)-S(\beta-h u_j)}{2h}.
$$

$u_j$ has 1 in position $j$ and 0 elsewhere. Comparing this numerical estimate with `-2 * X.T @ (y - X @ beta)` helps catch derivative mistakes. Very large $h$ gives approximation error; extremely small $h$ suffers floating-point cancellation.

## 5.5 Existence, uniqueness, convexity, and rank

If $X^TX$ is invertible:

$$
\hat{\beta}=(X^TX)^{-1}X^Ty.
$$

This requires **full column rank**: no design column can be made exactly from the others, and the data contain enough independent directions to estimate every coefficient.

The Hessian is:

$$
\nabla_\beta^2S(\beta)=2X^TX.
$$

For any vector $z$:

$$
z^TX^TXz=(Xz)^T(Xz)=\lVert Xz\rVert_2^2\geq0.
$$

Thus $X^TX$ is **positive semidefinite** and SSR is convex: it has no deceptive local minima. With full column rank, $Xz\neq0$ for every nonzero $z$, so the surface is strictly convex and the parameter minimiser is unique.

Predictions can still have a unique closest projection when coefficients are non-unique. Rank deficiency mainly destroys the unique interpretation of individual coefficients.

## 5.6 Why code should use `lstsq`

Three related implementations are:

```python
# Useful as mathematical notation, usually avoid in application code
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

# Solve the normal-equation system without explicit inversion
beta_hat = np.linalg.solve(X.T @ X, X.T @ y)

# Preferred here: solve least squares directly
beta_hat, residual_sums, rank, singular_values = np.linalg.lstsq(
    X, y, rcond=None
)
```

Forming $X^TX$ squares the condition number and can magnify numerical instability. Explicit inversion also does unnecessary work. `lstsq` uses a decomposition designed for least squares and reports rank and singular values. It solves the same objective; it is not replacing the mathematics with magic.

`residual_sums` may be empty in exact-fit or rank-deficient cases, so the estimator computes training metrics explicitly from predictions.

## 5.7 Fitting assumptions are not inference assumptions

Keep three layers separate:

1. **Algebra:** Did the program find a least-squares minimum?
2. **Generalisation:** Will the relationship work on future or wider-population projects?
3. **Causation:** Does the design identify an intervention effect?

Normally distributed residuals are not required merely to calculate OLS. Full column rank is required for a unique coefficient vector. Claims about unbiasedness, confidence intervals, heteroskedasticity, clustering, sampling, or causal effects require additional assumptions and methods.

A perfect training calculation proves only the first layer.

## 5.8 Perfect multicollinearity and rank failure

If both kilometres and metres are columns:

$$
x_m=1000x_{km}.
$$

One column contains no new information. Infinitely many coefficient pairs can produce the same combined contribution, so the separate coefficients are unidentified.

```python
X_bad = np.column_stack([np.ones(x_km.size), x_km, 1000 * x_km])
print(np.linalg.matrix_rank(X_bad))
print(X_bad.shape[1])
```

If rank is less than the number of columns, the design is rank deficient. `np.linalg.solve` may fail because $X^TX$ is singular. `np.linalg.lstsq` can return one minimum-norm solution, but returned numbers do not make redundant coefficients meaningful.

Usually remove or redesign the redundant feature. More columns do not necessarily mean more information.

Near-collinearity is not exact rank failure but can create unstable coefficients and long, narrow SSR contours. Small data changes may then cause large coefficient changes while predictions move much less.

## 5.9 Read the error-surface figure

With an intercept and one slope, every point $(\beta_0,\beta_1)$ is a candidate model. A contour joins candidate pairs with equal SSR. The centre marks the minimum.

- Well-separated information produces compact elliptical contours.
- Highly correlated design columns produce a long, narrow valley.
- Perfect collinearity produces a flat direction with no unique centre.

The contour plot is the two-parameter version of the one-parameter error bowl.

## 5.10 The complete estimator, method by method

The final `MHPCostEstimator` in the chapter has these responsibilities:

| Component | Responsibility |
|---|---|
| `AnalysisContract` | Record purpose, target, unit of observation, and intended use |
| `__init__` | Create empty learned attributes and fitted state |
| `_validate_inputs` | Enforce numerical shapes and finite values |
| `_add_intercept_column` | Prepend exactly one ones column |
| `_metrics_from_predictions` | Calculate residual-based metrics |
| `fit` | Check names, build design, call `lstsq`, reject rank deficiency, store results |
| `predict` | Require fitted state and compatible new features |
| `evaluate` | Predict and calculate metrics for supplied observations |
| `coefficients` | Pair intercept and feature names with learned values |

### Important safeguards in `fit`

- Feature-name count must equal raw feature count.
- Feature names cannot be empty or duplicated.
- The intercept is added internally; callers should not add another one.
- Required rank equals the number of design columns.
- Learned state is stored only after rank passes.
- Training predictions and metrics are calculated from the fitted design.
- Returning `self` permits `model = MHPCostEstimator(contract).fit(...)`.

### Important safeguards in `predict`

- Calling before `fit` raises `RuntimeError`.
- New rows must have the same feature count and order as training data.
- Inputs must be finite.
- A single project becomes shape `(1, p)`.

The implementation checks feature counts, not semantic column order. In production, labeled tables and schema checks should prevent accidentally swapping capacity and road-distance columns.

## 5.11 Interpret the synthetic KP fit carefully

The chapter fits eight projects using road distance, planned capacity, and terrain index. This tiny synthetic set is inspectable, not deployment-ready.

If the road-distance coefficient is positive, a prediction-oriented reading is:

> Holding the other included numerical features fixed, the fitted rule adds the coefficient's number of million PKR for each additional kilometre of recorded road distance.

It does not establish that building a kilometre of road changes cost by that amount. “Distance from a road” and “constructing a road” are different variables and the latter is an intervention.

Also ask whether fixed-capacity, fixed-terrain comparisons at different distances actually occur in the data. Regression can calculate contrasts in weakly supported regions.

## 5.12 Tests that connect code to mathematics

The chapter verifies:

```python
assert np.allclose(model.parameters_, beta_from_normal_equations)
assert np.allclose(X_design.T @ residuals, 0.0, atol=1e-9)
assert np.isclose(residuals.sum(), 0.0, atol=1e-9)
assert np.isclose(metrics["ssr"], residuals @ residuals)
assert np.isclose(metrics["rmse"], np.sqrt(np.mean(residuals ** 2)))
```

Each test corresponds to a claim:

- two solution methods agree for this full-rank example;
- residuals are perpendicular to every design column;
- the intercept makes training residuals sum to zero;
- code matches the SSR definition; and
- code matches the RMSE definition.

## 5.13 What this estimator still cannot do

It does not yet:

- create training, validation, and test splits;
- handle categorical districts, contractors, or project types;
- standardise features or thoroughly diagnose numerical conditioning;
- quantify coefficient or prediction uncertainty;
- diagnose nonlinearity, heteroskedasticity, or influential observations;
- model inflation, time trends, repeated districts, or clustering;
- enforce data lineage, access permissions, and revision history;
- protect automatically against project-phase leakage;
- establish causal effects; or
- monitor drift when future construction conditions change.

Acknowledging these limitations is part of a correct analysis, not an apology added after it.

---

# Week 1 capstone: a beginner's execution plan

The capstone asks you to build, break, and rebuild an early-stage estimator for 40 synthetic projects.

## Pass 1: define

Write an `AnalysisContract` and a short data note answering:

- What exactly is one row?
- What is the target, unit, and price basis?
- At what project stage is the prediction made?
- Which decision will it support?
- Is the job prediction, explanation, or causation?
- What is the precise estimand?

## Pass 2: build

1. Select at least three features genuinely known at prediction time.
2. Preserve project IDs while aligning rows.
3. Validate shapes, finite values, units, and names.
4. Fit the estimator.
5. Report coefficients with units and conditional wording.
6. Report training SSR, MSE, RMSE, and MAE.
7. Verify orthogonality and the metric equations.

## Pass 3: break

Create one controlled example of each failure and explain why it matters:

1. reverse target row order while leaving features unchanged;
2. insert `NaN` or infinity;
3. add a redundant metres version of a kilometre feature;
4. add information recorded after construction begins;
5. add one extreme target value.

Record whether validation catches the failure. Row misalignment and leakage are especially dangerous because plain array-shape validation may not detect them.

## Pass 4: rebuild

1. restore identifier alignment;
2. investigate and document missing/extreme values;
3. remove redundant information;
4. remove leaked features or redefine prediction time;
5. hold out projects not used for fitting;
6. compare training and held-out RMSE and MAE; and
7. write a recommendation separating supported prediction claims from unsupported explanatory or causal claims.

High held-out error relative to training error is a warning about overfitting, dataset differences, or both. One small held-out set also has uncertainty; do not treat it as a permanent truth.

## Self-assessment rubric

| Dimension | Ready when you can... |
|---|---|
| Question | state purpose, estimand, timing, target, and intended use precisely |
| Mathematics | reproduce key derivations and explain every symbol |
| Code | validate shapes and rank and tie tests to equations |
| Geometry | connect column space, projection, orthogonality, and minimum distance |
| Context | discuss units, provenance, fairness, and institutional consequences |
| Communication | report results, uncertainty, and limits in plain language |

---

# Formula sheet with reading instructions

| Concept | Formula | Read it as... |
|---|---|---|
| Linear prediction | $\hat{y}=X\beta$ | Build predictions from weighted design columns |
| Residual | $e=y-\hat{y}$ | Actual minus predicted |
| SSR | $e^Te=\sum_i e_i^2$ | Total squared training error |
| MSE | $SSR/n$ | Average squared error |
| RMSE | $\sqrt{SSR/n}$ | Squared-error summary in target units |
| MAE | $\frac1n\sum_i|e_i|$ | Average absolute miss |
| Orthogonality | $X^Te=0$ | Residual is perpendicular to every design column |
| Normal equations | $X^TX\hat{\beta}=X^Ty$ | Zero-gradient/perpendicularity condition |
| Closed form | $(X^TX)^{-1}X^Ty$ | Unique coefficients when full column rank holds |
| Through-origin slope | $\frac{\sum_i x_iy_i}{\sum_i x_i^2}$ | One-feature slope when no intercept is allowed |
| Intercept-model slope | $\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}{\sum_i(x_i-\bar{x})^2}$ | Co-movement divided by feature variation |
| Intercept | $\bar{y}-\hat{\beta}_1\bar{x}$ | Make the fitted line pass through the mean point |
| Gradient | $-2X^Ty+2X^TX\beta$ | Direction and rate of local SSR increase |
| Hessian | $2X^TX$ | Curvature of the SSR surface |

## Shape sheet

For $n$ observations, $p$ raw features, and an intercept:

| Object | Shape |
|---|---|
| Raw features | $(n,p)$ |
| Ones column | $(n,1)$ |
| Design matrix | $(n,p+1)$ |
| Target | $(n,)$ |
| Parameters | $(p+1,)$ |
| Predictions | $(n,)$ |
| Residuals | $(n,)$ |
| $X^TX$ | $(p+1,p+1)$ |
| $X^Ty$ | $(p+1,)$ |
| $X^Te$ | $(p+1,)$ |

---

# Glossary in beginner language

**Affine model:** A linear combination plus a constant translation; regression with an intercept is affine in its inputs but conventionally called linear regression.

**Association:** A numerical relationship observed between variables. It need not be causal.

**Broadcasting:** NumPy's rules for expanding compatible shapes during elementwise operations. Convenient, but not a substitute for matrix multiplication.

**Causal effect:** Difference between outcomes under two interventions or conditions for a defined population and time horizon.

**Column space:** Every vector obtainable as a weighted combination of a matrix's columns.

**Condition number:** A measure of sensitivity to small numerical changes. A large value warns that coefficient calculations may be unstable.

**Confounder:** A common cause of an exposure or decision and an outcome.

**Convex objective:** A bowl-shaped objective with no deceptive local minima.

**Counterfactual:** The outcome that would have occurred under an alternative action or condition.

**Data dictionary:** Documentation of fields, meanings, units, timing, provenance, valid values, and missing-value rules.

**Design matrix:** The rectangular matrix whose columns are the building blocks of predictions.

**Estimator:** A procedure that converts data into parameter estimates; OLS is an estimator.

**Estimand:** The exact quantity an analysis intends to estimate.

**Feature:** An input variable used to form model predictions.

**Finite difference:** A numerical approximation to a derivative made by slightly changing an input.

**Full column rank:** No design column is an exact linear combination of the others.

**Gradient:** Vector containing all first partial derivatives of a scalar function.

**Hessian:** Matrix containing second partial derivatives and describing curvature.

**Hyperparameter:** A modeling setting chosen outside the fitting calculation.

**Intercept:** Constant amount added to every linear prediction.

**Loss function:** Numerical rule that penalises errors during model fitting.

**Matrix rank:** Number of independent directions represented by a matrix.

**Metric:** Numerical summary used to evaluate performance.

**Multicollinearity:** Linear dependence among features. Perfect multicollinearity causes rank deficiency; near-multicollinearity can make coefficients unstable.

**Normal equations:** $X^TX\hat{\beta}=X^Ty$, the OLS condition obtained from geometry or calculus.

**Observation:** One unit represented by one dataset row.

**OLS:** Ordinary least squares, which chooses parameters that minimise the sum of squared residuals.

**Orthogonal:** Perpendicular; two vectors have dot product zero.

**Parameter:** Numerical value learned by fitting, such as an intercept or slope.

**Positive semidefinite:** A matrix $A$ for which $z^TAz\geq0$ for every $z$.

**Prediction:** Estimated target value produced by the fitted model.

**Projection:** Closest point in a subspace to a given vector under Euclidean distance.

**Provenance:** Where, when, why, and by whom data were created or changed.

**Rank deficiency:** Fewer independent design directions than columns, so coefficients are not uniquely identified.

**Residual:** Observed target minus fitted prediction.

**Scalar:** One number.

**Simpson's paradox:** Aggregated and subgroup relationships differ or reverse because group composition matters.

**Singular value:** A quantity reported by matrix decompositions that helps describe independent directions and numerical conditioning.

**Target:** Outcome the model estimates.

**Target leakage:** Use of information unavailable or illegitimate at the real prediction time.

**Tensor:** An array with any number of axes; do not confuse tensor order with matrix rank.

**Training data:** Observations used to learn model parameters.

**Validation/test data:** Observations withheld from fitting and used to assess generalisation.

**Vector:** Ordered one-dimensional collection of numbers.

---

# Suggested five-day study schedule

| Activity each day | Approximate minutes |
|---|---:|
| Recall yesterday's main idea | 15 |
| Work through today's derivation | 35 |
| Type, run, and alter proof code | 35 |
| Create or inspect the figure | 20 |
| Extend the estimator | 35 |
| Break and repair one failure | 25 |
| Complete exit check in writing | 15 |
| **Total** | **180** |

Three hours is a guide, not a race. If you cannot explain yesterday's exit check, retrieve that idea before adding new notation.

Morning retrieval prompts:

- After Day 1: name the three regression jobs and one risk of each.
- After Day 2: draw the shapes of $X$, $y$, and $\beta$.
- After Day 3: expand one row of $X\beta$ and explain the ones column.
- After Day 4: explain why the OLS residual is perpendicular to $X$.
- After Day 5: state the normal equations and the condition for unique coefficients.

---

# Responsible interpretation in the KP context

Ask these questions throughout the project:

- At what project stage is each field known?
- Who recorded it, using which definition and instrument?
- Does one row mean the same thing across districts and implementing partners?
- Are all costs expressed in the same currency and price year?
- Does a terrain index of 4 mean the same thing to every assessor?
- Does missingness reflect geography, staffing, or project difficulty?
- Could the estimator systematically disadvantage remote communities?
- Could a coefficient be used to unfairly judge a field team for structural conditions?
- Are comparisons supported by similar observed projects, or are they extrapolations?
- Has the analysis confused a predictive signal with an intervention effect?

Technical correctness is necessary but not sufficient. Public-infrastructure analysis also requires traceable data, appropriate comparisons, careful language, and awareness of who bears the consequences of error.

---

# Where Week 2 begins

Week 1 gives you a transparent OLS calculation and an honest statement of its limits. The next topics naturally address weaknesses exposed here:

1. feature scaling and numerical conditioning;
2. QR and singular-value decompositions;
3. probability models and maximum likelihood;
4. gradient descent built from the OLS gradient;
5. training, validation, and test separation;
6. coefficient and prediction uncertainty; and
7. residual diagnostics and model revision.

The enduring chain is:

```text
precise question
    -> defensible data representation
    -> transparent mathematical model
    -> independent code and geometric checks
    -> bounded, responsible judgement
```

If you can move both ways through that chain—from a decision to equations and code, and from output back to a careful decision statement—you have achieved the central goal of Chapter 1.
