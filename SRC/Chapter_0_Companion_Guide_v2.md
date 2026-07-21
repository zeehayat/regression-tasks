# Chapter 0 Companion Guide: From Nothing to Ready
## A First-Principles Mathematical & Computational Reference Manual for Beginners

This companion guide is designed as an exhaustive, first-principles reference manual to accompany **Chapter 0: From Nothing to Ready**. It provides step-by-step mathematical derivations, intuitive mental models, line-by-line code breakdowns, and explicit structural mappings to ensure you never encounter a "black box" symbol or code snippet.

All mathematical notation is strictly formatted using standard LaTeX delimiters (`$inline$` and `$$display$$`), making it immediately compatible with Markdown renderers, MathJax, and KaTeX.

---

## Table of Contents
1. **Pedagogical Philosophy & Operational Contract**
2. **Day 0A — Python Foundations Refresher**
3. **Day 0B — NumPy, Array Shapes, and Matrix Algebra**
4. **Day 0C — Algebra, Summation Notation, and Loss Functions**
5. **Day 0D — First-Principles Statistics**
6. **Day 0E — Calculus, Gradients, Optimization, and Debugging**
7. **Level 0 Capstone: Complete Step-by-Step Reference Solution**
8. **Master Rosetta Stone & Traceback Cheat Sheet**

---

---

## 1. Pedagogical Philosophy & Operational Contract

The primary goal of Chapter 0 is to construct the mathematical, computational, and statistical intuition required before encountering matrix regression, multivariate calculus, and statistical inference in Chapter 1.

### The Five Rules of the Pedagogical Contract
1. **One Central Idea Per Day:** Structured, incremental progression over five foundational days without skipping steps.
2. **No Mathematical Hand-Waving:** Every algebraic symbol, subscript, and operator is unpacked into concrete arithmetic and visible Python outputs.
3. **Code as Proof:** If a mathematical property or equation is claimed to be true, clean Python code will prove it numerically.
4. **Build, Break, Rebuild:** To truly understand how systems work, you must deliberately break them (e.g., shape mismatches, missing keys, learning rate explosions) and read the resulting errors calmly.
5. **Type Everything:** Muscle memory in writing Python syntax and NumPy indexing is essential for effortless problem-solving.

---

---

## 2. Day 0A — Python Foundations Refresher

### 2.1 Variables and Functions: State vs. Behavior

A **variable** assigns a symbolic label to a location in memory holding a specific value. A **function** encapsulates a reusable sequence of computations (a behavior) that accepts inputs, executes a contract, and explicitly returns an output.

```python
# Variable assignment: Storing a scalar value
capacity_kw = 250.0

# Function definition: Defining an operational contract
def kw_to_mw(capacity_kw: float) -> float:
    """Convert electrical capacity from kilowatts (kW) to megawatts (MW)."""
    return capacity_kw / 1000.0

# Calling the function
result_mw = kw_to_mw(capacity_kw)
print(result_mw)  # Output: 0.25
```

#### Line-by-Line Breakdown:
* `def kw_to_mw(capacity_kw: float) -> float:`
  * `def` tells Python we are defining a new function named `kw_to_mw`.
  * `capacity_kw` is the **parameter** (the input variable expected by the function contract).
  * `: float` and `-> float` are optional type hints indicating that the expected input and output are floating-point numbers.
* `"""Convert electrical capacity..."""`: The **docstring**, providing documentation accessible via `help(kw_to_mw)`.
* `return capacity_kw / 1000.0`: The `return` statement explicitly sends the calculated result back to the caller. If `return` is omitted, Python returns `None` by default.

---

### 2.2 Loops and List Iteration

A `for` loop iterates over a sequential collection (such as a list), binding each element to a loop variable one step at a time.

```python
capacities_kw = [100.0, 250.0, 500.0, 1800.0, 60.0]

capacities_mw = []
for c in capacities_kw:
    capacities_mw.append(kw_to_mw(c))

print(capacities_mw)  # Output: [0.1, 0.25, 0.5, 1.8, 0.06]
```

#### Execution Mechanics:
1. `capacities_mw = []` allocates an empty list in memory.
2. In the first iteration, `c` becomes `100.0`. `kw_to_mw(100.0)` evaluates to `0.1`, which is appended to `capacities_mw`.
3. This process repeats sequentially for every element in `capacities_kw`.

---

### 2.3 List Comprehensions: Syntactic Unrolling

A **list comprehension** is a concise way to create lists. It performs the exact same operation as a standard `for` loop with `.append()`, but in a single line.

```python
# List comprehension syntax: [expression for item in iterable]
capacities_mw = [kw_to_mw(c) for c in capacities_kw]
```

#### Translation Mapping:
* **Expression:** `kw_to_mw(c)` (what gets placed into the new list).
* **Loop target:** `c`.
* **Iterable:** `capacities_kw`.

If a list comprehension ever feels confusing, unroll it back into a multi-line `for` loop with an explicit empty list and `.append()`.

---

### 2.4 Dictionaries: Label-Value Mapping

A standard list stores items by numerical index (`list[0]`, `list[1]`), which can lead to index-confusion errors when managing tabular data. A **dictionary** (`dict`) stores data using key-value pairs, maintaining explicit labels for every value.

```python
project = {
    "site_id": "MHP-0042",
    "cable_length_km": 15.0,
    "planned_capacity_kw": 100.0,
    "terrain_index": 3,
}

# Accessing a value by key
print(project["planned_capacity_kw"])  # Output: 100.0
```

---

### 2.5 Object-Oriented Encapsulation: The `self` Keyword

A **class** is a blueprint that combines state (data attributes) and behavior (methods) into a single object.

```python
class UnitConverter:
    def __init__(self, factor: float):
        # __init__ is the constructor method.
        # It runs automatically when a new object instance is created.
        self.factor = factor  # Store factor inside this specific instance.

    def convert(self, value: float) -> float:
        # Access the instance's stored factor using self.factor
        return value * self.factor

# Instantiate the class with a specific conversion factor (1/1000 = 0.001)
kw_to_mw_converter = UnitConverter(factor=0.001)

# Call the method on the instantiated object
print(kw_to_mw_converter.convert(250.0))  # Output: 0.25
```

#### What is `self`?
* `self` refers to the **current instance** of the object.
* When you execute `kw_to_mw_converter.convert(250.0)`, Python automatically translates it under the hood to `UnitConverter.convert(kw_to_mw_converter, 250.0)`.
* `self.factor` ensures that the method reads the specific factor stored inside *that* converter instance.

---

### 2.6 The Standard List Multiplication Trap

```python
# Standard Python List
capacities_kw = [100.0, 250.0, 500.0]

# Unexpected Behavior:
print(capacities_kw * 2)
# Output: [100.0, 250.0, 500.0, 100.0, 250.0, 500.0]
```

**Why does this happen?** In standard Python, the `*` operator applied to a list means **sequence repetition**, not mathematical scaling. To perform mathematical scaling on every element without writing explicit loops, we must use vectorized array operations provided by **NumPy**.

---

---

## 3. Day 0B — NumPy, Array Shapes, and Matrix Algebra

### 3.1 Why NumPy? Contiguous Memory and Vectorization

Standard Python lists are arrays of pointers to scattered memory locations. A **NumPy array** (`np.ndarray`) stores homogeneous numerical data in a **contiguous block of memory**. This allows NumPy to perform arithmetic operations using low-level, vectorized CPU instructions.

```python
import numpy as np

# Creating a 1D NumPy array
capacities_kw = np.array([100.0, 250.0, 500.0])

# True vectorized elementwise scaling
capacities_watts = capacities_kw * 1000.0
print(capacities_watts)
# Output: [100000. 250000. 500000.]
```

---

### 3.2 Array Dimensions and the `.shape` Attribute

Every NumPy array has a `.shape` tuple that defines its dimensions. It is vital to distinguish between a 1D vector, a 2D row matrix, and a 2D column matrix.

```python
import numpy as np

# 1D Array (Vector of 3 elements)
a = np.array([1.0, 2.0, 3.0])
print(a.shape)  # Output: (3,)

# 2D Array with 1 Row and 3 Columns (Row Matrix)
b = np.array([[1.0, 2.0, 3.0]])
print(b.shape)  # Output: (1, 3)

# 2D Array with 3 Rows and 1 Column (Column Matrix)
c = np.array([[1.0], [2.0], [3.0]])
print(c.shape)  # Output: (3, 1)
```

#### Structural Differences Table:

| Array Representation | `.shape` | Number of Dimensions (`.ndim`) | Conceptual Meaning |
| :--- | :--- | :--- | :--- |
| `np.array([1, 2, 3])` | `(3,)` | 1 | Flat 1D collection |
| `np.array([[1, 2, 3]])` | `(1, 3)` | 2 | Matrix with 1 row, 3 columns |
| `np.array([[1], [2], [3]])` | `(3, 1)` | 2 | Matrix with 3 rows, 1 column |

These shapes are **not interchangeable** in linear algebra operations. A `(3,)` vector behaves differently than a `(3, 1)` matrix during matrix multiplication and broadcasting.

---

### 3.3 Matrix Slicing and Subscript Indexing

For a 2D array (matrix $X$), indexing uses the syntax `X[row_slice, column_slice]`.

```python
X = np.array([
    [15.0, 100.0, 3.0],   # Row 0: Site MHP-001
    [30.0, 250.0, 4.0],   # Row 1: Site MHP-002
    [5.0,  500.0, 1.0],   # Row 2: Site MHP-003
])

print(X.shape)        # Output: (3, 3) -> 3 rows (projects), 3 columns (features)
print(X[0, :])        # Output: array([15., 100., 3.]) -> Entire Row 0
print(X[:, 0])        # Output: array([15., 30., 5.])  -> Entire Column 0
print(X[0, 1])        # Output: 100.0                 -> Element at Row 0, Column 1
print(X[:2, :2])      # Output: First 2 rows and first 2 columns (2x2 submatrix)
```

---

### 3.4 The Mechanics of Broadcasting

**Broadcasting** is NumPy's set of rules for performing elementwise operations on arrays of different shapes without copying data unnecessarily.

#### The Official Broadcasting Rule:
NumPy compares the shapes of two arrays **from right to left** (trailing dimensions forward). Two dimensions are compatible if:
1. They are **equal**, or
2. One of them is **1**.

If a dimension is `1`, it is stretched along that dimension to match the size of the other array.

#### Concrete Example: Mean Centering Matrix $X$
Suppose $X$ has shape `(3, 2)` (3 project observations, 2 features) and `means` has shape `(2,)`.

```python
X = np.array([
    [15.0, 100.0],
    [30.0, 250.0],
    [5.0,  500.0]
])  # Shape: (3, 2)

means = np.array([16.7, 283.3])  # Shape: (2,)

# Subtract means from X
centred = X - means
```

#### Step-by-Step Shape Alignment Check:
1. Shape of $X$: `(3, 2)`
2. Shape of `means`: `(2,)` $ightarrow$ Right-aligned: `(1, 2)`
3. Compare trailing dimension: $2 == 2$ (Match!)
4. Compare leading dimension: $3$ vs $1$. The dimension of size $1$ stretches from $1 ightarrow 3$.
5. Final broadcasted subtraction subtracts `16.7` from Column 0 and `283.3` from Column 1 for all rows.

---

### 3.5 The Dot Product ($\cdot$) at Three Abstraction Levels

The **dot product** of two equal-length 1D vectors $\mathbf{a}$ and $\mathbf{b}$ of length $p$ is defined mathematically as:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{p} a_i b_i = a_1 b_1 + a_2 b_2 + \dots + a_p b_p$$

Here are three ways to compute the exact same result in Python:

```python
features = np.array([15.0, 100.0])       # Cable length (km), Capacity (kW)
weights  = np.array([0.05, 0.002])       # Cost weights per unit

# Level 1: Manual Scalar Term-by-Term
manual = features[0] * weights[0] + features[1] * weights[1]

# Level 2: Explicit Python Loop with zip
loop_total = 0.0
for f, w in zip(features, weights):
    loop_total += f * w

# Level 3: Vectorized NumPy Operator (@)
dot_total = features @ weights  # Equivalently: np.dot(features, weights)

print(manual, loop_total, dot_total)
# Output: 0.95 0.95 0.95
```

---

### 3.6 Matrix Prediction ($Xoldsymbol{eta}$)

When we multiply a design matrix $X$ of shape $(N 	imes P)$ by a parameter vector $oldsymbol{eta}$ of shape $(P,)$ using the `@` operator, NumPy performs a dot product between **each row of $X$** and the weight vector $oldsymbol{eta}$.

```python
X = np.array([
    [15.0, 100.0],  # Project 0
    [30.0, 250.0],  # Project 1
    [5.0,  500.0],  # Project 2
])  # Shape: (3, 2)

beta = np.array([0.05, 0.002])  # Shape: (2,)

# Matrix-vector multiplication
predictions = X @ beta
print(predictions)
# Output: [0.95  2.0   1.25 ]
```

#### Matrix Dimension Rule:
$$(N 	imes P) 	imes (P 	imes 1) \longrightarrow (N 	imes 1)$$
The inner dimensions ($P$) must match exactly, and the resulting prediction vector has length $N$ (one predicted output per observation row).

---

---

## 4. Day 0C — Algebra, Summation Notation, and Loss Functions

### 4.1 Functions and Equations of Lines

A linear relationship between an input variable $x$ and an output variable $y$ is expressed as:

$$y = m x + c$$

* $m$: **Slope** — The change in $y$ given a $1$-unit increase in $x$ ($rac{\Delta y}{\Delta x}$).
* $c$: **Intercept** — The value of $y$ when $x = 0$.

In multiple regression, we extend this equation to $P$ features:

$$\hat{y} = eta_0 + eta_1 x_1 + eta_2 x_2 + \dots + eta_p x_p$$

---

### 4.2 Deconstructing Summation Notation ($\sum$)

The summation symbol $\sum$ represents an accumulator loop.

$$\sum_{i=1}^{n} x_i$$

* $i = 1$: **Lower limit** (starting index of the loop).
* $n$: **Upper limit** (stopping index of the loop).
* $x_i$: **Expression** evaluated at step $i$ and added to the total.

#### Translating Summation to Python Code
Consider the **Sum of Squared Residuals (SSR)** formula:

$$	ext{SSR} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Where $y_i$ is the actual target value, and $\hat{y}_i$ is the predicted target value.

```python
import numpy as np

y_actual    = np.array([12.0, 30.0, 8.0])
y_predicted = np.array([11.2, 31.5, 9.0])

# 1. Formula expressed directly as a loop
ssr_loop = 0.0
for i in range(len(y_actual)):
    ssr_loop += (y_actual[i] - y_predicted[i]) ** 2

# 2. Formula expressed as a single vectorized NumPy line
ssr_numpy = np.sum((y_actual - y_predicted) ** 2)

print(ssr_loop, ssr_numpy)  # Output: 2.89 2.89
```

---

### 4.3 Why We Square Errors in Loss Functions

When evaluating prediction errors $e_i = y_i - \hat{y}_i$:
1. **Cancellation Problem:** Unsquared errors can be positive or negative. A model with errors $+10$ and $-10$ would sum to zero error ($\sum e_i = 0$), falsely appearing perfect.
2. **Squaring Eliminates Signs:** $(+10)^2 = 100$ and $(-10)^2 = 100$, ensuring all deviations contribute positively to total loss.
3. **Non-Linear Penalty for Large Outliers:** Squaring penalizes large errors much more severely than small ones:
   * Error = $2 \implies 	ext{Penalty} = 2^2 = 4$
   * Error = $10 \implies 	ext{Penalty} = 10^2 = 100$ (a 5x increase in error results in a 25x increase in penalty).

---

### 4.4 Critical Distinction: Sum of Squares vs. Square of Sum

$$\sum_{i=1}^{n} x_i^2 
eq \left( \sum_{i=1}^{n} x_i ight)^2$$

Let $x = [2.0, -3.0, 5.0]$:

#### 1. Sum of Squares ($\sum x_i^2$):
$$\sum x_i^2 = (2.0)^2 + (-3.0)^2 + (5.0)^2 = 4.0 + 9.0 + 25.0 = 38.0$$

#### 2. Square of Sum ($(\sum x_i)^2$):
$$\left( \sum x_i ight)^2 = (2.0 + (-3.0) + 5.0)^2 = (4.0)^2 = 16.0$$

```python
x = np.array([2.0, -3.0, 5.0])

sum_of_squares = np.sum(x ** 2)  # Output: 38.0
square_of_sum  = np.sum(x) ** 2  # Output: 16.0
```

---

---

## 5. Day 0D — First-Principles Statistics

### 5.1 Sample Mean ($ar{x}$)

The mean represents the arithmetic average (or balancing point) of a distribution.

$$ar{x} = rac{1}{n} \sum_{i=1}^{n} x_i$$

```python
costs = np.array([12.0, 30.0, 8.0, 45.0, 15.0])
mean_manual = np.sum(costs) / len(costs)  # Output: 22.0
```

---

### 5.2 Variance ($\sigma^2$) and Standard Deviation ($\sigma$)

**Variance** measures the average squared distance of data points from their mean:

$$\sigma^2 = rac{1}{n} \sum_{i=1}^{n} (x_i - ar{x})^2$$

> **Why can't we just average raw deviations $(x_i - ar{x})$?**
> The sum of raw deviations around the mean is **always zero** ($\sum_{i=1}^n (x_i - ar{x}) = 0$) because positive and negative deviations cancel out.

**Standard Deviation** ($\sigma$) takes the square root of variance to return the spread metric to the original units of the dataset:

$$\sigma = \sqrt{\sigma^2}$$

```python
# First principles computation
deviations = costs - costs.mean()
variance_manual = np.sum(deviations ** 2) / len(costs)
std_manual = np.sqrt(variance_manual)

print(variance_manual, std_manual)
# Output: 182.8 13.520355024924517
```

#### Population Variance vs. Sample Variance ($n$ vs $n-1$):
* **Population Variance (`ddof=0`):** Divides by $n$. Used when analyzing an entire dataset.
* **Sample Variance (`ddof=1`):** Divides by $n - 1$ (Bessel's correction). Used when estimating population variance from a sample to eliminate downward bias.

---

### 5.3 Covariance ($	ext{cov}(x, y)$)

Covariance measures whether two continuous variables tend to deviate from their respective means in the same direction at the same time.

$$	ext{cov}(x, y) = rac{1}{n} \sum_{i=1}^{n} (x_i - ar{x})(y_i - ar{y})$$

```python
cable_km = np.array([12.0, 30.0, 5.0, 40.0, 15.0])

x_dev = costs - costs.mean()
y_dev = cable_km - cable_km.mean()

cov_manual = np.sum(x_dev * y_dev) / len(costs)
# Output: 162.8
```

* **Positive Covariance:** When $x$ is above its mean, $y$ tends to be above its mean.
* **Negative Covariance:** When $x$ is above its mean, $y$ tends to be below its mean.
* **Zero Covariance:** No linear co-movement between $x$ and $y$.

---

### 5.4 Pearson Correlation Coefficient ($r$)

Because covariance depends on the units of measurement, it cannot be used to compare strength across different features. **Pearson Correlation** ($r$) normalizes covariance by dividing it by the product of the standard deviations of both variables:

$$r = rac{	ext{cov}(x, y)}{\sigma_x \sigma_y}$$

```python
r_manual = cov_manual / (costs.std() * cable_km.std())
print(r_manual)  # Output: 0.9413
```

#### Properties of $r$:
* $r \in [-1, +1]$.
* $r = +1$: Perfect positive linear relationship.
* $r = -1$: Perfect negative linear relationship.
* $r = 0$: No linear relationship.

---

### 5.5 The Zero Variance Trap and `NaN` Poisoning

If a feature vector has zero variation (e.g., $x = [5.0, 5.0, 5.0, 5.0]$), its standard deviation is zero ($\sigma_x = 0.0$).

```python
constant_feature = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
r_result = np.corrcoef(constant_feature, costs)[0, 1]
print(r_result)  # Output: nan
```

#### Why does this happen?
Calculating $r$ requires dividing by $\sigma_x$. Division by zero produces `NaN` (Not a Number). In machine learning pipelines, a single `NaN` will silently propagate through matrix multiplications, corrupting downstream calculations.

---

---

## 6. Day 0E — Calculus, Gradients, Optimization, and Debugging

### 6.1 The Definition of a Derivative

A **derivative** measures the instantaneous rate of change (the slope of the tangent line) of a function $f(x)$ at a specific point $x$.

$$rac{dy}{dx} = f'(x) = \lim_{h 	o 0} rac{f(x+h) - f(x)}{h}$$

#### Code Proof: Finite Difference Approximation
We can approximate derivatives numerically by choosing an extremely small step size $h$ (e.g., $h = 10^{-6}$).

```python
def f(x):
    return x ** 2

def numerical_slope(f, x, h=1e-6):
    return (f(x + h) - f(x)) / h

x_val = 3.0
print(numerical_slope(f, x_val))
# Output: 6.000000999922336  (Converges to analytical derivative 2*x = 6.0)
```

---

### 6.2 Essential Analytical Derivative Rules

1. **Power Rule:**
   $$rac{d}{dx} [x^n] = n x^{n-1}$$
   * Example: $rac{d}{dx} [x^2] = 2x$
2. **Constant Multiple Rule:**
   $$rac{d}{dx} [c \cdot f(x)] = c \cdot rac{d}{dx}[f(x)]$$
3. **Sum Rule:**
   $$rac{d}{dx} [f(x) + g(x)] = rac{d}{dx}[f(x)] + rac{d}{dx}[g(x)]$$

#### Combined Example:
For $y = 3x^2 + 5x$:

$$rac{dy}{dx} = rac{d}{dx}[3x^2] + rac{d}{dx}[5x] = 3(2x) + 5(1) = 6x + 5$$

---

### 6.3 Partial Derivatives ($\partial$) and the Gradient ($
abla$)

When an error function depends on multiple parameters—such as slope $m$ and intercept $c$—a **partial derivative** measures how the output changes when we adjust *one parameter* while keeping all others constant. We use the symbol $\partial$ (curly "d") instead of $d$.

$$	ext{Gradient } 
abla E(m, c) = egin{bmatrix} rac{\partial E}{\partial m} \ rac{\partial E}{\partial c} \end{bmatrix}$$

```python
def error(m, c):
    # Loss for predicting y=10.0 when x=2.0
    prediction = m * 2.0 + c
    return (10.0 - prediction) ** 2

def partial_wrt_m(m, c, h=1e-6):
    return (error(m + h, c) - error(m, c)) / h

def partial_wrt_c(m, c, h=1e-6):
    return (error(m, c + h) - error(m, c)) / h

m, c = 1.0, 1.0
print(partial_wrt_m(m, c))  # Output: -27.9999
print(partial_wrt_c(m, c))  # Output: -13.9999
```

---

### 6.4 Gradient Descent Mechanics

**Gradient Descent** is an iterative optimization algorithm that minimizes a loss function by taking steps in the direction opposite to the gradient.

$$	heta_{	ext{new}} = 	heta_{	ext{old}} - lpha 
abla E(	heta)$$

Where $lpha$ is the **learning rate** (step size hyperparameter).

```python
m, c = 0.0, 0.0          # Initial bad guess
learning_rate = 0.01     # Step size alpha

for step in range(50):
    grad_m = partial_wrt_m(m, c)
    grad_c = partial_wrt_c(m, c)
    
    # Step opposite to the gradient
    m -= learning_rate * grad_m
    c -= learning_rate * grad_c

print(f"Optimized parameters: m = {m:.3f}, c = {c:.3f}")
# Output: m and c settle into values where m*2 + c ≈ 10.0
```

#### What happens if the learning rate $lpha$ is too large?
If $lpha$ is set too high (e.g., $lpha = 5.0$), parameter updates will overshoot the minimum, causing error to explode toward infinity.

---

---

## 7. Level 0 Capstone: Complete Step-by-Step Reference Solution

Here is the complete, execution-verified Python script solving all 7 steps of the Level 0 Capstone.

```python
import numpy as np

# Capstone Input Data
projects = np.array([
    [12.0, 15.0],   # Row 0: [cable_km, terrain_index]
    [30.0, 25.0],   # Row 1
    [5.0,  8.0],    # Row 2
    [40.0, 45.0],   # Row 3
    [15.0, 12.0],   # Row 4
])
costs = np.array([12.0, 30.0, 8.0, 45.0, 15.0])   # Actual costs (million PKR)

# -------------------------------------------------------------------------
# STEP 1: Shape Inspection
# -------------------------------------------------------------------------
print("STEP 1: Shape of projects =", projects.shape)
# Explanation: Shape is (5, 2), meaning 5 project observations (rows) 
# and 2 features per project (columns: cable length and terrain index).

# -------------------------------------------------------------------------
# STEP 2: Custom Loop Means and Standard Deviations vs NumPy
# -------------------------------------------------------------------------
num_rows, num_cols = projects.shape
custom_means = []
custom_stds = []

for j in range(num_cols):
    col_data = projects[:, j]
    col_mean = sum(col_data) / num_rows
    col_var  = sum((x - col_mean) ** 2 for x in col_data) / num_rows
    custom_means.append(col_mean)
    custom_stds.append(np.sqrt(col_var))

print(f"STEP 2: Custom Means = {custom_means} | NumPy Means = {projects.mean(axis=0)}")
print(f"        Custom Stds  = {custom_stds} | NumPy Stds  = {projects.std(axis=0)}")

# -------------------------------------------------------------------------
# STEP 3: Pearson Correlation from First Principles (costs vs projects[:, 0])
# -------------------------------------------------------------------------
x = projects[:, 0]
mean_x = custom_means[0]
mean_y = sum(costs) / len(costs)

cov_xy = sum((x[i] - mean_x) * (costs[i] - mean_y) for i in range(len(costs))) / len(costs)
std_y  = np.sqrt(sum((y - mean_y) ** 2 for y in costs) / len(costs))

r_custom = cov_xy / (custom_stds[0] * std_y)
r_numpy  = np.corrcoef(costs, x)[0, 1]

print(f"STEP 3: Custom Correlation r = {r_custom:.4f} | NumPy r = {r_numpy:.4f}")

# -------------------------------------------------------------------------
# STEP 4: Matrix Predictions using Custom Loop and Vectorized @
# -------------------------------------------------------------------------
beta = np.array([0.5, 0.2])  # Made-up weights

pred_loop = np.zeros(num_rows)
for i in range(num_rows):
    pred_loop[i] = projects[i, 0] * beta[0] + projects[i, 1] * beta[1]

pred_matrix = projects @ beta

print("STEP 4: Loop Predictions   =", pred_loop)
print("        Matrix Predictions =", pred_matrix)
print("        Match check:", np.allclose(pred_loop, pred_matrix))

# -------------------------------------------------------------------------
# STEP 5: Sum of Squared Errors (SSE) Function
# -------------------------------------------------------------------------
def sse(y_actual, y_predicted):
    return np.sum((y_actual - y_predicted) ** 2)

current_sse = sse(costs, pred_matrix)
print(f"STEP 5: Current SSE = {current_sse:.4f}")

# -------------------------------------------------------------------------
# STEP 6: Finite Difference Numerical Partial Derivative wrt beta[0]
# -------------------------------------------------------------------------
h = 1e-5
beta_nudged = beta.copy()
beta_nudged[0] += h

nudged_predictions = projects @ beta_nudged
nudged_sse = sse(costs, nudged_predictions)

partial_beta_0 = (nudged_sse - current_sse) / h
print(f"STEP 6: Numerical d(SSE)/d(beta_0) = {partial_beta_0:.4f}")

# -------------------------------------------------------------------------
# STEP 7: Deliberate Shape Mismatch Error
# -------------------------------------------------------------------------
print("STEP 7: Intentional Error Triggering:")
try:
    bad_beta = np.array([0.5, 0.2, 0.9])  # Length 3 instead of 2
    invalid_pred = projects @ bad_beta
except ValueError as e:
    print("        CAUGHT EXPECTED ERROR:", e)
    print("        DIAGNOSTIC: projects has shape (5, 2); bad_beta has shape (3,).")
    print("        Matrix multiplication requires the inner dimension of projects (2)")
    print("        to match the dimension of bad_beta (3).")
```

---

---

## 8. Master Rosetta Stone & Traceback Cheat Sheet

### 8.1 Mathematical Symbol $ightarrow$ Python Loop $ightarrow$ NumPy Expression

| Mathematical Concept | LaTeX Symbol | Raw Python Loop Expression | Vectorized NumPy Expression |
| :--- | :--- | :--- | :--- |
| Vector Dot Product | $\sum_{i=1}^{p} a_i b_i$ | `sum(a[i] * b[i] for i in range(p))` | `a @ b` |
| Sample Mean | $ar{x} = rac{1}{n} \sum x_i$ | `sum(x) / len(x)` | `x.mean()` |
| Mean Centering | $x_i - ar{x}$ | `[val - mean_x for val in x]` | `x - x.mean()` |
| Sum of Squared Errors | $\sum (y_i - \hat{y}_i)^2$ | `sum((y[i] - y_hat[i])**2 for i in range(n))` | `np.sum((y - y_hat)**2)` |
| Population Variance | $rac{1}{n}\sum (x_i - ar{x})^2$ | `sum((x[i] - mean_x)**2 for i in range(n)) / n` | `x.var()` |
| Covariance | $rac{1}{n}\sum (x_i - ar{x})(y_i - ar{y})$ | `sum((x[i]-mx)*(y[i]-my) for i in range(n)) / n` | `np.cov(x, y, ddof=0)[0,1]` |
| Derivative Limit | $\lim_{h 	o 0} rac{f(x+h) - f(x)}{h}$ | `(f(x + h) - f(x)) / h` | Numerical finite difference |

---

### 8.2 Traceback Diagnostic Reference

#### 1. Shape Mismatch Error (`ValueError`)
```text
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, 
with gufunc signature (n?,k),(k?,m?)->(n?,m?) (size 3 is different from 2)
```
* **Translation:** You attempted matrix multiplication (`@`) between two arrays whose inner dimensions do not match.
* **Fix:** Check `array1.shape` and `array2.shape`. Ensure that the column count of the first array equals the row count of the second array.

#### 2. Non-existent Key Error (`KeyError`)
```text
KeyError: 'cost_per_km'
```
* **Translation:** You requested a key from a dictionary that does not exist.
* **Fix:** Check for typos in string keys or print `dictionary.keys()` to verify available labels.

#### 3. Data Type Mismatch Error (`TypeError`)
```text
TypeError: can only concatenate str (not "float") to str
```
* **Translation:** Python refused to automatically combine text strings with numeric floating-point values.
* **Fix:** Explicitly convert numbers to strings using `str(val)` or use f-strings (`f"Cost: {val}"`).
