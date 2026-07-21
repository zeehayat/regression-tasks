# Chapter 0 — From Nothing to Ready

## Level 0 Base Camp: five short days before the MHP Cost Estimator begins
<button class="read-details-btn" data-section="base-camp">✦ Read Details</button>

> **Central promise.** By the end of this chapter, a summation sign will not make you flinch, a NumPy shape error will feel like a solvable puzzle instead of a personal insult, and the sentence "take the derivative of the squared error with respect to the slope" will sound like an instruction you could actually follow. You will not yet know any machine learning. You will know enough Python, algebra, statistics, and calculus that Chapter 1 reads like a hard but fair climb, not a cliff.

This chapter exists because Chapter 1 does not slow down for you. By its second day it is talking about vectors, matrices, and tensors. By its fifth day it is deriving a matrix gradient and proving a matrix is positive semidefinite. That is exactly as it should be — this book is aiming you at research-level regression, causal inference, and survival analysis by Chapter 6, and there is no version of that destination that skips the math.

So we are going to spend five days building the floor you'll stand on. Same rules as every other chapter in this book: no black boxes, code proves every claim, you build something small each day, you break it on purpose, and you don't move on until the exit check passes.

If you already know some of this — if `for` loops and `numpy` arrays are old friends — skim fast, but still run every code block. Muscle memory matters more than novelty here.

---

## The pedagogical contract (same one Chapter 1 uses)
<button class="read-details-btn" data-section="contract">✦ Read Details</button>

1. **One central concept per day.**
2. **No mathematical hand-waving.** Every symbol gets unpacked and connected to a number you can see.
3. **Code as proof.** If a rule is true, you will watch Python confirm it.
4. **Build, break, rebuild.** Each day adds one working piece to a tiny toolkit; each day you also break something on purpose.
5. **No copy-pasting.** Type it. Your fingers need to learn this as much as your eyes do.

## Level 0 learning outcomes

At the end of five days, you should be able to:

- write and reason about Python functions, loops, comprehensions, dictionaries, and a basic class, without looking up syntax for the basics;
- create, index, slice, reshape, and broadcast NumPy arrays, and explain *why* broadcasting did what it did instead of guessing;
- compute a dot product by hand, by loop, and with `@`, and know they are the same operation at three levels of abstraction;
- read and write summation notation ($\sum$), and translate between a formula and a line of code in either direction;
- compute a mean, variance, standard deviation, covariance, and correlation coefficient from raw definitions — not from a library you don't understand yet;
- compute a derivative and a partial derivative of a simple function, both by the limit definition and by rule, and explain what "gradient" means in one sentence;
- read a Python traceback methodically instead of panicking at it; and
- set up an isolated Python environment and generate the KP datasets this book runs on.

## The five-day route

```mermaid
flowchart TD
    D0["Day 0A: Python you actually need"] --> D0B["Day 0B: NumPy, shapes, and the dot product"]
    D0B --> D0C["Day 0C: Algebra and summation notation"]
    D0C --> D0D["Day 0D: Statistics from raw definitions"]
    D0D --> D0E["Day 0E: Slope, derivatives, and surviving errors"]
```

### Minimal software setup

Same environment the rest of the book uses. If you already did this while reading the orientation chapter, skip ahead.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy matplotlib
```

Keep a scratch file — `level0_notes.py` or a notebook — and never delete a failed experiment. A wrong answer with a note explaining why it was wrong is worth more than a clean run you don't understand.

---

# Day 0A — Python You Actually Need

> **Today's central idea:** You don't need to be a "real programmer." You need six tools — variables, functions, loops, comprehensions, dictionaries, and one small class — used correctly and often.

## 0A.1 Variables and functions: naming a thought
<button class="read-details-btn" data-section="0a-1">✦ Read Details</button>

Suppose your manager in Peshawar hands you the planned capacity of one microhydro power (MHP) site: 250 kilowatts. You want to convert that to megawatts.

```python
capacity_kw = 250.0
capacity_mw = capacity_kw / 1000.0
print(capacity_mw)
# 0.25
```

That's a variable: a name you gave to a number so you don't have to keep re-typing `250.0` and hoping you remembered it correctly. Now suppose you need to do this conversion for every project that lands on your desk, forever. You don't write the same two lines a hundred times. You wrap the *behaviour* in a function, and give the *specific number* a name only when you call it.

```python
def kw_to_mw(capacity_kw):
    """Convert kilowatts to megawatts."""
    return capacity_kw / 1000.0

print(kw_to_mw(250.0))   # 0.25
print(kw_to_mw(1800.0))  # 1.8
```

A function has a contract: it takes something in, and it promises something specific out. Every regression model you build in this book — starting with `MHPCostEstimator` in Chapter 1 — is, underneath, a pile of functions with contracts like this one, just with more numbers going in.

## 0A.2 Loops: doing the same thing many times, honestly
<button class="read-details-btn" data-section="0a-2">✦ Read Details</button>

You have planned capacities for five projects and want each one in megawatts.

```python
capacities_kw = [100.0, 250.0, 500.0, 1800.0, 60.0]

capacities_mw = []
for c in capacities_kw:
    capacities_mw.append(kw_to_mw(c))

print(capacities_mw)
# [0.1, 0.25, 0.5, 1.8, 0.06]
```

Walk through this out loud (yes, actually say it): "For each value `c` in the list, convert it, and stick the result onto the end of a new list." That is the entire idea of a `for` loop. Nothing hides inside it.

## 0A.3 List comprehensions: the same loop, said more tersely
<button class="read-details-btn" data-section="0a-3">✦ Read Details</button>

Python lets you write the loop above in one line. This is not a different idea — it is the *identical* operation, just written compactly enough that you'll see it constantly in other people's code (and in later chapters of this book).

```python
capacities_mw = [kw_to_mw(c) for c in capacities_kw]
print(capacities_mw)
# [0.1, 0.25, 0.5, 1.8, 0.06]
```

Read it right to left inside the brackets: "for `c` in `capacities_kw`, compute `kw_to_mw(c)`, collect all of those." If a comprehension ever confuses you, the fix is always the same: mentally unroll it back into the `for` loop from §0A.2. They compute the same thing.

## 0A.4 Dictionaries: a record, not just a list
<button class="read-details-btn" data-section="0a-4">✦ Read Details</button>

A list of numbers loses the *meaning* of each number. A single MHP project is not just `[15.0, 100.0, 3]` — it's cable length, capacity, and terrain difficulty, and you'll forget which number is which within a week. A dictionary keeps the label attached to the value.

```python
project = {
    "site_id": "MHP-0042",
    "cable_length_km": 15.0,
    "planned_capacity_kw": 100.0,
    "terrain_index": 3,
}

print(project["planned_capacity_kw"])
# 100.0
```

This matters more than it looks like it should. Chapter 1, Day 2 introduces a "data dictionary" as part of the model itself — the discipline of never letting a raw number float around without its meaning attached starts here.

## 0A.5 A minimal class: bundling data and behaviour together
<button class="read-details-btn" data-section="0a-5">✦ Read Details</button>

A function does one thing to whatever you hand it. A class is a way of saying: "here is a *thing* — it has some data that belongs to it, and some behaviour that belongs to it." You will build classes named `MHPCostEstimator` and `GaussianOLS` later in this book. Here is the smallest possible ancestor of those, so the syntax is not a surprise when it matters.

```python
class UnitConverter:
    def __init__(self, factor):
        # This runs once, when the object is created.
        # It stores 'factor' so every method below can use it.
        self.factor = factor

    def convert(self, value):
        return value * self.factor

kw_to_mw_converter = UnitConverter(factor=0.001)
print(kw_to_mw_converter.convert(250.0))
# 0.25
```

`self` is just "this specific object." When you call `kw_to_mw_converter.convert(250.0)`, Python quietly hands the object itself in as `self`, so `self.factor` means "*this* converter's factor," not some factor floating in space. That's the whole trick. Everything else about classes in this book builds on exactly this.

## 0A.6 Build: a tiny project-cost sketch

Combine everything above into one small tool — not a real model yet, just proof you can chain these pieces together.

```python
projects = [
    {"site_id": "MHP-0001", "cable_length_km": 12.0, "cost_per_km_million_pkr": 0.9},
    {"site_id": "MHP-0002", "cable_length_km": 30.0, "cost_per_km_million_pkr": 0.9},
    {"site_id": "MHP-0003", "cable_length_km": 5.0,  "cost_per_km_million_pkr": 1.4},
]

def rough_cable_cost(project):
    return project["cable_length_km"] * project["cost_per_km_million_pkr"]

for p in projects:
    cost = rough_cable_cost(p)
    print(f"{p['site_id']}: approx {cost:.2f} million PKR of cable cost")
```

This is not a regression. It's a fixed formula, not a fitted one — the whole rest of the book is about the difference between "a formula I made up" and "a formula the data told me to use." Keep that distinction in your head; it will matter by Day 5.

## 0A.7 Break it deliberately
<button class="read-details-btn" data-section="0a-7">✦ Read Details</button>

Run each of these, read the *last line* of the error only, and write one sentence translating it into English before you fix it.

```python
# Break 1
def kw_to_mw(capacity_kw)   # missing colon
    return capacity_kw / 1000.0
```

```python
# Break 2
project = {"cable_length_km": 15.0}
print(project["cost_per_km_million_pkr"])  # key that doesn't exist
```

```python
# Break 3
capacities_kw = [100.0, 250.0, 500.0]
print(capacities_kw * 1000)   # not what you think it does
```

The third one is the important one. `capacities_kw * 1000` does not multiply every number by 1000 — it repeats the *list* a thousand times, because `*` on a Python list means "repeat," not "scale." This single misunderstanding is the entire reason NumPy exists, and it's where Day 0B starts.

### Day 0A exit check

You should be able to, without looking anything up:
- write a function with a docstring and a return value;
- write the same loop two ways (`for` and a comprehension);
- explain what `self` refers to inside a class method;
- explain, out loud, why `[1, 2, 3] * 2` is not `[2, 4, 6]`.

---

# Day 0B — NumPy, Shapes, and the Dot Product

> **Today's central idea:** A NumPy array is not "a faster list." It's a different kind of object with its own arithmetic rules, and almost every bug you'll hit in this book for the next six chapters is a shape mismatch.

## 0B.1 Why lists fail us
<button class="read-details-btn" data-section="0b-1">✦ Read Details</button>

You saw it already: `capacities_kw * 1000` repeats a list instead of scaling it. NumPy arrays fix this by defining `*`, `+`, `-`, `/` to mean *elementwise* arithmetic.

```python
import numpy as np

capacities_kw = np.array([100.0, 250.0, 500.0])
capacities_watts = capacities_kw * 1000.0
print(capacities_watts)
# [100000. 250000. 500000.]
```

## 0B.2 Shape is the first thing you check, always
<button class="read-details-btn" data-section="0b-2">✦ Read Details</button>

Every array has a `.shape`. Get comfortable reading it before you do anything else with an array.

```python
a = np.array([1.0, 2.0, 3.0])
print(a.shape)          # (3,)   — a 1-D array of 3 numbers, a "vector"

b = np.array([[1.0, 2.0, 3.0]])
print(b.shape)          # (1, 3) — a 2-D array: 1 row, 3 columns

c = np.array([[1.0], [2.0], [3.0]])
print(c.shape)          # (3, 1) — a 2-D array: 3 rows, 1 column
```

`(3,)`, `(1, 3)`, and `(3, 1)` hold the same numbers and are **not interchangeable**. This looks pedantic until it silently ruins a calculation two chapters from now. Chapter 1's design matrices depend on you already having this reflex.

## 0B.3 Indexing and slicing
<button class="read-details-btn" data-section="0b-3">✦ Read Details</button>

```python
X = np.array([
    [15.0, 100.0, 3],   # cable_km, capacity_kw, terrain_index
    [30.0, 250.0, 4],
    [5.0,  500.0, 1],
])

print(X.shape)        # (3, 3) — 3 projects, 3 features
print(X[0])            # first row: array([ 15., 100.,   3.])
print(X[:, 0])          # first column (all rows): array([15., 30.,  5.])
print(X[0, 1])          # row 0, column 1: 100.0
print(X[:2, :2])        # first two rows, first two columns
```

The comma inside `[ ]` separates "which rows" from "which columns." `:` alone means "all of them." This exact slicing vocabulary is what Chapter 1 uses to pull a single feature column out of a design matrix — get it into your hands now, not while also learning what a design matrix is.

## 0B.4 Broadcasting: NumPy's rule for mismatched shapes
<button class="read-details-btn" data-section="0b-4">✦ Read Details</button>

Broadcasting is *not* magic. It's one precise rule: NumPy compares shapes from the right, and a dimension of size 1 (or a missing dimension) is allowed to stretch to match. Watch it work, then watch it correctly refuse.

```python
X = np.array([
    [15.0, 100.0],
    [30.0, 250.0],
    [5.0,  500.0],
])   # shape (3, 2): cable_km, capacity_kw

means = np.array([16.7, 283.3])   # shape (2,) — one mean per column

centred = X - means
print(centred)
# Each row of X has 'means' subtracted from it, column by column.
```

`(3, 2)` and `(2,)` are compatible because the trailing dimension `2` matches `2`. Now break it on purpose:

```python
weights = np.array([1.5, 0.8, 2.1])   # shape (3,) — wrong length on purpose

result = X @ weights
```

```text
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0,
with gufunc signature (n?,k),(k?,m?)->(n?,m?) (size 3 is different from 2)
```

Translate the last line, not the jargon: *X has 2 columns; `weights` has 3 entries; matrix multiplication needs those to match.* Fix it by giving `weights` exactly 2 entries, one per column of `X`. This is the single most common error you will produce for the next six chapters, so get used to reading it calmly.

## 0B.5 The dot product, at three levels of honesty
<button class="read-details-btn" data-section="0b-5">✦ Read Details</button>

A "weighted sum" is the single most important operation in this entire book — every regression coefficient you will ever fit multiplies a feature by a weight and adds the results up. Here it is proven three ways, so you know they're the same thing.

```python
features = np.array([15.0, 100.0])       # cable_km, capacity_kw
weights  = np.array([0.05, 0.002])       # million PKR per unit

# Level 1: by hand, term by term
manual = features[0] * weights[0] + features[1] * weights[1]

# Level 2: by an explicit loop (works for any length)
loop_total = 0.0
for f, w in zip(features, weights):
    loop_total += f * w

# Level 3: NumPy's dot product
dot_total = features @ weights   # equivalently: np.dot(features, weights)

print(manual, loop_total, dot_total)
# 0.95 0.95 0.95
```

All three give `0.95` — 0.75 (cable contribution) + 0.2 (capacity contribution) million PKR. When Chapter 1 writes `X @ beta`, this is exactly what it means: do this weighted sum, once per row of `X`, all at once.

## 0B.6 Build: a manual prediction, matrix style
<button class="read-details-btn" data-section="0b-6">✦ Read Details</button>

```python
X = np.array([
    [15.0, 100.0],
    [30.0, 250.0],
    [5.0,  500.0],
])   # 3 projects, 2 features each

beta = np.array([0.05, 0.002])   # made-up weights, not fitted — just practising the mechanics

predictions = X @ beta
print(predictions)
# one predicted cost per project, in the same units as beta implies
```

## 0B.7 Break it deliberately

```python
row_vector = np.array([[1.0, 2.0, 3.0]])   # shape (1, 3)
col_vector = np.array([[1.0], [2.0], [3.0]])  # shape (3, 1)

print(row_vector + col_vector)
```

This does **not** error — it broadcasts to a `(3, 3)` result, which surprises almost everyone the first time. Before running it, predict the shape of the output. Then run it and see if you were right. If you were wrong, walk through §0B.4's rule again until the result stops being a surprise.

### Day 0B exit check

You should be able to, without looking anything up:
- state the shape of any array you just created, without running `.shape`, and then confirm it;
- explain broadcasting as one sentence about matching dimensions from the right;
- compute a dot product three ways and get the same number each time;
- read a `matmul` shape-mismatch error and say, in plain English, which two numbers disagree.

---

# Day 0C — Algebra and Summation Notation

> **Today's central idea:** $\sum$ is not a foreign symbol. It's a `for` loop that adds things up, written in a more compressed alphabet.

## 0C.1 A function is a recipe with a name
<button class="read-details-btn" data-section="0c-1">✦ Read Details</button>

You already know this from Python — `def kw_to_mw(capacity_kw): return capacity_kw / 1000.0` is a recipe. Algebra just writes recipes with single letters instead of English words:

$$f(x) = \frac{x}{1000}$$

says exactly what the Python function said: give me a number $x$, I'll divide it by 1000 and call the result $f(x)$. Every formula in this book is a recipe like this one — the discipline is learning to read the recipe before panicking about the symbols.

## 0C.2 The equation of a line, and why it matters here
<button class="read-details-btn" data-section="0c-2">✦ Read Details</button>

$$y = mx + c$$

$m$ is the slope: how much $y$ changes when $x$ increases by exactly 1. $c$ is the intercept: the value of $y$ when $x$ is 0. This is the entire idea behind linear regression with one feature — Chapter 1 spends its first three days building up to exactly this equation, generalised to many features. If you can already sketch $y = 2x + 5$ on paper and say what happens as $x$ grows, you have the geometric intuition; the rest is notation.

## 0C.3 Summation notation, built from a loop you already wrote
<button class="read-details-btn" data-section="0c-3">✦ Read Details</button>

Recall Day 0A's loop:

```python
loop_total = 0.0
for f, w in zip(features, weights):
    loop_total += f * w
```

In algebra, if there are $p$ features indexed $i = 1, 2, \dots, p$, this exact loop is written:

$$\sum_{i=1}^{p} f_i w_i$$

Read it left to right, the same way you'd read the `for` loop: "start a running total at 0; for each $i$ from 1 to $p$, add $f_i$ times $w_i$; stop." The little number under $\sum$ is where the loop starts, the number on top is where it ends, and everything to the right of $\sum$ is what gets added at each step.

**Code proof — the formula and the loop must agree:**

```python
import numpy as np

f = np.array([15.0, 100.0, 3.0])
w = np.array([0.05, 0.002, -1.5])

# The formula, as a loop
total_loop = 0.0
for i in range(len(f)):
    total_loop += f[i] * w[i]

# The formula, as NumPy
total_numpy = f @ w

print(total_loop, total_numpy)
# 0.55 0.55
```

$\sum_{i=1}^{p} f_i w_i$, the `for` loop, and `f @ w` are three spellings of one idea. When Chapter 1, Day 4 writes the sum of squared residuals as

$$\text{SSR} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

you should now be able to read it as: "loop over every project $i$; take the actual cost minus the predicted cost; square it; add it to a running total." Try writing that as Python *before* Chapter 1 does it for you.

## 0C.4 Exponents and why we square errors (a preview)
<button class="read-details-btn" data-section="0c-4">✦ Read Details</button>

$x^2$ means $x$ multiplied by itself. Two properties matter for everything downstream:

- squaring always produces a non-negative number, so a positive error and an equally-sized negative error contribute the same amount once squared;
- squaring punishes large errors much more than small ones — an error of 10 contributes 100, an error of 2 contributes only 4.

Chapter 1, Day 4 spends real time justifying *why* regression squares errors instead of just adding them up. You don't need the full argument yet — just the two facts above, so the argument doesn't start from zero.

## 0C.5 Build: SSR from the formula, three ways
<button class="read-details-btn" data-section="0c-5">✦ Read Details</button>

```python
y_actual    = np.array([12.0, 30.0, 8.0])
y_predicted = np.array([11.2, 31.5, 9.0])

# Level 1: by hand
errors = y_actual - y_predicted
squared_errors = errors ** 2
ssr_manual = squared_errors.sum()

# Level 2: as a literal loop mirroring the sigma
ssr_loop = 0.0
for a, p in zip(y_actual, y_predicted):
    ssr_loop += (a - p) ** 2

# Level 3: one line
ssr_oneline = np.sum((y_actual - y_predicted) ** 2)

print(ssr_manual, ssr_loop, ssr_oneline)
```

All three must match. If they don't, you have a bug — and finding it is exactly the kind of debugging Chapter 1 will ask of you constantly.

## 0C.6 Break it deliberately
<button class="read-details-btn" data-section="0c-6">✦ Read Details</button>

Predict the output on paper before running:

```python
values = np.array([2.0, -3.0, 5.0])
print(np.sum(values ** 2))     # sum of squares
print(np.sum(values) ** 2)     # square of the sum
```

These are **not equal**, and the difference between "sum of squares" and "square of a sum" is a mistake that will cost you real debugging time in later chapters (variance, in particular, depends on getting this order right). Say out loud why they differ before moving on.

### Day 0C exit check

You should be able to:
- translate a $\sum$ formula into a `for` loop and into one line of NumPy, and get matching numbers all three ways;
- explain why $y = mx + c$ is the same shape of idea as Chapter 1's regression equation;
- explain the difference between $\sum x_i^2$ and $(\sum x_i)^2$ using a concrete 3-number example.

---

# Day 0D — Statistics From Raw Definitions

> **Today's central idea:** Mean, variance, and correlation are not library functions you call. They are formulas you can build from Day 0C's summation notation in about four lines each — and Chapter 2's probability content assumes you already know what they mean.

## 0D.1 The mean: the balancing point
<button class="read-details-btn" data-section="0d-1">✦ Read Details</button>

Suppose five MHP projects had these actual costs, in million PKR: `[12.0, 30.0, 8.0, 45.0, 15.0]`.

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

"Add everything up, divide by how many there are."

```python
costs = np.array([12.0, 30.0, 8.0, 45.0, 15.0])

n = len(costs)
mean_manual = np.sum(costs) / n
mean_numpy = costs.mean()

print(mean_manual, mean_numpy)
# 22.0 22.0
```

## 0D.2 Variance: how far, on average, things sit from the mean
<button class="read-details-btn" data-section="0d-2">✦ Read Details</button>

You can't just average the raw deviations from the mean — they always sum to zero (that's *what* "mean" means). So we square them first, borrowing exactly the trick from Day 0C.

$$\sigma^2 = \frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})^2$$

```python
deviations = costs - costs.mean()
print(deviations)               # [-10. 8. -14. 23. -7.]
print(deviations.sum())          # 0.0 — always, by construction

variance_manual = np.sum(deviations ** 2) / n
variance_numpy = costs.var()     # NumPy's default matches this population formula

print(variance_manual, variance_numpy)
```

> **A warning that will save you real confusion later:** you'll sometimes see variance computed by dividing by $n - 1$ instead of $n$ (the "sample" variance, `ddof=1` in NumPy). Chapter 2, Day 6 explains exactly why that correction exists. For now, just notice that `costs.var()` and `costs.var(ddof=1)` disagree, and don't assume it's a bug when you see it — it's a deliberate choice depending on what you're trying to estimate.

## 0D.3 Standard deviation: getting the units back
<button class="read-details-btn" data-section="0d-3">✦ Read Details</button>

Variance is in squared units (million PKR², which means nothing to anyone). Standard deviation undoes the squaring:

$$\sigma = \sqrt{\sigma^2}$$

```python
std_manual = np.sqrt(variance_manual)
std_numpy = costs.std()
print(std_manual, std_numpy)
```

## 0D.4 Covariance: do two things move together?
<button class="read-details-btn" data-section="0d-4">✦ Read Details</button>

Now bring in a second variable — cable length per project, in km: `[12.0, 30.0, 5.0, 40.0, 15.0]`. Covariance asks: when one variable is above its mean, is the other one usually above its mean too?

$$\text{cov}(x, y) = \frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$$

```python
cable_km = np.array([12.0, 30.0, 5.0, 40.0, 15.0])

x_dev = costs - costs.mean()
y_dev = cable_km - cable_km.mean()

cov_manual = np.sum(x_dev * y_dev) / n
cov_numpy = np.cov(costs, cable_km, ddof=0)[0, 1]

print(cov_manual, cov_numpy)
```

If both deviations tend to have the same sign (both above their means, or both below), each product is positive and covariance comes out positive: as cable length rises, cost tends to rise too. That's not a coincidence for this dataset — it's the whole reason regression will later find cable length a useful feature.

## 0D.5 Correlation: covariance with the units washed out
<button class="read-details-btn" data-section="0d-5">✦ Read Details</button>

Covariance's *size* depends on the units you happened to measure in, which makes two covariances hard to compare. Correlation fixes that by dividing out each variable's own spread:

$$r = \frac{\text{cov}(x, y)}{\sigma_x \sigma_y}$$

```python
r_manual = cov_manual / (costs.std() * cable_km.std())
r_numpy = np.corrcoef(costs, cable_km)[0, 1]

print(r_manual, r_numpy)
```

$r$ always sits between $-1$ and $1$. Close to $1$: strong positive relationship. Close to $-1$: strong negative. Close to $0$: little linear relationship. This single number is going to matter enormously in Chapter 1, Day 2, where you'll learn that two *nearly identical* correlated features (like literacy rate and school enrollment in the orientation chapter's development dataset) can quietly break a regression — that's called multicollinearity, and it's just "correlation close to 1" between features, wearing a bigger hat.

## 0D.6 Build: a tiny stats report

```python
def describe(values, label):
    print(f"{label}: mean={values.mean():.2f}, std={values.std():.2f}, "
          f"min={values.min():.2f}, max={values.max():.2f}")

describe(costs, "cost (million PKR)")
describe(cable_km, "cable length (km)")
print(f"correlation(cost, cable_km) = {np.corrcoef(costs, cable_km)[0,1]:.3f}")
```

This four-line report is, in essence, what Exercise 0.1 in the orientation chapter asked you to build for the full KP datasets. Now you know what every number in it actually means, instead of trusting `.describe()` blindly.

## 0D.7 Break it deliberately
<button class="read-details-btn" data-section="0d-7">✦ Read Details</button>

```python
constant = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
print(np.corrcoef(constant, costs))
```

This produces `nan`, not an error — correlation divides by `constant.std()`, which is `0.0`, and dividing by zero silently poisons the whole calculation with `nan` instead of crashing loudly. This is a real trap: a feature with zero variance (every project has the same terrain grade, say) will not throw an exception, it will quietly corrupt anything downstream that touches it. Get used to checking `.std()` for zero *before* you trust a correlation number.

### Day 0D exit check

You should be able to:
- compute mean, variance, standard deviation, covariance, and correlation from the raw summation formulas, and match NumPy's built-ins;
- explain in one sentence why variance squares the deviations instead of just averaging them;
- explain what a correlation near $1$ between two features will eventually mean for a regression model;
- explain why a zero-variance feature produces `nan` instead of an error, and why that's more dangerous.

---

# Day 0E — Slope, Derivatives, and Surviving Errors

> **Today's central idea:** A derivative is just the slope of a curve at one exact point, and every model you train in this book is, underneath, a search for the point where that slope hits zero.

## 0E.1 From a straight slope to a curved one
<button class="read-details-btn" data-section="0e-1">✦ Read Details</button>

You already know $y = mx + c$ has one constant slope, $m$, everywhere. Now consider a curve: $y = x^2$. Its steepness is different at every point — flat near $x=0$, steep far from it. The **derivative** is a formula that tells you the exact slope at any single point you pick.

For $y = x^2$, the derivative is:

$$\frac{dy}{dx} = 2x$$

- At $x = 0$: slope is $2(0) = 0$ — flat, the bottom of the bowl.
- At $x = 3$: slope is $2(3) = 6$ — climbing steeply.
- At $x = -3$: slope is $2(-3) = -6$ — descending steeply, mirror image.

## 0E.2 Where that formula actually comes from (the limit definition)
<button class="read-details-btn" data-section="0e-2">✦ Read Details</button>

You don't have to memorise derivative rules as magic — they fall out of one idea: *look at the average slope between two points that are getting closer and closer together.*

$$\frac{dy}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Code proof — approximate this numerically and watch it converge to $2x$:**

```python
def f(x):
    return x ** 2

def numerical_slope(f, x, h):
    return (f(x + h) - f(x)) / h

x = 3.0
for h in [1.0, 0.1, 0.01, 0.0001, 0.000001]:
    print(f"h={h:<10} slope estimate={numerical_slope(f, x, h):.6f}")
# As h shrinks, the estimate creeps toward exactly 6.0 = 2 * 3
```

This is not a party trick. Chapter 1, Day 5 checks its hand-derived matrix gradient against exactly this kind of numerical approximation ("finite differences"), and Chapter 2, Day 9 builds gradient descent directly on top of it. You just ran a miniature version of both.

## 0E.3 Two derivative rules you'll actually use
<button class="read-details-btn" data-section="0e-3">✦ Read Details</button>

You don't need a full calculus course — this book only leans on a handful of rules, repeatedly:

- **Power rule:** the derivative of $x^n$ is $n x^{n-1}$. (This is where $x^2 \to 2x$ came from.)
- **Constant multiple:** the derivative of $a \cdot f(x)$ is $a$ times the derivative of $f(x)$.
- **Sum rule:** the derivative of $f(x) + g(x)$ is just the derivative of $f(x)$ plus the derivative of $g(x)$ — you can differentiate term by term.

```python
# Example: derivative of  y = 3x^2 + 5x
# Power rule on 3x^2  -> 6x
# Power rule on 5x    -> 5   (since x^1's derivative is 1*x^0 = 1)
# Sum rule combines them -> 6x + 5

def y(x):
    return 3 * x**2 + 5 * x

def dy_dx_by_rule(x):
    return 6 * x + 5

x = 2.0
print(dy_dx_by_rule(x))                                  # 17.0
print(numerical_slope(y, x, h=0.000001))                  # ~17.0, confirms the rule
```

## 0E.4 Partial derivatives: slope when there's more than one dial to turn
<button class="read-details-btn" data-section="0e-4">✦ Read Details</button>

Regression rarely has one adjustable number — it has one per feature. A **partial derivative** asks: "if I nudge *just this one* number and freeze everything else, how does the output change?" Notation-wise, $\partial$ replaces $d$ to signal "there are other variables here, and I'm holding them still."

Take a toy error surface: $E(m, c) = (10 - (m \cdot 2 + c))^2$ — the squared error of predicting $y=10$ at $x=2$ using slope $m$ and intercept $c$.

```python
def error(m, c):
    prediction = m * 2 + c
    return (10 - prediction) ** 2

def partial_wrt_m(error_fn, m, c, h=1e-6):
    return (error_fn(m + h, c) - error_fn(m, c)) / h

def partial_wrt_c(error_fn, m, c, h=1e-6):
    return (error_fn(m, c + h) - error_fn(m, c)) / h

m, c = 1.0, 1.0
print("slope if we nudge m:", partial_wrt_m(error, m, c))
print("slope if we nudge c:", partial_wrt_c(error, m, c))
```

Both numbers together form the **gradient** — a small arrow made of "how much does the error change per dial, holding the others still." Chapter 1, Day 5 derives this gradient by hand for the full OLS objective, with one partial derivative per feature instead of two. You just did the two-dial version.

## 0E.5 Why any of this matters: descending toward zero error
<button class="read-details-btn" data-section="0e-5">✦ Read Details</button>

The core trick behind fitting almost every model in this book: start somewhere, compute the gradient, take a small step in the direction that *decreases* error, and repeat until the gradient is (close to) zero.

```python
m, c = 0.0, 0.0        # start with a bad guess
learning_rate = 0.01

for step in range(50):
    grad_m = partial_wrt_m(error, m, c)
    grad_c = partial_wrt_c(error, m, c)
    m -= learning_rate * grad_m
    c -= learning_rate * grad_c

print(f"m={m:.3f}, c={c:.3f}, error={error(m, c):.6f}")
```

Run it and watch the error shrink toward zero as `m` and `c` settle into values that make `m * 2 + c` land near `10`. This is a hand-built, miniature version of gradient descent — the real one arrives in Chapter 2, Day 9, with real data instead of one toy point.

## 0E.6 Surviving the traceback
<button class="read-details-btn" data-section="0e-6">✦ Read Details</button>

You will see red error text constantly for the rest of this book. A practitioner reads an error the way an investigator reads a scene: bottom line first, ignore the jargon, translate to English.

**Shape mismatch (you already met this on Day 0B):**

```text
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0...
(size 4 is different from 2)
```
→ "Two things that need matching lengths don't have matching lengths."

**Missing key:**

```python
project = {"cable_length_km": 15.0}
project["terrain_index"]
```
```text
KeyError: 'terrain_index'
```
→ "You asked the dictionary for a label it doesn't have. Check spelling, check whether the data actually contains that field."

**Wrong type:**

```python
"cost: " + 15.0
```
```text
TypeError: can only concatenate str (not "float") to str
```
→ "Python won't silently guess how to combine a string and a number. Convert one of them explicitly: `"cost: " + str(15.0)`."

The instinct to panic and immediately paste the whole traceback into a search bar is understandable but skips the useful step: read the last line first, translate it into a plain sentence, *then* decide if you need help.

## 0E.7 The rule of the rubber duck

Put a mug, a duck, or an unimpressed cat on your desk. When something breaks, explain your code to it out loud, line by line, as if it has never seen Python. "Okay, here I create an array with three rows. Then I multiply it by a vector with... wait, how many entries does that vector have?" In a large fraction of cases you will catch the bug mid-sentence, before the duck says a word.

## 0E.8 Break it deliberately

```python
def error(m, c):
    prediction = m * 2 + c
    return (10 - prediction) ** 2

# Break: learning rate far too large
m, c = 0.0, 0.0
learning_rate = 5.0   # was 0.01

for step in range(10):
    grad_m = partial_wrt_m(error, m, c)
    grad_c = partial_wrt_c(error, m, c)
    m -= learning_rate * grad_m
    c -= learning_rate * grad_c
    print(step, m, c, error(m, c))
```

Watch the error *grow* instead of shrink, possibly exploding toward infinity. You just reproduced, in miniature, the single most common failure in Chapter 2, Day 9 — a learning rate that's too large overshoots the bottom of the valley on every step instead of settling into it. File that feeling away; you'll need to recognise it fast later.

### Day 0E exit check

You should be able to:
- compute a derivative numerically (finite differences) and confirm it against a rule-based derivative;
- explain a partial derivative in one sentence to someone who's never seen the word;
- explain, without notes, what gradient descent is doing and why a too-large learning rate breaks it;
- read a `KeyError`, `TypeError`, and shape-mismatch `ValueError` and translate each into plain English before trying to fix it.

---

# Level 0 Capstone — Prove You're Ready
<button class="read-details-btn" data-section="capstone">✦ Read Details</button>

Do this without peeking at earlier sections. This mirrors the format every later chapter uses for its own capstone, so it's worth taking seriously now.

**Given:**

```python
projects = np.array([
    [12.0, 15.0],   # cable_km, terrain_index
    [30.0, 25.0],
    [5.0,  8.0],
    [40.0, 45.0],
    [15.0, 12.0],
])
costs = np.array([12.0, 30.0, 8.0, 45.0, 15.0])   # million PKR
```

1. Report the shape of `projects` and explain what each dimension means, in a sentence.
2. Compute the mean and standard deviation of each column of `projects`, without using `.mean(axis=...)` — write the loop or comprehension yourself first, then confirm with NumPy.
3. Compute the correlation between `costs` and column 0 of `projects` from the raw covariance/std formulas, then check it with `np.corrcoef`.
4. Pick any made-up weight vector `beta` of length 2 and compute `predictions = projects @ beta` by hand-loop and with `@`, and confirm they match.
5. Write a function `sse(y_actual, y_predicted)` that returns $\sum (y_i - \hat{y}_i)^2$, and test it against your `predictions` from step 4.
6. Using finite differences, numerically estimate how `sse` changes if you nudge just the first entry of `beta` by a small amount. (You now have one entry of the gradient Chapter 1, Day 5 will derive properly.)
7. Deliberately break something — pass `beta` with the wrong length into step 4 — and write, in one sentence, what the resulting error is telling you.

If you can do all seven without reopening this chapter, you are ready for Chapter 1.

---

## Glossary (symbols you'll now recognise on sight)

| Symbol | Name | Meaning |
|---|---|---|
| $x_i$ | indexed variable | the $i$-th value in a collection |
| $\sum_{i=1}^{n}$ | summation | add up a formula for every $i$ from 1 to $n$ |
| $\bar{x}$ | mean | average value |
| $\sigma^2$ | variance | average squared distance from the mean |
| $\sigma$ | standard deviation | square root of variance; same units as the data |
| $r$ | correlation | covariance rescaled to sit between $-1$ and $1$ |
| $\frac{dy}{dx}$, $f'(x)$ | derivative | exact slope of $y=f(x)$ at a point |
| $\frac{\partial E}{\partial m}$ | partial derivative | slope of $E$ with respect to $m$ alone, holding other variables fixed |
| gradient | — | the collection of all partial derivatives, one per adjustable number |

## Instructor and self-study notes

- **Suggested timebox:** one focused day per section (0A–0E), or two half-days if math is genuinely new to you. Do not compress this into a single afternoon — the exit checks exist because rushing here is exactly what makes Chapter 1 feel impossible later.
- **Most common failure point:** learners skim Day 0B's broadcasting section because arrays "look like lists." Don't. Every day from Chapter 1 onward assumes shape-checking is reflexive, not effortful.
- **If the capstone is a struggle:** repeat Day 0C and 0D before moving on. Chapter 1 will not re-teach summation notation or variance; it will simply use them.

## Where Chapter 1 begins

Chapter 1 opens with a question you're now equipped to actually think about: *what is a regression for — prediction, explanation, or causation?* Everything you built this week — functions, shapes, sums, means, slopes — is about to get one name each: features, design matrix, objective function, gradient. Turn the page.
<style>
.read-details-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(34, 211, 238, 0.08);
    border: 1px solid rgba(34, 211, 238, 0.3);
    color: #22d3ee;
    padding: 0.4rem 0.8rem;
    font-size: 0.75rem;
    font-weight: 600;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: all 0.2s ease;
    margin: 0.5rem 0 1rem 0;
    font-family: inherit;
}
.read-details-btn:hover {
    background: rgba(34, 211, 238, 0.2);
    border-color: #22d3ee;
    box-shadow: 0 0 10px rgba(34, 211, 238, 0.2);
}
.comp-modal {
    display: none;
    position: fixed;
    z-index: 10000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background-color: rgba(11, 15, 25, 0.85);
    backdrop-filter: blur(8px);
    opacity: 0;
    transition: opacity 0.3s ease;
    align-items: center;
    justify-content: center;
}
.comp-modal.open {
    display: flex;
    opacity: 1;
}
.comp-modal-content {
    background-color: #151d30;
    border: 1px solid #223150;
    border-radius: 0.75rem;
    width: 90%;
    max-width: 48rem;
    max-height: 85vh;
    overflow-y: auto;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.6), 0 10px 10px -5px rgba(0, 0, 0, 0.6);
    position: relative;
    padding: 2.5rem 2rem 2rem 2rem;
    transform: scale(0.95);
    transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
.comp-modal.open .comp-modal-content {
    transform: scale(1);
}
.comp-modal-close {
    position: absolute;
    top: 0.75rem;
    right: 1.25rem;
    color: #cbd5e1;
    font-size: 2rem;
    font-weight: 300;
    cursor: pointer;
    transition: color 0.2s ease;
    line-height: 1;
}
.comp-modal-close:hover {
    color: #fff;
}
#comp-modal-body {
    color: #cbd5e1;
    font-size: 0.95rem;
}
#comp-modal-body h1, #comp-modal-body h2 {
    color: #fff;
    border-bottom: 1px solid #223150;
    padding-bottom: 0.5rem;
    margin-top: 0;
    margin-bottom: 1.5rem;
}
#comp-modal-body h3, #comp-modal-body h4 {
    color: #fff;
    margin-top: 1.75rem;
    margin-bottom: 0.75rem;
}
#comp-modal-body p {
    margin-bottom: 1rem;
}
#comp-modal-body code {
    color: #67e8f9;
    background: #0b0f19;
    padding: 0.125rem 0.375rem;
    border-radius: 0.25rem;
    font-size: 0.875rem;
}
#comp-modal-body pre {
    background: #0b0f19;
    border: 1px solid #223150;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
    overflow-x: auto;
}
#comp-modal-body pre code {
    background: transparent;
    padding: 0;
    color: #e2e8f0;
}
#comp-modal-body table {
    width: 100%;
    font-size: 0.875rem;
    border-collapse: collapse;
    margin: 1.5rem 0;
}
#comp-modal-body th, #comp-modal-body td {
    border: 1px solid #223150;
    padding: 0.5rem 0.75rem;
    text-align: left;
}
#comp-modal-body th {
    background: #151d30;
    color: #fff;
    font-weight: 600;
}
.companion-details-area, .companion-details-area ~ * {
    display: none;
}
</style>

<div id="companion-modal" class="comp-modal">
    <div class="comp-modal-content">
        <span class="comp-modal-close">&times;</span>
        <div id="comp-modal-body"></div>
    </div>
</div>

<script>
document.addEventListener("DOMContentLoaded", function() {
    var store = {};
    var area = document.querySelector('.companion-details-area');
    if (!area) return;
    
    var currentSection = null;
    var currentElements = [];
    var sibling = area.nextElementSibling;
    
    while (sibling) {
        if (sibling.classList.contains('companion-detail-heading')) {
            if (currentSection) {
                store[currentSection] = currentElements;
            }
            currentSection = sibling.getAttribute('data-section');
            currentElements = [];
        } else {
            if (currentSection) {
                currentElements.push(sibling.cloneNode(true));
            }
        }
        sibling = sibling.nextElementSibling;
    }
    if (currentSection) {
        store[currentSection] = currentElements;
    }
    
    // Wire up buttons
    document.querySelectorAll('.read-details-btn').forEach(function(btn) {
        btn.addEventListener('click', function() {
            var section = this.getAttribute('data-section');
            var elements = store[section];
            var modalBody = document.getElementById('comp-modal-body');
            modalBody.innerHTML = '';
            
            if (elements && elements.length > 0) {
                elements.forEach(function(el) {
                    el.style.display = '';
                    modalBody.appendChild(el);
                });
                
                // Trigger MathJax typesetting if loaded
                if (window.MathJax && window.MathJax.typesetPromise) {
                    window.MathJax.typesetPromise([modalBody]);
                }
            } else {
                modalBody.innerHTML = '<p>No details found for this section.</p>';
            }
            
            var modal = document.getElementById('companion-modal');
            modal.style.display = 'flex';
            // Reflow
            modal.offsetHeight;
            modal.classList.add('open');
            document.body.style.overflow = 'hidden';
        });
    });
    
    // Close modal function
    function closeModal() {
        var modal = document.getElementById('companion-modal');
        modal.classList.remove('open');
        setTimeout(function() {
            modal.style.display = 'none';
            document.body.style.overflow = '';
        }, 300);
    }
    
    document.querySelector('.comp-modal-close').addEventListener('click', closeModal);
    window.addEventListener('click', function(event) {
        var modal = document.getElementById('companion-modal');
        if (event.target === modal) {
            closeModal();
        }
    });
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            closeModal();
        }
    });
});
</script>


---

# Companion Details Area {: .companion-details-area}
## Detail BASE CAMP {: .companion-detail-heading data-section="base-camp"}
## Overview

This chapter serves as the foundational preparation for regression analysis in Machine Learning. The tutorial is structured to build essential skills in Python, mathematical notation, statistics, and calculus over five focused days.

## Pedagogical Framework

## 1. Pedagogical Philosophy & Operational Contract

The primary goal of Chapter 0 is to construct the mathematical, computational, and statistical intuition required before encountering matrix regression, multivariate calculus, and statistical inference in Chapter 1.


## Detail CONTRACT {: .companion-detail-heading data-section="contract"}
### Learning Contract
The tutorial follows specific principles:
1. **One central concept per day** – Each day focuses on a single foundational concept
2. **No mathematical hand-waving** – Every symbol is connected to concrete computations
3. **Code as proof** – Every mathematical concept is verified with Python code
4. **Build, break, rebuild** – Learners create, intentionally break, and fix code to build understanding
5. **No copy-pasting** – Learners type everything, building muscle memory

## 1. Pedagogical Philosophy & Operational Contract

The primary goal of Chapter 0 is to construct the mathematical, computational, and statistical intuition required before encountering matrix regression, multivariate calculus, and statistical inference in Chapter 1.


## Detail 0A 1 {: .companion-detail-heading data-section="0a-1"}
#### Variables and Functions
**Core Concept**: Variables are containers for storing data, while functions encapsulate reusable operations.

**Python Implementation**:
```python
# Variable definition
capacity_kw = 250.0

# Function definition with docstring
def kw_to_mw(capacity_kw):
    """Convert kilowatts to megawatts."""
    return capacity_kw / 1000.0
```

**Algebraic Analogy**: Functions in mathematics follow the same pattern: f(x) = x/1000 describes a transformation rule applied to input values.

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


## Detail 0A 2 {: .companion-detail-heading data-section="0a-2"}
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


## Detail 0A 3 {: .companion-detail-heading data-section="0a-3"}
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


## Detail 0A 4 {: .companion-detail-heading data-section="0a-4"}
#### Dictionary Data Structures
**Core Concept**: Key-value pairs that preserve context and meaning.

**Python Structure**:
```python
project = {
    "site_id": "MHP-0042",
    "cable_length_km": 15.0,
    "planned_capacity_kw": 100.0,
    "terrain_index": 3,
}

# Access with keys
print(project["planned_capacity_kw"])  # Returns 100.0
```

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


## Detail 0A 5 {: .companion-detail-heading data-section="0a-5"}
#### Object-Oriented Programming Basics
**Core Concept**: Classes bundle data and behavior together.

**Python Class Structure**:
```python
class UnitConverter:
    def __init__(self, factor):
        # Initialize with factor
        self.factor = factor
    
    def convert(self, value):
        # Define conversion method
        return value * self.factor
```

**Key Concept**: The self parameter represents the specific object instance, enabling each instance to maintain its own state.

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


## Detail 0A 7 {: .companion-detail-heading data-section="0a-7"}
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


## Detail 0B 1 {: .companion-detail-heading data-section="0b-1"}
#### Array Fundamentals
**Core Concept**: NumPy arrays provide efficient numerical computation with elementwise operations.

**Dimensional Structure**:
```python
import numpy as np

# 1D array - vector
vector_1d = np.array([1.0, 2.0, 3.0])
print(vector_1d.shape)  # (3,)

# 2D array - matrix
matrix_2d = np.array([[1.0, 2.0, 3.0]])
print(matrix_2d.shape)  # (1, 3)

# Transposed structure
matrix_transposed = np.array([[1.0], [2.0], [3.0]])
print(matrix_transposed.shape)  # (3, 1)
```

**Critical Note**: (3,), (1, 3), and (3, 1) are dimensionally incompatible despite holding the same values.

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


## Detail 0B 2 {: .companion-detail-heading data-section="0b-2"}
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


## Detail 0B 3 {: .companion-detail-heading data-section="0b-3"}
#### Indexing and Slicing
**Core Concept**: Accessing array elements via row/column specifications.

**Slicing Syntax**: array[row_slice, column_slice]
```python
X = np.array([
    [15.0, 100.0, 3],   # cable_km, capacity_kw, terrain_index
    [30.0, 250.0, 4],
    [5.0,  500.0, 1],
])

# Row operations
print(X[0])  # First row: [15., 100., 3.]

# Column operations
print(X[:, 0])  # All rows, first column: [15., 30., 5.]

# Specific element
print(X[0, 1])  # Row 0, column 1: 100.0
```

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


## Detail 0B 4 {: .companion-detail-heading data-section="0b-4"}
#### Broadcasting Rules
**Core Concept**: Automatic shape alignment for operations with compatible dimensions.

**Compatibility Pattern**: Match from the trailing (rightmost) dimension.
```python
X = np.array([
    [15.0, 100.0],
    [30.0, 250.0],
    [5.0,  500.0],
])  # shape (3, 2)

means = np.array([16.7, 283.3])  # shape (2,)

# Broadcasting occurs
result = X - means  # Each row subtracts the mean vector
```

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
2. Shape of `means`: `(2,)` $
ightarrow$ Right-aligned: `(1, 2)`
3. Compare trailing dimension: $2 == 2$ (Match!)
4. Compare leading dimension: $3$ vs $1$. The dimension of size $1$ stretches from $1 
ightarrow 3$.
5. Final broadcasted subtraction subtracts `16.7` from Column 0 and `283.3` from Column 1 for all rows.

---


## Detail 0B 5 {: .companion-detail-heading data-section="0b-5"}
#### Dot Products and Matrix Multiplication
**Core Concept**: Weighted sum operation at the heart of linear regression.

**Three-Level Implementation**:
```python
features = np.array([15.0, 100.0])
weights  = np.array([0.05, 0.002])

# Level 1: Manual computation
manual = features[0] * weights[0] + features[1] * weights[1]

# Level 2: Explicit loop
loop_total = 0.0
for f, w in zip(features, weights):
    loop_total += f * w

# Level 3: NumPy operator
dot_total = features @ weights  # np.dot(features, weights)
```


## Detail 0B 6 {: .companion-detail-heading data-section="0b-6"}
### 3.6 Matrix Prediction ($X\boldsymbol{\beta}$)

When we multiply a design matrix $X$ of shape $(N 	imes P)$ by a parameter vector $\boldsymbol{\beta}$ of shape $(P,)$ using the `@` operator, NumPy performs a dot product between **each row of $X$** and the weight vector $\boldsymbol{\beta}$.

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


## Detail 0C 1 {: .companion-detail-heading data-section="0c-1"}
### 4.1 Functions and Equations of Lines

A linear relationship between an input variable $x$ and an output variable $y$ is expressed as:

$$y = m x + c$$

* $m$: **Slope** — The change in $y$ given a $1$-unit increase in $x$ ($\frac{\Delta y}{\Delta x}$).
* $c$: **Intercept** — The value of $y$ when $x = 0$.

In multiple regression, we extend this equation to $P$ features:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p$$

---


## Detail 0C 2 {: .companion-detail-heading data-section="0c-2"}
### 4.1 Functions and Equations of Lines

A linear relationship between an input variable $x$ and an output variable $y$ is expressed as:

$$y = m x + c$$

* $m$: **Slope** — The change in $y$ given a $1$-unit increase in $x$ ($\frac{\Delta y}{\Delta x}$).
* $c$: **Intercept** — The value of $y$ when $x = 0$.

In multiple regression, we extend this equation to $P$ features:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p$$

---


## Detail 0C 3 {: .companion-detail-heading data-section="0c-3"}
#### Summation Notation
**Core Concept**: ∑ represents iteration and accumulation.

**Notation Meaning**: ∑_{i=1}^{p} f_i w_i corresponds to iteratively summing products of indexed elements.

**Code Implementation**:
```python
import numpy as np

f = np.array([15.0, 100.0, 3.0])
w = np.array([0.05, 0.002, -1.5])

# Iterative implementation
total_loop = 0.0
for i in range(len(f)):
    total_loop += f[i] * w[i]

# Vectorized implementation
total_numpy = f @ w  # Same as np.dot(f, w)
```

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


## Detail 0C 4 {: .companion-detail-heading data-section="0c-4"}
### 4.3 Why We Square Errors in Loss Functions

When evaluating prediction errors $e_i = y_i - \hat{y}_i$:
1. **Cancellation Problem:** Unsquared errors can be positive or negative. A model with errors $+10$ and $-10$ would sum to zero error ($\sum e_i = 0$), falsely appearing perfect.
2. **Squaring Eliminates Signs:** $(+10)^2 = 100$ and $(-10)^2 = 100$, ensuring all deviations contribute positively to total loss.
3. **Non-Linear Penalty for Large Outliers:** Squaring penalizes large errors much more severely than small ones:
   * Error = $2 \implies 	ext{Penalty} = 2^2 = 4$
   * Error = $10 \implies 	ext{Penalty} = 10^2 = 100$ (a 5x increase in error results in a 25x increase in penalty).

---


## Detail 0C 5 {: .companion-detail-heading data-section="0c-5"}
#### Linear Models and Regression
**Core Concept**: Linear relationships form the foundation of regression analysis.

**Equation of a Line**: y = mx + c
- m represents the slope (rate of change)
- c represents the intercept (starting value)

**Generalization to Multiple Features**: y = β₀ + β₁x₁ + … + βₚxₚ


## Detail 0C 6 {: .companion-detail-heading data-section="0c-6"}
### 4.4 Critical Distinction: Sum of Squares vs. Square of Sum

$$\sum_{i=1}^{n} x_i^2 
eq \left( \sum_{i=1}^{n} x_i 
ight)^2$$

Let $x = [2.0, -3.0, 5.0]$:

#### 1. Sum of Squares ($\sum x_i^2$):
$$\sum x_i^2 = (2.0)^2 + (-3.0)^2 + (5.0)^2 = 4.0 + 9.0 + 25.0 = 38.0$$

#### 2. Square of Sum ($(\sum x_i)^2$):
$$\left( \sum x_i 
ight)^2 = (2.0 + (-3.0) + 5.0)^2 = (4.0)^2 = 16.0$$

```python
x = np.array([2.0, -3.0, 5.0])

sum_of_squares = np.sum(x ** 2)  # Output: 38.0
square_of_sum  = np.sum(x) ** 2  # Output: 16.0
```

---

---


## Detail 0D 1 {: .companion-detail-heading data-section="0d-1"}
### 5.1 Sample Mean ($\bar{x}$)

The mean represents the arithmetic average (or balancing point) of a distribution.

$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

```python
costs = np.array([12.0, 30.0, 8.0, 45.0, 15.0])
mean_manual = np.sum(costs) / len(costs)  # Output: 22.0
```

---


## Detail 0D 4 {: .companion-detail-heading data-section="0d-4"}
### 5.3 Covariance ($	ext{cov}(x, y)$)

Covariance measures whether two continuous variables tend to deviate from their respective means in the same direction at the same time.

$$	ext{cov}(x, y) = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$$

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


## Detail 0D 5 {: .companion-detail-heading data-section="0d-5"}
### 5.4 Pearson Correlation Coefficient ($r$)

Because covariance depends on the units of measurement, it cannot be used to compare strength across different features. **Pearson Correlation** ($r$) normalizes covariance by dividing it by the product of the standard deviations of both variables:

$$r = \frac{	ext{cov}(x, y)}{\sigma_x \sigma_y}$$

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


## Detail 0D 7 {: .companion-detail-heading data-section="0d-7"}
#### Special Cases and Edge Conditions
**Zero Variance Warning**: Features with constant values produce nan in correlation calculations because division by zero occurs.

**Code Demonstration**:
```python
constant = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
print(np.corrcoef(constant, costs))  # Returns array with nan values
```

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


## Detail 0E 1 {: .companion-detail-heading data-section="0e-1"}
### 6.1 The Definition of a Derivative

A **derivative** measures the instantaneous rate of change (the slope of the tangent line) of a function $f(x)$ at a specific point $x$.

$$\frac{dy}{dx} = f'(x) = \lim_{h 	o 0} \frac{f(x+h) - f(x)}{h}$$

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


## Detail 0E 2 {: .companion-detail-heading data-section="0e-2"}
#### Derivatives and Gradients
**Core Concept**: Derivatives measure instantaneous rates of change.

**Limit Definition**: dy/dx = lim_{h→0} [f(x+h) - f(x)]/h

**Practical Application**:
```python
def f(x):
    return x ** 2

def numerical_slope(f, x, h=1e-6):
    return (f(x + h) - f(x)) / h

# Finite difference approximation
x = 3.0
for h in [1.0, 0.1, 0.01, 0.0001]:
    print(f"h={h}: slope={numerical_slope(f, x, h):.6f}")
```

**Power Rule**: Derivative of xⁿ is n·xⁿ⁻¹

**Implementation**:
```python
def y(x):
    return 3 * x**2 + 5 * x

def dy_dx_by_rule(x):
    # Power rule applied to 3x² gives 6x
    # Power rule applied to 5x gives 5
    return 6 * x + 5
```

### 6.1 The Definition of a Derivative

A **derivative** measures the instantaneous rate of change (the slope of the tangent line) of a function $f(x)$ at a specific point $x$.

$$\frac{dy}{dx} = f'(x) = \lim_{h 	o 0} \frac{f(x+h) - f(x)}{h}$$

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


## Detail 0E 3 {: .companion-detail-heading data-section="0e-3"}
### 6.2 Essential Analytical Derivative Rules

1. **Power Rule:**
   $$\frac{d}{dx} [x^n] = n x^{n-1}$$
   * Example: $\frac{d}{dx} [x^2] = 2x$
2. **Constant Multiple Rule:**
   $$\frac{d}{dx} [c \cdot f(x)] = c \cdot \frac{d}{dx}[f(x)]$$
3. **Sum Rule:**
   $$\frac{d}{dx} [f(x) + g(x)] = \frac{d}{dx}[f(x)] + \frac{d}{dx}[g(x)]$$

#### Combined Example:
For $y = 3x^2 + 5x$:

$$\frac{dy}{dx} = \frac{d}{dx}[3x^2] + \frac{d}{dx}[5x] = 3(2x) + 5(1) = 6x + 5$$

---


## Detail 0E 4 {: .companion-detail-heading data-section="0e-4"}
#### Partial Derivatives
**Core Concept**: Derivatives with respect to specific variables while holding others constant.

**Toy Example**: Error surface E(m, c) = (10 - (m·2 + c))²

**Numerical Partial Derivatives**:
```python
def error(m, c):
    prediction = m * 2 + c
    return (10 - prediction) ** 2

def partial_wrt_m(error_fn, m, c, h=1e-6):
    return (error_fn(m + h, c) - error_fn(m, c)) / h

def partial_wrt_c(error_fn, m, c, h=1e-6):
    return (error_fn(m, c + h) - error_fn(m, c)) / h
```


## Detail 0E 5 {: .companion-detail-heading data-section="0e-5"}
#### Gradient Descent
**Core Concept**: Iterative optimization by following the steepest descent direction.

**Update Rule**: θₙₑw = θₒₗd - α·∇J(θ)

**Practical Implementation**:
```python
m, c = 0.0, 0.0
learning_rate = 0.01

for step in range(50):
    grad_m = partial_wrt_m(error, m, c)
    grad_c = partial_wrt_c(error, m, c)
    m -= learning_rate * grad_m
    c -= learning_rate * grad_c

print(f"Optimized m={m:.3f}, c={c:.3f}, error={error(m, c):.6f}")
```

### 6.4 Gradient Descent Mechanics

**Gradient Descent** is an iterative optimization algorithm that minimizes a loss function by taking steps in the direction opposite to the gradient.

$$	heta_{	ext{new}} = 	heta_{	ext{old}} - \alpha 
abla E(	heta)$$

Where $\alpha$ is the **learning rate** (step size hyperparameter).

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

#### What happens if the learning rate $\alpha$ is too large?
If $\alpha$ is set too high (e.g., $\alpha = 5.0$), parameter updates will overshoot the minimum, causing error to explode toward infinity.

---

---


## Detail 0E 6 {: .companion-detail-heading data-section="0e-6"}
#### Error Handling and Debugging
**Common Error Types**:
- **Shape Mismatch**: Dimension size conflicts in matrix operations
- **KeyError**: Attempted access to missing dictionary keys
- **TypeError**: Type incompatibility in operations

**Error Translation Strategies**:
```python
# Shape mismatch interpretation
ValueError: matmul: Input operand 1 has a mismatch
# Translation: Two arrays need matching dimensions but have different sizes

# Missing key interpretation
KeyError: 'terrain_index'
# Translation: The dictionary does not contain the requested key

# Type conversion interpretation
TypeError: can only concatenate str to float
# Translation: Explicit type conversion required for string + numeric operations
```

## 8. Master Rosetta Stone & Traceback Cheat Sheet


## Detail CAPSTONE {: .companion-detail-heading data-section="capstone"}
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

### Comprehensive Exercises

### Assessment Criteria

## Appendix: Mathematical Reference

