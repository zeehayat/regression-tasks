# Interlude — From a Single Moment to a History Through Time

## Two short days between Chapter 4 and Chapter 5

> **Central promise.** Chapter 4 asked "did the event happen by month 12?" — a single yes/no, with a single probability attached. Chapter 5 asks "*when* does the event happen?" — and answering that means working with a probability that's spread out continuously over time, plus two pieces of mathematics this book hasn't used yet: integrals and formal limits. By the end of this interlude, a sentence like "the hazard is the instantaneous event rate among those still event-free" will read as an idea you can compute, not a wall of symbols.

## Why this exists

Look at what Chapter 5, Day 31 does in its opening section. In about thirty lines it introduces a cumulative distribution function, a probability density, a hazard defined as a limit of a conditional probability divided by a shrinking interval, an integral linking hazard to cumulative hazard, and the identity $S(t) = \exp[-H(t)]$. Chapter 4 prepared you for the *conditional probability* and the *exp/log* parts — logistic regression uses both constantly. It did not prepare you for **continuous** random variables, for **integrals**, or for a **formal limit**. Those three ideas arrive on Day 31 with no runway, and everything from Kaplan–Meier to the Cox model is built on top of them.

This interlude is short on purpose — you're not starting from zero the way Chapter 0 did. You already know discrete probability, likelihood, and derivatives cold. We're extending three things you know into a fourth thing you don't, not starting over.

Same rules as always: code proves every claim, you build something small, you break it, you don't move on until the exit check passes.

---

# Interlude Day A — From a Single Probability to a Curve Over Time

> **Today's central idea:** When an outcome can land anywhere on a continuous scale instead of just "yes" or "no," you can't ask "what's the probability of exactly this value?" — the honest question becomes "what's the probability of landing in this small window?"

## A.1 What Chapter 4 gave you, and where it stops

Chapter 4, Day 21 built everything on a Bernoulli outcome: a project either triggers a cost-overrun warning by month 12, or it doesn't. One probability, $p$, describes the whole thing.

```python
import numpy as np

# Chapter 4's world: did it happen by month 12? One number answers everything.
p_warning_by_12 = 0.30
```

But a project that triggers a warning at month 3 and one that triggers a warning at month 11 both get coded as `1` in that world. Chapter 5's whole reason for existing is that this throws away real information — *when* it happened, and for the ones that haven't happened yet, *how long they've already survived without it*. To keep that information, the outcome can no longer be a single 0-or-1; it has to be a number that can land anywhere on the timeline: 3.2 months, 11.87 months, 40.001 months. That's a **continuous random variable**, and it needs new machinery.

## A.2 Why you can't just ask "what's the probability of exactly month 11.870000...?"

Here's the uncomfortable fact first: for a continuous outcome, the probability of hitting any *exact* value is zero. There are infinitely many possible values between month 11 and month 12 — 11.1, 11.01, 11.001, forever — so no single one of them can carry a positive share of the total probability. This isn't a technicality to wave away; it's the reason the whole toolkit looks different from Chapter 4's.

**Code proof — watch the probability of "close to 11.87" shrink as "close" gets stricter:**

```python
rng = np.random.default_rng(0)

# Simulate 200,000 synthetic warning times (months), just to look at the shape.
# (This is illustrative data, not the real KP mechanism Chapter 5 will use.)
warning_times = rng.exponential(scale=20.0, size=200_000)

for window in [2.0, 0.5, 0.05, 0.005]:
    hits = np.sum(np.abs(warning_times - 11.87) < window)
    prob_estimate = hits / len(warning_times)
    print(f"window=±{window:<6} estimated P ≈ {prob_estimate:.5f}")
```

As the window shrinks, the estimated probability keeps shrinking toward zero — but the probability *per unit width* of that window settles down to a stable number. That stable number — probability per unit width, not probability itself — is exactly what a **density** is.

## A.3 The density: probability, rescaled by width

If $f(t)$ is the density of event time $T$, then the probability of landing in a small window of width $\Delta t$ around $t$ is approximately $f(t) \cdot \Delta t$ — density times width, the same "rate times interval" idea you'll recognise from Day 0E's slope work, just for probability instead of cost.

```python
# Estimate the density near t=11.87 directly: (probability of a window) / (window width)
window = 0.5
hits = np.sum(np.abs(warning_times - 11.87) < window)
prob_in_window = hits / len(warning_times)
density_estimate = prob_in_window / (2 * window)   # window extends both directions
print(f"estimated density near t=11.87: {density_estimate:.5f}")
```

A density is **not** a probability — it can be larger than 1, because it's a rate, not a share. Only density multiplied by a width gives you back an actual probability. This is exactly analogous to Chapter 5's warning about hazard later: *a rate is not a probability, even though it's built from one.*

## A.4 The cumulative distribution function and the survival function

You've actually already met the discrete cousin of this in Chapter 4: $P(T \le t)$. For a continuous $T$, this accumulated probability is called the **cumulative distribution function**:

$$F(t) = P(T \le t)$$

Chapter 5 will spend far more time on its complement, the **survival function** — directly meaningful in the KP running case as "the probability a project has *not yet* triggered a warning by month $t$":

$$S(t) = P(T > t) = 1 - F(t)$$

**Code proof — build both from raw simulated data, no library shortcuts:**

```python
sample = np.sort(warning_times)[:1000]   # a manageable sample, sorted

def empirical_cdf(sample, t):
    return np.mean(sample <= t)

def empirical_survival(sample, t):
    return np.mean(sample > t)

for t in [5, 10, 20, 40]:
    print(f"t={t:>3}  F(t)≈{empirical_cdf(sample, t):.3f}  "
          f"S(t)≈{empirical_survival(sample, t):.3f}  "
          f"sum={empirical_cdf(sample, t) + empirical_survival(sample, t):.3f}")
```

Notice `F(t) + S(t)` always comes out to (approximately) 1 — they're complements by construction, the same way "probability of rain" and "probability of no rain" always summed to 1 back in Chapter 4. This crude "count how many are past $t$" estimator is exactly the starting intuition for Kaplan–Meier — the only thing Day 31 adds is a correction for the projects whose exact time you never got to see because they were censored.

## A.5 Build: a tiny empirical survival curve

```python
def survival_curve(sample, time_grid):
    return np.array([empirical_survival(sample, t) for t in time_grid])

time_grid = np.linspace(0, 60, 13)
curve = survival_curve(sample, time_grid)

for t, s in zip(time_grid, curve):
    bar = "#" * int(s * 40)
    print(f"t={t:5.1f}  S(t)={s:.3f}  {bar}")
```

Run this and watch the curve fall from 1.0 toward 0.0 as time passes — every project eventually triggers a warning in this synthetic world, so survival decays to zero. This bar chart *is* the shape of a Kaplan–Meier curve, minus the censoring correction Day 31 adds.

## A.6 Break it deliberately

```python
# A tempting but wrong shortcut: treat the density as if it were a probability.
window = 0.01
hits = np.sum(np.abs(warning_times - 11.87) < window)
density_estimate = (hits / len(warning_times)) / (2 * window)
print(density_estimate)   # this number can easily exceed 1.0
```

If this prints something bigger than 1, that's not a bug — a density genuinely can exceed 1 (it's a *rate per unit time*, and if your unit of time is coarse relative to how fast probability accumulates, the rate number can be large). The mistake would be reporting this number as "the probability of triggering a warning around month 11.87." It isn't one. Only density × width is a probability. Keep this straight now, because Day 31 makes the identical distinction for hazard, and conflating the two is the single most common survival-analysis mistake beginners make.

### Interlude Day A exit check

You should be able to:
- explain, in one sentence, why the probability of a continuous outcome hitting one exact value is zero;
- explain what a density is, in terms of "probability per unit width," and why it isn't itself a probability;
- build an empirical CDF and survival function from a raw sample using nothing but comparisons and `.mean()`;
- state, without notes, why $F(t) + S(t) = 1$.

---

# Interlude Day B — Rates, Limits, and Integrals

> **Today's central idea:** An integral is a sum, the same way $\sum$ is a sum — just added over infinitely many infinitely thin slices instead of a finite list. Once you see it that way, $H(t) = \int_0^t h(u)\,du$ stops being unfamiliar notation and becomes "add up the hazard rate over every instant from 0 to $t$."

## B.1 You've already used a limit — now we name it properly

Back in Day 0E, you approximated a derivative like this:

```python
def numerical_slope(f, x, h):
    return (f(x + h) - f(x)) / h
```

and watched the estimate settle down as `h` shrank toward zero. That *is* a limit — you were computing $\lim_{h \to 0} \frac{f(x+h)-f(x)}{h}$ without the formal notation. Chapter 5's hazard function is defined the same way, just wrapped around a conditional probability instead of a plain function:

$$h(t) = \lim_{\Delta t \to 0} \frac{P(t \le T < t + \Delta t \mid T \ge t)}{\Delta t}$$

Read it exactly the way you read the derivative: "take the probability of the event landing in a shrinking window just after $t$, given that it hasn't happened yet by $t$; divide by the width of that window; watch what happens as the window shrinks to nothing." $\lim_{\Delta t \to 0}$ means precisely what `h` shrinking toward `0` meant in your derivative code.

**Code proof — estimate a hazard numerically, the same way you estimated a derivative numerically:**

```python
def numerical_hazard(sample, t, window):
    at_risk = sample[sample >= t]                 # everyone still event-free at t
    if len(at_risk) == 0:
        return np.nan
    events_in_window = np.sum((at_risk >= t) & (at_risk < t + window))
    conditional_prob = events_in_window / len(at_risk)
    return conditional_prob / window

for window in [5.0, 1.0, 0.1, 0.01]:
    print(f"window={window:<6} hazard estimate at t=10: {numerical_hazard(sample, 10, window):.4f}")
```

As `window` shrinks, this settles toward a stable number — the true instantaneous hazard at $t=10$ — exactly the way your derivative estimate settled toward $2x$ back in Day 0E. Same limiting process, applied to a conditional probability instead of a plain function.

## B.2 The integral: summation notation for infinitely thin slices

Recall $\sum_{i=1}^{n} x_i$ from Day 0C — add up a formula over a finite list, indexed by whole numbers. An integral does the same *adding-up*, but over a continuous stretch of a number line instead of a finite list of items, and it's written $\int$.

The bridge between the two is a **Riemann sum**: chop the interval into many thin rectangles, add up their areas, and watch the total settle down as the rectangles get thinner.

```python
def area_under(func, a, b, n_slices):
    xs = np.linspace(a, b, n_slices, endpoint=False)
    width = (b - a) / n_slices
    return np.sum(func(xs) * width)     # a literal sum of thin rectangle areas

def constant_hazard(t, lam=0.03):
    return np.full_like(t, lam)

for n in [10, 100, 10_000, 1_000_000]:
    approx = area_under(constant_hazard, 0, 12, n)
    print(f"n_slices={n:<10} approx integral = {approx:.6f}")
```

For a constant hazard $\lambda = 0.03$, the true value of $\int_0^{12} \lambda\, dt$ is exactly $\lambda \times 12 = 0.36$, and you should see the approximation converge to that as `n_slices` grows. This is the entire idea behind $\int$: it's `np.sum(values * width)` in the limit of infinitely many, infinitely thin slices — the continuous sibling of the $\sum$ you already know.

## B.3 Why cumulative hazard needs an integral, and hazard alone doesn't

Chapter 5 defines cumulative hazard as

$$H(t) = \int_0^t h(u)\, du$$

which is exactly the pattern in §B.2: add up the instantaneous rate $h(u)$ over every instant from 0 to $t$, in infinitely thin slices. This mirrors something you already did in Chapter 4: an overall risk over a *fixed* window is built by accumulating a rate; here, instead of one fixed window, you're accumulating continuously across the whole timeline up to $t$.

## B.4 Closing the loop: hazard, cumulative hazard, and survival

Three quantities, one identity, and you now have the pieces for all of it:

$$H(t) = -\log S(t) \qquad \Longleftrightarrow \qquad S(t) = \exp[-H(t)]$$

This uses exactly the exp/log machinery Chapter 4's logistic regression already put in your hands — $\log$ turns a product into a sum (useful because survival to time $t$ is a *product* of surviving each small interval along the way), and $\exp$ undoes it.

**Code proof — build a survival curve two independent ways and check they agree:**

```python
lam = 0.03
time_grid = np.linspace(0, 60, 13)

# Route 1: the closed-form constant-hazard survival function
survival_direct = np.exp(-lam * time_grid)

# Route 2: integrate the hazard numerically, then apply S(t) = exp(-H(t))
def cumulative_hazard_numeric(t, lam, n_slices=10_000):
    return area_under(lambda u: np.full_like(u, lam), 0, t, n_slices)

H_values = np.array([cumulative_hazard_numeric(t, lam) if t > 0 else 0.0 for t in time_grid])
survival_via_integral = np.exp(-H_values)

for t, s1, s2 in zip(time_grid, survival_direct, survival_via_integral):
    print(f"t={t:5.1f}  direct S(t)={s1:.4f}  via integral S(t)={s2:.4f}")
```

The two routes should agree to several decimal places. This is the single identity that Chapter 5, Day 31 onward leans on constantly: model the *rate* (hazard), integrate it, and you get back an *absolute probability* (survival) — the same "rate → total" logic as converting a speed into a distance travelled.

## B.5 Build: a hazard-to-survival mini pipeline

```python
def hazard_to_survival(hazard_fn, time_grid, n_slices=5_000):
    H = np.array([
        area_under(hazard_fn, 0, t, n_slices) if t > 0 else 0.0
        for t in time_grid
    ])
    return np.exp(-H)

def rising_hazard(t):
    # A hazard that increases over time — riskier the longer a project runs.
    return 0.01 + 0.002 * t

time_grid = np.linspace(0, 40, 9)
survival = hazard_to_survival(rising_hazard, time_grid)

for t, s in zip(time_grid, survival):
    print(f"t={t:5.1f}  S(t)={s:.4f}")
```

Unlike the constant-hazard case, this survival curve falls faster as time goes on, because the risk itself is rising — exactly the kind of hazard shape Chapter 5, Day 33 will teach you to detect and model when the constant-risk assumption fails.

## B.6 Break it deliberately

```python
# A believable but wrong claim: "the hazard at t=10 is 0.4, so there's a 40% chance
# of the event happening in the next month."
window = 12.0   # a whole year, not an instant
hazard_at_10 = numerical_hazard(sample, 10, window)
print(hazard_at_10)
print("naive probability guess:", hazard_at_10 * window)   # can exceed 1
```

If `hazard_at_10 * window` comes out above 1, you've just produced an impossible "probability." The bug isn't in the arithmetic — it's in forgetting that $h(t)\Delta t$ is only a good *approximation* to a probability when $\Delta t$ is small. Stretch the window out and the approximation breaks, the same way a straight-line approximation to a curve breaks if you extend it too far past the point you built it at.

### Interlude Day B exit check

You should be able to:
- explain a hazard as a limit, in the same terms you used to explain a derivative in Day 0E;
- explain an integral as "a sum over infinitely thin slices," and connect it to $\sum$ from Day 0C;
- compute a Riemann-sum approximation of an integral in code and watch it converge as slices get finer;
- derive a survival curve from a hazard function two independent ways ($\exp(-\lambda t)$ directly, and numerically integrating first) and get matching answers;
- explain why $h(t) \cdot \Delta t$ stops being a good probability approximation once $\Delta t$ is large.

---

## Interlude Capstone

Using `rising_hazard` from §B.5:

1. Compute $S(24)$ two ways — direct numeric integration, and by first tabulating $H(t)$ at a fine grid and reading off $\exp(-H(24))$ — and confirm they match.
2. Estimate the density $f(t)$ at $t=20$ by using $f(t) = h(t)\cdot S(t)$, then cross-check it against a direct "windowed probability ÷ width" estimate the way you did in §A.3.
3. Deliberately pick a `window` too large in `numerical_hazard` (say, 20 months) and show the "hazard × window" arithmetic producing a value the size of a real probability shouldn't reach. Write one sentence explaining why.
4. In your own words (no formulas), explain to an imaginary colleague the difference between "the probability a project triggers a warning by month 12" (Chapter 4's question) and "the hazard of triggering a warning at month 12" (Chapter 5's question).

If you can do all four without reopening this interlude, Chapter 5, Day 31 will read as a continuation of something you already understand, not a new subject.

---

## Glossary (new symbols for this interlude)

| Symbol | Name | Meaning |
|---|---|---|
| $f(t)$ | density | probability per unit time near $t$; not itself a probability |
| $F(t)$ | cumulative distribution function | $P(T \le t)$ |
| $S(t)$ | survival function | $P(T > t) = 1 - F(t)$ |
| $h(t)$ | hazard | instantaneous event rate among those still event-free at $t$ |
| $H(t)$ | cumulative hazard | $\int_0^t h(u)\,du$; accumulated rate from 0 to $t$ |
| $\lim_{\Delta t \to 0}$ | limit | what a quantity settles toward as $\Delta t$ shrinks to nothing — same idea as Day 0E's derivative limit |
| $\int_a^b \cdots\, du$ | integral | a sum over infinitely many, infinitely thin slices between $a$ and $b$ — the continuous sibling of $\sum$ |

## Where Chapter 5 continues

Day 30 picks up right where Chapter 4 left off — the censoring, risk-set, and truncation content there is already built from scratch and doesn't need a bridge. Day 31 is where this interlude pays off: §31.1's density, survival, and hazard definitions, and §31.2's $H(t) = -\log S(t)$ identity, are now the exact ideas you just built by hand, wearing their formal notation for the first time.
