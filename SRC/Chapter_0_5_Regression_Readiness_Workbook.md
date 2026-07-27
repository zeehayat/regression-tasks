# Chapter 0.5 — Regression Readiness Workout

## A one-evening bridge into Chapter 1, with 180+ exercises

> **Promise.** This workbook teaches only the mathematics, graph reading, and numerical habits that Chapter 1 assumes. It is designed for active work: read a small idea, do several short repetitions, check your understanding, and then use the idea in regression.

### Separate explanation pages

If you want the teaching text separately from the drills, read [Chapter 0.5 — Regression Readiness Explanations](Chapter_0_5_Regression_Readiness_Explanations.md) first, then return here for practice. The explanations are also available as a matching PDF.

The running context is the same as Chapter 1: fictional microhydro power (MHP) projects in Khyber Pakhtunkhwa (KP). Costs are measured in **million PKR**, distances in **kilometres**, capacity in **kilowatts**, and terrain difficulty on an index from 1 to 5.

This is a bridge, not a new statistics course. You will practise:

1. graphs, coordinates, straight lines, residuals, and contours;
2. algebra needed to expand and rearrange OLS equations;
3. vectors, distance, dot products, perpendicularity, and projection;
4. matrices, transpose, multiplication, column space, rank, and systems;
5. derivatives, partial derivatives, gradients, Hessians, and minima;
6. floating-point comparison and a simple held-out evaluation; and
7. one cumulative readiness challenge.

You do **not** need probability distributions, hypothesis tests, QR decomposition, SVD, regularisation, gradient descent, or causal diagrams yet.

---

## How to use this tonight

There are more exercises than you need for a single evening. That is deliberate: repetition is available wherever you feel weak.

Use three marks:

- **Core** — do this tonight;
- **Repeat** — do this if the idea is not automatic;
- **Stretch** — save it if time is short.

Suggested elapsed-time route:

| Block | Work | Time |
|---|---|---:|
| 0 | Setup and diagnostic | 15 min |
| 1 | Graphs and straight lines | 55 min |
| 2 | Algebra | 65 min |
| Break | Move, drink water, no screen | 10 min |
| 3 | Vector geometry | 60 min |
| 4 | Matrices and rank | 70 min |
| Break | Food and rest | 20 min |
| 5 | Calculus for minimisation | 80 min |
| 6 | Numerical computing and held-out data | 45 min |
| 7 | Cumulative challenge and exit check | 50 min |

That is about 7 hours including breaks. If you have less time, complete only exercises marked **Core**, then do the final exit check. Accuracy matters more than racing through every item.

### The attempt rule

For every exercise:

1. write an answer before looking at the key;
2. if wrong, write one sentence explaining the error;
3. redo it without copying;
4. revisit it after 30–60 minutes.

Use a calculator or Python only when the exercise explicitly says so. The point of the short arithmetic drills is to make the notation feel ordinary.

### Minimal setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib
```

Create `chapter_0_5_work.py` and begin it with:

```python
import numpy as np
import matplotlib.pyplot as plt
```

---

# Block 0 — Fifteen-Minute Diagnostic

Do these without notes. Do not worry if several are unfamiliar; the results tell you where to spend repetitions.

1. **[Core D1]** On a graph, which axis is horizontal?
2. **[Core D2]** In the point $(4,9)$, what are the $x$- and $y$-coordinates?
3. **[Core D3]** Evaluate $3 + 2(5-1)$.
4. **[Core D4]** Expand $2(x+3)$.
5. **[Core D5]** Expand $(a-b)^2$.
6. **[Core D6]** Solve $3x+2=14$.
7. **[Core D7]** Find the length of the vector $(3,4)$.
8. **[Core D8]** Calculate $(1,2)\cdot(3,4)$.
9. **[Core D9]** What does a zero dot product say about two nonzero vectors?
10. **[Core D10]** If $X$ has shape $(8,3)$ and $\beta$ has shape $(3,)$, what is the shape of $X\beta$?
11. **[Core D11]** What does matrix rank count informally?
12. **[Core D12]** Differentiate $x^2$.
13. **[Core D13]** If $f'(4)=0$, must $x=4$ be a minimum? Explain in one clause.
14. **[Core D14]** Why can `0.1 + 0.2 == 0.3` be false in Python?
15. **[Core D15]** Why should a model be evaluated on observations not used to fit it?

Score one point per sound answer.

- 13–15: use the Core exercises and move quickly.
- 9–12: do Core plus Repeat exercises in weak blocks.
- 0–8: do every Core and Repeat exercise; postpone Stretch exercises.

---

# Block 1 — Graphs, Coordinates, Lines, Residuals, and Contours

## 1.1 Axes, coordinates, scales, and units

A two-dimensional graph uses:

- the horizontal **$x$-axis** for an input such as road distance;
- the vertical **$y$-axis** for an outcome such as project cost;
- an ordered pair $(x,y)$ to locate a point; and
- a scale and unit on each axis.

If one project has road distance 3 km and cost 22 million PKR, its point is $(3,22)$. The coordinate order matters.

The **origin** is $(0,0)$. A point can be observed even when the axes do not begin at zero. Always inspect the tick labels before judging a visual difference: a truncated axis can make a small difference look dramatic.

### Exercises: coordinate literacy

1. **[Core G1]** For $(7,31)$, state the horizontal and vertical coordinates.
2. **[Core G2]** Write the point for a project 12 km from a road costing 48 million PKR.
3. **[Core G3]** What point is the origin?
4. **[Core G4]** Which point is farther right: $(2,50)$ or $(9,20)$?
5. **[Core G5]** Which point is higher: $(8,25)$ or $(3,40)$?
6. **[Repeat G6]** Plot by hand: $(0,8)$, $(1,10)$, $(2,12)$, $(3,14)$.
7. **[Repeat G7]** If the $x$-axis changes from kilometres to metres, what happens to the coordinate 2 km?
8. **[Repeat G8]** A cost axis starts at 95 rather than 0. Explain one visual danger.
9. **[Stretch G9]** Two charts show the same costs. One spans 0–100; the other spans 80–100. Which makes differences appear larger, and why?

## 1.2 Scatter plots and visible relationships

A **scatter plot** places one point per observation.

- An upward cloud suggests a positive relationship.
- A downward cloud suggests a negative relationship.
- A shapeless cloud suggests no clear visible relationship.
- A curve suggests the relationship may not be well described by one straight line.

A visible relationship is not automatically causal. Road distance and cost may rise together because remote projects also face difficult terrain.

### Exercises: reading a point cloud

10. **[Core G10]** Costs rise as road distance rises. Is the visible relationship positive or negative?
11. **[Core G11]** Capacity rises while cost falls. Is the visible relationship positive or negative?
12. **[Core G12]** Points form a nearly horizontal cloud. What relationship is visible?
13. **[Core G13]** Points form a U-shape. Why is one straight line incomplete?
14. **[Repeat G14]** Give one reason an extreme point deserves inspection rather than automatic deletion.
15. **[Repeat G15]** State why an upward scatter plot does not prove that increasing road distance causes higher cost.
16. **[Stretch G16]** Imagine two terrain groups, each with an upward trend, but the combined data slope downward. Name this phenomenon.

## 1.3 Straight lines: slope and intercept

A line is

$$
\hat y=b_0+b_1x.
$$

Here:

- $\hat y$ (“y-hat”) is the predicted outcome;
- $b_0$ is the intercept, the predicted outcome at $x=0$;
- $b_1$ is the slope, the predicted change in $\hat y$ for a one-unit increase in $x$.

Slope can also be read from two points:

$$
\text{slope}=\frac{\text{change in }y}{\text{change in }x}
=\frac{y_2-y_1}{x_2-x_1}.
$$

For $\hat y=8+4x$, the intercept is 8 million PKR and the slope is 4 million PKR per km.

### Worked example

For the points $(2,14)$ and $(5,26)$:

$$
b_1=\frac{26-14}{5-2}=\frac{12}{3}=4.
$$

Using $(2,14)$:

$$
14=b_0+4(2)\quad\Rightarrow\quad b_0=6.
$$

The line is $\hat y=6+4x$.

### Exercises: lines

17. **[Core G17]** In $\hat y=10+3x$, identify the intercept and slope with units if $x$ is km and $y$ is million PKR.
18. **[Core G18]** Predict at $x=0$, $x=2$, and $x=5$ for $\hat y=10+3x$.
19. **[Core G19]** Find the slope through $(1,7)$ and $(4,16)$.
20. **[Core G20]** Find the line through $(0,5)$ and $(3,11)$.
21. **[Core G21]** A line falls by 8 units when $x$ rises by 4. Find its slope.
22. **[Repeat G22]** Find the slope through $(2,20)$ and $(6,12)$.
23. **[Repeat G23]** A line has slope 5 and passes through $(2,17)$. Find its intercept.
24. **[Repeat G24]** Does $\hat y=4x$ include an intercept? What prediction does it force at $x=0$?
25. **[Repeat G25]** Convert the slope 4 million PKR per km into million PKR per metre.
26. **[Stretch G26]** Two points have the same $x$ but different $y$. Why is the ordinary slope formula undefined?

## 1.4 Observations, fitted values, and residuals

An observed value is $y_i$. A fitted or predicted value is $\hat y_i$. The residual is

$$
e_i=y_i-\hat y_i.
$$

On a graph, the residual is the **vertical signed distance** from the fitted line to the observed point.

- $e_i>0$: the point is above the line; the model underpredicted.
- $e_i<0$: the point is below the line; the model overpredicted.
- $e_i=0$: the point is on the line.

For a cost of 25 and prediction of 21, the residual is $25-21=4$ million PKR.

### Exercises: residuals

27. **[Core G27]** Observed cost is 31 and predicted cost is 28. Find and interpret the residual.
28. **[Core G28]** Observed cost is 20 and predicted cost is 26. Find and interpret the residual.
29. **[Core G29]** Calculate predictions and residuals for $x=(0,1,2)$, $y=(7,11,14)$, and $\hat y=6+4x$.
30. **[Core G30]** Which has the larger absolute error: residual $-7$ or residual $5$?
31. **[Repeat G31]** A point lies 3 units below a fitted line. What is its residual?
32. **[Repeat G32]** If every residual is positive, what systematic problem does the line have on these observations?
33. **[Repeat G33]** Why is a horizontal distance not the OLS residual used in Chapter 1?
34. **[Stretch G34]** Sketch two different lines through the same scatter plot. Mark the residuals for one observation under both lines.

## 1.5 Contour lines

Imagine the squared-error value depends on two adjustable parameters, $b_0$ and $b_1$. A **contour line** joins parameter pairs with the same error, just as a map contour joins places with the same elevation.

In an OLS contour plot:

- the horizontal and vertical axes are parameters, not observations;
- each loop is one equal-error level;
- moving toward smaller nested loops usually moves toward lower error; and
- the centre of elliptical loops is the minimum.

A long, narrow valley means many parameter pairs give similar error. A perfectly flat direction signals non-uniqueness.

### Exercises: contours and graph synthesis

35. **[Core G35]** What quantity stays constant along one error contour?
36. **[Core G36]** In a contour plot with axes $b_0$ and $b_1$, does one dot represent a project or a parameter pair?
37. **[Core G37]** Where is the minimum in a set of nested elliptical contours?
38. **[Repeat G38]** What does a long, narrow contour valley suggest?
39. **[Repeat G39]** Explain the difference between a scatter plot point $(x_i,y_i)$ and an error-surface point $(b_0,b_1)$.
40. **[Stretch G40]** In one sentence, connect a fitted line in data space to one point in parameter space.

### Code lab G

Type, run, and then change the slope from 4 to 2 before rerunning.

```python
x = np.array([0.0, 1.0, 2.0, 3.0])
y = np.array([7.0, 11.0, 14.0, 19.0])
b0, b1 = 6.0, 4.0
y_hat = b0 + b1 * x
residuals = y - y_hat

plt.scatter(x, y, label="observed")
plt.plot(x, y_hat, label="fitted")
for xi, yi, yhi in zip(x, y, y_hat):
    plt.vlines(xi, yhi, yi, colors="tab:red")
plt.xlabel("Road distance (km)")
plt.ylabel("Cost (million PKR)")
plt.legend()
plt.show()

print("predictions:", y_hat)
print("residuals:", residuals)
```

41. **[Core G41]** Before running, predict the four fitted values.
42. **[Core G42]** Confirm that each red segment has magnitude $|e_i|$.
43. **[Repeat G43]** With slope 2, which observations are underpredicted?
44. **[Stretch G44]** Add a horizontal line at zero to a separate residual plot.

### Block 1 mastery check

Without notes, explain:

1. what slope, intercept, and residual mean;
2. why graph axes and units must be read before the visual pattern;
3. why association in a scatter plot is not an intervention effect; and
4. what a contour line represents.

If any explanation takes more than two sentences, do G17–G40 again tomorrow.

---

# Block 2 — Algebra for Regression Equations

## 2.1 Signs, fractions, order of operations, and powers

Use the order:

1. brackets;
2. powers;
3. multiplication and division;
4. addition and subtraction.

Remember:

$$
-(-a)=a,\qquad a^2=a\cdot a,\qquad \sqrt{a^2}=|a|.
$$

The square of a negative number is positive: $(-3)^2=9$. But $-3^2=-(3^2)=-9$ unless brackets make the negative part of the base.

### Exercises: arithmetic fluency

1. **[Core A1]** Evaluate $5-8$.
2. **[Core A2]** Evaluate $-4-(-7)$.
3. **[Core A3]** Evaluate $3+2(5-1)$.
4. **[Core A4]** Evaluate $(3+2)^2-4$.
5. **[Core A5]** Evaluate $(-3)^2$ and $-3^2$.
6. **[Core A6]** Simplify $12/18$.
7. **[Repeat A7]** Evaluate $2-\{3-[4-6]\}$.
8. **[Repeat A8]** Evaluate $|{-5}|+\sqrt{16}$.
9. **[Repeat A9]** Evaluate $(10-4)^2/3$.
10. **[Stretch A10]** If $a=-2$, evaluate $3a^2-4a+1$.

## 2.2 Distributing and collecting like terms

The distributive law is

$$
a(b+c)=ab+ac.
$$

It works with subtraction:

$$
a(b-c)=ab-ac.
$$

Only like terms combine. For example, $3x+2x=5x$, but $3x+2$ cannot be reduced.

### Exercises: distribution

11. **[Core A11]** Expand $3(x+4)$.
12. **[Core A12]** Expand $5(2x-3)$.
13. **[Core A13]** Expand $-2(x-6)$.
14. **[Core A14]** Simplify $3x+5x-2$.
15. **[Core A15]** Simplify $4(x+2)+3x$.
16. **[Repeat A16]** Simplify $2(3x-1)-4(x+5)$.
17. **[Repeat A17]** Simplify $a(b+c)-ab$.
18. **[Repeat A18]** Expand $-3(2a-b+4)$.
19. **[Stretch A19]** Simplify $2(x-y)-3(y-x)$.

## 2.3 Expanding squares

These patterns matter because OLS squares errors:

$$
(a-b)^2=a^2-2ab+b^2,
$$

$$
(a+b)^2=a^2+2ab+b^2.
$$

The middle term is essential. $(a-b)^2$ is **not** $a^2-b^2$.

For one residual:

$$
(y-bx)^2=y^2-2ybx+b^2x^2.
$$

### Exercises: squared expressions

20. **[Core A20]** Expand $(x-3)^2$.
21. **[Core A21]** Expand $(x+5)^2$.
22. **[Core A22]** Expand $(y-bx)^2$.
23. **[Core A23]** Expand $(7-2b)^2$.
24. **[Core A24]** Check A23 by substituting $b=1$ into both forms.
25. **[Repeat A25]** Expand $(a-3c)^2$.
26. **[Repeat A26]** Expand $(2x+4)^2$.
27. **[Repeat A27]** Explain the error in $(a-b)^2=a^2-b^2$.
28. **[Stretch A28]** Expand and simplify $(y-b_0-b_1x)^2$.

## 2.4 Rearranging and isolating an unknown

An equation is a balance. Perform the same operation on both sides.

Example:

$$
3x+5=20
$$

Subtract 5:

$$
3x=15.
$$

Divide by 3:

$$
x=5.
$$

“Moving a term across” is shorthand, not magic: its sign changes because you add or subtract it on both sides.

### Exercises: solve and rearrange

29. **[Core A29]** Solve $x+7=12$.
30. **[Core A30]** Solve $4x=28$.
31. **[Core A31]** Solve $3x-5=16$.
32. **[Core A32]** Solve $5-2x=11$.
33. **[Core A33]** Solve $ax=c$ for $x$, assuming $a\ne0$.
34. **[Core A34]** Rearrange $y=b_0+b_1x$ to isolate $b_0$.
35. **[Repeat A35]** Rearrange $y=b_0+b_1x$ to isolate $x$, assuming $b_1\ne0$.
36. **[Repeat A36]** Solve $2(x+3)=14$.
37. **[Repeat A37]** Solve $3x/4=9$.
38. **[Stretch A38]** Solve $(x-2)/5=(x+6)/9$.

## 2.5 Simultaneous equations

Two unknowns require two independent equations. Use substitution or elimination.

Example:

$$
b_0+b_1=5,\qquad b_0+2b_1=8.
$$

Subtract the first equation from the second:

$$
b_1=3.
$$

Substitute back: $b_0+3=5$, so $b_0=2$.

### Exercises: systems

39. **[Core A39]** Solve $a+b=9$ and $a-b=3$.
40. **[Core A40]** Solve $b_0+b_1=7$ and $b_0+3b_1=15$.
41. **[Core A41]** Solve $2x+y=11$ and $x+y=7$.
42. **[Repeat A42]** Solve $3a+2b=12$ and $a+2b=8$.
43. **[Repeat A43]** Show that $x+y=4$ and $2x+2y=8$ do not determine one unique pair.
44. **[Stretch A44]** Interpret A43 geometrically as two equations describing lines.

## 2.6 Summations and OLS-style algebra

The sigma symbol means repeated addition:

$$
\sum_{i=1}^{n}a_i=a_1+a_2+\cdots+a_n.
$$

Useful rules are:

$$
\sum_i(a_i+b_i)=\sum_i a_i+\sum_i b_i,
$$

$$
\sum_i ca_i=c\sum_i a_i,
$$

where $c$ does not change with $i$.

For the no-intercept loss

$$
S(b)=\sum_{i=1}^{n}(y_i-bx_i)^2,
$$

expansion gives

$$
S(b)=\sum_i y_i^2-2b\sum_i x_i y_i+b^2\sum_i x_i^2.
$$

### Exercises: sigma fluency

45. **[Core A45]** Expand $\sum_{i=1}^{3}x_i$ without sigma notation.
46. **[Core A46]** For $x=(2,4,6)$, calculate $\sum_i x_i$ and $\sum_i x_i^2$.
47. **[Core A47]** For $x=(1,2)$ and $y=(3,5)$, calculate $\sum_i x_i y_i$.
48. **[Core A48]** Expand $\sum_i(y_i-bx_i)^2$ using the pattern above.
49. **[Repeat A49]** Why may $b$ move outside $\sum_i bx_i$?
50. **[Repeat A50]** Why may $x_i$ not generally move outside $\sum_i x_i y_i$?
51. **[Repeat A51]** For $x=(1,2)$, $y=(2,5)$, write $S(b)$ as a quadratic in $b$.
52. **[Stretch A52]** Evaluate the A51 quadratic at $b=2$ and verify it by calculating residuals directly.

### Code lab A

```python
x = np.array([1.0, 2.0])
y = np.array([2.0, 5.0])

for b in [0.0, 1.0, 2.0, 3.0]:
    direct = np.sum((y - b * x) ** 2)
    expanded = (
        np.sum(y ** 2)
        - 2 * b * np.sum(x * y)
        + b ** 2 * np.sum(x ** 2)
    )
    print(b, direct, expanded)
    assert np.isclose(direct, expanded)
```

53. **[Core A53]** Predict the output for `b=2.0` before running.
54. **[Repeat A54]** Add `b=2.4`. Which tested value has the smallest loss?
55. **[Stretch A55]** Explain why numerical agreement supports the algebra but is not a proof for every possible dataset.

### Block 2 mastery check

On blank paper:

1. expand $(y-bx)^2$;
2. solve $4-2x=10$;
3. solve $a+b=8$ and $a+3b=14$; and
4. translate $\sum_{i=1}^{3}(y_i-\hat y_i)^2$ into ordinary addition.

Do not continue until all four are correct.

---

# Block 3 — Vectors, Length, Angles, and Projection

## 3.1 A vector is a list and an arrow

A vector such as

$$
v=\begin{bmatrix}3\\4\end{bmatrix}
$$

can mean:

- an ordered list of two numbers;
- an arrow from the origin to $(3,4)$; or
- a direction and magnitude.

Vector addition and scalar multiplication are componentwise:

$$
\begin{bmatrix}a\\b\end{bmatrix}
+
\begin{bmatrix}c\\d\end{bmatrix}
=
\begin{bmatrix}a+c\\b+d\end{bmatrix},
\qquad
k\begin{bmatrix}a\\b\end{bmatrix}
=
\begin{bmatrix}ka\\kb\end{bmatrix}.
$$

### Exercises: vector operations

1. **[Core V1]** Add $(2,3)+(4,-1)$.
2. **[Core V2]** Subtract $(5,2)-(1,7)$.
3. **[Core V3]** Calculate $3(2,-4)$.
4. **[Core V4]** Find $2u-v$ for $u=(1,3)$ and $v=(4,2)$.
5. **[Repeat V5]** Draw $(3,1)$ and $2(3,1)$ from the origin. What stays the same?
6. **[Repeat V6]** Draw $v=(2,4)$ and $-v$. What changes?
7. **[Stretch V7]** Find $a$ and $b$ if $a(1,0)+b(0,1)=(7,-2)$.

## 3.2 Length, distance, and Pythagoras

The Euclidean length or norm of

$$
v=(v_1,v_2)
$$

is

$$
\|v\|=\sqrt{v_1^2+v_2^2}.
$$

In more dimensions:

$$
\|v\|=\sqrt{\sum_i v_i^2}.
$$

The distance between $a$ and $b$ is $\|a-b\|$. Squared length is especially important:

$$
\|v\|^2=v^Tv=\sum_i v_i^2.
$$

OLS minimises the squared residual-vector length.

### Exercises: norms and distances

8. **[Core V8]** Find $\|(3,4)\|$.
9. **[Core V9]** Find $\|(5,12)\|$.
10. **[Core V10]** Find $\|(1,2,2)\|$.
11. **[Core V11]** Find the distance between $(1,2)$ and $(4,6)$.
12. **[Core V12]** If residuals are $(3,-4)$, find SSR and residual-vector length.
13. **[Repeat V13]** Compare $\|(6,8)\|$ with $\|2(3,4)\|$.
14. **[Repeat V14]** Is $\|(-3,-4)\|$ negative? Explain.
15. **[Repeat V15]** Find all values of $a$ for which $\|(a,0)\|=7$.
16. **[Stretch V16]** Verify $\|v\|^2=v\cdot v$ for $v=(2,-3,6)$.

## 3.3 The dot product as arithmetic and geometry

The dot product is

$$
u\cdot v=u^Tv=\sum_i u_i v_i.
$$

It also obeys

$$
u\cdot v=\|u\|\|v\|\cos\theta,
$$

where $\theta$ is the angle between nonzero vectors.

Therefore:

- $u\cdot v>0$: acute angle, broadly similar direction;
- $u\cdot v<0$: obtuse angle, broadly opposing direction;
- $u\cdot v=0$: right angle; the vectors are **orthogonal**.

### Exercises: dot products and angles

17. **[Core V17]** Calculate $(1,2)\cdot(3,4)$.
18. **[Core V18]** Calculate $(2,-1)\cdot(3,6)$.
19. **[Core V19]** Show that $(1,2)$ and $(2,-1)$ are perpendicular.
20. **[Core V20]** Is the angle between $(1,0)$ and $(1,1)$ acute, right, or obtuse?
21. **[Core V21]** Is the angle between $(1,0)$ and $(-2,1)$ acute, right, or obtuse?
22. **[Repeat V22]** Find $a$ so that $(3,a)\cdot(2,-1)=0$.
23. **[Repeat V23]** Give a nonzero vector perpendicular to $(4,7)$.
24. **[Repeat V24]** Explain why a zero dot product cannot by itself define an angle if one vector is the zero vector.
25. **[Stretch V25]** Find $\cos\theta$ for $u=(1,1)$ and $v=(1,0)$.

## 3.4 Projection onto a line

The projection of $y$ onto a nonzero vector $x$ is

$$
\operatorname{proj}_x(y)
=
\frac{x^Ty}{x^Tx}x.
$$

The scalar

$$
\hat b=\frac{x^Ty}{x^Tx}
$$

tells how much of direction $x$ is needed. The projected vector is $\hat y=x\hat b$. The residual is $e=y-\hat y$, and it is perpendicular to $x$:

$$
x^Te=0.
$$

### Worked example

Let $x=(1,1)$ and $y=(3,1)$. Then

$$
\hat b=\frac{(1)(3)+(1)(1)}{1^2+1^2}=\frac{4}{2}=2.
$$

So $\hat y=(2,2)$ and $e=(1,-1)$. Check:

$$
x^Te=(1)(1)+(1)(-1)=0.
$$

### Exercises: projection

26. **[Core V26]** Project $y=(4,2)$ onto $x=(1,0)$.
27. **[Core V27]** Project $y=(3,1)$ onto $x=(1,1)$.
28. **[Core V28]** For V27, calculate the residual and verify orthogonality.
29. **[Core V29]** Project $y=(2,5)$ onto $x=(0,1)$.
30. **[Repeat V30]** Project $y=(5,1)$ onto $x=(1,2)$.
31. **[Repeat V31]** For V30, verify that $y=\hat y+e$.
32. **[Repeat V32]** Why is projection called the closest point on the line?
33. **[Stretch V33]** If $x$ is multiplied by 10, show that the projected vector does not change.

## 3.5 The Pythagorean reason projection minimises error

Let $\hat y$ be the projection of $y$ onto a line. Any other point $q$ on the line differs from $\hat y$ by a vector lying along the line. The residual $y-\hat y$ is perpendicular to that difference. Therefore:

$$
\|y-q\|^2
=
\|y-\hat y\|^2+\|\hat y-q\|^2.
$$

The second term cannot be negative. Thus no other $q$ is closer than $\hat y$.

### Exercises: geometric reasoning

34. **[Core V34]** In the Pythagorean identity above, which term is zero only when $q=\hat y$?
35. **[Core V35]** Why can adding $\|\hat y-q\|^2$ never reduce the distance?
36. **[Core V36]** Translate “the residual is orthogonal to the fitted direction” into a dot-product equation.
37. **[Repeat V37]** Use $y=(3,1)$, $\hat y=(2,2)$, and $q=(1,1)$ to verify the identity numerically.
38. **[Repeat V38]** Explain in ordinary language why perpendicularity identifies the nearest point.
39. **[Stretch V39]** Draw the right triangle formed by $y$, $\hat y$, and another attainable $q$.

### Code lab V

```python
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 2.0, 5.0])

b_hat = (x @ y) / (x @ x)
y_hat = x * b_hat
e = y - y_hat

print("b_hat:", b_hat)
print("y_hat:", y_hat)
print("residual:", e)
print("orthogonality:", x @ e)
assert np.isclose(x @ e, 0.0)
```

40. **[Core V40]** Compute `b_hat` by hand.
41. **[Core V41]** Why is `np.isclose` used rather than `==`?
42. **[Repeat V42]** Replace `y` with `2*x`. Predict the residual before running.
43. **[Stretch V43]** Test three other values of `b` and confirm none has smaller SSR.

---

# Block 4 — Matrices, Column Space, Rank, and Linear Systems

## 4.1 Matrix shape and transpose

A matrix is a rectangular table. If $X$ has $n$ rows and $p$ columns, its shape is $n\times p$.

In regression:

- rows are observations;
- columns are features;
- $y$ contains one target per row;
- $\beta$ contains one coefficient per design column.

The transpose $X^T$ swaps rows and columns. If $X$ is $n\times p$, then $X^T$ is $p\times n$.

### Exercises: shapes and transpose

1. **[Core M1]** State the shape of a matrix with 8 observations and 3 features.
2. **[Core M2]** If $X$ is $8\times3$, what is the shape of $X^T$?
3. **[Core M3]** Transpose $\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}$.
4. **[Core M4]** If $\beta$ has one coefficient for each of 4 design columns, what is its shape as a one-dimensional NumPy array?
5. **[Repeat M5]** Explain the difference between NumPy shapes `(3,)`, `(3, 1)`, and `(1, 3)`.
6. **[Repeat M6]** If one intercept column is added to 5 features, how many design columns are there?
7. **[Stretch M7]** Why must the order of feature columns at prediction match the order used during fitting?

## 4.2 Matrix–vector multiplication

For

$$
X=
\begin{bmatrix}
x_{11}&x_{12}\\
x_{21}&x_{22}\\
x_{31}&x_{32}
\end{bmatrix},
\qquad
\beta=
\begin{bmatrix}
\beta_1\\
\beta_2
\end{bmatrix},
$$

the product is

$$
X\beta=
\begin{bmatrix}
x_{11}\beta_1+x_{12}\beta_2\\
x_{21}\beta_1+x_{22}\beta_2\\
x_{31}\beta_1+x_{32}\beta_2
\end{bmatrix}.
$$

Each output is a row–coefficient dot product. Inner dimensions must match:

$$
(n\times p)(p\times1)=(n\times1).
$$

### Exercises: products and predictions

8. **[Core M8]** Calculate
$
\begin{bmatrix}1&2\\3&4\end{bmatrix}
\begin{bmatrix}5\\6\end{bmatrix}.
$
9. **[Core M9]** Calculate predictions for
$
X=\begin{bmatrix}1&0\\1&2\\1&5\end{bmatrix}
$
and
$
\beta=(10,3)^T.
$
10. **[Core M10]** In M9, what role does the first column play?
11. **[Core M11]** Can a $4\times3$ matrix multiply a length-4 coefficient vector on the right? Why?
12. **[Repeat M12]** State the output shape of $(12\times4)(4\times1)$.
13. **[Repeat M13]** Expand the second row of
$
\begin{bmatrix}1&2&3\\1&4&5\end{bmatrix}\beta
$
term by term.
14. **[Repeat M14]** Write a plain Python loop that computes each row dot product.
15. **[Stretch M15]** Explain why elementwise multiplication `X * beta` is not the same object as `X @ beta`.

## 4.3 Matrix–matrix multiplication

If $A$ is $m\times n$ and $B$ is $n\times p$, then $AB$ is $m\times p$. Entry $(i,j)$ is row $i$ of $A$ dotted with column $j$ of $B$.

### Exercises: matrix products

16. **[Core M16]** State whether $(3\times2)(2\times4)$ is valid and give the output shape.
17. **[Core M17]** State whether $(3\times2)(3\times4)$ is valid.
18. **[Core M18]** Calculate
$
\begin{bmatrix}1&2\\0&1\end{bmatrix}
\begin{bmatrix}3&1\\4&2\end{bmatrix}.
$
19. **[Repeat M19]** Show with a small example that generally $AB\ne BA$.
20. **[Repeat M20]** If $X$ is $n\times p$, state the shapes of $X^TX$ and $X^Ty$.
21. **[Stretch M21]** Why is $X^TX$ always square?

## 4.4 Linear combinations, span, and column space

Multiplying $X\beta$ forms a linear combination of the columns of $X$:

$$
X\beta=\beta_1X_{\cdot1}+\cdots+\beta_pX_{\cdot p}.
$$

The **span** of some vectors is every vector obtainable from their linear combinations. The **column space** of $X$, written $\mathcal C(X)$, is the span of its columns.

Every fitted vector $\hat y=X\hat\beta$ lies in $\mathcal C(X)$. OLS selects the point in that space closest to $y$.

### Exercises: span and column space

22. **[Core M22]** Write $X\beta$ as a combination of the columns when
$
X=[c_1\ c_2]
$
and $\beta=(3,-2)^T$.
23. **[Core M23]** Is $(5,0)$ in the span of $(1,0)$? If so, give the multiplier.
24. **[Core M24]** Is $(2,3)$ in the span of $(1,0)$?
25. **[Core M25]** Show that $(5,7)$ is in the span of $(1,0)$ and $(0,1)$.
26. **[Repeat M26]** Describe the span of one nonzero vector in $\mathbb R^2$.
27. **[Repeat M27]** Describe the span of two nonparallel vectors in $\mathbb R^2$.
28. **[Repeat M28]** Why must $\hat y=X\hat\beta$ belong to the column space?
29. **[Stretch M29]** With an intercept and one feature for three observations, which two vectors span the attainable predictions?

## 4.5 Independence and rank

Columns are linearly independent when no column can be constructed exactly from the others. Matrix **rank** counts independent directions.

For a design matrix with $p$ columns:

- full column rank means $\operatorname{rank}(X)=p$;
- rank deficiency means $\operatorname{rank}(X)<p$;
- full column rank makes $X^TX$ invertible and the OLS coefficient vector unique.

Example of dependence:

$$
X=
\begin{bmatrix}
1&2\\
2&4\\
3&6
\end{bmatrix}.
$$

The second column is twice the first, so the rank is 1, not 2.

### Exercises: independence and rank

30. **[Core M30]** Find the rank of
$
\begin{bmatrix}1&0\\0&1\end{bmatrix}.
$
31. **[Core M31]** Find the rank of
$
\begin{bmatrix}1&2\\2&4\end{bmatrix}.
$
32. **[Core M32]** Is a column of metres independent of the same distance column in kilometres?
33. **[Core M33]** A design has 3 columns and rank 2. Is the coefficient vector unique?
34. **[Core M34]** Can an $n\times p$ matrix have rank greater than $\min(n,p)$?
35. **[Repeat M35]** Give a $3\times2$ full-column-rank matrix.
36. **[Repeat M36]** Make M35 rank deficient by changing only its second column.
37. **[Repeat M37]** Explain why a numerical solver can return one answer even when coefficients are not uniquely identified.
38. **[Stretch M38]** Explain why duplicate information harms coefficient interpretation even when predictions remain possible.

## 4.6 Linear systems and inverses

A system can be written

$$
A\beta=c.
$$

If square $A$ has an inverse,

$$
A^{-1}A=I,
$$

then

$$
\beta=A^{-1}c.
$$

An inverse “undoes” a linear transformation. Not every square matrix has one. A matrix with dependent columns is singular and not invertible.

Chapter 1 derives

$$
(X^TX)\hat\beta=X^Ty.
$$

These are the normal equations. Although the symbolic solution is

$$
\hat\beta=(X^TX)^{-1}X^Ty,
$$

application code should use a least-squares solver, not explicitly compute the inverse.

### Exercises: systems and normal equations

39. **[Core M39]** Solve
$
\begin{bmatrix}1&0\\0&2\end{bmatrix}
\begin{bmatrix}a\\b\end{bmatrix}
=
\begin{bmatrix}5\\8\end{bmatrix}.
$
40. **[Core M40]** What does $I\beta$ equal?
41. **[Core M41]** Why does dependent-column matrix $X$ make $X^TX$ non-invertible?
42. **[Core M42]** State the normal equations.
43. **[Repeat M43]** For
$
X=\begin{bmatrix}1\\2\\3\end{bmatrix}
$
and $y=(2,2,5)^T$, calculate $X^TX$ and $X^Ty$.
44. **[Repeat M44]** Solve the one-parameter normal equation from M43.
45. **[Stretch M45]** Distinguish the mathematical closed form from the numerically preferred implementation.

## 4.7 OLS geometry with several columns

The fitted vector is the projection of $y$ onto $\mathcal C(X)$. The residual

$$
e=y-X\hat\beta
$$

is perpendicular to **every design column**:

$$
X^Te=0.
$$

When the design includes a column of ones, one row of this equation says

$$
\sum_i e_i=0.
$$

### Exercises: multi-column geometry

46. **[Core M46]** If $X$ has 3 columns, how many orthogonality conditions are in $X^Te=0$?
47. **[Core M47]** Explain why $X^Te=0$ implies each design column has zero dot product with the residual.
48. **[Core M48]** What special residual property follows from an intercept column?
49. **[Repeat M49]** If $e=(1,-2,1)$, verify it is perpendicular to the intercept vector $(1,1,1)$.
50. **[Repeat M50]** Let
$
X=\begin{bmatrix}1&0\\1&1\\1&2\end{bmatrix}
$
and $e=(1,-2,1)^T$. Verify $X^Te=0$.
51. **[Stretch M51]** Explain why satisfying $X^Te=0$ locates the least-squares projection when the objective is convex.

### Code lab M

```python
X = np.array([
    [1.0, 0.0],
    [1.0, 1.0],
    [1.0, 2.0],
    [1.0, 3.0],
])
y = np.array([7.0, 11.0, 14.0, 19.0])

beta_hat, _, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)
y_hat = X @ beta_hat
e = y - y_hat

print("shape X:", X.shape)
print("beta:", beta_hat)
print("rank:", rank)
print("X.T @ e:", X.T @ e)
assert rank == X.shape[1]
assert np.allclose(X.T @ e, 0.0, atol=1e-10)
```

52. **[Core M52]** Predict the shapes of `beta_hat`, `y_hat`, and `X.T @ e`.
53. **[Core M53]** Explain both assertions in words.
54. **[Repeat M54]** Add a third column `2 * X[:, 1]`. Predict the new rank.
55. **[Stretch M55]** Run the deficient design and explain why the two slope-like coefficients should not be interpreted separately.

---

# Block 5 — Calculus for Finding the OLS Minimum

## 5.1 A derivative is local slope

For a function $f(x)$, the derivative

$$
f'(x)=\frac{df}{dx}
$$

describes the instantaneous rate of change.

- $f'(x)>0$: the function rises locally as $x$ increases.
- $f'(x)<0$: the function falls locally.
- $f'(x)=0$: the graph is locally flat; this is a stationary point.

Useful rules:

$$
\frac{d}{dx}c=0,\qquad
\frac{d}{dx}x^k=kx^{k-1},
$$

$$
\frac{d}{dx}[af(x)]=af'(x),\qquad
\frac{d}{dx}[f(x)+g(x)]=f'(x)+g'(x).
$$

### Exercises: derivative rules

1. **[Core C1]** Differentiate $7$.
2. **[Core C2]** Differentiate $x$.
3. **[Core C3]** Differentiate $x^2$.
4. **[Core C4]** Differentiate $x^3$.
5. **[Core C5]** Differentiate $5x^2$.
6. **[Core C6]** Differentiate $3x^2+4x-8$.
7. **[Repeat C7]** Differentiate $2x^4-3x^2+6$.
8. **[Repeat C8]** Evaluate the derivative of $x^2$ at $x=-3$ and interpret the sign.
9. **[Stretch C9]** Find where $f(x)=x^2-6x+5$ is stationary.

## 5.2 Chain rule for squared error

If

$$
L(b)=[g(b)]^2,
$$

the chain rule gives

$$
\frac{dL}{db}=2g(b)g'(b).
$$

For one squared residual:

$$
L(b)=(y-bx)^2,
$$

so

$$
\frac{dL}{db}=2(y-bx)(-x)=-2x(y-bx).
$$

The negative sign comes from differentiating $y-bx$ with respect to $b$.

### Exercises: chain rule

10. **[Core C10]** Differentiate $(b-3)^2$.
11. **[Core C11]** Differentiate $(7-2b)^2$.
12. **[Core C12]** Differentiate $(y-bx)^2$ with respect to $b$.
13. **[Core C13]** In C12, which symbols are constants with respect to $b$?
14. **[Repeat C14]** Differentiate $(a+3b)^2$ with respect to $b$.
15. **[Repeat C15]** Differentiate $(y-b_0-b_1x)^2$ with respect to $b_0$.
16. **[Repeat C16]** Differentiate the same expression with respect to $b_1$.
17. **[Stretch C17]** Expand $(7-2b)^2$ first and confirm that ordinary power-rule differentiation matches C11.

## 5.3 From derivative zero to a minimum

A stationary point is a candidate, not automatically a minimum. The second derivative helps:

- $f''(x)>0$: curves upward; local minimum;
- $f''(x)<0$: curves downward; local maximum;
- $f''(x)=0$: inconclusive.

For $f(x)=(x-3)^2$:

$$
f'(x)=2(x-3),\qquad f''(x)=2>0.
$$

Thus $x=3$ is the minimum.

### Exercises: stationary points

18. **[Core C18]** Find and classify the stationary point of $f(x)=x^2-4x+7$.
19. **[Core C19]** Find and classify the stationary point of $f(x)=-x^2+6x$.
20. **[Core C20]** Does $f'(x)=0$ alone prove a minimum? Give a counterexample.
21. **[Repeat C21]** Find the minimum of $3(b-5)^2+2$.
22. **[Repeat C22]** Complete the square or use derivatives to minimise $b^2+8b+20$.
23. **[Stretch C23]** For $f(x)=x^3$, explain why $f'(0)=0$ but $x=0$ is not a minimum.

## 5.4 Deriving no-intercept OLS

The loss is

$$
S(b)=\sum_i(y_i-bx_i)^2.
$$

Differentiate:

$$
\frac{dS}{db}
=
-2\sum_i x_i(y_i-bx_i).
$$

Set it to zero:

$$
\sum_i x_i y_i-b\sum_i x_i^2=0.
$$

Therefore:

$$
\hat b=\frac{\sum_i x_i y_i}{\sum_i x_i^2},
$$

provided $\sum_i x_i^2>0$.

### Exercises: scalar OLS

24. **[Core C24]** Use the formula for $x=(1,2)$ and $y=(2,5)$.
25. **[Core C25]** Calculate the fitted values and residuals for C24.
26. **[Core C26]** Verify $\sum_i x_i e_i=0$ for C24.
27. **[Core C27]** What data condition makes $\sum_i x_i^2=0$?
28. **[Repeat C28]** Fit no-intercept OLS for $x=(1,2,3)$ and $y=(2,4,6)$.
29. **[Repeat C29]** Fit it for $x=(1,2,3)$ and $y=(1,1,1)$.
30. **[Repeat C30]** Differentiate the expanded loss from A51 and solve for its minimum.
31. **[Stretch C31]** Explain why the no-intercept line need not have residuals summing to zero.

## 5.5 Partial derivatives and gradients

When a function has several inputs, a partial derivative changes one input while holding the others fixed.

For

$$
f(a,b)=a^2+3ab+2b^2,
$$

$$
\frac{\partial f}{\partial a}=2a+3b,
\qquad
\frac{\partial f}{\partial b}=3a+4b.
$$

The gradient collects the partial derivatives:

$$
\nabla f=
\begin{bmatrix}
\partial f/\partial a\\
\partial f/\partial b
\end{bmatrix}.
$$

It points in the direction of steepest local increase. At an unconstrained smooth minimum, $\nabla f=0$.

### Exercises: partials and gradients

32. **[Core C32]** For $f(a,b)=a^2+b^2$, find both partial derivatives.
33. **[Core C33]** Find $\nabla f$ for $f(a,b)=a^2+3ab+2b^2$.
34. **[Core C34]** Evaluate the C33 gradient at $(a,b)=(1,2)$.
35. **[Core C35]** Find the stationary point of $f(a,b)=(a-2)^2+(b+1)^2$.
36. **[Repeat C36]** Find the gradient of $f(a,b)=4a^2-2ab+b^2$.
37. **[Repeat C37]** Explain why a zero gradient is the multivariable analogue of a horizontal tangent.
38. **[Stretch C38]** Solve $\nabla f=0$ for $f(a,b)=a^2+ab+b^2-3a$.

## 5.6 The OLS gradient and normal equations

The matrix loss is

$$
S(\beta)=(y-X\beta)^T(y-X\beta).
$$

Its gradient is

$$
\nabla_\beta S=-2X^T(y-X\beta).
$$

At a stationary point:

$$
-2X^T(y-X\hat\beta)=0.
$$

Rearranging gives:

$$
X^TX\hat\beta=X^Ty.
$$

Notice that the calculus result and the geometry result are the same: $X^Te=0$.

### Exercises: connect calculus and geometry

39. **[Core C39]** Substitute $e=y-X\hat\beta$ into the zero-gradient equation.
40. **[Core C40]** Rearrange $X^T(y-X\hat\beta)=0$ into the normal equations.
41. **[Core C41]** Translate $X^T(y-X\hat\beta)=0$ geometrically.
42. **[Repeat C42]** What are the shapes of the gradient and $\beta$?
43. **[Repeat C43]** Why is the gradient zero at the OLS solution?
44. **[Stretch C44]** Explain why the same condition appears from two different routes—projection and differentiation.

## 5.7 Hessians and convexity

The Hessian collects second partial derivatives. For OLS:

$$
H=2X^TX.
$$

For any vector $z$:

$$
z^THz=2z^TX^TXz=2\|Xz\|^2\ge0.
$$

Therefore the OLS loss is convex: there are no deceptive local minima.

- If $X$ has full column rank, $\|Xz\|^2>0$ for every nonzero $z$; the loss is strictly convex and the minimiser is unique.
- If $X$ is rank deficient, some nonzero $z$ has $Xz=0$; the loss has a flat direction and coefficients are not unique.

### Exercises: curvature and uniqueness

45. **[Core C45]** State the OLS Hessian.
46. **[Core C46]** Why is $\|Xz\|^2$ never negative?
47. **[Core C47]** What condition makes the OLS minimiser unique?
48. **[Core C48]** What shape does a positive second derivative suggest in one dimension?
49. **[Repeat C49]** Connect a rank-deficient design to a flat contour direction.
50. **[Repeat C50]** Distinguish convex from strictly convex in terms of uniqueness here.
51. **[Stretch C51]** For
$
X=\begin{bmatrix}1&0\\0&1\end{bmatrix},
$
calculate $2X^TX$.

## 5.8 Finite-difference checking

A central finite difference estimates a derivative:

$$
f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}.
$$

For a gradient, nudge one parameter at a time. Finite differences are a check, not a replacement for understanding. If $h$ is too large, the approximation is crude; if it is extremely tiny, floating-point cancellation can dominate.

### Exercises: derivative checks

52. **[Core C52]** Use $h=0.01$ to approximate the derivative of $f(x)=x^2$ at $x=3$.
53. **[Core C53]** Compare C52 with the exact derivative.
54. **[Repeat C54]** Predict what happens if $h=1$.
55. **[Repeat C55]** Why is “make $h$ as tiny as possible” not a perfect rule?
56. **[Stretch C56]** Write a Python function that finite-difference checks one component of the OLS gradient.

### Code lab C

```python
X = np.array([
    [1.0, 0.0],
    [1.0, 1.0],
    [1.0, 2.0],
])
y = np.array([1.0, 2.0, 2.0])
beta = np.array([0.5, 0.8])

def ssr(beta_value):
    e = y - X @ beta_value
    return e @ e

analytic = -2.0 * X.T @ (y - X @ beta)
numeric = np.zeros_like(beta)
h = 1e-6

for j in range(beta.size):
    step = np.zeros_like(beta)
    step[j] = h
    numeric[j] = (ssr(beta + step) - ssr(beta - step)) / (2 * h)

print("analytic:", analytic)
print("numeric:", numeric)
assert np.allclose(analytic, numeric, rtol=1e-6, atol=1e-8)
```

57. **[Core C57]** Explain what `step[j] = h` accomplishes.
58. **[Core C58]** Explain why the numerator uses two loss evaluations.
59. **[Repeat C59]** Change `h` to `1e-2` and compare.
60. **[Stretch C60]** Fit `beta_hat`, calculate the analytic gradient there, and explain the result.

---

# Block 6 — Floating-Point Discipline and Held-Out Evaluation

## 6.1 Why decimal arithmetic is approximate

Computers store most real-valued calculations in finite binary form. Many ordinary decimals cannot be represented exactly. Thus:

```python
0.1 + 0.2 == 0.3
```

may be `False`. This is not random; it is a representation limitation.

Use:

```python
np.isclose(a, b)
np.allclose(array_a, array_b)
```

Important arguments:

- `atol`: absolute tolerance, useful near zero;
- `rtol`: relative tolerance, useful when magnitudes vary.

Conceptually, closeness is judged by a rule like:

$$
|a-b|\le \text{atol}+\text{rtol}|b|.
$$

### Exercises: numerical comparison

1. **[Core N1]** Why is exact `==` risky for calculated floats?
2. **[Core N2]** Which function compares two scalar floats approximately?
3. **[Core N3]** Which function compares arrays approximately?
4. **[Core N4]** Which tolerance is especially important when the expected value is zero?
5. **[Repeat N5]** Would exact equality be reasonable for two integer row counts? Why?
6. **[Repeat N6]** Why should tolerances not be made so wide that every result passes?
7. **[Repeat N7]** Interpret `np.allclose(X.T @ e, 0, atol=1e-10)`.
8. **[Stretch N8]** Compare `0.1 + 0.2` and `0.3` with `repr` in Python.

## 6.2 Stable least-squares solving and rank decisions

The closed form is valuable mathematics:

$$
\hat\beta=(X^TX)^{-1}X^Ty.
$$

But application code should use:

```python
beta_hat, residual_sums, rank, singular_values = np.linalg.lstsq(
    X, y, rcond=None
)
```

Why:

- explicit inversion adds avoidable numerical error and work;
- `lstsq` is designed for the least-squares problem;
- it returns rank information;
- nearly dependent columns require a numerical threshold, not only symbolic reasoning.

Full numerical conditioning belongs in Chapter 2. For now, inspect the reported rank and avoid interpreting coefficients from a rank-deficient design.

### Exercises: solver literacy

9. **[Core N9]** Which NumPy function should Chapter 1 application code use to fit OLS?
10. **[Core N10]** What rank is required for a design with 4 columns to have full column rank?
11. **[Core N11]** If `rank == 3` for 4 columns, what should happen before interpreting coefficients?
12. **[Core N12]** Does `lstsq` returning numbers prove each coefficient is meaningful?
13. **[Repeat N13]** Give two reasons not to explicitly calculate the inverse in application code.
14. **[Repeat N14]** What do very small singular values warn about informally?
15. **[Stretch N15]** Create a nearly duplicate column and compare its singular values with an exactly duplicate column.

## 6.3 Training and held-out data

If you fit and evaluate on the same observations, the evaluation rewards the model for data it already saw. A **held-out set** provides a small test of performance on unseen observations.

For this bridge:

1. shuffle row indices with a fixed random seed;
2. assign, for example, 75% to training and 25% to held-out data;
3. fit all coefficients using training data only;
4. calculate predictions and metrics separately on both sets; and
5. never move held-out rows back into training because their result is disappointing.

For very small or structured data, one random split is unstable. Chapter 2 develops validation, test separation, and more careful evaluation. Tonight’s goal is only to prevent direct training-data reuse.

### Exercises: split discipline

16. **[Core N16]** Which rows may influence fitted coefficients?
17. **[Core N17]** Which rows estimate performance on unseen data?
18. **[Core N18]** Why set a random seed?
19. **[Core N19]** Name one danger of a single split on a tiny dataset.
20. **[Core N20]** Why is repeatedly checking the held-out result and changing the model a form of leakage?
21. **[Repeat N21]** With 20 rows and a 75/25 split, how many rows are in each set?
22. **[Repeat N22]** Should scaling values such as means be computed before or after splitting? On which set should they be learned?
23. **[Repeat N23]** A rare remote district appears only in the held-out set. What limitation should be reported?
24. **[Stretch N24]** Explain why chronological data may require a time-respecting split rather than random shuffling.

## 6.4 Metrics from definitions

For $m$ evaluation observations:

$$
\operatorname{MSE}=\frac{1}{m}\sum_i e_i^2,
$$

$$
\operatorname{RMSE}=\sqrt{\operatorname{MSE}},
$$

$$
\operatorname{MAE}=\frac{1}{m}\sum_i|e_i|.
$$

RMSE and MAE have the same unit as $y$. RMSE reacts more strongly to large errors because errors are squared.

### Exercises: held-out metrics

25. **[Core N25]** For residuals $(2,-1,3)$, calculate MSE, RMSE, and MAE.
26. **[Core N26]** Which metric is more sensitive to one very large residual: RMSE or MAE?
27. **[Core N27]** If cost is million PKR, what are the units of MSE and RMSE?
28. **[Core N28]** Training RMSE is 1.2 and held-out RMSE is 8.9. What does the gap suggest?
29. **[Repeat N29]** Can a low held-out RMSE establish causation? Why?
30. **[Repeat N30]** Why should training and held-out metrics use the same formula?
31. **[Stretch N31]** Construct two residual sets with the same MAE but different RMSE.

### Code lab N: a clean held-out split

```python
rng = np.random.default_rng(42)

X_features = np.array([
    [2.0, 40.0],
    [4.0, 45.0],
    [5.0, 50.0],
    [7.0, 55.0],
    [9.0, 62.0],
    [11.0, 70.0],
    [13.0, 76.0],
    [15.0, 85.0],
    [17.0, 92.0],
    [20.0, 100.0],
    [22.0, 110.0],
    [25.0, 120.0],
])
y = np.array([18.0, 21.0, 24.0, 28.0, 31.0, 36.0,
              39.0, 44.0, 49.0, 55.0, 59.0, 67.0])

indices = rng.permutation(len(y))
cut = int(0.75 * len(y))
train_idx = indices[:cut]
held_idx = indices[cut:]

X_train_raw = X_features[train_idx]
X_held_raw = X_features[held_idx]
y_train = y[train_idx]
y_held = y[held_idx]

X_train = np.column_stack([np.ones(len(train_idx)), X_train_raw])
X_held = np.column_stack([np.ones(len(held_idx)), X_held_raw])

beta_hat, _, rank, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
if rank < X_train.shape[1]:
    raise ValueError("Training design is rank deficient")

def metrics(y_true, y_pred):
    e = y_true - y_pred
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
    }

print("training:", metrics(y_train, X_train @ beta_hat))
print("held-out:", metrics(y_held, X_held @ beta_hat))
```

32. **[Core N32]** How many training and held-out rows are created?
33. **[Core N33]** Identify the only line that learns coefficients.
34. **[Core N34]** Confirm that held-out targets do not influence `beta_hat`.
35. **[Repeat N35]** Run twice and explain why the split repeats.
36. **[Repeat N36]** Change the seed and explain why the metrics change.
37. **[Stretch N37]** Add a check that both design matrices have the same number of columns.

---

# Block 7 — Cumulative Regression Readiness Challenge

Do this section without the answer key. It deliberately combines every block.

## Dataset

Eight fictional projects have one feature, road distance $x$ in km, and cost $y$ in million PKR:

| Project | $x$ | $y$ |
|---|---:|---:|
| A | 1 | 11 |
| B | 2 | 13 |
| C | 3 | 16 |
| D | 4 | 18 |
| E | 6 | 23 |
| F | 7 | 25 |
| G | 8 | 29 |
| H | 10 | 32 |

Use A–F as training data and G–H as held-out data. This fixed split is for arithmetic practice, not a general recommendation.

## Part 1 — See and describe

1. **[Core X1]** Label suitable axes, units, and scales for a scatter plot.
2. **[Core X2]** Plot the eight points by hand or with Matplotlib.
3. **[Core X3]** Describe the visible direction and whether one line seems plausible.
4. **[Core X4]** Explain why the graph does not prove that moving a site farther from a road would cause its cost to rise.

## Part 2 — Build the training design

5. **[Core X5]** Write the $6\times2$ training design matrix with an intercept.
6. **[Core X6]** State the shapes of $X_{\text{train}}$, $y_{\text{train}}$, and $\beta$.
7. **[Core X7]** Explain what each design column means.
8. **[Core X8]** Determine whether the design has full column rank and explain without software.

## Part 3 — Fit from centred scalar formulas

For one feature with an intercept:

$$
\hat b_1=
\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}
{\sum_i(x_i-\bar x)^2},
\qquad
\hat b_0=\bar y-\hat b_1\bar x.
$$

9. **[Core X9]** Calculate training $\bar x$ and $\bar y$.
10. **[Core X10]** Make columns for $x_i-\bar x$, $y_i-\bar y$, their product, and $(x_i-\bar x)^2$.
11. **[Core X11]** Calculate $\hat b_1$ and $\hat b_0$.
12. **[Core X12]** Interpret both coefficients with units and appropriate caution.

## Part 4 — Predictions, residuals, and geometry

13. **[Core X13]** Calculate all six training fitted values and residuals.
14. **[Core X14]** Verify that the training residuals sum to approximately zero.
15. **[Core X15]** Verify that $\sum_i x_i e_i$ is approximately zero.
16. **[Core X16]** Explain what X14 and X15 mean as dot products.
17. **[Core X17]** Calculate training RMSE and MAE.

## Part 5 — Held-out evaluation

18. **[Core X18]** Predict projects G and H without refitting.
19. **[Core X19]** Calculate held-out residuals, RMSE, and MAE.
20. **[Core X20]** Compare training and held-out results without claiming that two held-out projects are definitive evidence.

## Part 6 — Code proof and deliberate break

21. **[Core X21]** Verify your coefficients using `np.linalg.lstsq`.
22. **[Core X22]** Verify `X_train.T @ residuals` with `np.allclose`.
23. **[Core X23]** Add a third design column equal to `1000 * road_distance`.
24. **[Core X24]** inspect the new rank and explain the failure.
25. **[Repeat X25]** Fit the deficient matrix anyway. Compare predictions with the full-rank model and explain why individual distance coefficients are not interpretable.
26. **[Stretch X26]** Draw or compute an error contour grid for $b_0$ and $b_1$.

## Part 7 — One-page memo

27. **[Core X27]** Write a memo of no more than 150 words containing:

- the prediction question;
- units and training/held-out split;
- the fitted equation;
- training and held-out RMSE/MAE;
- one limitation of the tiny synthetic dataset;
- a clear statement that association is not causation; and
- one justified next step.

---

# Final Exit Check

Answer aloud without notes. You are ready to start Chapter 1 when you can answer at least 16 of 18 clearly.

1. What are the axes and one point in a scatter plot?
2. What do slope and intercept mean, including units?
3. What is a residual, and what does its sign mean?
4. What does a contour join?
5. Expand $(a-b)^2$.
6. Solve a two-equation, two-unknown system.
7. What are the arithmetic and geometric meanings of the dot product?
8. What is a vector norm?
9. What makes two vectors orthogonal?
10. What does projection have to do with minimum distance?
11. What does $X\beta$ mean row-wise and column-wise?
12. What is the column space?
13. What does full column rank mean?
14. What are the normal equations?
15. What does a derivative measure?
16. Why is a zero gradient not the whole uniqueness argument?
17. Why use `allclose` and `lstsq`?
18. Why evaluate on held-out observations?

If you miss:

- 1–4: repeat Block 1;
- 5–6: repeat Block 2;
- 7–10: repeat Block 3;
- 11–14: repeat Block 4;
- 15–16: repeat Block 5;
- 17–18: repeat Block 6.

---

# Concise Answer Key

Attempt first. A concise answer is provided so you can check quickly; where an explanation is requested, your wording may differ if the reasoning is sound.

## Diagnostic

D1 horizontal. D2 $x=4,y=9$. D3 11. D4 $2x+6$. D5 $a^2-2ab+b^2$. D6 $x=4$. D7 5. D8 11. D9 perpendicular/orthogonal. D10 `(8,)`. D11 independent directions. D12 $2x$. D13 no; it can be a maximum or neither. D14 finite binary representation. D15 to estimate performance on unseen observations.

## Block 1

- G1: 7 horizontal, 31 vertical. G2 $(12,48)$. G3 $(0,0)$. G4 $(9,20)$. G5 $(3,40)$. G7 2000 m. G8 truncated scale exaggerates differences. G9 80–100.
- G10 positive. G11 negative. G12 little/no visible relationship. G13 direction changes across $x$. G14 it may be genuine, informative, or a data error requiring investigation. G15 association may reflect other variables. G16 Simpson’s paradox.
- G17 intercept 10 million PKR; slope 3 million PKR/km. G18 10, 16, 25. G19 3. G20 $\hat y=5+2x$. G21 $-2$. G22 $-2$. G23 7. G24 no; forces 0. G25 0.004 million PKR/m. G26 division by $x_2-x_1=0$.
- G27 $3$, underprediction. G28 $-6$, overprediction. G29 predictions $(6,10,14)$; residuals $(1,1,0)$. G30 $-7$. G31 $-3$. G32 systematic underprediction. G33 OLS defines error in the outcome coordinate.
- G35 error/loss. G36 parameter pair. G37 centre/smallest loop. G38 weak identification or highly correlated parameters. G39 one is an observation; the other selects an entire fitted line. G40 each parameter-space point defines one fitted line. G41 $(6,10,14,18)$. G43 all except possibly the first are underpredicted; with slope 2, predictions are $(6,8,10,12)$ and all observed values exceed them.

## Block 2

- A1 $-3$. A2 3. A3 11. A4 21. A5 9 and $-9$. A6 $2/3$. A7 7. A8 9. A9 12. A10 21.
- A11 $3x+12$. A12 $10x-15$. A13 $-2x+12$. A14 $8x-2$. A15 $7x+8$. A16 $2x-22$. A17 $ac$. A18 $-6a+3b-12$. A19 $5x-5y$.
- A20 $x^2-6x+9$. A21 $x^2+10x+25$. A22 $y^2-2ybx+b^2x^2$. A23 $49-28b+4b^2$. A24 both give 25. A25 $a^2-6ac+9c^2$. A26 $4x^2+16x+16$. A27 missing cross-term; subtraction does not distribute through squaring. A28 $y^2+b_0^2+b_1^2x^2-2yb_0-2yb_1x+2b_0b_1x$.
- A29 5. A30 7. A31 7. A32 $-3$. A33 $c/a$. A34 $b_0=y-b_1x$. A35 $x=(y-b_0)/b_1$. A36 4. A37 12. A38 7.
- A39 $a=6,b=3$. A40 $b_0=3,b_1=4$. A41 $x=4,y=3$. A42 $a=2,b=3$. A43 infinitely many pairs on the same line. A44 identical lines.
- A45 $x_1+x_2+x_3$. A46 12 and 56. A47 13. A48 $\sum_i y_i^2-2b\sum_i x_iy_i+b^2\sum_i x_i^2$. A49 $b$ is constant across $i$. A50 $x_i$ varies with $i$. A51 $S(b)=29-24b+5b^2$. A52 at $b=2$, loss 1; residuals $(0,1)$. A53 `1.0 1.0`. A54 $b=2.4$ has loss 0.2, the smallest of those tested.

## Block 3

- V1 $(6,2)$. V2 $(4,-5)$. V3 $(6,-12)$. V4 $(-2,4)$. V5 direction; length doubles. V6 direction reverses, length stays. V7 $a=7,b=-2$.
- V8 5. V9 13. V10 3. V11 5. V12 SSR 25, length 5. V13 both 10. V14 no; norms are nonnegative. V15 $a=\pm7$. V16 both equal 49.
- V17 11. V18 0. V19 dot product $2-2=0$. V20 acute. V21 obtuse. V22 $a=6$. V23 e.g. $(7,-4)$. V24 the angle formula divides by zero norm. V25 $1/\sqrt2$.
- V26 $\hat y=(4,0)$. V27 $\hat y=(2,2)$. V28 $e=(1,-1)$ and dot product 0. V29 $(0,5)$. V30 multiplier $7/5$; $\hat y=(7/5,14/5)$. V31 $e=(18/5,-9/5)$ and sums correctly. V32 all alternative points add a nonnegative perpendicular squared distance. V33 scaling cancels between numerator, denominator, and vector.
- V34 $\|\hat y-q\|^2$. V35 squared norms are nonnegative. V36 $x^Te=0$. V37 left side 4; right side $2+2=4$. V40 $21/14=1.5$. V42 zero vector.

## Block 4

- M1 $8\times3$. M2 $3\times8$. M3 $\begin{bmatrix}1&4\\2&5\\3&6\end{bmatrix}$. M4 `(4,)`. M5 vector, column matrix, row matrix. M6 6. M7 coefficient meaning depends on position.
- M8 $(17,39)^T$. M9 $(10,16,25)^T$. M10 intercept. M11 no; inner dimensions 3 and 4 disagree. M12 $12\times1$. M13 $\beta_1+4\beta_2+5\beta_3$. M15 elementwise result retains/broadcasts matrix dimensions rather than producing row dot products.
- M16 valid, $3\times4$. M17 invalid. M18 $\begin{bmatrix}11&5\\4&2\end{bmatrix}$. M20 $p\times p$ and $p\times1$. M21 the same $p$ appears as rows and columns.
- M22 $3c_1-2c_2$. M23 yes, multiplier 5. M24 no. M25 $5(1,0)+7(0,1)$. M26 a line through the origin. M27 all of $\mathbb R^2$. M28 it is literally a column linear combination. M29 the ones vector and feature vector.
- M30 2. M31 1. M32 no; one is exactly 1000 times the other. M33 no. M34 no. M35 e.g. $\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}$. M36 make column 2 twice column 1. M37 a solver can choose one member of infinitely many minimisers. M38 coefficient allocation across duplicate directions is arbitrary/unstable.
- M39 $a=5,b=4$. M40 $\beta$. M41 dependence provides a nonzero direction mapped to zero. M42 $X^TX\hat\beta=X^Ty$. M43 14 and 21. M44 $\hat b=1.5$. M45 derive with the inverse; compute with a least-squares solver.
- M46 3. M47 each entry of the product is a column–residual dot product. M48 residuals sum to zero. M49 $1-2+1=0$. M50 both dot products are zero. M52 `(2,)`, `(4,)`, `(2,)`. M53 full rank and residual orthogonality. M54 rank remains 2, not 3.

## Block 5

- C1 0. C2 1. C3 $2x$. C4 $3x^2$. C5 $10x$. C6 $6x+4$. C7 $8x^3-6x$. C8 $-6$, locally decreasing as $x$ increases. C9 $x=3$.
- C10 $2(b-3)$. C11 $-4(7-2b)=8b-28$. C12 $-2x(y-bx)$. C13 $x$ and $y$. C14 $6(a+3b)$. C15 $-2(y-b_0-b_1x)$. C16 $-2x(y-b_0-b_1x)$.
- C18 $x=2$, minimum; $f''=2$. C19 $x=3$, maximum; $f''=-2$. C20 no; $-x^2$ at zero. C21 $b=5$, minimum value 2. C22 $b=-4$, minimum 4. C23 the function continues increasing through zero.
- C24 $12/5=2.4$. C25 $(2.4,4.8)$ and $(-0.4,0.2)$. C26 $-0.4+0.4=0$. C27 every $x_i=0$. C28 2 with zero residuals. C29 $6/14=3/7$. C30 derivative $10b-24=0$, so $b=2.4$. C31 without a ones column, orthogonality is to $x$, not necessarily to ones.
- C32 $2a,2b$. C33 $(2a+3b,3a+4b)^T$. C34 $(8,11)^T$. C35 $(2,-1)$. C36 $(8a-2b,-2a+2b)^T$. C38 $(2,-1)$.
- C39 $X^Te=0$. C40 expand to $X^Ty-X^TX\hat\beta=0$ and rearrange. C41 residual perpendicular to every design column. C42 same shape, one entry per coefficient. C43 no local first-order change reduces the loss. C44 both describe the same closest point.
- C45 $2X^TX$. C46 it is a sum of squares. C47 full column rank. C48 upward bowl. C49 a nonzero parameter direction does not change predictions/loss. C50 strict convexity guarantees one minimiser; ordinary convexity may have several. C51 $2I$.
- C52 $[(3.01)^2-(2.99)^2]/0.02=6$. C53 exact derivative is 6. C54 still 6 for this quadratic with central difference. C55 cancellation/rounding eventually dominates. C57 nudges only component $j$. C58 it estimates symmetric change around the current value. C60 gradient is numerically zero.

## Block 6

- N1 finite binary approximation. N2 `np.isclose`. N3 `np.allclose`. N4 `atol`. N5 yes; exact counts are integers. N6 loose tests can hide real mistakes. N7 every design-column/residual dot product is within the chosen tolerance of zero.
- N9 `np.linalg.lstsq`. N10 4. N11 reject/flag rank deficiency before coefficient interpretation. N12 no. N13 greater numerical error and unnecessary work; solver also provides diagnostics. N14 near dependence/poor conditioning.
- N16 training rows. N17 held-out rows. N18 reproducibility. N19 result varies greatly with which few rows are selected. N20 the held-out set begins guiding choices and is no longer unseen. N21 15 and 5. N22 after splitting; learn transformations from training only. N23 the result may be dominated by a group the model never saw. N24 future rows must not inform a model evaluated on the past.
- N25 MSE $14/3\approx4.667$, RMSE $\approx2.160$, MAE 2. N26 RMSE. N27 squared million PKR for MSE; million PKR for RMSE. N28 possible overfitting, shift, outliers, or unstable tiny split; investigate. N29 no; predictive performance does not identify interventions. N31 e.g. $(0,2)$ and $(1,1)$ both have MAE 1, with RMSE $\sqrt2$ and 1.
- N32 9 and 3. N33 the `np.linalg.lstsq` line. N34 only `X_train` and `y_train` enter it. N35 the fixed seed reproduces the permutation. N36 different observations are assigned to each set.

## Cumulative challenge

Use software to confirm arithmetic after attempting it by hand.

- X5:
$
X=
\begin{bmatrix}
1&1\\1&2\\1&3\\1&4\\1&6\\1&7
\end{bmatrix}.
$
X6: $(6,2)$, $(6,)$, `(2,)`. X8 full rank because the distance column is not a constant multiple of ones.
- X9: $\bar x=23/6\approx3.8333$ and $\bar y=106/6\approx17.6667$.
- X10 numerator $\sum(x_i-\bar x)(y_i-\bar y)=191/3\approx63.6667$; denominator $\sum(x_i-\bar x)^2=161/6\approx26.8333$.
- X11 $\hat b_1=382/161\approx2.3727$ and $\hat b_0=60/7\approx8.5714$.
- X12 predicted cost at zero distance is about 8.57 million PKR; each additional km is associated with about 2.37 million PKR higher predicted cost in these training data. This is not a causal effect.
- X13 fitted values are approximately $(10.944,13.317,15.689,18.062,22.807,25.180)$; residuals $(0.056,-0.317,0.311,-0.062,0.193,-0.180)$.
- X14 and X15 are zero apart from rounding. They express orthogonality to the ones and distance columns.
- X17 training RMSE $\approx0.213$ and MAE $\approx0.186$ million PKR.
- X18 predictions for G and H are approximately 27.553 and 32.298.
- X19 held-out residuals are approximately $1.447$ and $-0.298$; RMSE $\approx1.045$ and MAE $\approx0.873$ million PKR.
- X20 the held-out errors are larger, but two observations cannot establish stable generalisation performance.
- X24 rank is 2 for 3 columns because metres are 1000 times kilometres.

---

# What to Carry Into Chapter 1

Keep these six sentences beside you:

1. A graph is meaningless until I read both axes, scales, and units.
2. A residual is observed minus predicted, and OLS minimises their squared length.
3. $X\beta$ is both one row-wise prediction per observation and a linear combination of design columns.
4. OLS projects $y$ onto the column space, so $X^Te=0$.
5. Differentiating the squared-error objective produces the same normal equations as the geometry.
6. A computed coefficient is not automatically unique, stable, causal, or useful on unseen data.

You are now prepared to read Chapter 1 in two passes: first for meaning and computation, then for geometry and derivation.
