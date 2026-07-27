# Chapter 0.5 — Regression Readiness Explanations

## The companion reading pages for the exercise workbook

This document contains the explanatory teaching pages that sit beside **Chapter 0.5 — Regression Readiness Workout**. Read one section, then complete the matching exercises in the workbook. The labels below point to the relevant exercise families.

The aim is not to make you memorise formulas. It is to make every formula say something ordinary:

> What objects are present? What operation is being performed? What does the result mean for a prediction?

---

## 1. Graphs: seeing data before calculating

A graph is a map with a coordinate system. The horizontal axis is usually called the $x$-axis and the vertical axis the $y$-axis. A point $(x,y)$ records two numbers in that order: move $x$ units horizontally, then $y$ units vertically.

For an MHP project, a point might be $(12,48)$: 12 km from an all-weather road and a cost of 48 million PKR. The axes must state units. A chart showing “12” without saying whether it means kilometres or metres is not yet an interpretable chart.

The origin $(0,0)$ is useful as a reference, but an axis does not have to begin there. A chart that begins its cost axis at 80 million PKR and ends at 100 million PKR can make a small difference look enormous. Before interpreting a pattern, read the labels, tick spacing, scale, and units.

### Scatter plots

A scatter plot places one point per observation. An upward cloud means that larger values of the input tend to accompany larger values of the outcome. A downward cloud means they tend to accompany smaller values. A shapeless cloud contains no obvious straight-line pattern. A curve warns that one straight line may be too simple.

These are descriptions of association. They are not automatically causal claims. Remote sites may have longer road distances and more difficult terrain. If both features rise together with cost, road distance can be associated with cost even when terrain is the more important practical factor.

Simpson’s paradox is the particularly important warning in Chapter 1: a pattern within each group can reverse when groups are combined. Always ask which groups were pooled and whether group membership is related to both the input and the outcome.

### Straight lines

A fitted line has the form

$$
\hat y=b_0+b_1x.
$$

The hat means “predicted,” not observed. The intercept $b_0$ is the prediction at $x=0$. The slope $b_1$ is the predicted change for a one-unit increase in $x$.

If cost is measured in million PKR and distance in km, then $b_1$ has units of million PKR per km. If distance is changed to metres, the numerical coefficient changes by a factor of 1,000, but the predicted costs do not change when the conversion is done correctly.

The slope through two points is rise over run:

$$
b_1=\frac{y_2-y_1}{x_2-x_1}.
$$

This is a visual idea and an algebraic operation at the same time.

### Residuals

For observation $i$, the residual is

$$
e_i=y_i-\hat y_i.
$$

Observed minus predicted is the safest order to remember. A positive residual means the model underpredicted. A negative residual means it overpredicted. On a scatter plot, the residual is the signed vertical distance from the fitted line to the point.

OLS will choose the line that makes the total squared residual distance as small as possible. That choice is about prediction error in the observed outcome direction; it is not about minimising horizontal distance.

### Contours

When the adjustable parameters are $b_0$ and $b_1$, the sum of squared residuals is a surface over parameter space. A contour is a line joining parameter pairs with equal loss. Nested contours around a centre indicate lower loss toward the centre. A long, thin valley means several parameter combinations give similar predictions. A perfectly flat direction indicates that some coefficients are not uniquely identified.

**Read next:** workbook G1–G44.

---

## 2. Algebra: making the OLS derivation readable

Algebra is the language used to move from a verbal objective (“make squared errors small”) to a computable coefficient. Most OLS mistakes are sign, bracket, or shape mistakes rather than deep statistical mistakes.

### Order, signs, and squares

Use brackets first, then powers, multiplication/division, and finally addition/subtraction. Brackets matter especially with negative numbers:

$$
(-3)^2=9,\qquad -3^2=-9.
$$

The square of a residual is nonnegative. That is why a large positive and a large negative error both count as large mistakes after squaring.

### Distributing and expanding

The distributive law is

$$
a(b+c)=ab+ac.
$$

The square identity used constantly in regression is

$$
(a-b)^2=a^2-2ab+b^2.
$$

The middle term is not optional. For a no-intercept line, one residual is $y-bx$, so:

$$
(y-bx)^2=y^2-2ybx+b^2x^2.
$$

Summing over observations gives:

$$
\sum_i(y_i-bx_i)^2
=\sum_i y_i^2-2b\sum_i x_iy_i+b^2\sum_i x_i^2.
$$

The quantities involving the data can be calculated once. The remaining expression is a quadratic in $b$.

### Rearranging equations

An equation is a balance. Whatever operation is applied to one side must be applied to the other. “Moving a term across” is shorthand for adding or subtracting it on both sides, which changes its sign.

For example:

$$
3x+5=20
\Rightarrow 3x=15
\Rightarrow x=5.
$$

The same discipline applies to the normal equations. Starting with

$$
X^T(y-X\hat\beta)=0,
$$

distribute $X^T$:

$$
X^Ty-X^TX\hat\beta=0,
$$

then move one term:

$$
X^TX\hat\beta=X^Ty.
$$

### Simultaneous equations

Two unknowns require two independent equations. Substitution or elimination removes one unknown. If the equations are identical, there are infinitely many solutions; if they contradict one another, there is no solution. This is the small-system version of matrix rank.

**Read next:** workbook A1–A55.

---

## 3. Vectors and geometry: why projection is OLS

A vector is both a list of numbers and an arrow. The list interpretation supports calculation; the arrow interpretation supplies intuition.

For vectors $u$ and $v$, addition is component by component. Multiplication by a scalar stretches, shrinks, or reverses an arrow. The Euclidean length is

$$
\|v\|=\sqrt{v_1^2+\cdots+v_p^2}.
$$

The squared length is

$$
\|v\|^2=v^Tv=\sum_jv_j^2.
$$

That last expression is exactly the form used for a sum of squared residuals.

### Dot products

The dot product is first a weighted sum:

$$
u\cdot v=u^Tv=\sum_j u_jv_j.
$$

It is also an angle measurement:

$$
u\cdot v=\|u\|\|v\|\cos\theta.
$$

For nonzero vectors, a zero dot product means a right angle. We call the vectors orthogonal. A positive dot product indicates a generally similar direction; a negative one indicates a generally opposing direction.

### Projection onto a line

Suppose the only predictions you can make are multiples of a nonzero vector $x$. Every attainable prediction has the form $xb$. The projection of $y$ onto that line is obtained by choosing

$$
\hat b=\frac{x^Ty}{x^Tx},\qquad \hat y=x\hat b.
$$

The residual is $e=y-\hat y$. The crucial property is

$$
x^Te=0.
$$

The residual is perpendicular to the line of attainable predictions.

Why does that mean “closest”? Take another attainable point $q$ on the line. The three vectors form a right triangle, so:

$$
\|y-q\|^2=\|y-\hat y\|^2+\|\hat y-q\|^2.
$$

The second term is nonnegative. Any other point is therefore at least as far away, with equality only at the projection.

### Several columns

With a design matrix $X$, the attainable predictions are all linear combinations of its columns. Their set is the column space $\mathcal C(X)$. OLS projects $y$ onto this space. The residual is perpendicular to every column:

$$
X^Te=0.
$$

If one column is the vector of ones, one of these orthogonality statements says the residuals sum to zero.

**Read next:** workbook V1–V43 and M46–M51.

---

## 4. Matrices: the bookkeeping of many predictions

A matrix is a rectangular table. In a regression design matrix, rows represent observations and columns represent features or the intercept. If there are $n$ observations and $p$ design columns, $X$ has shape $n\times p$.

### Shapes are meaning

NumPy distinguishes `(p,)`, `(p, 1)`, and `(1, p)`. They may contain related numbers but participate in different broadcasting and multiplication rules. Check shapes before interpreting a result.

The transpose $X^T$ swaps rows and columns. Thus an $n\times p$ design becomes $p\times n$.

### Matrix–vector multiplication

If $X$ is $n\times p$ and $\beta$ is length $p$, then:

$$
X\beta
$$

is a length-$n$ vector of predictions. Row by row, each prediction is the dot product of one observation’s feature values with the coefficient vector. Column by column, the same product is a linear combination of the design columns.

The `@` operator means matrix multiplication in NumPy. The `*` operator means elementwise multiplication with broadcasting. They are different operations.

### Column space and rank

The column space is every vector that can be built by combining the columns. Rank counts the independent directions among those columns. Full column rank means no design column is an exact linear combination of the others.

If a distance column is recorded once in kilometres and once in metres, one column is exactly 1,000 times the other. The matrix can still produce predictions, but the two coefficients cannot be interpreted separately: many coefficient pairs produce the same combined contribution.

Full column rank gives a unique coefficient vector and makes $X^TX$ invertible. Rank deficiency creates a flat direction in parameter space.

### Normal equations and solvers

The OLS normal equations are

$$
X^TX\hat\beta=X^Ty.
$$

The symbolic closed form is $(X^TX)^{-1}X^Ty$ when full column rank holds. In application code, use `np.linalg.lstsq`. It avoids explicitly forming an inverse and reports the estimated rank and singular values.

**Read next:** workbook M1–M55.

---

## 5. Calculus: why the loss has a minimum

A derivative is a local slope. A positive derivative means the function rises as its input increases; a negative derivative means it falls. A stationary point has derivative zero, but that alone does not tell us whether the point is a minimum, maximum, or flat turning point.

The power rule is

$$
\frac{d}{dx}x^k=kx^{k-1}.
$$

Constants differentiate to zero. Derivatives distribute over sums and constant multiples.

### Squared error and the chain rule

For one residual loss:

$$
L(b)=(y-bx)^2.
$$

The inner expression is $y-bx$ and its derivative with respect to $b$ is $-x$. The chain rule therefore gives:

$$
\frac{dL}{db}=-2x(y-bx).
$$

Summing over observations and setting the derivative to zero gives:

$$
\hat b=\frac{\sum_i x_iy_i}{\sum_i x_i^2},
$$

provided the denominator is not zero.

### Partial derivatives and gradients

With several parameters, a partial derivative changes one parameter while holding the others fixed. The gradient stacks all partial derivatives into one vector:

$$
\nabla f=
\begin{bmatrix}
\partial f/\partial\beta_1\\
\vdots\\
\partial f/\partial\beta_p
\end{bmatrix}.
$$

At a smooth unconstrained minimum, the gradient is zero. For matrix OLS:

$$
S(\beta)=(y-X\beta)^T(y-X\beta),
$$

$$
\nabla_\beta S=-2X^T(y-X\beta).
$$

Setting this to zero reproduces the normal equations. Calculus and geometry are describing the same fitted point from different angles.

### Hessian and convexity

The Hessian is the matrix of second derivatives. For OLS:

$$
H=2X^TX.
$$

For any vector $z$:

$$
z^THz=2\|Xz\|^2\ge0.
$$

So the loss is convex: it has no misleading local minima. Full column rank makes it strictly convex and gives one unique minimiser. Rank deficiency leaves a flat direction, so several coefficient vectors can minimise the loss equally well.

Finite differences provide a numerical check of a symbolic gradient. They are evidence that the implementation matches the formula, not a substitute for deriving or understanding it.

**Read next:** workbook C1–C60.

---

## 6. Numerical discipline and held-out evaluation

### Floating-point comparison

Most decimal fractions are not represented exactly in binary floating-point. Tiny discrepancies such as `0.1 + 0.2` differing from `0.3` are expected. Use `np.isclose` for scalars and `np.allclose` for arrays. A tolerance should reflect the scale and purpose of the calculation: too strict can reject harmless rounding, while too loose can conceal a real bug.

### Rank and stable computation

The mathematical inverse formula is useful for deriving OLS, but direct inversion is not the preferred numerical implementation. `np.linalg.lstsq(X, y, rcond=None)` solves the least-squares problem and returns rank information. If the reported rank is lower than the number of design columns, pause before interpreting coefficients.

### Training versus held-out observations

Training data determine the coefficients. Held-out data are not used to fit the coefficients; they provide a small check of performance on observations the model did not see.

The clean sequence is:

1. split indices with a documented random seed;
2. build training and held-out arrays;
3. fit using training arrays only;
4. calculate predictions and metrics separately; and
5. report how small, unusual, or unrepresentative the split may be.

If the held-out result is repeatedly inspected and used to make model choices, it gradually stops being genuinely held out. That is leakage. Chapter 2 develops more careful train/validation/test practice.

### Metrics

For residuals $e_i$ over $m$ evaluation observations:

$$
\operatorname{MSE}=\frac1m\sum_i e_i^2,
\qquad
\operatorname{RMSE}=\sqrt{\operatorname{MSE}},
\qquad
\operatorname{MAE}=\frac1m\sum_i|e_i|.
$$

RMSE and MAE have the same units as the outcome. RMSE reacts more strongly to an unusually large error because it squares before averaging. A low held-out error supports a predictive statement; it does not establish a causal effect.

**Read next:** workbook N1–N37.

---

## 7. How the ideas join in OLS

The full chain is now short enough to say aloud:

1. A dataset gives observations, features, and a target with units.
2. A design matrix $X$ arranges the features and, when requested, a column of ones for the intercept.
3. A coefficient vector $\beta$ produces predictions $X\beta$.
4. The residual vector is $e=y-X\beta$.
5. Squared residual length is $e^Te$.
6. OLS chooses the attainable prediction with the smallest squared length.
7. Geometry says the residual is perpendicular to every design column: $X^Te=0$.
8. Calculus says the gradient is zero: $-2X^Te=0$.
9. Both routes produce $X^TX\hat\beta=X^Ty$.
10. Rank, numerical tolerances, and held-out evaluation determine whether the answer is unique, computed reliably, and useful beyond the rows used to fit it.

This is the mental bridge into Chapter 1. The formulas are not separate tricks; they are the same projection problem written in graph, algebraic, geometric, matrix, and calculus languages.

## Reading order

Read sections 1–2 before Chapter 1 Days 1–3, sections 3–4 before Day 4, section 5 before Day 5, and section 6 before the capstone. Use the exercise workbook’s **Core** labels as the minimum route and **Repeat** labels whenever an explanation does not yet feel automatic.
