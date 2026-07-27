# DAY 0E VIDEO — PART 1 SHOOTING SCRIPT
## "The Derivative, From Nothing" — Executed Verbatim

**Target Runtime:** 38–42 minutes (Part 1 of 3)
**Format:** 1920x1080 (TITLE_BAR: 1920x70, STAGE: 1150x930, PANEL: 770x930)
**Accent Color:** `#EC4899` (Currently changing element)
**Background/Static:** `#0F172A` / `#E5E7EB`
**Error/Breakage Color:** `#EF4444`

---

### S1.01 — Cold Open (0:00–0:35)
- **STAGE:** Black screen. Text fades in (54pt): `"take the derivative of the squared error with respect to the slope"`. Text dissolves, leaving `derivative` in `#EC4899`.
- **PANEL:** Empty black.
- **NARRATION:**
> That sentence is going to appear in Chapter One of this book, and right now it might read like a foreign language. By the end of this video, it will read like an instruction. Not an impressive instruction. A boring one. Something you could carry out on paper in about forty seconds. That is the entire goal here. Not to make you a mathematician — to make one specific sentence stop being frightening.

---

### S1.02 — The One-Sentence Thesis (0:35–1:10)
- **STAGE:** Centred 60pt: `A derivative is the slope of a curve at one exact point.` Underline `one exact point` in `#EC4899`.
- **PANEL:** 3-item roadmap:
  - `Part 1 — one dial to turn`
  - `Part 2 — many dials at once`
  - `Part 3 — turning them automatically`
- **NARRATION:**
> Here is the whole idea, and everything else in this video is a consequence of it. A derivative is the slope of a curve at one exact point. That is it. That is the definition. Machine learning uses it because training a model means turning dials until the error stops going down, and the slope is what tells you which way to turn. This video comes in three parts. Part one: one dial. Part two: many dials at once. Part three: turning them automatically, which is what a computer actually does when you train a model.

---

### S1.03 — A Straight Line Has One Slope Everywhere (1:10–2:20)
- **STAGE:** Plot `cost = 1.1 * cable_km + 2`. Points at x=10 and x=30 in `#EC4899`. Triangle: `run = 20`, `rise = 22`. Ratio `22/20 = 1.1`. Move triangle to x=5 and x=25, ratio stays `1.1`.
- **PANEL:** `cost = 1.1 * cable_km + 2`, `slope m = 1.1`.
- **NARRATION:**
> Start with something you already understand. Here is a straight-line model of project cost. Every extra kilometre of cable adds one point one million rupees. The slope is one point one, and here is the important part — it is one point one everywhere. Measure it here. Measure it over here instead. Same number. A straight line has one slope, and that slope is the whole story of the line. This is the comfortable case, and it is about to stop being available.

---

### S1.04 — A Curve Does Not Have One Slope (2:20–3:30)
- **STAGE:** Morph into `y = x²`. Attempt triangle trick at x=0, x=1.5, x=3. Ratios stacked in corner in `#EC4899`, flash red `?` in `#EF4444`.
- **PANEL:** `slope at x = 0 → ?`, `slope at x = 1.5 → ?`, `slope at x = 3 → ?`.
- **NARRATION:**
> Now a curve. Same trick, three places. Near zero the curve is nearly flat. Out at three it is climbing steeply. And halfway between, something in between. Three measurements, three different answers. So the question "what is the slope of this curve" is not a question with an answer. It is a badly formed question. The well-formed question is: what is the slope of this curve at this specific point I am pointing at. And answering that is what a derivative does.

---

### S1.05 — The Secant Line (3:30–4:40)
- **STAGE:** Zoomed curve `y = x²` at x=3. Fixed point `A = (3, 9)` white dot. Point `B = (3+h, (3+h)²)` in `#EC4899` with `h = 1`. Secant line drawn, slope `7.0`.
- **PANEL:** `A = (3, 9)`, `B = (3+h, (3+h)²)`, `slope = (f(3+h) - f(3))/h`, `h = 1 → slope = 7.0`.
- **NARRATION:**
> Here is the trick, and it is genuinely the only trick in this whole subject. I cannot measure the slope at a single point, because a slope needs two points. So I will cheat. I will keep point A fixed where I actually care — at x equals three — and put a second point B a little further along, a distance h away. Two points, so now I can draw a line, and a line has a slope. With h equal to one, that slope is seven. That is not the answer I want. But watch what happens when I make h smaller.

---

### S1.06 — Sliding B Toward A (The Key Animation) (4:40–6:00)
- **STAGE:** Continuous animation of B sliding toward A. h counts down 1.0, 0.5, 0.25, 0.1, 0.05, 0.01. At merge: flash line to `#EC4899`, label `tangent`, print `slope → 6`.
- **PANEL:** Table: `h=1.00 -> 7.00`, `0.50 -> 6.50`, `0.25 -> 6.25`, `0.10 -> 6.10`, `0.05 -> 6.05`, `0.01 -> 6.01`.
- **NARRATION:**
> Watch the line. As B slides toward A, the secant swings around, and the slope readout falls — seven, six point five, six point two five, six point one. B is getting closer and closer to A, and the number is closing in on something. Six. And when B finally arrives at A, the line stops cutting through the curve and just touches it. That line is called the tangent, and its slope is six. Six is the slope of this curve at exactly the point x equals three. Which means we have answered the question that seemed badly formed a minute ago.

---

### S1.07 — Why It Was Always Going to Be 6 + h (6:00–7:30)
- **STAGE:** Worked algebra: `((3+h)² - 3²)/h = (9 + 6h + h² - 9)/h = 6 + h`. Animate `h → 0` shrinking `+ h` to leave `6`.
- **PANEL:** Table with third column `6 + h` matching measured slope.
- **NARRATION:**
> The animation showed it. Now here is why it had to happen, in four lines of ordinary algebra — nothing you have not done before. The slope of the secant is the change in y over the change in x. Expand the square on top. The nine and the minus nine cancel. Divide everything by h. And what is left is six plus h. Look at that. The slope of the secant is always exactly six plus whatever h is. So of course it approached six — it was six plus a number I was deliberately shrinking to nothing. There is no mystery left in this. There never was one.

---

### S1.08 — The Limit Definition, Written Down (7:30–8:40)
- **STAGE:** Build formula `\frac{dy}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}` piece by piece.
- **PANEL:** 2-column dictionary mapping symbols to spoken form and meaning.
- **NARRATION:**
> Everyone writes this down the same way, so let us decode it once and never be bothered by it again. The top, f of x plus h minus f of x — that is the rise. How much y changed. The bottom, h — that is the run. How much x changed. Wrapped around the whole thing, lim as h goes to zero — that means: shrink the gap to nothing, exactly like the animation. And on the left, d y by d x, which is just the name we give to the answer. Four symbols, and you have already watched every one of them happen on screen. That is the limit definition of the derivative, and it is the only definition there is.

---

### S1.09 — Code Proof: Watch It Converge (8:40–10:20)
- **STAGE:** Formula `\frac{dy}{dx}` dimmed at top.
- **PANEL:** `p1_01_secant_table.py` stdout:
  `h=1.0 -> 7.000000`, `h=0.1 -> 6.100000`, `h=0.01 -> 6.010000`, `h=0.0001 -> 6.000100`, `h=1e-06 -> 6.000001` `[VERIFIED]`.
- **NARRATION:**
> This is the book's own code, and it is nine lines long. A function f that squares its input. A function that computes rise over run for whatever h you hand it. And a loop that hands it smaller and smaller values of h. Run it. Seven. Six point one. Six point zero one. Six point zero zero zero one. Six point zero zero zero zero zero one. That is not a proof in the mathematician's sense, but it is proof in the sense this book cares about — you have now watched Python confirm a claim rather than being asked to take it on faith. Type this file yourself. Do not copy it.

---

### S1.10 — BREAK IT: h Too Small (10:20–11:50)
- **STAGE:** Red banner `BREAK IT DELIBERATELY` in `#EF4444`. Number line showing `h=1e-16` rounding to 3.0.
- **PANEL:** `p1_02_h_too_small.py` stdout: `h=1e-16 slope estimate=0.0` `[FROM-OUTPUT]`. Highlight `0.0` in red `#EF4444`.
- **NARRATION:**
> Now let us break it on purpose, because that is how this book works. The story so far says smaller h is better. So let us take that seriously and make h absurdly small. And it falls apart. The estimates stop improving, they start wobbling, and at the very bottom the answer collapses to exactly zero — a slope of zero for a curve that is visibly climbing. Here is why. Your computer stores numbers with finite precision. When h gets small enough, three plus h is not slightly more than three — it rounds to exactly three. So the top of the fraction becomes three squared minus three squared, which is zero, and zero divided by anything is zero. The maths was never wrong. The arithmetic ran out of room.

---

### S1.11 — The Practical Rule (11:50–12:35)
- **STAGE:** Boxed rule: `For numerical slopes, use h ≈ 1e-6.`
- **PANEL:** Forward reference card for `gradient_check.py` and Chapter 1 Day 5.
- **NARRATION:**
> So there is a sweet spot, and it is not a deep truth, it is an engineering fact worth memorising: for numerical slopes, use h around ten to the minus six. Too big and your secant is not close enough to the tangent. Too small and floating point noise drowns the signal. And this is not a detail you will use once — in part three of this video we build a tool out of exactly this, and Chapter One of the book checks a hand-derived gradient against exactly this method. You have just met a technique you will use for the rest of the course.

---

### S1.12 — EXERCISE 1 (PAUSE CARD) (12:35–13:00 + 15s Countdown)
- **STAGE:** Full-screen Exercise 1 card (`p1_ex1_numerical_slope.py`).
- **NARRATION:**
> Your turn. Pause the video here — actually pause it, do not just watch me do it next. Write the numerical slope function from scratch, without scrolling back. Then use it on x cubed at x equals two, where the true answer is twelve, and hunt for the h that gets closest. The starter file has the assertions already written, so you will know when you are right. Pause now.

---

### S1.13 — Exercise 1 Walkthrough (13:00–14:40)
- **STAGE:** Plot of `y = x³` with tangent at x=2, slope 12.
- **PANEL:** `solutions/p1_ex1_numerical_slope.py` code & output (`h=1e-6 -> 12.000000`).
- **NARRATION:**
> Line by line. Define f as x cubed. Define numerical slope as f of x plus h, minus f of x, all over h — the same shape as before, and notice it does not care which function you hand it; that is the whole point of writing it as a function. Then loop over candidate h values and print the error against twelve. And the winner sits somewhere around ten to the minus six, exactly as the rule predicted. If yours landed nearby, you are right. If yours landed at ten to the minus twelve, re-watch the breakage — you have found the noise, not the answer.

---

### S1.14 — Deriving the Power Rule (14:40–16:30)
- **STAGE:** Derive `x² -> 2x` and `x³ -> 3x²` from limit definition. Pattern box: `xⁿ -> n*xⁿ⁻¹`.
- **PANEL:** Exponent table: `n=2 -> 2x`, `n=3 -> 3x²`, `n=4 -> 4x³`, `n=1 -> 1`, `n=0 -> 0`.
- **NARRATION:**
> Doing that limit by hand every single time would be miserable, so let us do it once with a general x and see if a shortcut falls out. Same four lines. Expand. Cancel. Divide by h. And what is left is two x plus h — so as h vanishes, the derivative of x squared is two x. Now do the identical thing for x cubed, and you get three x squared. Two from the square, three from the cube. The exponent comes down to the front, and the exponent left behind drops by one. That is the power rule. Notice we did not memorise it — we watched it fall out of the same limit you have now seen three times.

---

### S1.15 — Constant Multiple and Sum Rules (16:30–17:40)
- **STAGE:** Boxed rules: `d/dx[a*f(x)] = a*f'(x)` and `d/dx[f(x)+g(x)] = f'(x) + g'(x)`. Compare `y = x²` vs `y = 3x²`.
- **PANEL:** Worked example: `y = 3x² + 5x -> 6x + 5`.
- **NARRATION:**
> Two more rules, and then we have everything Part One needs. First: if you multiply a function by three, its slope is multiplied by three. Look at the two tangents — same shape, one is exactly three times steeper. That is obvious once you see it, and it is called the constant multiple rule. Second: if a function is two things added together, its derivative is the two derivatives added together. You can go term by term. So three x squared plus five x becomes six x plus five. Power rule on each piece, constant multiple to scale it, sum rule to glue it back together. Three rules, and they compose.

---

### S1.16 — Code Proof: The Rule Agrees with Measurement (17:40–19:00)
- **STAGE:** Worked result `6x + 5 at x=2 -> 17`.
- **PANEL:** `p1_03_rule_check.py` output: `by rule: 17.0`, `by finite difference: 17.000000` `[VERIFIED]`.
- **NARRATION:**
> And now the habit this book wants you to build for life. We derived six x plus five using rules. At x equals two, that says seventeen. But we also have a completely independent way of getting that number — the numerical method from ten minutes ago, which knows nothing about rules and just measures the curve. Run both. Seventeen, and seventeen. Two roads, same destination. Whenever you derive a derivative by hand for the rest of your life, check it this way. It costs three lines and it catches almost every mistake you will make.

---

### S1.17 — EXERCISE 2 (PAUSE CARD) + Walkthrough (19:00–21:30)
- **STAGE (card):** Full-screen Exercise 2 card (`p1_ex2_five_polynomials.py`).
- **NARRATION (before countdown):**
> Five of them. Rules first, numerical check second, every single time. Number five is not a trick question, but do think about what it means before you answer it. Pause now.
- **NARRATION (walkthrough):**
> Twenty-eight x cubed. Five x to the fourth plus four x. Ten — a straight line has a constant slope, which is where we started this video. x minus four. And number five: zero. The function y equals six is a flat horizontal line. It never changes. Its slope is zero everywhere. If that one felt strange, sit with it — a derivative measures change, and a constant does not change.

---

### S1.18 — The Chain Rule: Peeling the Onion (21:30–23:30)
- **STAGE:** Nested box diagram: outer `square it`, inner `10 - (2m + 1)`. Formula: `d/dm f(g(m)) = f'(g(m)) * g'(m)`.
- **PANEL:** The Onion Method rules card.
- **NARRATION:**
> This is the most important rule in the video, so we are going to take it slowly. Look at this function. To compute it, you do two things in order: first you work out ten minus two m plus one, and then you square the result. It is a function inside a function. Like an onion. The chain rule says: differentiate the outer layer while leaving the inner layer completely alone, and then multiply by the derivative of the inner layer. Outside first, then multiply by the inside. Say it a few times, because you will use it more than every other rule combined — every activation function, every loss function, and all of backpropagation is this one rule, applied over and over.

---

### S1.19 — Chain Rule on the Squared Error (23:30–25:40)
- **STAGE:** Derivation of `E(m) = (y - mx - c)²`: outer `2(y - mx - c)`, inner `-x`, result `dE/dm = -2x(y - mx - c)`.
- **PANEL:** Side-by-side comparison: Chain Rule vs Expand First (Chapter 0 §0E.5a).
- **NARRATION:**
> Now let us aim it at the thing that actually matters. This is the squared error of one prediction — how wrong the model is, squared. We want to know how the error changes as we turn the slope dial m. Outer layer: something squared, so its derivative is two times that something, inside untouched. Inner layer: y minus m x minus c — as m changes, that changes by minus x. Multiply them. Minus two x, times y minus m x minus c. And here is the satisfying part: Chapter Zero of your book gets to this same result a completely different way, by expanding the square first and differentiating term by term. Two methods. One answer. That is what it looks like when mathematics is not lying to you.

---

### S1.20 — Code Proof: −28, Three Ways (25:40–27:10)
- **STAGE:** `dE/dm = -2x(y - mx - c)` for `x=2, y=10, c=1, m=1` -> `-28.0` `[VERIFIED]`.
- **PANEL:** `p1_04_chain_squared_error.py` output: `-28.0` three ways `[VERIFIED]`.
- **NARRATION:**
> Put numbers in. x is two, y is ten, c is one, m is one. Minus two, times two, times ten minus two minus one. Minus four times seven. Minus twenty-eight. Now the code confirms it three separate ways — by chain rule, by expanding the square first, and by measuring the slope numerically. All three say minus twenty-eight. And notice what minus twenty-eight actually tells you: the error is decreasing steeply as m increases, which means m is too small and should be turned up. That is not an abstract number. That is an instruction.

---

### S1.21 — BREAK IT: The Dropped Inner Derivative (27:10–28:50)
- **STAGE:** Red banner `BREAK IT DELIBERATELY` in `#EF4444`. Correct `-28` vs Broken `14`. Red warning: `NO ERROR WAS RAISED. NOTHING CRASHED.`
- **PANEL:** `p1_05_dropped_factor.py` output: `dropped inner derivative: 14.0`, `correct: -28.0` `[VERIFIED]`.
- **NARRATION:**
> Here is the single most common chain rule mistake, and I want you to see it now rather than at three in the morning six weeks from now. Forget to multiply by the inner derivative. That is all. And look at what happens: you get fourteen instead of minus twenty-eight. Wrong magnitude, and worse, wrong sign — a model using this would confidently turn the dial in exactly the wrong direction. Now look at the terminal. No error. No traceback. No red text. Python cannot tell that you did calculus incorrectly; it can only tell you that you did arithmetic. This is precisely why we check derivatives numerically, and it is why Part Three of this video builds a tool that does nothing but catch this.

---

### S1.22 — EXERCISE 3 (PAUSE CARD) + Walkthrough (28:50–31:40)
- **STAGE (card):** Full-screen Exercise 3 card (`p1_ex3_chain_proof.py`).
- **NARRATION (before countdown):**
> Four of these, and number three is the interesting one — do it by chain rule, then do it again by multiplying the bracket out first, and confirm the two agree. That is the same two-roads check we have now done twice, and doing it yourself is the point. Number four uses the formula we just derived, with different numbers. Pause now, and take your time.
- **NARRATION (walkthrough):**
> Twelve times three x plus one cubed — four from the power, times three from the inside. Minus six times five minus two x squared — and note the minus sign comes from the inside, which is exactly the factor people drop. Four x times x squared plus one, and both routes agree. And number four: minus two, times four, times twenty minus twelve minus two. Minus eight times six. Minus forty-eight. Steeper than before, because the error is larger.

---

### S1.23 — The Second Derivative: Curvature (31:40–34:00)
- **STAGE:** Plot `E(m) = 81 - 36m + 4m²`. Tangent slope transitions from `-36 -> 0 -> +36`. Second derivative `E''(m) = 8.0` is constant positive.
- **PANEL:** `p1_06_second_derivative.py` output: `E''(m) = 8.0` `[VERIFIED]`.
- **NARRATION:**
> One last idea for Part One, and it is short. If the derivative is the slope, then the derivative of the derivative is the rate at which the slope is changing. Watch the tangent rotate as the dot travels. At the left it points steeply down. It flattens. It passes through horizontal at the bottom. Then it tips upward. The slope is changing steadily the whole way, and the second derivative is the number that measures that. For this function the second derivative is eight — a constant, positive number. Positive means the curve bends upward everywhere. Which means it is a bowl.

---

### S1.24 — Why "Bowl" is the Whole Ballgame (34:00–36:00)
- **STAGE:** Split screen. Left: Convex bowl (`E'' > 0`) with ball settling at single minimum. Right: Non-convex curve with 3 dips.
- **PANEL:** Summary card: "Squared error for linear regression has E'' > 0 everywhere -> exactly one minimum -> this is why least squares has one answer."
- **NARRATION:**
> And here is why anyone cares. Roll a ball into the bowl on the left. It does not matter where you release it — it always ends up at the same single lowest point. Now the shape on the right, which bends up in some places and down in others. Release three balls, and they settle in three different places. Where you started decided where you finished. That is the nightmare case in optimisation. Squared error for linear regression is always the bowl. Its second derivative is positive everywhere, so there is exactly one bottom, and no matter where the training starts it finds the same answer. When Part Two talks about a positive semidefinite Hessian, that is all it means: it is a bowl in every direction at once.

---

### S1.25 — Part 1 Exit Check and Handoff (36:00–38:30)
- **STAGE:** 5 exit check self-assessment questions appearing one at a time.
- **PANEL:** Handoff card for Part 2 (Many Dials At Once: Partial derivatives, Gradients, Hessians, Normal Equations).
- **NARRATION:**
> That is Part One. Before you move on, five honest questions — and honest is the operative word, because nobody is marking this but you. Can you explain what a derivative is in one sentence. Can you compute one numerically, and say why h around ten to the minus six rather than ten to the minus sixteen. Can you apply the four rules and check each numerically. Can you derive d E by d m for the squared error, from memory, without notes. And can you say what a positive second derivative means and why regression cares. If any of those is a no, the timestamps are in the description — go back to that one. Nothing in Part Two is hard, but all of it assumes these. Part Two is where the second dial arrives.
