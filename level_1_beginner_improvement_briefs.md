# Improvement Briefs: level_1_data_explorer.md (Beginner Experience)

> **Status: ✅ ALL BRIEFS EXECUTED.** [level_1_data_explorer.md](level_1_data_explorer.md) now follows this file's 11-day structure end to end. Three figures were generated via [generate_level1_figures.py](generate_level1_figures.py) into `figures/`. Every code block in the chapter (including the new exercise starter-code skeletons) was verified to at least parse, and every runnable snippet was executed against the real datasets in `data/` to confirm it works. See the per-brief status notes below for what was actually done and any deviations from the original plan. The one item under "Explicitly NOT decided here" remains undecided, as instructed.

**Context for whoever picks these up:** [level_1_data_explorer.md](level_1_data_explorer.md) is Chapter 1 of a self-paced regression curriculum (see [level_0_orientation.md](level_0_orientation.md) for the book's tone, rules, and existing per-chapter rhythm: read the theory → explain it back to a rubber duck → run the experiments → break it on purpose → solve the exercises). The chapter promises **zero prior expertise** in ML, statistics, or linear algebra, but currently delivers all 8 of its topics as one continuous ~800-line wall of material. That's the core beginner-experience problem this set of briefs solves.

**The organizing theme for this pass: "One topic per day, but excel into it."** Instead of one monolithic chapter, level_1_data_explorer.md should read as a sequence of **Day** units. Each day covers exactly one topic, goes deep enough that the reader genuinely masters it (not a shallow pass), and ends with its own recap and exercise before the reader moves on. This is a direct extension of level_0's existing per-chapter rhythm applied at finer granularity — don't invent a new pedagogical system, apply the one the book already has, one notch deeper.

**Ground rules for every brief:**
- Do this work **inside level_1_data_explorer.md's structure** (headers, ordering, exercises) — don't touch `regression_study_guide.html` or `HTML BOOK/`, those are compiled outputs (see [compile_book.py](compile_book.py)).
- Section numbers currently in the file (`Section 4`, `6.1`...`6.9`, `7`) are not referenced by any other file in the repo (verified via grep) — they're safe to restructure into `Day N` headers as long as internal cross-references within this file (e.g. "as covered in Section 6.7") are updated to match the new `Day N` labels.
- Keep the established tone: conversational, light humor, `> [!TIP]` **Comic Relief** and `> [!NOTE]` callouts — never at the expense of rigor. Every formula still gets derived, not asserted.
- Verify any numeric claim (SSR values, log-likelihoods, optimizer output) by actually running the code via Bash before committing the surrounding text.
- Do these briefs **in order** (Brief 0 first) — every per-day brief assumes Brief 0's template and day count already exist.

---

## Brief 0 (Master): Restructure the chapter into 11 Day units

**✅ DONE.** The chapter now opens with the Day 1–11 table, a "How to Pace Yourself" callout citing Chapter 0's five-step rhythm, the standalone exercises section was removed and redistributed per the mapping table below, a `## Chapter 1 Capstone` was added (touches Days 5, 6, 7, 9, and 11 in one integrative script), and `## Chapter 1 Recap`, `## Formula Cheat Sheet`, and `## References` remain at the very end. Every internal reference to the old `Section 6.X` numbering was rewritten to `Day N`.

**Day breakdown** (mapped from the chapter's current sections — do not renumber sections further until this mapping is implemented):

| Day | Title | Current source |
|---|---|---|
| 1 | What Regression Is — Prediction vs. Explanation vs. Causation | Section 4 |
| 2 | Regression Vocabulary | Section 5 |
| 3 | Linear Algebra Primer — Vectors, Matrices, and the Design Matrix | Section 6.1 |
| 4 | Measuring Error — SSR and the Geometric View of OLS | Sections 6.2 + 6.3 |
| 5 | Deriving OLS — The Algebra and the Full Hand Calculation | Sections 6.4 + 6.5 |
| 6 | Matrix Inversion & Multicollinearity — Why Models Break | Section 6.6 |
| 7 | Maximum Likelihood Estimation — OLS's Secret Origin Story | Section 6.7 |
| 8 | Loss Functions — Choosing What "Wrong" Means | Section 6.8 |
| 9 | Gradient Descent Fundamentals — Batch, Stochastic, Mini-batch | Section 6.9.1–6.9.3 |
| 10 | *(Bonus/Advanced)* Adaptive Optimizers — Momentum, AdaGrad, RMSprop, Adam | Section 6.9.4–6.9.6 |
| 11 | Exploratory Data Analysis — Getting to Know Your Data | Section 7 |

Splitting Gradient Descent into Day 9 (core) + Day 10 (explicitly labeled bonus/advanced) is deliberate: it resolves the earlier concern that Lipschitz bounds, Robbins-Monro conditions, and Adam's full derivation are graduate-level material inside a beginner chapter — instead of a defensive footnote, it becomes an honestly-labeled bonus day the reader can return to later.

**Per-day template — every day section should follow this exact shape**, using a `## Day N: Title` heading:

1. **Opening (2–4 sentences):** one line callback to what the previous day covered, one line on why today's topic matters, one line on what "excelling into it" will look like by the end of the day. Skip the callback on Day 1.
2. **The deep dive:** the existing rigorous content for that topic, unchanged in substance, but audited against the specific gap for that day (see per-day briefs below — most days have one).
3. **Worked example + code:** already present in most sections; keep it, verify it still runs.
4. **Day N Recap:** 3–5 bullet points, the "if you remember nothing else" version of the day.
5. **Quick Check:** 1–2 short questions answerable from that day's material alone, with the answer given immediately after (self-check, not a hidden-answer quiz).
6. **Day N Exercise:** exactly one exercise per day (redistribute the chapter's existing Exercises 1.1–1.6 into their matching days per the table below; write new light exercises for days that don't have one yet).

**Other chapter-level changes required by this restructuring:**
- Rewrite the chapter's opening "We will cover: 1...8..." list (currently lines 7–15) into the Day 1–11 table above, framed explicitly as a suggested one-day-per-topic pace — but say plainly it's a *suggested* pace, not a deadline (matches level_0's self-paced framing; don't contradict it).
- Add a short "How to Pace Yourself" callout right after the day list, explicitly pointing back to level_0 Section 1's five-step rhythm and saying each day below is that rhythm applied to one topic. Don't reinvent the rhythm — cite it.
- Remove the standalone `## Level 1 Exercises` section at the end (its 6 exercises move into their matching days per the table below) and replace it with a short `## Chapter 1 Capstone` — one integrative exercise that touches at least three different days' material (e.g., fit OLS on the real KP dataset by hand-derived formula, verify against MLE, then run an EDA pass to sanity-check the result), so there's still a synthesizing finale.
- Keep `## Chapter 1 Recap` and `## References` (and the Formula Cheat Sheet from Brief 9 below) at the very end, after the Capstone.

**Existing exercise → Day mapping:**

| Exercise | Moves to |
|---|---|
| 1.1 OLS by Hand | Day 5 |
| 1.2 Scratch Optimizers Lab | Split: batch/SGD portion → Day 9, Adam portion → Day 10 |
| 1.3 Eigenvalue and Condition Number Analysis | Day 6 (cross-reference forward to Day 10 where condition number reappears) |
| 1.4 Anscombe's Quartet Lab | Day 11 |
| 1.5 MLE vs. OLS Showdown | Day 7 |
| 1.6 EDA Deep Dive | Day 11 |

Days 1, 2, 3, 4, 8 currently have **no** matching exercise — each needs one written from scratch (small, single-sitting scope; see per-day briefs).

**Acceptance criteria for Brief 0:** the chapter's table of contents, headers, and exercise placement all reflect the 11-day structure; every internal "Section 6.X" cross-reference elsewhere in the file is updated to "Day N"; nothing from the current content is deleted, only relocated and (per the specific briefs below) deepened.

---

## Day 1: What Regression Is

**✅ DONE.** Added one additional worked scenario per subsection (1.1–1.3) drawn from the agricultural and infrastructure datasets, and wrote "Day 1 Exercise" with three cross-dataset scenarios to label plus a confounder-naming question. Also added a Day 1 Recap and Quick Check per the Brief 0 template.

**Source:** current Section 4 (Prediction / Explanation / Causation).
**Excel-into-it task:** this day is conceptual, not mathematical — depth here means more worked real-world scenarios, not more formulas. Add one additional worked scenario per subsection (4.1–4.3 equivalents) using the *other* two KP datasets (agricultural yields, infrastructure projects — see [level_0_orientation.md](level_0_orientation.md) Section 3), not just the development index, so the reader sees the prediction/explanation/causation distinction generalize.
**New exercise needed:** write "Exercise Day 1.1" — give the reader three short scenario descriptions (one prediction, one explanation, one causation question) drawn from the infrastructure or agricultural datasets and ask them to label which is which and justify why a naive regression would mislead them on the causal one.

## Day 2: Regression Vocabulary

**✅ DONE.** Checked Level 2–4 files via grep before writing forward pointers, as instructed. Confirmed: "Data Leakage" resurfaces concretely in Level 2 Section 9.1 ("The Danger of Test Set Leakage") and Level 3's Exercise 3.1 ("Preprocessing & Leakage Audit") — both cited. "Hyperparameters" resurfaces in Level 4 Section 22 ("Hyperparameter Tuning") and its gradient-boosting learning rate discussion — cited, plus tied to Day 9's learning rate. Added "Day 2 Exercise" (label feature/target/identifier on a fictional 6-column dataset, spot the leakage column).

**Source:** current Section 5.
**Excel-into-it task:** for each vocabulary term, add a one-line forward pointer to where it becomes load-bearing later (e.g., "Hyperparameters" → "you'll tune your first one, the learning rate, in Day 9"; "Data Leakage" → "you'll deliberately induce this in an exercise in Level 2"). Check Level 2–4 files for accurate forward references before writing them — don't invent a claim about where a term reappears without confirming it.
**New exercise needed:** write "Exercise Day 2.1" — give the reader a short (5–6 column) fictional dataset description and ask them to label each column as feature/target/identifier, and name one column that would cause data leakage if included.

## Day 3: Linear Algebra Primer

**✅ DONE.** Added the Notation & Symbol Glossary (covers $\sum$, $\prod$, $\in$/$\mathbb{R}^n$, subscript-vs-superscript, $\hat{}$, $\partial$, $\nabla$, $\|\cdot\|_2$, $\sim$/iid, $\arg\max$/$\arg\min$ — expanded slightly beyond the brief's minimum list since those symbols needed to be pre-glossed for later days too). Added the 5-line concrete preview using `sklearn.linear_model.LinearRegression` on the real dataset — ran it via Bash first (R² = 0.672, coefficients confirmed sane) before writing it in. Added "Day 3 Exercise" (3×2 by 2×1 matrix multiplication + transpose, verified by hand and in Python).

**Source:** current Section 6.1.
**Excel-into-it tasks (this is the densest day to get right):**
1. Add a **Notation & Symbol Glossary** at the start of this day, before "What is a Vector?" — define $\sum$, $\in$ (e.g. $y \in \mathbb{R}^n$), subscript-vs-superscript convention, and $\hat{}$ ("estimate of") with one-line plain-English translations and tiny concrete examples. (Later days will each need their own new symbols glossed the first time they appear — see Days 5, 7, 9 below.)
2. Add a **5-line "concrete preview"** before the glossary: load `data/kp_subdistrict_development_index.csv`, fit `sklearn.linear_model.LinearRegression` on `literacy_rate` and `health_facility_distance_km` predicting `development_score`, print coefficients and R². One sentence: "by the end of Day 5 you'll understand exactly what that `.fit()` call did internally, by hand." Verify this code actually runs against the real dataset before writing it into the chapter.
**Exercise:** none currently exists; write "Exercise Day 3.1" — a short manual matrix multiplication and transpose problem (3×2 by 2×1, different numbers than the chapter's own worked example) with a Python verification step.

## Day 4: Measuring Error — SSR and the Geometric View of OLS

**✅ DONE.** [generate_level1_figures.py](generate_level1_figures.py) generates `figures/day4_ols_residuals.png` (toy dataset, OLS line, dashed vertical residuals annotated with exact residual values) — kept the existing ASCII diagram alongside it rather than removing it, since it's still a useful abstract mental model. Added "Day 4 Exercise" (SSR by hand for a new candidate line $\hat y=0.5+1.6x$, verified in Python: SSR = 0.59).

**Source:** current Sections 6.2 + 6.3.
**Excel-into-it task:** generate and embed a real plot (not ASCII art) of the toy 3-point dataset with the OLS line and dashed vertical residual segments, replacing or supplementing the existing ASCII projection diagram. Write a small script (e.g. `generate_level1_figures.py`) that produces this as a PNG saved under an `assets/` or `figures/` directory, referenced via standard markdown image syntax.
**New exercise needed:** write "Exercise Day 4.1" — ask the reader to compute SSR by hand for a different toy candidate line than the one already worked in the chapter, then verify in Python (mirrors the chapter's existing worked pattern so it's gradeable against a known-correct answer).

## Day 5: Deriving OLS — The Algebra and the Full Hand Calculation

**✅ DONE.** Added the derivative primer (1-D slope intuition, $f(x)=x^2$ example verified numerically via Bash: $f'(3)=6$, matches $(f(3.001)-f(3))/0.001 = 6.001$), then bridged to the vector calculus rules. Exercise 1.1 relocated here verbatim as "Day 5 Exercise."

**Source:** current Sections 6.4 + 6.5.
**Excel-into-it task:** before the "Vector Calculus Rules to Know" box, add a ~150–250 word derivative primer: explain a derivative in one dimension first ("the slope of $f$ at a point — how fast $f$ changes as you nudge $x$"), with a trivial numeric example ($f(x) = x^2 \Rightarrow f'(x) = 2x$, verified by evaluating $f$ at $x=3$ and $x=3.001$), explain that setting a derivative to zero finds a flat spot (bottom of a bowl), then bridge to the vector-calculus rules as the same idea generalized to a vector of parameters. A reader with zero calculus background should be able to follow *why* setting $\frac{\partial S(\beta)}{\partial \beta}=0$ finds a minimum, not just accept it as a rule.
**Exercise:** relocate Exercise 1.1 (OLS by Hand) here, unchanged.

## Day 6: Matrix Inversion & Multicollinearity

**✅ DONE.** Added the forward cross-reference to Day 10's condition number discussion. Exercise 1.3 relocated here verbatim as "Day 6 Exercise," with a note to hang onto the answer for Day 10.

**Source:** current Section 6.6.
**Excel-into-it task:** none required beyond what's already there — this section is already appropriately deep for a beginner. Just add the forward cross-reference to Day 10 where condition number reappears in the gradient descent context, so the reader notices the callback later.
**Exercise:** relocate Exercise 1.3 (Eigenvalue and Condition Number Analysis) here, unchanged.

## Day 7: Maximum Likelihood Estimation

**✅ DONE.** [generate_level1_figures.py](generate_level1_figures.py) generates `figures/day7_mle_gaussian.png` (Gaussian centered at the predicted mean for toy point 2, observed value marked with its density height annotated). Exercise 1.5 relocated here verbatim as "Day 7 Exercise."

**Source:** current Section 6.7.
**Excel-into-it task:** generate and embed a real plot of a Gaussian PDF curve annotated with the MLE generative story (mark the mean, mark one residual as a deviation from it), as part of the same figure-generation script from Day 4's brief.
**Exercise:** relocate Exercise 1.5 (MLE vs. OLS Showdown) here, unchanged.

## Day 8: Loss Functions

**✅ DONE.** No structural gap to fix, so applied the standard day template plus wrote "Day 8 Exercise" (30-point synthetic dataset + 3 planted outliers, compare fitted slopes under MSE/MAE/Huber).

**Source:** current Section 6.8.
**Excel-into-it task:** this section is already well-scoped for a single day (four loss functions, each with formula + properties + code). No structural gap identified — just apply the standard day template (recap + quick check).
**New exercise needed:** write "Exercise Day 8.1" — generate a small synthetic dataset with 2–3 planted outliers, fit models minimizing MSE vs. MAE vs. Huber, and have the reader compare how much each model's slope shifts toward the outliers. This gives loss functions a hands-on payoff they currently lack.

## Day 9: Gradient Descent Fundamentals

**✅ DONE.** Added inline reminders pointing $\nabla$ and $\|\cdot\|_2$ back to the Day 3 glossary. Relocated the batch/SGD half of Exercise 1.2 here with `BatchGradientDescentRegressor` / `SGDRegressorFromScratch` starter skeletons (method signatures + docstrings + `# TODO`). **Deviation from plan, caught during verification:** the first draft of the skeletons used bare `# TODO` comments as method bodies, which is invalid Python (a comment alone isn't a statement) — confirmed via `compile()` on every code block in the chapter. Fixed by adding `pass  # TODO` to each stub; re-verified all 16 code blocks in the chapter now parse cleanly.

**Source:** current Section 6.9.1–6.9.3 (geometry of the gradient, deriving the OLS gradient, batch/SGD/mini-batch).
**Excel-into-it task:** no new gap beyond the general notation-as-introduced rule (gloss $\nabla$ and $\|\cdot\|_2$ the first time they appear here, per Day 3's glossary pattern — add a two-line inline reminder rather than a full second glossary).
**Exercise:** relocate the batch/SGD portion of Exercise 1.2 (Scratch Optimizers Lab) here — i.e., have the reader implement `BatchGradientDescentRegressor` and `SGDRegressorFromScratch` only. Add starter-code class skeletons (method signatures + docstrings + `# TODO` markers, no implementations) since the current exercise has zero scaffolding despite being a bigger leap than every guided example before it.

## Day 10 (Bonus/Advanced): Adaptive Optimizers

**✅ DONE.** Added the `> [!NOTE]` framing box. **Honest finding from checking Level 3–6 first (as instructed):** Adam/adaptive optimizers do **not** resurface by name anywhere later in the book (grepped all level_2 through level_6 files) — `scikit-learn`'s regressors handle their own optimization internally. The callout says this plainly rather than overclaiming relevance, while noting the underlying "badly-scaled features slow convergence" intuition does keep mattering. [generate_level1_figures.py](generate_level1_figures.py) generates `figures/day10_condition_number.png` (side-by-side contour plots, $\kappa=40$ zig-zagging path vs. $\kappa\approx1.1$ direct path — tuned the learning rate/step count until the zig-zag was visually obvious, not just a single big jump). Relocated the Adam half of Exercise 1.2 here with the same starter-skeleton pattern as Day 9 (same `pass  # TODO` fix applied).

**Source:** current Section 6.9.4–6.9.6 (Lipschitz bounds and convergence, Polyak-Ruppert averaging, Momentum/AdaGrad/RMSprop/Adam).
**Excel-into-it task:** add an opening `> [!NOTE]` explicitly framing this as the chapter's one deliberately advanced/optional day — honest about the fact that it's graduate-level material, explicit that it's fine to skim formulas on a first pass and return after finishing Level 2, but *not* claiming it's unimportant (check Level 3/4 files first for where Adam or adaptive optimizers actually get used hands-on, and say so here if it's genuinely load-bearing later). Also generate and embed the "narrow valley vs. symmetric bowl" contour plot pair for the condition-number discussion (unscaled vs. scaled features, gradient descent path sketched on each) as part of the same figure script.
**Exercise:** relocate the Adam portion of Exercise 1.2 here — have the reader implement `AdamRegressorFromScratch` and plot its loss trajectory against Day 9's batch/SGD results. Provide the same starter-code skeleton pattern as Day 9.

## Day 11: Exploratory Data Analysis

**✅ DONE.** Added Day 11 Recap and Quick Check (previously missing). Relocated Exercises 1.4 and 1.6 here as two exercises for the day, as planned. All EDA code (histplot, boxplot, scatterplot, heatmap, pairplot, countplot, groupby, Anscombe load) was executed against the real datasets to confirm it runs without error.

**Source:** current Section 7 (already substantially expanded in a prior pass — 7.1 through 7.5).
**Excel-into-it task:** none identified — this day is already appropriately deep. Just apply the standard day template (it currently ends with the Anscombe discussion and no explicit recap/quick-check box; add those).
**Exercise:** relocate Exercises 1.4 (Anscombe's Quartet Lab) and 1.6 (EDA Deep Dive) here as two exercises for this one day — that's acceptable since Day 11 is explicitly the synthesis/practice day for the whole EDA topic.

---

## Brief 9 (Cross-cutting, do last): Formula Cheat Sheet appendix

**✅ DONE.** Added `## Formula Cheat Sheet` after `## References`... actually placed it *before* `## References` in the final chapter (References reads better as the very last section) — one table, [Concept | Formula | Day] columns, covers all 16 boxed equations across the chapter.

**Problem:** even with a per-day Recap box (from Brief 0's template), there's no single compact lookup table across the whole chapter.
**Deliverable:** add a `## Formula Cheat Sheet` section after `## References`, one table with columns [Concept | Formula | Day]. Include: SSR, Normal Equations, $\hat\beta$ closed form, Gaussian log-likelihood, $\hat\sigma^2_{MLE}$, MSE/MAE/Huber/Quantile, the gradient of MSE, the three GD update rules (batch/SGD/mini-batch), and the Adam update rules.
**Acceptance criteria:** every boxed/numbered equation anywhere in the chapter has exactly one row here, and the Day column lets the reader jump back to the full derivation.

---

## Resolved: Day 10 relocated to a new standalone chapter

**✅ DONE.** The user decided Day 10 should move to a new chapter rather than stay as an in-place bonus day. Two integration options were presented — insert as a renumbered "Level 2" (cascading renumber of every subsequent chapter file) vs. a standalone "Level 1.5" that doesn't touch any other file — and the user chose the standalone option for lowest risk.

**What changed:**
- New file [level_1_5_adaptive_optimizers.md](level_1_5_adaptive_optimizers.md): the full former Day 10 content (Lipschitz convergence bounds, Polyak-Ruppert averaging, Momentum/AdaGrad/RMSprop/Adam), with its own recap, quick check, exercise (Adam-from-scratch skeleton), formula cheat sheet, and references (including the Robbins-Monro citation, moved out of Level 1's reference list since it's no longer discussed there — plus an added citation to the actual Adam paper, Kingma & Ba 2015, which the original chapter never cited despite covering Adam in depth).
- [level_1_data_explorer.md](level_1_data_explorer.md) now has **10 days**, not 11: the old Day 11 (EDA) shifted down to Day 10. Every cross-reference was updated: the Day 1–10 table, the pacing note, Day 6's two forward-pointers to the condition number (now pointing at Level 1.5 instead of "Day 10"), Day 9's exercise notes about the Adam half moving to Level 1.5, the Capstone, the Chapter Recap, and the Formula Cheat Sheet (the condition-number row now cites only Day 6; the Adam row was removed and replaced with a pointer to Level 1.5's own cheat sheet).
- The condition-number figure was renamed from `figures/day10_condition_number.png` to `figures/adaptive_optimizers_condition_number.png` (both in the generator script and its embed in the new chapter) since it no longer belongs to a "Day 10."
- Verified: [generate_level1_figures.py](generate_level1_figures.py) regenerates all three figures cleanly from scratch; every Python code block in both chapter files (16 total) parses without error; no leftover "Day 11," "eleven days," or stale figure-path references remain anywhere in level_1_data_explorer.md (checked via grep).
