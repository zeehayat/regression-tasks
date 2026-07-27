# SRC/ Beginner-Readiness Review — Findings and Fix Instructions

**Audience for the book:** absolute beginners in Python, ML, math, and statistics — readers who have never coded, never seen calculus, linear algebra, or statistics before Chapter 0.

**Audience for this document:** an implementing agent. Each chapter section below lists concrete, prioritized fixes grounded in specific line numbers in `SRC/*.md`. Work chapter by chapter, top to bottom within each chapter's list (item 1 is highest priority). Re-read the surrounding context before editing — line numbers may have shifted since this review if other edits landed first; search for the quoted text/section heading to relocate.

**Ground rules for the agent doing this work:**
- Preserve the book's existing "no black boxes / code as proof" voice and structure (Orient → Construct → Prove → See → Build → Break → Reflect). New scaffolding should match this pattern: a worked numeric example before the symbolic generalization, then a code `assert` that proves they agree.
- Do not weaken existing correct material to make room — add short bridging sections, glosses, or one-paragraph asides. Most fixes below are additive (a few sentences or a small worked example), not rewrites.
- After editing any `SRC/*.md` file, re-run the build (`python compile_src.py` and/or `python compile_book.py`) and spot-check the rendered HTML output before considering a fix done — several findings below are rendering bugs, not just prose gaps.
- Where a fix references "the companion guide," check both `SRC/chapter*.md` and the matching `SRC/Chapter_N_Companion_Guide.md` — some gaps are already patched in the companion but not in the main chapter text (or vice versa); decide per-case whether the fix belongs in the main chapter (if it blocks reading the core lesson) or the companion (if it's supplementary depth).

---

## Chapter 0 — `SRC/chapter0.md` (+ `SRC/level_0_orientation.md`)

1. **Add a symbolic-calculus bridge.** Chapter 0's calculus content (§0E) is entirely numeric finite-difference partials of a two-variable toy function; it never manipulates a symbolic algebraic expression by hand. Add a short section (could sit at the end of §0E, before the capstone) that takes a symbolic derivative of something slightly more composite than $x^2$ — e.g. expand and differentiate $(y - mx - c)^2$ with respect to $m$, term by term, algebraically — so that "manipulating a symbolic expression, not just plugging into a rule" isn't itself a new skill when Chapter 1 Day 5 needs it.
2. **Reconcile tone with `level_0_orientation.md`.** The orientation chapter is breezy/jokey ("crying over dependency conflicts at 2 AM," "beaten to death" Boston housing aside); `chapter0.md` is controlled and contract-driven ("Central promise..."). Pick one register for a reader's first ten minutes with the book and adjust the other file to match. Recommend softening `chapter0.md`'s opening slightly rather than stripping the orientation chapter's personality, since the humor helps an anxious beginner relax.
3. **De-duplicate the embedded companion content.** `chapter0.md` hand-embeds a full second copy of its own explanations starting around the "Companion Details Area" heading near the end of the file (self-authored `## Detail 0A 1`, `## Detail 0A 2`, etc. sections feeding the click-to-expand modal), largely re-deriving material already taught earlier in the same file. Every other chapter (1–6) gets this modal content generated automatically from a separate `Chapter_N_Companion_Guide.md` via `compile_src.py`'s `EMBEDDED_COMPANION_CHAPTERS` mechanism. Either:
   - (preferred) wire Chapter 0 into that same pipeline, sourcing modal content from `SRC/Chapter_0_Companion_Guide_v2.md` instead of the hand-duplicated block, and delete the duplicated prose from `chapter0.md`; or
   - if that's not feasible without a compile_src.py change, at minimum add a comment at the top of the duplicated block noting it must be kept in sync manually with the chapter's primary teaching text above it.
4. **Delete or archive orphaned files**, confirmed unused by both `compile_src.py` and `compile_book.py`:
   - `SRC/companion1.md`
   - `SRC/chapter0-companion-guide.md`
   - `SRC/companion_guide_ch0.html`
   Move them into `SRC/redundant/` (which already holds superseded drafts of chapter1.md and chapter6.md) rather than deleting outright, in case content needs to be salvaged later.

---

## Chapter 1 — `SRC/chapter1.md` (+ `SRC/Chapter_1_Companion_Guide.md`)

1. **(Highest priority in the whole book) Add a worked numeric expansion before §5.4** (the section deriving $\nabla_\beta(\beta^TA\beta)=(A+A^T)\beta$ via the double sum $\beta^TA\beta=\sum_j\sum_k\beta_jA_{jk}\beta_k$, around line 1647). Insert, before the symbolic derivation:
   - A concrete 2×2 matrix $A$ and 2-entry vector $\beta$ with actual numbers.
   - Write out all four terms of the double sum by hand.
   - Take $\partial/\partial\beta_1$ and $\partial/\partial\beta_2$ by direct substitution into the expanded numeric polynomial (no index notation yet).
   - Show in code that the result equals $(A+A^T)\beta$ via `assert np.allclose(...)`.
   - *Then* present the general symbolic argument currently at that location, now as a generalization of something the reader just did by hand rather than the first thing they see.
2. **Add a Matplotlib primer.** Matplotlib functions (`scatter`, `plot`, `axhline`, `contour`/`clabel`, `quiver`, `plot_surface`, `meshgrid`, `Axes3D`) are used roughly 49 times starting around line 259, with zero API introduction anywhere — Chapter 0 only `pip install`s matplotlib and never calls it. Add a short primer (`plt.figure`, `.scatter`, `.plot`, `.legend`, `plt.show()` on one trivial example) at the start of Day 3, before the first figure. Separately flag the 3D plotting code in §4.8 (~lines 1293–1333) as optional/advanced, since it's the densest matplotlib in the chapter and not essential to the core lesson.
3. **Gloss `@dataclass(frozen=True)` and `Literal[...]`.** These appear cold in Day 1's first code block (~lines 317–340); Chapter 0 only ever taught a plain `class` with `__init__`/`self`. Add a two-sentence aside: what `frozen=True` buys (immutability, auto-generated `__init__`/`__repr__`) and what `Literal["prediction","explanation","causation"]` communicates to a reader.
4. **Minor:** add a one-line footnote explaining `array[..., None]` broadcasting where it first appears (~line 1321), used to expand axes in the 3D surface-patch code.

---

## Chapter 2 — `SRC/chapter2.md` (+ `SRC/Chapter_2_Companion_Guide.md`)

Scope confirmed: Day 6 scaling/conditioning, Day 7 QR/SVD/pseudoinverse, Day 8 probability/MLE/inference, Day 9 gradient descent, Day 10 generalisation/baselines/trees, Day 11 splitting/leakage/CV, Day 12 metrics/diagnostics, plus a capstone and research-paper discussions.

1. **Scaffold class inheritance before it's used.** `class GaussianOLS(StableOLS):` with `super().fit(X, y)` appears around lines 1139–1141 with no prior introduction — Chapter 0 built only one standalone class. Add a short "inheritance in one paragraph" note before this section (what `super()` does, why a subclass reuses a parent's method), or restructure to avoid inheritance and duplicate the small number of affected methods instead.
2. **Gloss `np.einsum`** at its first appearance (~lines 1176–1178, inside `GaussianOLS.intervals_for`): a one-line aside — "this computes $x_0^T(X^TX)^{-1}x_0$ row-by-row without an explicit Python loop" — resolves it cheaply. (See also the cross-cutting note at the bottom of this document — this exact gap recurs in Chapter 3.)
3. **Gloss `scipy.stats.t.ppf`** at first use (~lines 1136, 1166, 1181): one sentence — "`ppf` (percent-point function) returns the value below which a given probability mass falls; it's the inverse of the CDF."
4. **Pull integral/expectation notation into the main chapter.** `P(a≤Y≤b)=∫f(y)dy` (~lines 900–904) and $\mathbb{E}_{(X,Y)\sim P_{\text{deployment}}}[\cdot]$ (~lines 1640–1647) are only actually scaffolded in the companion guide's §0.8. Pull a condensed version of that scaffolding directly into `chapter2.md` as a short "Day 8 prerequisites" box, since a reader working only from the main chapter currently hits this notation unexplained.
5. **Cross-reference `OneHotEncoder(drop="first")`** (~lines 2159, 3064) back to the rank-deficiency/multicollinearity material from Days 6–7: one sentence — "dropping one column avoids the exact linear dependence described in §7.8" — turns an unmotivated library default into a reinforcement of an earlier lesson.
6. **Break up Day 11's nested pipeline example.** The block combining `Pipeline` + `ColumnTransformer` + `GroupKFold` + `GridSearchCV` + `clone` (~lines 2317–2384) is the single densest code jump in the chapter. Add a smaller intermediate example first — just `GridSearchCV` with one plain estimator, no pipeline — before combining all five abstractions at once.

---

## Chapter 3 — `SRC/chapter3.md` (+ `SRC/Chapter_3_Companion_Guide.md`)

1. **Add a symbolic-derivative bridge before Day 14** (~line 649, where `d/dx`, `∂/∂x`, `∇` notation starts being used freely for regularization derivations). This compounds the Chapter 1 Day 5 gap rather than closing it — fixing the root cause once (see cross-cutting notes) is preferred to patching this locally, but at minimum add a forward/backward cross-reference here.
2. **Fix broken LaTeX in the Companion Details Area**, roughly lines 4013–4153 of `chapter3.md`. `\lVert`, `\rVert`, `X^\top` are rendering as literal text (`Vert`, stray tab characters) instead of proper math — this breaks the "Read Details" modal's math for exactly the content it's meant to expand on. Check for missing backslash-escaping or a markdown-to-HTML conversion step eating backslashes.
3. **Gloss `np.einsum`** on first use (~line 1937, 2147) — same fix as Chapter 2 item 2; write the explanation once and reuse the phrasing across chapters for consistency.
4. **Implement Rubin's multiple-imputation pooling rules in code**, or explicitly relabel §13.6 (~lines 331–378) as conceptual/optional. Currently formulas are shown but never implemented — the only worked imputation code in that section is single imputation — which breaks the chapter's own "derived, not just named" standard (stated at line 34).
5. **Add one sentence defining "posterior"** before the Bayesian MAP aside in §15.11 (~lines 1276–1303) and §16.9 (~lines 1647–1657) — currently dropped in with a prior $N(0,\tau^2 I)$ and posterior mode with zero groundwork on what a posterior is.
6. **Derive the coordinate-descent update formula** (~lines 1507–1513) from the lasso subgradient conditions rather than stating it — small gap in an otherwise fully-derived chapter.
7. **Retitle the "research paper" sections** (13.11, 15.13, 16.12, 18.12, 19.13–14, 20.13) as reading-guide/discussion-prompt sections rather than implying close primary-source engagement. The chapter's "research-grade" framing (line 1) is mostly earned by Day 20's pre-registered benchmark (lines 2733–3389); these sections currently oversell relative to what they actually ask the reader to do.

---

## Chapter 4 + Interlude — `SRC/chapter4.md`, `SRC/chapter4_5_interlude.md` (+ `SRC/Chapter_4_Companion_Guide.md`)

1. **Add a derivation-lite walkthrough of the SVM dual/kernel trick before Day 25** (~lines 1550–1602). Currently "the dual solution depends on inner products… replace with a kernel" is asserted without derivation — the only algorithm in the chapter treated this way. Even an informal walkthrough (why the dual only needs inner products, what substituting a kernel function buys you) would match the chapter's own standard set elsewhere (e.g. the Gini-split and Shapley worked examples).
2. **Add a numeric 3-class worked example for softmax/cross-entropy** (~lines 1615–1632), matching the treatment already given to Gini splits (~1756–1785) and the two-feature Shapley game (~2533–2567).
3. **Explain what `SplineTransformer` actually builds** at its use around line 1467/1485–1489 — knot placement, resulting basis-matrix shape — rather than only presenting the abstract $B_m(x)$ formula from §25.3. Lower priority since splines were reportedly introduced in Chapter 3 already.
4. **Add a short "reading this pipeline" warm-up before the Day 29 capstone** (~lines 2836–3240), which jumps sharply in code density (custom `BaseEstimator`/`TransformerMixin` subclassing, `clone()`, nested `ColumnTransformer`s) relative to Days 21–24's pace.
5. **No changes needed to `chapter4_5_interlude.md`** — confirmed as a well-targeted prerequisite bridge (continuous random variables, limits, integrals), explicitly naming its own prerequisites and reusing Day 0E's numerical-derivative code and Day 0C's summation idiom. Use this file as the structural template when building the Ch0→Ch1 and Ch1→Ch3 calculus bridges above.

---

## Chapter 5 — `SRC/chapter5.md` (+ `SRC/Chapter_5_Companion_Guide.md`)

1. **Address chapter scope before line-level fixes.** Nine days currently cover KM, Nelson–Aalen, Cox partial likelihood, Schoenfeld residuals, AFT, RMST, Aalen–Johansen, Fine–Gray, multi-state models, IPCW Brier score, and target-trial emulation — closer to a full graduate survival-analysis-plus-causal course than an "apprentice" module. Decide (with the user) between: (a) splitting this into two chapters, or (b) explicitly downgrading some days to "conceptual survey, not build" status (see the labeling convention proposed in the cross-cutting notes). Do the smaller fixes below regardless of which path is chosen.
2. **Derive or name the log-rank test's variance formula** (~line 665) — currently invoked as "hypergeometric variance" with no derivation or explicit connection to the hypergeometric distribution.
3. **Gloss `scipy.optimize.minimize(method="L-BFGS-B")`** (~lines 876–882), used as the workhorse optimizer for every Cox fit in the chapter. At minimum, one paragraph on what quasi-Newton curvature approximation is doing — this is a regression from the "derive gradient descent from scratch" standard set in Chapter 2.
4. **Comment the late-binding closure idiom** in `make_candidates()` (~lines 2302–2311): `lambda penalty=l2: ...` is a classic Python gotcha (default-argument capture to avoid late binding) used with zero comment — silently smuggles in an advanced idiom.
5. **Derive cluster-robust "sandwich" covariance** (§32.10, ~line 984) rather than naming it only.
6. Gloss dense NumPy/pandas idioms used without walkthrough: `np.searchsorted(..., side=...)`, `pd.qcut`, `dict(zip(...))`, `np.select`, `np.union1d` (concentrated around `cif_from_cause_specific_models`, ~lines 2201–2245).
7. **Check whether the companion guide actually decompresses this chapter**, or just restates it more briefly (flagged at its §32.4–32.9) — if the latter, it isn't serving its scaffolding purpose here and should be expanded with slower, more worked-example-driven content rather than condensed restatement.

---

## Chapter 6 — `SRC/chapter6.md` (+ `SRC/Chapter_6_Companion_Guide.md`)

1. **Audit and fix the prerequisite checkpoint** (~lines 17–31). It currently requires familiarity with "a fitted propensity-like probability" before Day 39, but propensity scores aren't taught until §42.3 — internally inconsistent, reads as copy-pasted rather than derived from what Chapters 0–5 actually cover. Rewrite it to match the chapter's real prerequisites.
2. **(Highest priority in this chapter) Add executable code to Days 44–46, or explicitly relabel them.** Days 39–43 (estimands through AIPW) are genuinely no-black-box — g-formula derived line-by-line (~lines 335–346), AIPW coded from scratch with cross-fitting (~994–1051). Days 44–46 break this pattern entirely: the parametric g-formula (~1222–1235), clone-censor-weight (~1260–1264), marginal structural models (~1237–1258), DiD/RD/IV/synthetic control (Day 45), and meta-learners/causal forests (~1527–1553) are prose/bullet-list only, with zero executable code — yet the exercises immediately after (§6.12–6.16) instruct the reader to "simulate," "fit," and "compare" exactly these designs with no worked template anywhere in the chapter. Either add minimal worked-code templates for each design, or relabel these days as an explicit "conceptual survey" (see cross-cutting notes) and adjust the exercises to match what the chapter actually equips the reader to do.
3. **Add a setup/installation step for causal forest tooling**, or drop the hands-on framing. Exercise 46.12 asks the reader to "fit... a causal forest," but EconML/grf are never installed or introduced anywhere in the chapter — only name-dropped in the final references (~lines 2393–2396).
4. **Expand TMLE and Double ML beyond one paragraph each** (§43.6, ~lines 1055–1061) — currently a single black-box paragraph per method, which directly contradicts the chapter's own line "these are frameworks, not magic brands" that immediately follows.
5. **Fix broken MathJax** at line 967: `\hat\phi_i=hat m_1(X_i)-...` is missing a backslash before the second `hat`, breaking rendering of the chapter's central derived influence-function formula.
6. **Reconsider the chapter's framing relative to its actual content.** "Research practitioner" is earned by Days 39–43 and 47–48; Days 44–46 read as a survey, not hands-on competence. Consider labeling accordingly rather than implying uniform mastery across all ten days.

---

## Cross-cutting fixes (do these once, they resolve findings in multiple chapters above)

1. **`np.einsum` is used unglossed in at least Chapters 2 and 3.** Write one clear paragraph explaining einsum's index-string syntax (once, wherever it's most natural — recommend Chapter 2, its first real appearance), and cross-reference it briefly at each later occurrence instead of re-explaining or leaving unglossed.
2. **The symbolic-calculus cliff appears at Chapter 1 Day 5, again at Chapter 3 Day 14, and is implicitly assumed from Chapter 5 onward.** Rather than patching each occurrence separately, build one well-constructed bridge section — modeled directly on `chapter4_5_interlude.md`, which reviewers independently rated as the best-executed prerequisite bridge in the book — and insert it either at the end of Chapter 0 or as a short interlude between Chapters 1 and 2. This is the single highest-leverage fix in this document: it's the root cause behind Chapter 1 item 1, Chapter 3 item 1, and part of what makes Chapters 5–6 feel steep.
3. **Scope inflation from Chapter 3 onward.** Every chapter reviewer independently flagged that chapter's title/level claim as somewhat oversold relative to what a true beginner-turned-through-N-prior-chapters reader could do unsupervised afterward. Recommend one editorial pass, book-wide, that explicitly labels each day/section as either "build it yourself" (working code, derived from scratch, matches the book's core promise) or "read and recognize" (conceptual survey, names tools and ideas without full hands-on construction) — and makes sure exercises never ask for "build it yourself" competence on a "read and recognize" day (this exact mismatch is the Chapter 6 Days 44–46 problem, and a milder version of the Chapter 5 scope problem).
