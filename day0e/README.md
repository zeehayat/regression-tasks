# Project: "Day 0E Masterclass" — Calculus for Machine Learning

**KP Regression Book, Chapter 0, Day 0E ("Slope, Derivatives, and Surviving Errors")**

---

## 📁 Directory Layout

```
day0e/
├── video/
│   ├── day0e_calculus_masterclass.vtt      # Full captions, sentence-aligned
│   ├── chapters.json                        # 24 segment breakdown with timestamps
│   └── script.md                            # Comprehensive spoken narration script
├── deck/
│   ├── index.html                           # Reveal.js offline slide deck with MathJax
│   └── day0e_deck.pdf                       # PDF export layout
├── code/
│   ├── 01_limit_definition.py              # Segment 1 & 2: limit definition & h-ladder
│   ├── 03_power_sum_rules.py               # Segment 3: power/sum rules
│   ├── 09_partial_derivatives.py           # Segment 9: MHP cost partial derivatives
│   ├── 14_ols_normal_equations.py          # Segment 14: OLS normal equations derivation
│   ├── 16_gradient_descent_handbuilt.py    # Segment 16: GD loop & lr=5.0 breakage
│   ├── 21_autodiff_engine.py               # Segment 21: Autodiff engine from scratch
│   ├── gradient_check.py                   # Segment 19: Reusable gradient check tool
│   └── test_all.py                         # Master test suite runner
└── README.md
```

---

## 🚀 How to Run the Code & Tests

To execute all code demos and verify numeric assertions:

```bash
cd /var/www/documentation/regression-tasks/day0e/code
python3 test_all.py
```

### Verified Numeric Assertions:
1. `01_limit_definition.py`: Slope estimate of $f(x)=x^2$ at $x=3.0 \implies \mathbf{6.000000}$
2. `03_power_sum_rules.py`: `dy_dx_by_rule(2.0)` for $y = 3x^2 + 5x \implies \mathbf{17.0}$
3. `09_partial_derivatives.py`: `symbolic_gradient_m(2, 10, 1, 1)` $\implies \mathbf{-28.0}$
4. `14_ols_normal_equations.py`: Closed-form solution $\hat{\beta} = (X^\top X)^{-1}X^\top y$ minimizes gradient norm to $\mathbf{< 10^{-10}}$
5. `16_gradient_descent_handbuilt.py`: Stable $lr=0.01$ decreases loss $564.0 \to 0.2028$; exploding $lr=5.0$ diverges to $\mathbf{3.38 \times 10^{47}}$
6. `21_autodiff_engine.py`: Reverse-mode autodiff matches analytical gradient $dL/dm = \mathbf{-28.0}$

---

## 📖 Recommended Viewing Schedule (2-Sitting Split)

Due to the exhaustive depth of this 24-segment masterclass (~117 minutes total):

- **Sitting 1 (Parts I–III, Segments 1–12)**: Single-variable derivatives, power rule, activation functions, convexity, partial derivatives, and gradient vectors.
- **Sitting 2 (Parts IV–VIII, Segments 13–24)**: Matrix calculus, OLS normal equations derivation, handbuilt gradient descent, autodiff engine, and traceback debugging.

---

## 🔗 Integration with Web Application

The `openDay0EVideoModal()` handler under the `# Day 0E — Slope, Derivatives, and Surviving Errors` section header launches the interactive video slide presentation engine with female voice narration, slide controls (`Play`, `Pause`, `Stop`), and pop-out/maximize capability.
