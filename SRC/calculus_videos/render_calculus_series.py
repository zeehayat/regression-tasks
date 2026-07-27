"""Render a beginner calculus-for-ML micro-lesson series.

The videos are intentionally caption-led and offline: every slide carries the
explanation, example, and takeaway, while a matching markdown script can be
used for narration or recording.  Requires Pillow and ffmpeg.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
FRAMES = ROOT / "frames"
OUT = ROOT / "mp4"
W, H = 1280, 720
FPS = 30
BG = "#FAF7F0"
TEXT = "#252622"
FOREST = "#3F6652"
TERRA = "#A45F45"
PANEL = "#ECE8DF"
MUTED = "#686B63"

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


LESSONS = [
    {
        "slug": "01_slope_and_rate_of_change",
        "title": "1 · Slope: the first idea behind a derivative",
        "script": """# Video 1 — Slope and rate of change\n\nA derivative begins with a simple question: how quickly is one quantity changing when another quantity changes?\n\nFor two points, slope is change in output divided by change in input: Δy/Δx. If a model's loss falls by 6 when the learning rate changes by 2 units, the average rate is -3 loss units per learning-rate unit. The negative sign says the loss is going down.\n\nA line has one constant slope. Real functions curve, so we make the two points closer and closer. That leads to the derivative at one point.\n\nMachine-learning connection: before we optimize a model, we need a language for how the loss responds to a small parameter change.\n""",
        "slides": [
            ("The question", ["How much does the output change", "when the input changes?", "That is a rate of change."]),
            ("Average slope", ["slope = change in output / change in input", "m = Δy / Δx", "Example: loss changes −6 while x changes +2", "m = −6 / 2 = −3"]),
            ("Read the sign", ["positive slope → output rises", "negative slope → output falls", "zero slope → locally flat", "The sign already tells us a direction."]),
            ("From a line to a curve", ["A curve has a different slope at different places.", "Use two nearby points.", "Bring them closer until the average slope becomes the local slope."]),
            ("Machine-learning connection", ["x = a model parameter", "y = the loss", "the derivative asks: if x moves a little,", "which way does the loss move?"]),
            ("Pause and try", ["If y = 2x + 1, what is its slope?", "If x increases by 3, how much does y change?", "Answer: slope 2; y changes by 6."]),
        ],
    },
    {
        "slug": "02_derivative_as_local_sensitivity",
        "title": "2 · The derivative: local sensitivity",
        "script": """# Video 2 — The derivative as local sensitivity\n\nThe derivative is the slope of a function at one exact input. We write f'(x).\n\nThe formal definition uses a limit: f'(x) = lim as h approaches zero of [f(x+h)-f(x)]/h. You do not need to fear the limit. It means: measure the slope over a tiny interval, then imagine that interval shrinking toward zero.\n\nFor f(x)=x², the derivative is 2x. At x=3 the local slope is 6. A small positive move in x produces an approximately six-times-as-large positive change in f.\n\nThis is sensitivity: how responsive is the output to a small parameter perturbation right now?\n""",
        "slides": [
            ("Derivative = local slope", ["f′(x) tells the slope at one exact x.", "It is not the slope of the entire curve."]),
            ("The limit idea", ["average slope = [f(x+h) − f(x)] / h", "make h smaller and smaller", "the limiting value is f′(x)"]),
            ("Example: f(x) = x²", ["f′(x) = 2x", "At x = 3: f′(3) = 6", "A tiny increase in x increases f by about 6 times that tiny amount."]),
            ("Approximation", ["Δf ≈ f′(x) · Δx", "At x=3, Δx=0.01", "Δf ≈ 6 · 0.01 = 0.06", "The approximation improves as Δx gets smaller."]),
            ("Why ML cares", ["A parameter is a knob.", "The derivative measures loss sensitivity to that knob.", "Sensitivity gives optimization a direction."]),
            ("Check yourself", ["For f(x)=x² at x=−2:", "is the slope positive or negative?", "Answer: −4, so the graph falls as x increases."]),
        ],
    },
    {
        "slug": "03_derivative_rules",
        "title": "3 · Derivative rules you will use constantly",
        "script": """# Video 3 — Core derivative rules\n\nRules let us differentiate useful expressions without returning to the limit definition every time.\n\nA constant has derivative zero. The power rule says the derivative of x to the n is n times x to the n minus 1. A constant multiplier stays in front. For sums, differentiate each term.\n\nFor f(x)=3x²−4x+7, the derivative is 6x−4. Notice that the constant 7 disappears because it never changes.\n\nThese rules are the algebraic engine behind gradients in linear models and neural networks.\n""",
        "slides": [
            ("Constant rule", ["d/dx [c] = 0", "A fixed number never changes.", "d/dx [7] = 0"]),
            ("Power rule", ["d/dx [xⁿ] = n·xⁿ⁻¹", "Example: d/dx[x³] = 3x²", "Bring the exponent down; reduce it by one."]),
            ("Multiplier and sum rules", ["d/dx[c·f(x)] = c·f′(x)", "d/dx[f+g] = f′+g′", "Differentiate term by term."]),
            ("Worked example", ["f(x) = 3x² − 4x + 7", "f′(x) = 6x − 4 + 0", "f′(x) = 6x − 4"]),
            ("At a point", ["At x=2: f′(2)=12−4=8", "The curve rises locally.", "A small positive x move increases f by about 8 times that move."]),
            ("ML connection", ["Loss functions are sums of terms.", "Rules turn a large derivative into manageable pieces.", "That is how gradients stay computable."]),
        ],
    },
    {
        "slug": "04_chain_rule",
        "title": "4 · The chain rule: derivatives of nested functions",
        "script": """# Video 4 — The chain rule\n\nMachine-learning models are compositions: one function feeds another. The chain rule tells us how a small change travels through the chain.\n\nIf y=f(u) and u=g(x), then dy/dx = (dy/du)(du/dx). Read it as: sensitivity of the final output to the intermediate quantity, multiplied by sensitivity of the intermediate quantity to the original input.\n\nFor y=(3x+1)², let u=3x+1. Then dy/du=2u and du/dx=3, so dy/dx=6(3x+1).\n\nBackpropagation is the chain rule applied repeatedly from the output of a neural network back toward its parameters.\n""",
        "slides": [
            ("Nested functions", ["x → u=g(x) → y=f(u)", "The output is built in stages.", "Most ML models have many stages."]),
            ("Chain rule", ["dy/dx = (dy/du) · (du/dx)", "Multiply the local sensitivities along the path."]),
            ("Worked example", ["y = (3x + 1)²", "u = 3x + 1", "dy/du = 2u", "du/dx = 3"]),
            ("Finish it", ["dy/dx = 2u · 3", "= 6u", "= 6(3x+1)"]),
            ("Backpropagation", ["Output loss → layer output → layer input → weight", "At every link, multiply a local derivative.", "That repeated chain is backpropagation."]),
            ("Common mistake", ["Do not add sensitivities through a chain.", "Multiply them.", "Add only when independent paths are summed."]),
        ],
    },
    {
        "slug": "05_partial_derivatives_and_gradients",
        "title": "5 · Partial derivatives and gradients",
        "script": """# Video 5 — Partial derivatives and gradients\n\nA model has many parameters, so the loss depends on many inputs. A partial derivative changes one variable while holding the others fixed.\n\nFor L(w1,w2)=w1²+3w2², the partial derivatives are 2w1 and 6w2. Put them into a vector: the gradient ∇L = [2w1, 6w2].\n\nThe gradient points in the direction of steepest increase. Therefore the negative gradient points toward the steepest local decrease. Gradient descent uses that direction to reduce loss.\n\nThe gradient is not a single number; it is one sensitivity per parameter.\n""",
        "slides": [
            ("Many parameters", ["L depends on w₁, w₂, …, wₚ.", "We need one sensitivity for each parameter."]),
            ("Partial derivative", ["∂L/∂w₁ changes w₁", "while holding w₂, w₃, … fixed.", "It isolates one direction."]),
            ("Worked example", ["L(w₁,w₂)=w₁²+3w₂²", "∂L/∂w₁=2w₁", "∂L/∂w₂=6w₂"]),
            ("Build the gradient", ["∇L = [∂L/∂w₁, ∂L/∂w₂]", "∇L = [2w₁, 6w₂]", "one number per parameter"]),
            ("Direction", ["+gradient → steepest local increase", "−gradient → steepest local decrease", "The arrow is a direction in parameter space."]),
            ("ML connection", ["A neural network may have millions of weights.", "Backpropagation computes the whole gradient efficiently."]),
        ],
    },
    {
        "slug": "06_gradient_descent",
        "title": "6 · Gradient descent: learning by taking careful steps",
        "script": """# Video 6 — Gradient descent\n\nGradient descent updates parameters in the negative-gradient direction. The basic rule is w_new = w_old − η∇L(w), where η is the learning rate.\n\nIf the gradient is positive, subtracting it moves the parameter down. If the gradient is negative, subtracting it moves the parameter up. The learning rate controls the step size.\n\nToo small means slow learning. Too large can overshoot or diverge. The gradient is local, so descent is an iterative process: recompute the gradient after each move.\n\nFor L(w)=(w−3)² starting at w=0, the gradient is −6. With learning rate 0.1, the next value is 0−0.1(−6)=0.6.\n""",
        "slides": [
            ("The update", ["w_new = w_old − η∇L(w)", "η is the learning rate.", "Recompute the gradient after every step."]),
            ("Read the signs", ["gradient positive → move w downward", "gradient negative → move w upward", "Both moves aim to reduce loss locally."]),
            ("Worked example", ["L(w)=(w−3)²", "start w=0", "L′(w)=2(w−3)", "gradient at 0 = −6"]),
            ("Take one step", ["η = 0.1", "w_new = 0 − 0.1(−6)", "w_new = 0.6", "The parameter moved toward 3."]),
            ("Learning-rate trade-off", ["too small → slow", "too large → overshoot or diverge", "good training uses a schedule or careful tuning"]),
            ("The whole loop", ["1. predict", "2. calculate loss", "3. calculate gradient", "4. update parameters", "5. repeat"]),
        ],
    },
    {
        "slug": "07_second_derivatives_and_curvature",
        "title": "7 · Second derivatives: curvature and step size",
        "script": """# Video 7 — Second derivatives and curvature\n\nThe second derivative is the derivative of the derivative. It tells us how the slope itself is changing: this is curvature.\n\nFor f(x)=x², f′(x)=2x and f′′(x)=2, so the curve bends upward everywhere. A positive second derivative indicates local bowl-like curvature; a negative one indicates a cap-like shape.\n\nNear a minimum, curvature helps determine how aggressive a step can be. Newton-style methods use both gradient and curvature, while ordinary gradient descent uses only the gradient.\n\nIn multiple dimensions, the matrix of second partial derivatives is the Hessian. You only need the core idea now: first derivatives give direction; second derivatives describe the shape around you.\n""",
        "slides": [
            ("Differentiate again", ["f′(x) = slope", "f′′(x) = how the slope changes", "That changing slope is curvature."]),
            ("Example: x²", ["f(x)=x²", "f′(x)=2x", "f′′(x)=2", "constant positive curvature → bowl shape"]),
            ("Shape and optimization", ["flat curvature → broad, gentle bowl", "high curvature → narrow, steep bowl", "the same step may be safe in one direction and too large in another"]),
            ("Hessian", ["For many parameters, collect second partials", "into the Hessian matrix.", "It describes curvature in every parameter direction."]),
            ("Two kinds of information", ["gradient → which way is downhill?", "curvature → how does the ground bend?", "first order vs second order"]),
            ("Final check", ["At a smooth local minimum:", "gradient is approximately zero", "curvature is usually positive", "but always inspect the actual problem."]),
        ],
    },
]


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def wrapped(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=fnt) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def render_slide(path: Path, lesson: dict, slide_no: int, heading: str, lines: list[str], total: int) -> None:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, W, 18), fill=FOREST)
    draw.rectangle((0, H - 12, W, H), fill=TERRA)
    draw.text((70, 48), "DERIVATIVES FOR MACHINE LEARNING", font=font(BOLD_PATH, 22), fill=TERRA)
    draw.text((70, 92), lesson["title"], font=font(BOLD_PATH, 34), fill=FOREST)
    draw.line((70, 150, W - 70, 150), fill="#D8D1C5", width=2)
    draw.text((85, 188), heading, font=font(BOLD_PATH, 40), fill=TEXT)
    y = 270
    body_font = font(FONT_PATH, 31)
    for raw in lines:
        for line in wrapped(draw, raw, body_font, 1020):
            draw.text((105, y), line, font=body_font, fill=TEXT)
            y += 48
        y += 22
    draw.text((W - 180, 54), f"{slide_no}/{total}", font=font(BOLD_PATH, 22), fill=FOREST)
    draw.text((70, H - 70), "Pause • write the rule • try the example", font=font(FONT_PATH, 22), fill=MUTED)
    image.save(path)


def render_lesson(lesson: dict) -> None:
    frame_dir = FRAMES / lesson["slug"]
    frame_dir.mkdir(parents=True, exist_ok=True)
    slides = lesson["slides"]
    for i, (heading, lines) in enumerate(slides, 1):
        render_slide(frame_dir / f"slide_{i:02d}.png", lesson, i, heading, lines, len(slides))
    out = OUT / f"{lesson['slug']}.mp4"
    OUT.mkdir(parents=True, exist_ok=True)
    # Eight seconds per slide gives time to read and pause.  The video is
    # caption-led; the adjacent .md file is the narration script.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", "1/8",
        "-i", str(frame_dir / "slide_%02d.png"), "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-r", str(FPS), "-movflags", "+faststart", str(out),
    ]
    subprocess.run(cmd, check=True)
    (OUT / f"{lesson['slug']}.md").write_text(lesson["script"], encoding="utf-8")


def main() -> None:
    for lesson in LESSONS:
        render_lesson(lesson)
        print(f"rendered {lesson['slug']}")
    index = ["# Derivatives and Differential Calculus for Machine Learning", "", "Caption-led beginner micro-lessons. Each video is about 48 seconds and includes a matching narration script.", ""]
    for lesson in LESSONS:
        index.append(f"- [{lesson['title']}]({lesson['slug']}.mp4) · [narration script]({lesson['slug']}.md)")
    (OUT / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
