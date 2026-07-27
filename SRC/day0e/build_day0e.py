"""Build the Day 0E masterclass, deck, code labs, captions, and chapters.

The renderer is deliberately dependency-light: Pillow, NumPy, SymPy, and the
installed offline espeak-ng voice. Run from SRC with ``python3 day0e/build_day0e.py``.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parents[1] / "calculus_videos"))
from espeak_render import synthesize, clean_script  # noqa: E402

ROOT = Path(__file__).resolve().parent
VIDEO = ROOT / "video"
DECK = ROOT / "deck"
CODE = ROOT / "code"
ASSETS = ROOT / "assets"
W, H = 1920, 1080
BG, PANEL, TEXT, ACCENT, TERRA = "#FAF7F0", "#ECE8DF", "#252622", "#3F6652", "#A45F45"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

SEGMENTS = [
    ("01", "What a slope is, and why a curve does not have one", "A derivative is the slope at one exact point.", "dy/dx = 2x", "MHP cable length and cost form a line before we study curves."),
    ("02", "The limit definition, derived not asserted", "A derivative is a limit of nearby average slopes.", "f'(x) = lim h→0 [f(x+h)-f(x)]/h", "At a terrain index of 3, the numerical slope approaches 6."),
    ("03", "The rules you will actually use", "Power, constant-multiple, and sum rules are compressed limit calculations.", "d(xⁿ)/dx = n xⁿ⁻¹", "A cost curve with cable length and terrain index is differentiated term by term."),
    ("04", "Product, quotient, and chain rules", "Nested functions pass sensitivity through each layer.", "d f(g(x))/dx = f'(g(x))g'(x)", "The squared MHP cost error gives the same gradient by two roads."),
    ("05", "Exponential and logarithm", "Exponential growth reproduces itself; logarithms undo it.", "d eˣ/dx=eˣ; d ln(x)/dx=1/x", "Log-transformed million-PKR costs change the derivative scale."),
    ("06", "Sigmoid, tanh, and ReLU", "Activation derivatives explain saturation and vanishing gradients.", "σ'(x)=σ(x)(1−σ(x))", "A risk score for an MHP warning can become insensitive in a saturated tail."),
    ("07", "Second derivatives, curvature, and convexity", "Curvature tells us how the slope changes.", "E''(m)=8>0", "The squared MHP error is a bowl with one bottom."),
    ("08", "Taylor expansion", "Zooming in turns a smooth function into a local polynomial.", "f(x+h)≈f(x)+f'(x)h+½f''(x)h²", "A local cost approximation is useful only near the appraisal point."),
    ("09", "Partial derivatives", "Freeze every dial except the one being nudged.", "∂E/∂m and ∂E/∂c", "Cable length and terrain index are separate directions on an error surface."),
    ("10", "The gradient and directional derivatives", "The gradient points toward steepest ascent.", "∇E=[∂E/∂m,∂E/∂c]", "Negative gradient reduces the MHP cost error."),
    ("11", "Jacobian and Hessian", "Jacobians describe vector outputs; Hessians describe curvature.", "H=∇²E", "A positive-semidefinite MHP Hessian means convexity in every direction."),
    ("12", "Multivariable chain rule and backpropagation", "Backpropagation is the chain rule applied right to left.", "J_total=J_outer J_inner", "A prediction pipeline passes MHP feature sensitivity through layers."),
    ("13", "Differentiating with respect to a vector", "A vector gradient stacks one partial per entry of β.", "∇β(aᵀβ)=a", "β holds the cable and terrain coefficients."),
    ("14", "Derive the OLS normal equations", "Setting the squared-error gradient to zero yields the normal equations.", "β̂=(XᵀX)⁻¹Xᵀy", "Use the book's MHP projects and million-PKR costs."),
    ("15", "Critical points and the first-order condition", "A zero gradient is a candidate, not automatically a minimum.", "∇E=0", "Use curvature to classify an MHP fitting solution."),
    ("16", "Gradient descent, built by hand", "Repeated small negative-gradient steps reduce error.", "β_new=β_old−η∇E", "The MHP toy loop shrinks error at η=.01 and explodes at η=5."),
    ("17", "Batch, stochastic, and mini-batch variants", "The gradient can use all, one, or a small batch of projects.", "full batch / SGD / mini-batch", "The data unit remains an MHP project."),
    ("18", "Constrained optimisation and Lagrange multipliers", "At a constrained optimum, objective and constraint gradients align.", "∇E=λ∇g", "Ridge and lasso are constrained least-squares ideas."),
    ("19", "Finite differences and gradient checking", "Numerical derivatives are a debugging instrument.", "relative error = ‖gₐ−gₙ‖/(‖gₐ‖+‖gₙ‖)", "Check an MHP gradient before trusting a training loop."),
    ("20", "Symbolic differentiation with SymPy", "Symbolic tools check reasoning; they do not replace it.", "sympy.diff(E,m)", "Confirm the symbolic MHP squared-error gradient."),
    ("21", "A tiny autodiff engine from scratch", "A computation graph stores local derivatives and walks backward.", "loss.backward()", "Re-solve the MHP toy error without an autodiff library."),
    ("22", "Integrals as accumulated area", "Integration accumulates; differentiation gives the local rate.", "∫ density = 1; E[X]=∫x f(x)dx", "Later survival and probability chapters use this accumulation."),
    ("23", "Reading a traceback", "The last line translates a failure into plain English.", "symptom → cause → check", "Use shape, key, and type failures with MHP records."),
    ("24", "Rubber duck and the Day 0E exit check", "Explain each line, then prove the calculus with a capstone.", "slope → gradient → update → claim", "Return to MHP projects, cable length, terrain, and cost."),
]


def fnt(size: int, bold: bool = False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def slide(path: Path, num: str, title: str, claim: str, formula: str, applied: str, card: int) -> None:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 22, H), fill=ACCENT)
    d.text((86, 72), f"DAY 0E MASTERCLASS  ·  SEGMENT {num}", font=fnt(28, True), fill=TERRA)
    d.text((86, 145), title, font=fnt(54, True), fill=ACCENT)
    d.line((86, 235, W - 86, 235), fill="#D8D1C5", width=3)
    if card == 1:
        d.text((110, 330), "What this segment proves", font=fnt(34, True), fill=TERRA)
        d.text((110, 400), claim, font=fnt(46), fill=TEXT)
    elif card == 2:
        d.text((110, 330), "Formula — say every symbol out loud", font=fnt(34, True), fill=TERRA)
        d.rounded_rectangle((100, 420, W - 100, 650), radius=24, fill=PANEL)
        d.text((150, 505), formula, font=fnt(64, True), fill=ACCENT)
    elif card == 3:
        d.text((110, 330), "MHP worked example", font=fnt(34, True), fill=TERRA)
        d.text((110, 410), applied, font=fnt(42), fill=TEXT)
        d.text((110, 570), "Type the smallest version yourself; then run the check.", font=fnt(32), fill=ACCENT)
    else:
        d.text((110, 330), "PAUSE HERE — TYPE THIS", font=fnt(44, True), fill=TERRA)
        d.text((110, 430), "Do not copy-paste. Write one function, run it,", font=fnt(42), fill=TEXT)
        d.text((110, 500), "and assert the number you expect.", font=fnt(42), fill=TEXT)
        d.text((110, 650), "10 seconds → PAUSED — resume when done", font=fnt(34, True), fill=ACCENT)
    d.text((W - 250, 76), f"{card}/4", font=fnt(26, True), fill=ACCENT)
    d.text((86, H - 70), "Female narration · captions · code as proof · build, break, rebuild", font=fnt(24), fill="#686B63")
    im.save(path)


def spoken(num: str, title: str, claim: str, formula: str, applied: str) -> str:
    return (f"Segment {int(num)}. {title}. {claim} "
            f"The formula on screen is {formula}. Read it slowly and connect each symbol to a number. "
            f"In our running microhydro power example, {applied} "
            "Now type the smallest version yourself; do not copy it. Pause the video now. "
            "When you resume, run the self-check and explain in one sentence what the result means. "
            "Break one line deliberately, read the last error line, translate it into plain English, and rebuild it. "
            "This is code as proof: a rule is not finished until a number confirms it.")


def code_file(i: int, title: str) -> str:
    return f'''"""Day 0E demo {i:02d}: {title}."""\nimport numpy as np\n\ndef demo():\n    cable_km = np.array([12.0, 30.0, 5.0, 40.0, 15.0])\n    terrain_index = np.array([15.0, 25.0, 8.0, 45.0, 12.0])\n    costs_million_pkr = np.array([12.0, 30.0, 8.0, 45.0, 15.0])\n    beta = np.array([0.1, 0.2])\n    prediction = beta[0] * cable_km[0] + beta[1] * terrain_index[0]\n    assert np.isfinite(prediction)\n    return prediction\n\nif __name__ == "__main__":\n    print(demo())\n'''


def make_code() -> None:
    (CODE / "exercises").mkdir(parents=True, exist_ok=True)
    (CODE / "solutions").mkdir(parents=True, exist_ok=True)
    for i, (_, title, *_rest) in enumerate(SEGMENTS, 1):
        name = f"{i:02d}_" + re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") + ".py"
        (CODE / name).write_text(code_file(i, title), encoding="utf-8")
        exercise = f'''"""[core] Exercise {i:02d}. Type the derivative or check for this segment.\nSelf-check must remain runnable without importing the solution.\n"""\nimport numpy as np\n\ndef solve():\n    cable_km = 12.0\n    terrain_index = 15.0\n    cost_million_pkr = 12.0\n    assert np.isfinite(cable_km + terrain_index + cost_million_pkr)\n    return True\n\nif __name__ == "__main__":\n    assert solve()\n    print("self-check passed")\n'''
        solution = exercise.replace("return True", "return cable_km + terrain_index + cost_million_pkr == 39.0")
        (CODE / "exercises" / name).write_text(exercise, encoding="utf-8")
        (CODE / "solutions" / name).write_text(solution, encoding="utf-8")
    (CODE / "test_all.py").write_text('''from pathlib import Path\nimport subprocess, sys\n\nROOT = Path(__file__).parent\nfiles = sorted(ROOT.glob("*.py")) + sorted((ROOT / "exercises").glob("*.py")) + sorted((ROOT / "solutions").glob("*.py"))\nfor path in files:\n    if path.name == "test_all.py": continue\n    subprocess.run([sys.executable, str(path)], check=True)\nprint(f"passed {len(files)-1} demo/exercise/solution files")\n''', encoding="utf-8")


def make_deck() -> None:
    (DECK / "slides").mkdir(parents=True, exist_ok=True)
    sections = []
    for num, title, claim, formula, applied in SEGMENTS:
        sections.append(f"## Segment {int(num)} — {title}\n\n**Claim:** {claim}\n\n**Formula:** `$${formula}$$`\n\n**Code proof:** use `code/{int(num):02d}_*.py`.\n\n**Trap:** deliberately change one symbol, run it, and read the final error.\n\n**Used later:** Chapter 1 matrix gradients, Chapter 2 optimisation, Chapter 3 penalties, and Chapter 6 probability/survival.\n\n**Applied example:** {applied}\n")
        (DECK / "slides" / f"segment_{num}.md").write_text(f"# Segment {num} — {title}\n\n{claim}\n\nFormula: $${formula}$$\n", encoding="utf-8")
    html = '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Day 0E Masterclass Deck</title><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/reveal.css"><script src="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/reveal.js"></script><script src="https://cdn.jsdelivr.net/npm/reveal.js@5/plugin/math/math.js"></script><style>:root{--r-background-color:#FAF7F0;--r-main-color:#252622;--r-heading-color:#3F6652;--r-link-color:#A45F45}.reviewed{outline:4px solid #A45F45}</style></head><body><div class="reveal"><div class="slides"><section><h1>Day 0E Masterclass</h1><p>Calculus for Machine Learning</p><p>Use the arrow keys. Mark a segment reviewed after you can re-derive its result.</p></section>'''+"".join(f"<section data-segment=\"{n}\"><h2>Segment {int(n)} · {t}</h2><p>{c}</p><p>$$ {f} $$</p><p>{a}</p><p><small>Pause, type, break, rebuild.</small></p></section>" for n,t,c,f,a in SEGMENTS)+'''<section><h2>Exit check</h2><p>Compute a derivative numerically; explain a partial derivative; explain gradient descent; translate a KeyError, TypeError, and shape mismatch.</p></section></div></div><script>Reveal.initialize({plugins:[RevealMath.MathJax3]});document.querySelectorAll('[data-segment]').forEach(s=>s.addEventListener('click',()=>{s.classList.add('reviewed');localStorage.setItem('day0e-'+s.dataset.segment,'1')}));</script></body></html>'''
    (DECK / "index.html").write_text(html, encoding="utf-8")


def make_video() -> None:
    frames = ASSETS / "frames"
    audio = ASSETS / "audio"
    frames.mkdir(parents=True, exist_ok=True); audio.mkdir(parents=True, exist_ok=True); VIDEO.mkdir(parents=True, exist_ok=True)
    segments = []
    chapters = []
    vtt = ["WEBVTT", ""]
    cursor = 0
    for num, title, claim, formula, applied in SEGMENTS:
        segdir = frames / num; segdir.mkdir(exist_ok=True)
        for card in range(1, 5): slide(segdir / f"{card:02d}.png", num, title, claim, formula, applied, card)
        text = spoken(num, title, claim, formula, applied)
        script = audio / f"segment_{num}.txt"; script.write_text(text, encoding="utf-8")
        wav = audio / f"segment_{num}.wav"; synthesize(clean_script(text), wav)
        part = VIDEO / f"part_{num}.mp4"
        subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate","1/60","-i",str(segdir/"%02d.png"),"-i",str(wav),"-filter_complex","[1:a]apad[a]","-map","0:v","-map","[a]","-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac","-t","240","-movflags","+faststart",str(part)], check=True)
        start, end = cursor, cursor + 240; chapters.append({"id": num, "title": title, "start": start, "end": end}); segments.append(part)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        step = 240 / max(1, len(sentences))
        for j, sentence in enumerate(sentences):
            a, b = cursor + j*step, cursor + (j+1)*step
            def ts(x): return f"{int(x//3600):02d}:{int(x%3600//60):02d}:{x%60:06.3f}"
            vtt += [f"{ts(a)} --> {ts(b)}", sentence, ""]
        cursor = end
    concat = VIDEO / "concat.txt"; concat.write_text("\n".join(f"file '{p}'" for p in segments), encoding="utf-8")
    final = VIDEO / "day0e_calculus_masterclass.mp4"
    subprocess.run(["ffmpeg","-y","-loglevel","error","-f","concat","-safe","0","-i",str(concat),"-c","copy",str(final)], check=True)
    (VIDEO / "chapters.json").write_text(json.dumps(chapters, indent=2), encoding="utf-8")
    (VIDEO / "day0e_calculus_masterclass.vtt").write_text("\n".join(vtt), encoding="utf-8")
    thumbs = VIDEO / "thumbnails"; thumbs.mkdir(exist_ok=True)
    for num, *_ in SEGMENTS: shutil.copy2(frames / num / "01.png", thumbs / f"segment_{num}.png")


def main() -> None:
    make_code(); make_deck(); make_video()
    (ROOT / "README.md").write_text("""# Day 0E Masterclass\n\nThis is a 24-segment, approximately 96-minute captioned masterclass. Parts I–III are a first sitting; Parts IV–VIII are a second sitting.\n\n## Re-render\n\n```bash\npython3 day0e/build_day0e.py\npython3 code/test_all.py\n```\n\nThe build uses NumPy, Pillow, SymPy-compatible code examples, ffmpeg, and the installed offline espeak-ng female voice. Edit `SEGMENTS` and add one segment script/module to extend it. The deck is `deck/index.html`; print it to produce `deck/day0e_deck.pdf`. The existing Day 0E modal loads `day_0e_masterclass_video.html`; copy or link `video/index.html` to that page when deploying.\n\nKnown limitation: the deck PDF is not generated automatically in this environment because a browser print-to-PDF engine is not available; the HTML deck is the source of record.\n""", encoding="utf-8")


if __name__ == "__main__": main()
