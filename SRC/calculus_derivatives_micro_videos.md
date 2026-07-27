# 🎥 Interactive Video Series: Calculus & Differential Derivatives for Machine Learning
> **Designed Specially for Beginners in Math & Machine Learning**  
> *5 In-Depth Video Modules with Full Audio Narration, Step-by-Step Conceptual Explanations, Visual Canvas Animations, and Worked Machine Learning Examples.*

---

<style>
.video-player-card {
    background: #151d30 !important;
    border: 1px solid #223150 !important;
    border-radius: 16px !important;
    padding: 24px !important;
    margin-bottom: 32px !important;
    box-shadow: 0 15px 35px rgba(0,0,0,0.6) !important;
    color: #f1f5f9 !important;
}
.video-player-card p, .video-player-card li, .video-player-card span, .video-player-card td {
    color: #f1f5f9 !important;
    font-size: 1.05rem !important;
    line-height: 1.75 !important;
}
.video-player-card h1, .video-player-card h2, .video-player-card h3, .video-player-card h4 {
    color: #ffffff !important;
}
.video-player-card strong {
    color: #38bdf8 !important;
    font-weight: 700 !important;
}
.video-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #223150;
    padding-bottom: 14px;
    margin-bottom: 18px;
}
.video-badge {
    background: rgba(34, 211, 238, 0.15);
    color: #22d3ee !important;
    border: 1px solid rgba(34, 211, 238, 0.3);
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.05em;
}
.video-canvas-container {
    background: #0b0f19;
    border: 1px solid #223150;
    border-radius: 12px;
    position: relative;
    width: 100%;
    height: 320px;
    overflow: hidden;
    margin-bottom: 16px;
}
canvas.video-canvas {
    width: 100%;
    height: 100%;
    display: block;
}
.video-controls {
    display: flex;
    align-items: center;
    gap: 12px;
    background: #0b0f19;
    border: 1px solid #223150;
    padding: 10px 16px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.video-btn {
    background: #223150;
    color: #22d3ee !important;
    border: 1px solid rgba(34, 211, 238, 0.3);
    padding: 7px 16px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s ease;
}
.video-btn:hover {
    background: rgba(34, 211, 238, 0.2);
    border-color: #22d3ee;
}
.audio-btn {
    background: linear-gradient(135deg, #0284c7, #0891b2);
    color: #ffffff !important;
    border: none;
    padding: 7px 16px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(8, 145, 178, 0.4);
}
.video-scrubber {
    flex: 1;
    accent-color: #22d3ee;
    cursor: pointer;
}
.concept-explanation-card {
    background: #0b0f19 !important;
    border: 1px solid #223150 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    margin-top: 18px !important;
}
.worked-example-box {
    background: #0b0f19 !important;
    border-left: 4px solid #22d3ee !important;
    padding: 20px !important;
    border-radius: 0 10px 10px 0 !important;
    margin-top: 18px !important;
}
.worked-example-box p, .worked-example-box li, .worked-example-box span, .worked-example-box strong {
    color: #f1f5f9 !important;
}
</style>

## 📌 Course Introduction: Why Calculus Powers Machine Learning

If math felt abstract in school, don't worry! In Machine Learning, calculus is **not** about doing tedious manual algebra; it is the **steering wheel** of artificial intelligence.

When an AI model makes predictions (e.g. predicting house prices or classifying images), it measures its error using a **Loss Function**. Calculus gives us the exact formula to calculate:
$$\text{"If I adjust weight } w \text{ by a tiny fraction, how much will my error decrease?"}$$

Below are 5 comprehensive video modules. Each lesson includes an **Interactive Visual Canvas**, a **Full Audio Narration**, a **Deep Concept Breakdown**, and **Worked Machine Learning Examples**.

---

### 🎬 Video Lesson 1: What is a Derivative? (Intuition, Tangent Lines & Rates of Change)

<div class="video-player-card" id="lesson-1-card">
    <div class="video-header">
        <div>
            <span class="video-badge">LESSON 1 OF 5 • DURATION: 4:30</span>
            <h3 style="margin: 6px 0 0 0; color: #ffffff; font-weight: 800;">What is a Derivative? The Speedometer Analogy</h3>
        </div>
    </div>

    <div class="video-canvas-container">
        <canvas id="canvas-lesson-1" class="video-canvas" width="700" height="320"></canvas>
    </div>

    <div class="video-controls">
        <button class="video-btn" id="btn-play-1">▶ Play Animation</button>
        <button class="audio-btn" id="btn-audio-1">🔊 Listen to Audio Lesson</button>
        <button class="video-btn" id="btn-reset-1">🔄 Reset</button>
        <input type="range" min="0" max="100" value="0" class="video-scrubber" id="scrub-1">
        <span style="font-size: 11px; color: #94a3b8; font-weight: 600;" id="time-1">0:00 / 4:30</span>
    </div>

    <div class="concept-explanation-card">
        <h4 style="color: #22d3ee; margin-top: 0;">📖 Deep Concept Explanation: From Average Rate to Instantaneous Rate</h4>
        
        <p><strong>1. The Real-World Speedometer Analogy:</strong><br>
        Imagine you drive 100 miles on a highway and arrive in 2 hours. Your <em>average speed</em> was 50 mph ($100 \text{ miles} / 2 \text{ hours}$). But did you drive at exactly 50 mph every single second? No! At minute 42, you looked down at your speedometer and saw <strong>65 mph</strong>. That single-moment speed is an <strong>instantaneous rate of change</strong>—which is precisely what a <strong>derivative</strong> is!</p>

        <p><strong>2. Secant Line to Tangent Line (The Math Magic):</strong><br>
        To find average speed between time $t$ and $t + h$, we draw a <strong>secant line</strong> connecting two points on a graph:
        $$\text{Average Slope} = \frac{\Delta y}{\Delta x} = \frac{f(x+h) - f(x)}{h}$$
        When we shrink the time gap $h$ closer and closer to zero ($h \to 0$), the two points collapse into a single point, and the secant line turns into a perfectly balanced <strong>tangent line</strong>. The slope of this tangent line is the <strong>derivative</strong>:
        $$\frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$
        </p>

        <p><strong>3. Machine Learning Connection:</strong><br>
        In Machine Learning, $x$ represents a weight $w$ inside a neural network, and $f(x)$ represents the model's Error (Loss). The derivative $\frac{d\text{Loss}}{dw}$ tells us:
        <ul>
            <li>If derivative is <strong>positive ($+$)</strong>: Increasing weight $w$ increases error. We should <em>decrease</em> $w$!</li>
            <li>If derivative is <strong>negative ($-$)</strong>: Increasing weight $w$ decreases error. We should <em>increase</em> $w$!</li>
            <li>If derivative is <strong>zero ($0$)</strong>: We are at the bottom of the curve (minimum error)!</li>
        </ul>
        </p>
    </div>

    <div class="worked-example-box">
        <h4 style="color: #38bdf8; margin-top: 0;">✍️ Detailed Worked Example: Proving $f(x) = x^2$ at $x = 3$</h4>
        <p>Let's calculate the exact slope of $f(x) = x^2$ at the point $x = 3$ step-by-step using first principles:</p>
        <ol>
            <li><strong>Step 1: Write down the limit definition formula:</strong>
                $$\frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$
            </li>
            <li><strong>Step 2: Plug $f(x) = x^2$ into the formula:</strong>
                $$\frac{df}{dx} = \lim_{h \to 0} \frac{(x+h)^2 - x^2}{h}$$
            </li>
            <li><strong>Step 3: Expand the squared term $(x+h)^2 = x^2 + 2xh + h^2$:</strong>
                $$\frac{df}{dx} = \lim_{h \to 0} \frac{(x^2 + 2xh + h^2) - x^2}{h}$$
            </li>
            <li><strong>Step 4: Cancel out $x^2 - x^2 = 0$:</strong>
                $$\frac{df}{dx} = \lim_{h \to 0} \frac{2xh + h^2}{h}$$
            </li>
            <li><strong>Step 5: Factor out $h$ from the numerator and cancel with the denominator:</strong>
                $$\frac{df}{dx} = \lim_{h \to 0} \frac{h(2x + h)}{h} = \lim_{h \to 0} (2x + h)$$
            </li>
            <li><strong>Step 6: Evaluate the limit as $h \to 0$:</strong>
                $$\frac{df}{dx} = 2x + 0 = \mathbf{2x}$$
            </li>
            <li><strong>Step 7: Evaluate at $x = 3$:</strong>
                $$\text{Slope at } x = 3 \text{ is } 2(3) = \mathbf{6}$$
            </li>
        </ol>
        <p><strong>Conclusion:</strong> At $x = 3$, the function $y = x^2$ is increasing at a rate of 6 units of $y$ per 1 unit of $x$.</p>
    </div>
</div>

---

### 🎬 Video Lesson 2: The Power Rule & Fast Differentiation Shortcuts

<div class="video-player-card" id="lesson-2-card">
    <div class="video-header">
        <div>
            <span class="video-badge">LESSON 2 OF 5 • DURATION: 5:00</span>
            <h3 style="margin: 6px 0 0 0; color: #ffffff; font-weight: 800;">Fast Derivatives: The Power Rule & Polynomial Loss</h3>
        </div>
    </div>

    <div class="video-canvas-container">
        <canvas id="canvas-lesson-2" class="video-canvas" width="700" height="320"></canvas>
    </div>

    <div class="video-controls">
        <button class="video-btn" id="btn-play-2">▶ Play Animation</button>
        <button class="audio-btn" id="btn-audio-2">🔊 Listen to Audio Lesson</button>
        <button class="video-btn" id="btn-reset-2">🔄 Reset</button>
        <input type="range" min="0" max="100" value="0" class="video-scrubber" id="scrub-2">
        <span style="font-size: 11px; color: #94a3b8; font-weight: 600;" id="time-2">0:00 / 5:00</span>
    </div>

    <div class="concept-explanation-card">
        <h4 style="color: #22d3ee; margin-top: 0;">📖 Deep Concept Explanation: Differentiating Without Limits</h4>
        
        <p><strong>1. Why We Need Shortcuts:</strong><br>
        Computing the limit $\lim_{h \to 0} \frac{f(x+h)-f(x)}{h}$ by hand every time is slow. Mathematicians discovered patterns that allow us to differentiate instantly. The most important rule in Machine Learning is the <strong>Power Rule</strong>.</p>

        <p><strong>2. The 3 Core Rules of Differentiation:</strong>
        <ul>
            <li><strong>Rule 1: The Power Rule</strong><br>
                To differentiate $x^n$, bring exponent $n$ to the front and subtract 1 from the power:
                $$\frac{d}{dx}[x^n] = n \cdot x^{n-1}$$
            </li>
            <li><strong>Rule 2: Constant Multiple Rule</strong><br>
                A constant number multiplying a variable stays attached:
                $$\frac{d}{dx}[c \cdot x^n] = c \cdot (n \cdot x^{n-1})$$
            </li>
            <li><strong>Rule 3: Constant Term Rule</strong><br>
                The derivative of a standalone constant (fixed number) is always <strong>zero</strong> ($\frac{d}{dx}[7] = 0$) because a flat number never changes!
            </li>
        </ul>
        </p>

        <p><strong>3. Machine Learning Connection:</strong><br>
        Mean Squared Error (MSE) loss $L(w) = (w \cdot x - y)^2$ is a quadratic polynomial. Using the Power Rule, the gradient of MSE is simply $2(w \cdot x - y) \cdot x$.</p>
    </div>

    <div class="worked-example-box">
        <h4 style="color: #38bdf8; margin-top: 0;">✍️ Detailed Worked Example: Polynomial Loss Function Derivative</h4>
        <p>Differentiate the polynomial loss function: $L(w) = 4w^3 - 5w^2 + 7w - 19$</p>
        <ol>
            <li><strong>Differentiate Term 1 ($4w^3$):</strong><br>
                Bring down $3$: $4 \times (3w^{3-1}) = \mathbf{12w^2}$
            </li>
            <li><strong>Differentiate Term 2 ($-5w^2$):</strong><br>
                Bring down $2$: $-5 \times (2w^{2-1}) = \mathbf{-10w}$
            </li>
            <li><strong>Differentiate Term 3 ($7w^1$):</strong><br>
                Bring down $1$: $7 \times (1w^{1-1}) = 7 \times w^0 = \mathbf{7}$ (since $w^0 = 1$)
            </li>
            <li><strong>Differentiate Term 4 ($-19$):</strong><br>
                Standalone constant derivative is $\mathbf{0}$.
            </li>
            <li><strong>Combine all terms:</strong>
                $$\frac{dL}{dw} = \mathbf{12w^2 - 10w + 7}$$
            </li>
        </ol>
    </div>
</div>

---

### 🎬 Video Lesson 3: Partial Derivatives & Multi-Variable Loss Surfaces

<div class="video-player-card" id="lesson-3-card">
    <div class="video-header">
        <div>
            <span class="video-badge">LESSON 3 OF 5 • DURATION: 5:40</span>
            <h3 style="margin: 6px 0 0 0; color: #ffffff; font-weight: 800;">Partial Derivatives: Handling Multiple Weights $\frac{\partial L}{\partial w_1}$</h3>
        </div>
    </div>

    <div class="video-canvas-container">
        <canvas id="canvas-lesson-3" class="video-canvas" width="700" height="320"></canvas>
    </div>

    <div class="video-controls">
        <button class="video-btn" id="btn-play-3">▶ Play Animation</button>
        <button class="audio-btn" id="btn-audio-3">🔊 Listen to Audio Lesson</button>
        <button class="video-btn" id="btn-reset-3">🔄 Reset</button>
        <input type="range" min="0" max="100" value="0" class="video-scrubber" id="scrub-3">
        <span style="font-size: 11px; color: #94a3b8; font-weight: 600;" id="time-3">0:00 / 5:40</span>
    </div>

    <div class="concept-explanation-card">
        <h4 style="color: #22d3ee; margin-top: 0;">📖 Deep Concept Explanation: Slicing Multivariable Surfaces</h4>
        
        <p><strong>1. Single Variable vs. Multivariable Real World:</strong><br>
        In single-variable calculus, we have 1 input $x$. But a Linear Regression model has weight $w$ AND bias $b$: $y = w \cdot x + b$. Neural networks have thousands or millions of weights! We need to know how Loss changes when we tweak <em>one specific weight</em> while leaving all other weights untouched.</p>

        <p><strong>2. The Partial Derivative Golden Rule:</strong><br>
        We use the curly partial symbol $\partial$ (pronounced "del"). When taking $\frac{\partial L}{\partial w_1}$:
        $$\mathbf{\text{Treat all other variables (like } w_2, b) \text{ as if they are plain fixed numbers (constants)!}}$$
        </p>

        <p><strong>3. Geometrical Meaning (The 3D Mountain Slice):</strong><br>
        Imagine a 3D bowl surface representing Error $L(w_1, w_2)$. Taking $\frac{\partial L}{\partial w_1}$ cuts a flat 2D slice through the bowl along the $w_1$ direction and measures the steepness of that single slice.</p>
    </div>

    <div class="worked-example-box">
        <h4 style="color: #38bdf8; margin-top: 0;">✍️ Detailed Worked Example: 2-Weight Partial Derivative Calculation</h4>
        <p>Given the 2-weight Loss function: $L(w_1, w_2) = 5w_1^2 + 4w_1 w_2 + 3w_2^2 - 8w_1 + 2$</p>
        <ol>
            <li><strong>Find Partial Derivative with respect to $w_1$ ($\frac{\partial L}{\partial w_1}$):</strong><br>
                Treat $w_2$ as a constant number!
                $$\frac{\partial L}{\partial w_1} = \frac{\partial}{\partial w_1}[5w_1^2] + \frac{\partial}{\partial w_1}[(4w_2) \cdot w_1] + \frac{\partial}{\partial w_1}[3w_2^2 - 8w_1 + 2]$$
                $$\frac{\partial L}{\partial w_1} = 10w_1 + 4w_2 + 0 - 8 + 0 = \mathbf{10w_1 + 4w_2 - 8}$$
            </li>
            <li><strong>Find Partial Derivative with respect to $w_2$ ($\frac{\partial L}{\partial w_2}$):</strong><br>
                Treat $w_1$ as a constant number!
                $$\frac{\partial L}{\partial w_2} = \frac{\partial}{\partial w_2}[5w_1^2] + \frac{\partial}{\partial w_2}[(4w_1) \cdot w_2] + \frac{\partial}{\partial w_2}[3w_2^2 - 8w_1 + 2]$$
                $$\frac{\partial L}{\partial w_2} = 0 + 4w_1 + 6w_2 - 0 + 0 = \mathbf{4w_1 + 6w_2}$$
            </li>
            <li><strong>Evaluate Gradient at point $(w_1 = 1, w_2 = 2)$:</strong>
                $$\frac{\partial L}{\partial w_1} = 10(1) + 4(2) - 8 = 10 + 8 - 8 = \mathbf{10}$$
                $$\frac{\partial L}{\partial w_2} = 4(1) + 6(2) = 4 + 12 = \mathbf{16}$$
            </li>
        </ol>
    </div>
</div>

---

### 🎬 Video Lesson 4: The Chain Rule & Backpropagation (The Neural Network Engine)

<div class="video-player-card" id="lesson-4-card">
    <div class="video-header">
        <div>
            <span class="video-badge">LESSON 4 OF 5 • DURATION: 6:10</span>
            <h3 style="margin: 6px 0 0 0; color: #ffffff; font-weight: 800;">The Chain Rule: How Backpropagation Flow Works</h3>
        </div>
    </div>

    <div class="video-canvas-container">
        <canvas id="canvas-lesson-4" class="video-canvas" width="700" height="320"></canvas>
    </div>

    <div class="video-controls">
        <button class="video-btn" id="btn-play-4">▶ Play Animation</button>
        <button class="audio-btn" id="btn-audio-4">🔊 Listen to Audio Lesson</button>
        <button class="video-btn" id="btn-reset-4">🔄 Reset</button>
        <input type="range" min="0" max="100" value="0" class="video-scrubber" id="scrub-4">
        <span style="font-size: 11px; color: #94a3b8; font-weight: 600;" id="time-4">0:00 / 6:10</span>
    </div>

    <div class="concept-explanation-card">
        <h4 style="color: #22d3ee; margin-top: 0;">📖 Deep Concept Explanation: Multiplying Local Rates Across Layers</h4>
        
        <p><strong>1. The Bicycle Gears Analogy:</strong><br>
        Imagine a bicycle with 3 connected gears: Pedal Gear A turns Middle Gear B 2x as fast. Middle Gear B turns Rear Wheel C 3x as fast. How fast does turning Pedal Gear A rotate Rear Wheel C?
        $$\text{Total Speed Ratio} = 2 \times 3 = \mathbf{6}$$
        That simple multiplication of connected rates is the **Chain Rule**!</p>

        <p><strong>2. Chain Rule Math Definition:</strong><br>
        When a variable $y$ depends on $u$, and $u$ depends on $x$ ($y = f(u)$ where $u = g(x)$):
        $$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$
        </p>

        <p><strong>3. Backpropagation in Neural Networks:</strong><br>
        In Deep Learning, a network passes inputs forward through layers:
        $$\text{Weight } w \longrightarrow \text{Weighted Sum } z = w\cdot x + b \longrightarrow \text{Activation } a = \sigma(z) \longrightarrow \text{Loss } L$$
        To find how a weight in layer 1 affects final Loss, we multiply local derivatives backward:
        $$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$
        </p>
    </div>

    <div class="worked-example-box">
        <h4 style="color: #38bdf8; margin-top: 0;">✍️ Detailed Worked Example: Single Artificial Neuron Gradient</h4>
        <p>Let Loss $L = (a - y)^2$, activation $a = \sigma(z)$, and $z = w \cdot x + b$. Calculate $\frac{\partial L}{\partial w}$:</p>
        <ol>
            <li><strong>Link 1 ($\frac{\partial L}{\partial a}$):</strong> Derivative of loss with respect to activation $a$:
                $$\frac{\partial}{\partial a}[(a - y)^2] = \mathbf{2(a - y)}$$
            </li>
            <li><strong>Link 2 ($\frac{\partial a}{\partial z}$):</strong> Derivative of Sigmoid activation $\sigma(z)$:
                $$\frac{\partial \sigma}{\partial z} = \mathbf{a(1 - a)}$$
            </li>
            <li><strong>Link 3 ($\frac{\partial z}{\partial w}$):</strong> Derivative of linear sum $z = w\cdot x + b$ with respect to weight $w$:
                $$\frac{\partial}{\partial w}[w \cdot x + b] = \mathbf{x}$$
            </li>
            <li><strong>Multiply the 3 links together (Chain Rule):</strong>
                $$\frac{\partial L}{\partial w} = \mathbf{2(a - y) \cdot a(1 - a) \cdot x}$$
            </li>
        </ol>
    </div>
</div>

---

### 🎬 Video Lesson 5: Gradient Descent in Action (Stepping Down Error Curves)

<div class="video-player-card" id="lesson-5-card">
    <div class="video-header">
        <div>
            <span class="video-badge">LESSON 5 OF 5 • DURATION: 5:10</span>
            <h3 style="margin: 6px 0 0 0; color: #ffffff; font-weight: 800;">Gradient Descent: Stepping Down to Minimum Error</h3>
        </div>
    </div>

    <div class="video-canvas-container">
        <canvas id="canvas-lesson-5" class="video-canvas" width="700" height="320"></canvas>
    </div>

    <div class="video-controls">
        <button class="video-btn" id="btn-play-5">▶ Play Animation</button>
        <button class="audio-btn" id="btn-audio-5">🔊 Listen to Audio Lesson</button>
        <button class="video-btn" id="btn-reset-5">🔄 Reset</button>
        <input type="range" min="0" max="100" value="0" class="video-scrubber" id="scrub-5">
        <span style="font-size: 11px; color: #94a3b8; font-weight: 600;" id="time-5">0:00 / 5:10</span>
    </div>

    <div class="concept-explanation-card">
        <h4 style="color: #22d3ee; margin-top: 0;">📖 Deep Concept Explanation: How AI Learns</h4>
        
        <p><strong>1. The Gradient Vector:</strong><br>
        The <strong>gradient</strong> $\nabla L$ collects all partial derivatives into a vector:
        $$\nabla L = \left[ \frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}, \dots, \frac{\partial L}{\partial w_n} \right]$$
        The gradient vector points in the direction of steepest <em>increase</em> (uphill).</p>

        <p><strong>2. Walking Downhill (The Weight Update Rule):</strong><br>
        Since we want to <em>minimize</em> Error, we take steps in the <strong>opposite direction</strong> of the gradient:
        $$w_{\text{new}} = w_{\text{old}} - \alpha \cdot \frac{\partial L}{\partial w}$$
        where $\alpha$ is the <strong>learning rate</strong> (step size, e.g. 0.01 or 0.1).</p>

        <p><strong>3. Convergence:</strong><br>
        As the weight moves closer to the optimal point, the slope $\frac{\partial L}{\partial w}$ shrinks to near $0$. The update steps naturally slow down until the model settles at minimum loss!</p>
    </div>

    <div class="worked-example-box">
        <h4 style="color: #38bdf8; margin-top: 0;">✍️ Detailed Worked Example: 2 Complete Gradient Descent Steps</h4>
        <p>Given Loss function $L(w) = w^2 - 6w + 10$, initial weight $w_0 = 0.0$, learning rate $\alpha = 0.1$:</p>
        <ol>
            <li><strong>Derivative Formula:</strong> $\frac{dL}{dw} = 2w - 6$</li>
            <li><strong>Iteration 1 ($w_0 = 0.0$):</strong>
                <ul>
                    <li>Gradient: $\frac{dL}{dw}(0) = 2(0) - 6 = -6$</li>
                    <li>Update: $w_1 = 0.0 - (0.1)(-6) = 0.0 + 0.6 = \mathbf{0.6}$</li>
                </ul>
            </li>
            <li><strong>Iteration 2 ($w_1 = 0.6$):</strong>
                <ul>
                    <li>Gradient: $\frac{dL}{dw}(0.6) = 2(0.6) - 6 = 1.2 - 6 = -4.8$</li>
                    <li>Update: $w_2 = 0.6 - (0.1)(-4.8) = 0.6 + 0.48 = \mathbf{1.08}$</li>
                </ul>
            </li>
        </ol>
        <p><strong>Result:</strong> Weight $w$ rapidly moves from $0.0 \to 0.6 \to 1.08 \dots$ heading straight toward the true minimum at $w^* = 3.0$!</p>
    </div>
</div>

---

<!-- Interactive Canvas Animation Scripts & Speech Narration -->
<script>
(function() {
    // Audio Speech Narration Helper
    function speakText(text) {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            var utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.95;
            utterance.pitch = 1.0;
            window.speechSynthesis.speak(utterance);
        }
    }

    var audio1Text = "Lesson 1: What is a Derivative? Imagine you drive 100 miles in 2 hours. Your average speed is 50 miles per hour. But at minute 42, your speedometer showed 65 miles per hour. That instantaneous speed at one exact moment is a derivative. In Machine Learning, derivatives tell us how much our loss or error changes when we adjust a weight inside our model.";
    var audio2Text = "Lesson 2: The Power Rule. Differentiating functions using limits can be slow. The Power Rule gives us an instant shortcut. To differentiate x to the power of n, bring n to the front and subtract 1 from the exponent. For example, the derivative of x cubed is 3 x squared.";
    var audio3Text = "Lesson 3: Partial Derivatives. Machine learning models have multiple weights. A partial derivative measures how loss changes when we tweak one single weight, while holding all other weights constant as fixed numbers.";
    var audio4Text = "Lesson 4: The Chain Rule. In neural networks, information flows through layers from inputs to outputs. The Chain Rule allows us to calculate how weights in early layers affect final loss by multiplying local rates of change across layers backward.";
    var audio5Text = "Lesson 5: Gradient Descent. The gradient vector points in the direction of steepest increase. To minimize error, gradient descent takes small steps in the opposite direction of the gradient using a learning rate.";

    document.getElementById('btn-audio-1')?.addEventListener('click', function() { speakText(audio1Text); });
    document.getElementById('btn-audio-2')?.addEventListener('click', function() { speakText(audio2Text); });
    document.getElementById('btn-audio-3')?.addEventListener('click', function() { speakText(audio3Text); });
    document.getElementById('btn-audio-4')?.addEventListener('click', function() { speakText(audio4Text); });
    document.getElementById('btn-audio-5')?.addEventListener('click', function() { speakText(audio5Text); });

    // --- Canvas 1: Tangent Line Animation ---
    var c1 = document.getElementById('canvas-lesson-1');
    var ctx1 = c1 ? c1.getContext('2d') : null;
    var scrub1 = document.getElementById('scrub-1');
    var play1 = document.getElementById('btn-play-1');
    var reset1 = document.getElementById('btn-reset-1');
    var timer1 = null;

    function renderCanvas1(progress) {
        if (!ctx1) return;
        var w = c1.width, h = c1.height;
        ctx1.clearRect(0, 0, w, h);

        ctx1.strokeStyle = '#1e293b'; ctx1.lineWidth = 1;
        for (var x = 0; x < w; x += 40) { ctx1.beginPath(); ctx1.moveTo(x, 0); ctx1.lineTo(x, h); ctx1.stroke(); }
        for (var y = 0; y < h; y += 40) { ctx1.beginPath(); ctx1.moveTo(0, y); ctx1.lineTo(w, y); ctx1.stroke(); }

        ctx1.strokeStyle = '#475569'; ctx1.lineWidth = 2;
        ctx1.beginPath(); ctx1.moveTo(60, 260); ctx1.lineTo(640, 260); ctx1.stroke();
        ctx1.beginPath(); ctx1.moveTo(350, 20); ctx1.lineTo(350, 280); ctx1.stroke();

        ctx1.fillStyle = '#94a3b8'; ctx1.font = '12px sans-serif';
        ctx1.fillText('x (Weight)', 600, 278); ctx1.fillText('f(x) = x²', 360, 35);

        ctx1.strokeStyle = '#38bdf8'; ctx1.lineWidth = 3;
        ctx1.beginPath();
        for (var px = 100; px <= 600; px += 2) {
            var py = 260 - (0.0035 * Math.pow(px - 350, 2));
            if (px === 100) ctx1.moveTo(px, py); else ctx1.lineTo(px, py);
        }
        ctx1.stroke();

        var targetX = 180 + (progress / 100) * (520 - 180);
        var targetY = 260 - (0.0035 * Math.pow(targetX - 350, 2));
        var slope = -2 * 0.0035 * (targetX - 350);

        var lineLen = 90;
        ctx1.strokeStyle = '#f43f5e'; ctx1.lineWidth = 2.5;
        ctx1.beginPath();
        ctx1.moveTo(targetX - lineLen, targetY - slope * (-lineLen));
        ctx1.lineTo(targetX + lineLen, targetY - slope * (lineLen));
        ctx1.stroke();

        ctx1.fillStyle = '#f43f5e'; ctx1.beginPath(); ctx1.arc(targetX, targetY, 7, 0, Math.PI * 2); ctx1.fill();

        ctx1.fillStyle = '#0f172a'; ctx1.fillRect(20, 20, 230, 75);
        ctx1.strokeStyle = '#334155'; ctx1.strokeRect(20, 20, 230, 75);
        var mathX = ((targetX - 350) / 40).toFixed(1);
        var mathSlope = (-slope * 40).toFixed(2);
        ctx1.fillStyle = '#38bdf8'; ctx1.font = 'bold 13px sans-serif';
        ctx1.fillText('Weight x: ' + mathX, 32, 42);
        ctx1.fillStyle = '#f43f5e';
        ctx1.fillText('Derivative (Slope): ' + mathSlope, 32, 64);
        ctx1.fillStyle = '#cbd5e1'; ctx1.font = '11px sans-serif';
        ctx1.fillText(mathSlope < 0 ? 'Slope < 0 (Decrease Weight)' : 'Slope > 0 (Increase Weight)', 32, 82);
    }

    if (scrub1) scrub1.addEventListener('input', function() { renderCanvas1(parseFloat(scrub1.value)); });
    if (play1) play1.addEventListener('click', function() {
        if (timer1) { clearInterval(timer1); timer1 = null; play1.textContent = '▶ Play Animation'; return; }
        play1.textContent = '⏸ Pause';
        timer1 = setInterval(function() {
            var val = parseFloat(scrub1.value) + 0.8;
            if (val > 100) val = 0;
            scrub1.value = val;
            renderCanvas1(val);
        }, 30);
    });
    if (reset1) reset1.addEventListener('click', function() {
        if (timer1) { clearInterval(timer1); timer1 = null; }
        if (play1) play1.textContent = '▶ Play Animation';
        scrub1.value = 0; renderCanvas1(0);
    });
    renderCanvas1(0);

    // --- Canvas 2 ---
    var c2 = document.getElementById('canvas-lesson-2');
    var ctx2 = c2 ? c2.getContext('2d') : null;
    var scrub2 = document.getElementById('scrub-2');
    var play2 = document.getElementById('btn-play-2');
    var reset2 = document.getElementById('btn-reset-2');
    var timer2 = null;

    function renderCanvas2(progress) {
        if (!ctx2) return;
        var w = c2.width, h = c2.height;
        ctx2.clearRect(0, 0, w, h);

        ctx2.strokeStyle = '#1e293b'; ctx2.lineWidth = 1;
        for (var x = 0; x < w; x += 40) { ctx2.beginPath(); ctx2.moveTo(x, 0); ctx2.lineTo(x, h); ctx2.stroke(); }
        for (var y = 0; y < h; y += 40) { ctx2.beginPath(); ctx2.moveTo(0, y); ctx2.lineTo(w, y); ctx2.stroke(); }

        ctx2.strokeStyle = '#a855f7'; ctx2.lineWidth = 3;
        ctx2.beginPath();
        for (var px = 100; px <= 600; px += 2) {
            var py = 250 - (0.0025 * Math.pow(px - 350, 2));
            if (px === 100) ctx2.moveTo(px, py); else ctx2.lineTo(px, py);
        }
        ctx2.stroke();

        ctx2.strokeStyle = '#22c55e'; ctx2.lineWidth = 2; ctx2.setLineDash([5, 5]);
        ctx2.beginPath();
        ctx2.moveTo(100, 250 - (0.005 * (100 - 350) * 15));
        ctx2.lineTo(600, 250 - (0.005 * (600 - 350) * 15));
        ctx2.stroke(); ctx2.setLineDash([]);

        var curX = 140 + (progress / 100) * (420);
        var curY = 250 - (0.0025 * Math.pow(curX - 350, 2));

        ctx2.fillStyle = '#fbbf24'; ctx2.beginPath(); ctx2.arc(curX, curY, 8, 0, Math.PI * 2); ctx2.fill();

        ctx2.fillStyle = '#0f172a'; ctx2.fillRect(20, 20, 280, 80);
        ctx2.strokeStyle = '#334155'; ctx2.strokeRect(20, 20, 280, 80);
        ctx2.fillStyle = '#a855f7'; ctx2.font = 'bold 13px sans-serif';
        ctx2.fillText('Loss Function: L(w) = 0.5 * (w - 3)²', 32, 42);
        ctx2.fillStyle = '#22c55e';
        ctx2.fillText('Power Rule Derivative: dL/dw = w - 3', 32, 64);
        ctx2.fillStyle = '#fbbf24';
        ctx2.fillText('Gradient Value: ' + (((curX - 350) / 40)).toFixed(2), 32, 86);
    }
    if (scrub2) scrub2.addEventListener('input', function() { renderCanvas2(parseFloat(scrub2.value)); });
    if (play2) play2.addEventListener('click', function() {
        if (timer2) { clearInterval(timer2); timer2 = null; play2.textContent = '▶ Play Animation'; return; }
        play2.textContent = '⏸ Pause';
        timer2 = setInterval(function() {
            var val = parseFloat(scrub2.value) + 0.8;
            if (val > 100) val = 0;
            scrub2.value = val;
            renderCanvas2(val);
        }, 30);
    });
    if (reset2) reset2.addEventListener('click', function() {
        if (timer2) { clearInterval(timer2); timer2 = null; }
        if (play2) play2.textContent = '▶ Play Animation';
        scrub2.value = 0; renderCanvas2(0);
    });
    renderCanvas2(0);

    // --- Canvas 3 ---
    var c3 = document.getElementById('canvas-lesson-3');
    var ctx3 = c3 ? c3.getContext('2d') : null;
    var scrub3 = document.getElementById('scrub-3');
    var play3 = document.getElementById('btn-play-3');
    var reset3 = document.getElementById('btn-reset-3');
    var timer3 = null;

    function renderCanvas3(progress) {
        if (!ctx3) return;
        var w = c3.width, h = c3.height;
        ctx3.clearRect(0, 0, w, h);

        for (var r = 120; r > 10; r -= 15) {
            ctx3.strokeStyle = 'rgba(34, 211, 238, ' + (1 - r/130) + ')';
            ctx3.lineWidth = 1.5;
            ctx3.beginPath(); ctx3.ellipse(350, 160, r * 1.8, r * 0.8, 0, 0, Math.PI * 2); ctx3.stroke();
        }

        var sliceY = 160;
        ctx3.strokeStyle = '#f43f5e'; ctx3.lineWidth = 2; ctx3.setLineDash([4, 4]);
        ctx3.beginPath(); ctx3.moveTo(100, sliceY); ctx3.lineTo(600, sliceY); ctx3.stroke(); ctx3.setLineDash([]);

        var px = 150 + (progress / 100) * 400;
        ctx3.fillStyle = '#f43f5e'; ctx3.beginPath(); ctx3.arc(px, sliceY, 7, 0, Math.PI * 2); ctx3.fill();

        ctx3.fillStyle = '#0f172a'; ctx3.fillRect(20, 20, 270, 75);
        ctx3.strokeStyle = '#334155'; ctx3.strokeRect(20, 20, 270, 75);
        ctx3.fillStyle = '#22d3ee'; ctx3.font = 'bold 13px sans-serif';
        ctx3.fillText('Partial Derivative ∂L / ∂w₁', 32, 42);
        ctx3.fillStyle = '#f43f5e';
        ctx3.fillText('w₂ Fixed at Constant = 1.0', 32, 64);
        ctx3.fillStyle = '#cbd5e1'; ctx3.font = '11px sans-serif';
        ctx3.fillText('Measures steepness along w₁ axis only', 32, 82);
    }
    if (scrub3) scrub3.addEventListener('input', function() { renderCanvas3(parseFloat(scrub3.value)); });
    if (play3) play3.addEventListener('click', function() {
        if (timer3) { clearInterval(timer3); timer3 = null; play3.textContent = '▶ Play Animation'; return; }
        play3.textContent = '⏸ Pause';
        timer3 = setInterval(function() {
            var val = parseFloat(scrub3.value) + 0.8;
            if (val > 100) val = 0;
            scrub3.value = val;
            renderCanvas3(val);
        }, 30);
    });
    if (reset3) reset3.addEventListener('click', function() {
        if (timer3) { clearInterval(timer3); timer3 = null; }
        if (play3) play3.textContent = '▶ Play Animation';
        scrub3.value = 0; renderCanvas3(0);
    });
    renderCanvas3(0);

    // --- Canvas 4 ---
    var c4 = document.getElementById('canvas-lesson-4');
    var ctx4 = c4 ? c4.getContext('2d') : null;
    var scrub4 = document.getElementById('scrub-4');
    var play4 = document.getElementById('btn-play-4');
    var reset4 = document.getElementById('btn-reset-4');
    var timer4 = null;

    function renderCanvas4(progress) {
        if (!ctx4) return;
        var w = c4.width, h = c4.height;
        ctx4.clearRect(0, 0, w, h);

        var nodes = [
            { x: 120, y: 160, label: 'Weight (w)', color: '#38bdf8' },
            { x: 280, y: 160, label: 'z = w·x+b', color: '#a855f7' },
            { x: 440, y: 160, label: 'a = σ(z)', color: '#ec4899' },
            { x: 600, y: 160, label: 'Loss (L)', color: '#f43f5e' }
        ];

        for (var i = 0; i < nodes.length - 1; i++) {
            ctx4.strokeStyle = '#475569'; ctx4.lineWidth = 3;
            ctx4.beginPath(); ctx4.moveTo(nodes[i].x + 35, nodes[i].y); ctx4.lineTo(nodes[i+1].x - 35, nodes[i+1].y); ctx4.stroke();
        }

        nodes.forEach(function(n) {
            ctx4.fillStyle = n.color; ctx4.beginPath(); ctx4.arc(n.x, n.y, 25, 0, Math.PI * 2); ctx4.fill();
            ctx4.fillStyle = '#ffffff'; ctx4.font = 'bold 11px sans-serif'; ctx4.textAlign = 'center';
            ctx4.fillText(n.label, n.x, n.y + 42);
        });

        var backX = 600 - (progress / 100) * (600 - 120);
        ctx4.fillStyle = '#fbbf24'; ctx4.beginPath(); ctx4.arc(backX, 160, 9, 0, Math.PI * 2); ctx4.fill();

        ctx4.fillStyle = '#0f172a'; ctx4.fillRect(20, 20, 310, 65);
        ctx4.strokeStyle = '#334155'; ctx4.strokeRect(20, 20, 310, 65);
        ctx4.fillStyle = '#fbbf24'; ctx4.font = 'bold 13px sans-serif'; ctx4.textAlign = 'left';
        ctx4.fillText('Backpropagation Gradient Flow', 32, 42);
        ctx4.fillStyle = '#cbd5e1'; ctx4.font = '11px sans-serif';
        ctx4.fillText('dL/dw = (dL/da) × (da/dz) × (dz/dw)', 32, 64);
    }
    if (scrub4) scrub4.addEventListener('input', function() { renderCanvas4(parseFloat(scrub4.value)); });
    if (play4) play4.addEventListener('click', function() {
        if (timer4) { clearInterval(timer4); timer4 = null; play4.textContent = '▶ Play Animation'; return; }
        play4.textContent = '⏸ Pause';
        timer4 = setInterval(function() {
            var val = parseFloat(scrub4.value) + 0.8;
            if (val > 100) val = 0;
            scrub4.value = val;
            renderCanvas4(val);
        }, 30);
    });
    if (reset4) reset4.addEventListener('click', function() {
        if (timer4) { clearInterval(timer4); timer4 = null; }
        if (play4) play4.textContent = '▶ Play Animation';
        scrub4.value = 0; renderCanvas4(0);
    });
    renderCanvas4(0);

    // --- Canvas 5 ---
    var c5 = document.getElementById('canvas-lesson-5');
    var ctx5 = c5 ? c5.getContext('2d') : null;
    var scrub5 = document.getElementById('scrub-5');
    var play5 = document.getElementById('btn-play-5');
    var reset5 = document.getElementById('btn-reset-5');
    var timer5 = null;

    function renderCanvas5(progress) {
        if (!ctx5) return;
        var w = c5.width, h = c5.height;
        ctx5.clearRect(0, 0, w, h);

        ctx5.strokeStyle = '#22d3ee'; ctx5.lineWidth = 3;
        ctx5.beginPath();
        for (var px = 100; px <= 600; px += 2) {
            var py = 260 - (0.003 * Math.pow(px - 350, 2));
            if (px === 100) ctx5.moveTo(px, py); else ctx5.lineTo(px, py);
        }
        ctx5.stroke();

        var stepCount = Math.floor((progress / 100) * 8);
        var currentW = 120;
        ctx5.fillStyle = '#34d399';
        for (var s = 0; s <= stepCount; s++) {
            var stepY = 260 - (0.003 * Math.pow(currentW - 350, 2));
            ctx5.beginPath(); ctx5.arc(currentW, stepY, 6, 0, Math.PI * 2); ctx5.fill();
            currentW += (350 - currentW) * 0.35;
        }

        var activeY = 260 - (0.003 * Math.pow(currentW - 350, 2));
        ctx5.fillStyle = '#f43f5e'; ctx5.beginPath(); ctx5.arc(currentW, activeY, 9, 0, Math.PI * 2); ctx5.fill();

        ctx5.fillStyle = '#0f172a'; ctx5.fillRect(20, 20, 290, 70);
        ctx5.strokeStyle = '#334155'; ctx5.strokeRect(20, 20, 290, 70);
        ctx5.fillStyle = '#34d399'; ctx5.font = 'bold 13px sans-serif'; ctx5.textAlign = 'left';
        ctx5.fillText('Gradient Descent Weight Update', 32, 42);
        ctx5.fillStyle = '#cbd5e1'; ctx5.font = '11px sans-serif';
        ctx5.fillText('w_new = w_old - α · (dL / dw)', 32, 62);
    }
    if (scrub5) scrub5.addEventListener('input', function() { renderCanvas5(parseFloat(scrub5.value)); });
    if (play5) play5.addEventListener('click', function() {
        if (timer5) { clearInterval(timer5); timer5 = null; play5.textContent = '▶ Play Animation'; return; }
        play5.textContent = '⏸ Pause';
        timer5 = setInterval(function() {
            var val = parseFloat(scrub5.value) + 1.2;
            if (val > 100) val = 0;
            scrub5.value = val;
            renderCanvas5(val);
        }, 40);
    });
    if (reset5) reset5.addEventListener('click', function() {
        if (timer5) { clearInterval(timer5); timer5 = null; }
        if (play5) play5.textContent = '▶ Play Animation';
        scrub5.value = 0; renderCanvas5(0);
    });
    renderCanvas5(0);
})();
</script>
