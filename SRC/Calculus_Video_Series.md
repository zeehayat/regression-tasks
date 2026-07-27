# Derivatives and Differential Calculus for Machine Learning

## Beginner micro-lessons

Seven short, caption-led lessons. Pause after each slide, copy the rule, and solve the check question before continuing.

<style>
.video-library { display:grid; gap:1.25rem; }
.video-card { background:var(--panel); border:1px solid var(--border); border-left:5px solid var(--terracotta); border-radius:1rem; padding:1.25rem; }
.video-card h3 { margin:.1rem 0 .45rem; color:var(--accent); }
.video-card p { margin:.35rem 0 .8rem; }
.video-open { border:0; border-radius:.65rem; padding:.7rem 1rem; background:var(--accent); color:#fff; font:inherit; font-weight:700; cursor:pointer; }
.video-open:hover { background:var(--terracotta); }
.video-modal[hidden] { display:none; }
.video-modal { position:fixed; inset:0; z-index:100; display:grid; place-items:center; padding:1rem; background:rgba(37,38,34,.82); }
.video-modal__panel { width:min(1100px,96vw); max-height:95vh; overflow:auto; background:var(--bg); border:1px solid var(--border); border-radius:1rem; padding:1rem; box-shadow:0 20px 70px rgba(0,0,0,.35); }
.video-modal__head { display:flex; justify-content:space-between; align-items:center; gap:1rem; margin-bottom:.8rem; }
.video-modal__head h2 { margin:0; border:0; padding:0; font-size:1.3rem; }
.video-close, .video-fullscreen { border:1px solid var(--border); border-radius:.5rem; padding:.45rem .7rem; background:var(--panel); color:var(--text); cursor:pointer; font:inherit; font-weight:700; }
.video-modal video { display:block; width:100%; max-height:78vh; background:#171b18; border-radius:.6rem; }
</style>

<div class="video-library">
<article class="video-card"><h3>1 · Slope: the first idea behind a derivative</h3><p>Average rate of change, signs, and why curves need a local slope.</p><button class="video-open" data-video="calculus_videos/mp4/01_slope_and_rate_of_change.mp4" data-title="1 · Slope: the first idea behind a derivative">Open video</button></article>
<article class="video-card"><h3>2 · The derivative: local sensitivity</h3><p>Limits, local slope, and the sensitivity approximation Δf ≈ f′(x)Δx.</p><button class="video-open" data-video="calculus_videos/mp4/02_derivative_as_local_sensitivity.mp4" data-title="2 · The derivative: local sensitivity">Open video</button></article>
<article class="video-card"><h3>3 · Derivative rules</h3><p>Constants, powers, sums, and a complete polynomial example.</p><button class="video-open" data-video="calculus_videos/mp4/03_derivative_rules.mp4" data-title="3 · Derivative rules">Open video</button></article>
<article class="video-card"><h3>4 · The chain rule</h3><p>How local sensitivities multiply through nested functions and neural networks.</p><button class="video-open" data-video="calculus_videos/mp4/04_chain_rule.mp4" data-title="4 · The chain rule">Open video</button></article>
<article class="video-card"><h3>5 · Partial derivatives and gradients</h3><p>One sensitivity per parameter, assembled into a gradient vector.</p><button class="video-open" data-video="calculus_videos/mp4/05_partial_derivatives_and_gradients.mp4" data-title="5 · Partial derivatives and gradients">Open video</button></article>
<article class="video-card"><h3>6 · Gradient descent</h3><p>The update rule, signs, learning rate, and a worked step.</p><button class="video-open" data-video="calculus_videos/mp4/06_gradient_descent.mp4" data-title="6 · Gradient descent">Open video</button></article>
<article class="video-card"><h3>7 · Second derivatives and curvature</h3><p>Curvature, the Hessian, and why shape affects optimization.</p><button class="video-open" data-video="calculus_videos/mp4/07_second_derivatives_and_curvature.mp4" data-title="7 · Second derivatives and curvature">Open video</button></article>
</div>

<div class="video-modal" id="video-modal" hidden aria-hidden="true">
  <div class="video-modal__panel" role="dialog" aria-modal="true" aria-labelledby="video-modal-title">
    <div class="video-modal__head"><h2 id="video-modal-title">Video lesson</h2><div><button class="video-fullscreen" id="video-fullscreen" type="button">Fullscreen</button> <button class="video-close" id="video-close" type="button">Close</button></div></div>
    <video id="video-player" controls preload="metadata"></video>
  </div>
</div>

<script>
(function () {
  const modal = document.getElementById('video-modal');
  const player = document.getElementById('video-player');
  const title = document.getElementById('video-modal-title');
  const close = document.getElementById('video-close');
  const fullscreen = document.getElementById('video-fullscreen');
  function hide() { player.pause(); player.removeAttribute('src'); player.load(); modal.hidden = true; modal.setAttribute('aria-hidden', 'true'); }
  document.querySelectorAll('.video-open').forEach(function (button) {
    button.addEventListener('click', function () { title.textContent = button.dataset.title; player.src = button.dataset.video; modal.hidden = false; modal.setAttribute('aria-hidden', 'false'); player.play().catch(function () {}); });
  });
  close.addEventListener('click', hide);
  modal.addEventListener('click', function (event) { if (event.target === modal) hide(); });
  fullscreen.addEventListener('click', function () { if (player.requestFullscreen) player.requestFullscreen(); else if (player.webkitEnterFullscreen) player.webkitEnterFullscreen(); });
  document.addEventListener('keydown', function (event) { if (event.key === 'Escape' && !modal.hidden) hide(); });
})();
</script>
