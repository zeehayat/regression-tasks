# Vendor assets for fully offline SRC_HTML pages

Compiled pages in SRC_HTML/ work offline already (no Tailwind CDN, no
Google Fonts). Diagram source (```mermaid fences) is shown as readable
plain text, and math ($...$) is shown as raw LaTeX, with no JS required.

To additionally get live-rendered mermaid diagrams, typeset math, and the
Quill note editor while fully offline, drop these files in this directory:

- `vendor/mermaid.min.js` — from https://www.jsdelivr.com/package/npm/mermaid
  (dist/mermaid.min.js)
- `vendor/mathjax/tex-mml-svg.js` — pinned MathJax 3.2.2 SVG bundle from
  https://www.jsdelivr.com/package/npm/mathjax (es5/tex-mml-svg.js). The SVG
  output is self-contained and does not require a separate webfont directory.
- `vendor/quill/quill.js` — from https://www.jsdelivr.com/package/npm/quill
  (dist/quill.js)
- `vendor/quill/quill.snow.css` — from https://www.jsdelivr.com/package/npm/quill
  (dist/quill.snow.css)

The compiled pages check for these local files first and only reach out
to a CDN as a fallback if they're missing and the machine has network
access. Re-run `python3 compile_src.py` after adding the files (or just
refresh the page — the paths are already wired in).
