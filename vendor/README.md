# Vendor assets for fully offline SRC_HTML pages

Compiled pages in SRC_HTML/ work offline already (no Tailwind CDN, no
Google Fonts). Diagram source (```mermaid fences) is shown as readable
plain text, and math ($...$) is shown as raw LaTeX, with no JS required.

To additionally get live-rendered mermaid diagrams, typeset math, and the
Quill note editor while fully offline, drop these files in this directory:

- `vendor/mermaid.min.js` — from https://www.jsdelivr.com/package/npm/mermaid
  (dist/mermaid.min.js)
- `vendor/mathjax/tex-mml-chtml.js` — from https://www.jsdelivr.com/package/npm/mathjax
  (es5/tex-mml-chtml.js)
- `vendor/quill/quill.js` — from https://www.jsdelivr.com/package/npm/quill
  (dist/quill.js)
- `vendor/quill/quill.snow.css` — from https://www.jsdelivr.com/package/npm/quill
  (dist/quill.snow.css)

The compiled pages check for these local files first and only reach out
to a CDN as a fallback if they're missing and the machine has network
access. Re-run `python3 compile_src.py` after adding the files (or just
refresh the page — the paths are already wired in).
