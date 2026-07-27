"""Build a print-ready HTML edition from the maintained SRC/ files.

The resulting HTML is intentionally self-contained apart from vendored local
MathJax and Mermaid scripts.  Each companion follows its chapter because PDF
cannot preserve the web edition's contextual Read More modal interaction.
"""

from __future__ import annotations

import argparse
import html
import shutil
import subprocess
import tempfile
from pathlib import Path

from compile_src import (
    CHAPTER_ORDER,
    COMPANION_FILES,
    clean_control_characters,
    convert_mermaid_fences,
    render_markdown,
    strip_legacy_companion_markup,
    title_from_markdown,
)


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "SRC"
BUILD_DIR = ROOT / "book_build"
OUTPUT_HTML = BUILD_DIR / "KP_Regression_Book_Beginner_Edition.html"
OUTPUT_PDF = ROOT / "KP_Regression_Book_Beginner_Edition.pdf"


def render_source(path: Path, *, strip_legacy: bool = False) -> tuple[str, str]:
    markdown_text = clean_control_characters(path.read_text(encoding="utf-8"))
    if strip_legacy:
        markdown_text = strip_legacy_companion_markup(markdown_text)
    title = title_from_markdown(markdown_text, path.stem)
    rendered = render_markdown(convert_mermaid_fences(markdown_text))
    return title, rendered


def build_sections() -> tuple[str, str]:
    toc_entries: list[str] = []
    sections: list[str] = []

    for index, slug in enumerate(CHAPTER_ORDER, start=1):
        source_path = SRC_DIR / f"{slug}.md"
        title, body = render_source(source_path, strip_legacy=True)
        anchor = f"part-{index}-{slug}"
        toc_entries.append(
            f'<li><a href="#{anchor}">{html.escape(title)}</a></li>'
        )
        sections.append(
            f'<section class="book-part chapter" id="{anchor}">{body}</section>'
        )

        companion_filename = COMPANION_FILES.get(slug)
        if companion_filename:
            companion_path = SRC_DIR / companion_filename
            companion_title, companion_body = render_source(companion_path)
            companion_anchor = f"{anchor}-companion"
            toc_entries.append(
                '<li class="toc-companion">'
                f'<a href="#{companion_anchor}">{html.escape(companion_title)}</a>'
                "</li>"
            )
            sections.append(
                f'<section class="book-part companion" id="{companion_anchor}">'
                '<p class="edition-label">BEGINNER COMPANION</p>'
                f"{companion_body}</section>"
            )

    return "\n".join(toc_entries), "\n".join(sections)


def build_print_html() -> None:
    toc, content = build_sections()
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KP Regression — Beginner Edition</title>
<style>
@page {{ size: A4; margin: 17mm 16mm 19mm; }}
* {{ box-sizing: border-box; }}
html {{ font-family: "Plus Jakarta Sans", "Inter", Arial, Helvetica, sans-serif; color: #172033; background: #fff; }}
body {{ margin: 0; font-size: 11pt; line-height: 1.6; }}
.cover {{ min-height: 245mm; display: flex; flex-direction: column; justify-content: center;
          text-align: center; break-after: page; padding: 18mm; }}
.cover-kicker, .edition-label {{ color: #087f8c; font-size: 10pt; font-weight: 700;
                                letter-spacing: .14em; }}
.cover h1 {{ border: 0; font-size: 34pt; margin: 8mm 0 4mm; color: #10243e; }}
.cover .subtitle {{ font-size: 16pt; color: #47627b; }}
.cover .scope {{ margin-top: 20mm; color: #52697f; }}
.toc {{ break-after: page; }}
.toc h1 {{ break-before: auto; }}
.toc ol {{ columns: 2; column-gap: 12mm; padding-left: 6mm; }}
.toc li {{ break-inside: avoid; margin: 0 0 2.5mm; font-size: 10.5pt; }}
.toc-companion {{ margin-left: 4mm !important; font-size: 9.5pt; color: #536d81; }}
a {{ color: #087f8c; text-decoration: none; }}
.book-part {{ break-before: page; }}
.companion {{ border-top: 2.5mm solid #0e8e94; padding-top: 4mm; }}
.edition-label {{ margin: 0 0 3mm; }}
h1, h2, h3, h4 {{ color: #122c49; line-height: 1.25; break-after: avoid; font-family: "Plus Jakarta Sans", "Inter", sans-serif; }}
h1 {{ font-size: 24pt; border-bottom: .6mm solid #b7d9dd; padding-bottom: 3mm; margin: 0 0 7mm; }}
h2 {{ font-size: 17pt; margin: 8mm 0 3mm; }}
h3 {{ font-size: 14pt; margin: 6mm 0 2mm; }}
h4 {{ font-size: 12pt; margin: 5mm 0 2mm; }}
p, li {{ orphans: 3; widows: 3; font-size: 11pt; line-height: 1.6; }}
blockquote {{ margin: 4mm 0; padding: 3mm 4.5mm; border-left: 1.2mm solid #15949d;
              background: #eef8f8; color: #263f55; break-inside: avoid; font-size: 10.5pt; }}
pre {{ margin: 3.5mm 0; padding: 3.5mm; border: .25mm solid #cbd6df; border-radius: 1.5mm;
       background: #f5f7f9; color: #16212d; font: 9pt/1.45 "JetBrains Mono", "DejaVu Sans Mono", monospace;
       white-space: pre-wrap; overflow-wrap: anywhere; tab-size: 4; }}
code {{ font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace; font-size: .92em; }}
p code, li code {{ background: #eef1f4; padding: .2mm .7mm; border-radius: .6mm; }}
table {{ width: 100%; border-collapse: collapse; margin: 3.5mm 0 5.5mm; font-size: 9.5pt; }}
thead {{ display: table-header-group; }}
tr {{ break-inside: avoid; }}
th, td {{ border: .22mm solid #becbd6; padding: 1.5mm 1.8mm; vertical-align: top; }}
th {{ background: #e7f2f4; color: #18364d; text-align: left; }}
img, svg, mjx-container {{ max-width: 100% !important; }}
mjx-container[display="true"] {{ margin: 3.5mm 0 !important; break-inside: avoid; }}
.diagram-panel {{ margin: 4mm 0; padding: 3mm; border: .3mm solid #bed5d8; background: #fbfefe;
                  break-inside: avoid; }}
.diagram-panel__label {{ display: none; }}
.mermaid {{ text-align: center; }}
.mermaid svg {{ max-height: 175mm; }}
hr {{ border: 0; border-top: .25mm solid #c9d5de; margin: 7mm 0; }}
.read-details-btn, .companion-more-button, script {{ display: none !important; }}
@media print {{
  a {{ color: inherit; }}
  .cover, .book-part, .toc {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
}}
</style>
<script>
window.MathJax = {{
  tex: {{ inlineMath: [['$', '$']], displayMath: [['$$', '$$']] }},
  startup: {{ typeset: false }}
}};
</script>
<script src="../vendor/mathjax/tex-svg-full.js"></script>
<script src="../vendor/mermaid.min.js"></script>
</head>
<body>
<section class="cover">
  <p class="cover-kicker">ZERO TO RESEARCH</p>
  <h1>KP Regression</h1>
  <p class="subtitle">A First-Principles Beginner Edition</p>
  <p class="scope">Python · Mathematics · Statistics · Regression · Survival Analysis · Causal Inference</p>
  <p>Complete chapters with integrated companion guides</p>
</section>
<nav class="toc">
  <h1>Contents</h1>
  <ol>{toc}</ol>
</nav>
<main>{content}</main>
<script>
(async function preparePrint() {{
  try {{
    await window.MathJax.startup.promise;
    await window.MathJax.typesetPromise();
    if (window.mermaid) {{
      window.mermaid.initialize({{ startOnLoad: false, theme: 'neutral', securityLevel: 'strict' }});
      document.querySelectorAll('.mermaid-source').forEach(function (source) {{
        var diagram = document.createElement('div');
        diagram.className = 'mermaid';
        diagram.textContent = source.textContent;
        source.replaceWith(diagram);
      }});
      await window.mermaid.run({{ nodes: document.querySelectorAll('.mermaid') }});
    }}
    document.documentElement.dataset.printReady = 'true';
  }} catch (error) {{
    console.error('Print preparation failed:', error);
    document.documentElement.dataset.printReady = 'error';
  }}
}})();
</script>
</body>
</html>
"""
    BUILD_DIR.mkdir(exist_ok=True)
    OUTPUT_HTML.write_text(document, encoding="utf-8")
    print(f"Built {OUTPUT_HTML.relative_to(ROOT)}")


def export_pdf() -> None:
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        raise RuntimeError("Google Chrome or Chromium is required for PDF export")

    with tempfile.TemporaryDirectory(prefix="kp-regression-pdf-") as profile:
        subprocess.run(
            [
                chrome,
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--disable-crash-reporter",
                "--disable-breakpad",
                "--allow-file-access-from-files",
                f"--user-data-dir={profile}",
                "--virtual-time-budget=30000",
                "--run-all-compositor-stages-before-draw",
                "--no-pdf-header-footer",
                f"--print-to-pdf={OUTPUT_PDF}",
                OUTPUT_HTML.as_uri(),
            ],
            check=True,
        )
    print(f"Built {OUTPUT_PDF.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Build the print-ready HTML without launching Chrome for PDF export.",
    )
    args = parser.parse_args()
    build_print_html()
    if not args.html_only:
        export_pdf()


if __name__ == "__main__":
    main()
