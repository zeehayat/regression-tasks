"""Build a print-ready PDF from one Markdown file in this project.

The generated HTML uses the same Markdown, MathJax, and Mermaid helpers as the
book compiler, then Chrome prints the prepared page to PDF. This keeps display
math and Mermaid diagrams browser-rendered before export.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

from compile_src import (
    VENDOR_MATHJAX_REL,
    VENDOR_MERMAID_REL,
    clean_control_characters,
    convert_mermaid_fences,
    title_from_markdown,
)


ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "book_build"


PRINT_CSS = """
@page {
  size: A4;
  margin: 17mm 16mm 19mm;
}
* {
  box-sizing: border-box;
}
html {
  color: #182536;
  background: #fff;
  font-family: "Source Serif Pro", "Cambria", "Georgia", serif;
}
body {
  margin: 0;
  font-size: 11pt;
  line-height: 1.62;
}
.cover {
  min-height: 246mm;
  display: flex;
  flex-direction: column;
  justify-content: center;
  text-align: center;
  break-after: page;
  padding: 16mm;
}
.cover-kicker {
  color: #087f8c;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 10pt;
  font-weight: 700;
  letter-spacing: .14em;
  text-transform: uppercase;
}
.cover h1 {
  margin: 7mm 0 5mm;
  color: #10243e;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 30pt;
  line-height: 1.12;
  border: 0;
}
.cover .subtitle {
  max-width: 130mm;
  margin: 0 auto;
  color: #47627b;
  font-size: 14pt;
}
.cover .source {
  margin-top: 18mm;
  color: #64748b;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 9.5pt;
}
main {
  max-width: 178mm;
  margin: 0 auto;
}
h1,
h2,
h3,
h4 {
  color: #122c49;
  font-family: Arial, Helvetica, sans-serif;
  line-height: 1.25;
  break-after: avoid;
}
h1 {
  margin: 0 0 7mm;
  padding-bottom: 3mm;
  border-bottom: .6mm solid #b7d9dd;
  font-size: 23pt;
}
h2 {
  margin: 8mm 0 3mm;
  font-size: 16.5pt;
}
h3 {
  margin: 6mm 0 2mm;
  font-size: 13.5pt;
}
h4 {
  margin: 5mm 0 2mm;
  font-size: 12pt;
}
p,
li {
  orphans: 3;
  widows: 3;
}
blockquote {
  margin: 4mm 0;
  padding: 3mm 4.5mm;
  border-left: 1.2mm solid #15949d;
  background: #eef8f8;
  color: #263f55;
  break-inside: avoid;
}
pre {
  margin: 3.5mm 0;
  padding: 3.5mm;
  border: .25mm solid #cbd6df;
  border-radius: 1.5mm;
  background: #f5f7f9;
  color: #16212d;
  font: 9pt/1.45 "DejaVu Sans Mono", "Consolas", monospace;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  tab-size: 4;
}
code {
  font-family: "DejaVu Sans Mono", "Consolas", monospace;
  font-size: .92em;
}
p code,
li code,
td code {
  background: #eef1f4;
  padding: .2mm .7mm;
  border-radius: .6mm;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 3.5mm 0 5.5mm;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 9.2pt;
}
thead {
  display: table-header-group;
}
tr {
  break-inside: avoid;
}
th,
td {
  border: .22mm solid #becbd6;
  padding: 1.5mm 1.8mm;
  vertical-align: top;
}
th {
  background: #e7f2f4;
  color: #18364d;
  text-align: left;
}
hr {
  border: 0;
  border-top: .25mm solid #c9d5de;
  margin: 7mm 0;
}
a {
  color: #087f8c;
  text-decoration: none;
}
img,
svg,
mjx-container {
  max-width: 100% !important;
}
mjx-container[display="true"] {
  margin: 3.5mm 0 !important;
  break-inside: avoid;
}
.diagram-panel {
  margin: 4mm 0;
  padding: 3mm;
  border: .3mm solid #bed5d8;
  background: #fbfefe;
  break-inside: avoid;
}
.diagram-panel__label {
  display: none;
}
.mermaid {
  text-align: center;
}
.mermaid svg {
  width: 100%;
  height: auto;
  max-height: 175mm;
}
@media print {
  a {
    color: inherit;
  }
  .cover,
  main,
  blockquote,
  th {
    print-color-adjust: exact;
    -webkit-print-color-adjust: exact;
  }
}
"""


def render_markdown_for_pdf(markdown_text: str) -> str:
    """Render Markdown while preserving MathJax display and inline TeX."""
    math_spans: list[str] = []

    def stash(match: "re.Match[str]") -> str:
        token = f"CODEXMATH{len(math_spans):06d}PLACEHOLDER"
        math_spans.append("\n".join(line.rstrip() for line in match.group(0).splitlines()))
        return f"\n\n{token}\n\n" if match.group(0).startswith("$$") else token

    protected = re.sub(r"(?<!\\)\$\$(.*?)(?<!\\)\$\$", stash, markdown_text, flags=re.DOTALL)
    protected = re.sub(r"(?<!\\)\$(?!\$)([^$\n]+?)(?<!\\)\$", stash, protected)
    rendered = markdown.markdown(
        protected,
        extensions=["fenced_code", "tables", "toc", "sane_lists", "attr_list", "nl2br"],
    )
    for index, math_span in enumerate(math_spans):
        rendered = rendered.replace(
            f"CODEXMATH{index:06d}PLACEHOLDER",
            html.escape(math_span, quote=False),
        )
    return rendered


def default_output_path(source: Path) -> Path:
    return ROOT / f"{source.stem}.pdf"


def build_print_html(source: Path, html_path: Path) -> str:
    markdown_text = clean_control_characters(source.read_text(encoding="utf-8"))
    title = title_from_markdown(markdown_text, source.stem)
    body = render_markdown_for_pdf(convert_mermaid_fences(markdown_text))
    source_label = html.escape(str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source))

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{PRINT_CSS}</style>
<script>
window.MathJax = {{
  tex: {{ inlineMath: [['$', '$']], displayMath: [['$$', '$$']] }},
  startup: {{ typeset: false }}
}};
</script>
<script src="{VENDOR_MATHJAX_REL}"></script>
<script src="{VENDOR_MERMAID_REL}"></script>
</head>
<body>
<section class="cover">
  <p class="cover-kicker">Regression Study Guide</p>
  <h1>{html.escape(title)}</h1>
  <p class="subtitle">Print-ready study guide with browser-rendered mathematics and diagrams</p>
  <p class="source">Source: {source_label}</p>
</section>
<main>{body}</main>
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
    html_path.write_text(document, encoding="utf-8")
    return title


def export_pdf(html_path: Path, pdf_path: Path) -> None:
    chrome = (
        shutil.which("google-chrome-real")
        or shutil.which("google-chrome")
        or shutil.which("chromium")
    )
    if not chrome:
        raise RuntimeError("Google Chrome or Chromium is required for PDF export")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="markdown-pdf-") as profile:
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
                f"--print-to-pdf={pdf_path}",
                html_path.as_uri(),
            ],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Markdown file to export")
    parser.add_argument("--output", type=Path, help="PDF output path")
    parser.add_argument("--html-only", action="store_true", help="Only build the intermediate print HTML")
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() != ".md":
        raise ValueError("source must be a Markdown .md file")

    BUILD_DIR.mkdir(exist_ok=True)
    html_path = BUILD_DIR / f"{source.stem}.print.html"
    pdf_path = (args.output.resolve() if args.output else default_output_path(source))

    title = build_print_html(source, html_path)
    print(f"Built print HTML for {title}: {html_path.relative_to(ROOT)}")

    if not args.html_only:
        export_pdf(html_path, pdf_path)
        print(f"Built PDF: {pdf_path.relative_to(ROOT) if pdf_path.is_relative_to(ROOT) else pdf_path}")


if __name__ == "__main__":
    main()
