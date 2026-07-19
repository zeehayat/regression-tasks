import re
from pathlib import Path
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
HTML_BOOK_DIR = ROOT / "HTML BOOK"
OUTPUT_FILE = ROOT / "regression_study_guide.html"

print("Compiling HTML Book into monolithic regression_study_guide.html...")

# 1. Read index.html as the base template
index_path = HTML_BOOK_DIR / "index.html"
with open(index_path, "r", encoding="utf-8") as f:
    template = f.read()

# 2. Extract chapters and compile content
chapters = [
    "level_0_orientation.html",
    "level_1_data_explorer.html",
    "level_1_5_adaptive_optimizers.html",
    "level_2_beginner_modeler.html",
    "level_3_practitioner.html",
    "level_4_advanced_modeler.html",
    "level_5_research_apprentice.html",
    "level_6_research_practitioner.html",
    "appendices_and_resources.html"
]

sections_html = []

for chapter in chapters:
    chapter_path = HTML_BOOK_DIR / chapter
    if not chapter_path.exists():
        print(f"Warning: {chapter} not found!")
        continue
    
    with open(chapter_path, "r", encoding="utf-8") as f:
        ch_content = f.read()
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(ch_content, "html.parser")
    content_div = soup.find(id="content")
    
    if not content_div:
        # Fallback to body content if #content not found
        content_div = soup.body if soup.body else soup
    
    clean_name = chapter.replace(".html", "")
    # Wrap in a section container
    sections_html.append(
        f'<div id="section-{clean_name}" class="chapter-content-section hidden">\n'
        f'{content_div.encode_contents().decode("utf-8")}\n'
        f'</div>'
    )

combined_sections = "\n\n".join(sections_html)

# 3. Inject combined sections into the main content div
soup_template = BeautifulSoup(template, "html.parser")
main_content_div = soup_template.find(id="main-content")
if main_content_div:
    # Clear placeholder contents
    main_content_div.clear()
    # Insert compiled sections
    sections_soup = BeautifulSoup(combined_sections, "html.parser")
    main_content_div.append(sections_soup)
else:
    print("Error: Could not find #main-content in template index.html!")
    exit(1)

# Get the updated HTML string
compiled_html = str(soup_template)

# 4. Modify the Javascript Router to work monolithically (without fetch calls)
# We will replace the async navigate function and openConceptModal function in the javascript block.

old_navigate = """        async function navigate() {
            const hash = window.location.hash;
            const file = routes[hash] || 'level_0_orientation.html';

            showLoader(true);

            try {
                const response = await fetch(file);
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                const text = await response.text();

                // Extract contents within body, or target element if specified
                const parser = new DOMParser();
                const doc = parser.parseFromString(text, 'text/html');
                const pageContent = doc.getElementById('content');

                if (pageContent) {
                    mainContent.innerHTML = pageContent.innerHTML;
                } else {
                    mainContent.innerHTML = text; // fallback
                }

                // Update active state in nav menu
                document.querySelectorAll('.nav-btn').forEach(btn => {
                    btn.classList.remove('text-cyan-400', 'bg-[#151d30]/80', 'border-cyan-500/20');
                    btn.classList.add('text-[#cbd5e1]', 'border-transparent');
                });

                const activeId = activeNavItemId[hash] || 'nav-chapter0';
                const activeBtn = document.getElementById(activeId);
                if (activeBtn) {
                    activeBtn.classList.remove('text-[#cbd5e1]', 'border-transparent');
                    activeBtn.classList.add('text-cyan-400', 'bg-[#151d30]/80', 'border-cyan-500/20');
                }

                // Scroll to top
                window.scrollTo({ top: 0, behavior: 'instant' });

                // Mark progress
                markRouteCompleted(hash);

                // Run MathJax
                if (window.MathJax && MathJax.typesetPromise) {
                    MathJax.typesetPromise([mainContent]);
                }

            } catch (err) {
                console.error("Navigation error:", err);
                mainContent.innerHTML = `
                    <div class="bg-red-500/10 border border-red-500/20 text-[#ef4444] rounded-xl p-6 text-center space-y-3">
                        <i class="fa-solid fa-circle-exclamation text-3xl"></i>
                        <h3 class="text-lg font-bold">Failed to Load Page</h3>
                        <p class="text-sm text-[#cbd5e1]">There was an error loading the requested chapter. Make sure you are running a local web server (e.g., <code>python3 -m http.server 8000</code>) and not opening index.html directly from file explorer.</p>
                        <p class="text-xs text-red-400/70 font-mono">${err.message}</p>
                    </div>
                `;
            } finally {
                showLoader(false);
            }
        }"""

new_navigate = """        async function navigate() {
            const hash = window.location.hash;
            const targetFile = routes[hash] || 'level_0_orientation.html';
            const cleanName = targetFile.replace('.html', '');

            showLoader(true);

            try {
                // Hide all sections
                document.querySelectorAll('.chapter-content-section').forEach(sec => {
                    sec.classList.add('hidden');
                });

                // Show target section
                const targetSec = document.getElementById(`section-${cleanName}`);
                if (targetSec) {
                    targetSec.classList.remove('hidden');
                }

                // Update active state in nav menu
                document.querySelectorAll('.nav-btn').forEach(btn => {
                    btn.classList.remove('text-cyan-400', 'bg-[#151d30]/80', 'border-cyan-500/20');
                    btn.classList.add('text-[#cbd5e1]', 'border-transparent');
                });

                const activeId = activeNavItemId[hash] || 'nav-chapter0';
                const activeBtn = document.getElementById(activeId);
                if (activeBtn) {
                    activeBtn.classList.remove('text-[#cbd5e1]', 'border-transparent');
                    activeBtn.classList.add('text-cyan-400', 'bg-[#151d30]/80', 'border-cyan-500/20');
                }

                // Scroll to top
                window.scrollTo({ top: 0, behavior: 'instant' });

                // Mark progress
                markRouteCompleted(hash);

                // Run MathJax
                if (window.MathJax && MathJax.typesetPromise) {
                    MathJax.typesetPromise();
                }

            } catch (err) {
                console.error("Navigation error:", err);
            } finally {
                showLoader(false);
            }
        }"""

old_concept_modal = """        async function openConceptModal(file, id) {
            modalBadge.textContent = "CONCEPT EXPLANATION";
            modalBadge.className = "px-2 py-0.5 text-[10px] font-mono font-bold tracking-wider text-cyan-400 bg-cyan-400/10 rounded-md border border-cyan-400/20 uppercase";
            modalTitle.textContent = "Loading concept explanation...";
            modalBody.innerHTML = `
                <div class="flex items-center justify-center py-10">
                    <div class="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-cyan-400"></div>
                </div>
            `;
            showModal(true);

            try {
                // Fetch target page
                const response = await fetch(file);
                if (!response.ok) throw new Error(`Failed to load target file: ${file}`);
                const text = await response.text();

                // Parse target section
                const parser = new DOMParser();
                const doc = parser.parseFromString(text, 'text/html');
                const targetElement = doc.getElementById(id);

                if (!targetElement) throw new Error(`Concept element with ID "${id}" not found.`);

                // Extract titles and cleanup content
                const header = targetElement.querySelector('h1, h2, h3, h4');
                const headerText = header ? header.textContent : "Concept Explanation";
                
                // Copy node to remove title inside modal body if present
                const clone = targetElement.cloneNode(true);
                const cloneHeader = clone.querySelector('h1, h2, h3, h4');
                if (cloneHeader) cloneHeader.remove();

                modalTitle.textContent = headerText;
                modalBody.innerHTML = clone.innerHTML;

                // Render math inside modal content
                if (window.MathJax && MathJax.typesetPromise) {
                    MathJax.typesetPromise([modalBody]);
                }

            } catch (err) {
                console.error("Failed to load modal concept: ", err);
                modalTitle.textContent = "Error";
                modalBody.innerHTML = `
                    <div class="bg-red-500/10 border border-red-500/20 text-[#ef4444] rounded-xl p-4 text-sm">
                        <i class="fa-solid fa-circle-exclamation mr-2"></i> Failed to retrieve concept link: ${err.message}
                    </div>
                `;
            }
        }"""

new_concept_modal = """        async function openConceptModal(file, id) {
            modalBadge.textContent = "CONCEPT EXPLANATION";
            modalBadge.className = "px-2 py-0.5 text-[10px] font-mono font-bold tracking-wider text-cyan-400 bg-cyan-400/10 rounded-md border border-cyan-400/20 uppercase";
            modalTitle.textContent = "Concept Explanation";

            // Monolithic mode: target element is already present in the DOM
            const targetElement = document.getElementById(id);
            if (!targetElement) {
                modalBody.innerHTML = `
                    <div class="bg-red-500/10 border border-red-500/20 text-[#ef4444] rounded-xl p-4 text-sm">
                        <i class="fa-solid fa-circle-exclamation mr-2"></i> Concept element with ID "${id}" not found in document.
                    </div>
                `;
                showModal(true);
                return;
            }

            // Extract titles and cleanup content
            const header = targetElement.querySelector('h1, h2, h3, h4');
            const headerText = header ? header.textContent : "Concept Explanation";
            
            // Copy node to remove title inside modal body if present
            const clone = targetElement.cloneNode(true);
            const cloneHeader = clone.querySelector('h1, h2, h3, h4');
            if (cloneHeader) cloneHeader.remove();

            modalTitle.textContent = headerText;
            modalBody.innerHTML = clone.innerHTML;
            showModal(true);

            // Render math inside modal content
            if (window.MathJax && MathJax.typesetPromise) {
                MathJax.typesetPromise([modalBody]);
            }
        }"""

# Apply replacements to change routing logic from fetch-based to DOM-based
compiled_html = compiled_html.replace(old_navigate, new_navigate)
compiled_html = compiled_html.replace(old_concept_modal, new_concept_modal)

# 4b. Fix SRC chapter links: the template (HTML BOOK/index.html) is one
# directory below the repo root, so its SRC links are "../SRC_HTML/...".
# The monolithic file lives at the repo root itself, so those links need to
# drop the leading "../" to keep resolving to the real SRC_HTML/ directory.
compiled_html = compiled_html.replace('href="../SRC_HTML/', 'href="SRC_HTML/')

# 5. Write the compiled content to regression_study_guide.html in the root
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(compiled_html)

print("HTML Book successfully recompiled into regression_study_guide.html!")
