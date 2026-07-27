"""Compile the Markdown chapters in SRC/ into standalone HTML pages.

Each SRC/*.md file is rendered into SRC_HTML/<same-basename>.html using a shared
dark-themed template. All styling is plain, hand-authored CSS embedded directly
in the page (no Tailwind CDN runtime, no Google Fonts CDN) so the pages render
correctly with zero network access.

```mermaid fences render as a clearly-labeled "Diagram source" panel by
default (readable offline, not styled like a code block). If a local copy of
mermaid.js is dropped into vendor/mermaid.min.js (see vendor/README.md), the
page automatically upgrades to a live-rendered SVG diagram. The same
local-first/CDN-fallback pattern is used for MathJax.
"""
import html
import re
import shutil
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "SRC"
OUTPUT_DIR = ROOT / "SRC_HTML"
VENDOR_DIR = ROOT / "vendor"

# Chapters in reading order (interlude sits between chapter4 and chapter5).
CHAPTER_ORDER = [
    "chapter0",
    "level_0_orientation",
    "calculus_derivatives_micro_videos",
    "day_0e_masterclass_video",
    "chapter1",
    "chapter2",
    "chapter3",
    "chapter4",
    "chapter4_5_interlude",
    "chapter5",
    "chapter6",
]

COMPANION_FILES = {
    "chapter0": "Chapter_0_Companion_Guide_v2.md",
    "chapter1": "Chapter_1_Companion_Guide.md",
    "chapter2": "Chapter_2_Companion_Guide.md",
    "chapter3": "Chapter_3_Companion_Guide.md",
    "chapter4": "Chapter_4_Companion_Guide.md",
    "chapter5": "Chapter_5_Companion_Guide.md",
    "chapter6": "Chapter_6_Companion_Guide.md",
}
READINESS_SLUG = "chapter1_read_first"
READINESS_FILE = "Chapter_0_5_Regression_Readiness_Explanations.md"
EXERCISE_SLUG = "chapter1_exercise_notebook"
EXERCISE_FILE = "Chapter_0_5_Regression_Readiness_Workbook.md"
CHAPTER2_HELP_SLUG = "chapter2_concept_help"
CHAPTER2_HELP_FILE = "Chapter_2_Concept_Help.md"
CHAPTER3_HELP_SLUG = "chapter3_concept_help"
CHAPTER3_HELP_FILE = "Chapter_3_Concept_Help.md"
CHAPTER4_HELP_SLUG = "chapter4_concept_help"
CHAPTER4_HELP_FILE = "Chapter_4_Concept_Help.md"
CHAPTER5_HELP_SLUG = "chapter5_concept_help"
CHAPTER5_HELP_FILE = "Chapter_5_Concept_Help.md"
CHAPTER6_HELP_SLUG = "chapter6_concept_help"
CHAPTER6_HELP_FILE = "Chapter_6_Concept_Help.md"
CALCULUS_VIDEO_SLUG = "calculus_video_series"
CALCULUS_VIDEO_FILE = "Calculus_Video_Series.md"
EMBEDDED_COMPANION_CHAPTERS = {
    "chapter0", "chapter1", "chapter2", "chapter3", "chapter4", "chapter5", "chapter6"
}

# Local vendor paths (relative to SRC_HTML/*.html) checked before falling
# back to a CDN. Drop the real files in vendor/ to go fully offline.
VENDOR_MERMAID_REL = "../vendor/mermaid.min.js"
VENDOR_MATHJAX_REL = "../vendor/mathjax/tex-svg-full.js"
VENDOR_QUILL_JS_REL = "../vendor/quill/quill.js"
VENDOR_QUILL_CSS_REL = "../vendor/quill/quill.snow.css"

PLATFORM_CSS = """
        .platform-shell {
            position: fixed;
            right: 1rem;
            bottom: 1rem;
            width: min(25rem, calc(100vw - 2rem));
            max-height: calc(100vh - 2rem);
            display: flex;
            flex-direction: column;
            background: rgba(12, 18, 30, 0.96);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.35);
            overflow: hidden;
            z-index: 50;
        }
        .platform-shell.collapsed { width: auto; }
        .platform-shell.collapsed .platform-body { display: none; }
        .platform-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            padding: 0.75rem 0.9rem;
            border-bottom: 1px solid var(--border);
            background: rgba(21, 29, 48, 0.9);
        }
        .platform-title { font-size: 1.05rem; font-weight: 700; color: #fff; }
        .platform-section { font-size: 0.85rem; color: var(--text-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .platform-body { overflow-y: auto; padding: 1rem; }
        .platform-group { border-top: 1px solid var(--border); padding-top: 1rem; margin-top: 1rem; }
        .platform-group:first-child { border-top: 0; padding-top: 0; margin-top: 0; }
        .platform-row { display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; }
        .platform-label { display: block; margin-bottom: 0.4rem; font-size: 0.85rem; font-weight: 700; color: #fff; text-transform: uppercase; letter-spacing: 0.05em; }
        .platform-button {
            border: 1px solid rgba(34, 211, 238, 0.45);
            background: rgba(34, 211, 238, 0.08);
            color: var(--accent);
            border-radius: 0.375rem;
            padding: 0.5rem 0.8rem;
            font-size: 0.875rem;
            font-weight: 650;
            cursor: pointer;
        }
        .platform-button:hover:not(:disabled) { background: rgba(34, 211, 238, 0.14); }
        .platform-button:disabled {
            cursor: not-allowed;
            color: #94a3b8;
            border-color: #334155;
            background: rgba(148, 163, 184, 0.08);
        }
        .platform-textarea {
            width: 100%;
            min-height: 5rem;
            resize: vertical;
            border: 1px solid var(--border);
            border-radius: 0.375rem;
            background: #080c14;
            color: var(--text);
            padding: 0.65rem 0.75rem;
            font: inherit;
            font-size: 0.95rem;
            line-height: 1.5;
        }
        .platform-message {
            margin-top: 0.5rem;
            color: #fbbf24;
            font-size: 0.875rem;
        }
        .platform-output {
            margin-top: 0.65rem;
            white-space: pre-wrap;
            color: var(--text-muted);
            font-size: 0.925rem;
            line-height: 1.6;
        }
        .platform-history {
            max-height: 14rem;
            overflow: auto;
            display: grid;
            gap: 0.55rem;
            margin: 0.65rem 0;
        }
        .platform-turn {
            border: 1px solid var(--border);
            border-radius: 0.375rem;
            padding: 0.55rem 0.7rem;
            background: rgba(21, 29, 48, 0.55);
            color: var(--text-muted);
            font-size: 0.925rem;
            line-height: 1.5;
        }
        .platform-turn strong { color: #fff; }
        .paper-card {
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1.25rem 0;
            background: rgba(21, 29, 48, 0.55);
        }
        .paper-card h4 { margin: 0 0 0.4rem; font-size: 1.15rem; }
        .paper-card p { margin: 0.4rem 0; font-size: 0.975rem; line-height: 1.6; }
        .paper-card .paper-meta { color: #93c5fd; font-size: 0.875rem; }
        .paper-card .paper-status { color: #fbbf24; font-size: 0.875rem; }
        .settings-grid {
            display: grid;
            gap: 1rem;
        }
        .settings-card {
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            background: rgba(21, 29, 48, 0.55);
            padding: 1rem;
        }
        .settings-card h2 {
            margin-top: 0;
        }
        .settings-form-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
        }
        .settings-field {
            display: grid;
            gap: 0.25rem;
        }
        .settings-field.full {
            grid-column: 1 / -1;
        }
        .settings-field label,
        .settings-check {
            color: var(--text-muted);
            font-size: 0.85rem;
        }
        .settings-field input,
        .settings-field select {
            width: 100%;
            border: 1px solid var(--border);
            border-radius: 0.35rem;
            background: #080c14;
            color: var(--text);
            padding: 0.5rem;
            font: inherit;
        }
        .settings-actions {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }
        .settings-status {
            color: #fbbf24;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        .settings-file-viewer {
            margin-top: 1rem;
            display: grid;
            gap: 1rem;
        }
        .settings-file-card {
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            background: rgba(21, 29, 48, 0.55);
            padding: 1rem;
        }
        .settings-file-card h2 {
            margin: 0 0 0.35rem;
        }
        .settings-file-card pre {
            max-height: 26rem;
            overflow: auto;
            margin: 0.75rem 0 0;
        }
        .settings-file-meta {
            color: var(--text-muted);
            font-size: 0.85rem;
        }
        .notes-list {
            display: grid;
            gap: 0.35rem;
            margin-top: 0.55rem;
        }
        .note-link {
            display: block;
            width: 100%;
            text-align: left;
            border: 1px solid var(--border);
            border-radius: 0.35rem;
            background: rgba(21, 29, 48, 0.5);
            color: var(--accent);
            padding: 0.45rem 0.55rem;
            cursor: pointer;
            font: inherit;
        }
        .note-link:hover { border-color: rgba(34, 211, 238, 0.5); }
        .note-link small {
            display: block;
            color: var(--text-muted);
            margin-top: 0.15rem;
        }
        .modal-backdrop {
            position: fixed;
            inset: 0;
            display: none;
            align-items: center;
            justify-content: center;
            padding: 1rem;
            background: rgba(0, 0, 0, 0.68);
            z-index: 100;
        }
        .modal-backdrop.open { display: flex; }
        .note-modal {
            width: min(58rem, 100%);
            max-height: min(46rem, calc(100vh - 2rem));
            display: flex;
            flex-direction: column;
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            background: #0b0f19;
            box-shadow: 0 25px 70px rgba(0, 0, 0, 0.45);
            overflow: hidden;
        }
        .note-modal-header,
        .note-modal-footer {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            padding: 0.75rem;
            border-bottom: 1px solid var(--border);
            background: rgba(21, 29, 48, 0.9);
        }
        .note-modal-footer {
            border-top: 1px solid var(--border);
            border-bottom: 0;
        }
        .note-title-input {
            width: 100%;
            border: 1px solid var(--border);
            border-radius: 0.35rem;
            background: #080c14;
            color: var(--text);
            padding: 0.55rem;
            font: inherit;
            font-weight: 700;
        }
        .note-toolbar,
        .note-modal .ql-toolbar.ql-snow {
            border: 0;
            border-bottom: 1px solid var(--border);
            background: rgba(21, 29, 48, 0.45);
        }
        .note-modal .ql-toolbar.ql-snow .ql-picker,
        .note-modal .ql-toolbar.ql-snow button {
            color: var(--text-muted);
        }
        .note-modal .ql-toolbar.ql-snow .ql-stroke {
            stroke: var(--text-muted);
        }
        .note-modal .ql-toolbar.ql-snow .ql-fill {
            fill: var(--text-muted);
        }
        .note-modal .ql-toolbar.ql-snow .ql-picker-options {
            background: #0b0f19;
            border-color: var(--border);
        }
        .note-modal .ql-container.ql-snow {
            border: 0;
            font: inherit;
        }
        .note-editor {
            min-height: 18rem;
            flex: 1 1 auto;
            overflow: auto;
            color: var(--text);
            background: #080c14;
            outline: none;
        }
        .note-editor .ql-editor,
        .note-editor[contenteditable="true"] {
            min-height: 18rem;
            color: var(--text);
            font-size: 1rem;
            line-height: 1.6;
            padding: 1rem;
        }
        .note-editor .ql-editor.ql-blank::before {
            color: #64748b;
            font-style: normal;
        }
        .note-editor:empty::before {
            content: attr(data-placeholder);
            color: #64748b;
        }
        .note-attachments {
            border-top: 1px solid var(--border);
            background: rgba(8, 12, 20, 0.95);
            padding: 0.75rem;
            max-height: 11rem;
            overflow: auto;
        }
        .attachment-input {
            max-width: 22rem;
            color: var(--text-muted);
            font: inherit;
        }
        .attachment-list {
            display: grid;
            gap: 0.5rem;
            margin-top: 0.65rem;
        }
        .attachment-item {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            border: 1px solid var(--border);
            border-radius: 0.35rem;
            background: rgba(21, 29, 48, 0.45);
            padding: 0.5rem;
        }
        .attachment-preview {
            width: 3.5rem;
            height: 3.5rem;
            flex: 0 0 3.5rem;
            display: grid;
            place-items: center;
            border-radius: 0.35rem;
            background: #080c14;
            color: var(--text-muted);
            overflow: hidden;
            font-size: 0.78rem;
        }
        .attachment-preview img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .attachment-meta {
            min-width: 0;
        }
        .attachment-meta a {
            display: block;
            color: var(--accent);
            overflow-wrap: anywhere;
        }
        .attachment-meta small {
            color: var(--text-muted);
        }
        @media (min-width: 72.01rem) {
            body { padding-right: 27rem; }
        }
        @media (max-width: 72rem) {
            .platform-shell { position: static; width: auto; max-height: none; margin: 0 1rem 1rem; }
            .platform-shell.collapsed .platform-body { display: none; }
        }
"""

PLATFORM_JS = r"""
    <script>
        window.KP_PLATFORM = {
            chapterSlug: "{chapter_slug}",
            serviceStatus: {},
            statusTimer: null,
            currentSectionId: null,
            currentHeading: null,
            noteQuill: null,
            quillPromise: null
        };

        (function platform() {
            var state = window.KP_PLATFORM;
            var reasonText = {
                no_connectivity: "no internet connection",
                no_api_key: "no API key configured",
                provider_error: "provider error",
                disabled: "service disabled"
            };
            var featureNames = {
                tutor: "AI Tutor",
                grading: "Exercise grading",
                papers: "Research paper lookup",
                tts: "Read-aloud narration",
                sync: "Cross-device sync"
            };

            function qs(sel) { return document.querySelector(sel); }
            function qsa(sel) { return Array.prototype.slice.call(document.querySelectorAll(sel)); }
            function sectionKey(sectionId) { return state.chapterSlug + "::" + sectionId; }
            function loadStylesheet(localHref, cdnHref) {
                if (document.querySelector("link[href='" + localHref + "'], link[href='" + cdnHref + "']")) return;
                var link = document.createElement("link");
                link.rel = "stylesheet";
                link.href = localHref;
                link.onerror = function () {
                    var fallback = document.createElement("link");
                    fallback.rel = "stylesheet";
                    fallback.href = cdnHref;
                    document.head.appendChild(fallback);
                };
                document.head.appendChild(link);
            }
            function loadScript(localSrc, cdnSrc) {
                return new Promise(function (resolve, reject) {
                    if (window.Quill) {
                        resolve(window.Quill);
                        return;
                    }
                    var script = document.createElement("script");
                    script.src = localSrc;
                    script.onload = function () { resolve(window.Quill); };
                    script.onerror = function () {
                        var fallback = document.createElement("script");
                        fallback.src = cdnSrc;
                        fallback.onload = function () { resolve(window.Quill); };
                        fallback.onerror = reject;
                        document.head.appendChild(fallback);
                    };
                    document.head.appendChild(script);
                });
            }
            function loadQuill() {
                if (!state.quillPromise) {
                    loadStylesheet("../vendor/quill/quill.snow.css", "https://cdn.jsdelivr.net/npm/quill@2.0.3/dist/quill.snow.css");
                    state.quillPromise = loadScript("../vendor/quill/quill.js", "https://cdn.jsdelivr.net/npm/quill@2.0.3/dist/quill.js");
                }
                return state.quillPromise;
            }
            function displayReason(reason) { return reasonText[reason] || "provider error"; }
            function disabledMessage(service) {
                return featureNames[service] + " is not available — " + displayReason((state.serviceStatus[service] || {}).reason) + ".";
            }
            function setMessage(id, text) {
                var el = qs(id);
                if (el) el.textContent = text || "";
            }
            function setAvailable(service, selectors) {
                var status = state.serviceStatus[service] || { available: false, reason: "no_connectivity" };
                selectors.forEach(function (selector) {
                    qsa(selector).forEach(function (el) { el.disabled = !status.available; });
                });
                var msg = qs("[data-service-message='" + service + "']");
                if (msg) msg.textContent = status.available ? "" : disabledMessage(service);
            }
            function statusFallback() {
                ["tutor", "grading", "papers", "tts", "sync"].forEach(function (service) {
                    state.serviceStatus[service] = { available: false, reason: "no_connectivity" };
                });
            }
            async function fetchJson(url, options, timeoutMs) {
                var controller = new AbortController();
                var timer = setTimeout(function () { controller.abort(); }, timeoutMs || 3000);
                try {
                    var response = await fetch(url, Object.assign({}, options || {}, { signal: controller.signal }));
                    var text = await response.text();
                    var data = text ? JSON.parse(text) : {};
                    if (!response.ok) {
                        var error = new Error(data.error || response.statusText);
                        error.status = response.status;
                        error.data = data;
                        throw error;
                    }
                    return data;
                } finally {
                    clearTimeout(timer);
                }
            }
            async function refreshStatus() {
                try {
                    state.serviceStatus = await fetchJson("/api/services/status", {}, 6500);
                } catch (err) {
                    statusFallback();
                }
                applyStatus();
                loadPaperCards();
            }
            function applyStatus() {
                setAvailable("tutor", ["#tutor-submit", "#tutor-explain", "#tutor-check"]);
                setAvailable("grading", ["#grade-submit"]);
                setAvailable("tts", ["#tts-submit"]);
                setAvailable("sync", ["#sync-push", "#sync-pull"]);
                qsa(".paper-card[data-live='pending']").forEach(function (card) {
                    card.querySelector(".paper-status").textContent = disabledMessage("papers");
                });
            }

            function headingText(el) { return el ? el.textContent.replace(/\s+/g, " ").trim() : "Current section"; }
            function chooseInitialSection() {
                var headings = qsa(".markdown-body h1[id], .markdown-body h2[id], .markdown-body h3[id]");
                state.currentHeading = headings[0] || null;
                state.currentSectionId = state.currentHeading ? state.currentHeading.id : "top";
                updateSectionUi();
                if ("IntersectionObserver" in window && headings.length) {
                    var visible = new Map();
                    var observer = new IntersectionObserver(function (entries) {
                        entries.forEach(function (entry) { visible.set(entry.target, entry.isIntersecting); });
                        var active = headings.find(function (h) {
                            return h.getBoundingClientRect().top >= 0 && h.getBoundingClientRect().top < window.innerHeight * 0.45;
                        }) || headings.filter(function (h) { return h.getBoundingClientRect().top < window.innerHeight * 0.45; }).pop();
                        if (active && active.id !== state.currentSectionId) {
                            state.currentHeading = active;
                            state.currentSectionId = active.id;
                            updateSectionUi();
                        }
                    }, { rootMargin: "-10% 0px -65% 0px", threshold: [0, 1] });
                    headings.forEach(function (h) { observer.observe(h); });
                }
            }
            function getSectionText() {
                var heading = state.currentHeading;
                if (!heading) return qs(".markdown-body").textContent.slice(0, 5000);
                var level = Number(heading.tagName.substring(1));
                var parts = [headingText(heading)];
                var node = heading.nextElementSibling;
                while (node) {
                    if (/^H[1-3]$/.test(node.tagName) && Number(node.tagName.substring(1)) <= level) break;
                    parts.push(node.innerText || node.textContent || "");
                    node = node.nextElementSibling;
                }
                return parts.join("\n").replace(/\n{3,}/g, "\n\n").slice(0, 9000);
            }
            function updateSectionUi() {
                var id = state.currentSectionId || "top";
                qs("#platform-section").textContent = headingText(state.currentHeading);
                loadNote(id);
                loadProgress(id);
                loadTutorHistory(id);
                setMessage("#grade-output", "");
                setMessage("#tts-output", "");
            }

            async function loadNote(sectionId) {
                var list = qs("#notes-list");
                if (!list) return;
                list.innerHTML = '<div class="platform-output">Loading notes...</div>';
                try {
                    var data = await fetchJson("/api/notes?chapter_slug=" + encodeURIComponent(state.chapterSlug) + "&section_id=" + encodeURIComponent(sectionId), {}, 1200);
                    renderNotesList(data.notes || []);
                } catch (err) {
                    var localNotes = JSON.parse(localStorage.getItem("kp-notes::" + sectionKey(sectionId)) || "[]");
                    renderNotesList(localNotes);
                }
            }
            function stripHtml(html) {
                var div = document.createElement("div");
                div.innerHTML = html || "";
                return div.textContent || div.innerText || "";
            }
            function renderNotesList(notes) {
                var list = qs("#notes-list");
                list.innerHTML = "";
                if (!notes.length) {
                    list.innerHTML = '<div class="platform-output">No notes for this section yet.</div>';
                    return;
                }
                notes.forEach(function (note) {
                    var button = document.createElement("button");
                    button.type = "button";
                    button.className = "note-link";
                    button.dataset.noteId = note.id || "";
                    var preview = stripHtml(note.preview || note.body || "").slice(0, 90);
                    button.innerHTML = escapeHtml(note.title || "Untitled note") + (preview ? "<small>" + escapeHtml(preview) + "</small>" : "");
                    button.addEventListener("click", function () { openNote(note.id); });
                    list.appendChild(button);
                });
            }
            async function ensureNoteEditor() {
                var editor = qs("#note-editor");
                if (state.noteQuill) return state.noteQuill;
                editor.removeAttribute("contenteditable");
                try {
                    var QuillLib = await loadQuill();
                    if (!QuillLib) throw new Error("Quill unavailable");
                    state.noteQuill = new QuillLib("#note-editor", {
                        theme: "snow",
                        placeholder: "Write a full note for this section",
                        modules: {
                            toolbar: "#note-quill-toolbar",
                            history: { delay: 800, maxStack: 150, userOnly: true }
                        }
                    });
                    qs("#note-quill-toolbar").style.display = "";
                    qs("#note-modal-status").textContent = "";
                    return state.noteQuill;
                } catch (err) {
                    qs("#note-quill-toolbar").style.display = "none";
                    editor.setAttribute("contenteditable", "true");
                    qs("#note-modal-status").textContent = "Quill editor is not available; using basic editor.";
                    return null;
                }
            }
            function setEditorHtml(html) {
                var editor = qs("#note-editor");
                if (state.noteQuill) {
                    state.noteQuill.clipboard.dangerouslyPasteHTML(html || "");
                } else {
                    editor.innerHTML = html || "";
                }
            }
            function getEditorHtml() {
                if (state.noteQuill) return state.noteQuill.root.innerHTML;
                return qs("#note-editor").innerHTML;
            }
            function formatBytes(bytes) {
                var value = Number(bytes || 0);
                if (value < 1024) return value + " B";
                if (value < 1024 * 1024) return (value / 1024).toFixed(1) + " KB";
                return (value / (1024 * 1024)).toFixed(1) + " MB";
            }
            function renderAttachments(attachments) {
                var list = qs("#note-attachments-list");
                if (!list) return;
                list.innerHTML = "";
                if (!attachments.length) {
                    list.innerHTML = '<div class="platform-output">No attachments yet.</div>';
                    return;
                }
                attachments.forEach(function (item) {
                    var row = document.createElement("div");
                    row.className = "attachment-item";
                    var preview = item.is_image
                        ? '<img src="' + encodeURI(item.url) + '" alt="">'
                        : escapeHtml((item.original_name || "file").split(".").pop().toUpperCase().slice(0, 6) || "FILE");
                    row.innerHTML =
                        '<div class="attachment-preview">' + preview + '</div>' +
                        '<div class="attachment-meta">' +
                            '<a href="' + encodeURI(item.url) + '" target="_blank" rel="noopener">' + escapeHtml(item.original_name || "Attachment") + '</a>' +
                            '<small>' + escapeHtml(item.content_type || "file") + " · " + formatBytes(item.size_bytes) + '</small>' +
                        '</div>';
                    list.appendChild(row);
                });
            }
            function setAttachmentMessage(message) {
                var list = qs("#note-attachments-list");
                if (list) list.innerHTML = '<div class="platform-output">' + escapeHtml(message) + '</div>';
            }
            async function loadAttachments(noteId) {
                if (!noteId || String(noteId).indexOf("local-") === 0) {
                    setAttachmentMessage("Save this note to the local server before attaching files.");
                    return;
                }
                setAttachmentMessage("Loading attachments...");
                try {
                    var data = await fetchJson("/api/notes/" + encodeURIComponent(noteId) + "/attachments", {}, 2500);
                    renderAttachments(data.attachments || []);
                } catch (err) {
                    setAttachmentMessage("Attachments are unavailable because the local server could not be reached.");
                }
            }
            async function uploadAttachments() {
                var input = qs("#note-attachment-input");
                var files = Array.prototype.slice.call((input && input.files) || []);
                if (!files.length) return;
                var modal = qs("#note-modal");
                var noteId = modal.dataset.noteId || "";
                if (!noteId || noteId.indexOf("local-") === 0) {
                    noteId = await saveNoteModal();
                }
                if (!noteId || String(noteId).indexOf("local-") === 0) {
                    qs("#note-modal-status").textContent = "Attachments require a saved SQLite note. Start the local server and try again.";
                    input.value = "";
                    return;
                }
                qs("#note-modal-status").textContent = "Uploading attachments...";
                try {
                    for (var i = 0; i < files.length; i += 1) {
                        var form = new FormData();
                        form.append("file", files[i], files[i].name);
                        await fetchJson("/api/notes/" + encodeURIComponent(noteId) + "/attachments", {
                            method: "POST",
                            body: form
                        }, 60000);
                    }
                    input.value = "";
                    await loadAttachments(noteId);
                    loadNote(state.currentSectionId || "top");
                    qs("#note-modal-status").textContent = "Attachments uploaded.";
                } catch (err) {
                    qs("#note-modal-status").textContent = err.status === 413 ? "Attachment is too large. Maximum size is 25 MB per file." : "Could not upload attachment.";
                }
            }
            async function openNote(noteId) {
                qs("#note-modal-backdrop").classList.add("open");
                qs("#note-modal-status").textContent = "";
                await ensureNoteEditor();
                if (!noteId) {
                    qs("#note-modal").dataset.noteId = "";
                    qs("#note-title").value = "Untitled note";
                    setEditorHtml("");
                    loadAttachments(null);
                    if (state.noteQuill) state.noteQuill.focus();
                    else qs("#note-editor").focus();
                    return;
                }
                try {
                    var data = await fetchJson("/api/notes/" + encodeURIComponent(noteId), {}, 1500);
                    var note = data.note;
                    qs("#note-modal").dataset.noteId = note.id;
                    qs("#note-title").value = note.title || "Untitled note";
                    setEditorHtml(note.body || "");
                    loadAttachments(note.id);
                    if (state.noteQuill) state.noteQuill.focus();
                    else qs("#note-editor").focus();
                } catch (err) {
                    qs("#note-modal-status").textContent = "Could not load this note.";
                }
            }
            function closeNoteModal() {
                qs("#note-modal-backdrop").classList.remove("open");
            }
            function commandNote(action, value) {
                document.execCommand(action, false, value || null);
                qs("#note-editor").focus();
            }
            async function saveNoteModal() {
                var sectionId = state.currentSectionId || "top";
                var modal = qs("#note-modal");
                var noteId = modal.dataset.noteId;
                var title = qs("#note-title").value.trim() || "Untitled note";
                var body = getEditorHtml();
                var payload = { chapter_slug: state.chapterSlug, section_id: sectionId, title: title, body: body };
                qs("#note-modal-status").textContent = "Saving...";
                try {
                    var url = noteId ? "/api/notes/" + encodeURIComponent(noteId) : "/api/notes";
                    var data = await fetchJson(url, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload)
                    }, 3000);
                    if (!noteId && data.id) {
                        modal.dataset.noteId = data.id;
                        noteId = data.id;
                    }
                    qs("#note-modal-status").textContent = "Saved.";
                    loadNote(sectionId);
                    loadAttachments(noteId);
                    return noteId;
                } catch (err) {
                    var key = "kp-notes::" + sectionKey(sectionId);
                    var notes = JSON.parse(localStorage.getItem(key) || "[]");
                    var localId = noteId || "local-" + Date.now();
                    var existing = notes.find(function (n) { return String(n.id) === String(localId); });
                    if (existing) {
                        existing.title = title;
                        existing.body = body;
                        existing.preview = body;
                    } else {
                        notes.unshift({ id: localId, title: title, body: body, preview: body });
                    }
                    localStorage.setItem(key, JSON.stringify(notes));
                    modal.dataset.noteId = localId;
                    qs("#note-modal-status").textContent = "Saved locally in this browser. Start the server to sync to SQLite.";
                    renderNotesList(notes);
                    loadAttachments(localId);
                    return localId;
                }
            }
            async function loadProgress(sectionId) {
                var checkbox = qs("#progress-complete");
                var localKey = "kp-progress::" + sectionKey(sectionId);
                try {
                    var data = await fetchJson("/api/progress?chapter_slug=" + encodeURIComponent(state.chapterSlug) + "&section_id=" + encodeURIComponent(sectionId), {}, 1200);
                    checkbox.checked = Boolean(data.completed);
                } catch (err) {
                    checkbox.checked = localStorage.getItem(localKey) === "1";
                }
            }
            function saveProgress() {
                var sectionId = state.currentSectionId || "top";
                var completed = qs("#progress-complete").checked;
                localStorage.setItem("kp-progress::" + sectionKey(sectionId), completed ? "1" : "0");
                fetchJson("/api/progress", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ chapter_slug: state.chapterSlug, section_id: sectionId, completed: completed })
                }, 1500).catch(function () {});
            }

            function renderHistory(messages) {
                var history = qs("#tutor-history");
                history.innerHTML = "";
                messages.forEach(function (msg) {
                    var div = document.createElement("div");
                    div.className = "platform-turn";
                    div.innerHTML = "<strong>" + (msg.role === "assistant" ? "Tutor" : "You") + ":</strong> " + escapeHtml(msg.body);
                    history.appendChild(div);
                });
                history.scrollTop = history.scrollHeight;
            }
            async function loadTutorHistory(sectionId) {
                try {
                    var data = await fetchJson("/api/tutor/history?chapter_slug=" + encodeURIComponent(state.chapterSlug) + "&section_id=" + encodeURIComponent(sectionId), {}, 1400);
                    renderHistory(data.messages || []);
                } catch (err) {
                    renderHistory([]);
                }
            }
            async function askTutor(mode) {
                if (!(state.serviceStatus.tutor || {}).available) return;
                var input = qs("#tutor-question");
                var question = input.value.trim();
                if (!question && mode === "explain") question = "Explain this section differently, with a concrete example.";
                if (!question) return;
                setMessage("#tutor-output", "Thinking...");
                var payload = {
                    chapter_slug: state.chapterSlug,
                    section_id: state.currentSectionId || "top",
                    question: question,
                    mode: mode || "ask",
                    section_text: getSectionText()
                };
                try {
                    var response = await fetch("/api/tutor/ask", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload)
                    });
                    if (!response.ok) throw new Error("Tutor request failed");
                    var reader = response.body.getReader();
                    var decoder = new TextDecoder();
                    var text = "";
                    while (true) {
                        var chunk = await reader.read();
                        if (chunk.done) break;
                        text += decoder.decode(chunk.value, { stream: true });
                        setMessage("#tutor-output", text);
                    }
                    input.value = "";
                    loadTutorHistory(payload.section_id);
                } catch (err) {
                    setMessage("#tutor-output", disabledMessage("tutor"));
                }
            }
            async function gradeSubmission() {
                if (!(state.serviceStatus.grading || {}).available) return;
                var submission = qs("#grade-submission").value.trim();
                if (!submission) return;
                setMessage("#grade-output", "Grading...");
                try {
                    var data = await fetchJson("/api/exercises/grade", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            chapter_slug: state.chapterSlug,
                            section_id: state.currentSectionId || "top",
                            submission: submission,
                            exercise_text: getSectionText()
                        })
                    }, 30000);
                    setMessage("#grade-output", data.feedback || "");
                } catch (err) {
                    setMessage("#grade-output", disabledMessage("grading"));
                }
            }
            async function playTts() {
                if (!(state.serviceStatus.tts || {}).available) return;
                setMessage("#tts-output", "Preparing audio...");
                try {
                    var response = await fetch("/api/tts", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            chapter_slug: state.chapterSlug,
                            section_id: state.currentSectionId || "top",
                            section_text: getSectionText()
                        })
                    });
                    if (!response.ok) throw new Error("TTS request failed");
                    var blob = await response.blob();
                    var audio = new Audio(URL.createObjectURL(blob));
                    audio.play();
                    setMessage("#tts-output", "Playing this section.");
                } catch (err) {
                    setMessage("#tts-output", disabledMessage("tts"));
                }
            }
            function slugify(text) {
                return text.toLowerCase().replace(/&/g, " and ").replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 90) || "paper";
            }
            function createPaperCards() {
                if (state.chapterSlug === "settings") return;
                qsa(".markdown-body h2, .markdown-body h3").forEach(function (heading) {
                    if (!/Research paper/i.test(heading.textContent) || heading.dataset.paperCard) return;
                    heading.dataset.paperCard = "1";
                    var card = document.createElement("div");
                    card.className = "paper-card";
                    card.dataset.key = slugify(heading.textContent);
                    card.dataset.query = heading.textContent.replace(/Research papers?:?/i, "").trim() || heading.textContent.trim();
                    card.dataset.live = "pending";
                    card.innerHTML = '<div class="paper-status">Research paper lookup is loading.</div>';
                    heading.insertAdjacentElement("afterend", card);
                });
            }
            async function loadPaperCards() {
                createPaperCards();
                qsa(".paper-card").forEach(async function (card) {
                    var key = card.dataset.key;
                    var query = card.dataset.query;
                    try {
                        var data = await fetchJson("/api/papers/" + encodeURIComponent(key) + "?query=" + encodeURIComponent(query), {}, 4500);
                        var meta = data.metadata || {};
                        card.dataset.live = "loaded";
                        card.innerHTML = '<div class="paper-meta">' + (data.cached ? "cached — available offline" : "live scholarly lookup") + '</div>' +
                            '<h4>' + escapeHtml(meta.title || query || "Research paper") + '</h4>' +
                            '<p>' + escapeHtml([meta.authors, meta.year].filter(Boolean).join(" · ")) + '</p>' +
                            '<p>' + escapeHtml(meta.abstract || "No abstract returned by the provider.") + '</p>' +
                            (meta.url ? '<p><a href="' + meta.url + '" target="_blank" rel="noopener">Open paper</a></p>' : "");
                    } catch (err) {
                        card.dataset.live = "pending";
                        card.innerHTML = '<div class="paper-status">' + disabledMessage("papers") + '</div><button class="platform-button paper-retry" type="button">Retry</button>';
                        var retry = card.querySelector(".paper-retry");
                        retry.addEventListener("click", function () { refreshStatus(); });
                    }
                });
            }
            async function sync(direction) {
                var service = state.serviceStatus.sync || {};
                if (!service.available) return;
                setMessage("#sync-output", direction === "push" ? "Pushing..." : "Pulling...");
                try {
                    var data = await fetchJson("/api/sync/" + direction, { method: "POST" }, 15000);
                    setMessage("#sync-output", data.message || "Sync complete.");
                } catch (err) {
                    setMessage("#sync-output", disabledMessage("sync"));
                }
            }
            function escapeHtml(value) {
                return String(value || "").replace(/[&<>"']/g, function (ch) {
                    return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch];
                });
            }
            function bindUi() {
                qs("#platform-toggle").addEventListener("click", function () {
                    qs("#platform-shell").classList.toggle("collapsed");
                });
                qs("#status-retry").addEventListener("click", refreshStatus);
                qs("#note-new").addEventListener("click", function () { openNote(null); });
                qs("#note-close").addEventListener("click", closeNoteModal);
                qs("#note-save").addEventListener("click", saveNoteModal);
                qs("#note-attachment-input").addEventListener("change", uploadAttachments);
                qs("#note-modal-backdrop").addEventListener("click", function (event) {
                    if (event.target.id === "note-modal-backdrop") closeNoteModal();
                });
                qs("#progress-complete").addEventListener("change", saveProgress);
                qs("#tutor-submit").addEventListener("click", function () { askTutor("ask"); });
                qs("#tutor-explain").addEventListener("click", function () { askTutor("explain"); });
                qs("#tutor-check").addEventListener("click", function () { askTutor("check"); });
                qs("#grade-submit").addEventListener("click", gradeSubmission);
                qs("#tts-submit").addEventListener("click", playTts);
                qs("#sync-push").addEventListener("click", function () { sync("push"); });
                qs("#sync-pull").addEventListener("click", function () { sync("pull"); });
            }
            document.addEventListener("DOMContentLoaded", function () {
                bindUi();
                chooseInitialSection();
                createPaperCards();
                statusFallback();
                applyStatus();
                refreshStatus();
                state.statusTimer = setInterval(refreshStatus, 60000);
            });
        })();
    </script>
"""

PLATFORM_PANEL = """
    <aside id="platform-shell" class="platform-shell" aria-label="Learning tools">
        <div class="platform-header">
            <div>
                <div class="platform-title">Learning Tools</div>
                <div id="platform-section" class="platform-section">Current section</div>
            </div>
            <button id="platform-toggle" class="platform-button" type="button" title="Collapse tools">Tools</button>
        </div>
        <div class="platform-body">
            <section class="platform-group">
                <label class="platform-row"><input id="progress-complete" type="checkbox"> Mark current section complete</label>
                <div class="platform-row">
                    <div class="platform-label">Section notes</div>
                    <button id="note-new" class="platform-button" type="button">New note</button>
                </div>
                <div id="notes-list" class="notes-list"></div>
            </section>
            <section class="platform-group">
                <div class="platform-label">AI Tutor</div>
                <div class="platform-row">
                    <button id="tutor-explain" class="platform-button" type="button">Explain differently</button>
                    <button id="tutor-check" class="platform-button" type="button">Check reasoning</button>
                </div>
                <div id="tutor-history" class="platform-history"></div>
                <textarea id="tutor-question" class="platform-textarea" placeholder="Ask about this section or paste your reasoning"></textarea>
                <button id="tutor-submit" class="platform-button" type="button">Ask tutor</button>
                <div class="platform-message" data-service-message="tutor"></div>
                <div id="tutor-output" class="platform-output"></div>
            </section>
            <section class="platform-group">
                <div class="platform-label">Exercise grading</div>
                <textarea id="grade-submission" class="platform-textarea" placeholder="Paste your answer, code, or derivation"></textarea>
                <button id="grade-submit" class="platform-button" type="button">Grade my answer</button>
                <div class="platform-message" data-service-message="grading"></div>
                <div id="grade-output" class="platform-output"></div>
            </section>
            <section class="platform-group">
                <div class="platform-label">Narration</div>
                <button id="tts-submit" class="platform-button" type="button">Listen to this section</button>
                <div class="platform-message" data-service-message="tts"></div>
                <div id="tts-output" class="platform-output"></div>
            </section>
            <section class="platform-group">
                <div class="platform-label">Cross-device sync</div>
                <div class="platform-row">
                    <button id="sync-push" class="platform-button" type="button">Push</button>
                    <button id="sync-pull" class="platform-button" type="button">Pull</button>
                    <button id="status-retry" class="platform-button" type="button">Retry services</button>
                </div>
                <div class="platform-message" data-service-message="sync"></div>
                <div id="sync-output" class="platform-output">Opt-in only. Notes, progress, and tutor history stay local unless you push.</div>
            </section>
            <section class="platform-group">
                <div class="platform-label">Spaced repetition</div>
                <div class="platform-output">Stretch goal: reminder delivery is not built in this pass.</div>
            </section>
        </div>
    </aside>
    <div id="note-modal-backdrop" class="modal-backdrop">
        <div id="note-modal" class="note-modal" role="dialog" aria-modal="true" aria-label="Edit note">
            <div class="note-modal-header">
                <input id="note-title" class="note-title-input" type="text" value="Untitled note" aria-label="Note title">
                <button id="note-close" class="platform-button" type="button">Close</button>
            </div>
            <div id="note-quill-toolbar" class="note-toolbar">
                <span class="ql-formats">
                    <select class="ql-header">
                        <option value="1"></option>
                        <option value="2"></option>
                        <option selected></option>
                    </select>
                </span>
                <span class="ql-formats">
                    <button class="ql-bold" type="button"></button>
                    <button class="ql-italic" type="button"></button>
                    <button class="ql-underline" type="button"></button>
                    <button class="ql-strike" type="button"></button>
                </span>
                <span class="ql-formats">
                    <button class="ql-list" value="ordered" type="button"></button>
                    <button class="ql-list" value="bullet" type="button"></button>
                    <button class="ql-blockquote" type="button"></button>
                    <button class="ql-code-block" type="button"></button>
                </span>
                <span class="ql-formats">
                    <button class="ql-link" type="button"></button>
                    <button class="ql-clean" type="button"></button>
                </span>
            </div>
            <div id="note-editor" class="note-editor" data-placeholder="Write a full note for this section"></div>
            <div class="note-attachments">
                <div class="platform-row">
                    <div class="platform-label">Attachments</div>
                    <input id="note-attachment-input" class="attachment-input" type="file" multiple>
                </div>
                <div id="note-attachments-list" class="attachment-list"></div>
            </div>
            <div class="note-modal-footer">
                <div id="note-modal-status" class="settings-status"></div>
                <button id="note-save" class="platform-button" type="button">Save note</button>
            </div>
        </div>
    </div>
"""

SETTINGS_BODY = r"""
<h1>Settings</h1>
<p>Configure online services for the learning platform. Values are saved to local <code>config.yaml</code>; real keys are not committed and existing keys are never echoed back into this page.</p>
<div class="settings-actions">
    <button id="settings-save" class="platform-button" type="button">Save settings</button>
    <button id="settings-reload" class="platform-button" type="button">Reload</button>
    <button id="settings-test" class="platform-button" type="button">Test services</button>
    <button id="settings-files-reload" class="platform-button" type="button">View config files</button>
</div>
<div id="settings-status" class="settings-status"></div>
<div id="settings-grid" class="settings-grid"></div>
<h1>Config files</h1>
<p>This viewer is read-only and masks API keys. <code>config.yaml</code> is the default; if <code>config.json</code> exists, the server loads it too and saves future settings there.</p>
<div id="settings-file-viewer" class="settings-file-viewer"></div>

<script>
    (function settingsPage() {
        var services = ["tutor", "grading", "papers", "tts", "sync"];
        var labels = {
            tutor: "AI Tutor",
            grading: "Exercise grading",
            papers: "Research paper lookup",
            tts: "Read-aloud narration",
            sync: "Cross-device sync"
        };
        var providerOptions = {
            tutor: ["anthropic", "gemini", "lmstudio", "openai-compatible"],
            grading: ["anthropic", "gemini", "lmstudio", "openai-compatible"],
            papers: ["semantic_scholar", "crossref"],
            tts: ["openai", "google_tts", "elevenlabs"],
            sync: ["custom_rest", "supabase", "firebase"]
        };
        var defaults = {
            anthropic: "https://api.anthropic.com/v1/messages",
            gemini: "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            lmstudio: "http://127.0.0.1:1234/v1/chat/completions",
            "openai-compatible": "http://127.0.0.1:1234/v1/chat/completions",
            semantic_scholar: "https://api.semanticscholar.org/graph/v1/paper/search",
            crossref: "https://api.crossref.org/works",
            openai: "https://api.openai.com/v1/audio/speech",
            google_tts: "https://texttospeech.googleapis.com/v1/text:synthesize",
            elevenlabs: "https://api.elevenlabs.io/v1/text-to-speech",
            custom_rest: "",
            supabase: "",
            firebase: ""
        };

        function qs(sel) { return document.querySelector(sel); }
        function clean(value) {
            if (value === null || value === undefined) return "";
            if (typeof value === "object") return "";
            if (value === "[object Object]") return "";
            return value;
        }
        function el(tag, attrs, children) {
            var node = document.createElement(tag);
            Object.keys(attrs || {}).forEach(function (key) {
                if (key === "class") node.className = attrs[key];
                else if (key === "text") node.textContent = attrs[key];
                else node.setAttribute(key, attrs[key]);
            });
            (children || []).forEach(function (child) {
                node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
            });
            return node;
        }
        function input(name, label, value, type, full) {
            var id = "settings-" + name.replace(/\./g, "-");
            value = clean(value);
            return el("div", { class: "settings-field" + (full ? " full" : "") }, [
                el("label", { for: id, text: label }),
                el("input", { id: id, name: name, type: type || "text", value: value == null ? "" : value })
            ]);
        }
        function select(name, label, value, options) {
            var id = "settings-" + name.replace(/\./g, "-");
            value = clean(value);
            var selectEl = el("select", { id: id, name: name });
            options.forEach(function (option) {
                var optionEl = el("option", { value: option, text: option });
                if (option === value) optionEl.selected = true;
                selectEl.appendChild(optionEl);
            });
            return el("div", { class: "settings-field" }, [el("label", { for: id, text: label }), selectEl]);
        }
        function checkbox(name, label, checked) {
            var id = "settings-" + name.replace(/\./g, "-");
            var box = el("input", { id: id, name: name, type: "checkbox" });
            box.checked = Boolean(checked);
            return el("label", { class: "settings-check" }, [box, " " + label]);
        }
        async function fetchJson(url, options) {
            var response = await fetch(url, options || {});
            var text = await response.text();
            var data = text ? JSON.parse(text) : {};
            if (!response.ok) throw new Error(data.error || response.statusText);
            return data;
        }
        function render(settings) {
            var grid = qs("#settings-grid");
            grid.innerHTML = "";
            services.forEach(function (service) {
                var cfg = settings.services[service] || {};
                var card = el("section", { class: "settings-card", "data-service": service });
                card.appendChild(el("h2", { text: labels[service] }));
                card.appendChild(checkbox(service + ".enabled", "Enabled", cfg.enabled));
                var form = el("div", { class: "settings-form-grid" });
                form.appendChild(select(service + ".provider", "Provider", cfg.provider || providerOptions[service][0], providerOptions[service]));
                form.appendChild(input(service + ".timeout_seconds", "Probe timeout seconds", cfg.timeout_seconds || 3, "number"));
                form.appendChild(input(service + ".endpoint", "Endpoint", clean(cfg.endpoint), "text", true));
                form.appendChild(input(service + ".model", "Model", clean(cfg.model), "text"));
                form.appendChild(input(service + ".voice", "Voice", clean(cfg.voice), "text"));
                form.appendChild(input(service + ".mailto", "Contact email / mailto", clean(cfg.mailto), "email"));
                form.appendChild(input(service + ".api_key", cfg.api_key_configured ? "API key (configured; leave blank to keep)" : "API key", "", "password", true));
                form.appendChild(checkbox(service + ".clear_api_key", "Clear saved API key", false));
                card.appendChild(form);
                grid.appendChild(card);
            });
            document.querySelectorAll("select[name$='.provider']").forEach(function (selectEl) {
                selectEl.addEventListener("change", function () {
                    var service = selectEl.name.split(".")[0];
                    var endpoint = document.querySelector("[name='" + service + ".endpoint']");
                    if (endpoint && !endpoint.value.trim()) endpoint.value = defaults[selectEl.value] || "";
                });
            });
        }
        function collect() {
            var result = { services: {} };
            services.forEach(function (service) {
                var card = document.querySelector("[data-service='" + service + "']");
                var cfg = {};
                card.querySelectorAll("input, select").forEach(function (field) {
                    var key = field.name.split(".")[1];
                    if (field.type === "checkbox") cfg[key] = field.checked;
                    else if (field.type === "number") cfg[key] = Number(field.value || 0);
                    else cfg[key] = field.value.trim();
                });
                result.services[service] = cfg;
            });
            return result;
        }
        async function load() {
            qs("#settings-status").textContent = "Loading settings...";
            try {
                render(await fetchJson("/api/settings"));
                qs("#settings-status").textContent = "";
            } catch (err) {
                qs("#settings-status").textContent = "Settings are not available — start the local server with python3 server.py.";
            }
        }
        async function save() {
            qs("#settings-status").textContent = "Saving settings...";
            try {
                render(await fetchJson("/api/settings", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(collect())
                }));
                qs("#settings-status").textContent = "Saved to config.yaml. Service checks now use the updated values.";
            } catch (err) {
                qs("#settings-status").textContent = "Save failed — " + err.message;
            }
        }
        async function test() {
            qs("#settings-status").textContent = "Testing services...";
            try {
                var status = await fetchJson("/api/services/status");
                qs("#settings-status").textContent = Object.keys(status).map(function (name) {
                    return labels[name] + ": " + (status[name].available ? "available" : status[name].reason);
                }).join(" · ");
            } catch (err) {
                qs("#settings-status").textContent = "Service status is not available.";
            }
        }
        function escapeHtml(value) {
            return String(value || "").replace(/[&<>"']/g, function (ch) {
                return ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch];
            });
        }
        async function loadConfigFiles() {
            var viewer = qs("#settings-file-viewer");
            viewer.innerHTML = "";
            try {
                var data = await fetchJson("/api/settings/config-files");
                data.files.forEach(function (file) {
                    var card = el("section", { class: "settings-file-card" });
                    card.appendChild(el("h2", { text: file.name }));
                    card.appendChild(el("div", {
                        class: "settings-file-meta",
                        text: (file.exists ? "Found" : "Not found") + (file.active_write_target ? " · active save target" : "")
                    }));
                    var pre = el("pre", {}, [el("code", {})]);
                    pre.querySelector("code").innerHTML = escapeHtml(file.exists ? file.content : "No local " + file.name + " file exists yet.");
                    card.appendChild(pre);
                    viewer.appendChild(card);
                });
            } catch (err) {
                viewer.innerHTML = '<div class="settings-status">Config files are not available — start the local server with python3 server.py.</div>';
            }
        }
        document.addEventListener("DOMContentLoaded", function () {
            qs("#settings-save").addEventListener("click", save);
            qs("#settings-reload").addEventListener("click", load);
            qs("#settings-test").addEventListener("click", test);
            qs("#settings-files-reload").addEventListener("click", loadConfigFiles);
            load();
            loadConfigFiles();
        });
    })();
</script>
"""

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400;1,700&family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
    <!-- Offline Vendored Quill WYSIWYG Editor -->
    <link rel="stylesheet" href="../vendor/quill/quill.snow.css">
    <script src="../vendor/quill/quill.min.js"></script>
    <style>
        :root {{
            --bg: #FAF7F0;
            --panel: #F1EEE6;
            --border: #D8D1C5;
            --text: #252622;
            --text-muted: #686B63;
            --accent: #3F6652;
            --terracotta: #A45F45;
            --code: #ECE8DF;
        }}
        * {{ box-sizing: border-box; }}
        html, body {{ margin: 0; padding: 0; overflow-x: hidden; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: "Atkinson Hyperlegible", "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 1.0625rem;
            line-height: 1.75;
            -webkit-font-smoothing: antialiased;
        }}
        code, pre, .mono {{
            font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Consolas, monospace;
        }}
        #content {{
            width: auto;
            max-width: none;
            margin: 0 0 0 300px;
            padding: 3.5rem 2.75rem;
        }}
        .chapter-sidebar {{
            position: fixed;
            z-index: 20;
            inset: 0 auto 0 0;
            width: 280px;
            overflow-y: auto;
            padding: 1.25rem 1rem 1.5rem;
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.55), rgba(241, 238, 230, 0.92)),
                var(--panel);
            border-right: 1px solid var(--border);
            box-shadow: 12px 0 32px rgba(37, 38, 34, 0.04);
        }}
        .sidebar-brand {{
            display: grid;
            gap: 0.2rem;
            padding: 0.75rem 0.85rem 1rem;
            margin-bottom: 1rem;
            border-radius: 0.9rem;
            background: rgba(255, 255, 255, 0.45);
            border: 1px solid rgba(216, 209, 197, 0.8);
        }}
        .sidebar-brand__eyebrow {{
            color: var(--terracotta);
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }}
        .sidebar-brand__title {{
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 800;
            line-height: 1.25;
        }}
        .sidebar-brand__subtitle {{
            color: var(--text-muted);
            font-size: 0.86rem;
            line-height: 1.45;
        }}
        .sidebar-section {{
            padding: 0.9rem 0 0;
        }}
        .sidebar-section + .sidebar-section {{
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(216, 209, 197, 0.9);
        }}
        .sidebar-section-title {{
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin: 0 0 0.75rem;
            color: var(--accent);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }}
        .sidebar-section-title::after {{
            content: "";
            flex: 1 1 auto;
            height: 1px;
            background: rgba(63, 102, 82, 0.18);
        }}
        nav.chapters {{
            display: flex;
            flex-direction: column;
            align-items: stretch;
            gap: 0.45rem;
            width: auto;
            margin: 0 !important;
            padding: 0 !important;
            border: 0 !important;
        }}
        .chapter-sidebar nav.chapters a {{
            display: block;
            width: 100%;
            padding: 0.65rem 0.8rem;
            border-radius: 0.85rem;
            border: 1px solid transparent;
            background: rgba(255, 255, 255, 0.55);
            color: var(--text-muted);
            text-decoration: none;
            font-family: "Atkinson Hyperlegible", sans-serif;
            font-size: 0.93rem;
            font-weight: 700;
            line-height: 1.35;
            letter-spacing: 0.01em;
            transition: transform 0.15s ease, background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
        }}
        .chapter-sidebar nav.chapters a:hover {{
            color: var(--accent);
            background: rgba(255, 255, 255, 0.78);
            border-color: rgba(63, 102, 82, 0.25);
            transform: translateX(2px);
        }}
        .chapter-sidebar nav.chapters a.active {{
            color: var(--accent);
            background: rgba(63, 102, 82, 0.12);
            border-color: rgba(63, 102, 82, 0.35);
            box-shadow: inset 3px 0 0 var(--accent);
        }}
        .chapter-sidebar nav.chapters a:focus-visible {{
            outline: 2px solid var(--terracotta);
            outline-offset: 2px;
        }}
        @media (max-width: 1100px) {{
            #content {{ margin: 0 0 0 260px; padding: 2.5rem 1.5rem; }}
            .chapter-sidebar {{ width: 240px; }}
        }}
        @media (max-width: 760px) {{
            #content {{ margin: 0; padding: 1.5rem 1rem; }}
            .chapter-sidebar {{
                position: static;
                width: auto;
                max-height: none;
                overflow: visible;
                border-right: none;
                border-bottom: 1px solid var(--border);
                padding: 1rem;
                box-shadow: none;
            }}
        }}
        .markdown-body h1 {{ font-family: "Atkinson Hyperlegible", "Inter", sans-serif; font-size: 2.6rem; font-weight: 800; color: var(--accent); margin: 0.5rem 0 1.5rem; line-height: 1.2; }}
        .markdown-body h2 {{ font-family: "Atkinson Hyperlegible", "Inter", sans-serif; font-size: 1.85rem; font-weight: 700; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; margin: 2.5rem 0 1rem; line-height: 1.3; }}
        .markdown-body h3 {{ font-family: "Atkinson Hyperlegible", "Inter", sans-serif; font-size: 1.45rem; font-weight: 600; color: var(--accent); margin: 2rem 0 0.75rem; line-height: 1.35; }}
        .markdown-body h4 {{ font-family: "Atkinson Hyperlegible", "Inter", sans-serif; font-size: 1.2rem; font-weight: 600; color: var(--accent); margin: 1.5rem 0 0.5rem; line-height: 1.4; }}
        .markdown-body p {{ color: var(--text); font-size: 1.0625rem; line-height: 1.75; margin: 0 0 1.25rem; }}
        .markdown-body ul, .markdown-body ol {{ color: var(--text); font-size: 1.0625rem; line-height: 1.75; margin: 0 0 1.25rem; padding-left: 1.5rem; }}
        .markdown-body li {{ margin-bottom: 0.35rem; }}
        .markdown-body blockquote {{
            border-left: 4px solid var(--terracotta);
            padding: 0.85rem 1.25rem;
            font-style: italic;
            font-size: 1.05rem;
            line-height: 1.7;
            background: var(--panel);
            border-radius: 0 0.5rem 0.5rem 0;
            margin: 1.25rem 0;
        }}
        .markdown-body a {{ color: var(--accent); text-decoration: underline; }}
        .markdown-body a:hover {{ color: #67e8f9; }}
        .markdown-body :not(pre) > code {{
            font-size: 0.925em;
            color: var(--text);
            background: var(--code);
            padding: 0.15rem 0.4rem;
            border-radius: 0.25rem;
        }}
        .markdown-body pre {{
            background: var(--code);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 1.25rem;
            margin: 1.25rem 0;
            overflow-x: auto;
            font-size: 0.925rem;
            line-height: 1.65;
        }}
        .markdown-body pre code {{
            background: transparent;
            padding: 0;
            color: var(--text);
            font-size: 0.925rem;
        }}
        .markdown-body table {{ width: 100%; font-size: 0.975rem; border-collapse: collapse; margin: 1.5rem 0; }}
        .markdown-body th, .markdown-body td {{ border: 1px solid var(--border); padding: 0.6rem 0.85rem; text-align: left; }}
        .markdown-body th {{ background: var(--panel); color: var(--accent); font-weight: 600; font-size: 1rem; }}
        .markdown-body td {{ color: var(--text); font-size: 0.975rem; }}
        .markdown-body hr {{ border: none; border-top: 1px solid var(--border); margin: 2rem 0; }}

        /* Mermaid diagram panel: readable offline as labeled source; mermaid.js
           (if vendored) replaces the <pre> contents with a rendered <svg>. */
        .diagram-panel {{
            background: var(--code);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            margin: 1.5rem 0;
            overflow: hidden;
        }}
        .diagram-panel__label {{
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: var(--accent);
            background: rgba(34, 211, 238, 0.08);
            padding: 0.4rem 0.9rem;
            border-bottom: 1px solid var(--border);
        }}
        .diagram-panel .mermaid {{
            padding: 1rem;
            display: flex;
            justify-content: center;
            overflow-x: auto;
        }}
        .diagram-panel pre.mermaid-source {{
            margin: 0;
            border: none;
            border-radius: 0;
            padding: 1rem;
            color: var(--text);
            font-size: 0.8rem;
            white-space: pre-wrap;
        }}
{platform_css}
        /* Warm Paper overrides for the learning-tools panel. */
        .platform-shell {{ background: var(--panel) !important; border-color: var(--border) !important; color: var(--text) !important; font-size: 1rem; }}
        .platform-header, .platform-turn, .paper-card {{ background: var(--panel) !important; border-color: var(--border) !important; }}
        .platform-title, .platform-label, .platform-turn strong, .platform-output, .platform-textarea,
        .platform-history, .paper-card h4, .paper-card p {{ color: var(--text) !important; }}
        .platform-section, .platform-message, .platform-status, .paper-card .paper-status {{ color: var(--terracotta) !important; }}
        .platform-textarea {{ background: var(--code) !important; border-color: var(--border) !important; }}
        .platform-button {{ color: var(--accent) !important; border-color: var(--accent) !important; background: rgba(63, 102, 82, .08) !important; }}
        .platform-button:hover:not(:disabled) {{ background: rgba(63, 102, 82, .16) !important; }}
        .platform-button:disabled {{ color: var(--text-muted) !important; border-color: var(--border) !important; background: var(--paper) !important; }}
    </style>
</head>
<body>
    <div id="content">
{nav_links}
        <article class="markdown-body">
{body}
        </article>
    </div>
{platform_panel}
{platform_js}

    <!-- MathJax is loaded synchronously after the chapter markup.  At this
         point every equation is already in the DOM, so startup reliably
         typesets the initial page as well as later modal content. -->
    <script>
        window.MathJax = {{
            tex: {{ inlineMath: [['$', '$']], displayMath: [['$$', '$$']] }},
            startup: {{ typeset: false }}
        }};
    </script>
    <script id="MathJax-script" src="{mathjax_local}"></script>
    <script>
        window.MathJax.startup.promise
            .then(function () {{ return window.MathJax.typesetPromise(); }})
            .catch(function (error) {{ console.error('MathJax typesetting failed:', error); }});
    </script>

    <!-- Diagrams: local vendor copy preferred, CDN used only as a fallback when online.
         If neither loads (fully offline, no vendor file), the labeled diagram-source
         panels above remain readable as plain text. -->
    <script>
        (function loadMermaid() {{
            function init(mermaid) {{
                mermaid.initialize({{ startOnLoad: false, theme: 'dark' }});
                document.querySelectorAll('.diagram-panel').forEach(function (panel) {{
                    var sourceEl = panel.querySelector('.mermaid-source');
                    if (!sourceEl) return;
                    var container = document.createElement('div');
                    container.className = 'mermaid';
                    container.textContent = sourceEl.textContent;
                    panel.replaceChild(container, sourceEl);
                }});
                mermaid.run();
            }}

            var local = document.createElement('script');
            local.src = '{mermaid_local}';
            local.onload = function () {{ init(window.mermaid); }};
            local.onerror = function () {{
                var cdn = document.createElement('script');
                cdn.src = 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js';
                cdn.onload = function () {{ init(window.mermaid); }};
                document.head.appendChild(cdn);
            }};
            document.head.appendChild(local);
        }})();
    </script>

    <!-- Right-Click Custom Context Menu -->
    <div id="custom-context-menu" style="display:none; position:fixed; z-index:9999999; background:#0f1524; border:1px solid #223150; border-radius:14px; padding:6px; width:220px; box-shadow:0 20px 40px rgba(0,0,0,0.8); font-family:sans-serif;">
        <button id="ctx-add-note" style="width:100%; text-align:left; background:transparent; border:none; color:#22d3ee; padding:9px 12px; font-size:12px; font-weight:700; border-radius:8px; cursor:pointer; display:flex; align-items:center; gap:8px;" onmouseover="this.style.background='rgba(34,211,238,0.15)'" onmouseout="this.style.background='transparent'">
            <span>🖊️ Add Note / Notion</span>
        </button>
        <div style="height:1px; background:#223150; margin:4px 0;"></div>
        <button id="ctx-copy" style="width:100%; text-align:left; background:transparent; border:none; color:#cbd5e1; padding:9px 12px; font-size:12px; font-weight:600; border-radius:8px; cursor:pointer; display:flex; align-items:center; gap:8px;" onmouseover="this.style.background='rgba(255,255,255,0.08)'" onmouseout="this.style.background='transparent'">
            <span>📋 Copy Selected Text</span>
        </button>
    </div>

    <!-- MS Word Minimized Comments Sidebar Toggle & Drawer -->
    <button id="word-comments-toggle" style="display:none; position:fixed; top:20px; right:20px; z-index:999980; background:#151d30; color:#eab308; border:1px solid #eab308; border-radius:9999px; padding:8px 16px; font-size:12px; font-weight:700; cursor:pointer; box-shadow:0 10px 25px rgba(0,0,0,0.5); align-items:center; gap:6px;">
        <span>💬 Comments</span>
        <span id="word-comments-badge" style="background:#eab308; color:#000000; font-size:10px; padding:1px 6px; border-radius:9999px; font-weight:800;">0</span>
    </button>

    <div id="word-comments-drawer" style="display:none; position:fixed; top:0; right:0; bottom:0; width:340px; max-width:85vw; z-index:999999; background:#0f1524; border-left:1px solid #223150; box-shadow:-10px 0 30px rgba(0,0,0,0.7); padding:20px; overflow-y:auto; font-family:sans-serif;">
        <div style="display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid #223150; padding-bottom:12px; margin-bottom:16px;">
            <span style="font-size:13px; font-weight:800; color:#eab308; display:flex; align-items:center; gap:6px;">💬 CHAPTER COMMENTS</span>
            <button id="word-comments-drawer-close" style="background:transparent; border:none; color:#cbd5e1; font-size:18px; cursor:pointer;">✕</button>
        </div>
        <div id="word-comments-list">
            <!-- Dynamically loaded chapter notes -->
        </div>
    </div>

    <!-- Floating Selection Trigger Button -->
    <button id="inline-comment-trigger" style="display:none; position:fixed; z-index:999999; background:#151d30; color:#22d3ee; border:1px solid rgba(34,211,238,0.5); border-radius:9999px; padding:7px 16px; font-size:12px; font-weight:700; box-shadow:0 10px 25px rgba(0,0,0,0.6); cursor:pointer; align-items:center; gap:6px;">
        <span>🖊️ Comment / Sync Notion</span>
    </button>

    <!-- Persistent Bottom-Right Note Button -->
    <button id="persistent-note-btn" style="position:fixed; bottom:24px; right:24px; z-index:999990; background:#151d30; color:#22d3ee; border:1px solid #223150; border-radius:9999px; padding:11px 20px; font-size:13px; font-weight:700; box-shadow:0 10px 30px rgba(0,0,0,0.7); cursor:pointer; display:flex; align-items:center; gap:8px; transition:all 0.2s ease;">
        <span>🖊️ Add Note / Notion</span>
        <span id="selection-badge" style="display:none; background:#0284c7; color:#ffffff; font-size:10px; padding:2px 7px; border-radius:9999px; text-transform:uppercase; font-weight:800; letter-spacing:0.05em;">Text Selected</span>
    </button>

    <!-- Note Card Modal with Offline Quill WYSIWYG Editor -->
    <div id="inline-comment-box" style="display:none; position:fixed; z-index:999999; top:50%; left:50%; transform:translate(-50%, -50%); background:#0f1524; border:1px solid #223150; border-radius:20px; padding:24px; width:680px; max-width:92vw; max-height:88vh; overflow-y:auto; box-shadow:0 25px 60px rgba(0,0,0,0.85); font-family:sans-serif; text-align:left;">
        <div style="display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid #223150; padding-bottom:14px; margin-bottom:16px;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-size:14px; font-weight:800; color:#22d3ee; display:flex; align-items:center; gap:6px;">🖊️ INLINE NOTE & NOTION SYNC</span>
                <span style="font-size:10px; background:rgba(34,211,238,0.15); color:#22d3ee; border:1px solid rgba(34,211,238,0.3); padding:2px 8px; border-radius:9999px; font-weight:700;">QUILL RICH TEXT</span>
            </div>
            <button id="inline-comment-close" style="background:transparent; border:none; color:#cbd5e1; font-size:18px; cursor:pointer; padding:4px 8px; border-radius:6px;">✕</button>
        </div>
        
        <div style="background:#151d30; border:1px solid #223150; padding:12px 14px; border-radius:12px; margin-bottom:16px;">
            <span style="font-size:10px; text-transform:uppercase; font-weight:700; color:#22d3ee; display:block; margin-bottom:4px;">Note Title (Selected Text)</span>
            <h4 id="inline-quote-text" style="font-size:14px; font-weight:700; color:#ffffff; margin:0; line-height:1.4;">
                <!-- Highlighted text formatted as title -->
            </h4>
        </div>

        <div style="margin-bottom:18px;">
            <label style="font-size:11px; text-transform:uppercase; font-weight:700; color:#94a3b8; display:block; margin-bottom:6px;">Your Thoughts & Rich Notes</label>
            
            <!-- Quill Editor Container -->
            <div id="quill-editor-wrapper" style="background:#0b0f19; border:1px solid #223150; border-radius:12px; overflow:hidden;">
                <div id="quill-editor" style="height:220px; font-size:14px; color:#ffffff; border:none;"></div>
            </div>
        </div>

        <div style="display:flex; align-items:center; justify-content:space-between;">
            <span id="inline-comment-status" style="font-size:12px; color:#fbbf24; font-weight:600;"></span>
            <div style="display:flex; align-items:center; gap:10px;">
                <button id="inline-comment-cancel" style="background:transparent; color:#94a3b8; border:1px solid #223150; padding:8px 16px; border-radius:10px; font-size:12px; font-weight:600; cursor:pointer;">Cancel</button>
                <button id="inline-comment-submit" style="background:linear-gradient(135deg, #0284c7, #0891b2); color:#ffffff; border:none; padding:9px 20px; border-radius:10px; font-size:13px; font-weight:700; cursor:pointer; box-shadow:0 4px 15px rgba(8,145,178,0.4);">
                    Save Note & Sync Notion
                </button>
            </div>
        </div>
    </div>

    <!-- Maximizable Day 0E Video Masterclass Modal -->
    <div id="day0e-video-modal" style="display:none; position:fixed; z-index:9999999; top:50%; left:50%; transform:translate(-50%, -50%); background:#0f1524; border:1px solid #223150; border-radius:20px; width:96vw; max-width:1400px; height:94vh; box-shadow:0 30px 80px rgba(0,0,0,0.9); font-family:sans-serif; overflow:hidden; transition:all 0.25s cubic-bezier(0.16, 1, 0.3, 1);">
        <!-- Modal Header -->
        <div id="day0e-modal-header" style="display:flex; align-items:center; justify-content:space-between; background:#151d30; border-bottom:1px solid #223150; padding:12px 20px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:14px; font-weight:800; color:#f472b6; display:flex; align-items:center; gap:6px;">🎬 DAY 0E MASTERCLASS VIDEO</span>
                <span style="font-size:10px; background:rgba(236,72,153,0.18); color:#f472b6; border:1px solid rgba(236,72,153,0.4); padding:2px 8px; border-radius:9999px; font-weight:700;">FEMALE VOICE & INTERACTIVE EXERCISES</span>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
                <button id="day0e-modal-maximize" style="background:#223150; border:1px solid rgba(255,255,255,0.1); color:#cbd5e1; font-size:12px; font-weight:700; padding:6px 14px; border-radius:8px; cursor:pointer;" onclick="toggleDay0EMaximize()">
                    <span id="max-icon">🗖 Maximize</span>
                </button>
                <button id="day0e-modal-close" style="background:transparent; border:none; color:#cbd5e1; font-size:20px; cursor:pointer; padding:4px 8px; border-radius:6px;" onclick="closeDay0EVideoModal()">✕</button>
            </div>
        </div>

        <!-- Modal Body Iframe -->
        <div style="width:100%; height:calc(100% - 54px); background:#0b0f19;">
            <iframe id="day0e-modal-iframe" src="" style="width:100%; height:100%; border:none;" title="Day 0E Masterclass Video" allow="fullscreen"></iframe>
        </div>
    </div>

    <script>
        var isDay0EMaximized = false;

        function openDay0EVideoModal() {{
            var modal = document.getElementById('day0e-video-modal');
            var iframe = document.getElementById('day0e-modal-iframe');
            if (iframe && (!iframe.src || iframe.src.indexOf('day_0e_masterclass_video') === -1)) {{
                iframe.src = 'day_0e_masterclass_video.html';
            }}
            if (modal) modal.style.display = 'block';
        }}

        function closeDay0EVideoModal() {{
            var modal = document.getElementById('day0e-video-modal');
            if ('speechSynthesis' in window) window.speechSynthesis.cancel();
            if (modal) modal.style.display = 'none';
        }}

        function toggleDay0EMaximize() {{
            var modal = document.getElementById('day0e-video-modal');
            var maxBtn = document.getElementById('max-icon');
            if (!modal) return;
            isDay0EMaximized = !isDay0EMaximized;
            if (isDay0EMaximized) {{
                modal.style.width = '100vw';
                modal.style.height = '100vh';
                modal.style.maxWidth = 'none';
                modal.style.top = '0';
                modal.style.left = '0';
                modal.style.transform = 'none';
                modal.style.borderRadius = '0';
                if (maxBtn) maxBtn.textContent = '🗗 Restore';
            }} else {{
                modal.style.width = '96vw';
                modal.style.height = '94vh';
                modal.style.maxWidth = '1400px';
                modal.style.top = '50%';
                modal.style.left = '50%';
                modal.style.transform = 'translate(-50%, -50%)';
                modal.style.borderRadius = '20px';
                if (maxBtn) maxBtn.textContent = '🗖 Maximize';
            }}
        }}

        window.openDay0EVideoModal = openDay0EVideoModal;
        window.closeDay0EVideoModal = closeDay0EVideoModal;
        window.toggleDay0EMaximize = toggleDay0EMaximize;

        document.addEventListener('keydown', function(e) {{
            var modal = document.getElementById('day0e-video-modal');
            if (!modal || modal.style.display === 'none') return;
            if (e.key === 'Escape') closeDay0EVideoModal();
            if (e.key === 'f' || e.key === 'F') toggleDay0EMaximize();
        }});

        (function() {{
            var currentSelectedText = '';
            var quillEditor = null;

            function initQuillInstance() {{
                if (quillEditor) return;
                var el = document.getElementById('quill-editor');
                if (el && typeof Quill !== 'undefined') {{
                    quillEditor = new Quill('#quill-editor', {{
                        theme: 'snow',
                        placeholder: 'Write down your notes, bullet points, key takeaways, or drop images here...',
                        modules: {{
                            toolbar: [
                                [{{ 'header': [2, 3, false] }}],
                                ['bold', 'italic', 'underline', 'strike'],
                                [{{ 'list': 'ordered'}}, {{ 'list': 'bullet' }}],
                                ['blockquote', 'code-block'],
                                ['image', 'link'],
                                ['clean']
                            ]
                        }}
                    }});
                }}
            }}

            function getSelectionText() {{
                var sel = window.getSelection();
                return sel ? sel.toString().trim() : '';
            }}

            function updateSelectionState() {{
                var text = getSelectionText();
                var trigger = document.getElementById('inline-comment-trigger');
                var badge = document.getElementById('selection-badge');
                var fab = document.getElementById('persistent-note-btn');

                if (text.length >= 1) {{
                    currentSelectedText = text;
                    if (badge) badge.style.display = 'inline-block';
                    if (fab) {{
                        fab.style.borderColor = 'rgba(34,211,238,0.8)';
                        fab.style.boxShadow = '0 0 20px rgba(34,211,238,0.4)';
                    }}
                    try {{
                        var sel = window.getSelection();
                        var range = sel ? sel.getRangeAt(0) : null;
                        var rect = range ? range.getBoundingClientRect() : null;
                        if (rect && rect.width > 0 && trigger) {{
                            trigger.style.top = Math.max(10, rect.top - 45) + 'px';
                            trigger.style.left = Math.max(10, Math.min(window.innerWidth - 220, rect.left + (rect.width / 2) - 80)) + 'px';
                            trigger.style.display = 'flex';
                        }}
                    }} catch(e) {{}}
                }} else {{
                    if (badge) badge.style.display = 'none';
                    if (fab) {{
                        fab.style.borderColor = '#223150';
                        fab.style.boxShadow = '0 10px 30px rgba(0,0,0,0.7)';
                    }}
                }}
            }}

            document.addEventListener('mouseup', function() {{ setTimeout(updateSelectionState, 50); }});
            document.addEventListener('keyup', function() {{ setTimeout(updateSelectionState, 50); }});

            document.addEventListener('contextmenu', function(e) {{
                var ctxMenu = document.getElementById('custom-context-menu');
                var text = getSelectionText();
                if (text) currentSelectedText = text;
                if (ctxMenu) {{
                    e.preventDefault();
                    ctxMenu.style.top = Math.min(window.innerHeight - 120, e.clientY) + 'px';
                    ctxMenu.style.left = Math.min(window.innerWidth - 230, e.clientX) + 'px';
                    ctxMenu.style.display = 'block';
                }}
            }});

            document.addEventListener('click', function(e) {{
                var ctxMenu = document.getElementById('custom-context-menu');
                if (ctxMenu && !ctxMenu.contains(e.target)) ctxMenu.style.display = 'none';
            }});

            function openNoteModal(e) {{
                if (e && e.stopPropagation) e.stopPropagation();
                var box = document.getElementById('inline-comment-box');
                var quote = document.getElementById('inline-quote-text');
                var status = document.getElementById('inline-comment-status');
                var trigger = document.getElementById('inline-comment-trigger');
                var ctxMenu = document.getElementById('custom-context-menu');

                if (trigger) trigger.style.display = 'none';
                if (ctxMenu) ctxMenu.style.display = 'none';
                if (quote) quote.textContent = currentSelectedText || 'General Chapter Note';
                if (status) status.textContent = '';
                if (box) {{
                    box.style.display = 'block';
                    initQuillInstance();
                    if (quillEditor) quillEditor.focus();
                }}
            }}

            window.openNoteModal = openNoteModal;

            var triggerBtn = document.getElementById('inline-comment-trigger');
            var fabBtn = document.getElementById('persistent-note-btn');
            var ctxBtn = document.getElementById('ctx-add-note');
            var ctxCopy = document.getElementById('ctx-copy');
            var closeBtn = document.getElementById('inline-comment-close');
            var cancelBtn = document.getElementById('inline-comment-cancel');
            var submitBtn = document.getElementById('inline-comment-submit');

            if (triggerBtn) triggerBtn.onclick = openNoteModal;
            if (fabBtn) fabBtn.onclick = openNoteModal;
            if (ctxBtn) ctxBtn.onclick = openNoteModal;
            if (ctxCopy) ctxCopy.onclick = function() {{
                var ctxMenu = document.getElementById('custom-context-menu');
                if (ctxMenu) ctxMenu.style.display = 'none';
                if (currentSelectedText) navigator.clipboard.writeText(currentSelectedText);
            }};
            if (closeBtn) closeBtn.onclick = function() {{
                var box = document.getElementById('inline-comment-box');
                if (box) box.style.display = 'none';
            }};
            if (cancelBtn) cancelBtn.onclick = function() {{
                var box = document.getElementById('inline-comment-box');
                if (box) box.style.display = 'none';
            }};

            if (submitBtn) submitBtn.onclick = async function() {{
                var status = document.getElementById('inline-comment-status');
                var htmlComment = quillEditor ? quillEditor.root.innerHTML : '';
                var textComment = quillEditor ? quillEditor.getText().trim() : '';

                if (!textComment && !currentSelectedText && (!htmlComment || htmlComment === '<p><br></p>')) return;
                if (status) status.textContent = 'Saving note & syncing to Notion...';

                try {{
                    var resp = await fetch('/api/notion/comment', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            title: currentSelectedText || (textComment ? textComment.slice(0, 60) : "Study Note"),
                            selection_text: currentSelectedText,
                            comment: htmlComment,
                            chapter_slug: window.location.pathname.split('/').pop() || 'chapter'
                        }})
                    }});
                    var res = await resp.json();
                    if (res.ok) {{
                        if (status) {{
                            status.style.color = '#34d399';
                            status.textContent = (res.notion && res.notion.ok) ? 'Pushed to Notion & Local DB!' : 'Saved to Local Database!';
                        }}
                        setTimeout(function() {{
                            var box = document.getElementById('inline-comment-box');
                            if (box) box.style.display = 'none';
                            if (quillEditor) quillEditor.setContents([]);
                            loadSavedChapterNotes();
                        }}, 1200);
                    }}
                }} catch (err) {{
                    if (status) {{
                        status.style.color = '#fbbf24';
                        status.textContent = 'Saved to Local Session!';
                    }}
                    setTimeout(function() {{
                        var box = document.getElementById('inline-comment-box');
                        if (box) box.style.display = 'none';
                    }}, 1200);
                }}
            }};

            async function loadSavedChapterNotes() {{
                try {{
                    var slug = window.location.pathname.split('/').pop() || 'chapter';
                    var resp = await fetch('/api/notes?chapter_slug=' + encodeURIComponent(slug));
                    var data = await resp.json();
                    var notes = data.notes || [];

                    var toggleBtn = document.getElementById('word-comments-toggle');
                    var badge = document.getElementById('word-comments-badge');
                    var list = document.getElementById('word-comments-list');

                    if (badge) badge.textContent = notes.length;
                    if (toggleBtn) toggleBtn.style.display = notes.length > 0 ? 'inline-flex' : 'none';

                    if (list) {{
                        if (notes.length === 0) {{
                            list.innerHTML = '<div style="color:#94a3b8; font-size:12px; text-align:center; padding:20px;">No comments yet. Highlight text and right-click to add a note!</div>';
                        }} else {{
                            list.innerHTML = notes.map(function(n) {{
                                var title = n.title || 'Note';
                                var body = n.body ? n.body : '';
                                return '<div style="background:#151d30; border:1px solid #223150; border-radius:12px; padding:14px; margin-bottom:12px;">' +
                                    '<div style="font-size:10px; font-weight:800; color:#eab308; text-transform:uppercase; margin-bottom:4px;">💬 Comment</div>' +
                                    '<h4 style="font-size:13px; font-weight:700; color:#ffffff; margin:0 0 6px 0; line-height:1.4;">' + escapeHtml(title) + '</h4>' +
                                    (body ? '<p style="font-size:12px; color:#cbd5e1; margin:0; line-height:1.5;">' + escapeHtml(body) + '</p>' : '') +
                                    '<div style="font-size:10px; color:#64748b; margin-top:8px; display:flex; justify-content:space-between; align-items:center;">' +
                                        '<span>' + (n.created_at || '') + '</span>' +
                                        '<span style="color:#34d399; font-weight:700;">Notion Synced</span>' +
                                    '</div>' +
                                '</div>';
                            }}).join('');
                        }}
                    }}

                    notes.forEach(function(note) {{
                        if (!note.title || note.title === 'Study Note' || note.title.length < 3) return;
                        highlightTextSnippet(note.title, note);
                    }});
                }} catch(e) {{}}
            }}

            function highlightTextSnippet(searchText, note) {{
                try {{
                    var body = document.querySelector('.markdown-body') || document.body;
                    if (!body) return;
                    var walker = document.createTreeWalker(body, NodeFilter.SHOW_TEXT, null, false);
                    var node;
                    while (node = walker.nextNode()) {{
                        if (node.parentElement && node.parentElement.closest('.word-comment-highlight, script, style, textarea, button')) continue;
                        var idx = node.nodeValue ? node.nodeValue.toLowerCase().indexOf(searchText.toLowerCase().slice(0, 30)) : -1;
                        if (idx !== -1) {{
                            var matchedText = node.nodeValue.substr(idx, searchText.length);
                            var mark = document.createElement('mark');
                            mark.className = 'word-comment-highlight';
                            mark.style.cssText = 'background:rgba(234,179,8,0.25); border-bottom:2px solid #eab308; border-radius:4px; padding:1px 4px; color:inherit; cursor:pointer;';
                            mark.innerHTML = escapeHtml(matchedText) + '<span style="display:inline-flex; align-items:center; justify-content:center; background:#eab308; color:#000; font-size:10px; font-weight:800; border-radius:9999px; width:16px; height:16px; margin-left:4px; vertical-align:middle;" title="MS Word Comment">💬</span>';
                            mark.addEventListener('click', function(e) {{
                                e.stopPropagation();
                                var drawer = document.getElementById('word-comments-drawer');
                                if (drawer) drawer.style.display = 'block';
                            }});
                            var parent = node.parentNode;
                            var after = node.splitText(idx);
                            after.nodeValue = after.nodeValue.substr(searchText.length);
                            parent.insertBefore(mark, after);
                            break;
                        }}
                    }}
                }} catch(err) {{}}
            }}

            function escapeHtml(val) {{
                return String(val || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
            }}

            var wordToggle = document.getElementById('word-comments-toggle');
            var wordClose = document.getElementById('word-comments-drawer-close');

            if (wordToggle) wordToggle.onclick = function() {{
                var drawer = document.getElementById('word-comments-drawer');
                if (drawer) drawer.style.display = drawer.style.display === 'none' ? 'block' : 'none';
            }};

            if (wordClose) wordClose.onclick = function() {{
                var drawer = document.getElementById('word-comments-drawer');
                if (drawer) drawer.style.display = 'none';
            }};

            window.addEventListener('load', loadSavedChapterNotes);
        }})();
    </script>
</body>
</html>
"""

NAV_LINK = '            <a href="{href}" class="{classes}">{label}</a>'


def build_nav(current_slug: str) -> str:
    # 1. Main book chapters navigation group
    chapter_links = []
    for slug in CHAPTER_ORDER:
        if slug == "level_0_orientation":
            label = "Level 0 Orientation"
        elif slug == "calculus_derivatives_micro_videos":
            label = "🎥 Calculus Micro-Videos"
        elif slug == "day_0e_masterclass_video":
            label = "🎬 Day 0E Masterclass Video"
        else:
            label = slug.replace("chapter", "Ch. ").replace("_5_interlude", ".5 Interlude").replace("_", " ")
        if slug == "chapter1":
            readiness_active = READINESS_SLUG == current_slug
            readiness_classes = "active" if readiness_active else ""
            chapter_links.append(
                NAV_LINK.format(
                    href=f"{READINESS_SLUG}.html",
                    classes=readiness_classes,
                    label="Read this before jumping into Chapter 1",
                )
            )
            exercise_active = EXERCISE_SLUG == current_slug
            exercise_classes = "active" if exercise_active else ""
            chapter_links.append(
                NAV_LINK.format(
                    href=f"{EXERCISE_SLUG}.html",
                    classes=exercise_classes,
                    label="Chapter 1 exercise notebook",
                )
            )
        if slug == "chapter2":
            help_active = CHAPTER2_HELP_SLUG == current_slug
            help_classes = "active" if help_active else ""
            chapter_links.append(
                NAV_LINK.format(
                    href=f"{CHAPTER2_HELP_SLUG}.html",
                    classes=help_classes,
                    label="Chapter 2 concept help (open only when stuck)",
                )
            )
        if slug == "chapter3":
            help_active = CHAPTER3_HELP_SLUG == current_slug
            help_classes = "active" if help_active else ""
            chapter_links.append(
                NAV_LINK.format(
                    href=f"{CHAPTER3_HELP_SLUG}.html",
                    classes=help_classes,
                    label="Chapter 3 concept help (open only when stuck)",
                )
            )
        if slug == "chapter4":
            help_active = CHAPTER4_HELP_SLUG == current_slug
            help_classes = "active" if help_active else ""
            chapter_links.append(
                NAV_LINK.format(
                    href=f"{CHAPTER4_HELP_SLUG}.html",
                    classes=help_classes,
                    label="Chapter 4 concept help (open only when stuck)",
                )
            )
        if slug == "chapter5":
            help_active = CHAPTER5_HELP_SLUG == current_slug
            help_classes = "active" if help_active else ""
            chapter_links.append(
                NAV_LINK.format(
                    href=f"{CHAPTER5_HELP_SLUG}.html",
                    classes=help_classes,
                    label="Chapter 5 concept help (open only when stuck)",
                )
            )
        if slug == "chapter6":
            help_active = CHAPTER6_HELP_SLUG == current_slug
            help_classes = "active" if help_active else ""
            chapter_links.append(
                NAV_LINK.format(
                    href=f"{CHAPTER6_HELP_SLUG}.html",
                    classes=help_classes,
                    label="Chapter 6 concept help (open only when stuck)",
                )
            )
        active = slug == current_slug
        classes = "active" if active else ""
        chapter_links.append(NAV_LINK.format(href=f"{slug}.html", classes=classes, label=label))
    video_active = CALCULUS_VIDEO_SLUG == current_slug
    chapter_links.append(
        NAV_LINK.format(
            href=f"{CALCULUS_VIDEO_SLUG}.html",
            classes="active" if video_active else "",
            label="Calculus video lessons",
        )
    )
    chapter_links.append(NAV_LINK.format(href="settings.html", classes="active" if current_slug == "settings" else "", label="Settings"))
    
    # 2. Companion guides navigation group
    companion_links = []
    companion_order = ["chapter0_companion", "chapter1_companion", "chapter2_companion", "chapter3_companion", "chapter4_companion", "chapter5_companion", "chapter6_companion"]
    for slug in companion_order:
        label = slug.replace("chapter", "Ch. ").replace("_companion", " Companion")
        active = slug == current_slug
        classes = "active" if active else ""
        companion_links.append(NAV_LINK.format(href=f"{slug}.html", classes=classes, label=label))
        
    joined_chapters = "\n".join(chapter_links)
    joined_companions = "\n".join(companion_links)
    
    nav_html = f'''
    <aside class="chapter-sidebar">
        <div class="sidebar-brand">
            <div class="sidebar-brand__eyebrow">Regression Study Guide</div>
            <div class="sidebar-brand__title">Navigation</div>
            <div class="sidebar-brand__subtitle">Jump between chapters, companion guides, and study aids.</div>
        </div>
        <section class="sidebar-section">
            <div class="sidebar-section-title">Book Chapters</div>
            <nav class="chapters">
                {joined_chapters}
            </nav>
        </section>
        <section class="sidebar-section">
            <div class="sidebar-section-title">Companion Guides</div>
            <nav class="chapters">
                {joined_companions}
            </nav>
        </section>
    </aside>
    '''
    return nav_html


def title_from_markdown(text: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else fallback


def convert_mermaid_fences(md_text: str) -> str:
    """Swap ```mermaid fences for a labeled diagram panel before markdown conversion.

    Renders as readable plain text with no JS. If mermaid.js loads (local vendor
    file or CDN), loadMermaid() in the template swaps the <pre> for a live <div
    class="mermaid"> and re-renders it as an SVG diagram.
    """

    def replacer(match: "re.Match[str]") -> str:
        diagram = match.group(1)
        return (
            '<div class="diagram-panel">'
            '<div class="diagram-panel__label">Diagram</div>'
            f'<pre class="mermaid-source">{diagram}</pre>'
            '</div>'
        )

    return re.sub(r"```mermaid\n(.*?)```", replacer, md_text, flags=re.DOTALL)


def clean_control_characters(text: str) -> str:
    return text.replace("\x08", "\\b").replace("\x0c", "\\f").replace("\x07", "\\a")


def render_markdown(md_text: str) -> str:
    """Render Markdown without letting it reinterpret display-math syntax.

    Python-Markdown does not understand MathJax delimiters.  In a block such
    as ``$$\\hat{y}_i=f(x_{i1}, ...)$$`` it can therefore turn underscores
    into ``<em>`` tags before MathJax runs.  Replace display equations with
    inert tokens during Markdown conversion, then restore escaped TeX into the
    generated HTML for MathJax to typeset in the browser.
    """
    # Python-Markdown deliberately leaves raw HTML blocks untouched.  That is
    # useful for the ``<details>`` wrapper used by the concept-help pages, but
    # it also means Markdown inside an expanded card would be shown literally.
    # Render each card body first, then let the outer pass handle the document.
    def render_detail_body(match: "re.Match[str]") -> str:
        summary = match.group(1)
        body = match.group(2)
        return f"<details>\n<summary>{summary}</summary>\n{render_markdown(body)}\n</details>"

    if re.search(r"<details>.*?</details>", md_text, flags=re.DOTALL):
        md_text = re.sub(
            r"<details>\s*<summary>(.*?)</summary>(.*?)</details>",
            render_detail_body,
            md_text,
            flags=re.DOTALL,
        )

    math_blocks: list[str] = []

    def stash(match: "re.Match[str]") -> str:
        token = f"CODEXDISPLAYMATH{len(math_blocks):06d}PLACEHOLDER"
        math_blocks.append("\n".join(line.rstrip() for line in match.group(0).splitlines()))
        return f"\n\n{token}\n\n"

    protected = re.sub(r"(?<!\\)\$\$(.*?)(?<!\\)\$\$", stash, md_text, flags=re.DOTALL)
    rendered = markdown.markdown(
        protected,
        extensions=["fenced_code", "tables", "toc", "sane_lists", "attr_list", "nl2br"],
    )
    for index, math_block in enumerate(math_blocks):
        token = f"CODEXDISPLAYMATH{index:06d}PLACEHOLDER"
        rendered = rendered.replace(token, html.escape(math_block, quote=False))
    return rendered


def strip_legacy_companion_markup(md_text: str) -> str:
    """Remove the old hand-copied modal implementation from a chapter.

    Chapters 1–3 historically copied a small companion summary, its CSS, and
    its JavaScript into the end of each Markdown file.  The compiled modal now
    reads the maintained companion guide instead, so retaining that block would
    duplicate content and register two competing click handlers.
    """
    md_text = re.sub(
        r'^\s*<button class="read-details-btn"[^>]*>.*?</button>\s*$',
        "",
        md_text,
        flags=re.MULTILINE,
    )
    legacy_style = re.search(r"^<style>\s*\n\.read-details-btn\s*\{", md_text, re.MULTILINE)
    if legacy_style:
        md_text = md_text[: legacy_style.start()].rstrip() + "\n"
    return md_text


def companion_sections(md_text: str) -> dict[str, str]:
    """Split a companion guide into overview, day, capstone, and reference parts."""
    headings = list(re.finditer(r"^#\s+(.+?)\s*$", md_text, re.MULTILINE))
    if not headings:
        return {"overview": md_text}

    chunks: list[tuple[str, str]] = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(md_text)
        chunks.append((heading.group(1).strip(), md_text[heading.start():end].strip()))

    first_day = next(
        (index for index, (title, _) in enumerate(chunks) if re.match(r"Day\s+\d+\b", title, re.I)),
        len(chunks),
    )
    # Chapter 0's maintained guide predates the others and uses numbered H2
    # headings such as "2. Day 0A" rather than H1 day headings.
    if first_day == len(chunks):
        chapter_zero_headings = list(re.finditer(
            r"^##\s+(?:\d+\.\s+)?(?:Day\s+0[A-E]\b|Level\s+0\s+Capstone\b|Master\s+Rosetta\b).*$",
            md_text,
            re.MULTILINE | re.IGNORECASE,
        ))
        if chapter_zero_headings:
            sections = {"overview": md_text[:chapter_zero_headings[0].start()].strip()}
            for index, heading in enumerate(chapter_zero_headings):
                end = chapter_zero_headings[index + 1].start() if index + 1 < len(chapter_zero_headings) else len(md_text)
                title = heading.group(0)
                chunk = md_text[heading.start():end].strip()
                day = re.search(r"Day\s+(0[A-E])\b", title, re.I)
                if day:
                    sections[f"day-{day.group(1).lower()}"] = chunk
                elif "capstone" in title.lower():
                    sections["capstone"] = chunk
                else:
                    sections["reference"] = chunk
            return {key: value for key, value in sections.items() if value.strip()}
    sections: dict[str, str] = {
        "overview": "\n\n".join(chunk for _, chunk in chunks[:first_day])
    }
    reference_chunks: list[str] = []

    for title, chunk in chunks[first_day:]:
        day = re.match(r"Day\s+(\d+)\b", title, re.I)
        if day:
            sections[f"day-{day.group(1)}"] = chunk
        elif "capstone" in title.lower():
            sections["capstone"] = chunk
        else:
            reference_chunks.append(chunk)

    if reference_chunks:
        sections["reference"] = "\n\n".join(reference_chunks)
    return {key: value for key, value in sections.items() if value.strip()}


def add_companion_buttons(md_text: str, available: set[str]) -> str:
    """Add contextual modal launchers after chapter-level Markdown headings."""
    if "overview" in available:
        md_text = re.sub(
            r"^(#\s+.+)$",
            r'\1\n\n<button type="button" class="companion-more-button" '
            r'data-companion-section="overview">✦ Read the beginner companion overview</button>',
            md_text,
            count=1,
            flags=re.MULTILINE,
        )

    def add_day_button(match: "re.Match[str]") -> str:
        heading, day = match.group(1), match.group(2)
        key = f"day-{day.lower()}"
        if key not in available:
            return heading
        return (
            f'{heading}\n\n<button type="button" class="companion-more-button" '
            f'data-companion-section="{key}">✦ Read the beginner companion for Day {day}</button>'
        )

    md_text = re.sub(
        r"^(#\s+Day\s+(\d+[A-Z]?)\s+.+)$",
        add_day_button,
        md_text,
        flags=re.MULTILINE,
    )

    if "capstone" in available:
        md_text = re.sub(
            r"^(#{1,3}\s+[^\n]*Capstone[^\n]*)$",
            r'\1\n\n<button type="button" class="companion-more-button" '
            r'data-companion-section="capstone">✦ Read the beginner capstone companion</button>',
            md_text,
            count=1,
            flags=re.MULTILINE | re.IGNORECASE,
        )

    if "reference" in available:
        reference_heading = re.compile(
            r"^(#\s+(?:Formula\s+sheet|Glossary|Chapter\s+\d+\s+Synthesis).*)$",
            re.MULTILINE | re.IGNORECASE,
        )
        md_text = reference_heading.sub(
            r'\1\n\n<button type="button" class="companion-more-button" '
            r'data-companion-section="reference">✦ Open the companion reference material</button>',
            md_text,
            count=1,
        )
    return md_text


def build_companion_modal(slug: str) -> tuple[str, set[str]]:
    """Compile a chapter's maintained companion guide into hidden templates."""
    filename = COMPANION_FILES.get(slug)
    if not filename:
        return "", set()
    companion_path = SRC_DIR / filename
    if not companion_path.exists():
        return "", set()

    guide_text = clean_control_characters(companion_path.read_text(encoding="utf-8"))
    sections = companion_sections(guide_text)
    templates = []
    for key, section_md in sections.items():
        section_html = render_markdown(convert_mermaid_fences(section_md))
        templates.append(
            f'<template id="companion-template-{key}">{section_html}</template>'
        )

    modal = f"""
<style>
.companion-more-button {{
    display: inline-flex; align-items: center; gap: .45rem;
    margin: .5rem 0 1.25rem; padding: .55rem .85rem;
    border: 1px solid rgba(34, 211, 238, .42); border-radius: .45rem;
    background: rgba(34, 211, 238, .09); color: #67e8f9;
    font: inherit; font-size: .82rem; font-weight: 700; cursor: pointer;
}}
.companion-more-button:hover, .companion-more-button:focus-visible {{
    background: rgba(34, 211, 238, .18); border-color: #22d3ee;
}}
.companion-more-modal[hidden] {{ display: none; }}
.companion-more-modal {{
    position: fixed; inset: 0; z-index: 10000; display: grid; place-items: center;
    padding: 1rem; background: rgba(5, 9, 17, .88); backdrop-filter: blur(7px);
}}
.companion-more-dialog {{
    width: min(64rem, 96vw); max-height: 92vh; display: flex; flex-direction: column;
    overflow: hidden; border: 1px solid #334155; border-radius: .8rem;
    background: #111827; box-shadow: 0 24px 70px rgba(0, 0, 0, .55);
}}
.companion-more-header {{
    display: flex; align-items: center; justify-content: space-between; gap: 1rem;
    padding: .8rem 1rem; border-bottom: 1px solid #334155; background: #172033;
}}
.companion-more-header strong {{ color: #fff; }}
.companion-more-actions {{ display: flex; align-items: center; gap: .75rem; }}
.companion-more-actions a {{ color: #67e8f9; font-size: .82rem; }}
.companion-more-close {{
    width: 2.1rem; height: 2.1rem; border: 1px solid #475569; border-radius: .4rem;
    background: #0b1220; color: #e2e8f0; font-size: 1.35rem; cursor: pointer;
}}
.companion-more-body {{ overflow: auto; padding: 1.25rem 1.5rem 2rem; }}
.companion-more-body > :first-child {{ margin-top: 0; }}
.companion-more-body table {{ display: block; overflow-x: auto; }}
.companion-more-body pre {{ max-width: 100%; overflow-x: auto; }}
@media (max-width: 640px) {{
    .companion-more-modal {{ padding: 0; }}
    .companion-more-dialog {{ width: 100vw; max-height: 100vh; height: 100vh; border-radius: 0; }}
    .companion-more-body {{ padding: 1rem; }}
    .companion-more-actions a {{ display: none; }}
}}
</style>
<div class="companion-more-modal" id="companion-more-modal" hidden>
  <section class="companion-more-dialog" role="dialog" aria-modal="true" aria-labelledby="companion-more-title">
    <header class="companion-more-header">
      <strong id="companion-more-title">Beginner companion</strong>
      <div class="companion-more-actions">
        <a href="{slug}_companion.html">Open the complete guide</a>
        <button type="button" class="companion-more-close" aria-label="Close companion">&times;</button>
      </div>
    </header>
    <div class="companion-more-body" id="companion-more-body"></div>
  </section>
</div>
{''.join(templates)}
<script>
(function () {{
    var modal = document.getElementById('companion-more-modal');
    var body = document.getElementById('companion-more-body');
    var close = modal && modal.querySelector('.companion-more-close');
    var previousFocus = null;
    if (!modal || !body || !close) return;

    function closeModal() {{
        modal.hidden = true;
        body.replaceChildren();
        document.body.style.overflow = '';
        if (previousFocus) previousFocus.focus();
    }}

    document.querySelectorAll('.companion-more-button').forEach(function (button) {{
        button.addEventListener('click', function () {{
            var key = button.getAttribute('data-companion-section');
            var template = document.getElementById('companion-template-' + key);
            if (!template) return;
            previousFocus = button;
            body.replaceChildren(template.content.cloneNode(true));
            modal.hidden = false;
            document.body.style.overflow = 'hidden';
            close.focus();
            if (window.MathJax && window.MathJax.typesetPromise) {{
                window.MathJax.typesetPromise([body]);
            }}
        }});
    }});
    close.addEventListener('click', closeModal);
    modal.addEventListener('click', function (event) {{
        if (event.target === modal) closeModal();
    }});
    document.addEventListener('keydown', function (event) {{
        if (event.key === 'Escape' && !modal.hidden) closeModal();
    }});
    var requestedSection = new URLSearchParams(window.location.search).get('companion');
    if (requestedSection) {{
        var requestedButton = Array.from(
            document.querySelectorAll('.companion-more-button')
        ).find(function (button) {{
            return button.getAttribute('data-companion-section') === requestedSection;
        }});
        if (requestedButton) requestedButton.click();
    }}
}})();
</script>
"""
    return modal, set(sections)


def compile_chapter(slug: str) -> None:
    src_path = SRC_DIR / f"{slug}.md"
    if not src_path.exists():
        print(f"Warning: {src_path.name} not found, skipping.")
        return

    md_text = src_path.read_text(encoding="utf-8")
    md_text = clean_control_characters(md_text)
    companion_modal, companion_keys = ("", set())
    if slug in EMBEDDED_COMPANION_CHAPTERS:
        companion_modal, companion_keys = build_companion_modal(slug)
    if companion_modal:
        md_text = strip_legacy_companion_markup(md_text)
        md_text = add_companion_buttons(md_text, companion_keys)
    title = title_from_markdown(md_text, fallback=slug)
    md_text = convert_mermaid_fences(md_text)

    body_html = render_markdown(md_text)
    body_html += companion_modal

    html = TEMPLATE.format(
        title=title,
        nav_links=build_nav(slug),
        body=body_html,
        mermaid_local=VENDOR_MERMAID_REL,
        mathjax_local=VENDOR_MATHJAX_REL,
        platform_css=PLATFORM_CSS,
        platform_panel=PLATFORM_PANEL,
        platform_js=PLATFORM_JS.replace("{chapter_slug}", slug),
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{slug}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Compiled {src_path.name} -> {out_path.relative_to(ROOT)}")


def compile_companion(slug: str, filename: str) -> None:
    src_path = SRC_DIR / filename
    if not src_path.exists():
        print(f"Warning: {src_path.name} not found, skipping.")
        return

    md_text = src_path.read_text(encoding="utf-8")
    md_text = clean_control_characters(md_text)
    title = title_from_markdown(md_text, fallback=slug)
    md_text = convert_mermaid_fences(md_text)

    body_html = render_markdown(md_text)

    html = TEMPLATE.format(
        title=title,
        nav_links=build_nav(slug),
        body=body_html,
        mermaid_local=VENDOR_MERMAID_REL,
        mathjax_local=VENDOR_MATHJAX_REL,
        platform_css=PLATFORM_CSS,
        platform_panel="",
        platform_js="",
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{slug}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Compiled {src_path.name} -> {out_path.relative_to(ROOT)}")


def compile_settings_page() -> None:
    html = TEMPLATE.format(
        title="Settings",
        nav_links=build_nav("settings"),
        body=SETTINGS_BODY,
        mermaid_local=VENDOR_MERMAID_REL,
        mathjax_local=VENDOR_MATHJAX_REL,
        platform_css=PLATFORM_CSS,
        platform_panel=PLATFORM_PANEL,
        platform_js=PLATFORM_JS.replace("{chapter_slug}", "settings"),
    )
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "settings.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Compiled settings page -> {out_path.relative_to(ROOT)}")


def compile_day0e_masterclass_page() -> None:
    """Compile SRC/day_0e_masterclass_video.md into standalone video overlay SRC_HTML/day_0e_masterclass_video.html."""
    src_file = SRC_DIR / "day_0e_masterclass_video.md"
    if src_file.exists():
        raw = src_file.read_text(encoding="utf-8")
        body = "<style>#content { margin: 0 !important; padding: 0 !important; max-width: 100% !important; } .chapter-sidebar, aside { display: none !important; }</style>\n" + render_markdown(raw)
    else:
        body = "<h1>Day 0E Masterclass</h1>"
    html = TEMPLATE.format(title="Day 0E Masterclass Video Player", nav_links="", body=body, mermaid_local=VENDOR_MERMAID_REL, mathjax_local=VENDOR_MATHJAX_REL, platform_css=PLATFORM_CSS, platform_panel="", platform_js="")
    out_path = OUTPUT_DIR / "day_0e_masterclass_video.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Compiled standalone Day 0E video overlay -> {out_path.relative_to(ROOT)}")


def ensure_vendor_readme() -> None:
    VENDOR_DIR.mkdir(exist_ok=True)
    readme = VENDOR_DIR / "README.md"
    if readme.exists():
        return
    readme.write_text(
        "# Vendor assets for fully offline SRC_HTML pages\n\n"
        "Compiled pages in SRC_HTML/ work offline already (no Tailwind CDN, no\n"
        "Google Fonts). Diagram source (```mermaid fences) is shown as readable\n"
        "plain text, and math ($...$) is shown as raw LaTeX, with no JS required.\n\n"
        "To additionally get live-rendered mermaid diagrams and typeset math\n"
        "while fully offline, drop these two files in this directory:\n\n"
        "- `vendor/mermaid.min.js` — from https://www.jsdelivr.com/package/npm/mermaid\n"
        "  (dist/mermaid.min.js)\n"
        "- `vendor/mathjax/tex-svg-full.js` — pinned MathJax 3.2.2 full TeX/SVG bundle from\n"
        "  https://www.jsdelivr.com/package/npm/mathjax (es5/tex-svg-full.js)\n\n"
        "The compiled pages check for these local files first and only reach out\n"
        "to a CDN as a fallback if they're missing and the machine has network\n"
        "access. Re-run `python3 compile_src.py` after adding the files (or just\n"
        "refresh the page — the paths are already wired in).\n",
        encoding="utf-8",
    )


def main() -> None:
    print("Compiling SRC/ markdown chapters into SRC_HTML/...")
    ensure_vendor_readme()
    for slug in CHAPTER_ORDER:
        compile_chapter(slug)
    compile_companion(READINESS_SLUG, READINESS_FILE)
    compile_companion(EXERCISE_SLUG, EXERCISE_FILE)
    compile_companion(CHAPTER2_HELP_SLUG, CHAPTER2_HELP_FILE)
    compile_companion(CHAPTER3_HELP_SLUG, CHAPTER3_HELP_FILE)
    compile_companion(CHAPTER4_HELP_SLUG, CHAPTER4_HELP_FILE)
    compile_companion(CHAPTER5_HELP_SLUG, CHAPTER5_HELP_FILE)
    compile_companion(CHAPTER6_HELP_SLUG, CHAPTER6_HELP_FILE)
    compile_companion(CALCULUS_VIDEO_SLUG, CALCULUS_VIDEO_FILE)
    # Keep the video assets under STATIC_ROOT so the application can serve
    # them with the same ordinary relative URLs as its HTML pages.
    video_source = SRC_DIR / "calculus_videos" / "mp4"
    video_target = OUTPUT_DIR / "calculus_videos" / "mp4"
    if video_source.exists():
        video_target.mkdir(parents=True, exist_ok=True)
        for video_file in video_source.glob("*.mp4"):
            shutil.copy2(video_file, video_target / video_file.name)
    day0e_source = SRC_DIR / "day0e" / "video"
    day0e_target = OUTPUT_DIR / "day0e_video"
    if day0e_source.exists():
        if day0e_target.exists():
            shutil.rmtree(day0e_target)
        shutil.copytree(day0e_source, day0e_target)
        deck_pdf = SRC_DIR / "day0e" / "deck" / "day0e_deck.pdf"
        if deck_pdf.exists():
            shutil.copy2(deck_pdf, day0e_target / "day0e_deck.pdf")
    compile_day0e_masterclass_page()
    
    for chapter_slug, filename in COMPANION_FILES.items():
        compile_companion(f"{chapter_slug}_companion", filename)
    compile_settings_page()
    print("SRC markdown compilation complete.")


if __name__ == "__main__":
    main()
