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
EMBEDDED_COMPANION_CHAPTERS = {
    "chapter1", "chapter2", "chapter3", "chapter4", "chapter5", "chapter6"
}

# Local vendor paths (relative to SRC_HTML/*.html) checked before falling
# back to a CDN. Drop the real files in vendor/ to go fully offline.
VENDOR_MERMAID_REL = "../vendor/mermaid.min.js"
VENDOR_MATHJAX_REL = "../vendor/mathjax/tex-mml-svg.js"
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
        .platform-title { font-size: 0.85rem; font-weight: 700; color: #fff; }
        .platform-section { font-size: 0.72rem; color: var(--text-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .platform-body { overflow-y: auto; padding: 0.85rem; }
        .platform-group { border-top: 1px solid var(--border); padding-top: 0.85rem; margin-top: 0.85rem; }
        .platform-group:first-child { border-top: 0; padding-top: 0; margin-top: 0; }
        .platform-row { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
        .platform-label { display: block; margin-bottom: 0.35rem; font-size: 0.72rem; font-weight: 700; color: #fff; text-transform: uppercase; letter-spacing: 0.04em; }
        .platform-button {
            border: 1px solid rgba(34, 211, 238, 0.45);
            background: rgba(34, 211, 238, 0.08);
            color: var(--accent);
            border-radius: 0.35rem;
            padding: 0.4rem 0.6rem;
            font-size: 0.78rem;
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
            min-height: 4.5rem;
            resize: vertical;
            border: 1px solid var(--border);
            border-radius: 0.35rem;
            background: #080c14;
            color: var(--text);
            padding: 0.55rem;
            font: inherit;
            font-size: 0.85rem;
        }
        .platform-message {
            margin-top: 0.45rem;
            color: #fbbf24;
            font-size: 0.78rem;
        }
        .platform-output {
            margin-top: 0.55rem;
            white-space: pre-wrap;
            color: var(--text-muted);
            font-size: 0.84rem;
        }
        .platform-history {
            max-height: 12rem;
            overflow: auto;
            display: grid;
            gap: 0.45rem;
            margin: 0.55rem 0;
        }
        .platform-turn {
            border: 1px solid var(--border);
            border-radius: 0.35rem;
            padding: 0.45rem 0.55rem;
            background: rgba(21, 29, 48, 0.55);
            color: var(--text-muted);
            font-size: 0.82rem;
        }
        .platform-turn strong { color: #fff; }
        .paper-card {
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 0.8rem;
            margin: 1rem 0;
            background: rgba(21, 29, 48, 0.55);
        }
        .paper-card h4 { margin: 0 0 0.3rem; font-size: 1rem; }
        .paper-card p { margin: 0.35rem 0; font-size: 0.9rem; }
        .paper-card .paper-meta { color: #93c5fd; font-size: 0.82rem; }
        .paper-card .paper-status { color: #fbbf24; font-size: 0.8rem; }
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
    <style>
        :root {{
            --bg: #0b0f19;
            --panel: #151d30;
            --border: #223150;
            --text: #f1f5f9;
            --text-muted: #cbd5e1;
            --accent: #22d3ee;
        }}
        * {{ box-sizing: border-box; }}
        html, body {{ margin: 0; padding: 0; overflow-x: hidden; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }}
        code, pre, .mono {{
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        }}
        #content {{
            width: 100%;
            max-width: 56rem;
            margin: 0 auto;
            padding: 3rem 1.5rem;
        }}
        nav.chapters {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 2.5rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
        }}
        nav.chapters a {{
            padding: 0.375rem 0.75rem;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            border-radius: 999px;
            border: 1px solid var(--border);
            color: var(--text-muted);
            text-decoration: none;
        }}
        nav.chapters a:hover {{ border-color: rgba(34, 211, 238, 0.4); }}
        nav.chapters a.active {{
            color: var(--accent);
            background: rgba(21, 29, 48, 0.8);
            border-color: rgba(34, 211, 238, 0.4);
        }}
        .markdown-body h1 {{ font-size: 2.25rem; font-weight: 800; color: #fff; margin: 0.5rem 0 1.5rem; }}
        .markdown-body h2 {{ font-size: 1.5rem; font-weight: 700; color: #fff; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; margin: 2.5rem 0 1rem; }}
        .markdown-body h3 {{ font-size: 1.25rem; font-weight: 600; color: #fff; margin: 2rem 0 0.75rem; }}
        .markdown-body h4 {{ font-size: 1.1rem; font-weight: 600; color: #fff; margin: 1.5rem 0 0.5rem; }}
        .markdown-body p {{ color: var(--text-muted); margin: 0 0 1rem; }}
        .markdown-body ul, .markdown-body ol {{ color: var(--text-muted); margin: 0 0 1rem; padding-left: 1.5rem; }}
        .markdown-body li {{ margin-bottom: 0.25rem; }}
        .markdown-body blockquote {{
            border-left: 4px solid var(--accent);
            padding: 0.5rem 1rem;
            font-style: italic;
            background: rgba(34, 211, 238, 0.05);
            border-radius: 0 0.5rem 0.5rem 0;
            margin: 1rem 0;
        }}
        .markdown-body a {{ color: var(--accent); text-decoration: underline; }}
        .markdown-body a:hover {{ color: #67e8f9; }}
        .markdown-body :not(pre) > code {{
            font-size: 0.875rem;
            color: #67e8f9;
            background: var(--panel);
            padding: 0.125rem 0.375rem;
            border-radius: 0.25rem;
        }}
        .markdown-body pre {{
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 1rem 0;
            overflow-x: auto;
        }}
        .markdown-body pre code {{
            background: transparent;
            padding: 0;
            color: #e2e8f0;
            font-size: 0.875rem;
        }}
        .markdown-body table {{ width: 100%; font-size: 0.875rem; border-collapse: collapse; margin: 1.5rem 0; }}
        .markdown-body th, .markdown-body td {{ border: 1px solid var(--border); padding: 0.5rem 0.75rem; text-align: left; }}
        .markdown-body th {{ background: var(--panel); color: #fff; font-weight: 600; }}
        .markdown-body td {{ color: var(--text-muted); }}
        .markdown-body hr {{ border: none; border-top: 1px solid var(--border); margin: 2rem 0; }}

        /* Mermaid diagram panel: readable offline as labeled source; mermaid.js
           (if vendored) replaces the <pre> contents with a rendered <svg>. */
        .diagram-panel {{
            background: var(--bg);
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
            color: var(--text-muted);
            font-size: 0.8rem;
            white-space: pre-wrap;
        }}
{platform_css}
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

    <!-- Math: local vendor copy preferred, CDN used only as a fallback when online. -->
    <script>
        window.MathJax = {{
            tex: {{ inlineMath: [['$', '$']], displayMath: [['$$', '$$']] }}
        }};
        (function loadMathJax() {{
            var local = document.createElement('script');
            local.src = '{mathjax_local}';
            local.async = true;
            local.onerror = function () {{
                var cdn = document.createElement('script');
                cdn.src = 'https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-svg.js';
                cdn.async = true;
                document.head.appendChild(cdn);
            }};
            document.head.appendChild(local);
        }})();
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
        else:
            label = slug.replace("chapter", "Ch. ").replace("_5_interlude", ".5 Interlude").replace("_", " ")
        active = slug == current_slug
        classes = "active" if active else ""
        chapter_links.append(NAV_LINK.format(href=f"{slug}.html", classes=classes, label=label))
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
    <nav class="chapters" style="margin-bottom: 1rem; padding-bottom: 1rem;">
        <span style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: var(--accent); letter-spacing: 0.05em; width: 100%; margin-bottom: 0.5rem; display: block;">Book Chapters</span>
        {joined_chapters}
    </nav>
    <nav class="chapters" style="margin-bottom: 2.5rem; padding-bottom: 1.5rem;">
        <span style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: var(--accent); letter-spacing: 0.05em; width: 100%; margin-bottom: 0.5rem; display: block;">Companion Guides</span>
        {joined_companions}
    </nav>
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
        key = f"day-{day}"
        if key not in available:
            return heading
        return (
            f'{heading}\n\n<button type="button" class="companion-more-button" '
            f'data-companion-section="{key}">✦ Read the beginner companion for Day {day}</button>'
        )

    md_text = re.sub(
        r"^(#\s+Day\s+(\d+)\s+.+)$",
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
        "- `vendor/mathjax/tex-mml-svg.js` — pinned MathJax 3.2.2 bundle from\n"
        "  https://www.jsdelivr.com/package/npm/mathjax (es5/tex-mml-svg.js)\n\n"
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
    
    for chapter_slug, filename in COMPANION_FILES.items():
        compile_companion(f"{chapter_slug}_companion", filename)
    compile_settings_page()
    print("SRC markdown compilation complete.")


if __name__ == "__main__":
    main()
