"use strict";

// Everything shown on the page is built with textContent / append, never
// innerHTML, so text from the model or the catalog can never run as HTML.

const byId = (id) => document.getElementById(id);
const form = byId("ask-form");
const questionBox = byId("question");
const sendButton = byId("send");
const modeSelect = byId("mode");
const statusLine = byId("status");
const resultBox = byId("result");
const answerBox = byId("answer");
const sourcesPanel = byId("sources-panel");
const sourcesList = byId("sources");
const sourceCount = byId("source-count");
const otherSourcesPanel = byId("other-sources-panel");
const otherSourcesList = byId("other-sources");
const otherSourcesCount = byId("other-source-count");
const metaLine = byId("meta");

const CATALOG_PREFIX = "https://catalog.utk.edu/";
const MODE_LABELS = {
  hybrid_rerank: "Hybrid + rerank (default)",
  hybrid: "Hybrid",
  vector: "Meaning only (vector)",
  bm25: "Keywords only (BM25)",
};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Links only ever point at the catalog site, whatever the response contains.
function catalogUrl(url) {
  return typeof url === "string" && url.startsWith(CATALOG_PREFIX) ? url : null;
}

function showStatus(text, isError) {
  statusLine.textContent = text;
  statusLine.classList.toggle("error", Boolean(isError));
  statusLine.hidden = !text;
}

function citationNode(number, citations) {
  const citation = citations.get(number);
  const url = citation && catalogUrl(citation.url);
  if (!url) return document.createTextNode("[" + number + "]");
  const link = element("a", "cite", String(citation.shown));
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.title = citation.section ? citation.title + " / " + citation.section : citation.title;
  return link;
}

// Turns "**bold**" and "[1]" / "[1, 2]" inside one line into nodes.
function appendInline(parent, text, citations) {
  const pattern = /\*\*(.+?)\*\*|\[(\d+(?:\s*,\s*\d+)*)\]/g;
  let last = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) parent.append(text.slice(last, match.index));
    if (match[1] !== undefined) {
      const strong = element("strong");
      appendInline(strong, match[1], citations);
      parent.append(strong);
    } else {
      for (const raw of match[2].split(",")) parent.append(citationNode(Number(raw.trim()), citations));
    }
    last = pattern.lastIndex;
  }
  if (last < text.length) parent.append(text.slice(last));
}

// The model cites excerpts by their position in what it was sent, which can
// be [28] or [41] when whole program pages are included. For reading, cited
// sources are renumbered 1, 2, 3 in the order the answer first uses them.
function numberForDisplay(citationList) {
  return citationList.map((c, index) => ({ ...c, shown: index + 1 }));
}

function renderAnswer(text, citationList) {
  const citations = new Map(citationList.map((c) => [c.n, c]));
  answerBox.replaceChildren();
  let list = null;
  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();
    if (!line) { list = null; continue; }
    const bullet = /^[-*]\s+/.exec(line);
    if (bullet) {
      if (!list) { list = element("ul"); answerBox.append(list); }
      const item = element("li");
      appendInline(item, line.slice(bullet[0].length), citations);
      list.append(item);
    } else {
      list = null;
      const paragraph = element("p");
      appendInline(paragraph, line, citations);
      answerBox.append(paragraph);
    }
  }
}

function sourceItem(source, shownNumber) {
  const item = element("li", shownNumber ? "cited" : "");
  if (shownNumber) item.value = shownNumber;
  const label = source.section ? source.title + " / " + source.section : source.title;
  const url = catalogUrl(source.url);
  if (url) {
    const link = element("a", "", label);
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    item.append(link);
  } else {
    item.append(label);
  }
  const excerpt = element("details", "excerpt");
  excerpt.append(element("summary", "", "Show excerpt"), element("pre", "", source.text));
  item.append(excerpt);
  return item;
}

// Cited sources first, numbered as in the answer; then everything else that
// was searched and sent to the model, unnumbered.
function renderSources(sources, citationList) {
  const shown = new Map(citationList.map((c) => [c.n, c.shown]));
  const cited = sources.filter((s) => shown.has(s.n)).sort((a, b) => shown.get(a.n) - shown.get(b.n));
  const others = sources.filter((s) => !shown.has(s.n));
  sourcesList.replaceChildren(...cited.map((s) => sourceItem(s, shown.get(s.n))));
  otherSourcesList.replaceChildren(...others.map((s) => sourceItem(s, 0)));
  otherSourcesPanel.hidden = others.length === 0;
  otherSourcesPanel.open = false;
  otherSourcesCount.textContent = "(" + others.length + ")";
  sourceCount.textContent = "(" + cited.length + " cited)";
  sourcesPanel.open = false;
}

async function ask(question) {
  sendButton.disabled = true;
  resultBox.hidden = true;
  showStatus("Searching the catalog and writing an answer…", false);
  const payload = { question };
  if (modeSelect.value) payload.mode = modeSelect.value;
  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    let body = null;
    try { body = await response.json(); } catch (parseError) { body = null; }
    if (!body || !body.success) {
      const message = body && body.error && body.error.message;
      showStatus(message || "The server gave an unexpected reply. Please try again.", true);
      return;
    }
    const data = body.data;
    const citations = numberForDisplay(data.citations);
    renderAnswer(data.answer, citations);
    renderSources(data.sources, citations);
    metaLine.textContent = typeof data.latency_s === "number" ? "Answered in " + data.latency_s.toFixed(1) + " s" : "";
    showStatus("", false);
    resultBox.hidden = false;
  } catch (networkError) {
    showStatus("Could not reach the server. Check your connection and try again.", true);
  } finally {
    sendButton.disabled = false;
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionBox.value.trim();
  if (question && !sendButton.disabled) ask(question);
});

// Enter sends; Shift+Enter makes a new line.
questionBox.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

for (const chip of document.querySelectorAll(".chip")) {
  chip.addEventListener("click", () => {
    questionBox.value = chip.textContent;
    form.requestSubmit();
  });
}

// Fill the search-method dropdown from the server, so the page never lists a
// mode the server does not have. If this fails the default mode still works.
fetch("/health")
  .then((response) => response.json())
  .then((health) => {
    for (const mode of health.modes || []) {
      if (mode === health.default_mode) continue;
      const option = element("option", "", MODE_LABELS[mode] || mode);
      option.value = mode;
      modeSelect.append(option);
    }
  })
  .catch(() => {});
