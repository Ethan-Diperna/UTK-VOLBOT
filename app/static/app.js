"use strict";

// Everything shown on the page is built with textContent / append, never
// innerHTML, so text from the model or the catalog can never run as HTML.

const byId = (id) => document.getElementById(id);
const app = byId("app");
const thread = byId("thread");
const form = byId("ask-form");
const questionBox = byId("question");
const sendButton = byId("send");

const CATALOG_PREFIX = "https://catalog.utk.edu/";
const PENDING_TEXT = "Searching the catalog and writing an answer…";

// Chat memory. The server keeps nothing between questions, so the page sends
// the last few answered turns along with each new question; the server uses
// them to work out what a follow-up like "what are its prerequisites?" means.
// Both limits match the server's (src/config.py); it rejects anything larger.
const MAX_HISTORY_TURNS = 3;
const MAX_HISTORY_ANSWER_CHARS = 4000;

// Adds a turn and returns a new list (the old one is left untouched).
function withTurn(history, question, answer) {
  const turn = { question, answer: answer.slice(0, MAX_HISTORY_ANSWER_CHARS) };
  return [...history, turn].slice(-MAX_HISTORY_TURNS);
}

let history = [];

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

function catalogLink(url, className, text) {
  const link = element("a", className, text);
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

function citationNode(number, citations) {
  const citation = citations.get(number);
  const url = citation && catalogUrl(citation.url);
  if (!url) return document.createTextNode("[" + number + "]");
  const link = catalogLink(url, "cite", String(citation.shown));
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

function answerNode(text, citationList) {
  const citations = new Map(citationList.map((c) => [c.n, c]));
  const box = element("div", "answer");
  let list = null;
  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();
    if (!line) { list = null; continue; }
    const bullet = /^[-*]\s+/.exec(line);
    if (bullet) {
      if (!list) { list = element("ul"); box.append(list); }
      const item = element("li");
      appendInline(item, line.slice(bullet[0].length), citations);
      list.append(item);
    } else {
      list = null;
      const paragraph = element("p");
      appendInline(paragraph, line, citations);
      box.append(paragraph);
    }
  }
  return box;
}

function sourceItem(source, shownNumber) {
  const item = element("li", shownNumber ? "cited" : "");
  if (shownNumber) item.value = shownNumber;
  const label = source.section ? source.title + " / " + source.section : source.title;
  const url = catalogUrl(source.url);
  item.append(url ? catalogLink(url, "", label) : label);
  const excerpt = element("details", "excerpt");
  excerpt.append(element("summary", "", "Show excerpt"), element("pre", "", source.text));
  item.append(excerpt);
  return item;
}

function countedSummary(label, count) {
  const summary = element("summary", "", label + " ");
  summary.append(element("span", "count", "(" + count + ")"));
  return summary;
}

// Cited sources first, numbered as in the answer; then everything else that
// was searched and sent to the model, unnumbered. Both start collapsed.
function sourcesNode(sources, citationList) {
  const shown = new Map(citationList.map((c) => [c.n, c.shown]));
  const cited = sources.filter((s) => shown.has(s.n)).sort((a, b) => shown.get(a.n) - shown.get(b.n));
  const others = sources.filter((s) => !shown.has(s.n));
  const panel = element("details", "sources-panel");
  const citedList = element("ol", "sources");
  citedList.append(...cited.map((s) => sourceItem(s, shown.get(s.n))));
  panel.append(countedSummary("Sources", cited.length + " cited"), citedList);
  if (others.length > 0) {
    const otherPanel = element("details", "other-sources");
    const otherList = element("ul");
    otherList.append(...others.map((s) => sourceItem(s, 0)));
    otherPanel.append(countedSummary("Other excerpts that were searched", others.length), otherList);
    panel.append(otherPanel);
  }
  return panel;
}

// One question and its reply. The first one switches the page from the
// centred start screen to the chat layout. Returns the (still waiting) reply.
function addTurn(question) {
  app.classList.remove("is-empty");
  thread.hidden = false;
  const turn = element("article", "turn");
  const reply = element("div", "bot-msg");
  reply.append(element("p", "pending", PENDING_TEXT));
  turn.append(element("div", "user-msg", question), reply);
  thread.append(turn);
  turn.scrollIntoView({ block: "start", behavior: "smooth" });
  return reply;
}

function showReply(reply, data) {
  const citations = numberForDisplay(data.citations);
  const parts = [answerNode(data.answer, citations), sourcesNode(data.sources, citations)];
  // Show how a follow-up was understood, so a wrong guess is easy to spot.
  if (typeof data.searched_as === "string" && data.searched_as) {
    parts.unshift(element("p", "understood", "Understood as: " + data.searched_as));
  }
  if (typeof data.latency_s === "number") {
    parts.push(element("p", "meta", "Answered in " + data.latency_s.toFixed(1) + " s"));
  }
  reply.replaceChildren(...parts);
}

function showError(reply, message) {
  reply.replaceChildren(element("p", "error", message));
}

// The Ask button turns into three spinning dots while a question is being
// answered. The look is all CSS (see #send in style.css); this only flips the
// class. A reply that comes back almost at once (a rejected question, say)
// would make the button flicker, so the dots always stay for a short minimum.
const MIN_LOADING_MS = 700;

function setLoading(isLoading) {
  sendButton.classList.toggle("is-loading", isLoading);
  sendButton.setAttribute("aria-busy", String(isLoading));
}

async function ask(question) {
  const startedAt = performance.now();
  sendButton.disabled = true;
  setLoading(true);
  const reply = addTurn(question);
  questionBox.value = "";
  // No "mode" is sent, so the server uses its configured search method.
  const payload = { question, history };
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
      showError(reply, message || "The server gave an unexpected reply. Please try again.");
      return;
    }
    showReply(reply, body.data);
    history = withTurn(history, question, body.data.answer);
  } catch (networkError) {
    showError(reply, "Could not reach the server. Check your connection and try again.");
  } finally {
    const remaining = MIN_LOADING_MS - (performance.now() - startedAt);
    if (remaining > 0) await new Promise((resolve) => setTimeout(resolve, remaining));
    setLoading(false);
    sendButton.disabled = false;
    questionBox.focus();
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
