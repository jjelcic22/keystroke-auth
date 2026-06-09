
"use strict";

// ── Configuration ────────────────────────────────────────────────────────────

const WINDOW_SIZE   = 40;
const HISTORY_SIZE  = 5;

const THRESHOLD_WARNING = 2;
const THRESHOLD_LOCKED  = 3;

const IGNORED_KEYS = new Set([
  "Shift", "Control", "Alt", "Meta", "CapsLock",
  "Tab", "Escape", "ArrowLeft", "ArrowRight",
  "ArrowUp", "ArrowDown"
]);

// ── State ────────────────────────────────────────────────────────────────────

let currentUser    = null;
let eventBuffer    = [];       // paired events: {key, keydown, keyup}
let pressedKeys    = {};       // key -> keydown timestamp
let backspaceCount = 0;
let totalWindows   = 0;
let windowHistory  = [];
let sessionLocked  = false;

// ── DOM shortcuts ─────────────────────────────────────────────────────────────

const screenLogin    = document.getElementById("screen-login");
const screenSession  = document.getElementById("screen-session");
const typingArea     = document.getElementById("typing-area");
const displayUser    = document.getElementById("display-user");
const windowCounter  = document.getElementById("window-counter");
const lockOverlay    = document.getElementById("lock-overlay");
const statusBadge    = document.getElementById("status-badge");
const statusIcon     = document.getElementById("status-icon");
const statusText     = document.getElementById("status-text");
const statusSub      = document.getElementById("status-sub");
const windowHistory$ = document.getElementById("window-history");
const statKeys       = document.getElementById("stat-keys");
const statBs         = document.getElementById("stat-bs");
const statWindows    = document.getElementById("stat-windows");
const statScore      = document.getElementById("stat-score");

// ── Login / Logout ────────────────────────────────────────────────────────────

async function loginAs(user) {
  try {
    const resp = await fetch("/login", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ user }),
    });

    const data = await resp.json();

    if (!resp.ok) {
      alert(data.error || "Login failed.");
      return;
    }

    currentUser = user;
    _initSession(user);
  } catch (err) {
    console.error("Login error:", err);
    alert("Could not connect to the server.");
  }
}

async function logout() {
  try {
    await fetch("/logout", { method: "POST" });
  } catch (_) {
    // best-effort
  }

  _resetState();
  screenSession.classList.remove("active");
  screenLogin.classList.add("active");
}

function resetSession() {
  logout();
}

// ── Session initialisation ────────────────────────────────────────────────────

function _initSession(user) {
  _resetState();
  currentUser = user;

  screenLogin.classList.remove("active");
  screenSession.classList.add("active");

  displayUser.textContent = user.charAt(0).toUpperCase() + user.slice(1);

  typingArea.addEventListener("keydown", _onKeyDown);
  typingArea.addEventListener("keyup", _onKeyUp);

  typingArea.disabled = false;
  typingArea.focus();

  _renderStatus();
}

function _resetState() {
  currentUser    = null;
  eventBuffer    = [];
  pressedKeys    = {};
  backspaceCount = 0;
  totalWindows   = 0;
  windowHistory  = [];
  sessionLocked  = false;

  typingArea.removeEventListener("keydown", _onKeyDown);
  typingArea.removeEventListener("keyup", _onKeyUp);

  typingArea.value = "";
  typingArea.disabled = false;

  lockOverlay.classList.add("hidden");

  statKeys.textContent    = "0";
  statBs.textContent      = "0";
  statWindows.textContent = "0";
  statScore.textContent   = "—";
  windowCounter.textContent = `window 0 / keystroke 0 / ${WINDOW_SIZE}`;
  windowHistory$.innerHTML = '<div class="history-empty">No windows analysed yet.</div>';

  _applyStatus("authenticated", "✓", "AUTHENTICATED", "Waiting for data…");
}

// ── Keystroke capture ─────────────────────────────────────────────────────────

function _onKeyDown(e) {
  if (sessionLocked) return;
  if (IGNORED_KEYS.has(e.key)) return;
  if (e.repeat) return;

  const ts = performance.now();

  if (e.key === "Backspace") {
    backspaceCount++;
  }

  pressedKeys[e.key] = ts;
  _updateCounters();
}

function _onKeyUp(e) {
  if (sessionLocked) return;
  if (IGNORED_KEYS.has(e.key)) return;

  const keydownTime = pressedKeys[e.key];

  if (keydownTime === undefined) return;

  const keyupTime = performance.now();

  if (keyupTime >= keydownTime) {
    eventBuffer.push({
      key: e.key,
      keydown: keydownTime,
      keyup: keyupTime,
    });
  }

  delete pressedKeys[e.key];

  _updateCounters();

  if (eventBuffer.length >= WINDOW_SIZE) {
    _flushWindow();
  }
}

function _updateCounters() {
  statKeys.textContent = eventBuffer.length;
  statBs.textContent   = backspaceCount;
  windowCounter.textContent =
    `window ${totalWindows} / keystroke ${eventBuffer.length} / ${WINDOW_SIZE}`;
}

// ── Window flush & prediction ─────────────────────────────────────────────────

async function _flushWindow() {
  const eventsToSend = eventBuffer.slice(0, WINDOW_SIZE);
  const bsCount      = backspaceCount;

  eventBuffer    = eventBuffer.slice(WINDOW_SIZE);
  pressedKeys    = {};
  backspaceCount = 0;
  totalWindows++;

  statWindows.textContent = totalWindows;
  _updateCounters();

  try {
    const resp = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        events: eventsToSend,
        backspace_count: bsCount,
      }),
    });

    const result = await resp.json();

    if (!resp.ok) {
      console.warn("Predict endpoint returned", resp.status, result);
      _handlePrediction({
        accepted: false,
        score: 0,
        status: "error",
        message: result.message || result.error || "Backend error",
      });
      return;
    }

    _handlePrediction(result);
  } catch (err) {
    console.error("Predict request failed:", err);
    _handlePrediction({
      accepted: false,
      score: 0,
      status: "error",
      message: "Predict request failed",
    });
  }
}

// ── Prediction handling & sliding-window state machine ───────────────────────

function _handlePrediction(result) {
  windowHistory.push(result);

  if (windowHistory.length > HISTORY_SIZE) {
    windowHistory.shift();
  }

  statScore.textContent =
    typeof result.score === "number" ? (result.score * 100).toFixed(1) + "%" : "—";

  const suspiciousCount = windowHistory.filter(r => !r.accepted).length;

  let newState, icon, label, sub;

  if (suspiciousCount >= THRESHOLD_LOCKED) {
    newState = "locked";
    icon     = "⊘";
    label    = "LOCKED";
    sub      = `${suspiciousCount}/${windowHistory.length} suspicious — session suspended`;
    _lockSession();
  } else if (suspiciousCount >= THRESHOLD_WARNING) {
    newState = "warning";
    icon     = "△";
    label    = "WARNING";
    sub      = `${suspiciousCount}/${windowHistory.length} suspicious — monitoring closely`;
  } else {
    newState = "authenticated";
    icon     = "✓";
    label    = "AUTHENTICATED";
    sub      = `${suspiciousCount}/${windowHistory.length} suspicious — behaviour normal`;
  }

  if (result.status === "error") {
    newState = "warning";
    icon = "!";
    label = "ERROR";
    sub = result.message || "Prediction error";
  }

  _applyStatus(newState, icon, label, sub);
  _renderHistoryTiles();
}

function _lockSession() {
  sessionLocked = true;
  typingArea.disabled = true;
  lockOverlay.classList.remove("hidden");
}

// ── UI helpers ────────────────────────────────────────────────────────────────

function _applyStatus(state, icon, label, sub) {
  statusBadge.className = `status-badge status-${state}`;
  statusIcon.textContent = icon;
  statusText.textContent = label;
  statusSub.textContent  = sub;
}

function _renderHistoryTiles() {
  if (windowHistory.length === 0) {
    windowHistory$.innerHTML =
      '<div class="history-empty">No windows analysed yet.</div>';
    return;
  }

  windowHistory$.innerHTML = windowHistory
    .map((r, i) => {
      const cls   = r.accepted ? "accepted" : "suspicious";
      const icon  = r.accepted ? "✓" : "✗";
      const score = typeof r.score === "number" ? (r.score * 100).toFixed(0) + "%" : "—";
      return `
        <div class="hist-tile ${cls}" title="Window ${i + 1}: score ${score}">
          <span class="hist-score">${icon}</span>
          <span class="hist-label">${score}</span>
        </div>`;
    })
    .join("");
}

function _renderStatus() {
  _applyStatus("authenticated", "✓", "AUTHENTICATED", "Waiting for data…");
  _renderHistoryTiles();
}
