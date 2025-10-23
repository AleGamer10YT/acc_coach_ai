const WS_URL = (window.OVERLAY_WS_URL || "ws://localhost:8080/ws/feedback").replace(
  /^http/,
  "ws"
);

const sessionEl = document.getElementById("session-id");
const lapEl = document.getElementById("lap-info");
const deltaEl = document.getElementById("delta-value");
const throttleEl = document.getElementById("throttle-value");
const streamEl = document.getElementById("feedback-stream");

let socket;
let reconnectTimeout;

function connect() {
  socket = new WebSocket(WS_URL);

  socket.addEventListener("open", () => {
    clearTimeout(reconnectTimeout);
    streamEl.classList.remove("disconnected");
  });

  socket.addEventListener("message", (event) => {
    const payload = JSON.parse(event.data);
    renderFeedback(payload);
  });

  socket.addEventListener("close", () => scheduleReconnect());
  socket.addEventListener("error", () => socket.close());
}

function scheduleReconnect() {
  clearTimeout(reconnectTimeout);
  streamEl.classList.add("disconnected");
  reconnectTimeout = setTimeout(connect, 2000);
}

function formatDelta(ms) {
  if (typeof ms !== "number") return "+0.000 s";
  const sign = ms >= 0 ? "+" : "-";
  const value = Math.abs(ms / 1000).toFixed(3);
  return `${sign}${value} s`;
}

function renderFeedback(event) {
  const placeholder = streamEl.querySelector(".feedback-placeholder");
  if (placeholder) {
    placeholder.remove();
  }

  sessionEl.textContent = `Sessione: ${event.session_id ?? "--"}`;
  lapEl.textContent = `Lap: ${event.lap ?? "--"}`;

  if (event.metrics?.delta_best_ms !== undefined) {
    deltaEl.textContent = formatDelta(event.metrics.delta_best_ms);
  }
  if (event.metrics?.throttle_avg !== undefined) {
    throttleEl.textContent = `${Math.round(event.metrics.throttle_avg * 100)}%`;
  }

  const card = document.createElement("article");
  card.className = `feedback-card ${event.severity || "suggestion"}`;

  card.innerHTML = `
    <div class="feedback-card__meta">
      <span>${event.section ?? "Generale"}</span>
      <span>${event.severity ?? "info"}</span>
    </div>
    <h3 class="feedback-card__title">${event.message}</h3>
    ${
      event.metrics
        ? `<div class="feedback-card__metrics">
          ${Object.entries(event.metrics)
            .map(([key, value]) => `<span>${key}: <strong>${value}</strong></span>`)
            .join("")}
        </div>`
        : ""
    }
  `;

  streamEl.prepend(card);
  const cards = streamEl.querySelectorAll(".feedback-card");
  const limit = 4;
  if (cards.length > limit) {
    cards[cards.length - 1].remove();
  }
}

connect();
