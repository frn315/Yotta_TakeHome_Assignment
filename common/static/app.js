// Helpers shared by the builder and RAG pages. Each page defines KIND:
// event kind -> [label, css class, summary(event)] for the Activity timeline.
const $ = id => document.getElementById(id);
const icon = name => `<svg class="icon" aria-hidden="true"><use href="/static/icons.svg#${name}"/></svg>`;

// Top bar, the same on both pages: the two sections of the app as large tiles.
const SECTIONS = [["/", "builder", "Website Builder", "Describe a site, edit it by chat"],
                  ["/rag", "chat", "Document Chat", "Ask your documents, see the sources"]];
document.body.insertAdjacentHTML("afterbegin", `<header class="top"><a class="logo" href="/">${icon("logo")}PromptSite</a>
  <nav class="sections" aria-label="Sections">${SECTIONS.map(([href, name, title, sub]) =>
    `<a class="tile" href="${href}"${location.pathname === href ? ' aria-current="page"' : ""}>${icon(name)}<span><b>${title}</b><small>${sub}</small></span></a>`).join("")}</nav>
  <span class="badge" id="provider">checking provider...</span></header>`);

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

async function api(path, body) {
  let res;
  try { res = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); }
  catch (e) { return { ok: false, data: { error: { code: "network", message: "Cannot reach the server. Is it still running?" } } }; }
  let data = {}; try { data = await res.json(); } catch (e) { data = { error: { code: "bad_response", message: "Server returned an unreadable response (" + res.status + ")." } }; }
  if (!res.ok && !data.error) data.error = { code: "http_" + res.status, message: (data.detail && JSON.stringify(data.detail)) || "Request failed." };
  return { ok: res.ok, data };
}

// A chat bubble. cls: user | bot | err | "bot busy".
function bubble(cls, text) {
  $("chat").querySelector(".hint")?.remove();
  const div = document.createElement("div"); div.className = "msg " + cls; div.textContent = text;
  $("chat").appendChild(div); $("chat").scrollTop = $("chat").scrollHeight;
  return div;
}

// Tabs: <button role="tab" aria-controls="paneId">. One pane visible at a time.
function showTab(tabId) {
  for (const tab of document.querySelectorAll("[role=tab]")) {
    const on = tab.id === tabId;
    tab.setAttribute("aria-selected", on); $(tab.getAttribute("aria-controls")).hidden = !on;
  }
}
for (const tab of document.querySelectorAll("[role=tab]")) tab.onclick = () => showTab(tab.id);

// Activity timeline: one row per event, every field of the event on demand.
let eventCount = 0;
function addEvents(events) {
  if (!events || !events.length) return;
  if (!eventCount) $("log").innerHTML = "";
  eventCount += events.length; $("logCount").textContent = eventCount;
  for (const e of events) {
    const [label, cls, summary] = KIND[e.kind] || [e.kind, "", () => ""];
    const { ts, source, kind, site_id, ...rest } = e;
    const rows = Object.entries(rest).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${k === "sent" || k === "reply" ? `<pre>${esc(v)}</pre>` : esc(typeof v === "string" ? v : JSON.stringify(v))}</dd>`).join("");
    $("log").insertAdjacentHTML("beforeend", `<li><details class="ev ${cls}"><summary><time>${new Date(ts * 1000).toLocaleTimeString([], { hour12: false })}</time><span class="kind">${esc(label)}</span><span class="sum">${esc(summary(e))}</span></summary><dl>${rows}</dl></details></li>`);
  }
}

// Provider badge in the header; a red message in the chat when the provider is not usable.
function showProvider(data) {
  const badge = $("provider");
  badge.textContent = (data.provider || data.llm) ? `${(data.llm || data).provider} · ${(data.llm || data).model}` : "server unreachable";
  const problem = data.ready === false ? data.problem : data.error && data.error.message;
  if (problem) { badge.classList.add("bad"); bubble("err", "Provider not ready: " + problem); }
}
