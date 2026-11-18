"""Business-friendly HTML report for a generated graph.

Written for process owners, operations and compliance readers, not engineers:
plain wording ("steps", "who", "what happens next"), no internal ids or
pipeline terms, gaps phrased as questions, source text on request, and the
technical details folded away at the bottom for analysts.
"""
import html
import json
import re
from datetime import datetime

# Fill / border colours for roles (cycled when there are more roles).
PALETTE = [
    ("#dbeafe", "#2563eb"), ("#dcfce7", "#16a34a"), ("#fef3c7", "#d97706"),
    ("#fce7f3", "#db2777"), ("#ede9fe", "#7c3aed"), ("#cffafe", "#0891b2"),
    ("#ffedd5", "#ea580c"), ("#e0e7ff", "#4f46e5"), ("#ecfccb", "#65a30d"),
    ("#fae8ff", "#c026d3"),
]
NO_OWNER = ("#f1f5f9", "#94a3b8")
TYPE_LABELS = {"start": "Start", "end": "Outcome", "decision": "Decision",
               "activity": "Step", "event": "Event"}


# ---------------------------------------------------------------- helpers
def _s(value, default=""):
    """Null-safe string: LLM output may contain null for any field."""
    return default if value is None else str(value)


def _e(value, default=""):
    return html.escape(_s(value, default))


def _plural(count, word, plural=None):
    return f"{count} {word if count == 1 else (plural or word + 's')}"


def _clean(text):
    """Remove pipeline internals from model-written sentences."""
    text = re.sub(r"\s*\(?\bC\d+:[\w-]+\)?", "", _s(text))
    text = re.sub(r"\b(in|from) (the|this) excerpt\b", r"\1 the document", text, flags=re.I)
    text = re.sub(r"\bprocess graph\b", "process map", text, flags=re.I)
    return re.sub(r"\bexcerpt\b", "document", text, flags=re.I).strip()


def _owner(node):
    actor = _s(node.get("actor")).strip()
    return "" if actor.lower() in ("", "unknown", "none", "n/a") else actor


def _step_name(node):
    name = _s(node.get("name")).strip() or _s(node.get("id"))
    if _s(node.get("type")).lower() == "decision" and not name.endswith("?"):
        name = re.sub(r"\s*decision$", "", name, flags=re.I).strip() + "?"
    return name


def _label(text, limit=70):
    """Text that is safe inside a quoted Mermaid label."""
    text = re.sub(r"[\"<>|]", "'", _s(text)).replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _date(value):
    try:
        return datetime.fromisoformat(_s(value)).strftime("%d %b %Y")
    except ValueError:
        return ""


# ---------------------------------------------------------------- model
class Report:
    """Everything the page needs, derived once from the graph."""

    def __init__(self, g):
        self.g = g
        self.process = _s(g.get("document_type")) == "business_process"
        self.nodes = [n for n in g.get("nodes", []) if isinstance(n, dict)]
        ids = {_s(n.get("id")) for n in self.nodes}
        self.edges = [e for e in g.get("edges", []) if isinstance(e, dict)
                      and _s(e.get("from")) in ids and _s(e.get("to")) in ids]
        self.number = {}
        for index, node in enumerate(self.nodes, 1):
            match = re.fullmatch(r"N(\d+)", _s(node.get("id")))
            self.number[_s(node.get("id"))] = int(match.group(1)) if match else index
        self.by_id = {_s(n.get("id")): n for n in self.nodes}
        self.out = {i: [] for i in ids}
        self.inc = {i: [] for i in ids}
        for e in self.edges:
            self.out[_s(e["from"])].append(e)
            self.inc[_s(e["to"])].append(e)
        roles = []
        for node in self.nodes:
            if _owner(node) and _owner(node) not in roles:
                roles.append(_owner(node))
        self.roles = roles
        self.colour = {role: PALETTE[i % len(PALETTE)] for i, role in enumerate(roles)}

    def kind(self, node):
        return _s(node.get("type")).lower()

    def title(self, node_id):
        return f"{self.number[node_id]}. {_step_name(self.by_id[node_id])}"

    def outcomes(self):
        return [n for n in self.nodes if self.kind(n) == "end"]

    def decisions(self):
        return [n for n in self.nodes if self.kind(n) == "decision"]

    def attention(self):
        """Things a reader should check, in plain business language."""
        items = []
        if self.process and self.nodes:
            if not any(self.kind(n) == "start" for n in self.nodes):
                items.append(("Gap", "It isn't clear where the process starts."))
            if not self.outcomes():
                items.append(("Gap", "No final outcome is described."))
        for n in self.nodes:
            node_id, name = _s(n.get("id")), f"Step {self.title(_s(n.get('id')))}"
            if not self.out[node_id] and not self.inc[node_id]:
                items.append(("Gap", f"{name} isn't connected to the rest of the process."))
                continue
            if not self.process:
                continue
            if self.kind(n) != "end" and not self.out[node_id]:
                items.append(("Gap", f"{name}: the document doesn't say what happens next."))
            if self.kind(n) not in ("start",) and not self.inc[node_id]:
                items.append(("Gap", f"{name}: it isn't clear how the process gets here."))
            if self.kind(n) == "decision" and len(self.out[node_id]) < 2:
                items.append(("Gap", f"{name}: only one outcome is described. What happens otherwise?"))
        if self.process:
            unowned = [str(self.number[_s(n.get("id"))]) for n in self.nodes if not _owner(n)]
            if unowned:
                items.append(("Missing owner",
                              f"The document doesn't say who is responsible for "
                              f"{'step' if len(unowned) == 1 else 'steps'} {', '.join(unowned)}."))
        for example in (self.g.get("grounding") or {}).get("ungrounded_examples", []):
            where = _s(example.get("where"))
            label = f"Step {self.title(where)}" if where in self.by_id else "A link between steps"
            items.append(("Verify", f"{label}: its supporting quote couldn't be found word-for-word "
                                    "in the document. Please check it against the source."))
        for question in self.g.get("open_questions") or []:
            text = _clean(question)
            # owner gaps are already summarised above, and dropped/duplicate-id notes are technical
            if not text or re.search(r"\bactor\b|\bowner\b|edge with unknown|node id", text, re.I):
                continue
            items.append(("Question", text))
        return items


# ---------------------------------------------------------------- diagram
def mermaid_source(g):
    report = Report(g)
    safe = {}
    lines = ["flowchart TD"]
    for index, node in enumerate(report.nodes):
        node_id = _s(node.get("id"))
        safe[node_id] = f"n{index}"
        label = _label(report.title(node_id))
        kind = report.kind(node)
        if kind == "decision":  # hexagon: as distinct as a diamond, but stays compact
            lines.append(f'  n{index}{{{{"{label}"}}}}')
        elif kind in ("start", "end"):
            lines.append(f'  n{index}(["{label}"])')
        else:
            lines.append(f'  n{index}["{label}"]')
    for edge in report.edges:
        source, target = safe[_s(edge["from"])], safe[_s(edge["to"])]
        # Branch labels carry meaning; labels on plain sequence links are mostly noise.
        from_decision = report.kind(report.by_id[_s(edge["from"])]) == "decision"
        label = _label(edge.get("label"), 45) if from_decision else ""
        lines.append(f'  {source} -->|"{label}"| {target}' if label else f"  {source} --> {target}")
    for index, node in enumerate(report.nodes):
        fill, stroke = report.colour.get(_owner(node), NO_OWNER)
        width = "3px" if report.kind(node) in ("start", "end") else "1.5px"
        lines.append(f"  style n{index} fill:{fill},stroke:{stroke},stroke-width:{width},color:#0f172a")
    return "\n".join(lines)


# ---------------------------------------------------------------- page
def render_html(g):
    r = Report(g)
    word = "step" if r.process else "item"
    nodes, edges = r.nodes, r.edges

    # ----- header facts
    title = _e(g.get("graph_title"), "Process overview" if r.process else "Document overview")
    summary = _e(_clean(g.get("description")))
    facts = [_plural(len(nodes), word)]
    if r.process:
        facts += [_plural(len(r.decisions()), "decision"), _plural(len(r.roles), "role"),
                  _plural(len(r.outcomes()), "possible outcome")]
    else:
        facts += [_plural(len(edges), "connection")]
    source_line = " · ".join(x for x in [
        f"Source: {_e(g.get('source_document'))}" if g.get("source_document") else "",
        f"Generated {_date(g.get('generated_at'))}" if _date(g.get("generated_at")) else "",
    ] if x)
    grounding = g.get("grounding") or {}
    checked, found = grounding.get("quotes_checked", 0), grounding.get("quotes_grounded", 0)
    if checked and found == checked:
        trust = f"Every {word} is backed by a quote from the source document."
    elif checked:
        trust = f"{found} of {checked} supporting quotes match the document; the rest are listed below to verify."
    else:
        trust = ""
    reviewed = ('<span class="badge ok">Reviewed by a person</span>' if g.get("reviewed")
                else '<span class="badge">Not yet reviewed by a person</span>')

    # ----- needs attention
    attention = r.attention()
    if attention:
        attention_html = (
            f'<section class="attention"><h2>Needs your attention <span class="count">{len(attention)}</span></h2>'
            '<p class="muted">Gaps or unclear points found while reading the document. '
            'Check these with the process owner before relying on the map.</p><ol>'
            + "".join(f'<li><span class="tag tag-{_e(tag.split()[0].lower())}">{_e(tag)}</span> {_e(text)}</li>'
                      for tag, text in attention)
            + "</ol></section>")
    else:
        attention_html = ('<section class="attention clear"><h2>Needs your attention</h2>'
                          '<p>Nothing found: every step is connected, has an owner and is backed by the document.</p></section>')

    def step_link(node):
        node_id = _s(node.get("id"))
        return f'<a href="#step-{_e(node_id)}" data-step="{_e(node_id)}">{_e(r.title(node_id))}</a>'

    unowned = sum(1 for n in nodes if not _owner(n))

    # ----- legend for the map
    legend = "".join(
        f'<span class="key"><span class="dot" style="background:{r.colour[role][0]};border-color:{r.colour[role][1]}"></span>{_e(role)}</span>'
        for role in r.roles)
    if unowned:
        legend += (f'<span class="key"><span class="dot" style="background:{NO_OWNER[0]};border-color:{NO_OWNER[1]}"></span>'
                   'No owner stated</span>')
    shapes = ('<span class="key"><span class="shape pill"></span>Start / outcome</span>'
              '<span class="key"><span class="shape hex"></span>Decision</span>'
              '<span class="key"><span class="shape box"></span>Step</span>') if r.process else ""

    # ----- step-by-step rows
    rows = []
    for node in nodes:
        node_id = _s(node.get("id"))
        kind = r.kind(node)
        nexts = []
        for e in r.out[node_id]:
            target = step_link(r.by_id[_s(e["to"])])
            label = _s(e.get("label")).strip()
            if kind == "decision" and label:
                nexts.append(f'<li><span class="branch">{_e(label)}</span> → {target}</li>')
            else:
                nexts.append(f"<li>→ {target}</li>")
        if not nexts:
            nexts_html = ('<span class="muted">The process ends here.</span>' if kind == "end" or not r.process
                          else '<span class="warn">Not described in the document</span>')
        else:
            nexts_html = f'<ul class="next">{"".join(nexts)}</ul>'
        evidence = node.get("evidence") or []
        evidence = [evidence] if isinstance(evidence, str) else evidence
        source_html = ('<details><summary>Show quote</summary>'
                       + "".join(f"<blockquote>{_e(q)}</blockquote>" for q in evidence) + "</details>"
                       if evidence else '<span class="muted">—</span>')
        owner = _owner(node)
        fill, stroke = r.colour.get(owner, NO_OWNER)
        description = _s(node.get("description")).strip()
        if description.lower().startswith("decision point:"):
            description = description.split(":", 1)[1].strip()
        rows.append(
            f'<tr id="step-{_e(node_id)}" data-role="{_e(owner)}">'
            f'<td class="num">{r.number[node_id]}</td>'
            f'<td><div class="step-name">{_e(_step_name(node))}</div>'
            f'<span class="type type-{_e(kind)}">{_e(TYPE_LABELS.get(kind, kind.title() or "Item"))}</span>'
            f'<div class="desc">{_e(description)}</div></td>'
            f'<td><span class="dot" style="background:{fill};border-color:{stroke}"></span>'
            f'{_e(owner) if owner else "<i class=muted>Not stated</i>"}</td>'
            f"<td>{nexts_html}</td><td>{source_html}</td></tr>")

    # ----- interpretation + technical details
    assumptions = [_clean(a) for a in g.get("assumptions") or [] if _clean(a)]
    assumptions_html = ("<section><h2>How the document was interpreted</h2>"
                        "<p class='muted'>Choices made where the document was not explicit.</p><ul>"
                        + "".join(f"<li>{_e(a)}</li>" for a in assumptions) + "</ul></section>") if assumptions else ""
    classification = g.get("classification") or {}
    parts = "".join(
        f"<li>{_e(r.title(_s(n.get('id'))))}: document part {', '.join(_s(c) for c in n.get('source_chunks') or []) or '—'}</li>"
        for n in nodes)
    technical = (
        "<details class='technical'><summary>Technical details (for analysts)</summary>"
        f"<p>Graph type: <code>{_e(g.get('document_type'))}</code>"
        + (f" · classifier confidence {_e(classification.get('confidence'))}" if classification.get("confidence") is not None else "")
        + (f"<br>Classifier reason: {_e(classification.get('reason'))}" if classification.get("reason") else "")
        + f"<br>Supporting quotes found in the document: {found} of {checked}</p>"
        + "<h4>Automated checks</h4><ul>"
        + ("".join(f"<li>{_e(_clean(i))}</li>" for i in g.get("validation_issues") or []) or "<li>All passed</li>")
        + f"</ul><h4>Where each {word} came from</h4><ul class='parts'>{parts}</ul>"
        + "<p class='muted'>The full data is in document_graph.json.</p></details>")

    diagram = html.escape(mermaid_source(g))
    data = html.escape(json.dumps({"roles": r.roles}))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
:root{{--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--bg:#f8fafc;--accent:#2563eb}}
*{{box-sizing:border-box}} body{{font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--ink);margin:0}}
main{{max-width:1180px;margin:0 auto;padding:28px 20px 60px}}
header,section,.technical{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:22px 24px;margin-bottom:16px}}
h1{{font-size:26px;margin:0 0 6px}} h2{{font-size:19px;margin:0 0 10px}} h3{{font-size:15px;margin:0 0 8px}} h4{{margin:16px 0 6px}}
.muted{{color:var(--muted)}} .summary{{font-size:16px;max-width:900px;margin:6px 0 14px}}
.facts{{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}} .fact{{background:#eff6ff;color:#1e3a8a;border-radius:999px;padding:4px 12px;font-weight:600;font-size:14px}}
.meta{{display:flex;flex-wrap:wrap;gap:10px;align-items:center;font-size:13px;color:var(--muted)}}
.badge{{border:1px solid #f59e0b;color:#92400e;background:#fffbeb;border-radius:999px;padding:2px 10px;font-weight:600}}
.badge.ok{{border-color:#16a34a;color:#166534;background:#f0fdf4}}
.attention{{border-left:5px solid #f59e0b}} .attention.clear{{border-left-color:#16a34a}}
.attention ol{{margin:8px 0 0;padding-left:22px}} .attention li{{margin:6px 0}}
.count{{background:#f59e0b;color:#fff;border-radius:999px;padding:0 9px;font-size:14px;vertical-align:middle}}
.tag{{display:inline-block;font-size:11.5px;font-weight:700;text-transform:uppercase;letter-spacing:.03em;border-radius:6px;padding:1px 7px;margin-right:4px}}
.tag-gap{{background:#fef3c7;color:#92400e}} .tag-missing{{background:#ede9fe;color:#5b21b6}} .tag-verify{{background:#fee2e2;color:#991b1b}} .tag-question{{background:#e0f2fe;color:#075985}}
.dot{{display:inline-block;width:12px;height:12px;border-radius:4px;border:2px solid;vertical-align:-1px;margin-right:7px}}
a{{color:var(--accent);text-decoration:none}} a:hover{{text-decoration:underline}}
.map-tools{{display:flex;flex-wrap:wrap;justify-content:space-between;gap:10px;align-items:center;margin-bottom:10px}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px}} .key{{white-space:nowrap}}
.shape{{display:inline-block;width:16px;height:11px;border:2px solid #64748b;margin-right:6px;vertical-align:-1px}}
.shape.pill{{border-radius:8px}} .shape.hex{{width:18px;clip-path:polygon(25% 0,75% 0,100% 50%,75% 100%,25% 100%,0 50%);background:#64748b;border:0}}
button.tool{{border:1px solid var(--line);background:#fff;border-radius:8px;padding:5px 12px;font:inherit;font-size:13px;cursor:pointer}}
.map{{overflow:auto;border:1px solid var(--line);border-radius:12px;background:#fff;padding:12px;max-height:85vh}}
.map svg{{display:block;margin:0 auto}} .map.fit svg{{max-width:100%!important;height:auto}} .map .node{{cursor:pointer}}
.map .edgeLabel, .map .edgeLabel p{{background:#fff!important;color:#334155;font-size:13px}}
.hint{{font-size:13px;color:var(--muted);margin-top:6px}}
.filters{{display:flex;flex-wrap:wrap;gap:10px;margin:4px 0 12px}} .filters select,.filters input{{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:8px}}
.filters input{{flex:1;min-width:200px}}
table{{width:100%;border-collapse:collapse}} th{{text-align:left;font-size:12.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);border-bottom:2px solid var(--line);padding:8px}}
td{{border-bottom:1px solid var(--line);padding:12px 8px;vertical-align:top}} td.num{{font-weight:700;color:var(--muted);width:40px}}
th:nth-child(2){{width:38%}} th:nth-child(3){{width:170px}} th:nth-child(5){{width:120px}} td:nth-child(3){{white-space:nowrap}}
tr.hl td{{background:#fefce8;transition:background .3s}}
.step-name{{font-weight:650}} .desc{{color:#334155;margin-top:4px}}
.type{{display:inline-block;font-size:11.5px;font-weight:700;border-radius:6px;padding:0 7px;margin-top:3px;background:#f1f5f9;color:#475569}}
.type-decision{{background:#fef3c7;color:#92400e}} .type-end{{background:#dcfce7;color:#166534}} .type-start{{background:#dbeafe;color:#1e40af}}
.next{{list-style:none;margin:0;padding:0}} .next li{{margin:2px 0}} .branch{{font-weight:600}}
.warn{{color:#b45309;font-weight:600}}
blockquote{{margin:8px 0 0;padding:6px 10px;border-left:3px solid #cbd5e1;background:#f8fafc;color:#334155;font-size:14px}}
details summary{{cursor:pointer;color:var(--accent);font-size:13.5px}}
.technical summary{{font-weight:600;color:var(--muted)}} .parts{{columns:2;font-size:13.5px}}
@media (max-width:760px){{
  main{{padding:16px 10px 40px}} header,section,.technical{{padding:18px 16px}}
  thead{{display:none}} table,tbody,tr,td{{display:block;width:auto!important}}
  tr{{border-bottom:1px solid var(--line);padding:10px 0}} td{{border:0;padding:3px 0;white-space:normal!important}}
  td.num{{float:left;width:auto;margin-right:8px}} .parts{{columns:1}} .map{{max-height:70vh}}
}}
@media print{{ .filters,.map-tools button,.hint{{display:none}} .map{{max-height:none;overflow:visible}} details{{display:block}} }}
</style></head><body><main>
<header>
  <h1>{title}</h1>
  <p class="summary">{summary}</p>
  <div class="facts">{"".join(f'<span class="fact">{_e(f)}</span>' for f in facts)}</div>
  <div class="meta">{reviewed}<span>{_e(trust)}</span>{f"<span>{source_line}</span>" if source_line else ""}</div>
</header>
{attention_html}
<section>
  <h2>{"Process map" if r.process else "Map"}</h2>
  <div class="map-tools"><div class="legend">{legend}{shapes}</div>
    <button class="tool" id="fit">Show actual size</button></div>
  <div class="map fit" id="map"><pre class="mermaid">{diagram}</pre></div>
  <p class="hint">Read from top to bottom. Click a {word} to see its details below.</p>
</section>
<section>
  <h2>{"Step by step" if r.process else "All items"}</h2>
  <div class="filters">
    <select id="role-filter" aria-label="Filter by role"><option value="*">Everyone</option>
      {"".join(f'<option value="{_e(role)}">{_e(role)}</option>' for role in r.roles)}
      {'<option value="">No owner stated</option>' if unowned else ''}</select>
    <input id="search" type="search" placeholder="Search {word}s, e.g. credit report, decline, signature">
  </div>
  <table><thead><tr><th>#</th><th>{"Step" if r.process else "Item"}</th><th>Who</th>
    <th>{"What happens next" if r.process else "Connected to"}</th><th>Source</th></tr></thead>
  <tbody id="rows">{"".join(rows)}</tbody></table>
  <p class="muted" id="no-match" hidden>No {word}s match the filter.</p>
</section>
{assumptions_html}
{technical}
</main>
<script type="application/json" id="report-data">{data}</script>
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
mermaid.initialize({{startOnLoad:false, securityLevel:'strict', maxTextSize:5000000, maxEdges:20000,
  theme:'base', themeVariables:{{fontFamily:'system-ui,-apple-system,Segoe UI,sans-serif', fontSize:'15px', lineColor:'#64748b', edgeLabelBackground:'#ffffff'}},
  flowchart:{{useMaxWidth:false, htmlLabels:true, curve:'basis', nodeSpacing:28, rankSpacing:44, wrappingWidth:220}}}});
await mermaid.run({{querySelector:'.mermaid'}});
const rows = [...document.querySelectorAll('#rows tr')];
const ids = rows.map(r => r.id.replace('step-', ''));
function show(stepId) {{
  const row = document.getElementById('step-' + stepId); if (!row) return;
  document.getElementById('role-filter').value = '*'; document.getElementById('search').value = ''; filter();
  rows.forEach(r => r.classList.remove('hl')); row.classList.add('hl');
  row.scrollIntoView({{behavior:'smooth', block:'center'}});
}}
document.querySelectorAll('#map g.node').forEach(g => {{
  const m = g.id.match(/flowchart-n(\\d+)-/); if (!m) return;
  g.addEventListener('click', () => show(ids[Number(m[1])]));
}});
document.querySelectorAll('a[data-step]').forEach(a => a.addEventListener('click', ev => {{ ev.preventDefault(); show(a.dataset.step); }}));
function filter() {{
  const role = document.getElementById('role-filter').value, q = document.getElementById('search').value.toLowerCase();
  let shown = 0;
  rows.forEach(r => {{ const ok = (role === '*' || r.dataset.role === role) && (!q || r.textContent.toLowerCase().includes(q));
    r.hidden = !ok; shown += ok; }});
  document.getElementById('no-match').hidden = shown > 0;
}}
document.getElementById('role-filter').addEventListener('change', filter);
document.getElementById('search').addEventListener('input', filter);
const map = document.getElementById('map'), fit = document.getElementById('fit');
fit.addEventListener('click', () => {{ map.classList.toggle('fit'); fit.textContent = map.classList.contains('fit') ? 'Show actual size' : 'Fit to screen'; }});
</script></body></html>"""
