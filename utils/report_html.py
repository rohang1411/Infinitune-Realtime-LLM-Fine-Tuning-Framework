"""
Offline HTML report generator for InfiniTune evaluation artifacts.

The HTML report is the primary analytical surface. It renders from the shared
presentation spec in the artifact manifest so it no longer infers structure
from filenames or repeats the summary dashboard image.
"""

from __future__ import annotations

import csv
import html
import json
import os
import sys
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.report_utils import build_presentation_spec, presentation_to_dict


def _safe_read_csv_as_rows(csv_path: str) -> List[Dict[str, Any]]:
    if not csv_path or not os.path.isfile(csv_path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _relpath(from_dir: str, path: Optional[str]) -> str:
    if not path:
        return ""
    try:
        return os.path.relpath(path, from_dir).replace("\\", "/")
    except Exception:
        return os.path.basename(path)


def _kpi_to_display(kpi: Dict[str, Any]) -> Dict[str, str]:
    unit = kpi.get("unit", "")
    value = float(kpi.get("value", 0.0))
    status = kpi.get("status", "neutral")
    if unit == "%":
        value_str = f"{value:.1f}%"
    else:
        value_str = f"{value:.3f}" if abs(value) <= 1 else f"{value:.2f}"
    delta_line = kpi.get("delta_display") or ""
    comparison = kpi.get("comparison_basis") or ""
    supporting = kpi.get("supporting_caption") or ""
    return {
        "label": _escape(kpi.get("label", "")),
        "value": _escape(value_str),
        "delta_line": _escape(delta_line),
        "comparison": _escape(comparison),
        "supporting": _escape(supporting),
        "source_note": _escape(kpi.get("source_note", "") or ""),
        "status": _escape(status),
    }


def _metadata_cards_html(cards: List[Dict[str, Any]]) -> str:
    if not cards:
        return ""
    rendered = []
    for card in cards:
        items = []
        for item in card.get("items", []) or []:
            label = _escape(item.get("label", ""))
            value = _escape(item.get("value", ""))
            if not label or not value:
                continue
            items.append(
                '<div class="metaItem">'
                f'  <div class="metaLabel">{label}</div>'
                f'  <div class="metaValue">{value}</div>'
                '</div>'
            )
        if not items:
            continue
        rendered.append(
            '<article class="metaCard">'
            f'  <div class="metaCardHeader"><h3>{_escape(card.get("title", ""))}</h3><p>{_escape(card.get("description", ""))}</p></div>'
            f'  <div class="metaItemGrid">{"".join(items)}</div>'
            '</article>'
        )
    return "".join(rendered)


def _themed_img(artifact_root: str, dark_path: Optional[str], light_path: Optional[str], alt: str, css_class: str = "chartMedia") -> str:
    if not dark_path and not light_path:
        return '<div class="chartFallback">This chart was not generated for this run.</div>'
    dark_rel = _escape(_relpath(artifact_root, dark_path))
    light_rel = _escape(_relpath(artifact_root, light_path))
    src = dark_rel or light_rel
    return f'<img class="{css_class}" src="{src}" data-dark="{dark_rel}" data-light="{light_rel}" alt="{_escape(alt)}" />'


def _load_verbose_text(manifest: Optional[Dict[str, Any]], artifact_root: str) -> str:
    candidates = [
        os.path.join(artifact_root, "verbose_samples.md"),
        os.path.join(os.path.dirname(((manifest or {}).get("metrics_csv_path", "")) or ""), "verbose_samples.md"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                return ""
    return ""


def _presentation_from_inputs(rows: List[Dict[str, Any]], config: Optional[Dict[str, Any]], manifest: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    presentation = ((manifest or {}).get("presentation") or {})
    if presentation:
        return presentation
    return presentation_to_dict(build_presentation_spec(rows, config))


def render_html_report(rows: List[Dict[str, Any]], artifact_root: str, config: Optional[Dict[str, Any]] = None, manifest: Optional[Dict[str, Any]] = None) -> str:
    if not rows:
        return ""

    artifact_root = os.path.abspath(artifact_root)
    os.makedirs(artifact_root, exist_ok=True)
    presentation = _presentation_from_inputs(rows, config, manifest)
    header = presentation.get("header", {})
    profile = presentation.get("profile", {}) or {}
    kpis = presentation.get("kpi_cards", []) or []
    takeaways = presentation.get("takeaway_cards", []) or []
    sections = presentation.get("sections", []) or []
    chart_lookup = {chart.get("id"): chart for chart in (presentation.get("chart_specs", []) or [])}
    metadata_cards = header.get("cards", []) or []
    details_blocks = presentation.get("details_blocks", []) or []
    plotly_js = ""
    has_plotly = False
    try:
        from plotly.offline.offline import get_plotlyjs

        plotly_js = get_plotlyjs()
        has_plotly = True
    except Exception:
        has_plotly = False

    hero_chart = None
    for section_id in ("quantitative", "qualitative", "training"):
        for chart in presentation.get("chart_specs", []) or []:
            if chart.get("section") == section_id and chart.get("role") == "hero":
                hero_chart = chart
                break
        if hero_chart:
            break

    def chart_card(chart: Dict[str, Any], prefer_plotly: bool = True) -> str:
        chart_id = chart.get("id", "")
        title = _escape(chart.get("title", "Chart"))
        subtitle = _escape(chart.get("subtitle", ""))
        dark_path = (chart.get("fallback_paths") or {}).get("dark")
        light_path = (chart.get("fallback_paths") or {}).get("light")
        if prefer_plotly:
            body = f'<div class="plotlyChart" data-chart-id="{_escape(chart_id)}"></div>'
        else:
            body = _themed_img(artifact_root, dark_path, light_path, title)
        return (
            f'<article class="chartCard" data-chart-card="{_escape(chart_id)}">'
            f'  <div class="chartHeader"><h3>{title}</h3><p>{subtitle}</p></div>'
            f"  {body}"
            f"</article>"
        )

    nav_links = ['<a href="#summary">Summary</a>']
    if hero_chart:
        nav_links.append('<a href="#hero">Hero Chart</a>')
    rendered_sections: List[str] = []
    for section in sections:
        section_id = section.get("id", "")
        charts = [chart for chart in section.get("charts", []) if chart.get("id") != (hero_chart or {}).get("id")]
        if not charts:
            continue
        nav_links.append(f'<a href="#{_escape(section_id)}">{_escape(section.get("title", section_id.title()))}</a>')
        cards = "".join(chart_card(chart, prefer_plotly=has_plotly) for chart in charts)
        grid_class = "chartGrid single" if len(charts) == 1 else "chartGrid"
        rendered_sections.append(
            f'<section id="{_escape(section_id)}" class="section">'
            f'  <div class="sectionHeader"><h2>{_escape(section.get("title", ""))}</h2><p>{_escape(section.get("description", ""))}</p></div>'
            f'  <div class="{grid_class}">{cards}</div>'
            f"</section>"
        )
    nav_links.append('<a href="#details">Details</a>')

    primary_kpis = kpis[:4]
    secondary_kpis = kpis[4:6]

    kpi_html = "".join(
        (
            lambda d: (
                f'<article class="kpiCard {d["status"]}">'
                f'  <div class="kpiLabel">{d["label"]}</div>'
                f'  <div class="kpiValue">{d["value"]}</div>'
                f'  <div class="kpiDelta">{d["delta_line"]}</div>'
                f'  <div class="kpiBasis">{d["comparison"]}</div>'
                f'  <div class="kpiCaption">{d["supporting"]}</div>'
                f"</article>"
            )
        )(_kpi_to_display(kpi))
        for kpi in primary_kpis
    )
    def _render_mini_kpi(d: Dict[str, str]) -> str:
        meta = d["delta_line"] or d["comparison"] or d["supporting"]
        submeta = d["comparison"] if d["delta_line"] else (d["supporting"] or d["source_note"])
        extra = f'  <div class="miniKpiSubmeta">{submeta}</div>' if submeta else ""
        return (
            f'<article class="miniKpi {d["status"]}">'
            f'  <div class="miniKpiLabel">{d["label"]}</div>'
            f'  <div class="miniKpiValue">{d["value"]}</div>'
            f'  <div class="miniKpiMeta">{meta}</div>'
            f"{extra}"
            f"</article>"
        )

    secondary_kpi_html = "".join(
        _render_mini_kpi(_kpi_to_display(kpi))
        for kpi in secondary_kpis
    )

    takeaway_html = "".join(
        f'<article class="takeawayCard {_escape(card.get("tone", "neutral"))}">'
        f'  <div class="takeawayTitle">{_escape(card.get("title", ""))}</div>'
        f'  <div class="takeawayBadge">{_escape(card.get("badge", ""))}</div>'
        f'  <p>{_escape(card.get("body", ""))}</p>'
        f"</article>"
        for card in takeaways[:3]
    )

    hero_html = ""
    if hero_chart:
        hero_body = (
            f'<div class="plotlyChart heroPlot" data-chart-id="{_escape(hero_chart.get("id", ""))}"></div>'
            if has_plotly
            else _themed_img(
                artifact_root,
                (hero_chart.get("fallback_paths") or {}).get("dark"),
                (hero_chart.get("fallback_paths") or {}).get("light"),
                hero_chart.get("title", "Hero chart"),
            )
        )
        hero_html = (
            f'<section id="hero" class="section heroSection">'
            f'  <div class="sectionHeader"><h2>{_escape(hero_chart.get("title", ""))}</h2><p>{_escape(hero_chart.get("subtitle", ""))}</p></div>'
            f'  <div class="heroCard">'
            f'    {hero_body}'
            f'  </div>'
            f"</section>"
        )

    verbose_text = _load_verbose_text(manifest, artifact_root)
    config_json = _escape(json.dumps(config or {}, indent=2, sort_keys=True))
    manifest_json = _escape(json.dumps(manifest or {}, indent=2, sort_keys=True))
    extra_details = "".join(
        f'    <details><summary>{_escape(block.get("title", "Validation"))}</summary><ul>'
        + "".join(f"<li>{_escape(item)}</li>" for item in (block.get("items") or []))
        + "</ul></details>"
        for block in details_blocks
        if (block.get("items") or [])
    )
    details_html = (
        '<section id="details" class="section">'
        '  <div class="sectionHeader"><h2>Details</h2><p>Collapsed reproducibility details so the report stays focused on analysis first.</p></div>'
        '  <div class="detailsStack">'
        + extra_details
        +
        f'    <details><summary>Config snapshot</summary><pre>{config_json}</pre></details>'
        +
        f'    <details><summary>Artifact manifest</summary><pre>{manifest_json}</pre></details>'
        + (f'    <details><summary>Verbose evaluation samples</summary><pre>{_escape(verbose_text)}</pre></details>' if verbose_text else "")
        + "  </div></section>"
    )

    summary_copy = "Start with the strongest proof-of-learning signals, then move into the charts that explain why the model improved and where the remaining gaps still are."

    theme_tokens = {
        "dark": {
            "paper": "#101826",
            "plot": "#141E2E",
            "grid": "#27324A",
            "text": "#F6F8FF",
            "subtext": "#A8B4CB",
            "border": "#27324A",
            "lines": {
                "accuracy": "#59C08A",
                "mcc": "#4E9DFF",
                "f1": "#9C87E2",
                "kappa": "#B084EB",
                "perplexity": "#F7C97B",
                "eval_loss": "#F08AA5",
                "loss": "#F08AA5",
                "forgetting": "#FF9770",
                "coverage": "#33C6A6",
                "consistency": "#A48DFF",
                "secondary": "#FF6B6B",
                "response": "#F4A261",
                "semantic": "#6DD3CE",
                "keyword": "#78D686",
                "diversity": "#6DD3CE",
                "neutral": "#86B7FF",
                "warning": "#F7C97B",
            },
        },
        "light": {
            "paper": "#FFFFFF",
            "plot": "#FFFFFF",
            "grid": "#DCE3EF",
            "text": "#142033",
            "subtext": "#51607A",
            "border": "#DCE3EF",
            "lines": {
                "accuracy": "#1C8F52",
                "mcc": "#236CE5",
                "f1": "#7757D8",
                "kappa": "#8D63D5",
                "perplexity": "#B87400",
                "eval_loss": "#CA4F7B",
                "loss": "#CA4F7B",
                "forgetting": "#D86A34",
                "coverage": "#0B9E7C",
                "consistency": "#6B57D8",
                "secondary": "#D14A4A",
                "response": "#C27400",
                "semantic": "#16858A",
                "keyword": "#1C8F52",
                "diversity": "#16858A",
                "neutral": "#236CE5",
                "warning": "#B87400",
            },
        },
    }

    plotly_block = f"<script>{plotly_js}</script>" if has_plotly else ""
    fallback_script = ""
    if not has_plotly:
        fallback_script = """
<script>
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    document.querySelectorAll('img[data-dark][data-light]').forEach((img) => {
      const dark = img.getAttribute('data-dark');
      const light = img.getAttribute('data-light');
      img.src = theme === 'dark' ? (dark || img.src) : (light || img.src);
    });
  }
  function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    applyTheme(current === 'dark' ? 'light' : 'dark');
  }
  applyTheme((window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light');
</script>
"""
    else:
        fallback_script = f"""
<script>
  const PRESENTATION = {json.dumps(presentation, ensure_ascii=False)};
  const THEMES = {json.dumps(theme_tokens, ensure_ascii=False)};
  function resolveColor(themeName, key) {{
    return (THEMES[themeName].lines || {{}})[key] || (THEMES[themeName].lines || {{}}).neutral || '#4E9DFF';
  }}
  function parseLabels(chart) {{
    if (!chart.note) return [];
    try {{
      const parsed = JSON.parse(chart.note);
      return parsed.labels || [];
    }} catch (err) {{
      return [];
    }}
  }}
  function buildFigure(chart, themeName) {{
    const theme = THEMES[themeName];
    const isBar = chart.chart_type === 'barh';
    const baseMargin = isBar ? {{l: 126, r: 40, t: 20, b: 76}} : {{l: 86, r: chart.y2_label ? 96 : 38, t: 20, b: 76}};
    const layout = {{
      paper_bgcolor: theme.paper,
      plot_bgcolor: theme.plot,
      font: {{color: theme.text, family: 'Aptos, Segoe UI Variable, Segoe UI, system-ui, sans-serif'}},
      margin: baseMargin,
      legend: {{orientation: 'h', y: 1.16, x: 0, font: {{color: theme.text}}}},
      hovermode: 'x unified',
      hoverlabel: {{bgcolor: theme.plot, bordercolor: theme.border, font: {{color: theme.text}}}},
      xaxis: {{
        title: {{text: chart.x_label || 'Step', standoff: 18, font: {{color: theme.subtext, size: 13}}}},
        gridcolor: theme.grid,
        zerolinecolor: theme.grid,
        tickfont: {{color: theme.subtext, size: 12}},
        ticklabelposition: 'outside',
        ticks: 'outside',
        ticklen: 6,
        tickcolor: theme.border,
        ticklabeloverflow: 'allow',
        automargin: true
      }},
      yaxis: {{
        title: {{text: chart.y_label || '', standoff: 18, font: {{color: theme.subtext, size: 13}}}},
        gridcolor: theme.grid,
        zerolinecolor: theme.grid,
        tickfont: {{color: theme.subtext, size: 12}},
        ticklabelposition: 'outside',
        ticks: 'outside',
        ticklen: 6,
        tickcolor: theme.border,
        ticklabeloverflow: 'allow',
        automargin: true
      }},
      height: chart.role === 'hero' ? 470 : chart.preferred_aspect === 'square' ? 390 : 400,
    }};
    const data = [];
    if (isBar) {{
      const trace = chart.traces[0] || {{x: [], y: []}};
      const labels = parseLabels(chart);
      const values = trace.x || [];
      const colors = values.map((v) => resolveColor(themeName, trace.color_key || 'coverage'));
      if (values.length > 1) {{
        const minValue = Math.min(...values);
        const minIndex = values.indexOf(minValue);
        colors[minIndex] = resolveColor(themeName, 'secondary');
      }}
      data.push({{
        type: 'bar',
        orientation: 'h',
        x: values,
        y: labels.length ? labels : trace.y,
        marker: {{color: colors}},
        hovertemplate: (chart.x_label || 'Value') + ': %{{x:.2f}}<br>' + (chart.y_label || trace.name || 'Category') + ': %{{y}}<extra></extra>',
      }});
      layout.xaxis.title = {{text: chart.x_label || 'Value', standoff: 18, font: {{color: theme.subtext, size: 13}}}};
      layout.yaxis = {{
        tickfont: {{color: theme.subtext, size: 12}},
        automargin: true,
        ticklabelposition: 'outside',
        ticks: '',
        title: {{text: chart.y_label || '', standoff: 14, font: {{color: theme.subtext, size: 13}}}}
      }};
    }} else {{
      let needsSecondary = false;
      (chart.traces || []).forEach((trace) => {{
        if (trace.axis === 'secondary') needsSecondary = true;
        data.push({{
          type: 'scatter',
          mode: 'lines+markers',
          name: trace.name,
          x: trace.x,
          y: trace.y,
          line: {{
            color: resolveColor(themeName, trace.color_key || 'neutral'),
            width: 3.4,
            dash: trace.style === 'dashed' ? 'dash' : trace.style === 'dotted' ? 'dot' : 'solid',
            shape: trace.style === 'dashed' ? 'linear' : 'spline',
            smoothing: trace.style === 'dashed' ? 0 : 0.72
          }},
          marker: {{size: 6.4}},
          fill: trace.fill ? 'tozeroy' : 'none',
          yaxis: trace.axis === 'secondary' ? 'y2' : 'y',
          hovertemplate: (chart.x_label || 'Step') + ': %{{x}}<br>' + trace.name + ': %{{y:.2f}}<extra></extra>',
        }});
      }});
      if (needsSecondary) {{
        layout.yaxis2 = {{
          title: {{text: chart.y2_label || '', standoff: 18, font: {{color: theme.subtext, size: 13}}}},
          overlaying: 'y',
          side: 'right',
          tickfont: {{color: theme.subtext, size: 12}},
         ticklabelposition: 'outside',
          ticks: 'outside',
          ticklen: 6,
          tickcolor: theme.border,
          ticklabeloverflow: 'allow',
          automargin: true,
          showgrid: false,
        }};
      }}
      layout.shapes = [];
      layout.annotations = [];
      (chart.thresholds || []).forEach((threshold) => {{
        const axis = threshold.axis === 'secondary' ? 'y2' : 'y';
        layout.shapes.push({{
          type: 'line',
          xref: 'paper',
          x0: 0,
          x1: 1,
          yref: axis,
          y0: threshold.value,
          y1: threshold.value,
          line: {{color: resolveColor(themeName, threshold.color_key || 'warning'), dash: threshold.style === 'dashed' ? 'dash' : 'solid', width: 1.2}},
        }});
      }});
    }}
    return {{data, layout}};
  }}
  function renderCharts(themeName) {{
    document.querySelectorAll('.plotlyChart[data-chart-id]').forEach((node) => {{
      const chartId = node.getAttribute('data-chart-id');
      const chart = (PRESENTATION.chart_specs || []).find((item) => item.id === chartId);
      if (!chart) return;
      const fig = buildFigure(chart, themeName);
      Plotly.react(node, fig.data, fig.layout, {{displayModeBar: false, responsive: true}});
    }});
  }}
  function applyTheme(theme) {{
    document.documentElement.setAttribute('data-theme', theme);
    renderCharts(theme);
  }}
  function toggleTheme() {{
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    applyTheme(current === 'dark' ? 'light' : 'dark');
  }}
  applyTheme((window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light');
</script>
"""

    html_out = f"""<!doctype html>
<html data-theme="dark">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>InfiniTune Evaluation Report</title>
  <style>
    :root {{
      --bg: #0B1020;
      --panel: #101826;
      --card: #141E2E;
      --text: #F6F8FF;
      --subtext: #A8B4CB;
      --muted: #7E8AA5;
      --border: #27324A;
      --shadow: 0 22px 60px rgba(0, 0, 0, 0.24);
      --good: #78D686;
      --bad: #FF7F79;
      --warning: #F7C97B;
      --neutral: #86B7FF;
    }}
    [data-theme="light"] {{
      --bg: #F4F7FB;
      --panel: #FFFFFF;
      --card: #FFFFFF;
      --text: #142033;
      --subtext: #51607A;
      --muted: #6D7A92;
      --border: #DCE3EF;
      --shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
      --good: #1C8F52;
      --bad: #D14A4A;
      --warning: #B87400;
      --neutral: #236CE5;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      padding: 24px;
      background: radial-gradient(circle at top, rgba(81, 96, 122, 0.10), transparent 28%), var(--bg);
      color: var(--text);
      font-family: "Aptos", "Segoe UI Variable", "Segoe UI", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      line-height: 1.64;
    }}
    .shell {{ max-width: 1440px; margin: 0 auto; }}
    .topbar {{ display: grid; grid-template-columns: 1fr auto; gap: 18px; align-items: start; margin-bottom: 18px; padding: 22px 24px; background: linear-gradient(160deg, color-mix(in srgb, var(--panel) 90%, transparent), color-mix(in srgb, var(--card) 92%, transparent)); border: 1px solid var(--border); border-radius: 28px; box-shadow: var(--shadow); }}
    .headline h1 {{ margin: 0 0 10px; font-family: "Aptos Display", "Segoe UI Variable Display", "Aptos", "Segoe UI", system-ui, sans-serif; font-size: clamp(34px, 3.6vw, 46px); line-height: 0.98; letter-spacing: -0.045em; }}
    .headline p {{ margin: 0; color: var(--subtext); font-size: 15px; max-width: 920px; line-height: 1.65; }}
    .metaGrid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin: 18px 0 24px; }}
    .metaCard {{ background: linear-gradient(180deg, color-mix(in srgb, var(--panel) 88%, transparent), var(--panel)); border: 1px solid color-mix(in srgb, var(--border) 92%, transparent); border-radius: 26px; padding: 20px; box-shadow: var(--shadow); position: relative; overflow: hidden; }}
    .metaCard::after {{ content: ""; position: absolute; inset: auto -20% 72% 42%; height: 120px; background: radial-gradient(circle, color-mix(in srgb, var(--neutral) 18%, transparent), transparent 66%); opacity: 0.9; pointer-events: none; }}
    .metaCardHeader h3 {{ margin: 0 0 7px; font-family: "Aptos Display", "Segoe UI Variable Display", "Aptos", "Segoe UI", system-ui, sans-serif; font-size: 17px; line-height: 1.15; letter-spacing: -0.02em; }}
    .metaCardHeader p {{ margin: 0 0 15px; color: var(--muted); font-size: 13px; line-height: 1.55; }}
    .metaItemGrid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .metaItem {{ min-width: 0; }}
    .metaLabel {{ color: var(--muted); font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 5px; }}
    .metaValue {{ color: var(--text); font-size: 14px; font-weight: 700; line-height: 1.48; word-break: break-word; }}
    .actions {{ display: flex; gap: 10px; }}
    .btn {{ border: 1px solid var(--border); background: linear-gradient(180deg, color-mix(in srgb, var(--panel) 92%, transparent), var(--panel)); color: var(--text); padding: 11px 15px; border-radius: 14px; cursor: pointer; font-weight: 700; box-shadow: var(--shadow); transition: transform 180ms ease, box-shadow 220ms ease, border-color 180ms ease, background 180ms ease; }}
    .nav {{ position: sticky; top: 10px; z-index: 5; display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 26px; padding: 12px; background: color-mix(in srgb, var(--bg) 78%, transparent); backdrop-filter: blur(10px); border-radius: 18px; }}
    .nav a {{ color: var(--subtext); text-decoration: none; padding: 8px 12px; border-radius: 999px; border: 1px solid var(--border); background: var(--panel); font-size: 13px; font-weight: 700; transition: transform 180ms ease, border-color 180ms ease, color 180ms ease, background 180ms ease; }}
    .summaryGrid {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 18px; margin-bottom: 24px; }}
    .summaryMain {{ grid-column: span 12; }}
    .section {{ margin-bottom: 36px; }}
    .sectionHeader {{ margin-bottom: 16px; }}
    .sectionHeader h2 {{ margin: 0 0 8px; font-family: "Aptos Display", "Segoe UI Variable Display", "Aptos", "Segoe UI", system-ui, sans-serif; font-size: 25px; line-height: 1.08; letter-spacing: -0.03em; }}
    .sectionHeader p {{ margin: 0; color: var(--subtext); font-size: 15px; line-height: 1.65; }}
     .kpiGrid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin-bottom: 16px; }}
     .kpiCard {{ background: linear-gradient(180deg, color-mix(in srgb, var(--card) 86%, transparent), var(--card)); border: 1px solid var(--border); border-radius: 28px; padding: 22px 22px 20px; box-shadow: var(--shadow); position: relative; overflow: hidden; }}
    .kpiCard::after {{ content: ""; position: absolute; inset: auto -12% 58% 44%; height: 160px; background: radial-gradient(circle, color-mix(in srgb, var(--neutral) 16%, transparent), transparent 68%); opacity: 0.95; pointer-events: none; }}
    .kpiCard::before {{ content: ""; position: absolute; inset: 0 auto auto 0; width: 100%; height: 4px; background: color-mix(in srgb, var(--neutral) 88%, transparent); }}
    .kpiCard.good {{ background: linear-gradient(180deg, color-mix(in srgb, var(--good) 11%, var(--card)), var(--card)); border-color: color-mix(in srgb, var(--good) 34%, var(--border)); }}
    .kpiCard.good::before {{ background: color-mix(in srgb, var(--good) 88%, transparent); }}
    .kpiCard.bad {{ background: linear-gradient(180deg, color-mix(in srgb, var(--bad) 10%, var(--card)), var(--card)); border-color: color-mix(in srgb, var(--bad) 34%, var(--border)); }}
    .kpiCard.bad::before {{ background: color-mix(in srgb, var(--bad) 88%, transparent); }}
    .kpiCard.warning {{ background: linear-gradient(180deg, color-mix(in srgb, var(--warning) 12%, var(--card)), var(--card)); border-color: color-mix(in srgb, var(--warning) 34%, var(--border)); }}
    .kpiCard.warning::before {{ background: color-mix(in srgb, var(--warning) 88%, transparent); }}
    .kpiCard.neutral {{ background: linear-gradient(180deg, color-mix(in srgb, var(--neutral) 10%, var(--card)), var(--card)); border-color: color-mix(in srgb, var(--neutral) 34%, var(--border)); }}
    .kpiCard.neutral::before {{ background: color-mix(in srgb, var(--neutral) 88%, transparent); }}
    .kpiLabel {{ color: var(--subtext); font-size: 12px; font-weight: 800; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .kpiValue {{ font-family: "Aptos Display", "Segoe UI Variable Display", "Aptos", "Segoe UI", system-ui, sans-serif; font-size: clamp(36px, 3vw, 46px); line-height: 0.96; letter-spacing: -0.055em; font-weight: 900; margin-bottom: 12px; }}
    .kpiDelta {{ color: var(--text); font-size: 14px; font-weight: 800; margin-bottom: 10px; line-height: 1.56; }}
     .kpiBasis {{ color: var(--subtext); font-size: 12px; font-weight: 700; margin-bottom: 8px; line-height: 1.56; }}
     .kpiCaption {{ color: var(--muted); font-size: 12px; line-height: 1.58; }}
     .miniKpiGrid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-bottom: 22px; }}
     .miniKpi {{ background: color-mix(in srgb, var(--panel) 90%, transparent); border: 1px solid var(--border); border-radius: 22px; padding: 16px 18px; position: relative; overflow: hidden; }}
     .miniKpi::before {{ content: ""; position: absolute; inset: 0 auto auto 0; width: 100%; height: 3px; background: color-mix(in srgb, var(--neutral) 78%, transparent); }}
     .miniKpi.good {{ background: linear-gradient(180deg, color-mix(in srgb, var(--good) 10%, var(--panel)), var(--panel)); border-color: color-mix(in srgb, var(--good) 28%, var(--border)); }}
     .miniKpi.good::before {{ background: color-mix(in srgb, var(--good) 78%, transparent); }}
     .miniKpi.bad {{ background: linear-gradient(180deg, color-mix(in srgb, var(--bad) 10%, var(--panel)), var(--panel)); border-color: color-mix(in srgb, var(--bad) 28%, var(--border)); }}
     .miniKpi.bad::before {{ background: color-mix(in srgb, var(--bad) 78%, transparent); }}
     .miniKpi.warning {{ background: linear-gradient(180deg, color-mix(in srgb, var(--warning) 12%, var(--panel)), var(--panel)); border-color: color-mix(in srgb, var(--warning) 28%, var(--border)); }}
     .miniKpi.warning::before {{ background: color-mix(in srgb, var(--warning) 78%, transparent); }}
     .miniKpi.neutral {{ background: linear-gradient(180deg, color-mix(in srgb, var(--neutral) 10%, var(--panel)), var(--panel)); border-color: color-mix(in srgb, var(--neutral) 28%, var(--border)); }}
     .miniKpi.neutral::before {{ background: color-mix(in srgb, var(--neutral) 78%, transparent); }}
     .miniKpiLabel {{ color: var(--subtext); font-size: 11px; font-weight: 800; margin-bottom: 7px; text-transform: uppercase; letter-spacing: 0.08em; }}
     .miniKpiValue {{ color: var(--text); font-family: "Aptos Display", "Segoe UI Variable Display", "Aptos", "Segoe UI", system-ui, sans-serif; font-size: 28px; line-height: 0.98; letter-spacing: -0.05em; font-weight: 900; margin-bottom: 7px; }}
     .miniKpiMeta {{ color: var(--muted); font-size: 12px; line-height: 1.55; }}
     .miniKpiSubmeta {{ color: var(--subtext); font-size: 11px; line-height: 1.55; margin-top: 6px; }}
    .takeawayGrid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .takeawayCard {{ background: var(--panel); border: 1px solid var(--border); border-radius: 24px; padding: 20px; box-shadow: var(--shadow); position: relative; overflow: hidden; }}
    .takeawayCard::after {{ content: ""; position: absolute; inset: auto -10% 64% 48%; height: 120px; background: radial-gradient(circle, color-mix(in srgb, var(--good) 18%, transparent), transparent 70%); opacity: 0.7; pointer-events: none; }}
    .takeawayTitle {{ font-family: "Aptos Display", "Segoe UI Variable Display", "Aptos", "Segoe UI", system-ui, sans-serif; font-size: 19px; font-weight: 800; margin-bottom: 8px; letter-spacing: -0.02em; }}
    .takeawayBadge {{ display: inline-block; margin-bottom: 12px; padding: 6px 10px; border-radius: 999px; background: color-mix(in srgb, var(--neutral) 18%, var(--panel)); color: var(--text); font-size: 12px; font-weight: 800; }}
    .takeawayCard.good .takeawayBadge {{ background: color-mix(in srgb, var(--good) 18%, var(--panel)); }}
    .takeawayCard.warning .takeawayBadge {{ background: color-mix(in srgb, var(--warning) 22%, var(--panel)); }}
    .takeawayCard.bad .takeawayBadge {{ background: color-mix(in srgb, var(--bad) 18%, var(--panel)); }}
    .takeawayCard p {{ margin: 0; color: var(--subtext); font-size: 15px; line-height: 1.66; }}
    .heroCard, .chartCard {{ background: var(--panel); border: 1px solid var(--border); border-radius: 26px; padding: 18px; box-shadow: var(--shadow); transition: transform 220ms ease, box-shadow 240ms ease, border-color 200ms ease, background 200ms ease; }}
    .chartGrid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; align-items: start; }}
    .chartGrid.single {{ grid-template-columns: minmax(0, 860px); }}
    .chartHeader h3 {{ margin: 0 0 7px; font-family: "Aptos Display", "Segoe UI Variable Display", "Aptos", "Segoe UI", system-ui, sans-serif; font-size: 19px; line-height: 1.14; letter-spacing: -0.02em; }}
    .chartHeader p {{ margin: 0 0 15px; color: var(--subtext); font-size: 14px; line-height: 1.6; }}
     .plotlyChart {{ width: 100%; min-height: 380px; }}
     .heroPlot {{ min-height: 460px; }}
     .chartMedia {{ width: 100%; display: block; border-radius: 18px; }}
    .chartFallback {{ border: 1px dashed var(--border); border-radius: 18px; padding: 24px; color: var(--subtext); background: color-mix(in srgb, var(--subtext) 8%, transparent); }}
    .detailsStack {{ display: grid; gap: 14px; }}
    details {{ background: var(--panel); border: 1px solid var(--border); border-radius: 20px; padding: 14px 16px; box-shadow: var(--shadow); transition: transform 200ms ease, box-shadow 220ms ease, border-color 180ms ease; }}
    summary {{ cursor: pointer; font-weight: 800; }}
    pre {{ margin: 14px 0 0; padding: 14px; border-radius: 16px; border: 1px solid var(--border); background: color-mix(in srgb, var(--panel) 88%, black 2%); color: var(--text); overflow: auto; font-size: 12px; line-height: 1.5; white-space: pre-wrap; }}
    @media (hover: hover) {{
      .metaCard, .kpiCard, .miniKpi, .takeawayCard, .chartCard, .heroCard, details {{
        transition: transform 220ms ease, box-shadow 240ms ease, border-color 200ms ease, background 200ms ease;
      }}
      .metaCard:hover, .kpiCard:hover, .miniKpi:hover, .takeawayCard:hover, .chartCard:hover, .heroCard:hover, details:hover {{
        transform: translateY(-4px);
        box-shadow: 0 28px 70px rgba(0, 0, 0, 0.26);
        border-color: color-mix(in srgb, var(--neutral) 44%, var(--border));
      }}
      .chartCard:hover .chartHeader h3, .heroCard:hover .chartHeader h3, .metaCard:hover .metaCardHeader h3 {{
        color: var(--neutral);
      }}
      .nav a:hover {{
        transform: translateY(-2px);
        border-color: color-mix(in srgb, var(--neutral) 48%, var(--border));
        color: var(--text);
        background: color-mix(in srgb, var(--panel) 80%, var(--neutral) 7%);
      }}
      .btn:hover {{
        transform: translateY(-2px);
        border-color: color-mix(in srgb, var(--neutral) 48%, var(--border));
        background: color-mix(in srgb, var(--panel) 80%, var(--neutral) 7%);
      }}
    }}
     @media (max-width: 980px) {{
       body {{ padding: 16px; }}
       .topbar {{ grid-template-columns: 1fr; }}
       .chartGrid {{ grid-template-columns: 1fr; }}
       .kpiGrid, .takeawayGrid, .miniKpiGrid, .metaGrid, .metaItemGrid {{ grid-template-columns: 1fr; }}
     }}
     @media (max-width: 1260px) {{
       .kpiGrid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
       .metaGrid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
     }}
  </style>
  {plotly_block}
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <div class="headline">
        <h1>{_escape(header.get("product", "InfiniTune"))} Evaluation Report</h1>
        <p>{_escape(header.get("subtitle", ""))} Usecase: {_escape(profile.get("label", "Evaluation"))}. Evidence focus: {_escape(profile.get("evidence_label", ""))}.</p>
      </div>
      <div class="actions">
        <button class="btn" onclick="toggleTheme()">Toggle theme</button>
      </div>
    </div>

    {('<section class="section"><div class="metaGrid">' + _metadata_cards_html(metadata_cards) + '</div></section>') if metadata_cards else ''}

    <nav class="nav">{''.join(nav_links)}</nav>

    <section id="summary" class="section">
      <div class="sectionHeader">
        <h2>Summary</h2>
        <p>{summary_copy}</p>
      </div>
      <div class="kpiGrid">{kpi_html or '<div class="chartFallback">No KPI cards were available for this run.</div>'}</div>
      {'<div class="miniKpiGrid">' + secondary_kpi_html + '</div>' if secondary_kpi_html else ''}
      <div class="takeawayGrid">{takeaway_html or '<div class="chartFallback">No takeaways were generated for this run.</div>'}</div>
    </section>

    {hero_html}
    {''.join(rendered_sections)}
    {details_html}
  </div>
  {fallback_script}
</body>
</html>"""

    out_path = os.path.join(artifact_root, "report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_out)
    return out_path


def generate_html_report(metrics_csv_path: str, out_dir: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> str:
    from utils.evaluation_artifacts import generate_evaluation_artifacts

    run_root = out_dir or os.path.dirname(os.path.abspath(metrics_csv_path))
    manifest = generate_evaluation_artifacts(
        metrics_csv_path=metrics_csv_path,
        run_root=run_root,
        config=config,
        context="standalone",
    )
    return (((manifest or {}).get("generated_files", {}) or {}).get("report", "")) or ""
