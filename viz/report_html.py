import os
import csv
import html
import time
from datetime import datetime

from detectors.base import MatchResult
from core.loader import ImageInfo


def generate_report(matches: list[MatchResult], all_infos: list[ImageInfo],
                    output_dir: str, scan_directory: str, config: dict,
                    vis_dir: str = "vis"):
    report_dir = os.path.join(output_dir, "report")
    vis_abs = os.path.join(report_dir, vis_dir)
    os.makedirs(vis_abs, exist_ok=True)

    total_images = len(all_infos)
    total_pairs = total_images * (total_images - 1) // 2
    sev_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    type_counts = {}
    for m in matches:
        sev_counts[m.severity] = sev_counts.get(m.severity, 0) + 1
        type_counts[m.match_type] = type_counts.get(m.match_type, 0) + 1

    rows = []
    for i, m in enumerate(matches):
        vis_filename = f"{i + 1:04d}_{m.severity.upper()[:3]}_{os.path.basename(m.image1)[:20]}_vs_{os.path.basename(m.image2)[:20]}.jpg"
        vis_rel = f"{vis_dir}/{vis_filename}"
        rows.append((m, vis_rel, vis_abs, vis_filename))

    html_content = _build_html(rows, scan_directory, total_images, total_pairs,
                               sev_counts, type_counts, config)
    csv_path = os.path.join(report_dir, "comparisons.csv")
    _write_csv(rows, csv_path)

    index_path = os.path.join(report_dir, "index.html")
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return index_path, csv_path


def _build_html(rows, scan_dir, total_images, total_pairs,
                sev_counts, type_counts, config) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_matches = len(rows)
    scan_dir_esc = html.escape(scan_dir)

    sev_bars = ""
    for sev in ["critical", "high", "medium"]:
        cnt = sev_counts.get(sev, 0)
        pct = (cnt / max(total_matches, 1)) * 100
        color = {"critical": "#dc3545", "high": "#e67e22", "medium": "#f1c40f"}[sev]
        label = {"critical": "严重", "high": "高", "medium": "中"}[sev]
        sev_bars += f"""
        <div class="sev-row">
          <span class="sev-label">{label}</span>
          <div class="sev-bar-bg"><div class="sev-bar" style="width:{pct}%;background:{color}"></div></div>
          <span class="sev-count">{cnt}</span>
        </div>"""

    type_items = "".join(
        f'<span class="type-tag">{html.escape(t)} ({c})</span>'
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1])
    )

    card_html = ""
    for i, (m, vis_rel, vis_abs, vis_fn) in enumerate(rows):
        card_html += _build_card(m, vis_rel, i)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>科研图片查重报告</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, 'Segoe UI', sans-serif; background: #f5f6fa; color: #2c3e50; padding: 20px; }}
.container {{ max-width: 1400px; margin: 0 auto; }}
.header {{ background: linear-gradient(135deg, #2c3e50, #34495e); color: white; padding: 28px 32px; border-radius: 12px; margin-bottom: 20px; }}
.header h1 {{ font-size: 24px; margin-bottom: 6px; }}
.header .meta {{ font-size: 13px; opacity: 0.8; line-height: 1.8; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 20px; }}
.stat-card {{ background: white; padding: 18px; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); text-align: center; }}
.stat-card .num {{ font-size: 28px; font-weight: 700; }}
.stat-card .lab {{ font-size: 12px; color: #888; margin-top: 4px; }}
.stat-card.critical .num {{ color: #dc3545; }}
.stat-card.high .num {{ color: #e67e22; }}
.stat-card.medium .num {{ color: #f1c40f; }}
.sev-summary {{ background: white; padding: 18px 22px; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 20px; }}
.sev-row {{ display: flex; align-items: center; margin: 6px 0; }}
.sev-label {{ width: 40px; font-size: 13px; }}
.sev-bar-bg {{ flex: 1; height: 18px; background: #ecf0f1; border-radius: 9px; margin: 0 10px; overflow: hidden; }}
.sev-bar {{ height: 100%; border-radius: 9px; transition: width 0.6s; min-width: 4px; }}
.sev-count {{ width: 30px; text-align: right; font-weight: 600; font-size: 14px; }}
.type-tags {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
.type-tag {{ background: #ecf0f1; padding: 4px 12px; border-radius: 12px; font-size: 12px; }}
.filters {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; align-items: center; }}
.filter-btn {{ padding: 6px 16px; border: 1px solid #ddd; border-radius: 16px; background: white; cursor: pointer; font-size: 13px; transition: 0.15s; }}
.filter-btn:hover {{ background: #eef; }}
.filter-btn.active {{ background: #3498db; color: white; border-color: #3498db; }}
.filter-btn.critical.active {{ background: #dc3545; border-color: #dc3545; }}
.filter-btn.high.active {{ background: #e67e22; border-color: #e67e22; }}
.filter-btn.medium.active {{ background: #f1c40f; border-color: #f1c40f; color: #333; }}
.cross-toggle {{ margin-left: auto; font-size: 13px; }}
.card {{ background: white; border-radius: 12px; box-shadow: 0 1px 6px rgba(0,0,0,0.08); margin-bottom: 16px; overflow: hidden; }}
.card-header {{ display: flex; align-items: center; padding: 12px 18px; cursor: pointer; user-select: none; }}
.card-header .badge {{ padding: 3px 12px; border-radius: 10px; font-size: 11px; font-weight: 700; color: white; margin-right: 12px; flex-shrink: 0; }}
.badge-critical {{ background: #dc3545; }}
.badge-high {{ background: #e67e22; }}
.badge-medium {{ background: #f1c40f; color: #333; }}
.card-header .fnames {{ font-size: 13px; font-weight: 500; flex: 1; }}
.card-header .sim {{ font-size: 12px; color: #888; margin-left: 12px; }}
.card-header .type-tag-sm {{ font-size: 11px; color: #888; margin-left: 8px; }}
.card-body {{ display: none; padding: 0 18px 18px; }}
.card.expanded .card-body {{ display: block; }}
.comp-img {{ width: 100%; border-radius: 6px; cursor: pointer; }}
.details {{ display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; font-size: 13px; margin-top: 12px; padding: 12px; background: #f8f9fa; border-radius: 6px; }}
.details .key {{ color: #888; white-space: nowrap; }}
.details .val {{ word-break: break-all; }}
.lightbox {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); z-index: 9999; justify-content: center; align-items: center; }}
.lightbox.show {{ display: flex; }}
.lightbox img {{ max-width: 95%; max-height: 95%; border-radius: 4px; }}
.lightbox .close {{ position: absolute; top: 20px; right: 30px; color: white; font-size: 36px; cursor: pointer; }}
.expand-icon {{ margin-left: auto; font-size: 14px; color: #bbb; }}
.cross-badge {{ background: #fff3cd; color: #856404; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-left: 8px; }}
@media(max-width:768px) {{ .header h1 {{ font-size: 18px; }} .stat-card .num {{ font-size: 22px; }} }}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>🔬 科研图片查重报告</h1>
  <div class="meta">
    扫描目录: <code>{scan_dir_esc}</code><br>
    图片: {total_images} 张 &nbsp;|&nbsp; 比对: {total_pairs:,} 对 &nbsp;|&nbsp; 疑似重复: {total_matches} 对<br>
    生成时间: {ts}
  </div>
</div>

<div class="stats">
  <div class="stat-card"><div class="num">{total_images}</div><div class="lab">总图片数</div></div>
  <div class="stat-card"><div class="num">{total_pairs:,}</div><div class="lab">总比对对数</div></div>
  <div class="stat-card critical"><div class="num">{sev_counts.get("critical", 0)}</div><div class="lab">严重 (Critical)</div></div>
  <div class="stat-card high"><div class="num">{sev_counts.get("high", 0)}</div><div class="lab">高 (High)</div></div>
  <div class="stat-card medium"><div class="num">{sev_counts.get("medium", 0)}</div><div class="lab">中 (Medium)</div></div>
</div>

<div class="sev-summary">
  <div style="font-weight:600;margin-bottom:6px;">严重程度分布</div>
  {sev_bars}
  <div class="type-tags">{type_items}</div>
</div>

<div class="filters">
  <button class="filter-btn active" data-filter="all">全部 ({total_matches})</button>
  <button class="filter-btn critical active" data-filter="critical">严重 ({sev_counts.get("critical",0)})</button>
  <button class="filter-btn high active" data-filter="high">高 ({sev_counts.get("high",0)})</button>
  <button class="filter-btn medium active" data-filter="medium">中 ({sev_counts.get("medium",0)})</button>
  <label class="cross-toggle"><input type="checkbox" id="crossToggle" checked> 显示跨通道比对</label>
</div>

<div id="cardList">
{card_html}
</div>

</div>

<div class="lightbox" id="lightbox" onclick="this.classList.remove('show')">
  <span class="close">&times;</span>
  <img id="lightboxImg" src="">
</div>

<script>
document.querySelectorAll('.filter-btn').forEach(btn => {{
  btn.onclick = () => {{
    btn.classList.toggle('active');
    applyFilters();
  }};
}});
document.getElementById('crossToggle').onchange = applyFilters;

function applyFilters() {{
  const showSev = new Set();
  document.querySelectorAll('.filter-btn.active').forEach(b => showSev.add(b.dataset.filter));
  const showCross = document.getElementById('crossToggle').checked;
  document.querySelectorAll('.card').forEach(c => {{
    const sev = c.dataset.severity;
    const cross = c.dataset.cross === 'true';
    const matchSev = showSev.has('all') || showSev.has(sev);
    const matchCross = showCross || !cross;
    c.style.display = (matchSev && matchCross) ? '' : 'none';
  }});
}}

document.querySelectorAll('.card-header').forEach(h => {{
  h.onclick = () => h.parentElement.classList.toggle('expanded');
}});

document.querySelectorAll('.comp-img').forEach(img => {{
  img.onclick = (e) => {{
    e.stopPropagation();
    document.getElementById('lightboxImg').src = img.src;
    document.getElementById('lightbox').classList.add('show');
  }};
}});
</script>
</body>
</html>"""


def _build_card(m: MatchResult, vis_rel: str, idx: int) -> str:
    sev = m.severity
    badge_cls = f"badge-{sev}" if sev in ("critical", "high", "medium") else "badge-medium"
    sev_label = {"critical": "严重", "high": "高", "medium": "中", "low": "低"}.get(sev, sev)
    cross_attr = 'true' if m.is_cross_channel else 'false'
    cross_tag = '<span class="cross-badge">跨通道</span>' if m.is_cross_channel else ''

    fn1 = html.escape(os.path.basename(m.image1))
    fn2 = html.escape(os.path.basename(m.image2))
    dir1 = html.escape(os.path.dirname(m.image1))
    dir2 = html.escape(os.path.dirname(m.image2))
    mtype = html.escape(m.match_type)
    mdetails = html.escape(m.details)

    return f"""<div class="card" data-severity="{sev}" data-cross="{cross_attr}">
  <div class="card-header">
    <span class="badge {badge_cls}">{sev_label}</span>
    <span class="fnames">{fn1}  vs  {fn2}{cross_tag}</span>
    <span class="sim">{m.similarity * 100:.1f}%</span>
    <span class="type-tag-sm">{mtype}</span>
    <span class="expand-icon">▼</span>
  </div>
  <div class="card-body">
    <img class="comp-img" src="{html.escape(vis_rel)}" alt="comparison" loading="lazy">
    <div class="details">
      <span class="key">匹配类型</span><span class="val">{mtype}</span>
      <span class="key">相似度</span><span class="val">{m.similarity * 100:.2f}%</span>
      <span class="key">路径 A</span><span class="val">{dir1}</span>
      <span class="key">路径 B</span><span class="val">{dir2}</span>
      <span class="key">文件名 A</span><span class="val">{fn1}</span>
      <span class="key">文件名 B</span><span class="val">{fn2}</span>
      <span class="key">详细</span><span class="val">{mdetails}</span>
    </div>
  </div>
</div>"""


def _write_csv(rows, csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(["序号", "严重程度", "匹配类型", "相似度", "跨通道",
                     "文件A", "目录A", "文件B", "目录B", "详细"])
        for i, (m, vis_rel, _, _) in enumerate(rows, 1):
            w.writerow([
                i, m.severity, m.match_type, m.similarity,
                "是" if m.is_cross_channel else "否",
                os.path.basename(m.image1), os.path.dirname(m.image1),
                os.path.basename(m.image2), os.path.dirname(m.image2),
                m.details,
            ])
