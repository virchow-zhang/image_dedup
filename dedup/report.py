"""
交互式 HTML 报告
================
目标：把「哪里重复」标得一眼可见、可放大细看、可对比。

关键设计：
  * 标注层用 **SVG 覆盖层** 而不是把框画进像素里 —— 缩放时框线保持锐利，
    且可点击、可高亮、可开关。
  * 查看器支持 **滚轮缩放 / 拖拽平移**，直接看原始分辨率细节。
  * 每条线索提供 A/B 两处区域的 **高清裁剪对照** 与 **闪烁对比**
    （同一位置交替显示两处内容，是识别「同源」最直观的方式）。
  * 单文件自包含（图片 base64 内嵌），无外部 CDN，双击即可离线打开。
"""

import base64
import html
import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np

MAX_VIEW_PX = 1400      # 报告内嵌原图的最长边
MAX_CROP_PX = 460       # 区域裁剪图的最长边

SEVERITY_LABEL = {'critical': '严重', 'high': '高风险', 'medium': '中等', 'low': '低'}
SEVERITY_COLOR = {'critical': '#e5484d', 'high': '#f76b15',
                  'medium': '#e8b931', 'low': '#46a758'}


def _b64_png(img: np.ndarray, max_dim: int) -> str:
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return ''
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode('ascii')


def _crop(img: np.ndarray, box: List[int]) -> np.ndarray:
    x, y, w, h = [int(v) for v in box]
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return np.zeros((8, 8, 3), np.uint8)
    return img[y0:y1, x0:x1].copy()


def build_intra_item(figure_path: str, finding, panels=None) -> Optional[dict]:
    img = cv2.imread(figure_path)
    if img is None:
        return None
    H, W = img.shape[:2]
    scale = min(1.0, MAX_VIEW_PX / max(H, W))
    view = cv2.resize(img, (int(W * scale), int(H * scale)),
                      interpolation=cv2.INTER_AREA) if scale < 1 else img

    a, b = finding['a_bbox'], finding['b_bbox']
    item = {
        'kind': 'intra',
        'figure': figure_path,
        'name': os.path.basename(figure_path),
        'image': _b64_png(view, MAX_VIEW_PX),
        'w': int(view.shape[1]), 'h': int(view.shape[0]),
        'ow': int(W), 'oh': int(H),
        'scale': round(scale, 6),
        'a': [round(v * scale, 2) for v in a],
        'b': [round(v * scale, 2) for v in b],
        'crop_a': _b64_png(_crop(img, a), MAX_CROP_PX),
        'crop_b': _b64_png(_crop(img, b), MAX_CROP_PX),
        'transform': finding.get('transform', ''),
        'ncc': finding.get('ncc', 0),
        'n_inliers': finding.get('n_inliers', 0),
        'score': finding.get('score', 0),
        'panel_a': finding.get('a_panel'),
        'panel_b': finding.get('b_panel'),
    }
    if panels:
        item['panels'] = [[int(p.x), int(p.y), int(p.w), int(p.h)] for p in panels]
        item['panel_scale'] = round(scale, 6)
    return item


def build_cross_item(match, rel_root: str = '') -> Optional[dict]:
    img1 = cv2.imread(match.image1)
    img2 = cv2.imread(match.image2)
    if img1 is None or img2 is None:
        return None
    return {
        'kind': 'cross',
        'name': f'{os.path.basename(match.image1)}  ↔  {os.path.basename(match.image2)}',
        'image1': _b64_png(img1, MAX_VIEW_PX),
        'image2': _b64_png(img2, MAX_VIEW_PX),
        'path1': match.image1, 'path2': match.image2,
        'severity': match.severity,
        'similarity': match.similarity,
        'confidence': match.confidence,
        'match_type': match.match_type,
        'details': match.details,
    }


# ---------------------------------------------------------------- 模板

_CSS = """
:root{--bg:#0f1115;--card:#171a21;--line:#272b35;--fg:#e6e8ee;--dim:#9aa1b1;
      --a:#4c8dff;--b:#ff7a45;--crit:#e5484d;--high:#f76b15;--mid:#e8b931;--low:#46a758}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.55 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:40;background:rgba(15,17,21,.94);
       backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:14px 22px}
h1{margin:0 0 4px;font-size:18px}
.sub{color:var(--dim);font-size:13px}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px}
.chip{background:var(--card);border:1px solid var(--line);border-radius:999px;
      padding:4px 12px;font-size:12.5px}
.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;align-items:center}
input[type=search],select{background:var(--card);border:1px solid var(--line);
      color:var(--fg);border-radius:8px;padding:6px 10px;font-size:13px}
main{padding:18px 22px 60px;max-width:1500px;margin:0 auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
      margin-bottom:18px;overflow:hidden}
.card>.hd{display:flex;gap:12px;align-items:center;padding:12px 16px;
      border-bottom:1px solid var(--line);flex-wrap:wrap}
.tag{border-radius:6px;padding:2px 9px;font-size:12px;font-weight:600;color:#fff}
.tname{font-weight:600}
.meta{color:var(--dim);font-size:12.5px}
.hd .sp{flex:1}
button{background:#222733;border:1px solid var(--line);color:var(--fg);
      border-radius:8px;padding:5px 11px;font-size:12.5px;cursor:pointer}
button:hover{background:#2c323f}
button.on{background:var(--a);border-color:var(--a);color:#fff}
.body{padding:14px 16px}
.viewer{position:relative;overflow:hidden;background:#0a0c10;border-radius:10px;
        border:1px solid var(--line);height:520px;cursor:grab}
.viewer.drag{cursor:grabbing}
.stage{position:absolute;top:0;left:0;transform-origin:0 0;will-change:transform}
.stage img{display:block;user-select:none;-webkit-user-drag:none}
.stage svg{position:absolute;left:0;top:0;pointer-events:none}
.recta{fill:rgba(76,141,255,.16);stroke:var(--a);stroke-width:2.5;
       vector-effect:non-scaling-stroke}
.rectb{fill:rgba(255,122,69,.16);stroke:var(--b);stroke-width:2.5;
       vector-effect:non-scaling-stroke;stroke-dasharray:7 4}
.prect{fill:none;stroke:#5a6274;stroke-width:1;stroke-dasharray:3 3;
       vector-effect:non-scaling-stroke}
.hint{position:absolute;right:10px;bottom:8px;color:#7b8395;font-size:11.5px;
      background:rgba(10,12,16,.75);padding:3px 8px;border-radius:6px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
.box{background:#0a0c10;border:1px solid var(--line);border-radius:10px;
     padding:8px;text-align:center}
.box .lbl{font-size:12px;color:var(--dim);margin-bottom:6px}
.box img{max-width:100%;display:block;margin:0 auto;border-radius:6px}
.side{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.blink{position:relative;display:inline-block;line-height:0}
.blink img{max-width:100%;display:block;border-radius:6px}
.blink img.b{position:absolute;inset:0}
.empty{color:var(--dim);text-align:center;padding:60px}
.legend{display:flex;gap:14px;font-size:12.5px;color:var(--dim);flex-wrap:wrap}
.legend i{display:inline-block;width:20px;height:0;border-top:3px solid;
          vertical-align:middle;margin-right:5px}
"""

_JS = r"""
function makeViewer(root){
  const stage = root.querySelector('.stage');
  const img = stage.querySelector('img');
  let s = 1, tx = 0, ty = 0, dragging = false, px = 0, py = 0;
  function apply(){
    stage.style.transform = `translate(${tx}px,${ty}px) scale(${s})`;
  }
  function fit(){
    const b = root.getBoundingClientRect();
    const iw = img.naturalWidth || img.width, ih = img.naturalHeight || img.height;
    s = Math.min(b.width / iw, b.height / ih) * 0.98;
    tx = (b.width - iw * s) / 2; ty = (b.height - ih * s) / 2;
    apply();
  }
  root.addEventListener('wheel', e => {
    e.preventDefault();
    const b = root.getBoundingClientRect();
    const mx = e.clientX - b.left, my = e.clientY - b.top;
    const k = e.deltaY < 0 ? 1.15 : 1/1.15;
    const ns = Math.min(30, Math.max(0.05, s * k));
    tx = mx - (mx - tx) * (ns / s);
    ty = my - (my - ty) * (ns / s);
    s = ns; apply();
  }, {passive:false});
  root.addEventListener('mousedown', e => {
    dragging = true; px = e.clientX; py = e.clientY; root.classList.add('drag');
  });
  window.addEventListener('mousemove', e => {
    if(!dragging) return;
    tx += e.clientX - px; ty += e.clientY - py;
    px = e.clientX; py = e.clientY; apply();
  });
  window.addEventListener('mouseup', () => { dragging = false; root.classList.remove('drag'); });
  root.querySelectorAll('button[data-act]').forEach(b => {
    b.addEventListener('click', () => {
      const a = b.dataset.act;
      if(a === 'fit'){ fit(); }
      else if(a === 'reset'){ s = 1; tx = 0; ty = 0; apply(); }
      else if(a === 'zoomin'){ s = Math.min(30, s*1.3); apply(); }
      else if(a === 'zoomout'){ s = Math.max(0.05, s/1.3); apply(); }
      else if(a === 'toggleA' || a === 'toggleB'){
        const el = root.querySelector('.' + (a === 'toggleA' ? 'recta' : 'rectb'));
        if(el){ const on = el.style.display !== 'none'; el.style.display = on ? 'none' : ''; b.classList.toggle('on', !on); }
      }
    });
  });
  if(img.complete) fit(); else img.addEventListener('load', fit);
  window.addEventListener('resize', () => { /* keep user transform */ });
  return {fit};
}
function makeBlink(root){
  const a = root.querySelector('img.a'), b = root.querySelector('img.b');
  if(!a || !b) return;
  let on = false, timer = null;
  const btn = root.closest('.body').querySelector('button[data-blink]');
  function step(){ on = !on; a.style.visibility = on ? 'hidden' : 'visible'; }
  if(btn) btn.addEventListener('click', () => {
    if(timer){ clearInterval(timer); timer = null; a.style.visibility='visible';
               btn.classList.remove('on'); btn.textContent='闪烁对比'; }
    else { timer = setInterval(step, 600); btn.classList.add('on');
           btn.textContent='停止闪烁'; }
  });
}
document.querySelectorAll('.viewer').forEach(makeViewer);
document.querySelectorAll('.blink').forEach(makeBlink);

// 过滤
const q = document.getElementById('q');
const sev = document.getElementById('sev');
function refilter(){
  const t = (q.value || '').toLowerCase();
  const s = sev ? sev.value : '';
  let shown = 0;
  document.querySelectorAll('.card').forEach(c => {
    const okT = !t || (c.dataset.name || '').toLowerCase().includes(t);
    const okS = !s || (c.dataset.sev || '') === s;
    const vis = okT && okS;
    c.style.display = vis ? '' : 'none';
    if(vis) shown++;
  });
  document.getElementById('shown').textContent = shown;
}
if(q) q.addEventListener('input', refilter);
if(sev) sev.addEventListener('change', refilter);
document.addEventListener('keydown', e => { if(e.key === 'Escape') refilter(); });
"""

_VIEWER = """
<div class="viewer">
  <div class="stage">
    <img src="__IMG__" width="__IW__" height="__IH__">
    <svg width="__IW__" height="__IH__" viewBox="0 0 __IW__ __IH__">__SHAPES__</svg>
  </div>
  <div class="toolbar" style="position:absolute;left:10px;top:10px;margin:0">
    <button data-act="fit">适应</button>
    <button data-act="reset">1:1</button>
    <button data-act="zoomin">＋</button>
    <button data-act="zoomout">－</button>
    <button data-act="toggleA" class="on">A 区</button>
    <button data-act="toggleB" class="on">B 区</button>
  </div>
  <div class="hint">滚轮缩放 · 拖拽平移</div>
</div>
"""


def _rects(a, b, panels, pscale) -> str:
    out = []
    if panels:
        for p in panels:
            out.append(f'<rect class="prect" x="{p[0]*pscale:.1f}" y="{p[1]*pscale:.1f}" '
                       f'width="{p[2]*pscale:.1f}" height="{p[3]*pscale:.1f}"/>')
    for box, cls in ((a, 'recta'), (b, 'rectb')):
        out.append(f'<rect class="{cls}" rx="3" x="{box[0]:.1f}" y="{box[1]:.1f}" '
                   f'width="{box[2]:.1f}" height="{box[3]:.1f}"/>')
    return ''.join(out)


def _intra_card(it: dict) -> str:
    shapes = _rects(it['a'], it['b'], it.get('panels'), it.get('panel_scale', 1.0))
    viewer = (_VIEWER.replace('__IMG__', it['image'])
                      .replace('__IW__', str(it['w'])).replace('__IH__', str(it['h']))
                      .replace('__SHAPES__', shapes))
    pa, pb = it.get('panel_a'), it.get('panel_b')
    where = ''
    if pa is not None and pb is not None:
        where = f'<span class="meta">区域归属: P{pa} ↔ P{pb}</span>'
    return f"""
<div class="card" data-name="{html.escape(it['name'])}" data-sev="critical">
  <div class="hd">
    <span class="tag" style="background:{SEVERITY_COLOR['critical']}">图内复用</span>
    <span class="tname">{html.escape(it['name'])}</span>
    <span class="meta">变换: {html.escape(str(it['transform']))} · 
      相似度 {it['ncc']:.3f} · 内点 {it['n_inliers']}</span>
    {where}
    <span class="sp"></span>
    <span class="meta">原图 {it['ow']}×{it['oh']}</span>
  </div>
  <div class="body">
    <div class="legend">
      <span><i style="border-color:#4c8dff"></i>A 区</span>
      <span><i style="border-color:#ff7a45;border-top-style:dashed"></i>B 区</span>
      <span><i style="border-color:#5a6274;border-top-style:dotted"></i>切分出的 panel</span>
    </div>
    {viewer}
    <div class="pair">
      <div class="box"><div class="lbl">A 区（原始分辨率裁剪）</div>
        <img src="{it['crop_a']}"></div>
      <div class="box"><div class="lbl">B 区（原始分辨率裁剪）</div>
        <img src="{it['crop_b']}"></div>
    </div>
    <div style="margin-top:12px">
      <button data-blink>闪烁对比</button>
      <span class="meta">（在同一位置交替显示 A/B 两处内容，同源时几乎看不出切换）</span>
    </div>
    <div class="blink" style="margin-top:8px;max-width:{MAX_CROP_PX}px">
      <img class="a" src="{it['crop_a']}">
      <img class="b" src="{it['crop_b']}">
    </div>
  </div>
</div>"""


def _cross_card(it: dict) -> str:
    color = SEVERITY_COLOR.get(it['severity'], '#888')
    return f"""
<div class="card" data-name="{html.escape(it['name'])}" data-sev="{it['severity']}">
  <div class="hd">
    <span class="tag" style="background:{color}">{SEVERITY_LABEL.get(it['severity'], '?')}</span>
    <span class="tname">{html.escape(it['name'])}</span>
    <span class="meta">{html.escape(it['match_type'])} · 相似度 {it['similarity']*100:.1f}% ·
      置信度 {it['confidence']*100:.1f}%</span>
  </div>
  <div class="body">
    <div class="meta" style="margin-bottom:8px">{html.escape(it['details'])}</div>
    <div class="side">
      <div class="box"><div class="lbl">{html.escape(os.path.basename(it['path1']))}</div>
        <img src="{it['image1']}"></div>
      <div class="box"><div class="lbl">{html.escape(os.path.basename(it['path2']))}</div>
        <img src="{it['image2']}"></div>
    </div>
  </div>
</div>"""


# ---------------------------------------------------------------- 组装


def generate_report(intra_items: List[dict], cross_items: List[dict],
                    output_path: str, meta: Dict = None) -> str:
    meta = meta or {}
    # 图内复用放在最前：这是 v3 的核心新能力，也是最容易被人眼忽略的一类问题
    cards = ''.join(_intra_card(i) for i in intra_items) + \
            ''.join(_cross_card(i) for i in cross_items)
    if not cards:
        cards = '<div class="empty">未发现可疑重复。</div>'

    n_intra = len(intra_items)
    n_cross = len(cross_items)
    sev_opts = ''.join(f'<option value="{k}">{v}</option>'
                       for k, v in SEVERITY_LABEL.items())

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>科研图片查重报告</title><style>{_CSS}</style></head>
<body>
<header>
  <h1>🔬 科研图片查重报告</h1>
  <div class="sub">{html.escape(str(meta.get('directory', '')))} ·
      扫描 {meta.get('n_images', 0)} 张 · 耗时 {meta.get('elapsed', 0)}s</div>
  <div class="stats">
    <span class="chip">图内复用/重叠 <b>{n_intra}</b> 处</span>
    <span class="chip">跨文件重复 <b>{n_cross}</b> 对</span>
    <span class="chip">当前显示 <b id="shown">{n_intra + n_cross}</b></span>
  </div>
  <div class="toolbar">
    <input type="search" id="q" placeholder="按文件名过滤…" style="min-width:240px">
    <select id="sev"><option value="">全部严重程度</option>{sev_opts}</select>
  </div>
</header>
<main>{cards}</main>
<script>{_JS}</script>
</body></html>"""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(doc)
    return output_path
