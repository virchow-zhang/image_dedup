import os
import cv2
import numpy as np
from PIL import Image
from detectors.base import MatchResult


RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_BG = (45, 45, 45)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
CYAN = (255, 255, 0)


def _imread_unicode(path: str):
    import numpy as np
    import cv2
    with open(path, 'rb') as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _imwrite_unicode(path: str, img: np.ndarray, params=None):
    import cv2
    success, buf = cv2.imencode('.jpg', img, params or [cv2.IMWRITE_JPEG_QUALITY, 92])
    if success:
        with open(path, 'wb') as f:
            f.write(buf.tobytes())
        return True
    return False


def annotate_pair(match: MatchResult, output_path: str, thumb_height: int = 500) -> str:
    img_a = _imread_unicode(match.image1)
    img_b = _imread_unicode(match.image2)
    if img_a is None or img_b is None:
        return _fallback_pil(match, output_path, thumb_height)

    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    h1, w1 = img_a.shape[:2]
    h2, w2 = img_b.shape[:2]
    scale1 = thumb_height / h1
    scale2 = thumb_height / h2
    disp_w1 = int(w1 * scale1)
    disp_w2 = int(w2 * scale2)

    disp1 = cv2.resize(img_a, (disp_w1, thumb_height))
    disp2 = cv2.resize(img_b, (disp_w2, thumb_height))

    has_match_points = (match.match_points1 is not None and len(match.match_points1) > 0
                        and match.match_points2 is not None and len(match.match_points2) > 0)

    if has_match_points:
        _draw_feature_matches(disp1, disp2, match, scale1, scale2)
    elif "子图" in match.match_type or "裁剪" in match.match_type:
        _draw_subimage(gray_a, gray_b, disp1, disp2, match, scale1, scale2)
    elif "边缘" in match.match_type:
        _draw_edge_overlap(gray_a, gray_b, disp1, disp2, match, scale1, scale2)
    else:
        _draw_diff_highlight(gray_a, gray_b, disp1, disp2, match, scale1, scale2)

    canvas = _compose_canvas(disp1, disp2, match, thumb_height)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _imwrite_unicode(output_path, canvas)
    return output_path


def _draw_feature_matches(disp1, disp2, match: MatchResult, s1: float, s2: float):
    h1, w1 = disp1.shape[:2]
    h2, w2 = disp2.shape[:2]
    pts1 = np.float32([(int(x * s1), int(y * s1)) for x, y in match.match_points1]).reshape(-1, 1, 2)
    pts2 = np.float32([(int(x * s2), int(y * s2)) for x, y in match.match_points2]).reshape(-1, 1, 2)

    n = min(len(pts1), len(pts2), 100)
    top_n = min(n, 10)

    for i in range(n):
        x1, y1 = int(pts1[i][0][0]), int(pts1[i][0][1])
        x2, y2 = int(pts2[i][0][0]), int(pts2[i][0][1])

        if i < top_n:
            thickness = 2
            color = RED
        else:
            thickness = 1
            color = (0, 0, 180)

        cv2.line(disp1, (x1, y1), (x2 + w1, y2), color, thickness)
        cv2.circle(disp1, (x1, y1), 3, WHITE, -1)
        cv2.circle(disp1, (x1, y1), 3, RED, 1)
        cv2.circle(disp2, (x2, y2), 3, WHITE, -1)
        cv2.circle(disp2, (x2, y2), 3, RED, 1)

    if len(pts1) > 5:
        xs1 = [int(p[0][0]) for p in pts1]
        ys1 = [int(p[0][1]) for p in pts1]
        xs2 = [int(p[0][0]) for p in pts2]
        ys2 = [int(p[0][1]) for p in pts2]
        _draw_bounding_box(disp1, min(xs1), min(ys1), max(xs1) - min(xs1), max(ys1) - min(ys1))
        _draw_bounding_box(disp2, min(xs2), min(ys2), max(xs2) - min(xs2), max(ys2) - min(ys2))


def _draw_bounding_box(img, x, y, w, h):
    if w > 5 and h > 5:
        cv2.rectangle(img, (x, y), (x + w, y + h), RED, 2)
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(img, (cx, cy), 4, YELLOW, -1)


def _draw_subimage(gray_a, gray_b, disp1, disp2, match: MatchResult, s1: float, s2: float):
    try:
        big = disp1 if gray_a.shape[0] * gray_a.shape[1] >= gray_b.shape[0] * gray_b.shape[1] else disp2
        small = disp2 if big is disp1 else disp1

        big_gray = gray_a if big is disp1 else gray_b
        small_gray = gray_b if big is disp1 else gray_a

        sw, sh = small_gray.shape[1], small_gray.shape[0]
        s_scale = thumb_height = max(disp1.shape[0], disp2.shape[0])
        _sb = cv2.resize(small_gray, (int(sw * s_scale / sh), thumb_height))

        res = cv2.matchTemplate(big_gray, _sb, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > 0.7:
            x, y = max_loc
            bw, bh = _sb.shape[1], _sb.shape[0]
            scale_big = s1 if big is disp1 else s2
            rx, ry = int(x * scale_big), int(y * scale_big)
            rw, rh = int(bw * scale_big), int(bh * scale_big)
            cv2.rectangle(big, (rx, ry), (rx + rw, ry + rh), RED, 3)
            cv2.rectangle(small, (5, 5), (small.shape[1] - 5, small.shape[0] - 5), RED, 3)
            _draw_center_line(big, small, rx + rw // 2, ry + rh // 2, small.shape[1] // 2, small.shape[0] // 2)
    except Exception:
        cv2.rectangle(disp1, (5, 5), (disp1.shape[1] - 5, disp1.shape[0] - 5), RED, 2)
        cv2.rectangle(disp2, (5, 5), (disp2.shape[1] - 5, disp2.shape[0] - 5), RED, 2)


def _draw_center_line(img1, img2, x1, y1, x2, y2):
    h1 = img1.shape[0]
    cv2.line(img1, (x1, y1), (x2 + img1.shape[1], y2), RED, 2, cv2.LINE_AA)


def _draw_edge_overlap(gray_a, gray_b, disp1, disp2, match, s1: float, s2: float):
    h = min(disp1.shape[0], disp2.shape[0])
    w1, w2 = disp1.shape[1], disp2.shape[1]

    strip_w = min(30, w1 // 6, w2 // 6)
    if strip_w < 5:
        strip_w = 5

    overlay = disp1.copy()
    cv2.rectangle(overlay, (w1 - strip_w, 0), (w1 - 1, h - 1), (0, 100, 255), -1)
    cv2.addWeighted(overlay, 0.35, disp1, 0.65, 0, disp1)
    cv2.rectangle(disp1, (w1 - strip_w, 0), (w1 - 1, h - 1), (0, 100, 255), 2)
    cv2.putText(disp1, "重叠边缘", (w1 - strip_w, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

    overlay2 = disp2.copy()
    cv2.rectangle(overlay2, (0, 0), (strip_w - 1, h - 1), (0, 100, 255), -1)
    cv2.addWeighted(overlay2, 0.35, disp2, 0.65, 0, disp2)
    cv2.rectangle(disp2, (0, 0), (strip_w - 1, h - 1), (0, 100, 255), 2)
    cv2.putText(disp2, "重叠边缘", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

    num_lines = 4
    for i in range(num_lines):
        y = int(h * (i + 1) / (num_lines + 1))
        cv2.line(disp1, (w1 - strip_w - 5, y), (w1 - 1, y), (0, 200, 255), 1, cv2.LINE_AA)
        cv2.line(disp2, (0, y), (strip_w + 5, y), (0, 200, 255), 1, cv2.LINE_AA)
        cv2.line(disp1, (w1 - 1, y), (w2, y), (0, 100, 255), 1, cv2.LINE_AA)

    bracket_y1 = int(h * 0.15)
    bracket_y2 = int(h * 0.85)
    edge_x = w1
    for y in [bracket_y1, bracket_y2]:
        cv2.line(disp1, (w1 - 3, y), (w1 + 3, y), (0, 255, 255), 2)
        cv2.line(disp2, (w2 - 3, y), (w2 + 3, y), (0, 255, 255), 2)

    _draw_center_line(disp1, disp2, w1 - strip_w // 2, h // 2, strip_w // 2, h // 2)


def _draw_diff_highlight(gray_a, gray_b, disp1, disp2, match, s1: float, s2: float):
    try:
        h, w = 400, 600
        ra = cv2.resize(gray_a, (w, h))
        rb = cv2.resize(gray_b, (w, h))
        diff = cv2.absdiff(ra, rb)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        kernel = np.ones((7, 7), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=3)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_pts = np.vstack(contours)
            x, y, bw, bh = cv2.boundingRect(all_pts)
            if bw > 20 and bh > 20:
                sx1 = int(x * disp1.shape[1] / w)
                sy1 = int(y * disp1.shape[0] / h)
                sw1 = int(bw * disp1.shape[1] / w)
                sh1 = int(bh * disp1.shape[0] / h)
                cv2.rectangle(disp1, (sx1, sy1), (sx1 + sw1, sy1 + sh1), RED, 2)
                sx2 = int(x * disp2.shape[1] / w)
                sy2 = int(y * disp2.shape[0] / h)
                sw2 = int(bw * disp2.shape[1] / w)
                sh2 = int(bh * disp2.shape[0] / h)
                cv2.rectangle(disp2, (sx2, sy2), (sx2 + sw2, sy2 + sh2), RED, 2)
                _draw_center_line(disp1, disp2, sx1 + sw1 // 2, sy1 + sh1 // 2, sx2 + sw2 // 2, sy2 + sh2 // 2)
                return
    except Exception:
        pass
    cv2.rectangle(disp1, (5, 5), (disp1.shape[1] - 5, disp1.shape[0] - 5), RED, 2)
    cv2.rectangle(disp2, (5, 5), (disp2.shape[1] - 5, disp2.shape[0] - 5), RED, 2)


def _compose_canvas(disp1, disp2, match: MatchResult, thumb_height: int) -> np.ndarray:
    header_h = 90
    gap = 12
    footer_h = 80
    img_w1, img_w2 = disp1.shape[1], disp2.shape[1]
    total_w = img_w1 + gap + img_w2
    total_h = header_h + thumb_height + footer_h

    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    canvas[:] = DARK_BG

    canvas[header_h:header_h + thumb_height, :img_w1] = disp1
    canvas[header_h:header_h + thumb_height, img_w1 + gap:] = disp2

    font = cv2.FONT_HERSHEY_SIMPLEX

    _draw_source_info(canvas, match, img_w1, gap, font)
    _draw_divider(canvas, header_h, header_h + thumb_height, total_w)
    _draw_footer(canvas, match, header_h + thumb_height, total_w, font)

    return canvas


def _draw_source_info(canvas, match: MatchResult, img_w1, gap, font):
    p1 = os.path.basename(match.image1)
    p2 = os.path.basename(match.image2)
    cv2.putText(canvas, p1, (10, 25), font, 0.65, CYAN, 2)
    cv2.putText(canvas, p2, (img_w1 + gap + 10, 25), font, 0.65, CYAN, 2)

    d1 = os.path.dirname(match.image1)
    d2 = os.path.dirname(match.image2)
    cv2.putText(canvas, d1[-50:], (10, 48), font, 0.45, GRAY, 1)
    cv2.putText(canvas, d2[-50:], (img_w1 + gap + 10, 48), font, 0.45, GRAY, 1)

    tag = "⚡ CROSS-CHANNEL" if match.is_cross_channel else ""
    if tag:
        cv2.putText(canvas, tag, (img_w1 + gap + 10, 75), font, 0.5, YELLOW, 1)

    mid_x = img_w1 + gap // 2
    cv2.putText(canvas, "VS", (mid_x - 18, 50), font, 1.0, RED, 2)


def _draw_divider(canvas, header_h, img_bottom, total_w):
    cv2.line(canvas, (0, header_h - 2), (total_w, header_h - 2), (80, 80, 80), 1)
    cv2.line(canvas, (0, img_bottom), (total_w, img_bottom), (80, 80, 80), 1)


def _draw_footer(canvas, match: MatchResult, y_base, total_w, font):
    y = y_base + 5
    sim_text = f"Similarity: {match.similarity * 100:.1f}%"
    sim_color = RED if match.severity == "critical" else (0, 140, 255) if match.severity == "high" else YELLOW
    cv2.putText(canvas, sim_text, (10, y + 25), font, 0.7, sim_color, 2)

    cv2.putText(canvas, f"Type: {match.match_type}", (10, y + 50), font, 0.5, GRAY, 1)
    cv2.putText(canvas, match.details[:60], (10, y + 70), font, 0.4, GRAY, 1)

    sev_labels = {"critical": "CRITICAL", "high": "HIGH", "medium": "MEDIUM", "low": "LOW"}
    sev_colors = {"critical": (0, 0, 200), "high": (0, 100, 200), "medium": (0, 180, 180), "low": (0, 150, 0)}
    label = sev_labels.get(match.severity, "UNKNOWN")
    color = sev_colors.get(match.severity, (100, 100, 100))
    (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
    lx = total_w - tw - 20
    cv2.rectangle(canvas, (lx - 5, y + 10), (lx + tw + 10, y + 16 + th), color, -1)
    cv2.putText(canvas, label, (lx, y + 14 + th), font, 0.55, WHITE, 1)


def _fallback_pil(match: MatchResult, output_path: str, thumb_height: int) -> str:
    try:
        img1 = Image.open(match.image1).convert('RGB')
        img2 = Image.open(match.image2).convert('RGB')
        s1 = thumb_height / img1.height
        s2 = thumb_height / img2.height
        d1 = img1.resize((int(img1.width * s1), thumb_height))
        d2 = img2.resize((int(img2.width * s2), thumb_height))
        from PIL import ImageDraw
        draw1 = ImageDraw.Draw(d1)
        draw2 = ImageDraw.Draw(d2)
        draw1.rectangle([5, 5, d1.width - 5, d1.height - 5], outline='red', width=3)
        draw2.rectangle([5, 5, d2.width - 5, d2.height - 5], outline='red', width=3)

        header_h, gap, footer_h = 90, 12, 80
        total_w = d1.width + gap + d2.width
        total_h = header_h + thumb_height + footer_h
        canvas = Image.new('RGB', (total_w, total_h), (45, 45, 45))
        canvas.paste(d1, (0, header_h))
        canvas.paste(d2, (d1.width + gap, header_h))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        canvas.save(output_path, quality=92)
        return output_path
    except Exception:
        return ""
