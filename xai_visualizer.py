"""
KhmerXScore XAI Visualizer
============================
Generates PNG visualization cards for each XAI sample showing:
  - Student answer with per-character saliency highlights
  - Saliency bar chart (one bar per character position)
  - Score chips, uncertainty bar, subject/question info
  - Reference answer panel

Uses PIL (Pillow) + Unifont for proper Khmer Unicode rendering.
Saves: one PNG per sample + combined summary sheet.

Run:
    python3 xai_visualizer.py                       # all available models
    python3 xai_visualizer.py --model bilstm_ar      # single model
    python3 xai_visualizer.py --model all --n 10     # first 10 samples per model

Discovers results/xai_*_data.json files produced by generate_xai_all.py.
Saves one report per model to results/xai_visuals/{model_key}/.
"""

import argparse
import glob as _glob
import json
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Paths ──────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FONT_PATH   = "/usr/share/fonts/opentype/unifont/unifont.otf"

# ── Color palette ──────────────────────────────────────────────────────
BG        = (13,  15,  20)     # Dark background
SURFACE   = (21,  24,  32)     # Card surface
BORDER    = (42,  47,  62)     # Borders
TEXT      = (232, 234, 240)    # Primary text
MUTED     = (107, 114, 128)    # Secondary text
ACCENT    = (79,  142, 247)    # Blue accent
GREEN     = (34,  197, 94)     # Correct
RED       = (239, 68,  68)     # Wrong
AMBER     = (245, 158, 11)     # Warning / incorrect
WHITE     = (255, 255, 255)


def lerp_color(c1, c2, t):
    """Linear interpolate between two RGB tuples."""
    t = max(0, min(1, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def sal_to_rgb(sal, correct):
    """Map saliency [0,1] to an RGB color."""
    s = max(0, min(1, sal))
    if s < 0.04:
        return (30, 34, 45)       # Near-invisible for negligible saliency
    if correct:
        # Low: dim blue → High: bright green
        mid = lerp_color((40, 80, 160), (79, 142, 247), s)
        return lerp_color(mid, (34, 197, 94), max(0, (s - 0.4) / 0.6))
    else:
        # Low: dim amber → High: bright red
        mid = lerp_color((100, 60, 10), (245, 158, 11), s)
        return lerp_color(mid, (239, 68, 68), max(0, (s - 0.4) / 0.6))


def with_alpha(rgb, alpha, bg=BG):
    """Blend rgb color with alpha over bg."""
    return tuple(int(bg[i] + (rgb[i] - bg[i]) * alpha) for i in range(3))


def load_fonts():
    """Load font sizes we need."""
    try:
        return {
            "sm":  ImageFont.truetype(FONT_PATH, 14),
            "md":  ImageFont.truetype(FONT_PATH, 17),
            "lg":  ImageFont.truetype(FONT_PATH, 20),
            "xl":  ImageFont.truetype(FONT_PATH, 28),
            "xs":  ImageFont.truetype(FONT_PATH, 11),
        }
    except Exception as e:
        print(f"  Font load failed: {e}  — falling back to default")
        default = ImageFont.load_default()
        return {k: default for k in ["sm","md","lg","xl","xs"]}


def text_size(draw, text, font):
    """Get (width, height) of rendered text."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_rounded_rect(draw, xy, radius=6, fill=None, outline=None, width=1):
    """Draw a rounded rectangle."""
    x0, y0, x1, y1 = xy
    r = min(radius, (x1-x0)//2, (y1-y0)//2)
    if fill:
        draw.rectangle([x0+r, y0, x1-r, y1], fill=fill)
        draw.rectangle([x0, y0+r, x1, y1-r], fill=fill)
        draw.ellipse([x0, y0, x0+2*r, y0+2*r], fill=fill)
        draw.ellipse([x1-2*r, y0, x1, y0+2*r], fill=fill)
        draw.ellipse([x0, y1-2*r, x0+2*r, y1], fill=fill)
        draw.ellipse([x1-2*r, y1-2*r, x1, y1], fill=fill)
    if outline:
        draw.rounded_rectangle([x0, y0, x1, y1], radius=r, outline=outline, width=width)


def draw_chip(draw, fonts, x, y, value, label, color=ACCENT, w=90):
    """Draw a metric chip."""
    h = 56
    draw_rounded_rect(draw, [x, y, x+w, y+h], radius=6, fill=SURFACE)
    draw_rounded_rect(draw, [x, y, x+w, y+h], radius=6, outline=BORDER, width=1)
    vw, vh = text_size(draw, str(value), fonts["md"])
    draw.text((x + (w-vw)//2, y+8), str(value), font=fonts["md"], fill=color)
    lw, lh = text_size(draw, label, fonts["xs"])
    draw.text((x + (w-lw)//2, y+h-lh-7), label, font=fonts["xs"], fill=MUTED)


def draw_label(draw, fonts, x, y, text, width=None):
    """Draw a section label with trailing line."""
    draw.text((x, y), text.upper(), font=fonts["xs"], fill=MUTED)
    tw, _ = text_size(draw, text.upper(), fonts["xs"])
    if width:
        lx = x + tw + 8
        draw.line([(lx, y+5), (x+width, y+5)], fill=BORDER, width=1)


def draw_saliency_bar(draw, img, x, y, char_sal, width, height=24, window=62):
    """Draw per-character saliency bar chart."""
    n = len(char_sal)
    if n == 0:
        return

    bar_w = max(1, width / n)

    for i, (ch, sal) in enumerate(char_sal):
        bx = x + int(i * bar_w)
        bw = max(1, int(bar_w) + (1 if i < n-1 else 0))
        beyond = i >= window
        color  = BORDER if beyond else sal_to_rgb(sal, True)
        alpha  = 0.4 if beyond else max(0.1, sal)
        blended = with_alpha(color, alpha)
        draw.rectangle([bx, y, bx+bw, y+height], fill=blended)

    # Window marker
    wx = x + int(window * bar_w)
    if wx < x + width:
        draw.line([(wx, y), (wx, y+height)], fill=AMBER, width=1)
        draw.text((wx+2, y), f"win={window}", font=draw._image.info.get("xs_font"),
                  fill=AMBER) if False else None


def draw_highlighted_text(img, draw, fonts, x, y, char_sal, window, correct, max_width, line_height=28):
    """
    Draw answer text with per-character background highlights.
    Handles Khmer by drawing character by character with colored backgrounds.
    Returns the final y position.
    """
    font = fonts["md"]
    cur_x, cur_y = x, y

    for i, (ch, sal) in enumerate(char_sal):
        beyond = i >= window

        # Measure this character
        try:
            cw, ch_h = text_size(draw, ch, font)
            cw = max(cw, 6)
        except Exception:
            cw, ch_h = 10, 18

        # Wrap to next line if needed
        if cur_x + cw > x + max_width and cur_x > x:
            cur_x = x
            cur_y += line_height

        # Draw colored background
        if beyond:
            bg_color = with_alpha(BORDER, 0.3)
        else:
            rgb   = sal_to_rgb(sal, correct)
            alpha = max(0.05, sal * 0.85)
            bg_color = with_alpha(rgb, alpha)

        pad = 2
        draw.rectangle(
            [cur_x - pad, cur_y - 2, cur_x + cw + pad, cur_y + ch_h + 2],
            fill=bg_color
        )

        # Draw character
        text_color = TEXT if not beyond else tuple(int(c * 0.45) for c in TEXT)
        try:
            draw.text((cur_x, cur_y), ch, font=font, fill=text_color)
        except Exception:
            pass

        cur_x += cw + 1

    return cur_y + line_height


def render_sample_card(sample, idx, fonts, card_w=900):
    """Render a single XAI sample as a PIL Image."""

    # Layout constants
    PAD     = 24
    INNER_W = card_w - 2 * PAD

    # Pre-measure text heights
    def estimate_text_h(text, font, max_w):
        """Rough estimate of wrapped text height."""
        words = list(text)  # character level for Khmer
        dummy = Image.new("RGB", (1,1))
        dd = ImageDraw.Draw(dummy)
        lines, line_w = 1, 0
        for ch in words:
            try:
                cw,_ = text_size(dd, ch, font)
            except:
                cw = 10
            if line_w + cw > max_w:
                lines += 1; line_w = cw
            else:
                line_w += cw + 1
        return lines * 26 + 12

    ans_h = estimate_text_h(sample["student_answer"], fonts["md"], INNER_W) + 40
    ref_h = estimate_text_h(sample["reference"], fonts["sm"], INNER_W//2 - 10) + 30
    q_h   = estimate_text_h(sample["question"],  fonts["sm"], INNER_W) + 24

    HEADER_H  = 52
    INFO_H    = 72
    Q_H       = max(q_h, 50) + 32
    ANS_H     = max(ans_h, 70) + 48
    REF_H     = max(ref_h, 50) + 40
    SALBAR_H  = 56
    CHIPS_H   = 88
    FOOTER_H  = 36

    total_h = (HEADER_H + INFO_H + Q_H + ANS_H +
               SALBAR_H + REF_H + CHIPS_H + FOOTER_H + PAD * 7)

    img  = Image.new("RGB", (card_w, total_h), BG)
    draw = ImageDraw.Draw(img)

    ok     = sample["correct"]
    dc     = GREEN if sample["true_label"] == sample["pred_label"] else \
             AMBER if abs(sample["true_label"] - sample["pred_label"]) == 1 else RED
    unc    = sample["uncertainty"]
    uc     = GREEN if unc < 0.05 else AMBER if unc < 0.15 else RED
    border = ACCENT if ok else AMBER

    # ── Header ───────────────────────────────────────────────────────
    draw.rectangle([0, 0, card_w, HEADER_H], fill=SURFACE)
    draw.line([(0, HEADER_H), (card_w, HEADER_H)], fill=BORDER, width=1)
    # Left accent bar
    draw.rectangle([0, 0, 4, HEADER_H], fill=border)

    num_text = f"#{String(idx+1)}"
    draw.text((PAD, 15), f"Sample {idx+1:02d}", font=fonts["md"], fill=TEXT)

    subj_w, _ = text_size(draw, sample["subject"], fonts["sm"])
    sx = PAD + 120
    draw_rounded_rect(draw, [sx, 14, sx+subj_w+16, 36], radius=4,
                      fill=with_alpha(ACCENT, 0.12), outline=with_alpha(ACCENT, 0.3))
    draw.text((sx+8, 15), sample["subject"], font=fonts["sm"], fill=ACCENT)

    correct_text = "✓ Correct" if ok else "✗ Incorrect"
    ct_w, _ = text_size(draw, correct_text, fonts["sm"])
    ctx = card_w - PAD - ct_w - 16
    ct_bg  = with_alpha(GREEN, 0.12) if ok else with_alpha(RED, 0.12)
    ct_col = GREEN if ok else RED
    draw_rounded_rect(draw, [ctx-8, 14, ctx+ct_w+8, 36], radius=4,
                      fill=ct_bg, outline=with_alpha(ct_col, 0.3))
    draw.text((ctx, 15), correct_text, font=fonts["sm"], fill=ct_col)

    draw.text((card_w - PAD - 60, HEADER_H//2 - 5),
              f"idx={sample['idx']}", font=fonts["xs"], fill=MUTED)

    cy = HEADER_H + PAD  # current y cursor

    # ── Score chips row ────────────────────────────────────────────────
    chip_labels = [
        (str(sample["true_label"]),                  "True label",    MUTED),
        (str(sample["pred_label"]),                  "Predicted",     dc),
        (f"{sample['true_score']:.2f}",              "True score",    TEXT),
        (f"{sample['pred_score']:.2f}",              "Pred score",    dc),
    ]
    chip_w = 82
    for ci, (val, lbl, col) in enumerate(chip_labels):
        draw_chip(draw, fonts, PAD + ci * (chip_w + 10), cy, val, lbl, col, chip_w)

    # Uncertainty bar chip
    unc_x = PAD + len(chip_labels) * (chip_w + 10)
    unc_w = card_w - PAD - unc_x
    draw_rounded_rect(draw, [unc_x, cy, card_w - PAD, cy + 56],
                      radius=6, fill=SURFACE, outline=BORDER)
    draw.text((unc_x + 10, cy + 8), "Uncertainty (MC Dropout T=10)",
              font=fonts["xs"], fill=MUTED)
    bar_y = cy + 26
    draw.rectangle([unc_x+10, bar_y, card_w-PAD-10, bar_y+8],
                   fill=with_alpha(BORDER, 0.6))
    unc_fill_w = int((unc_w - 20) * min(1.0, unc / 0.3))
    draw.rectangle([unc_x+10, bar_y, unc_x+10+unc_fill_w, bar_y+8], fill=uc)
    unc_label = "Low — confident" if unc < 0.05 else \
                "Medium — borderline" if unc < 0.15 else "High — defer to teacher"
    draw.text((unc_x + 10, bar_y + 12),
              f"σ={unc:.4f}   {unc_label}", font=fonts["xs"], fill=uc)

    cy += CHIPS_H

    # ── Question ──────────────────────────────────────────────────────
    draw_label(draw, fonts, PAD, cy, "Question", INNER_W)
    cy += 16
    draw.rectangle([PAD, cy, card_w-PAD, cy + Q_H - 20],
                   fill=with_alpha(SURFACE, 0.8))
    draw.line([(PAD, cy), (PAD, cy + Q_H - 20)],
              fill=with_alpha(ACCENT, 0.5), width=2)
    draw.text((PAD + 10, cy + 8), sample["question"],
              font=fonts["sm"], fill=with_alpha(TEXT, 0.85))
    cy += Q_H

    # ── Highlighted student answer ─────────────────────────────────────
    has_sal  = sample.get("has_saliency", True)
    inp_fmt  = sample.get("input_format", "ar")
    if not has_sal:
        sal_label = "Student Answer (No Gradient Saliency)"
    elif inp_fmt == "qar":
        sal_label = "Question + Answer — Gradient Saliency"
    else:
        sal_label = "Student Answer — Gradient Saliency"
    draw_label(draw, fonts, PAD, cy, sal_label, INNER_W)
    cy += 16
    ans_bg = with_alpha(SURFACE, 0.6)
    draw.rectangle([PAD, cy, card_w-PAD, cy + ANS_H - 10], fill=ans_bg)
    draw.line([(PAD, cy), (PAD, cy + ANS_H - 10)],
              fill=border, width=2)

    final_y = draw_highlighted_text(
        img, draw, fonts,
        x=PAD + 10, y=cy + 10,
        char_sal=sample["char_saliency"],
        window=sample["model_window"],
        correct=ok,
        max_width=INNER_W - 20,
        line_height=30,
    )
    cy += ANS_H

    # ── Saliency bar chart ─────────────────────────────────────────────
    draw_label(draw, fonts, PAD, cy, "Token Attribution Intensity (per character position)", INNER_W)
    cy += 16

    bar_area_h = 30
    # Background
    draw.rectangle([PAD, cy, card_w-PAD, cy+bar_area_h], fill=SURFACE)

    char_sal = sample["char_saliency"]
    n = len(char_sal)
    if n > 0:
        bar_unit = INNER_W / n
        for i, (ch, sal) in enumerate(char_sal):
            bx = PAD + int(i * bar_unit)
            bw = max(1, int(bar_unit) + 1)
            beyond = i >= sample["model_window"]
            if beyond:
                h = 4
                color = BORDER
            else:
                h = max(2, int(bar_area_h * sal))
                color = sal_to_rgb(sal, ok)
            by = cy + bar_area_h - h
            draw.rectangle([bx, by, bx+bw, cy+bar_area_h], fill=color)

        # Window line
        wx = PAD + int(sample["model_window"] * bar_unit)
        if wx < PAD + INNER_W:
            draw.line([(wx, cy), (wx, cy+bar_area_h)], fill=AMBER, width=1)
            draw.text((wx+2, cy+1), f"window={sample['model_window']}",
                      font=fonts["xs"], fill=AMBER)

    # Axis labels
    draw.text((PAD, cy+bar_area_h+4), "char position →", font=fonts["xs"], fill=MUTED)
    draw.text((card_w-PAD-60, cy+bar_area_h+4), f"n={n}", font=fonts["xs"], fill=MUTED)
    cy += bar_area_h + 22

    # ── Reference answer ──────────────────────────────────────────────
    draw_label(draw, fonts, PAD, cy, "Reference Answer (full-score example)", INNER_W)
    cy += 16
    draw.rectangle([PAD, cy, card_w-PAD, cy + REF_H - 10], fill=SURFACE)
    draw.line([(PAD, cy), (PAD, cy + REF_H - 10)],
              fill=with_alpha(GREEN, 0.5), width=2)
    draw.text((PAD+10, cy+8), sample["reference"],
              font=fonts["sm"], fill=with_alpha(TEXT, 0.75))
    cy += REF_H

    # ── Footer ────────────────────────────────────────────────────────
    draw.line([(0, total_h - FOOTER_H), (card_w, total_h - FOOTER_H)],
              fill=BORDER, width=1)
    mname = sample.get("model_name", "KhmerXScore")
    if sample.get("has_saliency", True):
        footer_text = f"KhmerXScore  ·  {mname}  ·  CORN+SCL  ·  Saliency: ||d_loss/d_emb||_2"
    else:
        footer_text = f"KhmerXScore  ·  {mname}  ·  No gradient saliency (deterministic model)"
    fw, _ = text_size(draw, footer_text, fonts["xs"])
    draw.text(((card_w - fw)//2, total_h - FOOTER_H + 10),
              footer_text, font=fonts["xs"], fill=MUTED)

    return img


def String(n):
    return str(n)


def make_legend_strip(fonts, width=900, height=50):
    """Create a reusable legend strip."""
    img  = Image.new("RGB", (width, height), SURFACE)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, height], fill=SURFACE)
    draw.line([(0, height-1), (width, height-1)], fill=BORDER)

    # Correct gradient swatch
    for i in range(120):
        t   = i / 119
        col = sal_to_rgb(t, True)
        draw.line([(20+i, 16), (20+i, 32)], fill=col)
    draw.text((20, 35), "Low", font=fonts["xs"], fill=MUTED)
    draw.text((110, 35), "High saliency — Correct ✓", font=fonts["xs"], fill=TEXT)

    # Incorrect gradient swatch
    for i in range(120):
        t   = i / 119
        col = sal_to_rgb(t, False)
        draw.line([(300+i, 16), (300+i, 32)], fill=col)
    draw.text((300, 35), "Low", font=fonts["xs"], fill=MUTED)
    draw.text((390, 35), "High saliency — Incorrect ✗", font=fonts["xs"], fill=TEXT)

    # Amber window marker
    draw.line([(600, 10), (600, 40)], fill=AMBER, width=2)
    draw.text((606, 20), "= model window boundary", font=fonts["xs"], fill=AMBER)

    return img


def _run_model(data_path, output_dir, fonts, max_samples=None):
    """Render all sample cards for one model and save combined report."""
    with open(data_path, encoding="utf-8") as f:
        samples = json.load(f)
    if max_samples is not None:
        samples = samples[:max_samples]

    model_label = samples[0].get("model_name", os.path.basename(data_path)) if samples else ""
    print(f"  {len(samples)} samples  [{model_label}]")
    os.makedirs(output_dir, exist_ok=True)

    all_cards = []
    for i, sample in enumerate(samples):
        tag = "OK" if sample["correct"] else "--"
        print(f"  [{i+1:2d}/{len(samples)}] idx={sample['idx']}  "
              f"true={sample['true_label']}  pred={sample['pred_label']}  {tag}")
        card = render_sample_card(sample, i, fonts, card_w=900)
        out_path = os.path.join(
            output_dir,
            f"sample_{i+1:02d}_true{sample['true_label']}_pred{sample['pred_label']}.png",
        )
        card.save(out_path, "PNG", dpi=(150, 150))
        all_cards.append(card)

    if not all_cards:
        return

    GAP     = 12
    HDR_H   = 50
    total_h = HDR_H + sum(c.height + GAP for c in all_cards) + GAP
    sheet   = Image.new("RGB", (900, total_h), BG)
    ds      = ImageDraw.Draw(sheet)
    ds.rectangle([0, 0, 900, HDR_H], fill=SURFACE)
    title = f"KhmerXScore — XAI Report: {model_label}"
    tw, _ = text_size(ds, title, fonts["lg"])
    ds.text(((900 - tw) // 2, 12), title, font=fonts["lg"], fill=ACCENT)
    ds.line([(0, HDR_H - 1), (900, HDR_H - 1)], fill=BORDER)
    cy = HDR_H + GAP
    for card in all_cards:
        sheet.paste(card, (0, cy))
        cy += card.height + GAP

    combined_path = os.path.join(output_dir, "xai_report_combined.png")
    sheet.save(combined_path, "PNG", dpi=(150, 150))
    print(f"  Combined: {combined_path}  ({sheet.width}x{sheet.height}px)")

    correct  = sum(1 for s in samples if s["correct"])
    unc_vals = [s["uncertainty"] for s in samples]
    sal_vals = [v for s in samples for _, v in s["char_saliency"] if v > 0.05]
    print(f"  Correct: {correct}/{len(samples)}  "
          f"mean_unc={np.mean(unc_vals):.4f}" +
          (f"  mean_active_sal={np.mean(sal_vals):.4f}" if sal_vals else ""))


def main():
    parser = argparse.ArgumentParser(description="KhmerXScore XAI Visualizer")
    parser.add_argument(
        "--model", default="all",
        help="Model key (e.g. bilstm_ar) or 'all' to process all xai_*_data.json files.",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Max samples to render per model (default: all).",
    )
    args = parser.parse_args()

    print("KhmerXScore XAI Visualizer")
    print("=" * 50)

    if args.model == "all":
        data_files = sorted(_glob.glob(os.path.join(RESULTS_DIR, "xai_*_data.json")))
        if not data_files:
            print(f"No xai_*_data.json files found in {RESULTS_DIR}")
            print("Run first:  python generate_xai_all.py")
            return
    else:
        p = os.path.join(RESULTS_DIR, f"xai_{args.model}_data.json")
        if not os.path.exists(p):
            # Backward compat: also accept bare filename passed as model key
            p2 = os.path.join(RESULTS_DIR, args.model)
            if os.path.exists(p2):
                p = p2
            else:
                print(f"ERROR: {p} not found.")
                print("Run first:  python generate_xai_all.py")
                return
        data_files = [p]

    fonts = load_fonts()
    print(f"Fonts loaded  |  models to render: {len(data_files)}\n")

    for data_path in data_files:
        model_key = (os.path.basename(data_path)
                     .replace("xai_", "").replace("_data.json", ""))
        output_dir = os.path.join(RESULTS_DIR, "xai_visuals", model_key)
        print(f"-- {model_key} --")
        _run_model(data_path, output_dir, fonts, max_samples=args.n)
        print()


if __name__ == "__main__":
    main()
