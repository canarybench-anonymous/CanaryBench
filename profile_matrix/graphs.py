r"""
Archived dependency-free SVG implementation. Kept below for reference; the
paper figures now use the matplotlib/pandas implementation at the end of this
file.

import argparse
import csv
import html
import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from typing import Iterable


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_RE = re.compile(r"^(?P<model>.+?)__(?P<run>baseline|dp_eps[0-9.]+)\.json$")
EPS_ORDER = [3.0, 5.0, 8.0, 16.0, 64.0]
MODEL_LABELS = {
    "gemma3_1b": "Gemma 3 1B",
    "gemma_2b_it": "Gemma 2 2B IT",
    "llama3_2_1b_it": "Llama 3.2 1B IT",
    "vault_gemma": "VaultGemma",
}
COLORS = {
    "gemma3_1b": "#2F7D6D",
    "gemma_2b_it": "#B4472A",
    "llama3_2_1b_it": "#4568A9",
    "vault_gemma": "#8A5A99",
}


@dataclass(frozen=True)
class Run:
    model: str
    run: str
    epsilon: float | None
    exposure_mean: float
    exposure_p50: float
    exposure_p95: float
    exposure_max: float
    gap: float
    wbc_auc: float
    wbc_lo: float | None
    wbc_hi: float | None
    gap_lo: float | None
    gap_hi: float | None
    pii_extraction_rate: float
    avg_rouge1: float
    perplexity: float
    member_exposures: tuple[float, ...]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fmt_num(value: float | None, digits: int = 3) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def eps_label(run: Run | str) -> str:
    if isinstance(run, Run):
        return "Baseline" if run.epsilon is None else f"eps={run.epsilon:g}"
    return "Baseline" if run == "baseline" else run.replace("dp_eps", "eps=")


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def percentile(values: Iterable[float], q: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def load_cis(path: str) -> dict[tuple[str, str], dict[str, float]]:
    if not os.path.exists(path):
        return {}
    out: dict[tuple[str, str], dict[str, float]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[(row["model"], row["eps"])] = {
                key: float(row[key])
                for key in (
                    "gap_ci_lower",
                    "gap_ci_upper",
                    "wbc_ci_lower",
                    "wbc_ci_upper",
                )
                if row.get(key)
            }
    return out


def load_runs(data_dir: str, ci_path: str | None = None) -> list[Run]:
    cis = load_cis(ci_path or os.path.join(data_dir, "bootstrap_cis.csv"))
    runs: list[Run] = []
    for name in sorted(os.listdir(data_dir)):
        match = RESULT_RE.match(name)
        if not match:
            continue
        path = os.path.join(data_dir, name)
        with open(path) as fh:
            data = json.load(fh)
        ce = data.get("canary_exposure") or {}
        wbc = data.get("wbc") or {}
        pii = data.get("pii_extraction") or {}
        ppl = data.get("perplexity") or {}
        run = data.get("run") or match.group("run")
        epsilon = data.get("epsilon")
        ci_key = (data.get("model_key") or match.group("model"), run)
        ci = cis.get(ci_key, {})
        runs.append(
            Run(
                model=data.get("model_key") or match.group("model"),
                run=run,
                epsilon=None if epsilon is None else float(epsilon),
                exposure_mean=float(ce.get("exposure_mean", 0.0) or 0.0),
                exposure_p50=float(ce.get("exposure_p50", 0.0) or 0.0),
                exposure_p95=float(ce.get("exposure_p95", 0.0) or 0.0),
                exposure_max=float(ce.get("exposure_max", 0.0) or 0.0),
                gap=float(ce.get("gap", 0.0) or 0.0),
                wbc_auc=float(wbc.get("wbc_auc", 0.0) or 0.0),
                wbc_lo=ci.get("wbc_ci_lower"),
                wbc_hi=ci.get("wbc_ci_upper"),
                gap_lo=ci.get("gap_ci_lower"),
                gap_hi=ci.get("gap_ci_upper"),
                pii_extraction_rate=float(pii.get("pii_extraction_rate", 0.0) or 0.0),
                avg_rouge1=float(pii.get("avg_rouge1", 0.0) or 0.0),
                perplexity=float(ppl.get("perplexity", 0.0) or 0.0),
                member_exposures=tuple(float(v) for v in (ce.get("member_exposures") or [])),
            )
        )
    return sorted(runs, key=lambda r: (r.model, -1 if r.epsilon is None else r.epsilon))


def write_text(fh, x: float, y: float, text: object, *, size=13, anchor="middle", weight="400", fill="#222", rotate=None):
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate else ""
    fh.write(
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="{fill}"{transform}>{esc(text)}</text>\n'
    )


def svg_start(fh, width: int, height: int, title: str):
    fh.write(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
    )
    fh.write("<style>text{font-family:Arial,Helvetica,sans-serif}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.tick{stroke:#333;stroke-width:1}</style>\n")
    fh.write(f"<title>{esc(title)}</title>\n")


def svg_end(fh):
    fh.write("</svg>\n")


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / max(n - 1, 1)
    power = 10 ** math.floor(math.log10(raw))
    step = min((1, 2, 5, 10), key=lambda m: abs(m * power - raw)) * power
    start = math.floor(lo / step) * step
    vals = []
    v = start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.5:
            vals.append(v)
        v += step
    return vals


def y_scale(value: float, lo: float, hi: float, top: float, bottom: float) -> float:
    return bottom - (value - lo) / (hi - lo) * (bottom - top)


def x_scale(value: float, lo: float, hi: float, left: float, right: float) -> float:
    return left + (value - lo) / (hi - lo) * (right - left)


def draw_axes(fh, left, top, right, bottom, y_lo, y_hi, y_label, x_label=None):
    fh.write(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>\n')
    fh.write(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>\n')
    for tick in nice_ticks(y_lo, y_hi):
        y = y_scale(tick, y_lo, y_hi, top, bottom)
        fh.write(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>\n')
        fh.write(f'<line class="tick" x1="{left-4}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}"/>\n')
        write_text(fh, left - 8, y + 4, f"{tick:g}", size=11, anchor="end", fill="#444")
    write_text(fh, left - 56, (top + bottom) / 2, y_label, size=13, rotate=-90, fill="#222")
    if x_label:
        write_text(fh, (left + right) / 2, bottom + 54, x_label, size=13, fill="#222")


def write_summary_csv(runs: list[Run], out_path: str):
    with open(out_path, "w", newline="") as fh:
        fieldnames = [
            "model",
            "run",
            "epsilon",
            "gap",
            "gap_ci_lower",
            "gap_ci_upper",
            "exposure_mean",
            "exposure_p50",
            "exposure_p95",
            "exposure_max",
            "wbc_auc",
            "wbc_ci_lower",
            "wbc_ci_upper",
            "pii_extraction_rate",
            "avg_rouge1",
            "perplexity",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in runs:
            writer.writerow(
                {
                    "model": r.model,
                    "run": r.run,
                    "epsilon": "" if r.epsilon is None else r.epsilon,
                    "gap": r.gap,
                    "gap_ci_lower": "" if r.gap_lo is None else r.gap_lo,
                    "gap_ci_upper": "" if r.gap_hi is None else r.gap_hi,
                    "exposure_mean": r.exposure_mean,
                    "exposure_p50": r.exposure_p50,
                    "exposure_p95": r.exposure_p95,
                    "exposure_max": r.exposure_max,
                    "wbc_auc": r.wbc_auc,
                    "wbc_ci_lower": "" if r.wbc_lo is None else r.wbc_lo,
                    "wbc_ci_upper": "" if r.wbc_hi is None else r.wbc_hi,
                    "pii_extraction_rate": r.pii_extraction_rate,
                    "avg_rouge1": r.avg_rouge1,
                    "perplexity": r.perplexity,
                }
            )


def plot_baseline_vs_dp(runs: list[Run], out_path: str):
    metrics = [
        ("gap", "Memorization Gap", lambda r: r.gap, lambda r: (r.gap_lo, r.gap_hi)),
        ("wbc", "WBC AUC", lambda r: r.wbc_auc, lambda r: (r.wbc_lo, r.wbc_hi)),
        ("exposure", "Mean Exposure", lambda r: r.exposure_mean, lambda r: (None, None)),
        ("ppl", "Perplexity", lambda r: r.perplexity, lambda r: (None, None)),
    ]
    models = sorted({r.model for r in runs})
    best_dp = {
        m: min((r for r in runs if r.model == m and r.epsilon is not None), key=lambda r: r.perplexity)
        for m in models
    }
    baseline = {r.model: r for r in runs if r.epsilon is None}
    width, height = 1180, 760
    left, right, top, bottom = 76, 1130, 84, 650
    panel_w = (right - left - 42) / 2
    panel_h = (bottom - top - 46) / 2
    with open(out_path, "w") as fh:
        svg_start(fh, width, height, "Baseline versus DP privacy profile")
        write_text(fh, width / 2, 34, "Baseline vs DP Privacy Profile", size=22, weight="700")
        write_text(fh, width / 2, 58, "DP point uses the lowest-perplexity DP run for each model", size=13, fill="#555")
        for i, (_, title, getter, ci_getter) in enumerate(metrics):
            px = left + (i % 2) * (panel_w + 42)
            py = top + (i // 2) * (panel_h + 46)
            vals = [getter(baseline[m]) for m in models] + [getter(best_dp[m]) for m in models]
            y_lo, y_hi = 0, max(vals) * 1.18
            if title == "WBC AUC":
                y_lo, y_hi = 0.45, max(vals) * 1.08
            draw_axes(fh, px, py, px + panel_w, py + panel_h, y_lo, y_hi, title)
            write_text(fh, px + panel_w / 2, py - 14, title, size=15, weight="700")
            group_w = panel_w / len(models)
            bar_w = min(28, group_w * 0.26)
            for j, m in enumerate(models):
                cx = px + group_w * (j + 0.5)
                for dx, run, fill, label in (
                    (-bar_w * 0.6, baseline[m], "#555555", "Baseline"),
                    (bar_w * 0.6, best_dp[m], COLORS.get(m, "#888"), "DP"),
                ):
                    val = getter(run)
                    y = y_scale(val, y_lo, y_hi, py, py + panel_h)
                    fh.write(f'<rect x="{cx+dx-bar_w/2:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{py+panel_h-y:.2f}" fill="{fill}"/>\n')
                    lo, hi = ci_getter(run)
                    if lo is not None and hi is not None:
                        ylo = y_scale(lo, y_lo, y_hi, py, py + panel_h)
                        yhi = y_scale(hi, y_lo, y_hi, py, py + panel_h)
                        fh.write(f'<line x1="{cx+dx:.2f}" y1="{yhi:.2f}" x2="{cx+dx:.2f}" y2="{ylo:.2f}" stroke="#222"/>\n')
                        fh.write(f'<line x1="{cx+dx-5:.2f}" y1="{yhi:.2f}" x2="{cx+dx+5:.2f}" y2="{yhi:.2f}" stroke="#222"/>\n')
                        fh.write(f'<line x1="{cx+dx-5:.2f}" y1="{ylo:.2f}" x2="{cx+dx+5:.2f}" y2="{ylo:.2f}" stroke="#222"/>\n')
                write_text(fh, cx, py + panel_h + 22, model_label(m).replace(" ", "\n"), size=10, fill="#333")
        fh.write('<rect x="456" y="710" width="16" height="16" fill="#555"/>\n')
        write_text(fh, 478, 723, "Baseline", size=12, anchor="start")
        fh.write('<rect x="548" y="710" width="16" height="16" fill="#2F7D6D"/>\n')
        write_text(fh, 570, 723, "DP selected per model", size=12, anchor="start")
        svg_end(fh)


def plot_epsilon_curves(runs: list[Run], out_path: str):
    metrics = [
        ("gap", "Memorization Gap", lambda r: r.gap),
        ("wbc", "WBC AUC", lambda r: r.wbc_auc),
        ("exposure", "Mean Exposure", lambda r: r.exposure_mean),
        ("ppl", "Perplexity", lambda r: r.perplexity),
    ]
    width, height = 1180, 760
    left, right, top, bottom = 82, 1125, 84, 650
    panel_w = (right - left - 46) / 2
    panel_h = (bottom - top - 48) / 2
    eps_values = EPS_ORDER
    x_lo, x_hi = math.log2(min(eps_values)) - 0.25, math.log2(max(eps_values)) + 0.25
    models = sorted({r.model for r in runs})
    by_model_eps = {(r.model, r.epsilon): r for r in runs if r.epsilon is not None}
    baseline = {r.model: r for r in runs if r.epsilon is None}
    with open(out_path, "w") as fh:
        svg_start(fh, width, height, "Privacy metrics across DP epsilon")
        write_text(fh, width / 2, 36, "DP Epsilon Sweep", size=22, weight="700")
        write_text(fh, width / 2, 60, "Dashed horizontal ticks show each model baseline", size=13, fill="#555")
        for i, (_, title, getter) in enumerate(metrics):
            px = left + (i % 2) * (panel_w + 46)
            py = top + (i // 2) * (panel_h + 48)
            vals = [getter(r) for r in runs]
            y_lo = 0 if title != "WBC AUC" else 0.45
            y_hi = max(vals) * 1.18 if title != "WBC AUC" else max(vals) * 1.08
            draw_axes(fh, px, py, px + panel_w, py + panel_h, y_lo, y_hi, title, "Privacy budget epsilon")
            write_text(fh, px + panel_w / 2, py - 14, title, size=15, weight="700")
            for eps in eps_values:
                x = x_scale(math.log2(eps), x_lo, x_hi, px, px + panel_w)
                fh.write(f'<line class="tick" x1="{x:.2f}" y1="{py+panel_h:.2f}" x2="{x:.2f}" y2="{py+panel_h+4:.2f}"/>\n')
                write_text(fh, x, py + panel_h + 19, f"{eps:g}", size=11)
            for m in models:
                color = COLORS.get(m, "#666")
                points = []
                for eps in eps_values:
                    r = by_model_eps.get((m, eps))
                    if r:
                        points.append((x_scale(math.log2(eps), x_lo, x_hi, px, px + panel_w), y_scale(getter(r), y_lo, y_hi, py, py + panel_h), r))
                if len(points) > 1:
                    d = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in points)
                    fh.write(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2.2"/>\n')
                for x, y, _ in points:
                    fh.write(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{color}" stroke="white" stroke-width="1"/>\n')
                if m in baseline:
                    y = y_scale(getter(baseline[m]), y_lo, y_hi, py, py + panel_h)
                    fh.write(f'<line x1="{px:.2f}" y1="{y:.2f}" x2="{px+panel_w:.2f}" y2="{y:.2f}" stroke="{color}" stroke-width="1.2" stroke-dasharray="5 5" opacity="0.55"/>\n')
        lx = 392
        for i, m in enumerate(models):
            x = lx + i * 142
            fh.write(f'<circle cx="{x}" cy="718" r="5" fill="{COLORS.get(m, "#666")}"/>\n')
            write_text(fh, x + 10, 722, model_label(m), size=12, anchor="start")
        svg_end(fh)


def plot_privacy_utility(runs: list[Run], out_path: str):
    width, height = 780, 620
    left, top, right, bottom = 92, 72, 720, 525
    x_vals = [r.perplexity for r in runs]
    y_vals = [r.wbc_auc for r in runs]
    x_lo, x_hi = min(x_vals) * 0.96, max(x_vals) * 1.04
    y_lo, y_hi = 0.48, max(y_vals) * 1.08
    with open(out_path, "w") as fh:
        svg_start(fh, width, height, "Privacy utility tradeoff")
        write_text(fh, width / 2, 34, "Privacy-Utility Tradeoff", size=22, weight="700")
        draw_axes(fh, left, top, right, bottom, y_lo, y_hi, "WBC AUC", "Perplexity")
        for tick in nice_ticks(x_lo, x_hi):
            x = x_scale(tick, x_lo, x_hi, left, right)
            fh.write(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}"/>\n')
            fh.write(f'<line class="tick" x1="{x:.2f}" y1="{bottom}" x2="{x:.2f}" y2="{bottom+4}"/>\n')
            write_text(fh, x, bottom + 21, f"{tick:g}", size=11)
        for r in runs:
            x = x_scale(r.perplexity, x_lo, x_hi, left, right)
            y = y_scale(r.wbc_auc, y_lo, y_hi, top, bottom)
            color = COLORS.get(r.model, "#666")
            shape = "rect" if r.epsilon is None else "circle"
            if shape == "rect":
                fh.write(f'<rect x="{x-5:.2f}" y="{y-5:.2f}" width="10" height="10" fill="{color}" stroke="#111"/>\n')
            else:
                radius = 3.5 + min(r.epsilon or 0, 64) / 64 * 4
                fh.write(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{color}" opacity="0.82" stroke="white"/>\n')
        write_text(fh, 115, 565, "Squares = baseline; circles = DP runs; larger circle = larger epsilon", size=12, anchor="start", fill="#555")
        lx = 430
        for i, m in enumerate(sorted({r.model for r in runs})):
            y = 548 + (i // 2) * 22
            x = lx + (i % 2) * 150
            fh.write(f'<circle cx="{x}" cy="{y}" r="5" fill="{COLORS.get(m, "#666")}"/>\n')
            write_text(fh, x + 10, y + 4, model_label(m), size=12, anchor="start")
        svg_end(fh)


def plot_exposure_boxplots(runs: list[Run], out_path: str):
    models = sorted({r.model for r in runs})
    selected = []
    for m in models:
        model_runs = [r for r in runs if r.model == m]
        selected.extend([r for r in model_runs if r.epsilon is None])
        selected.extend([min((r for r in model_runs if r.epsilon is not None), key=lambda r: r.perplexity)])
    width, height = 980, 620
    left, top, right, bottom = 88, 70, 925, 500
    y_lo, y_hi = 0, max((max(r.member_exposures) for r in selected if r.member_exposures), default=1) * 1.08
    with open(out_path, "w") as fh:
        svg_start(fh, width, height, "Member canary exposure distributions")
        write_text(fh, width / 2, 34, "Member Canary Exposure Distributions", size=22, weight="700")
        draw_axes(fh, left, top, right, bottom, y_lo, y_hi, "Exposure")
        group_w = (right - left) / len(models)
        box_w = min(34, group_w * 0.22)
        for i, m in enumerate(models):
            base_x = left + group_w * (i + 0.5)
            for dx, r, fill in (
                (-box_w * 0.85, next(rr for rr in selected if rr.model == m and rr.epsilon is None), "#555555"),
                (box_w * 0.85, next(rr for rr in selected if rr.model == m and rr.epsilon is not None), COLORS.get(m, "#888")),
            ):
                vals = r.member_exposures
                q1, med, q3 = percentile(vals, 0.25), percentile(vals, 0.5), percentile(vals, 0.75)
                low, high = percentile(vals, 0.05), percentile(vals, 0.95)
                x = base_x + dx
                yq1, yq3 = y_scale(q1, y_lo, y_hi, top, bottom), y_scale(q3, y_lo, y_hi, top, bottom)
                ymed = y_scale(med, y_lo, y_hi, top, bottom)
                ylow, yhigh = y_scale(low, y_lo, y_hi, top, bottom), y_scale(high, y_lo, y_hi, top, bottom)
                fh.write(f'<line x1="{x:.2f}" y1="{yhigh:.2f}" x2="{x:.2f}" y2="{ylow:.2f}" stroke="#333"/>\n')
                fh.write(f'<rect x="{x-box_w/2:.2f}" y="{yq3:.2f}" width="{box_w:.2f}" height="{yq1-yq3:.2f}" fill="{fill}" opacity="0.82" stroke="#333"/>\n')
                fh.write(f'<line x1="{x-box_w/2:.2f}" y1="{ymed:.2f}" x2="{x+box_w/2:.2f}" y2="{ymed:.2f}" stroke="white" stroke-width="2"/>\n')
                fh.write(f'<line x1="{x-box_w/3:.2f}" y1="{yhigh:.2f}" x2="{x+box_w/3:.2f}" y2="{yhigh:.2f}" stroke="#333"/>\n')
                fh.write(f'<line x1="{x-box_w/3:.2f}" y1="{ylow:.2f}" x2="{x+box_w/3:.2f}" y2="{ylow:.2f}" stroke="#333"/>\n')
            write_text(fh, base_x, bottom + 24, model_label(m), size=12)
        fh.write('<rect x="370" y="560" width="16" height="16" fill="#555"/>\n')
        write_text(fh, 392, 573, "Baseline", size=12, anchor="start")
        fh.write('<rect x="462" y="560" width="16" height="16" fill="#2F7D6D"/>\n')
        write_text(fh, 484, 573, "Lowest-perplexity DP run", size=12, anchor="start")
        svg_end(fh)


def plot_dp_reduction_heatmap(runs: list[Run], out_path: str):
    models = sorted({r.model for r in runs})
    metrics = [
        ("gap", "Gap reduction", lambda b, d: 1 - d.gap / b.gap if b.gap else 0),
        ("wbc", "WBC-AUC reduction", lambda b, d: 1 - (d.wbc_auc - 0.5) / (b.wbc_auc - 0.5) if b.wbc_auc > 0.5 else 0),
        ("exp", "Exposure reduction", lambda b, d: 1 - d.exposure_mean / b.exposure_mean if b.exposure_mean else 0),
        ("ppl", "Perplexity increase", lambda b, d: d.perplexity / b.perplexity - 1 if b.perplexity else 0),
    ]
    baseline = {r.model: r for r in runs if r.epsilon is None}
    dp = {
        m: min((r for r in runs if r.model == m and r.epsilon is not None), key=lambda r: r.perplexity)
        for m in models
    }
    width, height = 860, 500
    left, top = 220, 82
    cell_w, cell_h = 140, 58
    with open(out_path, "w") as fh:
        svg_start(fh, width, height, "DP relative changes heatmap")
        write_text(fh, width / 2, 36, "Relative Change From Baseline", size=22, weight="700")
        write_text(fh, width / 2, 60, "DP run selected by lowest perplexity for each model", size=13, fill="#555")
        for j, (_, label, _) in enumerate(metrics):
            write_text(fh, left + j * cell_w + cell_w / 2, top - 14, label, size=11)
        for i, m in enumerate(models):
            write_text(fh, left - 18, top + i * cell_h + 34, model_label(m), size=12, anchor="end")
            for j, (_, _, fn) in enumerate(metrics):
                val = fn(baseline[m], dp[m])
                if j == 3:
                    intensity = min(max(val / 1.2, 0), 1)
                    color = f"rgb({230 - int(55*intensity)},{244 - int(90*intensity)},{232 - int(70*intensity)})"
                    display = f"+{val*100:.0f}%"
                else:
                    intensity = min(max(val, 0), 1)
                    color = f"rgb({238 - int(120*intensity)},{246 - int(70*intensity)},{246 - int(120*intensity)})"
                    display = f"{val*100:.0f}%"
                x = left + j * cell_w
                y = top + i * cell_h
                fh.write(f'<rect x="{x}" y="{y}" width="{cell_w-4}" height="{cell_h-4}" fill="{color}" stroke="#fff"/>\n')
                write_text(fh, x + cell_w / 2 - 2, y + 34, display, size=16, weight="700")
        svg_end(fh)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper-ready SVG figures from profile_matrix JSON results.")
    parser.add_argument("--data-dir", default=SCRIPT_DIR, help="Directory containing profile_matrix *.json result files.")
    parser.add_argument("--ci", default=None, help="Optional bootstrap_cis.csv path.")
    parser.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "figures"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = load_runs(args.data_dir, args.ci)
    if not runs:
        raise SystemExit(f"No result JSONs found in {args.data_dir}")

    summary_path = os.path.join(args.out_dir, "profile_matrix_summary.csv")
    write_summary_csv(runs, summary_path)
    outputs = [
        ("baseline_vs_dp.svg", plot_baseline_vs_dp),
        ("epsilon_sweep.svg", plot_epsilon_curves),
        ("privacy_utility_tradeoff.svg", plot_privacy_utility),
        ("exposure_boxplots.svg", plot_exposure_boxplots),
        ("dp_relative_change_heatmap.svg", plot_dp_reduction_heatmap),
    ]
    for name, fn in outputs:
        path = os.path.join(args.out_dir, name)
        fn(runs, path)
        print(path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULT_RE = re.compile(r"^(?P<model>.+?)__(?P<run>baseline|dp_eps[0-9.]+)\.json$")
EPS_ORDER = [3.0, 5.0, 8.0, 16.0, 64.0]

MODEL_LABELS = {
    "gemma3_1b": "Gemma 3 1B",
    "gemma_2b_it": "Gemma 2 2B IT",
    "llama3_2_1b_it": "Llama 3.2 1B IT",
    "vault_gemma": "VaultGemma",
}

COLORS = {
    "gemma3_1b": "#2f7d6d",
    "gemma_2b_it": "#b4472a",
    "llama3_2_1b_it": "#4568a9",
    "vault_gemma": "#8a5a99",
}


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def run_label(row: pd.Series) -> str:
    if pd.isna(row["epsilon"]):
        return "Baseline"
    return f"eps={row['epsilon']:g}"


def selected_dp(df: pd.DataFrame) -> pd.DataFrame:
    """Pick one DP point per model for compact baseline-vs-DP plots."""
    dp = df[df["epsilon"].notna()].copy()
    return dp.sort_values(["model", "perplexity"]).groupby("model", as_index=False).first()


def load_cis(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["model", "run"])
    ci = pd.read_csv(path)
    ci = ci.rename(
        columns={
            "eps": "run",
            "gap_ci_lower": "gap_lo",
            "gap_ci_upper": "gap_hi",
            "wbc_ci_lower": "wbc_lo",
            "wbc_ci_upper": "wbc_hi",
        }
    )
    keep = ["model", "run", "gap_lo", "gap_hi", "wbc_lo", "wbc_hi"]
    return ci[[col for col in keep if col in ci.columns]]


def load_results(data_dir: Path, ci_path: Path | None = None) -> tuple[pd.DataFrame, dict[tuple[str, str], list[float]]]:
    rows: list[dict] = []
    exposure_samples: dict[tuple[str, str], list[float]] = {}
    ci = load_cis(ci_path or data_dir / "bootstrap_cis.csv")

    for path in sorted(data_dir.glob("*.json")):
        match = RESULT_RE.match(path.name)
        if not match:
            continue
        with path.open() as fh:
            data = json.load(fh)

        ce = data.get("canary_exposure") or {}
        wbc = data.get("wbc") or {}
        pii = data.get("pii_extraction") or {}
        ppl = data.get("perplexity") or {}
        model = data.get("model_key") or match.group("model")
        run = data.get("run") or match.group("run")
        epsilon = data.get("epsilon")
        exposure_samples[(model, run)] = [float(v) for v in ce.get("member_exposures", [])]
        rows.append(
            {
                "model": model,
                "model_label": model_label(model),
                "run": run,
                "epsilon": np.nan if epsilon is None else float(epsilon),
                "gap": float(ce.get("gap", 0.0) or 0.0),
                "exposure_mean": float(ce.get("exposure_mean", 0.0) or 0.0),
                "exposure_p50": float(ce.get("exposure_p50", 0.0) or 0.0),
                "exposure_p95": float(ce.get("exposure_p95", 0.0) or 0.0),
                "exposure_max": float(ce.get("exposure_max", 0.0) or 0.0),
                "wbc_auc": float(wbc.get("wbc_auc", 0.0) or 0.0),
                "tpr_at_fpr001": float(wbc.get("tpr_at_fpr001", 0.0) or 0.0),
                "pii_extraction_rate": float(pii.get("pii_extraction_rate", 0.0) or 0.0),
                "avg_rouge1": float(pii.get("avg_rouge1", 0.0) or 0.0),
                "perplexity": float(ppl.get("perplexity", 0.0) or 0.0),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No profile_matrix result JSON files found in {data_dir}")
    df = df.merge(ci, how="left", on=["model", "run"])
    df["display_run"] = df.apply(run_label, axis=1)
    return df.sort_values(["model", "epsilon"], na_position="first"), exposure_samples


def load_detailed_records(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load per-example logs: member/reference losses, WBC scores, and PII attempts."""
    loss_rows: list[dict] = []
    score_rows: list[dict] = []
    pii_rows: list[dict] = []

    for path in sorted(data_dir.glob("*.json")):
        match = RESULT_RE.match(path.name)
        if not match:
            continue
        with path.open() as fh:
            data = json.load(fh)

        model = data.get("model_key") or match.group("model")
        run = data.get("run") or match.group("run")
        epsilon = data.get("epsilon")
        ce = data.get("canary_exposure") or {}
        wbc = data.get("wbc") or {}
        pii = data.get("pii_extraction") or {}

        pii_by_id = {
            item.get("canary_id"): item
            for item in pii.get("details", [])
            if isinstance(item, dict) and item.get("canary_id")
        }

        for split, key in [("member", "member_losses"), ("reference", "reference_losses")]:
            for i, item in enumerate(ce.get(key, []) or []):
                if isinstance(item, dict):
                    canary_id = item.get("id", f"{split}_{i}")
                    pii_item = pii_by_id.get(canary_id, {})
                    loss_rows.append(
                        {
                            "model": model,
                            "model_label": model_label(model),
                            "run": run,
                            "epsilon": np.nan if epsilon is None else float(epsilon),
                            "display_run": "Baseline" if epsilon is None else f"eps={float(epsilon):g}",
                            "split": split,
                            "canary_id": canary_id,
                            "field_type": item.get("field_type", "UNKNOWN"),
                            "exposure_count": item.get("exposure", np.nan),
                            "loss": float(item.get("loss", np.nan)),
                            "exposure_score": float(item.get("exposure_score", np.nan))
                            if item.get("exposure_score") is not None
                            else np.nan,
                            "target": item.get("target", ""),
                            "prefix": item.get("prefix", ""),
                            "exact_hit": bool(pii_item.get("exact_hit", False)),
                            "rouge1": float(pii_item.get("rouge1", 0.0) or 0.0),
                            "completion": pii_item.get("completion", ""),
                        }
                    )

        for split, key in [("member", "member_scores"), ("reference", "reference_scores")]:
            for i, score in enumerate(wbc.get(key, []) or []):
                score_rows.append(
                    {
                        "model": model,
                        "model_label": model_label(model),
                        "run": run,
                        "epsilon": np.nan if epsilon is None else float(epsilon),
                        "display_run": "Baseline" if epsilon is None else f"eps={float(epsilon):g}",
                        "split": split,
                        "score": float(score),
                    }
                )

        for item in pii.get("details", []) or []:
            if not isinstance(item, dict):
                continue
            pii_rows.append(
                {
                    "model": model,
                    "model_label": model_label(model),
                    "run": run,
                    "epsilon": np.nan if epsilon is None else float(epsilon),
                    "display_run": "Baseline" if epsilon is None else f"eps={float(epsilon):g}",
                    "canary_id": item.get("canary_id", ""),
                    "field_type": item.get("field_type", "UNKNOWN"),
                    "exact_hit": bool(item.get("exact_hit", False)),
                    "rouge1": float(item.get("rouge1", 0.0) or 0.0),
                    "completion": item.get("completion", ""),
                    "target": item.get("target", ""),
                }
            )

    return pd.DataFrame(loss_rows), pd.DataFrame(score_rows), pd.DataFrame(pii_rows)


def savefig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_baseline_vs_dp(df: pd.DataFrame, out_dir: Path) -> None:
    base = df[df["epsilon"].isna()].copy()
    dp = selected_dp(df)
    compact = pd.concat([base.assign(kind="Baseline"), dp.assign(kind="DP")], ignore_index=True)
    compact["model_label"] = compact["model"].map(model_label)

    metrics = [
        ("gap", "Memorization gap"),
        ("wbc_auc", "WBC AUC"),
        ("exposure_mean", "Mean canary exposure"),
        ("perplexity", "Perplexity"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        x = np.arange(len(base["model"].unique()))
        width = 0.34
        models = sorted(base["model"].unique())
        base_vals = [float(base.loc[base["model"] == m, metric].iloc[0]) for m in models]
        dp_vals = [float(dp.loc[dp["model"] == m, metric].iloc[0]) for m in models]
        ax.bar(x - width / 2, base_vals, width, label="Baseline", color="#595959")
        ax.bar(x + width / 2, dp_vals, width, label="DP", color=[COLORS[m] for m in models])

        if metric in {"gap", "wbc_auc"}:
            for i, m in enumerate(models):
                for offset, frame in [(-width / 2, base), (width / 2, dp)]:
                    row = frame.loc[frame["model"] == m].iloc[0]
                    lo = row.get(f"{'gap' if metric == 'gap' else 'wbc'}_lo")
                    hi = row.get(f"{'gap' if metric == 'gap' else 'wbc'}_hi")
                    if pd.notna(lo) and pd.notna(hi):
                        ax.errorbar(i + offset, row[metric], yerr=[[row[metric] - lo], [hi - row[metric]]], color="#222", capsize=3, linewidth=1)

        ax.set_title(ylabel)
        ax.set_xticks(x, [model_label(m) for m in models], rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        style_axes(ax)

    axes[0, 0].legend(frameon=False, ncols=2)
    fig.suptitle("Baseline vs DP Privacy Profile", fontsize=16, fontweight="bold")
    fig.tight_layout()
    savefig(fig, out_dir, "baseline_vs_dp_privacy_profile")


def plot_epsilon_sweep(df: pd.DataFrame, out_dir: Path) -> None:
    dp = df[df["epsilon"].notna()].copy()
    base = df[df["epsilon"].isna()].copy()
    metrics = [
        ("gap", "Memorization gap"),
        ("wbc_auc", "WBC AUC"),
        ("exposure_mean", "Mean canary exposure"),
        ("perplexity", "Perplexity"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        for model, group in dp.groupby("model"):
            group = group.sort_values("epsilon")
            ax.plot(group["epsilon"], group[metric], marker="o", linewidth=2, label=model_label(model), color=COLORS.get(model))
            baseline = base.loc[base["model"] == model, metric]
            if not baseline.empty:
                ax.axhline(float(baseline.iloc[0]), color=COLORS.get(model), linestyle="--", linewidth=1, alpha=0.35)

        ax.set_xscale("log", base=2)
        ax.set_xticks(EPS_ORDER, [f"{e:g}" for e in EPS_ORDER])
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        style_axes(ax)

    axes[0, 0].legend(frameon=False, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("Privacy budget epsilon")
    fig.suptitle("DP Epsilon Sweep", fontsize=16, fontweight="bold")
    fig.tight_layout()
    savefig(fig, out_dir, "epsilon_sweep")


def plot_privacy_utility(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    for model, group in df.groupby("model"):
        base = group[group["epsilon"].isna()]
        dp = group[group["epsilon"].notna()]
        ax.scatter(dp["perplexity"], dp["wbc_auc"], s=55, color=COLORS.get(model), alpha=0.82, label=model_label(model))
        if not base.empty:
            ax.scatter(base["perplexity"], base["wbc_auc"], marker="s", s=85, color=COLORS.get(model), edgecolor="#222")

    ax.axhline(0.5, color="#777", linestyle=":", linewidth=1)
    ax.set_xlabel("Perplexity (lower is better)")
    ax.set_ylabel("WBC AUC (lower is more private)")
    ax.set_title("Privacy-Utility Tradeoff")
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    savefig(fig, out_dir, "privacy_utility_tradeoff")


def plot_exposure_boxplots(df: pd.DataFrame, exposure_samples: dict[tuple[str, str], list[float]], out_dir: Path) -> None:
    base = df[df["epsilon"].isna()].copy()
    dp = selected_dp(df)
    selected = pd.concat([base.assign(kind="Baseline"), dp.assign(kind="DP")], ignore_index=True)

    labels = []
    samples = []
    colors = []
    for model in sorted(selected["model"].unique()):
        for kind in ["Baseline", "DP"]:
            row = selected[(selected["model"] == model) & (selected["kind"] == kind)].iloc[0]
            labels.append(f"{model_label(model)}\n{kind}")
            samples.append(exposure_samples[(model, row["run"])])
            colors.append("#595959" if kind == "Baseline" else COLORS.get(model, "#777777"))

    fig, ax = plt.subplots(figsize=(11, 5.7))
    bp = ax.boxplot(samples, patch_artist=True, showfliers=False, whis=(5, 95))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
    for median in bp["medians"]:
        median.set_color("white")
        median.set_linewidth(2)

    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=20, ha="right")
    ax.set_ylabel("Canary exposure")
    ax.set_title("Member Canary Exposure Distribution")
    style_axes(ax)
    fig.tight_layout()
    savefig(fig, out_dir, "exposure_boxplots")


def plot_relative_change_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    base = df[df["epsilon"].isna()].set_index("model")
    dp = selected_dp(df).set_index("model")
    rows = []
    for model in sorted(base.index):
        b = base.loc[model]
        d = dp.loc[model]
        rows.append(
            {
                "model": model_label(model),
                "Gap reduction": 1 - d["gap"] / b["gap"],
                "WBC-AUC excess reduction": 1 - (d["wbc_auc"] - 0.5) / (b["wbc_auc"] - 0.5),
                "Exposure reduction": 1 - d["exposure_mean"] / b["exposure_mean"],
                "Perplexity increase": d["perplexity"] / b["perplexity"] - 1,
            }
        )
    mat = pd.DataFrame(rows).set_index("model")

    fig, ax = plt.subplots(figsize=(9, 4.4))
    image = ax.imshow(mat.values, cmap="YlGnBu", vmin=0, vmax=max(1.0, float(mat.max().max())))
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=20, ha="right")
    ax.set_yticks(range(mat.shape[0]), mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.iloc[i, j] * 100:.0f}%", ha="center", va="center", color="#111", fontweight="bold")
    ax.set_title("Relative Change From Baseline")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Relative change")
    fig.tight_layout()
    savefig(fig, out_dir, "dp_relative_change_heatmap")


def selected_run_keys(df: pd.DataFrame) -> set[tuple[str, str]]:
    base = df[df["epsilon"].isna()][["model", "run"]]
    dp = selected_dp(df)[["model", "run"]]
    return set(map(tuple, pd.concat([base, dp], ignore_index=True).to_numpy()))


def plot_loss_distributions(df: pd.DataFrame, loss_df: pd.DataFrame, out_dir: Path) -> None:
    """Full member/reference canary-loss histograms for baseline and selected DP."""
    chosen = selected_run_keys(df)
    plot_df = loss_df[loss_df.apply(lambda r: (r["model"], r["run"]) in chosen, axis=1)].copy()
    models = sorted(plot_df["model"].unique())
    fig, axes = plt.subplots(len(models), 2, figsize=(11.5, 10.5), sharex=True, sharey=True)
    bins = np.linspace(plot_df["loss"].quantile(0.005), plot_df["loss"].quantile(0.995), 42)

    for i, model in enumerate(models):
        model_runs = plot_df[plot_df["model"] == model]
        run_order = ["baseline"] + [r for r in model_runs["run"].unique() if r != "baseline"]
        for j, run in enumerate(run_order[:2]):
            ax = axes[i, j]
            group = model_runs[model_runs["run"] == run]
            member = group[group["split"] == "member"]["loss"]
            reference = group[group["split"] == "reference"]["loss"]
            ax.hist(reference, bins=bins, density=True, histtype="stepfilled", alpha=0.28, color="#777777", label="Reference")
            ax.hist(member, bins=bins, density=True, histtype="step", linewidth=2.0, color=COLORS.get(model), label="Member")
            ax.axvline(member.mean(), color=COLORS.get(model), linewidth=1.4)
            ax.axvline(reference.mean(), color="#555555", linewidth=1.2, linestyle="--")
            ax.set_title(f"{model_label(model)} - {run.replace('dp_eps', 'eps=')}")
            if j == 0:
                ax.set_ylabel("Density")
            if i == len(models) - 1:
                ax.set_xlabel("Target-token loss")
            style_axes(ax)

    axes[0, 0].legend(frameon=False)
    fig.suptitle("Full Canary Loss Distributions", fontsize=16, fontweight="bold")
    fig.tight_layout()
    savefig(fig, out_dir, "loss_distributions_member_vs_reference")


def plot_exposure_survival(df: pd.DataFrame, exposure_samples: dict[tuple[str, str], list[float]], out_dir: Path) -> None:
    """CCDF curves: fraction of member canaries with exposure >= x."""
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    interesting_eps = [np.nan, 3.0, 8.0, 64.0]
    line_styles = {np.nan: "-", 3.0: ":", 8.0: "-.", 64.0: "--"}

    for ax, model in zip(axes.flat, models):
        model_rows = df[df["model"] == model].copy()
        for eps in interesting_eps:
            if pd.isna(eps):
                row = model_rows[model_rows["epsilon"].isna()]
                label = "Baseline"
            else:
                row = model_rows[model_rows["epsilon"] == eps]
                label = f"eps={eps:g}"
            if row.empty:
                continue
            run = row.iloc[0]["run"]
            values = np.sort(np.asarray(exposure_samples[(model, run)], dtype=float))
            y = 1.0 - np.arange(len(values)) / len(values)
            ax.step(values, y, where="post", label=label, linewidth=2, linestyle=line_styles.get(eps, "-"), color=COLORS.get(model))
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1.05)
        ax.set_title(model_label(model))
        ax.set_xlabel("Exposure")
        ax.set_ylabel("Fraction of member canaries >= exposure")
        style_axes(ax)

    axes[0, 0].legend(frameon=False)
    fig.suptitle("Exposure Tail Curves From All Member Canaries", fontsize=16, fontweight="bold")
    fig.tight_layout()
    savefig(fig, out_dir, "exposure_survival_curves")


def plot_wbc_score_distributions(df: pd.DataFrame, score_df: pd.DataFrame, out_dir: Path) -> None:
    """Full member/reference WBC score histograms for baseline and selected DP."""
    chosen = selected_run_keys(df)
    plot_df = score_df[score_df.apply(lambda r: (r["model"], r["run"]) in chosen, axis=1)].copy()
    models = sorted(plot_df["model"].unique())
    fig, axes = plt.subplots(len(models), 2, figsize=(11.5, 10.5), sharex=True, sharey=True)
    bins = np.linspace(plot_df["score"].quantile(0.005), plot_df["score"].quantile(0.995), 42)

    for i, model in enumerate(models):
        model_runs = plot_df[plot_df["model"] == model]
        run_order = ["baseline"] + [r for r in model_runs["run"].unique() if r != "baseline"]
        for j, run in enumerate(run_order[:2]):
            ax = axes[i, j]
            group = model_runs[model_runs["run"] == run]
            member = group[group["split"] == "member"]["score"]
            reference = group[group["split"] == "reference"]["score"]
            ax.hist(reference, bins=bins, density=True, histtype="stepfilled", alpha=0.28, color="#777777", label="Reference")
            ax.hist(member, bins=bins, density=True, histtype="step", linewidth=2.0, color=COLORS.get(model), label="Member")
            ax.set_title(f"{model_label(model)} - {run.replace('dp_eps', 'eps=')}")
            if j == 0:
                ax.set_ylabel("Density")
            if i == len(models) - 1:
                ax.set_xlabel("WBC membership score")
            style_axes(ax)

    axes[0, 0].legend(frameon=False)
    fig.suptitle("Full WBC Membership-Score Distributions", fontsize=16, fontweight="bold")
    fig.tight_layout()
    savefig(fig, out_dir, "wbc_score_distributions")


def plot_field_type_exact_hits(df: pd.DataFrame, pii_df: pd.DataFrame, out_dir: Path) -> None:
    """Exact secret extraction rates by canary field type."""
    chosen = selected_run_keys(df)
    plot_df = pii_df[pii_df.apply(lambda r: (r["model"], r["run"]) in chosen, axis=1)].copy()
    plot_df["kind"] = np.where(plot_df["epsilon"].isna(), "Baseline", "DP")
    rates = (
        plot_df.groupby(["model", "field_type", "kind"], as_index=False)["exact_hit"]
        .mean()
        .sort_values(["model", "field_type", "kind"])
    )

    field_types = sorted(rates["field_type"].unique())
    models = sorted(rates["model"].unique())
    fig, axes = plt.subplots(1, len(field_types), figsize=(12, 4.4), sharey=True)
    if len(field_types) == 1:
        axes = [axes]
    for ax, field in zip(axes, field_types):
        sub = rates[rates["field_type"] == field]
        x = np.arange(len(models))
        width = 0.35
        base_vals = [sub[(sub["model"] == m) & (sub["kind"] == "Baseline")]["exact_hit"].mean() for m in models]
        dp_vals = [sub[(sub["model"] == m) & (sub["kind"] == "DP")]["exact_hit"].mean() for m in models]
        base_vals = [0 if pd.isna(v) else v for v in base_vals]
        dp_vals = [0 if pd.isna(v) else v for v in dp_vals]
        ax.bar(x - width / 2, base_vals, width, color="#595959", label="Baseline")
        ax.bar(x + width / 2, dp_vals, width, color=[COLORS.get(m, "#777777") for m in models], label="DP")
        ax.set_title(field)
        ax.set_xticks(x, [model_label(m) for m in models], rotation=20, ha="right")
        ax.set_ylabel("Exact extraction rate")
        style_axes(ax)
    axes[0].legend(frameon=False)
    fig.suptitle("Exact Secret Extraction by Field Type", fontsize=16, fontweight="bold")
    fig.tight_layout()
    savefig(fig, out_dir, "exact_hits_by_field_type")


def write_top_canary_tables(df: pd.DataFrame, loss_df: pd.DataFrame, out_dir: Path) -> None:
    """CSV evidence tables from individual member canaries, not aggregates."""
    chosen = selected_run_keys(df)
    member = loss_df[
        (loss_df["split"] == "member")
        & loss_df.apply(lambda r: (r["model"], r["run"]) in chosen, axis=1)
    ].copy()
    member["target_preview"] = member["target"].astype(str).str.slice(0, 80)
    member["completion_preview"] = member["completion"].astype(str).str.slice(0, 80)
    member["prefix_preview"] = member["prefix"].astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 120)

    cols = [
        "model",
        "run",
        "display_run",
        "canary_id",
        "field_type",
        "exposure_count",
        "loss",
        "exact_hit",
        "rouge1",
        "target_preview",
        "completion_preview",
        "prefix_preview",
    ]
    member.sort_values(["exact_hit", "loss"], ascending=[False, True]).head(100)[cols].to_csv(
        out_dir / "top_memorized_member_canaries.csv", index=False
    )
    member.sort_values("loss").groupby(["model", "run"], as_index=False).head(25)[cols].to_csv(
        out_dir / "lowest_loss_member_canaries_by_run.csv", index=False
    )


def plot_lowest_loss_canaries(df: pd.DataFrame, loss_df: pd.DataFrame, out_dir: Path) -> None:
    """Shows the low-loss tail for each selected run, where memorization is most visible."""
    chosen = selected_run_keys(df)
    member = loss_df[
        (loss_df["split"] == "member")
        & loss_df.apply(lambda r: (r["model"], r["run"]) in chosen, axis=1)
    ].copy()
    rows = []
    for (model, run), group in member.groupby(["model", "run"]):
        for rank, (_, row) in enumerate(group.nsmallest(30, "loss").iterrows(), start=1):
            rows.append({"model": model, "run": run, "rank": rank, "loss": row["loss"]})
    top = pd.DataFrame(rows)

    models = sorted(top["model"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, models):
        sub = top[top["model"] == model]
        for run, group in sub.groupby("run"):
            label = "Baseline" if run == "baseline" else run.replace("dp_eps", "eps=")
            ax.plot(group["rank"], group["loss"], marker="o", markersize=3.5, linewidth=1.8, label=label)
        ax.set_title(model_label(model))
        ax.set_xlabel("Rank among lowest-loss member canaries")
        ax.set_ylabel("Loss")
        ax.invert_yaxis()
        style_axes(ax)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Lowest-Loss Member Canary Tail", fontsize=16, fontweight="bold")
    fig.tight_layout()
    savefig(fig, out_dir, "lowest_loss_member_canary_tail")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create paper figures from profile_matrix JSON results.")
    parser.add_argument("--data-dir", type=Path, default=HERE)
    parser.add_argument("--ci", type=Path, default=None, help="Optional bootstrap_cis.csv path")
    parser.add_argument("--out-dir", type=Path, default=HERE / "figures")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df, exposure_samples = load_results(args.data_dir, args.ci)
    loss_df, score_df, pii_df = load_detailed_records(args.data_dir)
    df.to_csv(args.out_dir / "profile_matrix_summary.csv", index=False)
    loss_df.to_csv(args.out_dir / "per_canary_losses.csv", index=False)
    score_df.to_csv(args.out_dir / "per_example_wbc_scores.csv", index=False)
    pii_df.to_csv(args.out_dir / "per_canary_pii_attempts.csv", index=False)

    plot_baseline_vs_dp(df, args.out_dir)
    plot_epsilon_sweep(df, args.out_dir)
    plot_privacy_utility(df, args.out_dir)
    plot_exposure_boxplots(df, exposure_samples, args.out_dir)
    plot_relative_change_heatmap(df, args.out_dir)
    plot_loss_distributions(df, loss_df, args.out_dir)
    plot_exposure_survival(df, exposure_samples, args.out_dir)
    plot_wbc_score_distributions(df, score_df, args.out_dir)
    plot_field_type_exact_hits(df, pii_df, args.out_dir)
    plot_lowest_loss_canaries(df, loss_df, args.out_dir)
    write_top_canary_tables(df, loss_df, args.out_dir)

    print(f"Wrote figures to {args.out_dir}")
    print("Best paper candidates:")
    print("  1. baseline_vs_dp_privacy_profile.pdf")
    print("  2. epsilon_sweep.pdf")
    print("  3. privacy_utility_tradeoff.pdf")
    print("  4. exposure_boxplots.pdf")
    print("  5. dp_relative_change_heatmap.pdf")
    print("  6. loss_distributions_member_vs_reference.pdf")
    print("  7. exposure_survival_curves.pdf")
    print("  8. wbc_score_distributions.pdf")
    print("  9. exact_hits_by_field_type.pdf")
    print(" 10. lowest_loss_member_canary_tail.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
