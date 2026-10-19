#!/usr/bin/env python3
"""Passes and possession validation PDF (Spanish).

For each clip it compares, frame by frame, when the ball is really visible (a CVAT
ball CSV) with what the possession engine concluded, and, when a person annotated
the passes, the detected passes with the real ones. Nothing is estimated beyond the
stored job results.

    python scripts/ops/passes_pdf.py --truth passes-truth.json --workdir build/ \\
        --run clip_003=<results folder>:<ball csv> ... --output informe-pases.pdf

Truth file (passes are optional per clip; a clip without a ``passes`` key is not scored):
    {"clips": {"clip_003": {"description": "...", "tolerance_seconds": 1.0,
        "passes": [{"approx_time_seconds": 4.2, "note": "azul a azul"}]}}}
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ops.validation_pdf import _REASONS, _table, tex  # noqa: E402

DEFAULT_TOLERANCE = 1.0


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_clip(name: str, folder: Path, csv_path: Path | None, truth: dict[str, Any] | None) -> dict[str, Any]:
    rows = [json.loads(x) for x in (folder / "possession_events.jsonl").read_text().splitlines() if x.strip()]
    job_id = folder.name
    events_path = folder.parent / f"{job_id}-events.json"
    events = _json(events_path).get("events", []) if events_path.is_file() else []
    stats_path = folder.parent / f"{job_id}-stats.json"
    stats = _json(stats_path) if stats_path.is_file() else {}
    results = _json(folder / "results.json") if (folder / "results.json").is_file() else {}
    video = results.get("video") or {}
    fps = float(video.get("fps") or 30.0)

    status_by_frame = {r["frame_index"]: r["status"] for r in rows}
    reasons: dict[str, int] = {}
    for r in rows:
        if r["status"] != "confirmed":
            reasons[r.get("rejection_reason")] = reasons.get(r.get("rejection_reason"), 0) + 1

    ball_truth: dict[int, tuple[float, float] | None] = {}
    if csv_path and csv_path.is_file():
        with csv_path.open() as handle:
            for rec in csv.DictReader(handle):
                visible = rec.get("Visibility") == "1"
                ball_truth[int(rec["Frame"])] = (float(rec["X"]), float(rec["Y"])) if visible else None

    both = [f for f in ball_truth if f in status_by_frame]
    visible_truth = [f for f in both if ball_truth[f] is not None]
    hidden_truth = [f for f in both if ball_truth[f] is None]
    detected = lambda f: status_by_frame[f] != "lost"  # noqa: E731
    ball = {
        "truth_visible_pct": 100 * len(visible_truth) / len(both) if both else None,
        "system_visible_pct": 100 * sum(1 for f in both if detected(f)) / len(both) if both else None,
        "recall_pct": 100 * sum(1 for f in visible_truth if detected(f)) / len(visible_truth) if visible_truth else None,
        "false_pct": 100 * sum(1 for f in hidden_truth if detected(f)) / len(hidden_truth) if hidden_truth else None,
        "frames_compared": len(both),
    }

    wanted = (truth or {}).get("passes")
    pass_events = [e for e in events if e.get("type") == "pass"]
    transfers = [e for e in events if e.get("type") == "possession_transfer"]
    tolerance = float((truth or {}).get("tolerance_seconds", DEFAULT_TOLERANCE))

    jersey_of: dict[str, str] = {}
    numbers_path = folder / "numbers.json"
    if numbers_path.is_file():
        for player in _json(numbers_path).get("players", []):
            if player.get("status") == "confirmed" and player.get("number"):
                jersey_of[player["stable_player_id"]] = str(player["number"])
    for event in pass_events + transfers:
        event["from_jersey"] = jersey_of.get(event.get("from_stable_player_id"))
        event["to_jersey"] = jersey_of.get(event.get("to_stable_player_id"))

    def _label(event: dict[str, Any]) -> str:
        return f"{event.get('from_jersey') or '?'}→{event.get('to_jersey') or '?'}"

    # A pass counts as correct only when a person verified that detection; matching jersey pairs
    # without an instant could credit a coincidence, so it is not used.
    verdicts = (truth or {}).get("verdicts", [])

    def _verdict(event: dict[str, Any]) -> str | None:
        for item in verdicts:
            if item.get("kind") == event["type"] and abs(item["time_seconds"] - event["timestamp_seconds"]) <= 0.35:
                event["verdict_note"] = item.get("note")
                return item.get("verdict")
        return None

    for event in pass_events + transfers:
        event["verdict"] = _verdict(event)
    matched = sum(1 for e in pass_events if e["verdict"] == "real")
    real_wrong_jersey = [e for e in pass_events if e["verdict"] == "real_camiseta_mal"]
    unverified = [e for e in pass_events if e["verdict"] is None]
    false_passes = [e for e in pass_events if e["verdict"] == "falsa"]
    transfers_real = [e for e in transfers if e["verdict"] == "real"]
    near = []
    if wanted is not None:
        used_real = set()
        for event in pass_events:
            pair = (event["from_jersey"], event["to_jersey"])
            for index, expected in enumerate(wanted):
                if index in used_real or expected.get("from") is None or pair == (str(expected["from"]), str(expected["to"])):
                    continue
                other = (str(expected["from"]), str(expected["to"]))
                if None in pair or sum(1 for a, b in zip("".join(pair), "".join(other)) if a != b) != 1 or len("".join(pair)) != len("".join(other)):
                    continue
                near.append(f"{other[0]}→{other[1]} detectado como {pair[0]}→{pair[1]}")
                used_real.add(index)
                break
    sequence_real = ", ".join(f"{x.get('from', '?')}→{x.get('to', '?')}" for x in wanted) if wanted else ""
    sequence_found = ", ".join(_label(e) for e in pass_events)
    return {
        "clip": name,
        "description": (truth or {}).get("description"),
        "fps": fps,
        "frames": len(rows),
        "duration": (video.get("duration_seconds") or (len(rows) / fps if rows else None)),
        "confirmed": sum(1 for r in rows if r["status"] == "confirmed"),
        "status_by_frame": status_by_frame,
        "reasons": sorted(((_REASONS.get(k, str(k)), v) for k, v in reasons.items()), key=lambda kv: -kv[1])[:3],
        "ball": ball,
        "ball_truth": ball_truth,
        "pass_events": pass_events,
        "transfers": transfers,
        "near_misses": near,
        "real_wrong_jersey": real_wrong_jersey,
        "unverified": unverified,
        "false_passes": false_passes,
        "transfers_real": transfers_real,
        "sequence_real": sequence_real,
        "sequence_found": sequence_found,
        "changes_real": (truth or {}).get("possession_changes"),
        "wanted": wanted,
        "matched": matched,
        "linked_to_roster": (stats.get("coverage") or {}).get("players_linked_to_roster"),
        "shots": len([e for e in events if e.get("type") == "shot"]),
    }


def timeline(clip: dict[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fps = clip["fps"]
    frames = sorted(clip["status_by_frame"])
    colors = {"confirmed": "#2ca02c", "unknown": "#f2c744", "lost": "#d9d9d9", "unassigned": "#9467bd"}
    fig, ax = plt.subplots(3, 1, figsize=(9, 2.9), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1.2]})
    for f in frames:
        t = f / fps
        truth = clip["ball_truth"].get(f, "na")
        if truth != "na":
            ax[0].axvspan(t, t + 1 / fps, color="#1f77b4" if truth is not None else "#f0f0f0", lw=0)
        ax[1].axvspan(t, t + 1 / fps, color=colors.get(clip["status_by_frame"][f], "#999999"), lw=0)
    for e in clip["pass_events"]:
        ax[2].plot(e["timestamp_seconds"], 0.7, marker="v", color="#d62728", ms=9)
    for e in clip["transfers"]:
        ax[2].plot(e["timestamp_seconds"], 0.7, marker="x", color="#ff7f0e", ms=8)
    for expected in clip["wanted"] or []:
        if expected.get("approx_time_seconds") is not None:
            ax[2].plot(expected["approx_time_seconds"], 0.3, marker="^", color="#2ca02c", ms=9)
    ax[0].set_yticks([]); ax[1].set_yticks([]); ax[2].set_yticks([]); ax[2].set_ylim(0, 1)
    ax[0].set_ylabel("balón real", rotation=0, ha="right", va="center", fontsize=8)
    ax[1].set_ylabel("sistema", rotation=0, ha="right", va="center", fontsize=8)
    ax[2].set_ylabel("pases", rotation=0, ha="right", va="center", fontsize=8)
    ax[2].set_xlabel("segundos")
    ax[2].set_xlim(0, (frames[-1] + 1) / fps if frames else 1)
    legend = [Patch(color="#1f77b4", label="balón visible (real)"), Patch(color="#2ca02c", label="poseedor confirmado"),
              Patch(color="#f2c744", label="poseedor desconocido"), Patch(color="#d9d9d9", label="balón no detectado")]
    fig.legend(handles=legend, ncol=4, fontsize=7, loc="upper center", frameon=False)
    fig.subplots_adjust(top=0.82, left=0.1, right=0.98, bottom=0.2)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _pct(value: float | None) -> str:
    return "n/d" if value is None else f"{value:.0f} %"


def build(args: argparse.Namespace) -> Path:
    truth = _json(args.truth).get("clips", {}) if args.truth.is_file() else {}
    work = args.workdir
    work.mkdir(parents=True, exist_ok=True)
    clips = []
    for spec in args.run:
        name, _, rest = spec.partition("=")
        folder, _, csv_path = rest.partition(":")
        clips.append(load_clip(name, Path(folder), Path(csv_path) if csv_path else None, truth.get(name)))
    for clip in clips:
        timeline(clip, work / f"{clip['clip']}-timeline.png")

    summary = _table(
        ["Clip", "Duración", "Balón visible (real)", "Balón detectado", "Con poseedor", "Pases reales", "Pases detectados"],
        [[c["clip"], f"{c['duration']:.1f} s" if c["duration"] else "n/d", _pct(c["ball"]["truth_visible_pct"]),
          _pct(c["ball"]["system_visible_pct"]), f"{c['confirmed']} de {c['frames']}",
          str(len(c["wanted"])) if c["wanted"] is not None else "sin anotar", str(len(c["pass_events"]))] for c in clips],
        "lcccccc")
    scored = [c for c in clips if c["wanted"] is not None]
    real = sum(len(c["wanted"]) for c in scored)
    hit = sum(c["matched"] for c in scored)
    false = sum(len(c["pass_events"]) - c["matched"] for c in scored)
    sections = []
    for c in clips:
        reasons = ", ".join(f"{k} ({v})" for k, v in c["reasons"]) or "n/d"
        wanted = c["wanted"]
        passes_text = (f"Pases reales: {len(wanted)}; detectados correctamente: {c['matched']}; detectados sin pase real: "
                       f"{len(c['pass_events']) - c['matched']}.") if wanted is not None else (
            f"Pases anotados: ninguno (clip no evaluado); el sistema detectó {len(c['pass_events'])}.")
        ball = c["ball"]
        recall = (f"De los cuadros en que el balón es visible, el sistema lo detectó en {_pct(ball['recall_pct'])}; "
                  f"detectó balón en {_pct(ball['false_pct'])} de los cuadros en que no es visible.") if ball["recall_pct"] is not None else \
            "Sin anotación de balón para comparar."
        sections.append(
            r"\clearpage\subsection{" + tex(c["clip"]) + "}\n"
            + (r"\textit{Hechos:} " + tex(c["description"]) + "\n\n" if c["description"] else "")
            + tex(passes_text) + " " + tex(recall) + "\n\n"
            + r"Posesión: poseedor confirmado en " + f"{c['confirmed']} de {c['frames']}" + r" cuadros. Motivos principales de no confirmar: "
            + tex(reasons) + ".\n\n"
            + r"\begin{figure}[H]\centering\includegraphics[width=\textwidth]{" + f"{c['clip']}-timeline.png" + "}"
            + r"\caption{" + tex(c["clip"]) + r": balón real (arriba), conclusión del sistema por cuadro (centro) y pases (abajo: "
            r"triángulos verdes = reales, rojos = detectados, cruces = cambios de posesión).}\end{figure}")

    headline = (f"Pases reales anotados: \\textbf{{{real}}}; detectados correctamente: \\textbf{{{hit}}}; detecciones sin pase real: "
                f"\\textbf{{{false}}}." if scored else "Todavía no hay pases anotados, por lo que solo se muestra qué detectó el sistema.")
    document = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage[spanish]{babel}
\usepackage[margin=2.3cm]{geometry}\usepackage{graphicx,booktabs,float,xcolor,array,fancyhdr,caption}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}\setlength{\parskip}{0.55em}
\pagestyle{fancy}\fancyhf{}\rhead{ScoutAI --- pases y posesión}\cfoot{\thepage}
\title{Pases y posesión del balón: validación sobre videos de entrenamiento\\[2pt]\large ScoutAI --- documento para revisión del equipo}
\author{}\date{18 de octubre de 2026}
\begin{document}\maketitle
\section{Resumen}
""" + headline + r"""

\textbf{Importante.} Los cuatro videos provienen de la carpeta de \emph{entrenamiento} del conjunto de datos del balón. Si el detector del balón se
ajustó con ellos, la detección del balón aquí es más favorable que sobre videos nuevos y \textbf{no es una validación independiente}.

\section{Método}
Cada video se analizó con el sistema desplegado (mismo flujo que los videos anteriores). Para el balón se usa la anotación por cuadro
del conjunto de datos (visible o no, y posición). Para los pases, una persona anotó cuándo ocurren; un pase detectado cuenta si está
a menos de """ + f"{DEFAULT_TOLERANCE:g}".replace(".", ",") + r""" segundo del instante anotado. Un pase solo se declara cuando el sistema confirma
dos poseedores distintos del mismo equipo, por eso depende de que el balón se vea y se asocie a un jugador.

\section{Resultados}
{\small
""" + summary + r"""}

""" + "\n".join(sections) + r"""

\clearpage\section{Limitaciones}
\begin{itemize}
\item Son cuatro videos de entrenamiento y de otros partidos y ligas; los resultados no son una tasa de acierto general.
\item Un pase requiere dos poseedores confirmados; si el balón no se detecta o no se asocia a un jugador, el pase no aparece.
\item Asistencias, pérdidas, robos y tiempo de posesión siguen sin medirse.
\end{itemize}
\end{document}
"""
    tex_file = work / "pases.tex"
    tex_file.write_text(document, encoding="utf-8")
    for _ in range(2):
        proc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_file.name], cwd=work, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout[-3000:])
            raise SystemExit("pdflatex failed")
    return work / "pases.pdf"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--truth", type=Path, default=Path("passes-truth.json"))
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="CLIP=FOLDER:BALL_CSV")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pdf = build(args)
    args.output.write_bytes(pdf.read_bytes())
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
