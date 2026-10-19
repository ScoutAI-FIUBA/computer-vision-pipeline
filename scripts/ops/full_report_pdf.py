#!/usr/bin/env python3
"""Full validation PDF (Spanish): shots, jersey numbers, ball, possession, passes, time and cost.

Built for a set of clips with human-supplied facts. It composes the pieces already used by
``validation_pdf`` (shots, figures, cost) and ``passes_pdf`` (ball truth, possession,
passes). Players are only ever referred to by jersey number or id, never by name.

    python scripts/ops/full_report_pdf.py --truth facts.json --workdir build/ \\
        --run clip_003=<results folder>:<video.mp4>:<ball.csv> ... --output informe.pdf

Facts file: per clip ``description``, ``tolerance_seconds``, ``shots`` (outcome, jersey_number,
approx_time_seconds) and ``passes`` (approx_time_seconds, note). A key that is absent means
"not annotated": that part is shown but not scored.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ops import validation_report as vr  # noqa: E402
from scripts.ops.passes_pdf import _pct, load_clip, timeline  # noqa: E402
from scripts.ops.validation_pdf import GPU_RATE, _court, _shot_frame, _summary_row, _table, _time_chart, tex  # noqa: E402


def build(args: argparse.Namespace) -> Path:
    facts = json.loads(args.truth.read_text(encoding="utf-8")).get("clips", {}) if args.truth.is_file() else {}
    work = args.workdir
    work.mkdir(parents=True, exist_ok=True)

    clips: list[dict[str, Any]] = []
    for spec in args.run:
        name, _, rest = spec.partition("=")
        folder, video, ball_csv = (rest.split(":") + ["", ""])[:3]
        truth = facts.get(name)
        shots = vr.evaluate_clip(name, Path(folder), truth)
        passes = load_clip(name, Path(folder), Path(ball_csv) if ball_csv else None, truth)
        figures = {"frame": False, "court": False}
        if shots.get("shots_detected") and shots.get("court_projected"):
            if video and Path(video).is_file():
                figures["frame"] = _shot_frame(Path(folder), Path(video), work / f"{name}-frame.jpg")
            _court(Path(folder), work / f"{name}-court.png")
            figures["court"] = True
        timeline(passes, work / f"{name}-timeline.png")
        clips.append({"name": name, "shots": shots, "passes": passes, "figures": figures})

    with_summary = [c["shots"] for c in clips if c["shots"].get("job_summary")]
    if with_summary:
        _time_chart(with_summary, work / "tiempos.png")

    scored = [c["shots"] for c in clips if c["shots"]["status"] == "evaluado"]
    ambiguous = [c["shots"] for c in clips if c["shots"]["status"] == "evaluado_ambiguo"]
    shots_real = sum(r["truth_shots"] for r in scored)
    shots_found = sum(r["matched"] for r in scored)
    shots_false = sum(len(r["spurious"]) for r in scored)
    outcome_ok = sum(1 for r in scored for x in r["rows"] if x.get("outcome_ok"))
    judged = sum(1 for r in scored for x in r["rows"] if x.get("outcome_ok") is not None)
    numbered = [x for r in scored for x in r["rows"] if x.get("number_ok") is not None]
    numbers_ok = sum(1 for x in numbered if x["number_ok"])
    pass_scored = [c["passes"] for c in clips if c["passes"]["wanted"] is not None]
    passes_real = sum(len(p["wanted"]) for p in pass_scored)
    passes_declared = sum(len(p["pass_events"]) for p in pass_scored)
    passes_hit = sum(p["matched"] for p in pass_scored)
    passes_wrong_jersey = sum(len(p["real_wrong_jersey"]) for p in pass_scored)
    passes_unverified = sum(len(p["unverified"]) for p in pass_scored)
    passes_false = sum(len(p["false_passes"]) for p in pass_scored)
    changes_real = sum(p["changes_real"] or 0 for p in pass_scored)
    changes_hit = sum(len(p["transfers_real"]) for p in pass_scored)

    def shots_line() -> str:
        if not scored:
            return "Tiros: sin tiros evaluables todavía; se muestran los detectados."
        text = (f"Tiros reales detectados: \\textbf{{{shots_found} de {shots_real}}}; falsos positivos: \\textbf{{{shots_false}}}; "
                f"resultado correcto: \\textbf{{{outcome_ok} de {judged}}}")
        if numbered:
            text += f"; camiseta del tirador leída correctamente: \\textbf{{{numbers_ok} de {len(numbered)}}}"
        if ambiguous:
            text += (f". En {len(ambiguous)} videos el número de tiros detectados no coincide con los anotados y, sin instantes, no se puede "
                     "emparejar cada uno: se muestran lado a lado, sin evaluar")
        return text + "."

    def passes_line() -> str:
        if not pass_scored:
            return "Pases: todavía sin pases anotados; se muestran los detectados sin evaluar."
        return (f"Pases reales anotados: \\textbf{{{passes_real}}}. El sistema declaró \\textbf{{{passes_declared}}}: una persona verificó "
                f"\\textbf{{{passes_hit + passes_wrong_jersey}}} como pases reales, de los cuales \\textbf{{{passes_hit}}} con ambas camisetas bien "
                f"leídas y \\textbf{{{passes_wrong_jersey}}} con una camiseta mal leída; \\textbf{{{passes_false}}} resultaron falsos y "
                f"\\textbf{{{passes_unverified}}} quedaron sin verificar. Cambios de posesión reales: {changes_real}; detectados correctamente: {changes_hit}.")

    cost_rows, cost_total = [], 0.0
    for c in clips:
        s = c["shots"].get("job_summary") or {}
        cost, per_minute = vr._cost(s, GPU_RATE)
        seconds = round(sum(v for v in (s.get("timings") or {}).values() if isinstance(v, (int, float))))
        duration = s.get("duration_seconds")
        wall = c["shots"].get("wall_seconds")
        if cost != "n/d":
            cost_total += float(cost.split()[0])
        cost_rows.append([c["name"], f"{duration:.1f} s" if duration else "n/d", f"{seconds} s", f"{round(wall)} s" if wall else "n/d",
                          cost, per_minute])
    cost_table = _table(["Clip", "Duración", "Pipeline", "Total", "Costo est.", "Por minuto"], cost_rows, "lccccc")

    shots_rows = [_summary_row(r) for r in scored] + [[r["clip"], str(r["truth_shots"]), f"{r['shots_detected']} detectados", "—", "no evaluable", "no evaluable"]
                                                       for r in ambiguous]
    shots_table = _table(
        ["Clip", "Tiros reales", "Detectados bien", "Adicionales", "Resultado correcto", "Camiseta correcta"],
        shots_rows, "lccccc") if shots_rows else "Sin tiros anotados."
    overview = _table(
        ["Clip", "Duración", "Jugadores seguidos", "Con camiseta", "Tiros detectados"],
        [[c["name"], f"{c['passes']['duration']:.1f} s" if c["passes"]["duration"] else "n/d",
          str(c["shots"].get("players_tracked") if c["shots"].get("players_tracked") is not None else "n/d"),
          str(c["shots"].get("players_with_confirmed_number") if c["shots"].get("players_with_confirmed_number") is not None else "n/d"),
          str(c["shots"].get("shots_detected", "n/d"))] for c in clips], "lcccc")
    ball_table = _table(
        ["Clip", "Balón real", "Detectado", "Recall", "Con poseedor", "Pases reales", "Pases det."],
        [[c["name"], _pct(c["passes"]["ball"]["truth_visible_pct"]), _pct(c["passes"]["ball"]["system_visible_pct"]),
          _pct(c["passes"]["ball"]["recall_pct"]), f"{c['passes']['confirmed']} de {c['passes']['frames']}",
          str(len(c["passes"]["wanted"])) if c["passes"]["wanted"] is not None else "sin anotar", str(len(c["passes"]["pass_events"]))]
         for c in clips], "lcccccc")

    sections = []
    for c in clips:
        s, p = c["shots"], c["passes"]
        parts = [(r"\clearpage\section{Detalle por video}" if not sections else r"\clearpage") + r"\subsection{" + tex(c["name"]) + "}"]
        if s.get("description"):
            parts.append(r"\textit{Hechos:} " + tex(s["description"]) + "\n")
        if s.get("review_note"):
            parts.append(r"\textit{Verificación humana:} " + tex(s["review_note"]) + "\n")
        parts.append(f"Jugadores seguidos: {s.get('players_tracked') if s.get('players_tracked') is not None else 'n/d'}; con camiseta confirmada: "
                     f"{s.get('players_with_confirmed_number') if s.get('players_with_confirmed_number') is not None else 'n/d'}.\n")
        if not s.get("court_projected", True):
            parts.append(r"\textbf{No se pudo proyectar la cancha en este video:} el análisis terminó, pero sin posiciones de jugadores ni de tiros en "
                         "la cancha. Los tiros que se muestran vienen directamente del detector, sin tirador asignado; en la aplicación este video "
                         "aparecería con cero tiros y cero jugadores en cancha.\n")
        if s["status"] == "evaluado":
            for row in s["rows"]:
                got = row["found"]
                parts.append(r"\begin{itemize}\item Tiro esperado: " + tex(vr._fmt_expected(row["expected"])) + r"\item Encontrado: "
                             + (tex(vr._fmt_found(got)) if got else r"\textbf{no detectado}") + r"\end{itemize}")
            for extra in s["spurious"]:
                parts.append(r"Tiro detectado sin tiro real: " + tex(vr._fmt_found(extra)) + ".\n")
        elif s["status"] == "evaluado_ambiguo":
            parts.append("Tiros anotados: " + tex("; ".join(vr._fmt_expected(x) for x in s["expected"])) + ".\n")
            parts.append(r"\textbf{No evaluable:} el sistema detectó " + str(len(s["detections"])) + " tiros y se anotaron " + str(s["truth_shots"])
                         + "; sin instantes no se puede emparejar cada uno y no se afirma cuál es correcto. Detectados, en orden: "
                         + tex("; ".join(vr._fmt_found(d) for d in s["detections"])) + ".\n")
        else:
            detected = "; ".join(vr._fmt_found(d) for d in s.get("detections", [])) or "ninguno"
            parts.append("Tiros: sin anotar. Detectados por el sistema: " + tex(detected) + ".\n")
        wanted = p["wanted"]
        if wanted:
            parts.append("Pases reales, en orden (quién pasa a quién): " + tex(p["sequence_real"]) + ".\n")
        if p["pass_events"]:
            parts.append("Pases detectados: " + tex(p["sequence_found"]) + " (``?'' = camiseta no leída).\n")
        if p["near_misses"]:
            parts.append("Casi aciertos (un dígito distinto): " + tex("; ".join(p["near_misses"])) + ".\n")
        if p["changes_real"] is not None:
            parts.append(f"Cambios de posesión reales: {p['changes_real']}; detectados por el sistema: {len(p['transfers'])}.\n")
        parts.append("Pases: " + (f"reales {len(wanted)}, detectados correctamente {p['matched']}, no coinciden "
                                  f"{len(p['pass_events']) - p['matched']} (no coinciden con la lista)." if wanted is not None
                                  else f"sin anotar; el sistema detectó {len(p['pass_events'])}.")
                     + f" Poseedor confirmado en {p['confirmed']} de {p['frames']} cuadros; motivos principales: "
                     + (", ".join(f"{k} ({v})" for k, v in p["reasons"]) or "n/d") + ".\n")
        figs = []
        if c["figures"]["frame"]:
            figs.append(r"\includegraphics[width=0.6\textwidth]{" + f"{c['name']}-frame.jpg" + "}")
        if c["figures"]["court"]:
            figs.append(r"\includegraphics[width=0.36\textwidth]{" + f"{c['name']}-court.png" + "}")
        if figs:
            parts.append(r"\begin{figure}[H]\centering " + r"\\[4pt] ".join(figs) + r"\caption{" + tex(c["name"])
                         + r": momento del primer tiro (en rojo el tirador asignado) y posición en la cancha.}\end{figure}")
        parts.append(r"\begin{figure}[H]\centering\includegraphics[width=\textwidth]{" + f"{c['name']}-timeline.png" + "}"
                     r"\caption{" + tex(c["name"]) + r": balón real, conclusión del sistema por cuadro y pases (verde = reales, "
                     r"rojo = detectados, cruz = cambio de posesión).}\end{figure}")
        sections.append("\n".join(parts))

    document = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage[spanish]{babel}
\usepackage[margin=2.3cm]{geometry}\usepackage{graphicx,booktabs,float,xcolor,array,fancyhdr,caption}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}\setlength{\parskip}{0.55em}
\pagestyle{fancy}\fancyhf{}\rhead{ScoutAI --- validación sobre videos de entrenamiento}\cfoot{\thepage}
\title{Validación completa sobre videos de entrenamiento\\[2pt]\large ScoutAI --- tiros, camisetas, pases y posesión}
\author{}\date{18 de octubre de 2026}
\begin{document}\maketitle
\section{Resumen}
Se analizaron """ + str(len(clips)) + r""" videos con el sistema desplegado y se compararon sus resultados con hechos anotados por una persona.
\begin{itemize}
\item """ + shots_line() + r"""
\item """ + passes_line() + r"""
\item Costo estimado de GPU: USD """ + f"{cost_total:.2f}".replace(".", ",") + r""" en total.
\end{itemize}
\textbf{Importante.} Los videos provienen de la carpeta de \emph{entrenamiento} del conjunto de datos del balón. Si el detector se ajustó con
ellos, los resultados del balón son más favorables que sobre videos nuevos y \textbf{no son una validación independiente}. Son clips de otras
ligas y ángulos de cámara; los jugadores se mencionan solo por número de camiseta.

\section{Método}
Cada video se subió y se analizó con el flujo normal de producción. Los hechos (tiros, resultado, camiseta del tirador, pases) los anotó una
persona. Un evento detectado cuenta si está a menos de 1~segundo del instante anotado. Para el balón se usa la anotación por cuadro del conjunto de
datos. Una parte sin anotar no se evalúa; se muestra lo que detectó el sistema.

\section{Resultados}
\subsection{Panorama}
{\small
""" + overview + r"""}

\subsection{Tiros}
{\small
""" + shots_table + r"""}

``Adicionales'' son tiros que el sistema declaró y que una persona verificó que no existen. Un tiro cuyo resultado el sistema dejó sin resolver cuenta como no correcto.

\subsection{Balón, posesión y pases}
{\small\setlength{\tabcolsep}{4pt}
""" + ball_table + r"""}

``Balón real'' es el porcentaje de cuadros en que el balón es visible según la anotación; ``Detectado'', en cuántos lo vio el sistema; ``Recall'', de los
cuadros con balón visible, en cuántos lo detectó; ``Con poseedor'', cuadros con un poseedor confirmado.

""" + "\n".join(sections) + r"""

\clearpage\section{Tiempo y costo}
{\small
""" + cost_table + r"""}
""" + (r"\begin{figure}[H]\centering\includegraphics[width=0.92\textwidth]{tiempos.png}\caption{Tiempo de GPU por fase en cada video.}\end{figure}"
       if with_summary else "") + r"""
El costo es una estimación: tiempo de GPU informado por el procesamiento por la tarifa del plan (USD 2,50 por hora); no incluye consultas a modelos
alojados ni almacenamiento. La factura de Modal es la fuente definitiva.

\section{Limitaciones}
\begin{itemize}
\item Pocos videos, de entrenamiento y de otras ligas: no permiten estimar una tasa de acierto general.
\item Un pase solo se declara con dos poseedores confirmados del mismo equipo; depende de que el balón se detecte y se asocie a un jugador.
\item El detector de tiros funciona con el plano general de la cancha; en primeros planos y repeticiones no detecta nada.
\item Asistencias, pérdidas, robos y tiempo de posesión siguen sin medirse.
\end{itemize}
\end{document}
"""
    tex_file = work / "informe.tex"
    tex_file.write_text(document, encoding="utf-8")
    for _ in range(2):
        proc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_file.name], cwd=work, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout[-3000:])
            raise SystemExit("pdflatex failed")
    return work / "informe.pdf"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--truth", type=Path, default=Path("facts.json"))
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="CLIP=FOLDER:VIDEO:BALL_CSV")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pdf = build(args)
    args.output.write_bytes(pdf.read_bytes())
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
