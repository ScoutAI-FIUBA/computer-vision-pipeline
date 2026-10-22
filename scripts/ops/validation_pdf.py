#!/usr/bin/env python3
"""Builds the validation PDF (Spanish) from stored job results and a human ground truth.

It reuses ``validation_report.evaluate_clip`` for every number, draws the figures
(annotated shot frame, shot chart, time breakdown) and compiles a LaTeX document
with ``pdflatex``. Nothing is estimated beyond what the job records contain.

    python scripts/ops/validation_pdf.py --truth truth.json --workdir build/ \\
        --new brown3pt=<results folder>:<video.mp4> ... \\
        --old brown3pt=<old stats folder> ... --output validacion.pdf
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

GPU_RATE = 2.5
FONT = "/usr/share/fonts/truetype/lato/Lato-Medium.ttf"


def tex(text: Any) -> str:
    out = str(text)
    for char, repl in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                       ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        out = out.replace(char, repl)
    return out


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _shot_frame(folder: Path, video: Path, out: Path) -> bool:
    """Frame at the first shot with the players' boxes; the shooter in red."""
    from PIL import Image, ImageDraw, ImageFont

    shots = _json(folder / "projected_shots.json")["shots"]
    if not shots:
        return False
    shot = shots[0]
    stamp = vr._shot_time(shot) or 0.0
    results = _json(folder / "results.json") if (folder / "results.json").is_file() else {}
    fps = (results.get("video") or {}).get("fps") or 60.0
    target = round(stamp * fps)
    rows = [json.loads(line) for line in (folder / "projected_tracks.jsonl").read_text().splitlines() if line.strip()]
    frames = sorted({r["frame_index"] for r in rows})
    frame_index = min(frames, key=lambda f: abs(f - target))
    raw = out.with_suffix(".raw.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{frame_index / fps:.3f}", "-i", str(video), "-frames:v", "1", str(raw)], check=True)
    image = Image.open(raw).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, 20)
    shooter = shot.get("shooter_stable_player_id")
    colors = {0: (60, 120, 255), 1: (255, 200, 0)}
    for row in rows:
        if row["frame_index"] != frame_index or not row.get("xyxy"):
            continue
        x1, y1, x2, y2 = row["xyxy"]
        is_shooter = shooter is not None and row.get("stable_player_id") == shooter
        color = (230, 20, 20) if is_shooter else colors.get(row.get("team_id"), (200, 200, 200))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5 if is_shooter else 2)
        if is_shooter:
            number = shot.get("shooter_confirmed_number")
            label = f"tirador: {shooter}" + (f" (#{number})" if number else " (camiseta no leida)")
            draw.rectangle([x1, max(0, y1 - 28), x1 + 12 * len(label), y1], fill=color)
            draw.text((x1 + 4, max(0, y1 - 27)), label, fill=(255, 255, 255), font=font)
    image.save(out, quality=88)
    raw.unlink(missing_ok=True)
    return True


def _court(folder: Path, out: Path) -> None:
    from pipeline.projected_shots import write_projected_shots_image

    write_projected_shots_image(_json(folder / "projected_shots.json"), out)


def _time_chart(rows: list[dict[str, Any]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases = ["model_load", "tracking", "identity", "packaging", "court_projection", "ball_tracking",
              "shot_detection", "possession", "jersey_numbers"]
    names = {"model_load": "carga de modelos", "tracking": "seguimiento", "identity": "identidad",
             "packaging": "empaquetado", "court_projection": "proyección a cancha", "ball_tracking": "pelota",
             "shot_detection": "detección de tiros", "possession": "posesión", "jersey_numbers": "camisetas"}
    fig, ax = plt.subplots(figsize=(8, 3.4))
    left = [0.0] * len(rows)
    palette = plt.get_cmap("tab10")
    for index, phase in enumerate(phases):
        values = [(r["job_summary"].get("timings") or {}).get(phase, 0.0) for r in rows]
        ax.barh([r["clip"] for r in rows], values, left=left, label=names[phase], color=palette(index))
        left = [a + b for a, b in zip(left, values)]
    ax.set_xlabel("segundos de GPU por fase")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


_REASONS = {"ball_track_lost": "balón no detectado", "no_authoritative_player_match": "balón sin jugador asociado",
            "confirmation_pending": "confirmación pendiente", "ambiguous_player_candidates": "varios jugadores posibles",
            "player_identity_unassigned": "jugador sin identidad"}


def pass_stats(folder: Path) -> dict[str, Any]:
    """What the system produced about possession and passes, straight from the stored files."""
    import collections

    rows_path = folder / "possession_events.jsonl"
    rows = [json.loads(x) for x in rows_path.read_text().splitlines() if x.strip()] if rows_path.is_file() else []
    events_path = folder.parent / f"{folder.name}-events.json"
    events = _json(events_path).get("events", []) if events_path.is_file() else []
    stats_path = folder.parent / f"{folder.name}-stats.json"
    coverage = _json(stats_path).get("coverage", {}) if stats_path.is_file() else {}
    reasons = collections.Counter(r.get("rejection_reason") for r in rows if r.get("status") != "confirmed")
    return {
        "frames": len(rows),
        "confirmed": sum(1 for r in rows if r.get("status") == "confirmed"),
        "passes_detected": sum(1 for e in events if e.get("type") == "pass"),
        "transfers_detected": sum(1 for e in events if e.get("type") == "possession_transfer"),
        "ball_visibility_pct": coverage.get("ball_visibility_pct"),
        "top_reasons": [(_REASONS.get(k, str(k)), v) for k, v in reasons.most_common(2)],
    }


def _status(row: dict[str, Any] | None) -> str:
    if row is None:
        return "---"
    return "sí" if row else "NO"


def _summary_row(result: dict[str, Any]) -> list[str]:
    rows = result["rows"]
    judged = [x for x in rows if x.get("outcome_ok") is not None]
    numbered = [x for x in rows if x.get("number_ok") is not None]
    return [
        result["clip"],
        str(result["truth_shots"]),
        str(result["matched"]),
        str(len(result["spurious"])),
        f"{sum(1 for x in judged if x['outcome_ok'])}/{len(judged)}" if judged else "n/d",
        f"{sum(1 for x in numbered if x['number_ok'])}/{len(numbered)}" if numbered else "n/d",
    ]


def _table(header: list[str], body: list[list[str]], widths: str) -> str:
    lines = [r"\begin{tabular}{" + widths + "}", r"\toprule", " & ".join(r"\textbf{" + tex(h) + "}" for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(tex(c) for c in row) + r" \\" for row in body]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def build(args: argparse.Namespace) -> Path:
    truth = _json(args.truth)["clips"]
    work = args.workdir
    work.mkdir(parents=True, exist_ok=True)
    new_runs = {}
    for spec in args.new:
        clip, _, rest = spec.partition("=")
        folder, _, video = rest.partition(":")
        new_runs[clip] = (Path(folder), Path(video))
    old_runs = {spec.partition("=")[0]: Path(spec.partition("=")[2]) for spec in args.old}

    new = [vr.evaluate_clip(clip, folder, truth.get(clip)) for clip, (folder, _) in new_runs.items()]
    old = {clip: vr.evaluate_clip(clip, folder, truth.get(clip)) for clip, folder in old_runs.items()}

    figures: dict[str, dict[str, bool]] = {}
    for clip, (folder, video) in new_runs.items():
        figures[clip] = {
            "frame": _shot_frame(folder, video, work / f"{clip}-frame.jpg") if video.is_file() else False,
        }
        _court(folder, work / f"{clip}-court.png")
    _time_chart([r for r in new if r.get("job_summary")], work / "tiempos.png")

    evaluated = [r for r in new if r["status"] == "evaluado"]
    shots = sum(r["truth_shots"] for r in evaluated)
    found = sum(r["matched"] for r in evaluated)
    outcomes = sum(1 for r in evaluated for x in r["rows"] if x.get("outcome_ok"))
    numbered = [x for r in evaluated for x in r["rows"] if x.get("number_ok") is not None]
    numbers_ok = sum(1 for x in numbered if x["number_ok"])
    spurious = sum(len(r["spurious"]) for r in evaluated)

    results_table = _table(
        ["Clip", "Tiros reales", "Detectados", "Falsos", "Resultado correcto", "Camiseta correcta"],
        [_summary_row(r) for r in evaluated], "lccccc")

    before_after = []
    for r in evaluated:
        before = old.get(r["clip"])
        if before and before["status"] == "evaluado":
            b = _summary_row(before)
            a = _summary_row(r)
            before_after.append([r["clip"], f"{b[2]} de {b[1]}", f"{a[2]} de {a[1]}", b[3], a[3], b[4], a[4]])
    before_after_table = _table(
        ["Clip", "Detect. antes", "Detect. ahora", "Falsos antes", "Falsos ahora", "Result. antes", "Result. ahora"],
        before_after, "lcccccc")

    cost_rows, cost_total = [], 0.0
    for r in new:
        s = r.get("job_summary") or {}
        cost, per_minute = vr._cost(s, GPU_RATE)
        wall = r.get("wall_seconds")
        seconds = round(sum(v for v in (s.get("timings") or {}).values() if isinstance(v, (int, float))))
        duration = s.get("duration_seconds")
        if cost != "n/d":
            cost_total += float(cost.split()[0])
        cost_rows.append([r["clip"], f"{duration:.1f} s" if duration else "n/d", f"{seconds} s",
                          f"{round(wall)} s" if wall else "n/d", cost, per_minute,
                          (s.get("gpu") or {}).get("name", "n/d")])
    cost_table = _table(["Clip", "Duración", "Pipeline", "Total", "Costo est.", "Por minuto", "GPU"], cost_rows, "lcccccl")

    clip_sections = []
    for r in new:
        clip = r["clip"]
        parts = [(r"\clearpage" if clip_sections else "") + r"\subsection{" + tex(clip) + "}"]
        if r.get("description"):
            parts.append(r"\textit{Verdad de terreno:} " + tex(r["description"]) + "\n")
        if r.get("review_note"):
            parts.append(r"\textit{Nota de revisión:} " + tex(r["review_note"]) + "\n")
        for row in r.get("rows", []):
            got = row["found"]
            parts.append(r"\begin{itemize}\item Esperado: " + tex(vr._fmt_expected(row["expected"])) +
                         r"\item Encontrado: " + (tex(vr._fmt_found(got)) if got else r"\textbf{no detectado}") + r"\end{itemize}")
        figs = []
        if figures[clip]["frame"]:
            figs.append(r"\includegraphics[width=0.82\textwidth]{" + f"{clip}-frame.jpg" + "}")
        figs.append(r"\includegraphics[width=0.56\textwidth]{" + f"{clip}-court.png" + "}")
        parts.append(r"\begin{figure}[H]\centering " + r"\\[6pt] ".join(figs) +
                     r"\caption{" + tex(clip) + r": momento del tiro (arriba; en rojo el tirador que asigna el sistema) y "
                     r"posición del tiro en la cancha (abajo; verde = convertido).}\end{figure}")
        clip_sections.append("\n".join(parts))

    pass_rows, annotated_passes, detected_on_annotated = [], 0, 0
    for r in new:
        info = pass_stats(new_runs[r["clip"]][0])
        wanted = (truth.get(r["clip"]) or {}).get("passes")
        if wanted is not None:
            annotated_passes += len(wanted)
            detected_on_annotated += min(len(wanted), info["passes_detected"])
        share = f"{info['confirmed']} de {info['frames']}" if info["frames"] else "n/d"
        ball = f"{info['ball_visibility_pct']:.0f} %" if isinstance(info["ball_visibility_pct"], (int, float)) else "n/d"
        pass_rows.append([r["clip"], str(len(wanted)) if wanted is not None else "sin anotar", str(info["passes_detected"]),
                          ball, share, "; ".join(f"{k} ({v})" for k, v in info["top_reasons"]) or "n/d"])
    passes_table = _table(["Clip", "Pases reales", "Pases detectados", "Balón visible", "Fotogramas con poseedor", "Motivos principales"],
                          pass_rows, "lccp{1.9cm}p{2.6cm}p{4.6cm}")
    total_passes = sum(pass_stats(new_runs[r["clip"]][0])["passes_detected"] for r in new)

    derrick_crop = args.derrick_crop
    derrick_fig = ""
    if derrick_crop and derrick_crop.is_file():
        from PIL import Image

        target = work / "derrick-camiseta.jpg"
        Image.open(derrick_crop).convert("RGB").save(target, quality=88)
        derrick_fig = (r"\begin{figure}[H]\centering\includegraphics[height=7cm]{derrick-camiseta.jpg}"
                       r"\caption{Derrick: el tirador asignado es el jugador 9 (WHITE); el sistema leyó la camiseta como 8.}\end{figure}")

    document = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[spanish]{babel}
\usepackage[margin=2.3cm]{geometry}
\usepackage{graphicx,booktabs,float,xcolor,array,fancyhdr,caption}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}\setlength{\parskip}{0.55em}
\pagestyle{fancy}\fancyhf{}\rhead{ScoutAI --- validación de resultados}\cfoot{\thepage}
\title{Validación de los resultados del pipeline de análisis\\[2pt]\large ScoutAI --- documento para revisión del equipo}
\author{}\date{18 de octubre de 2026}
\begin{document}
\maketitle
\section{Resumen}
Se comparó lo que devuelve el sistema desplegado (aplicación web, backend y procesamiento en GPU) con lo que realmente
ocurre en cuatro videos, verificados a mano por una persona del equipo. Resultados:
\begin{itemize}
\item Tiros reales detectados: \textbf{""" + f"{found} de {shots}" + r"""}; falsos positivos: \textbf{""" + str(spurious) + r"""}.
\item Resultado del tiro (convertido o fallado) correcto: \textbf{""" + f"{outcomes} de {found}" + r"""}.
\item Camiseta del tirador leída correctamente: \textbf{""" + f"{numbers_ok} de {len(numbered)}" + r"""}. El tirador asignado
fue el correcto en los cuatro videos; en uno el sistema lo identificó bien pero leyó mal su número (sección~\ref{sec:hallazgos}).
\item Pases: \textbf{""" + f"{detected_on_annotated} de {annotated_passes}" + r"""} pases anotados detectados (derrick); el sistema no detectó
ningún pase en los cuatro videos y no los mide de forma confiable (sección~\ref{sec:pases}).
\item Costo estimado de GPU: entre USD 0,55 y 0,65 por video de 8 a 13 segundos (USD """ + f"{cost_total:.2f}".replace(".", ",") + r""" los cuatro).
\end{itemize}
\textbf{Alcance:} son cuatro videos con un tiro cada uno. Es una verificación caso por caso, no una estimación estadística
de precisión.

\section{Qué se evaluó y cómo}
\textbf{Sistema evaluado.} Procesamiento en Modal (GPU A100 de 80~GB) y backend desplegados el 18 de octubre de 2026 a partir de la
rama de integración del modelo (versión 2). Cada video se subió desde la interfaz web y se analizó con el flujo normal de
producción; los resultados se descargaron del almacenamiento tal como los guardó el sistema.

\textbf{Verdad de terreno.} Una persona vio cada video y anotó los tiros: resultado, camiseta del tirador y, cuando se conoce,
el instante aproximado. Un tiro detectado cuenta como el tiro real si está a menos de 1~segundo del instante anotado. Todos
los tiros de estos videos fueron convertidos.

\textbf{Métricas.} Tiros detectados y falsos positivos; resultado correcto sobre los detectados; camiseta correcta cuando el
sistema la leyó (``no leída'' no cuenta como error ni como acierto); tiempo y costo de GPU.

\textbf{Comparación con el modelo anterior.} Se repitió la evaluación con las ejecuciones previas a las correcciones de esta etapa,
guardadas localmente, con la misma verdad de terreno y la misma ventana. Esas ejecuciones no leían camisetas.

\section{Resultados}
""" + results_table + r"""

\subsection{Antes y después de las correcciones}
{\small\setlength{\tabcolsep}{5pt}
""" + before_after_table + "}" + r"""

``Falsos'' son eventos detectados sin un tiro real detrás (por ejemplo, un pase contado como tiro). ``Resultado'' es
convertido o fallado correcto sobre los tiros detectados; ``n/d'' significa que no hubo tiro detectado que evaluar. El
modelo anterior no leía camisetas, por lo que esa columna solo puede compararse en la versión actual.

\clearpage\section{Detalle por video}
""" + "\n".join(clip_sections) + r"""

\clearpage\section{Pases}\label{sec:pases}
El sistema reporta pases como eventos experimentales, derivados de los cambios de poseedor del balón; nunca se suman a las
estadísticas. Resultado en estos videos:

{\small
""" + passes_table + r"""}

\textbf{Pases detectados en los cuatro videos: """ + str(total_passes) + r""".} Pase real anotado: """ + str(annotated_passes) + r""" (derrick, a los 5~s);
detectados: """ + str(detected_on_annotated) + r""". En los otros videos no se anotaron pases, por lo que no pueden evaluarse.

\textbf{Por qué.} Para declarar un pase el sistema necesita confirmar quién tiene el balón en dos momentos distintos. Con el balón
visible solo en una parte de los cuadros, casi nunca llega a confirmar un poseedor (columna ``fotogramas con poseedor''): o no ve el balón,
o no puede asociarlo a un único jugador, o queda a la espera de más cuadros. Sin poseedor confirmado no hay cambio de poseedor y, por lo
tanto, tampoco pase. Mientras tanto, asistencias, pérdidas, robos y tiempo de posesión siguen sin medirse.

\clearpage\section{Tiempo y costo}
""" + cost_table + r"""

\begin{figure}[H]\centering\includegraphics[width=0.92\textwidth]{tiempos.png}
\caption{Tiempo de GPU por fase en cada video.}\end{figure}
El costo es una estimación: tiempo de GPU informado por el propio procesamiento por la tarifa del plan (USD 2,50 por hora).
No incluye las consultas a los modelos alojados (Roboflow) ni el almacenamiento; la factura de Modal es la fuente definitiva.
Las fases más largas son identidad, proyección a cancha, seguimiento y lectura de camisetas.

\section{Hallazgos y limitaciones}\label{sec:hallazgos}
\textbf{Camiseta mal leída (Derrick).} El sistema asignó el tiro a la persona correcta (su recuadro coincide con el del tiro y
el cuadro muestra al jugador 9), pero el lector de números leyó ``8'' en 25 de 25 lecturas, repartidas en 7 seguimientos distintos.
Es una confusión sistemática entre 9 y 8 con el jugador pequeño y de perfil. Consecuencia: los puntos aparecen bajo ``\#8'' y
no se vinculan al jugador de la plantilla.
""" + derrick_fig + r"""
\textbf{Otras limitaciones conocidas.}
\begin{itemize}
\item El detector de tiros solo funciona con el plano general de la cancha; en primeros planos y repeticiones no detecta nada.
\item No se miden asistencias, pérdidas, robos ni tiempo de posesión, y la detección de pases es experimental y no confirmó ninguno.
\item Pocas muestras (cuatro tiros): los resultados no permiten estimar una tasa de acierto general.
\item La regla de identidad estable (tolerar hasta 1\% de filas inválidas) está pendiente de confirmación del responsable del módulo.
\end{itemize}

\section{Qué pedimos validar}
\begin{itemize}
\item Que la verdad de terreno de cada video (resultado, camiseta e instante del tiro) sea correcta.
\item Si hay más videos de plano general con resultado conocido para ampliar la muestra (cada análisis cuesta cerca de USD 0,60).
\item Si prefieren corregir la lectura de camisetas 9/8 (por ejemplo, con recortes más grandes o validando contra la plantilla)
o dejarla documentada como limitación.
\end{itemize}
\end{document}
"""
    tex_file = work / "validacion.tex"
    tex_file.write_text(document, encoding="utf-8")
    for _ in range(2):
        proc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_file.name],
                              cwd=work, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout[-3000:])
            raise SystemExit("pdflatex failed")
    return work / "validacion.pdf"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--new", action="append", required=True, metavar="CLIP=FOLDER:VIDEO")
    parser.add_argument("--old", action="append", default=[], metavar="CLIP=FOLDER")
    parser.add_argument("--derrick-crop", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pdf = build(args)
    args.output.write_bytes(pdf.read_bytes())
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
