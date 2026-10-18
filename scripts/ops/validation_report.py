#!/usr/bin/env python3
"""Compares what the pipeline returned with what really happened in each clip.

Unlike ``acceptance_report.py`` (which checks that a staged run completed and its
artifacts are intact), this answers "was the result right?". It needs a ground
truth written by a person. Nothing is invented: a clip without truth is listed as
not evaluated.

Inputs per run: the folder with the result files the system stored for one job
(``projected_shots.json`` and, when present, ``numbers.json``,
``player_summary.json`` and ``job.json``).

    python scripts/ops/validation_report.py --truth truth.json \\
        --run brown3pt=analysis-results/<job-id> --run derrick=... --output informe.md

Truth file:
    {"clips": {"brown3pt": {"description": "...", "tolerance_seconds": 2.0,
        "shots": [{"outcome": "made", "jersey_number": "7", "approx_time_seconds": 6.6}]}}}
``jersey_number`` and ``approx_time_seconds`` are optional. An empty ``shots`` list
means "no shot happens in this clip".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_TOLERANCE_SECONDS = 2.0


def _load(folder: Path, name: str) -> Any | None:
    path = folder / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _shot_time(shot: dict[str, Any]) -> float | None:
    for key in ("release_timestamp_seconds", "start_timestamp_seconds", "resolved_timestamp_seconds"):
        if isinstance(shot.get(key), (int, float)):
            return float(shot[key])
    return None


def _match(truth: list[dict[str, Any]], detected: list[dict[str, Any]], tolerance: float):
    """Pairs each true shot with the nearest unused detection (by time when both have one)."""

    remaining = list(range(len(detected)))
    pairs: list[tuple[int, int | None]] = []
    for t_index, expected in enumerate(truth):
        when = expected.get("approx_time_seconds")
        best, best_gap = None, None
        for d_index in remaining:
            at = _shot_time(detected[d_index])
            gap = 0.0 if when is None or at is None else abs(at - when)
            if when is not None and at is not None and gap > tolerance:
                continue
            if best is None or gap < best_gap:  # type: ignore[operator]
                best, best_gap = d_index, gap
        if best is not None:
            remaining.remove(best)
        pairs.append((t_index, best))
    return pairs, remaining


def evaluate_clip(name: str, folder: Path, truth: dict[str, Any] | None) -> dict[str, Any]:
    shots_doc = _load(folder, "projected_shots.json")
    if shots_doc is None:
        return {"clip": name, "status": "sin_resultados", "detail": f"no hay projected_shots.json en {folder}"}
    detected = list(shots_doc.get("shots", []))
    summary_doc = _load(folder, "player_summary.json")
    numbers_doc = _load(folder, "numbers.json")
    job_doc = _load(folder, "job.json") or {}

    players = (summary_doc or {}).get("players", [])
    confirmed = [p for p in players if (p.get("jersey") or {}).get("status") == "confirmed"]
    result: dict[str, Any] = {
        "clip": name,
        "status": "evaluado" if truth is not None else "sin_verdad_terreno",
        "description": (truth or {}).get("description"),
        "shots_detected": len(detected),
        "players_tracked": len(players) if summary_doc else None,
        "players_with_confirmed_number": len(confirmed) if summary_doc else (
            sum(1 for p in (numbers_doc or {}).get("players", []) if p.get("status") == "confirmed")
            if numbers_doc else None
        ),
        "shots_needing_review": sum(1 for s in detected if s.get("needs_review")),
        "job_summary": job_doc.get("summary") if isinstance(job_doc, dict) else None,
    }
    if truth is None:
        result["detections"] = [_describe(s) for s in detected]
        return result

    expected = truth.get("shots", [])
    pairs, spurious = _match(expected, detected, float(truth.get("tolerance_seconds", DEFAULT_TOLERANCE_SECONDS)))
    rows = []
    for t_index, d_index in pairs:
        want = expected[t_index]
        got = detected[d_index] if d_index is not None else None
        row: dict[str, Any] = {"expected": want, "found": _describe(got) if got else None}
        if got is not None:
            row["outcome_ok"] = got.get("outcome") == want.get("outcome")
            if "jersey_number" in want:
                number = got.get("shooter_confirmed_number")
                row["number_ok"] = (number == want["jersey_number"]) if number is not None else None
                row["shooter_attributed"] = got.get("shooter_stable_player_id") is not None
        rows.append(row)
    result.update(
        {
            "truth_shots": len(expected),
            "matched": sum(1 for r in rows if r["found"]),
            "missed_by_system": sum(1 for r in rows if not r["found"]),
            "spurious": [_describe(detected[i]) for i in spurious],
            "rows": rows,
        }
    )
    return result


def _describe(shot: dict[str, Any] | None) -> dict[str, Any] | None:
    if shot is None:
        return None
    return {
        "time_seconds": _shot_time(shot),
        "outcome": shot.get("outcome"),
        "outcome_confidence": shot.get("outcome_confidence"),
        "shooter": shot.get("shooter_stable_player_id"),
        "jersey_number": shot.get("shooter_confirmed_number"),
        "attribution": shot.get("shooter_attribution_method"),
        "needs_review": shot.get("needs_review"),
    }


def _cost(summary: dict[str, Any] | None, rate: float | None) -> tuple[str, str]:
    if not summary or rate is None:
        return "n/d", "n/d"
    timings = summary.get("timings") or {}
    seconds = sum(v for v in timings.values() if isinstance(v, (int, float)))
    duration = summary.get("duration_seconds")
    if seconds <= 0:
        return "n/d", "n/d"
    cost = seconds / 3600 * rate
    per_minute = f"{cost / duration * 60:.2f} USD" if isinstance(duration, (int, float)) and duration else "n/d"
    return f"{cost:.2f} USD", per_minute


_OUTCOMES = {"made": "convertido", "missed": "fallado", "unknown": "sin resolver", None: "sin resolver"}


def _fmt_expected(want: dict[str, Any]) -> str:
    parts = [_OUTCOMES.get(want.get("outcome"), str(want.get("outcome")))]
    if want.get("jersey_number") is not None:
        parts.append(f"camiseta {want['jersey_number']}")
    if want.get("approx_time_seconds") is not None:
        parts.append(f"~{want['approx_time_seconds']} s")
    return ", ".join(parts)


def _fmt_found(got: dict[str, Any]) -> str:
    parts = [_OUTCOMES.get(got.get("outcome"), str(got.get("outcome")))]
    parts.append(f"camiseta {got['jersey_number']}" if got.get("jersey_number") else "camiseta no leída")
    parts.append(f"jugador {got['shooter']}" if got.get("shooter") else "sin jugador asignado")
    if got.get("time_seconds") is not None:
        parts.append(f"{got['time_seconds']:.1f} s")
    if got.get("needs_review"):
        parts.append("requiere revisión")
    return ", ".join(parts)


def _yn(value: Any) -> str:
    return "n/d" if value is None else ("sí" if value else "NO")


def render(results: list[dict[str, Any]], rate: float | None) -> str:
    lines = ["# Validación de resultados contra la verdad de terreno", ""]
    lines.append(
        "Cada fila compara lo que el sistema devolvió con lo que realmente ocurre en el video, "
        "según una verificación humana. `n/d` significa que el dato no está disponible; "
        "nunca se completa con un supuesto."
    )
    lines += ["", "## Resumen", "", "| Clip | Tiros reales | Detectados bien | No detectados | Falsos | Resultado correcto | Camiseta correcta | Tiros a revisar |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r["status"] != "evaluado":
            lines.append(f"| {r['clip']} | — | — | — | — | — | — | {r.get('shots_needing_review', '—')} |")
            continue
        rows = r["rows"]
        outcome = sum(1 for x in rows if x.get("outcome_ok"))
        with_number = [x for x in rows if x.get("number_ok") is not None]
        number = sum(1 for x in with_number if x["number_ok"])
        number_text = f"{number}/{len(with_number)}" if with_number else "n/d"
        lines.append(
            f"| {r['clip']} | {r['truth_shots']} | {r['matched']} | {r['missed_by_system']} | "
            f"{len(r['spurious'])} | {outcome}/{r['matched']} | {number_text} | {r['shots_needing_review']} |"
        )
    for r in results:
        lines += ["", f"## {r['clip']}", ""]
        if r["status"] == "sin_resultados":
            lines.append(f"Sin resultados: {r['detail']}.")
            continue
        if r.get("description"):
            lines.append(f"Verdad de terreno: {r['description']}")
            lines.append("")
        lines.append(
            f"- Jugadores seguidos: {r['players_tracked'] if r['players_tracked'] is not None else 'n/d'}; "
            f"con camiseta confirmada: {r['players_with_confirmed_number'] if r['players_with_confirmed_number'] is not None else 'n/d'}."
        )
        summary = r.get("job_summary") or {}
        cost, per_minute = _cost(summary, rate)
        if summary:
            lines.append(f"- Duración del video: {summary.get('duration_seconds', 'n/d')} s; GPU: {(summary.get('gpu') or {}).get('name', 'n/d')}; costo estimado: {cost} ({per_minute} por minuto de video).")
        if r["status"] == "sin_verdad_terreno":
            lines.append("- Sin verdad de terreno: tiros detectados listados, sin evaluar.")
            for d in r["detections"]:
                lines.append(f"  - {_fmt_found(d)}")
            continue
        lines += ["", "| Esperado | Encontrado | Resultado | Camiseta | Tirador asignado |", "|---|---|---|---|---|"]
        for row in r["rows"]:
            want, got = row["expected"], row["found"]
            if got is None:
                lines.append(f"| {_fmt_expected(want)} | **no detectado** | — | — | — |")
            else:
                lines.append(f"| {_fmt_expected(want)} | {_fmt_found(got)} | {_yn(row.get('outcome_ok'))} | {_yn(row.get('number_ok'))} | {_yn(row.get('shooter_attributed'))} |")
        for extra in r["spurious"]:
            lines.append(f"| (no hay tiro real) | **falso positivo:** {_fmt_found(extra)} | — | — | — |")
    lines += [
        "",
        "## Límites",
        "",
        "- La verdad de terreno la define una persona mirando cada video; son pocos clips, por lo que "
        "los porcentajes no son una estimación estadística, sino una verificación caso por caso.",
        "- El detector solo funciona sobre el plano general de la cancha; primeros planos y repeticiones no producen detecciones.",
        "- Asistencias, pérdidas, robos y tiempo de posesión no se miden.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="CLIP=FOLDER")
    parser.add_argument("--gpu-rate-usd-per-hour", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    truth = json.loads(args.truth.read_text(encoding="utf-8")).get("clips", {})
    results = []
    for spec in args.run:
        if "=" not in spec:
            parser.error(f"--run expects CLIP=FOLDER, got {spec!r}")
        name, _, folder = spec.partition("=")
        results.append(evaluate_clip(name, Path(folder), truth.get(name)))
    text = json.dumps(results, indent=2, ensure_ascii=False) if args.json else render(results, args.gpu_rate_usd_per_hour)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
