#!/usr/bin/env python3
"""Converts the report LaTeX written by the validation generators into one self-contained HTML file.

Only the small LaTeX subset those generators emit is supported (sections, lists, tables,
figures, verbatim, bold/italic/code). Images are embedded as base64 so the file can be sent
as is: text and tables copy cleanly into Docs or Word and images can be saved or pasted.

    python scripts/ops/tex_to_html.py build/informe.tex informe.html
"""
from __future__ import annotations

import argparse
import base64
import html
import re
import sys
from pathlib import Path

CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:920px;margin:2rem auto;padding:0 1rem;line-height:1.55;color:#1d1d1f}
h1{font-size:1.9rem;margin-bottom:.2rem}h1 small{display:block;font-size:1.05rem;font-weight:400;color:#555;margin-top:.3rem}
h2{margin-top:2.2rem;border-bottom:2px solid #e3e3e3;padding-bottom:.25rem}h3{margin-top:1.8rem}
table{border-collapse:collapse;margin:1rem 0;font-size:.92rem}th,td{border:1px solid #d4d4d4;padding:.35rem .6rem;text-align:left;vertical-align:top}
th{background:#f4f4f5}code,pre{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.88em}
pre{background:#f6f6f7;padding:.8rem;overflow-x:auto;border-radius:6px}code{background:#f1f1f2;padding:0 .25rem;border-radius:3px}
figure{margin:1.2rem 0;text-align:center}figure img{max-width:100%;height:auto}figcaption{font-size:.88rem;color:#555;margin-top:.3rem}
.date{color:#555}
@media (prefers-color-scheme:dark){body{background:#161618;color:#e8e8ea}th{background:#26262a}th,td{border-color:#3a3a3f}pre,code{background:#232327}h2{border-color:#333}figcaption,.date,h1 small{color:#aaa}figure img{background:#fff}}
@media print{body{max-width:none;margin:0}}
"""

_ESCAPES = {r"\&": "&", r"\%": "%", r"\$": "$", r"\#": "#", r"\_": "_", r"\{": "{", r"\}": "}",
            r"\textasciitilde{}": "~", r"\^{}": "^", r"\textbackslash{}": "\\"}
_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def _braced(text: str, start: int) -> tuple[str, int]:
    """Content of the {...} group whose '{' is at ``start``; returns (content, index after '}')."""
    if text[start] != "{":
        raise ValueError(f"expected '{{' at {start}: {text[start:start + 30]!r}")
    depth, i = 0, start
    while i < len(text):
        char = text[i]
        if char == "\\":
            i += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    raise ValueError("unbalanced braces")


def inline(text: str) -> str:
    """Inline LaTeX to HTML."""
    out: list[str] = []
    i = 0
    plain: list[str] = []

    def flush() -> None:
        if plain:
            chunk = html.escape("".join(plain), quote=False)
            chunk = chunk.replace("---", "—").replace("--", "–").replace("``", "“").replace("''", "”").replace("~", "\u00a0")
            out.append(chunk)
            plain.clear()

    while i < len(text):
        char = text[i]
        if char != "\\":
            if char in "{}":
                i += 1  # stray grouping brace
                continue
            plain.append(char)
            i += 1
            continue
        for seq, repl in _ESCAPES.items():
            if text.startswith(seq, i):
                plain.append(repl if repl != "~" else "\u0001")
                i += len(seq)
                break
        else:
            match = re.match(r"\\([A-Za-z]+)\*?", text[i:])
            if text.startswith("\\\\", i):
                flush()
                out.append("<br>")
                i += 2
                if i < len(text) and text[i] == "[":
                    i = text.index("]", i) + 1
                continue
            if not match:
                i += 1
                continue
            name = match.group(1)
            i += match.end()
            if name in {"textbf", "textit", "emph", "texttt"}:
                while i < len(text) and text[i] == " ":
                    i += 1
                arg, i = _braced(text, i)
                tag = {"textbf": "strong", "textit": "em", "emph": "em", "texttt": "code"}[name]
                flush()
                out.append(f"<{tag}>{inline(arg)}</{tag}>")
            elif name in {"ref", "label", "cite"}:
                _, i = _braced(text, i)
            elif name in {"small", "large", "centering", "noindent", "footnotesize"}:
                continue
            elif name == "hfill":
                continue
            else:
                continue  # unknown command: drop it rather than print it
    flush()
    return "".join(out).replace("\u0001", "~")


def _split_cells(row: str) -> list[str]:
    row = row.replace(r"\&", "\u0002")
    return [cell.replace("\u0002", r"\&").strip() for cell in row.split("&")]


def _table(spec_and_body: str) -> str:
    body = re.sub(r"\\(toprule|midrule|bottomrule)", "", spec_and_body)
    rows = [r.strip() for r in re.split(r"\\\\(?![A-Za-z])", body) if r.strip()]
    html_rows = []
    for index, row in enumerate(rows):
        tag = "th" if index == 0 else "td"
        cells = "".join(f"<{tag}>{inline(c)}</{tag}>" for c in _split_cells(row))
        html_rows.append(f"<tr>{cells}</tr>")
    return "<table>" + "".join(html_rows) + "</table>"


_FIGURES = [0]


def _figure(body: str, assets: Path) -> str:
    _FIGURES[0] += 1
    images = []
    for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
        path = assets / match.group(1)
        if path.is_file():
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            images.append(f'<img alt="{html.escape(match.group(1))}" src="data:{_MIME.get(path.suffix.lower(), "image/png")};base64,{data}">')
    caption = ""
    cap = body.find("\\caption")
    if cap != -1:
        text, _ = _braced(body, cap + len("\\caption"))
        caption = f"<figcaption>Figura {_FIGURES[0]}: {inline(text)}</figcaption>"
    return "<figure>" + "<br>".join(images) + caption + "</figure>"


def _strip_groups(text: str) -> str:
    """Removes the {\\small ...} wrappers (keeping their content)."""
    while True:
        match = re.search(r"\{\\small", text)
        if not match:
            return text
        _, end = _braced(text, match.start())
        inner = text[match.end():end - 1]
        inner = re.sub(r"^\s*\\setlength\{[^}]*\}\{[^}]*\}", "", inner)
        text = text[:match.start()] + inner + text[end:]


def convert(tex: str, assets: Path) -> str:
    _FIGURES[0] = 0
    title_match = re.search(r"\\title\{", tex)
    title_html, subtitle = "Informe", ""
    if title_match:
        raw, _ = _braced(tex, title_match.end() - 1)
        main, _, rest = raw.partition("\\\\")
        title_html = inline(main)
        subtitle = inline(re.sub(r"^\[[^\]]*\]\s*\\large", "", rest)).strip() if rest else ""
    date_match = re.search(r"\\date\{([^}]*)\}", tex)
    body = tex.split("\\begin{document}", 1)[1].rsplit("\\end{document}", 1)[0].replace("\\maketitle", "")
    body = _strip_groups(body)
    body = re.sub(r"\\(clearpage|newpage)", "", body)

    parts: list[str] = []
    token = re.compile(r"\\(section|subsection)\{|\\begin\{(figure|tabular|itemize|verbatim)\}")
    pos = 0

    def paragraphs(chunk: str) -> None:
        for para in re.split(r"\n\s*\n", chunk):
            para = para.strip()
            if para:
                parts.append(f"<p>{inline(para)}</p>")

    while True:
        match = token.search(body, pos)
        if not match:
            paragraphs(body[pos:])
            break
        paragraphs(body[pos:match.start()])
        if match.group(1):
            level = "h2" if match.group(1) == "section" else "h3"
            text, pos = _braced(body, match.end() - 1)
            parts.append(f"<{level}>{inline(text)}</{level}>")
            continue
        env = match.group(2)
        end_marker = f"\\end{{{env}}}"
        end = body.index(end_marker, match.end())
        content = body[match.end():end]
        pos = end + len(end_marker)
        if env == "verbatim":
            parts.append(f"<pre>{html.escape(content.strip(chr(10)))}</pre>")
        elif env == "tabular":
            _, after = _braced(content, content.index("{"))
            parts.append(_table(content[after:]))
        elif env == "itemize":
            items = [x.strip() for x in content.split("\\item")[1:] if x.strip()]
            parts.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>")
        else:
            parts.append(_figure(content, assets))
    date = f'<p class="date">{inline(date_match.group(1))}</p>' if date_match else ""
    heading = f"<h1>{title_html}" + (f"<small>{subtitle}</small>" if subtitle else "") + "</h1>"
    plain_title = re.sub(r"<[^>]+>", "", title_html)
    return (f'<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{plain_title}</title><style>{CSS}</style></head><body>{heading}{date}{''.join(parts)}</body></html>")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tex", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(convert(args.tex.read_text(encoding="utf-8"), args.tex.parent), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
