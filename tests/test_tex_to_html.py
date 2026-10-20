import base64

from scripts.ops import tex_to_html as th

DOC = r"""\documentclass{article}
\title{Informe de prueba\\[2pt]\large Subtítulo}
\date{1 de enero}
\begin{document}\maketitle
\section{Resumen}\label{sec:a}
Texto con \textbf{negrita}, \textit{cursiva}, \texttt{codigo\_1}, 50\% y ``comillas'' --- fin.

\begin{itemize}
\item Uno \& dos
\item Tres
\end{itemize}
{\small
\begin{tabular}{lc}
\toprule
\textbf{Clip} & \textbf{N} \\
\midrule
clip\_1 & 3 \\
\bottomrule
\end{tabular}}
\begin{verbatim}
python x.py --a "b" \
    --c
\end{verbatim}
\begin{figure}[H]\centering\includegraphics[width=0.5\textwidth]{f.png}\caption{Una figura 100\%.}\end{figure}
\end{document}
"""


def test_converts_the_subset_the_generators_emit(tmp_path):
    (tmp_path / "f.png").write_bytes(b"\x89PNG-fake")
    out = th.convert(DOC, tmp_path)
    assert "<h1>Informe de prueba<small>Subtítulo</small></h1>" in out
    assert "<h2>Resumen</h2>" in out
    assert "<strong>negrita</strong>" in out and "<em>cursiva</em>" in out and "<code>codigo_1</code>" in out
    assert "50%" in out and "“comillas”" in out and "—" in out
    assert "<li>Uno &amp; dos</li>" in out and "<li>Tres</li>" in out
    assert "<table>" in out and "<td>clip_1</td>" in out and "<th><strong>Clip</strong></th>" in out
    assert 'python x.py --a "b" \\' in out.replace("&quot;", '"')
    assert "Figura 1: Una figura 100%." in out
    assert base64.b64encode(b"\x89PNG-fake").decode() in out


def test_no_latex_is_left_behind_and_missing_images_do_not_break(tmp_path):
    out = th.convert(DOC, tmp_path)  # f.png is absent here
    body = out.split("</style>", 1)[1]
    assert "\\" not in body.replace("\\\n", "") or "<pre>" in body
    assert "\\textbf" not in body and "\\section" not in body and "\\begin" not in body


def test_figure_numbers_restart_for_each_document(tmp_path):
    th.convert(DOC, tmp_path)
    assert "Figura 1:" in th.convert(DOC, tmp_path)
