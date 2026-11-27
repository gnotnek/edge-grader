from edge_grader.paper_check import clean, main, span, table_rows


def test_span_writes_ranges_as_the_paper_does():
    assert span([3.8, 3.7]) == "3.7--3.8"
    assert span([146.46, 144.42]) == "144--146"  # whole joules above 100
    assert span([5.3, 5.34]) == "5.3"
    assert span([0.41, 0.4], 2) == "0.40--0.41"


def test_clean_strips_latex_markup():
    assert clean(r" \textbf{76.7} (68.3--84.9) ") == "76.7 (68.3--84.9)"
    assert clean(r"144--147$^\ast$") == "144--147"
    assert clean(r"\textit{E4B tweaked (reference)}") == "E4B tweaked (reference)"


def test_table_rows_reads_the_body_only():
    tex = r"""\label{tab:x}
\begin{tabular}{lc}
\toprule
Head & Col \\
\midrule
a & 1.0 \\
\textbf{b} & 2.0 \\
\bottomrule
\end{tabular}"""
    assert table_rows(tex, "x") == [["a", "1.0"], ["b", "2.0"]]


def test_the_manuscript_matches_the_committed_runs():
    assert main([]) == 0
