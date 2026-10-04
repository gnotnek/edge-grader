import pytest

from edge_grader.board_check import mcnemar, parse_server_log, threads, window_energy

LOG = """\
2026-10-04T07:54:38+0800 pi llama-server[0]: 0.25.103.060 I slot print_timing: id  3 | task 0 | prompt eval time =    2200.77 ms /   116 tokens (   18.97 ms per token,    52.71 tokens per second)
2026-10-04T07:54:38+0800 pi llama-server[0]: 0.25.103.067 I slot print_timing: id  3 | task 0 |        eval time =     163.74 ms /     2 tokens (  163.74 ms per token,     6.11 tokens per second)
2026-10-04T07:54:57+0800 pi llama-server[0]: 0.43.821.332 I slot print_timing: id  2 | task 35 | prompt processing, n_tokens =    229, progress = 0.98, t =   3.78 s / 60.65 tokens per second
2026-10-04T07:54:57+0800 pi llama-server[0]: 0.44.137.835 I slot print_timing: id  2 | task 35 | prompt eval time =    3941.66 ms /   233 tokens (   16.92 ms per token,    59.11 tokens per second)
2026-10-04T07:54:57+0800 pi llama-server[0]: 0.44.137.842 I slot print_timing: id  2 | task 35 |        eval time =     150.54 ms /     2 tokens (  150.54 ms per token,     6.64 tokens per second)
2026-10-04T07:54:58+0800 pi llama-server[0]: 0.45.000.000 I slot print_timing: id  2 | task 40 | prompt eval time =    1000.00 ms /   100 tokens (   10.00 ms per token,   100.00 tokens per second)
"""


def test_parse_server_log_pairs_prompt_and_decode_and_skips_progress_and_partial():
    reqs = parse_server_log(LOG)
    assert [r["task"] for r in reqs] == [0, 35]  # task 40 has no decode line yet
    assert reqs[0]["prompt_tokens"] == 116 and reqs[0]["prompt_ms"] == 2200.77
    assert reqs[1]["gen_tokens"] == 2 and reqs[1]["gen_ms"] == 150.54
    assert reqs[1]["end"] - reqs[0]["end"] == 19


def test_window_energy_integrates_only_inside_the_window():
    samples = [(0, 100.0), (1, 2.0), (2, 4.0), (3, 6.0), (4, 100.0)]
    assert window_energy(samples, 1, 3) == pytest.approx(3 + 5)


def test_mcnemar_counts_discordant_pairs():
    a = [True, True, True, False, True]
    b = [True, False, False, False, True]
    r = mcnemar(a, b)
    assert (r["only_first_right"], r["only_second_right"]) == (2, 0)
    assert r["p"] == 0.5
    assert mcnemar([True], [True])["p"] == 1.0


def test_threads_reads_only_the_t_flag():
    assert threads("llama-server -m x.gguf -t 3 -c 8192") == 3
    assert threads("llama-server -m x-t 9.gguf -c 8192 -ngl 0") is None
    assert threads("") is None
