import numpy as np
import pytest

from edge_grader.soup import soup_config, stack

ALPHA, R = 32, 16


def adapter(rng, inp=12, out=8, layers=2) -> dict[str, np.ndarray]:
    a = {}
    for i in range(layers):
        k = f"base_model.model.model.language_model.layers.{i}.self_attn.q_proj"
        a[f"{k}.lora_A.weight"] = rng.standard_normal((R, inp)).astype(np.float32)
        a[f"{k}.lora_B.weight"] = rng.standard_normal((out, R)).astype(np.float32)
    return a


def update(a: dict[str, np.ndarray], layer: str, r: int) -> np.ndarray:
    """dW as llama.cpp and PEFT apply it: alpha / rank, the rank read from B's shape."""
    b = a[f"{layer}.lora_B.weight"]
    assert b.shape[1] == r
    return ALPHA / r * b @ a[f"{layer}.lora_A.weight"]


def test_stacked_update_is_the_mean_of_the_updates():
    rng = np.random.default_rng(0)
    parts = [adapter(rng) for _ in range(5)]
    soup = stack(parts)
    for i in range(2):
        layer = f"base_model.model.model.language_model.layers.{i}.self_attn.q_proj"
        mean = np.mean([update(p, layer, R) for p in parts], axis=0)
        np.testing.assert_allclose(update(soup, layer, 5 * R), mean, rtol=1e-5, atol=1e-5)
        # the naive average of A and B is not the same thing
        naive = {k: np.mean([p[k] for p in parts], axis=0) for k in parts[0]}
        assert not np.allclose(update(naive, layer, R), mean, atol=1e-3)


def test_soup_of_one_is_that_adapter():
    a = adapter(np.random.default_rng(1))
    assert all(np.array_equal(stack([a])[k], a[k]) for k in a)


def test_mismatched_adapters_are_refused():
    rng = np.random.default_rng(2)
    a, b = adapter(rng), adapter(rng, layers=1)
    with pytest.raises(ValueError, match="same tensors"):
        stack([a, b])
    with pytest.raises(ValueError, match="LoRA A or B"):
        stack([{"x.weight": np.zeros(1)}])


def test_config_multiplies_the_rank_and_keeps_alpha():
    c = {"r": R, "lora_alpha": ALPHA, "use_rslora": False, "rank_pattern": {}, "alpha_pattern": {}}
    assert soup_config([c] * 5) == {**c, "r": 80}
    with pytest.raises(ValueError, match="different configs"):
        soup_config([c, {**c, "lora_alpha": 16}])
    with pytest.raises(ValueError, match="plain LoRA"):
        soup_config([{**c, "use_rslora": True}] * 2)
