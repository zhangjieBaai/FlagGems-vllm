# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tuned-config loader and dispatch-policy checks.

These exercise the YAML parsing and stage-selection rules, which are plain
metadata handling: no kernel is launched and no tensor is allocated.  A
visible device is still required because importing the package probes one.
"""

import importlib

import pytest

moe = importlib.import_module("flaggems_vllm.ops.fused_moe")

H100 = "NVIDIA_H100_80GB_HBM3"
MINIMAX_KEY = "128,384,int8_w8a8,0,0"


def parse(value):
    return moe._parse_moe_config_entry(value, "T")


def test_shared_list_applies_to_both_stages():
    entry = parse([16, 64, 128, 1, 4, 3])
    assert entry["gemm1"] == entry["gemm2"]
    assert entry["gemm1"] == {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3,
    }
    # Stages must not alias, or popping a flag in one would affect the other.
    assert entry["gemm1"] is not entry["gemm2"]


def test_trailing_flag_mapping_is_expanded():
    entry = parse([16, 64, 128, 1, 4, 3, {"SWAP_AB": True}])
    assert entry["gemm1"]["SWAP_AB"] is True


def test_per_stage_entry_keeps_stages_independent():
    entry = parse(
        {
            "gemm1": [64, 64, 64, 4, 4, 3],
            "gemm2": [64, 128, 64, 2, 4, 3, {"INT32_DOWN": True}],
        }
    )
    assert entry["gemm1"]["BLOCK_SIZE_N"] == 64
    assert entry["gemm2"]["BLOCK_SIZE_N"] == 128
    assert "INT32_DOWN" not in entry["gemm1"]
    assert entry["gemm2"]["INT32_DOWN"] is True


@pytest.mark.parametrize(
    "value, match",
    [
        ([16, 64, 128, 1], "tile values"),
        ([16, 64, 128, 1, 4, 3, "nope"], "flag mapping"),
        ([16, 64, 128, 1, 4, 3, {"FOO": True}], "unknown flags"),
        ({"gemm1": [16, 64, 128, 1, 4, 3]}, "missing"),
        (5, "list or mapping"),
    ],
)
def test_malformed_entries_rejected(value, match):
    with pytest.raises(ValueError, match=match):
        parse(value)


def test_per_stage_block_m_must_agree():
    # Expert alignment runs once with the GEMM1 block size and GEMM2 reuses
    # the result, so differing BLOCK_SIZE_M would mis-index sorted_token_ids.
    with pytest.raises(ValueError, match="share BLOCK_SIZE_M"):
        parse(
            {
                "gemm1": [16, 64, 128, 1, 4, 3],
                "gemm2": [32, 64, 128, 1, 4, 3],
            }
        )


def test_every_shipped_entry_parses():
    configs, _ = moe.get_embedded_moe_configs()
    assert configs, "embedded config table should not be empty"
    for device, table in configs.items():
        for key, by_m in table.items():
            for m, entry in by_m.items():
                assert isinstance(m, int), f"{device}[{key}] has non-int M {m!r}"
                assert set(entry) == {"gemm1", "gemm2"}
                for stage, config in entry.items():
                    where = f"{device}[{key}][{m}][{stage}]"
                    for name in moe._MOE_CONFIG_KEYS_ORDER:
                        assert name in config, f"{where} missing {name}"
                    extra = set(config) - set(moe._MOE_CONFIG_KEYS_ORDER)
                    assert extra <= moe._MOE_CONFIG_EXTRA_FLAGS, where


def test_minimax_tuned_profile_shape():
    """The migrated H100 profile: SWAP_AB on decode GEMM1, INT32 on prefill GEMM2."""
    configs, _ = moe.get_embedded_moe_configs()
    table = configs[H100][MINIMAX_KEY]

    for m in (16, 32, 64):
        gemm1, gemm2 = table[m]["gemm1"], table[m]["gemm2"]
        assert gemm1["SWAP_AB"] is True, m
        assert "SWAP_AB" not in gemm2, m
        assert gemm1["BLOCK_SIZE_K"] == 128, m
        assert gemm1["BLOCK_SIZE_M"] == gemm2["BLOCK_SIZE_M"], m

    for m in (1024, 2048, 4096, 8192):
        gemm1, gemm2 = table[m]["gemm1"], table[m]["gemm2"]
        assert gemm2["INT32_DOWN"] is True, m
        assert "INT32_DOWN" not in gemm1, m
        assert "SWAP_AB" not in gemm1, m
        assert gemm1["BLOCK_SIZE_K"] == 64, m
        assert gemm1["BLOCK_SIZE_M"] == gemm2["BLOCK_SIZE_M"], m

    # Untuned rows stay shared between the stages.
    for m in (1, 8, 128, 512):
        assert table[m]["gemm1"] == table[m]["gemm2"], m


def test_stage_selection_reaches_the_kernel_config(monkeypatch):
    monkeypatch.setattr(moe, "_get_device_name", lambda device_index=None: H100)
    common = dict(
        w1_shape=(128, 768, 6144),
        w2_shape=(128, 6144, 384),
        top_k=4,
        dtype="int8_w8a8",
        E=128,
    )
    gemm1 = moe.try_get_optimal_moe_config(M=8192, gemm_stage="gemm1", **common)
    gemm2 = moe.try_get_optimal_moe_config(M=8192, gemm_stage="gemm2", **common)
    assert gemm2["INT32_DOWN"] is True
    assert "INT32_DOWN" not in gemm1

    decode1 = moe.try_get_optimal_moe_config(M=64, gemm_stage="gemm1", **common)
    decode2 = moe.try_get_optimal_moe_config(M=64, gemm_stage="gemm2", **common)
    assert decode1["SWAP_AB"] is True
    assert "SWAP_AB" not in decode2


def test_returned_config_is_a_fresh_copy(monkeypatch):
    """invoke_fused_moe_triton_kernel pops keys, so it must not share state."""
    monkeypatch.setattr(moe, "_get_device_name", lambda device_index=None: H100)
    kwargs = dict(
        w1_shape=(128, 768, 6144),
        w2_shape=(128, 6144, 384),
        top_k=4,
        dtype="int8_w8a8",
        E=128,
        M=64,
        gemm_stage="gemm1",
    )
    first = moe.try_get_optimal_moe_config(**kwargs)
    first.pop("SWAP_AB")
    assert moe.try_get_optimal_moe_config(**kwargs)["SWAP_AB"] is True


def test_int8_w8a8_avoids_plain_half_config():
    # The plain-half GEMM fast paths do not apply INT8 scales.
    assert moe._get_config_dtype_str(use_int8_w8a8=True) == "int8_w8a8"
    assert "int8_w8a8" not in moe._PLAIN_HALF_CONFIG_DTYPES


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("value", [-1e39, 1e39])
def test_fp32_overflow_scalars_rejected(index, value):
    # Finite as a Python FP64 value, infinite once narrowed to FP32.
    params = [1.702, 1.0, 7.0]
    params[index] = value
    with pytest.raises(ValueError, match="finite FP32"):
        moe._validate_swigluoai_uninterleave_params(*params)


def test_fp32_representable_scalars_accepted():
    import torch

    limit = float(torch.finfo(torch.float32).max)
    assert moe._validate_swigluoai_uninterleave_params(1.702, 1.0, limit) == (
        1.702,
        1.0,
        limit,
    )


@pytest.mark.parametrize("value", [float("inf"), float("nan"), float("-inf")])
def test_non_finite_scalars_still_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        moe._validate_swigluoai_uninterleave_params(value, 1.0, 7.0)
