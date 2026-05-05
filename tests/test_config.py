from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pyssmf.config import RunnerConfig, build_default_config, load_config

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def test_runner_config_defaults() -> None:
    config = RunnerConfig()

    assert config.magnetic_mode == "paramagnetic"
    assert config.lattice == "bethe"
    assert config.orbitals == 2
    assert config.beta == 1000.0
    assert config.integration_points == 2000
    assert config.mixing == 0.1
    assert config.output_path is None


def test_runner_config_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RunnerConfig(mode="paramagnetic")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"u_step": 0.0}, "u_step must be strictly positive"),
        ({"u_step": -0.1}, "u_step must be strictly positive"),
        (
            {"u_start": 2.0, "u_stop": 1.0},
            "u_stop must be greater than or equal to u_start",
        ),
        ({"integration_points": 99}, "integration_points must be at least 100"),
        ({"mixing": 0.0}, "mixing must be in the interval"),
        ({"mixing": 1.1}, "mixing must be in the interval"),
        ({"max_iterations": 0}, "max_iterations must be strictly positive"),
    ],
)
def test_runner_config_validates_ranges(
    overrides: dict[str, float], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        RunnerConfig(**overrides)


@pytest.mark.parametrize("mode", ["paramagnetic", "antiferromagnetic"])
def test_build_default_config(mode: str) -> None:
    config = build_default_config(mode)

    assert config.magnetic_mode == mode
    if mode == "paramagnetic":
        assert config.beta == 1000.0
        assert config.u_stop == 4.0
        assert config.output_path == "results_bethe_2orbital_para.dat"
    else:
        assert config.beta == 10000.0
        assert config.u_stop == 1.0
        assert config.output_path == "results_bethe_2orbital_af.dat"


def test_load_config_without_path_returns_builtin_default() -> None:
    config = load_config()

    assert config == build_default_config()


def test_load_config_from_yaml_file() -> None:
    config = load_config(TEST_DATA_DIR / "example_config.yaml")

    assert config.magnetic_mode == "antiferromagnetic"
    assert config.beta == 1234.5
    assert config.u_start == 0.5
    assert config.u_stop == 1.5
    assert config.u_step == 0.25
    assert config.output_path == "custom-results.dat"


def test_load_config_empty_yaml_uses_model_defaults() -> None:
    config = load_config(TEST_DATA_DIR / "example_config_empty.yaml")

    assert config == RunnerConfig()
    assert config.output_path is None
