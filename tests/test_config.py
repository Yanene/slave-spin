from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pyssmf.config import RunnerConfig, build_default_config, load_config

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def test_runner_config_defaults() -> None:
    config = RunnerConfig()

    assert config.magnetic_mode == "paramagnetic"
    assert config.lattice.type == "bethe"
    assert config.lattice.orbitals == 2
    assert config.numerical.beta == 1000.0
    assert config.numerical.integration_points == 2000
    assert config.numerical.mixing == 0.1
    assert config.sweep.u_interaction.stop == 4.0
    assert config.output_path is None


def test_runner_config_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RunnerConfig(mode="paramagnetic")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"sweep": {"u_interaction": {"start": 0.0, "stop": 1.0, "step": 0.0}}},
            "Input should be greater than 0",
        ),
        (
            {"sweep": {"u_interaction": {"start": 2.0, "stop": 1.0, "step": 0.1}}},
            "stop must be >= start",
        ),
        (
            {"numerical": {"integration_points": 99}},
            "Input should be greater than or equal to 100",
        ),
        (
            {"numerical": {"mixing": 0.0}},
            "Input should be greater than 0",
        ),
        (
            {"numerical": {"mixing": 1.1}},
            "Input should be less than or equal to 1",
        ),
        (
            {"numerical": {"max_iterations": 0}},
            "Input should be greater than 0",
        ),
    ],
)
def test_runner_config_validates_ranges(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        RunnerConfig(**overrides)


@pytest.mark.parametrize("mode", ["paramagnetic", "antiferromagnetic"])
def test_build_default_config(mode: str) -> None:
    config = build_default_config(mode)

    assert config.magnetic_mode == mode

    if mode == "paramagnetic":
        assert config.numerical.beta == 1000.0
        assert config.sweep.u_interaction.stop == 4.0
        assert config.output_path == "results_bethe_2orbital_para.dat"
    else:
        assert config.numerical.beta == 10000.0
        assert config.sweep.u_interaction.stop == 1.0
        assert config.output_path == "results_bethe_2orbital_af.dat"


def test_load_config_without_path_returns_builtin_default() -> None:
    config = load_config()

    assert config == build_default_config()


def test_load_config_from_yaml_file() -> None:
    config = load_config(TEST_DATA_DIR / "example_config.yaml")

    assert config.magnetic_mode == "antiferromagnetic"
    assert config.numerical.beta == 8000.0
    assert config.sweep.u_interaction.start == 0.5
    assert config.sweep.u_interaction.stop == 3.0
    assert config.sweep.u_interaction.step == 0.25
    assert config.output_path == "results/custom_bethe_af_scan.dat"


def test_load_config_empty_yaml_uses_model_defaults() -> None:
    config = load_config(TEST_DATA_DIR / "example_config_empty.yaml")

    assert config == RunnerConfig()
    assert config.output_path is None
