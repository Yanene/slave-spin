from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from pyssmf.config import build_default_config, load_config
from pyssmf.runner import run_simulation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_config_is_paramagnetic() -> None:
    config = build_default_config()
    assert config.mode == "paramagnetic"
    assert config.orbitals == 2


def test_invalid_config_fails_validation(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("mode: bad-mode\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(config_path)


@pytest.mark.parametrize(
    ("example_name", "reference_name"),
    [
        ("bethe_2orbital_para.yaml", "PARA_bethe_2orbitals_J=0.dat"),
        ("bethe_2orbital_af.yaml", "AF_bethe_2orbitals_J=0.dat"),
    ],
)
def test_runner_matches_reference_data(example_name: str, reference_name: str) -> None:
    config = load_config(PROJECT_ROOT / "examples" / example_name)
    result = run_simulation(config)
    reference = np.loadtxt(PROJECT_ROOT / "data" / reference_name, comments="#")
    assert result.rows.shape == reference.shape
    assert np.allclose(result.rows, reference, atol=5e-3, rtol=5e-3)
