"""Configuration models and helpers for pySSMF runner inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


Mode = Literal["paramagnetic", "af"]


class RunnerConfig(BaseModel):
    """Validated configuration for the built-in Bethe lattice runners."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    lattice: Literal["bethe"] = "bethe"
    mode: Mode = "paramagnetic"
    orbitals: Literal[2] = 2
    hopping: float = 0.5
    beta: float = 1000.0
    tolerance: float = 1e-4
    integration_points: int = 2000
    mixing: float = 0.1
    max_iterations: int = 5000
    target_occupation: float = 2.0
    u_start: float = 0.0
    u_stop: float = 3.5
    u_step: float = 0.1
    hund_coupling_factor: float = 0.0
    output_path: str | None = None
    mu_guess: float = 0.0
    density_guess: float = 1.0
    z_guess: float = 1.0
    lambda_orbital_1_guess: float = 0.0
    lambda_orbital_2_guess: float = 0.0
    lambda_1_up_guess: float = -1.6
    lambda_1_down_guess: float = 1.6
    lambda_2_up_guess: float = -1.6
    lambda_2_down_guess: float = 1.6
    lambda_1_up_shift_guess: float = -3.16
    lambda_1_down_shift_guess: float = 3.16
    lambda_2_up_shift_guess: float = -3.16
    lambda_2_down_shift_guess: float = 3.16

    @model_validator(mode="after")
    def validate_ranges(self) -> "RunnerConfig":
        """Ensure the numerical setup is coherent before the solver starts."""
        if self.u_step <= 0:
            raise ValueError("u_step must be strictly positive.")
        if self.u_stop < self.u_start:
            raise ValueError("u_stop must be greater than or equal to u_start.")
        if self.integration_points < 100:
            raise ValueError("integration_points must be at least 100.")
        if not 0 < self.mixing <= 1:
            raise ValueError("mixing must be in the interval (0, 1].")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be strictly positive.")
        return self


def build_default_config(mode: Mode = "paramagnetic") -> RunnerConfig:
    """Return the default validated configuration for a built-in use case."""
    if mode == "paramagnetic":
        return RunnerConfig(
            mode="paramagnetic",
            beta=1000.0,
            output_path="results_bethe_2orbital_para.dat",
        )

    return RunnerConfig(
        mode="af",
        beta=10000.0,
        u_stop=1.0,
        output_path="results_bethe_2orbital_af.dat",
    )


def load_config(path: str | Path | None = None) -> RunnerConfig:
    """Load a YAML config file, or return the built-in default when absent."""
    if path is None:
        return build_default_config()

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle) or {}

    return RunnerConfig.model_validate(payload)
