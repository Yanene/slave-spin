from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

LATTICE_TYPES = Literal["bethe"]  # supported lattice type
MAGNETIC_MODES = Literal["paramagnetic", "antiferromagnetic"]  # supported magnetic mode


class RunnerConfig(BaseModel):
    """Validated configuration for the built-in Bethe lattice runners."""

    model_config = ConfigDict(extra="forbid")

    # Lattice information
    magnetic_mode: MAGNETIC_MODES = Field(
        default="paramagnetic", description="The magnetic mode of the simulation."
    )
    lattice: LATTICE_TYPES = Field(
        default="bethe", description="The lattice type to be used in the simulation."
    )
    orbitals: int = Field(
        default=2, description="The number of orbitals in the tight-binding model."
    )
    hopping: float = Field(
        default=1.0,
        description="The hopping parameter (t) for the tight-binding model.",
    )

    # Numerical parameters
    beta: float = Field(
        default=1000.0,
        description="The inverse temperature (beta) for the Fermi-Dirac distribution.",
    )
    tolerance: float = Field(
        default=1e-4,
        description="The convergence tolerance for the self-consistent loop.",
    )
    integration_points: int = Field(
        default=2000,
        description="The number of points to use in numerical integration.",
    )
    mixing: float = Field(
        default=0.1, description="The mixing parameter for the self-consistent loop."
    )
    max_iterations: int = Field(
        default=5000,
        description="The maximum number of iterations for the self-consistent loop.",
    )

    # Simulation parameters
    target_occupation: float = Field(
        default=2.0, description="The target occupation number for the simulation."
    )
    u_start: float = Field(
        default=0.0,
        description="The starting value of the interaction strength (U) for the simulation.",
    )
    u_stop: float = Field(
        default=4.0,
        description="The stopping value of the interaction strength (U) for the simulation.",
    )
    u_step: float = Field(
        default=0.1,
        description="The step size for the interaction strength (U) in the simulation.",
    )
    hund_coupling: float = Field(
        default=0.0, description="The Hund's coupling constant for the simulation."
    )

    # SSMF solver parameters
    mu_guess: float = Field(
        default=0.0,
        description="Initial guess for the chemical potential (mu) in the SSMF solver.",
    )
    density_guess: float = Field(
        default=1.0, description="Initial guess for the density in the SSMF solver."
    )
    z_guess: float = Field(
        default=1.0,
        description="Initial guess for the quasiparticle weight (Z) in the SSMF solver.",
    )
    lambda_orbital_1_guess: float = Field(
        default=0.0,
        description="Initial guess for the lambda parameter of orbital 1 in the SSMF solver.",
    )
    lambda_orbital_2_guess: float = Field(
        default=0.0,
        description="Initial guess for the lambda parameter of orbital 2 in the SSMF solver.",
    )
    lambda_1_up_guess: float = Field(
        default=-1.6,
        description="Initial guess for the lambda parameter of orbital 1 with spin up in the SSMF solver.",
    )
    lambda_1_down_guess: float = Field(
        default=1.6,
        description="Initial guess for the lambda parameter of orbital 1 with spin down in the SSMF solver.",
    )
    lambda_2_up_guess: float = Field(
        default=-1.6,
        description="Initial guess for the lambda parameter of orbital 2 with spin up in the SSMF solver.",
    )
    lambda_2_down_guess: float = Field(
        default=1.6,
        description="Initial guess for the lambda parameter of orbital 2 with spin down in the SSMF solver.",
    )
    lambda_1_up_shift_guess: float = Field(
        default=-3.16,
        description="Initial guess for the shifted lambda parameter of orbital 1 with spin up in the SSMF solver.",
    )
    lambda_1_down_shift_guess: float = Field(
        default=3.16,
        description="Initial guess for the shifted lambda parameter of orbital 1 with spin down in the SSMF solver.",
    )
    lambda_2_up_shift_guess: float = Field(
        default=-3.16,
        description="Initial guess for the shifted lambda parameter of orbital 2 with spin up in the SSMF solver.",
    )
    lambda_2_down_shift_guess: float = Field(
        default=3.16,
        description="Initial guess for the shifted lambda parameter of orbital 2 with spin down in the SSMF solver.",
    )

    output_path: str | None = None

    @model_validator(mode="after")
    def validate_ranges(self):
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


def build_default_config(
    magnetic_mode: MAGNETIC_MODES = "paramagnetic",
) -> RunnerConfig:
    """
    Return the default validated configuration for a built-in use case.

    Args:
        magnetic_mode: MAGNETIC_MODE: The magnetic mode for which to build the default configuration.
        Must be either "paramagnetic" or "antiferromagnetic". Defaults to "paramagnetic".

    Returns:
        RunnerConfig: The default validated configuration object for the specified magnetic mode.
    """
    if magnetic_mode == "paramagnetic":
        return RunnerConfig(
            magnetic_mode="paramagnetic",
            beta=1000.0,
            output_path="results_bethe_2orbital_para.dat",
        )

    return RunnerConfig(
        magnetic_mode="antiferromagnetic",
        beta=10000.0,
        u_stop=1.0,
        output_path="results_bethe_2orbital_af.dat",
    )


def load_config(path: str | Path | None = None) -> RunnerConfig:
    """
    Load a YAML config file, or return the built-in default when absent.

    Args:
        path: str | Path | None: The path to the YAML configuration file. If None, the default configuration is returned.

    Returns:
        RunnerConfig: The validated configuration object for the SSMF solver.
    """
    if path is None:
        return build_default_config()

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle) or {}

    return RunnerConfig.model_validate(payload)
