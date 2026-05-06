from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# supported types and modes
LATTICE_TYPES = Literal["bethe"]
MAGNETIC_MODES = Literal["paramagnetic", "antiferromagnetic"]


class LatticeConfig(BaseModel):
    type: LATTICE_TYPES = Field(
        "bethe", description="The type of lattice to be used in the simulation."
    )
    orbitals: int = Field(
        2,
        gt=0,
        le=10,
        description="The number of orbitals in the tight-binding model. Maximum allowed is 10.",
    )
    hopping: float = Field(
        1.0, gt=0, description="The hopping parameter (t) for the tight-binding model."
    )


class NumericalConfig(BaseModel):
    beta: float = Field(
        1000.0,
        gt=0,
        description="The inverse temperature (beta) for the Fermi-Dirac distribution.",
    )
    tolerance: float = Field(
        1e-4,
        gt=0,
        description="The convergence tolerance for the self-consistent loop.",
    )
    integration_points: int = Field(
        2000,
        ge=100,
        description="The number of points to use in numerical integration.",
    )
    mixing: float = Field(
        0.1,
        gt=0,
        le=1,
        description="The mixing parameter for the self-consistent loop.",
    )
    max_iterations: int = Field(
        5000,
        gt=0,
        description="The maximum number of iterations for the self-consistent loop.",
    )


class RangeSweep(BaseModel):
    start: float = Field(..., description="The starting value of the range sweep.")
    stop: float = Field(..., description="The stopping value of the range sweep.")
    step: float = Field(0.1, gt=0, description="The step size for the range sweep.")

    @model_validator(mode="after")
    def validate_range(self):
        if self.stop < self.start:
            raise ValueError("stop must be >= start")
        return self


class SweepConfig(BaseModel):
    u_interaction: RangeSweep = Field(
        default_factory=lambda: RangeSweep(start=0.0, stop=4.0, step=0.1),
        description="The range sweep configuration for the interaction strength (U).",
    )
    hund_coupling: RangeSweep | float = Field(
        default=0.0,
        description="The range sweep configuration for the Hund's coupling constant, or a fixed value if not sweeping.",
    )


class SolverGuesses(BaseModel):
    # ! soon to be deprecated
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


class RunnerConfig(BaseModel):
    """Validated configuration for the built-in Bethe lattice runners."""

    model_config = ConfigDict(extra="forbid")

    magnetic_mode: MAGNETIC_MODES = Field(
        default="paramagnetic", description="The magnetic mode of the simulation."
    )
    lattice: LatticeConfig = Field(default_factory=LatticeConfig)
    numerical: NumericalConfig = Field(default_factory=NumericalConfig)
    # interaction sweep parameters
    sweep: SweepConfig = Field(default_factory=SweepConfig)

    # SSMF solver guesses (SOON TO BE DEPRECATED)
    solver_guesses: SolverGuesses | None = None

    target_occupation: float = Field(
        default=2.0, description="The target occupation number for the simulation."
    )
    output_path: str | None = None


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
            numerical=NumericalConfig(beta=1000.0),
            output_path="results_bethe_2orbital_para.dat",
        )

    return RunnerConfig(
        magnetic_mode="antiferromagnetic",
        numerical=NumericalConfig(beta=10000.0),
        sweep=SweepConfig(
            u_interaction=RangeSweep(start=0.0, stop=1.0, step=0.1),
            hund_coupling=0.0,
        ),
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
