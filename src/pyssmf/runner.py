"""Command-line runner for the built-in pySSMF Bethe lattice examples."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.linalg
from scipy.integrate import simpson

from .config import RunnerConfig, load_config

GAUGE_EPSILON = 1e-8
GRADIENT_STEP = 1e-5


@dataclass(slots=True)
class RunResult:
    """Container for a completed simulation run."""

    config: RunnerConfig
    headers: list[str]
    rows: np.ndarray
    elapsed_seconds: float

    def write(self, destination: str | Path | None = None) -> Path:
        """Persist the numerical table to disk and return the written path."""
        target = Path(
            destination or self.config.output_path or default_output_name(self.config)
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        header = "# " + " ".join(self.headers)
        np.savetxt(target, self.rows, header=header, comments="")
        return target


def default_output_name(config: RunnerConfig) -> str:
    """Build a readable default filename from the validated mode."""
    suffix = "para" if config.mode == "paramagnetic" else "af"
    return f"results_bethe_2orbital_{suffix}.dat"


def make_u_values(config: RunnerConfig) -> np.ndarray:
    """Construct the inclusive interaction sweep used by the legacy scripts."""
    stop = config.u_stop + (config.u_step / 2)
    return np.arange(config.u_start, stop, config.u_step)


def fermi_distribution(energy: np.ndarray | float, beta: float) -> np.ndarray | float:
    """Evaluate the Fermi function with the legacy high-beta cutoff."""
    values = np.asarray(np.real(energy), dtype=float)
    exponent = beta * values
    occupation = np.zeros_like(exponent, dtype=float)
    mask = exponent < 10.0
    occupation[mask] = 1.0 / (np.exp(exponent[mask]) + 1.0)
    if np.isscalar(energy):
        return float(occupation)
    return occupation


def bethe_density_of_states(epsilon: np.ndarray, hopping: float) -> np.ndarray:
    """Return the non-interacting Bethe lattice density of states."""
    return (1.0 / (2.0 * np.pi * hopping**2)) * np.sqrt(
        np.clip(4.0 * hopping**2 - epsilon**2, 0.0, None)
    )


def estimate_gauge(density: float) -> float:
    """Estimate the gauge factor used in the slave-spin operators."""
    clipped_density = float(np.clip(density, GAUGE_EPSILON, 1.0 - GAUGE_EPSILON))
    return (
        1.0 / np.sqrt(clipped_density * (1.0 - clipped_density) + GAUGE_EPSILON)
    ) - 1.0


def btest(state: int, index: int) -> int:
    """Return the binary occupation of `state` at the selected bit position."""
    return (state >> index) & 1


def spin_z_operator(orbitals: int, index: int) -> np.ndarray:
    """Construct the local spin-z operator for a given spin-orbital index."""
    dimension = 4**orbitals
    matrix = np.zeros((dimension, dimension))
    for state in range(dimension):
        matrix[state, state] = 0.5 if btest(state, index) == 1 else -0.5
    return matrix


def spin_flip_operator(orbitals: int, index: int, gauge: float) -> np.ndarray:
    """Construct the generic slave-spin flip operator with its gauge factor."""
    dimension = 4**orbitals
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    flipper = 2**index
    for state in range(dimension):
        flipped_state = state ^ flipper
        matrix[flipped_state, state] = 1.0 if btest(state, index) == 1 else gauge
    return matrix


def spin_flip_dagger(orbitals: int, index: int, gauge: float) -> np.ndarray:
    """Return the Hermitian conjugate of the spin-flip operator."""
    return spin_flip_operator(orbitals, index, gauge).conj().T


def average_ground_state_value(
    operator: np.ndarray, eigenvectors: np.ndarray, eigenvalues: np.ndarray
) -> float:
    """Evaluate an operator in the ground state of the slave-spin Hamiltonian."""
    ground_state = eigenvectors[:, np.argmin(eigenvalues)]
    return float(np.real(np.vdot(ground_state, operator @ ground_state)))


def quasiparticle_weight(average: float) -> float:
    """Compute the quasiparticle weight from the slave-spin expectation value."""
    return average**2


def eta(density: float) -> float:
    """Return the AF lambda-shift prefactor used in the legacy solver."""
    return (2.0 * density - 1.0) / (4.0 * density * (1.0 - density) + GAUGE_EPSILON)


class ParamagneticSolver:
    """Numerical solver for the two-orbital paramagnetic Bethe use case."""

    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.epsilon = np.linspace(-1.0, 1.0, config.integration_points)
        self.dos = bethe_density_of_states(self.epsilon, config.hopping)
        self.orbitals = config.orbitals

    def fermionic_occupation(
        self, z_value: float, mu: float, lambda_value: float
    ) -> float:
        """Compute the occupation for one paramagnetic orbital."""
        energies = z_value * self.epsilon - mu - lambda_value
        integrand = self.dos * fermi_distribution(energies, self.config.beta)
        return float(simpson(integrand, x=self.epsilon))

    def calculate_h(self, mu: float, lambda_value: float, z_value: float) -> float:
        """Compute the renormalized kinetic field for one orbital."""
        energies = z_value * self.epsilon - mu - lambda_value
        integrand = (
            self.dos * self.epsilon * fermi_distribution(energies, self.config.beta)
        )
        return float(np.sqrt(z_value) * simpson(integrand, x=self.epsilon))

    def slave_hamiltonian(
        self,
        h1_up: float,
        h1_down: float,
        h2_up: float,
        h2_down: float,
        lambda1: float,
        lambda2: float,
        u_value: float,
        gauge1_up: float,
        gauge1_down: float,
        gauge2_up: float,
        gauge2_down: float,
    ) -> np.ndarray:
        """Construct the paramagnetic slave-spin Hamiltonian."""
        matrix = np.zeros((4**self.orbitals, 4**self.orbitals), dtype=np.complex128)
        matrix += lambda1 * (
            spin_z_operator(self.orbitals, 0) + spin_z_operator(self.orbitals, 1)
        )
        matrix += lambda2 * (
            spin_z_operator(self.orbitals, 2) + spin_z_operator(self.orbitals, 3)
        )
        matrix += h1_down * spin_flip_dagger(self.orbitals, 0, gauge1_down)
        matrix += h1_up * spin_flip_dagger(self.orbitals, 1, gauge1_up)
        matrix += np.conj(h1_down) * spin_flip_operator(self.orbitals, 0, gauge1_down)
        matrix += np.conj(h1_up) * spin_flip_operator(self.orbitals, 1, gauge1_up)
        matrix += h2_down * spin_flip_dagger(self.orbitals, 2, gauge2_down)
        matrix += h2_up * spin_flip_dagger(self.orbitals, 3, gauge2_up)
        matrix += np.conj(h2_down) * spin_flip_operator(self.orbitals, 2, gauge2_down)
        matrix += np.conj(h2_up) * spin_flip_operator(self.orbitals, 3, gauge2_up)
        spin_1_up = spin_z_operator(self.orbitals, 1)
        spin_1_down = spin_z_operator(self.orbitals, 0)
        spin_2_up = spin_z_operator(self.orbitals, 3)
        spin_2_down = spin_z_operator(self.orbitals, 2)
        matrix += u_value * (spin_1_up @ spin_1_down + spin_2_up @ spin_2_down)
        return matrix

    def gradient_descent_2d(
        self, function: Callable[[float, float], float], start: tuple[float, float]
    ) -> np.ndarray:
        """Numerically minimize a two-variable scalar objective."""
        vector = np.array(start, dtype=float)
        for _ in range(self.config.max_iterations):
            x_value, y_value = vector
            gradient = np.array(
                [
                    (
                        function(x_value + GRADIENT_STEP, y_value)
                        - function(x_value - GRADIENT_STEP, y_value)
                    )
                    / (2.0 * GRADIENT_STEP),
                    (
                        function(x_value, y_value + GRADIENT_STEP)
                        - function(x_value, y_value - GRADIENT_STEP)
                    )
                    / (2.0 * GRADIENT_STEP),
                ]
            )
            if np.all(np.abs(gradient) <= 1e-8):
                break
            vector -= self.config.mixing * gradient
        return vector

    def find_lambdas(
        self,
        old_lambda_1: float,
        old_lambda_2: float,
        u_value: float,
        h1_up: float,
        h1_down: float,
        h2_up: float,
        h2_down: float,
        gauge1_up: float,
        gauge1_down: float,
        gauge2_up: float,
        gauge2_down: float,
        occupation_1_up: float,
        occupation_1_down: float,
        occupation_2_up: float,
        occupation_2_down: float,
    ) -> tuple[float, float]:
        """Update the orbital Lagrange multipliers to match the fermionic densities."""

        def spread(lambda1: float, lambda2: float) -> float:
            evals, evecs = scipy.linalg.eigh(
                self.slave_hamiltonian(
                    h1_up,
                    h1_down,
                    h2_up,
                    h2_down,
                    lambda1,
                    lambda2,
                    u_value,
                    gauge1_up,
                    gauge1_down,
                    gauge2_up,
                    gauge2_down,
                )
            )
            spin_occ_1_up = (
                average_ground_state_value(
                    spin_z_operator(self.orbitals, 1), evecs, evals
                )
                + 0.5
            )
            spin_occ_1_down = (
                average_ground_state_value(
                    spin_z_operator(self.orbitals, 0), evecs, evals
                )
                + 0.5
            )
            spin_occ_2_up = (
                average_ground_state_value(
                    spin_z_operator(self.orbitals, 3), evecs, evals
                )
                + 0.5
            )
            spin_occ_2_down = (
                average_ground_state_value(
                    spin_z_operator(self.orbitals, 2), evecs, evals
                )
                + 0.5
            )
            return (
                (spin_occ_1_up - occupation_1_up) ** 2
                + (spin_occ_1_down - occupation_1_down) ** 2
                + (spin_occ_2_up - occupation_2_up) ** 2
                + (spin_occ_2_down - occupation_2_down) ** 2
            )

        result = self.gradient_descent_2d(spread, (old_lambda_1, old_lambda_2))
        return float(result[0]), float(result[1])

    def density_derivative(
        self,
        mu_new: float,
        mu_old: float,
        lambda1: float,
        lambda2: float,
        z1: float,
        z2: float,
    ) -> float:
        """Estimate `dn/dmu` for the next chemical potential update."""
        n_new = 2.0 * self.fermionic_occupation(
            z1, mu_new, lambda1
        ) + 2.0 * self.fermionic_occupation(z2, mu_new, lambda2)
        n_old = 2.0 * self.fermionic_occupation(
            z1, mu_old, lambda1
        ) + 2.0 * self.fermionic_occupation(z2, mu_old, lambda2)
        if mu_new == mu_old:
            return 0.0
        density = (n_new - n_old) / (mu_new - mu_old)
        return max(0.1, density) if density > 0 else min(-0.1, density)

    def find_mu(
        self,
        mu_guess: float,
        lambda1: float,
        lambda2: float,
        z1: float,
        z2: float,
        derivative: float,
    ) -> float:
        """Update the chemical potential using the legacy secant-style correction."""
        occupation = 2.0 * self.fermionic_occupation(
            z1, mu_guess, lambda1
        ) + 2.0 * self.fermionic_occupation(z2, mu_guess, lambda2)
        if derivative == 0:
            return mu_guess
        return float(
            mu_guess - (occupation - self.config.target_occupation) / derivative
        )

    def solve(self) -> tuple[list[str], np.ndarray]:
        """Run the full interaction sweep for the paramagnetic case."""
        rows: list[list[float]] = []
        mu_guess = self.config.mu_guess
        dens_guess = self.config.density_guess
        z1_up_guess = z1_down_guess = z2_up_guess = z2_down_guess = self.config.z_guess
        lambda1_guess = self.config.lambda_orbital_1_guess
        lambda2_guess = self.config.lambda_orbital_2_guess
        h1_up_guess = h1_down_guess = 0.2
        h2_up_guess = h2_down_guess = 0.2
        spin_1_up_old = spin_1_down_old = 0.5
        spin_2_up_old = spin_2_down_old = 0.5

        for u_value in make_u_values(self.config):
            for _ in range(self.config.max_iterations):
                f1_up = self.fermionic_occupation(z1_up_guess, mu_guess, lambda1_guess)
                f1_down = self.fermionic_occupation(
                    z1_down_guess, mu_guess, lambda1_guess
                )
                f2_up = self.fermionic_occupation(z2_up_guess, mu_guess, lambda2_guess)
                f2_down = self.fermionic_occupation(
                    z2_down_guess, mu_guess, lambda2_guess
                )
                gauge1_up = estimate_gauge(f1_up)
                gauge1_down = estimate_gauge(f1_down)
                gauge2_up = estimate_gauge(f2_up)
                gauge2_down = estimate_gauge(f2_down)
                h1_up_new = self.calculate_h(mu_guess, lambda1_guess, z1_up_guess)
                h1_down_new = self.calculate_h(mu_guess, lambda1_guess, z1_down_guess)
                h2_up_new = self.calculate_h(mu_guess, lambda2_guess, z2_up_guess)
                h2_down_new = self.calculate_h(mu_guess, lambda2_guess, z2_down_guess)
                lambda1_new, lambda2_new = self.find_lambdas(
                    lambda1_guess,
                    lambda2_guess,
                    u_value,
                    h1_up_new,
                    h1_down_new,
                    h2_up_new,
                    h2_down_new,
                    gauge1_up,
                    gauge1_down,
                    gauge2_up,
                    gauge2_down,
                    f1_up,
                    f1_down,
                    f2_up,
                    f2_down,
                )
                evals, evecs = scipy.linalg.eigh(
                    self.slave_hamiltonian(
                        h1_up_new,
                        h1_down_new,
                        h2_up_new,
                        h2_down_new,
                        lambda1_new,
                        lambda2_new,
                        u_value,
                        gauge1_up,
                        gauge1_down,
                        gauge2_up,
                        gauge2_down,
                    )
                )
                z1_up_new = quasiparticle_weight(
                    average_ground_state_value(
                        spin_flip_operator(self.orbitals, 1, gauge1_up), evecs, evals
                    )
                )
                z1_down_new = quasiparticle_weight(
                    average_ground_state_value(
                        spin_flip_operator(self.orbitals, 0, gauge1_down), evecs, evals
                    )
                )
                z2_up_new = quasiparticle_weight(
                    average_ground_state_value(
                        spin_flip_operator(self.orbitals, 3, gauge2_up), evecs, evals
                    )
                )
                z2_down_new = quasiparticle_weight(
                    average_ground_state_value(
                        spin_flip_operator(self.orbitals, 2, gauge2_down), evecs, evals
                    )
                )
                spin_occ_1_up = (
                    average_ground_state_value(
                        spin_z_operator(self.orbitals, 1), evecs, evals
                    )
                    + 0.5
                )
                spin_occ_1_down = (
                    average_ground_state_value(
                        spin_z_operator(self.orbitals, 0), evecs, evals
                    )
                    + 0.5
                )
                spin_occ_2_up = (
                    average_ground_state_value(
                        spin_z_operator(self.orbitals, 3), evecs, evals
                    )
                    + 0.5
                )
                spin_occ_2_down = (
                    average_ground_state_value(
                        spin_z_operator(self.orbitals, 2), evecs, evals
                    )
                    + 0.5
                )
                mu_new = self.find_mu(
                    mu_guess,
                    lambda1_new,
                    lambda2_new,
                    0.5 * (z1_up_new + z1_down_new),
                    0.5 * (z2_up_new + z2_down_new),
                    dens_guess,
                )
                dens_new = self.density_derivative(
                    mu_new,
                    mu_guess,
                    lambda1_new,
                    lambda2_new,
                    0.5 * (z1_up_new + z1_down_new),
                    0.5 * (z2_up_new + z2_down_new),
                )
                total_f = f1_up + f1_down + f2_up + f2_down

                converged = (
                    abs(z1_up_new - z1_up_guess) < self.config.tolerance
                    and abs(z1_down_new - z1_down_guess) < self.config.tolerance
                    and abs(z2_up_new - z2_up_guess) < self.config.tolerance
                    and abs(z2_down_new - z2_down_guess) < self.config.tolerance
                    and abs(lambda1_new - lambda1_guess) < self.config.tolerance
                    and abs(lambda2_new - lambda2_guess) < self.config.tolerance
                    and abs(h1_up_new - h1_up_guess) < self.config.tolerance
                    and abs(h1_down_new - h1_down_guess) < self.config.tolerance
                    and abs(h2_up_new - h2_up_guess) < self.config.tolerance
                    and abs(h2_down_new - h2_down_guess) < self.config.tolerance
                    and abs(mu_new - mu_guess) < self.config.tolerance
                    and abs(self.config.target_occupation - total_f)
                    < self.config.tolerance
                    and abs(spin_occ_1_up - spin_1_up_old) < self.config.tolerance
                    and abs(spin_occ_1_down - spin_1_down_old) < self.config.tolerance
                    and abs(spin_occ_2_up - spin_2_up_old) < self.config.tolerance
                    and abs(spin_occ_2_down - spin_2_down_old) < self.config.tolerance
                )
                if converged:
                    rows.append(
                        [
                            float(u_value),
                            mu_new,
                            0.5 * (z1_up_new + z1_down_new),
                            0.5 * (z2_up_new + z2_down_new),
                            lambda1_new,
                            lambda2_new,
                            0.5 * (h1_up_new + h1_down_new),
                            0.5 * (h2_up_new + h2_down_new),
                            0.5 * (f1_up + f1_down),
                            0.5 * (f2_up + f2_down),
                            0.5 * (spin_occ_1_up + spin_occ_1_down),
                            0.5 * (spin_occ_2_up + spin_occ_2_down),
                        ]
                    )
                    break

                alpha = self.config.mixing
                h1_up_guess = alpha * h1_up_new + (1.0 - alpha) * h1_up_guess
                h1_down_guess = alpha * h1_down_new + (1.0 - alpha) * h1_down_guess
                h2_up_guess = alpha * h2_up_new + (1.0 - alpha) * h2_up_guess
                h2_down_guess = alpha * h2_down_new + (1.0 - alpha) * h2_down_guess
                spin_1_up_old = alpha * spin_occ_1_up + (1.0 - alpha) * spin_1_up_old
                spin_1_down_old = (
                    alpha * spin_occ_1_down + (1.0 - alpha) * spin_1_down_old
                )
                spin_2_up_old = alpha * spin_occ_2_up + (1.0 - alpha) * spin_2_up_old
                spin_2_down_old = (
                    alpha * spin_occ_2_down + (1.0 - alpha) * spin_2_down_old
                )
                lambda1_guess = alpha * lambda1_new + (1.0 - alpha) * lambda1_guess
                lambda2_guess = alpha * lambda2_new + (1.0 - alpha) * lambda2_guess
                z1_up_guess = alpha * z1_up_new + (1.0 - alpha) * z1_up_guess
                z1_down_guess = alpha * z1_down_new + (1.0 - alpha) * z1_down_guess
                z2_up_guess = alpha * z2_up_new + (1.0 - alpha) * z2_up_guess
                z2_down_guess = alpha * z2_down_new + (1.0 - alpha) * z2_down_guess
                mu_guess = alpha * mu_new + (1.0 - alpha) * mu_guess
                dens_guess = alpha * dens_new + (1.0 - alpha) * dens_guess
            else:
                raise RuntimeError(
                    f"Paramagnetic solver did not converge for U={u_value}."
                )

        headers = [
            "U",
            "mu",
            "Z_1",
            "Z_2",
            "lamda_1",
            "lamda_2",
            "h_1",
            "h_2",
            "n_f_1",
            "n_f_2",
            "n_s_1",
            "n_s_2",
        ]
        return headers, np.array(rows, dtype=float)


class AntiferromagneticSolver:
    """Numerical solver for the two-orbital antiferromagnetic Bethe use case."""

    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.epsilon = np.linspace(-1.0, 1.0, config.integration_points)
        self.dos = bethe_density_of_states(self.epsilon, config.hopping)
        self.orbitals = config.orbitals

    def lambda_plus(
        self,
        epsilon: np.ndarray,
        lambda_up: float,
        lambda_down: float,
        z_up: float,
        z_down: float,
    ) -> np.ndarray:
        """Upper fermionic band of the AF effective Hamiltonian."""
        return (
            -(lambda_up + lambda_down)
            + np.sqrt((lambda_up - lambda_down) ** 2 + 4.0 * z_up * z_down * epsilon**2)
        ) / 2.0

    def lambda_minus(
        self,
        epsilon: np.ndarray,
        lambda_up: float,
        lambda_down: float,
        z_up: float,
        z_down: float,
    ) -> np.ndarray:
        """Lower fermionic band of the AF effective Hamiltonian."""
        return (
            -(lambda_up + lambda_down)
            - np.sqrt((lambda_up - lambda_down) ** 2 + 4.0 * z_up * z_down * epsilon**2)
        ) / 2.0

    def alpha_minus(
        self,
        z_up: float,
        z_down: float,
        epsilon: np.ndarray,
        lambda_up: float,
        lambda_down: float,
    ) -> np.ndarray:
        """Eigenvector coefficient for the lower AF fermionic band."""
        lambda_minus = self.lambda_minus(epsilon, lambda_up, lambda_down, z_up, z_down)
        numerator = np.sqrt(z_up * z_down) * epsilon * lambda_minus
        denominator = np.sqrt(
            z_up * z_down * epsilon**2 * lambda_minus**2
            + (
                z_up * z_down * epsilon**2
                - lambda_down * lambda_minus
                - lambda_up * lambda_down
            )
            ** 2
        )
        return numerator / denominator

    def beta_minus(
        self,
        z_up: float,
        z_down: float,
        epsilon: np.ndarray,
        lambda_up: float,
        lambda_down: float,
    ) -> np.ndarray:
        """Companion eigenvector coefficient for the lower AF fermionic band."""
        lambda_minus = self.lambda_minus(epsilon, lambda_up, lambda_down, z_up, z_down)
        numerator = (
            z_up * z_down * epsilon**2
            - lambda_down * lambda_minus
            - lambda_up * lambda_down
        )
        denominator = np.sqrt(
            z_up * z_down * epsilon**2 * lambda_minus**2
            + (
                z_up * z_down * epsilon**2
                - lambda_down * lambda_minus
                - lambda_up * lambda_down
            )
            ** 2
        )
        return numerator / denominator

    def alpha_plus(
        self,
        z_up: float,
        z_down: float,
        epsilon: np.ndarray,
        lambda_up: float,
        lambda_down: float,
    ) -> np.ndarray:
        """Eigenvector coefficient for the upper AF fermionic band."""
        lambda_plus = self.lambda_plus(epsilon, lambda_up, lambda_down, z_up, z_down)
        numerator = np.sqrt(z_up * z_down) * epsilon * lambda_plus
        denominator = np.sqrt(
            z_up * z_down * epsilon**2 * lambda_plus**2
            + (
                z_up * z_down * epsilon**2
                - lambda_down * lambda_plus
                - lambda_up * lambda_down
            )
            ** 2
        )
        return numerator / denominator

    def beta_plus(
        self,
        z_up: float,
        z_down: float,
        epsilon: np.ndarray,
        lambda_up: float,
        lambda_down: float,
    ) -> np.ndarray:
        """Companion eigenvector coefficient for the upper AF fermionic band."""
        lambda_plus = self.lambda_plus(epsilon, lambda_up, lambda_down, z_up, z_down)
        numerator = (
            z_up * z_down * epsilon**2
            - lambda_down * lambda_plus
            - lambda_up * lambda_down
        )
        denominator = np.sqrt(
            z_up * z_down * epsilon**2 * lambda_plus**2
            + (
                z_up * z_down * epsilon**2
                - lambda_down * lambda_plus
                - lambda_up * lambda_down
            )
            ** 2
        )
        return numerator / denominator

    def fermionic_up(
        self,
        lambda_up: float,
        lambda_down: float,
        z_up: float,
        z_down: float,
        mu: float,
    ) -> float:
        """Compute the spin-up fermionic occupation for one orbital."""
        alpha_plus = self.alpha_plus(z_up, z_down, self.epsilon, lambda_up, lambda_down)
        beta_plus = self.beta_plus(z_up, z_down, self.epsilon, lambda_up, lambda_down)
        alpha_minus = self.alpha_minus(
            z_up, z_down, self.epsilon, lambda_up, lambda_down
        )
        beta_minus = self.beta_minus(z_up, z_down, self.epsilon, lambda_up, lambda_down)
        denominator = alpha_plus * beta_minus - alpha_minus * beta_plus
        plus_term = (
            self.dos
            * np.abs(beta_minus / denominator) ** 2
            * fermi_distribution(
                self.lambda_plus(self.epsilon, lambda_up, lambda_down, z_up, z_down)
                - mu,
                self.config.beta,
            )
        )
        minus_term = (
            self.dos
            * np.abs(beta_plus / denominator) ** 2
            * fermi_distribution(
                self.lambda_minus(self.epsilon, lambda_up, lambda_down, z_up, z_down)
                - mu,
                self.config.beta,
            )
        )
        return float(simpson(plus_term + minus_term, x=self.epsilon))

    def fermionic_down(
        self,
        lambda_up: float,
        lambda_down: float,
        z_up: float,
        z_down: float,
        mu: float,
    ) -> float:
        """Compute the spin-down fermionic occupation for one orbital."""
        alpha_plus = self.alpha_plus(z_up, z_down, self.epsilon, lambda_up, lambda_down)
        beta_plus = self.beta_plus(z_up, z_down, self.epsilon, lambda_up, lambda_down)
        alpha_minus = self.alpha_minus(
            z_up, z_down, self.epsilon, lambda_up, lambda_down
        )
        beta_minus = self.beta_minus(z_up, z_down, self.epsilon, lambda_up, lambda_down)
        denominator = beta_plus * alpha_minus - alpha_plus * beta_minus
        plus_term = (
            self.dos
            * np.abs(alpha_minus / denominator) ** 2
            * fermi_distribution(
                self.lambda_plus(self.epsilon, lambda_up, lambda_down, z_up, z_down)
                - mu,
                self.config.beta,
            )
        )
        minus_term = (
            self.dos
            * np.abs(alpha_plus / denominator) ** 2
            * fermi_distribution(
                self.lambda_minus(self.epsilon, lambda_up, lambda_down, z_up, z_down)
                - mu,
                self.config.beta,
            )
        )
        return float(simpson(plus_term + minus_term, x=self.epsilon))

    def calculate_h_up(
        self,
        lambda_up: float,
        lambda_down: float,
        z_up: float,
        z_down: float,
        mu: float,
    ) -> float:
        """Compute the spin-up `h` field for one orbital."""
        alpha_plus = self.alpha_plus(z_up, z_down, self.epsilon, lambda_up, lambda_down)
        beta_plus = self.beta_plus(z_up, z_down, self.epsilon, lambda_up, lambda_down)
        alpha_minus = self.alpha_minus(
            z_up, z_down, self.epsilon, lambda_up, lambda_down
        )
        beta_minus = self.beta_minus(z_up, z_down, self.epsilon, lambda_up, lambda_down)
        denominator = (alpha_plus * beta_minus - alpha_minus * beta_plus) ** 2
        fermion_difference = fermi_distribution(
            self.lambda_plus(self.epsilon, lambda_up, lambda_down, z_up, z_down) - mu,
            self.config.beta,
        ) - fermi_distribution(
            self.lambda_minus(self.epsilon, lambda_up, lambda_down, z_up, z_down) - mu,
            self.config.beta,
        )
        integrand = (
            self.dos
            * self.epsilon
            * ((alpha_plus * beta_plus) / denominator)
            * fermion_difference
        )
        return float(np.sqrt(z_down) * simpson(integrand, x=self.epsilon))

    def calculate_h_down(
        self,
        lambda_up: float,
        lambda_down: float,
        z_up: float,
        z_down: float,
        mu: float,
    ) -> float:
        """Compute the spin-down `h` field for one orbital."""
        alpha_plus = self.alpha_plus(z_up, z_down, self.epsilon, lambda_up, lambda_down)
        beta_plus = self.beta_plus(z_up, z_down, self.epsilon, lambda_up, lambda_down)
        alpha_minus = self.alpha_minus(
            z_up, z_down, self.epsilon, lambda_up, lambda_down
        )
        beta_minus = self.beta_minus(z_up, z_down, self.epsilon, lambda_up, lambda_down)
        denominator = (alpha_plus * beta_minus - alpha_minus * beta_plus) ** 2
        fermion_difference = fermi_distribution(
            self.lambda_plus(self.epsilon, lambda_up, lambda_down, z_up, z_down) - mu,
            self.config.beta,
        ) - fermi_distribution(
            self.lambda_minus(self.epsilon, lambda_up, lambda_down, z_up, z_down) - mu,
            self.config.beta,
        )
        integrand = (
            self.dos
            * self.epsilon
            * ((alpha_plus * beta_plus) / denominator)
            * fermion_difference
        )
        return float(np.sqrt(z_up) * simpson(integrand, x=self.epsilon))

    def slave_hamiltonian(
        self,
        h1_up: float,
        h1_down: float,
        h2_up: float,
        h2_down: float,
        lambda1_up: float,
        lambda1_down: float,
        lambda2_up: float,
        lambda2_down: float,
        u_value: float,
        gauge1_up: float,
        gauge1_down: float,
        gauge2_up: float,
        gauge2_down: float,
    ) -> np.ndarray:
        """Construct the AF slave-spin Hamiltonian for two orbitals."""
        matrix = np.zeros((4**self.orbitals, 4**self.orbitals), dtype=np.complex128)
        matrix += lambda1_down * spin_z_operator(
            self.orbitals, 0
        ) + lambda1_up * spin_z_operator(self.orbitals, 1)
        matrix += lambda2_down * spin_z_operator(
            self.orbitals, 2
        ) + lambda2_up * spin_z_operator(self.orbitals, 3)
        matrix += h1_down * spin_flip_dagger(
            self.orbitals, 0, gauge1_down
        ) + h1_up * spin_flip_dagger(self.orbitals, 1, gauge1_up)
        matrix += np.conj(h1_down) * spin_flip_operator(
            self.orbitals, 0, gauge1_down
        ) + np.conj(h1_up) * spin_flip_operator(self.orbitals, 1, gauge1_up)
        matrix += h2_down * spin_flip_dagger(
            self.orbitals, 2, gauge2_down
        ) + h2_up * spin_flip_dagger(self.orbitals, 3, gauge2_up)
        matrix += np.conj(h2_down) * spin_flip_operator(
            self.orbitals, 2, gauge2_down
        ) + np.conj(h2_up) * spin_flip_operator(self.orbitals, 3, gauge2_up)
        spin_1_up = spin_z_operator(self.orbitals, 1)
        spin_1_down = spin_z_operator(self.orbitals, 0)
        spin_2_up = spin_z_operator(self.orbitals, 3)
        spin_2_down = spin_z_operator(self.orbitals, 2)
        matrix += u_value * (spin_1_up @ spin_1_down + spin_2_up @ spin_2_down)
        return matrix

    def gradient_descent(
        self, function: Callable[[np.ndarray], float], start: np.ndarray
    ) -> np.ndarray:
        """Numerically minimize a four-variable scalar objective."""
        vector = np.array(start, dtype=float)
        for _ in range(self.config.max_iterations):
            base = function(vector)
            if abs(base) <= 1e-8:
                break
            gradient = np.zeros_like(vector)
            for index in range(len(vector)):
                displaced = np.array(vector, copy=True)
                displaced[index] += GRADIENT_STEP
                gradient[index] = (function(displaced) - base) / GRADIENT_STEP
            vector -= self.config.mixing * gradient
        return vector

    def find_lambdas(
        self,
        guesses: np.ndarray,
        u_value: float,
        h_values: tuple[float, float, float, float],
        gauges: tuple[float, float, float, float],
        occupations: tuple[float, float, float, float],
    ) -> np.ndarray:
        """Update the AF Lagrange multipliers across both orbitals and spins."""

        def spread(values: np.ndarray) -> float:
            evals, evecs = scipy.linalg.eigh(
                self.slave_hamiltonian(
                    h_values[0],
                    h_values[1],
                    h_values[2],
                    h_values[3],
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                    u_value,
                    gauges[0],
                    gauges[1],
                    gauges[2],
                    gauges[3],
                )
            )
            spin_occ_1up = (
                average_ground_state_value(
                    spin_z_operator(self.orbitals, 1), evecs, evals
                )
                + 0.5
            )
            spin_occ_1down = (
                average_ground_state_value(
                    spin_z_operator(self.orbitals, 0), evecs, evals
                )
                + 0.5
            )
            spin_occ_2up = (
                average_ground_state_value(
                    spin_z_operator(self.orbitals, 3), evecs, evals
                )
                + 0.5
            )
            spin_occ_2down = (
                average_ground_state_value(
                    spin_z_operator(self.orbitals, 2), evecs, evals
                )
                + 0.5
            )
            return (
                (spin_occ_1up - occupations[0]) ** 2
                + (spin_occ_1down - occupations[1]) ** 2
                + (spin_occ_2up - occupations[2]) ** 2
                + (spin_occ_2down - occupations[3]) ** 2
            )

        return self.gradient_descent(function=spread, start=guesses)

    def density_derivative(
        self,
        mu_new: float,
        mu_old: float,
        lambda1_up: float,
        lambda1_down: float,
        z1_up: float,
        z1_down: float,
        lambda2_up: float,
        lambda2_down: float,
        z2_up: float,
        z2_down: float,
    ) -> float:
        """Estimate `dn/dmu` for the AF chemical potential update."""
        n_new = (
            self.fermionic_up(lambda1_up, lambda1_down, z1_up, z1_down, mu_new)
            + self.fermionic_down(lambda1_up, lambda1_down, z1_up, z1_down, mu_new)
            + self.fermionic_up(lambda2_up, lambda2_down, z2_up, z2_down, mu_new)
            + self.fermionic_down(lambda2_up, lambda2_down, z2_up, z2_down, mu_new)
        )
        n_old = (
            self.fermionic_up(lambda1_up, lambda1_down, z1_up, z1_down, mu_old)
            + self.fermionic_down(lambda1_up, lambda1_down, z1_up, z1_down, mu_old)
            + self.fermionic_up(lambda2_up, lambda2_down, z2_up, z2_down, mu_old)
            + self.fermionic_down(lambda2_up, lambda2_down, z2_up, z2_down, mu_old)
        )
        if mu_new == mu_old:
            return 0.0
        density = (n_new - n_old) / (mu_new - mu_old)
        return max(0.2, density) if density > 0 else min(-0.2, density)

    def find_mu(
        self,
        mu_guess: float,
        lambda1_up: float,
        lambda1_down: float,
        z1_up: float,
        z1_down: float,
        lambda2_up: float,
        lambda2_down: float,
        z2_up: float,
        z2_down: float,
        derivative: float,
    ) -> float:
        """Update the AF chemical potential using the legacy correction step."""
        occupation = (
            self.fermionic_up(lambda1_up, lambda1_down, z1_up, z1_down, mu_guess)
            + self.fermionic_down(lambda1_up, lambda1_down, z1_up, z1_down, mu_guess)
            + self.fermionic_up(lambda2_up, lambda2_down, z2_up, z2_down, mu_guess)
            + self.fermionic_down(lambda2_up, lambda2_down, z2_up, z2_down, mu_guess)
        )
        if derivative == 0:
            return mu_guess
        return float(
            mu_guess - (occupation - self.config.target_occupation) / derivative
        )

    def solve(self) -> tuple[list[str], np.ndarray]:
        """Run the full interaction sweep for the antiferromagnetic case."""
        rows: list[list[float]] = []
        mu_guess = self.config.mu_guess
        dens_guess = self.config.density_guess
        z1_up_guess = z1_down_guess = z2_up_guess = z2_down_guess = self.config.z_guess
        lambda_guesses = np.array(
            [
                self.config.lambda_1_up_guess,
                self.config.lambda_1_down_guess,
                self.config.lambda_2_up_guess,
                self.config.lambda_2_down_guess,
            ],
            dtype=float,
        )
        lambda_shift_guesses = np.array(
            [
                self.config.lambda_1_up_shift_guess,
                self.config.lambda_1_down_shift_guess,
                self.config.lambda_2_up_shift_guess,
                self.config.lambda_2_down_shift_guess,
            ],
            dtype=float,
        )
        h_old = np.array([-0.2, -0.2, -0.2, -0.2], dtype=float)
        spin_old = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)

        for u_value in make_u_values(self.config):
            for _ in range(self.config.max_iterations):
                lambda_tilde = lambda_guesses - lambda_shift_guesses
                f1_up = self.fermionic_up(
                    lambda_tilde[0],
                    lambda_tilde[1],
                    z1_up_guess,
                    z1_down_guess,
                    mu_guess,
                )
                f1_down = self.fermionic_down(
                    lambda_tilde[0],
                    lambda_tilde[1],
                    z1_up_guess,
                    z1_down_guess,
                    mu_guess,
                )
                f2_up = self.fermionic_up(
                    lambda_tilde[2],
                    lambda_tilde[3],
                    z2_up_guess,
                    z2_down_guess,
                    mu_guess,
                )
                f2_down = self.fermionic_down(
                    lambda_tilde[2],
                    lambda_tilde[3],
                    z2_up_guess,
                    z2_down_guess,
                    mu_guess,
                )
                occupations = (f1_up, f1_down, f2_up, f2_down)
                gauges = tuple(estimate_gauge(value) for value in occupations)
                h_values = (
                    self.calculate_h_up(
                        lambda_tilde[0],
                        lambda_tilde[1],
                        z1_up_guess,
                        z1_down_guess,
                        mu_guess,
                    ),
                    self.calculate_h_down(
                        lambda_tilde[0],
                        lambda_tilde[1],
                        z1_up_guess,
                        z1_down_guess,
                        mu_guess,
                    ),
                    self.calculate_h_up(
                        lambda_tilde[2],
                        lambda_tilde[3],
                        z2_up_guess,
                        z2_down_guess,
                        mu_guess,
                    ),
                    self.calculate_h_down(
                        lambda_tilde[2],
                        lambda_tilde[3],
                        z2_up_guess,
                        z2_down_guess,
                        mu_guess,
                    ),
                )
                lambda_new = self.find_lambdas(
                    lambda_guesses, u_value, h_values, gauges, occupations
                )
                evals, evecs = scipy.linalg.eigh(
                    self.slave_hamiltonian(
                        h_values[0],
                        h_values[1],
                        h_values[2],
                        h_values[3],
                        lambda_new[0],
                        lambda_new[1],
                        lambda_new[2],
                        lambda_new[3],
                        u_value,
                        gauges[0],
                        gauges[1],
                        gauges[2],
                        gauges[3],
                    )
                )
                average_1_up = average_ground_state_value(
                    spin_flip_operator(self.orbitals, 1, gauges[0]), evecs, evals
                )
                average_1_down = average_ground_state_value(
                    spin_flip_operator(self.orbitals, 0, gauges[1]), evecs, evals
                )
                average_2_up = average_ground_state_value(
                    spin_flip_operator(self.orbitals, 3, gauges[2]), evecs, evals
                )
                average_2_down = average_ground_state_value(
                    spin_flip_operator(self.orbitals, 2, gauges[3]), evecs, evals
                )
                z1_up_new = quasiparticle_weight(average_1_up)
                z1_down_new = quasiparticle_weight(average_1_down)
                z2_up_new = quasiparticle_weight(average_2_up)
                z2_down_new = quasiparticle_weight(average_2_down)
                spin_occ = np.array(
                    [
                        average_ground_state_value(
                            spin_z_operator(self.orbitals, 1), evecs, evals
                        )
                        + 0.5,
                        average_ground_state_value(
                            spin_z_operator(self.orbitals, 0), evecs, evals
                        )
                        + 0.5,
                        average_ground_state_value(
                            spin_z_operator(self.orbitals, 3), evecs, evals
                        )
                        + 0.5,
                        average_ground_state_value(
                            spin_z_operator(self.orbitals, 2), evecs, evals
                        )
                        + 0.5,
                    ]
                )
                lambda_shift_new = np.array(
                    [
                        4.0 * h_values[0] * average_1_up * eta(f1_up),
                        4.0 * h_values[1] * average_1_down * eta(f1_down),
                        4.0 * h_values[2] * average_2_up * eta(f2_up),
                        4.0 * h_values[3] * average_2_down * eta(f2_down),
                    ]
                )
                lambda_tilde_new = lambda_new - lambda_shift_new
                mu_new = self.find_mu(
                    mu_guess,
                    lambda_tilde_new[0],
                    lambda_tilde_new[1],
                    z1_up_new,
                    z1_down_new,
                    lambda_tilde_new[2],
                    lambda_tilde_new[3],
                    z2_up_new,
                    z2_down_new,
                    dens_guess,
                )
                dens_new = self.density_derivative(
                    mu_new,
                    mu_guess,
                    lambda_tilde_new[0],
                    lambda_tilde_new[1],
                    z1_up_new,
                    z1_down_new,
                    lambda_tilde_new[2],
                    lambda_tilde_new[3],
                    z2_up_new,
                    z2_down_new,
                )
                magnetizations = [abs(f1_up - f1_down), abs(f2_up - f2_down)]

                converged = (
                    abs(z1_up_new - z1_up_guess) < self.config.tolerance
                    and abs(z1_down_new - z1_down_guess) < self.config.tolerance
                    and abs(z2_up_new - z2_up_guess) < self.config.tolerance
                    and abs(z2_down_new - z2_down_guess) < self.config.tolerance
                    and np.all(
                        np.abs(lambda_new - lambda_guesses) < self.config.tolerance
                    )
                    and np.all(
                        np.abs(lambda_shift_new - lambda_shift_guesses)
                        < self.config.tolerance
                    )
                    and np.all(
                        np.abs(np.array(h_values) - h_old) < self.config.tolerance
                    )
                    and abs(mu_new - mu_guess) < self.config.tolerance
                    and abs(self.config.target_occupation - sum(occupations))
                    < self.config.tolerance
                    and np.all(np.abs(spin_occ - spin_old) < self.config.tolerance)
                )
                if converged:
                    rows.append(
                        [
                            float(u_value),
                            magnetizations[0],
                            magnetizations[1],
                            mu_new,
                            z1_up_new,
                            z1_down_new,
                            z2_up_new,
                            z2_down_new,
                            lambda_new[0],
                            lambda_new[2],
                            lambda_new[1],
                            lambda_new[3],
                            h_values[0],
                            h_values[1],
                            h_values[2],
                            h_values[3],
                            lambda_shift_new[0],
                            lambda_shift_new[1],
                            lambda_shift_new[2],
                            lambda_shift_new[3],
                            f1_up,
                            f1_down,
                            f2_up,
                            f2_down,
                            spin_occ[0],
                            spin_occ[1],
                            spin_occ[2],
                            spin_occ[3],
                        ]
                    )
                    break

                alpha = self.config.mixing
                h_old = alpha * np.array(h_values) + (1.0 - alpha) * h_old
                spin_old = alpha * spin_occ + (1.0 - alpha) * spin_old
                lambda_guesses = alpha * lambda_new + (1.0 - alpha) * lambda_guesses
                lambda_shift_guesses = (
                    alpha * lambda_shift_new + (1.0 - alpha) * lambda_shift_guesses
                )
                z1_up_guess = alpha * z1_up_new + (1.0 - alpha) * z1_up_guess
                z1_down_guess = alpha * z1_down_new + (1.0 - alpha) * z1_down_guess
                z2_up_guess = alpha * z2_up_new + (1.0 - alpha) * z2_up_guess
                z2_down_guess = alpha * z2_down_new + (1.0 - alpha) * z2_down_guess
                mu_guess = alpha * mu_new + (1.0 - alpha) * mu_guess
                dens_guess = alpha * dens_new + (1.0 - alpha) * dens_guess
            else:
                raise RuntimeError(f"AF solver did not converge for U={u_value}.")

        headers = [
            "U",
            "m1",
            "m2",
            "mu_new",
            "Z1_up_n",
            "Z1_down_n",
            "Z2_up_n",
            "Z2_down_n",
            "lamda1_up_new",
            "lamda2_up_new",
            "lamda1_down_new",
            "lamda2_down_new",
            "h1_new_up",
            "h1_new_down",
            "h2_new_up",
            "h2_new_down",
            "lamda1_0_up",
            "lamda1_0_down",
            "lamda2_0_up",
            "lamda2_0_down",
            "f1_up",
            "f1_down",
            "f2_up",
            "f2_down",
            "spin_occ_1up",
            "spin_occ_1down",
            "spin_occ_2up",
            "spin_occ_2down",
        ]
        return headers, np.array(rows, dtype=float)


def run_simulation(config: RunnerConfig) -> RunResult:
    """Execute one validated built-in simulation and return its numerical table."""
    start = perf_counter()
    solver = (
        ParamagneticSolver(config)
        if config.mode == "paramagnetic"
        else AntiferromagneticSolver(config)
    )
    headers, rows = solver.solve()
    return RunResult(
        config=config,
        headers=headers,
        rows=rows,
        elapsed_seconds=perf_counter() - start,
    )


def parse_args() -> argparse.Namespace:
    """Parse the package CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run pySSMF built-in Bethe lattice simulations."
    )
    parser.add_argument(
        "config", nargs="?", help="Optional path to a YAML config file."
    )
    parser.add_argument(
        "--output", help="Optional output path that overrides the config value."
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point for `python -m pyssmf.runner`."""
    args = parse_args()
    config = load_config(args.config)
    if args.output:
        config = config.model_copy(update={"output_path": args.output})
    result = run_simulation(config)
    destination = result.write()
    print(
        f"Completed {config.mode} Bethe lattice run with {len(result.rows)} points "
        f"in {result.elapsed_seconds:.2f}s -> {destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
