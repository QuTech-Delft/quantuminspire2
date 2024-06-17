# (H2) Hydrogen molecule ground state energy determined using VQE with a UCCSD-ansatz function.
# Compared with Hartee-Fock energies and with energies calculated by NumPyMinimumEigensolver
# This script is based on the Qiskit Chemistry tutorials
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile
from qiskit.primitives import BackendEstimator
from qiskit_algorithms import NumPyMinimumEigensolverResult, VQEResult
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper

from quantuminspire.sdk.models.hybrid_algorithm import HybridAlgorithm
from quantuminspire.sdk.qiskit.backend import QuantumInspireBackend
from quantuminspire.util.api.local_backend import LocalBackend
from quantuminspire.util.api.quantum_interface import QuantumInterface


@dataclass
class _GroundStateEnergyResults:
    result: VQEResult | NumPyMinimumEigensolverResult
    nuclear_repulsion_energy: float


def calculate_H0(backend: QuantumInspireBackend, distance: float = 0.735) -> _GroundStateEnergyResults:
    molecule = f"H 0.0 0.0 0.0; H 0.0 0.0 {distance}"
    driver = PySCFDriver(molecule)
    es_problem = driver.run()

    fermionic_op = es_problem.hamiltonian.second_q_op()
    n_particles = es_problem.num_particles
    n_spatial_orbitals = es_problem.num_spatial_orbitals
    nuclear_repulsion_energy = es_problem.nuclear_repulsion_energy

    mapper = ParityMapper(num_particles=(1, 1))
    qubit_op = mapper.map(fermionic_op)

    from qiskit.quantum_info import Pauli
    qubit_op = Pauli("X")

    estimator = BackendEstimator(backend=backend)
    initial_state = HartreeFock(n_spatial_orbitals, n_particles, mapper)
    ansatz = UCCSD(n_spatial_orbitals, n_particles, mapper, initial_state=initial_state)

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    params = [Parameter("t_0"), Parameter("t_1")]

    ansatz = QuantumCircuit(1)
    ansatz.ry(params[0], qubit=0)
    ansatz.rz(params[1], qubit=0)

    optimizer = COBYLA(maxiter=1)

    algo = VQE(estimator, ansatz, optimizer)
    result = algo.compute_minimum_eigenvalue(qubit_op)

    breakpoint()
    result.optimal_parameters

    # print(f"{distance=}: nuclear_repulsion_energy={nuclear_repulsion_energy:.2f}, eigenvalue={result.eigenvalue:.2f}")
    return _GroundStateEnergyResults(result, nuclear_repulsion_energy)


def execute(qi: QuantumInterface) -> None:
    # distances = np.arange(0.3, 2.5, 0.1)
    distances = np.arange(0.3, 0.5, 0.1)
    results = []
    for distance in distances:
        ground_state_energy_results = calculate_H0(backend=QuantumInspireBackend(qi), distance=distance)
        # result = dataclasses.asdict(ground_state_energy_results)
        result = {}
        # result["eigenvalue"] = ground_state_energy_results.result.eigenvalue
        # result["nuclear_repulsion_energy"] = ground_state_energy_results.nuclear_repulsion_energy
        result["total_energy"] = ground_state_energy_results.nuclear_repulsion_energy + ground_state_energy_results.result.eigenvalue
        result["distance"] = distance
        results.append(result)
        # print(f"{distance}, {result['total_energy']}")

    qi.results = {"results": results}
    # print(results)


def finalize(results: Any) -> dict[str, Any]:
    return results


if __name__ == "__main__":
    # Run the individual steps for debugging
    algorithm = HybridAlgorithm("test", "test")
    algorithm.read_file(Path(__file__))

    # from quantuminspire.util.api.remote_backend import RemoteBackend
    # backend = RemoteBackend()
    # job_id = backend.run(algorithm, backend_type_id=3, number_of_shots=1024)
    # results = backend.get_results(job_id)

    local_backend = LocalBackend()
    job_id = local_backend.run(algorithm, 0)
    results = local_backend.get_results(job_id)["results"]
    print("=== Execute ===\n", results)

    # plot total energy vs distance between atoms
    for y_name in ["total_energy"]:
        distances, energies = [[result[key] for result in results] for key in ["distance", y_name]]
        plt.plot(distances, energies, label=y_name)
    plt.legend()
    plt.show()
    plt.savefig(Path("~/repositories/quantuminspire2/env/test.png").expanduser())

    # draw optimal circuit of first distance as text
    circuit = transpile(results[0]["result"].optimal_circuit, basis_gates=['ry', 'h', 'cx', 'x', 'sdg', 'rz', 's'])
    print(circuit.draw("text"))


