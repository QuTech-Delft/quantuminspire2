from typing import Any
from quantuminspire.util.api.quantum_interface import QuantumInterface
import numpy as np


def execute(qi: QuantumInterface) -> None:
    distances = np.arange(0.3, 2.5, 0.1)
    results = []
    for distance in distances:
        from qiskit_nature.second_q.drivers import PySCFDriver


        molecule = f"H 0.0 0.0 0.0; H 0.0 0.0 {distance}"
        driver = PySCFDriver(molecule)
        es_problem = driver.run()

        fermionic_op = es_problem.hamiltonian.second_q_op()
        n_particles = es_problem.num_particles
        n_spatial_orbitals = es_problem.num_spatial_orbitals
        nuclear_repulsion_energy = es_problem.nuclear_repulsion_energy

        from qiskit_nature.second_q.mappers import ParityMapper
        from qiskit_nature.second_q.mappers import JordanWignerMapper

        mapper = JordanWignerMapper()
        qubit_op = mapper.map(fermionic_op)

        from qiskit.quantum_info import SparsePauliOp

        qubit_op = SparsePauliOp(["X", "Z"], coeffs=np.array([1, distance]))
        nuclear_repulsion_energy = 0

        from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

        initial_state = HartreeFock(n_spatial_orbitals, n_particles, mapper)
        ansatz = UCCSD(n_spatial_orbitals, n_particles, mapper, initial_state=initial_state)

        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
        params = [Parameter("t_0"), Parameter("t_1")]

        ansatz = QuantumCircuit(1)
        ansatz.ry(params[0], qubit=0)
        ansatz.rz(params[1], qubit=0)

        from qiskit.primitives import BackendEstimator
        from quantuminspire.sdk.qiskit.backend import QuantumInspireBackend

        backend = QuantumInspireBackend(qi)  # qi is passed into `execute`
        estimator = BackendEstimator(backend=backend)

        from qiskit_algorithms.optimizers import COBYLA

        optimizer = COBYLA(maxiter=100)

        from qiskit_algorithms import VQE

        algo = VQE(estimator, ansatz, optimizer)
        result = algo.compute_minimum_eigenvalue(qubit_op)

        result_dict = {"distance": distance, "total_energy": result.eigenvalue + nuclear_repulsion_energy}

        from qiskit import transpile

        transpiled_circuit = transpile(result.optimal_circuit, basis_gates=['ry', 'h', 'cx', 'x', 'sdg', 'rz', 's'])
        result_dict["circuit"] = repr(transpiled_circuit.draw()).replace("'", '"')
        result_dict["circuit_depth"] = transpiled_circuit.depth()
        result_dict["optimal_t_0"] = result.optimal_parameters[params[0]]
        result_dict["optimal_t_1"] = result.optimal_parameters[params[1]]

        # result_dict["circuit"] = result.optimal_circuit.decompose(reps=20).draw("text")
        results.append(result_dict)

    qi.results = {"results": results}


def finalize(results: Any) -> dict[str, Any]:
    return results
