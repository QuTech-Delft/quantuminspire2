# (H2) Hydrogen molecuole ground state energy determined using VQE with a UCCSD-ansatz function.
# Compared with Hartee-Fock energies and with energies calculated by NumPyMinimumEigensolver
# This script is based on the Qiskit Chemistry tutorials
import json
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from qiskit.primitives import Estimator, BackendEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector, Pauli
from qiskit_algorithms import VQEResult, NumPyMinimumEigensolverResult
from qiskit_algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper


from quantuminspire.util.api.quantum_interface import QuantumInterface
from quantuminspire.sdk.qiskit.backend import QuantumInspireBackend


class _IOMode(Enum):
    LOAD = "load"
    SAVE = "save"


class ExecutionMode(Enum):
    LOCAL = "local"
    REMOTE = "remote"
    LOCAL_EXACT = "local_exact"


class _IOHandler:  # todo: improve
    n_particles = (1, 1)
    n_spatial_orbitals = 2

    def __init__(self):
        with open("pauli.json") as fp:
            self._read_values = [
                (distance, nuclear_repulsion_energy, self._decode(encoded_pauli_list))
                for distance, nuclear_repulsion_energy, encoded_pauli_list in json.load(fp)
            ]

        self._write_values = list[tuple[float, float, list[tuple[str, dict[str, float]]]]]()

    def get(self, distance: float) -> tuple[float, SparsePauliOp]:
        idx = min(enumerate(self._read_values), key=lambda x: abs(x[1][0] - distance))[0]
        val_at_distance = self._read_values[idx]
        nuclear_repulsion_energy = val_at_distance[1]
        qubit_op = SparsePauliOp.from_list(val_at_distance[2])
        return nuclear_repulsion_energy, qubit_op

    @staticmethod
    def _decode(encoded_pauli_list):
        return [(op, val["r"] + 1.0j * val["i"]) for op, val in encoded_pauli_list]

    def append(self, distance: float, nuclear_repulsion_energy: float, qubit_op: SparsePauliOp) -> None:
        encoded = self._encode(qubit_op.to_list())
        self._write_values.append((distance, nuclear_repulsion_energy, encoded))

    @staticmethod
    def _encode(pauli_list: list[SparsePauliOp]) -> list[tuple[str, dict[str, float]]]:
        return [(op, {"r": np.real(val), "i": np.imag(val)}) for op, val in pauli_list]

    def save(self):
        if self._write_values:
            with open("pauli.json", "w") as fp:
                json.dump(self._write_values, fp)


@dataclass
class _GroundStateEnergyResults:
    result: VQEResult | NumPyMinimumEigensolverResult
    nuclear_repulsion_energy: float


class _GroundStateEnergyCalculator:
    def __init__(
        self, backend: QuantumInspireBackend,
            io_mode: _IOMode = _IOMode.SAVE,
            execution_mode: ExecutionMode = ExecutionMode.LOCAL,
            verbose: bool = True,
    ):
        self.backend = backend
        self._io_mode = io_mode
        self._execution_mode = execution_mode
        self._verbose = verbose
        self._io_handler = _IOHandler()

    def calculate(self, distance: float = 0.735) -> _GroundStateEnergyResults:
        # mapper = JordanWignerMapper()
        mapper = ParityMapper(num_particles=(1, 1))
        if self._io_mode == _IOMode.LOAD:
            nuclear_repulsion_energy, qubit_op = self._io_handler.get(distance)
            n_particles, n_spatial_orbitals = self._io_handler.n_particles, self._io_handler.n_spatial_orbitals
        else:
            molecule = f"H 0.0 0.0 0.0; H 0.0 0.0 {distance}"
            driver = PySCFDriver(molecule)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                es_problem = driver.run()

            fermionic_op = es_problem.hamiltonian.second_q_op()
            qubit_op = mapper.map(fermionic_op)
            n_particles = es_problem.num_particles
            n_spatial_orbitals = es_problem.num_spatial_orbitals

            nuclear_repulsion_energy = es_problem.nuclear_repulsion_energy
            self._io_handler.append(distance, nuclear_repulsion_energy, qubit_op)

        if self._execution_mode is not ExecutionMode.LOCAL_EXACT:
            initial_state = HartreeFock(n_spatial_orbitals, n_particles, mapper)
            ansatz = UCCSD(n_spatial_orbitals, n_particles, mapper, initial_state=initial_state)

            if self._execution_mode is ExecutionMode.LOCAL:
                optimizer = COBYLA(maxiter=10000)
                estimator = Estimator()
            else:
                optimizer = COBYLA(maxiter=1)  # 10 iterations take two minutes
                estimator = BackendEstimator(backend=self.backend)

            algo = VQE(estimator, ansatz, optimizer)
            result = algo.compute_minimum_eigenvalue(qubit_op)
        else:
            result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_op)

        if self._verbose:
            print(f"{distance=}: nuclear_repulsion_energy={nuclear_repulsion_energy}, eigenvalue={result.eigenvalue}")
        return _GroundStateEnergyResults(result, nuclear_repulsion_energy)


def execute(qi: QuantumInterface) -> None:

    calculator = _GroundStateEnergyCalculator(QuantumInspireBackend(qi),
                                              execution_mode=ExecutionMode.REMOTE)
    c = calculator.calculate()
    print(c)


def show_info(result: VQEResult | NumPyMinimumEigensolverResult):
    print(f"number of iterations: {result.cost_function_evals}")

    decomposed = result.optimal_circuit.decompose(reps=10)
    decomposed.draw("mpl", filename="ansatz.png", style="iqx")
    print(decomposed.draw("text"))

    circuit = result.optimal_circuit.assign_parameters(result.optimal_parameters)

    I, X, Y, Z = Pauli("I"), Pauli("X"), Pauli("Y"), Pauli("Z")
    operators = [I ^ I, I ^ Z, Z ^ I, Z ^ Z, X ^ X]
    for op in operators:
        psi = Statevector(circuit)
        expectation_value = psi.expectation_value(op)
        print(f"<~psi|{op}|psi> = {expectation_value.real}")

    qasm = circuit.qasm()
    print(qasm)
    Path("circuit.qasm").write_text(qasm)



if __name__ == "__main__":
    #QI.set_token_authentication("xxx")
    #backend = QI.get_backend("QX single-node simulator")
    _io_handler = _IOHandler()

    #results = run_single_at_default()
    #show_info(results.result)

    # df = run_multiple()
    # plot_multiple(df)

    _io_handler.save()