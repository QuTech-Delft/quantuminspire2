from typing import Any, Dict, List
from functools import partial
from quantuminspire.sdk.models.circuit import Circuit
from quantuminspire.util.api.quantum_interface import QuantumInterface
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit.circuit import Parameter
from opensquirrel.ir import Qubit, Measure
import numpy as np

number_of_qubits = 2


def counts_to_distr(counts) -> dict[int, float]:
    """Convert Qiskit result counts to a dictionary

    The dictionary has integers as keys, and pseudo-probabilities as values.
    """
    n_shots = sum(counts.values())
    output_distr = {int(k, 2): v / n_shots for k, v in counts.items()}
    return {k: output_distr[k] for k in sorted(output_distr)}


class AverageDecreaseTermination:
    def __init__(self, N: int, tolerance: float = 0.0):
        """Callback to terminate optimization based the average decrease

        The average decrease over the last N data points is compared to the specified tolerance.
        The average decrease is determined by a linear fit (least squares) to the data.

        This class can be used as an argument to the Qiskit SPSA optimizer.

        Args:
            N: Number of data points to use
            tolerance: Abort if the average decrease is smaller than the specified tolerance

        """
        self.N = N
        self.tolerance = tolerance
        self.reset()

    @property
    def parameters(self):
        return self._parameters

    @property
    def values(self):
        return self._values

    def reset(self):
        """Reset the data"""
        self._values = []
        self._parameters = []

    def __call__(self, nfev, parameters, value, update, accepted) -> bool:
        """
        Args:
            nfev: Number of evaluations
            parameters: Current parameters in the optimization
            value: Value of the objective function
            update: Update step
            accepted: Whether the update was accepted

        Returns:
            True if the optimization loop should be aborted
        """
        self._values.append(value)
        self._parameters.append(parameters)

        if len(self._values) > self.N:
            last_values = self._values[-self.N :]
            pp = np.polyfit(range(self.N), last_values, 1)
            slope = pp[0] / self.N

            if slope > self.tolerance:
                return True
        return False

def generate_ansatz(params):

    with Circuit(platform_name="spin-2", program_name="prgm1", number_of_qubits=2) as circuit:
        circuit.ir.H(Qubit(0))  #U
        circuit.ir.H(Qubit(1))  #U
        circuit.ir.CZ(Qubit(0), Qubit(1))
        circuit.ir.H(Qubit(0))  #U
        circuit.ir.H(Qubit(1))  #U
        for ii in range(number_of_qubits):
            Measure(ii, ii)

    return circuit


def objective_function(params, qi,
                       target_distribution, nshots=None):
    """Compares the output distribution of our circuit with
    parameters `params` to the target distribution."""
    qc = generate_ansatz(params)

    execute_result = qi.execute_circuit(qc.content, nshots)
    # Convert the result to a dictionary with probabilities
    output_distr = counts_to_distr(execute_result.results)
    # Calculate the cost as the distance between the output
    # distribution and the target distribution
    cost = sum(
        abs(target_distribution.get(i, 0) - output_distr.get(i, 0))
        for i in range(2 ** number_of_qubits)
    )
    return cost


def data_callback(iteration: int, parameters: Any, residual: float) -> None:
    """Callback used to store data

    Args:
        iteration: Iteration on the optimization procedure
        parameters: Current values of the parameters to be optimized
        residual: Current residual (value of the objective function)

    """
    # do nothing for now
    pass


def qiskit_callback(number_evaluations, parameters, value, stepsize, accepted):
    data_callback(number_evaluations, parameters, value)

def execute(qi: QuantumInterface) -> None:

    optimizer = SPSA(maxiter=500, callback=qiskit_callback,
                     termination_checker=AverageDecreaseTermination(N=35))

    m = 2 ** number_of_qubits
    p0 = np.random.random(m) + 0.2
    p0 = p0 / np.sum(p0)
    target_distribution = {k: p0[k] for k in range(m)}
    F = partial(
        objective_function, qi=qi, target_distribution=target_distribution, nshots=2400
    )

    number_of_parameters = 12
    initial_parameters = .85 * np.random.rand(number_of_parameters, )
    result = optimizer.minimize(fun=F, x0=initial_parameters)

    return result


def finalize(list_of_measurements: Dict[int, List[Any]]) -> Dict[str, Any]:
    return {"measurements": list_of_measurements}


if __name__ == "__main__":
    p = 0.5 + 0.25 * np.random.random()
    target_distribution = {0: p, 1: 1 - p}

    # Run the individual steps for debugging
    number_of_parameters = 12
    qc = generate_ansatz([Parameter(f"s_{i}") for i in range(number_of_parameters)])
    initial_parameters = .85 * np.random.rand(number_of_parameters, )
    print(qc.content)
    print(initial_parameters)

    optimizer = SPSA(maxiter=500, callback=qiskit_callback,
                     termination_checker=AverageDecreaseTermination(N=35))

    F = partial(
        objective_function, qi=None, target_distribution=target_distribution, nshots=2400
    )

    result = optimizer.minimize(fun=F, x0=initial_parameters)
    print(optimizer)