from quantuminspire.util.api.quantum_interface import QuantumInterface
from qiskit.providers import BackendV2 as Backend, Options, JobV1 as Job
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import CXGate, Measure


class QuantumInspireTarget(Target):
    def __init__(self) -> None:
        # this object establishes de properties of the quantum chip
        super().__init__(description="Quantum Inspire 2 Target",
                         num_qubits=4)
        cx_props = {
            (0, 1): InstructionProperties(duration=5.23e-7, error=0.00098115),
            (1, 0): InstructionProperties(duration=4.52e-7, error=0.00132115),
        }
        self.add_instruction(CXGate(), cx_props)

        measure_props = {
            (0,): InstructionProperties(duration=5.23e-7, error=0.00098115),
            (1,): InstructionProperties(duration=4.52e-7, error=0.00132115),
        }

        self.add_instruction(Measure(label="measure"), measure_props)



class QuantumInspireBackend(Backend):

    def __init__(self, qi: QuantumInterface) -> None:
        super().__init__(description="Quantum Inspire 2 Provider",
                         backend_version="0.1")

        self.qi = qi
        self.target = QuantumInspireTarget()

    def max_circuits(self) -> int:
        return 6

    def _default_options(cls) -> Options:
        return Options()

    def name(self) -> str:
        return "QuantumInspireBackend"

    def target(self) -> Target:
        return Target()

    def run(self, run_input: QuantumCircuit, **options) -> Job:
        # TODO: this method should also accept Schedule or ScheduleBlock or list
        del options
        return Job(backend=self, job_id="0")


