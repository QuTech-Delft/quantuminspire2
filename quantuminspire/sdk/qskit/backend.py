from quantuminspire.util.api.quantum_interface import QuantumInterface
from qiskit.providers import BackendV2, Options, BackendStatus


class QuantumInspireBackend(BackendV2):

    def __init__(self, qi: QuantumInterface) -> None:
        super().__init__(description="Quantum Inspire 2 Provider",
                         backend_version="0.1")

        self.qi = qi

    def max_circuits(self) -> int:
        return 6

    def _default_options(cls) -> Options:
        return Options()

    def name(self) -> str:
        return "QuantumInspireBackend"

    def status(self) -> BackendStatus:
        return BackendStatus(
            backend_name=self.name(),
            backend_version="1",
            operational=True,
            pending_jobs=0,
            status_msg="",
        )


