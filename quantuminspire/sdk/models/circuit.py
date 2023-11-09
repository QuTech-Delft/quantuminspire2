"""Module containing the Quantum Circuit class."""
from types import TracebackType
from typing import Any, Optional, Type

import opensquirrel
from compute_api_client import AlgorithmType, CompileStage

from quantuminspire.sdk.models.base_algorithm import BaseAlgorithm


class Circuit(BaseAlgorithm):
    """A container object, interacting with OpenSquirrel and building a (compiled) cQASM 3.0 string."""

    def __init__(self, number_of_qubits: int, program_name: str) -> None:
        super().__init__(f"{number_of_qubits}_qubits", program_name)
        self._opensquirrel_circuit_builder = opensquirrel.CircuitBuilder(opensquirrel.DefaultGates, number_of_qubits)
        self._cqasm: str = ""

    @property
    def content(self) -> str:
        return self._cqasm

    @property
    def content_type(self) -> str:
        return str(AlgorithmType.QUANTUM)

    @property
    def compile_stage(self) -> str:
        return str(CompileStage.NONE)

    def __enter__(self) -> "Circuit":
        self.initialize()
        return self

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        self.finalize()
        # return True ########## Why are exceptions suppressed here?

    def initialize(self) -> None:
        """Initialize the quantum circuit."""

    def finalize(self) -> None:
        """Finalize the quantum circuit.

        After finishing writing the quantum circuit using the circuit builder interface, the resulting program is
        compiled (or not) by OpenSquirrel. Finally, the compiled program is written to a cQASM3 string to an internal
        variable.
        """
        self._cqasm = str(self._opensquirrel_circuit_builder.to_circuit())

    def enter_section(self, name: str) -> opensquirrel.CircuitBuilder:
        """Enter a new section in the circuit, by skipping a line and adding a comment in the output cQASM3 string.

        Those "sections" are just here to help humans read and understand the output cQASM3 strings.
        They have no semantic meaning in the cQASM3 language.

        Args:
            name: Name of the section of the quantum circuit.

        Returns:
            The OpenSquirrel circuit builder.
        """

        self._opensquirrel_circuit_builder.comment(name)

        return self._opensquirrel_circuit_builder

    def __getattr__(self, attr: str) -> Any:
        return self._opensquirrel_circuit_builder.__getattr__(attr)
