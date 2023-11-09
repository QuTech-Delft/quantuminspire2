from typing import Generator
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from quantuminspire.sdk.models.circuit import Circuit

MOCK_QUANTUM_CIRCUIT = "quantum circuit"


@pytest.fixture
def opensquirrel(mocker: MockerFixture) -> Generator[MagicMock, None, None]:
    yield mocker.patch("quantuminspire.sdk.models.circuit.opensquirrel")


def test_create(opensquirrel: MagicMock) -> None:
    _ = Circuit(number_of_qubits=5, program_name="program")
    opensquirrel.CircuitBuilder.assert_called_once()


def test_get_program_name(opensquirrel: MagicMock) -> None:
    with Circuit(number_of_qubits=5, program_name="program") as c:
        pass

    assert c.program_name == "program"


def test_get_platform_name(opensquirrel: MagicMock) -> None:
    with Circuit(number_of_qubits=5, program_name="program") as c:
        pass

    assert c.platform_name == "5_qubits"


def test_get_content_type(opensquirrel: MagicMock) -> None:
    with Circuit(number_of_qubits=5, program_name="program") as c:
        pass

    assert c.content_type == "quantum"


def test_get_compile_stage(opensquirrel: MagicMock) -> None:
    with Circuit(number_of_qubits=5, program_name="program") as c:
        pass

    assert c.compile_stage == "none"


def test_create_empty_circuit(opensquirrel: MagicMock) -> None:
    opensquirrel.CircuitBuilder().to_circuit.return_value = MOCK_QUANTUM_CIRCUIT

    with Circuit(number_of_qubits=5, program_name="program") as c:
        pass

    opensquirrel.CircuitBuilder().to_circuit.assert_called_once()
    assert c.content == MOCK_QUANTUM_CIRCUIT


def test_create_circuit_with_kernel(opensquirrel: MagicMock) -> None:
    with Circuit(number_of_qubits=5, program_name="program") as c:
        c.enter_section("section1")
        c.x(0)

    opensquirrel.CircuitBuilder().comment.assert_called_once()


def test_create_circuit_with_multiple_kernels(opensquirrel: MagicMock) -> None:
    with Circuit(number_of_qubits=5, program_name="program") as c:
        c.enter_section("section1")
        c.x(0)
        c.enter_section("section2")

    assert len(opensquirrel.CircuitBuilder().comment.mock_calls) == 2


def test_create_circuit_reuse_kernel(opensquirrel: MagicMock) -> None:
    with Circuit(number_of_qubits=5, program_name="program") as c:
        s1 = c.enter_section("section1")
        s1.x(0)
        c.enter_section("section2")

    assert len(opensquirrel.CircuitBuilder().comment.mock_calls) == 2
