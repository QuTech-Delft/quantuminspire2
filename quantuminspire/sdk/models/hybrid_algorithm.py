"""Module containing the Hybrid Quantum/Classical Algorithm class."""

from compute_api_client import AlgorithmType

from quantuminspire.sdk.models.file_algorithm import FileAlgorithm


class HybridAlgorithm(FileAlgorithm):
    """A container object, reading the python algorithm and keeping metadata.

    The HybridAlgorithm reads the python file describing the algorithm and stores it in `.content`.
    """

    @property
    def content_type(self) -> AlgorithmType:
        return AlgorithmType.HYBRID

    @property
    def language_name(self) -> str:
        return "Python"
