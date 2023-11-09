#!/usr/bin/env python

import time

from quantuminspire.sdk.models.circuit import Circuit
from quantuminspire.util.api.remote_backend import RemoteBackend

with Circuit(number_of_qubits=2, program_name="prgm1") as c:
    c.enter_section("my_section")
    c.x(0)
    c.hadamard(1)
    c.measure(0).measure(1)

print(c.content)

backend = RemoteBackend()

startTime = time.time()
backend.run(c, backend_type_id=3)
executionTime = time.time() - startTime
print("Execution time in seconds: " + str(executionTime))
