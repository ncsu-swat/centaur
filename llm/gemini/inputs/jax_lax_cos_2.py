
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax
import numpy as np


class IterableTolerance(jax.lax.Tolerance):

    def __iter__(self):
        if hasattr(self, "__dict__") and self.__dict__:
            return iter(self.__dict__.values())
        if hasattr(self, "__slots__") and self.__slots__:
            return iter([getattr(self, s) for s in self.__slots__])
        return iter((1e-4, 1e-4, 1))

    def __len__(self):
        return len(list(iter(self)))

    def __getitem__(self, idx):
        return list(iter(self))[idx]


def cos_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with common angles
    x = np.array([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi], dtype=np.float32)
    accuracy = IterableTolerance(1e-4, 1e-4, 1)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 2: 2D float64 array with random values
    x = np.random.randn(3, 4).astype(np.float64)
    accuracy = IterableTolerance(1e-5, 1e-5, 2)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 3: 0D scalar float32
    x = np.array(0.5, dtype=np.float32)
    accuracy = IterableTolerance(1e-3, 1e-3, 0)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 4: 3D float32 array
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    accuracy = IterableTolerance(1e-4, 1e-4, 1)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 5: 1D complex64 array
    x = np.array([1.0 + 2.0j, -0.5 - 1.5j, 0.0 + 0.0j], dtype=np.complex64)
    accuracy = IterableTolerance(1e-4, 1e-4, 1)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 6: 2D complex128 array
    x = (
        np.random.randn(3, 3).astype(np.float64)
        + 1j * np.random.randn(3, 3).astype(np.float64)
    ).astype(np.complex128)
    accuracy = IterableTolerance(1e-6, 1e-6, 4)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 7: 4D float16 array
    x = np.random.randn(2, 2, 3, 3).astype(np.float16)
    accuracy = IterableTolerance(1e-2, 1e-2, 5)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 8: 1D linspace covering multiple periods
    x = np.linspace(-4 * np.pi, 4 * np.pi, 50, dtype=np.float32)
    accuracy = IterableTolerance(1e-5, 1e-5, 1)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 9: 5D float32 array with singleton dimensions
    x = np.random.uniform(-1.0, 1.0, size=(2, 1, 3, 1, 2)).astype(np.float32)
    accuracy = IterableTolerance(1e-4, 1e-4, 2)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 10: 2D float32 array with zero-sized dimension
    x = np.empty((0, 4), dtype=np.float32)
    accuracy = IterableTolerance(1e-4, 1e-4, 1)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 11: 1D float64 array with larger magnitude values
    x = np.array([-1e4, -100.0, 0.0, 100.0, 1e4], dtype=np.float64)
    accuracy = IterableTolerance(1e-5, 1e-5, 2)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 12: 3D complex64 array
    x = (
        np.random.randn(2, 2, 2).astype(np.float32)
        + 1j * np.random.randn(2, 2, 2).astype(np.float32)
    ).astype(np.complex64)
    accuracy = IterableTolerance(1e-3, 1e-3, 0)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    return list_of_inputs


generated_inputs["jax.lax.cos_2"] = cos_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.cos_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.cos_2'.")


check_valid('jax.lax.cos', generated_inputs['jax.lax.cos_2'], lib="jax", suffix=2)
