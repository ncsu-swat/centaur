
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.lax
import numpy as np


class Tolerance(jax.lax.Tolerance):

    def __iter__(self):
        return iter((self.atol, self.rtol, self.ulps))

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return (self.atol, self.rtol, self.ulps)[idx]


def tan_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive, negative, and zero values
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    accuracy = Tolerance(atol=0.1, rtol=0.1, ulps=100)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 2: 2D float64 array
    x = np.random.uniform(-1.2, 1.2, size=(3, 4)).astype(np.float64)
    accuracy = Tolerance(atol=0.2, rtol=0.2, ulps=200)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 3: 0D float32 scalar array
    x = np.array(0.7853982, dtype=np.float32)
    accuracy = Tolerance(atol=0.1, rtol=0.1, ulps=100)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 4: 3D float32 array
    x = np.random.uniform(-1.0, 1.0, size=(2, 3, 4)).astype(np.float32)
    accuracy = Tolerance(atol=0.5, rtol=0.5, ulps=500)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 5: 1D float16 array
    x = np.array([-1.2, -0.3, 0.0, 0.3, 1.2], dtype=np.float16)
    accuracy = Tolerance(atol=1.0, rtol=1.0, ulps=1000)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 6: 2D complex64 array
    real_part = np.random.uniform(-1.0, 1.0, size=(2, 3))
    imag_part = np.random.uniform(-1.0, 1.0, size=(2, 3))
    x = (real_part + 1j * imag_part).astype(np.complex64)
    accuracy = Tolerance(atol=0.1, rtol=0.1, ulps=100)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 7: 1D complex128 array
    x = np.array(
        [0.5 + 0.5j, -0.5 - 0.5j, 1.0 - 0.2j, -1.0 + 0.2j], dtype=np.complex128
    )
    accuracy = Tolerance(atol=0.1, rtol=0.1, ulps=100)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 8: 4D float32 array
    x = np.random.uniform(-0.8, 0.8, size=(1, 2, 2, 3)).astype(np.float32)
    accuracy = Tolerance(atol=0.2, rtol=0.2, ulps=200)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 9: Empty 1D float32 array
    x = np.array([], dtype=np.float32)
    accuracy = Tolerance(atol=0.1, rtol=0.1, ulps=100)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 10: 5D float64 array
    x = np.random.uniform(-1.0, 1.0, size=(2, 1, 2, 1, 2)).astype(np.float64)
    accuracy = Tolerance(atol=0.1, rtol=0.1, ulps=100)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    # Input 11: 2D float32 array with common angles
    x = np.array(
        [
            [-np.pi / 4, np.pi / 4],
            [-np.pi / 6, np.pi / 6],
            [-np.pi / 3, np.pi / 3],
        ],
        dtype=np.float32,
    )
    accuracy = Tolerance(atol=0.1, rtol=0.1, ulps=100)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": accuracy}))

    return list_of_inputs


generated_inputs["jax.lax.tan_3"] = tan_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.tan_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.tan_3'.")


check_valid('jax.lax.tan', generated_inputs['jax.lax.tan_3'], lib="jax", suffix=3)
