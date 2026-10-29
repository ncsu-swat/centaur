
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import jax.lax as lax
import numpy as np


class IterableTolerance(lax.Tolerance):

    def __init__(self, atol=0.0, rtol=0.0, ulps=0):
        super().__init__(atol=float(atol), rtol=float(rtol), ulps=int(ulps))

    def __iter__(self):
        return iter((float(self.atol), float(self.rtol), int(self.ulps)))

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return (float(self.atol), float(self.rtol), int(self.ulps))[idx]


def log_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    x = np.array(2.71828, dtype=np.float32)
    accuracy = IterableTolerance(atol=0.0, rtol=0.0, ulps=2)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 array
    x = np.array([0.5, 1.0, 2.0, 10.0, 100.0], dtype=np.float32)
    accuracy = IterableTolerance(atol=1e-3, rtol=1e-3, ulps=2)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 matrix
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    accuracy = IterableTolerance(atol=0.0, rtol=0.0, ulps=2)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array
    x = np.random.uniform(1.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    accuracy = IterableTolerance(atol=0.0, rtol=0.0, ulps=4)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float16 array
    x = np.array([0.1, 0.5, 1.0, 2.0], dtype=np.float16)
    accuracy = IterableTolerance(atol=1e-2, rtol=1e-2, ulps=0)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D complex64 array
    x = np.array(
        [[1.0 + 1.0j, 2.0 - 0.5j], [1.0 + 2.0j, 0.5 + 1.5j]],
        dtype=np.complex64,
    )
    accuracy = IterableTolerance(atol=1e-3, rtol=1e-3, ulps=0)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D complex128 array
    x = np.array([1.0 + 0.0j, 2.0 + 1.0j, 0.5 + 3.0j], dtype=np.complex128)
    accuracy = IterableTolerance(atol=0.0, rtol=0.0, ulps=4)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 array
    x = np.random.uniform(0.1, 5.0, size=(2, 2, 3, 3)).astype(np.float32)
    accuracy = IterableTolerance(atol=0.01, rtol=0.01, ulps=4)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D float32 array with small fractions
    x = np.array([0.01, 0.1, 0.25, 0.5, 1.0], dtype=np.float32)
    accuracy = IterableTolerance(atol=0.05, rtol=0.05, ulps=0)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float32 array
    x = np.random.uniform(1.0, 20.0, size=(1, 2, 2, 2, 2)).astype(np.float32)
    accuracy = IterableTolerance(atol=0.0, rtol=0.0, ulps=8)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D float64 array with small values
    x = np.array([[1e-10, 1e-5], [1e-3, 1e-1]], dtype=np.float64)
    accuracy = IterableTolerance(atol=1e-4, rtol=1e-4, ulps=2)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 1D float64 array with large values
    x = np.array([1e3, 1e6, 1e9], dtype=np.float64)
    accuracy = IterableTolerance(atol=1e-3, rtol=0.0, ulps=0)
    input_dict = {"x": x, "accuracy": accuracy}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.lax.log_2"] = log_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.log_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.log_2'.")


check_valid('jax.lax.log', generated_inputs['jax.lax.log_2'], lib="jax", suffix=2)
