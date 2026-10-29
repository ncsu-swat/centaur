
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def rsqrt_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    x = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": ""}))

    # Input 2: 2D float32 array
    x = np.array([[0.25, 1.0], [4.0, 16.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": ""}))

    # Input 3: Scalar (0D) float64 array
    x = np.array(2.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": ""}))

    # Input 4: 3D float64 array
    x = np.arange(1.0, 9.0, dtype=np.float64).reshape((2, 2, 2))
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": ""}))

    # Input 5: 1D complex64 array
    x = np.array([1.0 + 2.0j, -3.0 + 4.0j, 5.0 - 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": ""}))

    # Input 6: 2D complex128 array
    x = np.array([[2.0 + 0.0j, -4.0 + 1.0j], [0.5 - 1.5j, 3.0 + 2.0j]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": ""}))

    # Input 7: 4D float32 array
    x = np.random.uniform(0.1, 10.0, size=(2, 1, 3, 2)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": ""}))

    # Input 8: 1D float16 array
    x = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": ""}))

    # Input 9: 2D float64 array with large values
    x = np.array([[1e4, 1e8], [1e12, 1e16]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": ""}))

    # Input 10: 2D float32 array with small values
    x = np.array([[1e-4, 1e-6], [1e-8, 1e-10]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": ""}))

    # Input 11: 3D complex64 array
    x = (np.random.uniform(0.5, 5.0, size=(2, 3, 2)) + 1j * np.random.uniform(0.5, 5.0, size=(2, 3, 2))).astype(np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x, "accuracy": ""}))

    return list_of_inputs

generated_inputs["jax.lax.rsqrt_1"] = rsqrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.rsqrt_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.rsqrt_1'.")


check_valid('jax.lax.rsqrt', generated_inputs['jax.lax.rsqrt_1'], lib="jax", suffix=1)
