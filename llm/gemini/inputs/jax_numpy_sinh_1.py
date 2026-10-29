
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sinh_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 2D array of float64
    x = np.array([[-2.0, 0.5], [1.5, -4.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: Scalar (0D array)
    x = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 3D array of integers (promoted by JAX to inexact)
    x = np.arange(-6, 6, dtype=np.int32).reshape(2, 2, 3)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 1D complex64 array
    x = np.array([1.0 + 2.0j, -3.0 + 0.5j, 0.0 - 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D complex128 array
    x = np.array([[2.0 - 3.0j, -1.0 + 1.0j], [0.5 + 0.5j, -2.5 - 1.5j]], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 4D float16 array
    x = np.random.randn(2, 3, 1, 4).astype(np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: Array containing infinities and NaNs
    x = np.array([-np.inf, -1.0, 0.0, np.nan, np.inf], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: Large 1D array with values close to 0
    x = np.linspace(-1e-4, 1e-4, 100, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: Empty array
    x = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: Non-contiguous / transposed array
    x = np.arange(12, dtype=np.float32).reshape(3, 4).T
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.numpy.sinh_1"] = sinh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sinh_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sinh_1'.")


check_valid('jax.numpy.sinh', generated_inputs['jax.numpy.sinh_1'], lib="jax", suffix=1)
