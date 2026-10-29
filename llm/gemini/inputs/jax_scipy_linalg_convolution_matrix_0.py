
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def convolution_matrix_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, full mode
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    n = 4
    mode = "full"
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n, "mode": mode}))

    # Input 2: 1D float64 array, same mode
    a = np.array([0.5, -1.5, 2.0, -0.5], dtype=np.float64)
    n = 6
    mode = "same"
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n, "mode": mode}))

    # Input 3: 1D int32 array, valid mode
    a = np.array([1, -1, 2], dtype=np.int32)
    n = 5
    mode = "valid"
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n, "mode": mode}))

    # Input 4: Single element array, full mode
    a = np.array([3.14], dtype=np.float32)
    n = 3
    mode = "full"
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n, "mode": mode}))

    # Input 5: Single element array, valid mode
    a = np.array([-2.0], dtype=np.float64)
    n = 1
    mode = "valid"
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n, "mode": mode}))

    # Input 6: Complex64 array, same mode
    a = np.array([1.0 + 2.0j, -0.5 + 0.5j, 3.0 - 1.0j], dtype=np.complex64)
    n = 5
    mode = "same"
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n, "mode": mode}))

    # Input 7: Complex128 array, full mode
    a = np.array([2.0 - 3.0j, 1.0 + 1.0j], dtype=np.complex128)
    n = 4
    mode = "full"
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n, "mode": mode}))

    # Input 8: 1D int64 array, valid mode
    a = np.array([1, 0, -1, 2], dtype=np.int64)
    n = 7
    mode = "valid"
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n, "mode": mode}))

    # Input 9: Longer float32 array, same mode
    a = np.linspace(-2.0, 2.0, 7, dtype=np.float32)
    n = 10
    mode = "same"
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n, "mode": mode}))

    # Input 10: Float64 array with length equal to n, valid mode
    a = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float64)
    n = 4
    mode = "valid"
    list_of_inputs.append(copy.deepcopy({"a": a, "n": n, "mode": mode}))

    return list_of_inputs


generated_inputs["jax.scipy.linalg.convolution_matrix"] = (
    convolution_matrix_inputs()
)

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.convolution_matrix' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.convolution_matrix'.")


check_valid('jax.scipy.linalg.convolution_matrix', generated_inputs['jax.scipy.linalg.convolution_matrix'], lib="jax", suffix=0)
