
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_arccosh_inputs():
    list_of_inputs = []

    # Input 1: 1D array with values >= 1 (standard domain for real output)
    x1 = np.array([1.0, 1.5, 2.0, 5.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x1)})

    # Input 2: 2D array with values >= 1
    x2 = np.array([[1.0, 3.0], [5.0, 10.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x2)})

    # Input 3: 3D array with values >= 1
    x3 = np.array([[[2.0, 4.0], [6.0, 8.0]], [[1.1, 1.5], [1.9, 2.2]]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x3)})

    # Input 4: Float64 precision array
    x4 = np.array([1.0001, 1.001, 1.01, 1.1], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x4)})

    # Input 5: Complex-valued 1D array (supports full complex domain)
    x5 = np.array([1.0 + 2.0j, -5.0 + 0.0j, 0.0 + 1.0j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x5)})

    # Input 6: Complex-valued 2D array
    x6 = np.array([[-1.0 + 1.0j, 0.0 + 0.0j], [2.0 - 3.0j, -jnp_inf if 'jnp_inf' in locals() else -np.inf + 0j]], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x6)})

    # Input 7: Values in the domain < 1 (results in nan for real-valued, but valid input)
    x7 = np.array([-10.0, -1.0, 0.0, 0.5, 0.99], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x7)})

    # Input 8: Integer array (implicitly promoted to floating-point)
    x8 = np.array([1, 2, 5, 10, 100], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x8)})

    # Input 9: Scalar represented as a 0D array
    x9 = np.array(2.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x9)})

    # Input 10: Array containing infinity and NaN
    x10 = np.array([1.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x10)})

    # Input 11: High-dimensional array (4D)
    x11 = np.ones((2, 2, 2, 2), dtype=np.float32) * 3.0
    list_of_inputs.append({"x": copy.deepcopy(x11)})

    return list_of_inputs

generated_inputs["jax.numpy.arccosh_1"] = jax_numpy_arccosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arccosh_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arccosh_1'.")


check_valid('jax.numpy.arccosh', generated_inputs['jax.numpy.arccosh_1'], lib="jax", suffix=1)
