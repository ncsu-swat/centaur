
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cosh_inputs():
    list_of_inputs = []

    # Input 1: float32, 1D array, positive values
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: float32, 1D array, negative values
    x = np.array([-0.5, -1.5, -2.5, -3.5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: float64, 2D array, mixed values
    x = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: float16, 3D array, small values
    x = np.array([[[-0.1, 0.2], [0.3, -0.4]], [[0.5, -0.6], [0.7, -0.8]]], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: complex64, 1D array
    x = np.array([1.0 + 1.0j, -2.0 + 0.5j, 0.0 - 1.5j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: complex128, 2D array
    x = np.array([[0.5 + 0.5j, -0.5 - 0.5j], [1.5j, -1.5j]], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: float32, 0D array (scalar equivalent)
    x = np.array(1.23, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: float32, 1D array with values from a uniform distribution
    x = np.linspace(-5.0, 5.0, 20, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: float32, 4D array of random values
    x = np.random.uniform(-2.0, 2.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: float32, 2D array of zeros
    x = np.zeros((3, 3), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.lax.cosh"] = cosh_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.cosh' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.cosh'.")


check_valid('jax.lax.cosh', generated_inputs['jax.lax.cosh'], lib="jax", suffix=0)
