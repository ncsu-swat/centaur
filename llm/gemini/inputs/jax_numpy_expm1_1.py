
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expm1_inputs():
    list_of_inputs = []

    # Input 1: 1D array with values close to 0 (float32)
    x = np.array([1e-4, 1e-6, 2e-10, -1e-5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array with mixed values (float32)
    x = np.array([2.0, -4.0, 3.0, -1.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array of positive values (float64)
    x = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 2D array of negative values (float32)
    x = np.array([[-0.1, -2.3], [-10.5, -0.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D array of random standard normal values (float32)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 0D array (scalar array)
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D array with zeros (float32)
    x = np.zeros((5,), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 4D array with very small negative values (float64)
    x = -np.random.uniform(1e-15, 1e-8, size=(2, 2, 2, 2)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D array of integers (int32)
    x = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D array of complex numbers (complex64)
    x = np.array([[1.0 + 1.0j, -1.0 + 0.5j], [0.0 - 2.0j, 0.1 - 0.1j]], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.expm1_1"] = expm1_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.expm1_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.expm1_1'.")


check_valid('jax.numpy.expm1', generated_inputs['jax.numpy.expm1_1'], lib="jax", suffix=1)
