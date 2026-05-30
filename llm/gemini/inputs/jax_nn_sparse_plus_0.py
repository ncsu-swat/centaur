
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def sparse_plus_inputs():
    list_of_inputs = []

    # Input 1: 1D array with boundary values around -1 and 1
    x = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D array with negative values strictly less than -1
    x = np.random.uniform(-5.0, -1.1, size=(3, 3)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array with positive values strictly greater than 1
    x = np.random.uniform(1.1, 5.0, size=(4, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 0D array (scalar)
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: Large 3D array with mixed values
    x = np.random.uniform(-2.0, 2.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D array of float64
    x = np.array([-1.5, 0.0, 1.5], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D array of float16
    x = np.random.uniform(-1.0, 1.0, size=(5, 2)).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: All zeros
    x = np.zeros((3, 3), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 4D array of float32
    x = np.random.uniform(-3.0, 3.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Array with extreme values (very small and very large)
    x = np.array([-1000.0, -10.0, -1.0, 0.0, 1.0, 10.0, 1000.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 1D array representing a smooth range of inputs
    x = np.linspace(-3.0, 3.0, num=50, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.nn.sparse_plus"] = sparse_plus_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.sparse_plus' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.sparse_plus'.")


check_valid('jax.nn.sparse_plus', generated_inputs['jax.nn.sparse_plus'], lib="jax", suffix=0)
