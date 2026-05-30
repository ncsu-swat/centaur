
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def log1mexp_inputs():
    list_of_inputs = []

    # Input 1: 1D array of float32, small positive values
    x = np.array([0.1, 0.5, 1.0, 2.0, 5.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D array of float32, shape (3, 3)
    x = np.random.uniform(0.1, 10.0, size=(3, 3)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D array of float64
    x = np.random.uniform(0.5, 20.0, size=(2, 3, 4)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 0D array (scalar) of float32
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: Large 1D array with values up to 50
    x = np.random.uniform(1.0, 50.0, size=(100,)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D array of float32
    x = np.random.uniform(0.1, 2.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D array of float64 with tiny positive values
    x = np.random.uniform(1e-5, 1e-1, size=(4, 4)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D array with extremely large values (testing numerical stability)
    x = np.array([10.0, 100.0, 1000.0, 10000.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 5D array of float32
    x = np.random.uniform(0.2, 8.0, size=(1, 2, 1, 3, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D array of float64
    x = np.random.uniform(0.01, 5.0, size=(5, 2)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.nn.log1mexp"] = log1mexp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.log1mexp' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.log1mexp'.")


check_valid('jax.nn.log1mexp', generated_inputs['jax.nn.log1mexp'], lib="jax", suffix=0)
