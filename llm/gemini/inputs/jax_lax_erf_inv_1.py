
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def erf_inv_inputs():
    list_of_inputs = []

    # Input 1: 0D tensor (scalar), positive float32
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D tensor, mixed values in (-1, 1), float32
    x = np.array([-0.8, -0.3, 0.0, 0.3, 0.8], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D tensor, float64, random values in (-1, 1)
    x = np.random.uniform(-0.99, 0.99, size=(3, 3)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D tensor, float32, positive values
    x = np.random.uniform(0.1, 0.9, size=(2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D tensor, float64, values close to boundary
    x = np.array([-0.999, -0.99, 0.99, 0.999], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D tensor, float32, values close to 0
    x = np.random.uniform(-0.1, 0.1, size=(2, 1, 3, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D tensor, shape (1, 5), float32
    x = np.array([[-0.5, -0.25, 0.0, 0.25, 0.5]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 5D tensor, float32, shape (2, 2, 1, 2, 2)
    x = np.random.uniform(-0.7, 0.7, size=(2, 2, 1, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D tensor, single element, negative float32
    x = np.array([-0.75], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D tensor, float32, random values
    x = np.random.uniform(-0.95, 0.95, size=(5, 5)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.lax.erf_inv_1"] = erf_inv_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.erf_inv_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.erf_inv_1'.")


check_valid('jax.lax.erf_inv', generated_inputs['jax.lax.erf_inv_1'], lib="jax", suffix=1)
