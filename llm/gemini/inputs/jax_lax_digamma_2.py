
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def digamma_inputs():
    list_of_inputs = []

    # Input 1: Scalar float32 positive
    x = np.array(2.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array of float32 positive
    x = np.array([1.0, 2.0, 3.5, 4.2], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array of float64 positive
    x = np.random.uniform(0.1, 10.0, size=(3, 3)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D array of float32 positive
    x = np.random.uniform(1.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D array of float32 negative non-integers (to avoid poles)
    x = np.array([-0.5, -1.5, -2.5, -3.5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D array of float16 positive
    x = np.random.uniform(0.5, 2.5, size=(2, 2, 2, 2)).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D array of float32 large positive values
    x = np.random.uniform(100.0, 1000.0, size=(4, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D array of float64 small positive values
    x = np.array([1e-4, 1e-3, 5e-3], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Scalar float64 negative non-integer
    x = np.array(-0.1, dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 5D array of float32 positive
    x = np.random.uniform(0.5, 1.5, size=(1, 2, 2, 1, 3)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.lax.digamma_2"] = digamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.digamma_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.digamma_2'.")


check_valid('jax.lax.digamma', generated_inputs['jax.lax.digamma_2'], lib="jax", suffix=2)
