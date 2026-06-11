
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def digamma_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive values
    x = np.array([1.0, 2.0, 3.5, 4.2], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float64 array with positive values
    x = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 1D float32 array with negative non-integer values
    x = np.array([-0.5, -1.5, -2.5, -3.1], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: Scalar (0D) float32 array
    x = np.array(5.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D float32 array with random positive values
    x = np.random.uniform(0.1, 10.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D float64 array with very large positive values
    x = np.array([100.0, 1000.0, 10000.0], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D float32 array with very small positive values
    x = np.array([1e-5, 1e-3, 1e-1], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 2D float32 array with mixture of positive and negative non-integers
    x = np.array([[-1.2, 0.5], [2.3, -3.4]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 4D float32 array
    x = np.random.uniform(0.5, 5.5, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D float16 array
    x = np.array([0.2, 1.2, 2.2, 3.2], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.lax.digamma_1"] = digamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.digamma_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.digamma_1'.")


check_valid('jax.lax.digamma', generated_inputs['jax.lax.digamma_1'], lib="jax", suffix=1)
