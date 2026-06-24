
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def i1_inputs():
    list_of_inputs = []

    # Input 1: 1D array with positive float32 values
    x = np.array([0.1, 1.0, 2.0, 5.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array with negative float32 values
    x = np.array([-0.5, -1.5, -3.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: Scalar as a 0D array
    x = np.array(0.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 2D array with float64 values (positive, negative, and zero)
    x = np.array([[0.0, -1.0, 2.5], [3.1, -4.2, 0.5]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D array with small random float32 values
    x = np.random.uniform(-1.0, 1.0, (2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D array with large values (within reasonable bounds to avoid extreme overflow)
    x = np.array([10.0, -10.0, 15.0, -15.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D array with very small values close to zero
    x = np.array([1e-5, -1e-5, 1e-10, -1e-10], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 4D array with float32 values
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D array of zeros
    x = np.zeros((5,), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D array with float64 values in a larger range
    x = np.random.uniform(-8.0, 8.0, (5, 5)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 1D array containing a mix of integers cast to float
    x = np.arange(-5, 6).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.scipy.special.i1"] = i1_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.i1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.i1'.")


check_valid('jax.scipy.special.i1', generated_inputs['jax.scipy.special.i1'], lib="jax", suffix=0)
