
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isfinite_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with standard numbers
    x = np.array([-1.0, 0.0, 1.0, 2.5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D float32 array containing infinity and NaN
    x = np.array([-np.inf, 3.0, np.inf, np.nan], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D float64 array with some negative values, inf and NaN
    x = np.array([[1.0, -2.0, np.inf], [np.nan, -np.inf, 0.0]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 0D array (scalar representation)
    x = np.array(5.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D float32 array with random numbers
    x = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D int32 array (integers are always finite)
    x = np.array([-10, 0, 10, 20], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D complex64 array containing finite and infinite components
    x = np.array([[1+2j, np.inf+3j], [4-np.inf*1j, np.nan+1j]], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D float16 array
    x = np.array([0.1, -0.2, np.inf], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 4D float32 array
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D float64 array containing only NaN values
    x = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.isfinite_1"] = isfinite_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isfinite_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isfinite_1'.")


check_valid('jax.numpy.isfinite', generated_inputs['jax.numpy.isfinite_1'], lib="jax", suffix=1)
