
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_isinf_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with mix of inf, -inf, nan, and finite values
    x = np.array([1.0, -2.0, np.inf, -np.inf, np.nan, 0.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 0D array (scalar tensor) with inf
    x = np.array(np.inf, dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D float32 array
    x = np.array([[np.inf, 2.3], [-np.inf, np.nan]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D float16 array with random values and inf
    x = np.random.randn(2, 3, 4).astype(np.float16)
    x[0, 1, 2] = np.inf
    x[1, 2, 3] = -np.inf
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D complex64 array with complex infs
    x = np.array([1+2j, np.inf+3j, 4-np.inf*1j, np.nan+1j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 2D int32 array
    x = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D float64 array with inf and -inf
    x = np.ones((2, 2, 2, 2), dtype=np.float64)
    x[0, 0, 0, 0] = np.inf
    x[1, 1, 1, 1] = -np.inf
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D float32 array with all infs
    x = np.full((5,), np.inf, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 2D float32 array with finite negative numbers
    x = np.array([[-1.0, -100.0], [-0.001, -9999.9]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D float16 array with nan only
    x = np.full((3,), np.nan, dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.isinf_1"] = generate_isinf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isinf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isinf_1'.")


check_valid('jax.numpy.isinf', generated_inputs['jax.numpy.isinf_1'], lib="jax", suffix=1)
