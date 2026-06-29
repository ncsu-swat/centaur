
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def erfcx_inputs():
    list_of_inputs = []

    # Input 1: 0D array (scalar) - float32
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array with positive values - float32
    x = np.array([0.1, 1.0, 5.0, 10.0, 100.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 1D array with negative values - float32
    x = np.array([-0.1, -1.0, -5.0, -10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 2D array - standard normal distribution - float32
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 2D array - float64
    x = np.random.randn(5, 5).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 3D array with uniform values between -5 and 5 - float32
    x = np.random.uniform(-5.0, 5.0, (2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D array with very large positive values (testing numerical stability)
    x = np.array([100.0, 1000.0, 10000.0], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D array with zero and small values
    x = np.array([-0.0, 0.0, 1e-5, -1e-5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 4D array - float32
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D array with integers stored as floats
    x = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.scipy.special.erfcx"] = erfcx_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.erfcx' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.erfcx'.")


check_valid('jax.scipy.special.erfcx', generated_inputs['jax.scipy.special.erfcx'], lib="jax", suffix=0)
