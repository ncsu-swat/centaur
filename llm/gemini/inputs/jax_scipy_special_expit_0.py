
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expit_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive values
    x = np.array([0.0, 1.0, 2.0, 5.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D float32 array with negative values
    x = np.array([-0.5, -1.5, -3.0, -10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D float64 array with standard normal distribution
    x = np.random.randn(3, 4).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D float32 array
    x = np.random.uniform(-5.0, 5.0, (2, 3, 3)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 0D array (scalar equivalent)
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: Extreme values to test numerical stability
    x = np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D int32 array (which will be cast to float)
    x = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Larger 1D float32 array using arange
    x = np.arange(-10, 10, 0.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 2D float32 array with uniform values
    x = np.random.uniform(-1.0, 1.0, (5, 5)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 4D float32 array
    x = np.ones((2, 2, 3, 3), dtype=np.float32) * -2.5
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.scipy.special.expit"] = expit_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.expit' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.expit'.")


check_valid('jax.scipy.special.expit', generated_inputs['jax.scipy.special.expit'], lib="jax", suffix=0)
