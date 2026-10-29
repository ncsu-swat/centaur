
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def positive_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array
    x = np.array([-5.0, 4.0, 7.0, -9.5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D int32 array with mixed signs and zeros
    x = np.array([[-10, 0, 5], [20, -3, -1]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 0D scalar float64 array
    x = np.array(-42.5, dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D float32 array
    x = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D int64 array
    x = np.array([-10000000000, 0, 10000000000], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 2D float64 array
    x = np.array([[1.5, -2.5], [-3.5, 4.5]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D int32 array
    x = np.arange(-12, 12, dtype=np.int32).reshape(2, 2, 2, 3)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D float32 array with special values
    x = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 5D float32 array
    x = np.ones((1, 2, 1, 3, 1), dtype=np.float32) * -1.0
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 2D int64 array
    x = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs


generated_inputs["jax.numpy.positive_1"] = positive_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.positive_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.positive_1'.")


check_valid('jax.numpy.positive', generated_inputs['jax.numpy.positive_1'], lib="jax", suffix=1)
