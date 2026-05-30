
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def add_inputs():
    list_of_inputs = []

    # Input 1: Basic addition with 1D float32 array
    x = 5
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 2: Negative integer with 2D int32 array
    x = -10
    y = np.array([[1, 2], [3, 4]], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 3: Zero with 3D float64 array
    x = 0
    y = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 4: Large integer with 1D float16 array
    x = 1000
    y = np.array([0.5, -0.5, 10.5], dtype=np.float16)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 5: Negative integer with 4D float32 array
    x = -3
    y = np.random.randn(2, 2, 2, 2).astype(np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 6: Addition with 0D (scalar) array
    x = 42
    y = np.array(8.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 7: Positive integer with uint8 array (values capping/wrapping is handled by numpy)
    x = 15
    y = np.array([10, 20, 30], dtype=np.uint8)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 8: Addition with complex array
    x = 2
    y = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 9: Large dimensions, 5D float32 array
    x = -1
    y = np.random.randn(2, 1, 3, 1, 4).astype(np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 10: Addition with a boolean array (treated as 0 and 1)
    x = 1
    y = np.array([[True, False], [False, True]], dtype=bool)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.numpy.add_5"] = add_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.add_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.add_5'.")


check_valid('jax.numpy.add', generated_inputs['jax.numpy.add_5'], lib="jax", suffix=5)
