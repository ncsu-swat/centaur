
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_not_equal_inputs():
    list_of_inputs = []

    # Input 1: x is True, y is 1D boolean array
    x = True
    y = np.array([True, False, True, False], dtype=bool)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 2: x is False, y is 2D integer array
    x = False
    y = np.array([[0, 1], [2, 0]], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 3: x is True, y is 3D float array
    x = True
    y = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 4: x is False, y is 1D float array with negative numbers
    x = False
    y = np.array([-1.0, 0.0, 1.0, -2.5], dtype=np.float64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 5: x is True, y is 4D integer array
    x = True
    y = np.ones((2, 2, 2, 2), dtype=np.int64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 6: x is True, y is 0D scalar array
    x = True
    y = np.array(1, dtype=np.int32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 7: x is False, y is 2D boolean array
    x = False
    y = np.array([[True, False], [False, True]], dtype=bool)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 8: x is True, y is 3D integer array with zeros and ones
    x = True
    y = np.random.randint(0, 2, size=(3, 3, 3)).astype(np.int32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 9: x is False, y is 5D float array
    x = False
    y = np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 10: x is True, y is 1D uint8 array
    x = True
    y = np.array([0, 255, 128], dtype=np.uint8)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.numpy.not_equal_7"] = jax_numpy_not_equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.not_equal_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.not_equal_7'.")


check_valid('jax.numpy.not_equal', generated_inputs['jax.numpy.not_equal_7'], lib="jax", suffix=7)
