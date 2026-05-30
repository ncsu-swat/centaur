
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_not_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array
    x = np.array([1, 2, 3, 4, 10], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D boolean array
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 1D int8 array with negative and positive values
    x = np.array([-5, -1, 0, 1, 5], dtype=np.int8)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: Another 1D int8 array
    x = np.array([0, 127, -128], dtype=np.int8)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D int16 array using np.arange
    x = np.arange(-10, 10, dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 3D int32 array
    x = np.random.randint(-100, 100, size=(3, 3, 3), dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D int64 array with extreme values
    x = np.array([-9223372036854775808, 9223372036854775807, 0], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 2D int16 array
    x = np.random.randint(-1000, 1000, size=(4, 4), dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 3D boolean array
    x = np.random.choice([True, False], size=(2, 3, 4)).astype(np.bool_)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 0-D array (scalar) int32
    x = np.array(-42, dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 4D int32 array
    x = np.zeros((2, 2, 2, 2), dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.bitwise_not"] = bitwise_not_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_not' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_not'.")


check_valid('jax.numpy.bitwise_not', generated_inputs['jax.numpy.bitwise_not'], lib="jax", suffix=0)
