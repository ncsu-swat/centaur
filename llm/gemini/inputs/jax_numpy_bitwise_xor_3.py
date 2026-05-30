
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_xor_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D positive int32 array
    x = 5
    y = np.array([1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 2D array with negative values (int32)
    x = -1
    y = np.array([[-10, 20], [30, -40]], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: 3D array (int32)
    x = 0
    y = np.arange(8, dtype=np.int32).reshape(2, 2, 2)
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: 1D array (int32)
    x = 42
    y = np.array([100, 200, 300], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: 4D array (int64) with negative x
    x = -10
    y = np.ones((2, 2, 2, 2), dtype=np.int64) * 100
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: 2D array (int32) representing bit masks
    x = 255
    y = np.array([[0, 15], [240, 255]], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: 0D array (scalar tensor)
    x = 1
    y = np.array(42, dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: 2D array (int32) with negative values
    x = -3
    y = np.array([[-128, 127], [0, -1]], dtype=np.int32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: Large 1D array (int64)
    x = 123456
    y = np.linspace(0, 1000, 10, dtype=np.int64)
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: 3D array of zeros and ones (int32)
    x = 0
    y = np.random.randint(0, 2, size=(3, 3, 3)).astype(np.int32)
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs

generated_inputs["jax.numpy.bitwise_xor_3"] = bitwise_xor_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_xor_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_xor_3'.")


check_valid('jax.numpy.bitwise_xor', generated_inputs['jax.numpy.bitwise_xor_3'], lib="jax", suffix=3)
