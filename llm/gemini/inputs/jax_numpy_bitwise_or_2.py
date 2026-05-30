
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_or_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array and integer 1
    x = np.array([0, 1, 2, 3], dtype=np.int32)
    y = 1
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 2D int32 array and integer 2
    x = np.array([[1, 2], [3, 4]], dtype=np.int32)
    y = 2
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: 3D int64 array with negative values and integer -1
    x = np.array([[[ -1, -2], [3, 4]], [[5, 6], [-7, -8]]], dtype=np.int64)
    y = -1
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: 1D int32 array and integer 128
    x = np.array([0, 15, 240, 255], dtype=np.int32)
    y = 128
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: 4D int32 array and integer 15
    x = np.arange(16, dtype=np.int32).reshape((2, 2, 2, 2))
    y = 15
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: 1D int32 array with negative values and integer -5
    x = np.array([-128, -64, 0, 63, 127], dtype=np.int32)
    y = -5
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: 0D int32 array and integer 0
    x = np.array(42, dtype=np.int32)
    y = 0
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: 2D int64 array and positive integer
    x = np.array([[1000, 2000], [3000, 4000]], dtype=np.int64)
    y = 123456
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: 5D int64 array and integer 1024
    x = np.ones((1, 2, 1, 2, 1), dtype=np.int64)
    y = 1024
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: 3D int32 array and integer 65535
    x = np.array([[[1, 2], [3, 4]]], dtype=np.int32)
    y = 65535
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs

generated_inputs["jax.numpy.bitwise_or_2"] = bitwise_or_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.bitwise_or_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.bitwise_or_2'.")


check_valid('jax.numpy.bitwise_or', generated_inputs['jax.numpy.bitwise_or_2'], lib="jax", suffix=2)
