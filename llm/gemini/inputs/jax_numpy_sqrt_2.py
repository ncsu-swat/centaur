
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_sqrt_inputs():
    list_of_inputs = []

    # Input 1: Scalar python integer
    list_of_inputs.append({"x": 4})

    # Input 2: Negative scalar python integer
    list_of_inputs.append({"x": -9})

    # Input 3: Zero python integer
    list_of_inputs.append({"x": 0})

    # Input 4: Numpy scalar int32
    list_of_inputs.append({"x": np.int32(16)})

    # Input 5: Numpy scalar int64
    list_of_inputs.append({"x": np.int64(25)})

    # Input 6: 1D Numpy array of int32
    list_of_inputs.append({"x": np.array([1, 4, 9, 16], dtype=np.int32)})

    # Input 7: 2D Numpy array of int64
    list_of_inputs.append({"x": np.array([[0, 25], [36, 49]], dtype=np.int64)})

    # Input 8: 3D Numpy array of int16
    list_of_inputs.append({"x": np.array([[[1, 4], [9, 16]], [[25, 36], [49, 64]]], dtype=np.int16)})

    # Input 9: Numpy array of uint8
    list_of_inputs.append({"x": np.array([0, 1, 4, 9], dtype=np.uint8)})

    # Input 10: 1D Numpy array with negative integers (int32)
    list_of_inputs.append({"x": np.array([-4, -1, 0, 1, 4], dtype=np.int32)})

    # Input 11: 4D Numpy array of int32
    list_of_inputs.append({"x": np.ones((2, 2, 2, 2), dtype=np.int32) * 100})

    return [copy.deepcopy(inp) for inp in list_of_inputs]

generated_inputs["jax.numpy.sqrt_2"] = jax_numpy_sqrt_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sqrt_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sqrt_2'.")


check_valid('jax.numpy.sqrt', generated_inputs['jax.numpy.sqrt_2'], lib="jax", suffix=2)
