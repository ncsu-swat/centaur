
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def reciprocal_inputs():
    list_of_inputs = []

    # Input 1: Simple scalar integer
    list_of_inputs.append({"x": int(5)})

    # Input 2: Negative scalar integer
    list_of_inputs.append({"x": int(-10)})

    # Input 3: 1D numpy array of int32
    list_of_inputs.append({"x": np.array([1, 2, 3, 4], dtype=np.int32)})

    # Input 4: 2D numpy array of int64
    list_of_inputs.append({"x": np.array([[1, -2], [3, -4]], dtype=np.int64)})

    # Input 5: 3D numpy array of int16
    list_of_inputs.append({"x": np.array([[[1], [2]], [[3], [4]]], dtype=np.int16)})

    # Input 6: 1D numpy array of uint32
    list_of_inputs.append({"x": np.array([10, 20, 30], dtype=np.uint32)})

    # Input 7: Scalar 0 as integer
    list_of_inputs.append({"x": int(0)})

    # Input 8: 1D numpy array of int8 with negative and positive values
    list_of_inputs.append({"x": np.array([-128, -1, 1, 127], dtype=np.int8)})

    # Input 9: 4D numpy array of int32
    list_of_inputs.append({"x": np.ones((2, 2, 2, 2), dtype=np.int32) * 5})

    # Input 10: Scalar np.int64
    list_of_inputs.append({"x": np.int64(42)})

    # Input 11: 2D numpy array of uint8
    list_of_inputs.append({"x": np.array([[1, 2], [3, 4]], dtype=np.uint8)})

    return list_of_inputs

generated_inputs["jax.numpy.reciprocal_3"] = reciprocal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.reciprocal_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.reciprocal_3'.")


check_valid('jax.numpy.reciprocal', generated_inputs['jax.numpy.reciprocal_3'], lib="jax", suffix=3)
