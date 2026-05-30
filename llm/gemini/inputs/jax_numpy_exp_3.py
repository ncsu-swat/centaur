
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_exp_inputs():
    list_of_inputs = []

    # Input 1: Standard python integer (zero)
    list_of_inputs.append({"x": 0})

    # Input 2: Negative python integer
    list_of_inputs.append({"x": -5})

    # Input 3: Positive python integer
    list_of_inputs.append({"x": 10})

    # Input 4: Numpy int32 scalar
    list_of_inputs.append({"x": np.int32(3)})

    # Input 5: Numpy int64 scalar (negative)
    list_of_inputs.append({"x": np.int64(-4)})

    # Input 6: 1D Numpy array of int32
    list_of_inputs.append({"x": np.array([1, 2, 3], dtype=np.int32)})

    # Input 7: 2D Numpy array of int64 with negative and positive values
    list_of_inputs.append({"x": np.array([[-1, 0], [1, 2]], dtype=np.int64)})

    # Input 8: 3D Numpy array of uint8
    list_of_inputs.append({"x": np.array([[[1, 2], [3, 4]]], dtype=np.uint8)})

    # Input 9: Numpy int16 scalar
    list_of_inputs.append({"x": np.int16(5)})

    # Input 10: 1D Numpy array of int16 with negative values
    list_of_inputs.append({"x": np.array([-10, -5, 0, 5], dtype=np.int16)})

    # Input 11: 4D Numpy array of int32
    list_of_inputs.append({"x": np.ones((2, 2, 2, 2), dtype=np.int32)})

    return list_of_inputs

generated_inputs["jax.numpy.exp_3"] = jax_numpy_exp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.exp_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.exp_3'.")


check_valid('jax.numpy.exp', generated_inputs['jax.numpy.exp_3'], lib="jax", suffix=3)
