
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def equal_inputs():
    list_of_inputs = []

    # Input 1: 0D tensor, integer scalar comparison
    input_dict = {"x": np.array(0, dtype=np.int32), "y": 0}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 tensor, positive integer
    input_dict = {
        "x": np.array([1, 2, 3, 4, 1], dtype=np.int32),
        "y": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float32 tensor, negative integer
    input_dict = {
        "x": np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32),
        "y": -2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int64 tensor, positive integer
    input_dict = {
        "x": np.array([[5, 10], [15, 5]], dtype=np.int64),
        "y": 5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float64 tensor, zero
    input_dict = {
        "x": np.zeros((3, 3), dtype=np.float64),
        "y": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D int32 tensor, negative integer
    input_dict = {
        "x": np.full((2, 3, 4), -10, dtype=np.int32),
        "y": -10,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D int64 tensor, positive integer
    input_dict = {
        "x": np.arange(24, dtype=np.int64).reshape((2, 3, 4)),
        "y": 12,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 tensor
    input_dict = {
        "x": np.random.randint(-5, 5, size=(2, 2, 2, 2)).astype(np.float32),
        "y": 3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty 1D int32 tensor
    input_dict = {
        "x": np.array([], dtype=np.int32),
        "y": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float32 tensor
    input_dict = {
        "x": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "y": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 5D int32 tensor
    input_dict = {
        "x": np.zeros((1, 2, 1, 3, 2), dtype=np.int32),
        "y": -50,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.equal_3"] = equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.equal_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.equal_3'.")


check_valid('jax.numpy.equal', generated_inputs['jax.numpy.equal_3'], lib="jax", suffix=3)
