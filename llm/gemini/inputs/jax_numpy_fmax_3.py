
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fmax_inputs():
    list_of_inputs = []

    # Input 1: 1D float array with positive float scalar
    input_dict = {
        "x1": 5.0,
        "x2": np.array([1.0, 7.0, 9.0, 4.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float array with negative float scalar
    input_dict = {
        "x1": -2.5,
        "x2": np.array([[-3.0, 0.0], [4.5, -5.2]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Array with NaN values to test NaN handling
    input_dict = {
        "x1": 3.14,
        "x2": np.array([np.nan, 1.0, np.nan, 5.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar float with Inf / -Inf in the tensor
    input_dict = {
        "x1": 0.0,
        "x2": np.array([-np.inf, np.inf, 0.0, -1.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor with positive scalar
    input_dict = {
        "x1": 10.0,
        "x2": np.random.randn(2, 3, 4).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0D tensor (scalar array)
    input_dict = {
        "x1": -1.0,
        "x2": np.array(-0.5, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float scalar with integer tensor
    input_dict = {
        "x1": 4.5,
        "x2": np.array([[1, 5, 3], [7, 2, 4]], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar float with 4D tensor
    input_dict = {
        "x1": 1.25,
        "x2": np.ones((2, 2, 2, 2), dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large/extreme float values
    input_dict = {
        "x1": float(np.finfo(np.float32).max),
        "x2": np.array([1e30, 1e38, -1e38], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Scalar float NaN
    input_dict = {
        "x1": float("nan"),
        "x2": np.array([1.0, -2.0, np.nan, np.inf], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Float infinity scalar
    input_dict = {
        "x1": float("inf"),
        "x2": np.array([[-100.0, 50.0], [np.nan, 200.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.fmax_3"] = fmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmax_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmax_3'.")


check_valid('jax.numpy.fmax', generated_inputs['jax.numpy.fmax_3'], lib="jax", suffix=3)
