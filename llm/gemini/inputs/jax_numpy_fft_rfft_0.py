
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def rfft_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, default-like parameters
    input_dict = {
        "a": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "n": 4,
        "axis": -1,
        "norm": "backward",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 array, odd n, ortho norm
    input_dict = {
        "a": np.array([-1.5, 0.0, 2.5, -3.0, 4.0], dtype=np.float64),
        "n": 5,
        "axis": 0,
        "norm": "ortho",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, forward norm, transform along axis 0
    input_dict = {
        "a": np.random.randn(6, 4).astype(np.float32),
        "n": 6,
        "axis": 0,
        "norm": "forward",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, transform along axis 1 (or -1) with n larger than dimension (zero-padding)
    input_dict = {
        "a": np.random.randn(3, 5).astype(np.float32),
        "n": 8,
        "axis": 1,
        "norm": "backward",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, transform with n smaller than dimension (truncation)
    input_dict = {
        "a": np.random.randn(4, 10).astype(np.float64),
        "n": 4,
        "axis": -1,
        "norm": "ortho",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, axis=1, backward norm
    input_dict = {
        "a": np.random.randn(2, 8, 3).astype(np.float32),
        "n": 8,
        "axis": 1,
        "norm": "backward",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array with negative axis, forward norm
    input_dict = {
        "a": np.random.randn(3, 4, 7).astype(np.float64),
        "n": 7,
        "axis": -2,
        "norm": "forward",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Integer input array (real-valued)
    input_dict = {
        "a": np.array([[1, -2, 3], [4, -5, 6]], dtype=np.int32),
        "n": 3,
        "axis": -1,
        "norm": "ortho",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array, axis=2, odd n padding
    input_dict = {
        "a": np.random.randn(2, 3, 4, 5).astype(np.float32),
        "n": 7,
        "axis": 2,
        "norm": "backward",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array of size 1
    input_dict = {
        "a": np.array([3.14], dtype=np.float32),
        "n": 1,
        "axis": 0,
        "norm": "forward",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D float64 array, axis 0
    input_dict = {
        "a": np.random.randn(5, 2, 2, 2).astype(np.float64),
        "n": 6,
        "axis": 0,
        "norm": "ortho",
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.fft.rfft"] = rfft_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fft.rfft' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fft.rfft'.")


check_valid('jax.numpy.fft.rfft', generated_inputs['jax.numpy.fft.rfft'], lib="jax", suffix=0)
