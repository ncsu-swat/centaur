
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def jax_numpy_array_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, ndmin=0
    input_dict = {
        "object": np.array([1.0, -2.5, 3.2], dtype=np.float32),
        "dtype": np.dtype("float32"),
        "copy": True,
        "order": "K",
        "ndmin": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array with negative values, ndmin=2
    input_dict = {
        "object": np.array([[-10, 20], [30, -40]], dtype=np.int32),
        "dtype": np.dtype("int32"),
        "copy": False,
        "order": "K",
        "ndmin": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 0D scalar array expanded to 2D using ndmin
    input_dict = {
        "object": np.array(42, dtype=np.int32),
        "dtype": np.dtype("int32"),
        "copy": True,
        "order": "K",
        "ndmin": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array, ndmin=3
    input_dict = {
        "object": np.ones((2, 3, 4), dtype=np.float32),
        "dtype": np.dtype("float32"),
        "copy": False,
        "order": "K",
        "ndmin": 3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D boolean array, converted to bool dtype, ndmin=1
    input_dict = {
        "object": np.array([True, False, True, True], dtype=np.bool_),
        "dtype": np.dtype("bool"),
        "copy": True,
        "order": "K",
        "ndmin": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D complex64 array, ndmin=2
    input_dict = {
        "object": np.array(
            [[1.0 - 2.0j, 3.5j], [-0.5 + 1.2j, 0.0]], dtype=np.complex64
        ),
        "dtype": np.dtype("complex64"),
        "copy": True,
        "order": "K",
        "ndmin": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D uint8 array, ndmin=4
    input_dict = {
        "object": np.zeros((1, 2, 2, 3), dtype=np.uint8),
        "dtype": np.dtype("uint8"),
        "copy": False,
        "order": "K",
        "ndmin": 4,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D int16 array with large ndmin promotion
    input_dict = {
        "object": np.array([-32768, 0, 32767], dtype=np.int16),
        "dtype": np.dtype("int16"),
        "copy": True,
        "order": "K",
        "ndmin": 3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float16 array converted to float32
    input_dict = {
        "object": np.array(
            [[-1.5, 0.25], [3.125, -4.75]], dtype=np.float16
        ),
        "dtype": np.dtype("float32"),
        "copy": False,
        "order": "K",
        "ndmin": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D int8 array with ndmin=1
    input_dict = {
        "object": np.array([-128, 0, 127], dtype=np.int8),
        "dtype": np.dtype("int8"),
        "copy": True,
        "order": "K",
        "ndmin": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.array_1"] = jax_numpy_array_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.array_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.array_1'.")


check_valid('jax.numpy.array', generated_inputs['jax.numpy.array_1'], lib="jax", suffix=1)
