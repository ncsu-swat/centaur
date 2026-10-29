
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def asarray_inputs():
    list_of_inputs = []

    # Input 1: Zero float to float32
    input_dict = {
        "a": 0.0,
        "dtype": np.dtype(np.float32),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive float to float64
    input_dict = {
        "a": 1.0,
        "dtype": np.dtype(np.float64),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative float to float32
    input_dict = {
        "a": -3.14159,
        "dtype": np.dtype(np.float32),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive float to int32
    input_dict = {
        "a": 42.0,
        "dtype": np.dtype(np.int32),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large negative float to int64
    input_dict = {
        "a": -1000.0,
        "dtype": np.dtype(np.int64),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Mathematical constant to float16
    input_dict = {
        "a": 2.71828,
        "dtype": np.dtype(np.float16),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float to complex64
    input_dict = {
        "a": 0.5,
        "dtype": np.dtype(np.complex64),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative zero float with float32
    input_dict = {
        "a": -0.0,
        "dtype": np.dtype(np.float32),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small positive float to bool
    input_dict = {
        "a": 1e-4,
        "dtype": np.dtype(np.bool_),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Positive float to uint32
    input_dict = {
        "a": 65535.0,
        "dtype": np.dtype(np.uint32),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Infinity float to float32
    input_dict = {
        "a": float("inf"),
        "dtype": np.dtype(np.float32),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: NaN float to float64
    input_dict = {
        "a": float("nan"),
        "dtype": np.dtype(np.float64),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.asarray_5"] = asarray_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.asarray_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.asarray_5'.")


check_valid('jax.numpy.asarray', generated_inputs['jax.numpy.asarray_5'], lib="jax", suffix=5)
