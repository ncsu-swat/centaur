
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def full_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D with scalar float tensor
    input_dict = {
        "shape": 5,
        "fill_value": np.array(3.14, dtype=np.float32),
        "dtype": np.dtype("float32"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Integer shape with integer tensor
    input_dict = {
        "shape": 10,
        "fill_value": np.array(-7, dtype=np.int32),
        "dtype": np.dtype("int32"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Boolean fill value
    input_dict = {
        "shape": 3,
        "fill_value": np.array(True, dtype=np.bool_),
        "dtype": np.dtype("bool"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D tensor broadcasted to shape
    input_dict = {
        "shape": 4,
        "fill_value": np.array([2.5], dtype=np.float64),
        "dtype": np.dtype("float64"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Exact matching 1D tensor
    input_dict = {
        "shape": 3,
        "fill_value": np.array([10, 20, 30], dtype=np.int64),
        "dtype": np.dtype("int64"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex numbers
    input_dict = {
        "shape": 6,
        "fill_value": np.array(1.0 + 2.0j, dtype=np.complex64),
        "dtype": np.dtype("complex64"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Unsigned integer dtype
    input_dict = {
        "shape": 8,
        "fill_value": np.array(255, dtype=np.uint8),
        "dtype": np.dtype("uint8"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Zero shape (empty array)
    input_dict = {
        "shape": 0,
        "fill_value": np.array(0.0, dtype=np.float32),
        "dtype": np.dtype("float32"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative fill value with int16 dtype
    input_dict = {
        "shape": 12,
        "fill_value": np.array(-32768, dtype=np.int16),
        "dtype": np.dtype("int16"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Type casting via dtype
    input_dict = {
        "shape": 2,
        "fill_value": np.array([1.9, -0.4], dtype=np.float32),
        "dtype": np.dtype("int32"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Single element size
    input_dict = {
        "shape": 1,
        "fill_value": np.array(42.0, dtype=np.float16),
        "dtype": np.dtype("float16"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.full_4"] = full_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.full_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.full_4'.")


check_valid('jax.numpy.full', generated_inputs['jax.numpy.full_4'], lib="jax", suffix=4)
