
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def eye_inputs():
    list_of_inputs = []

    # Input 1: Standard square matrix with main diagonal
    input_dict = {
        "N": 3,
        "M": 3,
        "k": np.array(0, dtype=np.int32),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Rectangular matrix (more columns) with positive offset
    input_dict = {
        "N": 4,
        "M": 6,
        "k": np.array(1, dtype=np.int64),
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Rectangular matrix (more rows) with negative offset
    input_dict = {
        "N": 6,
        "M": 4,
        "k": np.array(-2, dtype=np.int32),
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boolean dtype
    input_dict = {
        "N": 5,
        "M": 5,
        "k": np.array(0, dtype=np.int64),
        "dtype": np.dtype(np.bool_),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex64 dtype with positive offset
    input_dict = {
        "N": 4,
        "M": 4,
        "k": np.array(2, dtype=np.int32),
        "dtype": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Single-element matrix
    input_dict = {
        "N": 1,
        "M": 1,
        "k": np.array(0, dtype=np.int32),
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty matrix (0 rows, 0 columns)
    input_dict = {
        "N": 0,
        "M": 0,
        "k": np.array(0, dtype=np.int32),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Unsigned integer dtype with large positive offset
    input_dict = {
        "N": 3,
        "M": 8,
        "k": np.array(4, dtype=np.int64),
        "dtype": np.dtype(np.uint32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Float16 dtype with negative offset
    input_dict = {
        "N": 7,
        "M": 5,
        "k": np.array(-3, dtype=np.int32),
        "dtype": np.dtype(np.float16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Offset outside matrix boundaries
    input_dict = {
        "N": 3,
        "M": 3,
        "k": np.array(5, dtype=np.int32),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Complex128 dtype with large square matrix
    input_dict = {
        "N": 8,
        "M": 8,
        "k": np.array(-1, dtype=np.int64),
        "dtype": np.dtype(np.complex128),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.eye_2"] = eye_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.eye_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.eye_2'.")


check_valid('jax.numpy.eye', generated_inputs['jax.numpy.eye_2'], lib="jax", suffix=2)
