
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def eye_inputs():
    list_of_inputs = []

    # Input 1: Basic square identity with float32
    list_of_inputs.append(
        {"N": 3, "M": 3, "k": 0, "dtype": np.dtype(np.float32)}
    )

    # Input 2: Rectangular matrix with M > N and positive k offset
    list_of_inputs.append({"N": 3, "M": 5, "k": 1, "dtype": np.dtype(np.int32)})

    # Input 3: Rectangular matrix with N > M and negative k offset
    list_of_inputs.append(
        {"N": 6, "M": 4, "k": -2, "dtype": np.dtype(np.float64)}
    )

    # Input 4: Boolean identity matrix
    list_of_inputs.append({"N": 4, "M": 4, "k": 0, "dtype": np.dtype(np.bool_)})

    # Input 5: Offset k larger than M (all zeros result)
    list_of_inputs.append(
        {"N": 3, "M": 3, "k": 5, "dtype": np.dtype(np.float32)}
    )

    # Input 6: Large negative offset k less than -N
    list_of_inputs.append({"N": 4, "M": 4, "k": -10, "dtype": np.dtype(np.int64)})

    # Input 7: Single element matrix (1x1)
    list_of_inputs.append(
        {"N": 1, "M": 1, "k": 0, "dtype": np.dtype(np.uint8)}
    )

    # Input 8: Complex dtype with offset
    list_of_inputs.append(
        {"N": 5, "M": 5, "k": -1, "dtype": np.dtype(np.complex64)}
    )

    # Input 9: Zero-sized dimension N=0
    list_of_inputs.append(
        {"N": 0, "M": 4, "k": 0, "dtype": np.dtype(np.float32)}
    )

    # Input 10: Zero-sized dimension M=0
    list_of_inputs.append({"N": 4, "M": 0, "k": 1, "dtype": np.dtype(np.int16)})

    # Input 11: Larger matrix with int8
    list_of_inputs.append(
        {"N": 10, "M": 12, "k": 3, "dtype": np.dtype(np.int8)}
    )

    # Input 12: Complex128 dtype
    list_of_inputs.append(
        {"N": 2, "M": 6, "k": -1, "dtype": np.dtype(np.complex128)}
    )

    return list_of_inputs


generated_inputs["jax.numpy.eye_1"] = eye_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.eye_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.eye_1'.")


check_valid('jax.numpy.eye', generated_inputs['jax.numpy.eye_1'], lib="jax", suffix=1)
