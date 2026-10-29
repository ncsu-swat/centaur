
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def tri_inputs():
    list_of_inputs = []

    # Input 1: Basic square matrix
    input_dict = {"N": 3, "M": 3, "k": 0, "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Rectangular matrix where M > N
    input_dict = {"N": 3, "M": 5, "k": 0, "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Rectangular matrix where N > M
    input_dict = {"N": 6, "M": 2, "k": 0, "dtype": np.dtype(np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Positive k (above main diagonal)
    input_dict = {"N": 4, "M": 4, "k": 1, "dtype": np.dtype(np.float64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative k (below main diagonal)
    input_dict = {"N": 5, "M": 5, "k": -2, "dtype": np.dtype(np.int64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Boolean dtype
    input_dict = {"N": 4, "M": 4, "k": -1, "dtype": np.dtype(np.bool_)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex dtype
    input_dict = {"N": 3, "M": 3, "k": 2, "dtype": np.dtype(np.complex64)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large positive k exceeding dimension
    input_dict = {"N": 3, "M": 4, "k": 10, "dtype": np.dtype(np.uint8)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large negative k exceeding dimension
    input_dict = {"N": 4, "M": 3, "k": -10, "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1x1 matrix
    input_dict = {"N": 1, "M": 1, "k": 0, "dtype": np.dtype(np.int16)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Zero rows/columns
    input_dict = {"N": 0, "M": 5, "k": 0, "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: Zero columns
    input_dict = {"N": 5, "M": 0, "k": 0, "dtype": np.dtype(np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.tri"] = tri_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tri' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tri'.")


check_valid('jax.numpy.tri', generated_inputs['jax.numpy.tri'], lib="jax", suffix=0)
