
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def tril_indices_inputs():
    list_of_inputs = []

    # Input 1: Basic square matrix, main diagonal
    list_of_inputs.append({"n": 3, "k": 0, "m": 3})

    # Input 2: Square matrix, sub-diagonal above main diagonal (k > 0)
    list_of_inputs.append({"n": 4, "k": 1, "m": 4})

    # Input 3: Square matrix, sub-diagonal below main diagonal (k < 0)
    list_of_inputs.append({"n": 5, "k": -1, "m": 5})

    # Input 4: Rectangular matrix (rows < cols), main diagonal
    list_of_inputs.append({"n": 3, "k": 0, "m": 6})

    # Input 5: Rectangular matrix (rows > cols), negative offset
    list_of_inputs.append({"n": 6, "k": -2, "m": 3})

    # Input 6: Rectangular matrix (rows < cols), positive offset
    list_of_inputs.append({"n": 3, "k": 2, "m": 5})

    # Input 7: Single element matrix
    list_of_inputs.append({"n": 1, "k": 0, "m": 1})

    # Input 8: Empty matrix (0 rows, 0 cols)
    list_of_inputs.append({"n": 0, "k": 0, "m": 0})

    # Input 9: Large positive k covering all elements
    list_of_inputs.append({"n": 4, "k": 10, "m": 4})

    # Input 10: Large negative k covering no elements
    list_of_inputs.append({"n": 4, "k": -10, "m": 4})

    # Input 11: Using numpy integer types
    list_of_inputs.append(
        {"n": int(np.int64(7)), "k": int(np.int64(-3)), "m": int(np.int64(8))}
    )

    return list_of_inputs


generated_inputs["jax.numpy.tril_indices"] = tril_indices_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tril_indices' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tril_indices'.")


check_valid('jax.numpy.tril_indices', generated_inputs['jax.numpy.tril_indices'], lib="jax", suffix=0)
