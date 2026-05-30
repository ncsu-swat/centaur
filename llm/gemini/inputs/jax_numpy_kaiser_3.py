
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_kaiser_inputs():
    list_of_inputs = []

    # Input 1: Window size 0, beta 0
    list_of_inputs.append({"M": 0, "beta": 0})

    # Input 2: Window size 1, beta 5
    list_of_inputs.append({"M": 1, "beta": np.int32(5)})

    # Input 3: Window size 5, beta 0 (as numpy int64)
    list_of_inputs.append({"M": np.int64(5), "beta": 0})

    # Input 4: Standard size and beta (as numpy int32)
    list_of_inputs.append({"M": np.int32(14), "beta": np.int32(6)})

    # Input 5: Larger window size, small beta (numpy int64)
    list_of_inputs.append({"M": 50, "beta": np.int64(2)})

    # Input 6: Medium size, large beta
    list_of_inputs.append({"M": np.int64(32), "beta": np.int64(12)})

    # Input 7: Small window size 2, beta 1
    list_of_inputs.append({"M": 2, "beta": np.int32(1)})

    # Input 8: Even window size 8, beta 8
    list_of_inputs.append({"M": np.int32(8), "beta": np.int32(8)})

    # Input 9: Odd window size 15, beta 3
    list_of_inputs.append({"M": 15, "beta": 3})

    # Input 10: Large window size 1000, beta 5
    list_of_inputs.append({"M": np.int32(1000), "beta": 5})

    # Input 11: Power of two window size, beta 10
    list_of_inputs.append({"M": 256, "beta": np.int64(10)})

    return list_of_inputs

generated_inputs["jax.numpy.kaiser_3"] = generate_kaiser_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.kaiser_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.kaiser_3'.")


check_valid('jax.numpy.kaiser', generated_inputs['jax.numpy.kaiser_3'], lib="jax", suffix=3)
