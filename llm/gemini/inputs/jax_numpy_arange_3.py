
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def arange_inputs():
    list_of_inputs = []

    # Input 1: Basic positive integer range
    input_dict = {
        "start": np.array(0, dtype=np.int32),
        "stop": np.array(10, dtype=np.int32),
        "step": np.array(1, dtype=np.int32),
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float range with fractional step
    input_dict = {
        "start": np.array(0.0, dtype=np.float32),
        "stop": np.array(5.0, dtype=np.float32),
        "step": np.array(0.5, dtype=np.float32),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative start to positive stop with step 2
    input_dict = {
        "start": np.array(-10, dtype=np.int32),
        "stop": np.array(10, dtype=np.int32),
        "step": np.array(2, dtype=np.int32),
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Decrementing range with negative step
    input_dict = {
        "start": np.array(10, dtype=np.int32),
        "stop": np.array(0, dtype=np.int32),
        "step": np.array(-1, dtype=np.int32),
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float range with decimal step
    input_dict = {
        "start": np.array(1.0, dtype=np.float32),
        "stop": np.array(3.0, dtype=np.float32),
        "step": np.array(0.25, dtype=np.float32),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Decrementing float range
    input_dict = {
        "start": np.array(5.0, dtype=np.float32),
        "stop": np.array(-5.0, dtype=np.float32),
        "step": np.array(-1.5, dtype=np.float32),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty range (start == stop)
    input_dict = {
        "start": np.array(5, dtype=np.int32),
        "stop": np.array(5, dtype=np.int32),
        "step": np.array(1, dtype=np.int32),
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Step larger than interval (single element)
    input_dict = {
        "start": np.array(0, dtype=np.int32),
        "stop": np.array(5, dtype=np.int32),
        "step": np.array(10, dtype=np.int32),
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative start and negative stop with negative step
    input_dict = {
        "start": np.array(-2, dtype=np.int32),
        "stop": np.array(-10, dtype=np.int32),
        "step": np.array(-3, dtype=np.int32),
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Negative float values
    input_dict = {
        "start": np.array(-4.0, dtype=np.float32),
        "stop": np.array(2.0, dtype=np.float32),
        "step": np.array(0.8, dtype=np.float32),
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.arange_3"] = arange_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.arange_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.arange_3'.")


check_valid('jax.numpy.arange', generated_inputs['jax.numpy.arange_3'], lib="jax", suffix=3)
