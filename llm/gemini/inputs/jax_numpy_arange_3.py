
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def arange_inputs():
    list_of_inputs = []

    # Input 1: Basic positive integer sequence
    input_dict = {
        "start": np.array(0, dtype=np.int32),
        "stop": np.array(10, dtype=np.int32),
        "step": np.array(1, dtype=np.int32),
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float32 sequence with fractional step
    input_dict = {
        "start": np.array(0.0, dtype=np.float32),
        "stop": np.array(1.0, dtype=np.float32),
        "step": np.array(0.1, dtype=np.float32),
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative integer range with step
    input_dict = {
        "start": np.array(-10, dtype=np.int32),
        "stop": np.array(-2, dtype=np.int32),
        "step": np.array(2, dtype=np.int32),
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative to positive float range with Float64
    input_dict = {
        "start": np.array(-5.0, dtype=np.float64),
        "stop": np.array(5.0, dtype=np.float64),
        "step": np.array(0.5, dtype=np.float64),
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Int64 large step sequence
    input_dict = {
        "start": np.array(100, dtype=np.int64),
        "stop": np.array(1000, dtype=np.int64),
        "step": np.array(100, dtype=np.int64),
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Downward integer sequence (negative step)
    input_dict = {
        "start": np.array(10, dtype=np.int32),
        "stop": np.array(0, dtype=np.int32),
        "step": np.array(-1, dtype=np.int32),
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Downward float sequence (negative step)
    input_dict = {
        "start": np.array(5.0, dtype=np.float64),
        "stop": np.array(1.0, dtype=np.float64),
        "step": np.array(-0.5, dtype=np.float64),
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float16 precision sequence
    input_dict = {
        "start": np.array(0.0, dtype=np.float16),
        "stop": np.array(2.0, dtype=np.float16),
        "step": np.array(0.2, dtype=np.float16),
        "dtype": np.float16
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single-element range (step is larger than difference)
    input_dict = {
        "start": np.array(0, dtype=np.int32),
        "stop": np.array(1, dtype=np.int32),
        "step": np.array(2, dtype=np.int32),
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixing input array types (will be cast to the specified dtype)
    input_dict = {
        "start": np.array(0, dtype=np.int32),
        "stop": np.array(5.5, dtype=np.float32),
        "step": np.array(0.5, dtype=np.float64),
        "dtype": np.float32
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
