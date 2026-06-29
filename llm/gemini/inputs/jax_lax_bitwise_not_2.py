
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_not_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar int32
    x = np.array(42, dtype=np.int32)
    list_of_inputs.append({"x": x})

    # Input 2: 1D array int64 with negative values
    x = np.array([-1, 0, 1, -100, 100], dtype=np.int64)
    list_of_inputs.append({"x": x})

    # Input 3: 2D array int16
    x = np.array([[10, -20], [30, -40]], dtype=np.int16)
    list_of_inputs.append({"x": x})

    # Input 4: 3D array int8 with negative values
    x = np.random.randint(-128, 128, size=(2, 3, 4), dtype=np.int8)
    list_of_inputs.append({"x": x})

    # Input 5: 4D array uint32
    x = np.random.randint(0, 1000, size=(2, 2, 2, 2), dtype=np.uint32)
    list_of_inputs.append({"x": x})

    # Input 6: 1D array uint16
    x = np.array([0, 65535, 1024, 2048], dtype=np.uint16)
    list_of_inputs.append({"x": x})

    # Input 7: 2D array uint8
    x = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    list_of_inputs.append({"x": x})

    # Input 8: Large 1D array int64
    x = np.arange(-50, 50, dtype=np.int64)
    list_of_inputs.append({"x": x})

    # Input 9: 0D scalar uint64
    x = np.array(18446744073709551615, dtype=np.uint64)
    list_of_inputs.append({"x": x})

    # Input 10: 5D array int32
    x = np.random.randint(-1000, 1000, size=(1, 2, 3, 2, 1), dtype=np.int32)
    list_of_inputs.append({"x": x})

    return list_of_inputs

generated_inputs["jax.lax.bitwise_not_2"] = bitwise_not_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bitwise_not_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bitwise_not_2'.")


check_valid('jax.lax.bitwise_not', generated_inputs['jax.lax.bitwise_not_2'], lib="jax", suffix=2)
