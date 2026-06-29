
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def bitwise_not_inputs():
    list_of_inputs = []

    # Input 1: bool 1D array
    x = np.array([True, False, True], dtype=np.bool_)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: int8 2D array with signed negative, zero and positive values
    x = np.array([[-128, -1, 0], [1, 127, 42]], dtype=np.int8)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: int16 1D array
    x = np.array([-32768, 0, 32767], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: int32 3D array
    x = np.random.randint(-1000, 1000, size=(2, 2, 2), dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: int64 2D array
    x = np.random.randint(-100000, 100000, size=(3, 3), dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: bool 3D array
    x = np.random.choice([True, False], size=(3, 3, 3)).astype(np.bool_)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: int32 1D array containing extremums and negative values
    x = np.array([-2147483648, -1, 0, 1, 2147483647], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: int64 4D array
    x = np.random.randint(-50, 50, size=(2, 2, 2, 2), dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: int16 2D array
    x = np.array([[-100, 200], [-300, 400]], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: bool 0D array (scalar)
    x = np.array(True, dtype=np.bool_)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.lax.bitwise_not_1"] = bitwise_not_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.bitwise_not_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.bitwise_not_1'.")


check_valid('jax.lax.bitwise_not', generated_inputs['jax.lax.bitwise_not_1'], lib="jax", suffix=1)
