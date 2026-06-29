
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def population_count_inputs():
    list_of_inputs = []

    # Input 1: 0D tensor, int32
    x = np.array(5, dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D tensor, int8 (negative and positive values)
    x = np.array([-1, 0, 1, 127, -128], dtype=np.int8)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D tensor, int16
    x = np.array([[10, 20], [30, 40]], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D tensor, int32
    x = np.array([[[1, -2], [3, -4]], [[5, -6], [7, -8]]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D tensor, int64
    x = np.array([0, 123456789, -123456789], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 2D tensor, int32 (containing min and max int32)
    x = np.array([[-2147483648, 2147483647], [0, -1]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D tensor, int16
    x = np.array([-32768, 32767, 0, 1], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 3D tensor, int64
    x = np.array([[[-9223372036854775808, 9223372036854775807]]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Large random 1D tensor, int32
    x = np.random.randint(-1000, 1000, size=(100,), dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 4D tensor, int32
    x = np.arange(16, dtype=np.int32).reshape((2, 2, 2, 2))
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.lax.population_count"] = population_count_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.population_count' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.population_count'.")


check_valid('jax.lax.population_count', generated_inputs['jax.lax.population_count'], lib="jax", suffix=0)
