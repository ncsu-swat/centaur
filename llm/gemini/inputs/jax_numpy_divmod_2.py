
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def divmod_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D int32 array, positive integer divisor
    x1 = np.array([10, 20, 30], dtype=np.int32)
    x2 = 3
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 2: 1D array with negative values, negative integer divisor
    x1 = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=np.int32)
    x2 = -3
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 3: 2D int64 array, positive integer divisor
    x1 = np.array([[12, 15], [22, 25]], dtype=np.int64)
    x2 = 5
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 4: 1D float32 array, positive integer divisor
    x1 = np.array([6.5, 7.8, 9.1], dtype=np.float32)
    x2 = 2
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 5: 3D int32 array, positive divisor
    x1 = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    x2 = 7
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 6: 2D uint8 array, positive divisor
    x1 = np.array([[100, 200], [50, 150]], dtype=np.uint8)
    x2 = 13
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 7: 1D float64 array, negative divisor
    x1 = np.array([-10.5, 0.0, 10.5], dtype=np.float64)
    x2 = -4
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 8: 0D array (scalar tensor), positive divisor
    x1 = np.array(42, dtype=np.int32)
    x2 = 10
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 9: 4D int16 array with negative elements, positive divisor
    x1 = np.random.randint(-100, 100, size=(2, 2, 2, 2), dtype=np.int16)
    x2 = 6
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 10: 1D int64 array with large values, large divisor
    x1 = np.array([1000000, 2000000, 3000000], dtype=np.int64)
    x2 = 123456
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.divmod_2"] = divmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.divmod_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.divmod_2'.")


check_valid('jax.numpy.divmod', generated_inputs['jax.numpy.divmod_2'], lib="jax", suffix=2)
