
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def power_inputs():
    list_of_inputs = []

    # Input 1: positive float, 1D float32 array
    x1 = 2.0
    x2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 2: positive float, 2D int32 array
    x1 = 3.5
    x2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 3: negative float, 1D int32 array (integer exponents are safe for negative bases)
    x1 = -2.0
    x2 = np.array([1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 4: positive float, 3D float64 array
    x1 = 1.5
    x2 = np.random.uniform(0.5, 2.5, size=(2, 2, 2)).astype(np.float64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 5: zero float, 2D int32 array (positive)
    x1 = 0.0
    x2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 6: positive float, 1D float16 array
    x1 = 10.0
    x2 = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float16)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 7: small positive float, 4D int64 array
    x1 = 0.5
    x2 = np.random.randint(1, 5, size=(2, 1, 3, 2)).astype(np.int64)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 8: positive float, 0D array (scalar tensor)
    x1 = 5.2
    x2 = np.array(3, dtype=np.int32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 9: negative float, 2D int16 array
    x1 = -1.5
    x2 = np.array([[2, 4], [6, 8]], dtype=np.int16)
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 10: large float, 1D float32 array (small values to avoid overflow)
    x1 = 100.0
    x2 = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.power_5"] = power_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.power_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.power_5'.")


check_valid('jax.numpy.power', generated_inputs['jax.numpy.power_5'], lib="jax", suffix=5)
