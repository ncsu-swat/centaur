
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def power_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, positive values, float exponent 2.0
    x1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x2 = 2.0
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 2: 2D float32 array, positive values, float exponent 0.5 (square root)
    x1 = np.array([[4.0, 9.0], [16.0, 25.0]], dtype=np.float32)
    x2 = 0.5
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 3: 3D float32 array, non-zero values, negative float exponent -1.0
    x1 = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    x2 = -1.0
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 4: 1D int32 array, positive values, float exponent 3.0
    x1 = np.array([1, 2, 3, 4], dtype=np.int32)
    x2 = 3.0
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 5: 2D float64 array, positive values, fractional float exponent 2.5
    x1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    x2 = 2.5
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 6: 1D float32 array with negative values, integer-valued float exponent 3.0
    x1 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    x2 = 3.0
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 7: 0D float32 array (scalar array), float exponent 1.5
    x1 = np.array(9.0, dtype=np.float32)
    x2 = 1.5
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 8: 2D float32 array, zero exponent 0.0
    x1 = np.random.uniform(0.1, 10.0, size=(3, 3)).astype(np.float32)
    x2 = 0.0
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 9: 4D float32 array, negative exponent -2.0
    x1 = np.random.uniform(1.0, 5.0, size=(1, 2, 2, 2)).astype(np.float32)
    x2 = -2.0
    list_of_inputs.append({"x1": x1, "x2": x2})

    # Input 10: 1D int64 array, positive exponent 2.0
    x1 = np.array([10, 20, 30], dtype=np.int64)
    x2 = 2.0
    list_of_inputs.append({"x1": x1, "x2": x2})

    return list_of_inputs

generated_inputs["jax.numpy.power_3"] = power_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.power_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.power_3'.")


check_valid('jax.numpy.power', generated_inputs['jax.numpy.power_3'], lib="jax", suffix=3)
